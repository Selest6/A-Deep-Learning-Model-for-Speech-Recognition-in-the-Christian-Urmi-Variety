"""
Обрабатывает все .txt в каталоге texts внутри 2_urmi_labeled (или заданном --input):
- «-», «=», неразрывный дефис ‑ (U+2011), а также  U+F1EA и ꞊ U+A78A → обычный пробел U+0020
- перенос из записи Khan 2016 в целевую систему (см. KHAN_REPLACEMENTS)
- снимает парные обозначения заимствований вокруг слова, как в разметке корпуса: R…R, Arm…Arm,
  Az…Az, E…E, Ge…Ge, P…P (сопоставление до приведения к нижнему регистру; у R/E/P после буквы
  маркера допускаются только Lm-модификаторы ˈ ʾ и т.п.), затем q → k, w → v (до фильтрации)
- сохраняет гласные (сводятся к базовому виду: a, e, i, o, u, ə и ряд близких букв) и согласные
  целевой системы; прочие символы отбрасываются
- результат: одна строка без перевода строки; только одиночные ASCII-пробелы между фрагментами

По умолчанию создаёт зеркало корпуса в --output (копирует audios и прочие файлы из корня
2_urmi_labeled, кроме texts). Для прогона без копирования wav --no-copy-rest.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path


# Целевой инвентарь; порядок сопоставления задаётся через _CLUSTER_ORDER.
ALLOWED_CONSONANT_CLUSTERS: tuple[str, ...] = (
    "\u010d\u0323",
    "c\u0323",
    "\u1e57",
    "\u1e6d",
    "\u0161",
    "\u017e",
    "\u01e7",
    "\u0263",
    "\u010d",
    "p",
    "b",
    "t",
    "d",
    "c",
    "g",
    "k",
    "f",
    "v",
    "s",
    "z",
    "x",
    "h",
    "y",
    "m",
    "n",
    "l",
    "r",
)

_CLUSTER_ORDER: tuple[str, ...] = tuple(
    sorted(ALLOWED_CONSONANT_CLUSTERS, key=lambda s: (-len(s), s))
)

# Гласные в выходе (как у _vowel_cluster_to_canonical); для статистики «по буквам», не по кодпоинтам.
_VOWEL_LETTER_TOKENS: tuple[str, ...] = ("ae", "oe", "\u0259", "a", "e", "i", "o", "u")

_LETTER_TOKEN_ORDER: tuple[str, ...] = tuple(
    sorted(
        frozenset(ALLOWED_CONSONANT_CLUSTERS).union(_VOWEL_LETTER_TOKENS),
        key=lambda s: (-len(s), s),
    )
)

# Базовая согласная (латиница), с которой можно снять только комбинирующие знаки (ударение и т.п.).
_LATIN_CONSONANT_STRIP_BASE: frozenset[str] = frozenset(
    "pbtdcgkfvszxhymnlr"
)

# Замены Khan - целевая система (длинные первыми).
KHAN_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\u010d\u032d", "\u010d\u0323"),  # č̭ → č̣ (через dot below)
    ("c\u032d", "c\u0323"),  # c̭ → c̣
    ("t\u0331", "\u1e6d"),  # t + macron below → ṭ
    ("t\u032d", "\u1e6d"),  # ṱ → ṭ
    ("k\u032d", "k"),  # k̭ → k
    ("k\u0331", "k"),
    ("p\u030c", "\u1e57"),  # p̌ → ṗ
    ("p\u0302", "\u1e57"),  # p̂ → ṗ
    ("\u1e71", "\u1e6d"),  # ṱ → ṭ
    ("g\u0307", "\u0263"),  # ġ → ɣ
    ("\u025f", "g"),  # ɟ → g
    ("j", "\u01e7"),  # j → ǧ (по заданию; после ɟ→g)
    ("\u0121", "\u0263"),  # ġ → ɣ (как ġ)
)

# Парные обёртки заимствований — как в .txt (капс/инициал в маркерах); длинные первыми.
# Основной текст в корпусе строчный, поэтому снятие делается до .lower() (см. process_text).
_BORROWING_WRAPPERS_LONGEST_FIRST: tuple[str, ...] = ("Arm", "Az", "Ge", "R", "E", "P")

# Ключи для статистики снятых обёрток (совпадают с обозначениями в разметке: R…R, P…P, …).
BORROWING_STAT_LABELS: tuple[str, ...] = tuple(
    f"{m}...{m}" for m in _BORROWING_WRAPPERS_LONGEST_FIRST
)
BORROWING_STAT_LABEL_SET: frozenset[str] = frozenset(BORROWING_STAT_LABELS)


def _normalize_separators(text: str) -> str:
    out = text.replace("-", " ")
    out = out.replace("\u2011", " ")  # ‑ NON-BREAKING HYPHEN (как дефис-разделитель)
    out = out.replace("=", " ")
    out = out.replace("\uf1ea", " ")  # «=» в виде 
    out = out.replace("\ua78a", " ")  # ꞊ modifier letter colon (= в корпусе)
    return out


def _apply_khan_map(text: str) -> str:
    """Подстановки Khan; порядок фиксирован (длинные цепочки уже первые в KHAN_REPLACEMENTS)."""
    s = text
    for a, b in KHAN_REPLACEMENTS:
        s = s.replace(a, b)
    return unicodedata.normalize("NFC", s)


def _skip_combining_cluster(s: str, i: int) -> int:
    """Возвращает индекс после кластера: базовая буква + все последующие combining (Mn/Mc)."""
    n = len(s)
    j = i + 1
    while j < n:
        if unicodedata.combining(s[j]):
            j += 1
        else:
            break
    return j


def _is_unicode_whitespace(ch: str) -> bool:
    if ch.isspace():
        return True
    # NBSP, thin space и др.
    return unicodedata.category(ch) in ("Zs", "Zl", "Zp")


def _apply_qw_map(text: str) -> str:
    """q → k, w → v (сохраняем целевые буквы в дальнейшей обработке)."""
    return text.replace("q", "k").replace("w", "v")


def _strip_one_borrowing_wrapper(token: str) -> tuple[str, bool, str | None]:
    """
    Если token — целиком обёртка вида marker…marker (см. _BORROWING_WRAPPERS_LONGEST_FIRST),
    возвращает (внутренность, True, метка для статистики «Arm…Arm», «P…P»).
    Иначе (исходный token, False, None).

    У одиночной буквы маркера (R, E, P) после неё в префиксе допускаются только Lm (ˈ ʾ …);
    закрытие — та же буква + необязательные Lm в конце (напр. E…Eˈ). Для E без Lm с обеих сторон
    не снимаем (ложные срабатывания на однотипные «E» внутри обычных слов).
    """
    if len(token) < 2:
        return token, False, None
    n = len(token)
    for marker in _BORROWING_WRAPPERS_LONGEST_FIRST:
        mlen = len(marker)
        if not token.startswith(marker):
            continue
        if mlen >= 2:
            j = mlen
        else:
            j = 1
            while j < n and unicodedata.category(token[j]) == "Lm":
                j += 1
        if j >= n:
            continue
        if mlen >= 2:
            if not token.endswith(marker):
                continue
            k = n - mlen
        else:
            k = n - 1
            while k >= j and unicodedata.category(token[k]) == "Lm":
                k -= 1
            if k < j or token[k] != marker:
                continue
        if k <= j:
            continue
        inner = token[j:k]
        if not inner:
            continue
        if mlen == 1:
            has_lm_open = j > 1
            has_lm_close = any(
                unicodedata.category(c) == "Lm" for c in token[k + 1 : n]
            )
            if marker == "E" and not has_lm_open and not has_lm_close:
                continue
            if not has_lm_open and not has_lm_close and len(inner) < 2:
                continue
        label = f"{marker}...{marker}"
        return inner, True, label
    return token, False, None


def _remove_borrowing_wrappers_in_text(text: str, removed_counter: Counter[str]) -> str:
    """
    По «словам» (куски между пробельными символами Unicode) снимает парные маркеры заимствований.
    Регистр значим: маркеры задаются как в корпусе (Arm, E, …). Учёт: removed_counter[label]
    для label из числа «Arm...Arm», «P...P», … (см. BORROWING_STAT_LABELS).
    """
    if not text:
        return text
    parts: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if _is_unicode_whitespace(text[i]):
            parts.append(text[i])
            i += 1
            continue
        start = i
        while i < n and not _is_unicode_whitespace(text[i]):
            i += 1
        raw_word = text[start:i]
        stripped, did, borrow_label = _strip_one_borrowing_wrapper(raw_word)
        if did and borrow_label is not None:
            removed_counter[borrow_label] += 1
        parts.append(stripped)
    return "".join(parts)


def _vowel_cluster_to_canonical(cluster_nfc: str) -> str | None:
    """Гласный кластер NFC → одна базовая гласная (aeiouə) или цепочка вида ae/oe."""
    if not cluster_nfc:
        return None
    d = unicodedata.normalize("NFD", cluster_nfc)
    base = d[0]
    rest = d[1:]
    if rest and not all(unicodedata.combining(c) for c in rest):
        return None

    if base in "aeiou":
        return base
    if base == "\u0259":  # ə
        return "\u0259"
    if base == "\u01dd":  # ǝ LATIN SMALL LETTER TURNED E → schwa
        return "\u0259"
    if base in "\u0251\u0250":  # ɑ ɐ
        return "a"
    if base == "\u0153":  # œ
        return "oe"
    if base in "\u025b\u025c":  # ɛ ɜ
        return "e"
    if base == "\u00e6":  # æ
        return "ae"
    if base in "\u026a\u0268":  # ɪ ɨ
        return "i"
    if base == "\u0131":  # ı
        return "i"
    if base in "\u00f6\u00f8":  # ö ø
        return "o"
    if base == "\u00fc":  # ü
        return "u"
    if base in "\u00e4\u00e5":  # ä å → a
        return "a"
    return None


def _latin_consonant_stress_only(cluster_nfc: str) -> str | None:
    """Например ý → y, ṇ не покрывается — только база из _LATIN_CONSONANT_STRIP_BASE + Mn."""
    if not cluster_nfc:
        return None
    d = unicodedata.normalize("NFD", cluster_nfc)
    base = d[0]
    if base not in _LATIN_CONSONANT_STRIP_BASE:
        return None
    i = 1
    while i < len(d) and unicodedata.combining(d[i]):
        i += 1
    if i < len(d):
        return None
    return base


def extract_allowed_segments(text: str, removed_counter: Counter[str]) -> str:
    """
    Согласные из _CLUSTER_ORDER + канонические гласные + только U+0020 между словами.
    Без символов новой строки в результате.
    """
    s = unicodedata.normalize("NFC", text.lower())
    n = len(s)
    out: list[str] = []
    i = 0
    while i < n:
        ch = s[i]
        if _is_unicode_whitespace(ch):
            out.append(" ")
            i += 1
            while i < n and _is_unicode_whitespace(s[i]):
                i += 1
            continue
        matched = False
        for tok in _CLUSTER_ORDER:
            if s.startswith(tok, i):
                out.append(tok)
                i += len(tok)
                matched = True
                break
        if matched:
            continue
        start = i
        i = _skip_combining_cluster(s, start)
        cluster_raw = s[start:i]
        cluster = unicodedata.normalize("NFC", cluster_raw)

        vowel = _vowel_cluster_to_canonical(cluster)
        if vowel is not None:
            out.append(vowel)
            continue

        cons = _latin_consonant_stress_only(cluster)
        if cons is not None:
            out.append(cons)
            continue

        for k in range(start, i):
            removed_counter[s[k]] += 1

    merged = " ".join("".join(out).split())
    # Для ASR и токенизаторов стабильнее канонически сложенный вид (NFC), не NFD.
    return unicodedata.normalize("NFC", merged)


def letter_and_nonletter_counts(processed_line: str) -> tuple[Counter[str], Counter[str]]:
    """
    Логические «буквы» (согласные кластеры + гласные, как один токен) и прочее
    (пробел и любые недопустимые фрагменты после обработки).
    """
    letters: Counter[str] = Counter()
    nonletters: Counter[str] = Counter()
    i = 0
    n = len(processed_line)
    while i < n:
        ch = processed_line[i]
        if ch == " ":
            nonletters[" "] += 1
            i += 1
            continue
        matched = False
        for tok in _LETTER_TOKEN_ORDER:
            if processed_line.startswith(tok, i):
                letters[tok] += 1
                i += len(tok)
                matched = True
                break
        if matched:
            continue
        start = i
        i = _skip_combining_cluster(processed_line, start)
        nonletters[processed_line[start:i]] += 1
    return letters, nonletters


def _text_unit_label(unit: str) -> str:
    """Подпись для ключа статистики: одна скаляр или цепочка (буква/кусок мусора)."""
    if len(unit) == 1:
        o = ord(unit)
        if unit.isprintable() and not unit.isspace():
            return repr(unit)
        return f"U+{o:04X} ({unicodedata.name(unit, '????')})"
    return repr(unit)


def process_text(raw: str, removed_counter: Counter[str]) -> str:
    s = unicodedata.normalize("NFC", raw)
    s = _normalize_separators(s)
    # Маркеры заимствований в корпусе с капсом — снимаем до lower(), затем Khan и q/w.
    s = _remove_borrowing_wrappers_in_text(s, removed_counter)
    s = s.lower()
    s = _apply_khan_map(s)
    s = _apply_qw_map(s)
    return extract_allowed_segments(s, removed_counter)


def mirror_non_texts(src_root: Path, dst_root: Path) -> None:
    """Копирует всё из src_root кроме каталога texts."""
    for item in src_root.iterdir():
        if item.name == "texts":
            continue
        dest = dst_root / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parent / "2_urmi_labeled",
        help="Каталог 2_urmi_labeled (с подпапкой texts)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "3_urmi_labeled",
        help="Куда записать зеркало корпуса с обработанными текстами",
    )
    ap.add_argument(
        "--no-copy-rest",
        action="store_true",
        help="Не копировать audios и прочие файлы (только каталог texts в выходе)",
    )
    args = ap.parse_args()
    src: Path = args.input.expanduser().resolve()
    dst: Path = args.output.expanduser().resolve()
    texts_src = src / "texts"
    texts_dst = dst / "texts"

    if not texts_src.is_dir():
        raise FileNotFoundError(f"Нет папки texts: {texts_src}")

    dst.mkdir(parents=True, exist_ok=True)
    texts_dst.mkdir(parents=True, exist_ok=True)

    if not args.no_copy_rest:
        mirror_non_texts(src, dst)

    removed_total: Counter[str] = Counter()
    letters_total: Counter[str] = Counter()
    nonletters_total: Counter[str] = Counter()
    n_files = 0

    for f in sorted(texts_src.glob("*.txt")):
        raw = f.read_text(encoding="utf-8", errors="replace")
        file_removed: Counter[str] = Counter()
        out = process_text(raw, file_removed)
        (texts_dst / f.name).write_text(out, encoding="utf-8", newline="")
        removed_total.update(file_removed)
        lo, nx = letter_and_nonletter_counts(out)
        letters_total.update(lo)
        nonletters_total.update(nx)
        n_files += 1

    def _sym_label(c: str) -> str:
        if len(c) != 1:
            return repr(c)
        o = ord(c)
        if c.isprintable() and not c.isspace():
            return repr(c)
        return f"U+{o:04X} ({unicodedata.name(c, '????')})"

    print(f"Обработано файлов: {n_files}")
    print(f"Выход: {dst}")
    print()
    print("Удалённые символы (сумма по всем файлам, по кодовым позициям)")
    borrow_rows = [(removed_total[lbl], lbl) for lbl in BORROWING_STAT_LABELS if removed_total[lbl]]
    if borrow_rows:
        print("  Снятые обозначения заимствований:")
        for cnt, lbl in sorted(borrow_rows, key=lambda x: -x[0]):
            print(f"    {repr(lbl)}: {cnt}")
    for ch, cnt in removed_total.most_common():
        if ch in BORROWING_STAT_LABEL_SET:
            continue
        print(f"  {_sym_label(ch)}: {cnt}")
    print()
    print("Буквы в обработанных текстах (логические токены; сумма)")
    for tok, cnt in letters_total.most_common():
        print(f"  {_text_unit_label(tok)}: {cnt}")
    print()
    print("Небуквенные единицы в обработанных текстах (пробел, прочее; сумма)")
    for unit, cnt in nonletters_total.most_common():
        print(f"  {_text_unit_label(unit)}: {cnt}")


if __name__ == "__main__":
    main()
