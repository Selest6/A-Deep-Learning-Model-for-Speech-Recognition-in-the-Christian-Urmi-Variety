"""
Делит выборку Urmi labeled (по таблице ASR model data - лист Urmi labeled process)
на train/test так, чтобы в тестовой выборке у каждого диалекта была по возможности
одинаковая доля суммарной длительности (≈ 1/N от общего времени теста для N диалектов;
доля общей выборки задаётся --test-fraction). Это баланс по времени, а не по числу файлов.
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

try:
    import pandas as pd
except ImportError as e:
    print("Нужны пакеты: pip install pandas openpyxl tqdm", file=sys.stderr)
    raise SystemExit(1) from e

try:
    from tqdm import tqdm
except ImportError as e:
    print("Нужны пакеты: pip install pandas openpyxl tqdm", file=sys.stderr)
    raise SystemExit(1) from e

EXCEL_GLOB_NAMES = "*asr*model*data*.xlsx"
SHEET = "Urmi labeled process"
COL_PARTS = "parts"
COL_TEXT_PARTS = "text parts"
COL_DIALECT = "dialect"
# разные апострофы в имени столбца
COL_LENGTH_CANDIDATES = (
    "parts' length",
    "parts’ length",
    "parts_length",
)

TEXTS_FALLBACK_DIRS = (
    Path("3_urmi_labeled") / "texts",
    Path("2_urmi_labeled") / "texts",
)


def canonical_suffix(filename: str) -> str | None:
    """Нормализованный суффикс имени после префикса audio / text (как при парах corpus)."""
    stem = Path(filename).stem.lower()
    if stem.startswith("audio"):
        return stem[len("audio") :]
    if stem.startswith("text"):
        return stem[len("text") :]
    return None


def build_suffix_to_txt_basename(text_dir: Path) -> dict[str, str]:
    """суффикс (нижний регистр) -> имя .txt файла."""
    out: dict[str, str] = {}
    for p in sorted(text_dir.glob("*.txt")):
        suf = canonical_suffix(p.name)
        if suf is None:
            continue
        key = suf.lower()
        if key not in out:
            out[key] = p.name
    return out


def resolve_default_text_src(base: Path) -> Path | None:
    """Первый из TEXTS_FALLBACK_DIRS, который существует."""
    for rel in TEXTS_FALLBACK_DIRS:
        p = (base / rel).resolve()
        if p.is_dir():
            return p
    return None


def resolve_txt_basename(
    row: dict,
    text_cat: Path,
    suffix_to_txt: dict[str, str],
) -> str | None:
    """
    Имя .txt: приоритет — ячейка «text parts» (basename), если такой файл есть;
    иначе парное имя из suffix_to_txt по имени WAV в «parts».
    """
    raw_tp = row.get("text_parts")
    if raw_tp not in (None, "") and (
        not (isinstance(raw_tp, float) and pd.isna(raw_tp))
    ):
        tn = Path(str(raw_tp).strip().replace("\\", "/")).name
        candidates: list[str] = []
        if tn:
            candidates.append(tn)
            if not tn.lower().endswith(".txt"):
                candidates.append(f"{Path(tn).stem}.txt")
        for cand in candidates:
            if cand and (text_cat / cand).is_file():
                return cand

    wav = basename_only(row.get("parts"))
    if not wav:
        return None
    suf = canonical_suffix(wav)
    if suf is None:
        return None
    cand = suffix_to_txt.get(suf.lower())
    if cand and (text_cat / cand).is_file():
        return cand
    return None


def find_excel(base: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Нет Excel: {p}")
        return p
    candidates = sorted(base.glob(EXCEL_GLOB_NAMES))
    if not candidates:
        candidates = sorted(base.glob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError(f"Не найден .xlsx в {base}")
    return candidates[0]


def resolve_length_column(columns: pd.Index) -> str:
    norm = {str(c).strip(): str(c).strip() for c in columns}
    for cand in COL_LENGTH_CANDIDATES:
        if cand in norm:
            return norm[cand]
    # апостроф/типография: сравниваем буквы
    for c in columns:
        s = str(c).strip().lower().replace("’", "'")
        if s.startswith("parts") and "length" in s:
            return str(c).strip()
    raise KeyError(
        "Не найден столбец длительности (ожидалось что-то вроде «parts' length»). "
        f"Есть столбцы: {list(columns)!r}"
    )


def parse_duration_seconds(val: object) -> float | None:
    """Секунды: float/int, timedelta, datetime.time из Excel или строки мм:ss / ч:мм:сс."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, dt.time):
        return (
            val.hour * 3600
            + val.minute * 60
            + val.second
            + val.microsecond / 1_000_000.0
        )
    if hasattr(val, "total_seconds"):
        try:
            return float(val.total_seconds())
        except (TypeError, ValueError):
            pass
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)

    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None

    if re.fullmatch(r"\d+(?:\.\d+)?", s):
        return float(s)

    parts = re.split(r"[:∶]", s)
    parts = [p.strip() for p in parts]
    try:
        nums = [float(p.replace(",", ".")) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, sec = nums
        return h * 3600 + m * 60 + sec
    if len(nums) == 2:
        m, sec = nums
        return m * 60 + sec
    return None


def is_aggregate_parts_row(parts_val: object) -> bool:
    if parts_val is None or (isinstance(parts_val, float) and pd.isna(parts_val)):
        return False
    t = str(parts_val).strip().lower()
    return "total amount" in t


def parts_is_no_parts(parts_val: object) -> bool:
    if parts_val is None or (isinstance(parts_val, float) and pd.isna(parts_val)):
        return True
    s = str(parts_val).strip().lower()
    return s == "" or s == "no parts"


def row_key(it: dict) -> tuple[str, str]:
    return (it["parts"], str(it.get("text_parts") or ""))


def stratified_test_by_duration(
    items_by_dialect: dict[str, list[dict]],
    test_fraction: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """
    Стремится дать каждому диалекту в тесте похожую суммарную длительность:
    цель на диалект = (test_fraction * общая длительность) / число диалектов.
    Выбор клипов — жадно в случайном порядке.
    """
    dialects_nonempty = [d for d, lst in items_by_dialect.items() if lst]
    if not dialects_nonempty:
        return [], []

    total_sec = sum(
        float(x["duration_sec"])
        for lst in items_by_dialect.values()
        for x in lst
    )
    n_d = len(dialects_nonempty)
    target_per_dialect = test_fraction * total_sec / n_d

    test_keys: set[tuple[str, str]] = set()
    for dname in dialects_nonempty:
        pool = list(items_by_dialect[dname])
        rng.shuffle(pool)
        dialect_total = sum(float(x["duration_sec"]) for x in pool)
        target = min(target_per_dialect, dialect_total)
        acc = 0.0
        for it in pool:
            if acc >= target:
                break
            test_keys.add(row_key(it))
            acc += float(it["duration_sec"])

    uniq: dict[tuple[str, str], dict] = {}
    for lst in items_by_dialect.values():
        for it in lst:
            k = row_key(it)
            uniq.setdefault(k, dict(it))

    train: list[dict] = []
    test: list[dict] = []
    for it in uniq.values():
        if row_key(it) in test_keys:
            test.append(it)
        else:
            train.append(it)
    return train, test


def print_stats(title: str, rows: list[dict], file=sys.stdout) -> None:
    by_d: dict[str, float] = defaultdict(float)
    count_by_d: dict[str, int] = defaultdict(int)
    for it in rows:
        d = str(it.get("_dialect", it.get("dialect", ""))).strip() or "(пусто)"
        by_d[d] += float(it["duration_sec"])
        count_by_d[d] += 1
    total = sum(by_d.values())
    print(title, file=file)
    print(f"  Файлов: {len(rows)}", file=file)
    print(f"  Суммарная длительность: {total:.2f} с ({total / 3600:.4f} ч)", file=file)
    if total <= 0:
        return
    print("  По диалектам:", file=file)
    for d in sorted(by_d.keys(), key=lambda x: x.lower()):
        sec = by_d[d]
        pct = 100.0 * sec / total
        print(f"    {d!r}: {sec:.2f} с ({pct:.2f}%), файлов: {count_by_d[d]}", file=file)


def basename_only(parts_value: object) -> str:
    raw = "" if parts_value is None else str(parts_value).strip()
    if not raw:
        return ""
    return Path(raw.replace("\\", "/")).name


def copy_audio_for_split(
    rows: list[dict],
    audio_src: Path,
    audio_dest: Path,
    label: str,
) -> tuple[int, int]:
    """
    Копирует файлы из audio_src по полю «parts» (имя файла или путь → basename).
    Возвращает (сколько скопировано, сколько ошибок — пустые имена или нет файла).
    """
    audio_dest.mkdir(parents=True, exist_ok=True)
    missing = 0
    seen_dest: set[str] = set()
    todo: list[tuple[Path, Path]] = []

    for it in rows:
        name = basename_only(it.get("parts"))
        if not name:
            print(f"[{label}] Пустое имя в parts, строка пропущена.", file=sys.stderr)
            missing += 1
            continue
        if name in seen_dest:
            continue
        seen_dest.add(name)
        src_file = audio_src / name
        if not src_file.is_file():
            print(f"[{label}] Нет файла в источнике: {src_file}", file=sys.stderr)
            missing += 1
            continue
        todo.append((src_file, audio_dest / name))

    ok = len(todo)
    for src_file, dst in tqdm(todo, desc=f"Аудио {label}", unit="файл"):
        shutil.copy2(src_file, dst)
    return ok, missing


def copy_text_for_split(
    rows: list[dict],
    text_cat: Path,
    suffix_to_txt: dict[str, str],
    text_dest: Path,
    label: str,
) -> tuple[int, int]:
    """
    Копирует в text_dest строки транскрипций из каталога text_cat (.txt имена см. resolve_txt_basename).
    Возвращает (число скопированных, число ошибок — не найдено имя или нет файла).
    """
    text_dest.mkdir(parents=True, exist_ok=True)
    missing = 0
    failed_sample: list[str] = []
    seen_txt: set[str] = set()
    todo: list[tuple[Path, Path]] = []

    for it in rows:
        txt_name = resolve_txt_basename(it, text_cat, suffix_to_txt)
        if not txt_name:
            missing += 1
            wav = basename_only(it.get("parts"))
            if len(failed_sample) < 25:
                failed_sample.append(str(wav) if wav else "?")
            continue
        if txt_name in seen_txt:
            continue
        seen_txt.add(txt_name)
        src_file = text_cat / txt_name
        if not src_file.is_file():
            missing += 1
            print(f"[{label}] Нет текстового файла: {src_file}", file=sys.stderr)
            continue
        todo.append((src_file, text_dest / txt_name))

    ok = len(todo)
    for src_file, dst in tqdm(todo, desc=f"Тексты {label}", unit="файл"):
        shutil.copy2(src_file, dst)
    if failed_sample:
        samp = "; ".join(failed_sample[:10])
        extra = f" (ещё строк: {missing - 10})" if missing > 10 else ""
        print(
            f"[{label}] Нет .txt по parts для {missing} строк{extra}. Примеры: {samp}",
            file=sys.stderr,
        )
    return ok, missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Каталог с Excel (по умолчанию — рядом со скриптом)",
    )
    ap.add_argument("--excel", type=Path, default=None, help="Путь к .xlsx")
    ap.add_argument(
        "--test-fraction",
        type=float,
        default=0.15,
        help="Доля всей выборки (по длительности), попадающая в тест (по умолчанию 0.15)",
    )
    ap.add_argument("--seed", type=int, default=42, help="Seed для случайного порядка клипов")
    ap.add_argument(
        "--stats-only",
        action="store_true",
        help="Только текстовая статистика: не записывать CSV и не копировать аудио",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Каталог для train.csv/test.csv "
            "(по умолчанию <base>/splits); игнорируется при --stats-only"
        ),
    )
    ap.add_argument(
        "--audio-src",
        type=Path,
        default=None,
        help="Откуда копировать wav: по умолчанию <base>/time_urmi_labeled",
    )
    ap.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Папка train: по умолчанию <base>/5h_train",
    )
    ap.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Папка test: по умолчанию <base>/5h_test",
    )
    ap.add_argument(
        "--no-audio-copy",
        action="store_true",
        help="Не копировать аудио в подпапку audios (CSV по-прежнему пишется, если не --stats-only)",
    )
    ap.add_argument(
        "--text-src",
        type=Path,
        default=None,
        help=(
            "Каталог исходных .txt (<base>/3_urmi_labeled/texts или 2_urmi_labeled/texts если не указан)"
        ),
    )
    ap.add_argument(
        "--no-text-copy",
        action="store_true",
        help="Не копировать транскрипции в подпапку texts",
    )
    args = ap.parse_args()

    if not (0 < args.test_fraction < 1):
        print("--test-fraction должен быть между 0 и 1.", file=sys.stderr)
        return 1

    base = args.base.expanduser().resolve()
    excel = find_excel(base, args.excel)

    df = pd.read_excel(excel, sheet_name=SHEET, engine="openpyxl")
    df.columns = pd.Index([str(c).strip() for c in df.columns])

    for col in (COL_PARTS, COL_TEXT_PARTS, COL_DIALECT):
        if col not in df.columns:
            print(
                f"Нет столбца {col!r}. Доступные: {list(df.columns)!r}",
                file=sys.stderr,
            )
            return 1
    len_col = resolve_length_column(df.columns)

    rng = random.Random(args.seed)

    items_by_dialect: dict[str, list[dict]] = defaultdict(list)
    skipped_no_len = 0
    agg_skipped = 0

    for _, row in df.iterrows():
        parts = row[COL_PARTS]
        if is_aggregate_parts_row(parts):
            agg_skipped += 1
            continue
        if parts_is_no_parts(parts):
            continue

        dur = parse_duration_seconds(row[len_col])
        if dur is None or dur < 0:
            skipped_no_len += 1
            continue

        dialect = row[COL_DIALECT]
        if dialect is None or (isinstance(dialect, float) and pd.isna(dialect)):
            dname = ""
        else:
            dname = str(dialect).strip()

        txt = row[COL_TEXT_PARTS]
        text_parts = None if pd.isna(txt) else str(txt).strip()

        audio_name = str(parts).strip()
        items_by_dialect[dname].append(
            {
                "parts": audio_name,
                "text_parts": text_parts or "",
                "duration_sec": dur,
                "dialect": dname,
            }
        )

    dialects_nonempty = [
        k for k, v in items_by_dialect.items() if sum(x["duration_sec"] for x in v) > 0
    ]
    if len(dialects_nonempty) < len(items_by_dialect):
        for dead in list(items_by_dialect.keys()):
            if dead not in dialects_nonempty:
                del items_by_dialect[dead]

    train, test = stratified_test_by_duration(dict(items_by_dialect), args.test_fraction, rng)

    for it in train:
        it["_dialect"] = it.get("dialect", "")
    for it in test:
        it["_dialect"] = it.get("dialect", "")

    print(f"Excel: {excel}", file=sys.stderr)
    print(f"Пропущено агрегирующих строк (total amount …): {agg_skipped}", file=sys.stderr)
    print(f"Строк без валидной длительности: {skipped_no_len}", file=sys.stderr)
    print(f"Диалектов в выборке: {len(items_by_dialect)}", file=sys.stderr)
    print(f"Стратегия: цель длительности теста на диалект ≈ общая цель × 1/N", file=sys.stderr)
    print()

    print_stats("Обучение (train)", train)
    print()
    print_stats("Тест (test)", test)
    total_test_sec = sum(float(x["duration_sec"]) for x in test)
    if total_test_sec > 1e-6:
        print()
        print("Оценка баланса в тесте (доля длительности по диалекту, должно быть близко к равному):")
        by_d: dict[str, float] = defaultdict(float)
        for it in test:
            d = str(it.get("dialect", "")).strip() or "(пусто)"
            by_d[d] += float(it["duration_sec"])
        ideal = 100.0 / len(by_d) if by_d else 0.0
        for d in sorted(by_d.keys(), key=lambda x: x.lower()):
            pct = 100.0 * by_d[d] / total_test_sec
            delta = pct - ideal
            print(f"  {d!r}: {pct:.2f}% целое ~ {ideal:.2f}%, Δ {delta:+.2f}%")

    if args.stats_only:
        return 0

    out = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else (base / "splits").resolve()
    )
    out.mkdir(parents=True, exist_ok=True)
    cols = ["parts", "text_parts", "dialect", "duration_sec"]
    pd.DataFrame(train)[cols].to_csv(out / "train.csv", index=False, encoding="utf-8")
    pd.DataFrame(test)[cols].to_csv(out / "test.csv", index=False, encoding="utf-8")
    print(file=sys.stderr)
    print(f"Записано: {out / 'train.csv'}, {out / 'test.csv'}", file=sys.stderr)

    # Раскладка дерева для train/test: <root>/audios и <root>/texts.
    audio_src = (
        args.audio_src.expanduser().resolve()
        if args.audio_src is not None
        else base / "time_urmi_labeled"
    )
    train_dir = (
        args.train_dir.expanduser().resolve()
        if args.train_dir is not None
        else base / "5h_train"
    )
    test_dir = (
        args.test_dir.expanduser().resolve()
        if args.test_dir is not None
        else base / "5h_test"
    )
    audio_train = train_dir / "audios"
    audio_test = test_dir / "audios"
    texts_train = train_dir / "texts"
    texts_test = test_dir / "texts"

    text_cat = (
        args.text_src.expanduser().resolve()
        if args.text_src is not None
        else resolve_default_text_src(base)
    )

    suffix_to_txt: dict[str, str] = {}
    if not args.no_text_copy and text_cat is not None and text_cat.is_dir():
        suffix_to_txt = build_suffix_to_txt_basename(text_cat)
    elif not args.no_text_copy:
        print(
            f"Нет каталога текстов ({', '.join(str(base / x) for x in TEXTS_FALLBACK_DIRS)}) — "
            "подставьте --text-src. Подпапка texts создаётся только при наличии источника.",
            file=sys.stderr,
        )

    if args.no_audio_copy:
        print("Аудио: пропуск (--no-audio-copy).", file=sys.stderr)
    elif not audio_src.is_dir():
        print(
            f"Нет каталога с аудио {audio_src} — аудио не копируются.",
            file=sys.stderr,
        )
    else:
        tr_ok, tr_miss = copy_audio_for_split(train, audio_src, audio_train, "train")
        te_ok, te_miss = copy_audio_for_split(test, audio_src, audio_test, "test")
        print(
            f"Аудио train: скопировано {tr_ok}, пропусков {tr_miss} → {audio_train}",
            file=sys.stderr,
        )
        print(
            f"Аудио test: скопировано {te_ok}, пропусков {te_miss} → {audio_test}",
            file=sys.stderr,
        )

    if args.no_text_copy:
        print("Тексты: пропуск (--no-text-copy).", file=sys.stderr)
    elif text_cat is None or not text_cat.is_dir():
        pass  # предупреждение выведено выше
    else:
        ttr_ok, ttr_miss = copy_text_for_split(
            train, text_cat, suffix_to_txt, texts_train, "train"
        )
        tte_ok, tte_miss = copy_text_for_split(
            test, text_cat, suffix_to_txt, texts_test, "test"
        )
        print(
            f"Тексты train: скопировано {ttr_ok}, пропусков {ttr_miss} → {texts_train}",
            file=sys.stderr,
        )
        print(
            f"Тексты test: скопировано {tte_ok}, пропусков {tte_miss} → {texts_test}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
