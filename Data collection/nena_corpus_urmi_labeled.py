import os
import shutil
import pandas as pd
from lxml import etree
import soundfile as sf


def seconds_to_hms(seconds: float) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"





SOURCE_DIR = "Texts_for_tsakorpus_final"
INPUT_XLSX = "Texts_expedition.xlsx"
START_INDEX = 63

VALID_DIALECTS = {"Urm", "Urm_Arm", "Urm_Geo", "Urm_old"}

OUT_DIR = "labeled_data"
AUDIO_DIR = os.path.join(OUT_DIR, "audio")
TEXT_DIR = os.path.join(OUT_DIR, "texts")



os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

df = pd.read_excel(INPUT_XLSX)

df = df[df["Диалект"].isin(VALID_DIALECTS)]

metadata_rows = []

counter = START_INDEX

for _, row in df.iterrows():
    base_name = row["Название файла"]

    wav_path = os.path.join(SOURCE_DIR, base_name + ".wav")
    flex_path = os.path.join(SOURCE_DIR, base_name + ".flextext")

    if not os.path.exists(wav_path):
        print(f"❌ Нет wav: {base_name}")
        continue

    if not os.path.exists(flex_path):
        print(f"❌ Нет flextext: {base_name}")
        continue


    audio_name = f"audio{counter}_urmi_labeled.wav"
    audio_out_path = os.path.join(AUDIO_DIR, audio_name)

    shutil.copyfile(wav_path, audio_out_path)


    with sf.SoundFile(audio_out_path) as f:
        duration_sec = len(f) / f.samplerate
        duration_hms = seconds_to_hms(duration_sec)


    text_name = f"text{counter}_urmi_labeled.xlsx"
    text_out_path = os.path.join(TEXT_DIR, text_name)

    tree = etree.parse(flex_path)
    root = tree.getroot()

    rows = []

    for phrase in root.xpath(".//phrase"):
        begin = phrase.get("begin-time-offset")
        end = phrase.get("end-time-offset")

        words = []
        for item in phrase.xpath(".//words/word/item[@type='txt']"):
            if item.text:
                words.append(item.text)

        phrase_text = " ".join(words)

        if phrase_text.strip():
            rows.append({
                "timecode": f"{begin}-{end}",
                "text": phrase_text
            })

    text_df = pd.DataFrame(rows, columns=["timecode", "text"])
    text_df.to_excel(text_out_path, index=False)


    metadata_rows.append({
        "title of the text": base_name,
        "text file": text_name,
        "audio file length (seconds)": duration_hms,
        "source": "экспедиционные данные",
        "dialect": row["Диалект"]
    })

    print(f"✅ Готово: {base_name} → {counter}")

    counter += 1


metadata_df = pd.DataFrame(metadata_rows)
metadata_df.to_excel(
    os.path.join(OUT_DIR, "expedition_labeled_metadata.xlsx"),
    index=False
)

print("🎉 Всё завершено!")
