import os
import shutil
import pandas as pd
import soundfile as sf

def seconds_to_hms(seconds: float) -> str:
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"



SOURCE_DIR = "Texts_for_tsakorpus_final"
INPUT_XLSX = "Texts_expedition.xlsx"
START_INDEX = 354

VALID_URMI_DIALECTS = {"Urm", "Urm_Arm", "Urm_Geo", "Urm_old"}

OUT_DIR = "nena_corpora_non_urmi_unlabeled_data"
AUDIO_DIR = os.path.join(OUT_DIR, "audio")



os.makedirs(AUDIO_DIR, exist_ok=True)

df = pd.read_excel(INPUT_XLSX)


df = df[~df["Диалект"].isin(VALID_URMI_DIALECTS)]

metadata_rows = []

counter = START_INDEX

for _, row in df.iterrows():
    base_name = row["Название файла"]

    wav_path = os.path.join(SOURCE_DIR, base_name + ".wav")

    if not os.path.exists(wav_path):
        print(f"❌ Нет wav: {base_name}")
        continue


    audio_name = f"audio{counter}_non-urmi_unlabeled.wav"
    audio_out_path = os.path.join(AUDIO_DIR, audio_name)

    shutil.copyfile(wav_path, audio_out_path)


    with sf.SoundFile(audio_out_path) as f:
        duration_sec = len(f) / f.samplerate
        duration_hms = seconds_to_hms(duration_sec)


    metadata_rows.append({
        "title of the text": base_name,
        "audio file": audio_name,
        "audio file length": duration_hms,
        "source": "экспедиционные данные",
        "dialect": row["Диалект"]
    })

    print(f"✅ Готово: {base_name} → {audio_name}")

    counter += 1


metadata_df = pd.DataFrame(metadata_rows)
metadata_df.to_excel(
    os.path.join(OUT_DIR, "non_urmi_unlabeled_metadata.xlsx"),
    index=False
)

print("🎉 Всё завершено!")
