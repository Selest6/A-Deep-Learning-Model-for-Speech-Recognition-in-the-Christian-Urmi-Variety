import os
import shutil
import wave
import contextlib
from datetime import timedelta
from openpyxl import Workbook

src_folder = "200_words"
dst_folder = "200_words_audios"

os.makedirs(dst_folder, exist_ok=True)

counter = 393

non_wav_files = []

wb = Workbook()
ws = wb.active
ws.title = "metadata"

ws.append([
    "title of the text",
    "audio file",
    "audio file length",
    "source",
    "dialect"
])

for filename in os.listdir(src_folder):
    src_path = os.path.join(src_folder, filename)

    if not os.path.isfile(src_path):
        continue

    if filename.lower().endswith(".wav"):
        new_name = f"audio{counter}_non-urmi_unlabeled.wav"
        dst_path = os.path.join(dst_folder, new_name)

        shutil.copy2(src_path, dst_path)

        try:
            with contextlib.closing(wave.open(src_path, 'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration_seconds = frames / float(rate)
                duration_str = str(timedelta(seconds=int(duration_seconds)))
        except:
            duration_str = "00:00:00"

        ws.append([
            filename,
            new_name,
            duration_str,
            "nudezne_vocabulary",
            "nudezne"
        ])

        counter += 1
    else:
        non_wav_files.append(filename)

wb.save("nudezne_metadata.xlsx")

print("Не WAV файлы:")
for f in non_wav_files:
    print(f)
