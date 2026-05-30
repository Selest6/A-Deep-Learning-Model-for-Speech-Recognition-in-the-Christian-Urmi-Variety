import os
import pandas as pd
import shutil
import wave
from rapidfuzz import process, fuzz




INPUT_FILES = ["Zvukovoj_slovar_2021_10_09.xlsx", "Zvukovoj_slovar_2022.xlsx", "Zvukovoj_slovar_2024.xlsx"]
AUDIO_FOLDER = "audio"

BASE_DIR = "sound_dict_data"
URMI_DIR = os.path.join(BASE_DIR, "urmi_data")
URMI_AUDIO = os.path.join(URMI_DIR, "audio")
URMI_TEXT = os.path.join(URMI_DIR, "texts")

URMI_UNLABELED_DIR = os.path.join(BASE_DIR, "urmi_unlabeled_data")
NON_URMI_DIR = os.path.join(BASE_DIR, "non-urmi_data")

os.makedirs(URMI_AUDIO, exist_ok=True)
os.makedirs(URMI_TEXT, exist_ok=True)
os.makedirs(URMI_UNLABELED_DIR, exist_ok=True)
os.makedirs(NON_URMI_DIR, exist_ok=True)

speaker_dialects = {
    "nlt": 'New urmēžnaya',
    "vms": 'Old urmēžnaya',
    "gij": 'šapətnaya',
    "jsb": 'New urmēžnaya',
    "gjo": 'šapətnaya',
    "nmj": 'nudeznāya'
}

URMI_TYPES = ['New urmēžnaya', 'Old urmēžnaya']




counter_labeled = 139
counter_unlabeled = 236
counter_non_urmi = 393




missing_files = []
fuzzy_logs = []

metadata_urmi = []
metadata_urmi_unlabeled = []
metadata_non_urmi = []




USED_FILES = set()




def normalize(name):
    return str(name).lower().replace(" ", "").replace("_", "").replace("-", "")




ALL_AUDIO_FILES = os.listdir(AUDIO_FOLDER)
FILE_MAP = {normalize(f): f for f in ALL_AUDIO_FILES}
ALL_AUDIO_FILES_NORMALIZED = list(FILE_MAP.keys())




def get_wav_duration(path):
    try:
        with wave.open(path, 'r') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            seconds = frames / float(rate)
            return pd.to_timedelta(seconds, unit='s')
    except:
        return None


def find_best_match(name):
    name_norm = normalize(name)

    available = [f for f in ALL_AUDIO_FILES_NORMALIZED if FILE_MAP[f] not in USED_FILES]

    if not available:
        return None, None

    matches = process.extract(name_norm, available, scorer=fuzz.ratio, limit=1)

    if not matches:
        return None, None

    best_norm, score, _ = matches[0]
    return FILE_MAP[best_norm], score


def process_audio(audio_name, sheet, column, table_name):

    if pd.isna(audio_name):
        return None, None

    audio_name = str(audio_name).replace('\xa0', ' ').strip()

    if audio_name == "" or audio_name.lower() in ["no", "nan"]:
        return None, None

    base_name = audio_name.replace(".wav", "").replace(".WAV", "")
    search_name = base_name + ".wav"
    search_norm = normalize(search_name)




    if search_norm in FILE_MAP:
        real_name = FILE_MAP[search_norm]

        if real_name in USED_FILES:
            fuzzy_logs.append((base_name, real_name, 100, "REJECTED (ALREADY USED)"))
            return None, None

        USED_FILES.add(real_name)

        return os.path.join(AUDIO_FOLDER, real_name), real_name.replace(".wav", "").replace(".WAV", "")




    best_match, score = find_best_match(search_name)

    speaker = column.split("_")[0]
    dialect = speaker_dialects.get(speaker)

    if best_match:
        if best_match in USED_FILES:
            fuzzy_logs.append((base_name, best_match, score, "REJECTED (ALREADY USED)"))
            return None, None

        if score < 95.65:
            fuzzy_logs.append((base_name, best_match, score, "REJECTED (LOW SCORE)"))
            return None, None


        USED_FILES.add(best_match)

        if dialect in URMI_TYPES:
            status = "ADDED (URMI_LABELED)"
        else:
            status = "ADDED (NON_URMI)"

        fuzzy_logs.append((base_name, best_match, score, status))

        return os.path.join(AUDIO_FOLDER, best_match), best_match.replace(".wav", "").replace(".WAV", "")


    fuzzy_logs.append((base_name, "NONE", 0, "REJECTED (NOT FOUND)"))
    missing_files.append((audio_name, table_name, sheet, column))

    return None, None





def process_sheet(df, sheet_name, table_name):

    global counter_labeled, counter_unlabeled, counter_non_urmi

    if sheet_name == "Nouns":
        patterns = ["SG", "PL1", "PL2"]
    elif sheet_name == "Verbs":
        patterns = ["PRS", "PROG", "PST"]
    elif sheet_name == "Adjectives":
        patterns = ["M", "F"]
    elif sheet_name == "Noninflected":
        patterns = [""]
    else:
        return

    for speaker, dialect in speaker_dialects.items():
        for _, row in df.iterrows():
            for p in patterns:

                text_col = f"{speaker}_{p}" if p else speaker
                sound_col = f"{text_col}_sound"

                if sound_col not in df.columns:
                    continue

                audio_file, original_name = process_audio(
                    row.get(sound_col), sheet_name, sound_col, table_name
                )

                if not audio_file:
                    continue

                duration = get_wav_duration(audio_file)

                if dialect in URMI_TYPES:
                    text_value = row.get(text_col)

                    if pd.notna(text_value) and str(text_value).strip() != "":
                        txt_name = f"text{counter_labeled}_urmi_labeled.txt"
                        wav_name = f"audio{counter_labeled}_urmi_labeled.wav"

                        with open(os.path.join(URMI_TEXT, txt_name), "w", encoding="utf-8") as f:
                            f.write(str(text_value))

                        shutil.copy(audio_file, os.path.join(URMI_AUDIO, wav_name))

                        metadata_urmi.append([original_name, txt_name, str(duration), "Sound dictionary of NENA varieties spoken in Russia", dialect])
                        counter_labeled += 1

                    else:
                        wav_name = f"audio{counter_unlabeled}_urmi_unlabeled.wav"

                        shutil.copy(audio_file, os.path.join(URMI_UNLABELED_DIR, wav_name))

                        metadata_urmi_unlabeled.append([original_name, wav_name, str(duration), "Sound dictionary of NENA varieties spoken in Russia", dialect])
                        counter_unlabeled += 1

                else:
                    wav_name = f"audio{counter_non_urmi}_non-urmi_unlabeled.wav"

                    shutil.copy(audio_file, os.path.join(NON_URMI_DIR, wav_name))

                    metadata_non_urmi.append([original_name, wav_name, str(duration), "Sound dictionary of NENA varieties spoken in Russia", dialect])
                    counter_non_urmi += 1





for file in INPUT_FILES:
    xls = pd.ExcelFile(file)
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        process_sheet(df, sheet, file)




pd.DataFrame(metadata_urmi).to_excel(os.path.join(URMI_DIR, "metadata.xlsx"), index=False)
pd.DataFrame(metadata_urmi_unlabeled).to_excel(os.path.join(URMI_UNLABELED_DIR, "metadata.xlsx"), index=False)
pd.DataFrame(metadata_non_urmi).to_excel(os.path.join(NON_URMI_DIR, "metadata.xlsx"), index=False)




print("\n=== FUZZY LOGS (FULL) ===")

for table_name, best_match, score, status in fuzzy_logs:
    print(
        f"Table: {table_name} | Match: {best_match} | Score: {score:.2f}% | Status: {status}"
    )
