import os
import pandas as pd
import shutil
from pydub import AudioSegment
from difflib import SequenceMatcher


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


folders = [
    "texts_2",
    "texts_k_2023",
    "texts_nn_2023",
    "texts",
    "Audio (Urmiya)"
]

excel_file = "Texts_NENA_all.xlsx"

urmi_output_folder = "unlabeled_urmi_data"
non_urmi_output_folder = "unlabeled_non_urmi_data"

os.makedirs(urmi_output_folder, exist_ok=True)
os.makedirs(non_urmi_output_folder, exist_ok=True)

low_files = set()
not_found_files = set()
non_wav_files = set()

urmi_dialects = {
    "Arm_Urm",
    "Urm",
    "Urm_Arm",
    "Urm_Arm, Urm_new",
    "Urm_Geo",
    "Urm_new",
    "Urm_new, Urm_Arm",
    "Urm_old"
}

files_to_ignore = {
    "190705_ena_kk_lz_otce nas",
    "190705_ena_zas_kk_lz_kupletny na svadbe",
    "190705_ena_zas_kk_lz_pesnja 2",
    "190705_ena_zas_kk_lz_pesnja vazo",
    "190705_zas_kk_lz_molitva bozja mater",
    "190705_zas_kk_lz_molitvy pered snom",
    "190709_ggo_kk_mo_ss_pesnja_2",
    "190709_ggo_kk_mo_ss_pesnja_vadra",
    "210810_lb_ar_es_pesna_o_neveste",
    "210810_lb_ar_es_pesna_o_vesne",
    "210811_nlt_ss_pesnja_o_vesne",
    "210815_tmj_lz_nl_molitva_deve_marii",
    "210815_tmj_lz_nl_molitva_pered_snom"
}


def normalize(name):
    name = name.strip().lower()
    if name.endswith(".wav"):
        name = name[:-4]
    return name



df1 = pd.read_excel(excel_file, sheet_name="Ready_for_transcription")
df2 = pd.read_excel(excel_file, sheet_name="V_rabote")
df = pd.concat([df1, df2])

df["Название файла"] = df["Название файла"].astype(str).str.strip()
df["Диалект"] = df["Диалект"].astype(str)


file_index = {}

for folder in folders:
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        if not os.path.isfile(path):
            continue

        name, ext = os.path.splitext(file)
        key = normalize(name)

        file_index[key] = (path, file, folder, ext.lower())


all_keys = list(file_index.keys())


urmi_counter = 2
non_urmi_counter = 234

urmi_records = []
non_urmi_records = []

def get_audio_length(path):
    try:
        audio = AudioSegment.from_file(path)
        return len(audio) / 1000
    except:
        return None


fuzzy_matches = []



for _, row in df.iterrows():

    name = row["Название файла"].strip()
    dialect = row["Диалект"]
    description = str(row.get("Описание", "")).lower()

    name_clean = normalize(name).upper()


    unsuitable_keywords = [
        "LOW", "MOLITVA", "PESNA", "PESNYA", "PESNJA", "OTCE_NAS",
        "211205_LAM_ES_SKAZKA_PRO_SYNA_PASTUXA_I_CHUDOVISHCH"
    ]


    if (
            name_clean.endswith("_RU") or
            any(x in name_clean for x in unsuitable_keywords) or
            "молитва" in description or
            "песня" in description
    ):
        low_files.add(name)
        continue

    key = normalize(name)

    if key not in file_index:
        not_found_files.add(name)
        continue

    file_path, original_file, folder, ext = file_index[key]


    if ext != ".wav":
        try:
            audio = AudioSegment.from_file(file_path)
            wav_path = os.path.splitext(file_path)[0] + ".wav"
            audio.export(wav_path, format="wav")

            file_path = wav_path
            ext = ".wav"
        except:
            non_wav_files.add((original_file, folder))
            continue

    length = get_audio_length(file_path)


    if dialect in urmi_dialects:
        new_name = f"audio{urmi_counter}_urmi_unlabeled.wav"
        new_path = os.path.join(urmi_output_folder, new_name)

        shutil.copy(file_path, new_path)

        urmi_records.append({
            "title of the text": name,
            "audio file": new_name,
            "audio file length": length,
            "source": "field data",
            "dialect": dialect
        })

        urmi_counter += 1


    else:
        new_name = f"audio{non_urmi_counter}_non-urmi_unlabeled.wav"
        new_path = os.path.join(non_urmi_output_folder, new_name)

        shutil.copy(file_path, new_path)

        non_urmi_records.append({
            "title of the text": name,
            "audio file": new_name,
            "audio file length": length,
            "source": "field data",
            "dialect": dialect
        })

        non_urmi_counter += 1


cleaned_not_found = {
    name for name in not_found_files
    if normalize(name) not in files_to_ignore
}

for name in cleaned_not_found:
    key = normalize(name)

    best_score = 0
    best_matches = []

    for candidate in all_keys:
        score = similarity(key, candidate)

        if score > best_score:
            best_score = score
            best_matches = [candidate]
        elif score == best_score:
            best_matches.append(candidate)


    if best_score >= 0.8947 and len(best_matches) == 1:
        matched_key = best_matches[0]
        file_path, original_file, folder, ext = file_index[matched_key]


        if ext != ".wav":
            try:
                audio = AudioSegment.from_file(file_path)
                wav_path = os.path.splitext(file_path)[0] + ".wav"
                audio.export(wav_path, format="wav")
                file_path = wav_path
            except:
                non_wav_files.add((original_file, folder))
                continue

        length = get_audio_length(file_path)


        row_match = df[df["Название файла"].str.strip() == name]
        dialect = row_match.iloc[0]["Диалект"]

        if dialect in urmi_dialects:
            new_name = f"audio{urmi_counter}_urmi_unlabeled.wav"
            new_path = os.path.join(urmi_output_folder, new_name)

            shutil.copy(file_path, new_path)

            urmi_records.append({
                "title of the text": name,
                "audio file": new_name,
                "audio file length": length,
                "source": "field data",
                "dialect": dialect
            })

            urmi_counter += 1
            status = "ADDED (URMI)"

        else:
            new_name = f"audio{non_urmi_counter}_non-urmi_unlabeled.wav"
            new_path = os.path.join(non_urmi_output_folder, new_name)

            shutil.copy(file_path, new_path)

            non_urmi_records.append({
                "title of the text": name,
                "audio file": new_name,
                "audio file length": length,
                "source": "field data",
                "dialect": dialect
            })

            non_urmi_counter += 1
            status = "ADDED (NON-URMI)"

    else:
        status = "NOT ADDED"

    fuzzy_matches.append({
        "original_name": name,
        "best_matches": best_matches,
        "score": best_score,
        "status": status
    })



pd.DataFrame(urmi_records).to_excel("unlabeled_urmi_data.xlsx", index=False)
pd.DataFrame(non_urmi_records).to_excel("unlabeled_non-urmi_data.xlsx", index=False)


print("Unsuitable files:")
print(low_files)

print("\nFailed to convert files:")
print(non_wav_files)

print("\nFuzzy matching results:")

for item in fuzzy_matches:
    print("\n---")
    print(f"Table name: {item['original_name']}")
    print(f"Best match(es): {item['best_matches']}")
    print(f"Similarity: {round(item['score'] * 100, 2)}%")
    print(f"Status: {item['status']}")
