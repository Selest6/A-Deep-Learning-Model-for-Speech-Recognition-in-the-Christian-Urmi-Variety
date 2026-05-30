# A Deep Learning Model for Speech Recognition in the Christian Urmi Variety

Code repository accompanying the bachelor's thesis *A Deep Learning Model for Speech Recognition in the Christian Urmi Variety*

This project develops an automatic speech recognition (ASR) system for the **Christian Urmi variety** of North-Eastern Neo-Aramaic (NENA) — a low-resource endangered language. The pipeline covers data collection, preprocessing, model training, and evaluation. The best-performing checkpoint and a web demo are published separately on Hugging Face (links below).

### Related resources

- **Model checkpoint:** [Selest/wav2vec2-bert_Assyrian_Urmi_ASR_model](https://huggingface.co/Selest/wav2vec2-bert_Assyrian_Urmi_ASR_model)
- **Web interface (Gradio on Hugging Face Spaces):** [Selest/wav2vec2-bert_Assyrian_Urmi_ASR_model_interface](https://huggingface.co/spaces/Selest/wav2vec2-bert_Assyrian_Urmi_ASR_model_interface) — upload or record audio and get transcriptions in three writing systems (Russian researchers', Geoffrey Khan's, or Cyrillic)

## Repository structure

```
.
├── Data collection/              # Download and organize raw audio from external sources
├── Data preprocessing/
│   ├── Labeled data preprocessing/   # Prepare paired audio–text data for supervised training
│   └── Unlabeled data preprocessing/ # Prepare unlabeled audio for continued pre-training
└── Model creation/               # Fine-tuning and experiment scripts
```

Scripts are designed to be run in pipeline order within each folder (numbered filenames indicate the recommended sequence). Most preprocessing notebooks were originally written for Google Colab and expect archives downloaded from Yandex Disk; update the placeholder links before running.

---

## Data collection

Scripts that gather audio from four main sources described in the thesis:

1. [The North-Eastern Neo-Aramaic Database Project](https://nena-staging.ames.cam.ac.uk/) (University of Cambridge)
2. [Corpus of NENA varieties spoken in Russia](https://nenadict.iling.spb.ru/corpus)
3. [Sound dictionary of NENA varieties spoken in Russia](https://nenadict.iling.spb.ru/dictionary)
4. Unpublished field recordings collected by Russian Urmi researchers

| File | Purpose |
|------|---------|
| `nena_corpus_urmi_labeled.py` | Extracts **labeled Urmi** audio and transcriptions from the NENA corpus. Outputs paired `audio/` and `texts/` folders with a metadata spreadsheet. |
| `nena_corpus_non-urmi_unlabeled.py` | Extracts **unlabeled Non-Urmi** audio from the same NENA corpus. Keeps audio only — no transcriptions. |
| `nena_db_urmi_labeled_unlabeled.py` | Scrapes the **NENA online database** for Urmi entries: downloads audio, extracts time-coded Aramaic segments when available, and splits output into labeled (with text) and unlabeled subsets. |
| `nena_db_non-urmi_unlabeled.py` | Scrapes the NENA database for **Non-Urmi varieties** without transcriptions. Downloads and converts audio to WAV. |
| `sound_dict_urmi_labeled_unlabeled_non-urmi_unlabeled.py` | Processes the **Sound Dictionary** files: matches entries to audio files (including extra audio folders), uses fuzzy matching for filename alignment. |
| `sound_dict_urmi_labeled_unlabeled_non-urmi_unlabeled_old.py` | Earlier version of the Sound Dictionary collector (single audio folder, simpler speaker–dialect mapping). Kept for reference. |
| `field_data_urmi_unlabeled_non-urmi_unlabeled.py` | Organizes **field recordings** from researcher folders using metadata. Splits files into Urmi and non-Urmi unlabeled sets based on dialect labels. |
| `field_data_nudezne_vocab_non-urmi_unlabeled.py` | Copies and renames **Nudezne vocabulary** recordings into a standardized unlabeled audio folder with metadata (non-Urmi variety). |

---

## Data preprocessing

### Labeled data preprocessing

Transforms raw labeled recordings into a clean train/test-ready corpus (`audios/` + `texts/` pairs).

| File | Purpose |
|------|---------|
| `1_labeled_data_audio_text_segmenter.ipynb` | **Segments** long recordings into shorter utterances using timestamp metadata. Cuts both audio (via `pydub`) and corresponding text. Drops segments containing Russian speech. |
| `2_labeled_data_metadata_inventory.ipynb` | Builds a **metadata inventory** for the segmented labeled corpus: file paths, durations, dialect labels, and source information (exported to Excel). |
| `3_labeled_data_audio_normalization.ipynb` | **Normalizes audio** to a unified format: mono, 16 kHz, 16-bit PCM WAV (via `ffmpeg` / `librosa`). |
| `4_labeled_data_texts_preprocessing.py` | **Normalizes transcriptions:** converts Geoffrey Khan's orthography to the Russian researchers' system; splits clitics and verb prefixes into separate tokens; removes borrowing markers (`R…R`, `Arm…Arm`, etc.); strips punctuation, stress marks, and suprasegmental emphasis markers; lowercases text; keeps only valid Urmi phoneme inventory. |
| `5_labeled_data_train_test_split.py` | **Splits** the labeled corpus into train and test sets with balanced total **duration** per dialect (not just file count). Copies `audios/` and `texts/` into separate train/test directories. |

### Unlabeled data preprocessing

Prepares unlabeled audio for continued self-supervised pre-training and for filtering non-target-language speech.

| File | Purpose |
|------|---------|
| `1_unlabeled_data_normalization.ipynb` | Converts all unlabeled audio to **mono 16 kHz 16-bit WAV** using `ffmpeg`. Skips corrupt or very short files; logs failures. |
| `2_unlabeled_data_vad_segmentation.ipynb` | **Segments** long files with **Silero VAD**: extracts speech regions, splits on pauses, outputs clips of 2–25 seconds. |
| `3_unlabeled_data_quality_filter.ipynb` | **Filters by quality:** removes clips with insufficient speech (VAD), low RMS (too quiet), or excessive noise. |
| `4_unlabeled_data_whisper_language_filter.ipynb` | Uses **Whisper (small)** to detect and **remove** clips dominated by Russian or English speech. |
| `5_unlabeled_data_metadata_inventory.ipynb` | Builds a final **metadata inventory** for the cleaned unlabeled corpus (durations, sources, file counts). |

---

## Model creation

Training scripts expect labeled data:

```
train_dir/
├── audios/
│   ├── utterance_001.wav
│   └── ...
└── texts/
    ├── utterance_001.txt
    └── ...
```

Each `.txt` file contains a single line of transcription matching the basename of the corresponding `.wav`.

### Shared utilities

| File | Purpose |
|------|---------|
| `asr_training_common.py` | Shared training utilities used by Wav2Vec2-BERT scripts: loading audio/text splits, character vocabulary extraction, WER/CER computation (`jiwer`), greedy and beam decoding, metric export (Excel/JSON), training plots, checkpoint management, and resume logic. |
| `requirements_asr.txt` | Python dependencies for model training (`torch`, `transformers`, `datasets`, `librosa`, `jiwer`, `wandb`, etc.). |

### MMS experiments

| File | Purpose |
|------|---------|
| `mms.py` | **Full fine-tuning** of `facebook/mms-1b-all` with CTC loss. Default: no SpecAugment, cosine LR schedule, checkpoint selection by minimum validation CER. |
| `mms_adapters-only.py` | **Adapter-only fine-tuning** of MMS: freezes the wav2vec2 encoder and trains only language attention adapters + CTC head. Used to compare adapter-only vs. full fine-tuning (thesis section 4.1.2). Enables SpecAugment time masking by default (`mask_time_prob=0.05`). |

### Wav2Vec2-BERT experiments

| File | Purpose |
|------|---------|
| `w2v2-bert.py` | **Baseline** full fine-tuning of `facebook/w2v-bert-2.0` without augmentation. Compared against MMS in thesis section 4.1.3. |
| `w2v2-bert_Specaugment.py` | Fine-tuning with **SpecAugment** (encoder-level time and frequency masking: `mask_time_prob=0.05`, `mask_feature_prob=0.02`). Thesis section 4.2. |
| `w2v2-bert_speed_perturbation.py` | Fine-tuning with **speed perturbation** on the waveform (±10%, probability 0.5 via `librosa.effects.time_stretch`). This configuration achieved the **best test results** and was chosen as the final model. Thesis section 4.2. |
| `w2v2-bert_urmi-pretrained_speed-perturbation.py` | **Continued self-supervised pre-training** on unlabeled Urmi audio (masked-frame reconstruction with a temporary MLP decoder), followed by CTC fine-tuning with speed perturbation. Thesis section 4.3. |
| `w2v2-bert_non-urmi-pretrained_speed-perturbation.py` | Same pipeline as above, but pre-training uses **Urmi + phonetically related non-Urmi** NENA varieties (Northern, Ashiret, Southern). Thesis section 4.3. |
