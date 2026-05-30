from __future__ import annotations
import argparse
import inspect
import json
import logging
import os
from contextlib import nullcontext
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC, Wav2Vec2Processor
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Fine-tune MMS ASR for Urmia (wav+txt folders).')
    p.add_argument('--model_name_or_path', type=str, default='facebook/mms-1b-all', help='Базовая модель MMS (или путь к локальной копии после huggingface-cli download).')
    p.add_argument('--train_dir', type=str, required=True, help='Папка с подкаталогами audios/ и texts/.')
    p.add_argument('--test_dir', type=str, required=True, help='Тестовая папка audios/ + texts/.')
    p.add_argument('--output_dir', type=str, default='./mms-urmi-out', help='Чекпоинты и логи.')
    p.add_argument('--num_train_epochs', type=float, default=30.0)
    p.add_argument('--per_device_train_batch_size', type=int, default=2, help='По умолчанию ниже для стабильности fp16.')
    p.add_argument('--per_device_eval_batch_size', type=int, default=2)
    p.add_argument('--learning_rate', type=float, default=4e-05, help='Ниже прошлого 1e-4 — меньше риска взрыва на fp16.')
    p.add_argument('--warmup_ratio', type=float, default=0.15, help='Доля warmup длиннее, чем 0.1 — мягче рост LR.')
    p.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Эффективный batch = × per_device_* (умолчанию 2×2=4/GPU).')
    p.add_argument('--precision', type=str, choices=('fp16', 'auto', 'bf16', 'fp32'), default='fp16', help='Смешанная точность: fp16 — по умолчанию (GradScaler); auto — bf16 если GPU поддерживает, иначе fp16; bf16 — принудительно (нет на старых GPU); fp32 — без AMP.')
    p.add_argument('--max_train_samples', type=int, default=None, help='Ограничить число обучающих примеров (отладка).')
    p.add_argument('--save_steps', type=int, default=500)
    p.add_argument('--eval_steps', type=int, default=500)
    p.add_argument('--logging_steps', type=int, default=50)
    p.add_argument('--freeze_feature_encoder', action='store_true', help='Заморозить CNN фронтенд.')
    p.add_argument('--train_adapters_only', action='store_true', help='Заморозить весь wav2vec2-энкодер; обучать только MMS attention adapters и lm_head (см. transformers examples run_speech_recognition_ctc_adapter). Требуется чекпоинт с adapter_attn_dim (facebook/mms-1b-all, mms-1b-l1107, mms-1b-fl102), не голый mms-1b.')
    p.add_argument('--wandb_project', type=str, default='mms-urmi-asr', help='Проект Weights & Biases.')
    p.add_argument('--wandb_run_name', type=str, default=None, help='Имя рана в W&B (опционально).')
    p.add_argument('--disable_wandb', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_total_limit', type=int, default=3, help='Сколько последних checkpoint-* держать на диске во время обучения (HF всегда сохраняет best при load_best_model_at_end). -1 или 0 — без ограничения.')
    p.add_argument('--keep_intermediate_checkpoints', action='store_true', help='Не удалять папки checkpoint-* после обучения (по умолчанию удаляются, остаются final_model и run_artifacts).')
    p.add_argument('--beam_width', type=int, default=10, help='Размер beam при финальной оценке pyctcdecode (лучшая модель).')
    p.add_argument('--skip_beam_eval', action='store_true', help='Не считать WER/CER с beam search (быстрее).')
    p.add_argument('--resume_from_checkpoint', type=str, default=None, metavar='PATH_OR_LAST', help='Продолжить Trainer с чекпоинта HF: последний сохранённый шаг из --output_dir — «last» (или «true», «1», «yes»); или полный путь к каталогу checkpoint-XXXX. Подтягиваются веса, optimizer и LR scheduler, global_step. Нужны те же --train_dir/--test_dir/--model_name_or_path/--train_adapters_only и т.д., что и при первом запуске; output_dir — тот же, где лежат checkpoint-*.')
    p.add_argument('--debug_trainable_grads', action='store_true', help='Перед trainer.train(): один forward+backward на первом train-батче, логирует норму градиентов по requires_grad=True, max_grad_norm из TrainingArguments и сколько параметров без grad. Не вызывает optimizer.step.')
    return p.parse_args()

def parse_resume_from_checkpoint_arg(value: Optional[str]) -> Union[bool, str, None]:
    if value is None:
        return None
    v = str(value).strip()
    if not v:
        return None
    if v.lower() in ('last', 'true', '1', 'yes'):
        return True
    return str(Path(v).expanduser())

def find_latest_checkpoint_dir(output_dir: Path) -> Optional[Path]:
    dirs = [p for p in output_dir.glob('checkpoint-*') if p.is_dir()]
    if not dirs:
        return None

    def step_key(p: Path) -> int:
        suf = p.name.split('-')[-1]
        return int(suf) if suf.isdigit() else -1
    return max(dirs, key=step_key)

def resolve_resume_checkpoint_dir(output_dir: Path, resume_parsed: Union[bool, str, None]) -> Optional[Path]:
    if resume_parsed is None:
        return None
    if resume_parsed is True:
        return find_latest_checkpoint_dir(output_dir)
    p = Path(str(resume_parsed)).expanduser().resolve()
    return p if p.is_dir() else None

def load_processor_for_run(args: argparse.Namespace, output_dir: Path, new_chars: List[str], resume_ckpt_dir: Optional[Path]) -> Wav2Vec2Processor:
    if resume_ckpt_dir is not None:
        try:
            processor = Wav2Vec2Processor.from_pretrained(str(resume_ckpt_dir))
        except Exception as e:
            logger.warning('resume: не удалось загрузить Wav2Vec2Processor из %s (%s). Поведение как при новом запуске — возможен MISMATCH lm_head при загрузке весов.', resume_ckpt_dir, e)
            processor = Wav2Vec2Processor.from_pretrained(args.model_name_or_path)
            return maybe_extend_vocab(processor, new_chars)
        vocab = processor.tokenizer.get_vocab()
        missing_chars = [c for c in new_chars if c not in vocab]
        if missing_chars:
            raise ValueError(f'resume_from_checkpoint: в train/test есть символы, которых нет в tokenizer из чекпоинта ({len(missing_chars)} шт., примеры: {missing_chars[:25]!r}). Тогда vocab_size не совпадёт с сохранённым lm_head. Уберите --resume_from_checkpoint или приведите данные к тому же алфавиту, что при сохранении чекпоинта.')
        logger.info('Processor загружен из чекпоинта %s (len(tokenizer)=%s) — размер головы совпадает с весами.', resume_ckpt_dir, len(processor.tokenizer))
        return processor
    processor = Wav2Vec2Processor.from_pretrained(args.model_name_or_path)
    return maybe_extend_vocab(processor, new_chars)

def wav_stem_to_txt_stem(wav_stem: str) -> str:
    if wav_stem.startswith('audio'):
        return 'text' + wav_stem[5:]
    return wav_stem.replace('audio', 'text', 1)

def load_split(root: Path) -> Dataset:
    audios_dir = root / 'audios'
    texts_dir = root / 'texts'
    if not audios_dir.is_dir() or not texts_dir.is_dir():
        raise FileNotFoundError(f'Ожидаются каталоги {audios_dir} и {texts_dir}')
    paths: List[str] = []
    texts: List[str] = []
    for wav in sorted(audios_dir.glob('*.wav')):
        tstem = wav_stem_to_txt_stem(wav.stem)
        txt = texts_dir / f'{tstem}.txt'
        if not txt.is_file():
            logger.warning('Нет текста для %s (ожидался %s)', wav.name, txt)
            continue
        text = txt.read_text(encoding='utf-8', errors='replace').strip()
        if not text:
            continue
        paths.append(str(wav.resolve()))
        texts.append(text)
    if not paths:
        raise RuntimeError(f'В {root} не найдено ни одной парой wav+txt.')
    return Dataset.from_dict({'path': paths, 'text': texts})

def extract_all_chars(ds: Dataset) -> List[str]:
    vocab = set()
    for t in ds['text']:
        vocab.update(t)
    return sorted(vocab)

def min_wav_samples_for_positive_ctc_frames(model: Wav2Vec2ForCTC) -> int:
    w2v = model.wav2vec2
    cfg = model.config
    mask_time_prob = float(getattr(cfg, 'mask_time_prob', 0.0))
    mask_time_len = int(getattr(cfg, 'mask_time_length', 10))
    min_frames_after_cnn = 1
    if mask_time_prob > 0:
        min_frames_after_cnn = max(min_frames_after_cnn, mask_time_len + 1)

    def frame_count(n: int) -> int:
        t = w2v._get_feat_extract_output_lengths(torch.tensor([n], dtype=torch.long))
        return int(t.item())
    lo, hi = (1, 480000)
    while lo < hi:
        mid = (lo + hi) // 2
        if frame_count(mid) >= min_frames_after_cnn:
            hi = mid
        else:
            lo = mid + 1
    return lo

def load_audio_mono(path: str, sr: int) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path, sr=sr, mono=True)
    if y.size > 0:
        return y
    import soundfile as sf
    data, file_sr = sf.read(path, dtype='float32', always_2d=False)
    if data.size == 0:
        return y
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    n_native = int(data.shape[0])
    if file_sr != sr:
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr)
    logger.debug('load_audio_mono: soundfile прочитал %d сэмплов @ %d Hz после пустого librosa → %d @ %d Hz: %s', n_native, file_sr, len(data), sr, path)
    return data

def prepare_asr_batch(batch: Dict[str, List], feature_extractor, processor: Wav2Vec2Processor, min_audio_samples: int) -> Dict[str, List]:
    audio_arrays = []
    target_sr = int(feature_extractor.sampling_rate)
    for p in batch['path']:
        y = load_audio_mono(p, target_sr)
        if y.size == 0:
            logger.warning('Пустой wav после librosa и soundfile: %s — подставляем тишину min_audio_samples=%d', p, min_audio_samples)
            y = np.zeros(min_audio_samples, dtype=np.float32)
        elif len(y) < min_audio_samples:
            ms = 1000.0 * len(y) / target_sr
            logger.debug('Короткое аудио ~%.1f ms (%d сэмплов), дополняем нулями до %d: %s', ms, len(y), min_audio_samples, p)
            y = np.pad(y, (0, min_audio_samples - len(y)), mode='constant')
        audio_arrays.append(y)
    labels = processor.tokenizer(text=batch['text'], return_attention_mask=False).input_ids
    lengths = [int(len(y)) for y in audio_arrays]
    return {'input_values': audio_arrays, 'labels': labels, 'length': lengths}

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_values': f['input_values']} for f in features]
        label_feats = [{'input_ids': f['labels']} for f in features]
        batch = self.processor.pad(input_features, padding=self.padding, return_tensors='pt')
        labels_batch = self.processor.tokenizer.pad(label_feats, padding=self.padding, return_tensors='pt')
        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch['labels'] = labels
        return batch

def normalize_text_strings(pred_str: List[str], label_str: List[str]) -> Tuple[List[str], List[str]]:
    import jiwer
    xf = jiwer.Compose([jiwer.RemoveMultipleSpaces(), jiwer.Strip()])
    pred_str = [xf(s) for s in pred_str]
    label_str = [xf(s) for s in label_str]
    return (pred_str, label_str)

def wer_cer_from_strings(pred_str: List[str], label_str: List[str]) -> Tuple[float, float]:
    import jiwer
    ps, ls = normalize_text_strings(pred_str, label_str)
    try:
        wer = float(jiwer.wer(ls, ps))
        cer = float(jiwer.cer(ls, ps))
    except ZeroDivisionError:
        wer, cer = (1.0, 1.0)
    return (wer, cer)

def build_compute_metrics(processor: Wav2Vec2Processor):

    def compute_metrics(pred) -> Dict[str, float]:
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        wer, cer = wer_cer_from_strings(pred_str, label_str)
        return {'wer': wer, 'cer': cer}
    return compute_metrics

def maybe_extend_vocab(processor: Wav2Vec2Processor, new_chars: List[str]) -> Wav2Vec2Processor:
    tok: Wav2Vec2CTCTokenizer = processor.tokenizer
    existing = set(tok.get_vocab().keys())
    added = [c for c in new_chars if c not in existing]
    if not added:
        return processor
    new_tokens = sorted(set(added))
    tok.add_tokens(new_tokens)
    fe = processor.feature_extractor
    return Wav2Vec2Processor(feature_extractor=fe, tokenizer=tok)

def labels_aligned_for_pyctc(processor: Wav2Vec2Processor) -> List[str]:
    tok = processor.tokenizer
    n = int(getattr(tok, 'vocab_size', len(tok.get_vocab())))
    out: List[str] = []
    for i in range(n):
        t = tok.convert_ids_to_tokens(i)
        out.append('' if t is None else str(t))
    return out

def save_training_plots(log_history: List[Dict[str, Any]], plots_dir: Path) -> List[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: List[str] = []

    def xy(key: str) -> Tuple[List[int], List[float]]:
        steps, ys = ([], [])
        for row in log_history:
            if key in row:
                ys.append(row[key])
                steps.append(row.get('step', len(steps)))
        return (steps, ys)
    plots = [('loss/train', 'Training loss', ('loss', 'train_loss')), ('eval_loss', 'Eval loss', ('eval_loss',)), ('eval_wer', 'Eval WER (greedy)', ('eval_wer',)), ('eval_cer', 'Eval CER (greedy)', ('eval_cer',))]
    for fname, title, keys in plots:
        fig, ax = plt.subplots(figsize=(7, 4))
        plotted = False
        for k in keys:
            st, ys = xy(k)
            if ys:
                ax.plot(st, ys, label=k)
                plotted = True
        if plotted:
            ax.set_title(title)
            ax.set_xlabel('step')
            ax.legend()
            ax.grid(True, alpha=0.3)
            p = plots_dir / f"{fname.replace('/', '_')}.png"
            fig.tight_layout()
            fig.savefig(p, dpi=150)
            plt.close(fig)
            saved.append(str(p))
        else:
            plt.close(fig)
    return saved

def _trainable_param_grad_norm_l2(model: torch.nn.Module) -> Tuple[float, int, int]:
    total_sq = 0.0
    n_with_grad = 0
    n_trainable = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n_trainable += 1
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        total_sq += float(torch.sum(g * g).item())
        n_with_grad += 1
    return (total_sq ** 0.5 if total_sq > 0 else 0.0, n_with_grad, n_trainable)

def debug_log_trainable_gradients(trainer: Trainer) -> None:
    args = trainer.args
    model = trainer.model
    model.train()
    batch = next(iter(trainer.get_train_dataloader()))
    batch = trainer._prepare_inputs(batch)
    model.zero_grad(set_to_none=True)
    if torch.cuda.is_available() and bool(getattr(args, 'bf16', False)):
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    elif torch.cuda.is_available() and bool(getattr(args, 'fp16', False)):
        amp_ctx = torch.autocast(device_type='cuda', dtype=torch.float16)
    else:
        amp_ctx = nullcontext()
    with amp_ctx:
        out = model(**batch)
    loss = getattr(out, 'loss', None)
    if loss is None:
        loss = out['loss']
    loss.backward()
    raw_norm, n_grad, n_train = _trainable_param_grad_norm_l2(model)
    trainable_with_none = n_train - n_grad
    clip_cap = float(getattr(args, 'max_grad_norm', 1.0))
    train_params = [p for p in model.parameters() if p.requires_grad]
    if train_params and clip_cap > 0:
        clipped_t = torch.nn.utils.clip_grad_norm_(train_params, clip_cap)
        clipped_v = float(clipped_t.item())
    elif train_params:
        clipped_t = torch.nn.utils.clip_grad_norm_(train_params, float('inf'))
        clipped_v = float(clipped_t.item())
    else:
        clipped_v = 0.0
    logger.info('debug_trainable_grads: loss=%s | grad_norm_trainable (до clip)=%.6f | норма после clip_grad_norm_(max=%s)=%.6f | trainable tensor-params=%d с ненулевым grad=%d, без grad=%d', loss.detach().float().item(), raw_norm, clip_cap if clip_cap > 0 else 'inf (clip откл.)', clipped_v, n_train, n_grad, trainable_with_none)
    if raw_norm == 0.0 and n_grad == 0:
        logger.error('debug_trainable_grads: все обучаемые параметры без градиента после backward — проверьте заморозку слоёв и loss.')
    elif raw_norm == 0.0:
        logger.warning('debug_trainable_grads: суммарная норма 0 при частичных grad — нетипично, стоит проверить FP16/overflow.')
    model.zero_grad(set_to_none=True)

def configure_trainable_parameters(model: Wav2Vec2ForCTC, args: argparse.Namespace) -> None:
    if args.train_adapters_only:
        if getattr(model.config, 'adapter_attn_dim', None) is None:
            raise ValueError('--train_adapters_only: в конфиге нет adapter_attn_dim. Используйте facebook/mms-1b-all (или mms-1b-l1107 / mms-1b-fl102), не facebook/mms-1b.')
        model.init_adapter_layers()
        model.freeze_base_model()
        for p in model._get_adapters().values():
            p.requires_grad = True
        n_train = sum((p.numel() for p in model.parameters() if p.requires_grad))
        logger.info('train_adapters_only: обучаются только adapters + lm_head (%s параметров с requires_grad=True).', n_train)
        lh = sum((p.numel() for p in model.lm_head.parameters() if p.requires_grad))
        logger.info('train_adapters_only: в lm_head обучаемых элементов: %s (если 0 — голову заморозили по ошибке).', lh)
        if args.freeze_feature_encoder:
            logger.info('--freeze_feature_encoder не нужен: база уже заморожена (train_adapters_only).')
        return
    if args.freeze_feature_encoder:
        model.freeze_feature_encoder()

def model_run_info(model: torch.nn.Module) -> Dict[str, Any]:
    cfg = getattr(model, 'config', None)
    n_params = sum((p.numel() for p in model.parameters()))
    meta = {'num_parameters_total': int(n_params), 'num_parameters_trainable': int(sum((p.numel() for p in model.parameters() if p.requires_grad))), 'torch_version': torch.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda_device_count': torch.cuda.device_count()}
    if torch.cuda.is_available():
        meta['cuda_device_name'] = torch.cuda.get_device_name(0)
    if cfg is not None and hasattr(cfg, 'to_dict'):
        meta['config'] = cfg.to_dict()
    return meta

def predict_logits_and_refs(trainer: Trainer, dataset: Dataset, processor: Wav2Vec2Processor) -> Tuple[np.ndarray, np.ndarray]:
    prev = trainer.compute_metrics
    trainer.compute_metrics = None
    try:
        po = trainer.predict(dataset)
    finally:
        trainer.compute_metrics = prev
    logits = po.predictions
    label_ids = po.label_ids
    if logits is None:
        raise RuntimeError('Trainer.predict вернул пустые predictions')
    label_ids_fixed = np.array(label_ids).copy()
    label_ids_fixed[label_ids_fixed == -100] = processor.tokenizer.pad_token_id
    return (logits, label_ids_fixed)

def decode_greedy_from_logits(logits: np.ndarray, processor: Wav2Vec2Processor) -> List[str]:
    pred_ids = np.argmax(logits, axis=-1)
    return processor.batch_decode(pred_ids)

def decode_refs_from_labels(label_row: np.ndarray, processor: Wav2Vec2Processor) -> str:
    return processor.batch_decode(np.expand_dims(label_row, 0), group_tokens=False)[0]

def _numpy_log_softmax(x: np.ndarray, axis: int=-1) -> np.ndarray:
    xm = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - xm)
    return x - xm - np.log(np.sum(ex, axis=axis, keepdims=True) + 1e-30)

def decode_beam_batch(logits: np.ndarray, decoder: Any, beam_width: int) -> List[str]:
    out: List[str] = []
    for i in range(logits.shape[0]):
        lp = _numpy_log_softmax(logits[i].astype(np.float64, copy=False), axis=-1)
        txt = decoder.decode(lp, beam_width=beam_width)
        out.append(txt)
    return out

def greedy_metrics_from_predictions(logits: np.ndarray, labels: np.ndarray, processor: Wav2Vec2Processor) -> Tuple[float, float]:
    pred_str = decode_greedy_from_logits(logits, processor)
    label_str = [decode_refs_from_labels(labels[i], processor) for i in range(labels.shape[0])]
    return wer_cer_from_strings(pred_str, label_str)

def beam_metrics_from_predictions(logits: np.ndarray, labels: np.ndarray, processor: Wav2Vec2Processor, decoder: Any, beam_width: int) -> Tuple[float, float]:
    pred_str = decode_beam_batch(logits, decoder, beam_width)
    label_str = [decode_refs_from_labels(labels[i], processor) for i in range(labels.shape[0])]
    return wer_cer_from_strings(pred_str, label_str)

def export_metrics_excel(path: Path, greedy: Dict[str, Tuple[float, float]], beam: Dict[str, Tuple[float, float]], meta: Dict[str, Any]) -> None:
    import pandas as pd
    rows = []
    for split in ('train', 'test'):
        g = greedy.get(split, (np.nan, np.nan))
        b = beam.get(split, (np.nan, np.nan))
        rows.append({'split': split, 'greedy_WER': g[0], 'greedy_CER': g[1], 'beam_WER': b[0], 'beam_CER': b[1]})
    df = pd.DataFrame(rows)
    tr_g = greedy.get('train', (np.nan, np.nan))
    ts_g = greedy.get('test', (np.nan, np.nan))
    tr_b = beam.get('train', (np.nan, np.nan))
    ts_b = beam.get('test', (np.nan, np.nan))
    overview = pd.DataFrame([{'train_greedy_WER': tr_g[0], 'train_greedy_CER': tr_g[1], 'test_greedy_WER': ts_g[0], 'test_greedy_CER': ts_g[1], 'train_beam_WER': tr_b[0], 'train_beam_CER': tr_b[1], 'test_beam_WER': ts_b[0], 'test_beam_CER': ts_b[1], 'beam_width_final_eval': meta.get('beam_width_used')}])
    meta_rows = []
    for k, v in sorted(meta.items()):
        if isinstance(v, (dict, list)):
            meta_rows.append({'key': k, 'value': json.dumps(v, ensure_ascii=False, default=str)[:32700]})
        else:
            meta_rows.append({'key': k, 'value': str(v)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine='openpyxl') as w:
        df.to_excel(w, sheet_name='by_split_greedy_beam', index=False)
        overview.to_excel(w, sheet_name='overview_single_row', index=False)
        pd.DataFrame(meta_rows).to_excel(w, sheet_name='run_meta_short', index=False)

def remove_intermediate_checkpoints(output_dir: Path) -> List[str]:
    removed: List[str] = []
    for p in sorted(output_dir.glob('checkpoint-*')):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p.name)
    return removed

def run_training(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_root = Path(args.train_dir)
    test_root = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    artifact_dir = output_dir / 'run_artifacts'
    plots_dir = artifact_dir / 'plots'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_train = load_split(train_root)
    raw_test = load_split(test_root)
    if args.max_train_samples:
        raw_train = raw_train.select(range(min(args.max_train_samples, len(raw_train))))
    logger.info('Train samples: %d | Test samples: %d', len(raw_train), len(raw_test))
    combined_for_vocab = Dataset.from_dict({'text': list(raw_train['text']) + list(raw_test['text'])})
    new_chars = extract_all_chars(combined_for_vocab)
    resume_ckpt = parse_resume_from_checkpoint_arg(args.resume_from_checkpoint)
    resume_ckpt_dir = resolve_resume_checkpoint_dir(output_dir, resume_ckpt)
    if resume_ckpt is True and resume_ckpt_dir is None:
        logger.warning('resume_from_checkpoint=last, но в %s нет каталога checkpoint-* — processor строится с нуля.', output_dir)
    processor = load_processor_for_run(args, output_dir, new_chars, resume_ckpt_dir)
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name_or_path, attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, mask_time_prob=0.05, layerdrop=0.0, ctc_loss_reduction='mean', pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer), ignore_mismatched_sizes=True)
    configure_trainable_parameters(model, args)
    model.gradient_checkpointing_enable()
    min_audio_samples = min_wav_samples_for_positive_ctc_frames(model)
    _sr = int(processor.feature_extractor.sampling_rate)
    logger.info('Минимальная длина WAV при загрузке: %d сэмплов (~%.0f ms @ %d Hz) — нужно для CTC и для SpecAugment (mask по времени), иначе возможны input_lengths < 0 или ValueError mask_length vs sequence_length.', min_audio_samples, 1000.0 * min_audio_samples / _sr, _sr)
    _map_kw = dict(batched=True, batch_size=8, fn_kwargs={'feature_extractor': processor.feature_extractor, 'processor': processor, 'min_audio_samples': min_audio_samples})
    train_ds = raw_train.map(prepare_asr_batch, remove_columns=raw_train.column_names, **_map_kw)
    test_ds = raw_test.map(prepare_asr_batch, remove_columns=raw_test.column_names, **_map_kw)
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    report_to: List[str] = []
    if not args.disable_wandb and os.environ.get('WANDB_DISABLED', '').lower() not in ('1', 'true', 'yes'):
        report_to.append('wandb')
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        wandb.config.update({'_notebook_sync_hint': 'После обучения rsync только: ' + str((output_dir / 'final_model').resolve()) + ' и ' + str(artifact_dir.resolve())}, allow_val_change=True)
    _ta_params = inspect.signature(TrainingArguments.__init__).parameters
    _eval_kw = 'eval_strategy' if 'eval_strategy' in _ta_params else 'evaluation_strategy'
    _group_kw: Dict[str, Any] = {}
    if 'train_sampling_strategy' in _ta_params:
        _group_kw['train_sampling_strategy'] = 'group_by_length'
    elif 'group_by_length' in _ta_params:
        _group_kw['group_by_length'] = True
    save_limit: Optional[int] = None if args.save_total_limit in (-1, 0) else int(args.save_total_limit)
    _cuda_ok = torch.cuda.is_available()
    _bf16_supported = bool(_cuda_ok and torch.cuda.is_bf16_supported())
    pref = getattr(args, 'precision', None) or 'fp16'
    if pref == 'fp32':
        _use_bf16, _use_fp16 = (False, False)
    elif pref == 'bf16':
        if not _bf16_supported:
            raise RuntimeError('--precision bf16 указан, но CUDA bf16 не поддерживается на этой машине.')
        _use_bf16, _use_fp16 = (True, False)
    elif pref == 'auto':
        _use_bf16 = _bf16_supported
        _use_fp16 = bool(_cuda_ok and (not _use_bf16))
    else:
        _use_bf16 = False
        _use_fp16 = bool(_cuda_ok)
    if _cuda_ok:
        logger.info('AMP (--precision=%s): bf16=%s fp16=%s', pref, _use_bf16, _use_fp16)
    training_args = TrainingArguments(output_dir=str(output_dir), **_group_kw, per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, **{_eval_kw: 'steps'}, save_strategy='steps', num_train_epochs=args.num_train_epochs, bf16=_use_bf16, fp16=_use_fp16, save_steps=args.save_steps, eval_steps=args.eval_steps, logging_steps=args.logging_steps, learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio, save_total_limit=save_limit, load_best_model_at_end=True, metric_for_best_model='eval_cer', greater_is_better=False, report_to=report_to, push_to_hub=False, gradient_checkpointing=True, seed=args.seed, max_grad_norm=1.0)
    compute_metrics = build_compute_metrics(processor)
    trainer = Trainer(model=model, data_collator=data_collator, args=training_args, train_dataset=train_ds, eval_dataset=test_ds, processing_class=processor, compute_metrics=compute_metrics)
    if resume_ckpt is not None:
        if resume_ckpt is True:
            logger.info('resume_from_checkpoint: берём последний checkpoint-* в %s', output_dir.resolve())
        else:
            logger.info('resume_from_checkpoint: %s', resume_ckpt)
    if args.debug_trainable_grads:
        debug_log_trainable_gradients(trainer)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    plots_saved = []
    try:
        plots_saved = save_training_plots(trainer.state.log_history, plots_dir)
    except Exception as e:
        logger.warning('Не удалось построить графики обучения: %s', e)
    try:
        (artifact_dir / 'trainer_log_history.json').write_text(json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    except Exception as e:
        logger.warning('Не сохранён trainer_log_history.json: %s', e)
    logger.info('Финальная оценка на train/test (greedy decoding, через Trainer.evaluate).')
    train_eval = trainer.evaluate(train_ds, metric_key_prefix='greedy_train')
    test_eval = trainer.evaluate(test_ds, metric_key_prefix='greedy_test')
    for name, blob in (('train_greedy', train_eval), ('test_greedy', test_eval)):
        logger.info('%s: %s', name, blob)
    if report_to:
        import wandb
        wandb.log({**train_eval, **test_eval})
    best_ckpt = getattr(trainer.state, 'best_model_checkpoint', None) or ''
    trainer.save_model(str(output_dir / 'final_model'))
    processor.save_pretrained(str(output_dir / 'final_model'))
    run_meta = {'train_adapters_only': bool(args.train_adapters_only), 'best_model_checkpoint_path_before_cleanup': best_ckpt, 'beam_width_used': getattr(args, 'beam_width', 10), 'metric_for_best_model': 'eval_cer', 'best_metric_during_training': getattr(trainer.state, 'best_metric', None), 'trainer_global_steps': getattr(trainer.state, 'global_step', None), 'train_samples': len(raw_train), 'test_samples': len(raw_test), **model_run_info(trainer.model), 'plots_saved': plots_saved, 'final_model_dir': str((output_dir / 'final_model').resolve()), 'artifacts_dir': str(artifact_dir.resolve()), 'evaluation_note': 'Во время training/eval Trainer использует greedy (argmax по времени для CTC). После обучения — финальная оценка greedy на train/test, затем beam (pyctcdecode) для той же (лучшей по минимальному eval_cer) модели при load_best_model_at_end=True.'}
    greedy_pack = {'train': (train_eval.get('greedy_train_wer', train_eval.get('eval_wer', float('nan'))), train_eval.get('greedy_train_cer', train_eval.get('eval_cer', float('nan')))), 'test': (test_eval.get('greedy_test_wer', test_eval.get('eval_wer', float('nan'))), test_eval.get('greedy_test_cer', test_eval.get('eval_cer', float('nan'))))}
    beam_pack: Dict[str, Tuple[float, float]] = {}
    beam_details: Dict[str, Any] = {}
    if not args.skip_beam_eval:
        try:
            from pyctcdecode import build_ctcdecoder
            vocab_labels = labels_aligned_for_pyctc(processor)
            logits_dim = getattr(trainer.model.config, 'vocab_size', None)
            if logits_dim is not None and int(logits_dim) != len(vocab_labels):
                logger.warning('vocab_size=%s ≠ len(pyctc labels)=%s', logits_dim, len(vocab_labels))
            decoder = build_ctcdecoder(vocab_labels)
            trainer.model.eval()
            bw = getattr(args, 'beam_width', 10)
            for split_key, ds in (('train', train_ds), ('test', test_ds)):
                logits, lids = predict_logits_and_refs(trainer, ds, processor)
                logits = logits.astype(np.float32, copy=False)
                wer_b, cer_b = beam_metrics_from_predictions(logits, lids, processor, decoder, bw)
                beam_pack[split_key] = (wer_b, cer_b)
                beam_details[f'beam_{split_key}'] = {'wer': wer_b, 'cer': cer_b, 'beam_width': bw}
                logger.info('Beam %s: WER=%.6f CER=%.6f (beam_width=%s)', split_key, wer_b, cer_b, bw)
            if report_to:
                import wandb
                flat = {}
                for k, (w, c) in beam_pack.items():
                    flat[f'beam_{k}_wer'] = w
                    flat[f'beam_{k}_cer'] = c
                wandb.log(flat)
        except Exception as e:
            logger.exception('Beam search оценка не выполнена: %s', e)
            beam_details['beam_error'] = repr(e)
    checkpoint_dirs_removed: List[str] = []
    if args.keep_intermediate_checkpoints:
        logger.info('Сохранены все оставшиеся checkpoint-* (--keep_intermediate_checkpoints); на ноутбук копируйте только final_model/ и run_artifacts/ если не нужна возобновляемость.')
    else:
        checkpoint_dirs_removed = remove_intermediate_checkpoints(output_dir)
        if checkpoint_dirs_removed:
            logger.info('Удалены промежуточные каталоги (экономия места под rsync): %s', checkpoint_dirs_removed)
        else:
            logger.info('Промежуточных checkpoint-* не найдено (уже ограничено save_total_limit или не сохранялись).')
    run_meta['checkpoint_dirs_removed_after_train'] = checkpoint_dirs_removed
    run_meta['kept_intermediate_checkpoints_flag'] = bool(args.keep_intermediate_checkpoints)
    excel_path = artifact_dir / 'best_model_wer_cer_greedy_and_beam.xlsx'
    try:
        export_metrics_excel(excel_path, greedy={k: greedy_pack[k] for k in greedy_pack}, beam=beam_pack, meta=run_meta)
    except Exception as e:
        logger.warning('Excel не сохранён: %s', e)
    metrics_json = artifact_dir / 'metrics_summary.json'
    summary = {'greedy': {'train': {'wer': greedy_pack['train'][0], 'cer': greedy_pack['train'][1]}, 'test': {'wer': greedy_pack['test'][0], 'cer': greedy_pack['test'][1]}}, 'beam': {split: {'wer': beam_pack.get(split, (np.nan, np.nan))[0], 'cer': beam_pack.get(split, (np.nan, np.nan))[1]} for split in ('train', 'test')}, 'meta': run_meta, 'beam_extra': beam_details}
    metrics_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    (artifact_dir / 'README_ARTIFACTS.txt').write_text('\n'.join(['Что снять на ноутбук (достаточно двух директорий):', '  • final_model/ — веса + processor лучшей модели по eval_cer (load_best_model_at_end).', '  • run_artifacts/ — графики, Excel WER/CER greedy+beam, metrics_summary.json, лог истории Trainer.', '', 'По умолчанию после обучения удаляются checkpoint-* (нет лишних гигабайт под rsync).', '  Держать их на сервере: --keep_intermediate_checkpoints', '', 'run_artifacts/best_model_wer_cer_greedy_and_beam.xlsx — листы by_split_greedy_beam и overview_single_row.', '', 'W&B: wandb login или export WANDB_API_KEY | отключить: --disable_wandb.', '']), encoding='utf-8')
    logger.info('Готово. Лучшая модель: %s | отчёты: %s', output_dir / 'final_model', artifact_dir)
    if report_to:
        try:
            import wandb
            sum_payload: Dict[str, Any] = {'metric_for_best_model': 'eval_cer', 'best_metric_during_training': getattr(trainer.state, 'best_metric', None), 'greedy_train_wer': greedy_pack['train'][0], 'greedy_train_cer': greedy_pack['train'][1], 'greedy_test_wer': greedy_pack['test'][0], 'greedy_test_cer': greedy_pack['test'][1], 'notebook_final_model_dir': str((output_dir / 'final_model').resolve()), 'notebook_artifacts_dir': str(artifact_dir.resolve()), 'notebook_excel_best_metrics': str(excel_path.resolve())}
            for split in ('train', 'test'):
                w_b, c_b = beam_pack.get(split, (None, None))
                sum_payload[f'beam_{split}_wer'] = w_b
                sum_payload[f'beam_{split}_cer'] = c_b
            wandb.summary.update(sum_payload)
            wandb.finish()
        except Exception:
            logger.exception('W&B finish/summary завершился с ошибкой.')

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run_training(parse_args())
if __name__ == '__main__':
    main()
