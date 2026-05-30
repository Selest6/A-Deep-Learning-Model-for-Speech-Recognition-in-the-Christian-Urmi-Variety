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
import argparse
import inspect
import json
import logging
import os
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from datasets import Dataset
from transformers import SeamlessM4TFeatureExtractor, Trainer, TrainingArguments, Wav2Vec2BertForCTC, Wav2Vec2BertProcessor, Wav2Vec2CTCTokenizer
from train_mms_urmi_full import RESUME_FROM_BEST_SENTINEL, beam_metrics_from_predictions, build_compute_metrics, debug_log_trainable_gradients, export_metrics_excel, extract_all_chars, find_best_checkpoint_dir, labels_aligned_for_pyctc, load_audio_mono, load_split, model_run_info, parse_resume_from_checkpoint_arg, predict_logits_and_refs, remove_intermediate_checkpoints, resolve_resume_checkpoint_dir, save_training_plots
logger = logging.getLogger(__name__)
RESUME_FROM_BEST_SENTINEL = object()

def find_best_checkpoint_dir(output_dir: Path) -> Optional[Path]:
    latest = find_latest_checkpoint_dir(output_dir)
    if latest is None:
        return None
    ts_path = latest / 'trainer_state.json'
    if not ts_path.is_file():
        return None
    text = ts_path.read_text(encoding='utf-8')
    text = re.sub(':\\s*NaN\\b', ': null', text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    best = data.get('best_model_checkpoint')
    if best:
        p = Path(str(best)).expanduser().resolve()
        if p.is_dir():
            return p
    best_cer = None
    best_step_dir: Optional[Path] = None
    for ck in output_dir.glob('checkpoint-*'):
        if not ck.is_dir():
            continue
        tsp = ck / 'trainer_state.json'
        if not tsp.is_file():
            continue
        try:
            ttxt = re.sub(':\\s*NaN\\b', ': null', tsp.read_text(encoding='utf-8'))
            td = json.loads(ttxt)
        except (OSError, json.JSONDecodeError):
            continue
        for row in reversed(td.get('log_history') or []):
            if 'eval_cer' in row:
                cer = float(row['eval_cer'])
                if best_cer is None or cer < best_cer:
                    best_cer = cer
                    best_step_dir = ck
                break
    return best_step_dir

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

def preprocess_ctc_logits_for_metrics(logits: Any, labels: Any) -> torch.Tensor:
    while isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.argmax(dim=-1)

def build_compute_metrics(processor: Wav2Vec2Processor):

    def compute_metrics(pred) -> Dict[str, float]:
        raw = pred.predictions
        if raw.ndim >= 3:
            pred_ids = np.asarray(np.argmax(raw, axis=-1))
        else:
            pred_ids = np.asarray(raw)
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

def predict_logits_and_refs(trainer: Trainer, dataset: Dataset, processor: Wav2Vec2Processor, *, need_full_logits: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    prev = trainer.compute_metrics
    prev_prep = getattr(trainer, 'preprocess_logits_for_metrics', None)
    trainer.compute_metrics = None
    if need_full_logits:
        trainer.preprocess_logits_for_metrics = None
    try:
        po = trainer.predict(dataset)
    finally:
        trainer.compute_metrics = prev
        trainer.preprocess_logits_for_metrics = prev_prep
    logits = po.predictions
    label_ids = po.label_ids
    if logits is None:
        raise RuntimeError('Trainer.predict вернул пустые predictions')
    label_ids_fixed = np.array(label_ids).copy()
    label_ids_fixed[label_ids_fixed == -100] = processor.tokenizer.pad_token_id
    return (logits, label_ids_fixed)

def predict_logits_and_refs_chunked(trainer: Trainer, dataset: Dataset, processor: Wav2Vec2Processor, chunk_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(dataset)
    cs = int(chunk_samples)
    if cs <= 0:
        return predict_logits_and_refs(trainer, dataset, processor, need_full_logits=True)
    cs = max(1, cs)
    if cs >= n:
        return predict_logits_and_refs(trainer, dataset, processor, need_full_logits=True)
    logits_parts: List[np.ndarray] = []
    labs_parts: List[np.ndarray] = []
    for start in range(0, n, cs):
        sub = dataset.select(range(start, min(start + cs, n)))
        lg, lids = predict_logits_and_refs(trainer, sub, processor, need_full_logits=True)
        logits_parts.append(lg)
        labs_parts.append(lids)
    return (np.concatenate(logits_parts, axis=0), np.concatenate(labs_parts, axis=0))

def decode_greedy_from_logits(logits: np.ndarray, processor: Wav2Vec2Processor) -> List[str]:
    pred_ids = np.argmax(logits, axis=-1)
    return processor.batch_decode(pred_ids)

def decode_refs_from_labels(label_row: np.ndarray, processor: Wav2Vec2Processor) -> str:
    return processor.batch_decode(np.expand_dims(label_row, 0), group_tokens=False)[0]

def _numpy_log_softmax(x: np.ndarray, axis: int=-1) -> np.ndarray:
    xm = np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x - xm)
    return x - xm - np.log(np.sum(ex, axis=axis, keepdims=True) + 1e-30)
SURROGATE_CTC_LABEL_BASE = 44032

def surrogate_ctc_label_strings(vocab_size: int) -> List[str]:
    if vocab_size <= 0:
        return []
    labels = ['']
    for i in range(1, vocab_size):
        labels.append(chr(SURROGATE_CTC_LABEL_BASE + i - 1))
    last_chr_ord = SURROGATE_CTC_LABEL_BASE + vocab_size - 2
    if last_chr_ord > 55203:
        raise ValueError(f'vocab_size={vocab_size} не помещается в суррогатный блок символов для pyctcdecode.')
    return labels

def build_pyctcdecoder_for_wav2vec_logits(vocab_size: int) -> Any:
    from pyctcdecode import build_ctcdecoder
    return build_ctcdecoder(surrogate_ctc_label_strings(vocab_size))

def decode_beam_batch(logits: np.ndarray, decoder: Any, beam_width: int) -> List[str]:
    out: List[str] = []
    for i in range(logits.shape[0]):
        lp = _numpy_log_softmax(logits[i].astype(np.float64, copy=False), axis=-1)
        txt = decoder.decode(lp, beam_width=beam_width)
        out.append(txt)
    return out

def decode_beam_batch_wav2vec2(logits: np.ndarray, decoder: Any, beam_width: int, processor: Wav2Vec2Processor) -> List[str]:
    n_vocab = int(logits.shape[-1])
    base = SURROGATE_CTC_LABEL_BASE
    out: List[str] = []
    for i in range(logits.shape[0]):
        lp = _numpy_log_softmax(logits[i].astype(np.float64, copy=False), axis=-1)
        sur = decoder.decode(lp, beam_width=beam_width)
        ids: List[int] = []
        for ch in sur:
            if ch == '':
                continue
            o = ord(ch)
            if base <= o <= base + n_vocab - 2:
                ids.append(o - base + 1)
        row = np.asarray(ids, dtype=np.int64)
        txt = processor.batch_decode(np.expand_dims(row, 0), group_tokens=False)[0]
        out.append(txt)
    return out

def greedy_metrics_from_predictions(logits: np.ndarray, labels: np.ndarray, processor: Wav2Vec2Processor) -> Tuple[float, float]:
    pred_str = decode_greedy_from_logits(logits, processor)
    label_str = [decode_refs_from_labels(labels[i], processor) for i in range(labels.shape[0])]
    return wer_cer_from_strings(pred_str, label_str)

def beam_metrics_from_predictions(logits: np.ndarray, labels: np.ndarray, processor: Wav2Vec2Processor, decoder: Any, beam_width: int) -> Tuple[float, float]:
    pred_str = decode_beam_batch_wav2vec2(logits, decoder, beam_width, processor)
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
    model = Wav2Vec2ForCTC.from_pretrained(args.model_name_or_path, attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, mask_time_prob=float(args.mask_time_prob), layerdrop=0.0, ctc_loss_reduction='mean', ctc_zero_infinity=True, pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer), ignore_mismatched_sizes=True)
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
    _ws = int(getattr(args, 'warmup_steps', 0) or 0)
    _warmup_kw: Dict[str, Any]
    if _ws > 0:
        _warmup_kw = {'warmup_steps': _ws}
    else:
        _warmup_kw = {'warmup_ratio': float(args.warmup_ratio)}
    _max_steps = int(getattr(args, 'max_steps', -1) or -1)
    _ta_build: Dict[str, Any] = dict(output_dir=str(output_dir), **_group_kw, per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, **{_eval_kw: 'steps'}, save_strategy='steps', num_train_epochs=args.num_train_epochs, bf16=_use_bf16, fp16=_use_fp16, save_steps=args.save_steps, eval_steps=args.eval_steps, logging_steps=args.logging_steps, learning_rate=args.learning_rate, weight_decay=float(args.weight_decay), save_total_limit=save_limit, load_best_model_at_end=True, metric_for_best_model='eval_cer', greater_is_better=False, report_to=report_to, push_to_hub=False, gradient_checkpointing=True, seed=args.seed, max_grad_norm=1.0)
    _ta_build.update(_warmup_kw)
    if _max_steps > 0:
        _ta_build['max_steps'] = _max_steps
        logger.info('max_steps=%s (глобальный лимит шагов; перекрывает num_train_epochs).', _max_steps)
    if 'lr_scheduler_type' in _ta_params:
        _ta_build['lr_scheduler_type'] = str(args.lr_scheduler_type)
    else:
        logger.warning('TrainingArguments без lr_scheduler_type — используется дефолт transformers.')
    training_args = TrainingArguments(**_ta_build)
    compute_metrics = build_compute_metrics(processor)
    trainer = Trainer(model=model, data_collator=data_collator, args=training_args, train_dataset=train_ds, eval_dataset=test_ds, processing_class=processor, compute_metrics=compute_metrics, preprocess_logits_for_metrics=preprocess_ctc_logits_for_metrics)
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
    run_meta = {'train_adapters_only': bool(args.train_adapters_only), 'learning_rate': float(args.learning_rate), 'lr_scheduler_type': str(args.lr_scheduler_type), 'warmup_steps': int(args.warmup_steps), 'warmup_ratio': float(args.warmup_ratio), 'weight_decay': float(args.weight_decay), 'mask_time_prob': float(args.mask_time_prob), 'best_model_checkpoint_path_before_cleanup': best_ckpt, 'beam_width_used': getattr(args, 'beam_width', 10), 'metric_for_best_model': 'eval_cer', 'best_metric_during_training': getattr(trainer.state, 'best_metric', None), 'trainer_global_steps': getattr(trainer.state, 'global_step', None), 'train_samples': len(raw_train), 'test_samples': len(raw_test), **model_run_info(trainer.model), 'plots_saved': plots_saved, 'final_model_dir': str((output_dir / 'final_model').resolve()), 'artifacts_dir': str(artifact_dir.resolve()), 'evaluation_note': 'Во время training/eval Trainer использует greedy (argmax по времени для CTC). После обучения — финальная оценка greedy на train/test, затем beam (pyctcdecode) для той же (лучшей по минимальному eval_cer) модели при load_best_model_at_end=True.'}
    greedy_pack = {'train': (train_eval.get('greedy_train_wer', train_eval.get('eval_wer', float('nan'))), train_eval.get('greedy_train_cer', train_eval.get('eval_cer', float('nan')))), 'test': (test_eval.get('greedy_test_wer', test_eval.get('eval_wer', float('nan'))), test_eval.get('greedy_test_cer', test_eval.get('eval_cer', float('nan'))))}
    beam_pack: Dict[str, Tuple[float, float]] = {}
    beam_details: Dict[str, Any] = {}
    if not args.skip_beam_eval:
        try:
            logits_dim = int(getattr(trainer.model.config, 'vocab_size', 0))
            if logits_dim <= 0:
                raise RuntimeError('config.vocab_size не задан для beam-декодера.')
            decoder = build_pyctcdecoder_for_wav2vec_logits(logits_dim)
            trainer.model.eval()
            bw = getattr(args, 'beam_width', 10)
            bchunk = getattr(args, 'beam_logits_chunk_samples', 256)
            for split_key, ds in (('train', train_ds), ('test', test_ds)):
                logits, lids = predict_logits_and_refs_chunked(trainer, ds, processor, chunk_samples=bchunk)
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
logger = logging.getLogger(__name__)

def load_encoder_weights_into_ctc(model: Wav2Vec2BertForCTC, pretrained_dir: Path) -> Dict[str, Any]:
    pretrained_dir = pretrained_dir.resolve()
    st_path = pretrained_dir / 'model.safetensors'
    pt_path = pretrained_dir / 'pytorch_model.bin'
    state: Dict[str, torch.Tensor]
    if st_path.is_file():
        from safetensors.torch import load_file
        state = load_file(str(st_path))
    elif pt_path.is_file():
        try:
            state = torch.load(str(pt_path), map_location='cpu', weights_only=True)
        except TypeError:
            state = torch.load(str(pt_path), map_location='cpu')
    else:
        raise FileNotFoundError(f'Нет model.safetensors или pytorch_model.bin в {pretrained_dir}')
    prefix = 'wav2vec2_bert.'
    enc_sd = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    if not enc_sd:
        raise RuntimeError(f'В чекпоинте нет ключей с префиксом {prefix!r}')
    res = model.wav2vec2_bert.load_state_dict(enc_sd, strict=False)
    info = {'pretrained_encoder_dir': str(pretrained_dir), 'weights_loaded_prefix_only': prefix.rstrip('.'), 'encoder_missing_keys_count': len(res.missing_keys), 'encoder_unexpected_keys_count': len(res.unexpected_keys), 'encoder_missing_keys_sample': res.missing_keys[:30], 'encoder_unexpected_keys_sample': res.unexpected_keys[:30]}
    logger.info('Загрузка энкодера из %s: missing=%d unexpected=%d', pretrained_dir, len(res.missing_keys), len(res.unexpected_keys))
    return info

def load_training_budget_from_checkpoint(checkpoint_dir: Path) -> Tuple[Optional[int], Optional[int]]:
    path = checkpoint_dir / 'trainer_state.json'
    if not path.is_file():
        return (None, None)
    text = path.read_text(encoding='utf-8')
    text = re.sub(':\\s*NaN\\b', ': null', text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning('Не разобрать trainer_state.json для бюджета шагов: %s', path)
        return (None, None)
    raw_mx = data.get('max_steps')
    raw_ep = data.get('num_train_epochs')
    max_steps_v: Optional[int] = None
    if raw_mx is not None:
        try:
            imx = int(raw_mx)
            if imx > 0:
                max_steps_v = imx
        except (TypeError, ValueError):
            pass
    epochs_v: Optional[int] = None
    if raw_ep is not None:
        try:
            iep = int(raw_ep)
            if iep > 0:
                epochs_v = iep
        except (TypeError, ValueError):
            pass
    return (max_steps_v, epochs_v)

class TrainerResumeNewHparamsSameStep(Trainer):

    def __init__(self, *args, resume_new_hparams_same_step: bool=False, **kwargs):
        self._resume_new_hparams_same_step = bool(resume_new_hparams_same_step)
        super().__init__(*args, **kwargs)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if self._resume_new_hparams_same_step:
            logger.info('resume_new_hparams_same_step: optimizer/scheduler из чекпоинта не загружаются — используется новый AdamW и расписание из TrainingArguments.')
            return
        return super()._load_optimizer_and_scheduler(checkpoint)

    def _load_scaler(self, checkpoint):
        if self._resume_new_hparams_same_step:
            logger.info('resume_new_hparams_same_step: GradScaler из чекпоинта не загружается (новое fp16-масштабирование).')
            return
        return super()._load_scaler(checkpoint)

    def create_scheduler(self, num_training_steps: int, optimizer: Optional[torch.optim.Optimizer]=None):
        sched = super().create_scheduler(num_training_steps, optimizer)
        if self._resume_new_hparams_same_step and self.lr_scheduler is not None:
            gs = int(getattr(self.state, 'global_step', 0) or 0)
            if gs > 0:
                cap = max(0, int(num_training_steps) - 1)
                ff = gs if gs <= cap else cap
                if ff < gs:
                    logger.warning('global_step=%d больше, чем num_training_steps−1=%d нового расписания; LR scheduler прокручивается только на %d шагов.', gs, cap, ff)
                logger.info('Синхронизация нового LR scheduler: %d быстрых .step() (ориентир: global_step из чекпоинта).', ff)
                try:
                    for _ in range(ff):
                        self.lr_scheduler.step()
                except Exception as exc:
                    logger.warning('Не удалось прокрутить LR scheduler (--resume_new_hparams_same_step): %s', exc)
        return sched

class TrainerResumeNewHparamsWeightOverride(TrainerResumeNewHparamsSameStep):

    def __init__(self, *args, model_weights_override_dir: Optional[Path]=None, **kwargs):
        self._model_weights_override_dir = Path(model_weights_override_dir).resolve() if model_weights_override_dir is not None else None
        super().__init__(*args, **kwargs)

    def _load_from_checkpoint(self, resume_from_checkpoint: str, model=None):
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)
        if self._model_weights_override_dir is not None:
            logger.info('model_weights_override_from: поверх resume загружаются веса из %s', self._model_weights_override_dir)
            super()._load_from_checkpoint(str(self._model_weights_override_dir), model=model)

def build_char_tokenizer(chars: List[str], work_dir: Path) -> Wav2Vec2CTCTokenizer:
    unk_t = '<unk>'
    pad_t = '<pad>'
    vocab: Dict[str, int] = {c: i for i, c in enumerate(sorted(set(chars)))}
    for t in (unk_t, pad_t):
        if t not in vocab:
            vocab[t] = len(vocab)
    if ' ' in vocab:
        delim = ' '
    else:
        delim = '|'
        if delim not in vocab:
            vocab[delim] = len(vocab)
    work_dir.mkdir(parents=True, exist_ok=True)
    vocab_path = work_dir / 'vocab_char.json'
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding='utf-8')
    return Wav2Vec2CTCTokenizer(str(vocab_path), unk_token=unk_t, pad_token=pad_t, word_delimiter_token=delim)

def load_or_build_processor(model_name: str, new_chars: List[str], output_dir: Path, resume_ckpt_dir: Optional[Path]) -> Wav2Vec2BertProcessor:
    feature_extractor = SeamlessM4TFeatureExtractor.from_pretrained(model_name)
    if resume_ckpt_dir is not None:
        try:
            proc = Wav2Vec2BertProcessor.from_pretrained(str(resume_ckpt_dir))
        except Exception as e:
            logger.warning('resume: не удалось загрузить Wav2Vec2BertProcessor из %s (%s)', resume_ckpt_dir, e)
            proc = None
        if proc is not None:
            vocab = proc.tokenizer.get_vocab()
            missing = [c for c in new_chars if c not in vocab]
            if missing:
                raise ValueError(f'resume_from_checkpoint: в данных есть символы, которых нет в tokenizer чекпоинта ({len(missing)} шт., примеры: {missing[:25]!r}).')
            logger.info('Processor из чекпоинта %s (len(tokenizer)=%s).', resume_ckpt_dir, len(proc.tokenizer))
            return proc
    tok_dir = output_dir / 'char_tokenizer_init'
    tokenizer = build_char_tokenizer(new_chars, tok_dir)
    return Wav2Vec2BertProcessor(feature_extractor=feature_extractor, tokenizer=tokenizer)

def min_wav_samples_for_frames(feature_extractor: SeamlessM4TFeatureExtractor, min_mel_frames: int) -> int:
    sr = int(feature_extractor.sampling_rate)
    stft_win = int(len(feature_extractor.window))
    lo, hi = (max(1, stft_win), 480000)

    def n_frames(n: int) -> int:
        if n < stft_win:
            return 0
        y = np.zeros(n, dtype=np.float32)
        out = feature_extractor(y, sampling_rate=sr, return_tensors='np')
        return int(out['input_features'][0].shape[0])
    while lo < hi:
        mid = (lo + hi) // 2
        if n_frames(mid) >= min_mel_frames:
            hi = mid
        else:
            lo = mid + 1
    return lo

def prepare_w2v_bert_batch(batch: Dict[str, List], processor: Wav2Vec2BertProcessor, min_audio_samples: int) -> Dict[str, List]:
    fe = processor.feature_extractor
    sr = int(fe.sampling_rate)
    feats: List[np.ndarray] = []
    lengths: List[int] = []
    for p in batch['path']:
        y = load_audio_mono(p, sr)
        if y.size == 0:
            logger.warning('Пустой wav — тишина min_audio_samples=%d: %s', min_audio_samples, p)
            y = np.zeros(min_audio_samples, dtype=np.float32)
        elif len(y) < min_audio_samples:
            y = np.pad(y, (0, min_audio_samples - len(y)), mode='constant')
        enc = processor(y, sampling_rate=sr, return_tensors='np')
        arr = enc.input_features[0]
        feats.append(arr)
        lengths.append(int(arr.shape[0]))
    labels = processor.tokenizer(text=batch['text'], return_attention_mask=False).input_ids
    return {'input_features': feats, 'labels': labels, 'length': lengths}

@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2BertProcessor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_feats = [{'input_features': f['input_features']} for f in features]
        label_feats = [{'input_ids': f['labels']} for f in features]
        batch = self.processor.pad(input_feats, padding=self.padding, return_tensors='pt')
        labels_batch = self.processor.pad(labels=label_feats, padding=self.padding, return_tensors='pt')
        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch['labels'] = labels
        return batch

def _apply_speed_perturb(y: np.ndarray, rate: float) -> np.ndarray:
    import librosa
    y = np.asarray(y, dtype=np.float32)
    if abs(rate - 1.0) < 1e-05:
        return y
    out = librosa.effects.time_stretch(y, rate=rate)
    if out.size == 0:
        return y
    return out.astype(np.float32, copy=False)

def _apply_gain_db(y: np.ndarray, gain_db: float) -> np.ndarray:
    g = 10.0 ** (gain_db / 20.0)
    return (y * g).astype(np.float32, copy=False)

def _mix_white_noise_snr(y: np.ndarray, snr_db: float, rng_np: np.random.Generator) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    if n == 0:
        return y
    noise = rng_np.standard_normal(n).astype(np.float32)
    p_sig = float(np.mean(y * y) + 1e-12)
    p_noise = float(np.mean(noise * noise) + 1e-12)
    target_noise = p_sig / (10.0 ** (snr_db / 10.0) + 1e-12)
    scale = float(np.sqrt(target_noise / p_noise))
    return (y + scale * noise).astype(np.float32, copy=False)

def add_path_length_for_grouping(batch: Dict[str, List], sr: int) -> Dict[str, List[int]]:
    import soundfile as sf
    lens: List[int] = []
    for p in batch['path']:
        try:
            info = sf.info(p)
            lens.append(max(1, int(info.duration * sr)))
        except Exception:
            y = load_audio_mono(p, sr)
            lens.append(max(1, int(len(y))))
    return {'length': lens}

@dataclass
class DataCollatorW2VBertDynamicTrain:
    processor: Wav2Vec2BertProcessor
    min_audio_samples: int
    augment_mode: str
    base_seed: int = 42
    speed_factors: Tuple[float, ...] = (0.9, 1.1)
    speed_prob: float = 0.5
    noise_prob: float = 0.35
    snr_min_db: float = 10.0
    snr_max_db: float = 20.0
    gain_db_max: float = 3.0
    padding: Union[bool, str] = True
    _call_counter: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.augment_mode not in ('speed', 'noise'):
            raise ValueError(self.augment_mode)

    def _augment_waveform(self, y: np.ndarray, rng_py: random.Random, rng_np: np.random.Generator) -> np.ndarray:
        y = np.asarray(y, dtype=np.float32)
        if self.augment_mode == 'speed':
            if rng_py.random() < self.speed_prob:
                rate = rng_py.choice(self.speed_factors)
                y = _apply_speed_perturb(y, rate)
        elif self.augment_mode == 'noise':
            if self.gain_db_max > 0 and rng_py.random() < 0.5:
                y = _apply_gain_db(y, rng_py.uniform(-self.gain_db_max, self.gain_db_max))
            if rng_py.random() < self.noise_prob:
                snr = rng_py.uniform(self.snr_min_db, self.snr_max_db)
                y = _mix_white_noise_snr(y, snr, rng_np)
        return y

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if features and 'input_features' in features[0] and ('path' not in features[0]):
            return DataCollatorCTCWithPadding(processor=self.processor, padding=self.padding)(features)
        fe = self.processor.feature_extractor
        sr = int(fe.sampling_rate)
        self._call_counter += 1
        seed = int(self.base_seed) * 1000003 + self._call_counter * 97 + len(features) * 13 & 2147483647
        rng_py = random.Random(seed)
        rng_np = np.random.default_rng(seed)
        feats_np: List[np.ndarray] = []
        texts: List[str] = []
        for i, f in enumerate(features):
            path, text = (f['path'], f['text'])
            y = load_audio_mono(str(path), sr)
            if y.size == 0:
                y = np.zeros(self.min_audio_samples, dtype=np.float32)
            sub_seed = seed + i * 1009
            y = self._augment_waveform(y, random.Random(sub_seed), np.random.default_rng(sub_seed))
            if len(y) < self.min_audio_samples:
                y = np.pad(y, (0, self.min_audio_samples - len(y)), mode='constant')
            enc = self.processor(y, sampling_rate=sr, return_tensors='np')
            feats_np.append(enc.input_features[0])
            texts.append(text)
        labels = self.processor.tokenizer(text=texts, return_attention_mask=False).input_ids
        collate_static = DataCollatorCTCWithPadding(processor=self.processor, padding=self.padding)
        pack = [{'input_features': a, 'labels': b} for a, b in zip(feats_np, labels)]
        return collate_static(pack)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Wav2Vec2-BERT CTC — эксперименты с аугментацией.')
    p.add_argument('--model_name_or_path', type=str, default='facebook/w2v-bert-2.0')
    p.add_argument('--train_dir', type=str, required=True)
    p.add_argument('--test_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='./w2v-bert-augment-exp-out')
    p.add_argument('--experiment_name', type=str, default='', help='Пишется в metrics JSON (для диплома).')
    p.add_argument('--num_train_epochs', type=float, default=30.0)
    p.add_argument('--per_device_train_batch_size', type=int, default=2)
    p.add_argument('--per_device_eval_batch_size', type=int, default=2)
    p.add_argument('--learning_rate', type=float, default=3e-05)
    p.add_argument('--warmup_ratio', type=float, default=0.15)
    p.add_argument('--lr_scheduler_type', type=str, default=None, help='Планировщик LR (HF): cosine, linear, constant_with_warmup, ... Если не задан — дефолт Transformers.')
    p.add_argument('--weight_decay', type=float, default=None, help='Weight decay в AdamW. Если не задан — дефолт TrainingArguments (обычно 0).')
    p.add_argument('--max_steps', type=int, default=None, help='Жёсткий лимит шагов обучения (>0 перекрывает num_train_epochs). При продолжении с checkpoint без этого аргумента лимит берётся из trainer_state.json того же прогона (тот же «деноминатор», что был при первом запуске).')
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--gradient_accumulation_steps', type=int, default=2)
    p.add_argument('--precision', type=str, choices=('fp16', 'auto', 'bf16', 'fp32'), default='fp16')
    p.add_argument('--max_train_samples', type=int, default=None)
    p.add_argument('--save_steps', type=int, default=500)
    p.add_argument('--eval_steps', type=int, default=500)
    p.add_argument('--logging_steps', type=int, default=50)
    p.add_argument('--mask_time_prob', type=float, default=0.0)
    p.add_argument('--mask_feature_prob', type=float, default=0.0, help='SpecAugment по mel-каналам внутри энкодера (частотное маскирование признаков).')
    p.add_argument('--train_augment_mode', type=str, choices=('none', 'speed', 'noise'), default='none', help='Аугментация на уровне волны в train collator (eval/test без неё).')
    p.add_argument('--speed_factors', type=str, default='0.9,1.1', help='Через запятую, для speed.')
    p.add_argument('--speed_prob', type=float, default=0.5)
    p.add_argument('--noise_prob', type=float, default=0.35)
    p.add_argument('--noise_snr_min_db', type=float, default=10.0)
    p.add_argument('--noise_snr_max_db', type=float, default=20.0)
    p.add_argument('--noise_gain_db_max', type=float, default=3.0, help='0 — выключить случайный gain.')
    p.add_argument('--add_adapter', action='store_true')
    p.add_argument('--freeze_base_model', action='store_true')
    p.add_argument('--wandb_project', type=str, default='w2v-bert-urmi-asr')
    p.add_argument('--wandb_run_name', type=str, default=None)
    p.add_argument('--disable_wandb', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_total_limit', type=int, default=3)
    p.add_argument('--keep_intermediate_checkpoints', action='store_true')
    p.add_argument('--beam_width', type=int, default=10)
    p.add_argument('--skip_beam_eval', action='store_true')
    p.add_argument('--resume_from_checkpoint', type=str, default=None, metavar='PATH_OR_LAST_OR_BEST', help='Продолжить тот же прогон HF: global_step, optimizer, LR scheduler как в checkpoint. Укажите каталог checkpoint-XXXX, «last» (последний в --output_dir), «best» (минимальный eval_cer — из trainer_state последнего checkpoint в --resume_best_search_dir или в --output_dir). Для смены learning_rate/planner см. --reset_optimizer_on_resume / --resume_new_hparams_same_step.')
    p.add_argument('--resume_best_search_dir', type=str, default=None, help='Каталог первого прогона с checkpoint-* для resume_from_checkpoint=best (если не задан — используется текущий --output_dir).')
    p.add_argument('--reset_optimizer_on_resume', action='store_true', help='Только с --resume_from_checkpoint: веса+processor из checkpoint, но без восстановления optimizer/scheduler/RNG — тогда действуют новые --learning_rate, --lr_scheduler_type и global_step начнётся с 0. Без этого флага обучение по шагам продолжается с того же момента, что в чекпоинте (и с тем же сохранённым расписанием LR). Несовместимо с --resume_new_hparams_same_step.')
    p.add_argument('--resume_new_hparams_same_step', action='store_true', help='С --resume_from_checkpoint: как обычный HF resume (тот же global_step, пропуск батчей), но optimizer, LR scheduler и fp16 GradScaler создаются заново из текущих TrainingArguments (learning_rate, lr_scheduler_type, weight_decay, …). Новый scheduler синхронизируется со сохранённым global_step. Несовместимо с --reset_optimizer_on_resume.')
    p.add_argument('--model_weights_override_from', type=str, default=None, metavar='PATH_OR_BEST', help='Только с --resume_new_hparams_same_step и resume_from_checkpoint=last (или путь к последнему checkpoint-*): после загрузки trainer_state/global_step с последнего чекпоинта подставить веса модели из другого каталога. Значение «best» — каталог с минимальным eval_cer (как у resume_from_checkpoint=best), поиск в --resume_best_search_dir или в --output_dir. Требуется новый optimizer (флаг выше).')
    p.add_argument('--copy_best_checkpoint_to', type=str, default=None, help='Общий каталог: после обучения лучший checkpoint-* (eval_cer) копируется в <каталог>/<experiment_name>/checkpoint-XXXX до удаления промежуточных checkpoint-*.')
    p.add_argument('--pretrained_encoder_dir', type=str, default=None, help='Каталог со снимком masked-frame предобучения (model.safetensors / pytorch_model.bin): в CTC подставляются только веса wav2vec2_bert.*; lm_head обучается с нуля. Игнорируется при продолжении с --resume_from_checkpoint (веса из CTC-чекпоинта).')
    p.add_argument('--ctc_zero_infinity', action='store_true', help='ctc_zero_infinity=True в конфиге CTC (меньше проблем с inf на пустых выравниваниях).')
    p.add_argument('--debug_trainable_grads', action='store_true')
    return p

def copy_best_checkpoint_to_shared_dir(best_ckpt_dir: Path, shared_root: Path, experiment_label: str) -> Optional[Path]:
    src = best_ckpt_dir.resolve()
    if not src.is_dir():
        logger.warning('copy_best_checkpoint_to: не каталог: %s', src)
        return None
    safe = ''.join((c if c.isalnum() or c in '-_.' else '_' for c in experiment_label.strip())).strip('_') or 'run'
    dest_parent = shared_root.expanduser().resolve() / safe
    dest_parent.mkdir(parents=True, exist_ok=True)
    dest = dest_parent / src.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    logger.info('Лучший чекпоинт скопирован в общую папку: %s → %s', src, dest)
    return dest

def run_training(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    train_root = Path(args.train_dir)
    test_root = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    artifact_dir = output_dir / 'run_artifacts'
    plots_dir = artifact_dir / 'plots'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_train = load_split(train_root)
    raw_test = load_split(test_root)
    if args.max_train_samples:
        raw_train = raw_train.select(range(min(int(args.max_train_samples), len(raw_train))))
    logger.info('Train samples: %d | Test samples: %d', len(raw_train), len(raw_test))
    combined_for_vocab = Dataset.from_dict({'text': list(raw_train['text']) + list(raw_test['text'])})
    new_chars = extract_all_chars(combined_for_vocab)
    resume_ckpt = parse_resume_from_checkpoint_arg(args.resume_from_checkpoint)
    if resume_ckpt == RESUME_FROM_BEST_SENTINEL:
        _best_root = Path(args.resume_best_search_dir).expanduser() if args.resume_best_search_dir else output_dir
        resume_ckpt_dir = find_best_checkpoint_dir(_best_root)
        if resume_ckpt_dir is None:
            raise ValueError(f'resume_from_checkpoint=best: не найдены checkpoint-* в {_best_root.resolve()} — укажите --resume_best_search_dir на каталог первого прогона.')
    else:
        resume_ckpt_dir = resolve_resume_checkpoint_dir(output_dir, resume_ckpt)
    if resume_ckpt is True and resume_ckpt_dir is None:
        logger.warning('resume_from_checkpoint=last, но нет checkpoint-* в %s', output_dir)
    if resume_ckpt == RESUME_FROM_BEST_SENTINEL:
        logger.info('resume_from_checkpoint=best: веса/шаг из %s', resume_ckpt_dir)
    reset_opt = bool(getattr(args, 'reset_optimizer_on_resume', False))
    resume_new_hp_same = bool(getattr(args, 'resume_new_hparams_same_step', False))
    weights_override_raw = getattr(args, 'model_weights_override_from', None)
    weights_override_raw = str(weights_override_raw).strip() if weights_override_raw else None
    if resume_new_hp_same and reset_opt:
        raise ValueError('Нельзя одновременно --resume_new_hparams_same_step и --reset_optimizer_on_resume.')
    if resume_new_hp_same and resume_ckpt is None:
        raise ValueError('--resume_new_hparams_same_step требует --resume_from_checkpoint (last, best или путь к checkpoint-* ).')
    if resume_new_hp_same and resume_ckpt_dir is None:
        raise ValueError('--resume_new_hparams_same_step: каталог checkpoint не найден — проверьте путь, resume_from_checkpoint=best и --resume_best_search_dir, или --output_dir с checkpoint-*.')
    if reset_opt and resume_ckpt is None:
        raise ValueError('--reset_optimizer_on_resume требует --resume_from_checkpoint (путь или last).')
    if reset_opt and resume_ckpt_dir is None:
        raise ValueError('--reset_optimizer_on_resume: не удалось найти каталог checkpoint (проверьте путь или output_dir с checkpoint-*).')
    model_weights_override_dir: Optional[Path] = None
    if weights_override_raw:
        if not resume_new_hp_same:
            raise ValueError('--model_weights_override_from требует --resume_new_hparams_same_step (новый AdamW под новые веса).')
        if reset_opt:
            raise ValueError('--model_weights_override_from несовместимо с --reset_optimizer_on_resume.')
        if resume_ckpt is None or resume_ckpt_dir is None:
            raise ValueError('--model_weights_override_from: задайте --resume_from_checkpoint last или явный путь к последнему checkpoint-*.')
        if resume_ckpt == RESUME_FROM_BEST_SENTINEL:
            raise ValueError('--model_weights_override_from: resume_from_checkpoint=best даёт шаг из лучшего чекпоинта, а не последний. Используйте last (или путь к последнему checkpoint-*) и при необходимости model_weights_override_from=best.')
        _ov_root = Path(args.resume_best_search_dir).expanduser() if args.resume_best_search_dir else output_dir
        if weights_override_raw.lower() == 'best':
            model_weights_override_dir = find_best_checkpoint_dir(_ov_root)
            if model_weights_override_dir is None:
                raise ValueError(f'--model_weights_override_from=best: не найдены checkpoint-* в {_ov_root.resolve()} (или задайте --resume_best_search_dir на каталог первого прогона).')
            logger.info('model_weights_override_from=best → %s', model_weights_override_dir)
        else:
            model_weights_override_dir = Path(weights_override_raw).expanduser().resolve()
            if not model_weights_override_dir.is_dir():
                raise ValueError(f'--model_weights_override_from: не каталог: {model_weights_override_dir}')
    processor = load_or_build_processor(args.model_name_or_path, new_chars, output_dir, resume_ckpt_dir)
    _mask_t = float(args.mask_time_prob)
    _mask_f = float(args.mask_feature_prob)
    pretrained_encoder_load_info: Optional[Dict[str, Any]] = None
    if reset_opt:
        logger.info('Продолжение с весами из %s без восстановления optimizer/scheduler (новые гиперпараметры применятся).', resume_ckpt_dir)
        model = Wav2Vec2BertForCTC.from_pretrained(str(resume_ckpt_dir))
    else:
        model = Wav2Vec2BertForCTC.from_pretrained(args.model_name_or_path, attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, mask_time_prob=_mask_t, mask_feature_prob=_mask_f, layerdrop=0.0, ctc_loss_reduction='mean', ctc_zero_infinity=bool(getattr(args, 'ctc_zero_infinity', False)), add_adapter=bool(args.add_adapter), pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer), ignore_mismatched_sizes=True)
        _enc_raw = getattr(args, 'pretrained_encoder_dir', None)
        if _enc_raw and resume_ckpt is None:
            enc_path = Path(str(_enc_raw)).expanduser().resolve()
            if not enc_path.is_dir():
                raise ValueError(f'--pretrained_encoder_dir: не каталог: {enc_path}')
            pretrained_encoder_load_info = load_encoder_weights_into_ctc(model, enc_path)
    if args.freeze_base_model:
        for p in model.wav2vec2_bert.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = True
    model.gradient_checkpointing_enable()
    min_mel_frames = 1
    if _mask_t > 0:
        min_mel_frames = max(min_mel_frames, int(getattr(model.config, 'mask_time_length', 10)) + 1)
    min_audio_samples = min_wav_samples_for_frames(processor.feature_extractor, min_mel_frames)
    _sr = int(processor.feature_extractor.sampling_rate)
    logger.info('Минимальная длина WAV: %d сэмплов (~%.0f ms @ %d Hz); mel-кадров >= %d; mask_time_prob=%s mask_feature_prob=%s.', min_audio_samples, 1000.0 * min_audio_samples / _sr, _sr, min_mel_frames, _mask_t, _mask_f)
    _map_kw = dict(batched=True, batch_size=8, fn_kwargs={'processor': processor, 'min_audio_samples': min_audio_samples})
    test_ds = raw_test.map(prepare_w2v_bert_batch, remove_columns=raw_test.column_names, **_map_kw)
    train_ds_eval = raw_train.map(prepare_w2v_bert_batch, remove_columns=raw_train.column_names, **_map_kw)
    mode = str(args.train_augment_mode)
    speed_factors_report: Optional[Tuple[float, ...]] = None
    if mode == 'none':
        train_ds_trainer = raw_train.map(prepare_w2v_bert_batch, remove_columns=raw_train.column_names, **_map_kw)
        data_collator = DataCollatorCTCWithPadding(processor=processor)
    else:
        raw_train_grouped = raw_train.map(add_path_length_for_grouping, batched=True, batch_size=32, fn_kwargs={'sr': _sr})
        train_ds_trainer = raw_train_grouped
        parsed_sf = tuple((float(x.strip()) for x in str(args.speed_factors).split(',') if x.strip()))
        speed_factors_coll = parsed_sf if parsed_sf else (0.9, 1.1)
        if mode == 'speed':
            speed_factors_report = speed_factors_coll
        data_collator = DataCollatorW2VBertDynamicTrain(processor=processor, min_audio_samples=min_audio_samples, augment_mode=mode, base_seed=int(args.seed), speed_factors=speed_factors_coll, speed_prob=float(args.speed_prob), noise_prob=float(args.noise_prob), snr_min_db=float(args.noise_snr_min_db), snr_max_db=float(args.noise_snr_max_db), gain_db_max=float(args.noise_gain_db_max))
        logger.info('Train: динамическая аугментация mode=%s (eval/test — без аугментации, закэшированные признаки).', mode)
    _msx_cli = getattr(args, 'max_steps', None)
    _msx_final: Optional[int] = None
    if _msx_cli is not None and int(_msx_cli) > 0:
        _msx_final = int(_msx_cli)
    _num_epochs_final = float(args.num_train_epochs)
    _budget_inherited_from_checkpoint = False
    if resume_ckpt_dir is not None and (not reset_opt) and (_msx_final is None):
        mx_ckpt, n_ep_ckpt = load_training_budget_from_checkpoint(resume_ckpt_dir)
        if mx_ckpt is not None:
            _msx_final = mx_ckpt
            _budget_inherited_from_checkpoint = True
            logger.info('Продолжение без --max_steps: общий лимит шагов как в первом запуске (trainer_state.json чекпоинта): %d', _msx_final)
            if n_ep_ckpt is not None:
                _num_epochs_final = float(n_ep_ckpt)
                logger.info('num_train_epochs для согласованности с чекпоинтом: %d', int(_num_epochs_final))
        else:
            logger.warning('Продолжение без --max_steps: в trainer_state.json нет max_steps — лимит шагов пересчитается по num_train_epochs и датасету (может не совпасть с первым прогоном). Задайте --max_steps явно, если нужен тот же потолок.')
    if _msx_final is not None:
        args.max_steps = _msx_final
    report_to: List[str] = []
    if not args.disable_wandb and os.environ.get('WANDB_DISABLED', '').lower() not in ('1', 'true', 'yes'):
        report_to.append('wandb')
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
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
            raise RuntimeError('--precision bf16, но bf16 на GPU недоступен.')
        _use_bf16, _use_fp16 = (True, False)
    elif pref == 'auto':
        _use_bf16 = _bf16_supported
        _use_fp16 = bool(_cuda_ok and (not _use_bf16))
    else:
        _use_bf16, _use_fp16 = (False, bool(_cuda_ok))
    _sched = getattr(args, 'lr_scheduler_type', None)
    _wd = getattr(args, 'weight_decay', None)
    training_args_kw: Dict[str, Any] = dict(output_dir=str(output_dir), **_group_kw, per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, **{_eval_kw: 'steps'}, save_strategy='steps', num_train_epochs=_num_epochs_final, bf16=_use_bf16, fp16=_use_fp16, save_steps=args.save_steps, eval_steps=args.eval_steps, logging_steps=args.logging_steps, learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio, save_total_limit=save_limit, load_best_model_at_end=True, metric_for_best_model='eval_cer', greater_is_better=False, report_to=report_to, push_to_hub=False, gradient_checkpointing=True, seed=args.seed, max_grad_norm=float(args.max_grad_norm), remove_unused_columns=False)
    if _msx_final is not None:
        training_args_kw['max_steps'] = int(_msx_final)
    if _sched:
        training_args_kw['lr_scheduler_type'] = str(_sched)
    if _wd is not None:
        training_args_kw['weight_decay'] = float(_wd)
    training_args = TrainingArguments(**training_args_kw)
    compute_metrics = build_compute_metrics(processor)
    if resume_new_hp_same and model_weights_override_dir is not None:
        trainer = TrainerResumeNewHparamsWeightOverride(model=model, data_collator=data_collator, args=training_args, train_dataset=train_ds_trainer, eval_dataset=test_ds, processing_class=processor, compute_metrics=compute_metrics, resume_new_hparams_same_step=True, model_weights_override_dir=model_weights_override_dir)
    elif resume_new_hp_same:
        trainer = TrainerResumeNewHparamsSameStep(model=model, data_collator=data_collator, args=training_args, train_dataset=train_ds_trainer, eval_dataset=test_ds, processing_class=processor, compute_metrics=compute_metrics, resume_new_hparams_same_step=True)
    else:
        trainer = Trainer(model=model, data_collator=data_collator, args=training_args, train_dataset=train_ds_trainer, eval_dataset=test_ds, processing_class=processor, compute_metrics=compute_metrics)
    if args.debug_trainable_grads:
        debug_log_trainable_gradients(trainer)
    if reset_opt:
        train_resume = None
    elif resume_ckpt_dir is not None:
        train_resume = str(resume_ckpt_dir.resolve())
    else:
        train_resume = None
    if reset_opt:
        logger.info('trainer.train() без HF-resume: global_step начнётся с 0, метрики в логах — новый проход (веса инициализированы из %s).', resume_ckpt_dir)
    elif resume_new_hp_same:
        logger.info('resume_new_hparams_same_step: тот же global_step и пропуск даталоадера как при resume; optimizer/LR/GradScaler созданы заново (learning_rate=%s, lr_scheduler_type=%s).', args.learning_rate, _sched or '(дефолт HF)')
    if resume_ckpt_dir is not None:
        logger.info('Продолжение обучения: чекпоинт %s | reset_optimizer_on_resume=%s | resume_new_hparams_same_step=%s | model_weights_override_from=%s | lr=%s warmup_ratio=%s weight_decay=%s max_grad_norm=%s lr_scheduler_type=%s', resume_ckpt_dir.resolve(), reset_opt, resume_new_hp_same, str(model_weights_override_dir) if model_weights_override_dir else None, args.learning_rate, args.warmup_ratio, _wd if _wd is not None else '(HF default)', args.max_grad_norm, _sched or '(HF default)')
    trainer.train(resume_from_checkpoint=train_resume)
    try:
        plots_saved = save_training_plots(trainer.state.log_history, plots_dir)
    except Exception as e:
        logger.warning('Графики: %s', e)
        plots_saved = []
    try:
        (artifact_dir / 'trainer_log_history.json').write_text(json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    except Exception as e:
        logger.warning('trainer_log_history.json: %s', e)
    logger.info('Финальная оценка greedy train/test (без аугментации).')
    train_eval = trainer.evaluate(train_ds_eval, metric_key_prefix='greedy_train')
    test_eval = trainer.evaluate(test_ds, metric_key_prefix='greedy_test')
    best_ckpt = getattr(trainer.state, 'best_model_checkpoint', None) or ''
    exp_label = (args.experiment_name or output_dir.name).strip() or 'unnamed'
    shared_copy_dest: Optional[Path] = None
    _copy_raw = getattr(args, 'copy_best_checkpoint_to', None)
    if _copy_raw and str(_copy_raw).strip() and best_ckpt:
        shared_copy_dest = copy_best_checkpoint_to_shared_dir(Path(best_ckpt), Path(str(_copy_raw).strip()), exp_label)
    elif _copy_raw and str(_copy_raw).strip() and (not best_ckpt):
        logger.warning('copy_best_checkpoint_to задан (%s), но best_model_checkpoint пуст — пропуск копирования.', _copy_raw)
    trainer.save_model(str(output_dir / 'final_model'))
    processor.save_pretrained(str(output_dir / 'final_model'))
    run_meta: Dict[str, Any] = {'experiment_name': exp_label, 'base_pretrained_model': args.model_name_or_path, 'pretrained_encoder_dir': str(Path(args.pretrained_encoder_dir).expanduser().resolve()) if getattr(args, 'pretrained_encoder_dir', None) else None, 'pretrained_encoder_load': pretrained_encoder_load_info, 'ctc_zero_infinity': bool(getattr(args, 'ctc_zero_infinity', False)), 'reset_optimizer_on_resume': reset_opt, 'resume_new_hparams_same_step': resume_new_hp_same, 'model_weights_override_from': str(model_weights_override_dir) if model_weights_override_dir else None, 'resume_from_checkpoint_path': str(resume_ckpt_dir.resolve()) if resume_ckpt_dir is not None else None, 'weights_initialized_from_checkpoint': str(resume_ckpt_dir.resolve()) if reset_opt and resume_ckpt_dir else None, 'learning_rate': float(args.learning_rate), 'warmup_ratio': float(args.warmup_ratio), 'max_grad_norm': float(args.max_grad_norm), 'gradient_accumulation_steps': int(args.gradient_accumulation_steps), 'lr_scheduler_type': _sched, 'weight_decay': _wd, 'max_steps': int(_msx_final) if _msx_final is not None else None, 'max_steps_inherited_from_checkpoint': _budget_inherited_from_checkpoint, 'add_adapter': bool(args.add_adapter), 'freeze_base_model': bool(args.freeze_base_model), 'mask_time_prob': _mask_t, 'mask_feature_prob': _mask_f, 'train_augment_mode': mode, 'speed_factors': list(speed_factors_report) if speed_factors_report else None, 'speed_prob': float(args.speed_prob) if mode == 'speed' else None, 'noise_prob': float(args.noise_prob) if mode == 'noise' else None, 'noise_snr_db_range': [float(args.noise_snr_min_db), float(args.noise_snr_max_db)] if mode == 'noise' else None, 'noise_gain_db_max': float(args.noise_gain_db_max) if mode == 'noise' else None, 'best_model_checkpoint_path_before_cleanup': best_ckpt, 'best_checkpoint_shared_copy_path': str(shared_copy_dest.resolve()) if shared_copy_dest else None, 'beam_width_used': int(args.beam_width), 'metric_for_best_model': 'eval_cer', 'best_metric_during_training': getattr(trainer.state, 'best_metric', None), 'trainer_global_steps': getattr(trainer.state, 'global_step', None), 'train_samples': len(raw_train), 'test_samples': len(raw_test), **model_run_info(trainer.model), 'plots_saved': plots_saved, 'final_model_dir': str((output_dir / 'final_model').resolve()), 'artifacts_dir': str(artifact_dir.resolve())}
    greedy_pack = {'train': (train_eval.get('greedy_train_wer', train_eval.get('eval_wer', float('nan'))), train_eval.get('greedy_train_cer', train_eval.get('eval_cer', float('nan')))), 'test': (test_eval.get('greedy_test_wer', test_eval.get('eval_wer', float('nan'))), test_eval.get('greedy_test_cer', test_eval.get('eval_cer', float('nan'))))}
    beam_pack: Dict[str, Tuple[float, float]] = {}
    beam_details: Dict[str, Any] = {}
    if not args.skip_beam_eval:
        try:
            from pyctcdecode import build_ctcdecoder
            vocab_labels = labels_aligned_for_pyctc(processor)
            decoder = build_ctcdecoder(vocab_labels)
            trainer.model.eval()
            bw = int(args.beam_width)
            for split_key, ds in (('train', train_ds_eval), ('test', test_ds)):
                logits, lids = predict_logits_and_refs(trainer, ds, processor)
                logits = logits.astype(np.float32, copy=False)
                wer_b, cer_b = beam_metrics_from_predictions(logits, lids, processor, decoder, bw)
                beam_pack[split_key] = (wer_b, cer_b)
                beam_details[f'beam_{split_key}'] = {'wer': wer_b, 'cer': cer_b, 'beam_width': bw}
                logger.info('Beam %s: WER=%.6f CER=%.6f (beam_width=%s)', split_key, wer_b, cer_b, bw)
        except Exception as e:
            logger.exception('Beam search: %s', e)
            beam_details['beam_error'] = repr(e)
    if args.keep_intermediate_checkpoints:
        logger.info('Сохранены все checkpoint-* (--keep_intermediate_checkpoints).')
    else:
        removed = remove_intermediate_checkpoints(output_dir)
        run_meta['checkpoint_dirs_removed_after_train'] = removed
    run_meta['kept_intermediate_checkpoints_flag'] = bool(args.keep_intermediate_checkpoints)
    excel_path = artifact_dir / 'best_model_wer_cer_greedy_and_beam.xlsx'
    try:
        export_metrics_excel(excel_path, greedy={k: greedy_pack[k] for k in greedy_pack}, beam=beam_pack, meta=run_meta)
    except Exception as e:
        logger.warning('Excel: %s', e)
    metrics_json = artifact_dir / 'metrics_summary.json'
    summary = {'greedy': {'train': {'wer': greedy_pack['train'][0], 'cer': greedy_pack['train'][1]}, 'test': {'wer': greedy_pack['test'][0], 'cer': greedy_pack['test'][1]}}, 'beam': {split: {'wer': beam_pack.get(split, (np.nan, np.nan))[0], 'cer': beam_pack.get(split, (np.nan, np.nan))[1]} for split in ('train', 'test')}, 'meta': run_meta, 'beam_extra': beam_details}
    metrics_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    logger.info('Готово: %s | артефакты: %s', output_dir / 'final_model', artifact_dir)
    if report_to:
        try:
            import wandb
            wandb.finish()
        except Exception:
            logger.exception('wandb.finish')
import logging

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = build_parser()
    p.set_defaults(output_dir='./w2v-bert_speed_perturbation', experiment_name='w2v-bert_speed_perturbation', train_augment_mode='speed', mask_time_prob=0.0, mask_feature_prob=0.0, speed_factors='0.9,1.1', speed_prob=0.5, add_adapter=True)
    run_training(p.parse_args())
if __name__ == '__main__':
    main()
