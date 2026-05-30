import argparse
import inspect
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from datasets import Dataset
from transformers import SeamlessM4TFeatureExtractor, Trainer, TrainingArguments, Wav2Vec2BertForCTC, Wav2Vec2BertProcessor, Wav2Vec2CTCTokenizer
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
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Fine-tune Wav2Vec2-BERT CTC ASR (w2v-bert-2.0) for Urmia.')
    p.add_argument('--model_name_or_path', type=str, default='facebook/w2v-bert-2.0', help='Предобученный Wav2Vec2-BERT (ASR требует дообучения CTC-головы).')
    p.add_argument('--train_dir', type=str, required=True)
    p.add_argument('--test_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='./w2v-bert-urmi-out')
    p.add_argument('--num_train_epochs', type=float, default=30.0)
    p.add_argument('--per_device_train_batch_size', type=int, default=2)
    p.add_argument('--per_device_eval_batch_size', type=int, default=2)
    p.add_argument('--learning_rate', type=float, default=3e-05)
    p.add_argument('--warmup_ratio', type=float, default=0.15)
    p.add_argument('--gradient_accumulation_steps', type=int, default=2)
    p.add_argument('--precision', type=str, choices=('fp16', 'auto', 'bf16', 'fp32'), default='fp16')
    p.add_argument('--max_train_samples', type=int, default=None)
    p.add_argument('--save_steps', type=int, default=500)
    p.add_argument('--eval_steps', type=int, default=500)
    p.add_argument('--logging_steps', type=int, default=50)
    p.add_argument('--mask_time_prob', type=float, default=0.0, help='Во время fine-tune SpecAugment по времени внутри энкодера. 0 — как в блоге HF для w2v-bert; >0 — поднимайте минимальную длину аудио (см. лог после старта).')
    p.add_argument('--add_adapter', action='store_true', help='config.add_adapter=True (см. пример HF); меняет размер скрытого состояния для lm_head.')
    p.add_argument('--freeze_base_model', action='store_true', help='Заморозить wav2vec2_bert; обучается только lm_head (быстро, но обычно хуже на ASR).')
    p.add_argument('--wandb_project', type=str, default='w2v-bert-urmi-asr')
    p.add_argument('--wandb_run_name', type=str, default=None)
    p.add_argument('--disable_wandb', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_total_limit', type=int, default=3)
    p.add_argument('--keep_intermediate_checkpoints', action='store_true')
    p.add_argument('--beam_width', type=int, default=10)
    p.add_argument('--skip_beam_eval', action='store_true')
    p.add_argument('--resume_from_checkpoint', type=str, default=None, metavar='PATH_OR_LAST')
    p.add_argument('--debug_trainable_grads', action='store_true')
    return p.parse_args()

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
        logger.warning('resume_from_checkpoint=last, но нет checkpoint-* в %s', output_dir)
    processor = load_or_build_processor(args.model_name_or_path, new_chars, output_dir, resume_ckpt_dir)
    model = Wav2Vec2BertForCTC.from_pretrained(args.model_name_or_path, attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, mask_time_prob=float(args.mask_time_prob), layerdrop=0.0, ctc_loss_reduction='mean', add_adapter=bool(args.add_adapter), pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer), ignore_mismatched_sizes=True)
    if args.freeze_base_model:
        for p in model.wav2vec2_bert.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = True
        logger.info('freeze_base_model: trainable=%s', sum((x.numel() for x in model.parameters() if x.requires_grad)))
    model.gradient_checkpointing_enable()
    min_mel_frames = 1
    if float(args.mask_time_prob) > 0:
        min_mel_frames = max(min_mel_frames, int(getattr(model.config, 'mask_time_length', 10)) + 1)
    min_audio_samples = min_wav_samples_for_frames(processor.feature_extractor, min_mel_frames)
    _sr = int(processor.feature_extractor.sampling_rate)
    logger.info('Минимальная длина WAV: %d сэмплов (~%.0f ms @ %d Hz); mel-кадров >= %d (mask_time_prob=%s).', min_audio_samples, 1000.0 * min_audio_samples / _sr, _sr, min_mel_frames, args.mask_time_prob)
    _map_kw = dict(batched=True, batch_size=8, fn_kwargs={'processor': processor, 'min_audio_samples': min_audio_samples})
    train_ds = raw_train.map(prepare_w2v_bert_batch, remove_columns=raw_train.column_names, **_map_kw)
    test_ds = raw_test.map(prepare_w2v_bert_batch, remove_columns=raw_test.column_names, **_map_kw)
    data_collator = DataCollatorCTCWithPadding(processor=processor)
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
    training_args = TrainingArguments(output_dir=str(output_dir), **_group_kw, per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, **{_eval_kw: 'steps'}, save_strategy='steps', num_train_epochs=args.num_train_epochs, bf16=_use_bf16, fp16=_use_fp16, save_steps=args.save_steps, eval_steps=args.eval_steps, logging_steps=args.logging_steps, learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio, save_total_limit=save_limit, load_best_model_at_end=True, metric_for_best_model='eval_cer', greater_is_better=False, report_to=report_to, push_to_hub=False, gradient_checkpointing=True, seed=args.seed, max_grad_norm=1.0)
    compute_metrics = build_compute_metrics(processor)
    trainer = Trainer(model=model, data_collator=data_collator, args=training_args, train_dataset=train_ds, eval_dataset=test_ds, processing_class=processor, compute_metrics=compute_metrics)
    if args.debug_trainable_grads:
        debug_log_trainable_gradients(trainer)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    try:
        plots_saved = save_training_plots(trainer.state.log_history, plots_dir)
    except Exception as e:
        logger.warning('Графики: %s', e)
        plots_saved = []
    try:
        (artifact_dir / 'trainer_log_history.json').write_text(json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    except Exception as e:
        logger.warning('trainer_log_history.json: %s', e)
    logger.info('Финальная оценка greedy train/test.')
    train_eval = trainer.evaluate(train_ds, metric_key_prefix='greedy_train')
    test_eval = trainer.evaluate(test_ds, metric_key_prefix='greedy_test')
    best_ckpt = getattr(trainer.state, 'best_model_checkpoint', None) or ''
    trainer.save_model(str(output_dir / 'final_model'))
    processor.save_pretrained(str(output_dir / 'final_model'))
    run_meta = {'base_pretrained_model': args.model_name_or_path, 'add_adapter': bool(args.add_adapter), 'freeze_base_model': bool(args.freeze_base_model), 'mask_time_prob': float(args.mask_time_prob), 'best_model_checkpoint_path_before_cleanup': best_ckpt, 'beam_width_used': int(args.beam_width), 'metric_for_best_model': 'eval_cer', 'best_metric_during_training': getattr(trainer.state, 'best_metric', None), 'trainer_global_steps': getattr(trainer.state, 'global_step', None), 'train_samples': len(raw_train), 'test_samples': len(raw_test), **model_run_info(trainer.model), 'plots_saved': plots_saved, 'final_model_dir': str((output_dir / 'final_model').resolve()), 'artifacts_dir': str(artifact_dir.resolve()), 'comparison_note': 'Сравнение с MMS: см. mms-urmi-out/run_artifacts/metrics_summary.json и этот файл; архитектуры разные, но greedy/beam WER/CER на одних и тех же 5h_* должны быть сопоставимы.'}
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
            for split_key, ds in (('train', train_ds), ('test', test_ds)):
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
        if removed:
            logger.info('Удалены checkpoint-*: %s', removed)
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

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run_training(parse_args())
if __name__ == '__main__':
    main()
