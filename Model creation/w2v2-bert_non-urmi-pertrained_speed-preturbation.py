from __future__ import annotations
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import inspect
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from datasets import Dataset
from transformers import SeamlessM4TFeatureExtractor, Trainer, TrainingArguments, Wav2Vec2BertForCTC, Wav2Vec2BertProcessor, Wav2Vec2CTCTokenizer

def build_char_tokenizer(chars: List[str], work_dir: Path) -> Wav2Vec2CTCTokenizer:
    unk_t = '<unk>'
    pad_t = '<pad>'
    vocab: Dict[str, int] = {pad_t: 0, unk_t: 1}
    nxt = 2
    for c in sorted(set(chars)):
        if c in vocab:
            continue
        vocab[c] = nxt
        nxt += 1
    if ' ' in vocab:
        delim = ' '
    else:
        delim = '|'
        if delim not in vocab:
            vocab[delim] = nxt
            nxt += 1
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

import random
import re
import shutil
from dataclasses import dataclass, field
from asr_training_common import RESUME_FROM_BEST_SENTINEL, beam_metrics_from_predictions, build_compute_metrics, debug_log_trainable_gradients, export_metrics_excel, extract_all_chars, find_best_checkpoint_dir, labels_aligned_for_pyctc, load_audio_mono, load_split, model_run_info, parse_resume_from_checkpoint_arg, predict_logits_and_refs, remove_intermediate_checkpoints, resolve_resume_checkpoint_dir, save_training_plots
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
            wandb.finish()
        except Exception:
            logger.exception('wandb.finish')

import torch.nn as nn
import torch.nn.functional as F
from transformers import SeamlessM4TFeatureExtractor, Trainer, TrainerCallback, TrainingArguments, Wav2Vec2BertConfig, Wav2Vec2BertForCTC, Wav2Vec2BertModel, Wav2Vec2BertProcessor, Wav2Vec2BertPreTrainedModel
from transformers.models.wav2vec2_bert.modeling_wav2vec2_bert import _compute_mask_indices
from asr_training_common import beam_metrics_from_predictions, build_compute_metrics, export_metrics_excel, extract_all_chars, labels_aligned_for_pyctc, load_split, model_run_info, parse_resume_from_checkpoint_arg, predict_logits_and_refs, predict_logits_and_refs_chunked, remove_intermediate_checkpoints, resolve_resume_checkpoint_dir, save_training_plots, debug_log_trainable_gradients
from asr_training_common import load_audio_mono
logger = logging.getLogger(__name__)

@dataclass
class PretrainSnapshotBundle:
    root: Path
    milestone_steps: Tuple[int, ...] = (5000, 10000, 15000)
    trainer: Optional[Trainer] = None
    plateau_saved: bool = False
    written: Dict[str, str] = field(default_factory=dict)

    def bind_trainer(self, trainer: Trainer) -> None:
        self.trainer = trainer

    def save_subdir(self, name: str, *, reason: str='') -> Optional[Path]:
        if self.trainer is None:
            logger.warning('PretrainSnapshotBundle: trainer не привязан, пропуск %s', name)
            return None
        dst = self.root / name
        if dst.exists():
            shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True, exist_ok=True)
        self.trainer.save_model(str(dst))
        self.written[name] = str(dst.resolve())
        logger.info('Снимок предобучения%s: %s → %s', f' ({reason})' if reason else '', name, dst)
        return dst

class PretrainMilestoneSnapshotCallback(TrainerCallback):

    def __init__(self, bundle: PretrainSnapshotBundle):
        self.bundle = bundle
        self._saved: set[int] = set()

    def on_step_end(self, args, state, control, **kwargs):
        step = int(state.global_step)
        for tgt in self.bundle.milestone_steps:
            if tgt in self._saved or step < tgt:
                continue
            self.bundle.save_subdir(f'step_{tgt}', reason=f'контрольная точка ≥{tgt} шагов')
            self._saved.add(tgt)
        return control

class PretrainEvalPlateauEarlyStopping(TrainerCallback):

    def __init__(self, *, metric_key: str, min_global_step: int, patience: int, rel_improvement_min: float, plateau_state: Dict[str, Any], snapshot_bundle: Optional[PretrainSnapshotBundle]=None, plateau_snapshot_dirname: str='plateau_early_stop'):
        self.metric_key = metric_key
        self.min_global_step = int(min_global_step)
        self.patience = int(patience)
        self.rel_improvement_min = float(rel_improvement_min)
        self.plateau_state = plateau_state
        self.snapshot_bundle = snapshot_bundle
        self.plateau_snapshot_dirname = plateau_snapshot_dirname
        self.slow_streak = 0
        self.prev_loss: Optional[float] = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or self.metric_key not in metrics:
            return control
        loss = float(metrics[self.metric_key])
        step = int(state.global_step)
        if step < self.min_global_step:
            self.prev_loss = loss
            self.slow_streak = 0
            self.plateau_state['phase'] = 'before_plateau_watch'
            return control
        if self.prev_loss is None:
            self.prev_loss = loss
            return control
        prev = self.prev_loss
        rel_imp = (prev - loss) / max(abs(prev), 1e-12)
        self.prev_loss = loss
        self.plateau_state['last_eval_loss'] = loss
        self.plateau_state['last_rel_improvement'] = rel_imp
        self.plateau_state['last_global_step'] = step
        self.plateau_state['phase'] = 'plateau_watch'
        if rel_imp >= self.rel_improvement_min:
            self.slow_streak = 0
            self.plateau_state['slow_streak'] = 0
        else:
            self.slow_streak += 1
            self.plateau_state['slow_streak'] = self.slow_streak
            logger.info('Предобучение (контроль плато): относит. улучшение eval_loss=%.6g (порог %s), серия «медленно» %d/%d, global_step=%d', rel_imp, self.rel_improvement_min, self.slow_streak, self.patience, step)
        if self.slow_streak >= self.patience:
            self.plateau_state['early_stopped'] = True
            self.plateau_state['early_stop_reason'] = 'eval_loss_plateau'
            self.plateau_state['early_stop_global_step'] = step
            if self.snapshot_bundle is not None:
                p = self.snapshot_bundle.save_subdir(self.plateau_snapshot_dirname, reason='ранний стоп: eval_loss перестал быстро падать')
                if p is not None:
                    self.snapshot_bundle.plateau_saved = True
                    self.plateau_state['plateau_snapshot_dir'] = str(p.resolve())
            logger.info('Ранняя остановка предобучения: %d подряд оценок без заметного падения eval_loss (global_step=%d). Для CTC будет использован снимок %s.', self.patience, step, self.plateau_snapshot_dirname)
            control.should_training_stop = True
        return control

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Pretrain Wav2Vec2-BERT on audio, then fine-tune CTC (Urmi).')
    p.add_argument('--phase', choices=('both', 'pretrain_only', 'finetune_only'), default='both')
    p.add_argument('--output_dir', type=str, default='./w2v-bert-pretrain-finetune-out')
    p.add_argument('--base_model_name_or_path', type=str, default='facebook/w2v-bert-2.0', help='Инициализация энкодера и feature extractor.')
    p.add_argument('--pretrained_encoder_checkpoint', type=str, default=None, help='Каталог checkpoint (phase finetune_only): лучший этап предобучения, см. pretrain/best_pretrained_model.')
    p.add_argument('--pretrain_audio_roots', type=str, nargs='+', default=['./5h_urmi_pretraining', './5h_train/audios'], help='Один или несколько каталогов с wav (рекурсивный поиск *.wav; если есть подкаталог audios/, берётся он).')
    p.add_argument('--pretrain_validation_ratio', type=float, default=0.02)
    p.add_argument('--pretrain_max_samples', type=int, default=None)
    p.add_argument('--pretrain_num_train_epochs', type=float, default=5.0)
    p.add_argument('--pretrain_learning_rate', type=float, default=1e-05, help='Continued pretrain поверх MMS: для малого доменного корпуса ниже LR снижает забывание и переадаптацию (типичный диапазон 5e-6–3e-5).')
    p.add_argument('--pretrain_weight_decay', type=float, default=0.01, help='AdamW weight decay на SSL-этапе; для ~18k файлов 0.01 стабилизирует энкодер.')
    p.add_argument('--pretrain_lr_scheduler_type', type=str, default='cosine', help='Расписание LR предобучения (HF Trainer): cosine обычно лучше linear после warmup.')
    p.add_argument('--pretrain_per_device_train_batch_size', type=int, default=2)
    p.add_argument('--pretrain_per_device_eval_batch_size', type=int, default=2)
    p.add_argument('--pretrain_gradient_accumulation_steps', type=int, default=4, help='Эффективный батч = per_device × world_size × это значение; больше — стабильнее градиенты при той же памяти.')
    p.add_argument('--pretrain_save_steps', type=int, default=250, help='Частые сохранения; при контроле плато удобнее иметь точки около 5k–15k шагов.')
    p.add_argument('--pretrain_eval_steps', type=int, default=250)
    p.add_argument('--pretrain_logging_steps', type=int, default=50)
    p.add_argument('--pretrain_warmup_ratio', type=float, default=0.1)
    p.add_argument('--pretrain_mask_time_prob', type=float, default=0.065)
    p.add_argument('--pretrain_mask_time_length', type=int, default=10)
    p.add_argument('--pretrain_decoder_num_layers', type=int, default=3, help='Число полносвязных слоёв в декодере предобучения (Linear с GELU между промежуточными).')
    p.add_argument('--pretrain_perceptron_hidden_dim', type=int, default=0, help='Размер скрытых слоёв декодера при decoder_num_layers>1; 0 — брать hidden_size энкодера. При decoder_num_layers=1 этот параметр не используется.')
    p.add_argument('--pretrain_seed', type=int, default=42)
    p.add_argument('--pretrain_snapshot_milestone_steps', type=int, nargs='+', default=[5000, 10000, 15000], help='После global_step ≥ каждого значения сохраняется каталог pretrain/snapshots/step_<N>/ (веса save_model).')
    p.add_argument('--pretrain_save_total_limit', type=int, default=4)
    p.add_argument('--pretrain_max_steps', type=int, default=0, help='Жёсткий потолок шагов оптимизатора предобучения (>0 задаёт HF max_steps и перевешивает num_train_epochs). 0 — только num_train_epochs и при необходимости ранняя остановка по плато eval_loss.')
    p.add_argument('--pretrain_disable_eval_plateau_early_stop', action='store_true', help='Не останавливаться по плато eval_loss; отработать все эпохи (и при необходимости max_steps).')
    p.add_argument('--pretrain_plateau_min_steps', type=int, default=4000, help='Не считать плато до этого global_step (обычно после начального быстрого падения loss).')
    p.add_argument('--pretrain_plateau_patience', type=int, default=3, help='Столько подряд оценок eval без «достаточного» относительного улучшения — стоп.')
    p.add_argument('--pretrain_plateau_rel_improvement_min', type=float, default=0.005, help='Минимум относительного падения eval_loss между соседними eval: (L_prev-L_curr)/|L_prev|. Ниже — считаем шаг «медленным» для plateau patience.')
    p.add_argument('--pretrain_keep_checkpoints', action='store_true')
    p.add_argument('--pretrain_resume_from_checkpoint', type=str, default=None, metavar='PATH_OR_LAST', help='Продолжить предобучение с HF чекпоинта: «last» — последний сохранённый шаг в каталоге pretrain (тот же --output_dir/pretrain); или полный путь к checkpoint-XXXX. Если обрыв случился во время сохранения, берите предыдущий по номеру шага каталог с полным trainer_state.json.')
    p.add_argument('--train_dir', type=str, default=None)
    p.add_argument('--test_dir', type=str, default=None)
    p.add_argument('--finetune_max_steps', type=int, default=0, help='Максимум шагов оптимизатора только на размеченном CTC-этапе (не считая предобучение). HF: max_steps перекрывает num_train_epochs при значении > 0. Дефолт 0 — лимит только по --finetune_num_train_epochs (удобнее при разном размере train).')
    p.add_argument('--finetune_num_train_epochs', type=float, default=35.0, help='При finetune_max_steps=0 обучение длится столько проходов по train (выбор лучшего по eval — load_best_model_at_end).')
    p.add_argument('--finetune_learning_rate', type=float, default=3e-05, help='Для ~4–5k размеченных высказываний 3e-5 с cosine и weight_decay обычно безопаснее, чем 5e-5.')
    p.add_argument('--finetune_weight_decay', type=float, default=0.01, help='AdamW на CTC-этапе; при малом размеченном наборе снижает переобучение энкодера.')
    p.add_argument('--finetune_lr_scheduler_type', type=str, default='cosine', help='Расписание LR финетюна; cosine распространён для дообучения wav2vec-подобных моделей.')
    p.add_argument('--finetune_warmup_steps', type=int, default=0, help='Если > 0 — линейный warmup ровно на столько шагов (поле warmup_ratio игнорируется). Дефолт 0: используйте warmup_ratio.')
    p.add_argument('--finetune_warmup_ratio', type=float, default=0.1, help='Доля шагов под линейный warmup при finetune_warmup_steps=0 (типично 0.05–0.15).')
    p.add_argument('--finetune_per_device_train_batch_size', type=int, default=2)
    p.add_argument('--finetune_per_device_eval_batch_size', type=int, default=2)
    p.add_argument('--finetune_gradient_accumulation_steps', type=int, default=4)
    p.add_argument('--finetune_save_steps', type=int, default=500)
    p.add_argument('--finetune_eval_steps', type=int, default=500)
    p.add_argument('--finetune_logging_steps', type=int, default=50)
    p.add_argument('--finetune_mask_time_prob', type=float, default=0.05, help='SpecAugment по времени при CTC; 0 отключает. ~0.05 — распространённый компромисс качество/устойчивость.')
    p.add_argument('--finetune_no_adapter', action='store_true')
    p.add_argument('--finetune_freeze_base_model', action='store_true')
    p.add_argument('--finetune_freeze_feature_projection', action='store_true')
    p.add_argument('--finetune_no_gradient_checkpointing', action='store_true')
    p.add_argument('--finetune_max_train_samples', type=int, default=None)
    p.add_argument('--finetune_save_total_limit', type=int, default=3)
    p.add_argument('--finetune_keep_checkpoints', action='store_true')
    p.add_argument('--finetune_beam_width', type=int, default=10)
    p.add_argument('--finetune_beam_eval', action='store_true', help='Включить финальную оценку beam search (долго на CPU). По умолчанию только greedy.')
    p.add_argument('--finetune_beam_logits_chunk_samples', type=int, default=256)
    p.add_argument('--finetune_resume_from_checkpoint', type=str, default=None)
    p.add_argument('--finetune_debug_trainable_grads', action='store_true')
    p.add_argument('--precision', type=str, choices=('fp16', 'auto', 'bf16', 'fp32'), default='auto', help='auto: FP16 при CUDA (см. precision_flags в скрипте); для устойчивости чисел на поддерживающих GPU см. bf16.')
    p.add_argument('--wandb_project', type=str, default='w2v-bert-pretrain-finetune-urmi')
    p.add_argument('--wandb_run_name', type=str, default=None)
    p.add_argument('--disable_wandb', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.phase in ('both', 'finetune_only'):
        if not args.train_dir or not args.test_dir:
            p.error('--train_dir и --test_dir обязательны для phase both или finetune_only.')
    if args.phase == 'finetune_only' and (not args.pretrained_encoder_checkpoint):
        p.error('--pretrained_encoder_checkpoint обязателен для phase finetune_only.')
    return args

def collect_wav_paths(roots: List[str]) -> List[str]:
    paths: List[str] = []
    for r in roots:
        root = Path(r).expanduser().resolve()
        if not root.exists():
            logger.warning('Каталог аудио не найден, пропуск: %s', root)
            continue
        scan = root / 'audios' if (root / 'audios').is_dir() else root
        for wav in sorted(scan.rglob('*.wav')):
            paths.append(str(wav.resolve()))
    seen: set[str] = set()
    out: List[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def prepare_pretrain_batch(batch: Dict[str, List], feature_extractor: SeamlessM4TFeatureExtractor, min_audio_samples: int) -> Dict[str, List]:
    sr = int(feature_extractor.sampling_rate)
    feats: List[np.ndarray] = []
    lengths: List[int] = []
    for p in batch['path']:
        y = load_audio_mono(p, sr)
        if y.size == 0:
            logger.warning('Пустой wav — тишина min_audio_samples=%d: %s', min_audio_samples, p)
            y = np.zeros(min_audio_samples, dtype=np.float32)
        elif len(y) < min_audio_samples:
            y = np.pad(y, (0, min_audio_samples - len(y)), mode='constant')
        enc = feature_extractor(y, sampling_rate=sr, return_tensors='np')
        arr = enc['input_features'][0]
        feats.append(arr)
        lengths.append(int(arr.shape[0]))
    return {'input_features': feats, 'length': lengths}

@dataclass
class DataCollatorAudioOnly:
    feature_extractor: SeamlessM4TFeatureExtractor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        return self.feature_extractor.pad([{'input_features': f['input_features']} for f in features], padding=True, return_tensors='pt')

class PerceptronDecoder(nn.Module):

    def __init__(self, hidden_size: int, out_dim: int, *, num_layers: int=3, hidden_mid: int=0):
        super().__init__()
        nl = max(1, int(num_layers))
        mid = int(hidden_mid) if int(hidden_mid) > 0 else int(hidden_size)
        layers: List[nn.Module] = []
        d_in = int(hidden_size)
        for _ in range(nl - 1):
            layers.extend([nn.Linear(d_in, mid), nn.GELU()])
            d_in = mid
        layers.append(nn.Linear(d_in, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states)

class Wav2Vec2BertForMaskedFramePretraining(Wav2Vec2BertPreTrainedModel):
    config_class = Wav2Vec2BertConfig
    base_model_prefix = 'wav2vec2_bert'

    def __init__(self, config: Wav2Vec2BertConfig, *, decoder_num_layers: int=3, decoder_hidden_dim: int=0):
        super().__init__(config)
        self.wav2vec2_bert = Wav2Vec2BertModel(config)
        mid = int(decoder_hidden_dim)
        self.perceptron_decoder = PerceptronDecoder(config.hidden_size, config.feature_projection_input_dim, num_layers=int(decoder_num_layers), hidden_mid=mid)
        self.post_init()

    def forward(self, input_features: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, return_loss: bool=True, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        mask_np = _compute_mask_indices((input_features.shape[0], input_features.shape[1]), mask_prob=float(getattr(self, 'pretrain_mask_time_prob', 0.065)), mask_length=int(getattr(self, 'pretrain_mask_time_length', 10)), attention_mask=attention_mask, min_masks=self.config.mask_time_min_masks)
        mask_bt = torch.tensor(mask_np, device=input_features.device, dtype=torch.bool)
        outputs = self.wav2vec2_bert(input_features=input_features, attention_mask=attention_mask, mask_time_indices=mask_bt, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=True)
        hidden = outputs.last_hidden_state
        extract = outputs.extract_features
        pred = self.perceptron_decoder(hidden)
        if mask_bt.any():
            tgt = extract.detach()[mask_bt]
            pr = pred[mask_bt]
            loss = F.mse_loss(pr, tgt)
        else:
            loss = pred.mean() * 0.0
        if not return_dict:
            return (loss, pred)
        return {'loss': loss, 'logits': pred}

def processor_for_finetune(base_model: str, new_chars: List[str], output_dir: Path, resume_ckpt_dir: Optional[Path]) -> Wav2Vec2BertProcessor:
    return load_or_build_processor(base_model, new_chars, output_dir, resume_ckpt_dir)

def precision_flags(pref: str) -> Tuple[bool, bool]:
    _cuda_ok = torch.cuda.is_available()
    _bf16_supported = bool(_cuda_ok and torch.cuda.is_bf16_supported())
    if pref == 'fp32':
        return (False, False)
    if pref == 'bf16':
        if not _bf16_supported:
            raise RuntimeError('--precision bf16, но bf16 на GPU недоступен.')
        return (True, False)
    if pref == 'auto':
        return (False, bool(_cuda_ok))
    return (False, bool(_cuda_ok))

def load_encoder_weights_into_ctc(model: Wav2Vec2BertForCTC, pretrained_dir: Path) -> Dict[str, Any]:
    pretrained_dir = pretrained_dir.resolve()
    st_path = pretrained_dir / 'model.safetensors'
    pt_path = pretrained_dir / 'pytorch_model.bin'
    state: Dict[str, torch.Tensor]
    if st_path.is_file():
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
    info = {'pretrained_checkpoint_dir': str(pretrained_dir), 'weights_loaded_prefix_only': prefix.rstrip('.'), 'note_ru': 'Ключи perceptron_decoder.* из предобучения не используются в CTC.', 'encoder_missing_keys_count': len(res.missing_keys), 'encoder_unexpected_keys_count': len(res.unexpected_keys), 'encoder_missing_keys_sample': res.missing_keys[:30], 'encoder_unexpected_keys_sample': res.unexpected_keys[:30]}
    logger.info('Загрузка энкодера из %s: missing=%d unexpected=%d', pretrained_dir, len(res.missing_keys), len(res.unexpected_keys))
    return info

def _trainer_state_json_ok(checkpoint_dir: Path) -> bool:
    ts = checkpoint_dir / 'trainer_state.json'
    try:
        raw = ts.read_text(encoding='utf-8').strip()
        if not raw:
            return False
        json.loads(raw)
        return True
    except (OSError, json.JSONDecodeError):
        return False

def find_latest_valid_checkpoint_dir_for_resume(output_dir: Path) -> Optional[Path]:
    candidates: List[Tuple[int, Path]] = []
    for p in output_dir.glob('checkpoint-*'):
        if not p.is_dir():
            continue
        suf = p.name.split('-')[-1]
        if not suf.isdigit():
            continue
        if not _trainer_state_json_ok(p):
            logger.warning('Пропуск %s: нет валидного trainer_state.json (возможна незаконченная запись после сбоя).', p)
            continue
        candidates.append((int(suf), p))
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]

def run_pretraining(args: argparse.Namespace, pretrain_root: Path) -> Path:
    torch.manual_seed(args.pretrain_seed)
    np.random.seed(args.pretrain_seed)
    artifact_dir = pretrain_root / 'run_artifacts'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = collect_wav_paths(list(args.pretrain_audio_roots))
    if not paths:
        raise RuntimeError('Нет ни одного .wav под --pretrain_audio_roots. Дождитесь загрузки данных или поправьте пути.')
    if args.pretrain_max_samples:
        paths = paths[:int(args.pretrain_max_samples)]
    rng = np.random.RandomState(args.pretrain_seed)
    n = len(paths)
    iv = max(1, int(round(n * float(args.pretrain_validation_ratio))))
    idx = np.arange(n)
    rng.shuffle(idx)
    val_idx = set(idx[:iv].tolist())
    train_paths = [paths[i] for i in range(n) if i not in val_idx]
    val_paths = [paths[i] for i in range(n) if i in val_idx]
    ds_tr = Dataset.from_dict({'path': train_paths})
    ds_va = Dataset.from_dict({'path': val_paths})
    fe = SeamlessM4TFeatureExtractor.from_pretrained(args.base_model_name_or_path)
    cfg = Wav2Vec2BertConfig.from_pretrained(args.base_model_name_or_path)
    cfg.mask_time_prob = max(float(cfg.mask_time_prob), 0.0001)
    cfg.mask_feature_prob = 0.0
    cfg.add_adapter = False
    model = Wav2Vec2BertForMaskedFramePretraining(cfg, decoder_num_layers=int(args.pretrain_decoder_num_layers), decoder_hidden_dim=int(args.pretrain_perceptron_hidden_dim))
    bm = Wav2Vec2BertModel.from_pretrained(args.base_model_name_or_path)
    model.wav2vec2_bert.load_state_dict(bm.state_dict(), strict=True)
    model.pretrain_mask_time_prob = float(args.pretrain_mask_time_prob)
    model.pretrain_mask_time_length = int(args.pretrain_mask_time_length)
    min_audio_samples = min_wav_samples_for_frames(fe, min_mel_frames=2)
    map_kw = dict(batched=True, batch_size=8, fn_kwargs={'feature_extractor': fe, 'min_audio_samples': min_audio_samples})
    train_ds = ds_tr.map(prepare_pretrain_batch, remove_columns=ds_tr.column_names, **map_kw)
    eval_ds = ds_va.map(prepare_pretrain_batch, remove_columns=ds_va.column_names, **map_kw)
    data_collator = DataCollatorAudioOnly(feature_extractor=fe)
    use_gc = torch.cuda.is_available()
    model.gradient_checkpointing_enable()
    report_to: List[str] = []
    if not args.disable_wandb and os.environ.get('WANDB_DISABLED', '').lower() not in ('1', 'true', 'yes'):
        report_to.append('wandb')
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or 'pretrain', config=vars(args))
    _ta_params = inspect.signature(TrainingArguments.__init__).parameters
    _eval_kw = 'eval_strategy' if 'eval_strategy' in _ta_params else 'evaluation_strategy'
    _group_kw: Dict[str, Any] = {}
    if 'train_sampling_strategy' in _ta_params:
        _group_kw['train_sampling_strategy'] = 'group_by_length'
    elif 'group_by_length' in _ta_params:
        _group_kw['group_by_length'] = True
    use_bf16, use_fp16 = precision_flags(args.precision)
    save_limit = None if args.pretrain_save_total_limit in (-1, 0) else int(args.pretrain_save_total_limit)
    plateau_state: Dict[str, Any] = {'enabled': not bool(args.pretrain_disable_eval_plateau_early_stop), 'early_stopped': False, 'early_stop_reason': None, 'min_global_step': int(args.pretrain_plateau_min_steps), 'patience': int(args.pretrain_plateau_patience), 'rel_improvement_min': float(args.pretrain_plateau_rel_improvement_min)}
    milestone_tuple = tuple(sorted({int(s) for s in args.pretrain_snapshot_milestone_steps}))
    snap_bundle = PretrainSnapshotBundle(root=pretrain_root / 'snapshots', milestone_steps=milestone_tuple)
    pretrain_callbacks: List[TrainerCallback] = [PretrainMilestoneSnapshotCallback(snap_bundle)]
    if plateau_state['enabled']:
        pretrain_callbacks.append(PretrainEvalPlateauEarlyStopping(metric_key='eval_loss', min_global_step=int(args.pretrain_plateau_min_steps), patience=int(args.pretrain_plateau_patience), rel_improvement_min=float(args.pretrain_plateau_rel_improvement_min), plateau_state=plateau_state, snapshot_bundle=snap_bundle))
    pretrain_max_steps_cap = int(args.pretrain_max_steps)
    max_steps_kw: Dict[str, Any] = {}
    if pretrain_max_steps_cap > 0:
        max_steps_kw['max_steps'] = pretrain_max_steps_cap
    training_args = TrainingArguments(output_dir=str(pretrain_root), **_group_kw, remove_unused_columns=False, label_names=[], per_device_train_batch_size=args.pretrain_per_device_train_batch_size, per_device_eval_batch_size=args.pretrain_per_device_eval_batch_size, gradient_accumulation_steps=args.pretrain_gradient_accumulation_steps, **{_eval_kw: 'steps'}, save_strategy='steps', num_train_epochs=float(args.pretrain_num_train_epochs), **max_steps_kw, bf16=use_bf16, fp16=use_fp16, save_steps=args.pretrain_save_steps, eval_steps=args.pretrain_eval_steps, logging_steps=args.pretrain_logging_steps, learning_rate=float(args.pretrain_learning_rate), lr_scheduler_type=str(args.pretrain_lr_scheduler_type), weight_decay=float(args.pretrain_weight_decay), warmup_ratio=float(args.pretrain_warmup_ratio), save_total_limit=save_limit, load_best_model_at_end=True, metric_for_best_model='eval_loss', greater_is_better=False, report_to=report_to, push_to_hub=False, gradient_checkpointing=use_gc, seed=int(args.pretrain_seed), max_grad_norm=1.0)
    trainer = Trainer(model=model, data_collator=data_collator, args=training_args, train_dataset=train_ds, eval_dataset=eval_ds, callbacks=pretrain_callbacks)
    snap_bundle.bind_trainer(trainer)
    if pretrain_max_steps_cap > 0:
        logger.info('Предобучение: max_steps=%d (HF ограничивает общее число шагов; эпохи могут не завершиться полностью).', pretrain_max_steps_cap)
    if plateau_state['enabled']:
        logger.info('Предобучение: ранняя остановка по плато eval_loss — после global_step≥%d нужно %d подряд «медленных» eval (относит. улучшение < %s между соседними оценками).', int(args.pretrain_plateau_min_steps), int(args.pretrain_plateau_patience), float(args.pretrain_plateau_rel_improvement_min))
    else:
        logger.info('Предобучение: плато early-stop выключен (--pretrain_disable_eval_plateau_early_stop).')
    pretrain_resume = parse_resume_from_checkpoint_arg(args.pretrain_resume_from_checkpoint)
    if pretrain_resume is True:
        valid_dir = find_latest_valid_checkpoint_dir_for_resume(pretrain_root)
        if valid_dir is None:
            raise RuntimeError(f'В {pretrain_root.resolve()} нет ни одного пригодного checkpoint-* (нужен непустой валидный trainer_state.json). Удалите/переименуйте битую последнюю папку или укажите путь явно, напр. --pretrain_resume_from_checkpoint ./w2v-bert-pipeline-out/pretrain/checkpoint-24500')
        pretrain_resume = str(valid_dir.resolve())
        logger.info('Предобучение: last → последний целый чекпоинт %s', pretrain_resume)
    elif isinstance(pretrain_resume, str):
        ck = Path(pretrain_resume).expanduser().resolve()
        if not ck.is_dir():
            raise RuntimeError(f'Нет каталога чекпоинта: {ck}')
        if not _trainer_state_json_ok(ck):
            raise RuntimeError(f'{ck}: пустой или битый trainer_state.json — нельзя resume. Выберите предыдущий checkpoint-*.')
        pretrain_resume = str(ck)
        logger.info('Предобучение: resume_from_checkpoint=%s', pretrain_resume)
    trainer.train(resume_from_checkpoint=pretrain_resume)
    if not snap_bundle.plateau_saved:
        snap_bundle.save_subdir('plateau_proxy_best_eval', reason='early-stop по плато не сработал; веса после train = лучший eval_loss (load_best_model_at_end)')
    plateau_snap_dir = snap_bundle.root / 'plateau_early_stop'
    proxy_snap_dir = snap_bundle.root / 'plateau_proxy_best_eval'
    hf_best = getattr(trainer.state, 'best_model_checkpoint', None) or ''
    if plateau_state.get('early_stopped') and plateau_snap_dir.is_dir():
        best_src = str(plateau_snap_dir.resolve())
        encoder_pick_reason = 'pretrain/snapshots/plateau_early_stop — ранний стоп: eval_loss перестал быстро падать; этот снимок идёт в CTC.'
    elif proxy_snap_dir.is_dir():
        best_src = str(proxy_snap_dir.resolve())
        encoder_pick_reason = 'pretrain/snapshots/plateau_proxy_best_eval — плато не сработало или контроль выключен; веса после обучения = лучший eval_loss (load_best_model_at_end).'
    elif hf_best and Path(hf_best).is_dir():
        best_src = hf_best
        encoder_pick_reason = 'HF checkpoint по минимальному eval_loss (резерв: нет каталогов snapshots/plateau_*).'
    else:
        best_src = ''
        encoder_pick_reason = ''
    best_dst = pretrain_root / 'best_pretrained_model'
    if best_src and Path(best_src).is_dir():
        if best_dst.exists():
            shutil.rmtree(best_dst, ignore_errors=True)
        shutil.copytree(best_src, best_dst)
        logger.info('Энкодер для downstream (best_pretrained_model) скопирован из: %s', best_src)
        logger.info('%s', encoder_pick_reason)
    else:
        raise RuntimeError('Не удалось собрать best_pretrained_model: нет snapshots/plateau_early_stop, plateau_proxy_best_eval и HF best_model_checkpoint.')
    best_hf_eval_loss_checkpoint = hf_best if hf_best and Path(hf_best).is_dir() else None
    last_hf_mirror_dir = artifact_dir / 'last_hf_training_checkpoint'
    last_hf_src = find_latest_valid_checkpoint_dir_for_resume(pretrain_root)
    last_hf_training_checkpoint_mirror: Optional[str] = None
    last_hf_training_checkpoint_source_dir: Optional[str] = None
    if last_hf_src is not None and last_hf_src.is_dir():
        last_hf_training_checkpoint_source_dir = str(last_hf_src.resolve())
        if last_hf_mirror_dir.exists():
            shutil.rmtree(last_hf_mirror_dir, ignore_errors=True)
        shutil.copytree(last_hf_src, last_hf_mirror_dir)
        last_hf_training_checkpoint_mirror = str(last_hf_mirror_dir.resolve())
        logger.info('Сохранена копия последнего HF checkpoint (до возможной очистки checkpoint-*): %s → %s', last_hf_src.name, last_hf_mirror_dir)
    elif last_hf_src is None:
        logger.warning('Не найден действительный checkpoint-* для зеркальной копии последнего шага (нет валидного trainer_state.json?).')
    try:
        plots = save_training_plots(trainer.state.log_history, artifact_dir / 'plots')
    except Exception as e:
        logger.warning('Графики предобучения: %s', e)
        plots = []
    pre_info = {'stage': 'pretrain_encoder_plus_perceptron_decoder', 'advisor_protocol_ru': 'К энкодеру прикреплён MLP-декодер предобучения; обучение на аудио train+неразмеченные; перед CTC декодер отбрасывается (в downstream грузится только wav2vec2_bert).', 'pretrain_decoder_num_layers': int(args.pretrain_decoder_num_layers), 'pretrain_decoder_hidden_dim_arg': int(args.pretrain_perceptron_hidden_dim), 'pretrain_decoder_hidden_dim_effective': int(cfg.hidden_size) if int(args.pretrain_perceptron_hidden_dim) <= 0 else int(args.pretrain_perceptron_hidden_dim), 'perceptron_decoder': f'{int(args.pretrain_decoder_num_layers)}_linear_layers_with_gelu_between' if int(args.pretrain_decoder_num_layers) > 1 else 'linear_single_layer', 'pretrain_perceptron_hidden_dim': int(args.pretrain_perceptron_hidden_dim), 'loss_description_ru': 'Маскирование по времени; MSE между выходом перцептрон-декодера и отсоединёнными extract_features на замаскированных позициях.', 'base_model_name_or_path': args.base_model_name_or_path, 'pretrain_audio_roots': [str(Path(x).resolve()) for x in args.pretrain_audio_roots], 'num_wav_files_total': n, 'num_train_files': len(train_paths), 'num_val_files': len(val_paths), 'validation_ratio': float(args.pretrain_validation_ratio), 'pretrain_mask_time_prob': float(args.pretrain_mask_time_prob), 'pretrain_mask_time_length': int(args.pretrain_mask_time_length), 'min_audio_samples': int(min_audio_samples), 'best_model_checkpoint_original': best_src, 'best_pretrained_model_dir': str(best_dst.resolve()), 'finetune_encoder_source_description_ru': encoder_pick_reason, 'hf_best_eval_loss_checkpoint_dir': best_hf_eval_loss_checkpoint, 'best_eval_loss': getattr(trainer.state, 'best_metric', None), 'global_step': getattr(trainer.state, 'global_step', None), 'precision_requested': args.precision, 'bf16': use_bf16, 'fp16': use_fp16, 'training_args_nondefault': {'num_train_epochs': args.pretrain_num_train_epochs, 'learning_rate': args.pretrain_learning_rate, 'lr_scheduler_type': args.pretrain_lr_scheduler_type, 'weight_decay': args.pretrain_weight_decay, 'per_device_train_batch_size': args.pretrain_per_device_train_batch_size, 'gradient_accumulation_steps': args.pretrain_gradient_accumulation_steps, 'save_steps': args.pretrain_save_steps, 'eval_steps': args.pretrain_eval_steps}, 'pretrain_max_steps': pretrain_max_steps_cap if pretrain_max_steps_cap > 0 else None, 'eval_plateau_early_stop': dict(plateau_state), 'pretrain_snapshots_dir': str(snap_bundle.root.resolve()), 'pretrain_snapshot_milestone_steps': list(milestone_tuple), 'pretrain_snapshot_writes': dict(snap_bundle.written), 'plots_saved': plots, 'trainer_log_history_path': str((artifact_dir / 'pretrain_log_history.json').resolve()), 'last_hf_training_checkpoint_source_dir': last_hf_training_checkpoint_source_dir, 'last_hf_training_checkpoint_mirror_dir': last_hf_training_checkpoint_mirror, **model_run_info(trainer.model)}
    (artifact_dir / 'pretrain_log_history.json').write_text(json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    (artifact_dir / 'pretraining_info.json').write_text(json.dumps(pre_info, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    if not args.pretrain_keep_checkpoints:
        removed = remove_intermediate_checkpoints(pretrain_root)
        pre_info['checkpoint_dirs_removed_after_pretrain'] = removed
        (artifact_dir / 'pretraining_info.json').write_text(json.dumps(pre_info, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
        if best_dst.exists():
            logger.info('Промежуточные checkpoint-* удалены; best_pretrained_model сохранён.')
    if report_to:
        try:
            wandb.finish()
        except Exception:
            logger.exception('wandb.finish (pretrain)')
    return best_dst.resolve()

def run_finetune(args: argparse.Namespace, finetune_root: Path, encoder_ckpt_dir: Path) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_root = Path(args.train_dir)
    test_root = Path(args.test_dir)
    artifact_dir = finetune_root / 'run_artifacts'
    plots_dir = artifact_dir / 'plots'
    artifact_dir.mkdir(parents=True, exist_ok=True)
    raw_train = load_split(train_root)
    raw_test = load_split(test_root)
    if args.finetune_max_train_samples:
        raw_train = raw_train.select(range(min(args.finetune_max_train_samples, len(raw_train))))
    combined_for_vocab = Dataset.from_dict({'text': list(raw_train['text']) + list(raw_test['text'])})
    new_chars = extract_all_chars(combined_for_vocab)
    resume_ckpt = parse_resume_from_checkpoint_arg(args.finetune_resume_from_checkpoint)
    resume_ckpt_dir = resolve_resume_checkpoint_dir(finetune_root, resume_ckpt)
    processor = processor_for_finetune(args.base_model_name_or_path, new_chars, finetune_root, resume_ckpt_dir)
    use_adapter = not bool(args.finetune_no_adapter)
    model = Wav2Vec2BertForCTC.from_pretrained(args.base_model_name_or_path, mask_time_prob=float(args.finetune_mask_time_prob), layerdrop=0.0, ctc_loss_reduction='mean', ctc_zero_infinity=True, add_adapter=use_adapter, pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer), ignore_mismatched_sizes=True)
    enc_load = load_encoder_weights_into_ctc(model, encoder_ckpt_dir)
    if use_adapter:
        logger.info('add_adapter=True: блоки adapter в CTC инициализированы заново.')
    if args.finetune_freeze_base_model:
        for p in model.wav2vec2_bert.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = True
    elif args.finetune_freeze_feature_projection:
        fp_mod = model.wav2vec2_bert.feature_projection
        for p in fp_mod.parameters():
            p.requires_grad = False
    use_gradient_checkpointing = not bool(args.finetune_no_gradient_checkpointing)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    min_mel_frames = 2
    if float(args.finetune_mask_time_prob) > 0:
        min_mel_frames = max(min_mel_frames, int(getattr(model.config, 'mask_time_length', 10)) + 1)
    min_audio_samples = min_wav_samples_for_frames(processor.feature_extractor, min_mel_frames)
    _map_kw = dict(batched=True, batch_size=8, fn_kwargs={'processor': processor, 'min_audio_samples': min_audio_samples})
    train_ds = raw_train.map(prepare_w2v_bert_batch, remove_columns=raw_train.column_names, **_map_kw)
    test_ds = raw_test.map(prepare_w2v_bert_batch, remove_columns=raw_test.column_names, **_map_kw)
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    compute_metrics = build_compute_metrics(processor)
    report_to: List[str] = []
    if not args.disable_wandb and os.environ.get('WANDB_DISABLED', '').lower() not in ('1', 'true', 'yes'):
        report_to.append('wandb')
        wandb.init(project=args.wandb_project, name=args.wandb_run_name or 'finetune', config=vars(args))
    _ta_params = inspect.signature(TrainingArguments.__init__).parameters
    _eval_kw = 'eval_strategy' if 'eval_strategy' in _ta_params else 'evaluation_strategy'
    _group_kw: Dict[str, Any] = {}
    if 'train_sampling_strategy' in _ta_params:
        _group_kw['train_sampling_strategy'] = 'group_by_length'
    elif 'group_by_length' in _ta_params:
        _group_kw['group_by_length'] = True
    use_bf16, use_fp16 = precision_flags(args.precision)
    save_limit = None if args.finetune_save_total_limit in (-1, 0) else int(args.finetune_save_total_limit)
    _warmup_kw: Dict[str, Any] = {}
    if int(args.finetune_warmup_steps) > 0:
        _warmup_kw['warmup_steps'] = int(args.finetune_warmup_steps)
    else:
        _warmup_kw['warmup_ratio'] = float(args.finetune_warmup_ratio)
    max_steps_cap = int(args.finetune_max_steps)
    max_steps_kw: Dict[str, Any] = {}
    if max_steps_cap > 0:
        max_steps_kw['max_steps'] = max_steps_cap
    training_args = TrainingArguments(output_dir=str(finetune_root), **_group_kw, remove_unused_columns=False, per_device_train_batch_size=args.finetune_per_device_train_batch_size, per_device_eval_batch_size=args.finetune_per_device_eval_batch_size, gradient_accumulation_steps=args.finetune_gradient_accumulation_steps, **{_eval_kw: 'steps'}, save_strategy='steps', num_train_epochs=args.finetune_num_train_epochs, **max_steps_kw, bf16=use_bf16, fp16=use_fp16, save_steps=args.finetune_save_steps, eval_steps=args.finetune_eval_steps, logging_steps=args.finetune_logging_steps, learning_rate=args.finetune_learning_rate, lr_scheduler_type=str(args.finetune_lr_scheduler_type), weight_decay=float(args.finetune_weight_decay), **_warmup_kw, save_total_limit=save_limit, load_best_model_at_end=True, metric_for_best_model='eval_cer', greater_is_better=False, report_to=report_to, push_to_hub=False, gradient_checkpointing=use_gradient_checkpointing, seed=args.seed, max_grad_norm=1.0)
    if max_steps_cap > 0:
        logger.info('CTC (размеченные данные): максимум %d шагов оптимизатора (global_step после предобучения считается с нуля); при max_steps>0 HF не ориентируется на num_train_epochs.', max_steps_cap)
    else:
        logger.info('CTC (размеченные данные): лимит по эпохам — num_train_epochs=%s при max_steps=0.', args.finetune_num_train_epochs)
    trainer = Trainer(model=model, data_collator=data_collator, args=training_args, train_dataset=train_ds, eval_dataset=test_ds, processing_class=processor, compute_metrics=compute_metrics)
    if args.finetune_debug_trainable_grads:
        debug_log_trainable_gradients(trainer)
    trainer.train(resume_from_checkpoint=resume_ckpt)
    try:
        plots_saved = save_training_plots(trainer.state.log_history, plots_dir)
    except Exception as e:
        logger.warning('Графики: %s', e)
        plots_saved = []
    try:
        (artifact_dir / 'finetune_trainer_log_history.json').write_text(json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    except Exception as e:
        logger.warning('finetune_trainer_log_history.json: %s', e)
    logger.info('Финальная оценка greedy train/test.')
    train_eval = trainer.evaluate(train_ds, metric_key_prefix='greedy_train')
    test_eval = trainer.evaluate(test_ds, metric_key_prefix='greedy_test')
    best_ckpt = getattr(trainer.state, 'best_model_checkpoint', None) or ''
    best_ft_dst = finetune_root / 'best_finetuned_model'
    if best_ckpt and Path(best_ckpt).is_dir():
        if best_ft_dst.exists():
            shutil.rmtree(best_ft_dst, ignore_errors=True)
        shutil.copytree(best_ckpt, best_ft_dst)
        logger.info('Лучший чекпоинт после CTC скопирован: %s', best_ft_dst)
    final_mdir = finetune_root / 'final_model'
    trainer.save_model(str(final_mdir))
    processor.save_pretrained(str(final_mdir))
    greedy_pack = {'train': (train_eval.get('greedy_train_wer', train_eval.get('eval_wer', float('nan'))), train_eval.get('greedy_train_cer', train_eval.get('eval_cer', float('nan')))), 'test': (test_eval.get('greedy_test_wer', test_eval.get('eval_wer', float('nan'))), test_eval.get('greedy_test_cer', test_eval.get('eval_cer', float('nan'))))}
    beam_pack: Dict[str, Tuple[float, float]] = {}
    beam_details: Dict[str, Any] = {}
    if args.finetune_beam_eval:
        try:
            vocab_labels = labels_aligned_for_pyctc(processor)
            decoder = build_ctcdecoder(vocab_labels)
            trainer.model.eval()
            bw = int(args.finetune_beam_width)
            chunk_n = int(args.finetune_beam_logits_chunk_samples)
            for split_key, ds in (('train', train_ds), ('test', test_ds)):
                if chunk_n and chunk_n > 0:
                    logits, lids = predict_logits_and_refs_chunked(trainer, ds, processor, chunk_n)
                else:
                    logits, lids = predict_logits_and_refs(trainer, ds, processor, need_full_logits=True)
                logits = logits.astype(np.float32, copy=False)
                wer_b, cer_b = beam_metrics_from_predictions(logits, lids, processor, decoder, bw)
                beam_pack[split_key] = (wer_b, cer_b)
                beam_details[f'beam_{split_key}'] = {'wer': wer_b, 'cer': cer_b, 'beam_width': bw}
                logger.info('Beam %s: WER=%.6f CER=%.6f (beam_width=%s)', split_key, wer_b, cer_b, bw)
        except Exception as e:
            logger.exception('Beam search: %s', e)
            beam_details['beam_error'] = repr(e)
    run_meta = {'stage': 'finetune_ctc', 'base_pretrained_model': args.base_model_name_or_path, 'encoder_initialized_from_pretrain_dir': str(encoder_ckpt_dir.resolve()), 'encoder_weight_load_report': enc_load, 'add_adapter': use_adapter, 'freeze_base_model': bool(args.finetune_freeze_base_model), 'freeze_feature_projection': bool(args.finetune_freeze_feature_projection), 'gradient_checkpointing': use_gradient_checkpointing, 'precision_mode': args.precision, 'bf16_enabled': use_bf16, 'fp16_enabled': use_fp16, 'warmup_training_kwargs': _warmup_kw, 'weight_decay': float(args.finetune_weight_decay), 'lr_scheduler_type': str(args.finetune_lr_scheduler_type), 'finetune_max_steps_cap': max_steps_cap if max_steps_cap > 0 else None, 'finetune_num_train_epochs_fallback': float(args.finetune_num_train_epochs) if max_steps_cap <= 0 else None, 'finetune_schedule_note_ru': f'Дообучение на размеченных данных остановится после {max_steps_cap} шагов оптимизатора.' if max_steps_cap > 0 else f'Дефолт: до {float(args.finetune_num_train_epochs):g} эпох по train (нет фикса max_steps).', 'mask_time_prob': float(args.finetune_mask_time_prob), 'best_model_checkpoint_path_before_cleanup': best_ckpt, 'best_finetuned_model_dir': str(best_ft_dst.resolve()) if best_ft_dst.exists() else None, 'beam_width_used': int(args.finetune_beam_width) if args.finetune_beam_eval else None, 'metric_for_best_model': 'eval_cer', 'best_metric_during_training': getattr(trainer.state, 'best_metric', None), 'trainer_global_steps': getattr(trainer.state, 'global_step', None), 'train_samples': len(raw_train), 'test_samples': len(raw_test), **model_run_info(trainer.model), 'plots_saved': plots_saved, 'final_model_dir': str(final_mdir.resolve()), 'artifacts_dir': str(artifact_dir.resolve())}
    if not args.finetune_keep_checkpoints:
        removed = remove_intermediate_checkpoints(finetune_root)
        run_meta['checkpoint_dirs_removed_after_train'] = removed
    excel_path = artifact_dir / 'best_model_wer_cer_greedy.xlsx'
    try:
        export_metrics_excel(excel_path, greedy={k: greedy_pack[k] for k in greedy_pack}, meta=run_meta, beam=beam_pack if beam_pack else None)
    except Exception as e:
        logger.warning('Excel: %s', e)
    metrics_json = artifact_dir / 'metrics_summary.json'
    summary: Dict[str, Any] = {'greedy': {'train': {'wer': greedy_pack['train'][0], 'cer': greedy_pack['train'][1]}, 'test': {'wer': greedy_pack['test'][0], 'cer': greedy_pack['test'][1]}}, 'meta': run_meta}
    if beam_pack:
        summary['beam'] = {split: {'wer': beam_pack.get(split, (np.nan, np.nan))[0], 'cer': beam_pack.get(split, (np.nan, np.nan))[1]} for split in ('train', 'test')}
    if beam_details:
        summary['beam_extra'] = beam_details
    metrics_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    (artifact_dir / 'finetuning_info.json').write_text(json.dumps(run_meta, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    if report_to:
        try:
            wandb.finish()
        except Exception:
            logger.exception('wandb.finish (finetune)')
    return summary

from typing import Optional
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
logger = logging.getLogger(__name__)

def parse_non_urmi_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument('--output_root', type=str, default=None)
    ap.add_argument('--base_model_name_or_path', type=str, default='facebook/w2v-bert-2.0')
    ap.add_argument('--disable_wandb', action='store_true')
    ap.add_argument('--pretrain_resume_from_checkpoint', type=str, default=None)
    ap.add_argument('--finetune_resume_from_checkpoint', type=str, default=None)
    ap.add_argument('--skip_pretrain', action='store_true')
    ap.add_argument('--skip_finetune', action='store_true')
    return ap.parse_args()

def run_non_urmi_pretrain(repo: Path, output_root: Path, base_model: str, disable_wandb: bool, resume: Optional[str]) -> None:
    roots = [repo / '5h_non-urmi_pretraining', repo / '5h_urmi_pretraining', repo / '5h_train' / 'audios']
    missing = [p for p in roots if not p.is_dir()]
    if missing:
        raise FileNotFoundError('Missing pretrain audio dirs: ' + ', '.join(map(str, missing)))
    pre_args = parse_args()
    pre_args.phase = 'pretrain_only'
    pre_args.output_dir = str(output_root.resolve())
    pre_args.base_model_name_or_path = base_model
    pre_args.disable_wandb = disable_wandb
    pre_args.pretrain_audio_roots = [str(p.resolve()) for p in roots]
    pre_args.pretrain_num_train_epochs = 14.0
    pre_args.pretrain_eval_steps = 500
    pre_args.pretrain_save_steps = 500
    pre_args.pretrain_plateau_min_steps = 12000
    pre_args.pretrain_snapshot_milestone_steps = [20000, 40000, 65000]
    pre_args.pretrain_gradient_accumulation_steps = 8
    pre_args.wandb_run_name = 'non_urmi_ssl_pretrain'
    if resume:
        pre_args.pretrain_resume_from_checkpoint = resume
    pretrain_root = output_root / 'pretrain'
    pretrain_root.mkdir(parents=True, exist_ok=True)
    run_pretraining(pre_args, pretrain_root)

def run_non_urmi_finetune(repo: Path, output_root: Path, base_model: str, disable_wandb: bool, resume: Optional[str]) -> None:
    encoder_dir = output_root / 'pretrain' / 'best_pretrained_model'
    if not encoder_dir.is_dir():
        raise FileNotFoundError(f'Missing encoder after pretrain: {encoder_dir}')
    ft_out = output_root / 'finetune_experiment_2_augment_style'
    p = build_parser()
    p.set_defaults(model_name_or_path=base_model, pretrained_encoder_dir=str(encoder_dir.resolve()), train_dir=str((repo / '5h_train').resolve()), test_dir=str((repo / '5h_test').resolve()), output_dir=str(ft_out.resolve()), experiment_name='w2v2-bert_non-urmi-pretrained_speed-preturbation', train_augment_mode='speed', mask_time_prob=0.0, mask_feature_prob=0.0, speed_factors='0.9,1.1', speed_prob=0.5, add_adapter=True, ctc_zero_infinity=True, lr_scheduler_type='cosine', weight_decay=0.01, learning_rate=1.5e-05, warmup_ratio=0.1, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8, max_grad_norm=0.5, num_train_epochs=30.0, wandb_run_name='non_urmi_ft_exp2_augment_style', precision='fp16', disable_wandb=disable_wandb)
    ns = p.parse_args([])
    if resume:
        ns.resume_from_checkpoint = resume
    run_training(ns)

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    cli = parse_non_urmi_args()
    repo = Path(__file__).resolve().parent
    out_root = Path(cli.output_root).expanduser().resolve() if cli.output_root else repo / 'non_urmi_pretrain'
    out_root.mkdir(parents=True, exist_ok=True)
    if not cli.skip_pretrain:
        run_non_urmi_pretrain(repo, out_root, cli.base_model_name_or_path, cli.disable_wandb, cli.pretrain_resume_from_checkpoint)
    if not cli.skip_finetune:
        run_non_urmi_finetune(repo, out_root, cli.base_model_name_or_path, cli.disable_wandb, cli.finetune_resume_from_checkpoint)
    summary = {'pretrain_root': str((out_root / 'pretrain').resolve()), 'best_encoder': str((out_root / 'pretrain' / 'best_pretrained_model').resolve()), 'finetune_out': str((out_root / 'finetune_experiment_2_augment_style').resolve())}
    (out_root / 'pipeline_non_urmi_manifest.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    logger.info('Done: %s', out_root)

if __name__ == "__main__":
    main()
