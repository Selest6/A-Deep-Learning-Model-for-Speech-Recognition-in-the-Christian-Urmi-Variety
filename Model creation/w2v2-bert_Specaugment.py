from __future__ import annotations
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

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run_training(build_parser().parse_args())

def main() -> None:
    logging.basicConfig(level=logging.INFO)
    p = build_parser()
    p.set_defaults(output_dir='./experiment_1_encoder_specaugment', experiment_name='experiment_1_encoder_specaugment', train_augment_mode='none', mask_time_prob=0.05, mask_feature_prob=0.02, add_adapter=True, copy_best_checkpoint_to='shared_best_checkpoints')
    run_training(p.parse_args())

if __name__ == "__main__":
    main()
