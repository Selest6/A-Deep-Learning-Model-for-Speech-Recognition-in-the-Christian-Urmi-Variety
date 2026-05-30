from __future__ import annotations
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
from asr_training_common import build_compute_metrics, decode_beam_batch_wav2vec2, decode_refs_from_labels, export_metrics_excel, extract_all_chars, labels_aligned_for_pyctc, load_split, load_audio_mono, model_run_info, parse_resume_from_checkpoint_arg, predict_logits_and_refs, remove_intermediate_checkpoints, resolve_resume_checkpoint_dir, save_training_plots, debug_log_trainable_gradients, wer_cer_from_strings
logger = logging.getLogger(__name__)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Fine-tune Wav2Vec2-BERT CTC ASR (w2v-bert-2.0) for Urmia.')
    p.add_argument('--model_name_or_path', type=str, default='facebook/w2v-bert-2.0', help='Предобученный Wav2Vec2-BERT (ASR требует дообучения CTC-головы).')
    p.add_argument('--train_dir', type=str, required=True)
    p.add_argument('--test_dir', type=str, required=True)
    p.add_argument('--output_dir', type=str, default='./w2v-bert-urmi-out')
    p.add_argument('--num_train_epochs', type=float, default=30.0)
    p.add_argument('--max_steps', type=int, default=-1, help='Как у asr_training_common.py: при >0 ограничивает глобальные шаги (перекрывает num_train_epochs). Для сопоставимости с MMS по числу шагов оптимизатора задайте --max_steps 22000 (в артефакте trainer_global_steps может быть 22001 после финального eval).')
    p.add_argument('--per_device_train_batch_size', type=int, default=2)
    p.add_argument('--per_device_eval_batch_size', type=int, default=2)
    p.add_argument('--learning_rate', type=float, default=2e-05, help='Как у MMS-артефакта metrics_summary.json (полный FT на малом корпусе).')
    p.add_argument('--lr_scheduler_type', type=str, default='cosine', help='Как у MMS (TrainingArguments).')
    p.add_argument('--weight_decay', type=float, default=0.01, help='AdamW weight decay, как у MMS.')
    p.add_argument('--warmup_steps', type=int, default=0, help='При 0 используется только warmup_ratio (как MMS). Если >0 — приоритет над warmup_ratio.')
    p.add_argument('--warmup_ratio', type=float, default=0.12, help='Доля warmup при warmup_steps==0 (как в MMS run_artifacts).')
    p.add_argument('--gradient_accumulation_steps', type=int, default=2)
    p.add_argument('--precision', type=str, choices=('fp16', 'auto', 'bf16', 'fp32'), default='fp16', help='fp16 по умолчанию (GradScaler на CUDA). auto: на CUDA — fp16, без CUDA — fp32. bf16 только по явному запросу (на части прогонов давал NaN grad_norm). Если снова NaN — попробуйте --precision fp32.')
    p.add_argument('--max_train_samples', type=int, default=None)
    p.add_argument('--save_steps', type=int, default=500)
    p.add_argument('--eval_steps', type=int, default=500)
    p.add_argument('--logging_steps', type=int, default=50)
    p.add_argument('--mask_time_prob', type=float, default=0.0, help='Во время fine-tune SpecAugment по времени внутри энкодера. 0 — как в блоге HF для w2v-bert; >0 — поднимайте минимальную длину аудио (см. лог после старта).')
    p.add_argument('--add_adapter', action='store_true', help='Включить adapter в энкодере (рецепт HF для w2v-bert). По умолчанию выключено — полный fine-tune, как у MMS без --train_adapters_only (релевантное сравнение режимов дообучения).')
    p.add_argument('--freeze_base_model', action='store_true', help='Заморозить wav2vec2_bert; обучается только lm_head (быстро, но обычно хуже на ASR).')
    p.add_argument('--freeze_feature_projection', action='store_true', help='Заморозить convolutional frontend (feature_projection); иногда стабилизирует старт при полном fine-tune.')
    p.add_argument('--no_gradient_checkpointing', action='store_true', help='Отключить gradient checkpointing (больше VRAM, иногда стабильнее градиенты; обычно не нужно при bf16).')
    p.add_argument('--wandb_project', type=str, default='w2v-bert-urmi-asr')
    p.add_argument('--wandb_run_name', type=str, default=None)
    p.add_argument('--disable_wandb', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save_total_limit', type=int, default=3)
    p.add_argument('--keep_intermediate_checkpoints', action='store_true')
    p.add_argument('--beam_width', type=int, default=10)
    p.add_argument('--beam_logits_chunk_samples', type=int, default=256, help='Beam по кускам датасета: на каждый chunk отдельный Trainer.predict (как у MMS), затем склейка **строк** для WER/CER — без np.concatenate по разной длине T (иначе NaN в beam, см. beam_error в старых артефактах).')
    p.add_argument('--enable_beam_eval', action='store_true', help='Включить финальную оценку beam search (долго). По умолчанию только greedy.')
    p.add_argument('--resume_from_checkpoint', type=str, default=None, metavar='PATH_OR_LAST')
    p.add_argument('--debug_trainable_grads', action='store_true')
    return p.parse_args()

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
    use_adapter = bool(args.add_adapter)
    model = Wav2Vec2BertForCTC.from_pretrained(args.model_name_or_path, attention_dropout=0.0, hidden_dropout=0.0, feat_proj_dropout=0.0, mask_time_prob=float(args.mask_time_prob), layerdrop=0.0, ctc_loss_reduction='mean', ctc_zero_infinity=True, add_adapter=use_adapter, pad_token_id=processor.tokenizer.pad_token_id, vocab_size=len(processor.tokenizer), ignore_mismatched_sizes=True)
    if use_adapter:
        logger.info('Режим с adapter: в facebook/w2v-bert-2.0 нет предобученных весов adapter — блок adapter и lm_head заполняются при загрузке; сообщения MISSING от transformers здесь ожидаемы (см. блог HF по ASR).')
    else:
        logger.info('Полный fine-tune энкодера (add_adapter=False), как у MMS без --train_adapters_only — сопоставимый режим.')
    if args.freeze_base_model:
        for p in model.wav2vec2_bert.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = True
        logger.info('freeze_base_model: trainable=%s', sum((x.numel() for x in model.parameters() if x.requires_grad)))
    elif args.freeze_feature_projection:
        fp_mod = model.wav2vec2_bert.feature_projection
        for p in fp_mod.parameters():
            p.requires_grad = False
        logger.info('freeze_feature_projection: заморожено параметров в feature_projection: %s (trainable всего=%s)', sum((p.numel() for p in fp_mod.parameters())), sum((x.numel() for x in model.parameters() if x.requires_grad)))
    use_gradient_checkpointing = not bool(args.no_gradient_checkpointing)
    if use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
    min_mel_frames = 2
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
    _max_steps = int(getattr(args, 'max_steps', -1) or -1)
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
        _use_bf16 = False
        _use_fp16 = bool(_cuda_ok)
    else:
        _use_bf16, _use_fp16 = (False, bool(_cuda_ok))
    _warmup_kw: Dict[str, Any] = {}
    if int(args.warmup_steps) > 0:
        _warmup_kw['warmup_steps'] = int(args.warmup_steps)
    else:
        _warmup_kw['warmup_ratio'] = float(args.warmup_ratio)
    logger.info('Параметры стабильности: precision=%s → bf16=%s fp16=%s | add_adapter=%s | gradient_checkpointing=%s | ctc_zero_infinity=%s | warmup=%s | lr_scheduler=%s weight_decay=%s max_steps=%s', pref, _use_bf16, _use_fp16, use_adapter, use_gradient_checkpointing, bool(getattr(model.config, 'ctc_zero_infinity', False)), _warmup_kw, args.lr_scheduler_type, args.weight_decay, _max_steps)
    _ta_kw: Dict[str, Any] = dict(output_dir=str(output_dir), **_group_kw, per_device_train_batch_size=args.per_device_train_batch_size, per_device_eval_batch_size=args.per_device_eval_batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps, **{_eval_kw: 'steps'}, save_strategy='steps', num_train_epochs=args.num_train_epochs, bf16=_use_bf16, fp16=_use_fp16, save_steps=args.save_steps, eval_steps=args.eval_steps, logging_steps=args.logging_steps, learning_rate=args.learning_rate, weight_decay=float(args.weight_decay), **_warmup_kw, save_total_limit=save_limit, load_best_model_at_end=True, metric_for_best_model='eval_cer', greater_is_better=False, report_to=report_to, push_to_hub=False, gradient_checkpointing=use_gradient_checkpointing, seed=args.seed, max_grad_norm=1.0)
    if _max_steps > 0:
        _ta_kw['max_steps'] = _max_steps
        logger.info('max_steps=%s — лимит глобальных шагов (перекрывает num_train_epochs), как в asr_training_common.py.', _max_steps)
    if 'lr_scheduler_type' in _ta_params:
        _ta_kw['lr_scheduler_type'] = str(args.lr_scheduler_type)
    else:
        logger.warning('TrainingArguments без lr_scheduler_type — планировщик LR из дефолта transformers.')
    training_args = TrainingArguments(**_ta_kw)
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
    run_meta = {'base_pretrained_model': args.model_name_or_path, 'train_adapters_only': False, 'add_adapter': use_adapter, 'learning_rate': float(args.learning_rate), 'lr_scheduler_type': str(args.lr_scheduler_type), 'warmup_steps': int(args.warmup_steps), 'warmup_ratio': float(args.warmup_ratio), 'weight_decay': float(args.weight_decay), 'max_steps_arg': int(_max_steps), 'freeze_base_model': bool(args.freeze_base_model), 'freeze_feature_projection': bool(args.freeze_feature_projection), 'gradient_checkpointing': use_gradient_checkpointing, 'precision_mode': pref, 'bf16_enabled': _use_bf16, 'fp16_enabled': _use_fp16, 'warmup_training_kwargs': _warmup_kw, 'mask_time_prob': float(args.mask_time_prob), 'ctc_zero_infinity': bool(getattr(trainer.model.config, 'ctc_zero_infinity', False)), 'best_model_checkpoint_path_before_cleanup': best_ckpt, 'beam_width_used': int(args.beam_width), 'beam_logits_chunk_samples': int(args.beam_logits_chunk_samples), 'metric_for_best_model': 'eval_cer', 'best_metric_during_training': getattr(trainer.state, 'best_metric', None), 'trainer_global_steps': getattr(trainer.state, 'global_step', None), 'train_samples': len(raw_train), 'test_samples': len(raw_test), **model_run_info(trainer.model), 'plots_saved': plots_saved, 'final_model_dir': str((output_dir / 'final_model').resolve()), 'artifacts_dir': str(artifact_dir.resolve()), 'comparison_note': 'Протокол выровнен с MMS (run_artifacts/metrics_summary.json): те же train/test каталоги, LR=2e-5, cosine, warmup_ratio=0.12, weight_decay=0.01, полный FT при отсутствии --add_adapter. Для того же бюджета шагов оптимизатора, что у MMS, передайте --max_steps 22000 (см. trainer_log_history: последний train-step 22000).'}
    greedy_pack = {'train': (train_eval.get('greedy_train_wer', train_eval.get('eval_wer', float('nan'))), train_eval.get('greedy_train_cer', train_eval.get('eval_cer', float('nan')))), 'test': (test_eval.get('greedy_test_wer', test_eval.get('eval_wer', float('nan'))), test_eval.get('greedy_test_cer', test_eval.get('eval_cer', float('nan'))))}
    beam_pack: Dict[str, Tuple[float, float]] = {}
    beam_details: Dict[str, Any] = {}
    if args.enable_beam_eval:
        try:
            from pyctcdecode import build_ctcdecoder
            vocab_labels = labels_aligned_for_pyctc(processor)
            decoder = build_ctcdecoder(vocab_labels)
            trainer.model.eval()
            bw = int(args.beam_width)
            bchunk = int(args.beam_logits_chunk_samples)
            for split_key, ds in (('train', train_ds), ('test', test_ds)):
                n = len(ds)
                pred_all: List[str] = []
                ref_all: List[str] = []
                if bchunk <= 0:
                    _starts = [0]
                    _ends = [n]
                else:
                    _starts = list(range(0, n, bchunk))
                    _ends = [min(s + bchunk, n) for s in _starts]
                for start, end in zip(_starts, _ends):
                    sub = ds.select(range(start, end))
                    logits, lids = predict_logits_and_refs(trainer, sub, processor)
                    logits = logits.astype(np.float32, copy=False)
                    pred_all.extend(decode_beam_batch_wav2vec2(logits, decoder, bw, processor))
                    ref_all.extend((decode_refs_from_labels(lids[i], processor) for i in range(lids.shape[0])))
                wer_b, cer_b = wer_cer_from_strings(pred_all, ref_all)
                beam_pack[split_key] = (wer_b, cer_b)
                beam_details[f'beam_{split_key}'] = {'wer': wer_b, 'cer': cer_b, 'beam_width': bw}
                logger.info('Beam %s: WER=%.6f CER=%.6f (beam_width=%s, chunks=%s)', split_key, wer_b, cer_b, bw, len(_starts))
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
