"""
VLexRIA Data Pipeline — Canonical DataLoader Factory

Single source of truth for creating train/val DataLoaders.
Uses the VLexRIA pipeline: Ingestion → Cleaning → Adaptation → Tokenization → DataLoader.

For Vietnamese legal domain with PhoBERT tokenizer, applies automatic
word segmentation before tokenization (if underthesea is installed).

Usage:
    from vlex_ria.data import create_dataloaders
    train_loader, val_loader = create_dataloaders(config.data, tokenizer, mode="pretrain")
"""

import torch
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from transformers import PreTrainedTokenizer

from vlex_ria.core.config import DataConfig
from .ingestion import HFDatasetDownloader

from .adapters import LegalDatasetAdapter
from .dataset import collate_fn
from .cleaners import VLexRIADatasetCleaner, preprocess_vietnamese, is_vietnamese_tokenizer

from vlex_ria.core.utils import get_logger

logger = get_logger(__name__)

def _maybe_preprocess_vn(texts, tokenizer_name: str):
    """Apply Vietnamese word segmentation if using a Vietnamese tokenizer."""
    if not is_vietnamese_tokenizer(tokenizer_name):
        return texts
    if isinstance(texts, list):
        return [preprocess_vietnamese(t) for t in texts]
    return preprocess_vietnamese(texts)

def create_dataloaders(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    mode: str = "pretrain",
    batch_size: int = 16,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders using the VLexRIA pipeline.
    
    Pipeline: HF download → VLexRIA adapt → clean → VN preprocess → tokenize → DataLoader
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer
        mode: Training mode — "pretrain", "sft", or "rl"
        batch_size: Batch size
        max_samples: Max samples per split (None = use all)
        
    Returns:
        (train_loader, val_loader)
    """
    downloader = HFDatasetDownloader()
    tokenizer_name = config.tokenizer_name
    
    # Select dataset name and split based on mode
    if mode == "pretrain":
        dataset_name = config.pretrain_dataset_name
        dataset_split = getattr(config, "pretrain_dataset_split", "train")
        config_name = getattr(config, "pretrain_dataset_config", None)
    elif mode == "sft":
        dataset_name = config.sft_dataset_name
        dataset_split = getattr(config, "sft_dataset_split", "train")
        config_name = None
    elif mode == "rl":
        dataset_name = config.rl_dataset_name
        dataset_split = getattr(config, "rl_dataset_split", "train")
        config_name = getattr(config, "rl_dataset_config", None)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'pretrain', 'sft', or 'rl'.")

    logger.info(f"[DataLoader] mode={mode}, dataset={dataset_name}, split={dataset_split}")
    logger.info(f"[DataLoader] tokenizer={tokenizer_name}, VN preprocess={is_vietnamese_tokenizer(tokenizer_name)}")

    # 1. Ingestion: Download from HuggingFace
    dataset = downloader.download(
        dataset_name, 
        split=dataset_split,
        config_name=config_name,
    )
    if max_samples:
        dataset = dataset.select(range(min(len(dataset), max_samples)))
        
    # Split train/val
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_data = split_dataset['train']
    val_data = split_dataset['test']
    
    # 2. Processing & 3. Adaptation → Preprocessing → Tokenization
    def process_split(data):
        if mode == "pretrain":
            # Use pretrain_text_column if specified in config
            text_column = getattr(config, "pretrain_text_column", None)
            adapted = LegalDatasetAdapter.adapt_for_pretrain(data, text_column=text_column)
            cleaned = VLexRIADatasetCleaner.clean_empty_samples(adapted, ["text"])
            
            def _tokenize(examples):
                # Apply Vietnamese word segmentation before tokenization
                texts = _maybe_preprocess_vn(examples["text"], tokenizer_name)
                encodings = tokenizer(
                    texts,
                    truncation=True,
                    max_length=config.pretrain_max_seq_length,
                    padding="max_length",
                )
                encodings["labels"] = encodings["input_ids"].copy()
                return encodings
            
            tokenized = cleaned.map(_tokenize, batched=True, remove_columns=["text"])
            tokenized.set_format("torch")
            return tokenized
            
        elif mode == "sft":
            adapted = LegalDatasetAdapter.adapt_for_sft(data)
            cleaned = VLexRIADatasetCleaner.clean_empty_samples(adapted, ["instruction", "output"])
            
            def _tokenize_sft(examples):
                res = {"input_ids": [], "attention_mask": [], "labels": []}
                for i in range(len(examples["instruction"])):
                    # Preprocess Vietnamese text before building prompt
                    instr = _maybe_preprocess_vn(examples['instruction'][i], tokenizer_name)
                    inp = _maybe_preprocess_vn(examples.get('input', [''])[i], tokenizer_name)
                    out = _maybe_preprocess_vn(examples['output'][i], tokenizer_name)
                    
                    prompt = (
                        f"### Instruction:\n{instr}\n\n"
                        f"### Input:\n{inp}\n\n"
                        f"### Response:\n"
                    )
                    full_text = prompt + out
                    
                    p_ids = tokenizer.encode(prompt, add_special_tokens=True)
                    f_ids = tokenizer.encode(full_text, add_special_tokens=True)
                    
                    if len(f_ids) > config.sft_max_seq_length:
                        f_ids = f_ids[:config.sft_max_seq_length]
                        p_ids = p_ids[:min(len(p_ids), len(f_ids))]
                        
                    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
                    labels = labels[:len(f_ids)]
                    
                    pad_len = config.sft_max_seq_length - len(f_ids)
                    if pad_len > 0:
                        f_ids += [tokenizer.pad_token_id] * pad_len
                        labels += [-100] * pad_len
                    attn = [1] * (config.sft_max_seq_length - pad_len) + [0] * pad_len
                    
                    res["input_ids"].append(f_ids)
                    res["attention_mask"].append(attn)
                    res["labels"].append(labels)
                return res
            
            tokenized = cleaned.map(
                _tokenize_sft, batched=True,
                remove_columns=cleaned.column_names,
            )
            tokenized.set_format("torch")
            return tokenized
            
        elif mode == "rl":
            adapted = LegalDatasetAdapter.adapt_for_rl(data)
            cleaned = VLexRIADatasetCleaner.clean_empty_samples(adapted, ["chosen"])
            
            def _tokenize_rl(examples):
                res = {"input_ids": [], "attention_mask": [], "prompt_text": []}
                for c in examples["chosen"]:
                    prompt = c.split("Assistant:")[0] + "Assistant:" if "Assistant:" in c else c
                    # Preprocess Vietnamese text before tokenization
                    prompt_processed = _maybe_preprocess_vn(prompt, tokenizer_name)
                    enc = tokenizer(
                        prompt_processed,
                        max_length=config.rl_max_seq_length,
                        padding="max_length",
                        truncation=True,
                    )
                    res["input_ids"].append(enc['input_ids'])
                    res["attention_mask"].append(enc['attention_mask'])
                    res["prompt_text"].append(prompt)  # Keep original text for generation
                return res
            
            tokenized = cleaned.map(
                _tokenize_rl, batched=True,
                remove_columns=cleaned.column_names,
            )
            tokenized.set_format(
                type="torch",
                columns=["input_ids", "attention_mask"],
                output_all_columns=True,
            )
            return tokenized
            
    train_processed = process_split(train_data)
    val_processed = process_split(val_data)
    
    train_loader = DataLoader(
        train_processed,
        batch_size=batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_processed,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    return train_loader, val_loader

__all__ = [
    "create_dataloaders",
]
