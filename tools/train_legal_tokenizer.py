#!/usr/bin/env python3
"""
LegalLM Tokenizer Training Script

Script để huấn luyện một BPE Tokenizer chuyên biệt cho văn bản luật pháp Việt Nam.
Corpus sử dụng: undertheseanlp/UTS_VLC (Vietnamese Legal Corpus).

Kết quả lưu tại: artifacts/tokenizer_vlex_ria
-> Có thể thay cho vinai/phobert-base
"""

import os
import argparse
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer(vocab_size: int = 32000, dataset_name: str = "undertheseanlp/UTS_VLC"):
    print(f"Đang tải corpus {dataset_name} từ HuggingFace...")
    try:
        dataset = load_dataset(dataset_name, split="2026")
    except Exception as e:
        print(f"Không thể tải 2026 split, thử tải train split... Lỗi: {e}")
        dataset = load_dataset(dataset_name, split="train")

    # Khởi tạo mô hình BPE
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()

    # Cấu hình trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"],
        min_frequency=2,
        show_progress=True
    )

    # Dùng Generator để feed text vào Tokenizer
    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["content"]

    print(f"Bắt đầu huấn luyện Tokenizer với vocab_size = {vocab_size}...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    
    # Custom post-processing if needed
    from tokenizers.processors import TemplateProcessing
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        pair="<s> $A </s> <s> $B </s>",
        special_tokens=[
            ("<s>", tokenizer.token_to_id("<s>")),
            ("</s>", tokenizer.token_to_id("</s>")),
        ],
    )

    # Lưu kết quả
    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "tokenizer_vlex_ria")
    os.makedirs(out_dir, exist_ok=True)
    
    out_file = os.path.join(out_dir, "tokenizer.json")
    tokenizer.save(out_file)
    print(f"Huấn luyện thành công! Tokenizer JSON đã lưu tại: {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Legal BPE Tokenizer")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Kích thước từ vựng cần train")
    parser.add_argument("--dataset", type=str, default="undertheseanlp/UTS_VLC", help="Tên dataset trên HuggingFace")
    args = parser.parse_args()
    
    train_tokenizer(vocab_size=args.vocab_size, dataset_name=args.dataset)
