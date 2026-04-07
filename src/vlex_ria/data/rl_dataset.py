"""
VLexRIA RL Dataset Module

Provides dataset classes for different RL algorithms:
1. DPODataset: Preference pairs (chosen, rejected) for DPO training
2. GRPODataset: Prompts for group relative policy optimization
3. PPODataset: Prompts with reward model for PPO training

All datasets use **lazy tokenization**: raw text is stored in __init__,
tokenization happens on-the-fly in __getitem__() to avoid OOM.
"""

import os
import json
import random
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

from vlex_ria.core.utils import get_logger

# Use local cache if exists, avoid downloading every time
# REUSE_DATASET_IF_EXISTS: Skip download and processing if cached dataset exists
DOWNLOAD_MODE = "reuse_dataset_if_exists"

from vlex_ria.core.config import DataConfig

# Initialize logger
logger = get_logger(__name__)


# Raw Data Containers (no tensors — just strings)
@dataclass
class DPOPair:
    """Raw DPO preference pair — strings only, no tensors."""
    prompt: str
    chosen_response: str
    rejected_response: str


# DPO Dataset (Lazy Tokenization)
class DPODataset(Dataset):
    """
    Dataset for Direct Preference Optimization (DPO).
    
    Contains preference pairs: (prompt, chosen_response, rejected_response)
    Used to train model to prefer chosen over rejected responses.
    
    **Lazy tokenization**: raw string triples are stored;
    tokenization happens in __getitem__().
    
    Shape:
        - prompt_input_ids: (L_prompt,) - tokenized prompt
        - chosen_input_ids: (L_full,) - prompt + chosen response
        - rejected_input_ids: (L_full,) - prompt + rejected response
        - chosen_labels: (L_full,) - labels for chosen (prompt=-100)
        - rejected_labels: (L_full,) - labels for rejected (prompt=-100)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DataConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = config.rl_max_seq_length
        self.split = split
        max_samples = max_samples or config.rl_max_samples
        
        # Store raw pairs (strings only — no tensors)
        self._raw_pairs: List[DPOPair] = self._load_raw_pairs(config, split, max_samples)
        
        logger.info(f"DPODataset ({split}): {len(self._raw_pairs)} preference pairs "
                     f"(lazy — tensors created on access)")
    
    def _load_raw_pairs(
        self,
        config: DataConfig,
        split: str,
        max_samples: int,
    ) -> List[DPOPair]:
        """Load raw DPO preference pairs as strings (no tokenization)."""
        try:
            from datasets import load_dataset
            
            try:
                dataset = load_dataset(
                    config.rl_dataset_name,
                    config.rl_dataset_config,
                    cache_dir=config.rl_data_dir,
                    download_mode=DOWNLOAD_MODE,
                )
            except Exception:
                dataset = load_dataset(
                    config.rl_dataset_name,
                    "default",
                    cache_dir=config.rl_data_dir,
                    download_mode=DOWNLOAD_MODE,
                )
            
            if split in dataset:
                data = dataset[split]
            else:
                data = dataset["train"]
            
            pairs = []
            for i, item in enumerate(data):
                if i >= max_samples:
                    break
                
                chosen = item.get("chosen", "")
                rejected = item.get("rejected", "")
                
                if not chosen or not rejected:
                    continue
                
                prompt, chosen_resp, rejected_resp = self._parse_conversation(chosen, rejected)
                
                if prompt and chosen_resp and rejected_resp:
                    pairs.append(DPOPair(
                        prompt=prompt,
                        chosen_response=chosen_resp,
                        rejected_response=rejected_resp,
                    ))
            
            # Split for validation
            if split == "validation":
                pairs = pairs[:int(len(pairs) * 0.1)]
            elif split == "train":
                pairs = pairs[int(len(pairs) * 0.1):]
            
            return pairs
            
        except Exception as e:
            logger.error(f"Failed to load DPO dataset: {e}")
            logger.warning("Generating synthetic preference data...")
            return self._generate_synthetic_data(split, max_samples)
    
    def _parse_conversation(
        self,
        chosen: str,
        rejected: str,
    ) -> Tuple[str, str, str]:
        """Parse HH-RLHF conversation format: 'Human: ... Assistant: ...'"""
        if "Human:" not in chosen:
            return "", "", ""
        
        parts = chosen.split("Assistant:")
        if len(parts) < 2:
            return "", "", ""
        
        prompt = parts[0].replace("Human:", "").strip()
        chosen_response = parts[1].strip() if len(parts) > 1 else ""
        
        rejected_parts = rejected.split("Assistant:")
        rejected_response = rejected_parts[1].strip() if len(rejected_parts) > 1 else ""
        
        # Clean up responses (remove trailing human turns)
        if "Human:" in chosen_response:
            chosen_response = chosen_response.split("Human:")[0].strip()
        if "Human:" in rejected_response:
            rejected_response = rejected_response.split("Human:")[0].strip()
        
        return prompt, chosen_response, rejected_response
    
    def _tokenize_pair(self, pair: DPOPair) -> Dict[str, torch.Tensor]:
        """Tokenize a single DPO pair on-the-fly."""
        prompt_text = f"Human: {pair.prompt}\n\nAssistant:"
        chosen_full = f"{prompt_text} {pair.chosen_response}"
        rejected_full = f"{prompt_text} {pair.rejected_response}"
        
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        
        chosen_ids = self.tokenizer.encode(
            chosen_full, max_length=self.max_seq_length,
            truncation=True, add_special_tokens=True,
        )
        rejected_ids = self.tokenizer.encode(
            rejected_full, max_length=self.max_seq_length,
            truncation=True, add_special_tokens=True,
        )
        
        # Labels: -100 for prompt tokens
        chosen_labels = [-100] * len(prompt_ids) + chosen_ids[len(prompt_ids):]
        rejected_labels = [-100] * len(prompt_ids) + rejected_ids[len(prompt_ids):]
        
        # Pad to same length
        max_len = min(max(len(chosen_ids), len(rejected_ids)), self.max_seq_length)
        
        def pad_sequence(ids, labels, target_len):
            pad_len = target_len - len(ids)
            if pad_len > 0:
                ids = ids + [self.tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len
            else:
                ids = ids[:target_len]
                labels = labels[:target_len]
            return ids, labels
        
        chosen_ids, chosen_labels = pad_sequence(chosen_ids, chosen_labels, max_len)
        rejected_ids, rejected_labels = pad_sequence(rejected_ids, rejected_labels, max_len)
        
        return {
            'prompt_input_ids': torch.tensor(prompt_ids, dtype=torch.long),
            'chosen_input_ids': torch.tensor(chosen_ids, dtype=torch.long),
            'rejected_input_ids': torch.tensor(rejected_ids, dtype=torch.long),
            'chosen_labels': torch.tensor(chosen_labels, dtype=torch.long),
            'rejected_labels': torch.tensor(rejected_labels, dtype=torch.long),
            'chosen_attention_mask': torch.tensor(
                [1 if t != self.tokenizer.pad_token_id else 0 for t in chosen_ids],
                dtype=torch.long,
            ),
            'rejected_attention_mask': torch.tensor(
                [1 if t != self.tokenizer.pad_token_id else 0 for t in rejected_ids],
                dtype=torch.long,
            ),
        }
    
    def _generate_synthetic_data(
        self,
        split: str,
        max_samples: int,
    ) -> List[DPOPair]:
        """Generate synthetic preference data for testing."""
        preference_pairs = [
            ("Tội lừa đảo chiếm đoạt tài sản bị xử phạt như thế nào?", 
             "Theo Điều 174 Bộ luật Hình sự 2015, người phạm tội lừa đảo chiếm đoạt tài sản có thể bị phạt cải tạo không giam giữ đến 03 năm hoặc phạt tù từ 06 tháng đến 03 năm. Tùy theo tính chất mức độ (như có tổ chức, giá trị tài sản lớn), hình phạt có thể lên đến tù chung thân.",
             "Lừa đảo thì đi tù, nhẹ thì vài tháng, nặng thì chung thân."),
            ("Điều kiện kinh doanh dịch vụ môi giới bất động sản là gì?",
             "Theo Luật Kinh doanh bất động sản hiện hành, tổ chức, cá nhân kinh doanh dịch vụ môi giới phải thành lập doanh nghiệp và phải có ít nhất 02 người có chứng chỉ hành nghề môi giới.",
             "Cứ mở công ty ra làm thoải mái không cần bằng cấp."),
            ("Người dưới 18 tuổi gây thiệt hại thì ai bồi thường?",
             "Theo Điều 586 Bộ luật Dân sự 2015, người chưa đủ 15 tuổi gây thiệt hại mà còn cha mẹ thì cha mẹ phải bồi thường. Người từ 15 đến dưới 18 tuổi tự bồi thường bằng tài sản của mình, nếu không đủ thì cha mẹ bồi thường phần còn thiếu.",
             "Con nít làm sai thì bố mẹ chịu hết."),
            ("Muốn lập di chúc hợp pháp cần những yếu tố nào?",
             "Theo Điều 630 Bộ luật Dân sự 2015, quá trình lập di chúc phải tự nguyện, minh mẫn. Nội dung không trái pháp luật, đạo đức xã hội. Hình thức phải theo quy định (bằng văn bản hoặc miệng có người làm chứng).",
             "Cứ viết ra giấy ký tên là được chia tài sản."),
            ("Xe máy vượt đèn đỏ bị phạt bao nhiêu tiền?",
             "Theo Nghị định 100/2019/NĐ-CP (sửa đổi bởi Nghị định 123/2021/NĐ-CP), người điều khiển xe mô tô, xe gắn máy vượt đèn đỏ sẽ bị phạt tiền từ 800.000 đồng đến 1.000.000 đồng, đồng thời tước bằng lái từ 1 đến 3 tháng.",
             "Chắc cỡ vài trăm ngàn thôi."),
        ]
        
        num_samples = min(max_samples, 100 if split == "train" else 10)
        pairs = []
        
        for _ in range(num_samples):
            prompt, chosen, rejected = random.choice(preference_pairs)
            pairs.append(DPOPair(prompt=prompt, chosen_response=chosen, rejected_response=rejected))
        
        return pairs
    
    def __len__(self) -> int:
        return len(self._raw_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Lazy: tokenize on access."""
        return self._tokenize_pair(self._raw_pairs[idx])



# GRPO Dataset (Lazy Tokenization)
class GRPODataset(Dataset):
    """
    Dataset for Group Relative Policy Optimization.
    
    Contains prompts for generating multiple responses per prompt.
    Responses are generated during training, not stored.
    
    **Lazy tokenization**: raw prompt strings are stored;
    tokenization happens in __getitem__().
    
    Shape:
        - input_ids: (L,) - tokenized prompt
        - attention_mask: (L,) - 1 for real tokens
        - prompt_text: str - original prompt text
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DataConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = config.rl_max_seq_length
        self.split = split
        max_samples = max_samples or config.rl_max_samples
        
        # Store raw prompts only (strings — no tensors)
        self.prompts: List[str] = self._load_prompts(config, split, max_samples)
        logger.info(f"GRPODataset ({split}): {len(self.prompts)} prompts "
                     f"(lazy — tensors created on access)")
    
    def _load_prompts(
        self,
        config: DataConfig,
        split: str,
        max_samples: int,
    ) -> List[str]:
        """Load raw prompt strings (no tokenization)."""
        try:
            from datasets import load_dataset
            
            try:
                dataset = load_dataset(
                    config.rl_dataset_name,
                    config.rl_dataset_config,
                    cache_dir=config.rl_data_dir,
                    download_mode=DOWNLOAD_MODE,
                )
            except Exception:
                dataset = load_dataset(
                    config.rl_dataset_name,
                    "default",
                    cache_dir=config.rl_data_dir,
                    download_mode=DOWNLOAD_MODE,
                )
            
            if split in dataset:
                data = dataset[split]
            else:
                data = dataset["train"]
            
            prompts = []
            for item in data:
                chosen = item.get("chosen", "")
                if "Human:" in chosen:
                    human_turn = chosen.split("Human:")[1].split("Assistant:")[0].strip()
                    if human_turn:
                        prompts.append(f"Human: {human_turn}\n\nAssistant:")
                
                if len(prompts) >= max_samples:
                    break
            
            # Split for validation
            if split == "validation":
                prompts = prompts[:int(len(prompts) * 0.1)]
            elif split == "train":
                prompts = prompts[int(len(prompts) * 0.1):]
            
            return prompts
            
        except Exception as e:
            logger.error(f"Failed to load GRPO dataset: {e}")
            return self._generate_synthetic_prompts(split, max_samples)
    
    def _generate_synthetic_prompts(
        self,
        split: str,
        max_samples: int,
    ) -> List[str]:
        """Generate synthetic prompts."""
        templates = [
            "Human: Cho tôi hỏi thủ tục đăng ký kết hôn hiện nay quy định thế nào?\n\nAssistant:",
            "Human: Quy định về thời gian thử việc theo Bộ luật Lao động mới nhất?\n\nAssistant:",
            "Human: Các trường hợp nào được miễn giảm thuế thu nhập nông nghiệp?\n\nAssistant:",
            "Human: Phân biệt tội trộm cắp và tội cướp tài sản?\n\nAssistant:",
            "Human: Quy định về thừa kế theo pháp luật khi không có di chúc?\n\nAssistant:",
            "Human: Quyền lợi của khách hàng khi mua phải hàng giả, hàng nhái?\n\nAssistant:",
            "Human: Hồ sơ thành lập công ty TNHH 1 thành viên gồm những gì?\n\nAssistant:",
            "Human: Quy định về tách thửa đất thổ cư ở nông thôn?\n\nAssistant:",
        ]
        
        num_samples = min(max_samples, 100 if split == "train" else 10)
        return [random.choice(templates) for _ in range(num_samples)]
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Lazy: tokenize on access."""
        prompt = self.prompts[idx]
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'prompt_text': prompt,
        }


# PPO Dataset (Lazy Tokenization)
class PPODataset(Dataset):
    """
    Dataset for Proximal Policy Optimization.
    
    Similar to GRPO but may include additional context for reward computation.
    
    **Lazy tokenization**: raw prompt strings are stored;
    tokenization happens in __getitem__().
    
    Shape:
        - input_ids: (L,) - tokenized prompt
        - attention_mask: (L,) - 1 for real tokens
        - prompt_text: str - original prompt
        - context: str - additional context (optional)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DataConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = config.rl_max_seq_length
        self.split = split
        max_samples = max_samples or config.rl_max_samples
        
        # Store raw data (strings — no tensors)
        self._raw_data: List[Dict[str, str]] = self._load_raw_data(config, split, max_samples)
        logger.info(f"PPODataset ({split}): {len(self._raw_data)} samples "
                     f"(lazy — tensors created on access)")
    
    def _load_raw_data(
        self,
        config: DataConfig,
        split: str,
        max_samples: int,
    ) -> List[Dict[str, str]]:
        """Load raw PPO data as string dicts (no tokenization)."""
        try:
            from datasets import load_dataset
            
            try:
                dataset = load_dataset(
                    config.rl_dataset_name,
                    config.rl_dataset_config,
                    cache_dir=config.rl_data_dir,
                    download_mode=DOWNLOAD_MODE,
                )
            except Exception:
                dataset = load_dataset(
                    config.rl_dataset_name,
                    "default",
                    cache_dir=config.rl_data_dir,
                    download_mode=DOWNLOAD_MODE,
                )
            
            if split in dataset:
                data = dataset[split]
            else:
                data = dataset["train"]
            
            samples = []
            for item in data:
                chosen = item.get("chosen", "")
                if "Human:" in chosen:
                    human_turn = chosen.split("Human:")[1].split("Assistant:")[0].strip()
                    if human_turn:
                        prompt = f"Human: {human_turn}\n\nAssistant:"
                        samples.append({
                            'prompt_text': prompt,
                            'context': human_turn,
                        })
                
                if len(samples) >= max_samples:
                    break
            
            # Split for validation
            if split == "validation":
                samples = samples[:int(len(samples) * 0.1)]
            elif split == "train":
                samples = samples[int(len(samples) * 0.1):]
            
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load PPO dataset: {e}")
            return self._generate_synthetic_data(split, max_samples)
    
    def _generate_synthetic_data(
        self,
        split: str,
        max_samples: int,
    ) -> List[Dict[str, str]]:
        """Generate synthetic PPO data."""
        contexts = [
            "luật doanh nghiệp", "hợp đồng thương mại", "tranh chấp đất đai",
            "luật sở hữu trí tuệ", "chính sách thuế", "luật hôn nhân gia đình",
        ]
        
        num_samples = min(max_samples, 100 if split == "train" else 10)
        samples = []
        
        for _ in range(num_samples):
            context = random.choice(contexts)
            prompt = f"Human: Tóm tắt các nội dung chính về {context}.\n\nAssistant:"
            samples.append({
                'prompt_text': prompt,
                'context': context,
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self._raw_data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Lazy: tokenize on access."""
        item = self._raw_data[idx]
        encoding = self.tokenizer(
            item['prompt_text'],
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'prompt_text': item['prompt_text'],
            'context': item['context'],
        }


# Collate Functions
def dpo_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate DPO batch with variable length handling."""
    # Find max lengths
    max_chosen_len = max(ex['chosen_input_ids'].shape[0] for ex in batch)
    max_rejected_len = max(ex['rejected_input_ids'].shape[0] for ex in batch)
    max_len = max(max_chosen_len, max_rejected_len)
    
    result = {
        'chosen_input_ids': [],
        'rejected_input_ids': [],
        'chosen_labels': [],
        'rejected_labels': [],
        'chosen_attention_mask': [],
        'rejected_attention_mask': [],
    }
    
    pad_token_id = batch[0]['chosen_input_ids'][0].item()  # Fallback
    
    for ex in batch:
        for key in result.keys():
            tensor = ex[key]
            pad_len = max_len - tensor.shape[0]
            if pad_len > 0:
                if 'labels' in key:
                    pad_value = -100
                elif 'mask' in key:
                    pad_value = 0
                else:
                    pad_value = pad_token_id
                tensor = torch.cat([
                    tensor,
                    torch.full((pad_len,), pad_value, dtype=tensor.dtype)
                ])
            result[key].append(tensor)
    
    return {k: torch.stack(v) for k, v in result.items()}


def rl_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate GRPO/PPO batch."""
    result = {}
    
    tensor_keys = [k for k in batch[0].keys() if isinstance(batch[0][k], torch.Tensor)]
    
    for key in tensor_keys:
        result[key] = torch.stack([ex[key] for ex in batch])
    
    for key in batch[0].keys():
        if key not in tensor_keys:
            result[key] = [ex[key] for ex in batch]
    
    return result


# DataLoader Factory
def create_rl_dataloaders(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    algorithm: str = "grpo",  # "dpo", "grpo", "ppo"
    batch_size: int = 4,
    max_samples: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create RL DataLoaders for specific algorithm.
    
    Args:
        config: Data configuration
        tokenizer: Tokenizer
        algorithm: RL algorithm type
        batch_size: Batch size
        max_samples: Max samples
        
    Returns:
        train_loader, val_loader
    """
    # Select dataset class
    if algorithm == "dpo":
        DatasetClass = DPODataset
        collate = dpo_collate_fn
    elif algorithm == "grpo":
        DatasetClass = GRPODataset
        collate = rl_collate_fn
    elif algorithm == "ppo":
        DatasetClass = PPODataset
        collate = rl_collate_fn
    else:
        raise ValueError(f"Unknown RL algorithm: {algorithm}")
    
    # Create datasets
    train_dataset = DatasetClass(
        tokenizer=tokenizer,
        config=config,
        split="train",
        max_samples=max_samples,
    )
    
    val_dataset = DatasetClass(
        tokenizer=tokenizer,
        config=config,
        split="validation",
        max_samples=max_samples // 10 if max_samples else None,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(config.num_workers, 2),
        pin_memory=config.pin_memory,
        collate_fn=collate,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(config.num_workers, 2),
        pin_memory=config.pin_memory,
        collate_fn=collate,
    )
    
    return train_loader, val_loader


# Test
def test_rl_datasets():
    """Test all RL dataset classes."""
    from vlex_ria.core.config import load_config
    from vlex_ria.data.dataset import get_tokenizer
    
    logger.info("=" * 100)
    logger.info("Testing RL Datasets")
    logger.info("=" * 100)
    
    config = load_config()
    tokenizer = get_tokenizer(config.data)
    
    # Test DPO
    logger.info("-" * 100)
    logger.info("Testing DPODataset...")
    dpo_train, dpo_val = create_rl_dataloaders(
        config.data, tokenizer, algorithm="dpo",
        batch_size=2, max_samples=20,
    )
    batch = next(iter(dpo_train))
    logger.info(f"  chosen_input_ids shape: {batch['chosen_input_ids'].shape}")
    logger.info(f"  rejected_input_ids shape: {batch['rejected_input_ids'].shape}")
    
    # Test GRPO
    logger.info("-" * 100)
    logger.info("Testing GRPODataset...")
    grpo_train, grpo_val = create_rl_dataloaders(
        config.data, tokenizer, algorithm="grpo",
        batch_size=2, max_samples=20,
    )
    batch = next(iter(grpo_train))
    logger.info(f"  input_ids shape: {batch['input_ids'].shape}")
    logger.info(f"  prompt_text: {batch['prompt_text'][0][:50]}...")
    
    # Test PPO
    logger.info("-" * 100)
    logger.info("Testing PPODataset...")
    ppo_train, ppo_val = create_rl_dataloaders(
        config.data, tokenizer, algorithm="ppo",
        batch_size=2, max_samples=20,
    )
    batch = next(iter(ppo_train))
    logger.info(f"  input_ids shape: {batch['input_ids'].shape}")
    
    logger.info("=" * 100)
    logger.info("All RL dataset tests passed!")
    logger.info("=" * 100)


if __name__ == "__main__":
    test_rl_datasets()
