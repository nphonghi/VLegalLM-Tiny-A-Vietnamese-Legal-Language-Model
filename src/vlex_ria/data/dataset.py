"""
VLexRIA Dataset Module

Handles data loading and preprocessing for:
1. Pretraining: undertheseanlp/UTS_VLC dataset
2. SFT: YuITC/Vietnamese-Legal-Documents instruction dataset
3. RL: YuITC/Vietnamese-Legal-Documents preference/prompt dataset

All datasets use lazy tokenization: raw text is stored, tokens are
produced on-the-fly in __getitem__(). This avoids OOM on machines
with limited RAM (e.g. Apple Silicon M1 with 16 GB unified memory).
"""

import os
import json
import random
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from vlex_ria.core.utils import get_logger

# Use local cache if exists, avoid downloading every time
# REUSE_DATASET_IF_EXISTS: Skip download and processing if cached dataset exists
DOWNLOAD_MODE = "reuse_dataset_if_exists"

from vlex_ria.core.config import DataConfig

# Initialize logger
logger = get_logger(__name__)


# Tokenizer Management
def get_tokenizer(config: DataConfig) -> PreTrainedTokenizer:
    """
    Load and configure tokenizer.
    
    Supports:
    - vinai/phobert-base (Vietnamese, recommended for legal VN)
    - gpt2 (English, legacy)
    - Any HuggingFace tokenizer
    
    For PhoBERT: automatically applies Vietnamese word segmentation
    using `underthesea` if available, for optimal tokenization.
    
    Args:
        config: DataConfig with tokenizer settings
        
    Returns:
        Configured tokenizer
    """
    import os
    
    tokenizer_name = config.tokenizer_name
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    
    # Try to load from HuggingFace with local caching
    try:
        # Set local cache directory
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "tokenizer")
        os.makedirs(cache_dir, exist_ok=True)
        
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            local_files_only=False,
            cache_dir=cache_dir,
        )
    except Exception as e:
        # If network fails, try local_only mode
        logger.warning(f"Failed to load tokenizer from HuggingFace: {e}")
        logger.info("Attempting to load from local cache...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e2:
            logger.error(f"Failed to load tokenizer from local cache: {e2}")
            raise RuntimeError(
                f"Failed to load tokenizer '{tokenizer_name}'. "
                f"Please ensure you have internet connection to download it, "
                f"or use a pre-downloaded tokenizer."
            ) from e
    
    # Configure pad token based on tokenizer type
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            logger.info(f"Set pad_token = eos_token ({tokenizer.pad_token})")
        else:
            logger.warning("Tokenizer has no pad_token or eos_token!")
    
    # Log tokenizer info
    logger.info(f"Tokenizer loaded: {tokenizer_name}")
    logger.info(f"  Vocab size: {len(tokenizer)}")
    logger.info(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    logger.info(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
    logger.info(f"  BOS token: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
    
    # Check Vietnamese word segmentation support for PhoBERT
    if "phobert" in tokenizer_name.lower():
        try:
            from underthesea import word_tokenize
            logger.info("[SUCCESS] underthesea available — Vietnamese word segmentation enabled")
        except ImportError:
            logger.warning(
                "[WARNING] underthesea not installed. PhoBERT works best with Vietnamese "
                "word segmentation. Install with: pip install underthesea"
            )
    
    return tokenizer


# Pretrain Dataset (Lazy Tokenization)
class PretrainDataset(Dataset):
    """
    Dataset for pretraining on raw text.
    
    Uses OpenWebText or WikiText for language modeling.
    Text is chunked into windows of max_seq_length tokens.
    
    **Lazy tokenization**: raw text chunks are stored as strings;
    tokenization happens in __getitem__() to avoid OOM.
    
    Supports:
    - undertheseanlp/UTS_VLC (~20GB) - Large scale pretraining
    - Local custom texts - Quick testing
    
    Shape:
        - input_ids: (L,) - token indices
        - attention_mask: (L,) - 1 for real tokens
        - labels: (L,) - same as input_ids for LM
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DataConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: Tokenizer for encoding text
            config: Data configuration
            split: Dataset split ("train", "validation", "test")
            max_samples: Maximum number of samples (for debugging)
        """
        self.tokenizer = tokenizer
        self.max_seq_length = config.pretrain_max_seq_length
        self.split = split
        self.max_samples = max_samples or config.pretrain_max_samples
        
        # Load raw text data (strings only — no tensors in memory)
        logger.info(f"Loading {config.pretrain_dataset_name} dataset (split={split})...")
        logger.info(f"This may take a while for large datasets...")
        raw_texts = self._load_data(config, split)
        
        # Pre-chunk: concatenate all text and split into windows.
        # We store token-ID lists (int32), NOT torch.Tensors, to save ~4× RAM.
        self._chunks = self._build_chunks(raw_texts)
        
        logger.info(f"PretrainDataset ({split}): {len(self._chunks)} examples "
                     f"(lazy — tensors created on access)")
    
    def _load_data(self, config: DataConfig, split: str) -> List[str]:
        """Load raw text data from various sources."""
        try:
            from datasets import load_dataset
            import time
            
            start_time = time.time()
            dataset_name = config.pretrain_dataset_name
            
            # Different loading strategies for different datasets
            if dataset_name == "undertheseanlp/UTS_VLC":
                # UTS_VLC is ~20GB, load with streaming or limit samples
                logger.info(f"Loading UTS_VLC dataset...")
                logger.info(f"Max samples: {self.max_samples}")
                
                if config.pretrain_streaming:
                    # Streaming mode for very large datasets
                    dataset = load_dataset(
                        "undertheseanlp/UTS_VLC",
                        split=split if split != "validation" else "train",
                        streaming=True,
                        cache_dir=config.pretrain_data_dir,
                        download_mode=DOWNLOAD_MODE,
                    )
                    texts = []
                    for i, item in enumerate(dataset):
                        if self.max_samples and i >= self.max_samples:
                            break
                        texts.append(item["text"].strip())
                        if (i + 1) % 100000 == 0:
                            logger.info(f"  Loaded {i + 1} samples...")
                else:
                    # Non-streaming mode
                    dataset = load_dataset(
                        "undertheseanlp/UTS_VLC",
                        split="train",  # UTS_VLC primarily uses train split
                        cache_dir=config.pretrain_data_dir,
                        download_mode=DOWNLOAD_MODE,
                    )
                    # For validation, use last 10% of data
                    total_size = len(dataset)
                    if split == "validation":
                        start_idx = int(total_size * 0.9)
                        indices = range(start_idx, min(start_idx + (self.max_samples or total_size) // 10, total_size))
                    else:
                        # Train uses first 90%
                        end_idx = min(self.max_samples or total_size, int(total_size * 0.9))
                        indices = range(0, end_idx)
                    
                    texts = []
                    for i in indices:
                        texts.append(dataset[i]["text"].strip())
                        if (len(texts)) % 100000 == 0:
                            logger.info(f"  Loaded {len(texts)} samples...")
                            
            elif dataset_name == "wikitext" or dataset_name == "test_local":
                # Quick small tests
                dataset = load_dataset(
                    config.pretrain_dataset_name,
                    config.pretrain_dataset_config,
                    cache_dir=config.pretrain_data_dir,
                    download_mode=DOWNLOAD_MODE,
                )
                
                if split == "train":
                    texts = dataset["train"]["text"]
                elif split == "validation":
                    texts = dataset["validation"]["text"]
                else:
                    texts = dataset["test"]["text"]
                
                texts = [t.strip() for t in texts if t.strip()]
                
            else:
                # Generic loading for other datasets
                dataset = load_dataset(
                    dataset_name,
                    config.pretrain_dataset_config if config.pretrain_dataset_config else None,
                    cache_dir=config.pretrain_data_dir,
                    download_mode=DOWNLOAD_MODE,
                )
                
                if split in dataset:
                    data_split = dataset[split]
                else:
                    data_split = dataset["train"]
                    
                # Try common text field names
                text_field = None
                for field in ["text", "content", "document", "sentence"]:
                    if field in data_split.features:
                        text_field = field
                        break
                
                if text_field is None:
                    text_field = list(data_split.features.keys())[0]
                    
                texts = []
                for i, item in enumerate(data_split):
                    if self.max_samples and i >= self.max_samples:
                        break
                    text = str(item[text_field]).strip()
                    if text:
                        texts.append(text)
            
            elapsed = time.time() - start_time
            logger.info(f"  Dataset loaded in {elapsed:.1f}s, {len(texts)} texts")
            
            # Filter empty lines
            texts = [t for t in texts if t]
            return texts
            
        except Exception as e:
            logger.error(f"Failed to load dataset from HuggingFace: {e}")
            logger.warning("Generating synthetic data for demonstration...")
            return self._generate_synthetic_data(split)
    
    def _generate_synthetic_data(self, split: str) -> List[str]:
        """Generate synthetic data for testing."""
        templates = [
            "Theo Hiến pháp nước Cộng hòa xã hội chủ nghĩa Việt Nam, mọi công dân đều bình đẳng trước pháp luật.",
            "Bộ luật Hình sự năm 2015 quy định cụ thể về các tội phạm xâm phạm tính mạng, sức khỏe, nhân phẩm, danh dự của con người.",
            "Hợp đồng dân sự là sự thỏa thuận giữa các bên về việc xác lập, thay đổi hoặc chấm dứt quyền, nghĩa vụ dân sự.",
            "Người lao động có quyền đơn phương chấm dứt hợp đồng lao động nhưng phải báo trước cho người sử dụng lao động.",
            "Tài sản chung của vợ chồng gồm tài sản do vợ, chồng tạo ra, thu nhập do lao động, hoạt động sản xuất, kinh doanh.",
            "Tòa án nhân dân là cơ quan xét xử của nước Cộng hòa xã hội chủ nghĩa Việt Nam, thực hiện quyền tư pháp.",
            "Thuế thu nhập cá nhân được tính dựa trên các khoản thu nhập chịu thuế sau khi đã trừ các khoản giảm trừ gia cảnh.",
            "Doanh nghiệp bảo hiểm phải bồi thường, trả tiền bảo hiểm khi xảy ra sự kiện bảo hiểm theo thỏa thuận trong hợp đồng.",
            "Đất đai thuộc sở hữu toàn dân do Nhà nước đại diện chủ sở hữu và thống nhất quản lý.",
            "Cơ quan nhà nước, người có thẩm quyền phải chịu trách nhiệm bồi thường khi gây thiệt hại cho cá nhân, tổ chức.",
        ]
        
        num_samples = 5000 if split == "train" else 500
        data = []
        for _ in range(num_samples):
            # Combine random templates
            num_sentences = random.randint(3, 10)
            text = " ".join(random.choices(templates, k=num_sentences))
            data.append(text)
        
        return data
    
    def _build_chunks(self, texts: List[str]) -> List[List[int]]:
        """
        Tokenize all text and split into fixed-length chunks.
        
        Stores token IDs as Python lists (int32 each ≈ 28 bytes/element)
        instead of torch.Tensors (which carry CUDA/grad metadata overhead).
        For 100K chunks × 512 tokens, this uses ~1.4 GB vs ~5.6 GB with tensors.
        """
        import time

        logger.info(f"Tokenizing and chunking data...")
        start_time = time.time()

        batch_size = 1000
        all_tokens: List[int] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_text = " ".join(batch)
            tokens = self.tokenizer.encode(batch_text, add_special_tokens=False)
            all_tokens.extend(tokens)

            if (i + batch_size) % 50000 == 0 or i + batch_size >= len(texts):
                elapsed = time.time() - start_time
                logger.info(
                    f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts, "
                    f"{len(all_tokens)} tokens, {elapsed:.1f}s"
                )

        # Chunk into fixed-length windows (store as plain Python lists)
        chunks: List[List[int]] = []
        max_chunks = self.max_samples if self.max_samples else float('inf')

        stride = self.max_seq_length // 2
        for start in range(0, max(1, len(all_tokens) - self.max_seq_length + 1), stride):
            chunk = all_tokens[start:start + self.max_seq_length]
            if len(chunk) < self.max_seq_length and len(all_tokens) >= self.max_seq_length:
                break
            chunks.append(chunk)
            if len(chunks) >= max_chunks:
                break

        elapsed = time.time() - start_time
        total_tokens = len(chunks) * self.max_seq_length
        ram_mb = total_tokens * 4 / 1024 / 1024  # int32 = 4 bytes
        logger.info(f"  Created {len(chunks)} chunks ({total_tokens:,} tokens) in {elapsed:.1f}s")
        logger.info(f"  Estimated RAM: {ram_mb:.1f} MB (plain int lists, no tensor overhead)")

        return chunks
    
    def __len__(self) -> int:
        return len(self._chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Lazy: create tensors only when accessed by the DataLoader."""
        chunk = self._chunks[idx]
        input_ids = torch.tensor(chunk, dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones(len(chunk), dtype=torch.long),
            'labels': input_ids.clone(),
        }


# SFT Dataset (Lazy Tokenization)
@dataclass
class SFTExample:
    """Single SFT example with instruction and response."""
    instruction: str
    input: str
    output: str


class SFTDataset(Dataset):
    """
    Dataset for Supervised Fine-Tuning.
    
    Uses legal-format instruction data with:
    - instruction: Task description
    - input: Optional context
    - output: Expected response
    
    **Lazy tokenization**: raw (instruction, input, output) triples are
    stored; tokenization happens in __getitem__().
    
    Shape:
        - input_ids: (L,) - token indices
        - attention_mask: (L,) - 1 for real tokens
        - labels: (L,) - -100 for prompt tokens, token ids for response
    """
    
    # Prompt template
    PROMPT_TEMPLATE = """### Câu hỏi pháp lý:
{instruction}

### Ngữ cảnh:
{input}

### Trả lời:
{output}"""
    
    PROMPT_TEMPLATE_NO_INPUT = """### Câu hỏi pháp lý:
{instruction}

### Trả lời:
{output}"""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DataConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: Tokenizer for encoding
            config: Data configuration
            split: Dataset split
            max_samples: Maximum samples to use
        """
        self.tokenizer = tokenizer
        self.max_seq_length = config.sft_max_seq_length
        self.split = split
        max_samples = max_samples or config.sft_max_samples
        
        # Store raw examples (strings only — no tensors)
        self._raw_examples: List[SFTExample] = self._load_raw_data(config, split, max_samples)
        
        logger.info(f"SFTDataset ({split}): {len(self._raw_examples)} examples "
                     f"(lazy — tensors created on access)")
    
    def _load_raw_data(
        self, 
        config: DataConfig, 
        split: str,
        max_samples: int,
    ) -> List[SFTExample]:
        """Load raw SFT data as string triples (no tokenization)."""
        try:
            from datasets import load_dataset
            
            # Load Legal Instruction dataset
            dataset = load_dataset(
                config.sft_dataset_name,
                cache_dir=config.sft_data_dir,
                download_mode=DOWNLOAD_MODE,
            )
            
            # Get data
            if "train" in dataset:
                data = dataset["train"]
            else:
                data = list(dataset.values())[0]
            
            # Collect raw examples
            examples = []
            for i, item in enumerate(data):
                if i >= max_samples:
                    break
                
                instr = item.get("instruction", "")
                inp = item.get("input", "")
                out = item.get("output", "")
                
                if instr and out:
                    examples.append(SFTExample(instruction=instr, input=inp, output=out))
            
            # Split for validation
            if split == "validation":
                examples = examples[:int(len(examples) * 0.1)]
            elif split == "train":
                examples = examples[int(len(examples) * 0.1):]
            
            return examples
            
        except Exception as e:
            logger.error(f"Failed to load SFT dataset: {e}")
            logger.warning("Generating synthetic SFT data...")
            return self._generate_synthetic_data(split, max_samples)
    
    def _tokenize_example(self, ex: SFTExample) -> Dict[str, torch.Tensor]:
        """Tokenize a single SFT example on-the-fly."""
        # Format prompt
        if ex.input.strip():
            prompt = self.PROMPT_TEMPLATE.format(
                instruction=ex.instruction, input=ex.input, output="",
            )
            full_text = self.PROMPT_TEMPLATE.format(
                instruction=ex.instruction, input=ex.input, output=ex.output,
            )
        else:
            prompt = self.PROMPT_TEMPLATE_NO_INPUT.format(
                instruction=ex.instruction, output="",
            )
            full_text = self.PROMPT_TEMPLATE_NO_INPUT.format(
                instruction=ex.instruction, output=ex.output,
            )
        
        # Tokenize
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=True)
        
        # Truncate if necessary
        if len(full_ids) > self.max_seq_length:
            full_ids = full_ids[:self.max_seq_length]
            prompt_ids = prompt_ids[:min(len(prompt_ids), len(full_ids))]
        
        # Create labels: -100 for prompt (ignored in loss), token ids for response
        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        labels = labels[:len(full_ids)]
        
        # Pad to max length
        padding_length = self.max_seq_length - len(full_ids)
        if padding_length > 0:
            full_ids = full_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
            attention_mask = [1] * (self.max_seq_length - padding_length) + [0] * padding_length
        else:
            attention_mask = [1] * len(full_ids)
        
        return {
            'input_ids': torch.tensor(full_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }
    
    def _generate_synthetic_data(
        self, 
        split: str, 
        max_samples: int,
    ) -> List[SFTExample]:
        """Generate synthetic SFT data."""
        instructions = [
            ("Luật sư cho tôi hỏi mức phạt đối với tội lừa đảo chiếm đoạt tài sản là bao nhiêu?", "", "Mức phạt phụ thuộc vào giá trị tài sản chiếm đoạt, từ cải tạo không giam giữ đến án tù chung thân theo Điều 174 Bộ luật Hình sự 2015."),
            ("Điều kiện để ly hôn thuận tình là gì?", "Cả hai vợ chồng cùng đồng ý.", "Cần có sự thỏa thuận tự nguyện của hai bên, đã giải quyết xong về tài sản và con chung theo Điều 55 Luật Hôn nhân và Gia đình 2014."),
            ("Tội vi phạm quy định về tham gia giao thông đường bộ?", "", "Theo Điều 260 BLHS 2015, người vi phạm có thể bị phạt tiền hoặc phạt tù tùy theo mức độ thiệt hại gây ra."),
            ("Lao động nữ thai sản được nghỉ bao lâu?", "", "Lao động nữ được nghỉ sinh con 6 tháng, trước sinh không quá 2 tháng theo Luật Bảo hiểm xã hội."),
            ("Công ty không trả lương đúng hạn bị phạt thế nào?", "", "Nếu chậm lương trên 15 ngày, công ty phải trả thêm tiền lãi cho kỷ luật thanh toán chậm."),
            ("Chi phí sang tên sổ đỏ gồm những gì?", "Chuyển nhượng quyền sử dụng đất.", "Gồm: Thuế thu nhập cá nhân 2%, Lệ phí trước bạ 0.5% và lệ phí cấp Giấy chứng nhận."),
            ("Người dưới 18 tuổi có được ký hợp đồng lao động?", "Lao động 16 tuổi", "Được ký nhưng phải có sự đồng ý bằng văn bản của người đại diện theo pháp luật đối với người chưa đủ 15 tuổi."),
            ("Các loại thuế doanh nghiệp phải nộp?", "", "1. Thuế thu nhập doanh nghiệp\n2. Thuế giá trị gia tăng (VAT)\n3. Thuế môn bài"),
        ]
        
        num_samples = min(max_samples, 1000 if split == "train" else 100)
        examples = []
        
        for _ in range(num_samples):
            instr, inp, out = random.choice(instructions)
            examples.append(SFTExample(instruction=instr, input=inp, output=out))
        
        return examples
    
    def __len__(self) -> int:
        return len(self._raw_examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Lazy: tokenize on access."""
        return self._tokenize_example(self._raw_examples[idx])


# RL Dataset (Lazy Tokenization)
class RLDataset(Dataset):
    """
    Dataset for Reinforcement Learning (GRPO).
    
    Uses HH-RLHF prompts for generating responses.
    Only contains prompts, responses are generated during RL training.
    
    **Lazy tokenization**: raw prompt strings are stored;
    tokenization happens in __getitem__().
    
    Shape:
        - input_ids: (L,) - tokenized prompt
        - attention_mask: (L,) - 1 for real tokens
        - prompt_text: str - original prompt text (for generation)
    """
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DataConfig,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            tokenizer: Tokenizer for encoding
            config: Data configuration
            split: Dataset split
            max_samples: Maximum samples
        """
        self.tokenizer = tokenizer
        self.max_seq_length = config.rl_max_seq_length
        self.split = split
        max_samples = max_samples or config.rl_max_samples
        
        # Store raw prompts only (strings — no tensors)
        self.prompts: List[str] = self._load_prompts(config, split, max_samples)
        
        logger.info(f"RLDataset ({split}): {len(self.prompts)} prompts "
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
            
            # Load HH-RLHF dataset
            dataset = load_dataset(
                config.rl_dataset_name,
                config.rl_dataset_config,
                cache_dir=config.rl_data_dir,
                download_mode=DOWNLOAD_MODE,
            )
            
            # Get split
            if split in dataset:
                data = dataset[split]
            else:
                data = dataset["train"]
            
            # Extract prompts (human turns from conversations)
            prompts = []
            for item in data:
                # HH-RLHF format: "Human: ... Assistant: ..."
                chosen = item.get("chosen", "")
                if "Human:" in chosen:
                    # Extract first human turn
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
            logger.error(f"Failed to load RL dataset: {e}")
            logger.warning("Generating synthetic RL prompts...")
            return self._generate_synthetic_prompts(split, max_samples)
    
    def _generate_synthetic_prompts(
        self,
        split: str,
        max_samples: int,
    ) -> List[str]:
        """Generate synthetic RL prompts."""
        prompt_templates = [
            "Human: Can you explain what machine learning is?\n\nAssistant:",
            "Human: Write a short poem about nature.\n\nAssistant:",
            "Human: What are the benefits of exercise?\n\nAssistant:",
            "Human: How do computers work?\n\nAssistant:",
            "Human: Tell me about space exploration.\n\nAssistant:",
            "Human: What is the meaning of life?\n\nAssistant:",
            "Human: Explain quantum physics simply.\n\nAssistant:",
            "Human: What makes a good leader?\n\nAssistant:",
            "Human: How can I learn to code?\n\nAssistant:",
            "Human: Describe a perfect day.\n\nAssistant:",
        ]
        
        num_samples = min(max_samples, 200 if split == "train" else 20)
        prompts = [random.choice(prompt_templates) for _ in range(num_samples)]
        
        return prompts
    
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


# Data Collator
def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    Collate batch of examples.
    
    Args:
        batch: List of example dicts
        
    Returns:
        Batched tensors
    """
    # Stack tensors
    result = {}
    
    # Get all keys except non-tensor fields
    tensor_keys = [k for k in batch[0].keys() if isinstance(batch[0][k], torch.Tensor)]
    
    for key in tensor_keys:
        result[key] = torch.stack([ex[key] for ex in batch])
    
    # Handle non-tensor fields (like prompt_text)
    for key in batch[0].keys():
        if key not in tensor_keys:
            result[key] = [ex[key] for ex in batch]
    
    return result


# Test Functions
def test_datasets():
    """Test all dataset classes using the unified VLexRIA data pipeline."""
    from vlex_ria.core.config import load_config
    from vlex_ria.data import create_dataloaders  # ← canonical source (dataloaders.py)
    
    logger.info("=" * 100)
    logger.info("Testing VLexRIA Datasets")
    logger.info("=" * 100)
    
    # Load config and tokenizer
    config = load_config()
    tokenizer = get_tokenizer(config.data)
    
    logger.info(f"Tokenizer: {config.data.tokenizer_name}")
    logger.info(f"Vocab size: {len(tokenizer)}")
    logger.info(f"Pad token: {tokenizer.pad_token}")
    
    # Test pretrain dataset
    logger.info("-" * 100)
    logger.info("Testing PretrainDataset...")
    pretrain_train, pretrain_val = create_dataloaders(
        config.data, tokenizer, mode="pretrain",
        batch_size=4, max_samples=100,
    )
    
    batch = next(iter(pretrain_train))
    logger.info(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    logger.info(f"  Batch attention_mask shape: {batch['attention_mask'].shape}")
    logger.info(f"  Batch labels shape: {batch['labels'].shape}")
    logger.info(f"  Sample text: {tokenizer.decode(batch['input_ids'][0][:50])}...")
    
    # Test SFT dataset
    logger.info("-" * 100)
    logger.info("Testing SFTDataset...")
    sft_train, sft_val = create_dataloaders(
        config.data, tokenizer, mode="sft",
        batch_size=4, max_samples=100,
    )
    
    batch = next(iter(sft_train))
    logger.info(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    logger.info(f"  Sample instruction: {tokenizer.decode(batch['input_ids'][0][:100])}...")
    
    # Test RL dataset
    logger.info("-" * 100)
    logger.info("Testing RLDataset...")
    rl_train, rl_val = create_dataloaders(
        config.data, tokenizer, mode="rl",
        batch_size=4, max_samples=50,
    )
    
    batch = next(iter(rl_train))
    logger.info(f"  Batch input_ids shape: {batch['input_ids'].shape}")
    logger.info(f"  Sample prompt: {batch['prompt_text'][0][:100]}...")
    
    logger.info("=" * 100)
    logger.info("All dataset tests passed!")
    logger.info("=" * 100)


if __name__ == "__main__":
    test_datasets()
