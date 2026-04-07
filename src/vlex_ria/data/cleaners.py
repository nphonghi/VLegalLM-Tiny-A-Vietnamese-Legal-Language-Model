"""
VLexRIA Data Cleaners and Preprocessing

Combines Vietnamese text segmentation (for PhoBERT)
and dataset batch cleaning utilities.
"""



from typing import Optional, Callable
from vlex_ria.core.utils import get_logger

logger = get_logger(__name__)

# Global flag to avoid repeated import attempts
_UNDERTHESEA_AVAILABLE: Optional[bool] = None
_word_tokenize: Optional[Callable] = None


def _check_underthesea() -> bool:
    """Check if underthesea is available (cached)."""
    global _UNDERTHESEA_AVAILABLE, _word_tokenize
    if _UNDERTHESEA_AVAILABLE is not None:
        return _UNDERTHESEA_AVAILABLE
    try:
        from underthesea import word_tokenize
        _word_tokenize = word_tokenize
        _UNDERTHESEA_AVAILABLE = True
    except ImportError:
        _UNDERTHESEA_AVAILABLE = False
    return _UNDERTHESEA_AVAILABLE


def preprocess_vietnamese(text: str, use_segmentation: bool = True) -> str:
    """
    Preprocess Vietnamese text for PhoBERT tokenization.
    
    Steps:
    1. Strip whitespace
    2. Apply Vietnamese word segmentation (if underthesea available)
       - "quyền sở hữu" -> "quyền sở_hữu"
       - "trách nhiệm hình sự" -> "trách_nhiệm hình_sự"
    
    Args:
        text: Raw Vietnamese text
        use_segmentation: Whether to apply word segmentation
        
    Returns:
        Preprocessed text ready for PhoBERT tokenization
    """
    if not text:
        return text
    
    text = text.strip()
    
    if not use_segmentation:
        return text
    
    if not _check_underthesea():
        return text
    
    try:
        # underthesea word_tokenize joins compound words with underscore
        # e.g., "quyền sở hữu" -> "quyền sở_hữu"
        segmented = _word_tokenize(text, format="text")
        return segmented
    except Exception:
        # Fallback to raw text if segmentation fails
        return text


def is_vietnamese_tokenizer(tokenizer_name: str) -> bool:
    """Check if tokenizer is Vietnamese-optimized."""
    vn_tokenizers = ["phobert", "vinai", "vietai", "bkai"]
    name_lower = tokenizer_name.lower()
    return any(vn in name_lower for vn in vn_tokenizers)

from typing import Dict, Any, List
from datasets import Dataset

class VLexRIADatasetCleaner:
    """
    Cleaner thực hiện các bước làm sạch và tiền xử lý cơ bản đối với dữ liệu VLexRIA.
    """
    
    @staticmethod
    def clean_empty_samples(dataset: Dataset, required_columns: List[str] = None) -> Dataset:
        """
        Lọc bỏ các mẫu tin rỗng ở các cột quan trọng.
        
        Args:
            dataset: Đầu vào kiểu `transformers.Dataset`
            required_columns: Danh sách các cột không được để trống (null hoặc rỗng).
            
        Returns:
            Dataset đã được chuẩn hóa.
        """
        if required_columns is None:
            return dataset
            
        def is_valid(example: Dict[str, Any]) -> bool:
            for col in required_columns:
                val = example.get(col, None)
                if val is None:
                    return False
                if isinstance(val, str) and str(val).strip() == "":
                    return False
            return True
            
        return dataset.filter(is_valid, desc="Cleaning empty samples")
        
    @staticmethod
    def normalize_whitespace(dataset: Dataset, text_columns: List[str]) -> Dataset:
        """
        Xoá khoảng trắng thừa (newline liên tiếp, space thừa) trong text columns.
        """
        import re
        
        def _normalize(example: Dict[str, Any]) -> Dict[str, Any]:
            for col in text_columns:
                if col in example and isinstance(example[col], str):
                    text = example[col]
                    # Loại bỏ khoảng trắng thừa
                    text = re.sub(r' +', ' ', text)
                    # Loại bỏ dòng trắng nhiều liên tiếp (3 dòng trắng trở lên)
                    text = re.sub(r'\n{3,}', '\n\n', text)
                    example[col] = text.strip()
            return example
            
        return dataset.map(_normalize, desc="Normalizing whitespace")
