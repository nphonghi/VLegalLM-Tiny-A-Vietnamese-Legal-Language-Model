import os
from typing import Optional, Any
from datasets import load_dataset
from vlex_ria.core.utils import get_logger

logger = get_logger(__name__)

class HFDatasetDownloader:
    """
    Trình tải ứng dụng dữ liệu từ HuggingFace cho VLexRIA.
    Cung cấp các cơ chế bộ nhớ đệm (caching) và stream data nếu cần.
    """
    def __init__(self, cache_dir: str = "data/vlex_ria_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def download(self, repo_id: str, split: str = "train", streaming: bool = False, config_name: Optional[str] = None, **kwargs) -> Any:
        """
        Tải dataset từ HuggingFace với repo_id cho trước.

        Args:
            repo_id: Tên kho (Ví dụ: 'YuITC/Vietnamese-Legal-Documents' hoặc 'undertheseanlp/UTS_VLC')
            split: Split dữ liệu muốn lấy (e.g. 'train', 'validation', 'test')
            streaming: Chế độ lấy streaming nếu dữ liệu lớn (trả về IterableDataset)
            config_name: Tên config cho dataset (e.g. 'wikitext-2-raw-v1')
            kwargs: Các tham số bổ sung khác cho `datasets.load_dataset`
        
        Returns:
            Một `Dataset` snapshot hoặc một `IterableDataset` nếu bật streaming = True
        """
        logger.info(f"Đang tải dataset {repo_id} (config={config_name}, split={split}) từ Hugging Face...")
        
        try:
            dataset = load_dataset(
                repo_id,
                config_name,
                split=split,
                cache_dir=self.cache_dir,
                streaming=streaming,
                **kwargs
            )
            logger.info(f"Tải thành công dataset {repo_id}.")
            return dataset
        except Exception as e:
            logger.error(f"Lỗi khi tải dataset {repo_id}: {e}")
            raise
