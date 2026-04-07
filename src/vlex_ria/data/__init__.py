"""
VLex-RIA Data Module

Contains dataset implementations for all training stages and VLexRIA data pipeline.
"""

from .dataset import (
    get_tokenizer,
    PretrainDataset,
    SFTDataset,
    RLDataset,
    collate_fn,
)

from .rl_dataset import (
    DPODataset,
    GRPODataset,
    PPODataset,
    create_rl_dataloaders,
)

from .dataloaders import (
    create_dataloaders,
)

from .ingestion import HFDatasetDownloader
from .cleaners import VLexRIADatasetCleaner, preprocess_vietnamese, is_vietnamese_tokenizer
from .adapters import LegalDatasetAdapter

__all__ = [
    "get_tokenizer",
    "create_dataloaders",
    "create_rl_dataloaders",
    "PretrainDataset",
    "SFTDataset",
    "RLDataset",
    "DPODataset",
    "GRPODataset",
    "PPODataset",
    "collate_fn",
    "HFDatasetDownloader",
    "VLexRIADatasetCleaner",
    "preprocess_vietnamese",
    "is_vietnamese_tokenizer",
    "LegalDatasetAdapter",
]
