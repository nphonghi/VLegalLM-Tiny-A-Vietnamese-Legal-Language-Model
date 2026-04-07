"""
VLex-RIA Model Module

Contains the VLexRIA model architecture:
- MLA
- MoE 
- MTP
"""

from .deepseekv3 import (
    DeepSeekV3Model,
    TransformerBlock,
    count_parameters,
    print_model_summary,
)

from .attention import (
    MultiHeadLatentAttention,
    StandardAttention,
    RotaryEmbedding,
    RMSNorm,
    apply_rotary_pos_emb,
)

from .moe import (
    DeepSeekMoE,
    Expert,
    SwiGLU,
)

from .mtp import MTPHead

# Convenience aliases
VLexRIAModel = DeepSeekV3Model

def create_model(config):
    """
    Factory function to create a VLexRIAModel instance.
    
    Args:
        config: ModelConfig instance with model architecture settings.
        
    Returns:
        VLexRIAModel instance.
    """
    return VLexRIAModel(config)

__all__ = [
    "VLexRIAModel",
    "TransformerBlock",
    "DeepSeekMoE",
    "Expert",
    "MTPHead",
    "SwiGLU",
    "count_parameters",
    "print_model_summary",
    "create_model",
    "MultiHeadLatentAttention",
    "StandardAttention",
    "RotaryEmbedding",
    "RMSNorm",
    "apply_rotary_pos_emb",
]
