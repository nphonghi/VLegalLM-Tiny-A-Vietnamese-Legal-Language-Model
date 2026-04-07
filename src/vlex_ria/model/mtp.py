import torch
import torch.nn as nn
from typing import Optional, List
from .attention import RMSNorm

class MTPHead(nn.Module):
    """
    Multi-Token Prediction Head.
    
    Predicts multiple future tokens simultaneously as an auxiliary objective.
    Each prediction head shares the embedding but has separate output projections.
    
    Shape:
        Input: (B, L, D)
        Output: List of (B, L, V) for each prediction depth
    """
    
    def __init__(
        self,
        hidden_size: int,          # D: hidden dimension
        vocab_size: int,           # V: vocabulary size
        num_predict_tokens: int,   # Number of additional tokens to predict
        tie_embeddings: bool = True,
        embedding_weight: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_predict_tokens = num_predict_tokens
        
        # Separate projection for each prediction depth
        # Each transforms hidden state to predict token at depth d
        self.projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_predict_tokens)
        ])
        
        # Output heads (share with embedding if tie_embeddings)
        if tie_embeddings and embedding_weight is not None:
            self.output_heads = nn.ModuleList([
                nn.Linear(hidden_size, vocab_size, bias=False)
                for _ in range(num_predict_tokens)
            ])
            # Tie weights
            for head in self.output_heads:
                head.weight = embedding_weight
        else:
            self.output_heads = nn.ModuleList([
                nn.Linear(hidden_size, vocab_size, bias=False)
                for _ in range(num_predict_tokens)
            ])
        
        self.layer_norms = nn.ModuleList([
            RMSNorm(hidden_size)
            for _ in range(num_predict_tokens)
        ])
    
    def forward(self, hidden_states: torch.Tensor) -> List[torch.Tensor]:
        """
        Compute multi-token predictions.
        
        Args:
            hidden_states: Final hidden states, shape (B, L, D)
            
        Returns:
            List of logits for each prediction depth, each shape (B, L, V)
        """
        predictions = []
        
        for i in range(self.num_predict_tokens):
            # Project hidden states for this depth
            h = self.projections[i](hidden_states)
            h = self.layer_norms[i](h)
            # Compute logits: (B, L, D) -> (B, L, V)
            logits = self.output_heads[i](h)
            predictions.append(logits)
        
        return predictions
