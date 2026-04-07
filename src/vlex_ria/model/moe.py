import math
from typing import Optional, Tuple, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from vlex_ria.core.config import MoEConfig

class SwiGLU(nn.Module):
    """
    SwiGLU activation: Swish-Gated Linear Unit.
    
    SwiGLU(x) = Swish(xW_gate) * (xW_up)
    where Swish(x) = x * sigmoid(x)
    
    Shape:
        Input: (B, L, D)
        Output: (B, L, D)
    """
    
    def __init__(
        self,
        hidden_size: int,      # D: input dimension
        intermediate_size: int, # I: FFN hidden dimension
        bias: bool = False,
    ):
        super().__init__()
        # Gate projection: D -> I
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        # Up projection: D -> I
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        # Down projection: I -> D
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (B, L, D)
        Returns:
            Output tensor, shape (B, L, D)
        """
        # gate: (B, L, I), up: (B, L, I)
        gate = F.silu(self.gate_proj(x))  # Swish activation
        up = self.up_proj(x)
        # (B, L, I) -> (B, L, D)
        return self.down_proj(gate * up)


class Expert(nn.Module):
    """
    Single Expert FFN module.
    
    Each expert is an independent FFN that processes tokens routed to it.
    Uses SwiGLU activation following LLaMA/DeepSeek design.
    
    Shape:
        Input: (num_tokens, D) - flattened tokens assigned to this expert
        Output: (num_tokens, D)
    """
    
    def __init__(
        self,
        hidden_size: int,       # D: model dimension
        expert_hidden_size: int, # Expert FFN hidden dimension
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ffn = SwiGLU(hidden_size, expert_hidden_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor, shape (num_tokens, D)
        Returns:
            Output tensor, shape (num_tokens, D)
        """
        return self.dropout(self.ffn(x))


class MoEGate(nn.Module):
    """
    MoE Routing Gate with auxiliary load balancing.
    
    Routes each token to top-K experts using softmax probabilities.
    Implements auxiliary loss for load balancing across experts.
    
    Shape:
        Input: (B, L, D)
        Output:
            - router_probs: (B, L, N) - routing probabilities
            - selected_experts: (B, L, K) - indices of selected experts
            - expert_weights: (B, L, K) - normalized weights for selected experts
    """
    
    def __init__(
        self,
        hidden_size: int,       # D: input dimension
        num_experts: int,       # N: total number of experts
        num_experts_per_tok: int,  # K: experts selected per token
        aux_loss_alpha: float = 0.001,
        router_bias: bool = False,
        router_jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.aux_loss_alpha = aux_loss_alpha
        self.router_jitter_noise = router_jitter_noise
        
        # Router: projects hidden states to expert scores
        # Shape: (D, N)
        self.gate = nn.Linear(hidden_size, num_experts, bias=router_bias)
        
        # Storage for auxiliary loss
        self.aux_loss = None
        self.expert_usage = None
    
    def forward(
        self, 
        hidden_states: torch.Tensor,  # Shape: (B, L, D)
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute routing for MoE layer.
        
        Args:
            hidden_states: Input tensor, shape (B, L, D)
            training: Whether in training mode
            
        Returns:
            router_probs: Full probability distribution, shape (B, L, N)
            selected_experts: Indices of top-K experts, shape (B, L, K)
            expert_weights: Normalized weights for selected, shape (B, L, K)
        """
        B, L, D = hidden_states.shape
        
        # Add jitter noise during training for load balancing
        if training and self.router_jitter_noise > 0:
            hidden_states = hidden_states + torch.randn_like(hidden_states) * self.router_jitter_noise
        
        # Compute router logits: (B, L, D) -> (B, L, N)
        router_logits = self.gate(hidden_states)
        
        # Softmax to get probabilities: (B, L, N)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        
        # Select top-K experts: (B, L, K)
        expert_weights, selected_experts = torch.topk(
            router_probs, self.num_experts_per_tok, dim=-1
        )
        
        # Normalize selected weights to sum to 1: (B, L, K)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights.to(hidden_states.dtype)
        
        # Compute auxiliary load balancing loss
        if training:
            self._compute_aux_loss(router_probs, selected_experts)
        
        return router_probs, selected_experts, expert_weights
    
    def _compute_aux_loss(
        self,
        router_probs: torch.Tensor,   # (B, L, N)
        selected_experts: torch.Tensor,  # (B, L, K)
    ):
        """
        Compute auxiliary loss for load balancing.
        
        The loss encourages balanced expert utilization:
        L_aux = alpha * sum_i(f_i * P_i)
        where f_i is frequency of expert i, P_i is average probability
        """
        B, L, N = router_probs.shape
        num_tokens = B * L
        
        # Count how many tokens each expert receives: (N,)
        expert_mask = F.one_hot(selected_experts, num_classes=N).sum(dim=2)  # (B, L, N)
        expert_counts = expert_mask.sum(dim=(0, 1)).float()  # (N,)
        
        # Fraction of tokens routed to each expert: (N,)
        f_i = expert_counts / (num_tokens * self.num_experts_per_tok)
        
        # Average probability for each expert: (N,)
        P_i = router_probs.mean(dim=(0, 1))
        
        # Auxiliary loss: encourage uniform f_i and P_i
        self.aux_loss = self.aux_loss_alpha * N * (f_i * P_i).sum()
        
        # Store for visualization
        self.expert_usage = expert_counts / expert_counts.sum()


class DeepSeekMoE(nn.Module):
    """
    DeepSeek Mixture of Experts layer.
    
    Combines:
    1. Shared experts (always activated) - provide common knowledge
    2. Routed experts (top-K selected) - provide specialized knowledge
    
    VLexRIA uses auxiliary-loss-free load balancing, but we include
    auxiliary loss for educational purposes.
    
    Shape:
        Input: (B, L, D)
        Output: (B, L, D)
    """
    
    def __init__(
        self,
        hidden_size: int,
        moe_config: MoEConfig,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = moe_config.num_experts
        self.num_experts_per_tok = moe_config.num_experts_per_tok
        self.num_shared_experts = moe_config.num_shared_experts
        self.routed_scaling_factor = moe_config.routed_scaling_factor
        self.layer_idx = layer_idx
        
        # Shared experts (always active)
        if self.num_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                Expert(
                    hidden_size=hidden_size,
                    expert_hidden_size=moe_config.expert_hidden_size,
                    dropout=moe_config.expert_dropout,
                )
                for _ in range(self.num_shared_experts)
            ])
        else:
            self.shared_experts = None
        
        # Routed experts
        self.experts = nn.ModuleList([
            Expert(
                hidden_size=hidden_size,
                expert_hidden_size=moe_config.expert_hidden_size,
                dropout=moe_config.expert_dropout if hasattr(moe_config, 'expert_dropout') else 0.0,
            )
            for _ in range(self.num_experts)
        ])
        
        # Router gate
        self.gate = MoEGate(
            hidden_size=hidden_size,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            aux_loss_alpha=moe_config.aux_loss_alpha,
            router_bias=moe_config.router_bias,
            router_jitter_noise=moe_config.router_jitter_noise,
        )
        
        # Storage for visualization
        self.last_router_probs = None
        self.last_expert_weights = None
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MoE layer.
        
        Args:
            hidden_states: Input tensor, shape (B, L, D)
            
        Returns:
            Output tensor, shape (B, L, D)
        """
        B, L, D = hidden_states.shape
        
        # Initialize output
        output = torch.zeros_like(hidden_states)
        
        # 1. Shared expert computation (always active)
        if self.shared_experts is not None:
            shared_output = torch.zeros_like(hidden_states)
            for expert in self.shared_experts:
                # Each shared expert processes all tokens
                shared_output = shared_output + expert(hidden_states)
            # Sum shared expert outputs (VLexRIA paper: additive, not averaged)
            output = output + shared_output
        
        # 2. Routed expert computation
        # Get routing: (B, L, N), (B, L, K), (B, L, K)
        router_probs, selected_experts, expert_weights = self.gate(
            hidden_states, training=self.training
        )
        
        # Store for visualization
        self.last_router_probs = router_probs.detach()
        self.last_expert_weights = expert_weights.detach()
        
        # Flatten batch and sequence: (B*L, D)
        hidden_flat = hidden_states.view(-1, D)
        selected_flat = selected_experts.view(-1, self.num_experts_per_tok)  # (B*L, K)
        weights_flat = expert_weights.view(-1, self.num_experts_per_tok)     # (B*L, K)
        
        # Process each expert
        routed_output = torch.zeros(B * L, D, device=hidden_states.device, dtype=hidden_states.dtype)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert: (B*L, K) -> boolean mask
            # For each position (token), check if any of the K slots selected this expert
            expert_mask = (selected_flat == expert_idx)  # (B*L, K)
            
            # Get indices of tokens that selected this expert in any slot
            token_indices = expert_mask.any(dim=-1).nonzero(as_tuple=True)[0]  # (num_tokens,)
            
            if len(token_indices) == 0:
                continue
            
            # Get the tokens for this expert: (num_tokens, D)
            expert_input = hidden_flat[token_indices]
            
            # Get weights for these tokens (sum across slots where expert was selected)
            # expert_mask[token_indices]: (num_tokens, K)
            # weights_flat[token_indices]: (num_tokens, K)
            token_weights = (expert_mask[token_indices].float() * weights_flat[token_indices]).sum(dim=-1, keepdim=True)  # (num_tokens, 1)
            
            # Process through expert: (num_tokens, D)
            expert_output = self.experts[expert_idx](expert_input)
            
            # Weighted output
            weighted_output = expert_output * token_weights
            
            # Accumulate to output
            routed_output.index_add_(0, token_indices, weighted_output)
        
        # Reshape and add scaled routed output
        routed_output = routed_output.view(B, L, D)
        output = output + self.routed_scaling_factor * routed_output
        
        return output
    
    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        """Get auxiliary loss from router."""
        return self.gate.aux_loss
