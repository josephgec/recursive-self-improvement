"""LNN attention layer: adds logical bias to attention computation."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LNNAttentionLayer(nn.Module):
    """Logical Neural Network attention layer.

    Augments standard multi-head attention with:
    1. Relation classification (identifies logical vs factual vs computational)
    2. Logical bias (boosts attention for logically related tokens)
    3. Gating (blends logical bias with standard attention)
    """

    def __init__(self, hidden_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Standard attention projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Relation classifier: maps pairs of hidden states to relation type scores
        self.relation_classifier = nn.Linear(hidden_dim * 2, 3)  # 3 relation types

        # Logical bias: learnable bias per relation type per head
        self.logical_bias = nn.Parameter(torch.zeros(num_heads, 3))

        # Gating: blends logical bias with standard attention
        self.gate = nn.Linear(hidden_dim, num_heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            attention_mask: Optional (batch, seq_len) mask.

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = hidden_states.shape

        # Standard QKV
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard attention scores
        scale = self.head_dim ** 0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, heads, seq, seq)

        # Relation classification and logical bias
        relation_probs = self._classify_relations(hidden_states)  # (batch, seq, seq, 3)
        logical_bias = self._compute_logical_bias(relation_probs)  # (batch, heads, seq, seq)

        # Gating: blend logical bias with attention
        gate_values = torch.sigmoid(self.gate(hidden_states))  # (batch, seq, heads)
        gate_values = gate_values.permute(0, 2, 1).unsqueeze(-1)  # (batch, heads, seq, 1)

        # Apply gated logical bias
        attn_scores = attn_scores + gate_values * logical_bias

        # Apply attention mask
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (batch, heads, seq, head_dim)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        output = self.out_proj(attn_output)

        return output

    def _classify_relations(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Classify pairwise relations between positions.

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            relation_probs: (batch, seq_len, seq_len, 3) — softmax probabilities
            for each relation type (logical, factual, computational).
        """
        batch_size, seq_len, hdim = hidden_states.shape

        # Create pairwise features by broadcasting
        h_i = hidden_states.unsqueeze(2).expand(-1, -1, seq_len, -1)
        h_j = hidden_states.unsqueeze(1).expand(-1, seq_len, -1, -1)
        pairs = torch.cat([h_i, h_j], dim=-1)  # (batch, seq, seq, 2*hdim)

        logits = self.relation_classifier(pairs)  # (batch, seq, seq, 3)
        return F.softmax(logits, dim=-1)

    def _compute_logical_bias(self, relation_probs: torch.Tensor) -> torch.Tensor:
        """Compute per-head logical bias from relation classifications.

        Args:
            relation_probs: (batch, seq_len, seq_len, 3)

        Returns:
            bias: (batch, num_heads, seq_len, seq_len)
        """
        # relation_probs: (batch, seq, seq, 3)
        # logical_bias: (num_heads, 3)
        # Einsum: sum over relation types weighted by per-head bias
        bias = torch.einsum("bijk,hk->bhij", relation_probs, self.logical_bias)
        return bias
