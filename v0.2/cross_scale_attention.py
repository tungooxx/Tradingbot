"""
Cross-Scale Attention
=====================
Attention mechanism that combines features from multiple timescales.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class CrossScaleAttention(nn.Module):
    """
    Cross-scale attention mechanism.
    
    Takes encoded features from multiple timescales and applies attention
    to combine them into a single representation.
    
    Uses custom attention with separate Q, K, V, O projections to match saved models.
    """
    
    def __init__(self, d_model: int = 128, nhead: int = 8, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dropout: Dropout rate
        """
        super(CrossScaleAttention, self).__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        # Query, Key, Value, Output projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale factor for attention
        self.scale = self.head_dim ** -0.5
    
    def forward(self, encoded_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Apply cross-scale attention to encoded features.
        
        Args:
            encoded_dict: Dictionary of encoded features by timescale
                Each tensor has shape (batch, d_model) or (batch, seq_len, d_model)
                
        Returns:
            Combined features: (batch, d_model)
        """
        if not encoded_dict:
            # Return zeros if no features
            batch_size = 1
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.d_model, device=device)
        
        # Collect all encoded features
        encoded_list = []
        for interval, encoded in encoded_dict.items():
            # Handle different input shapes
            if encoded.dim() == 2:
                # (batch, d_model) - use as is
                encoded_list.append(encoded)
            elif encoded.dim() == 3:
                # (batch, seq_len, d_model) - take mean over sequence
                encoded_list.append(encoded.mean(dim=1))
            else:
                # Flatten if needed
                encoded = encoded.view(encoded.size(0), -1)
                if encoded.size(1) == self.d_model:
                    encoded_list.append(encoded)
                else:
                    # Project to d_model if needed
                    if not hasattr(self, f'proj_{interval}'):
                        setattr(self, f'proj_{interval}', 
                               nn.Linear(encoded.size(1), self.d_model).to(encoded.device))
                    proj = getattr(self, f'proj_{interval}')
                    encoded_list.append(proj(encoded))
        
        if not encoded_list:
            batch_size = 1
            device = next(self.parameters()).device
            return torch.zeros(batch_size, self.d_model, device=device)
        
        # Stack features: (num_scales, batch, d_model)
        stacked = torch.stack(encoded_list, dim=0)  # (num_scales, batch, d_model)
        
        # Apply layer norm
        stacked = self.layer_norm(stacked)
        
        # Compute Q, K, V
        Q = self.W_q(stacked)  # (num_scales, batch, d_model)
        K = self.W_k(stacked)  # (num_scales, batch, d_model)
        V = self.W_v(stacked)  # (num_scales, batch, d_model)
        
        # Reshape for multi-head attention
        num_scales, batch_size, _ = Q.shape
        Q = Q.view(num_scales, batch_size, self.nhead, self.head_dim).permute(2, 0, 1, 3)  # (nhead, num_scales, batch, head_dim)
        K = K.view(num_scales, batch_size, self.nhead, self.head_dim).permute(2, 0, 1, 3)  # (nhead, num_scales, batch, head_dim)
        V = V.view(num_scales, batch_size, self.nhead, self.head_dim).permute(2, 0, 1, 3)  # (nhead, num_scales, batch, head_dim)
        
        # Compute attention scores: (nhead, num_scales, batch, num_scales)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (nhead, num_scales, batch, num_scales)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: (nhead, num_scales, batch, head_dim)
        attn_output = torch.matmul(attn_weights, V)  # (nhead, num_scales, batch, head_dim)
        
        # Concatenate heads: (num_scales, batch, d_model)
        attn_output = attn_output.permute(1, 2, 0, 3).contiguous().view(num_scales, batch_size, self.d_model)
        
        # Output projection
        output = self.W_o(attn_output)  # (num_scales, batch, d_model)
        output = self.dropout(output)
        
        # Aggregate across scales (mean pooling)
        combined = output.mean(dim=0)  # (batch, d_model)
        
        return combined



