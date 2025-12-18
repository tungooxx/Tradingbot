"""
Multi-Scale Transformer Encoders
=================================
Implements transformer encoders for multiple timescales (1h, 4h, 1d).

Based on:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (Zhou et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional, Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time series.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for a single timescale.
    
    Architecture:
    - Input embedding
    - Positional encoding
    - Multi-head self-attention
    - Feed-forward network
    - Layer normalization
    """
    
    def __init__(self, 
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 input_dim: int = 5):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feed-forward dimension
            dropout: Dropout rate
            input_dim: Input feature dimension (default: 5 for Close, Log_Ret, Volume, RSI, MACD)
        """
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input embedding: project features to d_model
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,  # (seq_len, batch, d_model)
            activation='gelu'
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        # Handle 2D input (add batch dimension) or ensure 3D
        if x.dim() == 2:
            # (seq_len, input_dim) -> (1, seq_len, input_dim)
            x = x.unsqueeze(0)
        elif x.dim() == 1:
            # (input_dim,) -> (1, 1, input_dim)
            x = x.unsqueeze(0).unsqueeze(0)
        
        batch_size, seq_len, input_dim = x.shape
        
        # Embed input: (batch, seq_len, input_dim) -> (batch, seq_len, d_model)
        # Don't scale by sqrt(d_model) - this can cause instability
        x = self.input_embedding(x)
        
        # Transpose for transformer: (batch, seq_len, d_model) -> (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding (scaled down to prevent large values)
        x = x + self.pos_encoder(x) * 0.1  # Scale down positional encoding
        x = self.dropout(x)
        
        # Transformer encoder: (seq_len, batch, d_model) -> (seq_len, batch, d_model)
        x = self.transformer_encoder(x)
        
        # Transpose back: (seq_len, batch, d_model) -> (batch, seq_len, d_model)
        x = x.transpose(0, 1)
        
        return x


class MultiScaleTransformerEncoder(nn.Module):
    """
    Multi-scale transformer encoder with parallel encoders for different timescales.
    
    For crypto: 1h, 4h, 1d
    For stocks: 1h, 1d, 1w (added 1h support)
    """
    
    def __init__(self,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 input_dim: int = 5,
                 mode: str = "stock"):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers per encoder
            dim_feedforward: Feed-forward dimension
            dropout: Dropout rate
            input_dim: Input feature dimension
            mode: "crypto" or "stock"
        """
        super(MultiScaleTransformerEncoder, self).__init__()
        
        self.mode = mode
        self.d_model = d_model
        
        if mode == "crypto":
            # Three encoders: 1h, 4h, 1d
            self.encoder_1h = TransformerEncoder(
                d_model, nhead, num_layers, dim_feedforward, dropout, input_dim
            )
            self.encoder_4h = TransformerEncoder(
                d_model, nhead, num_layers, dim_feedforward, dropout, input_dim
            )
            self.encoder_1d = TransformerEncoder(
                d_model, nhead, num_layers, dim_feedforward, dropout, input_dim
            )
            self.num_scales = 3
        else:
            # Three encoders: 1h, 1d, 1w (or 1wk) - Added 1h for stock mode
            self.encoder_1h = TransformerEncoder(
                d_model, nhead, num_layers, dim_feedforward, dropout, input_dim
            )
            self.encoder_1d = TransformerEncoder(
                d_model, nhead, num_layers, dim_feedforward, dropout, input_dim
            )
            self.encoder_1w = TransformerEncoder(
                d_model, nhead, num_layers, dim_feedforward, dropout, input_dim
            )
            self.encoder_1wk = self.encoder_1w  # Alias for 1wk
            self.num_scales = 3
    
    def forward(self, 
                features_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all encoders.
        
        Args:
            features_dict: Dictionary of feature tensors
                - For crypto: {"1h": (batch, seq_len, input_dim), "4h": ..., "1d": ...}
                - For stocks: {"1h": (batch, seq_len, input_dim), "1d": ..., "1w": ...}
        
        Returns:
            Dictionary of encoded features
                - Same keys as input, values are (batch, seq_len, d_model)
        """
        encoded = {}
        
        if self.mode == "crypto":
            if "1h" in features_dict:
                encoded["1h"] = self.encoder_1h(features_dict["1h"])
            if "4h" in features_dict:
                encoded["4h"] = self.encoder_4h(features_dict["4h"])
            if "1d" in features_dict:
                encoded["1d"] = self.encoder_1d(features_dict["1d"])
        else:
            # Stock mode: 1h, 1d, 1w (or 1wk)
            if "1h" in features_dict:
                encoded["1h"] = self.encoder_1h(features_dict["1h"])
            if "1d" in features_dict:
                encoded["1d"] = self.encoder_1d(features_dict["1d"])
            if "1w" in features_dict:
                encoded["1w"] = self.encoder_1w(features_dict["1w"])
            elif "1wk" in features_dict:
                # Handle 1wk key (from data preprocessor)
                encoded["1wk"] = self.encoder_1w(features_dict["1wk"])
        
        return encoded
    
    def get_last_hidden_state(self, encoded_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get the last hidden state from each encoder and concatenate.
        
        Args:
            encoded_dict: Dictionary of encoded features
            
        Returns:
            Concatenated last hidden states: (batch, num_scales * d_model)
        """
        hidden_states = []
        
        # For stock mode, ensure we get 1h, 1d, and 1w/1wk
        if self.mode == "stock":
            # Try to get all three scales - we need 3 for stock mode (1h, 1d, 1w)
            if "1h" in encoded_dict:
                last_state = encoded_dict["1h"][:, -1, :]
                hidden_states.append(last_state)
            else:
                # If 1h is missing, create zero tensor
                batch_size = next(iter(encoded_dict.values())).shape[0] if encoded_dict else 1
                device = next(iter(encoded_dict.values())).device if encoded_dict else torch.device('cpu')
                hidden_states.append(torch.zeros(batch_size, self.d_model, device=device))
            
            if "1d" in encoded_dict:
                last_state = encoded_dict["1d"][:, -1, :]
                hidden_states.append(last_state)
            else:
                # If 1d is missing, create zero tensor
                batch_size = next(iter(encoded_dict.values())).shape[0] if encoded_dict else 1
                device = next(iter(encoded_dict.values())).device if encoded_dict else torch.device('cpu')
                hidden_states.append(torch.zeros(batch_size, self.d_model, device=device))
            
            if "1w" in encoded_dict:
                last_state = encoded_dict["1w"][:, -1, :]
                hidden_states.append(last_state)
            elif "1wk" in encoded_dict:
                # Handle 1wk key
                last_state = encoded_dict["1wk"][:, -1, :]
                hidden_states.append(last_state)
            else:
                # If 1w/1wk is missing, create zero tensor
                batch_size = next(iter(encoded_dict.values())).shape[0] if encoded_dict else 1
                device = next(iter(encoded_dict.values())).device if encoded_dict else torch.device('cpu')
                hidden_states.append(torch.zeros(batch_size, self.d_model, device=device))
        else:
            # For crypto mode, get all available
            for key in sorted(encoded_dict.keys()):
                # Get last timestep: (batch, seq_len, d_model) -> (batch, d_model)
                last_state = encoded_dict[key][:, -1, :]
                hidden_states.append(last_state)
        
        if not hidden_states:
            # Return zeros if no states
            batch_size = 1
            device = next(self.parameters()).device if hasattr(self, 'encoder_1d') else torch.device('cpu')
            return torch.zeros(batch_size, self.num_scales * self.d_model, device=device)
        
        # Concatenate: (batch, num_scales * d_model)
        return torch.cat(hidden_states, dim=1)


def test_transformer_encoders():
    """Test the transformer encoders"""
    print("Testing Transformer Encoders...")
    
    # Test single encoder
    print("\n1. Testing single TransformerEncoder...")
    encoder = TransformerEncoder(d_model=128, nhead=8, num_layers=4, input_dim=5)
    x = torch.randn(2, 30, 5)  # (batch=2, seq_len=30, features=5)
    output = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (2, 30, 128), f"Expected (2, 30, 128), got {output.shape}"
    
    # Test multi-scale encoder (crypto)
    print("\n2. Testing MultiScaleTransformerEncoder (crypto)...")
    multi_encoder_crypto = MultiScaleTransformerEncoder(mode="crypto", d_model=128)
    features_crypto = {
        "1h": torch.randn(2, 30, 5),
        "4h": torch.randn(2, 30, 5),
        "1d": torch.randn(2, 30, 5)
    }
    encoded_crypto = multi_encoder_crypto(features_crypto)
    print(f"Encoded keys: {list(encoded_crypto.keys())}")
    for key, val in encoded_crypto.items():
        print(f"{key} encoded shape: {val.shape}")
    
    last_hidden = multi_encoder_crypto.get_last_hidden_state(encoded_crypto)
    print(f"Last hidden state shape: {last_hidden.shape}")
    assert last_hidden.shape == (2, 3 * 128), f"Expected (2, 384), got {last_hidden.shape}"
    
    # Test multi-scale encoder (stock)
    print("\n3. Testing MultiScaleTransformerEncoder (stock)...")
    multi_encoder_stock = MultiScaleTransformerEncoder(mode="stock", d_model=128)
    features_stock = {
        "1d": torch.randn(2, 30, 5),
        "1w": torch.randn(2, 30, 5)
    }
    encoded_stock = multi_encoder_stock(features_stock)
    print(f"Encoded keys: {list(encoded_stock.keys())}")
    for key, val in encoded_stock.items():
        print(f"{key} encoded shape: {val.shape}")
    
    last_hidden_stock = multi_encoder_stock.get_last_hidden_state(encoded_stock)
    print(f"Last hidden state shape: {last_hidden_stock.shape}")
    assert last_hidden_stock.shape == (2, 2 * 128), f"Expected (2, 256), got {last_hidden_stock.shape}"
    
    print("\n[SUCCESS] All tests passed!")


if __name__ == "__main__":
    test_transformer_encoders()

