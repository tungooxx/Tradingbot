"""
Meta-Strategy Agent (Transformer-Based)
========================================
High-level agent that selects which trading strategy to use based on:
- Multi-timescale market features
- Current market regime
- Portfolio state

Uses Transformer architecture for temporal pattern recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Optional, Tuple
import numpy as np

from transformer_encoders import MultiScaleTransformerEncoder
from cross_scale_attention import CrossScaleAttention


class MetaStrategyAgent(nn.Module):
    """
    Meta-strategy agent with Transformer backbone.
    
    Architecture:
    - Multi-scale Transformer encoders (1h, 4h, 1d)
    - Cross-scale attention
    - Regime features
    - Actor head (4 strategies)
    - Critic head (value estimation)
    """
    
    def __init__(self,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 input_dim: int = 5,
                 mode: str = "stock",
                 hidden_dim: int = 256):
        """
        Args:
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feed-forward dimension
            dropout: Dropout rate
            input_dim: Input feature dimension
            mode: "crypto" or "stock"
            hidden_dim: Hidden dimension for actor/critic heads
        """
        super(MetaStrategyAgent, self).__init__()
        
        self.d_model = d_model
        self.mode = mode
        self.num_strategies = 4  # TREND_FOLLOW, MEAN_REVERT, MOMENTUM, RISK_OFF
        
        # Multi-scale transformer encoder
        self.transformer_encoder = MultiScaleTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            input_dim=input_dim,
            mode=mode
        )
        
        # Cross-scale attention
        self.cross_attention = CrossScaleAttention(d_model=d_model, nhead=nhead, dropout=dropout)
        
        # Feature dimension after transformer + cross-attention
        # Transformer outputs: num_scales * d_model (concatenated)
        # Cross-attention outputs: d_model
        # Regime features: 4 (one-hot)
        # Stock mode now has 3 scales: 1h, 1d, 1w (added 1h support)
        num_scales = 3  # Both crypto and stock now have 3 scales
        transformer_dim = num_scales * d_model
        cross_attention_dim = d_model
        regime_dim = 4
        
        # Combine transformer features (concatenated) + cross-attention + regime
        combined_dim = transformer_dim + cross_attention_dim + regime_dim
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (strategy selection)
        self.actor_head = nn.Linear(hidden_dim, self.num_strategies)
        
        # Critic head (value estimation)
        self.critic_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, 
                features_dict: Dict[str, torch.Tensor],
                regime_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features_dict: Dictionary of multi-scale features
                - Keys: "1h", "4h", "1d" (crypto) or "1d", "1w" (stock)
                - Values: (batch, seq_len, input_dim)
            regime_features: Regime one-hot encoding (batch, 4)
        
        Returns:
            logits: Strategy logits (batch, 4)
            value: State value (batch, 1)
        """
        # Encode multi-scale features
        encoded = self.transformer_encoder(features_dict)  # Dict of (batch, seq_len, d_model)
        
        # Get concatenated last hidden states from transformer
        transformer_features = self.transformer_encoder.get_last_hidden_state(encoded)  # (batch, num_scales * d_model)
        
        # Cross-scale attention
        cross_features = self.cross_attention(encoded)  # (batch, d_model)
        
        # Concatenate all features
        combined = torch.cat([
            transformer_features,  # (batch, num_scales * d_model)
            cross_features,         # (batch, d_model)
            regime_features        # (batch, 4)
        ], dim=1)  # (batch, combined_dim)
        
        # Feature fusion
        fused = self.feature_fusion(combined)  # (batch, hidden_dim)
        
        # Actor and critic
        logits = self.actor_head(fused)  # (batch, 4)
        value = self.critic_head(fused)  # (batch, 1)
        
        return logits, value
    
    def act(self, 
            features_dict: Dict[str, torch.Tensor],
            regime_features: torch.Tensor,
            deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select strategy action.
        
        Args:
            features_dict: Multi-scale features
            regime_features: Regime features
            deterministic: If True, select best strategy; if False, sample from distribution
        
        Returns:
            action: Strategy index (0-3)
            log_prob: Log probability of selected action
            value: State value estimate
        """
        logits, value = self.forward(features_dict, regime_features)
        
        # Check for NaN in logits
        if torch.isnan(logits).any():
            # Replace NaN with uniform distribution
            logits = torch.ones_like(logits) / logits.shape[-1]
        
        # Clamp logits to prevent extreme values
        logits = torch.clamp(logits, min=-10, max=10)
        
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        # Handle value tensor properly
        value_squeezed = value.squeeze()
        if value_squeezed.dim() == 0:
            value_scalar = value_squeezed.item()
        else:
            value_scalar = value_squeezed[0].item() if len(value_squeezed) > 0 else value_squeezed.item()
        
        if action.dim() == 0:
            return action.item(), log_prob, value_scalar
        else:
            # Batch mode: return first element for single action
            return action[0].item(), log_prob[0], value_scalar
    
    def evaluate(self, 
                 features_dict: Dict[str, torch.Tensor],
                 regime_features: torch.Tensor,
                 actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            features_dict: Multi-scale features
            regime_features: Regime features
            actions: Selected actions (batch,)
        
        Returns:
            log_probs: Log probabilities of actions
            values: State values
            entropy: Distribution entropy
        """
        logits, values = self.forward(features_dict, regime_features)
        
        # Check for NaN in logits
        if torch.isnan(logits).any():
            # Replace NaN with uniform distribution
            logits = torch.ones_like(logits) / logits.shape[-1]
        
        # Clamp logits to prevent extreme values
        logits = torch.clamp(logits, min=-10, max=10)
        
        # Check for NaN in values
        if torch.isnan(values).any():
            values = torch.zeros_like(values)
        
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(), entropy
    
    def get_action_confidence(self,
                             features_dict: Dict[str, torch.Tensor],
                             regime_features: torch.Tensor) -> np.ndarray:
        """
        Get confidence scores for each strategy.
        
        Args:
            features_dict: Multi-scale features
            regime_features: Regime features
        
        Returns:
            Softmax probabilities for each strategy (4,)
        """
        with torch.no_grad():
            logits, _ = self.forward(features_dict, regime_features)
            probs = F.softmax(logits, dim=1)
            return probs.squeeze().cpu().numpy()


def test_meta_strategy_agent():
    """Test meta-strategy agent"""
    print("Testing Meta-Strategy Agent...")
    
    # Test with stock mode
    print("\n1. Testing MetaStrategyAgent (stock mode)...")
    agent_stock = MetaStrategyAgent(mode="stock", d_model=128)
    
    features_stock = {
        "1d": torch.randn(2, 30, 5),
        "1w": torch.randn(2, 30, 5)
    }
    regime_stock = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.]])  # One-hot regime
    
    logits, value = agent_stock(features_stock, regime_stock)
    print(f"Logits shape: {logits.shape}, Value shape: {value.shape}")
    
    action, log_prob, val = agent_stock.act(features_stock, regime_stock, deterministic=True)
    print(f"Action: {action}, Log prob: {log_prob.item():.4f}, Value: {val.item():.4f}")
    
    confidence = agent_stock.get_action_confidence(features_stock, regime_stock)
    print(f"Confidence: {confidence}")
    
    # Test with crypto mode
    print("\n2. Testing MetaStrategyAgent (crypto mode)...")
    agent_crypto = MetaStrategyAgent(mode="crypto", d_model=128)
    
    features_crypto = {
        "1h": torch.randn(2, 30, 5),
        "4h": torch.randn(2, 30, 5),
        "1d": torch.randn(2, 30, 5)
    }
    regime_crypto = torch.tensor([[0., 0., 1., 0.], [0., 0., 0., 1.]])  # One-hot regime
    
    logits, value = agent_crypto(features_crypto, regime_crypto)
    print(f"Logits shape: {logits.shape}, Value shape: {value.shape}")
    
    action, log_prob, val = agent_crypto.act(features_crypto, regime_crypto, deterministic=False)
    print(f"Action: {action}, Log prob: {log_prob.item():.4f}, Value: {val.item():.4f}")
    
    print("\n[SUCCESS] Meta-Strategy Agent tests passed!")


if __name__ == "__main__":
    test_meta_strategy_agent()

