"""
Execution Agents (Transformer-Based)
=====================================
Four specialized execution agents, one for each strategy type:
1. TrendFollowingAgent
2. MeanReversionAgent
3. MomentumAgent
4. RiskOffAgent

Each agent uses Transformer architecture and outputs BUY/SELL/HOLD actions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Optional, Tuple
import numpy as np

from transformer_encoders import MultiScaleTransformerEncoder
from cross_scale_attention import CrossScaleAttention


def replace_nan_preserve_grad(tensor: torch.Tensor, replacement: float = 0.0) -> torch.Tensor:
    """
    Replace NaN/Inf values while preserving gradient flow.
    
    CRITICAL: torch.nan_to_num() creates a NEW tensor without grad_fn, breaking gradients!
    This function uses torch.where() to preserve the computation graph connection.
    
    Strategy:
    1. If tensor has grad_fn, use torch.where() to maintain the connection
    2. If tensor doesn't have grad_fn (e.g., input data), that's OK - inputs don't need gradients
    
    Args:
        tensor: Input tensor (may have NaN/Inf)
        replacement: Value to replace NaN/Inf with (default: 0.0)
    
    Returns:
        Tensor with NaN/Inf replaced, but gradient connection preserved if it existed
    """
    # Check if tensor has NaN or Inf
    mask = torch.isnan(tensor) | torch.isinf(tensor)
    
    # If no NaN/Inf, return original (preserves gradient connection)
    if not mask.any():
        return tensor
    
    # Use torch.where to preserve gradient connection
    # Key insight: torch.where() maintains grad_fn from the original tensor
    # If tensor has grad_fn, the result will too (through the where operation)
    # If tensor doesn't have grad_fn (input data), result won't either (which is fine)
    replacement_tensor = torch.full_like(tensor, replacement)
    result = torch.where(mask, replacement_tensor, tensor)
    
    # DEBUG: Verify gradient connection is preserved (if original had it)
    # Note: This is just for debugging, not a requirement
    # original_has_grad_fn = tensor.grad_fn is not None
    # result_has_grad_fn = result.grad_fn is not None
    # if original_has_grad_fn and not result_has_grad_fn:
    #     # This shouldn't happen with torch.where, but log if it does
    #     import logging
    #     logging.warning("Gradient connection lost in replace_nan_preserve_grad!")
    
    return result


class ExecutionAgent(nn.Module):
    """
    Execution agent with Transformer backbone.
    
    Architecture:
    - Multi-scale Transformer encoders
    - Cross-scale attention
    - Strategy-specific feature processing
    - Actor head (BUY/SELL/HOLD)
    - Critic head (value estimation)
    """
    
    def __init__(self,
                 strategy_type: str,
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
            strategy_type: "TREND_FOLLOW", "MEAN_REVERT", "MOMENTUM", or "RISK_OFF"
            d_model: Transformer model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feed-forward dimension
            dropout: Dropout rate
            input_dim: Input feature dimension
            mode: "crypto" or "stock"
            hidden_dim: Hidden dimension for actor/critic heads
        """
        super(ExecutionAgent, self).__init__()
        
        self.strategy_type = strategy_type
        self.d_model = d_model
        self.mode = mode
        self.num_actions = 3  # BUY, SELL, HOLD
        
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
        # Stock mode now has 3 scales: 1h, 1d, 1w (added 1h support)
        num_scales = 3  # Both crypto and stock now have 3 scales
        transformer_dim = num_scales * d_model
        cross_attention_dim = d_model
        
        # Strategy-specific features (position state, etc.)
        strategy_feat_dim = 3  # shares > 0, shares == 0, entry_price_normalized
        
        combined_dim = transformer_dim + cross_attention_dim + strategy_feat_dim
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Strategy-specific feature processing (after fusion)
        self.strategy_processor = self._create_strategy_processor(strategy_type, hidden_dim)
        
        # Actor head (BUY/SELL/HOLD) - uses processed dimension
        processed_dim = hidden_dim // 2
        self.actor_head = nn.Linear(processed_dim, self.num_actions)
        
        # Critic head (value estimation) - uses processed dimension
        processed_dim = hidden_dim // 2
        self.critic_head = nn.Linear(processed_dim, 1)
    
    def _create_strategy_processor(self, strategy_type: str, hidden_dim: int) -> nn.Module:
        """
        Create strategy-specific feature processor.
        
        Each strategy may process features differently.
        """
        if strategy_type == "TREND_FOLLOW":
            # Trend following: Emphasize momentum features
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            )
        elif strategy_type == "MEAN_REVERT":
            # Mean reversion: Emphasize deviation from mean
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            )
        elif strategy_type == "MOMENTUM":
            # Momentum: Similar to trend but more aggressive
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            )
        else:  # RISK_OFF
            # Risk-off: Conservative processing
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU()
            )
    
    def _get_strategy_features(self, 
                               shares: float,
                               entry_price: float,
                               current_price: float) -> torch.Tensor:
        """
        Get strategy-specific features (position state).
        
        Args:
            shares: Current position size
            entry_price: Entry price
            current_price: Current price
        
        Returns:
            Strategy features (3,)
        """
        has_position = 1.0 if shares > 0 else 0.0
        no_position = 1.0 if shares == 0 else 0.0
        
        if entry_price > 0:
            price_ratio = current_price / entry_price
        else:
            price_ratio = 1.0
        
        return torch.tensor([has_position, no_position, price_ratio], dtype=torch.float32)
    
    def forward(self,
                features_dict: Dict[str, torch.Tensor],
                strategy_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features_dict: Dictionary of multi-scale features
            strategy_features: Strategy-specific features (batch, 3)
        
        Returns:
            logits: Action logits (batch, 3)
            value: State value (batch, 1)
        """
        # Encode multi-scale features
        encoded = self.transformer_encoder(features_dict)
        
        # Get concatenated last hidden states
        transformer_features = self.transformer_encoder.get_last_hidden_state(encoded)
        
        # DEBUG: Check for NaN/Inf in intermediate outputs
        # CRITICAL: Use gradient-preserving replacement to maintain computation graph!
        # torch.nan_to_num() breaks gradients - use replace_nan_preserve_grad() instead
        if torch.isnan(transformer_features).any() or torch.isinf(transformer_features).any():
            transformer_features = replace_nan_preserve_grad(transformer_features, replacement=0.0)
        
        # Cross-scale attention
        cross_features = self.cross_attention(encoded)
        
        if torch.isnan(cross_features).any() or torch.isinf(cross_features).any():
            cross_features = replace_nan_preserve_grad(cross_features, replacement=0.0)
        
        # Concatenate all features
        combined = torch.cat([
            transformer_features,
            cross_features,
            strategy_features
        ], dim=1)
        
        if torch.isnan(combined).any() or torch.isinf(combined).any():
            combined = replace_nan_preserve_grad(combined, replacement=0.0)
        
        # Feature fusion
        fused = self.feature_fusion(combined)
        
        if torch.isnan(fused).any() or torch.isinf(fused).any():
            fused = replace_nan_preserve_grad(fused, replacement=0.0)
        
        # Strategy-specific processing
        processed = self.strategy_processor(fused)
        
        if torch.isnan(processed).any() or torch.isinf(processed).any():
            processed = replace_nan_preserve_grad(processed, replacement=0.0)
        
        # Actor and critic
        logits = self.actor_head(processed)
        value = self.critic_head(processed)
        
        # Final NaN/Inf check - use gradient-preserving replacement
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # Use gradient-preserving replacement instead of zeros_like
            logits = replace_nan_preserve_grad(logits, replacement=0.0)  # Uniform distribution (logits=0)
        
        if torch.isnan(value).any() or torch.isinf(value).any():
            value = replace_nan_preserve_grad(value, replacement=0.0)
        
        return logits, value
    
    def act(self,
            features_dict: Dict[str, torch.Tensor],
            strategy_features: torch.Tensor,
            deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action (BUY/SELL/HOLD).
        
        Args:
            features_dict: Multi-scale features
            strategy_features: Strategy features
            deterministic: If True, select best action; if False, sample
        
        Returns:
            action: Action index (0=HOLD, 1=BUY, 2=SELL)
            log_prob: Log probability
            value: State value
        """
        logits, value = self.forward(features_dict, strategy_features)
        
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
        
        # Clamp action to valid range
        action = torch.clamp(action, 0, 2)
        
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
                 strategy_features: torch.Tensor,
                 actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            features_dict: Multi-scale features
            strategy_features: Strategy features
            actions: Selected actions (batch,)
        
        Returns:
            log_probs: Log probabilities
            values: State values
            entropy: Distribution entropy
        """
        logits, values = self.forward(features_dict, strategy_features)
        
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
                             strategy_features: torch.Tensor) -> np.ndarray:
        """
        Get confidence scores for each action.
        
        Args:
            features_dict: Multi-scale features
            strategy_features: Strategy features
        
        Returns:
            Softmax probabilities for each action (3,)
        """
        with torch.no_grad():
            logits, _ = self.forward(features_dict, strategy_features)
            probs = F.softmax(logits, dim=1)
            return probs.squeeze().cpu().numpy()


# Factory function to create all execution agents
def create_execution_agents(mode: str = "stock",
                           d_model: int = 128,
                           nhead: int = 8,
                           num_layers: int = 4,
                           dim_feedforward: int = 512,
                           dropout: float = 0.1,
                           input_dim: int = 5,
                           hidden_dim: int = 256) -> Dict[str, ExecutionAgent]:
    """
    Create all four execution agents.
    
    Returns:
        Dictionary of agents: {"TREND_FOLLOW": agent, ...}
    """
    strategies = ["TREND_FOLLOW", "MEAN_REVERT", "MOMENTUM", "RISK_OFF"]
    
    agents = {}
    for strategy in strategies:
        agents[strategy] = ExecutionAgent(
            strategy_type=strategy,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            input_dim=input_dim,
            mode=mode,
            hidden_dim=hidden_dim
        )
    
    return agents


def test_execution_agents():
    """Test execution agents"""
    print("Testing Execution Agents...")
    
    # Test single agent
    print("\n1. Testing single ExecutionAgent (TREND_FOLLOW)...")
    agent = ExecutionAgent(strategy_type="TREND_FOLLOW", mode="stock", d_model=128)
    
    features = {
        "1d": torch.randn(2, 30, 5),
        "1w": torch.randn(2, 30, 5)
    }
    strategy_feat = torch.tensor([[1., 0., 1.05], [0., 1., 1.0]])  # Position state
    
    logits, value = agent(features, strategy_feat)
    print(f"Logits shape: {logits.shape}, Value shape: {value.shape}")
    
    action, log_prob, val = agent.act(features, strategy_feat, deterministic=True)
    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    print(f"Action: {action_map.get(action, action)}, Log prob: {log_prob.item():.4f}, Value: {val.item():.4f}")
    
    confidence = agent.get_action_confidence(features, strategy_feat)
    print(f"Confidence: {confidence}")
    
    # Test all agents
    print("\n2. Testing all execution agents...")
    agents = create_execution_agents(mode="stock", d_model=128)
    
    for strategy_name, strategy_agent in agents.items():
        action, _, _ = strategy_agent.act(features, strategy_feat, deterministic=True)
        print(f"{strategy_name}: Action = {action_map.get(action, action)}")
    
    print("\n[SUCCESS] Execution Agents tests passed!")


if __name__ == "__main__":
    test_execution_agents()

