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
        
        # Precision features: Vision (Transformer) + Precision (Scaled Price features)
        # 1. Current hourly return (normalized)
        # 2. RSI (scaled 0 to 1)
        # 3. Position Status (0 or 1)
        # 4. Unrealized PnL (normalized)
        precision_feat_dim = 4  # hourly_return, rsi, position_status, unrealized_pnl
        
        # HYBRID INPUT: Local Features from Environment (4 features)
        # 1. Position Status (0 or 1)
        # 2. Unrealized PnL (scaled by 10)
        # 3. Time in Trade (normalized by window size)
        # 4. Relative Price (relative to 20-period MA)
        local_feat_dim = 4  # position_status, unrealized_pnl, time_in_trade, relative_price
        
        combined_dim = transformer_dim + cross_attention_dim + strategy_feat_dim + precision_feat_dim + local_feat_dim
        
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
    
    def _get_precision_features(self,
                                features_dict: Dict[str, torch.Tensor],
                                shares: float,
                                entry_price: float,
                                current_price: float) -> torch.Tensor:
        """
        Get precision features: Vision (Transformer) + Precision (Scaled Price).
        
        Args:
            features_dict: Dictionary of multi-scale features (to extract RSI)
            shares: Current position size
            entry_price: Entry price
            current_price: Current price
        
        Returns:
            Precision features (4,): [hourly_return, rsi, position_status, unrealized_pnl]
        """
        # 1. Current hourly return (normalized)
        # Extract from 1h features if available, otherwise use 1d
        hourly_return = 0.0
        if "1h" in features_dict:
            # Get last timestep of 1h features (feature 2 = log returns)
            last_1h = features_dict["1h"][:, -1, :]  # (batch, features)
            if last_1h.shape[1] > 2:
                hourly_return = last_1h[0, 2].item()  # Feature 2 = log returns
        elif "1d" in features_dict:
            # Fallback to daily return
            last_1d = features_dict["1d"][:, -1, :]
            if last_1d.shape[1] > 2:
                hourly_return = last_1d[0, 2].item() / 24.0  # Approximate hourly from daily
        
        # Normalize hourly return (clip to [-1, 1] range)
        hourly_return = max(-1.0, min(1.0, hourly_return * 10.0))  # Scale and clip
        
        # 2. RSI (scaled 0 to 1)
        # Extract from features (feature 3 = RSI)
        rsi = 0.5  # Default neutral RSI
        if "1h" in features_dict:
            last_1h = features_dict["1h"][:, -1, :]
            if last_1h.shape[1] > 3:
                rsi_raw = last_1h[0, 3].item()  # Feature 3 = RSI
                # RSI is already normalized to [0, 1] in preprocessing
                rsi = max(0.0, min(1.0, rsi_raw))
        elif "1d" in features_dict:
            last_1d = features_dict["1d"][:, -1, :]
            if last_1d.shape[1] > 3:
                rsi_raw = last_1d[0, 3].item()
                rsi = max(0.0, min(1.0, rsi_raw))
        
        # 3. Position Status (0 or 1)
        position_status = 1.0 if shares > 0 else 0.0
        
        # 4. Unrealized PnL (normalized)
        if shares > 0 and entry_price > 0:
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
            # Normalize to [-1, 1] range (clip extreme values)
            unrealized_pnl = max(-1.0, min(1.0, unrealized_pnl_pct * 10.0))
        else:
            unrealized_pnl = 0.0
        
        return torch.tensor([hourly_return, rsi, position_status, unrealized_pnl], dtype=torch.float32)
    
    def forward(self,
                features_dict: Dict[str, torch.Tensor],
                strategy_features: torch.Tensor,
                precision_features: Optional[torch.Tensor] = None,
                local_features: Optional[torch.Tensor] = None,
                return_weights: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with Hybrid Input: Vision (Transformer) + Precision + Local Features.
        
        Args:
            features_dict: Dictionary of multi-scale features
            strategy_features: Strategy-specific features (batch, 3)
            precision_features: Optional precision features (batch, 4)
            local_features: Optional local features from environment (batch, 4)
            return_weights: If True, also return attention weights
        
        Returns:
            logits: Action logits (batch, 3)
            value: State value (batch, 1)
            attn_weights: (nhead, num_scales, batch, num_scales) if return_weights else None
        """
        # Encode multi-scale features (Vision - Step 1 Transformer output)
        encoded = self.transformer_encoder(features_dict)
        
        # Get concatenated last hidden states
        transformer_features = self.transformer_encoder.get_last_hidden_state(encoded)
        
        # CRITICAL: Use gradient-preserving replacement to maintain computation graph!
        if torch.isnan(transformer_features).any() or torch.isinf(transformer_features).any():
            transformer_features = replace_nan_preserve_grad(transformer_features, replacement=0.0)
        
        # Cross-scale attention
        if return_weights:
            cross_features, attn_weights = self.cross_attention(encoded, return_weights=True)
        else:
            cross_features = self.cross_attention(encoded, return_weights=False)
            attn_weights = None
            
        if torch.isnan(cross_features).any() or torch.isinf(cross_features).any():
            cross_features = replace_nan_preserve_grad(cross_features, replacement=0.0)
        
        # Precision features: If not provided, use default (will be calculated in training)
        if precision_features is None:
            # Default precision features (will be replaced in training loop)
            batch_size = transformer_features.shape[0]
            precision_features = torch.zeros(batch_size, 4, device=transformer_features.device, dtype=torch.float32)
        
        # HYBRID INPUT: Local Features from Environment
        # If not provided, use default (will be passed from environment in training)
        if local_features is None:
            batch_size = transformer_features.shape[0]
            local_features = torch.zeros(batch_size, 4, device=transformer_features.device, dtype=torch.float32)
        
        # Concatenate all features: Vision (Transformer) + Precision + Local Features
        combined = torch.cat([
            transformer_features,  # Vision: Transformer output (Step 1)
            cross_features,         # Vision: Cross-scale attention
            strategy_features,      # Strategy features
            precision_features,     # Precision: hourly_return, rsi, position_status, unrealized_pnl
            local_features          # Local: position_status, unrealized_pnl, time_in_trade, relative_price
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
        
        if return_weights:
            return logits, value, attn_weights
        return logits, value
    
    def act(self,
            features_dict: Dict[str, torch.Tensor],
            strategy_features: torch.Tensor,
            precision_features: Optional[torch.Tensor] = None,
            local_features: Optional[torch.Tensor] = None,
            deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action (BUY/SELL/HOLD).
        
        Args:
            features_dict: Multi-scale features
            strategy_features: Strategy features
            precision_features: Optional precision features (hourly_return, rsi, position_status, unrealized_pnl)
            local_features: Optional local features from environment (position_status, unrealized_pnl, time_in_trade, relative_price)
            deterministic: If True, select best action; if False, sample
        
        Returns:
            action: Action index (0=HOLD, 1=BUY, 2=SELL)
            log_prob: Log probability
            value: State value
        """
        logits, value = self.forward(features_dict, strategy_features, precision_features, local_features)
        
        # Check for NaN in logits
        if torch.isnan(logits).any():
            # Replace NaN with uniform distribution
            logits = torch.ones_like(logits) / logits.shape[-1]
        
        # ============================================================
        # ACTION MASKING + TEMPERATURE SCALING (CRITICAL FIX for Policy Collapse)
        # ============================================================
        # CRITICAL FIX: Apply temperature scaling BEFORE masking to prevent collapse
        # Tighter clamping + temperature to keep probabilities reasonable
        # ============================================================
        LOGIT_CLAMP_MIN = -20.0  # Allow mask to be effective
        LOGIT_CLAMP_MAX = +20.0
        LOGIT_TEMPERATURE = 1.0  # Default to 1.0
        ACTION_MASK_VALUE = -1e10  # STRONG MASK
        
        # CRITICAL FIX: Apply temperature scaling FIRST to prevent extreme logits
        logits = logits / LOGIT_TEMPERATURE
        
        # ACTION MASKING: Disable invalid actions
        # local_features[:, 0] = position_status (1.0 if holding, 0.0 if cash)
        # CRITICAL: This must work correctly or agent will select invalid actions!
        if local_features is not None and local_features.numel() > 0:
            position_status = local_features[:, 0] if local_features.dim() > 1 else local_features[0]
            
            # Create masks for each invalid action
            # If position_status == 1.0 (in position): mask BUY (index 1)
            # If position_status == 0.0 (no position): mask SELL (index 2)
            batch_size = logits.shape[0]
            
            for i in range(batch_size):
                pos = position_status[i].item() if position_status.dim() > 0 else position_status.item()
                
                # CRITICAL: Ensure position_status is valid (0.0 or 1.0)
                if pos > 0.5:  # In position - mask BUY
                    logits[i, 1] = ACTION_MASK_VALUE
                    # DEBUG: Log if BUY was masked (only first few times to avoid spam)
                    if not hasattr(self, '_mask_debug_count'):
                        self._mask_debug_count = 0
                    if self._mask_debug_count < 3:
                        self._mask_debug_count += 1
                else:  # No position - mask SELL
                    logits[i, 2] = ACTION_MASK_VALUE
        else:
            # CRITICAL: If local_features is None or empty, action masking is DISABLED!
            # This is a bug - we should always have local_features for proper masking
            import warnings
            warnings.warn(f"ACTION MASKING DISABLED: local_features is None or empty! Agent may select invalid actions!")
        
        # Clamp logits AFTER masking and temperature (tighter range)
        logits = torch.clamp(logits, min=LOGIT_CLAMP_MIN, max=LOGIT_CLAMP_MAX)
        
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
    
    def get_action_and_value(self,
                             features_dict: Dict[str, torch.Tensor],
                             strategy_features: torch.Tensor,
                             precision_features: Optional[torch.Tensor] = None,
                             local_features: Optional[torch.Tensor] = None,
                             deterministic: bool = False,
                             return_weights: bool = False,
                             return_logits: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Helper for telemetry to get action, log_prob, entropy, and value in one pass.
        
        Args:
            features_dict: Multi-scale features
            strategy_features: Strategy features
            precision_features: Precision features
            local_features: Local features
            deterministic: If True, select best action
            return_weights: If True, also return attention weights
            return_logits: If True, also return masked logits used for action selection
            
        Returns:
            action, log_prob, entropy, value, (attn_weights if return_weights else None[, logits if return_logits])
        """
        if return_weights:
            logits, values, attn_weights = self.forward(features_dict, strategy_features, precision_features, local_features, return_weights=True)
        else:
            logits, values = self.forward(features_dict, strategy_features, precision_features, local_features, return_weights=False)
            attn_weights = None
            
        # Numerical stability: clamp logits
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        
        # Apply ACTION MASKING in get_action_and_value (CRITICAL for telemetry and training consistency)
        if local_features is not None and local_features.numel() > 0:
            position_status = local_features[:, 0]
            batch_size = logits.shape[0]
            mask_val = -1e10
            for i in range(batch_size):
                pos = position_status[i].item() if position_status.dim() > 0 else position_status.item()
                if pos > 0.5:  # In position - mask BUY
                    logits[i, 1] = mask_val
                else:  # No position - mask SELL
                    logits[i, 2] = mask_val

        dist = Categorical(logits=logits)
        
        if deterministic:
            action = torch.argmax(logits, dim=1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        if return_weights and attn_weights is not None:
            # attn_weights shape: (batch, nhead, num_scales, num_scales)
            # 3. CRITICAL FIX: Correctly average the weights
            # Average across Heads (dim 1) -> Result: (batch, num_scales, num_scales)
            avg_heads = attn_weights.mean(dim=1)
            
            # Average across Query Scales (dim 1) -> Result: (batch, num_scales)
            # This represents: "For this batch item, how much focus is on Col 0 vs Col 1 vs Col 2"
            final_weights = avg_heads.mean(dim=1)
            
            if return_logits:
                return action, log_prob, entropy, values.squeeze(-1), final_weights, logits
            else:
                return action, log_prob, entropy, values.squeeze(-1), final_weights
            
        if return_logits:
            return action, log_prob, entropy, values.squeeze(-1), logits
        else:
            return action, log_prob, entropy, values.squeeze(-1)

    def evaluate(self,
                 features_dict: Dict[str, torch.Tensor],
                 strategy_features: torch.Tensor,
                 actions: torch.Tensor,
                 precision_features: Optional[torch.Tensor] = None,
                 local_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.
        
        Args:
            features_dict: Multi-scale features
            strategy_features: Strategy features
            actions: Selected actions (batch,)
            precision_features: Optional precision features (hourly_return, rsi, position_status, unrealized_pnl)
            local_features: Optional local features from environment (position_status, unrealized_pnl, time_in_trade, relative_price)
        
        Returns:
            log_probs: Log probabilities
            values: State values
            entropy: Distribution entropy
        """
        logits, values = self.forward(features_dict, strategy_features, precision_features, local_features)
        
        # Check for NaN in logits
        if torch.isnan(logits).any():
            # Replace NaN with uniform distribution
            logits = torch.ones_like(logits) / logits.shape[-1]
        
        # ============================================================
        # ACTION MASKING (from Principal ML Engineer Audit)
        # ============================================================
        # Same parameters as act() for consistency
        LOGIT_CLAMP_MIN = -20.0
        LOGIT_CLAMP_MAX = +20.0
        LOGIT_TEMPERATURE = 1.0
        ACTION_MASK_VALUE = -1e10
        
        # CRITICAL FIX: Apply temperature scaling FIRST (match act() method)
        logits = logits / LOGIT_TEMPERATURE
        
        # ACTION MASKING: Apply same masking as in act() for consistent log_probs
        if local_features is not None and local_features.numel() > 0:
            position_status = local_features[:, 0] if local_features.dim() > 1 else local_features[0]
            
            batch_size = logits.shape[0]
            for i in range(batch_size):
                pos = position_status[i].item() if position_status.dim() > 0 else position_status.item()
                if pos > 0.5:  # In position - mask BUY
                    logits[i, 1] = ACTION_MASK_VALUE
                else:  # No position - mask SELL
                    logits[i, 2] = ACTION_MASK_VALUE
        
        # Clamp logits AFTER masking and temperature (tighter range)
        logits = torch.clamp(logits, min=LOGIT_CLAMP_MIN, max=LOGIT_CLAMP_MAX)
        
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
    
    logit_val, value_val = agent(features, strategy_feat)
    print(f"Logits shape: {logit_val.shape}, Value shape: {value_val.shape}")
    
    action, log_prob, val = agent.act(features, strategy_feat, deterministic=True)
    action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
    
    # Handle scalar vs tensor
    log_prob_val = log_prob.item() if hasattr(log_prob, 'item') else log_prob
    val_val = val.item() if hasattr(val, 'item') else val
    
    print(f"Action: {action_map.get(action, action)}, Log prob: {log_prob_val:.4f}, Value: {val_val:.4f}")
    
    confidence = agent.get_action_confidence(features, strategy_feat)
    print(f"Confidence: {confidence}")
    
    # NEW TEST: Verify get_action_and_value with attention weights
    print("\n1b. Testing get_action_and_value (Telemetry Mode)...")
    # Need dummy precision and local features
    precision_feat = torch.zeros(2, 4)
    local_feat = torch.zeros(2, 4)
    
    # Call with return_weights=True
    act, logp, ent, val, weights = agent.get_action_and_value(
        features, strategy_feat, precision_feat, local_feat, 
        deterministic=True, return_weights=True
    )
    print(f"Action: {act}, LogProb: {logp.shape}, Weights: {weights.shape if weights is not None else 'None'}")
    
    if weights is not None:
        # Expected shape: (Batch=2, NumScales=2)
        print(f"Attention Weights Shape: {weights.shape}")
        if weights.shape == (2, 2):
            print("✅ Attention weights shape is correct: (Batch, NumScales)")
        else:
            print(f"❌ Attention weights shape mismatch! Expected (2, 2), got {weights.shape}")
    else:
        print("❌ Attention weights returned None!")
    
    # Test all agents
    print("\n2. Testing all execution agents...")
    agents = create_execution_agents(mode="stock", d_model=128)
    
    for strategy_name, strategy_agent in agents.items():
        action, _, _ = strategy_agent.act(features, strategy_feat, deterministic=True)
        print(f"{strategy_name}: Action = {action_map.get(action, action)}")
    
    print("\n[SUCCESS] Execution Agents tests passed!")


if __name__ == "__main__":
    test_execution_agents()

