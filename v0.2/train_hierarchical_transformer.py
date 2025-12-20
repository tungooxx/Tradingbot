"""
Training Pipeline for Hierarchical Transformer Trading System
=============================================================
Implements curriculum learning with 4 stages:
1. Pre-train transformers (supervised, price prediction)
2. Pre-train execution agents (one per strategy)
3. Train meta-strategy agent (with frozen execution agents)
4. End-to-end fine-tuning (all trainable)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import logging
from typing import Dict, Optional, Tuple, List
import os
from datetime import datetime
import json
from collections import defaultdict

# Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

# Import from v0.1
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v0.1'))
# Add v0.2 to path (current directory) so we can import local modules
sys.path.insert(0, os.path.dirname(__file__))
from dapgio_improved import TradingConfig, setup_logging, StockTradingEnv

# Import from v0.2
from data_preprocessor import MultiTimescalePreprocessor
from regime_detector import EnhancedRegimeDetector
from transformer_encoders import MultiScaleTransformerEncoder
from cross_scale_attention import CrossScaleAttention
from meta_strategy_agent import MetaStrategyAgent
from execution_agents import create_execution_agents
from hierarchical_env import MetaStrategyEnv, ExecutionEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==============================================================================
# TRAINING STATISTICS TRACKER
# ==============================================================================

class TrainingStats:
    """Track training statistics: actions, profits, rewards, etc."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.reset()
    
    def reset(self):
        """Reset all statistics"""
        self.total_reward = 0.0
        self.total_profit = 0.0
        self.initial_balance = None
        self.current_balance = None
        self.action_counts = defaultdict(int)  # {action: count}
        self.strategy_counts = defaultdict(int)  # {strategy: count}
        self.trade_count = 0
        self.win_count = 0
        self.loss_count = 0
        self.rewards_history = []
        self.profits_history = []
        self.invalid_action_counts = defaultdict(int)  # Track invalid actions for reward breakdown
        # DEBUG: Reset reward breakdown tracking
        self.reward_breakdowns = []
        self.transaction_costs = []
        self.volatility_penalties = []
        self.stop_loss_penalties = []
        self.whipsaw_penalties = []
        self.pnl_rewards = []
        self.action_rewards = []
        self.holding_rewards = []
        self.stop_loss_rewards = []
        self.take_profit_rewards = []
        # DEBUG: Initialize reward breakdown tracking
        self.reward_breakdowns = []
        self.transaction_costs = []
        self.volatility_penalties = []
        self.stop_loss_penalties = []
        self.whipsaw_penalties = []
        self.pnl_rewards = []
    
    def update(self, reward: float, info: Dict, action: Optional[int] = None, 
               strategy: Optional[str] = None):
        """Update statistics from step"""
        self.total_reward += reward
        self.rewards_history.append(reward)
        
        # REWARD BREAKDOWN TRACKING: Track reward components for debugging
        if not hasattr(self, 'reward_breakdowns'):
            self.reward_breakdowns = []
            self.transaction_costs = []
            self.volatility_penalties = []
            self.stop_loss_penalties = []
            self.whipsaw_penalties = []
            self.pnl_rewards = []
            self.action_rewards = []
            self.holding_rewards = []
            self.stop_loss_rewards = []
            self.take_profit_rewards = []
        
        # Track reward breakdown components if available
        if 'reward_breakdown' in info and isinstance(info['reward_breakdown'], dict):
            breakdown = info['reward_breakdown']
            self.reward_breakdowns.append(breakdown)
            
            # Track detailed components (if available in environment)
            if 'transaction_cost' in breakdown:
                self.transaction_costs.append(breakdown['transaction_cost'])
            if 'transaction_cost_ratio' in breakdown:
                # Convert ratio to absolute if we have portfolio value
                if 'portfolio_value' in info:
                    self.transaction_costs.append(breakdown['transaction_cost_ratio'] * info['portfolio_value'])
            if 'volatility_penalty' in breakdown:
                self.volatility_penalties.append(breakdown['volatility_penalty'])
            if 'stop_loss_penalty' in breakdown:
                self.stop_loss_penalties.append(breakdown['stop_loss_penalty'])
            if 'whipsaw_penalty' in breakdown:
                self.whipsaw_penalties.append(breakdown['whipsaw_penalty'])
            if 'pnl_reward' in breakdown or 'pnl_component' in breakdown:
                pnl = breakdown.get('pnl_reward') or breakdown.get('pnl_component', 0.0)
                self.pnl_rewards.append(pnl)
            
            # Track action-based rewards (what's actually in the current reward_breakdown)
            if 'action_reward' in breakdown:
                self.action_rewards.append(breakdown['action_reward'])
            if 'holding_reward' in breakdown:
                self.holding_rewards.append(breakdown['holding_reward'])
            if 'stop_loss_reward' in breakdown:
                self.stop_loss_rewards.append(breakdown['stop_loss_reward'])
            if 'take_profit_reward' in breakdown:
                self.take_profit_rewards.append(breakdown['take_profit_reward'])
        
        # REWARD BREAKDOWN TRACKING: Track invalid actions and reward components
        if 'invalid_action' in info:
            if not hasattr(self, 'invalid_action_counts'):
                self.invalid_action_counts = defaultdict(int)
            self.invalid_action_counts[info['invalid_action']] += 1
        
        # Track action
        if action is not None:
            action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
            action_name = action_names.get(action, f"ACTION_{action}")
            self.action_counts[action_name] += 1
        
        # Track strategy
        if strategy is not None:
            self.strategy_counts[strategy] += 1
        elif 'strategy' in info:
            self.strategy_counts[info['strategy']] += 1
        
        # Track balance and profit
        if 'balance' in info:
            # CRITICAL FIX: Always use initial_balance from info if available (more reliable)
            # The info dict should have the correct initial_balance from environment reset
            if 'initial_balance' in info:
                # Use initial_balance from info (set by environment reset)
                # This ensures we always have the correct starting balance
                if self.initial_balance is None:
                    self.initial_balance = info['initial_balance']
                elif abs(self.initial_balance - info['initial_balance']) > 0.01:
                    # If initial_balance changed significantly (shouldn't happen, but fix it)
                    # Only warn if difference is > $0.01 to avoid false positives from floating point
                    self.logger.warning(f"Initial balance mismatch: stats={self.initial_balance:.2f}, info={info['initial_balance']:.2f}, using info value")
                    self.initial_balance = info['initial_balance']
            elif self.initial_balance is None:
                # Fallback: use balance from first step if initial_balance not in info
                self.initial_balance = info['balance']
            
            self.current_balance = info['balance']
            if self.initial_balance and self.initial_balance > 0:
                profit = self.current_balance - self.initial_balance
                self.total_profit = profit
                self.profits_history.append(profit)
        
        # Track trades
        # Check for trade flag in info (can be 'trade' or 'trade_executed')
        trade_executed = info.get('trade', False) or info.get('trade_executed', False)
        
        # Also detect trades from actions: BUY (1) or SELL (2) indicate trades
        if action is not None and action in [1, 2]:  # BUY or SELL
            trade_executed = True
        
        if trade_executed:
            self.trade_count += 1
            # Check for profit in info
            if 'profit' in info:
                profit_val = info['profit']
                if isinstance(profit_val, (int, float)) and profit_val > 0:
                    self.win_count += 1
                elif isinstance(profit_val, (int, float)) and profit_val < 0:
                    self.loss_count += 1
            # Also check if we can infer profit from balance change
            elif self.initial_balance and self.current_balance:
                # If balance increased, it's likely a win (simplified)
                if self.current_balance > self.initial_balance:
                    # This is approximate - actual profit should come from info
                    pass  # Don't count as win/loss without explicit profit info
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        win_rate = (self.win_count / self.trade_count * 100) if self.trade_count > 0 else 0.0
        return_pct = ((self.current_balance - self.initial_balance) / self.initial_balance * 100) if self.initial_balance and self.initial_balance > 0 else 0.0
        
        # FIX: Sort action_counts by consistent order (HOLD, BUY, SELL) for consistent display
        # Python dicts maintain insertion order, so order can vary based on which action appears first
        action_order = ['HOLD', 'BUY', 'SELL']
        sorted_action_counts = {action: self.action_counts.get(action, 0) for action in action_order if action in self.action_counts}
        # Add any other actions that might exist (shouldn't happen, but defensive)
        for action, count in self.action_counts.items():
            if action not in sorted_action_counts:
                sorted_action_counts[action] = count
        
        summary = {
            'total_reward': self.total_reward,
            'total_profit': self.total_profit,
            'return_pct': return_pct,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'action_counts': sorted_action_counts,  # Use sorted dict for consistent display
            'strategy_counts': dict(self.strategy_counts),
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'avg_reward': np.mean(self.rewards_history) if self.rewards_history else 0.0,
            'avg_profit': np.mean(self.profits_history) if self.profits_history else 0.0
        }
        
        # REWARD BREAKDOWN: Add invalid action counts if tracked
        if hasattr(self, 'invalid_action_counts'):
            summary['invalid_action_counts'] = dict(self.invalid_action_counts)
        
        return summary
    
    def log_summary(self, prefix: str = ""):
        """Log summary statistics"""
        summary = self.get_summary()
        prefix_str = f"{prefix} - " if prefix else ""
        
        self.logger.info(f"{prefix_str}=== Training Statistics ===")
        self.logger.info(f"{prefix_str}Total Reward: {summary['total_reward']:.2f}")
        self.logger.info(f"{prefix_str}Total Profit: ${summary['total_profit']:.2f}")
        self.logger.info(f"{prefix_str}Return: {summary['return_pct']:.2f}%")
        # CRITICAL FIX: Handle None values and Unicode encoding issue
        initial_bal = summary['initial_balance'] if summary['initial_balance'] is not None else 0.0
        current_bal = summary['current_balance'] if summary['current_balance'] is not None else 0.0
        # Use ASCII arrow instead of Unicode to avoid encoding errors
        self.logger.info(f"{prefix_str}Balance: ${initial_bal:.2f} -> ${current_bal:.2f}")
        self.logger.info(f"{prefix_str}Actions: {summary['action_counts']}")
        if summary['strategy_counts']:
            self.logger.info(f"{prefix_str}Strategies: {summary['strategy_counts']}")
        self.logger.info(f"{prefix_str}Trades: {summary['trade_count']} (Wins: {summary['win_count']}, Losses: {summary['loss_count']}, Win Rate: {summary['win_rate']:.1f}%)")
        self.logger.info(f"{prefix_str}Avg Reward: {summary['avg_reward']:.4f}, Avg Profit: ${summary['avg_profit']:.2f}")
        
        # REWARD BREAKDOWN: Log invalid action counts if available
        if 'invalid_action_counts' in summary and summary['invalid_action_counts']:
            self.logger.info(f"{prefix_str}Invalid Actions: {summary['invalid_action_counts']}")


# ==============================================================================
# STAGE 1: PRE-TRAIN TRANSFORMERS (SUPERVISED)
# ==============================================================================

def prepare_pretraining_data(preprocessor: MultiTimescalePreprocessor, 
                            window_size: int = 30,
                            logger: Optional[logging.Logger] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Prepare data for supervised pre-training (price prediction).
    
    Returns:
        X_dict: Dictionary of input sequences by timescale
        y_dict: Dictionary of target prices by timescale
    """
    aligned_data, features = preprocessor.process(window_size=window_size)
    
    if not aligned_data or not features:
        raise ValueError("Failed to process data")
    
    X_dict = {}
    y_dict = {}
    
    for interval, feat_array in features.items():
        if len(feat_array) < window_size + 1:
            continue
        
        X_list = []
        y_list = []
        
            # Create sequences: predict next Close price
        for i in range(window_size, len(feat_array) - 1):
            window = feat_array[i - window_size:i]  # (window_size, features)
            target = feat_array[i + 1, 0]  # Next Close price (first feature)
            
            # Check for NaN/Inf
            if np.isnan(window).any() or np.isinf(window).any():
                continue
            if np.isnan(target) or np.isinf(target):
                continue
            
            X_list.append(window)
            y_list.append(target)
        
        if X_list:
            X_array = np.array(X_list, dtype=np.float32)
            y_array = np.array(y_list, dtype=np.float32).reshape(-1, 1)
            
            # Final NaN check
            if np.isnan(X_array).any() or np.isnan(y_array).any():
                if logger:
                    logger.warning(f"Skipping {interval} due to NaN in processed data")
                continue
            
            # Normalize input features (per feature dimension)
            # X_array shape: (samples, window_size, features)
            for feat_idx in range(X_array.shape[2]):
                feat_data = X_array[:, :, feat_idx]
                feat_mean = feat_data.mean()
                feat_std = feat_data.std() + 1e-7
                X_array[:, :, feat_idx] = (feat_data - feat_mean) / feat_std
            
            # Normalize targets to prevent large values
            y_mean = y_array.mean()
            y_std = y_array.std() + 1e-7
            y_array = (y_array - y_mean) / y_std
            
            # Clip extreme values
            X_array = np.clip(X_array, -10, 10)
            y_array = np.clip(y_array, -10, 10)
            
            X_dict[interval] = X_array
            y_dict[interval] = y_array
    
    return X_dict, y_dict


def pretrain_transformer_encoders(preprocessor: MultiTimescalePreprocessor,
                                  mode: str = "stock",
                                  epochs: int = 100,
                                  batch_size: int = 256,  # Increased from 64 to 256 (4x) for better GPU utilization
                                  lr: float = 0.0001,
                                  window_size: int = 30,
                                  logger: Optional[logging.Logger] = None,
                                  use_wandb: bool = True) -> MultiScaleTransformerEncoder:
    """
    Pre-train transformer encoders on price prediction task.
    
    Returns:
        Trained transformer encoder
    """
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STAGE 1: Pre-training Transformer Encoders (Supervised)")
    logger.info("=" * 60)
    
    # Prepare data
    logger.info("Preparing pre-training data...")
    X_dict, y_dict = prepare_pretraining_data(preprocessor, window_size, logger)
    
    if not X_dict:
        raise ValueError("No data prepared for pre-training")
    
    logger.info(f"Prepared data for {len(X_dict)} timescales")
    for interval, X in X_dict.items():
        logger.info(f"  {interval}: {len(X)} samples")
    
    # Determine input_dim from actual data (stocks have 8 features, crypto have 5)
    # CRITICAL FIX: X_array has shape (samples, window_size, features)
    # So we need shape[2] for features, not shape[1] (which is window_size)
    if X_dict:
        sample_X = next(iter(X_dict.values()))
        if len(sample_X.shape) == 3:
            # Shape is (samples, window_size, features)
            input_dim = sample_X.shape[2]  # Feature dimension (last axis)
        elif len(sample_X.shape) == 2:
            # Shape is (samples, features) - flattened
            input_dim = sample_X.shape[1]  # Feature dimension
        else:
            # Fallback
            input_dim = 8 if mode == "stock" else 5
        logger.info(f"Detected input_dim={input_dim} from data shape {sample_X.shape} (mode={mode})")
    else:
        # Fallback: 5 for crypto, 8 for stocks (with market hours)
        input_dim = 8 if mode == "stock" else 5
        logger.info(f"Using default input_dim={input_dim} for mode={mode}")
    
    # Create transformer encoder with smaller architecture for stability
    transformer = MultiScaleTransformerEncoder(
        d_model=64,  # Reduced from 128
        nhead=4,  # Reduced from 8
        num_layers=2,  # Reduced from 4
        dim_feedforward=256,  # Reduced from 512
        dropout=0.1,
        input_dim=input_dim,  # Dynamic based on mode
        mode=mode
    ).to(device)
    
    # Verify transformer has required encoders for the mode
    if mode == "stock":
        if not hasattr(transformer, 'encoder_1h'):
            raise RuntimeError("Transformer missing encoder_1h for stock mode. Please check transformer_encoders.py")
        logger.info("Transformer created with 1h, 1d, 1w encoders for stock mode")
    
    # Create prediction heads (one per timescale) - match d_model=64
    prediction_heads = {}
    for interval in X_dict.keys():
        prediction_heads[interval] = nn.Sequential(
            nn.Linear(64, 32),  # Match d_model=64
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        ).to(device)
    
    # Optimizer with lower learning rate to prevent NaN
    all_params = list(transformer.parameters())
    for head in prediction_heads.values():
        all_params.extend(head.parameters())
    
    optimizer = optim.Adam(all_params, lr=lr * 0.01, weight_decay=1e-5)  # Much lower LR + weight decay
    criterion = nn.MSELoss()
    
    # Initialize weights properly - use Kaiming init for better stability
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    transformer.apply(init_weights)
    for head in prediction_heads.values():
        head.apply(init_weights)
    
    # Training loop
    logger.info(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        interval_losses = {}  # Track losses per interval for wandb
        
        # Train on each timescale
        for interval, X in X_dict.items():
            y = y_dict[interval]
            interval_loss = 0.0
            interval_batches = 0
            
            # Create dataloader (FIX: Ensure float32 dtype)
            # OPTIMIZATION: Add num_workers and pin_memory for faster data loading
            # num_workers=4-8 is optimal (too many causes overhead, too few causes CPU bottleneck)
            dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=4,  # Match physical CPU cores (4-8 is sweet spot)
                pin_memory=True,  # Speed up CPU->GPU transfer
                persistent_workers=True  # Keep workers alive between epochs
            )
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                # Check for NaN in input
                if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                    logger.warning(f"Skipping batch with NaN values in {interval}")
                    continue
                
                # Forward pass
                # Map interval key if needed (1wk -> 1w for transformer)
                transformer_key = interval.replace("1wk", "1w") if interval == "1wk" else interval
                features_dict = {transformer_key: batch_X}
                encoded = transformer(features_dict)
                
                # Get last hidden state (use transformer_key, with fallback)
                if transformer_key not in encoded:
                    # Try alternative key (e.g., if transformer returned different key)
                    if interval in encoded:
                        transformer_key = interval
                    elif "1w" in encoded and transformer_key == "1wk":
                        transformer_key = "1w"
                    else:
                        logger.warning(f"Key '{transformer_key}' not found in encoded dict. Available keys: {list(encoded.keys())}")
                        logger.warning(f"Skipping batch for interval {interval}")
                        continue
                
                last_hidden = encoded[transformer_key][:, -1, :]  # (batch, d_model)
                
                # Check for NaN in hidden state
                if torch.isnan(last_hidden).any():
                    logger.warning(f"NaN in hidden state for {interval}, skipping batch")
                    continue
                
                # Predict
                pred = prediction_heads[interval](last_hidden)
                
                # Check for NaN in prediction
                if torch.isnan(pred).any():
                    logger.warning(f"NaN in prediction for {interval}, skipping batch")
                    continue
                
                # Loss
                loss = criterion(pred, batch_y)
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"NaN/Inf loss for {interval}, skipping batch")
                    continue
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN gradients
                has_nan_grad = False
                for param in all_params:
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    logger.warning(f"NaN gradients detected for {interval}, skipping update")
                    optimizer.zero_grad()
                    continue
                
                grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                interval_loss += loss.item()
                num_batches += 1
                interval_batches += 1
            
            # Store interval loss
            if interval_batches > 0:
                interval_losses[interval] = interval_loss / interval_batches
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")
            
            # Log to wandb
            if use_wandb and WANDB_AVAILABLE:
                log_dict = {
                    "stage1/epoch": epoch + 1,
                    "stage1/total_loss": avg_loss,
                }
                # Add per-interval losses
                for interval, loss_val in interval_losses.items():
                    log_dict[f"stage1/loss_{interval}"] = loss_val
                wandb.log(log_dict)
    
    logger.info("Transformer pre-training complete!")
    return transformer


# ==============================================================================
# STAGE 2: PRE-TRAIN EXECUTION AGENTS
# ==============================================================================

def pretrain_execution_agents(transformer: MultiScaleTransformerEncoder,
                              preprocessor: MultiTimescalePreprocessor,
                              regime_detector: EnhancedRegimeDetector,
                              mode: str = "stock",
                              epochs_per_agent: int = 100,
                              steps_per_epoch: int = 500,
                              lr: float = 0.003,  # 3e-3: 10x increase from 3e-4 - Agent needs bigger updates to change its mind
                              window_size: int = 30,
                              logger: Optional[logging.Logger] = None,
                              use_wandb: bool = True) -> Dict[str, nn.Module]:
    """
    Pre-train execution agents (one per strategy).
    
    Returns:
        Dictionary of trained execution agents
    """
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STAGE 2: Pre-training Execution Agents")
    logger.info("=" * 60)
    
    from execution_agents import ExecutionAgent
    
    strategies = ["TREND_FOLLOW", "MEAN_REVERT", "MOMENTUM", "RISK_OFF"]
    agents = {}
    
    for strategy in strategies:
        logger.info(f"\nPre-training {strategy} agent...")
        
        # Determine input_dim from transformer (should match data)
        input_dim = transformer.encoder_1d.input_dim if hasattr(transformer, 'encoder_1d') else (8 if mode == "stock" else 5)
        
        # Create agent with pre-trained transformer (match transformer architecture)
        agent = ExecutionAgent(
            strategy_type=strategy,
            d_model=64,  # Match pre-trained transformer
            nhead=4,  # Match pre-trained transformer
            num_layers=2,  # Match pre-trained transformer
            dim_feedforward=256,  # Match pre-trained transformer
            dropout=0.1,
            input_dim=input_dim,  # Match transformer input_dim
            mode=mode,
            hidden_dim=128  # Reduced to match smaller model
        ).to(device)
        
        # Transfer transformer weights (only matching layers)
        pretrained_state = transformer.state_dict()
        agent_state = agent.transformer_encoder.state_dict()
        
        # Load only matching keys
        matched_state = {}
        for key, value in pretrained_state.items():
            if key in agent_state and agent_state[key].shape == value.shape:
                matched_state[key] = value
        
        agent.transformer_encoder.load_state_dict(matched_state, strict=False)
        logger.info(f"Loaded {len(matched_state)}/{len(pretrained_state)} transformer weights")
        
        # HYBRID INPUT PROTECTION: Freeze transformer weights or use very low learning rate
        # Transformer weights remain frozen - only Actor-Critic 'Hybrid Head' learns
        for param in agent.transformer_encoder.parameters():
            param.requires_grad = False  # Freeze transformer weights
        logger.info("Transformer weights frozen - only Hybrid Head (Actor-Critic) will learn")
        
        # Create execution environment with Alpaca-matching settings
        config = TradingConfig()
        config.mode = mode
        # CRITICAL: Match Alpaca settings for consistent training/live performance
        # Alpaca default: max_positions=5, position_size_pct=0.10 (10%)
        # User can override by setting config.max_positions and config.max_position_size_pct
        if not hasattr(config, 'max_positions') or getattr(config, 'max_positions', None) is None:
            config.max_positions = 5  # Alpaca default: 5 concurrent positions
        if not hasattr(config, 'max_position_size_pct') or getattr(config, 'max_position_size_pct', None) is None:
            config.max_position_size_pct = 0.10  # Alpaca default: 10% per position
        
        # TEMPORARY: Set fees to 0.0% for the next 50 epochs to boost learning
        # This removes transaction cost friction and allows agent to explore more freely
        FEE_FREE_EPOCHS = 50
        original_commission = config.commission_rate
        original_slippage = config.slippage_bps
        config.commission_rate = 0.0  # 0.0% commission
        config.slippage_bps = 0.0  # 0.0% slippage
        logger.info(f"[FEE-FREE MODE] Setting fees to 0.0% for first {FEE_FREE_EPOCHS} epochs")
        logger.info(f"  Original: commission={original_commission:.4f} ({original_commission*100:.2f}%), slippage={original_slippage:.1f} bps")
        logger.info(f"  Temporary: commission=0.0 (0.0%), slippage=0.0 bps")
        
        logger.info(f"Creating ExecutionEnv with Alpaca-matching settings:")
        logger.info(f"  max_positions={config.max_positions}, position_size_pct={config.max_position_size_pct:.1%}")
        
        exec_env = ExecutionEnv(
            ticker=preprocessor.ticker,
            window_size=window_size,
            config=config,
            strategy_type=strategy,
            logger=logger
        )
        
        # Initialize statistics tracker for this strategy
        stats = TrainingStats(logger)
        
        # Optimizer - only train non-frozen parameters (Hybrid Head)
        # Filter out frozen transformer parameters
        trainable_params = [p for p in agent.parameters() if p.requires_grad]
        optimizer = optim.Adam(trainable_params, lr=lr)
        logger.info(f"Optimizer: Training {len(trainable_params)} trainable parameters (transformer frozen)")
        
        # DEBUG: Ensure agent is in training mode
        agent.train()  # Set to training mode
        # Note: Transformer params have requires_grad=False (frozen), which is correct
        
        logger.debug(f"Agent training mode: {agent.training}")
        logger.debug(f"Agent parameters requiring grad: {sum(p.requires_grad for p in agent.parameters())}/{len(list(agent.parameters()))}")
        
        # Fetch data once before training (cache it)
        logger.info("Fetching and preprocessing data once for training...")
        aligned_data, features = preprocessor.process(window_size=window_size)
        if not aligned_data or not features:
            logger.error(f"Failed to fetch data for {strategy} agent")
            continue
        
        # Training loop with PPO-style updates
        memory_states = []
        memory_actions = []
        memory_logprobs = []
        memory_rewards = []
        memory_values = []
        
        # Calculate total steps for entropy annealing
        total_steps = epochs_per_agent * steps_per_epoch
        current_global_step = 0
        
        # Persistent entropy settings and state (do NOT reset every epoch)
        # ENTROPY COEFFICIENT: Reduced for convergence stage
        # Cut in half from ~0.0078 to 0.003 to allow agent to start converging
        ENTROPY_COEF_START = 0.003  # Reduced from 0.01 to allow convergence
        ENTROPY_COEF_MIN = 0.001  # Increased 10x from 0.0001 to maintain minimum exploration
        ENTROPY_COEF_DECAY = 0.99
        ENTROPY_COEF_MAX = ENTROPY_COEF_START * 20.0  # 0.003 * 20 = 0.06
        entropy_coef = ENTROPY_COEF_START
        
        for epoch in range(epochs_per_agent):
            # DEBUG: Check balance BEFORE reset
            if hasattr(exec_env.base_env, 'balance'):
                balance_before_reset = exec_env.base_env.balance
                if epoch % 10 == 0 or epoch < 3:
                    logger.debug(f"  Epoch {epoch}: Balance BEFORE reset() = ${balance_before_reset:.2f}")
            
            state, _ = exec_env.reset()
            episode_reward = 0.0
            
            # DEBUG: Check balance AFTER reset
            if hasattr(exec_env.base_env, 'balance'):
                balance_after_reset = exec_env.base_env.balance
                if epoch % 10 == 0 or epoch < 3:
                    logger.debug(f"  Epoch {epoch}: Balance AFTER reset() = ${balance_after_reset:.2f}")
            
            # CRITICAL FIX: Explicitly reset balance to initial_balance at start of each epoch
            # The environment should reset balance, but we ensure it's always $2000.00
            if hasattr(exec_env.base_env, 'balance') and hasattr(exec_env.base_env, 'initial_balance'):
                expected_initial_balance = exec_env.base_env.initial_balance
                balance_before_fix = exec_env.base_env.balance
                
                # DEBUG: Log balance state before and after reset
                if epoch % 10 == 0 or epoch < 3:  # Log every 10 epochs or first 3
                    logger.info(f"  Epoch {epoch} START: Balance check - Before fix: ${balance_before_fix:.2f}, Expected: ${expected_initial_balance:.2f}")
                
                if abs(exec_env.base_env.balance - expected_initial_balance) > 0.01:
                    logger.warning(f"  Epoch {epoch}: Balance not reset properly! Was ${exec_env.base_env.balance:.2f}, resetting to ${expected_initial_balance:.2f}")
                    logger.warning(f"    DEBUG: Checking where $1800 came from...")
                    logger.warning(f"    DEBUG: exec_env.base_env.balance = ${exec_env.base_env.balance:.2f}")
                    logger.warning(f"    DEBUG: exec_env.base_env.initial_balance = ${exec_env.base_env.initial_balance:.2f}")
                    logger.warning(f"    DEBUG: exec_env.base_env.shares = {exec_env.base_env.shares}")
                    logger.warning(f"    DEBUG: exec_env.base_env.entry_price = ${exec_env.base_env.entry_price:.2f}")
                    
                    # Force correct balance
                    exec_env.base_env.balance = expected_initial_balance
                    exec_env.base_env.shares = 0.0
                    exec_env.base_env.entry_price = 0.0
                    
                    logger.warning(f"    DEBUG: After fix - balance = ${exec_env.base_env.balance:.2f}, shares = {exec_env.base_env.shares}, entry_price = ${exec_env.base_env.entry_price:.2f}")
                
                epoch_start_balance = exec_env.base_env.balance
                if epoch % 10 == 0 or epoch < 3:  # Log every 10 epochs or first 3
                    logger.debug(f"  Epoch {epoch} START: Environment balance = ${epoch_start_balance:.2f} (should be ${expected_initial_balance:.2f})")
            
            # CRITICAL FIX: Reset statistics at start of each epoch
            # This ensures action counts and other stats are per-epoch, not accumulated
            stats.reset()
            
            # Update entropy coef persistently (do NOT reset per epoch)
            # Apply gentle decay each epoch but keep floor; any auto-increase from PPO stays
            entropy_coef = max(entropy_coef * ENTROPY_COEF_DECAY, ENTROPY_COEF_MIN)
            epoch_progress = epoch / epochs_per_agent  # For logging only
            if epoch % 10 == 0 or epoch < 5:
                logger.info(f"  Epoch {epoch}: Entropy Coef (persistent) = {entropy_coef:.6f} (Progress: {epoch_progress:.1%})")
            
            # Track action diversity to detect policy collapse
            action_history = []  # Track last 100 actions to detect collapse
            ACTION_HISTORY_SIZE = 100
            
            for step in range(steps_per_epoch):
                current_global_step += 1
                # Get windowed features from cached data (use current environment index)
                # The environment should track its position in the data
                windowed = preprocessor.get_window_features(features, window_size, None)
                
                if not windowed:
                    logger.warning("No windowed features available, skipping step")
                    continue
                
                # Convert to tensors
                # Map interval keys (1wk -> 1w for transformer)
                features_dict = {}
                for interval, feat in windowed.items():
                    # Check for NaN/Inf in numpy array before converting
                    if np.isnan(feat).any() or np.isinf(feat).any():
                        logger.warning(f"NaN/Inf detected in windowed features for {interval}, replacing with zeros")
                        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    transformer_key = interval.replace("1wk", "1w") if interval == "1wk" else interval
                    # FIX: Ensure float32 dtype to match model
                    feat_tensor = torch.from_numpy(feat).float().unsqueeze(0).to(device)
                    features_dict[transformer_key] = feat_tensor
                
                # Get strategy features
                shares = exec_env.base_env.shares if hasattr(exec_env.base_env, 'shares') else 0.0
                entry_price = exec_env.base_env.entry_price if hasattr(exec_env.base_env, 'entry_price') else 0.0
                
                # Get current price from environment data (index 0 = Close price)
                if hasattr(exec_env.base_env, 'data') and exec_env.base_env.data is not None:
                    current_step = min(exec_env.base_env.current_step, len(exec_env.base_env.data) - 1)
                    current_price = exec_env.base_env.data[current_step, 0]  # Index 0 = Close price
                else:
                    current_price = entry_price if entry_price > 0 else 1.0
                
                strategy_feat = agent._get_strategy_features(shares, entry_price, current_price).unsqueeze(0).to(device)
                
                # PRECISION FEATURES: Vision (Transformer) + Precision (Scaled Price)
                # Calculate precision features: hourly_return, rsi, position_status, unrealized_pnl
                precision_feat = agent._get_precision_features(
                    features_dict, shares, entry_price, current_price
                ).unsqueeze(0).to(device)
                
                # DEBUG: Check for NaN/Inf in inputs before forward pass
                # NOTE: Inputs don't need gradients (they're data), so nan_to_num is OK here
                # But we'll use it for consistency and to avoid issues
                for key, feat_tensor in features_dict.items():
                    if torch.isnan(feat_tensor).any() or torch.isinf(feat_tensor).any():
                        logger.warning(f"NaN/Inf detected in features_dict[{key}], replacing with zeros")
                        # Inputs don't need gradients, but use nan_to_num for safety
                        features_dict[key] = torch.nan_to_num(feat_tensor, nan=0.0, posinf=0.0, neginf=0.0)
                
                if torch.isnan(strategy_feat).any() or torch.isinf(strategy_feat).any():
                    logger.warning("NaN/Inf detected in strategy_feat, replacing with zeros")
                    # Inputs don't need gradients, but use nan_to_num for safety
                    strategy_feat = torch.nan_to_num(strategy_feat, nan=0.0, posinf=0.0, neginf=0.0)
                
                # PRECISION FEATURES: Vision (Transformer) + Precision (Scaled Price)
                # Calculate precision features: hourly_return, rsi, position_status, unrealized_pnl
                precision_feat = agent._get_precision_features(
                    features_dict, shares, entry_price, current_price
                ).unsqueeze(0).to(device)
                
                if torch.isnan(precision_feat).any() or torch.isinf(precision_feat).any():
                    logger.warning("NaN/Inf detected in precision_feat, replacing with zeros")
                    precision_feat = torch.nan_to_num(precision_feat, nan=0.0, posinf=0.0, neginf=0.0)
                
                # HYBRID INPUT: Local Features from Environment
                # Get local features from environment (position_status, unrealized_pnl, time_in_trade, relative_price)
                # CRITICAL: This is used for action masking - must be correct!
                if hasattr(exec_env.base_env, '_get_local_features'):
                    local_feat_np = exec_env.base_env._get_local_features()
                    local_feat = torch.from_numpy(local_feat_np).unsqueeze(0).float().to(device)
                    
                    # DEBUG: Verify position_status matches actual shares
                    position_status_from_feat = local_feat[0, 0].item() if local_feat.numel() > 0 else 0.0
                    actual_shares = exec_env.base_env.shares if hasattr(exec_env.base_env, 'shares') else 0.0
                    expected_pos_status = 1.0 if actual_shares > 0 else 0.0
                    
                    # CRITICAL: Log mismatch if position_status is wrong
                    if abs(position_status_from_feat - expected_pos_status) > 0.1:
                        if step % 50 == 0:  # Log every 50 steps to avoid spam
                            logger.warning(f"  Step {step}: [ACTION MASKING BUG] position_status mismatch!")
                            logger.warning(f"    Expected: {expected_pos_status:.1f} (shares={actual_shares:.4f})")
                            logger.warning(f"    Got from local_feat: {position_status_from_feat:.1f}")
                            logger.warning(f"    This will cause incorrect action masking!")
                            # FIX: Overwrite position_status with correct value
                            local_feat[0, 0] = expected_pos_status
                            logger.warning(f"    FIXED: Set position_status to {expected_pos_status:.1f}")
                else:
                    # Fallback: create default local features
                    # CRITICAL: If _get_local_features doesn't exist, we need to calculate it manually
                    actual_shares = exec_env.base_env.shares if hasattr(exec_env.base_env, 'shares') else 0.0
                    position_status = 1.0 if actual_shares > 0 else 0.0
                    logger.warning(f"  Step {step}: [WARNING] _get_local_features() not found! Using fallback with position_status={position_status:.1f}")
                    local_feat = torch.zeros(1, 4, device=device, dtype=torch.float32)
                    local_feat[0, 0] = position_status  # Set correct position_status
                
                if torch.isnan(local_feat).any() or torch.isinf(local_feat).any():
                    logger.warning("NaN/Inf detected in local_feat, replacing with zeros")
                    local_feat = torch.nan_to_num(local_feat, nan=0.0, posinf=0.0, neginf=0.0)
                
                # DEBUG: Get action and value for PPO
                # CRITICAL: We need to store the ORIGINAL features_dict and strategy_feat WITHOUT detaching
                # so that when we recompute during PPO, gradients can flow through model parameters
                # The inputs themselves don't need gradients, but we must not detach them before storing
                try:
                    # DEBUG: Ensure agent is in training mode
                    agent.train()
                    
                    # Forward pass WITH gradients (needed for PPO updates)
                    # Inputs don't need gradients (they're data), but model parameters do
                    logits, value_tensor = agent(features_dict, strategy_feat, precision_feat, local_feat)
                    
                    # DEBUG: Print logits every 100 steps to check if agent is making decisions
                    if step % 100 == 0:
                        logits_np = logits[0].detach().cpu().numpy()
                        logger.info(f"  Step {step}: DEBUG LOGITS: {logits_np} (Strategy: {strategy})")
                        # Check if logits are all identical (dead network)
                        if len(logits_np) > 1 and np.allclose(logits_np, logits_np[0], atol=1e-6):
                            logger.warning(f"  [WARNING] All logits are identical! Network may be dead. Logits: {logits_np}")
                    
                    # DEBUG: Check for NaN/Inf in outputs
                    # CRITICAL: Don't use torch.nan_to_num() or zeros_like() here - they break gradients!
                    # The model's forward() already handles NaN with gradient-preserving replacement
                    # If we still get NaN here, it means the model itself has issues
                    # ============================================================
                    # LOGIT PARAMETERS (CRITICAL FIX for Policy Collapse)
                    # ============================================================
                    # CRITICAL FIX: Extreme logits cause softmax collapse to single action
                    # Tighter clamping + temperature scaling to prevent deterministic behavior
                    # ============================================================
                    LOGIT_CLAMP_MIN = -5.0   # Tighter: Prevent extreme values
                    LOGIT_CLAMP_MAX = +5.0   # Tighter: Keep probabilities reasonable
                    LOGIT_TEMPERATURE = 2.5  # Temperature scaling: Increased from 1.5 to 2.5 for stronger exploration
                    
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.warning("NaN/Inf in logits after model forward - model may have issues")
                        # Try to maintain gradient connection if possible
                        logits = logits + 0.0 * torch.zeros_like(logits)  # Maintain connection
                        logits = torch.clamp(logits, min=LOGIT_CLAMP_MIN, max=LOGIT_CLAMP_MAX)
                    
                    if torch.isnan(value_tensor).any() or torch.isinf(value_tensor).any():
                        logger.warning("NaN/Inf in value_tensor after model forward - model may have issues")
                        # Maintain gradient connection
                        value_tensor = value_tensor + 0.0 * torch.zeros_like(value_tensor)
                    
                    # CRITICAL FIX: Clamp FIRST, then apply adaptive temperature scaling
                    # This prevents extreme logits from causing policy collapse
                    logits = torch.clamp(logits, min=LOGIT_CLAMP_MIN, max=LOGIT_CLAMP_MAX)
                    
                    # Adaptive temperature scaling to reach a target entropy for 3 actions
                    LOGIT_TEMPERATURE_BASE = 2.5  # starting temperature
                    TARGET_ACTION_ENTROPY = 0.5   # ~half of max (~1.0986) for 3-way action
                    TEMP_MAX = 10.0               # upper bound to avoid over-flattening
                    TEMP_GROWTH = 1.5             # multiplicative growth per adjustment
                    temp = LOGIT_TEMPERATURE_BASE
                    
                    used_logits = logits / temp
                    dist = torch.distributions.Categorical(logits=used_logits)
                    entropy_val_temp = dist.entropy().item()
                    adjust_count = 0
                    while entropy_val_temp < TARGET_ACTION_ENTROPY and temp < TEMP_MAX:
                        temp = min(temp * TEMP_GROWTH, TEMP_MAX)
                        used_logits = logits / temp
                        dist = torch.distributions.Categorical(logits=used_logits)
                        entropy_val_temp = dist.entropy().item()
                        adjust_count += 1
                    if adjust_count > 0:
                        logger.debug(f"Adaptive temperature: increased to {temp:.2f} to reach entropy {entropy_val_temp:.3f}")
                    
                    # Create distribution (with final temperature) and sample action
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    # CRITICAL FIX: Log action probabilities MORE FREQUENTLY to detect collapse early
                    # Log every 50 steps (not 100) and every epoch (not 10) to catch collapse quickly
                    if step % 50 == 0:  # More frequent logging
                        probs = F.softmax(used_logits, dim=-1)
                        entropy_val = dist.entropy().item()
                        logger.info(f"  Step {step}: Action probs = HOLD:{probs[0,0]:.3f}, BUY:{probs[0,1]:.3f}, SELL:{probs[0,2]:.3f}, Entropy={entropy_val:.4f}")
                        
                        # ============================================================
                        # POLICY COLLAPSE DETECTION (CRITICAL - Detect Early!)
                        # ============================================================
                        max_prob = probs.max(dim=-1).values.mean().item()
                        if max_prob > 0.90:  # Lower threshold: Detect collapse earlier (was 0.95)
                            logger.warning(f"[WARNING] POLICY COLLAPSE DETECTED: max_action_prob = {max_prob:.2%}. Agent is deterministic!")
                            logger.warning(f"   Action distribution: HOLD={probs[0,0]:.1%}, BUY={probs[0,1]:.1%}, SELL={probs[0,2]:.1%}")
                            logger.warning(f"   Entropy={entropy_val:.4f} (should be >0.5 for 3 actions)")
                            
                            # CRITICAL FIX: Reset entropy coefficient if collapse detected
                            # This forces agent to explore again
                            if entropy_val < 0.3:  # Very low entropy = collapse
                                old_entropy_coef = entropy_coef
                                entropy_coef = min(entropy_coef * 2.0, ENTROPY_COEF_MAX)  # Double it, capped by max
                                logger.warning(f"   RESETTING entropy_coef: {old_entropy_coef:.6f} -> {entropy_coef:.6f} to force exploration (cap={ENTROPY_COEF_MAX:.6f})")
                    
                    # DEBUG: Detach values for storage (old policy values)
                    # These are used for computing advantages and ratios, but don't need gradients
                    value_tensor_detached = value_tensor.detach()
                    log_prob_detached = log_prob.detach()
                    
                except Exception as e:
                    logger.error(f"Error in forward pass: {e}, using random action")
                    # Fallback to random action (detached, no gradients needed)
                    action = torch.tensor([exec_env.action_space.sample()], device=device)
                    log_prob_detached = torch.tensor([0.0], device=device)
                    value_tensor_detached = torch.tensor([0.0], device=device)
                    log_prob = log_prob_detached
                    value_tensor = value_tensor_detached
                
                # DEBUG: Convert action to Python int for environment step
                action_item = action.item() if action.dim() == 0 else action[0].item()
                
                # CRITICAL FIX: Track action diversity to detect policy collapse
                action_history.append(action_item)
                if len(action_history) > ACTION_HISTORY_SIZE:
                    action_history.pop(0)  # Keep only last 100 actions
                
                # Check action diversity every 50 steps
                if step % 50 == 0 and len(action_history) >= 50:
                    action_counts = {0: 0, 1: 0, 2: 0}  # HOLD, BUY, SELL
                    for a in action_history:
                        action_counts[a] = action_counts.get(a, 0) + 1
                    
                    total_actions = len(action_history)
                    action_proportions = {k: v/total_actions for k, v in action_counts.items()}
                    max_proportion = max(action_proportions.values())
                    
                    # CRITICAL: Warn if one action dominates (>80% of recent actions)
                    if max_proportion > 0.80:
                        dominant_action = max(action_proportions, key=action_proportions.get)
                        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
                        logger.warning(f"[WARNING] ACTION COLLAPSE DETECTED: {action_names[dominant_action]} is {max_proportion:.1%} of last {len(action_history)} actions!")
                        logger.warning(f"   Action distribution: HOLD={action_proportions[0]:.1%}, BUY={action_proportions[1]:.1%}, SELL={action_proportions[2]:.1%}")
                        logger.warning(f"   This indicates policy collapse - agent is not exploring!")
                
                # Step environment (this doesn't need gradients)
                next_state, reward, done, truncated, info = exec_env.step(action_item)
                
                # DEBUG: Log reward breakdown for analysis (every 10 steps or when reward/profit mismatch)
                if step % 10 == 0 or (step == 1 and epoch % 10 == 0):
                    # Get reward breakdown from info if available
                    reward_breakdown = info.get('reward_breakdown', {})
                    if isinstance(reward_breakdown, dict):
                        log_return = reward_breakdown.get('log_return', 'N/A')
                        transaction_cost_ratio = reward_breakdown.get('transaction_cost_ratio', 'N/A')
                        raw_reward = reward_breakdown.get('raw_reward', 'N/A')
                        multiplier_applied = reward_breakdown.get('multiplier_applied', 'N/A')
                        scaled_reward = reward_breakdown.get('scaled_reward', 'N/A')
                        volatility_penalty = reward_breakdown.get('volatility_penalty', 'N/A')
                        exit_reason = reward_breakdown.get('exit_reason', 'N/A')
                        bars_held = reward_breakdown.get('bars_held', 'N/A')
                        
                        logger.info(f"  Step {step}: REWARD BREAKDOWN:")
                        # Format numeric values, keep strings as-is
                        log_return_str = f"{log_return:.6f}" if isinstance(log_return, (int, float)) else str(log_return)
                        transaction_cost_str = f"{transaction_cost_ratio:.6f}" if isinstance(transaction_cost_ratio, (int, float)) else str(transaction_cost_ratio)
                        raw_reward_str = f"{raw_reward:.4f}" if isinstance(raw_reward, (int, float)) else str(raw_reward)
                        scaled_reward_str = f"{scaled_reward:.4f}" if isinstance(scaled_reward, (int, float)) else str(scaled_reward)
                        volatility_penalty_str = f"{volatility_penalty:.4f}" if isinstance(volatility_penalty, (int, float)) else str(volatility_penalty)
                        
                        logger.info(f"    log_return={log_return_str}, transaction_cost_ratio={transaction_cost_str}")
                        logger.info(f"    raw_reward={raw_reward_str}, multiplier={multiplier_applied}, scaled={scaled_reward_str}")
                        logger.info(f"    volatility_penalty={volatility_penalty_str}, exit_reason={exit_reason}, bars_held={bars_held}")
                        logger.info(f"    final_reward={reward:.4f}")
                    else:
                        logger.warning(f"  Step {step}: No reward_breakdown in info dict! Keys: {list(info.keys())}")
                
                # CRITICAL FIX: Match environment reward clipping
                # Environment clips to [-500, +500] (after REWARD_SCALE=10.0 update and widened range)
                # Training should match this to ensure consistency
                reward = np.clip(reward, -500.0, 500.0)
                
                # CRITICAL FIX: Use EXECUTED action, not intended action
                # The environment may convert invalid actions to HOLD (e.g., SELL when shares==0)
                # Stop loss/take profit may also trigger automatic SELLs
                # We need to count the ACTUAL executed action, not the intended one
                executed_action = action_item  # Default to intended action
                
                # Try to get executed action from info dict
                if 'executed_action' in info:
                    executed_action_str = info['executed_action']
                    # Convert string to int if needed
                    if isinstance(executed_action_str, str):
                        action_map = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
                        executed_action = action_map.get(executed_action_str.upper(), action_item)
                    elif isinstance(executed_action_str, int):
                        executed_action = executed_action_str
                elif 'reward_breakdown' in info and isinstance(info['reward_breakdown'], dict):
                    # Try to get from reward_breakdown
                    action_type = info['reward_breakdown'].get('action_type', None)
                    if action_type:
                        action_map = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
                        executed_action = action_map.get(action_type.upper(), action_item)
                
                # DEBUG: Log mismatch between intended and executed action
                if executed_action != action_item and step % 100 == 0:
                    action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
                    logger.warning(f"  Step {step}: Action mismatch! Intended: {action_names[action_item]}, Executed: {action_names[executed_action]}")
                    if executed_action == 0 and action_item != 0:
                        logger.warning(f"    Invalid action converted to HOLD (action masking should prevent this)")
                
                # CRITICAL FIX: Always ensure balance and initial_balance are in info dict
                # This ensures stats tracking has correct values, especially at start of epoch
                if hasattr(exec_env.base_env, 'balance'):
                    info['balance'] = exec_env.base_env.balance
                if hasattr(exec_env.base_env, 'initial_balance'):
                    info['initial_balance'] = exec_env.base_env.initial_balance
                
                # DEBUG: Verify net worth (portfolio value) is correct at start of epoch (first step only)
                # CRITICAL FIX: Compare net_worth (cash + stock value) to initial_balance, not just cash
                # When agent buys stock, cash decreases but net worth stays the same
                # This prevents false alarms when agent correctly converts cash to stock
                if step == 1 and hasattr(exec_env.base_env, 'balance') and hasattr(exec_env.base_env, 'initial_balance'):
                    # Calculate net worth (portfolio value = cash + stock value)
                    if hasattr(exec_env.base_env, '_get_portfolio_value') and hasattr(exec_env.base_env, 'data'):
                        current_step = min(exec_env.base_env.current_step, len(exec_env.base_env.data) - 1)
                        current_price = exec_env.base_env.data[current_step, 0]  # Index 0 = Close price
                        net_worth = exec_env.base_env._get_portfolio_value(current_price)
                    else:
                        # Fallback: calculate manually
                        current_price = exec_env.base_env.entry_price if exec_env.base_env.entry_price > 0 else 1.0
                        if hasattr(exec_env.base_env, 'data') and exec_env.base_env.data is not None:
                            current_step = min(exec_env.base_env.current_step, len(exec_env.base_env.data) - 1)
                            current_price = exec_env.base_env.data[current_step, 0]
                        net_worth = exec_env.base_env.balance + (exec_env.base_env.shares * current_price)
                    
                    # Allow small difference for transaction costs (e.g., commission on initial buy)
                    # Transaction costs are typically <0.5% of position, so allow up to $10 difference on $2000
                    tolerance = max(10.0, exec_env.base_env.initial_balance * 0.005)  # 0.5% or $10, whichever is larger
                    
                    if abs(net_worth - exec_env.base_env.initial_balance) > tolerance:
                        logger.warning(f"  Epoch {epoch}, Step {step}: Net worth mismatch at start! Net Worth=${net_worth:.2f}, Initial=${exec_env.base_env.initial_balance:.2f}")
                        logger.warning(f"    DEBUG: This suggests environment reset() didn't work properly or portfolio value was modified")
                        logger.warning(f"    DEBUG: Net Worth (cash + stock) = ${net_worth:.2f}")
                        logger.warning(f"    DEBUG: Cash (balance) = ${exec_env.base_env.balance:.2f}")
                        logger.warning(f"    DEBUG: Stock value = ${exec_env.base_env.shares * current_price:.2f} (shares={exec_env.base_env.shares:.4f}, price=${current_price:.2f})")
                        logger.warning(f"    DEBUG: Initial balance = ${exec_env.base_env.initial_balance:.2f}")
                        logger.warning(f"    DEBUG: Difference = ${abs(net_worth - exec_env.base_env.initial_balance):.2f} (tolerance=${tolerance:.2f})")
                        
                        # Only force reset if net worth is significantly different (not just cash converted to stock)
                        if abs(net_worth - exec_env.base_env.initial_balance) > exec_env.base_env.initial_balance * 0.01:  # >1% difference
                            logger.warning(f"    DEBUG: Large net worth difference detected, forcing reset...")
                            exec_env.base_env.balance = exec_env.base_env.initial_balance
                            exec_env.base_env.shares = 0.0
                            exec_env.base_env.entry_price = 0.0
                            info['balance'] = exec_env.base_env.balance
                            info['initial_balance'] = exec_env.base_env.initial_balance
                            logger.warning(f"    DEBUG: After force fix - balance = ${exec_env.base_env.balance:.2f}, shares = {exec_env.base_env.shares}")
                    else:
                        # Net worth is correct (within tolerance) - this is normal when agent buys stock
                        if exec_env.base_env.shares > 0:
                            logger.debug(f"  Epoch {epoch}, Step {step}: Net worth OK (${net_worth:.2f} ≈ ${exec_env.base_env.initial_balance:.2f}). Agent has position: {exec_env.base_env.shares:.4f} shares @ ${current_price:.2f}")
                
                stats.update(reward, info, action=executed_action, strategy=strategy)
                
                # DEBUG: Store in memory for PPO update
                # CRITICAL: Store ORIGINAL features_dict and strategy_feat WITHOUT detaching
                # They don't need gradients themselves, but we need the original tensors
                # so that when we recompute, the model can create gradients through its parameters
                # Only detach the policy outputs (log_prob, value), not the inputs!
                memory_states.append((features_dict, strategy_feat, precision_feat, local_feat))
                
                # CRITICAL FIX: Store EXECUTED action, not intended action
                # The environment may convert invalid actions (e.g., SELL when no position → HOLD)
                # PPO must learn from what ACTUALLY happened, not what the agent intended
                # Convert executed_action (int) to tensor for storage
                executed_action_tensor = torch.tensor(executed_action, dtype=torch.long, device=device)
                if executed_action_tensor.dim() == 0:
                    executed_action_tensor = executed_action_tensor.unsqueeze(0)
                memory_actions.append(executed_action_tensor.detach())
                
                # NOTE: We keep the original log_prob from the intended action
                # This is an approximation - ideally we'd recompute log_prob(executed_action)
                # But that requires another forward pass. For now, this is acceptable because:
                # 1. Action masking should prevent most invalid actions
                # 2. When action conversion happens, it's usually to HOLD (action 0)
                # 3. The importance sampling in PPO will partially account for this
                # Store detached log_prob (old policy)
                log_prob_to_store = log_prob_detached if log_prob_detached.dim() > 0 else log_prob_detached.unsqueeze(0)
                memory_logprobs.append(log_prob_to_store)
                memory_rewards.append(reward)
                # Store detached value (old policy)
                value_to_store = value_tensor_detached.squeeze()
                if value_to_store.dim() == 0:
                    value_to_store = value_to_store.unsqueeze(0)
                memory_values.append(value_to_store)
                
                episode_reward += reward
                state = next_state
                
                if done or truncated:
                    state, _ = exec_env.reset()
                    # CRITICAL FIX: Explicitly reset balance when episode ends
                    # This ensures balance is always reset to initial_balance between episodes
                    if hasattr(exec_env.base_env, 'balance') and hasattr(exec_env.base_env, 'initial_balance'):
                        expected_initial_balance = exec_env.base_env.initial_balance
                        if exec_env.base_env.balance != expected_initial_balance:
                            exec_env.base_env.balance = expected_initial_balance
                            exec_env.base_env.shares = 0.0
                            exec_env.base_env.entry_price = 0.0
                    # Reset stats for new episode
                    stats.reset()
                
                # DEBUG: PPO update with larger batch size for better GPU utilization
                # Increased from 100 to 400 (4x) to match increased batch_size in Stage 1
                PPO_BATCH_SIZE_STAGE2 = 400
                if len(memory_rewards) >= PPO_BATCH_SIZE_STAGE2:
                    logger.debug(f"PPO update triggered at step {step}, memory size: {len(memory_rewards)} (batch_size={PPO_BATCH_SIZE_STAGE2})")
                    
                    # DEBUG: Calculate discounted rewards (returns)
                    rewards = []
                    discounted_reward = 0
                    gamma = 0.99
                    for r in reversed(memory_rewards):
                        discounted_reward = r + gamma * discounted_reward
                        rewards.insert(0, discounted_reward)
                    
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                    # ============================================================
                    # REWARD CLIPPING (CRITICAL FIX - Match Environment)
                    # ============================================================
                    # Match the reward clip range in the environment [-500, +500]
                    # Environment uses REWARD_SCALE=10.0 and clips to [-500, +500] to handle -50% crashes
                    REWARD_CLIP_MIN = -500.0
                    REWARD_CLIP_MAX = +500.0
                    rewards = torch.clamp(rewards, min=REWARD_CLIP_MIN, max=REWARD_CLIP_MAX)
                    
                    # Log reward statistics for debugging
                    if step % 500 == 0:
                        reward_std = rewards.std().item()
                        logger.info(f"Reward Stats (step {step}): mean={rewards.mean().item():.3f}, std={reward_std:.3f}, min={rewards.min().item():.3f}, max={rewards.max().item():.3f}")
                        
                        # ============================================================
                        # REWARD SPARSITY DETECTION (from Principal ML Engineer Audit)
                        # ============================================================
                        if reward_std < 1.0:
                            logger.warning(f"⚠️ REWARD SPARSITY: std={reward_std:.2f}. All rewards look the same to agent!")
                    
                    # DEBUG: Stack tensors from memory (these are detached, old policy values)
                    # Ensure all are 1D tensors for stacking and properly detached
                    # Convert to tensors if needed and ensure they're on the right device
                    memory_actions_tensors = [a if isinstance(a, torch.Tensor) else torch.tensor(a, device=device, dtype=torch.long) for a in memory_actions]
                    memory_logprobs_tensors = [lp if isinstance(lp, torch.Tensor) else torch.tensor(lp, device=device, dtype=torch.float32) for lp in memory_logprobs]
                    memory_values_tensors = [v if isinstance(v, torch.Tensor) else torch.tensor(v, device=device, dtype=torch.float32) for v in memory_values]
                    
                    # Stack and ensure proper shape
                    old_actions = torch.stack(memory_actions_tensors).to(device).squeeze().detach()
                    old_logprobs = torch.stack(memory_logprobs_tensors).to(device).squeeze().detach()
                    old_values = torch.stack(memory_values_tensors).to(device).squeeze().detach()
                    
                    # Ensure all have same shape (1D) and are detached
                    if old_actions.dim() == 0:
                        old_actions = old_actions.unsqueeze(0)
                    if old_logprobs.dim() == 0:
                        old_logprobs = old_logprobs.unsqueeze(0)
                    if old_values.dim() == 0:
                        old_values = old_values.unsqueeze(0)
                    
                    # Ensure they're properly detached and don't require gradients
                    old_actions = old_actions.detach().requires_grad_(False)
                    old_logprobs = old_logprobs.detach().requires_grad_(False)
                    old_values = old_values.detach().requires_grad_(False)
                    
                    logger.debug(f"Stacked tensors - actions: {old_actions.shape}, logprobs: {old_logprobs.shape}, values: {old_values.shape}")
                    logger.debug(f"Old tensors requires_grad - actions: {old_actions.requires_grad}, logprobs: {old_logprobs.requires_grad}, values: {old_values.requires_grad}")
                    
                    # DEBUG: PPO optimization loop (multiple epochs for better updates)
                    for ppo_epoch in range(3):
                        optimizer.zero_grad()
                        
                        # DEBUG: Re-evaluate with current policy (WITH gradients for new policy)
                        # This computes new log_probs and values with gradients enabled
                        logprobs_list = []
                        values_list = []
                        entropies_list = []
                        
                        for idx, (state_tuple, act) in enumerate(zip(memory_states, old_actions)):
                            try:
                                # Unpack state: (features_dict, strategy_feat, precision_feat, local_feat)
                                feat_dict = state_tuple[0]
                                strat_feat = state_tuple[1]
                                prec_feat = state_tuple[2] if len(state_tuple) > 2 else None
                                local_feat = state_tuple[3] if len(state_tuple) > 3 else None
                                
                                # DEBUG: Use stored features - they're already tensors created from numpy
                                # CRITICAL: Inputs don't need gradients (they're data), but we must NOT detach them
                                # before passing to model. The model will create gradients through its parameters.
                                # Just ensure they're on the right device
                                feat_dict_ready = {}
                                for key, val in feat_dict.items():
                                    if isinstance(val, torch.Tensor):
                                        # Use original tensor, ensure it's on device
                                        # Don't detach! Just move to device if needed
                                        if val.device != device:
                                            feat_dict_ready[key] = val.to(device)
                                        else:
                                            feat_dict_ready[key] = val
                                        # Ensure it doesn't require grad (it's input data)
                                        feat_dict_ready[key] = feat_dict_ready[key].requires_grad_(False)
                                    else:
                                        # Convert to tensor if not already
                                        feat_dict_ready[key] = torch.tensor(val, device=device, requires_grad=False)
                                
                                if isinstance(strat_feat, torch.Tensor):
                                    if strat_feat.device != device:
                                        strat_feat_ready = strat_feat.to(device)
                                    else:
                                        strat_feat_ready = strat_feat
                                    strat_feat_ready = strat_feat_ready.requires_grad_(False)
                                else:
                                    strat_feat_ready = torch.tensor(strat_feat, device=device, requires_grad=False)
                                
                                # Prepare precision features
                                if prec_feat is not None and isinstance(prec_feat, torch.Tensor):
                                    if prec_feat.device != device:
                                        prec_feat_ready = prec_feat.to(device)
                                    else:
                                        prec_feat_ready = prec_feat
                                    prec_feat_ready = prec_feat_ready.requires_grad_(False)
                                else:
                                    prec_feat_ready = torch.zeros(1, 4, device=device, dtype=torch.float32)
                                
                                # Prepare local features (HYBRID INPUT)
                                if local_feat is not None and isinstance(local_feat, torch.Tensor):
                                    if local_feat.device != device:
                                        local_feat_ready = local_feat.to(device)
                                    else:
                                        local_feat_ready = local_feat
                                    local_feat_ready = local_feat_ready.requires_grad_(False)
                                else:
                                    local_feat_ready = torch.zeros(1, 4, device=device, dtype=torch.float32)
                                
                                # Forward pass WITH gradients (for new policy)
                                # Model parameters have gradients, inputs don't (which is correct)
                                # DEBUG: Ensure agent is in training mode
                                agent.train()
                                
                                # DEBUG: Verify model parameters require gradients BEFORE forward pass
                                params_require_grad = any(p.requires_grad for p in agent.parameters())
                                if not params_require_grad:
                                    logger.error(f"CRITICAL: Model parameters don't require gradients at idx {idx}!")
                                    # Force enable gradients on all parameters
                                    for p in agent.parameters():
                                        p.requires_grad = True
                                    logger.warning(f"Force-enabled gradients on all parameters")
                                
                                # Forward pass - this should create outputs connected to model parameters
                                logits, value = agent(feat_dict_ready, strat_feat_ready, prec_feat_ready, local_feat_ready)
                                
                                # DEBUG: Verify outputs are connected to computation graph
                                # CRITICAL: Outputs MUST have grad_fn to enable backpropagation
                                # Check grad_fn (more reliable than requires_grad for intermediate outputs)
                                logits_has_grad_fn = logits.grad_fn is not None
                                value_has_grad_fn = value.grad_fn is not None
                                
                                if not logits_has_grad_fn:
                                    logger.error(f"CRITICAL: logits at idx {idx} has no grad_fn - computation graph broken!")
                                    logger.error(f"  This means gradients cannot flow back to model parameters")
                                    logger.error(f"  Check if model forward() uses torch.nan_to_num() or other detached operations")
                                    # Try to diagnose: check if inputs have issues
                                    for k, v in feat_dict_ready.items():
                                        logger.error(f"  Input {k}: shape={v.shape}, device={v.device}, requires_grad={v.requires_grad}")
                                
                                if not value_has_grad_fn:
                                    logger.error(f"CRITICAL: value at idx {idx} has no grad_fn - computation graph broken!")
                                
                                # If both outputs lack grad_fn, this is a critical error
                                if not logits_has_grad_fn and not value_has_grad_fn:
                                    raise RuntimeError(
                                        f"Computation graph broken at idx {idx}: "
                                        f"Neither logits nor value have grad_fn. "
                                        f"Model outputs are not connected to parameters. "
                                        f"Check model forward() for detached operations (torch.nan_to_num, etc.)"
                                    )
                                
                                # DEBUG: Check for NaN/Inf
                                # If NaN/Inf found, replace but maintain computation graph connection
                                if torch.isnan(logits).any() or torch.isinf(logits).any():
                                    logger.warning(f"NaN/Inf in logits at idx {idx}, using uniform")
                                    # Create uniform logits but maintain gradient connection through model
                                    # Use a small operation to keep graph intact
                                    uniform_logits = torch.zeros_like(logits)
                                    # Maintain connection: uniform = 0 + 0*logits (keeps graph if logits had grad_fn)
                                    if logits.grad_fn is not None:
                                        uniform_logits = uniform_logits + 0.0 * logits  # Maintain gradient connection
                                    logits = uniform_logits
                                
                                if torch.isnan(value).any() or torch.isinf(value).any():
                                    logger.warning(f"NaN/Inf in value at idx {idx}, using zero")
                                    # Create zero value but maintain gradient connection
                                    zero_value = torch.zeros_like(value)
                                    if value.grad_fn is not None:
                                        zero_value = zero_value + 0.0 * value  # Maintain gradient connection
                                    value = zero_value
                                
                                # Clamp logits then apply adaptive temperature scaling to prevent collapse
                                LOGIT_CLAMP_MIN = -5.0
                                LOGIT_CLAMP_MAX = 5.0
                                logits = torch.clamp(logits, min=LOGIT_CLAMP_MIN, max=LOGIT_CLAMP_MAX)
                                
                                LOGIT_TEMPERATURE_BASE = 2.5
                                TARGET_ACTION_ENTROPY = 0.5
                                TEMP_MAX = 10.0
                                TEMP_GROWTH = 1.5
                                temp = LOGIT_TEMPERATURE_BASE
                                
                                used_logits = logits / temp
                                dist = torch.distributions.Categorical(logits=used_logits)
                                ev = dist.entropy()
                                entropy_val_temp = ev.mean().item() if ev.dim() > 0 else ev.item()
                                adjust_count = 0
                                while entropy_val_temp < TARGET_ACTION_ENTROPY and temp < TEMP_MAX:
                                    temp = min(temp * TEMP_GROWTH, TEMP_MAX)
                                    used_logits = logits / temp
                                    dist = torch.distributions.Categorical(logits=used_logits)
                                    ev = dist.entropy()
                                    entropy_val_temp = ev.mean().item() if ev.dim() > 0 else ev.item()
                                    adjust_count += 1
                                if adjust_count > 0:
                                    logger.debug(f"Adaptive temperature (re-eval): increased to {temp:.2f} to reach entropy {entropy_val_temp:.3f}")
                                
                                # Create distribution and compute new policy log_prob with final temperature
                                # act is from old_actions (detached), but log_prob is from new policy (with gradients)
                                # Ensure act is a proper tensor
                                act_tensor = act if isinstance(act, torch.Tensor) else torch.tensor(act, device=device, dtype=torch.long)
                                log_prob = dist.log_prob(act_tensor)
                                value_squeezed = value.squeeze()
                                entropy = dist.entropy()
                                
                                # DEBUG: These should be part of computation graph even if requires_grad is False
                                # The key is that they're computed from model outputs, which depend on model parameters
                                # When used in loss.backward(), gradients will flow through model parameters
                                # We don't need to check requires_grad on intermediate outputs
                                
                                logprobs_list.append(log_prob)
                                values_list.append(value_squeezed)
                                entropies_list.append(entropy)
                            except Exception as e:
                                logger.error(f"Error in re-evaluation at idx {idx}: {e}, skipping this sample")
                                # Skip this sample instead of using old values (which would break gradients)
                                # We'll handle this by checking list lengths before stacking
                                continue
                        
                        # DEBUG: Check if we have valid samples (some might have been skipped)
                        if len(logprobs_list) == 0:
                            logger.warning(f"No valid samples for PPO update at epoch {ppo_epoch}, skipping")
                            break
                        
                        # DEBUG: Stack new policy values
                        # IMPORTANT: Intermediate outputs (logprobs, values) might not show requires_grad=True
                        # even though they're part of the computation graph. This is normal!
                        # Gradients will flow when we use them in loss.backward() because they depend on
                        # model parameters that DO require gradients.
                        new_logprobs = torch.stack(logprobs_list)
                        new_values = torch.stack(values_list)
                        entropies = torch.stack(entropies_list)
                        
                        logger.debug(f"New policy - logprobs: {new_logprobs.shape}, values: {new_values.shape}, entropies: {entropies.shape}")
                        
                        # DEBUG: Verify model parameters require gradients (this is what matters for backprop)
                        # The outputs themselves might not show requires_grad=True, but if parameters do,
                        # gradients will flow when we compute the loss and call backward()
                        params_require_grad = sum(p.requires_grad for p in agent.parameters())
                        total_params = len(list(agent.parameters()))
                        if params_require_grad == 0:
                            logger.error("CRITICAL: No model parameters require gradients!")
                            raise RuntimeError("Model parameters don't require gradients - cannot train")
                        
                        logger.debug(f"Model parameters requiring grad: {params_require_grad}/{total_params}")
                        
                        # DEBUG: Test that computation graph is intact
                        # CRITICAL INSIGHT: Intermediate outputs might not show requires_grad=True,
                        # but they can still be part of computation graph (have grad_fn).
                        # What matters is that when we use them in a loss and call backward(),
                        # gradients flow to model parameters.
                        # 
                        # The test: Create a simple operation and check if it has grad_fn.
                        # If grad_fn exists, the computation graph is intact and gradients can flow.
                        try:
                            # Create a test operation using the outputs
                            test_loss = new_logprobs.mean() + new_values.mean()
                            
                            # Check if computation graph is intact
                            # grad_fn indicates the operation that created the tensor
                            # If it exists, the tensor is connected to the computation graph
                            has_grad_fn = test_loss.grad_fn is not None
                            
                            if not has_grad_fn:
                                # This is a problem - outputs aren't connected to model parameters
                                logger.error("CRITICAL: Loss doesn't have grad_fn - computation graph is broken!")
                                logger.error(f"new_logprobs.grad_fn: {new_logprobs.grad_fn}")
                                logger.error(f"new_values.grad_fn: {new_values.grad_fn}")
                                logger.error(f"new_logprobs.requires_grad: {new_logprobs.requires_grad}")
                                logger.error(f"new_values.requires_grad: {new_values.requires_grad}")
                                logger.error(f"Agent training mode: {agent.training}")
                                logger.error(f"Model params requiring grad: {params_require_grad}/{total_params}")
                                
                                # Try to diagnose: check if individual outputs have grad_fn
                                sample_logprob = logprobs_list[0] if len(logprobs_list) > 0 else None
                                sample_value = values_list[0] if len(values_list) > 0 else None
                                if sample_logprob is not None:
                                    logger.error(f"Sample logprob.grad_fn: {sample_logprob.grad_fn}")
                                if sample_value is not None:
                                    logger.error(f"Sample value.grad_fn: {sample_value.grad_fn}")
                                
                                raise RuntimeError("Computation graph is broken - outputs not connected to model parameters")
                            
                            logger.debug(f"✓ Computation graph intact: test_loss has grad_fn")
                            del test_loss
                        except RuntimeError:
                            raise  # Re-raise our custom error
                        except Exception as e:
                            logger.error(f"CRITICAL: Cannot create loss from outputs: {e}")
                            raise RuntimeError(f"Computation graph broken: {e}")
                        
                        # DEBUG: Old values are already detached from stacking, but ensure they're clones
                        # This prevents any potential issues with shared memory
                        old_logprobs_detached = old_logprobs.clone()
                        old_values_detached = old_values.clone()
                        
                        # DEBUG: Calculate advantages (rewards - old_values, old_values are detached)
                        # rewards is a tensor (no gradients), old_values_detached is detached (no gradients)
                        # advantages should not have gradients (they're computed from non-differentiable values)
                        advantages = rewards - old_values_detached
                        # ============================================================
                        # ADVANTAGE NORMALIZATION (CRITICAL FIX for Large Actor Loss)
                        # ============================================================
                        # CRITICAL FIX: Normalize advantages to prevent huge actor loss
                        # Large advantages (e.g., 50-100) cause actor_loss to be huge (38+)
                        # This makes entropy/actor ratio tiny even with good entropy
                        # Normalize advantages to have mean=0, std=1 (standard PPO practice)
                        # ============================================================
                        advantages_mean = advantages.mean()
                        advantages_std = advantages.std() + 1e-8  # Avoid division by zero
                        advantages = (advantages - advantages_mean) / advantages_std
                        
                        # ============================================================
                        # ADVANTAGE CLIPPING (from Principal ML Engineer Audit)
                        # ============================================================
                        # After normalization, clip to reasonable range (e.g., [-5, +5])
                        # This prevents extreme values while preserving signal
                        # ============================================================
                        ADVANTAGE_CLIP_MIN = -5.0   # Tighter: After normalization, ±5 std is reasonable
                        ADVANTAGE_CLIP_MAX = +5.0   # Tighter: Prevents extreme outliers
                        advantages = torch.clamp(advantages, min=ADVANTAGE_CLIP_MIN, max=ADVANTAGE_CLIP_MAX)
                        advantages = advantages.detach()  # Ensure advantages don't have gradients
                        
                        # Log advantage statistics for debugging
                        if step % 500 == 0:
                            logger.info(f"Advantage Stats (step {step}): mean={advantages.mean().item():.3f}, std={advantages.std().item():.3f}, min={advantages.min().item():.3f}, max={advantages.max().item():.3f}")
                        
                        # ============================================================
                        # PPO OBJECTIVE CALCULATION
                        # ============================================================
                        
                        # FIX 1: Ensure shapes match to prevent Broadcasting Explosion
                        if new_logprobs.shape != old_logprobs_detached.shape:
                            new_logprobs = new_logprobs.view_as(old_logprobs_detached)
                        
                        # Calculate ratios
                        ratios = torch.exp(new_logprobs - old_logprobs_detached)
                        
                        # Calculate Surrogates
                        surr1 = ratios * advantages
                        surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                        
                        # ==========================================
                        # DEBUG DIAGNOSTICS (Right before actor_loss)
                        # ==========================================
                        
                        # 1. Check if the model has actually changed since data collection
                        # If diff is 0.0, the model isn't learning (or learning rate is too low)
                        log_prob_diff = (new_logprobs - old_logprobs_detached).abs().mean().item()
                        
                        # 2. Check the raw scale of advantages
                        # If these are like 0.00001, your reward scale is too small relative to normalization
                        adv_mean = advantages.mean().item()
                        adv_std = advantages.std().item()
                        
                        # 3. Check the ratios directly
                        # If these are all exactly 1.0, your implementation of PPO is broken (detached incorrectly?)
                        ratio_mean = ratios.mean().item()
                        ratio_std = ratios.std().item()
                        
                        logger.info(f"--- DEBUG DIAGNOSTICS ---")
                        logger.info(f"LogProb Diff (New - Old): {log_prob_diff:.8f}")
                        logger.info(f"Advantages: Mean={adv_mean:.6f} | Std={adv_std:.6f}")
                        logger.info(f"Ratios: Mean={ratio_mean:.6f} | Std={ratio_std:.6f}")
                        logger.info(f"-------------------------")
                        
                        # Calculate PPO Loss (Negative because we want to maximize objective)
                        # Using .mean() here is correct
                        actor_loss = -torch.min(surr1, surr2).mean()
                        
                        # FIX 2: REMOVED "ACTOR_LOSS_MIN" SCALING BLOCK
                        # Reason: Artificially scaling loss creates unstable gradients. 
                        # If actor_loss is small, that is GOOD (means policy is stable). 
                        # Let the Entropy term handle exploration naturally.
                        
                        # Safety Clip (Optional but safe): Cap extreme losses if they still happen
                        if actor_loss.item() > 100.0:
                            logger.warning(f"Extreme Actor Loss detected: {actor_loss.item()}")
                            actor_loss = torch.clamp(actor_loss, max=100.0)
                        
                        # ============================================================
                        # CRITIC & TOTAL LOSS
                        # ============================================================
                        
                        # Critic loss (value function learning)
                        # new_values has gradients (from model), rewards is a tensor (no gradients)
                        # MSE loss will compute gradients through new_values
                        critic_loss = nn.MSELoss()(new_values, rewards)
                        
                        # Entropy bonus (encourages exploration)
                        entropy_bonus = entropies.mean()
                        entropy_value = entropy_bonus.item()  # Actual entropy value (not scaled)
                        
                        # Total loss (actor + critic - entropy)
                        # entropy_coef is persistent across epochs with gentle decay and adaptive updates
                        entropy_loss_term = entropy_coef * entropy_bonus
                        total_loss = actor_loss + 0.5 * critic_loss - entropy_loss_term
                        
                        # ============================================================
                        # ENTROPY RATIO MONITORING (CRITICAL FIX for Policy Collapse)
                        # ============================================================
                        # TARGET: Entropy should be 1-10% of actor loss
                        # If ratio > 1.0: Agent is random (entropy dominates)
                        # If ratio < 0.01: Agent has no exploration (POLICY COLLAPSE RISK!)
                        # ============================================================
                        actor_loss_val = abs(actor_loss.item())
                        entropy_loss_val = entropy_loss_term.item()
                        ratio = entropy_loss_val / max(0.0001, actor_loss_val)
                        
                        # CRITICAL DIAGNOSTIC: Log actual entropy value and entropy_coef
                        logger.debug(f"PPO epoch {ppo_epoch}: actor_loss={actor_loss_val:.6f}, "
                                   f"critic_loss={critic_loss.item():.6f}, "
                                   f"entropy_value={entropy_value:.4f}, entropy_coef={entropy_coef:.6f}, "
                                   f"entropy_loss={entropy_loss_val:.6f}, "
                                   f"entropy/actor_ratio={ratio:.2%}, total_loss={total_loss.item():.6f}")
                        
                        # ============================================================
                        # SMART ENTROPY TUNER (Patch for Normalized PPO)
                        # ============================================================
                        
                        # 1. Don't panic if Actor Loss is naturally small
                        # If the model is learning (LogProb Diff > 0.01), the "ratio" doesn't matter.
                        is_learning_fast = (new_logprobs - old_logprobs_detached).abs().mean().item() > 0.01
                        
                        # CRITICAL FIX: More aggressive auto-increase for low ratios
                        # 0.263% is way too low - we need at least 1% (0.01)
                        # Also handle entropy domination (ratio > 1.0 = 100%)
                        if ratio > 1.0:  # Entropy loss > Actor loss (entropy dominates)
                            logger.warning(f"[WARNING] ENTROPY DOMINATION: entropy/actor ratio = {ratio:.1%}. Reduce entropy_coef!")
                            logger.warning(f"   DIAGNOSTIC: entropy_value={entropy_value:.4f}, entropy_coef={entropy_coef:.6f}, actor_loss={actor_loss_val:.6f}")
                            
                            # SMART LOGIC: Only decrease if we are genuinely stuck (not learning)
                            if is_learning_fast:
                                # If we are learning fast, IGNORE the ratio. 
                                # The high ratio is just a math artifact of normalized advantages.
                                logger.info(f"   Model is learning fast (LogProb Diff > 0.01), ignoring high ratio")
                                pass
                            elif ratio > 2.0 and entropy_coef > 0.001:
                                logger.warning(f"   High Entropy Ratio ({ratio:.2f}), but checking learning speed first...")
                                # Only decrease if we are genuinely stuck (not learning)
                                old_entropy_coef = entropy_coef
                                entropy_coef *= 0.95
                                logger.warning(f"   AUTO-DECREASING entropy_coef: {old_entropy_coef:.6f} -> {entropy_coef:.6f} (gentle decrease)")
                        elif ratio < 0.01:  # Below 1% is problematic
                            logger.warning(f"NO EXPLORATION: entropy/actor ratio = {ratio:.3%}. Increase entropy_coef!")
                            logger.warning(f"   DIAGNOSTIC: entropy_value={entropy_value:.4f}, entropy_coef={entropy_coef:.6f}, actor_loss={actor_loss_val:.6f}")
                            
                            # CRITICAL: Auto-increase entropy more aggressively
                            # If ratio < 1%, we need to increase entropy_coef significantly
                            if ratio < 0.01:  # Below 1%
                                old_entropy_coef = entropy_coef
                                # Increase by factor needed to reach 2% ratio
                                target_ratio = 0.02  # Target 2% (safe middle ground)
                                increase_factor = target_ratio / max(ratio, 0.0001)  # Avoid division by zero
                                # CRITICAL: Allow much higher entropy_coef to reach 1-10% ratio
                                # With entropy=0.78 and actor_loss=0.25, we need entropy_coef ~0.003-0.03
                                # Cap at ENTROPY_COEF_MAX to allow proper exploration while preventing runaway
                                entropy_coef = min(entropy_coef * increase_factor, ENTROPY_COEF_MAX)
                                logger.warning(f"   AUTO-INCREASING entropy_coef: {old_entropy_coef:.6f} -> {entropy_coef:.6f} (factor={increase_factor:.2f}x, cap={ENTROPY_COEF_MAX:.6f})")
                                
                                # Also check if entropy itself is too low (distribution too peaked)
                                if entropy_value < 0.5:  # For 3 actions, max entropy is ~1.1, so 0.5 is low
                                    logger.warning(f"   [WARNING] ENTROPY VALUE TOO LOW: {entropy_value:.4f} (should be >0.5 for 3 actions)")
                                    logger.warning(f"   This suggests logits are too extreme - check temperature scaling!")
                        
                        # DEBUG: Check for NaN before backward
                        if torch.isnan(total_loss) or torch.isinf(total_loss):
                            logger.warning(f"NaN/Inf loss in {strategy} training at epoch {ppo_epoch}, skipping update")
                            break
                        
                        # DEBUG: Backward pass (compute gradients)
                        total_loss.backward()
                        
                        # GRADIENT CLIPPING (from Principal ML Engineer Audit)
                        # Prevents logit explosion from extreme gradients
                        GRAD_CLIP_NORM = 1.0
                        grad_norm = torch.nn.utils.clip_grad_norm_(agent.parameters(), GRAD_CLIP_NORM)
                        
                        # DEBUG: Optimizer step (update parameters)
                        optimizer.step()
                        
                        # Log PPO metrics to wandb (every 100 steps to avoid spam)
                        if use_wandb and WANDB_AVAILABLE and step % 100 == 0:
                            wandb.log({
                                f"stage2/{strategy}/ppo_epoch": ppo_epoch,
                                f"stage2/{strategy}/actor_loss": actor_loss_val,
                                f"stage2/{strategy}/critic_loss": critic_loss.item(),
                                f"stage2/{strategy}/entropy_value": entropy_value,
                                f"stage2/{strategy}/entropy_coef": entropy_coef,
                                f"stage2/{strategy}/entropy_loss": entropy_loss_val,
                                f"stage2/{strategy}/entropy_actor_ratio": ratio,
                                f"stage2/{strategy}/total_loss": total_loss.item(),
                                f"stage2/{strategy}/grad_norm": grad_norm.item(),
                                f"stage2/{strategy}/advantage_mean": advantages.mean().item(),
                                f"stage2/{strategy}/advantage_std": advantages.std().item(),
                                f"stage2/{strategy}/advantage_min": advantages.min().item(),
                                f"stage2/{strategy}/advantage_max": advantages.max().item(),
                                f"stage2/{strategy}/reward_mean": rewards.mean().item(),
                                f"stage2/{strategy}/reward_std": rewards.std().item(),
                            })
                    
                    logger.debug(f"PPO update complete, clearing memory")
                    # Clear memory after update
                    memory_states = []
                    memory_actions = []
                    memory_logprobs = []
                    memory_rewards = []
                    memory_values = []
            
            if (epoch + 1) % 10 == 0:
                # Get summary for THIS epoch only (stats were reset at start of epoch)
                summary = stats.get_summary()
                
                # CRITICAL FIX: Log balance change to verify it's actually changing
                initial_bal = summary['initial_balance'] if summary['initial_balance'] is not None else 0.0
                current_bal = summary['current_balance'] if summary['current_balance'] is not None else 0.0
                balance_change = current_bal - initial_bal
                
                logger.info(f"  Epoch {epoch + 1}/{epochs_per_agent}, Episode Reward: {episode_reward:.2f}")
                logger.info(f"    Actions: {summary['action_counts']}, Trades: {summary['trade_count']}, "
                          f"Return: {summary['return_pct']:.2f}%")
                logger.info(f"    Balance: ${initial_bal:.2f} -> ${current_bal:.2f} (Change: ${balance_change:+.2f})")
                
                # DEBUG: Analyze reward vs profit mismatch (both directions)
                if summary['total_reward'] < 0 and summary['total_profit'] > 0:
                    logger.warning(f"    [REWARD MISMATCH] Negative reward ({summary['total_reward']:.2f}) but positive profit (${summary['total_profit']:.2f})!")
                    logger.warning(f"    This suggests penalties/transaction costs are dominating the reward signal")
                elif summary['total_reward'] > 0 and summary['total_profit'] < 0:
                    logger.warning(f"    [REWARD MISMATCH] Positive reward ({summary['total_reward']:.2f}) but negative profit (${summary['total_profit']:.2f})!")
                    logger.warning(f"    This suggests rewards are not aligned with actual P&L - check reward calculation")
                    logger.warning(f"    Possible causes: reward calculation uses log_return which can be positive even with losses")
                    logger.warning(f"    Or: reward includes bonuses/multipliers that don't reflect actual balance change")
                    
                    # ALWAYS log reward statistics (even if empty, show what we have)
                    if 'reward_stats' in summary and summary['reward_stats']:
                        stats = summary['reward_stats']
                        logger.info(f"    Reward Stats: Min={stats['min_reward']:.4f}, Max={stats['max_reward']:.4f}, "
                                  f"Mean={stats['mean_reward']:.4f}, Std={stats['std_reward']:.4f}")
                        logger.info(f"    Reward Distribution: Positive={stats['positive_rewards']}, "
                                  f"Negative={stats['negative_rewards']}, Zero={stats['zero_rewards']} (Total steps: {stats['total_steps']})")
                    else:
                        logger.warning(f"    [DEBUG] No reward_stats available in summary")
                    
                    # Log reward components breakdown (show what we have, even if incomplete)
                    if 'reward_components' in summary and summary['reward_components']:
                        comps = summary['reward_components']
                        logger.info(f"    Reward Components Breakdown:")
                        
                        if 'total_pnl_reward' in comps:
                            logger.info(f"      PnL Reward: Total={comps['total_pnl_reward']:.4f}, "
                                      f"Avg={comps['avg_pnl_reward']:.4f}, Count={comps['count_pnl_reward']}")
                        else:
                            logger.warning(f"      [DEBUG] PnL Reward: Not tracked (reward_breakdown may not be in info dict)")
                        
                        if 'total_transaction_cost' in comps:
                            logger.info(f"      Transaction Costs: Total={comps['total_transaction_cost']:.4f}, "
                                      f"Avg={comps['avg_transaction_cost']:.4f}, Count={comps['count_transaction_cost']}")
                        else:
                            logger.warning(f"      [DEBUG] Transaction Costs: Not tracked")
                        
                        if 'total_volatility_penalty' in comps:
                            logger.info(f"      Volatility Penalties: Total={comps['total_volatility_penalty']:.4f}, "
                                      f"Avg={comps['avg_volatility_penalty']:.4f}, Count={comps['count_volatility_penalty']}")
                        else:
                            logger.warning(f"      [DEBUG] Volatility Penalties: Not tracked")
                        
                        if 'total_stop_loss_penalty' in comps:
                            logger.info(f"      Stop Loss Penalties: Total={comps['total_stop_loss_penalty']:.4f}, "
                                      f"Avg={comps['avg_stop_loss_penalty']:.4f}, Count={comps['count_stop_loss_penalty']}")
                        else:
                            logger.warning(f"      [DEBUG] Stop Loss Penalties: Not tracked")
                        
                        if 'total_whipsaw_penalty' in comps:
                            logger.info(f"      Whipsaw Penalties: Total={comps['total_whipsaw_penalty']:.4f}, "
                                      f"Avg={comps['avg_whipsaw_penalty']:.4f}, Count={comps['count_whipsaw_penalty']}")
                        else:
                            logger.warning(f"      [DEBUG] Whipsaw Penalties: Not tracked")
                        
                        # Show action-based rewards (what's actually available in current reward_breakdown)
                        if 'total_action_reward' in comps:
                            logger.info(f"      Action Rewards: Total={comps['total_action_reward']:.4f}, "
                                      f"Avg={comps['avg_action_reward']:.4f}, Count={comps['count_action_reward']}")
                        if 'total_holding_reward' in comps:
                            logger.info(f"      Holding Rewards: Total={comps['total_holding_reward']:.4f}, "
                                      f"Avg={comps['avg_holding_reward']:.4f}, Count={comps['count_holding_reward']}")
                        if 'total_stop_loss_reward' in comps:
                            logger.info(f"      Stop Loss Rewards: Total={comps['total_stop_loss_reward']:.4f}, "
                                      f"Avg={comps['avg_stop_loss_reward']:.4f}, Count={comps['count_stop_loss_reward']}")
                        if 'total_take_profit_reward' in comps:
                            logger.info(f"      Take Profit Rewards: Total={comps['total_take_profit_reward']:.4f}, "
                                      f"Avg={comps['avg_take_profit_reward']:.4f}, Count={comps['count_take_profit_reward']}")
                        
                        # Calculate what's causing the negative reward (use available data)
                        total_penalties = (
                            comps.get('total_transaction_cost', 0) +
                            comps.get('total_volatility_penalty', 0) +
                            comps.get('total_stop_loss_penalty', 0) +
                            comps.get('total_whipsaw_penalty', 0)
                        )
                        pnl_component = comps.get('total_pnl_reward', summary['total_reward'])
                        logger.info(f"    Estimated: PnL Component={pnl_component:.2f}, Total Penalties={total_penalties:.2f}")
                        logger.info(f"    Net Reward = {pnl_component:.2f} - {total_penalties:.2f} = {pnl_component - total_penalties:.2f}")
                    else:
                        logger.warning(f"    [DEBUG] No reward_components available - environment may not be providing reward_breakdown in info dict")
                        logger.warning(f"    [DEBUG] Available summary keys: {list(summary.keys())}")
                        logger.warning(f"    [DEBUG] This means the environment's step() method needs to include 'reward_breakdown' in the info dict")
                    
                    # Calculate reward/profit ratio
                    if summary['total_profit'] != 0:
                        reward_profit_ratio = summary['total_reward'] / summary['total_profit']
                        logger.warning(f"    Reward/Profit Ratio: {reward_profit_ratio:.4f} (should be positive if aligned)")
                        logger.warning(f"    This means reward is {abs(reward_profit_ratio)*100:.1f}% of profit magnitude")
                    
                    # Additional analysis: Show per-step average
                    if summary.get('avg_reward', 0) != 0:
                        logger.info(f"    Per-Step Analysis: Avg reward per step={summary['avg_reward']:.4f}, "
                                  f"Avg profit per step=${summary.get('avg_profit', 0):.4f}")
                        logger.info(f"    Total steps: {len(summary.get('rewards_history', []))}")
                
                # DEBUG: Warn if balance didn't change (might indicate environment reset issue)
                if abs(balance_change) < 0.01 and summary['trade_count'] > 0:
                    logger.warning(f"    [WARNING] Balance didn't change (${balance_change:.2f}) despite {summary['trade_count']} trades!")
                    logger.warning(f"    This may indicate balance is being reset or not updated properly in environment")
                
                # REWARD BREAKDOWN: Log invalid action counts if available
                if 'invalid_action_counts' in summary and summary['invalid_action_counts']:
                    logger.info(f"    Invalid Actions: {summary['invalid_action_counts']}")
                
                # Log epoch summary to wandb
                if use_wandb and WANDB_AVAILABLE:
                    action_counts = summary.get('action_counts', {})
                    wandb.log({
                        f"stage2/{strategy}/epoch": epoch + 1,
                        f"stage2/{strategy}/episode_reward": episode_reward,
                        f"stage2/{strategy}/total_profit": summary.get('total_profit', 0.0),
                        f"stage2/{strategy}/return_pct": summary.get('return_pct', 0.0),
                        f"stage2/{strategy}/balance": current_bal,
                        f"stage2/{strategy}/balance_change": balance_change,
                        f"stage2/{strategy}/trade_count": summary.get('trade_count', 0),
                        f"stage2/{strategy}/win_count": summary.get('win_count', 0),
                        f"stage2/{strategy}/loss_count": summary.get('loss_count', 0),
                        f"stage2/{strategy}/win_rate": summary.get('win_rate', 0.0),
                        f"stage2/{strategy}/action_HOLD": action_counts.get('HOLD', 0),
                        f"stage2/{strategy}/action_BUY": action_counts.get('BUY', 0),
                        f"stage2/{strategy}/action_SELL": action_counts.get('SELL', 0),
                        f"stage2/{strategy}/avg_reward": summary.get('avg_reward', 0.0),
                        f"stage2/{strategy}/avg_profit": summary.get('avg_profit', 0.0),
                    })
        
        # Log final statistics for this strategy
        stats.log_summary(prefix=f"{strategy}")
        agents[strategy] = agent
        logger.info(f"{strategy} agent pre-training complete!")
    
    return agents


# ==============================================================================
# STAGE 3: TRAIN META-STRATEGY AGENT
# ==============================================================================

def train_meta_strategy_agent(transformer: MultiScaleTransformerEncoder,
                              execution_agents: Dict[str, nn.Module],
                              preprocessor: MultiTimescalePreprocessor,
                              regime_detector: EnhancedRegimeDetector,
                              mode: str = "stock",
                              steps: int = 20000,
                              lr: float = 0.0003,  # 3e-4: Standard for PPO/Actor-Critic (RL needs higher LR than transformers)
                              window_size: int = 30,
                              logger: Optional[logging.Logger] = None,
                              use_wandb: bool = True) -> MetaStrategyAgent:
    """
    Train meta-strategy agent (with frozen execution agents).
    
    Returns:
        Trained meta-strategy agent
    """
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STAGE 3: Training Meta-Strategy Agent")
    logger.info("=" * 60)
    
    # Determine input_dim from transformer (should match data)
    input_dim = transformer.encoder_1d.input_dim if hasattr(transformer, 'encoder_1d') else (8 if mode == "stock" else 5)
    
    # Create meta-strategy agent with pre-trained transformer (match transformer architecture)
    meta_agent = MetaStrategyAgent(
        d_model=64,  # Match pre-trained transformer
        nhead=4,  # Match pre-trained transformer
        num_layers=2,  # Match pre-trained transformer
        dim_feedforward=256,  # Match pre-trained transformer
        dropout=0.1,
        input_dim=input_dim,  # Match transformer input_dim
        mode=mode,
        hidden_dim=128  # Reduced to match smaller model
    ).to(device)
    
    # Transfer transformer weights (only matching layers)
    pretrained_state = transformer.state_dict()
    agent_state = meta_agent.transformer_encoder.state_dict()
    
    # Load only matching keys
    matched_state = {}
    for key, value in pretrained_state.items():
        if key in agent_state and agent_state[key].shape == value.shape:
            matched_state[key] = value
    
    meta_agent.transformer_encoder.load_state_dict(matched_state, strict=False)
    logger.info(f"Loaded {len(matched_state)}/{len(pretrained_state)} transformer weights into meta-agent")
    
    # Freeze execution agents
    for agent in execution_agents.values():
        for param in agent.parameters():
            param.requires_grad = False
    
    # Create meta-strategy environment
    config = TradingConfig()
    config.mode = mode
    # TEMPORARY: Set fees to 0.0% for the next 50 steps (meta-strategy uses steps, not epochs)
    # This removes transaction cost friction and allows agent to explore more freely
    FEE_FREE_STEPS = 50
    original_commission = config.commission_rate
    original_slippage = config.slippage_bps
    config.commission_rate = 0.0  # 0.0% commission
    config.slippage_bps = 0.0  # 0.0% slippage
    logger.info(f"[FEE-FREE MODE] Setting fees to 0.0% for first {FEE_FREE_STEPS} steps")
    logger.info(f"  Original: commission={original_commission:.4f} ({original_commission*100:.2f}%), slippage={original_slippage:.1f} bps")
    logger.info(f"  Temporary: commission=0.0 (0.0%), slippage=0.0 bps")
    
    meta_env = MetaStrategyEnv(
        ticker=preprocessor.ticker,
        window_size=window_size,
        config=config,
        execution_horizon=10,
        execution_agents=execution_agents,  # Pass execution agents
        logger=logger
    )
    
    # Optimizer
    optimizer = optim.Adam(meta_agent.parameters(), lr=lr)
    
    # Training loop (PPO-style)
    logger.info(f"Training for {steps} steps...")
    
    # Initialize statistics tracker
    stats = TrainingStats(logger)
    
    memory_states = []
    memory_actions = []
    memory_logprobs = []
    memory_rewards = []
    
    # Fetch data once before training (cache it)
    logger.info("Fetching and preprocessing data once for meta-strategy training...")
    aligned_data, features = preprocessor.process(window_size=window_size)
    if not aligned_data or not features:
        logger.error("Failed to fetch data for meta-strategy agent")
        return meta_agent
    
    state, _ = meta_env.reset()
    episode_reward = 0.0
    
    # FIX: Meta-agent should decide every 10-20 steps (Trade Cycle), not every step
    # If it switches every step, it creates "Strategy Flicker" (Buy -> Sell -> Buy -> Sell), which burns money on fees
    META_DECISION_INTERVAL = 15  # Meta-agent makes decision every 15 steps (reduces flicker)
    last_meta_action = None  # Track last meta-action
    steps_since_meta_decision = 0  # Track steps since last meta-decision
    strategy_history = []  # Track last 10 strategy choices for flicker detection
    
    for step in range(1, steps + 1):
        # RESTORE FEES after 50 steps (if we're past the fee-free period)
        if step == FEE_FREE_STEPS + 1:
            config.commission_rate = original_commission
            config.slippage_bps = original_slippage
            logger.info(f"[FEE RESTORED] Step {step}: Restoring original fees")
            logger.info(f"  Commission: 0.0 -> {original_commission:.4f} ({original_commission*100:.2f}%)")
            logger.info(f"  Slippage: 0.0 -> {original_slippage:.1f} bps")
        
        # BOOST ENTROPY: Increased from 0.0001 to 0.01 to force exploration
        # SLOW ENTROPY DECAY: Keep agent curious for longer (until step 10000+)
        # Entropy coefficient: Start at 0.01 (boosted from 0.0001), decay slowly using 0.99 per step
        # Use slower decay (0.99 per step) so entropy stays above zero for 100+ epochs
        entropy_coef = 0.01 * (0.99 ** step)  # Slow exponential decay (0.99 per step), boosted 100x from 0.0001
        entropy_coef = max(entropy_coef, 0.001)  # Clamp to minimum (boosted 100x from 0.00001)
        progress = step / steps  # For logging only
        # Get windowed features from cached data
        windowed = preprocessor.get_window_features(features, window_size, None)
        
        # Convert to tensors (FIX: Ensure float32 dtype to match model)
        features_dict = {}
        for interval, feat in windowed.items():
            # Convert to float32 to match model dtype
            feat_tensor = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            features_dict[interval] = feat_tensor
        
        # Get regime
        if aligned_data:
            regime_key = "1d" if "1d" in aligned_data else list(aligned_data.keys())[0]
            regime_df = aligned_data[regime_key]
            if not regime_df.empty:
                # Extract Close prices for regime detection
                close_prices = regime_df['Close'].values if 'Close' in regime_df.columns else regime_df.iloc[:, 0].values
                # FIX: Ensure float32 dtype
                regime_features = torch.from_numpy(regime_detector.get_regime_features(close_prices)).float().unsqueeze(0).to(device)
            else:
                regime_features = torch.zeros(1, 4, dtype=torch.float32).to(device)
        else:
            regime_features = torch.zeros(1, 4, dtype=torch.float32).to(device)
        
        # FIX: Meta-agent should only make decision every META_DECISION_INTERVAL steps
        # This prevents "Strategy Flicker" (switching strategies too frequently)
        if steps_since_meta_decision >= META_DECISION_INTERVAL or last_meta_action is None:
            # Time for meta-agent to make a decision
            
            # Get regime label for logging
            regime_label = "UNKNOWN"
            if aligned_data:
                regime_key = "1d" if "1d" in aligned_data else list(aligned_data.keys())[0]
                regime_df = aligned_data[regime_key]
                if not regime_df.empty:
                    close_prices = regime_df['Close'].values if 'Close' in regime_df.columns else regime_df.iloc[:, 0].values
                    regime_label = regime_detector.detect_regime(close_prices)
            
            # Get action with epsilon-greedy exploration
            # FIX: Reduced epsilon for financial model - Manager should listen to Regime Detector, not explore randomly
            # Epsilon of 0.495 is way too high for a financial model
            epsilon_start = 0.1  # Reduced from 0.5 to 0.1 (or 0.2) - Manager doesn't need random exploration
            epsilon_min = 0.05  # Minimum epsilon
            epsilon = max(epsilon_min, epsilon_start * (1 - step / steps))  # Decay from 0.1 to 0.05
            if np.random.random() < epsilon:
                # Random exploration
                action = np.random.randint(0, 4)  # 4 strategies
                # Get log_prob and value for random action
                logits, value_tensor = meta_agent.forward(features_dict, regime_features)
                
                # DEBUG: Print logits every 100 steps to check if agent is making decisions
                if step % 100 == 0:
                    logits_np = logits[0].detach().cpu().numpy()
                    logger.info(f"Step {step}: DEBUG LOGITS (Meta-Strategy, Epsilon={epsilon:.3f}): {logits_np}")
                    # Check if logits are all identical (dead network)
                    if len(logits_np) > 1 and np.allclose(logits_np, logits_np[0], atol=1e-6):
                        logger.warning(f"  ⚠️  WARNING: All logits are identical! Network may be dead. Logits: {logits_np}")
                
                dist = torch.distributions.Categorical(logits=logits)
                # FIX: Ensure long dtype for action
                log_prob = dist.log_prob(torch.tensor(action, dtype=torch.long).to(device))
                value = value_tensor.squeeze().item() if value_tensor.dim() > 0 else value_tensor.item()
            else:
                # Get logits for debugging before calling act()
                logits_debug, _ = meta_agent.forward(features_dict, regime_features)
                
                # DEBUG: Print logits every 100 steps to check if agent is making decisions
                if step % 100 == 0:
                    logits_np = logits_debug[0].detach().cpu().numpy()
                    epsilon_start = 0.1
                    epsilon_min = 0.05
                    epsilon = max(epsilon_min, epsilon_start * (1 - step / steps))
                    logger.info(f"Step {step}: DEBUG LOGITS (Meta-Strategy, Epsilon={epsilon:.3f}): {logits_np}")
                    # Check if logits are all identical (dead network)
                    if len(logits_np) > 1 and np.allclose(logits_np, logits_np[0], atol=1e-6):
                        logger.warning(f"  ⚠️  WARNING: All logits are identical! Network may be dead. Logits: {logits_np}")
                
                action, log_prob, value = meta_agent.act(features_dict, regime_features, deterministic=False)
            
            # Update meta-decision tracking (outside epsilon if-else, but inside meta-decision check)
            last_meta_action = action
            steps_since_meta_decision = 0
            
            # Track strategy choice for flicker detection
            strategy_names = {0: "TREND_FOLLOW", 1: "MEAN_REVERT", 2: "MOMENTUM", 3: "RISK_OFF"}
            strategy_name = strategy_names.get(action, f"STRATEGY_{action}")
            strategy_history.append(action)
            if len(strategy_history) > 10:
                strategy_history.pop(0)  # Keep last 10 choices
            
            # LOG: Regime input vs action output (to see if Manager is hallucinating)
            logger.info(f"Meta-Step {step}: Detected Regime={regime_label} -> Selected Strategy={strategy_name} (Action={action})")
            
            # LOG: Strategy sequence for flicker detection
            if len(strategy_history) >= 5:
                strategy_sequence = [strategy_names.get(s, f"S{s}") for s in strategy_history[-10:]]
                logger.info(f"  Strategy History (last {len(strategy_history)}): {strategy_sequence}")
                # Check for flicker (rapid switching)
                unique_strategies = len(set(strategy_history[-5:]))  # Count unique strategies in last 5
                if unique_strategies >= 4:  # 4+ different strategies in last 5 = flicker
                    logger.warning(f"  ⚠️  STRATEGY FLICKER DETECTED: {unique_strategies} different strategies in last 5 decisions!")
        else:
            # Reuse last meta-action (don't make new decision)
            action = last_meta_action
            # Get log_prob and value for the reused action (needed for PPO)
            logits, value_tensor = meta_agent.forward(features_dict, regime_features)
            dist = torch.distributions.Categorical(logits=logits)
            log_prob = dist.log_prob(torch.tensor(action, dtype=torch.long).to(device))
            value = value_tensor.squeeze().item() if value_tensor.dim() > 0 else value_tensor.item()
            steps_since_meta_decision += 1
        
        # Step environment
        next_state, reward, done, truncated, info = meta_env.step(action)
        
        # Update statistics (define strategy_name before using it)
        strategy_names = {0: "TREND_FOLLOW", 1: "MEAN_REVERT", 2: "MOMENTUM", 3: "RISK_OFF"}
        strategy_name = strategy_names.get(action, f"STRATEGY_{action}")
        
        # Check if hard stop-loss was triggered (Manager penalty)
        hard_stop_loss_triggered = info.get('hard_stop_loss', False)
        manager_penalty = 0.0  # Initialize manager penalty
        if hard_stop_loss_triggered:
            # Manager penalty: If Manager picked wrong strategy and it hit hard stop-loss, Manager loses points
            manager_penalty = -2.0  # Heavy penalty for the Manager
            reward += manager_penalty
            logger.warning(f"  Step {step}: HARD STOP-LOSS triggered! Manager penalty: {manager_penalty:.2f} (Strategy: {strategy_name})")
        
            # Clip reward to prevent extreme values
            # FIX: Match environment clipping range [-500, +500] (after REWARD_SCALE=10.0)
            reward = np.clip(reward, -500.0, 500.0)
        
        # LOG: Meta-Reward Breakdown (to see if Manager is getting punished for volatility)
        if step % 50 == 0 or hard_stop_loss_triggered:  # Log every 50 steps or when stop-loss triggers
            pnl_component = info.get('return_pct', 0.0) * 100 if 'return_pct' in info else 0.0
            execution_reward = info.get('execution_reward', 0.0)
            volatility_penalty = 0.0  # Will be extracted from execution_reward if available
            logger.info(f"Meta-Reward Breakdown (Step {step}): Total={reward:.4f} | From_PnL={pnl_component:.4f} | Execution_Reward={execution_reward:.4f} | Hard_SL_Penalty={manager_penalty:.4f}")
        if 'balance' not in info:
            # Try to get balance from execution env
            if hasattr(meta_env, 'current_balance'):
                info['balance'] = meta_env.current_balance
            if hasattr(meta_env, 'initial_balance'):
                info['initial_balance'] = meta_env.initial_balance
        stats.update(reward, info, action=action, strategy=strategy_name)
        
        # Store in memory (FIX: Ensure correct dtypes and shapes)
        memory_states.append((features_dict, regime_features))
        memory_actions.append(torch.tensor(action, dtype=torch.long))
        # Ensure log_prob is always a 1D tensor with shape [1] for stacking
        if isinstance(log_prob, torch.Tensor):
            if log_prob.dim() == 0:  # Scalar
                log_prob_tensor = log_prob.unsqueeze(0)
            else:
                log_prob_tensor = log_prob.flatten()[:1]  # Take first element and ensure 1D
        else:
            log_prob_tensor = torch.tensor([log_prob], dtype=torch.float32)
        memory_logprobs.append(log_prob_tensor)
        memory_rewards.append(reward)
        
        state = next_state
        episode_reward += reward
        
        # PPO update with larger batch size for better GPU utilization
        # Increased batch size from implicit 500 to 2000 (4x) for Stage 3
        PPO_BATCH_SIZE_STAGE3 = 2000
        if step % 500 == 0 and len(memory_rewards) >= PPO_BATCH_SIZE_STAGE3:
            if len(memory_rewards) > 0:
                # Calculate discounted rewards
                rewards = []
                discounted_reward = 0
                for r in reversed(memory_rewards):
                    discounted_reward = r + 0.99 * discounted_reward
                    rewards.insert(0, discounted_reward)
                
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
                
                # PPO optimization
                # Ensure all tensors have same shape before stacking
                memory_actions_ready = []
                for a in memory_actions:
                    if isinstance(a, torch.Tensor):
                        if a.dim() == 0:
                            memory_actions_ready.append(a.unsqueeze(0))
                        else:
                            memory_actions_ready.append(a.flatten()[:1])
                    else:
                        memory_actions_ready.append(torch.tensor([a], dtype=torch.long))
                
                memory_logprobs_ready = []
                for lp in memory_logprobs:
                    if isinstance(lp, torch.Tensor):
                        if lp.dim() == 0:  # Scalar
                            memory_logprobs_ready.append(lp.unsqueeze(0))
                        else:
                            memory_logprobs_ready.append(lp.flatten()[:1])  # Ensure 1D
                    else:
                        memory_logprobs_ready.append(torch.tensor([lp], dtype=torch.float32))
                
                old_actions = torch.stack(memory_actions_ready).to(device).squeeze()
                old_logprobs = torch.stack(memory_logprobs_ready).to(device).squeeze()
                
                for _ in range(3):
                    # Re-evaluate
                    logprobs_list = []
                    values_list = []
                    entropies_list = []
                    
                    for (feat_dict, reg_feat), old_action in zip(memory_states, old_actions):
                        # FIX: Ensure all tensors are float32 before evaluation
                        feat_dict_ready = {}
                        for key, val in feat_dict.items():
                            if isinstance(val, torch.Tensor):
                                feat_dict_ready[key] = val.float() if val.dtype != torch.float32 else val
                            else:
                                feat_dict_ready[key] = torch.tensor(val, dtype=torch.float32, device=device)
                        
                        reg_feat_ready = reg_feat.float() if reg_feat.dtype != torch.float32 else reg_feat
                        old_action_ready = old_action.long() if old_action.dtype != torch.long else old_action
                        
                        # Check for NaN in features before evaluation
                        has_nan = False
                        for feat_tensor in feat_dict_ready.values():
                            if torch.isnan(feat_tensor).any() or torch.isinf(feat_tensor).any():
                                has_nan = True
                                break
                        if torch.isnan(reg_feat_ready).any() or torch.isinf(reg_feat_ready).any():
                            has_nan = True
                        
                        if has_nan:
                            logger.warning("Skipping batch with NaN/Inf features in meta-strategy evaluation")
                            continue
                        
                        try:
                            logprobs, values, entropy = meta_agent.evaluate(feat_dict_ready, reg_feat_ready, old_action_ready.unsqueeze(0))
                        except RuntimeError as e:
                            if "dtype" in str(e) or "Double" in str(e) or "Float" in str(e):
                                logger.error(f"Error in re-evaluation at idx {len(logprobs_list)}: {e}, skipping this sample")
                                continue
                            else:
                                raise
                        
                        # Check for NaN in outputs
                        if torch.isnan(logprobs).any() or torch.isnan(values).any() or torch.isnan(entropy).any():
                            logger.warning("NaN detected in meta-strategy evaluation outputs, skipping batch")
                            continue
                        
                        logprobs_list.append(logprobs)
                        values_list.append(values)
                        entropies_list.append(entropy)
                    
                    if len(logprobs_list) == 0:
                        logger.warning("No valid batches for PPO update, skipping")
                        continue
                    
                    logprobs = torch.stack(logprobs_list).squeeze()
                    values = torch.stack(values_list).squeeze()
                    entropy = torch.stack(entropies_list).mean()
                    
                    # Check for NaN before loss calculation
                    if torch.isnan(logprobs).any() or torch.isnan(values).any() or torch.isnan(entropy).any():
                        logger.warning("NaN detected in stacked tensors, skipping PPO update")
                        continue
                    
                    # PPO loss
                    ratios = torch.exp(logprobs - old_logprobs.detach())
                    advantages = rewards - values.detach()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = 0.5 * nn.MSELoss()(values, rewards)
                    entropy_loss_term = entropy_coef * entropy
                    
                    # CRITICAL DEBUG: Check for entropy domination (Stage 3)
                    actor_loss_val = actor_loss.item()
                    entropy_loss_val = entropy_loss_term.item()
                    ratio = entropy_loss_val / (actor_loss_val + 1e-8)
                    
                    if ratio > 10.0 and step % 1000 == 0:  # Log occasionally
                        logger.warning(f"[WARNING] ENTROPY DOMINATION (Stage 3): entropy_loss ({entropy_loss_val:.6f}) is {ratio:.1f}x larger than actor_loss ({actor_loss_val:.6f})!")
                    
                    # ENTROPY ANNEALING: Use decaying entropy coefficient (calculated above)
                    # Starts at 0.01 for exploration, decays to 0.001 for convergence
                    loss = actor_loss + critic_loss - entropy_loss_term
                    
                    # Check for NaN loss
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        logger.warning("NaN/Inf loss detected, skipping update")
                        continue
                    
                    optimizer.zero_grad()
                    loss.mean().backward()
                    
                    # Check for NaN gradients
                    has_nan_grad = False
                    for param in meta_agent.parameters():
                        if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                            has_nan_grad = True
                            break
                    
                    if has_nan_grad:
                        logger.warning("NaN gradients detected, skipping update")
                        optimizer.zero_grad()
                        continue
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(meta_agent.parameters(), max_norm=0.5)
                    optimizer.step()
                    
                    # Log to wandb (every PPO update)
                    if use_wandb and WANDB_AVAILABLE:
                        # Calculate advantages for logging
                        advantages_log = rewards - values.detach()
                        wandb.log({
                            "stage3/step": step,
                            "stage3/actor_loss": actor_loss_val,
                            "stage3/critic_loss": critic_loss.item(),
                            "stage3/entropy": entropy.item(),
                            "stage3/entropy_coef": entropy_coef,
                            "stage3/entropy_loss": entropy_loss_val,
                            "stage3/total_loss": loss.mean().item(),
                            "stage3/grad_norm": grad_norm.item(),
                            "stage3/entropy_actor_ratio": ratio,
                            "stage3/advantage_mean": advantages_log.mean().item() if len(advantages_log) > 0 else 0.0,
                            "stage3/advantage_std": advantages_log.std().item() if len(advantages_log) > 0 else 0.0,
                            "stage3/reward_mean": rewards.mean().item(),
                            "stage3/reward_std": rewards.std().item(),
                        })
                
                # Clear memory
                memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
            
            # Get summary statistics (accumulated since last reset)
            summary = stats.get_summary()
            logger.info(f"Step {step}/{steps}, Episode Reward: {episode_reward:.2f}")
            logger.info(f"  Total Reward: {summary['total_reward']:.2f}, Total Profit: ${summary['total_profit']:.2f}, "
                      f"Return: {summary['return_pct']:.2f}%")
            logger.info(f"  Actions: {summary['action_counts']}")
            logger.info(f"  Strategy Selections: {summary['strategy_counts']}")
            logger.info(f"  Trades: {summary['trade_count']} (Wins: {summary['win_count']}, Losses: {summary['loss_count']}, Win Rate: {summary['win_rate']:.1f}%)")
            # CRITICAL FIX: Handle None values and Unicode encoding issue
            initial_bal = summary['initial_balance'] if summary['initial_balance'] is not None else 0.0
            current_bal = summary['current_balance'] if summary['current_balance'] is not None else 0.0
            # Use ASCII arrow instead of Unicode to avoid encoding errors
            logger.info(f"  Balance: ${initial_bal:.2f} -> ${current_bal:.2f}")
            
            # Log summary to wandb
            if use_wandb and WANDB_AVAILABLE:
                strategy_counts = summary.get('strategy_counts', {})
                action_counts = summary.get('action_counts', {})
                wandb.log({
                    "stage3/step": step,
                    "stage3/episode_reward": episode_reward,
                    "stage3/total_reward": summary.get('total_reward', 0.0),
                    "stage3/total_profit": summary.get('total_profit', 0.0),
                    "stage3/return_pct": summary.get('return_pct', 0.0),
                    "stage3/balance": current_bal,
                    "stage3/trade_count": summary.get('trade_count', 0),
                    "stage3/win_count": summary.get('win_count', 0),
                    "stage3/loss_count": summary.get('loss_count', 0),
                    "stage3/win_rate": summary.get('win_rate', 0.0),
                    "stage3/strategy_TREND_FOLLOW": strategy_counts.get('TREND_FOLLOW', 0),
                    "stage3/strategy_MEAN_REVERT": strategy_counts.get('MEAN_REVERT', 0),
                    "stage3/strategy_MOMENTUM": strategy_counts.get('MOMENTUM', 0),
                    "stage3/strategy_RISK_OFF": strategy_counts.get('RISK_OFF', 0),
                    "stage3/avg_reward": summary.get('avg_reward', 0.0),
                    "stage3/avg_profit": summary.get('avg_profit', 0.0),
                })
            
            episode_reward = 0.0
            # Reset stats after logging to start fresh for next period
            stats.reset()
        
        if done or truncated:
            state, _ = meta_env.reset()
            # Stats already reset after logging above, but reset again for safety
            stats.reset()
    
    # Log final statistics
    stats.log_summary(prefix="Meta-Strategy")
    logger.info("Meta-strategy agent training complete!")
    return meta_agent


# ==============================================================================
# STAGE 4: END-TO-END FINE-TUNING
# ==============================================================================

def fine_tune_end_to_end(meta_agent: MetaStrategyAgent,
                         execution_agents: Dict[str, nn.Module],
                         preprocessor: MultiTimescalePreprocessor,
                         regime_detector: EnhancedRegimeDetector,
                         mode: str = "stock",
                         steps: int = 10000,
                         lr: float = 0.003,  # 3e-3: 10x increase from 3e-4 - Agent needs bigger updates to change its mind
                         window_size: int = 30,
                         logger: Optional[logging.Logger] = None,
                         use_wandb: bool = True):
    """
    Fine-tune entire system end-to-end.
    
    All agents are trainable.
    """
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("STAGE 4: End-to-End Fine-Tuning")
    logger.info("=" * 60)
    
    # Unfreeze execution agents
    for agent in execution_agents.values():
        for param in agent.parameters():
            param.requires_grad = True
    
    # Create optimizers
    meta_optimizer = optim.Adam(meta_agent.parameters(), lr=lr)  # 3e-3: 10x boost for bigger updates
    exec_optimizers = {}
    for strategy, agent in execution_agents.items():
        exec_optimizers[strategy] = optim.Adam(agent.parameters(), lr=lr)  # Same LR as meta (3e-3) - 10x boost for bigger updates
    
    # Create environment
    config = TradingConfig()
    config.mode = mode
    # TEMPORARY: Set fees to 0.0% for the next 50 steps (fine-tuning uses steps, not epochs)
    # This removes transaction cost friction and allows agent to explore more freely
    FEE_FREE_STEPS_FT = 50
    original_commission_ft = config.commission_rate
    original_slippage_ft = config.slippage_bps
    config.commission_rate = 0.0  # 0.0% commission
    config.slippage_bps = 0.0  # 0.0% slippage
    logger.info(f"[FEE-FREE MODE] Setting fees to 0.0% for first {FEE_FREE_STEPS_FT} steps")
    logger.info(f"  Original: commission={original_commission_ft:.4f} ({original_commission_ft*100:.2f}%), slippage={original_slippage_ft:.1f} bps")
    logger.info(f"  Temporary: commission=0.0 (0.0%), slippage=0.0 bps")
    
    meta_env = MetaStrategyEnv(
        ticker=preprocessor.ticker,
        window_size=window_size,
        config=config,
        execution_horizon=10,
        execution_agents=execution_agents,  # Pass execution agents
        logger=logger
    )
    
    logger.info(f"Fine-tuning for {steps} steps...")
    
    # Initialize statistics tracker
    stats = TrainingStats(logger)
    
    # Fetch data once
    aligned_data, features = preprocessor.process(window_size=window_size)
    if not aligned_data or not features:
        logger.error("Failed to fetch data for fine-tuning")
        return
    
    state, _ = meta_env.reset()
    episode_reward = 0.0
    
    memory_states = []
    memory_actions = []
    memory_logprobs = []
    memory_rewards = []
    
    for step in range(1, steps + 1):
        # RESTORE FEES after 50 steps (if we're past the fee-free period)
        if step == FEE_FREE_STEPS_FT + 1:
            config.commission_rate = original_commission_ft
            config.slippage_bps = original_slippage_ft
            logger.info(f"[FEE RESTORED] Step {step}: Restoring original fees")
            logger.info(f"  Commission: 0.0 -> {original_commission_ft:.4f} ({original_commission_ft*100:.2f}%)")
            logger.info(f"  Slippage: 0.0 -> {original_slippage_ft:.1f} bps")
        
        # ENTROPY ANNEALING: Calculate entropy coefficient for this step
        # BOOST ENTROPY: Increased from 0.0001 to 0.01 to force exploration
        # CRITICAL FIX: Faster exponential decay to force agent to use brain instead of random guessing
        # SLOW ENTROPY DECAY: Keep agent curious for longer (until step 10000+)
        # Entropy coefficient: Start at 0.01 (boosted from 0.0001), decay slowly using 0.99 per step
        # Use slower decay (0.99 per step) so entropy stays above zero for 100+ epochs
        entropy_coef = 0.01 * (0.99 ** step)  # Slow exponential decay (0.99 per step), boosted 100x from 0.0001
        entropy_coef = max(entropy_coef, 0.001)  # Clamp to minimum (boosted 100x from 0.00001)
        progress = step / steps  # For logging only
        # Get windowed features
        windowed = preprocessor.get_window_features(features, window_size, None)
        if not windowed:
            state, _ = meta_env.reset()
            continue
        
        # Prepare features dict (FIX: Use full window, not just last element)
        features_dict = {}
        for interval, feat_array in windowed.items():
            if len(feat_array) > 0:
                # FIX: feat_array should already be (window_size, features), ensure it's 3D
                if isinstance(feat_array, np.ndarray):
                    if feat_array.ndim == 2:
                        # (window_size, features) -> (1, window_size, features)
                        feat_tensor = torch.from_numpy(feat_array).float().unsqueeze(0).to(device)
                    elif feat_array.ndim == 1:
                        # (features,) -> (1, 1, features) - shouldn't happen but handle it
                        feat_tensor = torch.from_numpy(feat_array).float().unsqueeze(0).unsqueeze(0).to(device)
                    else:
                        # Already 3D or other, convert directly
                        feat_tensor = torch.from_numpy(feat_array).float().to(device)
                        if feat_tensor.dim() == 2:
                            feat_tensor = feat_tensor.unsqueeze(0)
                else:
                    feat_tensor = torch.FloatTensor(feat_array).unsqueeze(0).to(device)
                features_dict[interval.replace("1wk", "1w")] = feat_tensor
        
        # Get regime
        if aligned_data:
            regime_key = "1d" if "1d" in aligned_data else list(aligned_data.keys())[0]
            regime_df = aligned_data[regime_key]
            if not regime_df.empty:
                # Extract Close prices for regime detection
                close_prices = regime_df['Close'].values if 'Close' in regime_df.columns else regime_df.iloc[:, 0].values
                regime_features = torch.FloatTensor(regime_detector.get_regime_features(close_prices)).unsqueeze(0).to(device)
            else:
                regime_features = torch.zeros(1, 4, dtype=torch.float32).to(device)
        else:
            regime_features = torch.zeros(1, 4, dtype=torch.float32).to(device)
        
        # Get logits for debugging before calling act()
        logits_debug, _ = meta_agent.forward(features_dict, regime_features)
        
        # DEBUG: Print logits every 100 steps to check if agent is making decisions
        if step % 100 == 0:
            logits_np = logits_debug[0].detach().cpu().numpy()
            logger.info(f"Step {step}: DEBUG LOGITS (End-to-End Meta): {logits_np}")
            # Check if logits are all identical (dead network)
            if len(logits_np) > 1 and np.allclose(logits_np, logits_np[0], atol=1e-6):
                logger.warning(f"  ⚠️  WARNING: All logits are identical! Network may be dead. Logits: {logits_np}")
        
        # Get action from meta agent
        action, log_prob, value = meta_agent.act(features_dict, regime_features, deterministic=False)
        
        # Step environment
        next_state, reward, done, truncated, info = meta_env.step(action)
        
        # Clip reward
        reward = np.clip(reward, -500.0, 500.0)  # Match environment clipping [-500, +500] (after REWARD_SCALE=10.0)
        
        # Update statistics
        strategy_names = {0: "TREND_FOLLOW", 1: "MEAN_REVERT", 2: "MOMENTUM", 3: "RISK_OFF"}
        strategy_name = strategy_names.get(action, f"STRATEGY_{action}")
        if 'balance' not in info:
            if hasattr(meta_env, 'current_balance'):
                info['balance'] = meta_env.current_balance
            if hasattr(meta_env, 'initial_balance'):
                info['initial_balance'] = meta_env.initial_balance
        stats.update(reward, info, action=action, strategy=strategy_name)
        
        # Store in memory (FIX: Ensure correct dtypes and shapes)
        memory_states.append((features_dict, regime_features))
        memory_actions.append(torch.tensor(action, dtype=torch.long))
        # Ensure log_prob is always a 1D tensor with shape [1] for stacking
        if isinstance(log_prob, torch.Tensor):
            if log_prob.dim() == 0:  # Scalar
                log_prob_tensor = log_prob.unsqueeze(0)
            else:
                log_prob_tensor = log_prob.flatten()[:1]  # Take first element and ensure 1D
        else:
            log_prob_tensor = torch.tensor([log_prob], dtype=torch.float32)
        memory_logprobs.append(log_prob_tensor)
        memory_rewards.append(reward)
        
        state = next_state
        episode_reward += reward
        
        # Update with larger batch size for better GPU utilization
        # Increased batch size from implicit 500 to 2000 (4x) for Stage 4
        PPO_BATCH_SIZE_STAGE4 = 2000
        if step % 500 == 0 and len(memory_rewards) >= PPO_BATCH_SIZE_STAGE4:
            if len(memory_rewards) > 0:
                # Calculate discounted rewards
                rewards = []
                discounted_reward = 0
                for r in reversed(memory_rewards):
                    discounted_reward = r + 0.99 * discounted_reward
                    rewards.insert(0, discounted_reward)
                
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
                
                # Update meta agent
                # Ensure all tensors have same shape before stacking
                memory_actions_ready = []
                for a in memory_actions:
                    if isinstance(a, torch.Tensor):
                        if a.dim() == 0:
                            memory_actions_ready.append(a.unsqueeze(0))
                        else:
                            memory_actions_ready.append(a.flatten()[:1])
                    else:
                        memory_actions_ready.append(torch.tensor([a], dtype=torch.long))
                
                memory_logprobs_ready = []
                for lp in memory_logprobs:
                    if isinstance(lp, torch.Tensor):
                        if lp.dim() == 0:  # Scalar
                            memory_logprobs_ready.append(lp.unsqueeze(0))
                        else:
                            memory_logprobs_ready.append(lp.flatten()[:1])  # Ensure 1D
                    else:
                        memory_logprobs_ready.append(torch.tensor([lp], dtype=torch.float32))
                
                old_actions = torch.stack(memory_actions_ready).to(device).squeeze()
                old_logprobs = torch.stack(memory_logprobs_ready).to(device).squeeze()
                
                for _ in range(3):
                    logprobs_list = []
                    values_list = []
                    entropies_list = []
                    
                    for (feat_dict, reg_feat), old_action in zip(memory_states, old_actions):
                        # FIX: Ensure all tensors are float32 before evaluation
                        feat_dict_ready = {}
                        for key, val in feat_dict.items():
                            if isinstance(val, torch.Tensor):
                                feat_dict_ready[key] = val.float() if val.dtype != torch.float32 else val
                            else:
                                feat_dict_ready[key] = torch.tensor(val, dtype=torch.float32, device=device)
                        
                        reg_feat_ready = reg_feat.float() if reg_feat.dtype != torch.float32 else reg_feat
                        old_action_ready = old_action.long() if old_action.dtype != torch.long else old_action
                        
                        has_nan = False
                        for feat_tensor in feat_dict_ready.values():
                            if torch.isnan(feat_tensor).any() or torch.isinf(feat_tensor).any():
                                has_nan = True
                                break
                        if torch.isnan(reg_feat_ready).any() or torch.isinf(reg_feat_ready).any():
                            has_nan = True
                        
                        if has_nan:
                            continue
                        
                        try:
                            logprobs, values, entropy = meta_agent.evaluate(feat_dict_ready, reg_feat_ready, old_action_ready.unsqueeze(0))
                        except RuntimeError as e:
                            if "dtype" in str(e) or "Double" in str(e) or "Float" in str(e):
                                logger.error(f"Error in re-evaluation at idx {len(logprobs_list)}: {e}, skipping this sample")
                                continue
                            else:
                                raise
                        logprobs_list.append(logprobs)
                        values_list.append(values)
                        entropies_list.append(entropy)
                    
                    if len(logprobs_list) == 0:
                        continue
                    
                    logprobs = torch.stack(logprobs_list).squeeze()
                    values = torch.stack(values_list).squeeze()
                    entropy = torch.stack(entropies_list).mean()
                    
                    if torch.isnan(logprobs).any() or torch.isnan(values).any():
                        continue
                    
                    # PPO loss
                    ratios = torch.exp(logprobs - old_logprobs.detach())
                    advantages = rewards - values.detach()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = 0.5 * nn.MSELoss()(values, rewards)
                    entropy_loss_term = entropy_coef * entropy
                    
                    # CRITICAL DEBUG: Check for entropy domination (Stage 4)
                    actor_loss_val = actor_loss.item()
                    entropy_loss_val = entropy_loss_term.item()
                    ratio = entropy_loss_val / (actor_loss_val + 1e-8)
                    
                    if ratio > 10.0 and step % 1000 == 0:  # Log occasionally
                        logger.warning(f"[WARNING] ENTROPY DOMINATION (Stage 4): entropy_loss ({entropy_loss_val:.6f}) is {ratio:.1f}x larger than actor_loss ({actor_loss_val:.6f})!")
                    
                    # ENTROPY ANNEALING: Use decaying entropy coefficient (calculated above)
                    # Starts at 0.01 for exploration, decays to 0.001 for convergence
                    loss = actor_loss + critic_loss - entropy_loss_term
                    
                    if torch.isnan(loss).any():
                        continue
                    
                    meta_optimizer.zero_grad()
                    loss.mean().backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(meta_agent.parameters(), max_norm=0.5)
                    meta_optimizer.step()
                    
                    # Log to wandb (every PPO update)
                    if use_wandb and WANDB_AVAILABLE:
                        # Calculate advantages for logging
                        advantages_log = rewards - values.detach()
                        wandb.log({
                            "stage4/step": step,
                            "stage4/actor_loss": actor_loss_val,
                            "stage4/critic_loss": critic_loss.item(),
                            "stage4/entropy": entropy.item(),
                            "stage4/entropy_coef": entropy_coef,
                            "stage4/entropy_loss": entropy_loss_val,
                            "stage4/total_loss": loss.mean().item(),
                            "stage4/grad_norm": grad_norm.item(),
                            "stage4/entropy_actor_ratio": ratio,
                            "stage4/advantage_mean": advantages_log.mean().item() if len(advantages_log) > 0 else 0.0,
                            "stage4/advantage_std": advantages_log.std().item() if len(advantages_log) > 0 else 0.0,
                            "stage4/reward_mean": rewards.mean().item(),
                            "stage4/reward_std": rewards.std().item(),
                        })
                
                # Clear memory
                memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
            
            # Get summary statistics (accumulated since last reset)
            summary = stats.get_summary()
            logger.info(f"Step {step}/{steps}, Episode Reward: {episode_reward:.2f}")
            logger.info(f"  Total Reward: {summary['total_reward']:.2f}, Total Profit: ${summary['total_profit']:.2f}, "
                      f"Return: {summary['return_pct']:.2f}%")
            logger.info(f"  Actions: {summary['action_counts']}")
            logger.info(f"  Strategy Selections: {summary['strategy_counts']}")
            logger.info(f"  Trades: {summary['trade_count']} (Wins: {summary['win_count']}, Losses: {summary['loss_count']}, Win Rate: {summary['win_rate']:.1f}%)")
            # CRITICAL FIX: Handle None values and Unicode encoding issue
            initial_bal = summary['initial_balance'] if summary['initial_balance'] is not None else 0.0
            current_bal = summary['current_balance'] if summary['current_balance'] is not None else 0.0
            # Use ASCII arrow instead of Unicode to avoid encoding errors
            logger.info(f"  Balance: ${initial_bal:.2f} -> ${current_bal:.2f}")
            
            # Log summary to wandb
            if use_wandb and WANDB_AVAILABLE:
                strategy_counts = summary.get('strategy_counts', {})
                action_counts = summary.get('action_counts', {})
                wandb.log({
                    "stage4/step": step,
                    "stage4/episode_reward": episode_reward,
                    "stage4/total_reward": summary.get('total_reward', 0.0),
                    "stage4/total_profit": summary.get('total_profit', 0.0),
                    "stage4/return_pct": summary.get('return_pct', 0.0),
                    "stage4/balance": current_bal,
                    "stage4/trade_count": summary.get('trade_count', 0),
                    "stage4/win_count": summary.get('win_count', 0),
                    "stage4/loss_count": summary.get('loss_count', 0),
                    "stage4/win_rate": summary.get('win_rate', 0.0),
                    "stage4/strategy_TREND_FOLLOW": strategy_counts.get('TREND_FOLLOW', 0),
                    "stage4/strategy_MEAN_REVERT": strategy_counts.get('MEAN_REVERT', 0),
                    "stage4/strategy_MOMENTUM": strategy_counts.get('MOMENTUM', 0),
                    "stage4/strategy_RISK_OFF": strategy_counts.get('RISK_OFF', 0),
                    "stage4/avg_reward": summary.get('avg_reward', 0.0),
                    "stage4/avg_profit": summary.get('avg_profit', 0.0),
                })
            
            episode_reward = 0.0
            # Reset stats after logging to start fresh for next period
            stats.reset()
        
        if done or truncated:
            state, _ = meta_env.reset()
            # Stats already reset after logging above, but reset again for safety
            stats.reset()
    
    # Log final statistics
    stats.log_summary(prefix="End-to-End")
    logger.info("End-to-end fine-tuning complete!")


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================

def train_hierarchical_transformer(ticker: str = "^IXIC",
                                   mode: str = "stock",
                                   window_size: int = 30,
                                   save_dir: str = "models_v0.2",
                                   logger: Optional[logging.Logger] = None,
                                   use_wandb: bool = True,
                                   wandb_project: str = "hierarchical-transformer-trading",
                                   wandb_entity: Optional[str] = None):
    """
    Complete training pipeline with curriculum learning.
    
    Args:
        ticker: Stock/crypto ticker symbol
        mode: "stock" or "crypto"
        window_size: Input window size
        save_dir: Directory to save models
        logger: Optional logger instance
        use_wandb: Whether to use Weights & Biases for tracking
        wandb_project: W&B project name
        wandb_entity: W&B entity/team name (optional)
    """
    logger = logger or logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("HIERARCHICAL TRANSFORMER TRAINING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Ticker: {ticker}")
    logger.info(f"Mode: {mode.upper()}")
    logger.info(f"Window Size: {window_size}")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)
    
    # Initialize Weights & Biases
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            config={
                "ticker": ticker,
                "mode": mode,
                "window_size": window_size,
                "device": str(device),
                "save_dir": save_dir,
            },
            name=f"{ticker}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        # Define metrics for each stage (wandb handles grouping automatically)
        # Note: We don't need wildcards - wandb groups metrics by prefix automatically
        wandb.define_metric("stage1/epoch")
        wandb.define_metric("stage3/step")
        wandb.define_metric("stage4/step")
        logger.info(f"Initialized Weights & Biases: project={wandb_project}")
    elif use_wandb and not WANDB_AVAILABLE:
        logger.warning("wandb requested but not installed. Install with: pip install wandb")
        use_wandb = False
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize components
    preprocessor = MultiTimescalePreprocessor(ticker, mode=mode, logger=logger)
    regime_detector = EnhancedRegimeDetector(window=20, logger=logger)
    
    # Stage 1: Pre-train transformers
    transformer = pretrain_transformer_encoders(
        preprocessor, mode=mode, epochs=100, window_size=window_size, logger=logger, use_wandb=use_wandb
    )
    torch.save(transformer.state_dict(), os.path.join(save_dir, "transformer_pretrained.pth"))
    logger.info("Saved pre-trained transformer")
    
    # Stage 2: Pre-train execution agents
    execution_agents = pretrain_execution_agents(
        transformer, preprocessor, regime_detector, mode=mode, 
        epochs_per_agent=500, steps_per_epoch=2048, window_size=window_size, logger=logger, use_wandb=use_wandb
    )
    for strategy, agent in execution_agents.items():
        torch.save(agent.state_dict(), os.path.join(save_dir, f"execution_{strategy.lower()}.pth"))
    logger.info("Saved pre-trained execution agents")
    
    # Stage 3: Train meta-strategy
    meta_agent = train_meta_strategy_agent(
        transformer, execution_agents, preprocessor, regime_detector,
        mode=mode, steps=100000, window_size=window_size, logger=logger, use_wandb=use_wandb
    )
    torch.save(meta_agent.state_dict(), os.path.join(save_dir, "meta_strategy.pth"))
    logger.info("Saved meta-strategy agent")
    
    # Stage 4: End-to-end fine-tuning
    fine_tune_end_to_end(
        meta_agent, execution_agents, preprocessor, regime_detector,
        mode=mode, steps=5000, window_size=window_size, logger=logger, use_wandb=use_wandb
    )
    
    # Save final models
    torch.save(meta_agent.state_dict(), os.path.join(save_dir, "meta_strategy_final.pth"))
    for strategy, agent in execution_agents.items():
        torch.save(agent.state_dict(), os.path.join(save_dir, f"execution_{strategy.lower()}_final.pth"))
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Hierarchical Transformer Trading System")
    parser.add_argument("--ticker", type=str, default="^IXIC", help="Ticker symbol")
    parser.add_argument("--mode", type=str, default="stock", choices=["stock", "crypto"], help="Trading mode")
    parser.add_argument("--window-size", type=int, default=30, help="Window size")
    parser.add_argument("--save-dir", type=str, default="models_v0.2", help="Save directory")
    parser.add_argument("--use-wandb", action="store_true", default=True, help="Use Weights & Biases for tracking")
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false", help="Disable Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="hierarchical-transformer-trading", help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team name (optional)")
    
    args = parser.parse_args()
    
    # Setup logging with file handler for repetitive warnings
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Main logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler for ALL logs (including repetitive warnings)
    log_file = f"training_{args.ticker.replace('^', '').replace('-', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Stream handler (terminal) - filter out repetitive warnings
    class RepetitiveWarningFilter(logging.Filter):
        """Filter to send repetitive warnings to file only, not terminal"""
        REPETITIVE_PATTERNS = [
            'WHIPSAW PENALTY',
            'STUPIDITY PENALTY',
            'ENTRY FEE',
            'QUICK WIN BONUS',
            'Take profit triggered',
            'Stop loss triggered',
        ]
        
        def filter(self, record):
            # Allow all important warnings (like ENTROPY DOMINATION) to terminal
            message = record.getMessage()
            for pattern in self.REPETITIVE_PATTERNS:
                if pattern in message:
                    # This is repetitive - don't show in terminal
                    return False
            # Important logs - show in terminal
            return True
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(RepetitiveWarningFilter())  # Filter repetitive warnings
    logger.addHandler(stream_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Repetitive warnings saved to: {log_file}")
    
    # Train
    train_hierarchical_transformer(
        ticker=args.ticker,
        mode=args.mode,
        window_size=args.window_size,
        save_dir=args.save_dir,
        logger=logger,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity
    )

