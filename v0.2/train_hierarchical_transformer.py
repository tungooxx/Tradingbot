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
import matplotlib
matplotlib.use('Agg')  # Switch to non-interactive backend for thread safety
import matplotlib.pyplot as plt
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
                            multi_stock_tickers: Optional[List[str]] = None,
                            logger: Optional[logging.Logger] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Prepare data for supervised pre-training (price prediction).
    
    IMPROVEMENT: Multi-stock support for Stage 1
    - Stage 1: Multi-stock (learns general market dynamics)
    - Stages 2-4: Single-stock (specialized training)
    
    Args:
        preprocessor: Single-stock preprocessor (for backward compatibility)
        window_size: Lookback window size
        multi_stock_tickers: List of tickers for multi-stock training (Stage 1 only)
        logger: Optional logger
    
    Returns:
        X_dict: Dictionary of input sequences by timescale
        y_dict: Dictionary of target prices by timescale
    """
    from typing import List
    
    # STAGE 1 IMPROVEMENT: Multi-stock training for general market dynamics
    if multi_stock_tickers and len(multi_stock_tickers) > 1:
        logger.info(f"Multi-stock pre-training: Loading {len(multi_stock_tickers)} stocks")
        logger.info(f"  Tickers: {multi_stock_tickers}")
        logger.info(f"  IMPROVEMENT: Using percentage changes (scale-invariant) for multi-stock compatibility")
        logger.info(f"  This ensures $150 stock and $2000 stock are on same scale")
        
        all_X_dict = {}
        all_y_dict = {}
        
        # Process each stock
        for ticker in multi_stock_tickers:
            try:
                # Create preprocessor for this ticker
                stock_preprocessor = MultiTimescalePreprocessor(ticker, mode=preprocessor.mode, logger=logger)
                aligned_data, features = stock_preprocessor.process(window_size=window_size)
                
                if not aligned_data or not features:
                    logger.warning(f"Skipping {ticker}: Failed to process data")
                    continue
                
                # Process each interval
                for interval, feat_array in features.items():
                    if len(feat_array) < window_size + 1:
                        continue
                    
                    if interval not in all_X_dict:
                        all_X_dict[interval] = []
                        all_y_dict[interval] = []
                    
                    X_list = []
                    y_list = []
                    
                    # Create sequences: predict next period's log return
                    # IMPROVEMENT: Use log returns (scale-invariant) for multi-stock compatibility
                    # Log returns work across different price levels ($150 vs $2000)
                    for i in range(window_size, len(feat_array) - 1):
                        window = feat_array[i - window_size:i]  # (window_size, features)
                        
                        # Target: Next period's log return (feature index 2)
                        # Log returns are scale-invariant: log($2000/$1900) ≈ log($200/$190)
                        # This ensures $150 stock and $2000 stock are on same scale
                        if feat_array.shape[1] > 2:
                            target = feat_array[i + 1, 2]  # Log return (feature index 2)
                        else:
                            # Fallback: calculate from normalized prices if log return not available
                            current_price = feat_array[i, 0]
                            next_price = feat_array[i + 1, 0]
                            if abs(current_price) > 1e-7:
                                target = (next_price - current_price) / (abs(current_price) + 1e-7)
                            else:
                                target = 0.0
                        
                        # Check for NaN/Inf
                        if np.isnan(window).any() or np.isinf(window).any():
                            continue
                        if np.isnan(target) or np.isinf(target):
                            continue
                        
                        X_list.append(window)
                        y_list.append(target)
                    
                    if X_list:
                        all_X_dict[interval].extend(X_list)
                        all_y_dict[interval].extend(y_list)
                        logger.info(f"  {ticker} ({interval}): Added {len(X_list)} samples")
            
            except Exception as e:
                logger.warning(f"Failed to process {ticker}: {e}")
                continue
        
        # Combine all stocks and normalize across all stocks
        X_dict = {}
        y_dict = {}
        
        for interval in all_X_dict.keys():
            if len(all_X_dict[interval]) == 0:
                continue
            
            X_array = np.array(all_X_dict[interval], dtype=np.float32)
            y_array = np.array(all_y_dict[interval], dtype=np.float32).reshape(-1, 1)
            
            # Final NaN check
            if np.isnan(X_array).any() or np.isnan(y_array).any():
                if logger:
                    logger.warning(f"Skipping {interval} due to NaN in processed data")
                continue
            
            # IMPROVEMENT: Normalize across ALL stocks (not per-stock)
            # This ensures $150 stock and $2000 stock are on same scale
            # X_array shape: (total_samples, window_size, features)
            for feat_idx in range(X_array.shape[2]):
                feat_data = X_array[:, :, feat_idx]
                feat_mean = feat_data.mean()  # Mean across ALL stocks
                feat_std = feat_data.std() + 1e-7  # Std across ALL stocks
                X_array[:, :, feat_idx] = (feat_data - feat_mean) / feat_std
            
            # Normalize targets across all stocks
            y_mean = y_array.mean()
            y_std = y_array.std() + 1e-7
            y_array = (y_array - y_mean) / y_std
            
            # Clip extreme values
            X_array = np.clip(X_array, -10, 10)
            y_array = np.clip(y_array, -10, 10)
            
            X_dict[interval] = X_array
            y_dict[interval] = y_array
            
            logger.info(f"  Combined {interval}: {len(X_array)} total samples from {len(multi_stock_tickers)} stocks")
        
        return X_dict, y_dict
    
    else:
        # Single-stock mode (backward compatibility, used in Stages 2-4)
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
                                  batch_size: int = 512,  # Increased for better GPU utilization (reduce CPU overhead)
                                  lr: float = 0.0001,
                                  window_size: int = 30,
                                  multi_stock_tickers: Optional[List[str]] = None,
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
    X_dict, y_dict = prepare_pretraining_data(preprocessor, window_size, multi_stock_tickers, logger)
    
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
    
    # Split data into train/validation sets (80/20 split)
    logger.info("Splitting data into train/validation sets (80/20)...")
    X_train_dict = {}
    y_train_dict = {}
    X_val_dict = {}
    y_val_dict = {}
    
    for interval, X in X_dict.items():
        y = y_dict[interval]
        n_samples = len(X)
        split_idx = int(0.8 * n_samples)
        
        X_train_dict[interval] = X[:split_idx]
        y_train_dict[interval] = y[:split_idx]
        X_val_dict[interval] = X[split_idx:]
        y_val_dict[interval] = y[split_idx:]
        
        logger.info(f"  {interval}: {len(X_train_dict[interval])} train, {len(X_val_dict[interval])} val")
    
    # Training loop
    logger.info(f"Training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        transformer.train()
        total_train_loss = 0.0
        num_train_batches = 0
        interval_train_losses = {}  # Track losses per interval for wandb
        
        # Train on each timescale
        for interval, X in X_train_dict.items():
            y = y_train_dict[interval]
            interval_train_loss = 0.0
            interval_train_batches = 0
            
            # OPTIMIZATION: Pre-convert to tensors to reduce CPU overhead
            # Use num_workers=0 for small datasets or when CPU is bottleneck
            # For large datasets, num_workers=2 is better than 4 to reduce CPU load
            X_tensor = torch.from_numpy(X).float()
            y_tensor = torch.from_numpy(y).float()
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=0,  # Set to 0 to reduce CPU overhead - data is already in memory
                pin_memory=True if device.type == 'cuda' else False  # Only pin if using GPU
            )
            
            for batch_X, batch_y in dataloader:
                # Use non_blocking transfer for faster CPU->GPU
                batch_X = batch_X.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
                
                # Move NaN checks to GPU (faster than CPU checks)
                # Only check if we suspect issues (skip for performance)
                # if torch.isnan(batch_X).any() or torch.isnan(batch_y).any():
                #     logger.warning(f"Skipping batch with NaN values in {interval}")
                #     continue
                
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
                        # Only log occasionally to reduce CPU overhead
                        if num_train_batches % 100 == 0:
                            logger.warning(f"Key '{transformer_key}' not found in encoded dict. Available keys: {list(encoded.keys())}")
                        continue
                
                last_hidden = encoded[transformer_key][:, -1, :]  # (batch, d_model)
                
                # Predict
                pred = prediction_heads[interval](last_hidden)
                
                # Loss
                loss = criterion(pred, batch_y)
                
                # Only check for NaN/Inf occasionally (reduce CPU overhead)
                # Most batches are fine, so we can check less frequently
                if num_train_batches % 50 == 0:
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.warning(f"NaN/Inf loss for {interval}, skipping batch")
                        continue
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN gradients (only occasionally to reduce CPU overhead)
                if num_train_batches % 100 == 0:
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
                
                total_train_loss += loss.item()
                interval_train_loss += loss.item()
                num_train_batches += 1
                interval_train_batches += 1
            
            # Store interval train loss
            if interval_train_batches > 0:
                interval_train_losses[interval] = interval_train_loss / interval_train_batches
        
        # Validation phase
        transformer.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        interval_val_losses = {}
        
        with torch.no_grad():
            for interval, X_val in X_val_dict.items():
                y_val = y_val_dict[interval]
                interval_val_loss = 0.0
                interval_val_batches = 0
                
                # Create validation dataloader (optimized for GPU)
                X_val_tensor = torch.from_numpy(X_val).float()
                y_val_tensor = torch.from_numpy(y_val).float()
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                val_dataloader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,  # Set to 0 to reduce CPU overhead
                    pin_memory=True if device.type == 'cuda' else False
                )
                
                for batch_X, batch_y in val_dataloader:
                    # Use non_blocking transfer for faster CPU->GPU
                    batch_X = batch_X.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    
                    # Forward pass
                    transformer_key = interval.replace("1wk", "1w") if interval == "1wk" else interval
                    features_dict = {transformer_key: batch_X}
                    encoded = transformer(features_dict)
                    
                    # Get last hidden state
                    if transformer_key not in encoded:
                        if interval in encoded:
                            transformer_key = interval
                        elif "1w" in encoded and transformer_key == "1wk":
                            transformer_key = "1w"
                        else:
                            continue
                    
                    last_hidden = encoded[transformer_key][:, -1, :]
                    
                    # Predict
                    pred = prediction_heads[interval](last_hidden)
                    
                    # Loss
                    loss = criterion(pred, batch_y)
                    
                    total_val_loss += loss.item()
                    interval_val_loss += loss.item()
                    num_val_batches += 1
                    interval_val_batches += 1
                
                # Store interval validation loss
                if interval_val_batches > 0:
                    interval_val_losses[interval] = interval_val_loss / interval_val_batches
        
        # Calculate average losses
        avg_train_loss = total_train_loss / num_train_batches if num_train_batches > 0 else 0
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else 0
        
        # Log every epoch (or every 10 epochs for less verbose output)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # Log to wandb every epoch
        if use_wandb and WANDB_AVAILABLE:
            log_dict = {
                "stage1/epoch": epoch + 1,
                "stage1/train_loss": avg_train_loss,
                "stage1/val_loss": avg_val_loss,
            }
            # Add per-interval losses
            for interval, loss_val in interval_train_losses.items():
                log_dict[f"stage1/train_loss_{interval}"] = loss_val
            for interval, loss_val in interval_val_losses.items():
                log_dict[f"stage1/val_loss_{interval}"] = loss_val
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
        # ENTROPY COEFFICIENT: BOOSTED to force exploration and break "HOLD" loop
        # Increased significantly to make agent try different actions
        ENTROPY_COEF_START = 0.1  # Force 10x more randomness initially (Updated from 0.05)
        ENTROPY_COEF_MIN = 0.01    # Sustain minimum exploration
        ENTROPY_COEF_DECAY = 0.99
        ENTROPY_COEF_MAX = 0.05  # Increased to allow effective "Panic Button" boost
        entropy_coef = ENTROPY_COEF_START
        
        for epoch in range(epochs_per_agent):
            # FEE-FREE TRAINING LOGIC: Restore fees after FEE_FREE_EPOCHS
            if epoch == FEE_FREE_EPOCHS:
                config.commission_rate = original_commission
                config.slippage_bps = original_slippage
                logger.info(f"  Epoch {epoch}: [FEE-FREE ENDED] Restoring original fees:")
                logger.info(f"    commission={config.commission_rate:.4f}, slippage={config.slippage_bps:.1f} bps")
            
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
            
            # Track action diversity and entropy boost state
            action_history = []  # Track last 100 actions to detect collapse
            ACTION_HISTORY_SIZE = 100
            action_collapse_active = False  # Flag for entropy boost "Panic Button"
            
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
                    
                    # ---------------------------------------------------------------------
                    # FIXED ACTION SELECTION LOGIC
                    # ---------------------------------------------------------------------
                    
                    # FIX 1: Prevent Saturation (The "Clamp" Bug)
                    # Normalize logits so the max is 0. This preserves relative differences 
                    # (Softmax is translation invariant) but prevents hitting the clamp ceiling.
                    # Example: [62, 40, 36] -> [0, -22, -26]
                    logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
                    
                    # Now we can safely clamp to avoid -inf issues without destroying signal
                    logits = torch.clamp(logits, min=-50.0, max=50.0)
                    
                    # FIX 2: Apply Action Masking (The "Missing Mask" Bug)
                    # local_feat[:, 0] is position_status (1.0 = Held, 0.0 = Cash)
                    if local_feat is not None:
                        # Use a sufficiently large negative number, but safe for float16/32
                        MASK_VALUE = -1e9 
                        
                        # Extract position status (batch_size, )
                        pos_status = local_feat[:, 0]
                        
                        # Apply mask to the logits tensor directly
                        # If Held (pos > 0.5): Mask BUY (index 1)
                        # If Cash (pos < 0.5): Mask SELL (index 2)
                        
                        # Vectorized masking (faster and cleaner)
                        # Create boolean masks
                        has_position = pos_status > 0.5
                        no_position = ~has_position
                        
                        # Apply mask: logits[row, col] = -1e9
                        # Note: We must clone to avoid in-place operations if needed for gradient, 
                        # but here we are just sampling actions, so in-place is fine for 'logits' 
                        # as long as we use 'original_logits' for loss calculation if needed.
                        # (Standard PPO uses the *old* log_probs, new pass calculates fresh logits)
                        
                        # Mask BUY (Index 1) where we have position
                        logits[has_position, 1] = MASK_VALUE
                        
                        # Mask SELL (Index 2) where we have NO position
                        logits[no_position, 2] = MASK_VALUE

                    # FIX 3: Temperature Scaling & Distribution
                    # Apply temperature to the *masked* logits
                    temperature = getattr(agent, "temperature", 1.0)
                    if agent.training: 
                        temperature = max(temperature, 1.5) # Raised from 0.5 to 1.5 to maintain curiousity
                    used_logits = logits / temperature
                    
                    # FIX 4: Probability Floor (ε-greedy) "The Defibrillator"
                    # We force a 5% minimum probability for every VALID action to break deadlocks
                    probs = F.softmax(used_logits, dim=-1)
                    if agent.training:
                        epsilon = 0.05  # 5% floor
                        # Identify valid actions (those not masked with -1e9)
                        valid_mask = (logits > -1e8).float()
                        num_valid = valid_mask.sum(dim=-1, keepdim=True)
                        # Create uniform distribution across valid actions
                        uniform_probs = valid_mask / (num_valid + 1e-9)
                        # Mix network probabilities with uniform exploration
                        probs = (1.0 - epsilon) * probs + epsilon * uniform_probs
                    
                    # Create distribution from mixed probabilities
                    dist = torch.distributions.Categorical(probs=probs)
                    
                    # Sample action
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    # Debug Logging (Check if fixed)
                    # Debug Logging: Show ACTUAL mixed probabilities (with 5% floor)
                    if step % 100 == 0:
                         probs_np = probs[0].detach().cpu().numpy()
                         logger.info(f"  Step {step}: Mixed Probs (with floor): HOLD:{probs_np[0]:.3f}, BUY:{probs_np[1]:.3f}, SELL:{probs_np[2]:.3f}")
                    # ---------------------------------------------------------------------
                    
                    # Check for NaN/Inf in value_tensor
                    if torch.isnan(value_tensor).any() or torch.isinf(value_tensor).any():
                        logger.warning("NaN/Inf in value_tensor after model forward - model may have issues")
                        # Maintain gradient connection
                        value_tensor = value_tensor + 0.0 * torch.zeros_like(value_tensor)
                    
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
                    
                    # CRITICAL: Warn if one action dominates (>90% of recent actions)
                    if max_proportion > 0.90:
                        dominant_action = max(action_proportions, key=action_proportions.get)
                        action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
                        logger.warning(f"[WARNING] ACTION COLLAPSE DETECTED: {action_names[dominant_action]} is {max_proportion:.1%} of last {len(action_history)} actions!")
                        logger.warning(f"   Action distribution: HOLD={action_proportions[0]:.1%}, BUY={action_proportions[1]:.1%}, SELL={action_proportions[2]:.1%}")
                        logger.warning(f"   PANIC BUTTON: Activating Entropy Boost (3x) for next update!")
                        action_collapse_active = True
                    elif max_proportion < 0.70:
                        # Recovery: Reset flag if diversity improves
                        if action_collapse_active:
                             logger.info(f"   [RECOVERY] Action diversity restored ({max_proportion:.1%}), disabling Entropy Boost.")
                        action_collapse_active = False
                
                # Step environment (this doesn't need gradients)
                next_state, reward, done, truncated, info = exec_env.step(action_item)
                
                # DEBUG: Log reward breakdown for analysis (every 10 steps or when reward/profit mismatch)
                if step % 10 == 0 or (step == 1 and epoch % 10 == 0):
                    # Get reward breakdown from info if available
                    reward_breakdown = info.get('reward_breakdown', {})
                    if isinstance(reward_breakdown, dict):
                        log_return = reward_breakdown.get('log_return', 'N/A')
                        transaction_cost_ratio = reward_breakdown.get('transaction_cost_ratio', 'N/A')
                        fixed_penalty = reward_breakdown.get('fixed_penalty', 'N/A')
                        raw_reward = reward_breakdown.get('raw_reward', 'N/A')
                        scaled_reward = reward_breakdown.get('scaled_reward', 'N/A')
                        exit_reason = reward_breakdown.get('exit_reason', 'N/A')
                        bars_held = reward_breakdown.get('bars_held', 'N/A')
                        
                        logger.info(f"  Step {step}: REWARD BREAKDOWN:")
                        # Format numeric values
                        log_return_str = f"{log_return:.6f}" if isinstance(log_return, (int, float)) else str(log_return)
                        transaction_cost_str = f"{transaction_cost_ratio:.6f}" if isinstance(transaction_cost_ratio, (int, float)) else str(transaction_cost_ratio)
                        fixed_penalty_str = f"{fixed_penalty:.4f}" if isinstance(fixed_penalty, (int, float)) else str(fixed_penalty)
                        raw_reward_str = f"{raw_reward:.4f}" if isinstance(raw_reward, (int, float)) else str(raw_reward)
                        scaled_reward_str = f"{scaled_reward:.4f}" if isinstance(scaled_reward, (int, float)) else str(scaled_reward)
                        
                        logger.info(f"    log_return={log_return_str}, transaction_cost_ratio={transaction_cost_str}")
                        logger.info(f"    fixed_penalty={fixed_penalty_str}, raw_reward={raw_reward_str}, scaled={scaled_reward_str}")
                        logger.info(f"    exit_reason={exit_reason}, bars_held={bars_held}, reward={reward:.4f}")
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
                # FIX: Batch size must be clean divisor of steps_per_epoch (2048)
                # 2048 / 64 = 32 minibatches (standard and stable)
                # Old: 400 doesn't divide cleanly (2048 / 400 = 5.12, wastes data)
                PPO_BATCH_SIZE_STAGE2 = 64  # Changed from 400 to 64 for clean division
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
                    
                    # === VECTORIZED PPO UPDATE (GOOD CODE B) ===
                    # 1. Stack all inputs into tensors for parallel processing
                    keys = memory_states[0][0].keys()
                    stacked_features_dict = {}
                    for key in keys:
                        tensors = [s[0][key] if isinstance(s[0][key], torch.Tensor) else torch.tensor(s[0][key], dtype=torch.float32) for s in memory_states]
                        # Dynamically detect feature dimension from samples
                        feat_dim = tensors[0].shape[-1]
                        stacked_features_dict[key] = torch.stack(tensors).view(len(memory_states), -1, feat_dim).detach().to(device)
                    
                    strategy_feats = torch.stack([s[1] if isinstance(s[1], torch.Tensor) else torch.tensor(s[1], dtype=torch.float32) for s in memory_states]).view(len(memory_states), -1).detach().to(device)
                    precision_feats = torch.stack([s[2] if isinstance(s[2], torch.Tensor) else torch.tensor(s[2], dtype=torch.float32) for s in memory_states]).view(len(memory_states), -1).detach().to(device)
                    local_feats = torch.stack([s[3] if isinstance(s[3], torch.Tensor) else torch.tensor(s[3], dtype=torch.float32) for s in memory_states]).view(len(memory_states), -1).detach().to(device)

                    # 2. PPO Math: Calculate Advantages once (old policy Baseline)
                    advantages = rewards - old_values
                    
                    # --- WHITEBOX: ADVANTAGE HISTOGRAM ---
                    if current_global_step % 500 == 0:  # Save every 500 steps
                        plt.figure(figsize=(10, 5))
                        plt.hist(advantages.cpu().numpy(), bins=50, color='lightgreen', edgecolor='black')
                        plt.title(f"Advantage Distribution - {strategy} (Step {current_global_step})")
                        plt.xlabel("Advantage")
                        plt.ylabel("Frequency")
                        plt.grid(alpha=0.3)
                        plot_path = f"debug_advantages_stage2_{strategy.lower()}_step_{current_global_step}.png"
                        plt.savefig(plot_path)
                        plt.close()
                    
                    advantages_mean = advantages.mean()
                    advantages_std = advantages.std() + 1e-8
                    advantages = (advantages - advantages_mean) / advantages_std
                    advantages = torch.clamp(advantages, min=-5.0, max=5.0).detach()
                    
                    # 3. PPO Optimization Loop (multiple epochs for stability)
                    PPO_EPOCHS = 3
                    MINIBATCH_SIZE = 64
                    
                    for ppo_epoch in range(PPO_EPOCHS):
                        # Shuffling for this epoch
                        indices = torch.randperm(len(old_actions))
                        
                        epoch_actor_loss = 0.0
                        epoch_critic_loss = 0.0
                        epoch_entropy = 0.0
                        num_batches = 0
                        
                        for i in range(0, len(old_actions), MINIBATCH_SIZE):
                            idx = indices[i : i + MINIBATCH_SIZE]
                            if len(idx) < 2: continue # Skip tiny final batches
                            
                            # Slice Minibatch
                            b_features = {k: v[idx] for k, v in stacked_features_dict.items()}
                            b_strategy = strategy_feats[idx]
                            b_precision = precision_feats[idx]
                            b_local = local_feats[idx]
                            b_old_actions = old_actions[idx]
                            b_old_logprobs = old_logprobs[idx]
                            b_rewards = rewards[idx]
                            b_advantages = advantages[idx]

                            optimizer.zero_grad()
                            
                            # Vectorized forward pass
                            logits, values = agent(b_features, b_strategy, b_precision, b_local)
                            values = values.squeeze()
                            
                            # Normalize & Clamp logits
                            logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
                            logits = torch.clamp(logits, min=-50.0, max=50.0)
                            
                            # ACTION MASKING in PPO update (CRITICAL FIX)
                            # local_feat[:, 0] is position_status (1.0 = Held, 0.0 = Cash)
                            mask_val = -1e9
                            pos_status = b_local[:, 0]
                            has_pos = pos_status > 0.5
                            no_pos = ~has_pos
                            
                            # Apply mask to current logits
                            logits[has_pos, 1] = mask_val  # Mask BUY
                            logits[no_pos, 2] = mask_val   # Mask SELL
                            
                            # Temperature application (matching environment base logic)
                            temp = 2.5
                            used_logits = logits / temp
                            dist = torch.distributions.Categorical(logits=used_logits)
                            
                            new_logprobs = dist.log_prob(b_old_actions)
                            entropies = dist.entropy()
                            
                            # PPO Surrogate Objective
                            ratios = torch.exp(new_logprobs - b_old_logprobs)
                            surr1 = ratios * b_advantages
                            surr2 = torch.clamp(ratios, 0.8, 1.2) * b_advantages
                            actor_loss = -torch.min(surr1, surr2).mean()
                            
                            # Critic Loss
                            critic_loss = F.mse_loss(values, b_rewards)
                            
                            # Entropy Bonus
                            entropy_bonus = entropies.mean()
                            
                            # Apply Entropy Boost if collapse is active
                            active_entropy_coef = entropy_coef
                            if action_collapse_active:
                                active_entropy_coef = min(entropy_coef * 3.0, ENTROPY_COEF_MAX)

                            total_loss = actor_loss + 0.5 * critic_loss - active_entropy_coef * entropy_bonus
                            
                            # Backward & Step
                            if not torch.isnan(total_loss):
                                total_loss.backward()
                                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                                optimizer.step()
                                
                                epoch_actor_loss += actor_loss.item()
                                epoch_critic_loss += critic_loss.item()
                                epoch_entropy += entropy_bonus.item()
                                num_batches += 1
                            
                        # Logging & Entropy Tuning (once per epoch after minibatches)
                        if num_batches > 0:
                            avg_actor_loss = epoch_actor_loss / num_batches
                            avg_critic_loss = epoch_critic_loss / num_batches
                            avg_entropy = epoch_entropy / num_batches
                            
                            ratio = (active_entropy_coef * avg_entropy) / max(0.0001, abs(avg_actor_loss))
                            
                            if ppo_epoch == 0:
                                logger.info(f"  PPO Epoch {ppo_epoch}: actor_loss={avg_actor_loss:.6f}, entropy={avg_entropy:.4f}, ratio={ratio:.2%}")
                            
                            # Smart Entropy Tuner (gentle adjustment)
                            if ratio < 0.01 and entropy_coef < ENTROPY_COEF_MAX:
                                entropy_coef = min(entropy_coef * 1.2, ENTROPY_COEF_MAX)
                            elif ratio > 0.5 and entropy_coef > 0.00001:
                                entropy_coef *= 0.95
                                
                            # WandB Logging
                            if use_wandb and WANDB_AVAILABLE and step % 100 == 0:
                                wandb.log({
                                    f"stage2/{strategy}/ppo_epoch": ppo_epoch,
                                    f"stage2/{strategy}/actor_loss": avg_actor_loss,
                                    f"stage2/{strategy}/critic_loss": avg_critic_loss,
                                    f"stage2/{strategy}/entropy_value": avg_entropy,
                                    f"stage2/{strategy}/entropy_coef": entropy_coef,
                                    f"stage2/{strategy}/entropy_actor_ratio": ratio,
                                    f"stage2/{strategy}/total_loss": avg_actor_loss + 0.5 * avg_critic_loss - active_entropy_coef * avg_entropy
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
                logger.info(f"    Total Reward: {summary['total_reward']:.2f}, Total Profit: ${summary['total_profit']:.2f}")
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
                
                # --- PERIODIC WHITEBOX TELEMETRY (Every 10 Epochs) ---
                if (epoch + 1) % 10 == 0:
                    try:
                        # Update the same CSV to keep it fresh
                        periodic_filename = f"whitebox_stage2_{strategy.lower()}_telemetry.csv"
                        generate_execution_whitebox_report(
                            agent=agent,
                            strategy_type=strategy,
                            preprocessor=preprocessor,
                            mode=mode,
                            window_size=window_size,
                            filename=periodic_filename,
                            logger=logger
                        )
                    except Exception as e:
                        logger.error(f"  Failed to generate periodic Stage 2 whitebox report: {e}")
        
        # Log final statistics for this strategy
        stats.log_summary(prefix=f"{strategy}")
        agents[strategy] = agent
        logger.info(f"{strategy} agent pre-training complete!")
    
    return agents


def generate_stage2_report(agent, env, preprocessor, step_limit=1000, model_name="Agent"):
    """
    Runs a deterministic evaluation episode and prints a comprehensive report.
    """
    print(f"\n{'='*30} STAGE 2 REPORT: {model_name} {'='*30}")
    
    # 1. Setup
    obs, info = env.reset()
    done = False
    truncated = False
    step = 0
    
    # Tracking Data
    rewards = []
    values = []
    actions = []
    
    # Track portfolio values properly
    if hasattr(env, 'base_env') and hasattr(env.base_env, '_get_portfolio_value'):
        # Initial check
        current_step = min(env.base_env.current_step, len(env.base_env.data) - 1)
        current_price = env.base_env.data[current_step, 0]
        initial_val = env.base_env._get_portfolio_value(current_price)
    elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'balance'):
        initial_val = env.unwrapped.balance
    else:
        initial_val = env.balance if hasattr(env, 'balance') else 0.0
        
    portfolio_values = [initial_val]
    
    # Market Tracking (for Alpha)
    market_prices = [] 
    if hasattr(env, 'base_env') and hasattr(env.base_env, 'data'):
        # Capture initial market price
        current_step = min(env.base_env.current_step, len(env.base_env.data) - 1)
        market_prices.append(env.base_env.data[current_step, 0])
    
    # 2. Simulation Loop (Deterministic)
    while not (done or truncated) and step < step_limit:
        
        # Prepare inputs using preprocessor to match training exactly
        # Get raw features from cache/preprocessor
        # We need to access the underlying data at the current step
        
        # 1. Get current windowed features
        # Note: env.base_env.current_step points to the CURRENT step
        # We use preprocessor to extract features at this step
        if hasattr(env, 'base_env'):
            # This is efficient if preprocessor has cached data
            windowed = preprocessor.get_window_features(
                preprocessor.features, 
                env.base_env.window_size, 
                env.base_env.current_step
            )
        else:
             # Fallback
             pass
             
        # Convert to tensors
        features_dict = {}
        if windowed:
            for interval, feat in windowed.items():
                if np.isnan(feat).any() or np.isinf(feat).any():
                    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
                
                transformer_key = interval.replace("1wk", "1w") if interval == "1wk" else interval
                feat_tensor = torch.from_numpy(feat).float().unsqueeze(0).to(device)
                features_dict[transformer_key] = feat_tensor
        
        # Strategy features
        if hasattr(env, 'base_env'):
            shares = env.base_env.shares
            entry_price = env.base_env.entry_price
            
            # Get current price
            current_step = min(env.base_env.current_step, len(env.base_env.data) - 1)
            current_price = env.base_env.data[current_step, 0]
        else:
            shares = 0
            entry_price = 0
            current_price = 1.0
            
        strategy_feat = agent._get_strategy_features(shares, entry_price, current_price).unsqueeze(0).to(device)
        
        # Precision features
        precision_feat = agent._get_precision_features(
            features_dict, shares, entry_price, current_price
        ).unsqueeze(0).to(device)
        
        # Local features (from env if available)
        if hasattr(env, 'base_env') and hasattr(env.base_env, '_get_local_features'):
            local_feat_np = env.base_env._get_local_features()
            local_feat = torch.from_numpy(local_feat_np).unsqueeze(0).float().to(device)
        else:
            local_feat = torch.zeros(1, 4, device=device, dtype=torch.float32)
            if shares > 0:
                local_feat[0, 0] = 1.0
        
        # GET DETERMINISTIC ACTION
        with torch.no_grad():
            # act returns: action, log_prob, value
            action, _, value_pred = agent.act(
                features_dict, 
                strategy_feat, 
                precision_feat, 
                local_feat, 
                deterministic=True
            )

        # Execute
        obs, reward, done, truncated, info = env.step(action)
        
        # Store Data
        rewards.append(reward)
        values.append(value_pred) # value_pred is scalar from act()
        actions.append(action)
        
        # Track portfolio value
        if hasattr(env, 'base_env') and hasattr(env.base_env, '_get_portfolio_value'):
            current_step = min(env.base_env.current_step, len(env.base_env.data) - 1)
            current_price = env.base_env.data[current_step, 0]
            val = env.base_env._get_portfolio_value(current_price)
            portfolio_values.append(val)
            market_prices.append(current_price)
        else:
            portfolio_values.append(info.get('balance', 0))
            if 'price' in info: market_prices.append(info['price'])
        
        step += 1
    
    # 3. Calculations
    
    # --- Financial ---
    initial_bal = portfolio_values[0]
    final_bal = portfolio_values[-1]
    net_profit = final_bal - initial_bal
    # Avoid div by zero
    if initial_bal == 0: initial_bal = 1.0 
    net_profit_pct = (net_profit / initial_bal) * 100
    
    # Alpha (vs Buy & Hold)
    if len(market_prices) > 1:
        # Market return over the period
        market_return = (market_prices[-1] - market_prices[0]) / market_prices[0]
        # net_profit_pct is percentage (e.g. 5.0 for 5%), market_return is decimal (e.g. 0.05 for 5%)
        # Convert net_profit_pct to decimal for comparison
        alpha = (net_profit_pct / 100) - market_return
    else:
        market_return = 0.0
        alpha = 0.0 # Data unavailable
        
    # Max Drawdown
    peak = portfolio_values[0]
    max_dd = 0.0
    for val in portfolio_values:
        if val > peak: peak = val
        if peak > 0:
            dd = (peak - val) / peak
            if dd > max_dd: max_dd = dd
        
    # --- Strategy Behavior ---
    total_trades = actions.count(1) + actions.count(2) # Buy + Sell
    # Assuming win rate requires detailed trade analysis not easily avail in simple list
    # We'll rely on what we can compute or what env provides if possible.
    # For now, simplistic check: 
    # If we have trades, we really want to check 'closed' trades.
    # We will skip Win Rate calculation here unless we track specific trade entries/exits manually
    # or expose env.trades list.
    
    # --- RL Health (Value Accuracy) ---
    # Calculate returns-to-go
    returns = []
    G = 0
    gamma = 0.99 
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns_t = torch.tensor(returns)
    values_t = torch.tensor(values)
    
    # Explained Variance
    y_true_var = torch.var(returns_t)
    if y_true_var == 0: y_true_var = 1e-8
    
    explained_var = 1 - torch.var(returns_t - values_t) / y_true_var
    
    # 4. Printing
    print(f"\n[Financial Performance]")
    print(f"  Final Balance: ${final_bal:.2f}")
    print(f"  Net Profit:    {net_profit_pct:.2f}%")
    print(f"  Max Drawdown:  {max_dd:.2%}")
    print(f"  Alpha:         {alpha:.4f} (Market Return: {market_return:.2%})")
    
    print(f"\n[RL Health]")
    print(f"  Explained Var: {explained_var.item():.4f} (Close to 1.0 is good)")
    print(f"  Action Dist:   HOLD={actions.count(0)/len(actions):.1%} | BUY={actions.count(1)/len(actions):.1%} | SELL={actions.count(2)/len(actions):.1%}")
    print(f"  Avg Value:     {values_t.mean().item():.4f} (Predicted) vs {returns_t.mean().item():.4f} (Actual)")
    
    print(f"\n[Sanity Checks]")
    print(f"  Steps Taken:   {step}")
    print(f"  Total Trades:  {total_trades}")
    
    if hasattr(env, 'base_env') and hasattr(env.base_env, 'invalid_action_count'):
         print(f"  Invalid Acts:  {env.base_env.invalid_action_count}")
    
    print(f"{'='*80}\n")


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
    
    # Track negative EV streak for early stopping (Stage 3)
    negative_ev_streak_stage3 = 0
    
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
        
        # REDUCED ENTROPY: Prioritize profit signals over exploration
        # Entropy coefficient: Start at 0.0001, decay slowly using 0.99 per step
        # Reduced by order of magnitude to make agent care more about profit
        entropy_coef = 0.0001 * (0.99 ** step)  # Slow exponential decay (0.99 per step)
        entropy_coef = max(entropy_coef, 0.00005)  # Clamp to minimum (reduced from 0.001)
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
                    logger.warning(f"  STRATEGY FLICKER DETECTED: {unique_strategies} different strategies in last 5 decisions!")
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
                
                # Keep a copy of unnormalized returns for EV computation
                returns_for_ev_np = np.array(rewards, dtype=np.float32)
                
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
                    
                    # Compute Critic Explained Variance (EV) before loss calculation
                    try:
                        y_true = rewards.detach().cpu().numpy()
                        y_pred = values.detach().cpu().numpy()
                        var_y = np.var(y_true)
                        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                        logger.info(f"Critic Explained Variance: {explained_var:.4f}")
                        if not np.isnan(explained_var) and explained_var < 0.0:
                            negative_ev_streak_stage3 += 1
                        else:
                            negative_ev_streak_stage3 = 0
                        if negative_ev_streak_stage3 >= 10:
                            logger.error("Critic Explained Variance remained negative for 10 consecutive PPO updates. Early stopping Stage 3 training.")
                            return meta_agent
                    except Exception as e:
                        logger.warning(f"Failed to compute Explained Variance (Stage 3): {e}")
                    
                    # Check for NaN before loss calculation
                    if torch.isnan(logprobs).any() or torch.isnan(values).any() or torch.isnan(entropy).any():
                        logger.warning("NaN detected in stacked tensors, skipping PPO update")
                        continue
                    
                    # PPO loss
                    ratios = torch.exp(logprobs - old_logprobs.detach())
                    advantages = rewards - values.detach()
                    
                    # --- WHITEBOX: ADVANTAGE HISTOGRAM ---
                    if step % 1000 == 0:  # Save every 1000 steps
                        plt.figure(figsize=(10, 5))
                        plt.hist(advantages.cpu().numpy(), bins=50, color='skyblue', edgecolor='black')
                        plt.title(f"Advantage Distribution (Step {step})")
                        plt.xlabel("Advantage (Actual - Predicted)")
                        plt.ylabel("Frequency")
                        plt.grid(alpha=0.3)
                        plot_path = f"debug_advantages_step_{step}.png"
                        plt.savefig(plot_path)
                        plt.close()
                        logger.info(f"📊 Advantage Histogram saved: {plot_path}")
                    
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
                        # Also log EV if available from the recent computation block
                        try:
                            wandb.log({"stage3/explained_variance": float(explained_var)})
                        except Exception:
                            pass
                
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
    # Create optimizer with differential learning rates (Stage 4)
    optimizer = optim.AdamW([
        # High LR for the Manager (Meta-Agent) - needs to learn fast
        {'params': meta_agent.parameters(), 'lr': 1e-4},

        # Low LR for Sub-Agents - PRESERVE their pre-training
        {'params': execution_agents['TREND_FOLLOW'].parameters(), 'lr': 1e-6},
        {'params': execution_agents['MEAN_REVERT'].parameters(), 'lr': 1e-6},
        {'params': execution_agents['MOMENTUM'].parameters(), 'lr': 1e-6},
        {'params': execution_agents['RISK_OFF'].parameters(), 'lr': 1e-6},
    ], weight_decay=1e-5)
    
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
        # REDUCED ENTROPY: Prioritize profit signals over exploration
        # Reduced by order of magnitude to make agent care more about profit
        # REDUCED ENTROPY: Prioritize profit signals over exploration
        entropy_coef = 0.0001 * (0.99 ** step)  # Slow exponential decay (0.99 per step)
        entropy_coef = max(entropy_coef, 0.00005)  # Clamp to minimum (reduced from 0.001)
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
                    
                    # --- WHITEBOX: ADVANTAGE HISTOGRAM ---
                    if step % 500 == 0:  # Save every 500 steps during fine-tuning
                        plt.figure(figsize=(10, 5))
                        plt.hist(advantages.cpu().numpy(), bins=50, color='salmon', edgecolor='black')
                        plt.title(f"Advantage Distribution (Fine-Tuning Step {step})")
                        plt.xlabel("Advantage")
                        plt.ylabel("Frequency")
                        plt.grid(alpha=0.3)
                        plot_path = f"debug_advantages_ft_step_{step}.png"
                        plt.savefig(plot_path)
                        plt.close()
                    
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
                    
                    all_trainable_params = list(meta_agent.parameters())
                    for ag in execution_agents.values():
                        all_trainable_params.extend(list(ag.parameters()))
                    
                    optimizer.zero_grad()
                    loss.mean().backward()
                    
                    # Compute L2 gradient norm before clipping (Silent Killer monitoring)
                    total_norm = 0.0
                    for p in all_trainable_params:
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += (param_norm.item() ** 2)
                    total_norm = total_norm ** 0.5
                    logger.info(f"Gradient Norm: {total_norm:.4f}")
                    
                    # Clip gradients and apply update
                    grad_norm = torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=0.5)
                    optimizer.step()
                    
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



def generate_execution_whitebox_report(agent: nn.Module,
                                      strategy_type: str,
                                      preprocessor: MultiTimescalePreprocessor,
                                      mode: str = "stock",
                                      window_size: int = 30,
                                      step_limit: int = 1000,
                                      filename: str = "whitebox_execution_logs.csv",
                                      logger: Optional[logging.Logger] = None):
    """
    Expose internal decision metrics for an Execution Agent to a CSV.
    """
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Generating Execution Whitebox Report for {strategy_type}: {filename}...")
    
    config = TradingConfig()
    config.mode = mode
    
    env = ExecutionEnv(
        ticker=preprocessor.ticker,
        window_size=window_size,
        config=config,
        strategy_type=strategy_type,
        logger=logger
    )
    
    # Process data
    aligned_data, features = preprocessor.process(window_size=window_size)
    if not aligned_data or not features:
        logger.error("Failed to fetch data for whitebox report")
        return None
    
    history = []
    state, _ = env.reset()
    
    # Determine scale names
    scale_names = list(features.keys())
    
    for step in range(1, step_limit + 1):
        # Prepare features
        windowed = preprocessor.get_window_features(features, window_size, None)
        features_dict = {}
        for interval, feat in windowed.items():
            features_dict[interval.replace("1wk", "1w")] = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            
        # Get strategy and precision features from environment
        shares = env.base_env.shares
        entry_price = env.base_env.entry_price
        
        # Calculate current price from data at current step
        if env.base_env.current_step < len(env.base_env.data):
            current_price = env.base_env.data[env.base_env.current_step, 0]
        else:
            current_price = entry_price if entry_price > 0 else 1.0
        
        with torch.no_grad():
            strategy_features = agent._get_strategy_features(shares, entry_price, current_price).unsqueeze(0).to(device)
            precision_features = agent._get_precision_features(features_dict, shares, entry_price, current_price).unsqueeze(0).to(device)
            
            # Local features using environment's own logic (ensures perfect consistency)
            local_feat_np = env.base_env._get_local_features()
            local_features = torch.from_numpy(local_feat_np).unsqueeze(0).float().to(device)
            
            # Run Agent in Deterministic Mode
            # get_action_and_value returns action_idx, log_prob, entropy, value, [attn_weights, logits]
            action, log_prob, entropy, value, attn_weights, logits = agent.get_action_and_value(
                features_dict, strategy_features, precision_features, local_features, 
                deterministic=True, return_weights=True, return_logits=True
            )
            
            # Compute per-action probabilities (0=HOLD, 1=BUY, 2=SELL)
            probs = torch.softmax(logits[0], dim=-1).detach().cpu().numpy()
            logits_np = logits[0].detach().cpu().numpy()
            
        # Step environment
        next_state, reward, done, truncated, info = env.step(action.item())
        
        # Build log entry for this step
        selected_idx = int(action.item()) if hasattr(action, 'item') else int(action)
        uniform = 1.0 / 3.0
        # Compute signed margins relative to neutral (uniform) distribution
        margin_hold = float(probs[0] - uniform)
        margin_buy = float(probs[1] - uniform)
        margin_sell = float(probs[2] - uniform)
        # Top-vs-second confidence margin
        sorted_probs = np.sort(probs)
        top_action_margin = float(sorted_probs[-1] - sorted_probs[-2]) if len(sorted_probs) >= 2 else float('nan')

        entry = {
            'step': step,
            'price': info.get('price', 0.0),
            'shares': shares,
            'entry_price': float(entry_price) if isinstance(entry_price, (int, float)) else 0.0,
            'current_price': float(current_price) if isinstance(current_price, (int, float)) else 0.0,
            'action': selected_idx,
            'prob': torch.exp(log_prob).item() if isinstance(log_prob, torch.Tensor) else float(np.exp(log_prob)),
            'value_pred': value.item() if isinstance(value, torch.Tensor) else float(value),
            'actual_reward': float(reward),
            'entropy': entropy.item() if isinstance(entropy, torch.Tensor) else float(entropy),
            # Per-action logits (0=HOLD, 1=BUY, 2=SELL)
            'logit_hold': float(logits_np[0]),
            'logit_buy': float(logits_np[1]),
            'logit_sell': float(logits_np[2]),
            # Per-action probabilities
            'prob_hold': float(probs[0]),
            'prob_buy': float(probs[1]),
            'prob_sell': float(probs[2]),
            # Signed margins relative to neutral 1/3
            'margin_hold': margin_hold,
            'margin_buy': margin_buy,
            'margin_sell': margin_sell,
            # Selected action stats
            'selected_action_prob': float(probs[selected_idx]),
            'selected_action_logit': float(logits_np[selected_idx]),
            'selected_action_margin': float(probs[selected_idx] - uniform),
            # Overall confidence
            'top_action_margin': top_action_margin
        }

        # Extract attention weights summary (already averaged across heads/query scales)
        if attn_weights is not None:
            # attn_weights shape: (batch, num_scales)
            scale_weights = attn_weights[0].cpu().numpy()
            
            # Add attention weights
            for i, name in enumerate(scale_names):
                if i < len(scale_weights):
                    entry[f'attn_{name}'] = scale_weights[i]
        else:
             # Log warning only once per report
             if step == 1:
                 logger.warning("Attention weights missing in whitebox report!")
               
        history.append(entry)
        if done or truncated: break
        
    df_history = pd.DataFrame(history)
    df_history.to_csv(filename, index=False)
    logger.info(f"✅ Execution Whitebox Telemetry saved to {filename}")
    return df_history

def generate_whitebox_report(agent: MetaStrategyAgent, 
                             preprocessor: MultiTimescalePreprocessor, 
                             regime_detector: EnhancedRegimeDetector,
                             execution_agents: Dict[str, nn.Module],
                             mode: str = "stock",
                             window_size: int = 30,
                             step_limit: int = 1000,
                             filename: str = "whitebox_logs.csv",
                             logger: Optional[logging.Logger] = None):
    """
    Expose internal decision metrics to a CSV for auditing.
    """
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Generating Whitebox Telemetry Report: {filename}...")
    
    config = TradingConfig()
    config.mode = mode
    
    env = MetaStrategyEnv(
        ticker=preprocessor.ticker,
        window_size=window_size,
        config=config,
        execution_horizon=10,
        execution_agents=execution_agents,
        logger=logger
    )
    
    # Process data
    aligned_data, features = preprocessor.process(window_size=window_size)
    if not aligned_data or not features:
        logger.error("Failed to fetch data for whitebox report")
        return None
    
    history = []
    state, _ = env.reset()
    
    # Determine scale names (for attention logging)
    scale_names = list(features.keys())
    
    for step in range(1, step_limit + 1):
        # Prepare features (same logic as training loop)
        windowed = preprocessor.get_window_features(features, window_size, None)
        features_dict = {}
        for interval, feat in windowed.items():
            features_dict[interval.replace("1wk", "1w")] = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            
        # Get regime features
        regime_df = aligned_data.get("1d", next(iter(aligned_data.values())))
        close_prices = regime_df['Close'].values if 'Close' in regime_df.columns else regime_df.iloc[:, 0].values
        regime_features = torch.from_numpy(regime_detector.get_regime_features(close_prices)).float().unsqueeze(0).to(device)
        regime_label = regime_detector.detect_regime(close_prices)
        
        # Run Agent in Deterministic Mode for valid metrics
        with torch.no_grad():
            # Get action, prob, entropy, value, AND attention weights
            action, log_prob, entropy, value, attn_weights = agent.get_action_and_value(
                features_dict, regime_features, deterministic=True, return_weights=True
            )
        
        # Step environment
        next_state, reward, done, truncated, info = env.step(action.item())
        
        # Extract attention weights summary: already averaged across heads and query scales
        # attn_weights shape: (batch, num_scales)
        scale_weights = attn_weights[0].cpu().numpy()
        
        entry = {
            'step': step,
            'price': info.get('price', 0.0),
            'regime': regime_label,
            'action': ["TREND_FOLLOW", "MEAN_REVERT", "MOMENTUM", "RISK_OFF"][action.item()],
            'prob': torch.exp(log_prob).item(),
            'value_pred': value.item(),
            'actual_reward': reward,
            'entropy': entropy.item()
        }
        
        # Add attention weights to log
        for i, name in enumerate(scale_names):
            if i < len(scale_weights):
                entry[f'attn_{name}'] = scale_weights[i]
        
        history.append(entry)
        
        if done or truncated: break
        
    df_history = pd.DataFrame(history)
    df_history.to_csv(filename, index=False)
    logger.info(f"✅ Whitebox Telemetry saved to {filename}")
    return df_history

def train_hierarchical_transformer(ticker: str = "^IXIC",
                                   mode: str = "stock",
                                   window_size: int = 30,
                                   save_dir: str = "models_v0.2",
                                   multi_stock_tickers: Optional[List[str]] = None,
                                   logger: Optional[logging.Logger] = None,
                                   use_wandb: bool = True,
                                   wandb_project: str = "hierarchical-transformer-trading",
                                   wandb_entity: Optional[str] = None,
                                   start_stage: int = 1,
                                   load_pretrain_path: Optional[str] = None):
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
    
    # Load or train transformer (Stage 1)
    transformer = None
    if load_pretrain_path and os.path.exists(load_pretrain_path):
        logger.info("=" * 60)
        logger.info(f"LOADING PRETRAINED TRANSFORMER FROM: {load_pretrain_path}")
        logger.info("=" * 60)
        
        # Determine input_dim from data (needed to initialize model before loading weights)
        aligned_data, features = preprocessor.process(window_size=window_size)
        if not aligned_data or not features:
            raise ValueError("Failed to process data for input_dim detection")
        
        # Get sample feature to determine input_dim
        sample_feat = next(iter(features.values()))
        if len(sample_feat.shape) == 2:
            input_dim = sample_feat.shape[1]
        else:
            input_dim = 8 if mode == "stock" else 5
        
        logger.info(f"Detected input_dim={input_dim} from data (mode={mode})")
        
        # Create transformer with same architecture as training
        transformer = MultiScaleTransformerEncoder(
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.1,
            input_dim=input_dim,
            mode=mode
        ).to(device)
        
        # Load pretrained weights
        transformer.load_state_dict(torch.load(load_pretrain_path, map_location=device))
        logger.info(f"Successfully loaded pretrained transformer from {load_pretrain_path}")
        logger.info("Skipping Stage 1 (using pretrained model)")
        
    elif start_stage <= 1:
        # Stage 1: Pre-train transformers
        logger.info("=" * 60)
        logger.info("STAGE 1: Pre-training Transformer Encoders")
        logger.info("=" * 60)
        
        # IMPROVEMENT: Multi-stock training for general market dynamics
        # If multi_stock_tickers provided, use multi-stock; otherwise use single ticker
        if multi_stock_tickers is None:
            # Default: Use single ticker for backward compatibility
            multi_stock_tickers = [ticker]
            logger.info(f"Stage 1: Single-stock mode (ticker: {ticker})")
        else:
            logger.info(f"Stage 1: Multi-stock mode ({len(multi_stock_tickers)} stocks)")
            logger.info(f"  Stocks: {multi_stock_tickers}")
        
        transformer = pretrain_transformer_encoders(
            preprocessor, mode=mode, epochs=300, window_size=window_size, 
            multi_stock_tickers=multi_stock_tickers, logger=logger, use_wandb=use_wandb
        )
        torch.save(transformer.state_dict(), os.path.join(save_dir, "transformer_pretrained.pth"))
        logger.info("Saved pre-trained transformer")
    else:
        raise ValueError(f"Cannot start from stage {start_stage} without a pretrained transformer. "
                        f"Either provide --load_pretrain or start from stage 1.")
    
    if transformer is None:
        raise RuntimeError("Transformer not initialized. This should not happen.")
    
    # Stage 2: Pre-train execution agents
    execution_agents = None
    if start_stage <= 2:
        logger.info("=" * 60)
        logger.info("STAGE 2: Pre-training Execution Agents")
        logger.info("=" * 60)
        execution_agents = pretrain_execution_agents(
            transformer, preprocessor, regime_detector, mode=mode, 
            epochs_per_agent=50, steps_per_epoch=512, window_size=window_size, logger=logger, use_wandb=use_wandb
        )
        for strategy, agent in execution_agents.items():
            torch.save(agent.state_dict(), os.path.join(save_dir, f"execution_{strategy.lower()}.pth"))
        logger.info("Saved pre-trained execution agents")
        
        # --- STAGE 2 WHITEBOX TELEMETRY ---
        logger.info("Generating Stage 2 Whitebox Telemetry...")
        for strategy, agent in execution_agents.items():
            try:
                generate_execution_whitebox_report(
                    agent=agent,
                    strategy_type=strategy,
                    preprocessor=preprocessor,
                    mode=mode,
                    window_size=window_size,
                    filename=os.path.join(save_dir, f"whitebox_stage2_{strategy.lower()}_telemetry.csv"),
                    logger=logger
                )
            except Exception as e:
                logger.error(f"Failed to generate whitebox report for {strategy}: {e}")
        # ---------------------------------
    else:
        logger.info("Skipping Stage 2 (loading execution agents)")
        # Load execution agents if starting from later stage
        execution_agents = {}
        strategies = ["TREND_FOLLOW", "MEAN_REVERT", "MOMENTUM", "RISK_OFF"]
        for strategy in strategies:
            agent_path = os.path.join(save_dir, f"execution_{strategy.lower()}.pth")
            if os.path.exists(agent_path):
                logger.info(f"Loading execution agent: {strategy}")
                # Determine input_dim for agent creation
                aligned_data, features = preprocessor.process(window_size=window_size)
                sample_feat = next(iter(features.values())) if features else None
                input_dim = sample_feat.shape[1] if sample_feat is not None and len(sample_feat.shape) == 2 else (8 if mode == "stock" else 5)
                
                # Create temporary agents to get the agent architecture, then load weights
                temp_agents = create_execution_agents(transformer, mode=mode, input_dim=input_dim)
                agent = temp_agents[strategy]
                agent.load_state_dict(torch.load(agent_path, map_location=device))
                execution_agents[strategy] = agent
            else:
                raise FileNotFoundError(f"Execution agent not found: {agent_path}. Cannot start from stage {start_stage}.")
    
    if execution_agents is None:
        raise RuntimeError("Execution agents not initialized. This should not happen.")
    
    # Stage 3: Train meta-strategy
    meta_agent = None
    if start_stage <= 3:
        logger.info("=" * 60)
        logger.info("STAGE 3: Training Meta-Strategy Agent")
        logger.info("=" * 60)
        meta_agent = train_meta_strategy_agent(
            transformer, execution_agents, preprocessor, regime_detector,
            mode=mode, steps=100000, window_size=window_size, logger=logger, use_wandb=use_wandb
        )
        torch.save(meta_agent.state_dict(), os.path.join(save_dir, "meta_strategy.pth"))
        logger.info("Saved meta-strategy agent")
        
        # --- WHITEBOX TELEMETRY ---
        generate_whitebox_report(
            agent=meta_agent,
            preprocessor=preprocessor,
            regime_detector=regime_detector,
            execution_agents=execution_agents,
            mode=mode,
            window_size=window_size,
            filename=os.path.join(save_dir, "whitebox_stage3_telemetry.csv"),
            logger=logger
        )
    else:
        logger.info("Skipping Stage 3 (loading meta-strategy agent)")
        meta_path = os.path.join(save_dir, "meta_strategy.pth")
        if os.path.exists(meta_path):
            logger.info(f"Loading meta-strategy agent from {meta_path}")
            # Determine input_dim for meta agent creation
            aligned_data, features = preprocessor.process(window_size=window_size)
            sample_feat = next(iter(features.values())) if features else None
            input_dim = sample_feat.shape[1] if sample_feat is not None and len(sample_feat.shape) == 2 else (8 if mode == "stock" else 5)
            
            from meta_strategy_agent import MetaStrategyAgent
            meta_agent = MetaStrategyAgent(
                d_model=64, nhead=4, num_layers=2, dim_feedforward=256,
                dropout=0.1, input_dim=input_dim, mode=mode, hidden_dim=128
            ).to(device)
            meta_agent.load_state_dict(torch.load(meta_path, map_location=device))
        else:
            raise FileNotFoundError(f"Meta-strategy agent not found: {meta_path}. Cannot start from stage {start_stage}.")
    
    if meta_agent is None:
        raise RuntimeError("Meta-strategy agent not initialized. This should not happen.")
    
    # Stage 4: End-to-end fine-tuning
    if start_stage <= 4:
        logger.info("=" * 60)
        logger.info("STAGE 4: End-to-End Fine-Tuning")
        logger.info("=" * 60)
        fine_tune_end_to_end(
            meta_agent, execution_agents, preprocessor, regime_detector,
            mode=mode, steps=5000, window_size=window_size, logger=logger, use_wandb=use_wandb
        )
        
        # Save final models
        torch.save(meta_agent.state_dict(), os.path.join(save_dir, "meta_strategy_final.pth"))
        for strategy, agent in execution_agents.items():
            torch.save(agent.state_dict(), os.path.join(save_dir, f"execution_{strategy.lower()}_final.pth"))
        logger.info("Saved final fine-tuned models")
        
        # --- WHITEBOX TELEMETRY ---
        generate_whitebox_report(
            agent=meta_agent,
            preprocessor=preprocessor,
            regime_detector=regime_detector,
            execution_agents=execution_agents,
            mode=mode,
            window_size=window_size,
            filename=os.path.join(save_dir, "whitebox_stage4_telemetry.csv"),
            logger=logger
        )
    else:
        logger.info("Skipping Stage 4")
    
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
    parser.add_argument("--multi-stock", type=str, nargs="+", default=None, 
                       help="Multi-stock tickers for Stage 1 (e.g., --multi-stock AAPL MSFT GOOGL). "
                            "Stage 1 learns general market dynamics. Stages 2-4 use single ticker.")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3, 4],
                       help="Starting stage (1-4). Stage 1: Pre-train transformers, "
                            "Stage 2: Pre-train execution agents, Stage 3: Train meta-strategy, "
                            "Stage 4: End-to-end fine-tuning. Requires --load_pretrain if stage > 1.")
    parser.add_argument("--load-pretrain", type=str, default=None,
                       dest="load_pretrain",
                       help="Path to pretrained transformer model (.pt or .pth file). "
                            "Use this to skip Stage 1 and start from Stage 2. "
                            "Example: --load-pretrain weights/transformer_pretrain_ETH-USD.pt")
    
    args = parser.parse_args()
    
    # Parse multi-stock tickers if provided
    multi_stock_tickers = args.multi_stock if args.multi_stock else None
    
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
    
    # Validate arguments
    if args.stage > 1 and not args.load_pretrain:
        logger.error(f"Cannot start from stage {args.stage} without --load-pretrain.")
        logger.error("Please provide --load-pretrain path to a pretrained transformer model.")
        raise ValueError(f"Stage {args.stage} requires --load-pretrain")
    
    if args.load_pretrain and not os.path.exists(args.load_pretrain):
        logger.error(f"Pretrained model not found: {args.load_pretrain}")
        raise FileNotFoundError(f"Pretrained model not found: {args.load_pretrain}")
    
    # Train
    train_hierarchical_transformer(
        ticker=args.ticker,
        mode=args.mode,
        window_size=args.window_size,
        save_dir=args.save_dir,
        multi_stock_tickers=multi_stock_tickers,  # Multi-stock for Stage 1
        logger=logger,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        start_stage=args.stage,
        load_pretrain_path=args.load_pretrain
    )

