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
    
    def update(self, reward: float, info: Dict, action: Optional[int] = None, 
               strategy: Optional[str] = None):
        """Update statistics from step"""
        self.total_reward += reward
        self.rewards_history.append(reward)
        
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
            if self.initial_balance is None:
                self.initial_balance = info.get('initial_balance', info['balance'])
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
        
        return {
            'total_reward': self.total_reward,
            'total_profit': self.total_profit,
            'return_pct': return_pct,
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'action_counts': dict(self.action_counts),
            'strategy_counts': dict(self.strategy_counts),
            'trade_count': self.trade_count,
            'win_count': self.win_count,
            'loss_count': self.loss_count,
            'win_rate': win_rate,
            'avg_reward': np.mean(self.rewards_history) if self.rewards_history else 0.0,
            'avg_profit': np.mean(self.profits_history) if self.profits_history else 0.0
        }
    
    def log_summary(self, prefix: str = ""):
        """Log summary statistics"""
        summary = self.get_summary()
        prefix_str = f"{prefix} - " if prefix else ""
        
        self.logger.info(f"{prefix_str}=== Training Statistics ===")
        self.logger.info(f"{prefix_str}Total Reward: {summary['total_reward']:.2f}")
        self.logger.info(f"{prefix_str}Total Profit: ${summary['total_profit']:.2f}")
        self.logger.info(f"{prefix_str}Return: {summary['return_pct']:.2f}%")
        self.logger.info(f"{prefix_str}Balance: ${summary['initial_balance']:.2f} → ${summary['current_balance']:.2f}")
        self.logger.info(f"{prefix_str}Actions: {summary['action_counts']}")
        if summary['strategy_counts']:
            self.logger.info(f"{prefix_str}Strategies: {summary['strategy_counts']}")
        self.logger.info(f"{prefix_str}Trades: {summary['trade_count']} (Wins: {summary['win_count']}, Losses: {summary['loss_count']}, Win Rate: {summary['win_rate']:.1f}%)")
        self.logger.info(f"{prefix_str}Avg Reward: {summary['avg_reward']:.4f}, Avg Profit: ${summary['avg_profit']:.2f}")


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
                                  batch_size: int = 64,
                                  lr: float = 0.0001,
                                  window_size: int = 30,
                                  logger: Optional[logging.Logger] = None) -> MultiScaleTransformerEncoder:
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
        
        # Train on each timescale
        for interval, X in X_dict.items():
            y = y_dict[interval]
            
            # Create dataloader (FIX: Ensure float32 dtype)
            dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
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
                
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.6f}")
    
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
                              logger: Optional[logging.Logger] = None) -> Dict[str, nn.Module]:
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
        
        # Create execution environment
        config = TradingConfig()
        config.mode = mode
        exec_env = ExecutionEnv(
            ticker=preprocessor.ticker,
            window_size=window_size,
            config=config,
            strategy_type=strategy,
            logger=logger
        )
        
        # Initialize statistics tracker for this strategy
        stats = TrainingStats(logger)
        
        # Optimizer
        optimizer = optim.Adam(agent.parameters(), lr=lr)
        
        # DEBUG: Ensure agent is in training mode and parameters require gradients
        agent.train()  # Set to training mode
        for param in agent.parameters():
            if not param.requires_grad:
                logger.warning(f"Parameter {param.shape} does not require grad, enabling it")
                param.requires_grad = True
        
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
        
        for epoch in range(epochs_per_agent):
            state, _ = exec_env.reset()
            episode_reward = 0.0
            
            # CRITICAL FIX: Reset statistics at start of each epoch
            # This ensures action counts and other stats are per-epoch, not accumulated
            stats.reset()
            
            # SLOW ENTROPY DECAY: Keep agent curious for longer (until epoch 100+)
            # Entropy coefficient: Start at 0.0001, decay slowly to 0.00001 over epochs
            # Use slower decay (0.99 per epoch) so entropy stays above zero until epoch 100
            # Instead of linear progress, use exponential decay with base 0.99
            entropy_coef = 0.0001 * (0.99 ** epoch)  # Slow exponential decay (0.99 per epoch)
            entropy_coef = max(entropy_coef, 0.00001)  # Clamp to minimum
            epoch_progress = epoch / epochs_per_agent  # For logging only
            
            # DEBUG: Log entropy coefficient to verify decay
            if epoch % 10 == 0 or epoch < 5:  # Log every 10 epochs or first 5
                logger.info(f"  Epoch {epoch}: Entropy Coef = {entropy_coef:.4f} (Progress: {epoch_progress:.1%})")
            
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
                
                # DEBUG: Get action and value for PPO
                # CRITICAL: We need to store the ORIGINAL features_dict and strategy_feat WITHOUT detaching
                # so that when we recompute during PPO, gradients can flow through model parameters
                # The inputs themselves don't need gradients, but we must not detach them before storing
                try:
                    # DEBUG: Ensure agent is in training mode
                    agent.train()
                    
                    # Forward pass WITH gradients (needed for PPO updates)
                    # Inputs don't need gradients (they're data), but model parameters do
                    logits, value_tensor = agent(features_dict, strategy_feat)
                    
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
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        logger.warning("NaN/Inf in logits after model forward - model may have issues")
                        # Try to maintain gradient connection if possible
                        # Use a small operation that preserves grad_fn
                        logits = logits + 0.0 * torch.zeros_like(logits)  # Maintain connection
                        logits = torch.clamp(logits, min=-10, max=10)  # Clamp to reasonable range
                    
                    if torch.isnan(value_tensor).any() or torch.isinf(value_tensor).any():
                        logger.warning("NaN/Inf in value_tensor after model forward - model may have issues")
                        # Maintain gradient connection
                        value_tensor = value_tensor + 0.0 * torch.zeros_like(value_tensor)
                    
                    # Clamp logits to prevent extreme values
                    logits = torch.clamp(logits, min=-10, max=10)
                    
                    # Create distribution and sample action
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    # DEBUG: Log action probabilities to check if agent is learning
                    # (Only log occasionally to avoid spam)
                    if step % 100 == 0 and epoch % 10 == 0:  # Log every 100 steps, every 10 epochs
                        probs = F.softmax(logits, dim=-1)
                        logger.debug(f"  Step {step}: Action probs = HOLD:{probs[0,0]:.3f}, BUY:{probs[0,1]:.3f}, SELL:{probs[0,2]:.3f}, Entropy={dist.entropy().item():.3f}")
                    
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
                
                # Step environment (this doesn't need gradients)
                next_state, reward, done, truncated, info = exec_env.step(action_item)
                
                # Clip reward to prevent extreme values
                reward = np.clip(reward, -100, 100)
                
                # Update statistics
                if 'balance' not in info and hasattr(exec_env.base_env, 'balance'):
                    info['balance'] = exec_env.base_env.balance
                if 'balance' not in info and hasattr(exec_env.base_env, 'initial_balance'):
                    info['initial_balance'] = exec_env.base_env.initial_balance
                stats.update(reward, info, action=action_item, strategy=strategy)
                
                # DEBUG: Store in memory for PPO update
                # CRITICAL: Store ORIGINAL features_dict and strategy_feat WITHOUT detaching
                # They don't need gradients themselves, but we need the original tensors
                # so that when we recompute, the model can create gradients through its parameters
                # Only detach the policy outputs (log_prob, value), not the inputs!
                memory_states.append((features_dict, strategy_feat))
                # Store action (detached, no gradients needed for actions)
                action_to_store = action if action.dim() > 0 else action.unsqueeze(0)
                memory_actions.append(action_to_store.detach())
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
                    # Reset stats for new episode
                    stats.reset()
                
                # DEBUG: PPO update every 100 steps
                if len(memory_rewards) >= 100:
                    logger.debug(f"PPO update triggered at step {step}, memory size: {len(memory_rewards)}")
                    
                    # DEBUG: Calculate discounted rewards (returns)
                    rewards = []
                    discounted_reward = 0
                    gamma = 0.99
                    for r in reversed(memory_rewards):
                        discounted_reward = r + gamma * discounted_reward
                        rewards.insert(0, discounted_reward)
                    
                    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                    # Normalize rewards for stability
                    rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
                    
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
                        
                        for idx, ((feat_dict, strat_feat), act) in enumerate(zip(memory_states, old_actions)):
                            try:
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
                                logits, value = agent(feat_dict_ready, strat_feat_ready)
                                
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
                                
                                logits = torch.clamp(logits, min=-10, max=10)
                                
                                # Create distribution and compute new policy log_prob
                                dist = torch.distributions.Categorical(logits=logits)
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
                        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)
                        advantages = advantages.detach()  # Ensure advantages don't have gradients
                        
                        # DEBUG: PPO clipped objective
                        # Ratio = exp(new_logprob - old_logprob)
                        # new_logprobs has gradients (from model), old_logprobs_detached is detached
                        # ratios will have gradients (from new_logprobs)
                        ratios = torch.exp(new_logprobs - old_logprobs_detached)
                        # advantages are detached, so surr1 and surr2 will have gradients from ratios
                        surr1 = ratios * advantages
                        surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                        actor_loss = -torch.min(surr1, surr2).mean()
                        
                        # DEBUG: Critic loss (value function learning)
                        # new_values has gradients (from model), rewards is a tensor (no gradients)
                        # MSE loss will compute gradients through new_values
                        critic_loss = nn.MSELoss()(new_values, rewards)
                        
                        # RISK B FIX: Entropy bonus (encourages exploration)
                        # CRITICAL FIX: Use epoch-based entropy coefficient (faster decay)
                        # Calculate entropy coefficient based on epoch progress (already calculated above)
                        entropy_bonus = entropies.mean()
                        
                        # Total loss (actor + critic - entropy)
                        # entropy_coef is calculated per epoch above (line 532)
                        entropy_loss_term = entropy_coef * entropy_bonus
                        total_loss = actor_loss + 0.5 * critic_loss - entropy_loss_term
                        
                        # CRITICAL DEBUG: Check for entropy domination
                        # If entropy term >> actor_loss, agent will stay random
                        actor_loss_val = actor_loss.item()
                        entropy_loss_val = entropy_loss_term.item()
                        ratio = entropy_loss_val / (actor_loss_val + 1e-8)
                        
                        logger.debug(f"PPO epoch {ppo_epoch}: actor_loss={actor_loss_val:.6f}, "
                                   f"critic_loss={critic_loss.item():.6f}, entropy_loss={entropy_loss_val:.6f}, "
                                   f"entropy/actor_ratio={ratio:.2f}, total_loss={total_loss.item():.6f}")
                        
                        # WARNING: If entropy dominates, agent can't learn
                        if ratio > 10.0:
                            logger.warning(f"[WARNING] ENTROPY DOMINATION: entropy_loss ({entropy_loss_val:.6f}) is {ratio:.1f}x larger than actor_loss ({actor_loss_val:.6f})! "
                                         f"Agent will maximize entropy (stay random). Consider reducing entropy_coef or boosting rewards.")
                        
                        # DEBUG: Check for NaN before backward
                        if torch.isnan(total_loss) or torch.isinf(total_loss):
                            logger.warning(f"NaN/Inf loss in {strategy} training at epoch {ppo_epoch}, skipping update")
                            break
                        
                        # DEBUG: Backward pass (compute gradients)
                        total_loss.backward()
                        
                        # DEBUG: Gradient clipping for stability
                        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                        
                        # DEBUG: Optimizer step (update parameters)
                        optimizer.step()
                    
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
                logger.info(f"  Epoch {epoch + 1}/{epochs_per_agent}, Episode Reward: {episode_reward:.2f}")
                logger.info(f"    Actions: {summary['action_counts']}, Trades: {summary['trade_count']}, "
                          f"Return: {summary['return_pct']:.2f}%, Balance: ${summary['current_balance']:.2f}")
        
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
                              logger: Optional[logging.Logger] = None) -> MetaStrategyAgent:
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
    
    for step in range(1, steps + 1):
        # ENTROPY ANNEALING: Calculate entropy coefficient for this step
        # Entropy coefficient: Start at 0.0001, decay to 0.00001 over training (DRASTICALLY REDUCED)
        progress = step / steps  # 0.0 to 1.0
        # Exponential decay: starts at 0.0001, decays to 0.00001
        entropy_coef = 0.0001 * (0.00001 / 0.0001) ** progress  # Exponential decay from 0.0001 to 0.00001
        entropy_coef = max(entropy_coef, 0.00001)  # Clamp to minimum
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
        
        # Get action with epsilon-greedy exploration
        # FIX: Add exploration to prevent over-reliance on single strategy
        epsilon = max(0.1, 0.5 * (1 - step / steps))  # Decay from 0.5 to 0.1
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
                epsilon = max(0.1, 0.5 * (1 - step / steps))
                logger.info(f"Step {step}: DEBUG LOGITS (Meta-Strategy, Epsilon={epsilon:.3f}): {logits_np}")
                # Check if logits are all identical (dead network)
                if len(logits_np) > 1 and np.allclose(logits_np, logits_np[0], atol=1e-6):
                    logger.warning(f"  ⚠️  WARNING: All logits are identical! Network may be dead. Logits: {logits_np}")
            
            action, log_prob, value = meta_agent.act(features_dict, regime_features, deterministic=False)
        
        # Step environment
        next_state, reward, done, truncated, info = meta_env.step(action)
        
        # Clip reward to prevent extreme values
        reward = np.clip(reward, -100, 100)
        
        # Update statistics
        strategy_names = {0: "TREND_FOLLOW", 1: "MEAN_REVERT", 2: "MOMENTUM", 3: "RISK_OFF"}
        strategy_name = strategy_names.get(action, f"STRATEGY_{action}")
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
        
        # PPO update every 500 steps
        if step % 500 == 0:
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
                    
                    torch.nn.utils.clip_grad_norm_(meta_agent.parameters(), max_norm=0.5)
                    optimizer.step()
                
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
            logger.info(f"  Balance: ${summary['initial_balance']:.2f} → ${summary['current_balance']:.2f}")
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
                         logger: Optional[logging.Logger] = None):
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
        # ENTROPY ANNEALING: Calculate entropy coefficient for this step
        # CRITICAL FIX: Faster exponential decay to force agent to use brain instead of random guessing
        progress = step / steps  # 0.0 to 1.0
        # Exponential decay: starts at 0.05, decays to 0.001
        # Entropy coefficient: Start at 0.01, decay to 0.001 over training
        # Exponential decay: starts at 0.01, decays to 0.001
        entropy_coef = 0.01 * (0.001 / 0.01) ** progress  # Exponential decay from 0.01 to 0.001
        entropy_coef = max(entropy_coef, 0.001)  # Clamp to minimum
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
        reward = np.clip(reward, -100, 100)
        
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
        
        # Update every 500 steps
        if step % 500 == 0:
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
                    torch.nn.utils.clip_grad_norm_(meta_agent.parameters(), max_norm=0.5)
                    meta_optimizer.step()
                
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
            logger.info(f"  Balance: ${summary['initial_balance']:.2f} → ${summary['current_balance']:.2f}")
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
                                   logger: Optional[logging.Logger] = None):
    """
    Complete training pipeline with curriculum learning.
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
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize components
    preprocessor = MultiTimescalePreprocessor(ticker, mode=mode, logger=logger)
    regime_detector = EnhancedRegimeDetector(window=20, logger=logger)
    
    # Stage 1: Pre-train transformers
    transformer = pretrain_transformer_encoders(
        preprocessor, mode=mode, epochs=10, window_size=window_size, logger=logger
    )
    torch.save(transformer.state_dict(), os.path.join(save_dir, "transformer_pretrained.pth"))
    logger.info("Saved pre-trained transformer")
    
    # Stage 2: Pre-train execution agents
    execution_agents = pretrain_execution_agents(
        transformer, preprocessor, regime_detector, mode=mode, 
        epochs_per_agent=50, steps_per_epoch=200, window_size=window_size, logger=logger
    )
    for strategy, agent in execution_agents.items():
        torch.save(agent.state_dict(), os.path.join(save_dir, f"execution_{strategy.lower()}.pth"))
    logger.info("Saved pre-trained execution agents")
    
    # Stage 3: Train meta-strategy
    meta_agent = train_meta_strategy_agent(
        transformer, execution_agents, preprocessor, regime_detector,
        mode=mode, steps=10000, window_size=window_size, logger=logger
    )
    torch.save(meta_agent.state_dict(), os.path.join(save_dir, "meta_strategy.pth"))
    logger.info("Saved meta-strategy agent")
    
    # Stage 4: End-to-end fine-tuning
    fine_tune_end_to_end(
        meta_agent, execution_agents, preprocessor, regime_detector,
        mode=mode, steps=5000, window_size=window_size, logger=logger
    )
    
    # Save final models
    torch.save(meta_agent.state_dict(), os.path.join(save_dir, "meta_strategy_final.pth"))
    for strategy, agent in execution_agents.items():
        torch.save(agent.state_dict(), os.path.join(save_dir, f"execution_{strategy.lower()}_final.pth"))
    
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
        logger=logger
    )

