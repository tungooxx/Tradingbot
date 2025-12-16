"""
REAL-TIME TRAINING SYSTEM
=========================
Continuously trains the model with live data from yfinance.

Features:
1. Fetches live data periodically (every N minutes)
2. Updates environment with new data automatically
3. Continues training the model on fresh data
4. Saves checkpoints periodically
5. Monitors performance in real-time
6. Graceful shutdown (Ctrl+C saves model)

Usage:
    # Train for 8 hours with data updates every 15 minutes
    python train_realtime.py --ticker ETH-USD --mode crypto --interval 1h --duration 8
    
    # Train for 4 hours with data updates every 5 minutes
    python train_realtime.py --ticker NVDA --mode stock --interval 1d --duration 4 --data-update-interval 5
    
    # Continue training from existing model
    python train_realtime.py --ticker ETH-USD --mode crypto --interval 1h --duration 8 --model kan_agent_crypto.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import signal
import sys
from typing import Optional, Dict
import argparse
import logging

from dapgio_improved import (
    StockTradingEnv,
    KANPredictor,
    KANActorCritic,
    TradingConfig,
    setup_logging,
    TradingLogger
)

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_flag
    print("\n\n⚠️  Shutdown signal received. Saving model and exiting...")
    shutdown_flag = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class RealtimeDataUpdater:
    """Updates environment data with live data from yfinance"""
    
    def __init__(self, ticker: str, mode: str, interval: str, window_size: int, logger: logging.Logger):
        self.ticker = ticker
        self.mode = mode
        self.interval = interval
        self.window_size = window_size
        self.logger = logger
        self.last_update_time = None
        self.update_interval_minutes = 15  # Update data every 15 minutes
    
    def should_update(self) -> bool:
        """Check if data should be updated"""
        if self.last_update_time is None:
            return True
        
        elapsed = (datetime.now() - self.last_update_time).total_seconds() / 60
        return elapsed >= self.update_interval_minutes
    
    def fetch_live_data(self) -> Optional[pd.DataFrame]:
        """Fetch latest data from yfinance"""
        try:
            if self.mode == "crypto":
                period = "730d" if self.interval == "1h" else "120d"
            else:
                if self.interval == "1h":
                    period = "2y"
                else:
                    period = "2y"
            
            self.logger.info(f"Fetching live data for {self.ticker}...")
            df = yf.download(self.ticker, period=period, interval=self.interval, 
                           progress=False, auto_adjust=True)
            
            if df.empty:
                self.logger.warning(f"No data fetched for {self.ticker}")
                return None
            
            # Fix MultiIndex
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            self.last_update_time = datetime.now()
            self.logger.info(f"Fetched {len(df)} bars, latest: {df.index[-1]}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching live data: {e}")
            return None
    
    def update_environment(self, env: StockTradingEnv, df: pd.DataFrame) -> bool:
        """Update environment with new data - recalculates all features"""
        try:
            # Use the environment's feature calculation method
            # Recalculate features exactly as environment does
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Vol_Norm'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            import pandas_ta as ta
            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.bbands(length=20, std=2, append=True)
            df.dropna(inplace=True)
            
            # Normalize (must match training normalization exactly)
            df["RSI_14"] = df["RSI_14"] / 100.0
            
            # Use rolling normalization for MACD (matches training)
            if self.interval == "1h":
                rolling_window = 252 * 24
            elif self.interval == "4h":
                rolling_window = 252 * 6
            else:
                rolling_window = 252
            
            macd_mean = df["MACD_12_26_9"].rolling(rolling_window, min_periods=1).mean()
            macd_std = df["MACD_12_26_9"].rolling(rolling_window, min_periods=1).std()
            df["MACD_12_26_9"] = (df["MACD_12_26_9"] - macd_mean) / (macd_std + 1e-7)
            
            # Update environment data structures
            features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9"]
            env.df = df.copy()  # Store full dataframe
            env.data = df[features].values  # Update data array
            env.max_steps = len(env.data) - 1
            
            # Adjust current_step if it's beyond new data
            if env.current_step >= env.max_steps:
                env.current_step = env.window_size
            elif env.current_step < env.window_size:
                env.current_step = env.window_size
            
            # Validate data
            if len(env.data) < env.window_size + 10:
                self.logger.warning(f"Insufficient data after update: {len(env.data)} < {env.window_size + 10}")
                return False
            
            self.logger.info(f"✅ Environment updated: {len(env.data)} data points, max_steps={env.max_steps}, current_step={env.current_step}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating environment: {e}", exc_info=True)
            return False


def prepare_pretraining_data(env):
    """Extract windows (X) and next day Log Returns (y) from environment data"""
    X_list = []
    y_list = []

    for i in range(env.window_size, len(env.data) - 1):
        obs = env._get_observation(i)
        target = env.data[i][1]  # The Log_Ret of "today"

        X_list.append(obs)
        y_list.append(target)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32).reshape(-1, 1)


def train_realtime(
    ticker: str,
    mode: str,
    interval: str,
    window_size: int,
    hidden_dim: int,
    duration_hours: float = 8.0,
    model_path: Optional[str] = None,
    checkpoint_interval: int = 10000,  # Save checkpoint every N steps
    data_update_interval: int = 15  # Update data every N minutes
):
    """
    Real-time training with live data updates
    
    Args:
        ticker: Symbol to train on
        mode: "crypto" or "stock"
        interval: "1h", "4h" for crypto; "1d" for stock
        window_size: Lookback window size
        hidden_dim: Hidden dimension for model
        duration_hours: How long to train (hours)
        model_path: Path to existing model (optional, for continued training)
        checkpoint_interval: Save checkpoint every N steps
        data_update_interval: Update data every N minutes
    """
    global shutdown_flag
    
    print("=" * 70)
    print("🔄 REAL-TIME TRAINING SYSTEM")
    print("=" * 70)
    print(f"Ticker: {ticker}")
    print(f"Mode: {mode.upper()}")
    print(f"Interval: {interval}")
    print(f"Window Size: {window_size}")
    print(f"Duration: {duration_hours} hours")
    print(f"Data Update Interval: {data_update_interval} minutes")
    print(f"Checkpoint Interval: {checkpoint_interval} steps")
    print("=" * 70)
    
    # Load config
    config = TradingConfig()
    config.mode = mode
    config.interval = interval
    config.ticker = ticker
    
    if model_path is None:
        model_path = "kan_agent_crypto.pth" if mode == "crypto" else "kan_agent_stock.pth"
    
    logger = setup_logging(config)
    
    # Initialize data updater
    data_updater = RealtimeDataUpdater(ticker, mode, interval, window_size, logger)
    
    # Initialize environment
    print("\n📥 Initializing environment with live data...")
    env = StockTradingEnv(ticker=ticker, window_size=window_size, config=config, logger=logger)
    obs_dim = env.obs_shape
    action_dim = env.action_space.n
    
    print(f"   Observation dimension: {obs_dim}")
    print(f"   Action dimension: {action_dim}")
    print(f"   Initial data points: {len(env.data)}")
    
    # Initialize or load model
    if os.path.exists(model_path):
        print(f"\n📂 Loading existing model from {model_path}...")
        agent = KANActorCritic(obs_dim, action_dim, hidden_dim).to(device)
        agent.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print("✅ Model loaded successfully")
    else:
        print(f"\n🧠 Creating new model...")
        # Pre-train predictor first
        print("   Pre-training predictor...")
        X_train, y_train = prepare_pretraining_data(env)
        dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        predictor = KANPredictor(obs_dim, hidden_dim).to(device)
        optimizer_pred = optim.Adam(predictor.parameters(), lr=0.0001)
        criterion = nn.MSELoss()
        
        for epoch in range(20):  # Quick pre-training
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer_pred.zero_grad()
                preds = predictor(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer_pred.step()
        
        # Create agent with pre-trained body
        agent = KANActorCritic(obs_dim, action_dim, hidden_dim, pretrained_body=predictor.body).to(device)
        print("✅ New model created")
    
    optimizer_rl = optim.Adam(agent.parameters(), lr=0.0003)
    
    # Training state
    state, _ = env.reset()
    memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
    episode_reward = 0.0
    cumulative_reward = 0.0
    logger_trading = TradingLogger()
    episode_start_balance = env.initial_balance
    
    # Timing
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=duration_hours)
    step = 0
    last_checkpoint_step = 0
    last_data_update = datetime.now()
    
    print(f"\n🚀 Starting real-time training...")
    print(f"   Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Press Ctrl+C to stop early and save model\n")
    
    try:
        while datetime.now() < end_time and not shutdown_flag:
            # Check if we should update data
            elapsed_minutes = (datetime.now() - last_data_update).total_seconds() / 60
            if elapsed_minutes >= data_update_interval:
                logger.info(f"🔄 Updating with live data (last update: {elapsed_minutes:.1f} minutes ago)...")
                new_df = data_updater.fetch_live_data()
                if new_df is not None:
                    old_data_points = len(env.data)
                    if data_updater.update_environment(env, new_df):
                        new_data_points = len(env.data)
                        new_bars = new_data_points - old_data_points
                        logger.info(f"✅ Data updated: {old_data_points} → {new_data_points} bars (+{new_bars} new bars)")
                        
                        # Reset environment to use new data (but preserve training state)
                        state, _ = env.reset()
                        episode_reward = 0.0
                        episode_start_balance = env.initial_balance
                        logger_trading = TradingLogger()
                        
                        # Clear memory to avoid stale data
                        memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
                    else:
                        logger.warning("⚠️  Failed to update environment, continuing with existing data")
                else:
                    logger.warning("⚠️  Failed to fetch new data, continuing with existing data")
                last_data_update = datetime.now()
            
            step += 1
            
            # Get action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action, log_prob, _ = agent.act(state_tensor, deterministic=False)
            
            # Step environment
            next_state, reward, done, _, _ = env.step(action)
            
            # Store in memory
            memory_states.append(torch.FloatTensor(state))
            memory_actions.append(torch.tensor(action))
            memory_logprobs.append(log_prob)
            memory_rewards.append(reward)
            
            state = next_state
            episode_reward += reward
            cumulative_reward += reward
            
            # PPO Update
            if step % 500 == 0 and len(memory_states) > 0:
                old_states = torch.stack(memory_states).to(device)
                old_actions = torch.stack(memory_actions).to(device)
                old_logprobs = torch.stack(memory_logprobs).to(device)
                
                # Calculate discounted rewards
                rewards = []
                discounted_reward = 0
                for r in reversed(memory_rewards):
                    discounted_reward = r + 0.99 * discounted_reward
                    rewards.insert(0, discounted_reward)
                rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
                
                # PPO optimization
                for _ in range(3):
                    logprobs, state_values, dist_entropy = agent.evaluate(old_states, old_actions)
                    state_values = torch.squeeze(state_values)
                    ratios = torch.exp(logprobs - old_logprobs.detach())
                    advantages = rewards - state_values.detach()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                    loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards) - 0.02 * dist_entropy
                    
                    optimizer_rl.zero_grad()
                    loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                    optimizer_rl.step()
                
                # Clear memory
                memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
                
                elapsed = (datetime.now() - start_time).total_seconds() / 3600
                remaining = duration_hours - elapsed
                print(f"   Step {step:6d} | Episode Reward: {episode_reward:7.2f} | "
                      f"Elapsed: {elapsed:4.1f}h | Remaining: {remaining:4.1f}h")
            
            # Progress report every 10000 steps
            if step % 10000 == 0:
                current_balance = env.balance
                if env.shares > 0:
                    current_price = env.data[env.current_step][0] if env.current_step < len(env.data) else 0
                    current_balance += env.shares * current_price
                
                total_profit = current_balance - episode_start_balance
                profit_pct = (total_profit / episode_start_balance * 100) if episode_start_balance > 0 else 0.0
                
                elapsed = (datetime.now() - start_time).total_seconds() / 3600
                remaining = duration_hours - elapsed
                
                print(f"\n{'='*70}")
                print(f"📊 PROGRESS REPORT - Step {step:,}")
                print(f"{'='*70}")
                print(f"   Elapsed Time:        {elapsed:>6.2f} hours")
                print(f"   Remaining Time:      {remaining:>6.2f} hours")
                print(f"   Cumulative Reward:   {cumulative_reward:>12.2f}")
                print(f"   Episode Reward:      {episode_reward:>12.2f}")
                print(f"   Account Balance:     ${current_balance:>11.2f}")
                print(f"   Total Profit:        ${total_profit:>11.2f} ({profit_pct:>6.2f}%)")
                print(f"   Next Data Update:    {data_update_interval - (datetime.now() - last_data_update).total_seconds() / 60:.1f} minutes")
                print(f"{'='*70}\n")
            
            # Save checkpoint
            if step - last_checkpoint_step >= checkpoint_interval:
                checkpoint_path = model_path.replace('.pth', f'_checkpoint_step{step}.pth')
                torch.save(agent.state_dict(), checkpoint_path)
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
                last_checkpoint_step = step
            
            # Episode finished
            if done:
                current_balance = env.balance
                if env.shares > 0:
                    current_price = env.data[env.current_step][0] if env.current_step < len(env.data) else 0
                    current_balance += env.shares * current_price
                episode_profit = current_balance - episode_start_balance
                episode_profit_pct = (episode_profit / episode_start_balance * 100) if episode_start_balance > 0 else 0.0
                
                logger.info(f"🏁 Episode Complete: Reward={episode_reward:.2f}, Profit=${episode_profit:.2f} ({episode_profit_pct:+.2f}%)")
                
                # Reset for next episode
                state, _ = env.reset()
                episode_reward = 0.0
                episode_start_balance = env.initial_balance
                logger_trading = TradingLogger()
            
            # Small sleep to prevent CPU spinning
            if step % 100 == 0:
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    
    finally:
        # Save final model
        print(f"\n💾 Saving final model to {model_path}...")
        torch.save(agent.state_dict(), model_path)
        print(f"✅ Model saved successfully")
        
        elapsed = (datetime.now() - start_time).total_seconds() / 3600
        print(f"\n📊 Training Summary:")
        print(f"   Total Steps: {step:,}")
        print(f"   Total Time: {elapsed:.2f} hours")
        print(f"   Cumulative Reward: {cumulative_reward:.2f}")
        print(f"   Final Model: {model_path}")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time training with live data')
    parser.add_argument('--ticker', type=str, default='ETH-USD', help='Ticker symbol')
    parser.add_argument('--mode', type=str, default='crypto', choices=['crypto', 'stock'], help='Trading mode')
    parser.add_argument('--interval', type=str, default='1h', help='Data interval')
    parser.add_argument('--window-size', type=int, default=30, help='Window size')
    parser.add_argument('--hidden-dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--duration', type=float, default=8.0, help='Training duration in hours')
    parser.add_argument('--model', type=str, default=None, help='Path to existing model (optional)')
    parser.add_argument('--checkpoint-interval', type=int, default=10000, help='Save checkpoint every N steps')
    parser.add_argument('--data-update-interval', type=int, default=15, help='Update data every N minutes')
    
    args = parser.parse_args()
    
    train_realtime(
        ticker=args.ticker,
        mode=args.mode,
        interval=args.interval,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        duration_hours=args.duration,
        model_path=args.model,
        checkpoint_interval=args.checkpoint_interval,
        data_update_interval=args.data_update_interval
    )
