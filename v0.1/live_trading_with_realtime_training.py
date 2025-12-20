"""
LIVE TRADING WITH REAL-TIME TRAINING
=====================================
Combines live trading (Alpaca) with real-time model training.

This system:
1. Trades live via Alpaca API
2. Collects experience (state, action, reward, next_state) from each trade
3. Periodically retrains the model with collected experience
4. Updates the model in-place for future trades
5. Continuously adapts to market conditions

Usage:
    python live_trading_with_realtime_training.py --ticker QQQ --mode stock --interval 1d
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
from typing import Optional, Dict, List, Tuple
import argparse
import logging
from collections import deque
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from alpaca_client import AlpacaClient
from database import TradingDB
from dapgio_improved import (
    StockTradingEnv,
    KANPredictor,
    KANActorCritic,
    TradingConfig,
    setup_logging,
    TradingLogger
)
from predict import get_prediction
from trading_constraints_env import create_constrained_env

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading_realtime_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("LiveTradingRealtimeTraining")

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_flag
    logger.info("\nShutdown signal received. Saving model and exiting...")
    shutdown_flag = True


signal.signal(signal.SIGINT, signal_handler)


class ExperienceBuffer:
    """Buffer to store trading experience for training"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool, info: Dict):
        """Add experience tuple"""
        self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'info': info.copy()
        })
    
    def sample(self, batch_size: int) -> List[Dict]:
        """Sample random batch of experiences"""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()


class LiveTradingWithRealtimeTraining:
    """
    Live trading bot that continuously trains on real trading experience.
    """
    
    def __init__(
        self,
        ticker: str,
        model_path: str,
        mode: str = "stock",
        interval: str = "1d",
        window_size: int = 30,
        hidden_dim: int = 32,
        min_confidence: float = 0.50,
        max_positions: int = 3,
        position_size_pct: float = 0.10,
        training_interval_trades: int = 10,  # Retrain every N trades
        training_batch_size: int = 32,
        learning_rate: float = 0.0001
    ):
        """
        Initialize live trading with real-time training.
        
        Args:
            ticker: Stock/crypto symbol
            model_path: Path to trained model
            mode: 'stock' or 'crypto'
            interval: '1d' for stocks, '1h'/'4h' for crypto
            window_size: Lookback window
            hidden_dim: Model hidden dimension
            min_confidence: Minimum confidence to trade
            max_positions: Maximum concurrent positions
            position_size_pct: Position size as % of account
            training_interval_trades: Retrain every N trades
            training_batch_size: Batch size for training
            learning_rate: Learning rate for training
        """
        self.ticker = ticker
        self.model_path = model_path
        self.mode = mode
        self.interval = interval
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.min_confidence = min_confidence
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.training_interval_trades = training_interval_trades
        self.training_batch_size = training_batch_size
        self.learning_rate = learning_rate
        
        # Initialize Alpaca client
        logger.info("Connecting to Alpaca...")
        self.client = AlpacaClient(paper=True)
        
        # Initialize database
        self.db = TradingDB("trading_bot.db")
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model
        logger.info(f"Loading model from {model_path}...")
        self.model, self.predictor = self._load_model()
        
        # Create environment for experience collection
        config = TradingConfig()
        config.mode = mode
        config.interval = interval
        config.ticker = ticker
        
        self.env = create_constrained_env(
            ticker=ticker,
            window_size=window_size,
            config=config,
            min_confidence=min_confidence,
            max_positions=max_positions,
            position_size_pct=position_size_pct,
            logger=logger
        )
        
        # Experience buffer
        self.experience_buffer = ExperienceBuffer(max_size=10000)
        
        # Training state
        self.trade_count = 0
        self.last_training_time = datetime.now()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Performance tracking
        self.total_reward = 0.0
        self.total_profit = 0.0  # Realized P&L from closed trades
        self.total_trades_executed = 0
        self.initial_portfolio_value = None
        
        # Trading state
        self.last_signal_time = None
        self.current_state = None
        self.current_action = None
        
        logger.info("Live trading with real-time training initialized")
        logger.info(f"  Training interval: Every {training_interval_trades} trades")
        logger.info(f"  Batch size: {training_batch_size}")
        logger.info(f"  Learning rate: {learning_rate}")
    
    def _load_model(self) -> Tuple[KANActorCritic, Optional[KANPredictor]]:
        """Load trained model"""
        try:
            obs_dim = self.window_size * 5  # window_size * features
            action_dim = 3
            
            # Load model
            agent = KANActorCritic(obs_dim, action_dim, self.hidden_dim).to(self.device)
            
            if os.path.exists(self.model_path):
                state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                if 'actor_head.weight' in state_dict:
                    agent.load_state_dict(state_dict)
                    logger.info("Loaded KANActorCritic model")
                else:
                    logger.warning("Model format not recognized, using random initialization")
            else:
                logger.warning(f"Model file not found: {self.model_path}, using random initialization")
            
            agent.eval()  # Start in eval mode, switch to train mode during training
            
            # Predictor is part of agent body, not needed separately
            return agent, None
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def _save_model(self):
        """Save model checkpoint"""
        try:
            torch.save(self.model.state_dict(), self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def get_trading_signal(self) -> Optional[Dict]:
        """Get trading signal from already-loaded model"""
        try:
            import torch.nn.functional as F
            
            # Get historical data
            df = self._get_historical_data()
            if df is None or len(df) < self.window_size:
                logger.warning("Insufficient data for signal")
                return None
            
            # Fix MultiIndex columns (same as dapgio_improved.py)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Feature engineering (same as predict.py)
            df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Vol_Norm'] = df['Volume'] / df['Volume'].rolling(20).mean()
            
            # Technical indicators
            import pandas_ta as ta
            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.dropna(inplace=True)
            
            # Normalize (must match training)
            df["RSI_14"] = df["RSI_14"] / 100.0
            
            # Rolling normalization for MACD
            if self.interval == "1h":
                rolling_window = 252 * 24
            elif self.interval == "4h":
                rolling_window = 252 * 6
            else:
                rolling_window = 252
            
            macd_mean = df["MACD_12_26_9"].rolling(rolling_window, min_periods=1).mean()
            macd_std = df["MACD_12_26_9"].rolling(rolling_window, min_periods=1).std()
            
            # Handle case where std is 0 or NaN (all values same in window)
            macd_std = macd_std.fillna(1e-7)  # Replace NaN std with small value
            macd_std = macd_std.replace(0.0, 1e-7)  # Replace zero std with small value
            
            df["MACD_12_26_9"] = (df["MACD_12_26_9"] - macd_mean) / macd_std
            
            # Fill any remaining NaN values (can occur at the beginning of the series)
            df["MACD_12_26_9"].fillna(method='ffill', inplace=True)
            df["MACD_12_26_9"].fillna(method='bfill', inplace=True)
            df["MACD_12_26_9"].fillna(0.0, inplace=True)
            
            # Final dropna to remove any rows that still have NaN in other columns
            df.dropna(inplace=True)
            
            features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9"]
            
            # Prepare observation
            window = df[features].iloc[-self.window_size:].values
            obs = window.flatten().astype(np.float32)
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get prediction from loaded model
            self.model.eval()
            with torch.no_grad():
                action, log_prob, state_value = self.model.act(obs_tensor, deterministic=True)
                # Get confidence probabilities
                features_out = self.model.body(obs_tensor)
                logits = self.model.actor_head(features_out)
                confidence_probs = F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()
            
            action_names = ["HOLD", "BUY", "SELL"]
            action_name = action_names[int(action)]
            max_confidence = float(confidence_probs.max())
            current_price = float(df['Close'].iloc[-1])
            
            signal = {
                'signal': action_name,
                'action_id': int(action),
                'confidence': max_confidence,
                'price': current_price,
                'confidence_breakdown': {
                    'HOLD': float(confidence_probs[0]),
                    'BUY': float(confidence_probs[1]),
                    'SELL': float(confidence_probs[2])
                }
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Error getting signal: {e}", exc_info=True)
            return None
    
    def _get_historical_data(self) -> Optional[pd.DataFrame]:
        """Get historical data from yfinance"""
        try:
            if self.mode == 'crypto':
                interval = self.interval
                period = "60d" if interval == "1h" else "120d"
            else:
                interval = "1d"
                period = "1y"
            
            df = yf.download(self.ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            
            if df.empty:
                return None
            
            # Process data (same as training)
            df = df.dropna()
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def _calculate_reward_from_trade(self, signal: Dict, executed: bool, pnl: Optional[float] = None) -> float:
        """Calculate reward from trade outcome"""
        if not executed:
            # No penalty for not executing - model should wait for good opportunities
            # The environment's reward function already handles HOLD actions
            return 0.0  # Neutral reward for not executing
        
        if pnl is None:
            # Use signal confidence as proxy reward
            return signal.get('confidence', 0.0) * 0.1
        
        # Reward based on P&L
        if pnl > 0:
            return pnl / 1000.0  # Scale reward
        else:
            return pnl / 1000.0  # Negative reward for loss
    
    def collect_experience(self, state: np.ndarray, action: int, reward: float, 
                           next_state: np.ndarray, done: bool, info: Dict):
        """Collect trading experience"""
        self.experience_buffer.add(state, action, reward, next_state, done, info)
    
    def train_on_experience(self):
        """Train model on collected experience"""
        if len(self.experience_buffer) < self.training_batch_size:
            logger.debug(f"Not enough experience ({len(self.experience_buffer)} < {self.training_batch_size})")
            return
        
        logger.info(f"Training on {len(self.experience_buffer)} experiences...")
        
        # Sample batch
        batch = self.experience_buffer.sample(self.training_batch_size)
        
        # Prepare data
        states = torch.FloatTensor([exp['state'] for exp in batch]).to(self.device)
        actions = torch.LongTensor([exp['action'] for exp in batch]).to(self.device)
        rewards = torch.FloatTensor([exp['reward'] for exp in batch]).to(self.device)
        next_states = torch.FloatTensor([exp['next_state'] for exp in batch]).to(self.device)
        dones = torch.BoolTensor([exp['done'] for exp in batch]).to(self.device)
        
        # Switch to training mode
        self.model.train()
        
        # PPO-style update (simplified)
        # Get current policy
        action_probs, values = self.model(states)
        action_log_probs = torch.log(action_probs + 1e-8)
        selected_log_probs = action_log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Calculate advantages (simplified - using rewards directly)
        advantages = rewards
        
        # Policy loss (negative log prob weighted by advantage)
        policy_loss = -(selected_log_probs * advantages).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), rewards)
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        # Switch back to eval mode
        self.model.eval()
        
        logger.info(f"Training complete. Loss: {loss.item():.4f} (policy: {policy_loss.item():.4f}, value: {value_loss.item():.4f})")
        
        # Save model
        self._save_model()
    
    def execute_signal(self, signal: Dict) -> Tuple[bool, Optional[float]]:
        """Execute trading signal and return (success, pnl)"""
        try:
            # Convert signal to action
            action_map = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
            action = action_map.get(signal['signal'], 0)
            
            # Get current positions
            positions = self.client.get_positions()
            has_position = self.ticker in positions
            
            # Execute trade
            if signal['signal'] == 'BUY' and not has_position:
                if len(positions) >= self.max_positions:
                    logger.info(f"Max positions reached ({self.max_positions}), skipping BUY")
                    return False, None
                
                # Calculate position size
                account = self.client.get_account()
                buying_power = float(account['buying_power'])
                position_value = buying_power * self.position_size_pct
                price = signal['price']
                shares = int(position_value / price)
                
                if shares < 1:
                    logger.warning("Insufficient buying power")
                    return False, None
                
                # Place order
                order = self.client.place_order(
                    symbol=self.ticker,
                    qty=shares,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                
                logger.info(f"BUY order placed: {shares} shares @ ${price:.2f}")
                return True, None
            
            elif signal['signal'] == 'SELL' and has_position:
                position = positions[self.ticker]
                shares = int(position['qty'])
                price = signal['price']
                
                # Place order
                order = self.client.place_order(
                    symbol=self.ticker,
                    qty=shares,
                    side='sell',
                    type='market',
                    time_in_force='day'
                )
                
                # Calculate P&L
                entry_price = float(position['avg_entry_price'])
                pnl = (price - entry_price) * shares
                
                logger.info(f"SELL order placed: {shares} shares @ ${price:.2f}, P&L: ${pnl:.2f}")
                return True, pnl
            
            return False, None
            
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False, None
    
    def run_once(self):
        """Run one trading cycle with experience collection"""
        try:
            logger.info(f"Getting signal for {self.ticker}...")
            
            # Get current state from environment
            if self.current_state is None:
                obs, info = self.env.reset()
                self.current_state = obs
            
            # Get trading signal
            signal = self.get_trading_signal()
            if signal is None:
                logger.warning("No signal generated")
                return
            
            logger.info(f"Signal: {signal['signal']} | Confidence: {signal['confidence']:.2%} | Price: ${signal['price']:.2f}")
            
            # Convert signal to action
            action_map = {'HOLD': 0, 'BUY': 1, 'SELL': 2}
            action = action_map.get(signal['signal'], 0)
            action_confidence = signal['confidence']
            
            # Execute trade if confident enough
            executed = False
            pnl = None
            if signal['confidence'] >= self.min_confidence:
                executed, pnl = self.execute_signal(signal)
                if executed:
                    self.trade_count += 1
            else:
                logger.info(f"Signal confidence {signal['confidence']:.2%} below threshold {self.min_confidence:.0%}, skipping")
            
            # Step environment to get next state
            # The environment's reward function handles all rewards (including HOLD)
            next_obs, reward, done, truncated, info = self.env.step(action, action_confidence=action_confidence)
            
            # Use environment reward directly (it already handles HOLD/BUY/SELL properly)
            # Only add trade-specific bonus if trade was executed
            if executed and pnl is not None:
                # Add small bonus/penalty based on actual P&L
                pnl_reward = pnl / 100.0  # Scale P&L to reward (e.g., $100 profit = +1 reward)
                reward += pnl_reward
            
            # Update performance tracking
            self.total_reward += reward
            if pnl is not None:
                self.total_profit += pnl
                self.total_trades_executed += 1
            
            # Collect experience (use environment reward, not separate trade_reward)
            self.collect_experience(
                state=self.current_state,
                action=action,
                reward=reward,  # Use environment reward (already includes trade bonuses)
                next_state=next_obs,
                done=done,
                info=info
            )
            
            # Update state
            self.current_state = next_obs
            
            # Get current portfolio value from Alpaca
            try:
                account = self.client.get_account()
                current_portfolio_value = float(account['portfolio_value'])
                current_balance = float(account['balance'])
                
                # Initialize on first run
                if self.initial_portfolio_value is None:
                    self.initial_portfolio_value = current_portfolio_value
                
                # Calculate total return
                total_return = current_portfolio_value - self.initial_portfolio_value
                total_return_pct = (total_return / self.initial_portfolio_value * 100) if self.initial_portfolio_value > 0 else 0.0
                
                # Display performance stats
                logger.info(f"Performance Stats:")
                logger.info(f"   Total Reward: {self.total_reward:.4f}")
                logger.info(f"   Realized Profit (closed trades): ${self.total_profit:.2f}")
                logger.info(f"   Portfolio Value: ${current_portfolio_value:.2f}")
                logger.info(f"   Total Return: ${total_return:.2f} ({total_return_pct:+.2f}%)")
                logger.info(f"   Trades Executed: {self.total_trades_executed}")
                if self.total_trades_executed > 0:
                    avg_profit = self.total_profit / self.total_trades_executed
                    logger.info(f"   Avg Profit per Trade: ${avg_profit:.2f}")
            except Exception as e:
                logger.warning(f"Could not fetch portfolio value: {e}")
                logger.info(f"Performance Stats:")
                logger.info(f"   Total Reward: {self.total_reward:.4f}")
                logger.info(f"   Realized Profit: ${self.total_profit:.2f}")
                logger.info(f"   Trades Executed: {self.total_trades_executed}")
            
            # Train if enough trades
            if self.trade_count > 0 and self.trade_count % self.training_interval_trades == 0:
                logger.info(f"Retraining after {self.trade_count} trades...")
                self.train_on_experience()
            
            # Save signal to database
            self.db.add_prediction(
                ticker=self.ticker,
                prediction_date=datetime.now().strftime('%Y-%m-%d'),
                predicted_price=signal['price'],
                confidence=signal['confidence'],
                signal=signal['signal'],
                model_version="v0.1_realtime"
            )
            
            self.last_signal_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in run_once: {e}", exc_info=True)
    
    def run_continuous(self, check_interval_minutes: int = 60):
        """Run continuously, trading and training in real-time"""
        logger.info(f"Starting live trading with real-time training")
        logger.info(f"  Check interval: {check_interval_minutes} minutes")
        logger.info(f"  Training every {self.training_interval_trades} trades")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while not shutdown_flag:
                self.run_once()
                
                logger.info(f"Waiting {check_interval_minutes} minutes until next check...")
                logger.info(f"  Total trades attempted: {self.trade_count}")
                logger.info(f"  Total trades executed: {self.total_trades_executed}")
                logger.info(f"  Total Reward: {self.total_reward:.4f}")
                logger.info(f"  Realized Profit: ${self.total_profit:.2f}")
                
                # Show portfolio value if available
                try:
                    account = self.client.get_account()
                    current_portfolio_value = float(account['portfolio_value'])
                    if self.initial_portfolio_value is not None:
                        total_return = current_portfolio_value - self.initial_portfolio_value
                        total_return_pct = (total_return / self.initial_portfolio_value * 100) if self.initial_portfolio_value > 0 else 0.0
                        logger.info(f"  Portfolio Value: ${current_portfolio_value:.2f} (Return: ${total_return:.2f}, {total_return_pct:+.2f}%)")
                except:
                    pass
                
                logger.info(f"  Experience buffer: {len(self.experience_buffer)}")
                
                # Wait for next check
                for _ in range(check_interval_minutes * 60):
                    if shutdown_flag:
                        break
                    time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Stopping...")
        finally:
            # Final training and save
            if len(self.experience_buffer) > 0:
                logger.info("Final training before shutdown...")
                self.train_on_experience()
            self._save_model()
            
            # Final performance summary
            logger.info("\n" + "=" * 60)
            logger.info("FINAL PERFORMANCE SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total Trades Attempted: {self.trade_count}")
            logger.info(f"Total Trades Executed: {self.total_trades_executed}")
            logger.info(f"Total Reward: {self.total_reward:.4f}")
            logger.info(f"Realized Profit (closed trades): ${self.total_profit:.2f}")
            
            # Get final portfolio value
            try:
                account = self.client.get_account()
                final_portfolio_value = float(account['portfolio_value'])
                if self.initial_portfolio_value is not None:
                    total_return = final_portfolio_value - self.initial_portfolio_value
                    total_return_pct = (total_return / self.initial_portfolio_value * 100) if self.initial_portfolio_value > 0 else 0.0
                    logger.info(f"Initial Portfolio Value: ${self.initial_portfolio_value:.2f}")
                    logger.info(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
                    logger.info(f"Total Return: ${total_return:.2f} ({total_return_pct:+.2f}%)")
            except Exception as e:
                logger.warning(f"Could not fetch final portfolio value: {e}")
            
            if self.total_trades_executed > 0:
                avg_profit = self.total_profit / self.total_trades_executed
                logger.info(f"Average Profit per Trade: ${avg_profit:.2f}")
            logger.info(f"Experience Collected: {len(self.experience_buffer)}")
            logger.info("=" * 60)
            logger.info("Shutdown complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live trading with real-time training")
    parser.add_argument("--ticker", type=str, default="QQQ", help="Ticker symbol")
    parser.add_argument("--mode", type=str, default="stock", choices=["stock", "crypto"], help="Trading mode")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval")
    parser.add_argument("--model", type=str, default="kan_agent_stock.pth", help="Model path")
    parser.add_argument("--window-size", type=int, default=30, help="Window size")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--min-confidence", type=float, default=0.50, help="Min confidence")
    parser.add_argument("--max-positions", type=int, default=3, help="Max positions")
    parser.add_argument("--position-size", type=float, default=0.10, help="Position size %")
    parser.add_argument("--training-interval", type=int, default=10, help="Train every N trades")
    parser.add_argument("--check-interval", type=int, default=60, help="Check interval (minutes)")
    
    args = parser.parse_args()
    
    bot = LiveTradingWithRealtimeTraining(
        ticker=args.ticker,
        model_path=args.model,
        mode=args.mode,
        interval=args.interval,
        window_size=args.window_size,
        hidden_dim=args.hidden_dim,
        min_confidence=args.min_confidence,
        max_positions=args.max_positions,
        position_size_pct=args.position_size,
        training_interval_trades=args.training_interval,
        training_batch_size=32,
        learning_rate=0.0001
    )
    
    bot.run_continuous(check_interval_minutes=args.check_interval)

