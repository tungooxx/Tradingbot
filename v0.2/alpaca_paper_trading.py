"""
ALPACA PAPER TRADING BOT (v0.2)
================================
Automated paper trading using Alpaca API with hierarchical transformer model.

This script:
1. Loads trained hierarchical transformer model (transformer + execution agents + meta agent)
2. Gets historical data from yfinance (same as training, avoids Alpaca SIP subscription issues)
3. Generates trading signals using the hierarchical model
4. Executes paper trades via Alpaca API
5. Tracks performance

Usage:
    python alpaca_paper_trading.py --ticker TSLA --model-dir models_v0.2

Configuration:
    Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables

Note:
    Historical data is fetched from yfinance to avoid Alpaca SIP subscription requirements.
    Alpaca API is used only for order execution and account management.
"""

import os
import sys
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
from pathlib import Path
import pytz
import argparse

# Add v0.1 to path for TradingConfig
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v0.1'))
# Add v0.2 to path (current directory)
sys.path.insert(0, os.path.dirname(__file__))

from alpaca_client import AlpacaClient
from dapgio_improved import TradingConfig
from data_preprocessor import MultiTimescalePreprocessor
from regime_detector import EnhancedRegimeDetector
from transformer_encoders import MultiScaleTransformerEncoder
from execution_agents import ExecutionAgent, create_execution_agents
from meta_strategy_agent import MetaStrategyAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpaca_paper_trading_v0.2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AlpacaPaperTrading_v0.2")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AlpacaPaperTradingBot:
    """Automated paper trading bot using Alpaca API with hierarchical transformer model"""
    
    def __init__(self, ticker: str, model_dir: str = "models_v0.2", mode: str = "stock",
                 window_size: int = 30, position_size_pct: float = 0.10,
                 max_positions: int = 3):
        """
        Initialize paper trading bot
        
        Args:
            ticker: Stock symbol (e.g., 'TSLA', 'NVDA')
            model_dir: Directory containing trained models
            mode: 'stock' or 'crypto'
            window_size: Lookback window (must match training, default: 30)
            position_size_pct: Position size as % of buying power (0.0-1.0)
            max_positions: Maximum concurrent positions
        """
        self.ticker = ticker
        self.mode = mode
        self.window_size = window_size
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.model_dir = model_dir
        
        # Initialize Alpaca client
        logger.info("Connecting to Alpaca...")
        self.client = AlpacaClient(paper=True)
        
        # Load models
        logger.info(f"Loading models from {model_dir}...")
        self.transformer, self.execution_agents, self.meta_agent = self._load_models(model_dir)
        
        # Initialize preprocessor and regime detector
        self.preprocessor = MultiTimescalePreprocessor(ticker, mode=mode, logger=logger)
        self.regime_detector = EnhancedRegimeDetector(window=20, logger=logger)
        
        # Trading state
        self.last_signal_time = None
        self.signals_history = []
        self.entry_step = {}  # Track entry step for each ticker (for time_in_trade calculation)
        self.current_step = 0  # Track current step (for time_in_trade calculation)
        
        # Market hours (for stocks only, crypto is 24/7)
        self.market_timezone = pytz.timezone('US/Eastern')  # US market hours
        
        logger.info(f"Bot initialized for {ticker}")
        logger.info(f"Position size: {position_size_pct:.0%} of buying power")
        logger.info(f"Max positions: {max_positions}")
        if mode == "stock":
            logger.info("Market hours awareness: Enabled (9:30 AM - 4:00 PM ET, weekdays)")
        else:
            logger.info("Market hours awareness: Disabled (crypto runs 24/7)")
    
    def _load_models(self, model_dir: str):
        """Load all trained models"""
        # Determine input_dim: 8 for stocks (with market hours), 5 for crypto
        input_dim = 8 if self.mode == "stock" else 5
        
        # Load transformer
        transformer = MultiScaleTransformerEncoder(
            d_model=64, nhead=4, num_layers=2, dim_feedforward=256,
            dropout=0.1, input_dim=input_dim, mode=self.mode
        ).to(device)
        
        transformer_path = os.path.join(model_dir, "transformer_pretrained.pth")
        if os.path.exists(transformer_path):
            transformer.load_state_dict(torch.load(transformer_path, map_location=device))
            logger.info("Loaded pre-trained transformer")
            # Update input_dim from loaded transformer if available
            if hasattr(transformer, 'encoder_1d') and hasattr(transformer.encoder_1d, 'input_dim'):
                input_dim = transformer.encoder_1d.input_dim
                logger.info(f"Detected input_dim={input_dim} from loaded transformer")
        else:
            raise FileNotFoundError(f"Transformer not found at {transformer_path}")
        
        # Load execution agents
        execution_agents = {}
        strategies = ["TREND_FOLLOW", "MEAN_REVERT", "MOMENTUM", "RISK_OFF"]
        
        for strategy in strategies:
            # Try final model first, then regular
            for suffix in ["_final.pth", ".pth"]:
                agent_path = os.path.join(model_dir, f"execution_{strategy.lower()}{suffix}")
                if os.path.exists(agent_path):
                    agent = ExecutionAgent(
                        strategy_type=strategy,
                        d_model=64, nhead=4, num_layers=2, dim_feedforward=256,
                        dropout=0.1, input_dim=input_dim, mode=self.mode, hidden_dim=128
                    ).to(device)
                    agent.load_state_dict(torch.load(agent_path, map_location=device))
                    agent.eval()
                    execution_agents[strategy] = agent
                    logger.info(f"Loaded {strategy} execution agent")
                    break
        
        if not execution_agents:
            raise FileNotFoundError(f"No execution agents found in {model_dir}")
        
        # Load meta-strategy agent
        meta_agent = MetaStrategyAgent(
            d_model=64, nhead=4, num_layers=2, dim_feedforward=256,
            dropout=0.1, input_dim=input_dim, mode=self.mode, hidden_dim=128
        ).to(device)
        
        meta_path = os.path.join(model_dir, "meta_strategy_final.pth")
        if not os.path.exists(meta_path):
            meta_path = os.path.join(model_dir, "meta_strategy.pth")
        
        if os.path.exists(meta_path):
            meta_agent.load_state_dict(torch.load(meta_path, map_location=device))
            meta_agent.eval()
            logger.info("Loaded meta-strategy agent")
        else:
            raise FileNotFoundError(f"Meta-strategy agent not found in {model_dir}")
        
        return transformer, execution_agents, meta_agent
    
    def get_trading_signal(self) -> Optional[Dict]:
        """Get trading signal from hierarchical model"""
        try:
            # Process data (same as training)
            aligned_data, features = self.preprocessor.process(window_size=self.window_size)
            if not aligned_data or not features:
                logger.warning("Failed to get market data")
                return None
            
            # Get windowed features
            windowed = self.preprocessor.get_window_features(features, self.window_size, None)
            if not windowed:
                logger.warning("Failed to get windowed features")
                return None
            
            # Convert to tensors
            features_dict = {}
            for interval, feat_array in windowed.items():
                if len(feat_array) > 0:
                    feat_tensor = torch.FloatTensor(feat_array).unsqueeze(0).to(device)  # (1, window_size, feature_dim)
                    # Handle both "1wk" and "1w" keys - use "1w" for transformer
                    key = interval.replace("1wk", "1w")
                    features_dict[key] = feat_tensor
            
            # For stock mode, ensure we have the correct scales
            # The model expects: 1h, 1d, 1w (3 scales)
            # But if model was trained without 1h, it might only expect 1d, 1w (2 scales)
            # Check what scales we have and ensure consistency
            if self.mode == "stock":
                # Remove 1h if present (model might not have been trained with it)
                # Check if we have 1h - if model was trained without it, we need to remove it
                if "1h" in features_dict:
                    # Check if model expects 1h by checking the actual model weights
                    # For now, we'll keep 1h if it exists, but ensure we have 1d and 1w
                    pass
                
                # Ensure we have 1d and 1w (required)
                required_keys = ["1d", "1w"]
                for req_key in required_keys:
                    if req_key not in features_dict:
                        if features_dict:
                            sample_feat = next(iter(features_dict.values()))
                            features_dict[req_key] = torch.zeros_like(sample_feat)
                        else:
                            features_dict[req_key] = torch.zeros(1, self.window_size, 5, dtype=torch.float32).to(device)
                
                # If model was trained without 1h, remove it to match expected dimensions
                # We'll try without 1h first (most likely scenario)
                if "1h" in features_dict:
                    # Try removing 1h - the model might have been trained with only 1d and 1w
                    logger.debug(f"Found 1h data, but model might not expect it. Available scales: {list(features_dict.keys())}")
                    # Keep 1h for now - we'll see if the error persists
            
            # Get regime features
            regime_key = "1d" if "1d" in aligned_data else list(aligned_data.keys())[0]
            regime_df = aligned_data[regime_key]
            if 'Close' in regime_df.columns:
                close_prices = regime_df['Close'].values
            else:
                close_prices = regime_df.iloc[:, 0].values
            
            regime_features = torch.FloatTensor(
                self.regime_detector.get_regime_features(close_prices)
            ).unsqueeze(0).to(device)
            
            # Get action from meta agent (deterministic for trading)
            with torch.no_grad():
                meta_action, _, _ = self.meta_agent.act(features_dict, regime_features, deterministic=True)
            
            # Get selected strategy
            strategy_names = ["TREND_FOLLOW", "MEAN_REVERT", "MOMENTUM", "RISK_OFF"]
            selected_strategy = strategy_names[meta_action]
            
            # Get execution action from selected strategy's agent
            execution_agent = self.execution_agents.get(selected_strategy)
            if execution_agent is None:
                logger.warning(f"Execution agent for {selected_strategy} not found")
                return None
            
            # Get execution action
            with torch.no_grad():
                # Strategy features should be position state features (3 dims), not strategy one-hot
                # Position state: [has_position, no_position, price_ratio]
                # For paper trading, we don't have position state, so use default values
                positions = self.client.get_positions()
                has_position = 1.0 if self.ticker in positions else 0.0
                no_position = 1.0 if self.ticker not in positions else 0.0
                
                # Get entry price and current price if we have a position
                shares = 0.0
                entry_price = 0.0
                current_price = float(close_prices[-1])
                
                if self.ticker in positions:
                    position = positions[self.ticker]
                    shares = float(position.get('qty', 0))
                    entry_price = float(position.get('avg_entry_price', 0))
                    price_ratio = current_price / entry_price if entry_price > 0 else 1.0
                else:
                    price_ratio = 1.0
                
                strategy_features = torch.tensor([[has_position, no_position, price_ratio]], dtype=torch.float32).to(device)
                
                # HYBRID INPUT: Calculate precision features (hourly_return, rsi, position_status, unrealized_pnl)
                precision_feat = execution_agent._get_precision_features(
                    features_dict, shares, entry_price, current_price
                ).unsqueeze(0).to(device)
                
                # HYBRID INPUT: Calculate local features (position_status, unrealized_pnl, time_in_trade, relative_price)
                local_feat = self._calculate_local_features(
                    shares, entry_price, current_price, close_prices, self.window_size
                ).to(device)
                
                exec_action, _, _ = execution_agent.act(
                    features_dict, strategy_features, precision_feat, local_feat, deterministic=True
                )
            
            # CRITICAL: Check if action masking worked correctly
            # Get actual position status to verify action is valid
            positions = self.client.get_positions()
            has_position = self.ticker in positions
            
            # Check if action is invalid (action masking should have prevented this)
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            intended_action_name = action_map.get(exec_action, 'HOLD')
            
            # Validate action against actual position
            if exec_action == 1 and has_position:  # BUY when already in position
                logger.warning(f"Action mismatch! Intended: {intended_action_name}, but already have position")
                logger.warning(f"    Invalid action (action masking should prevent this)")
                logger.warning(f"    Position status from local_feat: {local_feat[0, 0].item():.2f} (should be 1.0)")
                logger.warning(f"    Actual position: {has_position} (ticker in positions)")
            elif exec_action == 2 and not has_position:  # SELL when no position
                logger.warning(f"Action mismatch! Intended: {intended_action_name}, but no position to sell")
                logger.warning(f"    Invalid action (action masking should prevent this)")
                logger.warning(f"    Position status from local_feat: {local_feat[0, 0].item():.2f} (should be 0.0)")
                logger.warning(f"    Actual position: {has_position} (ticker not in positions)")
            
            # Map action to signal
            signal = action_map.get(exec_action, 'HOLD')
            
            # Get confidence (from meta agent)
            with torch.no_grad():
                confidence_probs = self.meta_agent.get_action_confidence(features_dict, regime_features)
                confidence = float(np.max(confidence_probs))
            
            return {
                'ticker': self.ticker,
                'signal': signal,
                'execution_action': exec_action,
                'strategy': selected_strategy,
                'confidence': confidence,
                'price': current_price,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting signal: {e}", exc_info=True)
            return None
    
    def _calculate_local_features(self, shares: float, entry_price: float, current_price: float,
                                  close_prices: np.ndarray, window_size: int) -> torch.Tensor:
        """
        Calculate 4 normalized 'Local Features' for Hybrid Input architecture.
        
        Args:
            shares: Current position size (0 if no position)
            entry_price: Entry price of position (0 if no position)
            current_price: Current market price
            close_prices: Array of recent close prices (for MA calculation)
            window_size: Window size for normalization
        
        Returns:
            Local features tensor (1, 4): [position_status, unrealized_pnl, time_in_trade, relative_price]
        """
        # 1. Position Status: 0 if no position, 1 if holding
        position_status = 1.0 if shares > 0 else 0.0
        
        # 2. Unrealized PnL: (Current price / Entry price - 1) scaled by 10
        if shares > 0 and entry_price > 0:
            unrealized_pnl = ((current_price / entry_price) - 1.0) * 10.0
            # Normalize to [-1, 1] range
            unrealized_pnl = np.clip(unrealized_pnl, -1.0, 1.0)
        else:
            unrealized_pnl = 0.0
        
        # 3. Time in Trade: Number of steps since last buy, normalized by window size
        # For paper trading, we track entry step in self.entry_step dict
        if shares > 0 and self.ticker in self.entry_step:
            entry_step = self.entry_step[self.ticker]
            time_in_trade = (self.current_step - entry_step) / float(window_size)
            # Normalize to [0, 1] range
            time_in_trade = np.clip(time_in_trade, 0.0, 1.0)
        else:
            time_in_trade = 0.0
        
        # 4. Relative Price: Current close relative to 20-period Moving Average
        if len(close_prices) >= 20:
            # Calculate 20-period MA
            ma_window = close_prices[-20:]
            ma_20 = np.mean(ma_window)
            if ma_20 > 0:
                relative_price = (current_price / ma_20) - 1.0
                # Normalize to [-1, 1] range (scale by 5 to make it more sensitive)
                relative_price = np.clip(relative_price * 5.0, -1.0, 1.0)
            else:
                relative_price = 0.0
        else:
            relative_price = 0.0
        
        return torch.tensor([[position_status, unrealized_pnl, time_in_trade, relative_price]], 
                           dtype=torch.float32)
    
    def calculate_position_size(self, price: float) -> float:
        """Calculate position size based on buying power"""
        try:
            account = self.client.get_account()
            buying_power = account['buying_power']
            
            # Calculate position value
            position_value = buying_power * self.position_size_pct
            
            # Calculate shares
            shares = position_value / price
            
            # Round down to whole shares
            shares = int(shares)
            
            return max(1, shares)  # At least 1 share
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1
    
    def is_market_open(self) -> bool:
        """
        Check if market is currently open (stocks only).
        Crypto markets are always open (24/7).
        
        Returns:
            True if market is open, False otherwise
        """
        if self.mode == "crypto":
            return True  # Crypto markets are always open
        
        try:
            # Get current time in market timezone
            now = datetime.now(self.market_timezone)
            current_time = now.time()
            weekday = now.weekday()  # 0=Monday, 6=Sunday
            
            # Market closed on weekends
            if weekday >= 5:  # Saturday or Sunday
                return False
            
            # Market hours: 9:30 AM to 4:00 PM ET
            market_open = datetime.strptime("09:30", "%H:%M").time()
            market_close = datetime.strptime("16:00", "%H:%M").time()
            
            # Check if within market hours
            is_open = market_open <= current_time <= market_close
            return is_open
            
        except Exception as e:
            logger.warning(f"Error checking market hours: {e}, assuming market is open")
            return True  # Default to open if we can't determine
    
    def should_trade(self, signal: Dict) -> bool:
        """Check if we should execute the trade"""
        try:
            # MARKET HOURS CHECK: For stocks, only trade when market is open
            if self.mode == "stock" and not self.is_market_open():
                logger.info("Market is closed, skipping trade")
                return False
            
            # Only trade BUY or SELL (not HOLD)
            if signal['signal'] == 'HOLD':
                return False
            
            # Check if we have enough positions
            positions = self.client.get_positions()
            if len(positions) >= self.max_positions and signal['signal'] == 'BUY':
                logger.info(f"Max positions reached ({self.max_positions}), skipping BUY")
                return False
            
            # Check if we already have position
            if self.ticker in positions and signal['signal'] == 'BUY':
                logger.info(f"Already have position in {self.ticker}, skipping BUY")
                return False
            
            # Check if we have position to sell
            if signal['signal'] == 'SELL' and self.ticker not in positions:
                logger.info(f"No position in {self.ticker} to sell")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error checking trade conditions: {e}")
            return False
    
    def execute_signal(self, signal: Dict) -> bool:
        """Execute trading signal"""
        try:
            if not self.should_trade(signal):
                return False
            
            action = signal['signal']
            price = signal['price']
            
            if action == 'BUY':
                # Calculate position size
                shares = self.calculate_position_size(price)
                
                # Place order
                logger.info(f"Placing BUY order: {shares} shares of {self.ticker} @ ${price:.2f} (Strategy: {signal['strategy']})")
                order = self.client.place_order(
                    symbol=self.ticker,
                    qty=shares,
                    side='buy',
                    order_type='market'
                )
                
                # HYBRID INPUT: Track entry step for time_in_trade calculation
                self.entry_step[self.ticker] = self.current_step
                
                logger.info(f"Order placed: {order['id']} - Status: {order['status']}")
                return True
            
            elif action == 'SELL':
                # Get current position
                positions = self.client.get_positions()
                if self.ticker not in positions:
                    return False
                
                position = positions[self.ticker]
                shares = int(position['qty'])
                
                if shares <= 0:
                    return False
                
                # Place order
                logger.info(f"Placing SELL order: {shares} shares of {self.ticker} @ ${price:.2f} (Strategy: {signal['strategy']})")
                order = self.client.place_order(
                    symbol=self.ticker,
                    qty=shares,
                    side='sell',
                    order_type='market'
                )
                
                # HYBRID INPUT: Clear entry step when position is closed
                if self.ticker in self.entry_step:
                    del self.entry_step[self.ticker]
                
                # Calculate P&L
                entry_price = position['avg_entry_price']
                pnl = (price - entry_price) * shares
                pnl_pct = ((price - entry_price) / entry_price) * 100
                
                logger.info(f"Order placed: {order['id']} - Status: {order['status']}")
                logger.info(f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
    
    def run_once(self):
        """Run one trading cycle"""
        try:
            logger.info(f"Getting signal for {self.ticker}...")
            
            # HYBRID INPUT: Increment current step for time_in_trade calculation
            self.current_step += 1
            
            # Get signal
            signal = self.get_trading_signal()
            if signal is None:
                logger.warning("No signal generated")
                return
            
            logger.info(
                f"Signal: {signal['signal']} | "
                f"Strategy: {signal['strategy']} | "
                f"Confidence: {signal['confidence']:.2%} | "
                f"Price: ${signal['price']:.2f}"
            )
            
            # Execute signal
            if signal['signal'] != 'HOLD':
                self.execute_signal(signal)
            else:
                logger.info("Signal is HOLD, no action taken")
            
            # Store signal
            self.signals_history.append(signal)
            self.last_signal_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in run_once: {e}", exc_info=True)
    
    def run_continuous(self, check_interval_minutes: int = 60, scheduled_times: Optional[List[str]] = None):
        """
        Run continuously, checking for signals periodically
        
        Args:
            check_interval_minutes: Minutes between checks (if scheduled_times is None)
            scheduled_times: List of times to check (e.g., ['03:30', '09:30', '15:30'])
                            Times should be in 24-hour format (HH:MM)
        """
        if scheduled_times:
            logger.info(f"Starting scheduled trading bot (checking at: {', '.join(scheduled_times)})")
        else:
            logger.info(f"Starting continuous trading bot (checking every {check_interval_minutes} minutes)")
        logger.info("Press Ctrl+C to stop")
        
        try:
            while True:
                if scheduled_times:
                    # Calculate next scheduled time
                    now = datetime.now(pytz.timezone('US/Eastern'))  # Alpaca uses US market hours
                    next_time = self._get_next_scheduled_time(now, scheduled_times)
                    
                    if next_time:
                        wait_seconds = (next_time - now).total_seconds()
                        if wait_seconds > 0:
                            wait_minutes = wait_seconds / 60
                            logger.info(f"Next check at {next_time.strftime('%H:%M:%S')} (in {wait_minutes:.1f} minutes)")
                            time.sleep(wait_seconds)
                        else:
                            # Time has passed, check immediately
                            self.run_once()
                            # Wait a bit to avoid immediate re-check
                            time.sleep(60)
                    else:
                        # No more scheduled times today, wait until tomorrow
                        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                        wait_seconds = (tomorrow - now).total_seconds()
                        logger.info(f"No more scheduled times today. Waiting until tomorrow...")
                        time.sleep(min(wait_seconds, 3600))  # Check at most every hour
                else:
                    # Simple interval-based checking
                    self.run_once()
                    logger.info(f"Waiting {check_interval_minutes} minutes until next check...")
                    time.sleep(check_interval_minutes * 60)
        except KeyboardInterrupt:
            logger.info("Stopping bot...")
        finally:
            logger.info("Bot stopped")
    
    def _get_next_scheduled_time(self, now: datetime, scheduled_times: List[str]) -> Optional[datetime]:
        """Get the next scheduled time from now"""
        try:
            # Parse scheduled times
            time_objs = []
            for time_str in scheduled_times:
                hour, minute = map(int, time_str.split(':'))
                time_objs.append((hour, minute))
            
            # Sort by time
            time_objs.sort()
            
            # Find next time
            for hour, minute in time_objs:
                # Create datetime for today at this time
                scheduled = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                
                # If this time hasn't passed today, return it
                if scheduled > now:
                    return scheduled
            
            # If all times passed, return first time tomorrow
            if time_objs:
                hour, minute = time_objs[0]
                tomorrow = (now + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
                return tomorrow
            
            return None
        except Exception as e:
            logger.error(f"Error calculating next scheduled time: {e}")
            return None


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Alpaca Paper Trading Bot (v0.2)')
    parser.add_argument('--ticker', type=str, default='TSLA', help='Stock symbol')
    parser.add_argument('--model-dir', type=str, default='models_v0.2', help='Directory containing trained models')
    parser.add_argument('--mode', type=str, default='stock', choices=['stock', 'crypto'], help='Trading mode')
    parser.add_argument('--window-size', type=int, default=30, help='Lookback window size (must match training)')
    parser.add_argument('--position-size', type=float, default=0.10, help='Position size as % of buying power (0.0-1.0)')
    parser.add_argument('--max-positions', type=int, default=3, help='Maximum concurrent positions')
    parser.add_argument('--check-interval', type=int, default=60, help='Minutes between checks (for continuous mode)')
    parser.add_argument('--scheduled-times', type=str, nargs='+', help='Scheduled times to check (e.g., "09:30" "15:30")')
    parser.add_argument('--once', action='store_true', help='Run once and exit (for testing)')
    
    args = parser.parse_args()
    
    # Check API keys
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_SECRET_KEY'):
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("\nExample (PowerShell):")
        print('  $env:ALPACA_API_KEY = "YOUR_KEY"')
        print('  $env:ALPACA_SECRET_KEY = "YOUR_SECRET"')
        sys.exit(1)
    
    try:
        # Initialize bot
        bot = AlpacaPaperTradingBot(
            ticker=args.ticker,
            model_dir=args.model_dir,
            mode=args.mode,
            window_size=args.window_size,
            position_size_pct=args.position_size,
            max_positions=args.max_positions
        )
        
        # Get account info
        account = bot.client.get_account()
        logger.info(f"\nPaper Trading Account:")
        logger.info(f"  Balance: ${account['balance']:,.2f}")
        logger.info(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
        logger.info(f"  Buying Power: ${account['buying_power']:,.2f}")
        
        # Run
        if args.once:
            bot.run_once()
        else:
            scheduled_times = args.scheduled_times if args.scheduled_times else None
            bot.run_continuous(
                check_interval_minutes=args.check_interval,
                scheduled_times=scheduled_times
            )
    
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()



