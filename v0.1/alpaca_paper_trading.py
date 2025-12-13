"""
ALPACA PAPER TRADING BOT
=========================
Automated paper trading using Alpaca API with your trained model.

This script:
1. Loads your trained model
2. Gets real-time data from Alpaca
3. Generates trading signals
4. Executes paper trades automatically
5. Tracks performance in database

Usage:
    python alpaca_paper_trading.py

Configuration:
    Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables
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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from alpaca_client import AlpacaClient
from database import TradingDB
from dapgio_improved import KANActorCritic, TradingConfig, setup_logging
from predict import get_prediction, KANPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('alpaca_paper_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AlpacaPaperTrading")


class AlpacaPaperTradingBot:
    """Automated paper trading bot using Alpaca API"""
    
    def __init__(self, ticker: str, model_path: str, mode: str = "stock",
                 interval: str = "1d", window_size: int = 30,
                 min_confidence: float = 0.70, max_positions: int = 3,
                 position_size_pct: float = 0.10):
        """
        Initialize paper trading bot
        
        Args:
            ticker: Stock symbol (e.g., 'NVDA', 'AAPL') or crypto (e.g., 'ETHUSD', 'ETH-USD')
            model_path: Path to trained model (.pth file)
            mode: 'stock' or 'crypto'
            interval: '1d' for stocks, '1h'/'4h' for crypto
            window_size: Lookback window (must match training)
            min_confidence: Minimum confidence to trade (0.0-1.0)
            max_positions: Maximum concurrent positions
            position_size_pct: Position size as % of buying power (0.0-1.0)
        """
        # Store original ticker for logging/display
        self.original_ticker = ticker
        
        # Convert crypto ticker format for Alpaca (ETH-USD -> ETHUSD)
        if mode == "crypto" and "-" in ticker:
            self.ticker = ticker.replace("-", "")
            logger.info(f"Converted ticker {ticker} to {self.ticker} for Alpaca API")
        else:
            self.ticker = ticker
        self.mode = mode
        self.interval = interval
        self.window_size = window_size
        self.min_confidence = min_confidence
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        
        # Initialize Alpaca client
        logger.info("Connecting to Alpaca...")
        self.client = AlpacaClient(paper=True)
        
        # Initialize database
        self.db = TradingDB("trading_bot.db")
        
        # Load model
        logger.info(f"Loading model from {model_path}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.predictor = self._load_model(model_path)
        
        # Trading state
        self.last_signal_time = None
        self.signals_history = []
        
        logger.info(f"Bot initialized for {self.original_ticker} (Alpaca: {self.ticker})")
        logger.info(f"Min confidence: {min_confidence:.0%}")
        logger.info(f"Max positions: {max_positions}")
        logger.info(f"Position size: {position_size_pct:.0%} of buying power")
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        try:
            # Get observation dimension (from training)
            obs_dim = 150  # window_size * features (30 * 5)
            action_dim = 3  # skip, buy, sell
            hidden_dim = 32
            
            # Load actor-critic
            agent = KANActorCritic(obs_dim, action_dim, hidden_dim).to(self.device)
            agent.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            agent.eval()
            
            # Load predictor (if separate file exists)
            predictor_path = model_path.replace('_crypto.pth', '_predictor.pth').replace('_stock.pth', '_predictor.pth')
            if os.path.exists(predictor_path):
                predictor = KANPredictor(obs_dim, hidden_dim).to(self.device)
                predictor.load_state_dict(torch.load(predictor_path, map_location=self.device, weights_only=True))
                predictor.eval()
            else:
                # Use predictor from agent
                predictor = None
            
            logger.info("Model loaded successfully")
            return agent, predictor
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def get_market_data(self, days: int = 60) -> Optional[pd.DataFrame]:
        """Get historical market data from Alpaca"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get bars with appropriate timeframe
            if self.mode == 'stock':
                timeframe = '1Day'
            elif self.interval == '1h':
                timeframe = '1Hour'
            elif self.interval == '4h':
                timeframe = '1Hour'  # Alpaca doesn't have 4h, use 1h and aggregate later
            else:
                timeframe = '1Hour'
            
            # Get bars
            bars = self.client.get_historical_bars(
                symbol=self.ticker,
                timeframe=timeframe,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                limit=days * 24 if self.mode == 'crypto' else days  # More bars for hourly data
            )
            
            if not bars:
                logger.warning(f"No data received for {self.ticker}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(bars)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Add technical indicators (same as training)
            import pandas_ta as ta
            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.bbands(length=20, std=2, append=True)
            
            # Calculate features (same as dapgio_improved.py)
            df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
            df["Vol_Norm"] = df["Volume"] / df["Volume"].rolling(20).mean()
            
            # Normalize
            df.dropna(inplace=True)
            if 'RSI_14' in df.columns:
                df["RSI_14"] = df["RSI_14"] / 100.0
            if 'MACD_12_26_9' in df.columns:
                df["MACD_12_26_9"] = (df["MACD_12_26_9"] - df["MACD_12_26_9"].mean()) / (df["MACD_12_26_9"].std() + 1e-7)
            
            logger.info(f"Retrieved {len(df)} bars for {self.ticker}")
            return df
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            return None
    
    def prepare_observation(self, df: pd.DataFrame) -> Optional[torch.Tensor]:
        """Prepare observation tensor from DataFrame"""
        try:
            if len(df) < self.window_size:
                return None
            
            # Get last window_size rows
            window = df.iloc[-self.window_size:].copy()
            
            # Select features (same as training)
            features = ['Close', 'Log_Ret', 'Volume', 'RSI_14', 'MACD_12_26_9']
            if 'BBL_20_2.0' in window.columns:
                features.extend(['BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0'])
            
            # Get available features
            available_features = [f for f in features if f in window.columns]
            if not available_features:
                return None
            
            # Extract and flatten
            obs = window[available_features].values.flatten()
            
            # Pad if needed
            target_size = self.window_size * 5  # 5 features
            if len(obs) < target_size:
                obs = np.pad(obs, (0, target_size - len(obs)), mode='constant')
            elif len(obs) > target_size:
                obs = obs[:target_size]
            
            return torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        except Exception as e:
            logger.error(f"Error preparing observation: {e}")
            return None
    
    def get_trading_signal(self) -> Optional[Dict]:
        """Get trading signal from model"""
        try:
            # Get market data
            df = self.get_market_data(days=60)
            if df is None or len(df) < self.window_size:
                return None
            
            # Prepare observation
            obs = self.prepare_observation(df)
            if obs is None:
                return None
            
            # Get prediction
            with torch.no_grad():
                action, log_prob, value = self.model.act(obs, deterministic=True)
                action = action.item()
                
                # Get confidence from log probability
                confidence = torch.exp(log_prob).item()
            
            # Get current price
            current_price = float(df['Close'].iloc[-1])
            
            # Map action to signal
            action_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            signal = action_map.get(action, 'HOLD')
            
            # Check confidence threshold
            if confidence < self.min_confidence:
                signal = 'HOLD'
                logger.debug(f"Signal confidence {confidence:.2%} below threshold {self.min_confidence:.0%}")
            
            return {
                'ticker': self.ticker,
                'signal': signal,
                'action': action,
                'confidence': confidence,
                'price': current_price,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting signal: {e}")
            return None
    
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
    
    def should_trade(self, signal: Dict) -> bool:
        """Check if we should execute the trade"""
        try:
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
            
            # Check confidence
            if signal['confidence'] < self.min_confidence:
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
                logger.info(f"Placing BUY order: {shares} shares of {self.ticker} @ ${price:.2f}")
                order = self.client.place_order(
                    symbol=self.ticker,
                    qty=shares,
                    side='buy',
                    order_type='market'
                )
                
                # Save to database
                self.db.add_trade(
                    ticker=self.ticker,
                    action='BUY',
                    shares=shares,
                    price=price,
                    timestamp=datetime.now(),
                    notes=f"Confidence: {signal['confidence']:.2%}"
                )
                
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
                logger.info(f"Placing SELL order: {shares} shares of {self.ticker} @ ${price:.2f}")
                order = self.client.place_order(
                    symbol=self.ticker,
                    qty=shares,
                    side='sell',
                    order_type='market'
                )
                
                # Calculate P&L
                entry_price = position['avg_entry_price']
                pnl = (price - entry_price) * shares
                pnl_pct = ((price - entry_price) / entry_price) * 100
                
                # Save to database
                self.db.add_trade(
                    ticker=self.ticker,
                    action='SELL',
                    shares=shares,
                    price=price,
                    entry_price=entry_price,
                    exit_price=price,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    timestamp=datetime.now(),
                    notes=f"Confidence: {signal['confidence']:.2%}"
                )
                
                logger.info(f"Order placed: {order['id']} - Status: {order['status']}")
                logger.info(f"P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
                return True
            
            return False
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return False
    
    def update_portfolio_snapshot(self):
        """Update portfolio snapshot in database"""
        try:
            account = self.client.get_account()
            positions = self.client.get_positions()
            
            # Calculate unrealized P&L
            unrealized_pnl = sum(pos['unrealized_pl'] for pos in positions.values())
            
            # Calculate daily return (simplified)
            portfolio_value = account['portfolio_value']
            initial_balance = 100000.0  # Paper trading starts with $100k
            cumulative_return = ((portfolio_value - initial_balance) / initial_balance) * 100
            
            # Save to database
            self.db.add_portfolio_snapshot(
                date=datetime.now().strftime('%Y-%m-%d'),
                balance=account['balance'],
                total_value=portfolio_value,
                positions_count=len(positions),
                unrealized_pnl=unrealized_pnl,
                realized_pnl=0.0,  # Would need to track this separately
                cumulative_return=cumulative_return
            )
        except Exception as e:
            logger.error(f"Error updating portfolio snapshot: {e}")
    
    def run_once(self):
        """Run one trading cycle"""
        try:
            logger.info(f"Getting signal for {self.ticker}...")
            
            # Get signal
            signal = self.get_trading_signal()
            if signal is None:
                logger.warning("No signal generated")
                return
            
            logger.info(f"Signal: {signal['signal']} | Confidence: {signal['confidence']:.2%} | Price: ${signal['price']:.2f}")
            
            # Save prediction to database
            self.db.add_prediction(
                ticker=self.ticker,
                prediction_date=datetime.now().strftime('%Y-%m-%d'),
                predicted_price=signal['price'],
                confidence=signal['confidence'],
                signal=signal['signal'],
                model_version="v0.1"
            )
            
            # Execute if confident enough
            if signal['confidence'] >= self.min_confidence:
                self.execute_signal(signal)
            else:
                logger.info(f"Signal confidence {signal['confidence']:.2%} below threshold {self.min_confidence:.0%}, skipping")
            
            # Update portfolio snapshot
            self.update_portfolio_snapshot()
            
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
                    now = datetime.now(pytz.timezone('Asia/Ho_Chi_Minh'))
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
            self.cleanup()
    
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
            current_hour = now.hour
            current_minute = now.minute
            
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
    
    def cleanup(self):
        """Cleanup resources"""
        self.db.close()
        logger.info("Bot stopped")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Alpaca Paper Trading Bot')
    parser.add_argument('--ticker', type=str, default='NVDA', help='Stock symbol')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--mode', type=str, default='stock', choices=['stock', 'crypto'], help='Trading mode')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval')
    parser.add_argument('--confidence', type=float, default=0.70, help='Min confidence threshold (0.0-1.0)')
    parser.add_argument('--max-positions', type=int, default=3, help='Max concurrent positions')
    parser.add_argument('--position-size', type=float, default=0.10, help='Position size as % of buying power')
    parser.add_argument('--once', action='store_true', help='Run once instead of continuously')
    parser.add_argument('--check-interval', type=int, default=60, help='Minutes between checks (continuous mode)')
    parser.add_argument('--schedule', type=str, nargs='+', help='Scheduled times to check (e.g., --schedule 03:30 09:30 15:30). Times in 24h format (HH:MM)')
    
    args = parser.parse_args()
    
    # Check API keys
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_SECRET_KEY'):
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("\nExample (PowerShell):")
        print('  $env:ALPACA_API_KEY = "YOUR_KEY"')
        print('  $env:ALPACA_SECRET_KEY = "YOUR_SECRET"')
        sys.exit(1)
    
    # Initialize bot
    bot = AlpacaPaperTradingBot(
        ticker=args.ticker,
        model_path=args.model,
        mode=args.mode,
        interval=args.interval,
        min_confidence=args.confidence,
        max_positions=args.max_positions,
        position_size_pct=args.position_size
    )
    
    # Run
    if args.once:
        bot.run_once()
    else:
        bot.run_continuous(
            check_interval_minutes=args.check_interval,
            scheduled_times=args.schedule
        )
    
    bot.cleanup()


if __name__ == "__main__":
    main()


