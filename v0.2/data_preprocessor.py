"""
Multi-Timescale Data Preprocessor
==================================
Downloads and processes market data at multiple timescales for hierarchical trading.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
import pytz


class MultiTimescalePreprocessor:
    """
    Preprocesses market data at multiple timescales.
    
    For crypto: 1h, 4h, 1d
    For stocks: 1d, 1w
    """
    
    def __init__(self, ticker: str, mode: str = "stock", logger: Optional[logging.Logger] = None):
        """
        Args:
            ticker: Ticker symbol (e.g., "TSLA", "ETH-USD")
            mode: "stock" or "crypto"
            logger: Optional logger
        """
        self.ticker = ticker
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)
        
        # Define intervals based on mode
        if mode == "crypto":
            self.intervals = ["1h", "4h", "1d"]
        else:  # stock
            # Note: 1h data is limited to last 60 days by yfinance
            self.intervals = ["1h", "1d", "1wk"]  # Added 1h for stock mode
        
        # Cache for downloaded data
        self._data_cache = {}
    
    def _download_data(self, interval: str) -> Optional[pd.DataFrame]:
        """Download data for a specific interval"""
        try:
            if interval in self._data_cache:
                return self._data_cache[interval]
            
            self.logger.info(f"Downloading {self.ticker} data at {interval} interval...")
            
            # Download data
            ticker_obj = yf.Ticker(self.ticker)
            # Note: 1h data for stocks is limited to last 60 days by yfinance
            if interval == "1h" and self.mode == "stock":
                data = ticker_obj.history(period="60d", interval=interval)
            else:
                data = ticker_obj.history(period="2y", interval=interval)
            
            if data.empty:
                self.logger.warning(f"No data downloaded for {self.ticker} at {interval}")
                return None
            
            # MARKET HOURS FILTERING: For stocks, only keep regular trading hours (9:30 AM - 4:00 PM ET)
            # This removes extended hours data which is low-quality and confuses the agent
            if self.mode == "stock" and isinstance(data.index, pd.DatetimeIndex):
                original_len = len(data)
                
                # Convert to US/Eastern timezone if needed
                import pytz
                if data.index.tz is None:
                    # Assume data is in ET if no timezone
                    data.index = data.index.tz_localize('US/Eastern')
                elif data.index.tz != pytz.timezone('US/Eastern'):
                    # Convert to ET
                    data.index = data.index.tz_convert('US/Eastern')
                
                # For hourly data, filter to regular trading hours: 9:30 AM - 4:00 PM ET
                if interval == "1h":
                    data = data.between_time('09:30', '16:00')
                    # Filter out weekends (keep only Monday-Friday)
                    data = data[data.index.weekday < 5]
                    filtered_len = len(data)
                    if original_len > 0:
                        pct_kept = (filtered_len / original_len) * 100
                        self.logger.info(f"Market hours filter ({interval}): Kept {filtered_len}/{original_len} bars ({pct_kept:.1f}%) - Regular hours only")
                # For daily/weekly data, keep all (they represent full trading days)
                # But still filter out weekends if they exist
                elif interval in ["1d", "1wk"]:
                    data = data[data.index.weekday < 5]  # Remove weekends
                    filtered_len = len(data)
                    if original_len != filtered_len:
                        self.logger.info(f"Weekend filter ({interval}): Kept {filtered_len}/{original_len} bars - Weekdays only")
                
                if data.empty:
                    self.logger.warning(f"No data remaining after market hours filter for {self.ticker} at {interval}")
                    return None
            
            # Clean data
            data = data.dropna()
            
            if len(data) < 100:
                self.logger.warning(f"Insufficient data for {self.ticker} at {interval}: {len(data)} rows")
                return None
            
            self._data_cache[interval] = data
            return data
            
        except Exception as e:
            self.logger.error(f"Error downloading {self.ticker} at {interval}: {e}")
            return None
    
    def _compute_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Compute features from OHLCV data.
        
        Features:
        0: Close (normalized)
        1: Volume (normalized)
        2: Returns (log returns)
        3: RSI (14-period)
        4: MACD signal
        """
        if len(data) < 20:
            return np.array([])
        
        # Extract OHLCV
        close = data['Close'].values
        high = data['High'].values
        low = data['Low'].values
        volume = data['Volume'].values
        
        n = len(close)
        
        # Feature 0: Normalized Close
        close_norm = (close - close.mean()) / (close.std() + 1e-7)
        
        # Feature 1: Normalized Volume
        volume_norm = (volume - volume.mean()) / (volume.std() + 1e-7)
        
        # Feature 2: Log Returns
        log_returns = np.diff(np.log(close + 1e-8))
        log_returns = np.concatenate([[0], log_returns])  # Pad first value
        # Ensure same length
        if len(log_returns) != n:
            log_returns = np.pad(log_returns, (0, n - len(log_returns)), mode='constant', constant_values=0)[:n]
        
        # Feature 3: RSI (simplified)
        delta = np.diff(close)  # Length: n-1
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Simple moving average of gains/losses
        rsi_window = 14
        rsi = np.full(n, 50.0)  # Initialize with neutral RSI
        
        if len(gain) >= rsi_window:
            # Rolling mean on gain/loss (length: n-1)
            avg_gain = pd.Series(gain).rolling(window=rsi_window, min_periods=1).mean().values
            avg_loss = pd.Series(loss).rolling(window=rsi_window, min_periods=1).mean().values
            
            # Replace NaN with 0 for calculation
            avg_gain = np.nan_to_num(avg_gain, nan=0.0)
            avg_loss = np.nan_to_num(avg_loss, nan=0.0)
            
            # RSI calculation (length: n-1)
            rs = np.where(avg_loss != 0, avg_gain / (avg_loss + 1e-8), 100)
            rsi_calc = 100 - (100 / (1 + rs))
            
            # Map RSI values back to original length n
            # First value stays 50, rest come from rsi_calc (which has length n-1)
            if len(rsi_calc) == n - 1:
                rsi[1:] = rsi_calc
            elif len(rsi_calc) > n - 1:
                rsi[1:] = rsi_calc[:n-1]
            else:
                # If shorter, pad from position 1
                rsi[1:1+len(rsi_calc)] = rsi_calc
        
        # Feature 4: MACD signal (simplified)
        # EMA12 - EMA26
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
        macd = ema12 - ema26
        macd_signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
        macd_diff = macd - macd_signal
        
        # Ensure MACD has correct length
        if len(macd_diff) != n:
            macd_diff = np.pad(macd_diff, (0, n - len(macd_diff)), mode='constant', constant_values=0)[:n]
        
        # Stack base features - ensure all have same length
        features = np.column_stack([
            close_norm[:n],
            volume_norm[:n],
            log_returns[:n],
            rsi[:n] / 100.0,  # Normalize RSI to [0, 1]
            macd_diff[:n] / (close.std() + 1e-7)  # Normalize MACD
        ])
        
        # MARKET HOURS AWARENESS: Add time-of-day features for stocks only
        if self.mode == "stock":
            market_hours_features = self._compute_market_hours_features(data, n)
            features = np.column_stack([features, market_hours_features])
        
        # Replace NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def _compute_market_hours_features(self, data: pd.DataFrame, n: int) -> np.ndarray:
        """
        Compute market hours features (stocks only).
        
        Features:
        - hour_norm: Normalized hour of day (0-1)
        - minutes_since_open_norm: Normalized minutes since market open (0-1)
        - is_market_open: 1.0 if market is open, 0.0 if closed
        
        For daily/weekly data, use default values (midday, market open).
        """
        try:
            if not isinstance(data.index, pd.DatetimeIndex):
                # Not datetime index - use default values
                hour_norm = np.full(n, 0.5)  # Midday
                minutes_since_open_norm = np.full(n, 0.5)  # Midday
                is_market_open = np.full(n, 1.0)  # Assume open
                return np.column_stack([hour_norm, minutes_since_open_norm, is_market_open])
            
            # Extract time components
            hours = data.index.hour.values
            minutes = data.index.minute.values
            weekdays = data.index.weekday.values  # 0=Monday, 6=Sunday
            
            # Normalize hour (0-23 -> 0-1)
            hour_norm = hours / 23.0
            
            # Calculate minutes since market open (9:30 AM = 0)
            market_open_minutes = 9 * 60 + 30  # 570 minutes
            current_minutes = hours * 60 + minutes
            minutes_since_open = current_minutes - market_open_minutes
            # Clip to valid range (0 to 390 minutes = 6.5 hours)
            minutes_since_open = np.clip(minutes_since_open, 0, 390)
            minutes_since_open_norm = minutes_since_open / 390.0
            
            # Is market open? (9:30 AM to 4:00 PM, weekdays)
            is_after_open = (hours > 9) | ((hours == 9) & (minutes >= 30))
            is_before_close = hours < 16
            is_weekday = weekdays < 5  # Monday-Friday
            
            is_market_open = (is_after_open & is_before_close & is_weekday).astype(float)
            
            # For daily/weekly data, set to default (midday, open)
            # Check if this is daily/weekly data (no minute precision)
            if len(np.unique(minutes)) == 1 and minutes[0] == 0:
                # Likely daily/weekly data - use defaults
                hour_norm = np.full(n, 0.5)  # Midday
                minutes_since_open_norm = np.full(n, 0.5)  # Midday
                is_market_open = np.full(n, 1.0)  # Assume open
            
            return np.column_stack([hour_norm, minutes_since_open_norm, is_market_open])
            
        except Exception as e:
            self.logger.warning(f"Failed to compute market hours features: {e}, using defaults")
            # Return default values on error
            hour_norm = np.full(n, 0.5)
            minutes_since_open_norm = np.full(n, 0.5)
            is_market_open = np.full(n, 1.0)
            return np.column_stack([hour_norm, minutes_since_open_norm, is_market_open])
    
    def get_window_features(self, features: Dict[str, np.ndarray], 
                           window_size: int, 
                           current_idx: Optional[int] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract windowed features for a specific time index.
        
        Args:
            features: Dictionary of feature arrays by interval
            window_size: Size of the window
            current_idx: Current time index (None = use last available window)
            
        Returns:
            Dictionary of windowed feature arrays by interval, or None if insufficient data
        """
        windowed = {}
        
        for interval, feat_array in features.items():
            if len(feat_array) < window_size:
                # Not enough data for this interval
                continue
            
            if current_idx is None:
                # Use last available window
                start_idx = len(feat_array) - window_size
                end_idx = len(feat_array)
            else:
                # Use window ending at current_idx
                start_idx = max(0, current_idx - window_size + 1)
                end_idx = current_idx + 1
                
                # Check if we have enough data
                if end_idx > len(feat_array):
                    # Not enough data, skip this interval
                    continue
            
            # Extract window
            window = feat_array[start_idx:end_idx]
            
            # Ensure window has correct size
            if len(window) == window_size:
                windowed[interval] = window
            elif len(window) < window_size:
                # Pad with zeros if needed
                padding = np.zeros((window_size - len(window), feat_array.shape[1]))
                window = np.vstack([padding, window])
                windowed[interval] = window
        
        if not windowed:
            return None
        
        return windowed
    
    def process(self, window_size: int = 30) -> Tuple[Optional[Dict], Optional[Dict]]:
        """
        Process data at multiple timescales.
        
        Returns:
            aligned_data: Dictionary of aligned DataFrames by interval
            features: Dictionary of feature arrays by interval
        """
        aligned_data = {}
        features = {}
        
        # Download and process each interval
        for interval in self.intervals:
            data = self._download_data(interval)
            
            if data is None or len(data) < window_size + 10:
                self.logger.warning(f"Skipping {interval} - insufficient data")
                continue
            
            # Compute features
            feat_array = self._compute_features(data)
            
            if len(feat_array) == 0:
                self.logger.warning(f"Skipping {interval} - feature computation failed")
                continue
            
            aligned_data[interval] = data
            features[interval] = feat_array
            
            self.logger.info(f"Processed {interval}: {len(feat_array)} samples, {feat_array.shape[1]} features")
        
        if not aligned_data:
            self.logger.error("No data processed successfully")
            return None, None
        
        return aligned_data, features





