"""
Human-Like Trading Strategies
==============================
Makes the trading bot think and act more like a human trader:
- Waits after market open before trading
- Checks market conditions and bars
- Hesitates on unusual events (rapid drops, gaps, etc.)
- Looks for confirmation signals
- Avoids trading during high volatility periods
"""

import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
from typing import Dict, Optional, Tuple, List
import pytz
import logging

logger = logging.getLogger(__name__)


class HumanLikeStrategy:
    """Human-like trading strategy logic"""
    
    def __init__(self, mode: str = "stock", market_timezone: str = "America/New_York"):
        """
        Initialize human-like strategy
        
        Args:
            mode: 'stock' or 'crypto'
            market_timezone: Timezone for market hours (default: ET for US stocks)
        """
        self.mode = mode
        self.market_tz = pytz.timezone(market_timezone)
        self.wait_minutes_after_open = 6  # Wait 5-6 minutes after market open (human-like)
        self.min_bars_after_open = 5  # Need at least 5 bars to analyze after open
        
        # Market hours for stocks (ET)
        self.market_open_time = time(9, 30)  # 9:30 AM ET
        self.market_close_time = time(16, 0)  # 4:00 PM ET
        
        # Thresholds for anomaly detection (more sensitive)
        self.rapid_drop_threshold = -0.015  # -1.5% in short time (more sensitive)
        self.rapid_drop_window = 3  # Check last 3 bars for rapid drops
        self.unusual_volume_multiplier = 1.8  # 1.8x average volume (more sensitive)
        self.gap_threshold = 0.015  # 1.5% gap (more sensitive)
        self.high_volatility_threshold = 0.025  # 2.5% volatility (more sensitive)
        self.flash_crash_threshold = -0.04  # -4% in 3 bars
        
        # Pattern recognition thresholds
        self.support_resistance_window = 20  # Lookback for support/resistance
        self.trend_confirmation_bars = 5  # Bars needed to confirm trend
        self.reversal_confirmation_bars = 3  # Bars needed to confirm reversal
        
    def is_market_open(self, current_time: Optional[datetime] = None) -> bool:
        """Check if market is currently open"""
        if self.mode == "crypto":
            return True  # Crypto markets are always open
        
        if current_time is None:
            current_time = datetime.now(self.market_tz)
        else:
            # Ensure timezone-aware
            if current_time.tzinfo is None:
                current_time = self.market_tz.localize(current_time)
            else:
                current_time = current_time.astimezone(self.market_tz)
        
        current_time_only = current_time.time()
        current_weekday = current_time.weekday()  # 0=Monday, 6=Sunday
        
        # Market closed on weekends
        if current_weekday >= 5:  # Saturday or Sunday
            return False
        
        # Check if within market hours
        return self.market_open_time <= current_time_only <= self.market_close_time
    
    def should_wait_after_open(self, current_time: Optional[datetime] = None, 
                               df: Optional[pd.DataFrame] = None) -> Tuple[bool, str]:
        """
        Check if we should wait after market open (human-like: wait 5-6 min and check bars)
        
        Args:
            current_time: Current datetime
            df: DataFrame to check bars after market open
        
        Returns:
            (should_wait, reason)
        """
        if self.mode == "crypto":
            return False, "Crypto markets always open"
        
        if current_time is None:
            current_time = datetime.now(self.market_tz)
        else:
            if current_time.tzinfo is None:
                current_time = self.market_tz.localize(current_time)
            else:
                current_time = current_time.astimezone(self.market_tz)
        
        # Check if market is open
        if not self.is_market_open(current_time):
            return True, "Market is closed"
        
        # Check if we're within wait period after market open
        market_open_datetime = current_time.replace(
            hour=self.market_open_time.hour,
            minute=self.market_open_time.minute,
            second=0,
            microsecond=0
        )
        
        # If current time is before market open, wait
        if current_time.time() < self.market_open_time:
            return True, f"Market opens at {self.market_open_time.strftime('%H:%M')}"
        
        # Calculate minutes since market open
        minutes_since_open = (current_time - market_open_datetime).total_seconds() / 60
        
        # Human-like behavior: Wait 5-6 minutes AND check that we have enough bars
        if minutes_since_open < self.wait_minutes_after_open:
            remaining = self.wait_minutes_after_open - minutes_since_open
            return True, f"Waiting {remaining:.1f} minutes after market open (checking bars...)"
        
        # After wait period, check if we have enough bars to analyze
        if df is not None and len(df) > 0:
            # Count bars since market open (assuming 1-minute bars for intraday)
            # For daily data, this won't apply
            bars_since_open = self._count_bars_since_open(df, current_time)
            if bars_since_open < self.min_bars_after_open:
                return True, f"Need {self.min_bars_after_open} bars after open, have {bars_since_open} (checking market bars...)"
            
            # Check if early bars show normal patterns (human-like: analyze first few bars)
            if bars_since_open >= self.min_bars_after_open:
                early_bars_normal = self._check_early_bars_normal(df, bars_since_open)
                if not early_bars_normal:
                    return True, "Early market bars show unusual patterns, waiting for stabilization..."
        
        return False, "Market open wait period passed and bars checked"
    
    def _count_bars_since_open(self, df: pd.DataFrame, current_time: datetime) -> int:
        """Count bars since market open (for intraday data)"""
        if len(df) == 0:
            return 0
        
        # For daily data, return a high number (always enough)
        if self.mode == "stock" and hasattr(df.index, 'freq'):
            # Check if this is daily data
            if df.index.freq and 'D' in str(df.index.freq):
                return 999  # Daily data: always enough
        
        # For intraday: count bars from market open time
        market_open_today = current_time.replace(
            hour=self.market_open_time.hour,
            minute=self.market_open_time.minute,
            second=0,
            microsecond=0
        )
        
        # Count bars after market open
        bars_after_open = 0
        for idx in df.index[-10:]:  # Check last 10 bars
            if hasattr(idx, 'to_pydatetime'):
                try:
                    bar_time = idx.to_pydatetime()
                    if bar_time >= market_open_today:
                        bars_after_open += 1
                except:
                    pass
        
        return bars_after_open
    
    def _check_early_bars_normal(self, df: pd.DataFrame, bars_since_open: int) -> bool:
        """
        Check if early bars (first 5-6 minutes) show normal patterns.
        Human-like: analyze if bars are stable or showing weird patterns.
        """
        if len(df) < bars_since_open:
            return True  # Not enough data, assume normal
        
        # Get bars since market open
        early_bars = df.iloc[-bars_since_open:] if bars_since_open <= len(df) else df.iloc[-5:]
        
        # Check for extreme volatility in early bars
        if len(early_bars) >= 3:
            early_returns = early_bars['Close'].pct_change().dropna()
            if len(early_returns) > 0:
                early_volatility = early_returns.std()
                # If volatility is too high in early bars, wait
                if early_volatility > 0.02:  # 2% volatility in first few bars
                    return False
        
        # Check for rapid price movements
        if len(early_bars) >= 3:
            price_range = (early_bars['High'].max() - early_bars['Low'].min()) / early_bars['Close'].iloc[0]
            if price_range > 0.03:  # More than 3% range in early bars
                return False
        
        # Check for unusual volume spikes in early bars
        if 'Volume' in early_bars.columns and len(df) > 20:
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            early_volume = early_bars['Volume'].mean()
            if avg_volume > 0 and early_volume > avg_volume * 2.5:  # Very high volume in early bars
                return False
        
        return True  # Early bars look normal
    
    def detect_anomalies(self, df: pd.DataFrame, lookback_bars: int = 10) -> Dict[str, any]:
        """
        Detect market anomalies that would make a human trader hesitate
        
        Args:
            df: DataFrame with OHLCV data
            lookback_bars: Number of bars to analyze
        
        Returns:
            Dict with anomaly flags and details
        """
        if len(df) < lookback_bars:
            return {
                'has_anomalies': False,
                'reasons': ['Insufficient data']
            }
        
        recent_df = df.iloc[-lookback_bars:].copy()
        anomalies = []
        anomaly_flags = {
            'rapid_drop': False,
            'unusual_volume': False,
            'price_gap': False,
            'high_volatility': False,
            'flash_crash': False,
            'rapid_pump': False,
            'price_rejection_high': False,
            'price_rejection_low': False
        }
        
        # 1. Check for rapid price drop
        if len(recent_df) >= self.rapid_drop_window:
            price_changes = recent_df['Close'].pct_change().dropna()
            if len(price_changes) > 0:
                recent_change = price_changes.iloc[-self.rapid_drop_window:].sum()
                if recent_change < self.rapid_drop_threshold:
                    anomaly_flags['rapid_drop'] = True
                    anomalies.append(f"Rapid drop detected: {recent_change:.2%} in last {self.rapid_drop_window} bars")
        
        # 2. Check for unusual volume
        if 'Volume' in recent_df.columns:
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            recent_volume = recent_df['Volume'].iloc[-1]
            if avg_volume > 0 and recent_volume > avg_volume * self.unusual_volume_multiplier:
                anomaly_flags['unusual_volume'] = True
                volume_ratio = recent_volume / avg_volume
                anomalies.append(f"Unusual volume: {volume_ratio:.1f}x average volume")
        
        # 3. Check for price gaps
        if len(recent_df) >= 2:
            prev_close = recent_df['Close'].iloc[-2]
            curr_open = recent_df['Open'].iloc[-1]
            gap_pct = (curr_open - prev_close) / prev_close
            if abs(gap_pct) > self.gap_threshold:
                anomaly_flags['price_gap'] = True
                gap_direction = "up" if gap_pct > 0 else "down"
                anomalies.append(f"Price gap detected: {gap_pct:.2%} {gap_direction}")
        
        # 4. Check for high volatility
        if len(recent_df) >= 5:
            recent_returns = recent_df['Close'].pct_change().dropna()
            volatility = recent_returns.std()
            if volatility > self.high_volatility_threshold:
                anomaly_flags['high_volatility'] = True
                anomalies.append(f"High volatility: {volatility:.2%} std dev")
        
        # 5. Check for flash crash (very rapid drop)
        if len(recent_df) >= 3:
            # Check if price dropped >4% in 3 bars (more sensitive)
            price_3bars_ago = recent_df['Close'].iloc[-3]
            current_price = recent_df['Close'].iloc[-1]
            drop_pct = (current_price - price_3bars_ago) / price_3bars_ago
            if drop_pct < self.flash_crash_threshold:  # -4% in 3 bars
                anomaly_flags['flash_crash'] = True
                anomalies.append(f"Flash crash detected: {drop_pct:.2%} in 3 bars")
        
        # 6. Check for rapid price increase (might be pump, hesitate on SELL)
        if len(recent_df) >= 3:
            price_3bars_ago = recent_df['Close'].iloc[-3]
            current_price = recent_df['Close'].iloc[-1]
            increase_pct = (current_price - price_3bars_ago) / price_3bars_ago
            if increase_pct > 0.05:  # +5% in 3 bars (potential pump)
                anomaly_flags['rapid_pump'] = True
                anomalies.append(f"Rapid price increase detected: {increase_pct:.2%} in 3 bars")
        
        # 7. Check for price rejection (wick patterns - human-like pattern recognition)
        if len(recent_df) >= 1:
            last_bar = recent_df.iloc[-1]
            if 'High' in last_bar and 'Low' in last_bar and 'Close' in last_bar and 'Open' in last_bar:
                body_size = abs(last_bar['Close'] - last_bar['Open'])
                upper_wick = last_bar['High'] - max(last_bar['Close'], last_bar['Open'])
                lower_wick = min(last_bar['Close'], last_bar['Open']) - last_bar['Low']
                total_range = last_bar['High'] - last_bar['Low']
                
                if total_range > 0:
                    # Large upper wick = rejection at high (bearish)
                    if upper_wick > body_size * 2 and upper_wick / total_range > 0.4:
                        anomaly_flags['price_rejection_high'] = True
                        anomalies.append("Price rejection at high (large upper wick)")
                    
                    # Large lower wick = rejection at low (bullish, but might hesitate)
                    if lower_wick > body_size * 2 and lower_wick / total_range > 0.4:
                        anomaly_flags['price_rejection_low'] = True
                        anomalies.append("Price rejection at low (large lower wick)")
        
        return {
            'has_anomalies': len(anomalies) > 0,
            'reasons': anomalies,
            'flags': anomaly_flags
        }
    
    def check_market_conditions(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Check overall market conditions (like a human would)
        
        Returns:
            Dict with market condition assessment
        """
        if len(df) < 20:
            return {
                'is_normal': False,
                'reason': 'Insufficient data for market analysis',
                'should_trade': False
            }
        
        # Get recent data
        recent_df = df.iloc[-20:].copy()
        
        # Calculate basic metrics
        current_price = recent_df['Close'].iloc[-1]
        avg_price = recent_df['Close'].mean()
        price_deviation = abs(current_price - avg_price) / avg_price
        
        # Check if price is too far from average
        if price_deviation > 0.05:  # >5% deviation
            return {
                'is_normal': False,
                'reason': f'Price deviates {price_deviation:.2%} from recent average',
                'should_trade': False,
                'price_deviation': price_deviation
            }
        
        # Check for consistent trend (good for trading)
        if len(recent_df) >= 10:
            recent_trend = recent_df['Close'].iloc[-10:].pct_change().mean()
            trend_strength = abs(recent_trend)
            
            # If trend is too strong (>1% per bar), might be overextended
            if trend_strength > 0.01:
                return {
                    'is_normal': False,
                    'reason': f'Strong trend detected ({recent_trend:.2%} per bar), might be overextended',
                    'should_trade': False,
                    'trend': recent_trend
                }
        
        # Check for low volume (market might be illiquid)
        if 'Volume' in recent_df.columns:
            avg_volume = df['Volume'].rolling(20).mean().iloc[-1]
            recent_volume = recent_df['Volume'].iloc[-1]
            if avg_volume > 0 and recent_volume < avg_volume * 0.5:
                return {
                    'is_normal': False,
                    'reason': 'Low volume detected, market might be illiquid',
                    'should_trade': False,
                    'volume_ratio': recent_volume / avg_volume
                }
        
        # Market conditions look normal
        return {
            'is_normal': True,
            'reason': 'Market conditions appear normal',
            'should_trade': True
        }
    
    def should_hesitate(self, df: pd.DataFrame, signal: str) -> Tuple[bool, str]:
        """
        Determine if we should hesitate before trading (like a human would)
        Enhanced with pattern recognition and more nuanced logic
        
        Args:
            df: Market data DataFrame
            signal: Trading signal ('BUY', 'SELL', 'HOLD')
        
        Returns:
            (should_hesitate, reason)
        """
        if signal == 'HOLD':
            return False, "No action needed"
        
        # Check for anomalies
        anomalies = self.detect_anomalies(df)
        if anomalies['has_anomalies']:
            flags = anomalies.get('flags', {})
            
            # Signal-specific hesitation logic
            if signal == 'BUY':
                # Hesitate more on BUY if rapid drop or flash crash
                if flags.get('rapid_drop') or flags.get('flash_crash'):
                    reasons = "; ".join(anomalies['reasons'])
                    return True, f"Hesitating on BUY: {reasons}"
                # Price rejection at high = bearish, hesitate on BUY
                if flags.get('price_rejection_high'):
                    return True, "Hesitating on BUY: Price rejection at high (bearish pattern)"
            
            elif signal == 'SELL':
                # Hesitate on SELL if flash crash (might recover)
                if flags.get('flash_crash'):
                    reasons = "; ".join(anomalies['reasons'])
                    return True, f"Hesitating on SELL: {reasons}"
                # Rapid pump might be temporary, hesitate on SELL
                if flags.get('rapid_pump'):
                    return True, "Hesitating on SELL: Rapid price increase detected, might continue"
                # Price rejection at low = bullish, hesitate on SELL
                if flags.get('price_rejection_low'):
                    return True, "Hesitating on SELL: Price rejection at low (bullish pattern)"
            
            # General hesitation for other anomalies
            reasons = "; ".join(anomalies['reasons'])
            return True, f"Hesitating due to anomalies: {reasons}"
        
        # Check market conditions
        market_conditions = self.check_market_conditions(df)
        if not market_conditions['is_normal']:
            return True, f"Hesitating: {market_conditions['reason']}"
        
        # Pattern-based hesitation (human-like pattern recognition)
        if len(df) >= 10:
            pattern_hesitation, pattern_reason = self._check_patterns(df, signal)
            if pattern_hesitation:
                return True, pattern_reason
        
        # For BUY signals, be extra cautious on rapid drops
        if signal == 'BUY':
            if len(df) >= self.rapid_drop_window:
                recent_returns = df['Close'].iloc[-self.rapid_drop_window:].pct_change().dropna()
                if len(recent_returns) > 0:
                    total_drop = recent_returns.sum()
                    if total_drop < self.rapid_drop_threshold:
                        return True, f"Hesitating on BUY: Rapid drop detected ({total_drop:.2%} in {self.rapid_drop_window} bars)"
        
        # For SELL signals, check if we're in a flash crash (might want to wait)
        if signal == 'SELL':
            if len(df) >= 3:
                price_3bars_ago = df['Close'].iloc[-3]
                current_price = df['Close'].iloc[-1]
                drop_pct = (current_price - price_3bars_ago) / price_3bars_ago
                if drop_pct < self.flash_crash_threshold:  # Flash crash
                    return True, f"Hesitating on SELL: Flash crash detected ({drop_pct:.2%}), might recover"
        
        return False, "No hesitation needed"
    
    def _check_patterns(self, df: pd.DataFrame, signal: str) -> Tuple[bool, str]:
        """
        Human-like pattern recognition: check for support/resistance, trends, reversals
        
        Returns:
            (should_hesitate, reason)
        """
        if len(df) < self.support_resistance_window:
            return False, ""
        
        recent_df = df.iloc[-self.support_resistance_window:].copy()
        
        # 1. Check for support/resistance levels
        recent_highs = recent_df['High'].rolling(5).max()
        recent_lows = recent_df['Low'].rolling(5).min()
        current_price = recent_df['Close'].iloc[-1]
        
        # Check if price is near resistance (hesitate on BUY)
        if signal == 'BUY':
            resistance_level = recent_highs.max()
            distance_to_resistance = (resistance_level - current_price) / current_price
            if distance_to_resistance < 0.01:  # Within 1% of resistance
                return True, f"Hesitating on BUY: Price near resistance level (${resistance_level:.2f})"
        
        # Check if price is near support (hesitate on SELL)
        if signal == 'SELL':
            support_level = recent_lows.min()
            distance_to_support = (current_price - support_level) / current_price
            if distance_to_support < 0.01:  # Within 1% of support
                return True, f"Hesitating on SELL: Price near support level (${support_level:.2f})"
        
        # 2. Check for trend exhaustion (overextended moves)
        if len(recent_df) >= self.trend_confirmation_bars:
            trend_returns = recent_df['Close'].iloc[-self.trend_confirmation_bars:].pct_change().dropna()
            if len(trend_returns) > 0:
                avg_trend = trend_returns.mean()
                trend_strength = abs(avg_trend)
                
                # If strong uptrend and BUY signal, might be overextended
                if signal == 'BUY' and avg_trend > 0.005:  # Strong uptrend (>0.5% per bar)
                    if trend_strength > 0.01:  # Very strong
                        return True, f"Hesitating on BUY: Strong uptrend detected ({avg_trend:.2%} per bar), might be overextended"
                
                # If strong downtrend and SELL signal, might be overextended
                if signal == 'SELL' and avg_trend < -0.005:  # Strong downtrend
                    if trend_strength > 0.01:  # Very strong
                        return True, f"Hesitating on SELL: Strong downtrend detected ({avg_trend:.2%} per bar), might be overextended"
        
        # 3. Check for potential reversal patterns
        if len(recent_df) >= self.reversal_confirmation_bars:
            # Check for reversal candlestick patterns (simplified)
            last_3_bars = recent_df.iloc[-3:]
            
            # Bearish reversal pattern: up bars followed by down bar
            if signal == 'BUY':
                if (last_3_bars['Close'].iloc[-2] > last_3_bars['Close'].iloc[-3] and
                    last_3_bars['Close'].iloc[-1] < last_3_bars['Close'].iloc[-2]):
                    return True, "Hesitating on BUY: Potential bearish reversal pattern detected"
            
            # Bullish reversal pattern: down bars followed by up bar
            if signal == 'SELL':
                if (last_3_bars['Close'].iloc[-2] < last_3_bars['Close'].iloc[-3] and
                    last_3_bars['Close'].iloc[-1] > last_3_bars['Close'].iloc[-2]):
                    return True, "Hesitating on SELL: Potential bullish reversal pattern detected"
        
        return False, ""
    
    def get_trading_decision(self, df: pd.DataFrame, model_signal: str, 
                            model_confidence: float, current_time: Optional[datetime] = None) -> Dict:
        """
        Make final trading decision with human-like logic
        
        Args:
            df: Market data DataFrame
            model_signal: Signal from model ('BUY', 'SELL', 'HOLD')
            model_confidence: Model confidence (0.0-1.0)
            current_time: Current time (for market hours check)
        
        Returns:
            Dict with final decision and reasoning
        """
        decision = {
            'final_signal': 'HOLD',
            'original_signal': model_signal,
            'confidence': model_confidence,
            'reasons': [],
            'should_wait': False,
            'hesitation': False
        }
        
        # 1. Check if we should wait after market open (with bar checking)
        should_wait, wait_reason = self.should_wait_after_open(current_time, df)
        if should_wait:
            decision['should_wait'] = True
            decision['reasons'].append(wait_reason)
            decision['final_signal'] = 'HOLD'
            return decision
        
        # 2. Check market conditions
        market_conditions = self.check_market_conditions(df)
        if not market_conditions['is_normal']:
            decision['reasons'].append(f"Market not normal: {market_conditions['reason']}")
            decision['final_signal'] = 'HOLD'
            return decision
        
        # 3. Check for anomalies and hesitation
        should_hesitate, hesitation_reason = self.should_hesitate(df, model_signal)
        if should_hesitate:
            decision['hesitation'] = True
            decision['reasons'].append(hesitation_reason)
            # Still allow the signal but with lower confidence
            decision['final_signal'] = model_signal
            decision['confidence'] = model_confidence * 0.7  # Reduce confidence by 30%
            return decision
        
        # 4. If all checks pass, use model signal
        decision['final_signal'] = model_signal
        decision['reasons'].append("All checks passed, using model signal")
        
        return decision

