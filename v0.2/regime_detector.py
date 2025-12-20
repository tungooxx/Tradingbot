"""
Enhanced Regime Detector
========================
Detects market regimes (trending, mean-reverting, high-volatility, low-volatility).
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging


class EnhancedRegimeDetector:
    """
    Detects market regimes using statistical methods.
    
    Regimes:
    0: TREND_FOLLOW - Strong trend
    1: MEAN_REVERT - Mean-reverting
    2: MOMENTUM - High momentum
    3: RISK_OFF - High volatility, risk-off
    """
    
    def __init__(self, window: int = 20, logger: Optional[logging.Logger] = None):
        """
        Args:
            window: Window size for regime detection
            logger: Optional logger
        """
        self.window = window
        self.logger = logger or logging.getLogger(__name__)
        # Store historical volatilities for proper threshold calculation
        self.volatility_history = []
        self.max_history = 1000  # Keep last 1000 volatility readings
    
    def detect_regime(self, prices: np.ndarray) -> int:
        """
        Detect current market regime from price series.
        
        Args:
            prices: Price array (Close prices)
            
        Returns:
            Regime index (0-3)
        """
        if len(prices) < self.window:
            return 0  # Default to TREND_FOLLOW
        
        # Use last window_size prices
        recent_prices = prices[-self.window:]
        
        # Calculate returns
        returns = np.diff(np.log(recent_prices + 1e-8))
        
        if len(returns) == 0:
            return 0
        
        # 1. Trend strength (ADX-like)
        # Calculate directional movement
        price_changes = np.diff(recent_prices)
        up_moves = np.where(price_changes > 0, price_changes, 0)
        down_moves = np.where(price_changes < 0, -price_changes, 0)
        
        avg_up = np.mean(up_moves) if len(up_moves) > 0 else 0
        avg_down = np.mean(down_moves) if len(down_moves) > 0 else 0
        
        # Trend strength
        if avg_up + avg_down > 0:
            trend_strength = abs(avg_up - avg_down) / (avg_up + avg_down + 1e-8)
        else:
            trend_strength = 0
        
        # 2. Volatility
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # 3. Mean reversion strength (Hurst-like)
        # Calculate autocorrelation
        autocorr = 0
        if len(returns) > 1:
            try:
                corr_matrix = np.corrcoef(returns[:-1], returns[1:])
                if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]):
                    autocorr = corr_matrix[0, 1]
            except:
                autocorr = 0
        
        mean_reversion_strength = -autocorr  # Negative autocorr = mean reverting
        
        # 4. Momentum
        momentum = np.mean(returns) if len(returns) > 0 else 0
        
        # Regime classification
        # FIX: Store volatility in history for proper threshold calculation
        self.volatility_history.append(volatility)
        if len(self.volatility_history) > self.max_history:
            self.volatility_history.pop(0)
        
        # FIX: Calculate threshold from historical volatilities, not single value
        if len(self.volatility_history) >= 20:  # Need at least 20 samples
            vol_threshold = np.percentile(self.volatility_history, 70)
        else:
            # Fallback: use 70% of current volatility if no history
            vol_threshold = volatility * 0.7 if volatility > 0 else 0.02
        
        # RISK_OFF: High volatility (only if significantly above historical 70th percentile)
        # Use 1.5x multiplier to avoid triggering too easily
        if volatility > vol_threshold * 1.5:
            return 3  # RISK_OFF
        
        # TREND_FOLLOW: Strong trend, low mean reversion
        if trend_strength > 0.6 and mean_reversion_strength < 0.3:
            return 0  # TREND_FOLLOW
        
        # MEAN_REVERT: Strong mean reversion
        if mean_reversion_strength > 0.5:
            return 1  # MEAN_REVERT
        
        # MOMENTUM: Strong momentum
        if abs(momentum) > 0.01:
            return 2  # MOMENTUM
        
        # Default: TREND_FOLLOW
        return 0
    
    def get_regime_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Get regime features as one-hot encoding.
        
        Returns:
            One-hot encoded regime vector [TREND, MEAN_REVERT, MOMENTUM, RISK_OFF]
        """
        regime = self.detect_regime(prices)
        features = np.zeros(4)
        features[regime] = 1.0
        return features





