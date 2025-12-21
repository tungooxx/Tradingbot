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
        # Store historical metrics for dynamic threshold calculation
        self.history = {
            'trend': [],
            'mr': [],
            'vol': []
        }
        self.max_history = 1000  # Keep last 1000 readings
    
    def detect_regime(self, prices: np.ndarray) -> int:
        """
        Detect current market regime from price series using dynamic thresholds.
        
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
        
        # 1. Trend strength (UP/DOWN imbalance)
        price_changes = np.diff(recent_prices)
        up_moves = np.where(price_changes > 0, price_changes, 0)
        down_moves = np.where(price_changes < 0, -price_changes, 0)
        avg_up = np.mean(up_moves) if len(up_moves) > 0 else 0
        avg_down = np.mean(down_moves) if len(down_moves) > 0 else 0
        trend_strength = abs(avg_up - avg_down) / (avg_up + avg_down + 1e-8) if (avg_up + avg_down) > 0 else 0
        
        # 2. Volatility (Standard Deviation of log returns)
        volatility = np.std(returns) if len(returns) > 0 else 0
        
        # 3. Mean reversion strength (Negative Autocorrelation)
        autocorr = 0
        if len(returns) > 1:
            try:
                corr_matrix = np.corrcoef(returns[:-1], returns[1:])
                if corr_matrix.shape == (2, 2) and not np.isnan(corr_matrix[0, 1]):
                    autocorr = corr_matrix[0, 1]
            except:
                pass
        mean_reversion_strength = -autocorr
        
        # Update History
        self.history['trend'].append(trend_strength)
        self.history['mr'].append(mean_reversion_strength)
        self.history['vol'].append(volatility)
        
        for k in self.history:
            if len(self.history[k]) > self.max_history:
                self.history[k].pop(0)
        
        # Dynamic Thresholds (Rolling Quantiles)
        # Handle cases with insufficient history
        if len(self.history['vol']) < 20:
            # Static fallbacks until enough history accumulates
            if volatility > 0.02: return 3 # RISK_OFF
            if trend_strength > 0.6: return 0 # TREND_FOLLOW
            if mean_reversion_strength > 0.4: return 1 # MEAN_REVERT
            return 2 # MOMENTUM
        
        # Calculate thresholds
        trend_thresh = np.nanpercentile(self.history['trend'], 70)
        mr_thresh = np.nanpercentile(self.history['mr'], 70)
        vol_thresh = np.nanpercentile(self.history['vol'], 80) # High bar for Risk-Off
        
        # 1. Check Risk-Off First (Priority)
        if volatility > vol_thresh:
            return 3 # RISK_OFF

        # 2. Check Trend
        if trend_strength > trend_thresh:
            return 0 # TREND_FOLLOW

        # 3. Check Mean Reversion
        if mean_reversion_strength > mr_thresh:
            return 1 # MEAN_REVERT

        # 4. Default to Momentum
        return 2
    
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





