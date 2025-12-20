"""
Diagnostic script to analyze regime detection
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v0.1'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import yfinance as yf
from regime_detector import EnhancedRegimeDetector
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_regime_detection(ticker="TSLA", period="5y", interval="1d"):
    """Diagnose why regime detection always returns RISK_OFF"""
    
    logger.info(f"Downloading {ticker} data...")
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    
    if df.empty:
        logger.error("Failed to download data")
        return
    
    # Get Close prices
    if 'Close' in df.columns:
        prices = df['Close'].values
    else:
        prices = df.iloc[:, 0].values
    
    logger.info(f"Data: {len(prices)} points")
    
    # Initialize detector
    detector = EnhancedRegimeDetector(window=20, logger=logger)
    
    # Test regime detection on different windows
    test_windows = [50, 100, 200, len(prices)]
    regime_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    regime_names = {0: "TREND_FOLLOW", 1: "MEAN_REVERT", 2: "MOMENTUM", 3: "RISK_OFF"}
    
    logger.info("\n" + "="*60)
    logger.info("REGIME DETECTION DIAGNOSTICS")
    logger.info("="*60)
    
    # Test on last 100 points
    test_prices = prices[-100:] if len(prices) > 100 else prices
    
    for i in range(len(test_prices) - 20):
        window_prices = test_prices[:i+20]
        regime = detector.detect_regime(window_prices)
        regime_counts[regime] += 1
    
    logger.info(f"\nRegime distribution (last 100 points):")
    for regime_id, count in regime_counts.items():
        pct = (count / sum(regime_counts.values())) * 100 if sum(regime_counts.values()) > 0 else 0
        logger.info(f"  {regime_names[regime_id]}: {count} ({pct:.1f}%)")
    
    # Analyze volatility calculation
    logger.info("\n" + "="*60)
    logger.info("VOLATILITY ANALYSIS")
    logger.info("="*60)
    
    recent_prices = test_prices[-20:]
    returns = np.diff(np.log(recent_prices + 1e-8))
    volatility = np.std(returns) if len(returns) > 0 else 0
    
    # BUG: vol_threshold calculation
    vol_threshold_wrong = np.percentile([volatility], 70)  # WRONG - single element!
    vol_threshold_correct = volatility * 0.7  # Should use historical volatility
    
    logger.info(f"Recent volatility: {volatility:.6f}")
    logger.info(f"Current vol_threshold (WRONG): {vol_threshold_wrong:.6f}")
    logger.info(f"Correct vol_threshold (70% of current): {vol_threshold_correct:.6f}")
    logger.info(f"RISK_OFF check: volatility > vol_threshold * 1.5")
    logger.info(f"  = {volatility:.6f} > {vol_threshold_wrong * 1.5:.6f}")
    logger.info(f"  = {volatility > vol_threshold_wrong * 1.5}")
    
    # Calculate historical volatility percentiles
    all_volatilities = []
    for i in range(20, len(prices)):
        window_prices = prices[i-20:i]
        window_returns = np.diff(np.log(window_prices + 1e-8))
        window_vol = np.std(window_returns) if len(window_returns) > 0 else 0
        all_volatilities.append(window_vol)
    
    if len(all_volatilities) > 0:
        historical_70th = np.percentile(all_volatilities, 70)
        logger.info(f"\nHistorical volatility 70th percentile: {historical_70th:.6f}")
        logger.info(f"Current volatility vs historical 70th: {volatility:.6f} vs {historical_70th:.6f}")
        logger.info(f"Should trigger RISK_OFF: {volatility > historical_70th * 1.5}")
    
    # Test with fixed regime detector
    logger.info("\n" + "="*60)
    logger.info("TESTING WITH FIXED THRESHOLD")
    logger.info("="*60)
    
    if len(all_volatilities) > 0:
        fixed_threshold = np.percentile(all_volatilities, 70)
        logger.info(f"Using fixed threshold: {fixed_threshold:.6f}")
        logger.info(f"RISK_OFF trigger: {volatility > fixed_threshold * 1.5}")
    
    return regime_counts, volatility, all_volatilities

if __name__ == "__main__":
    diagnose_regime_detection("TSLA", "5y", "1d")





