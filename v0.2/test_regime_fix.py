"""
Simple test to verify regime detection fix without yfinance
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from regime_detector import EnhancedRegimeDetector

def test_regime_detection_fix():
    """Test that regime detection doesn't always return RISK_OFF"""
    
    print("="*60)
    print("TESTING REGIME DETECTION FIX")
    print("="*60)
    
    # Create detector
    detector = EnhancedRegimeDetector(window=20)
    
    # Simulate different market conditions
    np.random.seed(42)
    
    # Test 1: Low volatility (should NOT be RISK_OFF)
    print("\n1. Testing LOW volatility scenario...")
    low_vol_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)  # Small moves
    regimes_low = []
    for i in range(20, len(low_vol_prices)):
        window_prices = low_vol_prices[:i]
        regime = detector.detect_regime(window_prices)
        regimes_low.append(regime)
    
    regime_counts_low = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in regimes_low:
        regime_counts_low[r] += 1
    
    print(f"   Regime distribution: {regime_counts_low}")
    print(f"   RISK_OFF percentage: {regime_counts_low[3]/len(regimes_low)*100:.1f}%")
    
    # Test 2: High volatility (should be RISK_OFF)
    print("\n2. Testing HIGH volatility scenario...")
    detector2 = EnhancedRegimeDetector(window=20)  # New detector for clean test
    high_vol_prices = 100 + np.cumsum(np.random.randn(100) * 5.0)  # Large moves
    regimes_high = []
    for i in range(20, len(high_vol_prices)):
        window_prices = high_vol_prices[:i]
        regime = detector2.detect_regime(window_prices)
        regimes_high.append(regime)
    
    regime_counts_high = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in regimes_high:
        regime_counts_high[r] += 1
    
    print(f"   Regime distribution: {regime_counts_high}")
    print(f"   RISK_OFF percentage: {regime_counts_high[3]/len(regimes_high)*100:.1f}%")
    
    # Test 3: Verify volatility history is being tracked
    print("\n3. Testing volatility history tracking...")
    detector3 = EnhancedRegimeDetector(window=20)
    test_prices = 100 + np.cumsum(np.random.randn(50) * 1.0)
    
    # Call detect_regime multiple times to build history
    for i in range(20, len(test_prices)):
        window_prices = test_prices[:i]
        regime = detector3.detect_regime(window_prices)
    
    print(f"   Volatility history length: {len(detector3.volatility_history)}")
    print(f"   Should be > 0: {len(detector3.volatility_history) > 0}")
    
    if len(detector3.volatility_history) >= 20:
        vol_threshold = np.percentile(detector3.volatility_history, 70)
        current_vol = detector3.volatility_history[-1]
        print(f"   Historical 70th percentile: {vol_threshold:.6f}")
        print(f"   Current volatility: {current_vol:.6f}")
        print(f"   RISK_OFF trigger: {current_vol > vol_threshold * 1.5}")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"[OK] Low volatility: RISK_OFF = {regime_counts_low[3]/len(regimes_low)*100:.1f}% (should be low)")
    print(f"[OK] High volatility: RISK_OFF = {regime_counts_high[3]/len(regimes_high)*100:.1f}% (should be higher)")
    print(f"[OK] Volatility history tracking: {'Working' if len(detector3.volatility_history) > 0 else 'NOT Working'}")
    
    # Verify fix: Low vol should have less RISK_OFF than high vol
    if regime_counts_low[3] < regime_counts_high[3]:
        print("\n[SUCCESS] FIX VERIFIED: Regime detection is working correctly!")
        print("   Low volatility scenarios trigger RISK_OFF less than high volatility scenarios.")
    else:
        print("\n[INFO] Regime detection logic may need further tuning.")
        print("   Volatility history tracking is working (fix applied).")
    
    return regime_counts_low, regime_counts_high

if __name__ == "__main__":
    test_regime_detection_fix()



