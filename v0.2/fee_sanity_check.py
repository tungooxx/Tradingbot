"""
10% Fee Sanity Check
====================
This test verifies that fees are actually being applied correctly.

Expected Result:
- Agent should stop trading immediately (0 trades)
- Or go bankrupt ($0 balance)

If agent still makes 1,000 trades: Fee code is disconnected.
"""

import sys
import os
import logging
import torch
import numpy as np
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))
sys.path.insert(0, str(parent_dir / "v0.1"))
sys.path.insert(0, str(parent_dir / "v0.2"))

# Import directly
from dapgio_improved import TradingConfig, StockTradingEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_fee_sanity_check(ticker: str = "^IXIC", mode: str = "stock", window_size: int = 30, steps: int = 500):
    """
    Run sanity check with 10% fees.
    
    Args:
        ticker: Ticker symbol
        mode: "stock" or "crypto"
        window_size: Window size for environment
        steps: Number of steps to run
    """
    logger.info("=" * 60)
    logger.info("10% FEE SANITY CHECK")
    logger.info("=" * 60)
    logger.info(f"Ticker: {ticker}")
    logger.info(f"Mode: {mode}")
    logger.info(f"Steps: {steps}")
    logger.info("=" * 60)
    
    # Create config with 10% fees (INSANE!)
    config = TradingConfig()
    config.mode = mode
    config.commission_rate = 0.10  # 10% commission (INSANE!)
    config.slippage_bps = 0.0  # No slippage for simplicity
    config.interval = "1d" if mode == "stock" else "1h"  # Daily for stocks, hourly for crypto
    
    # For daily stock data, market hours filter removes all data (daily bars have no time)
    # We'll use crypto mode for the sanity check to avoid this issue
    if mode == "stock":
        logger.warning("⚠️  Stock mode with daily data may fail due to market hours filter")
        logger.warning("   Daily bars have no time component, so filter removes all data")
        logger.warning("   Consider using --mode crypto for this test")
    logger.info(f"⚠️  COMMISSION RATE SET TO 10% (INSANE!)")
    logger.info(f"   Position size: {config.max_position_size_pct * 100}%")
    logger.info(f"   Initial balance: ${config.initial_balance if hasattr(config, 'initial_balance') else 2000.0:.2f}")
    
    # Calculate expected fee per trade
    initial_balance = 2000.0
    position_value = initial_balance * config.max_position_size_pct
    fee_per_buy = position_value * config.commission_rate
    fee_per_round_trip = fee_per_buy * 2  # BUY + SELL
    logger.info(f"   Fee per BUY: ${fee_per_buy:.2f}")
    logger.info(f"   Fee per round trip: ${fee_per_round_trip:.2f}")
    logger.info(f"   Expected: Agent should stop trading or go bankrupt!")
    logger.info("=" * 60)
    
    # Create environment
    try:
        env = StockTradingEnv(
            ticker=ticker,
            window_size=window_size,
            config=config,
            logger=logger
        )
        logger.info("Environment created successfully")
    except Exception as e:
        logger.error(f"Failed to create environment: {e}", exc_info=True)
        return
    
    # Reset environment
    state, info = env.reset()
    initial_balance = env.balance
    logger.info(f"Initial balance: ${initial_balance:.2f}")
    
    # Track statistics
    total_trades = 0
    buy_count = 0
    sell_count = 0
    hold_count = 0
    total_fees_paid = 0.0
    balance_history = [initial_balance]
    
    # Run for specified steps
    logger.info(f"\nRunning {steps} steps with 10% fees...")
    logger.info("=" * 60)
    
    for step in range(steps):
        # Random action (to test if fees stop trading)
        action = env.action_space.sample()
        
        # Step environment
        try:
            next_state, reward, done, truncated, info = env.step(action)
            
            # Track actions
            if action == 0:
                hold_count += 1
            elif action == 1:
                buy_count += 1
            elif action == 2:
                sell_count += 1
            
            # Track trades
            if info.get('trade_executed', False):
                total_trades += 1
                # Estimate fees paid (approximate)
                if action == 1:  # BUY
                    position_value = env.balance * config.max_position_size_pct
                    fee = position_value * config.commission_rate
                    total_fees_paid += fee
                elif action == 2:  # SELL
                    # Estimate based on position value
                    if hasattr(env, 'shares') and env.shares > 0:
                        position_value = env.shares * env.entry_price if hasattr(env, 'entry_price') else 200
                        fee = position_value * config.commission_rate
                        total_fees_paid += fee
            
            # Track balance
            balance_history.append(env.balance)
            
            # Log every 50 steps
            if (step + 1) % 50 == 0:
                logger.info(f"Step {step + 1}/{steps}: Balance=${env.balance:.2f}, Trades={total_trades}, "
                          f"Actions: HOLD={hold_count}, BUY={buy_count}, SELL={sell_count}")
            
            # Check if bankrupt
            if env.balance <= 0:
                logger.warning(f"⚠️  BANKRUPT at step {step + 1}! Balance: ${env.balance:.2f}")
                break
            
            # Reset if done
            if done or truncated:
                state, info = env.reset()
                logger.info(f"Episode ended at step {step + 1}, resetting...")
        
        except Exception as e:
            logger.error(f"Error at step {step + 1}: {e}", exc_info=True)
            break
    
    # Final statistics
    logger.info("=" * 60)
    logger.info("SANITY CHECK RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Steps: {steps}")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Actions: HOLD={hold_count}, BUY={buy_count}, SELL={sell_count}")
    logger.info(f"Initial Balance: ${initial_balance:.2f}")
    logger.info(f"Final Balance: ${env.balance:.2f}")
    logger.info(f"Balance Change: ${env.balance - initial_balance:.2f}")
    logger.info(f"Estimated Fees Paid: ${total_fees_paid:.2f}")
    logger.info("=" * 60)
    
    # Determine result
    if total_trades == 0:
        logger.info("✅ PASS: Agent stopped trading (0 trades)")
        logger.info("   Fees are working correctly!")
    elif env.balance <= 0:
        logger.info("✅ PASS: Agent went bankrupt")
        logger.info("   Fees are working correctly!")
    elif total_trades < 10:
        logger.info("⚠️  PARTIAL PASS: Agent made very few trades")
        logger.info(f"   {total_trades} trades is reasonable with 10% fees")
    else:
        logger.error("❌ FAIL: Agent still made many trades!")
        logger.error(f"   {total_trades} trades with 10% fees suggests fees are NOT working!")
        logger.error("   Fee code may be disconnected!")
    
    logger.info("=" * 60)
    
    return {
        'total_trades': total_trades,
        'initial_balance': initial_balance,
        'final_balance': env.balance,
        'balance_change': env.balance - initial_balance,
        'estimated_fees': total_fees_paid,
        'passed': total_trades == 0 or env.balance <= 0 or total_trades < 10
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="10% Fee Sanity Check")
    parser.add_argument("--ticker", type=str, default="^IXIC", help="Ticker symbol")
    parser.add_argument("--mode", type=str, default="stock", choices=["stock", "crypto"], help="Trading mode")
    parser.add_argument("--window-size", type=int, default=30, help="Window size")
    parser.add_argument("--steps", type=int, default=500, help="Number of steps to run")
    
    args = parser.parse_args()
    
    result = run_fee_sanity_check(
        ticker=args.ticker,
        mode=args.mode,
        window_size=args.window_size,
        steps=args.steps
    )
    
    if result and result['passed']:
        logger.info("✅ SANITY CHECK PASSED")
        sys.exit(0)
    else:
        logger.error("❌ SANITY CHECK FAILED")
        sys.exit(1)
