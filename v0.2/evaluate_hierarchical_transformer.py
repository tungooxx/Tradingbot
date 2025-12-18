"""
Evaluation Framework for Hierarchical Transformer Trading Model
================================================================
Step 5: Comprehensive evaluation and comparison

Evaluates the trained hierarchical transformer model on out-of-sample data
and compares against baseline strategies.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse

# Add v0.1 to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v0.1'))
# Add v0.2 to path (current directory) so we can import local modules
sys.path.insert(0, os.path.dirname(__file__))
from dapgio_improved import StockTradingEnv, TradingConfig

# Import v0.2 components
from data_preprocessor import MultiTimescalePreprocessor
from regime_detector import EnhancedRegimeDetector
from transformer_encoders import MultiScaleTransformerEncoder
from execution_agents import ExecutionAgent, create_execution_agents
from meta_strategy_agent import MetaStrategyAgent
from hierarchical_env import MetaStrategyEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PerformanceMetrics:
    """Calculate trading performance metrics"""
    
    @staticmethod
    def calculate_returns(equity_curve: np.ndarray) -> np.ndarray:
        """Calculate period returns from equity curve"""
        if len(equity_curve) < 2:
            return np.array([])
        # Avoid division by zero - replace zeros with small epsilon
        denominator = equity_curve[:-1].copy()
        denominator[denominator == 0] = 1e-10
        returns = np.diff(equity_curve) / denominator
        # Replace inf/nan with 0
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        return returns
    
    @staticmethod
    def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    @staticmethod
    def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (downside deviation only)"""
        if len(returns) == 0:
            return 0.0
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    @staticmethod
    def max_drawdown(equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(equity_curve) == 0:
            return 0.0
        # Replace zeros with small epsilon to avoid division issues
        equity_curve_safe = equity_curve.copy()
        equity_curve_safe[equity_curve_safe == 0] = 1e-10
        peak = np.maximum.accumulate(equity_curve_safe)
        # Avoid division by zero
        peak[peak == 0] = 1e-10
        drawdown = (equity_curve_safe - peak) / peak
        min_dd = np.min(drawdown)
        # Clamp to reasonable range (-100% to 0%)
        return max(-1.0, min(0.0, min_dd))
    
    @staticmethod
    def total_return(equity_curve: np.ndarray) -> float:
        """Calculate total return"""
        if len(equity_curve) == 0:
            return 0.0
        return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
    
    @staticmethod
    def win_rate(trades: List[Dict]) -> float:
        """Calculate win rate from trades"""
        if len(trades) == 0:
            return 0.0
        winning_trades = sum(1 for t in trades if t.get('profit', 0) > 0)
        return winning_trades / len(trades)
    
    @staticmethod
    def profit_factor(trades: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = sum(t.get('profit', 0) for t in trades if t.get('profit', 0) > 0)
        gross_loss = abs(sum(t.get('profit', 0) for t in trades if t.get('profit', 0) < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss


def load_models(ticker: str, mode: str, model_dir: str = "models_v0.2", 
                logger: Optional[logging.Logger] = None) -> Tuple[MultiScaleTransformerEncoder, 
                                                                   Dict[str, ExecutionAgent], 
                                                                   MetaStrategyAgent]:
    """Load all trained models"""
    logger = logger or logging.getLogger(__name__)
    
    # Determine input_dim: 8 for stocks (with market hours), 5 for crypto
    input_dim = 8 if mode == "stock" else 5
    
    # Load transformer
    transformer = MultiScaleTransformerEncoder(
        d_model=64, nhead=4, num_layers=2, dim_feedforward=256,
        dropout=0.1, input_dim=input_dim, mode=mode
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
        logger.warning(f"Transformer not found at {transformer_path}")
    
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
                    dropout=0.1, input_dim=input_dim, mode=mode, hidden_dim=128
                ).to(device)
                agent.load_state_dict(torch.load(agent_path, map_location=device))
                execution_agents[strategy] = agent
                logger.info(f"Loaded {strategy} execution agent")
                break
    
    # Load meta-strategy agent
    meta_agent = MetaStrategyAgent(
        d_model=64, nhead=4, num_layers=2, dim_feedforward=256,
        dropout=0.1, input_dim=input_dim, mode=mode, hidden_dim=128
    ).to(device)
    
    meta_path = os.path.join(model_dir, "meta_strategy_final.pth")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(model_dir, "meta_strategy.pth")
    
    if os.path.exists(meta_path):
        meta_agent.load_state_dict(torch.load(meta_path, map_location=device))
        logger.info("Loaded meta-strategy agent")
    else:
        logger.warning(f"Meta-strategy agent not found at {meta_path}")
    
    return transformer, execution_agents, meta_agent


def backtest_hierarchical_model(ticker: str, mode: str, window_size: int = 30,
                                test_split: float = 0.3, model_dir: str = "models_v0.2",
                                logger: Optional[logging.Logger] = None) -> Dict:
    """
    Backtest the hierarchical transformer model on out-of-sample data.
    
    Args:
        ticker: Symbol to test
        mode: "stock" or "crypto"
        window_size: Lookback window
        test_split: Fraction of data to use for testing (last portion)
        model_dir: Directory containing trained models
    
    Returns:
        Dictionary with performance metrics and trade history
    """
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("BACKTESTING HIERARCHICAL TRANSFORMER MODEL")
    logger.info("=" * 60)
    
    # Load models
    transformer, execution_agents, meta_agent = load_models(ticker, mode, model_dir, logger)
    meta_agent.eval()
    for agent in execution_agents.values():
        agent.eval()
    
    # Prepare data
    preprocessor = MultiTimescalePreprocessor(ticker, mode=mode, logger=logger)
    regime_detector = EnhancedRegimeDetector(window=20, logger=logger)
    
    aligned_data, features = preprocessor.process(window_size=window_size)
    if not aligned_data or not features:
        logger.error("Failed to fetch data")
        return {}
    
    # Split data (use last test_split for testing)
    regime_key = "1d" if "1d" in aligned_data else list(aligned_data.keys())[0]
    test_data = aligned_data[regime_key]
    total_len = len(test_data)
    test_start_idx = int(total_len * (1 - test_split))
    test_data = test_data.iloc[test_start_idx:].copy()
    
    logger.info(f"Testing on {len(test_data)} data points (last {test_split*100:.0f}%)")
    logger.info(f"Test start index: {test_start_idx}, Total length: {total_len}")
    
    # Create environment
    config = TradingConfig()
    config.mode = mode
    meta_env = MetaStrategyEnv(
        ticker=ticker,
        window_size=window_size,
        config=config,
        execution_horizon=10,
        execution_agents=execution_agents,
        logger=logger
    )
    
    # Track performance
    equity_curve = []
    trades = []
    strategy_selections = []
    current_balance = 2000.0  # Match v0.1 initial balance
    equity_curve.append(current_balance)
    
    state, _ = meta_env.reset()
    step = 0
    # Calculate max steps based on available data
    max_steps = min(len(test_data) - window_size - 10, total_len - test_start_idx - window_size - 10)
    max_steps = max(0, max_steps)  # Ensure non-negative
    
    logger.info(f"Running backtest for {max_steps} steps...")
    
    while step < max_steps:
        # Get windowed features (use current index in full dataset)
        current_idx = test_start_idx + step + window_size  # Add window_size to account for lookback
        windowed = preprocessor.get_window_features(features, window_size, current_idx)
        if not windowed:
            break
        
        # Prepare features dict (use full window, not just last element)
        features_dict = {}
        for interval, feat_array in windowed.items():
            if len(feat_array) > 0:
                # feat_array shape: (window_size, feature_dim)
                # Transformer expects (seq_len, batch, feature_dim) or (batch, seq_len, feature_dim)
                feat_tensor = torch.FloatTensor(feat_array).unsqueeze(0).to(device)  # (1, window_size, feature_dim)
                # Handle both "1wk" and "1w" keys - transformer expects "1w" for stock mode
                key = interval.replace("1wk", "1w")
                features_dict[key] = feat_tensor
        
        # Ensure we have all required scales for the mode
        # This is critical for matching the model's expected input dimensions
        if mode == "stock":
            # Stock mode requires both "1d" and "1w"
            required_keys = ["1d", "1w"]
            for req_key in required_keys:
                if req_key not in features_dict:
                    # Create zero tensor matching the shape of existing features
                    if features_dict:
                        sample_feat = next(iter(features_dict.values()))
                        features_dict[req_key] = torch.zeros_like(sample_feat)
                    else:
                        features_dict[req_key] = torch.zeros(1, window_size, 5, dtype=torch.float32).to(device)
        
        # For stock mode, ensure we have both "1d" and "1w" scales
        # If missing, pad with zeros to match expected dimensions
        if mode == "stock":
            if "1d" not in features_dict:
                # Create zero tensor for missing 1d scale
                if features_dict:  # Use shape from existing feature
                    sample_feat = next(iter(features_dict.values()))
                    features_dict["1d"] = torch.zeros_like(sample_feat)
                else:
                    features_dict["1d"] = torch.zeros(1, window_size, 5, dtype=torch.float32).to(device)
            
            if "1w" not in features_dict and "1wk" not in features_dict:
                # Create zero tensor for missing 1w scale
                if features_dict:  # Use shape from existing feature
                    sample_feat = next(iter(features_dict.values()))
                    features_dict["1w"] = torch.zeros_like(sample_feat)
                else:
                    features_dict["1w"] = torch.zeros(1, window_size, 5, dtype=torch.float32).to(device)
        
        # Get regime (use full aligned data for regime detection)
        if aligned_data and regime_key in aligned_data:
            regime_df = aligned_data[regime_key]
            if current_idx < len(regime_df):
                # Extract Close prices up to current_idx
                close_prices = regime_df['Close'].values[:current_idx+1] if 'Close' in regime_df.columns else regime_df.iloc[:current_idx+1, 0].values
                regime_features = torch.FloatTensor(regime_detector.get_regime_features(close_prices)).unsqueeze(0).to(device)
            else:
                regime_features = torch.zeros(1, 4, dtype=torch.float32).to(device)
        else:
            regime_features = torch.zeros(1, 4, dtype=torch.float32).to(device)
        
        # Get action from meta agent (deterministic for evaluation)
        with torch.no_grad():
            action, _, _ = meta_agent.act(features_dict, regime_features, deterministic=True)
        
        # Step environment
        next_state, reward, done, truncated, info = meta_env.step(action)
        
        # Track balance with better logging
        prev_balance = current_balance
        if 'balance' in info:
            current_balance = info['balance']
        else:
            # Fallback: get balance from environment
            if hasattr(meta_env, 'current_balance') and meta_env.current_balance is not None:
                current_balance = meta_env.current_balance
        
        # Get detailed trade information for debugging
        strategy = info.get('strategy', 'UNKNOWN')
        regime = info.get('regime', 'UNKNOWN')
        return_pct = info.get('return_pct', 0.0)
        
        # Log balance drops to zero with more context
        if prev_balance > 0 and current_balance <= 0:
            logger.warning(
                f"Balance dropped to zero at step {step}! "
                f"Prev: ${prev_balance:.2f}, Current: ${current_balance:.2f}, "
                f"Strategy: {strategy}, Regime: {regime}, Return: {return_pct:.2%}"
            )
            # Try to get more details from execution environment
            if hasattr(meta_env, 'execution_envs') and strategy in meta_env.execution_envs:
                exec_env = meta_env.execution_envs[strategy]
                if hasattr(exec_env, 'base_env'):
                    base_env = exec_env.base_env
                    if hasattr(base_env, 'shares'):
                        logger.warning(f"  Execution env state: shares={base_env.shares}, "
                                     f"entry_price={getattr(base_env, 'entry_price', 0):.2f}")
        
        # Track trades with more detail
        if 'trade' in info and info['trade']:
            trade_detail = {
                'step': step,
                'action': info.get('action', 'UNKNOWN'),
                'price': info.get('price', 0.0),
                'shares': info.get('shares', 0.0),
                'balance_before': prev_balance,
                'balance_after': current_balance,
                'profit': info.get('profit', 0.0),
                'strategy': strategy,
                'regime': regime
            }
            trades.append(trade_detail)
            
            # Log significant trades and positions
            # FIX: Show execution action (BUY/SELL/HOLD) instead of meta action
            exec_action = info.get('action', 'UNKNOWN')  # This is now the execution action (BUY/SELL/HOLD)
            is_unrealized = info.get('unrealized', False)
            profit = info.get('profit', 0.0)
            
            # Log if it's a trade (BUY/SELL) or significant unrealized P&L
            if info.get('trade', False) or (is_unrealized and abs(profit) > 0.5):
                action_label = exec_action
                if is_unrealized:
                    action_label = f"HOLD (unrealized)"
                
                logger.info(
                    f"Trade at step {step}: {action_label} "
                    f"@ ${info.get('price', 0.0):.2f}, "
                    f"Profit: ${profit:.2f}, "
                    f"Balance: ${prev_balance:.2f} -> ${current_balance:.2f}, "
                    f"Strategy: {strategy}"
                )
        
        equity_curve.append(max(0.0, current_balance))  # Ensure non-negative
        
        # Track strategy selection
        strategy_selections.append({
            'step': step,
            'strategy': info.get('strategy', 'UNKNOWN'),
            'regime': info.get('regime', 'UNKNOWN'),
            'balance': current_balance,
            'return_pct': info.get('return_pct', 0.0)
        })
        
        state = next_state
        step += 1
        
        if done or truncated:
            state, _ = meta_env.reset()
            if step >= max_steps:
                break
    
    equity_curve = np.array(equity_curve)
    
    # Calculate metrics
    returns = PerformanceMetrics.calculate_returns(equity_curve)
    metrics = {
        'total_return': PerformanceMetrics.total_return(equity_curve),
        'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns),
        'sortino_ratio': PerformanceMetrics.sortino_ratio(returns),
        'max_drawdown': PerformanceMetrics.max_drawdown(equity_curve),
        'final_balance': equity_curve[-1] if len(equity_curve) > 0 else 20.0,
        'initial_balance': equity_curve[0] if len(equity_curve) > 0 else 20.0,
        'num_trades': len(trades),
        'equity_curve': equity_curve,
        'strategy_selections': strategy_selections
    }
    
    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    
    # Handle NaN/inf for ratios
    sharpe = metrics['sharpe_ratio']
    sortino = metrics['sortino_ratio']
    sharpe_str = f"{sharpe:.2f}" if not (np.isnan(sharpe) or np.isinf(sharpe)) else "N/A (insufficient variance)"
    sortino_str = f"{sortino:.2f}" if not (np.isnan(sortino) or np.isinf(sortino)) else "N/A (insufficient variance)"
    
    logger.info(f"Sharpe Ratio: {sharpe_str}")
    logger.info(f"Sortino Ratio: {sortino_str}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"Final Balance: ${metrics['final_balance']:.2f}")
    logger.info(f"Initial Balance: ${metrics['initial_balance']:.2f}")
    logger.info(f"Number of Trades: {metrics['num_trades']}")
    
    # Additional statistics
    if len(equity_curve) > 1:
        min_balance = np.min(equity_curve)
        max_balance = np.max(equity_curve)
        logger.info(f"Min Balance: ${min_balance:.2f}")
        logger.info(f"Max Balance: ${max_balance:.2f}")
        if min_balance <= 0:
            logger.warning(f"⚠️  Balance reached zero or negative! This explains the -100% drawdown.")
    
    # Trade statistics
    if len(trades) > 0:
        profitable_trades = [t for t in trades if t.get('profit', 0) > 0]
        losing_trades = [t for t in trades if t.get('profit', 0) < 0]
        logger.info(f"Profitable Trades: {len(profitable_trades)}/{len(trades)}")
        logger.info(f"Losing Trades: {len(losing_trades)}/{len(trades)}")
        if len(profitable_trades) > 0:
            avg_profit = np.mean([t['profit'] for t in profitable_trades])
            logger.info(f"Average Profit: ${avg_profit:.2f}")
        if len(losing_trades) > 0:
            avg_loss = np.mean([t['profit'] for t in losing_trades])
            logger.info(f"Average Loss: ${avg_loss:.2f}")
    else:
        logger.warning("⚠️  No trades executed during backtest!")
    
    # Strategy performance summary
    if len(strategy_selections) > 0:
        logger.info("\n" + "=" * 60)
        logger.info("STRATEGY PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        strategy_stats = {}
        for sel in strategy_selections:
            strategy = sel.get('strategy', 'UNKNOWN')
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'count': 0,
                    'total_return': 0.0,
                    'min_balance': float('inf'),
                    'max_balance': float('-inf')
                }
            strategy_stats[strategy]['count'] += 1
            strategy_stats[strategy]['total_return'] += sel.get('return_pct', 0.0)
            balance = sel.get('balance', 0.0)
            strategy_stats[strategy]['min_balance'] = min(strategy_stats[strategy]['min_balance'], balance)
            strategy_stats[strategy]['max_balance'] = max(strategy_stats[strategy]['max_balance'], balance)
        
        for strategy, stats in sorted(strategy_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            avg_return = stats['total_return'] / stats['count'] if stats['count'] > 0 else 0.0
            logger.info(
                f"{strategy}: Used {stats['count']} times, "
                f"Avg Return: {avg_return:.2%}, "
                f"Balance Range: ${stats['min_balance']:.2f} - ${stats['max_balance']:.2f}"
            )
    
    return metrics


def backtest_baseline(ticker: str, mode: str, strategy: str = "buy_hold",
                     window_size: int = 30, test_split: float = 0.3,
                     logger: Optional[logging.Logger] = None) -> Dict:
    """
    Backtest baseline strategies for comparison.
    
    Args:
        ticker: Symbol to test
        mode: "stock" or "crypto"
        strategy: "buy_hold", "random", or strategy name
        window_size: Lookback window
        test_split: Fraction of data for testing
    """
    logger = logger or logging.getLogger(__name__)
    logger.info(f"Backtesting {strategy} baseline strategy...")
    
    # Create environment
    config = TradingConfig()
    config.mode = mode
    # Bypass entry filters for baseline strategies (buy-hold, random)
    config.bypass_entry_filters = True
    env = StockTradingEnv(ticker=ticker, window_size=window_size, config=config, logger=logger)
    
    # Get test data range (same as hierarchical model)
    total_len = len(env.data)
    test_start_idx = int(total_len * (1 - test_split))
    
    # Reset environment and set to test start position
    state, _ = env.reset()
    env.current_step = test_start_idx + window_size  # Start from test period
    
    equity_curve = []
    initial_balance = env.balance
    current_balance = initial_balance
    equity_curve.append(current_balance)
    
    # Track if we've bought (for buy_hold)
    has_bought = False
    buy_price = None  # Track buy price for debugging
    
    # Calculate max steps (same as hierarchical model)
    max_steps = min(len(env.data) - window_size - 10, total_len - test_start_idx - window_size - 10)
    max_steps = max(0, max_steps)
    
    step = 0
    
    while step < max_steps:
        if strategy == "buy_hold":
            # Buy and hold: buy once at start of test period, then hold
            if not has_bought and env.shares == 0 and env.balance > 0:
                action = 1  # BUY
                has_bought = True
                # Get current price before buy for debugging
                current_price_before = env.data[min(env.current_step, len(env.data)-1), 0]
                logger.debug(f"Buy-hold: Attempting to buy at step {step}, balance=${env.balance:.2f}, price=${current_price_before:.2f}")
            else:
                action = 0  # HOLD
        elif strategy == "random":
            # Random strategy: random actions
            action = env.action_space.sample()
        else:
            # Default to HOLD for unknown strategies
            action = 0
        
        state, reward, done, truncated, info = env.step(action)
        
        # Track buy price for buy_hold strategy
        if strategy == "buy_hold":
            if has_bought and buy_price is None:
                if env.shares > 0:
                    buy_price = env.entry_price if hasattr(env, 'entry_price') and env.entry_price > 0 else env.data[min(env.current_step, len(env.data)-1), 0]
                    logger.info(f"Buy-hold: Successfully bought {env.shares:.4f} shares at ${buy_price:.2f}, balance=${env.balance:.2f}")
                else:
                    # Buy was attempted but didn't execute - log why
                    logger.warning(f"Buy-hold: Buy attempted at step {step} but shares=0. Balance=${env.balance:.2f}, action={action}")
                    # Check if entry filters blocked it
                    if hasattr(env, 'logger'):
                        env.logger.setLevel(logging.DEBUG)  # Enable debug to see filter messages
        
        # Calculate current balance (cash + position value)
        # CRITICAL FIX: Use index [0] for Close price, not [3] (RSI)
        if env.shares > 0:
            current_price = env.data[min(env.current_step, len(env.data)-1), 0]  # Index 0 = Close price
            current_balance = env.balance + (env.shares * current_price)
        else:
            current_balance = env.balance
        
        equity_curve.append(max(0.0, current_balance))  # Ensure non-negative
        
        step += 1
        if done or truncated:
            # Reset but continue from test period
            state, _ = env.reset()
            env.current_step = test_start_idx + window_size
            if step >= max_steps:
                break
    
    equity_curve = np.array(equity_curve)
    returns = PerformanceMetrics.calculate_returns(equity_curve)
    
    # Debug buy-hold calculation
    if strategy == "buy_hold":
        if len(equity_curve) > 0:
            start_equity = equity_curve[0]
            end_equity = equity_curve[-1]
            if buy_price is not None:
                final_price = env.data[min(env.current_step, len(env.data)-1), 0] if env.shares > 0 else None
                if final_price:
                    price_return = ((final_price / buy_price) - 1) * 100
                    logger.debug(f"Buy-hold debug: Start equity=${start_equity:.2f}, End equity=${end_equity:.2f}, "
                               f"Buy price=${buy_price:.2f}, Final price=${final_price:.2f}, "
                               f"Price return={price_return:.2f}%, Equity return={((end_equity/start_equity)-1)*100:.2f}%")
            else:
                logger.warning(f"Buy-hold: Buy price not tracked, shares={env.shares}, balance=${env.balance:.2f}")
    
    metrics = {
        'total_return': PerformanceMetrics.total_return(equity_curve),
        'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns),
        'sortino_ratio': PerformanceMetrics.sortino_ratio(returns),
        'max_drawdown': PerformanceMetrics.max_drawdown(equity_curve),
        'final_balance': equity_curve[-1] if len(equity_curve) > 0 else initial_balance,
        'initial_balance': initial_balance,
        'equity_curve': equity_curve,
        'num_trades': 1 if strategy == "buy_hold" else step  # Buy-hold has 1 trade
    }
    
    logger.info(f"{strategy} baseline results: Return={metrics['total_return']:.2%}, "
                f"Final Balance=${metrics['final_balance']:.2f}")
    
    return metrics


def compare_strategies(ticker: str, mode: str, window_size: int = 30,
                      test_split: float = 0.3, model_dir: str = "models_v0.2",
                      logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Compare hierarchical model against baseline strategies.
    
    Returns:
        DataFrame with comparison metrics
    """
    logger = logger or logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("COMPARING STRATEGIES")
    logger.info("=" * 60)
    
    results = []
    
    # Test hierarchical model
    logger.info("Testing Hierarchical Transformer Model...")
    hierarchical_metrics = backtest_hierarchical_model(
        ticker, mode, window_size, test_split, model_dir, logger
    )
    if hierarchical_metrics:
        # Handle NaN/inf for ratios
        sharpe = hierarchical_metrics['sharpe_ratio']
        sortino = hierarchical_metrics['sortino_ratio']
        sharpe_str = f"{sharpe:.2f}" if not (np.isnan(sharpe) or np.isinf(sharpe)) else "N/A"
        sortino_str = f"{sortino:.2f}" if not (np.isnan(sortino) or np.isinf(sortino)) else "N/A"
        
        results.append({
            'Strategy': 'Hierarchical Transformer',
            'Total Return': f"{hierarchical_metrics['total_return']:.2%}",
            'Sharpe Ratio': sharpe_str,
            'Sortino Ratio': sortino_str,
            'Max Drawdown': f"{hierarchical_metrics['max_drawdown']:.2%}",
            'Final Balance': f"${hierarchical_metrics['final_balance']:.2f}",
            'Initial Balance': f"${hierarchical_metrics.get('initial_balance', 20.0):.2f}",
            'Num Trades': hierarchical_metrics.get('num_trades', 0)
        })
    
    # Test baselines
    for baseline in ["buy_hold", "random"]:
        logger.info(f"\nTesting {baseline} baseline...")
        baseline_metrics = backtest_baseline(ticker, mode, baseline, window_size, test_split, logger)
        if baseline_metrics:
            # Handle NaN/inf for ratios
            sharpe = baseline_metrics['sharpe_ratio']
            sortino = baseline_metrics['sortino_ratio']
            sharpe_str = f"{sharpe:.2f}" if not (np.isnan(sharpe) or np.isinf(sharpe)) else "N/A"
            sortino_str = f"{sortino:.2f}" if not (np.isnan(sortino) or np.isinf(sortino)) else "N/A"
            
            results.append({
                'Strategy': baseline.replace('_', ' ').title(),
                'Total Return': f"{baseline_metrics['total_return']:.2%}",
                'Sharpe Ratio': sharpe_str,
                'Sortino Ratio': sortino_str,
                'Max Drawdown': f"{baseline_metrics['max_drawdown']:.2%}",
                'Final Balance': f"${baseline_metrics['final_balance']:.2f}",
                'Initial Balance': f"${baseline_metrics.get('initial_balance', 20.0):.2f}",
                'Num Trades': baseline_metrics.get('num_trades', 0)
            })
    
    df = pd.DataFrame(results)
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)
    logger.info("\n" + df.to_string(index=False))
    
    # Summary: Find best strategy by total return
    if len(results) > 0:
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        
        # Parse returns for comparison (remove % sign and convert)
        def parse_return(return_str):
            try:
                return float(return_str.replace('%', ''))
            except:
                return float('-inf')
        
        best_strategy = max(results, key=lambda x: parse_return(x.get('Total Return', '0%')))
        worst_strategy = min(results, key=lambda x: parse_return(x.get('Total Return', '0%')))
        
        logger.info(f"🏆 Best Strategy: {best_strategy['Strategy']} "
                   f"(Return: {best_strategy['Total Return']}, "
                   f"Final Balance: {best_strategy['Final Balance']})")
        logger.info(f"📉 Worst Strategy: {worst_strategy['Strategy']} "
                   f"(Return: {worst_strategy['Total Return']}, "
                   f"Final Balance: {worst_strategy['Final Balance']})")
        
        # Compare hierarchical model vs buy-hold
        hierarchical_result = next((r for r in results if 'Hierarchical' in r['Strategy']), None)
        buyhold_result = next((r for r in results if 'Buy Hold' in r['Strategy']), None)
        
        if hierarchical_result and buyhold_result:
            hier_return = parse_return(hierarchical_result['Total Return'])
            bh_return = parse_return(buyhold_result['Total Return'])
            diff = hier_return - bh_return
            
            if diff > 0:
                logger.info(f"\n✅ Hierarchical Transformer outperforms Buy-Hold by {diff:.2f} percentage points")
            elif diff < 0:
                logger.info(f"\n❌ Hierarchical Transformer underperforms Buy-Hold by {abs(diff):.2f} percentage points")
            else:
                logger.info(f"\n➖ Hierarchical Transformer matches Buy-Hold performance")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hierarchical Transformer Trading Model")
    parser.add_argument("--ticker", type=str, default="^IXIC", help="Ticker symbol")
    parser.add_argument("--mode", type=str, default="stock", choices=["stock", "crypto"], help="Trading mode")
    parser.add_argument("--window-size", type=int, default=30, help="Window size")
    parser.add_argument("--test-split", type=float, default=0.3, help="Test data split (last portion)")
    parser.add_argument("--model-dir", type=str, default="models_v0.2", help="Model directory")
    parser.add_argument("--compare", action="store_true", default=True, help="Compare against baselines (default: True)")
    parser.add_argument("--no-compare", dest="compare", action="store_false", help="Disable baseline comparison")
    
    args = parser.parse_args()
    
    # Compare with baselines by default
    if args.compare:
        compare_strategies(
            args.ticker, args.mode, args.window_size, 
            args.test_split, args.model_dir, logger
        )
    else:
        backtest_hierarchical_model(
            args.ticker, args.mode, args.window_size,
            args.test_split, args.model_dir, logger
        )


if __name__ == "__main__":
    main()

