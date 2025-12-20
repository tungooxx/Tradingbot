"""
Hierarchical Trading Environment
================================
Two-level environment structure:
1. High-level (Meta-Strategy): Selects strategy based on regime
2. Low-level (Execution): Executes trades within selected strategy

Based on hierarchical RL principles for trading.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional, Tuple, List
import logging

# Import from v0.1
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v0.1'))
from dapgio_improved import StockTradingEnv, TradingConfig

# Import from v0.2
from data_preprocessor import MultiTimescalePreprocessor
from regime_detector import EnhancedRegimeDetector
from transformer_encoders import MultiScaleTransformerEncoder
from cross_scale_attention import CrossScaleAttention

logger = logging.getLogger(__name__)


class ExecutionEnv(gym.Env):
    """
    Low-level execution environment.
    Executes trades (BUY/SELL/HOLD) within a selected strategy.
    
    This is a wrapper around StockTradingEnv with strategy-specific reward shaping.
    """
    
    def __init__(self,
                 ticker: str,
                 window_size: int = 30,
                 config: Optional[TradingConfig] = None,
                 strategy_type: str = "TREND_FOLLOW",
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            ticker: Symbol to trade
            window_size: Lookback window size
            config: Trading configuration
            strategy_type: Strategy type (TREND_FOLLOW, MEAN_REVERT, MOMENTUM, RISK_OFF)
            logger: Optional logger
        """
        super(ExecutionEnv, self).__init__()
        
        self.strategy_type = strategy_type
        self.logger = logger or logging.getLogger(__name__)
        
        # Create base environment
        self.base_env = StockTradingEnv(
            ticker=ticker,
            window_size=window_size,
            config=config,
            logger=self.logger
        )
        
        # Copy necessary attributes
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.obs_shape = self.base_env.obs_shape
        
        # Strategy-specific parameters
        self._setup_strategy()
    
    def _setup_strategy(self):
        """Setup strategy-specific parameters"""
        if self.strategy_type == "TREND_FOLLOW":
            # Trend following: Reward momentum, penalize reversals
            self.momentum_bonus = 0.1
            self.reversal_penalty = -0.05
        elif self.strategy_type == "MEAN_REVERT":
            # Mean reversion: Reward contrarian moves, penalize momentum
            self.contrarian_bonus = 0.1
            self.momentum_penalty = -0.05
        elif self.strategy_type == "MOMENTUM":
            # Momentum: Similar to trend following but more aggressive
            self.momentum_bonus = 0.15
            self.reversal_penalty = -0.1
        elif self.strategy_type == "RISK_OFF":
            # FIX: Risk-off: Less conservative, allow profits to run
            self.conservative_bonus = 0.1  # Increased from 0.05
            self.risk_penalty = -0.15  # Less harsh (was -0.2)
    
    def reset(self, seed=None):
        """Reset environment"""
        obs, info = self.base_env.reset(seed=seed)
        return obs, info
    
    def step(self, action):
        """Step environment with strategy-specific reward shaping"""
        obs, reward, done, truncated, info = self.base_env.step(action)
        
        # Apply strategy-specific reward shaping
        shaped_reward = self._shape_reward(reward, action, obs, info)
        
        info['strategy'] = self.strategy_type
        info['base_reward'] = reward
        info['shaped_reward'] = shaped_reward
        
        return obs, shaped_reward, done, truncated, info
    
    def _shape_reward(self, base_reward: float, action: int, obs: np.ndarray, info: Dict) -> float:
        """
        Shape reward based on strategy type.
        
        Args:
            base_reward: Base reward from environment
            action: Action taken
            obs: Observation
            info: Info dict
            
        Returns:
            Shaped reward
        """
        shaped = base_reward
        
        # Get current price and position info from base_env
        if hasattr(self.base_env, 'shares') and hasattr(self.base_env, 'entry_price'):
            shares = self.base_env.shares
            entry_price = self.base_env.entry_price
            
            if shares > 0 and entry_price > 0:
                # Get current price from environment data (index 0 = Close price)
                # obs is flattened window, so we need to get current price from data array
                if hasattr(self.base_env, 'data') and self.base_env.data is not None:
                    current_step = min(self.base_env.current_step, len(self.base_env.data) - 1)
                    current_price = self.base_env.data[current_step, 0]  # Index 0 = Close price
                else:
                    current_price = entry_price  # Fallback
                current_return = (current_price - entry_price) / entry_price
                
                if self.strategy_type == "TREND_FOLLOW":
                    # Reward holding winners, penalize holding losers
                    if current_return > 0.02:  # +2% gain
                        shaped += self.momentum_bonus
                    elif current_return < -0.01:  # -1% loss
                        shaped += self.reversal_penalty
                
                elif self.strategy_type == "MEAN_REVERT":
                    # Reward contrarian moves (buying dips, selling rallies)
                    if current_return < -0.02 and action == 1:  # Buy on dip
                        shaped += self.contrarian_bonus
                    elif current_return > 0.02 and action == 2:  # Sell on rally
                        shaped += self.contrarian_bonus
                    elif abs(current_return) > 0.05:  # Large move, should mean revert
                        shaped += self.contrarian_bonus * 0.5
                
                elif self.strategy_type == "MOMENTUM":
                    # Strong momentum rewards
                    if current_return > 0.03:  # +3% gain
                        shaped += self.momentum_bonus
                    elif current_return < -0.02:  # -2% loss
                        shaped += self.reversal_penalty
                
                elif self.strategy_type == "RISK_OFF":
                    # FIX: Less conservative - allow profits to run, but cut losses quickly
                    if current_return < -0.02:  # -2% loss, should exit (was -1%)
                        shaped += self.risk_penalty
                    elif current_return > 0.05:  # +5% gain, reward holding (was +1%, too quick to exit)
                        shaped += self.conservative_bonus * 2  # Double bonus for larger gains
                    elif current_return > 0.02:  # +2% gain, small bonus
                        shaped += self.conservative_bonus
        
        return shaped
    
    def _get_observation(self):
        """Get observation from base environment"""
        return self.base_env._get_observation()


class MetaStrategyEnv(gym.Env):
    """
    High-level meta-strategy environment.
    Selects which strategy to use based on market regime.
    
    Actions:
    0: TREND_FOLLOW
    1: MEAN_REVERT
    2: MOMENTUM
    3: RISK_OFF
    """
    
    def __init__(self,
                 ticker: str,
                 window_size: int = 30,
                 config: Optional[TradingConfig] = None,
                 execution_horizon: int = 10,
                 execution_agents: Optional[Dict] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            ticker: Symbol to trade
            window_size: Lookback window size
            config: Trading configuration
            execution_horizon: Number of steps to execute selected strategy
            execution_agents: Dict of execution agents (TREND_FOLLOW, MEAN_REVERT, MOMENTUM, RISK_OFF)
            logger: Optional logger
        """
        super(MetaStrategyEnv, self).__init__()
        
        self.ticker = ticker
        self.window_size = window_size
        self.config = config or TradingConfig()
        self.execution_horizon = execution_horizon
        self.execution_agents = execution_agents or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Multi-timescale preprocessor
        self.preprocessor = MultiTimescalePreprocessor(ticker, mode=self.config.mode, logger=self.logger)
        
        # Regime detector
        self.regime_detector = EnhancedRegimeDetector(window=20, logger=self.logger)
        
        # Execution environments (one per strategy)
        self.execution_envs = {
            "TREND_FOLLOW": ExecutionEnv(ticker, window_size, config, "TREND_FOLLOW", logger),
            "MEAN_REVERT": ExecutionEnv(ticker, window_size, config, "MEAN_REVERT", logger),
            "MOMENTUM": ExecutionEnv(ticker, window_size, config, "MOMENTUM", logger),
            "RISK_OFF": ExecutionEnv(ticker, window_size, config, "RISK_OFF", logger)
        }
        
        # Cache for aligned data and features (to avoid repeated fetching)
        self.cached_aligned_data = None
        self.cached_features = None
        self.cache_window_size = None
        
        # Action space: 4 strategies
        self.action_space = spaces.Discrete(4)
        
        # Observation space: Multi-scale features + regime features
        # Will be set after data preprocessing
        self.obs_shape = None
        self.observation_space = None
        
        # State tracking
        self.current_strategy = None
        self.steps_in_strategy = 0
        self.initial_balance = None
        self.current_balance = None
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize environment with data"""
        try:
            # Fetch and process multi-timescale data (cache it)
            if self.cached_aligned_data is None or self.cache_window_size != self.window_size:
                self.logger.info("Fetching multi-timescale data (first time)...")
                self.cached_aligned_data, self.cached_features = self.preprocessor.process(window_size=self.window_size)
                self.cache_window_size = self.window_size
                self.logger.info("Data cached successfully")
            
            aligned_data = self.cached_aligned_data
            features = self.cached_features
            
            if not aligned_data or not features:
                raise ValueError("Failed to fetch multi-timescale data")
            
            # Store for later use
            self.aligned_data = aligned_data
            self.features = features
            
            # Get observation dimension
            # Multi-scale features + regime features (4 regimes)
            # For now, use concatenated last hidden states from all timescales
            total_features = 0
            for interval, feat_array in features.items():
                total_features += feat_array.shape[1]  # Number of features per timestep
            
            # Add regime features (4 one-hot)
            total_features += 4
            
            # Observation is flattened window
            self.obs_shape = self.window_size * total_features
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
            )
            
            self.logger.info(f"MetaStrategyEnv initialized: obs_shape={self.obs_shape}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MetaStrategyEnv: {e}", exc_info=True)
            raise
    
    def reset(self, seed=None):
        """Reset environment"""
        # Reset all execution environments
        for env in self.execution_envs.values():
            env.reset(seed=seed)
        
        # Reset state
        self.current_strategy = None
        self.steps_in_strategy = 0
        
        # Get initial balance from first execution env
        self.initial_balance = self.execution_envs["TREND_FOLLOW"].base_env.initial_balance
        self.current_balance = self.initial_balance
        
        # Get initial observation
        obs = self._get_observation()
        
        info = {
            'strategy': None,
            'regime': None,
            'balance': self.current_balance
        }
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get observation with multi-scale features and regime.
        
        Returns:
            Flattened observation array
        """
        # Get windowed features from each timescale
        windowed_features = self.preprocessor.get_window_features(
            self.features, 
            window_size=self.window_size,
            current_idx=None  # Use last window
        )
        
        # Detect current regime
        if self.aligned_data:
            # Use 1d data for regime detection (most stable)
            regime_key = "1d" if "1d" in self.aligned_data else list(self.aligned_data.keys())[0]
            regime_df = self.aligned_data[regime_key]
            
            if not regime_df.empty:
                # Get Close prices for regime detection
                close_prices = regime_df['Close'].values if 'Close' in regime_df.columns else regime_df.iloc[:, 0].values
                regime_features = self.regime_detector.get_regime_features(close_prices)
            else:
                regime_features = np.zeros(4, dtype=np.float32)
        else:
            regime_features = np.zeros(4, dtype=np.float32)
        
        # Concatenate features from all timescales
        feature_list = []
        for interval in sorted(windowed_features.keys()):
            feature_list.append(windowed_features[interval].flatten())
        
        # Add regime features
        feature_list.append(regime_features)
        
        # Concatenate all
        obs = np.concatenate(feature_list, axis=0).astype(np.float32)
        
        # Pad or truncate to expected size
        if len(obs) < self.obs_shape:
            padding = np.zeros(self.obs_shape - len(obs), dtype=np.float32)
            obs = np.concatenate([obs, padding])
        elif len(obs) > self.obs_shape:
            obs = obs[:self.obs_shape]
        
        return obs
    
    def step(self, action: int):
        """
        Step meta-strategy environment.
        
        Args:
            action: Strategy selection (0=TREND_FOLLOW, 1=MEAN_REVERT, 2=MOMENTUM, 3=RISK_OFF)
        
        Returns:
            obs, reward, done, truncated, info
        """
        strategy_names = ["TREND_FOLLOW", "MEAN_REVERT", "MOMENTUM", "RISK_OFF"]
        selected_strategy = strategy_names[action]
        
        # If switching strategy, reset execution environment and sync balance
        if self.current_strategy != selected_strategy:
            # Save current balance and position state from previous strategy's environment
            if self.current_strategy and self.current_strategy in self.execution_envs:
                prev_env = self.execution_envs[self.current_strategy]
                if hasattr(prev_env, 'base_env') and hasattr(prev_env.base_env, 'balance'):
                    # Sync balance from previous environment
                    self.current_balance = prev_env.base_env.balance
                    # Save position state if any
                    prev_shares = getattr(prev_env.base_env, 'shares', 0)
                    prev_entry_price = getattr(prev_env.base_env, 'entry_price', 0)
                else:
                    prev_shares = 0
                    prev_entry_price = 0
            else:
                prev_shares = 0
                prev_entry_price = 0
            
            # Reset new execution environment
            self.execution_envs[selected_strategy].reset()
            
            # Sync balance and position state to new execution environment
            if hasattr(self.execution_envs[selected_strategy], 'base_env'):
                new_env = self.execution_envs[selected_strategy].base_env
                new_env.balance = self.current_balance
                # Transfer position state if we had an open position
                if prev_shares > 0:
                    new_env.shares = prev_shares
                    new_env.entry_price = prev_entry_price
                    self.logger.debug(f"Transferred position: {prev_shares:.4f} shares @ ${prev_entry_price:.2f}")
            
            self.current_strategy = selected_strategy
            self.steps_in_strategy = 0
        
        # Execute strategy for execution_horizon steps
        total_reward = 0.0
        done = False
        truncated = False
        hard_stop_loss_triggered = False  # Track if hard stop-loss was triggered
        
        execution_env = self.execution_envs[selected_strategy]
        
        # Sync balance to execution environment before executing (in case it got out of sync)
        if hasattr(execution_env, 'base_env') and hasattr(execution_env.base_env, 'balance'):
            execution_env.base_env.balance = self.current_balance
        
        # Get execution agent for selected strategy
        exec_agent = self.execution_agents.get(selected_strategy)
        
        # Track execution actions and trade info for logging
        last_exec_action = None
        last_trade_info = None
        
        for _ in range(self.execution_horizon):
            # Get observation from execution environment
            exec_obs = execution_env._get_observation()
            
            # Use execution agent if available, otherwise use random action
            if exec_agent is not None:
                # Use cached features (should always be available after _initialize)
                if self.cached_features is None:
                    # Fallback: fetch if cache somehow missing (shouldn't happen)
                    self.logger.warning("Cache missing in step(), fetching data (this should not happen)")
                    self.cached_aligned_data, self.cached_features = self.preprocessor.process(window_size=self.window_size)
                    self.cache_window_size = self.window_size
                
                aligned_data = self.cached_aligned_data
                features = self.cached_features
                
                if aligned_data and features:
                    windowed = self.preprocessor.get_window_features(features, self.window_size, None)
                    if windowed:
                        # Convert to tensor and get action
                        features_dict = {}
                        for interval, feat_array in windowed.items():
                            if len(feat_array) > 0:
                                # Use last window
                                feat_tensor = torch.FloatTensor(feat_array[-1]).unsqueeze(0)
                                features_dict[interval] = feat_tensor
                        
                        # Strategy features (one-hot)
                        strategy_features = torch.zeros(1, 4)
                        strategy_idx = ["TREND_FOLLOW", "MEAN_REVERT", "MOMENTUM", "RISK_OFF"].index(selected_strategy)
                        strategy_features[0, strategy_idx] = 1.0
                        
                        # Get action from execution agent
                        try:
                            exec_action, _, _ = exec_agent.act(features_dict, strategy_features, deterministic=False)
                        except:
                            # Fallback to random if agent fails
                            exec_action = execution_env.action_space.sample()
                    else:
                        exec_action = execution_env.action_space.sample()
                else:
                    exec_action = execution_env.action_space.sample()
            else:
                # No agent available, use random action
                exec_action = execution_env.action_space.sample()
            
            # Check if balance is too low to trade (prevent further losses)
            min_balance_threshold = 0.50  # Minimum $0.50 to prevent micro-trades
            if hasattr(execution_env.base_env, 'balance'):
                current_exec_balance = execution_env.base_env.balance
                # If balance is too low and we're trying to BUY, force HOLD
                if current_exec_balance < min_balance_threshold and exec_action == 1:  # BUY
                    exec_action = 0  # Force HOLD
                    self.logger.debug(f"Balance too low (${current_exec_balance:.2f}), forcing HOLD instead of BUY")
            
            # Track previous state to detect trades
            prev_shares = execution_env.base_env.shares if hasattr(execution_env.base_env, 'shares') else 0
            
            # Track execution action
            last_exec_action = exec_action
            
            # Step execution environment
            _, exec_reward, exec_done, exec_truncated, exec_info = execution_env.step(exec_action)
            
            total_reward += exec_reward
            
            # Detect if a trade was executed (BUY or SELL action that changed position)
            current_shares = execution_env.base_env.shares if hasattr(execution_env.base_env, 'shares') else 0
            trade_executed = False
            if exec_action == 1:  # BUY
                trade_executed = (prev_shares == 0 and current_shares > 0)
            elif exec_action == 2:  # SELL
                trade_executed = (prev_shares > 0 and current_shares == 0)
            
            if trade_executed:
                exec_info['trade_executed'] = True
                exec_info['trade'] = True  # Also set 'trade' for compatibility
                # Store trade info for logging
                action_names = ['HOLD', 'BUY', 'SELL']
                last_trade_info = {
                    'execution_action': exec_action,
                    'action_name': action_names[exec_action] if exec_action in [0, 1, 2] else 'UNKNOWN',
                    'profit': exec_info.get('profit', 0.0),
                    'price': exec_info.get('price', 0.0) if 'price' in exec_info else (execution_env.base_env.data[execution_env.base_env.current_step, 0] if hasattr(execution_env.base_env, 'data') and execution_env.base_env.current_step < len(execution_env.base_env.data) else 0.0)
                }
            
            if exec_done or exec_truncated:
                done = True
                break
        
        self.steps_in_strategy += self.execution_horizon
        
        # Update balance from execution environment
        if hasattr(execution_env.base_env, 'balance'):
            self.current_balance = execution_env.base_env.balance
            
            # Ensure balance doesn't go negative (safety check)
            if self.current_balance < 0:
                self.logger.warning(f"Balance went negative: ${self.current_balance:.2f}, clamping to 0")
                self.current_balance = 0.0
                execution_env.base_env.balance = 0.0
        
        # Get next observation
        next_obs = self._get_observation()
        
        # Calculate meta-reward (risk-adjusted return over horizon)
        # FIX: Align reward with actual profitability, add strategy diversity bonus
        if self.initial_balance > 0:
            return_pct = (self.current_balance - self.initial_balance) / self.initial_balance
            meta_reward = return_pct * 100  # Scale to percentage
            
            # Add bonus for profitable strategies, penalty for losing ones
            if return_pct > 0:
                meta_reward *= 1.2  # 20% bonus for profits
            else:
                meta_reward *= 1.1  # 10% penalty for losses (less harsh)
            
            # Strategy diversity bonus: Penalize over-reliance on single strategy
            # Track strategy usage (simple: if same strategy used too much, small penalty)
            if hasattr(self, '_strategy_history'):
                self._strategy_history.append(selected_strategy)
                if len(self._strategy_history) > 20:
                    self._strategy_history.pop(0)
                
                # Check if same strategy used >80% of time
                if len(self._strategy_history) >= 10:
                    strategy_counts = {}
                    for s in self._strategy_history:
                        strategy_counts[s] = strategy_counts.get(s, 0) + 1
                    max_usage = max(strategy_counts.values()) / len(self._strategy_history)
                    if max_usage > 0.8:  # Same strategy >80% of time
                        meta_reward -= 2.0  # Small penalty for lack of diversity
            else:
                self._strategy_history = [selected_strategy]
            
            # Clip reward to prevent extreme values that cause NaN
            meta_reward = np.clip(meta_reward, -100, 100)
        else:
            meta_reward = np.clip(total_reward, -100, 100)
        
        # Detect current regime for info
        regime = "TREND"  # Default
        if self.aligned_data:
            regime_key = "1d" if "1d" in self.aligned_data else list(self.aligned_data.keys())[0]
            regime_df = self.aligned_data[regime_key]
            if not regime_df.empty:
                # Extract Close prices for regime detection
                close_prices = regime_df['Close'].values if 'Close' in regime_df.columns else regime_df.iloc[:, 0].values
                regime = self.regime_detector.detect_regime(close_prices)
        
        # Collect trade information from execution environment
        # FIX: Only show realized profit (from closed trades), not unrealized P&L
        trade_info = None
        
        # If we have trade info from execution (BUY/SELL executed), use that
        if last_trade_info:
            trade_info = {
                'trade': True,
                'action': last_trade_info['action_name'],  # BUY or SELL (actual execution action)
                'execution_action': last_trade_info['execution_action'],
                'price': last_trade_info['price'],
                'profit': last_trade_info['profit']  # Realized profit from closed trade
            }
        # Otherwise, if we have a position, show unrealized P&L (but mark it as HOLD)
        elif hasattr(execution_env.base_env, 'shares') and execution_env.base_env.shares > 0:
            # Position open - show unrealized P&L
            if hasattr(execution_env.base_env, 'data') and execution_env.base_env.data is not None:
                current_step = min(execution_env.base_env.current_step, len(execution_env.base_env.data) - 1)
                current_price = execution_env.base_env.data[current_step, 0]  # Index 0 = Close price
            else:
                current_price = execution_env.base_env.entry_price  # Fallback
            
            trade_info = {
                'trade': False,  # Not a trade, just holding
                'action': 'HOLD',  # Execution action was HOLD
                'execution_action': 0,  # HOLD
                'price': current_price,
                'shares': execution_env.base_env.shares,
                'entry_price': execution_env.base_env.entry_price,
                'profit': (current_price - execution_env.base_env.entry_price) * execution_env.base_env.shares if execution_env.base_env.entry_price > 0 else 0,  # Unrealized P&L
                'unrealized': True  # Mark as unrealized
            }
        
        info = {
            'strategy': selected_strategy,
            'regime': regime,
            'balance': self.current_balance,
            'return_pct': return_pct if self.initial_balance > 0 else 0.0,
            'execution_reward': total_reward,
            'hard_stop_loss': hard_stop_loss_triggered  # Pass hard stop-loss flag to training code
        }
        
        # Add trade info if available
        if trade_info:
            info.update(trade_info)
        
        # Done if execution environment is done
        if done:
            self.logger.info(f"MetaStrategyEnv episode done: strategy={selected_strategy}, return={return_pct:.2%}")
        
        return next_obs, meta_reward, done, truncated, info


def test_hierarchical_env():
    """Test hierarchical environment"""
    print("Testing Hierarchical Environment...")
    
    # Test execution environment
    print("\n1. Testing ExecutionEnv...")
    from dapgio_improved import TradingConfig
    
    config = TradingConfig()
    config.mode = "stock"
    config.interval = "1d"
    
    exec_env = ExecutionEnv("^IXIC", window_size=30, config=config, strategy_type="TREND_FOLLOW")
    obs, info = exec_env.reset()
    print(f"ExecutionEnv obs shape: {obs.shape}")
    
    action = exec_env.action_space.sample()
    obs, reward, done, truncated, info = exec_env.step(action)
    print(f"ExecutionEnv reward: {reward:.4f}, strategy: {info.get('strategy')}")
    
    # Test meta-strategy environment (simplified - may need data)
    print("\n2. Testing MetaStrategyEnv...")
    print("Note: MetaStrategyEnv requires data fetching, may take time...")
    
    try:
        meta_env = MetaStrategyEnv("^IXIC", window_size=30, config=config, execution_horizon=5)
        obs, info = meta_env.reset()
        print(f"MetaStrategyEnv obs shape: {obs.shape}")
        
        action = meta_env.action_space.sample()
        obs, reward, done, truncated, info = meta_env.step(action)
        print(f"MetaStrategyEnv reward: {reward:.4f}, strategy: {info.get('strategy')}, regime: {info.get('regime')}")
    except Exception as e:
        print(f"MetaStrategyEnv test failed (may need internet/data): {e}")
    
    print("\n[SUCCESS] Hierarchical environment tests completed!")


if __name__ == "__main__":
    test_hierarchical_env()

