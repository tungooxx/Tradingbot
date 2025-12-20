"""
TRADING CONSTRAINTS ENVIRONMENT WRAPPER
========================================
Wraps the base trading environment with real trading constraints:
- Max positions limit
- Confidence thresholds
- Position sizing rules
- Real trading logic

This ensures the model learns to work within actual trading constraints.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Dict, Tuple
import logging

from dapgio_improved import StockTradingEnv, TradingConfig


class TradingConstraintsEnv(gym.Wrapper):
    """
    Wrapper that adds real trading constraints to the base environment.
    This makes training match the actual trading conditions.
    """
    
    def __init__(
        self,
        base_env: StockTradingEnv,
        min_confidence: float = 0.50,
        max_positions: int = 3,
        position_size_pct: float = 0.10,
        logger: Optional[logging.Logger] = None
    ):
        """
        Wrap base environment with trading constraints.
        
        Args:
            base_env: Base StockTradingEnv to wrap
            min_confidence: Minimum confidence to execute trades
            max_positions: Maximum concurrent positions
            position_size_pct: Position size as % of account (0.0-1.0)
            logger: Logger instance
        """
        super(TradingConstraintsEnv, self).__init__(base_env)
        self.min_confidence = min_confidence
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.logger = logger or logging.getLogger("TradingConstraintsEnv")
        
        # Expose base environment attributes (needed for training)
        # These are set after base_env is fully initialized
        # We'll use property accessors to ensure they're always available
        
        # Track open positions
        self.open_positions = []  # List of entry prices/timestamps
        
        # Track model confidence for each action
        self.last_action_confidence = 0.0
        
        # Temporary storage for balance restoration
        self._temp_balance_restore = None
        
        self.logger.info(f"Trading constraints enabled:")
        self.logger.info(f"  Min confidence: {min_confidence:.0%}")
        self.logger.info(f"  Max positions: {max_positions}")
        self.logger.info(f"  Position size: {position_size_pct:.0%}")
    
    @property
    def obs_shape(self):
        """Get observation shape from base environment"""
        return self.env.obs_shape
    
    @property
    def observation_space(self):
        """Get observation space from base environment"""
        return self.env.observation_space
    
    @property
    def action_space(self):
        """Get action space from base environment"""
        return self.env.action_space
    
    @property
    def data(self):
        """Get data from base environment"""
        return self.env.data
    
    @property
    def df(self):
        """Get dataframe from base environment"""
        return self.env.df
    
    @property
    def window_size(self):
        """Get window size from base environment"""
        return self.env.window_size
    
    @property
    def current_step(self):
        """Get current step from base environment"""
        return self.env.current_step
    
    @property
    def max_steps(self):
        """Get max steps from base environment"""
        return self.env.max_steps
    
    @property
    def balance(self):
        """Get balance from base environment"""
        return self.env.balance
    
    @property
    def shares(self):
        """Get shares from base environment"""
        return self.env.shares
    
    @property
    def entry_price(self):
        """Get entry price from base environment"""
        return self.env.entry_price
    
    @property
    def initial_balance(self):
        """Get initial balance from base environment"""
        return self.env.initial_balance
    
    def get_action_confidence(self, action_probs: np.ndarray) -> float:
        """Get confidence for the selected action"""
        return float(np.max(action_probs))
    
    def check_max_positions(self) -> bool:
        """Check if we can open a new position"""
        return len(self.open_positions) < self.max_positions
    
    def calculate_position_size(self, account_balance: float, price: float) -> float:
        """Calculate position size based on constraints"""
        position_value = account_balance * self.position_size_pct
        shares = position_value / price
        return shares
    
    def step(self, action, action_confidence: Optional[float] = None):
        """
        Step with trading constraints applied.
        
        Args:
            action: Action from agent (0=HOLD, 1=BUY, 2=SELL)
            action_confidence: Confidence score for the action (optional)
        """
        # Store confidence if provided
        if action_confidence is not None:
            self.last_action_confidence = action_confidence
        
        # Apply constraints before executing action
        constrained_action = action
        
        # BUY constraint: Check max positions and confidence
        if action == 1:  # BUY
            if not self.check_max_positions():
                # Max positions reached - force HOLD
                self.logger.debug(f"Max positions reached ({self.max_positions}), blocking BUY")
                constrained_action = 0  # Force HOLD
                # Small penalty for trying to buy when max positions reached
                obs, reward, done, truncated, info = self.env.step(0)
                reward -= 0.01  # Penalty for trying to exceed max positions
                info['action_blocked'] = 'max_positions'
                info['original_action'] = 'BUY'
                # Ensure reward_breakdown exists (base env should provide it, but ensure it's there)
                if 'reward_breakdown' not in info:
                    info['reward_breakdown'] = {
                        'action_type': 'HOLD',
                        'action_reward': reward,
                        'holding_reward': 0.0,
                        'stop_loss_reward': 0.0,
                        'take_profit_reward': 0.0
                    }
                info['executed_action'] = 'HOLD'
                return obs, reward, done, truncated, info
            
            if self.last_action_confidence < self.min_confidence:
                # Confidence too low - force HOLD
                self.logger.debug(f"Confidence too low ({self.last_action_confidence:.2%} < {self.min_confidence:.0%}), blocking BUY")
                constrained_action = 0  # Force HOLD
                # Small penalty for low confidence trade attempt
                obs, reward, done, truncated, info = self.env.step(0)
                reward -= 0.005  # Penalty for low confidence
                info['action_blocked'] = 'low_confidence'
                info['original_action'] = 'BUY'
                info['confidence'] = self.last_action_confidence
                # Ensure reward_breakdown exists
                if 'reward_breakdown' not in info:
                    info['reward_breakdown'] = {
                        'action_type': 'HOLD',
                        'action_reward': reward,
                        'holding_reward': 0.0,
                        'stop_loss_reward': 0.0,
                        'take_profit_reward': 0.0
                    }
                info['executed_action'] = 'HOLD'
                return obs, reward, done, truncated, info
            
            # Apply position sizing constraint: limit balance to position_size_pct
            if self.env.shares == 0 and self.env.balance > 0:
                # Store original balance to restore unused portion after
                original_balance = self.env.balance
                # Limit balance to position_size_pct
                max_position_value = self.env.balance * self.position_size_pct
                self.env.balance = max_position_value
                # Store for restoration after step
                self._temp_balance_restore = original_balance
        
        # SELL constraint: Only if we have a position
        if action == 2:  # SELL
            if self.env.shares == 0:
                # No position to sell - force HOLD
                self.logger.debug("No position to sell, blocking SELL")
                constrained_action = 0  # Force HOLD
                obs, reward, done, truncated, info = self.env.step(0)
                reward -= 0.001  # Small penalty
                info['action_blocked'] = 'no_position'
                info['original_action'] = 'SELL'
                # Ensure reward_breakdown exists
                if 'reward_breakdown' not in info:
                    info['reward_breakdown'] = {
                        'action_type': 'HOLD',
                        'action_reward': reward,
                        'holding_reward': 0.0,
                        'stop_loss_reward': 0.0,
                        'take_profit_reward': 0.0
                    }
                info['executed_action'] = 'HOLD'
                return obs, reward, done, truncated, info
        
        # Execute constrained action
        obs, reward, done, truncated, info = self.env.step(constrained_action)
        
        # Restore balance if we modified it for position sizing
        if action == 1 and hasattr(self, '_temp_balance_restore'):
            original_balance = self._temp_balance_restore
            # Calculate actual position cost (including commission)
            if self.env.shares > 0:
                # Base environment already deducted commission
                # Calculate what was actually spent
                position_cost = self.env.shares * self.env.entry_price
                commission = position_cost * 0.001  # 0.1% commission
                total_cost = position_cost + commission
                # Return unused balance (original - total_cost)
                self.env.balance = original_balance - total_cost
            else:
                # No position opened (blocked by filters), restore full balance
                self.env.balance = original_balance
            delattr(self, '_temp_balance_restore')
        
        # Track positions
        if constrained_action == 1 and self.env.shares > 0:
            # Position opened
            self.open_positions.append({
                'entry_price': self.env.entry_price,
                'shares': self.env.shares
            })
            info['position_opened'] = True
        
        if constrained_action == 2 and self.env.shares == 0:
            # Position closed
            if self.open_positions:
                self.open_positions.pop(0)
            info['position_closed'] = True
        
        # Add constraint info
        info['max_positions'] = self.max_positions
        info['current_positions'] = len(self.open_positions)
        info['min_confidence'] = self.min_confidence
        info['action_confidence'] = self.last_action_confidence
        
        return obs, reward, done, truncated, info
    
    def reset(self, seed=None):
        """Reset environment and constraints"""
        obs, info = self.env.reset(seed=seed)
        self.open_positions = []
        self.last_action_confidence = 0.0
        return obs, info


def create_constrained_env(
    ticker: str,
    window_size: int,
    config: Optional[TradingConfig] = None,
    min_confidence: float = 0.50,
    max_positions: int = 3,
    position_size_pct: float = 0.10,
    logger: Optional[logging.Logger] = None
) -> TradingConstraintsEnv:
    """
    Create a trading environment with real constraints.
    
    This environment matches the constraints used in alpaca_paper_trading.py,
    so the model learns to work within actual trading limits.
    """
    base_env = StockTradingEnv(ticker=ticker, window_size=window_size, config=config, logger=logger)
    constrained_env = TradingConstraintsEnv(
        base_env,
        min_confidence=min_confidence,
        max_positions=max_positions,
        position_size_pct=position_size_pct,
        logger=logger
    )
    return constrained_env

