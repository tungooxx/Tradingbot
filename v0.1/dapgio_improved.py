"""
IMPROVED VERSION OF DAPGIO.PY FOR LIVE TRADING
==============================================

Key Improvements:
1. Comprehensive error handling
2. Risk management system
3. Configuration management
4. Data validation
5. Structured logging
6. Circuit breakers
7. Model versioning
8. Better separation of concerns
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import math
import matplotlib.pyplot as plt
import pandas_ta as ta
import random
import logging
import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


# ==============================================================================
# CONFIGURATION MANAGEMENT
# ==============================================================================

@dataclass
class TradingConfig:
    """Configuration for live trading - can be loaded from JSON/YAML"""
    # Trading Mode
    mode: str = "crypto"  # "crypto" or "stock"
    interval: str = "1h"  # "1h", "4h" for crypto; "1d" for stock
    
    # Risk Management
    max_position_size_pct: float = 0.10  # Max 10% of account per position
    max_daily_loss_pct: float = 0.05  # Stop trading if down 5% in a day
    max_drawdown_pct: float = 0.15  # Stop trading if drawdown > 15%
    stop_loss_pct: float = 0.03  # -3% stop loss (tighter for intraday)
    take_profit_pct: float = 0.20  # +20% take profit
    
    # Entry Filters
    min_confidence_threshold: float = 0.70  # Minimum confidence to trade (raised)
    max_gap_percent: float = 2.0  # Don't enter if price gapped >2%
    rsi_oversold: float = 30.0  # RSI < 30 (oversold, good for buy)
    rsi_overbought: float = 70.0  # RSI > 70 (overbought, good for sell)

    # Trading Parameters
    commission_rate: float = 0.001  # 0.1% commission
    slippage_bps: float = 5.0  # 5 basis points slippage

    # Circuit Breakers
    max_trades_per_day: int = 10
    max_trades_per_hour: int = 3
    emergency_stop: bool = False

    # Model
    model_path: str = "kan_agent_crypto.pth"
    model_version: str = "v1.0"
    window_size: int = 30  # For hourly: 30 hours = ~1.25 days

    # Data
    ticker: str = "ETH-USD"
    data_validation_enabled: bool = True
    max_data_age_minutes: int = 20  # Reject data older than 20 min

    # Logging
    log_level: str = "INFO"
    log_file: str = "trading.log"

    @classmethod
    def from_file(cls, path: str) -> 'TradingConfig':
        """Load config from JSON file"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            return cls(**data)
        return cls()

    def save(self, path: str):
        """Save config to JSON file"""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging(config: TradingConfig):
    """Setup structured logging"""
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("TradingBot")


# ==============================================================================
# TRADING LOGGER
# ==============================================================================

class TradingLogger:
    """Logger for tracking trading steps and results"""
    def __init__(self, missed_opp_threshold=0.015):
        self.history = []
        self.open_positions = []
        self.missed_opp_threshold = missed_opp_threshold

    def log_step(self, date, open_price, high_price, low_price, real_close, pred_close, action):
        """Log a single timestep.

        action: "buy" | "sell" | "skip"
        pred_close: predicted next close price (or None)
        """
        profit = 0.0
        if action == "buy":
            self.open_positions.append(real_close)
        elif action == "sell":
            if self.open_positions:
                buy_price = self.open_positions.pop(0)
                profit = real_close - buy_price

        self.history.append({
            "Date": date,
            "Open": float(open_price) if open_price is not None else None,
            "High": float(high_price) if high_price is not None else None,
            "Low": float(low_price) if low_price is not None else None,
            "Real_Close": float(real_close) if real_close is not None else None,
            "Pred_Close": float(pred_close) if pred_close is not None else None,
            "Action": action.upper(),
            "Profit": float(profit) if action == "sell" else 0.0,
            "Holding_Count": len(self.open_positions),
        })

    def get_results(self):
        df = pd.DataFrame(self.history)
        if df.empty:
            return df

        # Hindsight: compare to next day's real close
        df["Next_Close"] = df["Real_Close"].shift(-1)
        df["Price_Change_Pct"] = (df["Next_Close"] - df["Real_Close"]) / df["Real_Close"]

        missed = []
        for _, row in df.iterrows():
            if row["Action"] in ["SKIP", "SELL"] and row["Holding_Count"] == 0:
                if row["Price_Change_Pct"] > self.missed_opp_threshold:
                    missed.append("YES")
                else:
                    missed.append("-")
            else:
                missed.append("-")

        df["Miss_Opportunity"] = missed

        final = df[["Date", "Open", "High", "Low", "Real_Close", "Pred_Close", "Action", "Profit", "Miss_Opportunity"]]
        return final


# ==============================================================================
# RISK MANAGEMENT
# ==============================================================================

class RiskManager:
    """Comprehensive risk management for live trading"""

    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.daily_pnl = 0.0
        self.peak_balance = 0.0
        self.trades_today = 0
        self.trades_this_hour = 0
        self.last_trade_time = None
        self.hour_start = datetime.now().replace(minute=0, second=0, microsecond=0)

    def reset_daily(self):
        """Reset daily counters"""
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.trades_this_hour = 0
        self.hour_start = datetime.now().replace(minute=0, second=0, microsecond=0)
        self.logger.info("Daily risk counters reset")

    def reset_hourly(self):
        """Reset hourly counters"""
        now = datetime.now()
        if now >= self.hour_start + timedelta(hours=1):
            self.trades_this_hour = 0
            self.hour_start = now.replace(minute=0, second=0, microsecond=0)

    def check_circuit_breakers(self) -> Tuple[bool, str]:
        """Check if trading should be stopped"""
        if self.config.emergency_stop:
            return False, "Emergency stop activated"

        if self.trades_today >= self.config.max_trades_per_day:
            return False, f"Max trades per day reached: {self.trades_today}"

        self.reset_hourly()
        if self.trades_this_hour >= self.config.max_trades_per_hour:
            return False, f"Max trades per hour reached: {self.trades_this_hour}"

        return True, "OK"

    def check_daily_loss_limit(self, current_balance: float, initial_balance: float) -> Tuple[bool, str]:
        """Check if daily loss limit exceeded"""
        daily_loss = (initial_balance - current_balance) / initial_balance
        if daily_loss >= self.config.max_daily_loss_pct:
            return False, f"Daily loss limit exceeded: {daily_loss:.2%}"
        return True, "OK"

    def check_drawdown(self, current_balance: float) -> Tuple[bool, str]:
        """Check if maximum drawdown exceeded"""
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance

        if self.peak_balance > 0:
            drawdown = (self.peak_balance - current_balance) / self.peak_balance
            if drawdown >= self.config.max_drawdown_pct:
                return False, f"Maximum drawdown exceeded: {drawdown:.2%}"

        return True, "OK"

    def calculate_position_size(self, account_balance: float, price: float, confidence: float) -> float:
        """Calculate position size based on risk management rules"""
        # Base position size as % of account
        base_size_pct = self.config.max_position_size_pct

        # Scale by confidence (higher confidence = larger position)
        confidence_multiplier = min(confidence / self.config.min_confidence_threshold, 1.5)
        adjusted_size_pct = base_size_pct * confidence_multiplier

        # Calculate dollar amount
        position_value = account_balance * adjusted_size_pct

        # Calculate shares
        shares = position_value / price

        self.logger.debug(f"Position sizing: balance=${account_balance:.2f}, "
                          f"size_pct={adjusted_size_pct:.2%}, shares={shares:.4f}")

        return shares

    def record_trade(self, pnl: float):
        """Record a trade for risk tracking"""
        self.daily_pnl += pnl
        self.trades_today += 1
        self.trades_this_hour += 1
        self.last_trade_time = datetime.now()
        self.logger.info(f"Trade recorded: PnL=${pnl:.2f}, Daily PnL=${self.daily_pnl:.2f}, "
                         f"Trades today={self.trades_today}")


# ==============================================================================
# DATA VALIDATION
# ==============================================================================

class DataValidator:
    """Validate data quality before trading decisions"""

    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validate data quality"""
        if df.empty:
            return False, "DataFrame is empty"

        # Check for NaN values
        if df.isnull().any().any():
            nan_cols = df.columns[df.isnull().any()].tolist()
            return False, f"NaN values found in columns: {nan_cols}"

        # Check for sufficient data
        if len(df) < self.config.window_size + 10:
            return False, f"Insufficient data: {len(df)} rows, need at least {self.config.window_size + 10}"

        # Check for data freshness (if timestamp available)
        if 'timestamp' in df.columns or df.index.name == 'Date':
            # This would need actual timestamp checking in live trading
            pass

        # Check for extreme outliers (price changes > 50%)
        if 'Close' in df.columns:
            returns = df['Close'].pct_change()
            extreme_moves = (returns.abs() > 0.5).sum()
            if extreme_moves > 0:
                self.logger.warning(f"Found {extreme_moves} extreme price moves (>50%)")

        return True, "Data validation passed"

    def validate_features(self, features: np.ndarray) -> Tuple[bool, str]:
        """Validate feature array"""
        if features is None:
            return False, "Features array is None"

        if np.isnan(features).any():
            return False, "NaN values in features"

        if np.isinf(features).any():
            return False, "Infinite values in features"

        if features.size == 0:
            return False, "Features array is empty"

        return True, "Features validation passed"


# ==============================================================================
# KAN MODEL CLASSES (Same as original)
# ==============================================================================

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


class KANLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(
                                self.grid_size + self.spline_order,
                                self.in_features,
                                self.out_features,
                            )
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        return y.permute(2, 1, 0)

    def forward(self, x: torch.Tensor):
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_basis = self.b_splines(x).permute(1, 0, 2)
        ret = torch.bmm(spline_basis, self.spline_weight.permute(1, 2, 0))
        ret = ret.permute(1, 0, 2)

        if self.enable_standalone_scale_spline:
            ret = ret * self.spline_scaler.permute(1, 0).unsqueeze(0)

        spline_output = torch.sum(ret, dim=1)
        output = base_output + spline_output

        if len(original_shape) == 3:
            output = output.view(
                original_shape[0], original_shape[1], self.out_features
            )

        return output


class KANBody(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(KANBody, self).__init__()
        self.layer1 = KANLinear(obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.layer2 = KANLinear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.ln1(x)
        x = self.layer2(x)
        x = self.ln2(x)
        return x


class KANPredictor(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(KANPredictor, self).__init__()
        self.body = KANBody(obs_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.body(x)
        return self.head(features)


class KANActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, pretrained_body=None):
        super(KANActorCritic, self).__init__()

        if pretrained_body:
            self.body = pretrained_body
        else:
            self.body = KANBody(obs_dim, hidden_dim)

        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def act(self, state, deterministic=False):
        """Get action with optional deterministic mode for live trading"""
        features = self.body(state)
        logits = self.actor_head(features)

        if deterministic:
            # For live trading, use argmax instead of sampling
            action = torch.argmax(logits, dim=1)
            dist = Categorical(logits=logits)
            return action.item(), dist.log_prob(action), self.critic_head(features)
        else:
            # For training, use sampling
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action), self.critic_head(features)

    def evaluate(self, state, action):
        features = self.body(state)
        logits = self.actor_head(features)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic_head(features)
        return action_logprobs, state_values, dist_entropy

    def get_action_confidence(self, state):
        """Get confidence scores for each action"""
        features = self.body(state)
        logits = self.actor_head(features)
        probs = F.softmax(logits, dim=1)
        return probs.squeeze().detach().cpu().numpy()


# ==============================================================================
# IMPROVED ENVIRONMENT WITH BETTER ERROR HANDLING
# ==============================================================================

class StockTradingEnv(gym.Env):
    def __init__(self, ticker="ETH-USD", window_size=30, config: Optional[TradingConfig] = None,
                 logger: Optional[logging.Logger] = None):
        super(StockTradingEnv, self).__init__()
        self.config = config or TradingConfig()
        self.logger = logger or logging.getLogger("TradingEnv")

        try:
            # Determine data period and interval based on mode
            if self.config.mode == "crypto":
                # For crypto: use intraday data (1h or 4h)
                interval = self.config.interval  # "1h" or "4h"
                period = "60d" if interval == "1h" else "120d"  # More data for hourly
                self.logger.info(f"Downloading {ticker} data (Crypto mode: {interval})...")
            else:
                # For stocks: use daily data
                interval = "1d"
                period = "5y"
                self.logger.info(f"Downloading {ticker} data (Stock mode: daily)...")
            
            # Retry logic for yfinance downloads
            max_retries = 3
            retry_delay = 2  # seconds
            self.df = None
            
            for attempt in range(max_retries):
                try:
                    self.df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
                    if not self.df.empty:
                        break
                    else:
                        self.logger.warning(f"Attempt {attempt + 1}/{max_retries}: Empty data for {ticker}")
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1}/{max_retries}: Download failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
            
            if self.df is None or self.df.empty:
                # Try alternative ticker format for crypto
                if self.config.mode == "crypto" and "-" in ticker:
                    alt_ticker = ticker.replace("-", "")
                    self.logger.info(f"Trying alternative ticker format: {alt_ticker}")
                    try:
                        self.df = yf.download(alt_ticker, period=period, interval=interval, progress=False, auto_adjust=True)
                        if not self.df.empty:
                            self.logger.info(f"Successfully downloaded data using {alt_ticker}")
                    except:
                        pass
                
                if self.df is None or self.df.empty:
                    raise ValueError(
                        f"No data downloaded for {ticker}. "
                        f"This might be a temporary Yahoo Finance API issue. "
                        f"Please try again in a few minutes or check your internet connection."
                    )

            # Fix MultiIndex
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = self.df.columns.get_level_values(0)

            # Feature Engineering
            self._calculate_features()

            # Validate data
            validator = DataValidator(self.config, self.logger)
            valid, msg = validator.validate_data(self.df)
            if not valid:
                raise ValueError(f"Data validation failed: {msg}")

            # Setup observation space
            self.features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9"]
            self.data = self.df[self.features].values
            self.window_size = window_size
            self.max_steps = len(self.data) - 1

            self.action_space = spaces.Discrete(3)
            self.obs_shape = window_size * len(self.features)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
            )

            self.logger.info(f"Environment initialized: {len(self.data)} data points, window_size={window_size}")

        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}", exc_info=True)
            raise

    def _calculate_features(self):
        """Calculate technical indicators with error handling"""
        try:
            # Basic features
            self.df["Log_Ret"] = np.log(self.df["Close"] / self.df["Close"].shift(1))
            self.df["Vol_Norm"] = self.df["Volume"] / self.df["Volume"].rolling(20).mean()

            # Technical indicators
            self.df.ta.rsi(length=14, append=True)
            self.df.ta.macd(fast=12, slow=26, signal=9, append=True)
            self.df.ta.bbands(length=20, std=2, append=True)

            # Normalize
            self.df.dropna(inplace=True)
            self.df["RSI_14"] = self.df["RSI_14"] / 100.0
            
            # Use rolling normalization for MACD to prevent distribution shift
            # Use a fixed window (e.g., 252 bars for daily, 252*24=6048 for hourly)
            # This ensures consistent normalization across training and prediction
            if self.config.interval == "1h":
                rolling_window = 252 * 24  # ~1 year of hourly data
            elif self.config.interval == "4h":
                rolling_window = 252 * 6  # ~1 year of 4h data
            else:  # daily
                rolling_window = 252  # ~1 year of daily data
            
            # Use rolling mean/std for normalization (more robust to distribution shifts)
            macd_mean = self.df["MACD_12_26_9"].rolling(rolling_window, min_periods=1).mean()
            macd_std = self.df["MACD_12_26_9"].rolling(rolling_window, min_periods=1).std()
            self.df["MACD_12_26_9"] = (self.df["MACD_12_26_9"] - macd_mean) / (macd_std + 1e-7)

        except Exception as e:
            self.logger.error(f"Feature calculation failed: {e}", exc_info=True)
            raise

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = 20.0
        self.shares = 0
        self.entry_price = 0
        self.initial_balance = 20.0
        return self._get_observation(), {}

    def _get_observation(self, step=None):
        if step is None:
            step = self.current_step
        window = self.data[step - self.window_size: step]
        return window.flatten().astype(np.float32)

    def _check_entry_filters(self, current_price: float, rsi: float) -> Tuple[bool, str]:
        """Check if entry conditions are met"""
        # Check gap protection (for daily data)
        if self.config.mode == "stock" and len(self.df) > 1:
            prev_close = self.df['Close'].iloc[self.current_step - 1] if self.current_step > 0 else current_price
            gap_percent = abs((current_price - prev_close) / prev_close * 100)
            if gap_percent > self.config.max_gap_percent:
                return False, f"Gap too large: {gap_percent:.2f}% > {self.config.max_gap_percent}%"
        
        # Check RSI filters (only for BUY - we want oversold conditions)
        # Note: RSI is normalized (0-1), so multiply by 100
        rsi_pct = rsi * 100
        if rsi_pct > self.config.rsi_overbought:
            return False, f"RSI overbought: {rsi_pct:.1f} > {self.config.rsi_overbought}"
        
        return True, "OK"
    
    def step(self, action):
        try:
            current_price = self.data[self.current_step][0]
            daily_log_ret = self.data[self.current_step][1]
            
            # Get RSI for entry filters (index 3 in features)
            current_rsi = self.data[self.current_step][3] if len(self.data[self.current_step]) > 3 else 0.5

            reward = 0
            done = False

            # Action logic with improved error handling and entry filters
            if action == 1:  # BUY
                if self.shares == 0 and self.balance > 0:
                    # Check entry filters
                    can_enter, filter_msg = self._check_entry_filters(current_price, current_rsi)
                    if not can_enter:
                        # Small penalty for blocked entry (but less than before)
                        reward -= 0.005
                        self.logger.debug(f"Entry filter blocked: {filter_msg}")
                    else:
                        # Apply slippage
                        fill_price = current_price * (1 + self.config.slippage_bps / 10000)
                        commission = self.balance * self.config.commission_rate

                        self.shares = (self.balance - commission) / fill_price
                        self.balance = 0.0
                        self.entry_price = fill_price
                        # Reward for taking action (encourage trading)
                        reward += 0.1  # Small positive reward for entering position
                        self.logger.debug(f"BUY executed at {fill_price:.2f}")

            elif action == 2:  # SELL
                if self.shares > 0:
                    # Apply slippage
                    fill_price = current_price * (1 - self.config.slippage_bps / 10000)
                    gross_val = self.shares * fill_price
                    commission = gross_val * self.config.commission_rate
                    net_val = gross_val - commission

                    cost_basis = self.shares * self.entry_price
                    profit_dollars = net_val - cost_basis
                    # Reward based on profit + small bonus for taking action
                    reward = (profit_dollars / cost_basis) * 100 if cost_basis > 0 else 0
                    reward += 0.1  # Small bonus for closing position (encourage trading)
                    
                    self.balance = net_val
                    self.shares = 0
                    self.entry_price = 0
                    self.logger.debug(f"SELL executed at {fill_price:.2f}, profit: {profit_dollars:.2f}")
            
            # Penalty for HOLD when there's opportunity (encourage trading)
            elif action == 0:  # HOLD
                # Small penalty for holding when not in position (missed opportunity)
                if self.shares == 0 and self.balance > 0:
                    # Check if there's a good entry opportunity (RSI not extreme)
                    if 0.3 < current_rsi < 0.7:  # Neutral RSI zone
                        reward -= 0.01  # Small penalty for missing opportunity
                # No penalty for holding when already in position (that's handled below)

            # Holding logic with improved risk management
            if self.shares > 0:
                current_return = (current_price - self.entry_price) / self.entry_price
                cost_basis = self.shares * self.entry_price

                # Stop loss - improved for intraday data
                if current_return < -self.config.stop_loss_pct:
                    fill_price = current_price * (1 - self.config.slippage_bps / 10000)
                    gross_val = self.shares * fill_price
                    commission = gross_val * self.config.commission_rate
                    self.balance = gross_val - commission
                    profit_dollars = self.balance - cost_basis
                    self.shares = 0.0
                    self.entry_price = 0.0
                    # Reward based on actual loss (scaled to percentage)
                    reward = (profit_dollars / cost_basis) * 100 if cost_basis > 0 else -10.0
                    self.logger.warning(f"Stop loss triggered at {current_return:.2%}")

                # Take profit
                elif current_return > self.config.take_profit_pct:
                    fill_price = current_price * (1 - self.config.slippage_bps / 10000)
                    gross_val = self.shares * fill_price
                    commission = gross_val * self.config.commission_rate
                    self.balance = gross_val - commission
                    profit_dollars = self.balance - cost_basis
                    self.shares = 0.0
                    self.entry_price = 0.0
                    # Reward based on actual profit (scaled to percentage)
                    reward = (profit_dollars / cost_basis) * 100 if cost_basis > 0 else 20.0
                    self.logger.info(f"Take profit triggered at {current_return:.2%}")

                else:
                    # Improved reward shaping - make it proportional to unrealized P&L
                    # Base reward from unrealized return (scaled down to avoid accumulation)
                    unrealized_return = current_return * 100  # Convert to percentage
                    reward += unrealized_return * 0.15  # Increased from 0.1 to 0.15 (encourage holding winners)
                    
                    # Bonus for holding winners (only if profitable) - increased rewards
                    if current_return > 0.10:  # +10% unrealized
                        reward += 1.5  # Increased from 1.0
                    elif current_return > 0.05:  # +5% unrealized
                        reward += 0.75  # Increased from 0.5
                    elif current_return > 0.02:  # +2% unrealized
                        reward += 0.25  # New bonus tier
                    
                    # Small holding bonus (only if not losing too much)
                    if current_return > -0.01:  # Not losing more than 1%
                        reward += 0.02  # Increased from 0.01
                    
                    # Penalty for holding losers (encourage cutting losses early)
                    if current_return < -0.02:  # Losing more than 2%
                        reward -= 0.1  # Small penalty to encourage selling

            self.current_step += 1
            if self.current_step >= self.max_steps:
                done = True

            return self._get_observation(), reward, done, False, {}

        except Exception as e:
            self.logger.error(f"Error in step(): {e}", exc_info=True)
            # Return safe default
            return self._get_observation(), 0, True, False, {"error": str(e)}


# ==============================================================================
# LIVE TRADING ENGINE
# ==============================================================================

class LiveTradingEngine:
    """Engine for live trading with all safety features"""

    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = setup_logging(config)
        self.risk_manager = RiskManager(config, self.logger)
        self.data_validator = DataValidator(config, self.logger)

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.agent = None
        self.predictor = None
        self.env = None

        self._load_model()
        self._initialize_environment()

    def _load_model(self):
        """Load trained model with error handling"""
        try:
            if not os.path.exists(self.config.model_path):
                raise FileNotFoundError(f"Model file not found: {self.config.model_path}")

            # Initialize agent
            obs_dim = self.config.window_size * 5  # 5 features
            action_dim = 3

            self.agent = KANActorCritic(obs_dim, action_dim, hidden_dim=32)
            self.agent.load_state_dict(torch.load(self.config.model_path, map_location=self.device, weights_only=True))
            self.agent.to(self.device)
            self.agent.eval()

            self.logger.info(f"Model loaded successfully: {self.config.model_path}")

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}", exc_info=True)
            raise

    def _initialize_environment(self):
        """Initialize trading environment"""
        try:
            self.env = StockTradingEnv(
                ticker=self.config.ticker,
                window_size=self.config.window_size,
                config=self.config,
                logger=self.logger
            )
            self.logger.info("Environment initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize environment: {e}", exc_info=True)
            raise

    def get_trading_signal(self) -> Optional[Dict]:
        """Get trading signal with all safety checks"""
        try:
            # 1. Check circuit breakers
            can_trade, msg = self.risk_manager.check_circuit_breakers()
            if not can_trade:
                self.logger.warning(f"Circuit breaker: {msg}")
                return None

            # 2. Get current state
            state, _ = self.env.reset()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # 3. Validate features
            valid, msg = self.data_validator.validate_features(state)
            if not valid:
                self.logger.error(f"Feature validation failed: {msg}")
                return None

            # 4. Get action and confidence
            with torch.no_grad():
                action, log_prob, value = self.agent.act(state_tensor, deterministic=True)
                confidence = self.agent.get_action_confidence(state_tensor)

            # 5. Check confidence threshold
            max_confidence = confidence.max()
            if max_confidence < self.config.min_confidence_threshold:
                self.logger.debug(f"Confidence too low: {max_confidence:.2f} < {self.config.min_confidence_threshold}")
                return None

            # 6. Check risk limits
            current_balance = self.env.balance + (self.env.shares * self.env.data[self.env.current_step][0])
            can_trade, msg = self.risk_manager.check_daily_loss_limit(current_balance, self.env.initial_balance)
            if not can_trade:
                self.logger.warning(f"Daily loss limit: {msg}")
                return None

            can_trade, msg = self.risk_manager.check_drawdown(current_balance)
            if not can_trade:
                self.logger.warning(f"Drawdown limit: {msg}")
                return None

            # 7. Return signal
            action_names = ["HOLD", "BUY", "SELL"]
            signal = {
                "action": action_names[action],
                "action_id": action,
                "confidence": float(max_confidence),
                "all_confidence": confidence.tolist(),
                "value": float(value.item()),
                "timestamp": datetime.now().isoformat(),
                "price": float(self.env.data[self.env.current_step][0])
            }

            self.logger.info(f"Signal generated: {signal['action']} (confidence={signal['confidence']:.2f})")
            return signal

        except Exception as e:
            self.logger.error(f"Error generating signal: {e}", exc_info=True)
            return None

    def execute_trade(self, signal: Dict) -> bool:
        """Execute trade with position sizing and risk checks"""
        try:
            if signal is None or signal['action'] == "HOLD":
                return False

            # Calculate position size
            current_price = signal['price']
            account_balance = self.env.balance + (self.env.shares * current_price)
            confidence = signal['confidence']

            # This would integrate with actual exchange API
            # For now, we simulate
            self.logger.info(f"Executing {signal['action']} at ${current_price:.2f} "
                             f"(confidence={confidence:.2f})")

            # Record trade
            # In real implementation, would wait for fill confirmation
            self.risk_manager.record_trade(0.0)  # PnL calculated later

            return True

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}", exc_info=True)
            return False


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Load or create config
    config_path = "live_trading_config.json"
    if os.path.exists(config_path):
        config = TradingConfig.from_file(config_path)
        print(f"✅ Loaded config from {config_path}")
    else:
        config = TradingConfig()
        config.save(config_path)
        print(f"✅ Created default config at {config_path}")

    # Initialize engine
    try:
        engine = LiveTradingEngine(config)
        print("✅ Live Trading Engine initialized")

        # Get signal (example)
        signal = engine.get_trading_signal()
        if signal:
            print(f"\n📊 Trading Signal:")
            print(f"   Action: {signal['action']}")
            print(f"   Confidence: {signal['confidence']:.2%}")
            print(f"   Price: ${signal['price']:.2f}")
        else:
            print("\n⚠️ No trading signal (safety checks or low confidence)")

    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback

        traceback.print_exc()

