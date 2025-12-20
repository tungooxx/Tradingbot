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
from typing import Optional
import json
import os
import time
from datetime import datetime, timedelta
import pytz
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
    stop_loss_pct: float = 0.05  # -5% stop loss (wider to avoid premature exits)
    take_profit_pct: float = 0.10  # +10% take profit (lower, more realistic target)
    
    # Entry Filters
    min_confidence_threshold: float = 0.70  # Minimum confidence to trade (raised)
    max_gap_percent: float = 2.0  # Don't enter if price gapped >2%
    rsi_oversold: float = 30.0  # RSI < 30 (oversold, good for buy)
    rsi_overbought: float = 70.0  # RSI > 70 (overbought, good for sell)

    # Trading Parameters
    commission_rate: float = 0.001  # 0.1% commission (standard)
    slippage_bps: float = 5.0  # 5 basis points slippage
    # CRITICAL FIX: Total transaction cost = commission + slippage
    # For realistic training, we need to account for both
    # Total cost per trade: ~0.15% (0.1% commission + 0.05% slippage)

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
    """Setup structured logging with separate file handler for repetitive warnings"""
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Main logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, config.log_level))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler for ALL logs (including repetitive warnings)
    file_handler = logging.FileHandler(config.log_file)
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Stream handler (terminal) - filter out repetitive warnings
    class RepetitiveWarningFilter(logging.Filter):
        """Filter to send repetitive warnings to file only, not terminal"""
        REPETITIVE_PATTERNS = [
            'WHIPSAW PENALTY',
            'STUPIDITY PENALTY',
            'ENTRY FEE',
            'QUICK WIN BONUS',
            'Take profit triggered',
            'Stop loss triggered',
        ]
        
        def filter(self, record):
            # Allow all important warnings (like ENTROPY DOMINATION) to terminal
            message = record.getMessage()
            for pattern in self.REPETITIVE_PATTERNS:
                if pattern in message:
                    # This is repetitive - don't show in terminal
                    return False
            # Important logs - show in terminal
            return True
    
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, config.log_level))
    stream_handler.setFormatter(formatter)
    stream_handler.addFilter(RepetitiveWarningFilter())  # Filter repetitive warnings
    logger.addHandler(stream_handler)
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
        action_lower = action.lower()
        
        if action_lower == "buy":
            self.open_positions.append(real_close)
        elif action_lower == "sell":
            if self.open_positions:
                buy_price = self.open_positions.pop(0)
                profit = real_close - buy_price
            else:
                # WARNING: Selling without a matching buy - this is a bug!
                # This can happen if logger wasn't properly reset between episodes
                import logging
                logging.warning(f"TradingLogger: SELL action without matching BUY! "
                              f"This may indicate logger wasn't reset properly. "
                              f"Profit set to 0.0 for this invalid trade.")
                profit = 0.0

        self.history.append({
            "Date": date,
            "Open": float(open_price) if open_price is not None else None,
            "High": float(high_price) if high_price is not None else None,
            "Low": float(low_price) if low_price is not None else None,
            "Real_Close": float(real_close) if real_close is not None else None,
            "Pred_Close": float(pred_close) if pred_close is not None else None,
            "Action": action.upper(),
            "Profit": float(profit) if action_lower == "sell" else 0.0,
            "Holding_Count": len(self.open_positions),
        })
    
    def reset(self):
        """Explicitly reset the logger (clear history and open positions)"""
        self.history = []
        self.open_positions = []
    
    def verify_clean(self):
        """Verify logger is clean (no leftover data)"""
        return len(self.history) == 0 and len(self.open_positions) == 0

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


class TemporalAttention(nn.Module):
    """Multi-head self-attention for temporal patterns"""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super(TemporalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim) or (batch, hidden_dim)
        original_shape = x.shape
        if len(x.shape) == 2:
            # Add sequence dimension for attention
            x = x.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        batch_size, seq_len, _ = x.shape
        
        # Multi-head attention
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection and residual
        output = self.output_proj(attn_output)
        output = self.norm(output + x)  # Residual connection
        
        # Remove sequence dimension if it was added
        if len(original_shape) == 2:
            output = output.squeeze(1)
        
        return output


class EnhancedKANBody(nn.Module):
    """Enhanced KAN body with deeper architecture, attention, and residual connections"""
    def __init__(self, obs_dim, hidden_dim=128, num_layers=3, use_attention=True, dropout=0.1):
        super(EnhancedKANBody, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        
        # Input projection
        self.input_proj = KANLinear(obs_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        
        # Hidden layers with residual connections
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(KANLinear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention = TemporalAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)
        
        # Hidden layers with residual connections
        for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.norms, self.dropouts)):
            residual = x
            x = layer(x)
            x = norm(x)
            x = dropout(x)
            x = x + residual  # Residual connection
        
        # Attention (if enabled)
        if self.use_attention:
            x = self.attention(x)
        
        # Output normalization
        x = self.output_norm(x)
        x = self.output_dropout(x)
        
        return x


class KANBody(nn.Module):
    """Original KAN body (kept for backward compatibility)"""
    def __init__(self, obs_dim, hidden_dim=64, use_enhanced=False, **kwargs):
        super(KANBody, self).__init__()
        if use_enhanced:
            # Use enhanced architecture
            self.body = EnhancedKANBody(obs_dim, hidden_dim, **kwargs)
        else:
            # Original simple architecture
            self.layer1 = KANLinear(obs_dim, hidden_dim)
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.layer2 = KANLinear(hidden_dim, hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
            self.body = None

    def forward(self, x):
        if self.body is not None:
            return self.body(x)
        else:
            x = self.layer1(x)
            x = self.ln1(x)
            x = self.layer2(x)
            x = self.ln2(x)
            return x


class KANPredictor(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64, use_enhanced=False):
        super(KANPredictor, self).__init__()
        if use_enhanced:
            self.body = EnhancedKANBody(obs_dim, hidden_dim, num_layers=3, use_attention=True, dropout=0.1)
        else:
            self.body = KANBody(obs_dim, hidden_dim)
        # Enhanced head with dropout
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        features = self.body(x)
        return self.head(features)


class KANActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, pretrained_body=None, 
                 use_enhanced=False, confidence_temperature=1.0):
        super(KANActorCritic, self).__init__()
        self.confidence_temperature = confidence_temperature  # For confidence calibration

        if pretrained_body:
            self.body = pretrained_body
        else:
            if use_enhanced:
                # Enhanced architecture with attention and deeper layers
                self.body = EnhancedKANBody(
                    obs_dim, 
                    hidden_dim=hidden_dim,
                    num_layers=3,  # Deeper network
                    use_attention=True,
                    dropout=0.1
                )
            else:
                self.body = KANBody(obs_dim, hidden_dim)

        # Enhanced actor head with dropout
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Enhanced critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def act(self, state, deterministic=False):
        """Get action with optional deterministic mode for live trading"""
        features = self.body(state)
        logits = self.actor_head(features)
        
        # Check for NaN or Inf values in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # Replace NaN/Inf with zeros and add small random noise to break ties
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                                torch.zeros_like(logits), logits)
            # If all logits are zero, add small random values
            if torch.all(logits == 0):
                logits = torch.randn_like(logits) * 0.01
            # Clamp to reasonable range to prevent future issues
            logits = torch.clamp(logits, min=-10.0, max=10.0)

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
        
        # Check for NaN or Inf values in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # Replace NaN/Inf with zeros and add small random noise to break ties
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                                torch.zeros_like(logits), logits)
            # If all logits are zero, add small random values
            if torch.all(logits == 0):
                logits = torch.randn_like(logits) * 0.01
            # Clamp to reasonable range to prevent future issues
            logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic_head(features)
        return action_logprobs, state_values, dist_entropy

    def get_action_confidence(self, state):
        """Get confidence scores for each action with temperature scaling"""
        features = self.body(state)
        # Handle both Sequential and Linear actor heads
        if isinstance(self.actor_head, nn.Sequential):
            logits = self.actor_head(features)
        else:
            logits = self.actor_head(features)
        
        # Check for NaN or Inf values in logits
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # Replace NaN/Inf with zeros and add small random noise to break ties
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                                torch.zeros_like(logits), logits)
            # If all logits are zero, add small random values
            if torch.all(logits == 0):
                logits = torch.randn_like(logits) * 0.01
            # Clamp to reasonable range to prevent future issues
            logits = torch.clamp(logits, min=-10.0, max=10.0)
        
        # Temperature scaling for better confidence calibration
        # Lower temperature (<1.0) makes the distribution more peaked (higher confidence)
        scaled_logits = logits / self.confidence_temperature
        probs = F.softmax(scaled_logits, dim=1)
        
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
                period = "5y"  # Get 5 years to ensure enough data after cleaning
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

            # MARKET HOURS FILTERING: For stocks, only keep regular trading hours (9:30 AM - 4:00 PM ET)
            # This removes extended hours data which is low-quality and confuses the agent
            # NOTE: Skip filter for daily data (1d interval) as it has no time component
            if self.config.mode == "stock" and isinstance(self.df.index, pd.DatetimeIndex):
                # Check if this is daily data (no time component in index)
                has_time_component = any(not (idx.hour == 0 and idx.minute == 0 and idx.second == 0) 
                                        for idx in self.df.index[:10] if len(self.df) > 0)
                
                # Only apply filter if data has time component (hourly/intraday data)
                if has_time_component:
                    original_len = len(self.df)
                    
                    # Convert to US/Eastern timezone if needed
                    if self.df.index.tz is None:
                        # Assume data is in ET if no timezone
                        self.df.index = self.df.index.tz_localize('US/Eastern')
                    elif self.df.index.tz != pytz.timezone('US/Eastern'):
                        # Convert to ET
                        self.df.index = self.df.index.tz_convert('US/Eastern')
                    
                    # Filter to regular trading hours: 9:30 AM - 4:00 PM ET
                    self.df = self.df.between_time('09:30', '16:00')
                    
                    # Filter out weekends (keep only Monday-Friday)
                    self.df = self.df[self.df.index.weekday < 5]
                    
                    filtered_len = len(self.df)
                    if original_len > 0:
                        pct_kept = (filtered_len / original_len) * 100
                        self.logger.info(f"Market hours filter: Kept {filtered_len}/{original_len} bars ({pct_kept:.1f}%) - Regular hours only (9:30 AM - 4:00 PM ET, weekdays)")
                    
                    if self.df.empty:
                        raise ValueError(f"No data remaining after market hours filter for {ticker}")
                else:
                    # Daily data - just filter out weekends
                    original_len = len(self.df)
                    self.df = self.df[self.df.index.weekday < 5]
                    filtered_len = len(self.df)
                    if original_len > 0:
                        pct_kept = (filtered_len / original_len) * 100
                        self.logger.info(f"Daily data detected: Kept {filtered_len}/{original_len} bars ({pct_kept:.1f}%) - Weekdays only (no time filter for daily bars)")

            # Feature Engineering
            self._calculate_features()

            # Validate data
            validator = DataValidator(self.config, self.logger)
            valid, msg = validator.validate_data(self.df)
            if not valid:
                raise ValueError(f"Data validation failed: {msg}")

            # Setup observation space
            # Base features (always included)
            base_features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9"]
            
            # Add market hours features for stocks only
            if self.config.mode == "stock":
                market_hours_features = ["hour_norm", "minutes_since_open_norm", "is_market_open"]
                self.features = base_features + market_hours_features
            else:
                # Crypto: no market hours (24/7)
                self.features = base_features
            
            self.data = self.df[self.features].values
            self.window_size = window_size
            self.max_steps = len(self.data) - 1

            self.action_space = spaces.Discrete(3)
            self.obs_shape = window_size * len(self.features)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
            )

            self.logger.info(f"Environment initialized: {len(self.data)} data points, window_size={window_size}")
            
            # CRITICAL: Store initial balance as class attribute (never changes)
            # This ensures reset() always restores to the correct starting balance
            self.initial_balance = 2000.0

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
            
            # Handle case where std is 0 or NaN (all values same in window)
            macd_std = macd_std.fillna(1e-7)  # Replace NaN std with small value
            macd_std = macd_std.replace(0.0, 1e-7)  # Replace zero std with small value
            
            self.df["MACD_12_26_9"] = (self.df["MACD_12_26_9"] - macd_mean) / macd_std
            
            # Fill any remaining NaN values (can occur at the beginning of the series)
            # Use forward fill then backward fill to handle edge cases
            self.df["MACD_12_26_9"].fillna(method='ffill', inplace=True)
            self.df["MACD_12_26_9"].fillna(method='bfill', inplace=True)
            # If still NaN (shouldn't happen), fill with 0
            self.df["MACD_12_26_9"].fillna(0.0, inplace=True)
            
            # Final dropna to remove any rows that still have NaN in other columns
            self.df.dropna(inplace=True)
            
            # MARKET HOURS AWARENESS: Add time-of-day features for stocks only
            if self.config.mode == "stock":
                self._add_market_hours_features()
            
            # Note: Data length validation is done by DataValidator after this method

        except Exception as e:
            self.logger.error(f"Feature calculation failed: {e}", exc_info=True)
            raise
    
    def _add_market_hours_features(self):
        """Add market hours features to DataFrame (stocks only)"""
        try:
            # Extract hour and minute from index (assumes datetime index)
            if not isinstance(self.df.index, pd.DatetimeIndex):
                # If not datetime, try to convert or skip
                self.logger.warning("DataFrame index is not DatetimeIndex, skipping market hours features")
                return
            
            # Extract time components
            self.df['hour'] = self.df.index.hour
            self.df['minute'] = self.df.index.minute
            self.df['weekday'] = self.df.index.weekday  # 0=Monday, 6=Sunday
            
            # Calculate minutes since market open (9:30 AM = 0)
            # Market opens at 9:30 AM (9*60 + 30 = 570 minutes from midnight)
            market_open_minutes = 9 * 60 + 30  # 570 minutes
            current_minutes = self.df['hour'] * 60 + self.df['minute']
            self.df['minutes_since_open'] = current_minutes - market_open_minutes
            # Clip to valid range (0 to 390 minutes = 6.5 hours)
            self.df['minutes_since_open'] = self.df['minutes_since_open'].clip(0, 390)
            
            # Normalize features
            self.df['hour_norm'] = self.df['hour'] / 23.0  # Normalize hour (0-23)
            self.df['minutes_since_open_norm'] = self.df['minutes_since_open'] / 390.0  # Normalize (0-390 minutes)
            
            # Is market open? (9:30 AM to 4:00 PM, weekdays only)
            # Market hours: 9:30 AM (9:30) to 4:00 PM (16:00)
            is_after_open = (self.df['hour'] > 9) | ((self.df['hour'] == 9) & (self.df['minute'] >= 30))
            is_before_close = self.df['hour'] < 16
            is_weekday = self.df['weekday'] < 5  # Monday-Friday
            
            self.df['is_market_open'] = (is_after_open & is_before_close & is_weekday).astype(float)
            
            # Drop intermediate columns (keep only normalized features)
            self.df.drop(['hour', 'minute', 'weekday', 'minutes_since_open'], axis=1, inplace=True, errors='ignore')
            
            self.logger.debug("Added market hours features for stock mode")
            
        except Exception as e:
            self.logger.warning(f"Failed to add market hours features: {e}, continuing without them")
            # Continue without market hours features if there's an error

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        # CRITICAL FIX: Explicitly force balance back to initial_balance
        # This ensures every episode starts with fresh money, not leftover from previous episode
        # RL requires independent episodes - starting "in the hole" breaks the math
        # Ensure initial_balance is set (should already be set in __init__, but double-check)
        if not hasattr(self, 'initial_balance') or self.initial_balance is None:
            self.initial_balance = 2000.0
        
        # Force balance to initial_balance (explicit reset, no matter what it was before)
        old_balance = getattr(self, 'balance', None)
        self.balance = self.initial_balance
        
        # Safety check: Log if balance was incorrect before reset
        if old_balance is not None and abs(old_balance - self.initial_balance) > 0.01:
            self.logger.warning(f"reset(): Balance was ${old_balance:.2f}, resetting to ${self.initial_balance:.2f}")
        
        self.shares = 0
        self.entry_price = 0
        # Reset consecutive holds counter
        # Reset trade tracking
        self._last_trade_executed = False
        self._last_trade_profit = 0.0
        self._consecutive_holds = 0
        # IMPROVED: Track portfolio value for log return calculation
        self._prev_portfolio_value = self.initial_balance
        self._return_history = []  # Rolling window for volatility calculation
        self._volatility_window = 20  # Window size for volatility penalty
        self._volatility_lambda = 0.1  # Volatility penalty factor
        # CRITICAL FIX: Track exit reason for penalty application
        self._stop_loss_triggered = False
        self._take_profit_triggered = False
        self._exit_reason = None  # "STOP_LOSS", "TAKE_PROFIT", "SELL", or None
        # ANTI-CHURN: Track shares before action to detect entry/exits
        self._shares_before_action = 0
        self._entry_step = None  # Track when position was entered
        self._reward_log_counter = 0  # Initialize reward breakdown logging counter
        return self._get_observation(), {}

    def _get_local_features(self, step=None) -> np.ndarray:
        """
        Get 4 normalized 'Local Features' for Hybrid Input architecture.
        
        Returns:
            Local features (4,): [position_status, unrealized_pnl, time_in_trade, relative_price]
        """
        if step is None:
            step = self.current_step
        
        # 1. Position Status: 0 if no position, 1 if holding
        position_status = 1.0 if self.shares > 0 else 0.0
        
        # 2. Unrealized PnL: (Current price / Entry price - 1) scaled by 10
        if self.shares > 0 and self.entry_price > 0 and step < len(self.data):
            current_price = self.data[step, 0]  # Index 0 = Close price
            unrealized_pnl = ((current_price / self.entry_price) - 1.0) * 10.0
            # Normalize to [-1, 1] range
            unrealized_pnl = np.clip(unrealized_pnl, -1.0, 1.0)
        else:
            unrealized_pnl = 0.0
        
        # 3. Time in Trade: Number of steps since last buy, normalized by window size
        if hasattr(self, '_entry_step') and self._entry_step is not None:
            time_in_trade = (step - self._entry_step) / float(self.window_size)
            # Normalize to [0, 1] range
            time_in_trade = np.clip(time_in_trade, 0.0, 1.0)
        else:
            time_in_trade = 0.0
        
        # 4. Relative Price: Current close relative to 20-period Moving Average
        if step >= 20 and step < len(self.data):
            # Calculate 20-period MA
            ma_window = self.data[max(0, step - 19):step + 1, 0]  # Last 20 closes
            ma_20 = np.mean(ma_window)
            current_price = self.data[step, 0]
            if ma_20 > 0:
                relative_price = (current_price / ma_20) - 1.0
                # Normalize to [-1, 1] range (scale by 5 to make it more sensitive)
                relative_price = np.clip(relative_price * 5.0, -1.0, 1.0)
            else:
                relative_price = 0.0
        else:
            relative_price = 0.0
        
        return np.array([position_status, unrealized_pnl, time_in_trade, relative_price], dtype=np.float32)
    
    def _get_observation(self, step=None):
        if step is None:
            step = self.current_step
        window = self.data[step - self.window_size: step]
        return window.flatten().astype(np.float32)
    
    def _is_market_open_at_step(self, step: int) -> bool:
        """
        Check if market is open at given step (stocks only).
        
        Args:
            step: Current step index
            
        Returns:
            True if market is open, False otherwise
        """
        if self.config.mode == "crypto":
            return True  # Crypto markets are always open
        
        # For stocks, check market hours features if available
        if step < len(self.data):
            # Check if we have market hours features (stocks have 8 features, crypto has 5)
            if len(self.features) > 5:
                # Market hours features are at indices 5, 6, 7 (hour_norm, minutes_since_open_norm, is_market_open)
                is_market_open_idx = len(self.features) - 1  # Last feature is is_market_open
                if step < len(self.data) and is_market_open_idx < self.data.shape[1]:
                    is_open = self.data[step, is_market_open_idx] > 0.5  # > 0.5 means open
                    return bool(is_open)
        
        # Fallback: Check DataFrame index if available
        if hasattr(self, 'df') and isinstance(self.df.index, pd.DatetimeIndex):
            if step < len(self.df):
                current_time = self.df.index[step]
                hour = current_time.hour
                minute = current_time.minute
                weekday = current_time.weekday()
                
                # Market hours: 9:30 AM to 4:00 PM, weekdays
                is_after_open = (hour > 9) or (hour == 9 and minute >= 30)
                is_before_close = hour < 16
                is_weekday = weekday < 5
                
                return is_after_open and is_before_close and is_weekday
        
        # Default: assume open if we can't determine
        return True
    
    def _get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value (cash + unrealized P&L)"""
        return self.balance + (self.shares * current_price)
    
    def _calculate_log_return(self, v_t: float, v_t_prev: float) -> float:
        """Calculate log return: log(V_t / V_{t-1})"""
        if v_t_prev <= 0:
            return 0.0
        return math.log(v_t / v_t_prev)
    
    def _calculate_transaction_cost(self, action: int, position_value: float) -> float:
        """
        Calculate transaction cost penalty.
        
        CRITICAL FIX: Penalty must be 2-3x actual fee to prevent scalping.
        With scaling * 100, a 0.1% fee = 0.1 in reward terms.
        We use 0.5 (5x) to force high-conviction trades only.
        """
        if action == 0:  # HOLD
            return 0.0
        # BUY or SELL incurs cost - HEAVY penalty to prevent over-trading
        # Real fee is commission_rate (e.g., 0.001 = 0.1%)
        # Penalty is 5x to force agent to only trade when move > 5x fee
        TRADING_PENALTY_MULTIPLIER = 5.0  # 5x actual fee
        return position_value * self.config.commission_rate * TRADING_PENALTY_MULTIPLIER
    
    def _calculate_volatility_penalty(self) -> float:
        """Calculate volatility penalty: lambda * StdDev(returns)"""
        if len(self._return_history) < 2:
            return 0.0
        
        # Use last 'window' returns
        recent_returns = self._return_history[-self._volatility_window:] if len(self._return_history) >= self._volatility_window else self._return_history
        
        if len(recent_returns) < 2:
            return 0.0
        
        std_dev = np.std(recent_returns)
        return self._volatility_lambda * std_dev

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

            # Initialize reward and done first
            reward = 0.0
            done = False
            transaction_cost = 0.0  # Will be calculated based on action
            market_closed_forced = False
            
            # MARKET HOURS VALIDATION: For stocks, check if market is open
            if self.config.mode == "stock":
                is_market_open = self._is_market_open_at_step(self.current_step)
                if not is_market_open and action != 0:  # Trying to trade when closed
                    # Force HOLD and apply small penalty
                    action = 0  # HOLD
                    reward = -0.1  # Small penalty for trying to trade when closed
                    market_closed_forced = True
                    self.logger.debug(f"Market closed at step {self.current_step}, forcing HOLD (penalty: -0.1)")

            # IMPROVED: Use previous portfolio value from last step (for log return)
            prev_portfolio_value = self._prev_portfolio_value
            
            # CRITICAL FIX: Track shares before action to detect stop loss
            self._shares_before_action = self.shares
            
            # ============================================================
            # HARD STOP-LOSS (-5%): Force SELL before any action logic
            # ============================================================
            # CRITICAL: This is a safety mechanism to prevent large losses
            # If unrealized PnL reaches -5%, immediately force SELL regardless of agent's action
            # This happens BEFORE action processing to ensure it always triggers
            # Only applies when market is open (can't sell when market is closed)
            hard_stop_loss_triggered = False  # Track if hard stop-loss was triggered
            if not market_closed_forced and self.shares > 0 and self.entry_price > 0:
                unrealized_pnl_pct = (current_price - self.entry_price) / self.entry_price
                if unrealized_pnl_pct < -0.05:  # -5% hard stop-loss
                    original_action = action
                    action = 2  # Force SELL
                    hard_stop_loss_triggered = True
                    self.logger.warning(f"HARD STOP-LOSS: Unrealized PnL = {unrealized_pnl_pct:.2%}, forcing SELL (agent wanted: {original_action})")
            
            # If action was forced to HOLD due to market hours, skip action logic
            if market_closed_forced:
                # Market closed - just update portfolio value and return
                current_portfolio_value = self._get_portfolio_value(current_price)
                log_return = self._calculate_log_return(current_portfolio_value, prev_portfolio_value)
                self._prev_portfolio_value = current_portfolio_value
                self.current_step += 1
                if self.current_step >= self.max_steps:
                    done = True
                return self._get_observation(), reward, done, False, {'market_closed': True}

            # Action logic with improved error handling and entry filters
            if action == 1:  # BUY
                # ACTION MASKING FIX: Instead of penalizing invalid actions, treat them as HOLD
                # The model should use action masking to prevent invalid actions in the first place
                # If an invalid action slips through, just treat it as HOLD (no penalty)
                if self.shares > 0:
                    # Invalid BUY action - already in position
                    # JUST TREAT AS HOLD - no penalty (action masking should prevent this)
                    self.logger.debug(f"INVALID BUY: Already in position - treating as HOLD (action masking should prevent this)")
                    action = 0  # Convert to HOLD
                    # Fall through to HOLD logic below (will be handled after this if-elif block)
                
                elif self.shares == 0 and self.balance > 0:
                    # RISK A FIX: Entry filters are OPTIONAL (default: disabled)
                    # Let agent learn from RSI as input, not hard-coded rules
                    # Only use filters if explicitly enabled (for safety/backtesting)
                    use_entry_filters = getattr(self.config, 'use_entry_filters', False)
                    if use_entry_filters:
                        can_enter, filter_msg = self._check_entry_filters(current_price, current_rsi)
                        if not can_enter:
                            # Small penalty for blocked entry (encourages learning)
                            reward -= 0.005
                            self.logger.debug(f"Entry filter blocked: {filter_msg}")
                    else:
                        # No filters - agent can learn any strategy (trend following, mean reversion, etc.)
                        can_enter = True
                    
                    if can_enter:
                        # FIX: Use position sizing instead of 100% of balance
                        # Calculate position value based on max_position_size_pct
                        position_value = self.balance * self.config.max_position_size_pct
                        
                        # Ensure minimum position size (at least $1 to avoid micro-trades)
                        min_position_value = 1.0
                        if position_value < min_position_value:
                            self.logger.warning(f"Position value ${position_value:.2f} too small, skipping BUY")
                            reward -= 0.01
                        else:
                            # CRITICAL FIX: BUY Logic - You pay MORE than the price
                            # 1. Slippage INCREASES the price you pay
                            fill_price = current_price * (1 + self.config.slippage_bps / 10000)
                            
                            # 2. Calculate shares you can buy with position_value
                            self.shares = position_value / fill_price
                            
                            # 3. Calculate Gross Value (Shares * Fill Price)
                            gross_value = self.shares * fill_price
                            
                            # 4. Calculate Commission (on the gross value)
                            commission = gross_value * self.config.commission_rate
                            
                            # 5. Total Cost to you (gross value + commission)
                            total_cost = gross_value + commission
                            
                            # 6. DEDUCT FROM BALANCE (The Fix)
                            # Ensure you are subtracting total_cost, NOT just gross_value
                            self.balance -= total_cost
                            
                            # Safety check: ensure balance doesn't go negative
                            if self.balance < 0:
                                self.logger.warning(f"Balance went negative: ${self.balance:.2f}, clamping to 0")
                                self.balance = 0.0
                            
                            # Log transaction cost for debugging
                            if self.logger.level <= logging.DEBUG:
                                self.logger.debug(f"BUY: shares={self.shares:.4f}, fill_price=${fill_price:.2f}, gross_value=${gross_value:.2f}, commission=${commission:.4f}, total_cost=${total_cost:.4f}, balance=${self.balance:.2f}")
                            
                            self.entry_price = fill_price
                            # Track entry step for stop loss/take profit timing analysis
                            self._entry_step = self.current_step
                            # Reset exit reason on new entry
                            self._exit_reason = None
                            
                            # IMPROVED: Calculate transaction cost (no action-based reward)
                            transaction_cost = self._calculate_transaction_cost(action, position_value)
                            
                            # ANTI-CHURN: Entry fee penalty will be applied in reward calculation
                            # (checked after all actions are executed)
                            
                            # Reset consecutive holds on BUY
                            if hasattr(self, '_consecutive_holds'):
                                self._consecutive_holds = 0
                            self.logger.debug(f"BUY executed: {self.shares:.4f} shares @ ${fill_price:.2f}, position_value=${position_value:.2f}, balance=${self.balance:.2f}")
                            # Mark trade as executed
                            self._last_trade_executed = True
                            self._last_trade_profit = 0.0  # No profit yet, just entry

            elif action == 2:  # SELL
                # ACTION MASKING FIX: Instead of penalizing invalid actions, treat them as HOLD
                # The model should use action masking to prevent invalid actions in the first place
                # If an invalid action slips through, just treat it as HOLD (no penalty)
                if self.shares == 0:
                    # Invalid SELL action - no position to sell
                    # JUST TREAT AS HOLD - no penalty (action masking should prevent this)
                    self.logger.debug(f"INVALID SELL: No position to sell - treating as HOLD (action masking should prevent this)")
                    action = 0  # Convert to HOLD
                    # Fall through to HOLD logic below (will be handled after this if-elif block)
                
                elif self.shares > 0:
                    # CRITICAL FIX: SELL Logic - You receive LESS than the price
                    # 1. Slippage DECREASES the price you get
                    fill_price = current_price * (1 - self.config.slippage_bps / 10000)
                    
                    # 2. Calculate Gross Value (Shares * Fill Price)
                    gross_value = self.shares * fill_price
                    
                    # 3. Calculate Commission (on the gross value)
                    commission = gross_value * self.config.commission_rate
                    
                    # 4. Net Proceeds to you (gross value - commission)
                    net_proceeds = gross_value - commission
                    
                    # 5. ADD TO BALANCE (The Fix)
                    self.balance += net_proceeds
                    
                    # Calculate profit/loss
                    cost_basis = self.shares * self.entry_price
                    profit_dollars = net_proceeds - cost_basis
                    realized_profit_pct = (profit_dollars / cost_basis) if cost_basis > 0 else 0
                    
                    # IMPROVED: Calculate transaction cost (no action-based reward)
                    transaction_cost = self._calculate_transaction_cost(action, gross_value)
                    self.shares = 0
                    
                    # Log transaction cost for debugging
                    if self.logger.level <= logging.DEBUG:
                        self.logger.debug(f"SELL: shares={self.shares:.4f}, fill_price=${fill_price:.2f}, gross_value=${gross_value:.2f}, commission=${commission:.4f}, net_proceeds=${net_proceeds:.4f}, balance=${self.balance:.2f}")
                    self.entry_price = 0
                    # Set exit reason for SELL action
                    self._exit_reason = "SELL"
                    # Reset consecutive holds on SELL
                    if hasattr(self, '_consecutive_holds'):
                        self._consecutive_holds = 0
                    self.logger.debug(f"SELL executed at {fill_price:.2f}, profit: {profit_dollars:.2f}")
                    # Mark trade as executed
                    self._last_trade_executed = True
                    self._last_trade_profit = profit_dollars
            
            # HOLD action with penalties for excessive holding
            elif action == 0:  # HOLD
                # Track consecutive holds
                if not hasattr(self, '_consecutive_holds'):
                    self._consecutive_holds = 0
                self._consecutive_holds += 1
                
                # FIX: Removed excessive holding penalty - it was too weak (0.005 scaled = 0.05)
                # If you want to punish holding, make it 0.1 or higher, but for now removed
                # The main issue is reward clipping, not holding behavior
                
                # Note: Next-bar logic moved to "Holding logic" section below
                # so it applies to ALL actions when holding, not just HOLD

            # Holding logic with improved risk management
            if self.shares > 0:
                current_return = (current_price - self.entry_price) / self.entry_price
                cost_basis = self.shares * self.entry_price

                # Stop loss - improved for intraday data
                if current_return < -self.config.stop_loss_pct:
                    # CRITICAL FIX: SELL Logic - You receive LESS than the price
                    # 1. Slippage DECREASES the price you get
                    fill_price = current_price * (1 - self.config.slippage_bps / 10000)
                    
                    # 2. Calculate Gross Value (Shares * Fill Price)
                    gross_value = self.shares * fill_price
                    
                    # 3. Calculate Commission (on the gross value)
                    commission = gross_value * self.config.commission_rate
                    
                    # 4. Net Proceeds to you (gross value - commission)
                    net_proceeds = gross_value - commission
                    
                    # 5. ADD TO BALANCE (The Fix)
                    self.balance += net_proceeds
                    profit_dollars = net_proceeds - cost_basis
                    realized_loss_pct = (profit_dollars / cost_basis) if cost_basis > 0 else -0.10
                    
                    # ANTI-CHURN: Track exit for whipsaw penalty (before resetting shares)
                    # This will be checked in reward calculation
                    
                    self.shares = 0.0
                    self.entry_price = 0.0
                    # Set exit reason for stop loss
                    self._exit_reason = "STOP_LOSS"
                    # Reset consecutive holds on stop loss
                    if hasattr(self, '_consecutive_holds'):
                        self._consecutive_holds = 0
                    
                    # IMPROVED: Calculate transaction cost (no action-based reward)
                    transaction_cost = self._calculate_transaction_cost(2, gross_value)  # SELL action
                    
                    # Note: Stop loss penalties are now handled in the main reward calculation
                    # to ensure they're applied even if stop loss is triggered automatically
                    
                    # Log to file only (repetitive warning)
                    self.logger.debug(f"Stop loss triggered at {current_return:.2%}")
                    # Mark trade as executed (stop loss is a trade)
                    self._last_trade_executed = True
                    self._last_trade_profit = profit_dollars

                # Take profit
                elif current_return > self.config.take_profit_pct:
                    # CRITICAL FIX: SELL Logic - You receive LESS than the price
                    # 1. Slippage DECREASES the price you get
                    fill_price = current_price * (1 - self.config.slippage_bps / 10000)
                    
                    # 2. Calculate Gross Value (Shares * Fill Price)
                    gross_value = self.shares * fill_price
                    
                    # 3. Calculate Commission (on the gross value)
                    commission = gross_value * self.config.commission_rate
                    
                    # 4. Net Proceeds to you (gross value - commission)
                    net_proceeds = gross_value - commission
                    
                    # 5. ADD TO BALANCE (The Fix)
                    self.balance += net_proceeds
                    cost_basis = self.shares * self.entry_price
                    profit_dollars = net_proceeds - cost_basis
                    realized_profit_pct = (profit_dollars / cost_basis) if cost_basis > 0 else 0.20
                    
                    # ANTI-CHURN: Track exit for whipsaw penalty (before resetting shares)
                    # This will be checked in reward calculation
                    
                    self.shares = 0.0
                    self.entry_price = 0.0
                    # Set exit reason for take profit
                    self._exit_reason = "TAKE_PROFIT"
                    # Reset consecutive holds on take profit
                    if hasattr(self, '_consecutive_holds'):
                        self._consecutive_holds = 0
                    
                    # IMPROVED: Calculate transaction cost (no action-based reward)
                    transaction_cost = self._calculate_transaction_cost(2, gross_value)  # SELL action
                    
                    # Log to file only (repetitive warning)
                    self.logger.debug(f"Take profit triggered at {current_return:.2%}")
                    # Mark trade as executed (take profit is a trade)
                    self._last_trade_executed = True
                    self._last_trade_profit = profit_dollars

                else:
                    # IMPROVED: No next-bar prediction in reward (future peaking removed)
                    # No unrealized P&L rewards (replaced by log return)
                    # Reward will be calculated based on portfolio value change
                    pass

            # IMPROVED: Calculate new reward structure: R_t = R_pnl - R_cost - R_volatility - R_stop_loss
            # Calculate current portfolio value AFTER all actions are executed
            current_portfolio_value = self._get_portfolio_value(current_price)
            
            # 1. Calculate log return (R_pnl)
            log_return = self._calculate_log_return(current_portfolio_value, prev_portfolio_value)
            
            # 2. Normalize transaction cost to ratio (R_cost)
            # transaction_cost is in dollars, normalize by portfolio value for consistency
            if prev_portfolio_value > 0:
                transaction_cost_ratio = transaction_cost / prev_portfolio_value
            else:
                transaction_cost_ratio = 0.0
            
            # 3. Calculate volatility penalty (R_volatility) - Keep for risk adjustment
            volatility_penalty = self._calculate_volatility_penalty()
            
            # ============================================================
            # REWARD CALCULATION - SOFT SCALING VERSION (NEW)
            # Replaces hard penalties with proportional multipliers
            # ============================================================
            # 
            # KEY FIX: Use "Pain Multipliers" instead of fixed penalties
            # This prevents penalties from dominating the reward signal
            # 
            # Formula: reward = (log_return * 100) * multiplier - volatility
            # Where multiplier scales with outcome quality
            # ============================================================
            
            # Step 1: Base Reward is simply the Percentage Return
            # If we made 1%, reward is 1.0. If we lost 5%, reward is -5.0.
            # Subtract transaction costs from the return
            raw_reward = (log_return - transaction_cost_ratio) * 100.0
            
            # Step 2: Get exit information for scaling
            exit_reason = getattr(self, '_exit_reason', None)
            bars_held = None
            if hasattr(self, '_entry_step') and self._entry_step is not None:
                entry_step = self._entry_step
                if isinstance(entry_step, (int, float)):
                    bars_held = self.current_step - entry_step
            
            # Step 3: Apply "Pain Multipliers" (Proportional Penalties)
            # Instead of subtracting fixed amounts, we make negative returns "feel" heavier
            if raw_reward < 0:
                # A. Stop Loss Penalty: Make the loss feel 10% worse
                # This discourages hitting SL, but doesn't panic the agent
                if exit_reason == "STOP_LOSS":
                    raw_reward *= 1.10
                    self.logger.debug(f"Stop loss penalty: Loss scaled by 1.10x (bars held: {bars_held})")
                
                # B. "Stupidity" (Quick Loss) Penalty: Make it feel 20% worse
                # If you lose money immediately, it hurts more
                elif bars_held is not None and bars_held <= 3:
                    raw_reward *= 1.20
                    self.logger.debug(f"Quick loss penalty: Loss scaled by 1.20x (bars held: {bars_held})")
            
            # Step 4: Apply "Bonus Multipliers" for positive returns
            elif raw_reward > 0:
                # A. Quick Win Bonus: Boost the reward by 5%
                # Encourages efficiency (quick profitable exits)
                if bars_held is not None and bars_held <= 5:
                    raw_reward *= 1.05
                    self.logger.debug(f"Quick win bonus: Profit scaled by 1.05x (bars held: {bars_held})")
            
            # Step 5: Apply volatility penalty (risk adjustment)
            reward = raw_reward - volatility_penalty
            
            # Step 6: Apply reward scale to make signals stronger for neural network
            # Rewards at 0.007 or 0.09 are "whispers" - we need to "shout" with scale 10.0
            REWARD_SCALE = 10.0
            reward = reward * REWARD_SCALE
            
            # Step 7: Clip to sanity limits (prevent gradient explosions)
            # FIX: Widen clip range to handle -50% crashes (-500 points after 10x scale)
            # With REWARD_SCALE=10.0, need wider range to catch big crashes
            # Old: [-200, +200] was too narrow - couldn't capture -20% crashes
            # New: [-500, +500] allows up to -50% crashes to be properly penalized
            REWARD_CLIP_MIN = -500.0
            REWARD_CLIP_MAX = +500.0
            reward = np.clip(reward, REWARD_CLIP_MIN, REWARD_CLIP_MAX)
            
            # Reset exit reason for next step
            if hasattr(self, '_exit_reason'):
                self._exit_reason = None
            
            # REWARD BREAKDOWN: Always calculate for info dict (needed for training)
            # Calculate multiplier applied
            multiplier_applied = 1.0
            if raw_reward < 0:
                if exit_reason == "STOP_LOSS":
                    multiplier_applied = 1.10
                elif bars_held is not None and bars_held <= 3:
                    multiplier_applied = 1.20
            elif raw_reward > 0:
                if bars_held is not None and bars_held <= 5:
                    multiplier_applied = 1.05
            
            # Always create reward_breakdown with expected keys for training code
            reward_breakdown = {
                'log_return': log_return,
                'transaction_cost_ratio': transaction_cost_ratio,
                'raw_reward': raw_reward,
                'multiplier_applied': multiplier_applied,
                'scaled_reward': raw_reward * multiplier_applied if multiplier_applied != 1.0 else raw_reward,
                'volatility_penalty': volatility_penalty,
                'exit_reason': exit_reason,
                'bars_held': bars_held,
                'final_reward': reward
            }
            
            # REWARD BREAKDOWN LOGGING: Only log every 50 steps to avoid spam
            if hasattr(self, '_reward_log_counter'):
                self._reward_log_counter += 1
            else:
                self._reward_log_counter = 0
            
            if self._reward_log_counter % 50 == 0:
                # self.logger.info(f"REWARD BREAKDOWN (step {self.current_step}): {reward_breakdown}")
                pass
            
            # 5. Update return history for next step
            self._return_history.append(log_return)
            if len(self._return_history) > self._volatility_window * 2:
                self._return_history = self._return_history[-self._volatility_window * 2:]
            
            # 6. Update previous portfolio value for next step
            self._prev_portfolio_value = current_portfolio_value
            
            self.current_step += 1
            if self.current_step >= self.max_steps:
                done = True

            # Build info dict with reward_breakdown (already created above with expected keys)
            action_type = ['HOLD', 'BUY', 'SELL'][action] if action in [0, 1, 2] else 'HOLD'
            
            # Add action_type to reward_breakdown for compatibility
            reward_breakdown['action_type'] = action_type
            
            info = {
                'reward_breakdown': reward_breakdown,  # Use the reward_breakdown with expected keys
                'executed_action': action_type,
                'hard_stop_loss': hard_stop_loss_triggered  # Flag for meta-agent penalty
            }
            
            # Add trade execution info if a trade occurred
            if hasattr(self, '_last_trade_executed') and self._last_trade_executed:
                info['trade_executed'] = True
                info['trade'] = True
                if hasattr(self, '_last_trade_profit'):
                    info['profit'] = self._last_trade_profit
                # Reset flag
                self._last_trade_executed = False
            else:
                info['trade_executed'] = False
                info['trade'] = False

            return self._get_observation(), reward, done, False, info

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

