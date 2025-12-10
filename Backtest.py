import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import warnings

# Suppress yfinance warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ==============================================================================
# 1. PASTE YOUR MODEL CLASSES HERE
# (Must match training EXACTLY so torch.load works)
# ==============================================================================

class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02,
                 grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features,
                                                                                                       -1).contiguous())
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.Tensor(out_features, in_features, grid_size + spline_order))
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
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + self.spline_order, self.in_features,
                                 self.out_features) - 1 / 2) * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + (
                        (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
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
            output = output.view(original_shape[0], original_shape[1], self.out_features)
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


class KANActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, pretrained_body=None):
        super(KANActorCritic, self).__init__()
        if pretrained_body:
            self.body = pretrained_body
        else:
            self.body = KANBody(obs_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def act(self, state):
        features = self.body(state)
        logits = self.actor_head(features)

        # --- THE FIX IS HERE ---
        # OLD WAY (Training): dist.sample() -> Random based on %
        # NEW WAY (Prediction): torch.argmax() -> Always picks the highest %
        action = torch.argmax(logits, dim=1)

        return action.item()


# ==============================================================================
# 2. THE PREDICTION LOGIC
# ==============================================================================

def get_recommendation(ticker="NVDA"):
    print(f"\n🔮 Analyzing {ticker} for Tomorrow...")

    # 1. Fetch Data
    df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Feature Engineering
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Norm'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    df.dropna(inplace=True)

    # Normalize
    df["RSI_14"] = df["RSI_14"] / 100.0
    df["MACD_12_26_9"] = (df["MACD_12_26_9"] - df["MACD_12_26_9"].mean()) / (df["MACD_12_26_9"].std() + 1e-7)

    features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9"]

    # 3. Check Data Length
    if len(df) < 30:
        print("❌ Not enough data.")
        return

    # 4. Prepare Observation
    window = df[features].iloc[-30:].values
    obs = window.flatten().astype(np.float32)
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

    # 5. Load Model
    # IMPORTANT: Ensure hidden_dim matches your saved model (likely 32 or 64)
    agent = KANActorCritic(obs_dim=150, action_dim=3, hidden_dim=32)
    try:
        agent.load_state_dict(torch.load("kan_agent.pth", weights_only=True))
        agent.eval()
    except FileNotFoundError:
        print("❌ Model 'kan_agent.pth' not found.")
        return

    # 6. Predict
    with torch.no_grad():
        action = agent.act(obs_tensor)

    # 7. DISPLAY RESULTS
    current_price = df['Close'].iloc[-1]
    rsi_display = df['RSI_14'].iloc[-1] * 100
    macd_display = df['MACD_12_26_9'].iloc[-1]

    print(f"   Price: ${current_price:.2f}")
    print(f"   RSI: {rsi_display:.1f}  |  MACD: {macd_display:.2f}")
    print("-" * 40)

    # --- THE NEW TRADE PLANNER ---
    if action == 1:  # BUY
        # Calculate Logic
        stop_loss_price = current_price * 0.95  # -5%
        take_profit_price = current_price * 1.20  # +20%

        print("🚀 SIGNAL: BUY")
        print("   Reasoning: Model detects upward momentum.")
        print("-" * 40)
        print("📋 TRADE PLAN (Enter this into Webull):")
        print(f"   1. ENTRY:      ${current_price:.2f} (Market Order)")
        print(f"   2. STOP LOSS:  ${stop_loss_price:.2f} (-5%)")
        print(f"   3. TAKE PROFIT:${take_profit_price:.2f} (+20%)")

    elif action == 2:  # SELL
        print("🔴 SIGNAL: SELL")
        print("   Reasoning: Model detects weakness/downtrend.")
        print("   Action: Close any open positions immediately.")

    else:  # SKIP
        print("💤 SIGNAL: HOLD / WAIT")
        print("   Reasoning: Market is indecisive.")
        print("   Action: If you hold shares, keep holding. If cash, stay cash.")
    print("-" * 40)


if __name__ == "__main__":
    # Multi-Stock Loop
    tickers = ["NVDA"]
    for t in tickers:
        try:
            get_recommendation(t)
        except Exception as e:
            print(f"⚠️ Error analyzing {t}: {e}")