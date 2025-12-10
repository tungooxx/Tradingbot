import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import pandas_ta as ta
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# 1. DEFINE MODELS (MUST MATCH TRAINING EXACTLY)
# ==============================================================================
class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features, -1).contiguous())
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
            noise = ((torch.rand(self.grid_size + self.spline_order, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_((self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])
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
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(KANActorCritic, self).__init__()
        self.body = KANBody(obs_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def act(self, state):
        features = self.body(state)
        logits = self.actor_head(features)
        # Use argmax for deterministic testing
        action = torch.argmax(logits, dim=1)
        return action.item(), None, self.critic_head(features)

# ==============================================================================
# 2. THE COUNCIL CLASS
# ==============================================================================
class Council(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Council, self).__init__()
        self.surfer = KANActorCritic(obs_dim, action_dim, hidden_dim)
        self.guardian = KANActorCritic(obs_dim, action_dim, hidden_dim)

    def load_experts(self, surfer_path, guardian_path):
        # Load Surfer
        try:
            self.surfer.load_state_dict(torch.load(surfer_path, weights_only=True))
        except Exception as e:
            print(f"❌ Error loading Surfer: {e}")
            return False

        # Load Guardian
        try:
            self.guardian.load_state_dict(torch.load(guardian_path, weights_only=True))
        except Exception as e:
            print(f"❌ Error loading Guardian: {e}")
            return False

        self.surfer.eval()
        self.guardian.eval()
        return True

    def get_decision(self, state_tensor, veto_threshold):
        with torch.no_grad():
            # 1. Ask Guardian (Safety)
            _, _, guardian_value = self.guardian.act(state_tensor)
            fear_level = guardian_value.item()

            # 2. Veto Check
            if fear_level < veto_threshold:
                return 2 # SELL (Vetoed)

            # 3. Ask Surfer (Profit)
            surfer_action, _, _ = self.surfer.act(state_tensor)
            return surfer_action

# ==============================================================================
# 3. THE OPTIMIZATION LOOP
# ==============================================================================
def run_optimization():
    # --- A. SETUP ---
    TICKER = "^IXIC" # or NVDA
    START_BAL = 50.0
    THRESHOLDS = [-0.01,-0.05,-0.1, -0.5, -1.0, -2.0, -5.0, 0, 0.01, 0.05] # Test these fear levels
    # -0.1 = Very Paranoid (Sells everything)
    # -5.0 = Very Relaxed (Only sells if market is dying)

    print(f"🏛️ COUNCIL OPTIMIZATION: {TICKER}")
    print(f"   Starting Balance: ${START_BAL}")
    print(f"   Testing Thresholds: {THRESHOLDS}")

    # --- B. LOAD DATA ---
    df = yf.download(TICKER, period="1y", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Feature Engineering (Must match training)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Norm'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.dropna(inplace=True)

    # Normalize
    df["RSI_14"] = df["RSI_14"] / 100.0
    df["MACD_12_26_9"] = (df["MACD_12_26_9"] - df["MACD_12_26_9"].mean()) / (df["MACD_12_26_9"].std() + 1e-7)

    features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9"]
    data_values = df[features].values
    window_size = 30

    # --- C. INIT COUNCIL ---
    # Ensure hidden_dim matches your saved files (try 32 or 64)
    council = Council(150, 3, hidden_dim=32)
    loaded = council.load_experts("kan_agent_nasdaq.pth", "kan_agent_guardian.pth")

    if not loaded:
        return

    # --- D. RUN SIMULATIONS ---
    results = {}

    plt.figure(figsize=(12, 6))

    # 1. Plot Buy & Hold for reference
    initial_price = data_values[window_size][0]
    bh_shares = START_BAL / initial_price
    bh_curve = [bh_shares * data_values[i][0] for i in range(window_size, len(data_values))]
    plt.plot(bh_curve, label='Buy & Hold', color='black', linestyle='--', alpha=0.5)

    # 2. Loop through Thresholds
    for thresh in THRESHOLDS:
        balance = START_BAL
        shares = 0.0
        equity_curve = []

        print(f"\n⚙️ Testing Veto Threshold: {thresh}")

        for step in range(window_size, len(data_values)):
            # Observe
            window = data_values[step - window_size : step]
            obs = window.flatten().astype(np.float32)
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)

            # Decide
            with torch.no_grad():
                # Ask Guardian directly to see the Value
                _, _, guard_val = council.guardian.act(state_tensor)
                fear = guard_val.item()

                # Ask Surfer directly to see the Action
                surf_act, _, _ = council.surfer.act(state_tensor)

            # Print status every 20 days so we don't spam console
            if step % 20 == 0:
                print(f"Step {step}: Fear={fear:.4f} | Surfer Action={surf_act}")
            # --- DEBUG PROBE END ---

            # Decide
            action = council.get_decision(state_tensor, veto_threshold=thresh)

            # Execute
            current_price = data_values[step][0]

            # BUY
            if action == 1:
                if shares == 0 and balance > 0:
                    shares = balance / current_price
                    balance = 0.0

            # SELL (Or Vetoed)
            elif action == 2:
                if shares > 0:
                    balance = shares * current_price * 0.999 # Fee
                    shares = 0.0

            # Calculate Equity
            val = balance + (shares * current_price)
            equity_curve.append(val)

        final_val = equity_curve[-1]
        profit_pct = (final_val - START_BAL) / START_BAL * 100
        results[thresh] = profit_pct

        print(f"   🏁 Final Balance: ${final_val:.2f} ({profit_pct:+.2f}%)")
        plt.plot(equity_curve, label=f'Thresh {thresh} ({profit_pct:.1f}%)')

    # --- E. FINALIZE ---
    plt.title(f"Council Strategy Optimization (Starting ${START_BAL})")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    best_thresh = max(results, key=results.get)
    print("\n" + "="*40)
    print(f"🏆 WINNER: Threshold {best_thresh}")
    print(f"   Profit: {results[best_thresh]:.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_optimization()