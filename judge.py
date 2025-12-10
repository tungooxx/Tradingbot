import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import pandas_ta as ta
import warnings
import torch.nn.functional as F

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# 1. SETUP: KAN & AGENT CLASSES
# ==============================================================================
# Use CPU for prediction (Simpler)
device = torch.device("cpu")

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
        torch.nn.init.kaiming_uniform_(self.base_weight, a=5**0.5 * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + self.spline_order, self.in_features, self.out_features) - 1 / 2) * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_((self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(self.grid.T[self.spline_order : -self.spline_order], noise))
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=5**0.5 * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + ((grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:])
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
        action = torch.argmax(logits, dim=1)
        return action.item(), None, self.critic_head(features)

# ==============================================================================
# 2. COUNCIL CLASS
# ==============================================================================
class Council(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(Council, self).__init__()
        self.surfer = KANActorCritic(obs_dim, action_dim, hidden_dim)
        self.guardian = KANActorCritic(obs_dim, action_dim, hidden_dim)

    def load_experts(self, surfer_path, guardian_path):
        try:
            self.surfer.load_state_dict(torch.load(surfer_path, map_location=device, weights_only=True))
            print("✅ Surfer Loaded.")
        except:
            print(f"❌ ERROR: Could not load {surfer_path}")
            return False

        try:
            self.guardian.load_state_dict(torch.load(guardian_path, map_location=device, weights_only=True))
            print("✅ Guardian Loaded.")
        except:
            print(f"❌ ERROR: Could not load {guardian_path}")
            return False

        self.surfer.eval()
        self.guardian.eval()
        return True

    def get_decision(self, state_tensor, veto_threshold):
        with torch.no_grad():
            # 1. Ask Guardian (Safety)
            _, _, guardian_value = self.guardian.act(state_tensor)
            fear_level = guardian_value.item()

            # 2. Ask Surfer (Profit)
            surfer_action, _, _ = self.surfer.act(state_tensor)

            # 3. Logic
            vetoed = False
            final_action = surfer_action

            if fear_level < veto_threshold:
                final_action = 2 # Forced Sell
                vetoed = True

            return final_action, surfer_action, fear_level, vetoed

# ==============================================================================
# 3. PREDICTION FUNCTION
# ==============================================================================
def predict_council_decision():
    # --- CONFIGURATION ---
    TICKER = "^IXIC"
    VETO_THRESHOLD = -0.5  # Set this to your "Winner" from optimization
    SURFER_FILE = "kan_agent.pth"
    GUARDIAN_FILE = "kan_agent_guardian.pth"

    print(f"\n🏛️ COUNCIL MEETING: Analyzing {TICKER}...")

    # 1. Fetch Data
    df = yf.download(TICKER, period="6mo", interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

    # 2. Features (Exact Match)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Norm'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.dropna(inplace=True)

    # Normalize
    df["RSI_14"] = df["RSI_14"] / 100.0
    df["MACD_12_26_9"] = (df["MACD_12_26_9"] - df["MACD_12_26_9"].mean()) / (df["MACD_12_26_9"].std() + 1e-7)

    features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9"]
    window = df[features].iloc[-30:].values
    obs = window.flatten().astype(np.float32)
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

    # 3. Init Council
    council = Council(150, 3, hidden_dim=32).to(device)
    if not council.load_experts(SURFER_FILE, GUARDIAN_FILE):
        return

    # 4. Get Decision
    final_action, surfer_act, fear, vetoed = council.get_decision(state_tensor, VETO_THRESHOLD)

    # 5. Report
    current_price = df['Close'].iloc[-1]

    print("\n" + "="*40)
    print(f"   📊 MARKET REPORT: {TICKER}")
    print(f"   Current Price: ${current_price:.2f}")
    print("-" * 40)

    print(f"   😱 Guardian Fear Level: {fear:.4f}")
    print(f"      (Veto Threshold: {VETO_THRESHOLD})")

    act_str = ["SKIP", "BUY", "SELL"]
    print(f"   🏄 Surfer Suggestion:   {act_str[surfer_act]}")

    if vetoed:
        print("   🛡️ STATUS: VETOED (Guardian blocked the trade)")

    print("-" * 40)
    print(f"   🏆 FINAL DECISION: {act_str[final_action]}")
    print("="*40)

    # 6. Trade Plan
    if final_action == 1: # BUY
        sl = current_price * 0.95 # -5% for ^IXIC
        tp = current_price * 1.10 # +10%
        print(f"   🚀 PLAN: Long Entry")
        print(f"      Stop Loss:   ${sl:.2f}")
        print(f"      Take Profit: ${tp:.2f}")
    elif final_action == 2:
        print("   🔴 PLAN: Close Positions / Stay Cash")

if __name__ == "__main__":
    predict_council_decision()