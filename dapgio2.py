import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import math
import pandas_ta as ta
import random
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

# ==============================================================================
# 1. KAN & AGENT CLASSES (Standard)
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

class KANPredictor(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(KANPredictor, self).__init__()
        self.body = KANBody(obs_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        return self.head(self.body(x))

class KANActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64, pretrained_body=None):
        super(KANActorCritic, self).__init__()
        if pretrained_body: self.body = pretrained_body
        else: self.body = KANBody(obs_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
    def act(self, state):
        features = self.body(state)
        logits = self.actor_head(features)
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

# ==============================================================================
# 2. THE BOOTCAMP ENVIRONMENT (Aggressive)
# ==============================================================================
class BootcampEnv(gym.Env):
    def __init__(self, ticker="NVDA", mode="train"):
        super(BootcampEnv, self).__init__()
        print(f"📥 Downloading {ticker} for {mode.upper()}...")

        # 1. Download ALL Data
        raw_df = yf.download(ticker, period="5y", interval="1d", progress=False, auto_adjust=True)
        if isinstance(raw_df.columns, pd.MultiIndex): raw_df.columns = raw_df.columns.get_level_values(0)

        # 2. THE CRITICAL SPLIT (Time Machine)
        # Calculate the split index (80% Train, 20% Test)
        split_idx = int(len(raw_df) * 0.8)

        if mode == "train":
            # Train on the PAST (First 4 years)
            self.df = raw_df.iloc[:split_idx].copy()
            print(f"   📅 Training Data: {self.df.index[0].date()} -> {self.df.index[-1].date()}")
        else:
            # Test on the FUTURE (Last 1 year)
            self.df = raw_df.iloc[split_idx:].copy()
            print(f"   📅 Testing Data:  {self.df.index[0].date()} -> {self.df.index[-1].date()}")

        # 3. Feature Engineering (Same as before)
        self.df["Log_Ret"] = np.log(self.df["Close"] / self.df["Close"].shift(1))
        self.df["Vol_Norm"] = self.df["Volume"] / self.df["Volume"].rolling(20).mean()
        self.df.ta.rsi(length=14, append=True)
        self.df.ta.macd(fast=12, slow=26, signal=9, append=True)
        self.df.dropna(inplace=True)

        # Normalize
        self.df["RSI_14"] = self.df["RSI_14"] / 100.0
        self.df["MACD_12_26_9"] = (self.df["MACD_12_26_9"] - self.df["MACD_12_26_9"].mean()) / (self.df["MACD_12_26_9"].std() + 1e-7)

        self.features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9"]
        self.data = self.df[self.features].values
        self.window_size = 30
        self.max_steps = len(self.data) - 1
        self.action_space = spaces.Discrete(3)
        self.obs_shape = self.window_size * len(self.features)

    # ... (Keep reset, _get_observation, and step exactly the same as your code) ...
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = 100.0
        self.shares = 0
        self.entry_price = 0
        return self._get_observation(), {}

    def _get_observation(self, step=None):
        if step is None: step = self.current_step
        window = self.data[step - self.window_size : step]
        return window.flatten().astype(np.float32)

    def step(self, action):
        # (Paste your step function with the Boredom Penalty here)
        # For brevity, I am assuming you use the one we just wrote
        current_price = self.data[self.current_step][0]
        daily_log_ret = self.data[self.current_step][1]
        reward = 0
        done = False

        if action == 1: # BUY
            if self.shares == 0:
                self.shares = self.balance / current_price
                self.balance = 0.0
                self.entry_price = current_price
                reward += 0.2
        elif action == 2: # SELL
            if self.shares > 0:
                gross_val = self.shares * current_price
                self.balance = gross_val * 0.999
                reward = ((gross_val - (self.shares * self.entry_price))/(self.shares * self.entry_price)) * 10
                self.shares = 0
                self.entry_price = 0
            else:
                reward -= 0.1

        if self.shares > 0:
            current_ret = (current_price - self.entry_price) / self.entry_price
            if current_ret < -0.08:
                self.balance = self.shares * current_price * 0.999
                self.shares = 0
                reward = -2.0
            elif current_ret > 0.25:
                self.balance = self.shares * current_price * 0.999
                self.shares = 0
                reward = +5.0
            else:
                reward += (daily_log_ret * 10) + 0.05
        else:
            reward -= 0.05 # Boredom

        self.current_step += 1
        if self.current_step >= self.max_steps: done = True
        return self._get_observation(), reward, done, False, {}

# ==============================================================================
# 3. TRAIN & VERIFY
# ==============================================================================
def train_and_verify():
    # 1. Initialize TRAIN Env (Past Data Only)
    env = BootcampEnv(ticker="NVDA", mode="train")

    # --- PHASE 1: PRE-TRAIN ---
    print("\n🧠 PHASE 1: Pre-training on PAST Data...")
    X_list, y_list = [], []
    for i in range(env.window_size, len(env.data) - 1):
        X_list.append(env.data[i - env.window_size : i].flatten())
        y_list.append(env.data[i][1] * 100)

    X_train = torch.tensor(np.array(X_list, dtype=np.float32)).to(device)
    y_train = torch.tensor(np.array(y_list, dtype=np.float32)).reshape(-1, 1).to(device)

    predictor = KANPredictor(env.obs_shape, hidden_dim=32).to(device)
    opt_pred = optim.Adam(predictor.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(2500):
        opt_pred.zero_grad()
        loss = loss_fn(predictor(X_train), y_train)
        loss.backward()
        opt_pred.step()
        if epoch % 50 == 0: print(f"   Epoch {epoch}: Loss {loss.item():.4f}")

    # --- PHASE 2: RL TRAINING ---
    print("\n🎮 PHASE 2: RL Bootcamp on PAST Data...")
    agent = KANActorCritic(env.obs_shape, env.action_space.n, hidden_dim=32, pretrained_body=predictor.body).to(device)
    opt_rl = optim.Adam(agent.parameters(), lr=0.0003)

    state, _ = env.reset()
    mem_s, mem_a, mem_lp, mem_r = [], [], [], []

    for step in range(1, 40000):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob, _ = agent.act(state_t)

        next_state, reward, done, _, _ = env.step(action) # action is int here

        mem_s.append(state_t)
        mem_a.append(torch.tensor(action).to(device)) # Convert to Tensor
        mem_lp.append(log_prob)
        mem_r.append(reward)
        state = next_state

        if step % 2000 == 0:
            # PPO Update Code...
            old_s = torch.cat(mem_s)
            old_a = torch.stack(mem_a)
            old_lp = torch.stack(mem_lp)

            rewards = []
            disc_r = 0
            for r in reversed(mem_r):
                disc_r = r + 0.99 * disc_r
                rewards.insert(0, disc_r)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            for _ in range(3):
                lp, vals, ent = agent.evaluate(old_s, old_a)
                vals = torch.squeeze(vals)
                ratios = torch.exp(lp - old_lp.detach())
                adv = rewards - vals.detach()
                loss = -torch.min(ratios * adv, torch.clamp(ratios, 0.8, 1.2) * adv) + 0.5 * nn.MSELoss()(vals, rewards) - 0.01 * ent
                opt_rl.zero_grad()
                loss.mean().backward()
                opt_rl.step()
            mem_s, mem_a, mem_lp, mem_r = [], [], [], []
            print(f"   Step {step}: Training...")

        if done: state, _ = env.reset()

    torch.save(agent.state_dict(), "kan_agent.pth")
    print("✅ Bootcamp Complete. Agent Saved.")

    # --- PHASE 3: THE HONEST TEST ---
    print("\n⚔️ VERIFYING on UNSEEN FUTURE DATA (Test Set)...")

    # Initialize TEST Env (Future Data Only)
    test_env = BootcampEnv(ticker="NVDA", mode="test")

    agent.eval()
    state, _ = test_env.reset()
    actions = []

    # Run through the entire test set
    while True:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = agent.actor_head(agent.body(state_t))
            action = torch.argmax(logits, dim=1).item()

        actions.append(action)
        next_state, _, done, _, _ = test_env.step(action)
        state = next_state

        if done: break

    print(f"   Actions Distribution on Unknown Data: {np.bincount(actions, minlength=3)}")

    sell_count = np.bincount(actions, minlength=3)[2]
    if sell_count > (len(actions) * 0.8):
        print("❌ AGENT IS SCARED of the future.")
    else:
        print("✅ AGENT IS BRAVE! (It traded on unseen data).")

if __name__ == "__main__":
    train_and_verify()