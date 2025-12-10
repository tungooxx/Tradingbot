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
import warnings

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# 0. CONFIGURATION & DEVICE
# ==============================================================================
# 🚀 GPU SETUP: This makes training 20x faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Training Device: {device}")

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==============================================================================
# 1. KAN NEURAL NETWORK LAYERS
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

# ==============================================================================
# 2. MODELS (PREDICTOR & AGENT)
# ==============================================================================
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
# 3. TRADING ENVIRONMENT (NASDAQ EDITION)
# ==============================================================================
class StockTradingEnv(gym.Env):
    def __init__(self, ticker="^IXIC", window_size=30):
        super(StockTradingEnv, self).__init__()
        print(f"📥 Downloading {ticker} (NASDAQ) data...")

        # Download Data
        self.df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)

        # Feature Engineering
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

        self.window_size = window_size
        self.max_steps = len(self.data) - 1
        self.action_space = spaces.Discrete(3)
        self.obs_shape = window_size * len(self.features)

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        self.balance = 20.0
        self.shares = 0
        self.entry_price = 0
        return self._get_observation(), {}

    def _get_observation(self, step=None):
        if step is None: step = self.current_step
        window = self.data[step - self.window_size : step]
        return window.flatten().astype(np.float32)

    def step(self, action):
        current_price = self.data[self.current_step][0]
        daily_log_ret = self.data[self.current_step][1]
        reward = 0
        done = False

        # --- DEBUG COMMEND: Track Action Logic ---
        # if self.current_step % 500 == 0:
        #    print(f"DEBUG Step {self.current_step}: Action {action}, Price {current_price:.2f}, Shares {self.shares}")

        # --- 1. ACTION LOGIC ---
        if action == 1: # BUY
            if self.shares == 0 and self.balance > 0:
                self.shares = self.balance / current_price
                self.balance = 0.0
                self.entry_price = current_price
                reward -= 0.05

        elif action == 2: # SELL
            if self.shares > 0:
                gross_val = self.shares * current_price
                net_val = gross_val * 0.999
                cost_basis = self.shares * self.entry_price
                profit_dollars = net_val - cost_basis
                # Reward Profit %
                reward = (profit_dollars / cost_basis) * 100
                self.balance = net_val
                self.shares = 0
                self.entry_price = 0

        # --- 2. HOLDING LOGIC ---
        if self.shares > 0:
            current_return = (current_price - self.entry_price) / self.entry_price

            # A. Stop Loss (-5%)
            if current_return < -0.05:
                gross_val = self.shares * current_price
                self.balance = gross_val * 0.999
                self.shares = 0
                reward = -5.0 # Punishment

            # B. Take Profit (+20%)
            elif current_return > 0.20:
                gross_val = self.shares * current_price
                self.balance = gross_val * 0.999
                self.shares = 0
                reward = +20.0 # Reward

            # C. Compass (Stability Sauce)
            else:
                # This logic is what makes your agent "Stable"
                # It gets paid to hold, as long as daily return isn't terrible
                reward += (daily_log_ret * 10) + 0.01

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, False, {}

# ==============================================================================
# 4. TRAINING PIPELINE
# ==============================================================================
def prepare_data_gpu(env):
    """Moves all data to GPU at once for speed."""
    X_list = []
    y_list = []
    for i in range(env.window_size, len(env.data) - 1):
        window = env.data[i - env.window_size : i]
        # TARGET SCALING: Multiply by 100 to make the number visible to the brain
        target = env.data[i][1] * 100
        X_list.append(window.flatten())
        y_list.append(target)

    X_t = torch.tensor(np.array(X_list, dtype=np.float32)).to(device)
    y_t = torch.tensor(np.array(y_list, dtype=np.float32)).reshape(-1, 1).to(device)
    return X_t, y_t

def run_training_pipeline():
    # 1. Init NASDAQ Environment
    TICKER = "^IXIC" # NASDAQ Composite
    env = StockTradingEnv(ticker=TICKER, window_size=30)

    # --- PHASE 1: PRE-TRAINING (Warm Up the Brain) ---
    print(f"\n🧠 PHASE 1: Pre-training KAN on {TICKER}...")
    X_train, y_train = prepare_data_gpu(env)

    predictor = KANPredictor(env.obs_shape, hidden_dim=32).to(device)
    optimizer_pred = optim.Adam(predictor.parameters(), lr=0.001) # Higher LR for pretraining
    criterion = nn.MSELoss()

    for epoch in range(150): # 150 Epochs Sweet Spot
        optimizer_pred.zero_grad()
        preds = predictor(X_train)
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer_pred.step()

        if (epoch+1) % 50 == 0:
            print(f"   Epoch {epoch+1} | Loss: {loss.item():.4f} (Lower is better)")

    # --- PHASE 2: RL TRAINING (Teach it to Trade) ---
    print("\n🎮 PHASE 2: Starting RL (PPO) on NASDAQ...")
    agent = KANActorCritic(env.obs_shape, env.action_space.n, hidden_dim=32, pretrained_body=predictor.body).to(device)
    optimizer_rl = optim.Adam(agent.parameters(), lr=0.0005)

    state, _ = env.reset()
    memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
    recent_rewards = []

    for step in range(1, 30000): # 30k steps is plenty for daily data
        # Action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob, _ = agent.act(state_tensor)

        # DEBUG: Peek at the brain
        if step % 5000 == 0:
            print(f"   🔎 DEBUG [Step {step}]: Agent chose Action {action}")

        # Step
        next_state, reward, done, _, _ = env.step(action)

        # Store
        memory_states.append(state_tensor)
        memory_actions.append(torch.tensor(action).to(device))
        memory_logprobs.append(log_prob)
        memory_rewards.append(reward)
        recent_rewards.append(reward)

        state = next_state

        # PPO Update (Learn every 1000 steps)
        if step % 1000 == 0:
            old_states = torch.cat(memory_states)
            old_actions = torch.stack(memory_actions)
            old_logprobs = torch.stack(memory_logprobs)

            # Discounted Rewards
            rewards = []
            discounted_reward = 0
            for r in reversed(memory_rewards):
                discounted_reward = r + 0.99 * discounted_reward
                rewards.insert(0, discounted_reward)

            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            # Optimization Steps
            for _ in range(3):
                logprobs, state_values, dist_entropy = agent.evaluate(old_states, old_actions)
                state_values = torch.squeeze(state_values)
                ratios = torch.exp(logprobs - old_logprobs.detach())
                advantages = rewards - state_values.detach()
                loss = -torch.min(ratios * advantages, torch.clamp(ratios, 0.8, 1.2) * advantages) + 0.5 * nn.MSELoss()(state_values, rewards) - 0.01 * dist_entropy

                optimizer_rl.zero_grad()
                loss.mean().backward()
                optimizer_rl.step()

            # Report
            avg_rew = sum(recent_rewards) / len(recent_rewards)
            print(f"   🔄 Step {step} | Avg Reward: {avg_rew:.4f} (Positive = Good)")

            # Clear Memory
            memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
            recent_rewards = []

        if done:
            state, _ = env.reset()

    # Save
    torch.save(agent.state_dict(), "kan_agent_nasdaq.pth")
    print("✅ NASDAQ Surfer Trained. Saved as 'kan_agent_nasdaq.pth'")

def run_backtest():
    TICKER = "^IXIC"
    TEST_START, TEST_END = "2025-01-01", "2025-12-30" # Let's test on 2024 data
    print(f"\n⚔️ BACKTEST: {TICKER} ({TEST_START} -> {TEST_END})")

    # Load Data
    df = yf.download(TICKER, start=TEST_START, end=TEST_END, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)

    # Prep Features
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Norm'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.dropna(inplace=True)

    # Normalize
    df["RSI_14"] = df["RSI_14"] / 100.0
    df["MACD_12_26_9"] = (df["MACD_12_26_9"] - df["MACD_12_26_9"].mean()) / (df["MACD_12_26_9"].std() + 1e-7)

    features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9"]
    env = StockTradingEnv(ticker=TICKER, window_size=30)
    env.df = df
    env.data = df[features].values
    env.max_steps = len(env.data) - 1

    # Load Agent
    agent = KANActorCritic(env.obs_shape, env.action_space.n, hidden_dim=32).to(device)
    try:
        agent.load_state_dict(torch.load("kan_agent_nasdaq.pth", map_location=device, weights_only=True))
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return

    agent.eval()
    state, _ = env.reset()
    done = False

    portfolio_history = []

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = agent.actor_head(agent.body(state_tensor))
            action = torch.argmax(logits, dim=1).item()

        next_state, _, done, _, _ = env.step(action)

        # Calculate Value
        current_idx = env.current_step
        if current_idx < len(env.df):
            current_price = env.data[current_idx][0]
            val = env.balance + (env.shares * current_price)
            portfolio_history.append(val)
        state = next_state

    if portfolio_history:
        start_bal = 20.0
        end_bal = portfolio_history[-1]
        ret = (end_bal - start_bal) / start_bal * 100
        print(f"📊 Final Return: {ret:.2f}% (Start: ${start_bal} -> End: ${end_bal:.2f})")
        plt.plot(portfolio_history)
        plt.title(f"NASDAQ Surfer Performance: {ret:.2f}%")
        plt.show()

if __name__ == "__main__":
    # 1. Train on NASDAQ
    run_training_pipeline()
    # 2. Test

    run_backtest()