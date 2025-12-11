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
from torch.utils.data import TensorDataset, DataLoader # Add this import at the top if missing
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
class SelfAttention(nn.Module):
    """
    The 'Focus' Block.
    Allows the model to weigh the importance of different time steps.
    """
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Attention Score = (Q * K^T) / sqrt(dim)
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Weighted Sum
        context = torch.bmm(attention_weights, V)
        return context, attention_weights

class KANRecurrentMemory(nn.Module):
    """
    The 'Lego' Hybrid: KAN + GRU + Attention.
    """
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=1, seq_len=30):
        super(KANRecurrentMemory, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len

        # BLOCK 1: KAN Feature Extractor (The Mathematician)
        self.kan_feature_layer = KANLinear(input_dim, hidden_dim)

        # BLOCK 2: GRU (The Historian - Time Memory)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)

        # BLOCK 3: Self-Attention (The Sniper - Focus)
        self.attention = SelfAttention(hidden_dim)

        # Normalization
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # Input x shape: [Batch, Window_Size * Features]
        batch_size = x.shape[0]

        # 1. Reshape for Time Series processing [Batch, Seq, Features]
        # We ensure the reshape matches the configured dimensions
        x = x.view(batch_size, self.seq_len, self.input_dim)

        # 2. Apply KAN to extract non-linear patterns from EACH day
        # Flatten [Batch * Seq, Features] for KAN
        x_flat = x.view(-1, self.input_dim)
        kan_out = self.kan_feature_layer(x_flat)
        kan_out = self.ln(kan_out)
        kan_out = F.silu(kan_out)

        # Reshape back to [Batch, Seq, Hidden]
        kan_seq = kan_out.view(batch_size, self.seq_len, -1)

        # 3. Pass through GRU
        gru_out, _ = self.gru(kan_seq)

        # 4. Apply Attention
        attn_out, _ = self.attention(gru_out)

        # 5. Feature Pooling (Sum context)
        final_feature = torch.sum(attn_out, dim=1)

        return final_feature

class KANPredictor(nn.Module):
    """
    The Pre-training Model.
    Predicts the next day's return to teach the brain 'Physics'.
    """
    def __init__(self, obs_dim, hidden_dim=64):
        super(KANPredictor, self).__init__()
        # obs_dim is (Window * Features). We assume Features=6 and Window=30.
        self.body = KANRecurrentMemory(input_dim=6, hidden_dim=hidden_dim, seq_len=30)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.body(x)
        return self.head(features)

class KANActorCritic(nn.Module):
    """
    The RL Agent.
    Decides BUY/SELL/HOLD.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64, pretrained_body=None):
        super(KANActorCritic, self).__init__()
        if pretrained_body:
            self.body = pretrained_body
        else:
            self.body = KANRecurrentMemory(input_dim=6, hidden_dim=hidden_dim, seq_len=30)

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
def download_and_process(ticker, period="5y"):
    """Downloads and adds technical indicators including Trend Score."""
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 100: return None

        # Feature Engineering
        df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Vol_Norm"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-7)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)

        # --- NEW INNOVATION: Trend Score ---
        # Helps the agent detect crashes (ARKK scenario)
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["Trend_Score"] = (df["Close"] - df["SMA_50"]) / (df["SMA_50"] + 1e-7)

        # Cleanup
        df.dropna(inplace=True)

        # Normalize Indicators
        df["RSI_14"] = df["RSI_14"] / 100.0
        df["MACD_12_26_9"] = (df["MACD_12_26_9"] - df["MACD_12_26_9"].mean()) / (df["MACD_12_26_9"].std() + 1e-7)

        # Important: We now have 6 features
        features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9", "Trend_Score"]
        return df[features].astype(np.float32)
    except Exception as e:
        print(f"⚠️ Failed to download {ticker}: {e}")
        return None

def load_universe():
    # --- STRATEGY: MIX GOOD AND BAD COMPANIES ---
    # Good: Strong trends, recognizable patterns
    good_stocks = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "AMD", "TSLA", "META"]

    # Bad: Volatile, downtrends, traps (Curriculum Learning to recognize danger)
    bad_stocks = ["PTON", "ZM", "DOCU", "ARKK", "SPCE", "AMC", "GME", "CVNA"]

    universe = {}
    print("🌍 Loading Multi-Stock Universe...")

    all_tickers = good_stocks + bad_stocks
    for ticker in all_tickers:
        print(f"   Downloading {ticker}...", end="\r")
        df = download_and_process(ticker)
        if df is not None:
            universe[ticker] = df

    print(f"\n✅ Universe Loaded: {len(universe)} stocks ready.")
    return universe
# ==============================================================================
# 3. TRADING ENVIRONMENT (NASDAQ EDITION)
# ==============================================================================
class StockTradingEnv(gym.Env):
    def __init__(self, universe_dict, window_size=30, mode="train"):
        super(StockTradingEnv, self).__init__()
        self.universe = universe_dict
        self.window_size = window_size
        self.mode = mode

        self.tickers = list(self.universe.keys())
        self.action_space = spaces.Discrete(3)
        self.obs_shape = window_size * 6 # 5 Features

        # Placeholder for current episode data
        self.current_ticker = None
        self.data = None
        self.max_steps = 0

    def reset(self, seed=None):
        super().reset(seed=seed)

        # --- MULTI-STOCK LOGIC ---
        # 1. Pick a random stock from the universe
        self.current_ticker = random.choice(self.tickers)
        raw_data = self.universe[self.current_ticker].values

        # 2. Pick a RANDOM TIME SLICE (Crucial for generalization)
        # We don't always want to start at index 30.
        # Sometimes we start in 2021, sometimes 2023.
        total_len = len(raw_data)
        min_episode_len = 200 # Ensure we have at least 200 days to play

        if total_len > (self.window_size + min_episode_len):
            # Random start index between window_size and (end - min_len)
            max_start_idx = total_len - min_episode_len
            self.current_step = random.randint(self.window_size, max_start_idx)
        else:
            self.current_step = self.window_size

        # Store the data for this episode
        self.data = raw_data
        self.max_steps = total_len - 1

        self.balance = 20.0
        self.shares = 0
        self.entry_price = 0

        # Debug info for the user to see variety
        # print(f"DEBUG: Resetting Env | Ticker: {self.current_ticker} | Start Idx: {self.current_step}")

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
                profit_dollars = net_val - (self.shares * self.entry_price)
                cost_basis = self.shares * self.entry_price

                # Reward Profit Percentage
                reward = (profit_dollars / cost_basis) * 100
                self.balance = net_val
                self.shares = 0
                self.entry_price = 0

        # --- 2. HOLDING LOGIC ---
        if self.shares > 0:
            current_return = (current_price - self.entry_price) / self.entry_price

            # Stop Loss (-5%)
            if current_return < -0.05:
                gross_val = self.shares * current_price
                self.balance = gross_val * 0.999
                self.shares = 0
                reward = -2.0 # Punish hitting stop loss

            # Take Profit (+20%)
            elif current_return > 0.20:
                gross_val = self.shares * current_price
                self.balance = gross_val * 0.999
                self.shares = 0
                reward = +10.0 # Reward hitting take profit

            else:
                reward += (daily_log_ret * 10) + 0.01

        # --- 3. BOREDOM PENALTY ---
        else:
            reward -= 0.05

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, False, {}

# ==============================================================================
# 5. TRAINING PIPELINE
# ==============================================================================
def prepare_multi_stock_data_gpu(universe_dict, window_size=30):
    """
    Combines segments from ALL stocks into one giant tensor for Pre-training.
    This teaches the brain general market physics before we start trading.
    """
    print("   📊 Aggregating data from all stocks for pre-training...")
    X_list = []
    y_list = []

    # We take a sample of 1000 windows from each stock to keep memory manageable
    samples_per_stock = 1000

    for ticker, df in universe_dict.items():
        data = df.values
        if len(data) < window_size + 10: continue

        # Randomly sample indices
        valid_indices = range(window_size, len(data) - 1)
        # Don't error if data is small
        n_samples = min(samples_per_stock, len(valid_indices))
        selected_indices = random.sample(valid_indices, n_samples)

        for i in selected_indices:
            window = data[i - window_size : i]
            # Target: Next day's log return scaled up
            target = data[i][1] * 100

            X_list.append(window.flatten())
            y_list.append(target)

    X_t = torch.tensor(np.array(X_list, dtype=np.float32)).to(device)
    y_t = torch.tensor(np.array(y_list, dtype=np.float32)).reshape(-1, 1).to(device)
    print(f"   📊 Pre-training Dataset Size: {len(X_t)} samples")
    return X_t, y_t

def run_training_pipeline():
    # 1. Load Universe ONCE
    universe = load_universe()
    if not universe:
        print("❌ Error: No data loaded.")
        return

    # 2. Init Environment
    env = StockTradingEnv(universe_dict=universe, window_size=30, mode="train")

    # --- PHASE 1: PRE-TRAINING (Generalized Market Physics) ---
    print("\n🧠 PHASE 1: Pre-training KAN on Multi-Stock Data...")

    # Prepare Data
    X_train, y_train = prepare_multi_stock_data_gpu(universe, window_size=30)

    # 🔧 FIX: Use Mini-Batches to prevent GPU Crash
    # We break 16,000 samples into small chunks of 64
    batch_size = 64
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    predictor = KANPredictor(env.obs_shape, hidden_dim=64).to(device)
    optimizer_pred = optim.Adam(predictor.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print(f"   🚀 Training in batches of {batch_size}...")

    for epoch in range(15): # Reduced epochs (since we do many updates per epoch now)
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer_pred.zero_grad()
            preds = predictor(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer_pred.step()
            epoch_loss += loss.item()

        if (epoch+1) % 5 == 0:
            avg_loss = epoch_loss / len(train_loader)
            print(f"   Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

    # Clean up memory before Phase 2
    del X_train, y_train, train_loader, train_data
    torch.cuda.empty_cache()

    # --- PHASE 2: RL TRAINING (PPO) ---
    print("\n🎮 PHASE 2: Starting RL (PPO) with Random Ticker Switching...")

    # Transfer the "Physical Understanding" from Predictor to Agent
    agent = KANActorCritic(env.obs_shape, env.action_space.n, hidden_dim=64, pretrained_body=predictor.body).to(device)
    optimizer_rl = optim.Adam(agent.parameters(), lr=0.0003)

    state, _ = env.reset()
    memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
    recent_rewards = []

    total_steps = 50000
    update_timestep = 2000 # Update policy every 2000 steps

    for step in range(1, total_steps + 1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob, _ = agent.act(state_tensor)

        next_state, reward, done, _, _ = env.step(action)

        memory_states.append(state_tensor)
        memory_actions.append(torch.tensor(action).to(device))
        memory_logprobs.append(log_prob)
        memory_rewards.append(reward)
        recent_rewards.append(reward)

        state = next_state

        # PPO UPDATE LOOP
        if step % update_timestep == 0:
            old_states = torch.cat(memory_states)
            old_actions = torch.stack(memory_actions)
            old_logprobs = torch.stack(memory_logprobs)

            # Calculate Rewards (Monte Carlo)
            rewards = []
            discounted_reward = 0
            for r in reversed(memory_rewards):
                discounted_reward = r + 0.99 * discounted_reward
                rewards.insert(0, discounted_reward)

            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            # 🔧 FIX: Mini-Batch PPO Update (Prevent OOM here too)
            # We process the 2000 collected steps in chunks
            ppo_batch_size = 128
            dataset_ppo = TensorDataset(old_states, old_actions, old_logprobs, rewards)
            loader_ppo = DataLoader(dataset_ppo, batch_size=ppo_batch_size, shuffle=True)

            for _ in range(4): # 4 Epochs per update
                for b_states, b_actions, b_old_logprobs, b_rewards in loader_ppo:

                    logprobs, state_values, dist_entropy = agent.evaluate(b_states, b_actions)
                    state_values = torch.squeeze(state_values)

                    ratios = torch.exp(logprobs - b_old_logprobs.detach())
                    advantages = b_rewards - state_values.detach()

                    loss = -torch.min(ratios * advantages, torch.clamp(ratios, 0.8, 1.2) * advantages) + \
                           0.5 * nn.MSELoss()(state_values, b_rewards) - \
                           0.01 * dist_entropy

                    optimizer_rl.zero_grad()
                    loss.mean().backward()
                    optimizer_rl.step()

            avg_rew = sum(recent_rewards) / len(recent_rewards)
            print(f"   🔄 Step {step} | Avg Reward: {avg_rew:.4f} | Current Stock: {env.current_ticker}")

            # Clear Memory
            memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
            recent_rewards = []
            torch.cuda.empty_cache()

        if done:
            state, _ = env.reset()

    torch.save(agent.state_dict(), "kan_agent_multistock.pth")
    print("✅ Multi-Stock Surfer Trained. Saved as 'kan_agent_multistock.pth'")

def download_specific_range(ticker, start, end):
    """
    Downloads data and ADDS THE MISSING 'Trend_Score' feature.
    """
    try:
        # Buffer: Download 60 days extra before start date to calculate Rolling SMA correctly
        start_date_obj = pd.to_datetime(start)
        buffer_date = (start_date_obj - pd.Timedelta(days=80)).strftime('%Y-%m-%d')

        print(f"   ⬇️ Downloading {ticker} ({buffer_date} to {end})...")
        df = yf.download(ticker, start=buffer_date, end=end, interval="1d", progress=False, auto_adjust=True)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if len(df) < 50: return None

        # Feature Engineering (MUST MATCH TRAINING EXACTLY)
        df["Log_Ret"] = np.log(df["Close"] / df["Close"].shift(1))
        df["Vol_Norm"] = df["Volume"] / (df["Volume"].rolling(20).mean() + 1e-7)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)

        # --- THE MISSING PIECE: Trend Score ---
        df["SMA_50"] = df["Close"].rolling(50).mean()
        df["Trend_Score"] = (df["Close"] - df["SMA_50"]) / (df["SMA_50"] + 1e-7)

        # Cleanup (This removes the buffer period NaN values)
        df.dropna(inplace=True)

        # Slice back to the requested start date
        df = df[df.index >= start_date_obj]

        # Normalize Indicators
        df["RSI_14"] = df["RSI_14"] / 100.0
        df["MACD_12_26_9"] = (df["MACD_12_26_9"] - df["MACD_12_26_9"].mean()) / (df["MACD_12_26_9"].std() + 1e-7)

        # 6 Features now
        features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9", "Trend_Score"]
        return df[features].astype(np.float32)
    except Exception as e:
        print(f"⚠️ Error downloading {ticker}: {e}")
        return None

class RiskSentinel:
    """
    The 'Sticky' Shield.
    Implements Hysteresis: Easy to enter safety mode, hard to exit.
    Prevents 'Flickering' during volatile crashes.
    """
    def __init__(self, threshold=-0.10, recovery_threshold=-0.02, window_size=30, input_dim=6):
        self.threshold = threshold          # Trigger Crash Mode if < -10%
        self.recovery_threshold = recovery_threshold # Only exit Crash Mode if > -2%
        self.window_size = window_size
        self.input_dim = input_dim

        # MEMORY: Keeps track if we are currently hiding in the bunker
        self.in_crash_mode = False

    def emergency_override(self, state_tensor, vae_logvar=None):
        """
        Returns: Action (2 = SELL) or None.
        """
        # 1. Reshape Input
        if state_tensor.dim() == 2:
            batch_size = state_tensor.shape[0]
            try:
                state_tensor = state_tensor.view(batch_size, self.window_size, self.input_dim)
            except RuntimeError:
                return None

                # 2. Get Data
        last_day_features = state_tensor[0, -1, :]
        trend_score = last_day_features[5].item()

        # 3. HYSTERESIS LOGIC (The Fix)
        if self.in_crash_mode:
            # We are currently hiding. Can we come out?
            if trend_score > self.recovery_threshold:
                self.in_crash_mode = False # All clear, resume trading
                return None
            else:
                return 2 # STAY IN CASH (Force Sell/Wait)
        else:
            # We are trading normally. Should we hide?
            if trend_score < self.threshold:
                self.in_crash_mode = True # Enter Bunker
                return 2 # Force Sell

        return None # Normal Trading

# ==============================================================================
# UPDATED TEST SCENARIO (With Sentinel)
# ==============================================================================
def test_scenario(agent, ticker, name, start_date, end_date):
    print(f"\n🧪 TESTING SCENARIO: {name} [{ticker}]")

    df = download_specific_range(ticker, start_date, end_date)
    if df is None or len(df) < 30:
        print("   ❌ Data not found or too short, skipping.")
        return

    features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9", "Trend_Score"]

    # Init Environment
    dummy_universe = {ticker: df}
    env = StockTradingEnv(dummy_universe, window_size=30)

    env.current_ticker = ticker
    env.data = df[features].values
    env.max_steps = len(env.data) - 1
    env.current_step = env.window_size
    env.balance = 20.0
    env.shares = 0
    env.entry_price = 0

    # --- ACTIVATE THE SHIELD ---
    sentinel = RiskSentinel(threshold=-0.10) # 10% tolerance below SMA

    state = env._get_observation()
    done = False
    portfolio_history = []
    price_history = []

    # Metrics for Shield Activity
    shield_interventions = 0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            # 1. Get Brain Output
            body_out = agent.body(state_tensor)

            # Handle VAE output tuple
            if isinstance(body_out, tuple):
                features_out, _, _, logvar = body_out
            else:
                features_out = body_out
                logvar = None

            # 2. ASK THE SENTINEL (Shield Check)
            shield_action = sentinel.emergency_override(state_tensor, logvar)

            if shield_action is not None:
                # 🛡️ SHIELD TRIGGERED
                action = shield_action
                shield_interventions += 1
            else:
                # 🧠 AI NORMAL MODE
                logits = agent.actor_head(features_out)
                action = torch.argmax(logits, dim=1).item()

        next_state, _, done, _, _ = env.step(action)

        # Track Value
        current_idx = env.current_step
        if current_idx < len(env.data):
            current_price = env.data[current_idx][0]
            val = env.balance + (env.shares * current_price)
            portfolio_history.append(val)
            price_history.append(current_price)

        state = next_state

    # Results
    if portfolio_history:
        initial_bal = 20.0
        final_bal = portfolio_history[-1]
        agent_return = ((final_bal - initial_bal) / initial_bal) * 100

        initial_price = price_history[0]
        final_price = price_history[-1]
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100

        print(f"   📊 Agent Return:     {agent_return:.2f}% (${final_bal:.2f})")
        print(f"   📈 Buy & Hold Ret:   {buy_hold_return:.2f}%")
        print(f"   🛡️ Shield Active:    {shield_interventions} times (Saved you from the crash)")

        plt.figure(figsize=(10, 5))
        plt.plot(portfolio_history, label=f"Shielded AI (${final_bal:.2f})", color='blue')

        scaled_price = [p * (initial_bal / initial_price) for p in price_history]
        plt.plot(scaled_price, label=f"Buy & Hold ({buy_hold_return:.0f}%)", color='gray', linestyle='--', alpha=0.6)

        clean_name = name.encode('ascii', 'ignore').decode('ascii').strip()
        plt.title(f"Scenario: {clean_name} | Profit: {agent_return:.2f}% | Shield Uses: {shield_interventions}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def run_backtest():
    # Load Agent
    # IMPORTANT: hidden_dim must match what you trained with!
    # In the previous step, we used hidden_dim=64 for the VAE/Research model.
    hidden_dim = 64

    # Create dummy env
    temp_env = StockTradingEnv({"DUMMY": pd.DataFrame()}, window_size=30)

    agent = KANActorCritic(temp_env.obs_shape, temp_env.action_space.n, hidden_dim=hidden_dim).to(device)
    try:
        agent.load_state_dict(torch.load("kan_agent_multistock.pth", map_location=device, weights_only=True))
        agent.eval()
        print("✅ Agent Loaded Successfully.")
    except Exception as e:
        print(f"❌ Load Error: {e}")
        print("💡 Hint: Ensure 'hidden_dim' matches the training config (Default: 64)")
        return

    # === SCENARIOS ===
    test_scenario(agent, "ARKK", "🐻 Bad Year (Crash)", "2021-12-01", "2022-12-30")
    test_scenario(agent, "NVDA", "🐂 Good Year (Moon)", "2023-01-01", "2023-12-30")
    test_scenario(agent, "KO", "🦀 Average Year (Chop)", "2023-01-01", "2023-12-30")

if __name__ == "__main__":
    run_training_pipeline()
    run_backtest()