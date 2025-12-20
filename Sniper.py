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
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, latent_dim=16):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def reparameterize(self, mu, logvar):
        # 🛑 FIX 1: Only add noise during TRAINING.
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu # Deterministic (No jitter) during eval/test

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        return recon_x, mu, logvar, z

class SelfAttention(nn.Module):
    """The 'Focus' Block."""
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim])).to(device)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.bmm(weights, V), weights

class ResearchAgentBody(nn.Module):
    """
    The 'God Mode' Hybrid: VAE + KAN + GRU + Attention.
    """
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=1, seq_len=30):
        super(ResearchAgentBody, self).__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.vae_input_dim = seq_len * input_dim
        self.vae = VAE(input_dim=self.vae_input_dim, latent_dim=16)
        self.kan_layer = KANLinear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.gru = nn.GRU(hidden_dim + 16, hidden_dim, num_layers, batch_first=True)
        # 🔧 FIX 2: LayerNorm after fusion before GRU
        self.fusion_norm = nn.LayerNorm(hidden_dim + 16)
        self.attention = SelfAttention(hidden_dim)

    def forward(self, x):
        batch_size = x.shape[0]

        # VAE Path
        recon_x, mu, logvar, z_regime = self.vae(x)

        # Sequence Path
        x_seq = x.view(batch_size, self.seq_len, self.input_dim)
        x_flat = x_seq.view(-1, self.input_dim)
        kan_out = self.kan_layer(x_flat)
        kan_out = self.ln(kan_out)
        kan_out = F.silu(kan_out)
        kan_seq = kan_out.view(batch_size, self.seq_len, -1)

        # Fusion
        z_expanded = z_regime.unsqueeze(1).repeat(1, self.seq_len, 1)
        gru_input = torch.cat([kan_seq, z_expanded], dim=2)

        # Apply LayerNorm to the fused input (FIX 2)
        gru_input = self.fusion_norm(gru_input)

        gru_out, _ = self.gru(gru_input)
        attn_out, _ = self.attention(gru_out)
        final_feature = torch.sum(attn_out, dim=1)

        return final_feature, recon_x, mu, logvar

class KANActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, num_experts=3, hidden_dim=64):
        super(KANActorCritic, self).__init__()
        self.num_experts = num_experts

        # Hardcoded for safety (matches your env)
        WINDOW_SIZE = 30
        SINGLE_STEP_FEATURE_DIM = 6

        # 1. THE GATE
        self.gate = nn.Sequential(
            nn.Linear(SINGLE_STEP_FEATURE_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=1)
        )

        # 2. THE EXPERTS
        self.experts = nn.ModuleList([
            ResearchAgentBody(
                input_dim=SINGLE_STEP_FEATURE_DIM,
                hidden_dim=hidden_dim,
                seq_len=WINDOW_SIZE
            )
            for _ in range(num_experts)
        ])

        # 3. OUTPUT HEADS
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # Handle shape (Batch, 180) -> (Batch, 30, 6)
        if x.dim() == 2:
            batch_size = x.shape[0]
            x_seq = x.view(batch_size, 30, 6)
        else:
            x_seq = x
            x = x.reshape(x.shape[0], -1)

            # Gate Decision on LAST candle
        current_market_state = x_seq[:, -1, :]
        gate_weights = self.gate(current_market_state)

        # Expert Opinions
        expert_outputs = []
        for expert in self.experts:
            features, _, _, _ = expert(x)
            expert_outputs.append(features)

        expert_outputs = torch.stack(expert_outputs, dim=1)
        mixed_features = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)

        logits = self.actor_head(mixed_features)
        value = self.critic_head(mixed_features)
        return logits, value, gate_weights

    def act(self, x, deterministic=False):
        logits, _, gate_weights = self.forward(x)

        if deterministic:
            # 🛑 FIX 2: Pick the BEST action, don't roll dice
            action = torch.argmax(logits, dim=1)
            return action.item(), None, gate_weights
        else:
            # Training Mode (Explore)
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action), gate_weights

    def evaluate(self, state, action):
        logits, value, gate_weights = self.forward(state)
        dist = Categorical(logits=logits)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        # Note: We return None for recon/mu/logvar here to simplify the loop
        # The VAE loss is handled inside the experts if needed, or ignored for now
        return action_logprobs, value, dist_entropy, None, None, None, gate_weights

class KANPredictor(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(KANPredictor, self).__init__()
        self.body = ResearchAgentBody(input_dim=6, hidden_dim=hidden_dim, seq_len=30)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features, _, _, _ = self.body(x)
        return self.head(features)


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



class Personality:
    def __init__(self, name, risk_penalty=0.1, profit_multiplier=1.0, bored_penalty=0.05, stop_loss_penalty=2.0):
        self.name = name
        self.risk_penalty = risk_penalty       # Penalty for volatility/drawdown
        self.profit_multiplier = profit_multiplier # Scaling factor for gains
        self.bored_penalty = bored_penalty     # Penalty for doing nothing
        self.stop_loss_penalty = stop_loss_penalty # Punishment for hitting stop loss

# Define the Squad
TEAM_PROFILES = [
    Personality("🐂 Aggressive", risk_penalty=0.0, profit_multiplier=2.0, stop_loss_penalty=1.0),
    Personality("🐻 Conservative", risk_penalty=0.5, profit_multiplier=1.0, stop_loss_penalty=5.0),
    Personality("🏄 Standard", risk_penalty=0.1, profit_multiplier=1.0, stop_loss_penalty=2.0),
]

# ==============================================================================
# 2. MULTI-AGENT ENVIRONMENT
# ==============================================================================
class MultiPersonaEnv(gym.Env):
    def __init__(self, universe_dict, window_size=30, personality=TEAM_PROFILES[2]):
        super(MultiPersonaEnv, self).__init__()
        self.universe = universe_dict
        self.window_size = window_size
        self.personality = personality
        self.tickers = list(self.universe.keys())
        self.action_space = spaces.Discrete(3) # 0=Hold, 1=Buy, 2=Sell
        self.obs_shape = window_size * 6

    def set_personality(self, personality):
        self.personality = personality

    def reset(self, seed=None):
        # 1. Randomize Stock and Time
        self.current_ticker = random.choice(self.tickers)
        self.data = self.universe[self.current_ticker].values
        self.max_steps = len(self.data) - 1

        # Crash Specialist: Force start in high volatility if applicable
        if self.personality.name == "🐻 Conservative" and random.random() < 0.3:
            # (Optional: Logic to find high vol periods would go here)
            pass

        # Ensure we have enough data
        if self.max_steps > self.window_size + 200:
            self.current_step = random.randint(self.window_size, self.max_steps - 200)
        else:
            self.current_step = self.window_size

        self.balance = 1000.0 # Standardize starting cash
        self.shares = 0
        self.entry_price = 0
        self.initial_balance = 1000.0

        return self._get_observation(), {}

    def _get_observation(self):
        window = self.data[self.current_step - self.window_size : self.current_step]
        # Flattening is handled by the Agent's reshaping logic now, but we send flat
        return window.flatten().astype(np.float32)

    def step(self, action):
        current_price = self.data[self.current_step][0]
        reward = 0
        done = False

        # Track previous portfolio value for basic PnL calculation
        prev_val = self.balance + (self.shares * current_price)
        reward = np.clip(reward, -2.0, 2.0)
        # ============================================================
        # 1. EXECUTION LOGIC (The Missing Piece)
        # ============================================================
        if action == 1: # BUY
            if self.shares == 0:
                self.shares = self.balance / current_price
                self.balance = 0.0
                self.entry_price = current_price
                reward -= 0.001 # Tiny friction penalty

        elif action == 2: # SELL
            if self.shares > 0:
                self.balance = self.shares * current_price
                self.shares = 0
                self.entry_price = 0
                # Reward is calculated in the Strategy section below

        # ============================================================
        # 2. DYNAMIC REWARD STRATEGY
        # ============================================================

        # Calculate raw percentage return of the trade (if holding or just sold)
        raw_profit_pct = 0.0
        if self.shares > 0:
            raw_profit_pct = (current_price - self.entry_price) / self.entry_price
        elif action == 2 and self.balance > 0: # Just Sold
            # We need to calculate what the profit WAS before we zeroed shares
            # (Approximation: Use prev step price vs current price logic)
            # Better: Track realized PnL.
            # For RL stability, we use the step-to-step portfolio change often
            pass

        # --- REFINED STRATEGIES ---

        # 🐂 AGGRESSIVE: "The Gambler"
        # Rewards Big Realized Gains. Ignores Drawdowns (mostly).
        if self.personality.name == "🐂 Aggressive":
            if action == 2: # Selling
                curr_val = self.balance
                profit = (curr_val - 1000.0) / 1000.0
                if profit > 0:
                    reward = profit * 100.0 * 2.0 # Huge reward for total profit
                else:
                    reward = profit * 100.0 * 0.5 # Small penalty for loss

        # 🐻 CONSERVATIVE: "The Risk Manager"
        # Punishes drawdown. Rewards preservation.
        elif self.personality.name == "🐻 Conservative":
            if self.shares > 0:
                if raw_profit_pct < -0.02: # 2% Drawdown
                    reward -= 1.0 # High Anxiety
                if raw_profit_pct < -0.05: # 5% Drawdown
                    reward -= 10.0 # Panic

            # Reward for simply NOT losing money
            if self.shares == 0:
                reward += 0.01

                # 🏄 STANDARD: "The Trend Follower" (Balanced)
        # Mixes PnL with Technical Alignment
        elif self.personality.name == "🏄 Standard":
            # Feature 5 is Trend_Score (Price vs SMA50)
            trend_score = self.data[self.current_step][5]

            # Rule: Be long if Trend is Positive
            if trend_score > 0 and self.shares > 0:
                reward += 0.1 # Good boy
            elif trend_score < 0 and self.shares > 0:
                reward -= 0.1 # Why are you holding in a downtrend?

            # Also reward Realized Profit (so it learns to take profit eventually)
            if action == 2:
                curr_val = self.balance
                profit = (curr_val - 1000.0) / 1000.0
                reward += profit * 50.0

        # ============================================================

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, False, {}

# ==============================================================================
# 3. SHARED EXPERIENCE TRAINER (The "Hive Mind")
# ==============================================================================
def train_hive_mind(universe):
    # 1. Initialize MoE Agent
    global_agent = KANActorCritic(30*6, 3, hidden_dim=64).to(device)
    optimizer = optim.Adam(global_agent.parameters(), lr=0.0001) # 🔽 Lowered LR to 0.0001 for stability

    # The Environment
    env = MultiPersonaEnv(universe)

    # Configuration
    total_updates = 200
    steps_per_agent = 500

    print("🧠 Initializing Hive Mind (MoE) Training...")

    for update in range(1, total_updates + 1):
        memory_states = []
        memory_actions = []
        memory_logprobs = []
        memory_rewards = []

        debug_returns = [] # To check if numbers are too big

        # --- STEP 1: ROUND ROBIN COLLECTION ---
        for persona in TEAM_PROFILES:
            env.set_personality(persona)
            state, _ = env.reset()

            agent_rewards_raw = []

            # 1. Collect Trajectory
            for _ in range(steps_per_agent):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                with torch.no_grad():
                    action, log_prob, _ = global_agent.act(state_tensor)

                next_state, reward, done, _, _ = env.step(action)

                memory_states.append(state_tensor)
                memory_actions.append(torch.tensor(action).to(device))
                memory_logprobs.append(log_prob)
                agent_rewards_raw.append(reward)

                state = next_state
                if done: state, _ = env.reset()

            # 2. CALCULATE RETURNS
            returns = []
            R = 0
            gamma = 0.95 # 🔽 Lowered gamma (0.99 -> 0.95) to reduce accumulation size
            for r in reversed(agent_rewards_raw):
                R = r + gamma * R
                returns.insert(0, R)

            # Debug: Track max return
            debug_returns.append(max(returns))

            returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
            memory_rewards.extend(returns_tensor.tolist())

        # --- STEP 2: GLOBAL BRAIN UPDATE (PPO) ---
        old_states = torch.cat(memory_states)
        old_actions = torch.stack(memory_actions)
        old_logprobs = torch.stack(memory_logprobs)
        old_rewards = torch.tensor(memory_rewards).to(device)

        # 🛑 SAFETY: Normalize Targets Batch-Wise
        # This keeps the Critic from seeing "100.0" and exploding.
        # It sees "1.5 standard deviations" instead.
        rewards_mean = old_rewards.mean()
        rewards_std = old_rewards.std() + 1e-5
        old_rewards = (old_rewards - rewards_mean) / rewards_std

        dataset = TensorDataset(old_states, old_actions, old_logprobs, old_rewards)
        loader = DataLoader(dataset, batch_size=64, shuffle=True) # Smaller batch size

        total_loss = 0
        for _ in range(4): # Epochs
            for b_states, b_actions, b_old_logprobs, b_rewards in loader:

                # Evaluate
                logprobs, state_values, dist_entropy, _, _, _, gate_weights = global_agent.evaluate(b_states, b_actions)

                # Gate Entropy
                gate_entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=1).mean()

                # Advantages
                advantages = b_rewards - state_values.squeeze()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = advantages.detach()

                # PPO Ratio
                ratios = torch.exp(logprobs - b_old_logprobs.detach())
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 🛑 CRITICAL FIX: VALUE FUNCTION CLIPPING
                # This prevents the critic from making massive jumps
                v_pred = state_values.squeeze()
                v_clip = b_rewards + torch.clamp(v_pred - b_rewards, -0.2, 0.2)
                v_loss1 = (v_pred - b_rewards) ** 2
                v_loss2 = (v_clip - b_rewards) ** 2
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                # Total Loss
                loss = policy_loss + (0.5 * value_loss) - (0.01 * dist_entropy.mean()) + (0.01 * gate_entropy)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(global_agent.parameters(), max_norm=0.5)
                optimizer.step()

                total_loss += loss.item()

        if update % 10 == 0:
            avg_return_debug = sum(debug_returns) / len(debug_returns)
            print(f"   🔄 Upd {update} | Loss: {total_loss:.4f} | Max Return Raw: {avg_return_debug:.2f}")

    return global_agent
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
        reward = 0
        done = False

        # [Execution Logic same as before...]
        if action == 1 and self.shares == 0: # BUY
            self.shares = self.balance / current_price
            self.balance = 0.0
            self.entry_price = current_price
            reward -= 0.001
        elif action == 2 and self.shares > 0: # SELL
            self.balance = self.shares * current_price
            self.shares = 0
            self.entry_price = 0

        # --- STRATEGIES ---

        # 1. AGGRESSIVE (Scaled Down)
        if self.personality.name == "🐂 Aggressive":
            if action == 2:
                curr_val = self.balance
                profit = (curr_val - self.initial_balance) / self.initial_balance
                if profit > 0:
                    # WAS: profit * 200.0 -> NOW: profit * 20.0
                    reward = profit * 2.0
                else:
                    reward = profit * 0.5

                    # 2. CONSERVATIVE (Scaled Down)
        elif self.personality.name == "🐻 Conservative":
            if self.shares > 0:
                # WAS: -1.0 / -10.0 -> NOW: -0.1 / -1.0
                raw_profit_pct = (current_price - self.entry_price) / self.entry_price
                if raw_profit_pct < -0.02: reward -= 0.1
                if raw_profit_pct < -0.05: reward -= 1.0
            if self.shares == 0:
                reward += 0.001 # WAS: 0.01

        # 3. STANDARD (Scaled Down)
        elif self.personality.name == "🏄 Standard":
            trend_score = self.data[self.current_step][5]
            # WAS: 0.1 -> NOW: 0.01
            if trend_score > 0 and self.shares > 0: reward += 0.01
            elif trend_score < 0 and self.shares > 0: reward -= 0.01

            if action == 2:
                curr_val = self.balance
                profit = (curr_val - self.initial_balance) / self.initial_balance
                reward += profit * 5.0 # WAS: 50.0

        # ============================================================
        # 🛑 CRITICAL FIX: TIGHTER CLIPPING
        # Clip to [-1, 1]. This keeps gradients stable.
        # ============================================================
        reward = np.clip(reward, -10.0, 10.0)

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

    total_steps = 100000
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

                    # Unpack 6 variables from evaluate()
                    logprobs, state_values, dist_entropy, recon, mu, logvar = agent.evaluate(b_states, b_actions)
                    state_values = torch.squeeze(state_values)

                    # PPO LOSS
                    ratios = torch.exp(logprobs - b_old_logprobs.detach())
                    advantages = b_rewards - state_values.detach()
                    policy_loss = -torch.min(ratios * advantages, torch.clamp(ratios, 0.8, 1.2) * advantages)
                    value_loss = 0.5 * nn.MSELoss()(state_values, b_rewards)

                    # VAE LOSS
                    recon_loss = nn.MSELoss()(recon, b_states)
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

                    # 🔧 CRITICAL FIX: Add .mean() to dist_entropy
                    loss = policy_loss.mean() + value_loss + (-0.01 * dist_entropy.mean()) + (0.01 * (recon_loss + kl_loss))
                    optimizer_rl.zero_grad()
                    loss.backward() # This will now work because 'loss' is a scalar
                    optimizer_rl.step()

            avg_rew = sum(recent_rewards) / len(recent_rewards)
            print(f"   🔄 Step {step} | Avg Reward: {avg_rew:.4f} | Current Stock: {env.current_ticker}")

            # Clear Memory
            memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
            recent_rewards = []
            torch.cuda.empty_cache()
        torch.save(agent.state_dict(), "kan_agent_multistock2.pth")
        if done:
            state, _ = env.reset()

    torch.save(agent.state_dict(), "kan_agent_multistock2.pth")
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
def test_hive_mind(ticker, model_path="kan_hive_mind.pth"):
    print(f"\n⚔️ STARTING HIVE MIND BACKTEST ON {ticker}")

    # 1. SETUP ENVIRONMENT
    df = download_and_process(ticker)
    if df is None: return

    # FIX: Use MultiPersonaEnv, NOT StockTradingEnv
    # We set the personality to "Standard" for the fair test (or "Aggressive" if you prefer)
    test_universe = {ticker: df}
    env = MultiPersonaEnv(test_universe, window_size=30, personality=TEAM_PROFILES[2]) # 2 = Standard

    # 2. INITIALIZE THE NEW MOE AGENT
    agent = KANActorCritic(obs_dim=30*6, action_dim=3, hidden_dim=64).to(device)

    try:
        # Load weights
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()
        print(f"✅ Brain Loaded.")
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return

    # 3. RUN SIMULATION
    state, _ = env.reset()
    done = False

    portfolio_vals = []
    gate_usage = []

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

        with torch.no_grad():
            action, _, gate_weights = agent.act(state_tensor, deterministic=True)
            gate_usage.append(gate_weights.cpu().numpy()[0])

        state, _, done, _, _ = env.step(action)

        # Track Value
        # Note: MultiPersonaEnv uses self.balance and self.shares
        current_price = env.data[env.current_step][0]
        val = env.balance + (env.shares * current_price)
        portfolio_vals.append(val)

    # 4. REPORT CARD
    initial = portfolio_vals[0]
    final = portfolio_vals[-1]
    roi = ((final - initial) / initial) * 100

    # Calculate Buy & Hold from the same start point (index 30)
    start_price = df.iloc[30]['Close']
    end_price = df.iloc[-1]['Close']
    buy_hold = ((end_price - start_price) / start_price) * 100

    avg_gate = np.mean(gate_usage, axis=0)

    print(f"   💰 Final Balance: ${final:.2f}")
    print(f"   📈 Agent ROI:     {roi:.2f}%")
    print(f"   📊 Buy & Hold:    {buy_hold:.2f}%")
    print(f"   🧠 BRAIN SCAN:    [Bull: {avg_gate[0]:.2f} | Bear: {avg_gate[1]:.2f} | Neutral: {avg_gate[2]:.2f}]")

    if roi > buy_hold:
        print("   🏆 RESULT: BEAT THE MARKET")
    else:
        print("   💀 RESULT: LOST TO MARKET")
def fine_tune_on_single_stock(ticker, pretrained_path="kan_hive_mind.pth"):
    print(f"\n🎯 STARTING SPECIALIST TRAINING FOR: {ticker}")

    # 1. DATA: Download ONLY the target stock
    # We get more history if possible to give it a rich experience
    df = download_and_process(ticker, period="10y")
    if df is None: return

    universe = {ticker: df} # The universe is now just ONE planet

    # 2. LOAD THE HIVE MIND (The Generalist)
    # We load the weights we already trained
    agent = KANActorCritic(obs_dim=30*6, action_dim=3, hidden_dim=64).to(device)

    try:
        agent.load_state_dict(torch.load(pretrained_path, map_location=device))
        print("✅ Generalist Brain Loaded. Beginning Specialization...")
    except Exception as e:
        print(f"❌ Error loading brain: {e}")
        return

    # 3. SETUP OPTIMIZER (CRITICAL STEP)
    # We use a VERY LOW Learning Rate.
    # Why? We don't want to break what it already learned. We just want to nudge it.
    # Normal LR: 0.0003 -> Fine-Tune LR: 0.00001
    optimizer = optim.Adam(agent.parameters(), lr=0.00001)

    # 4. TRAINING LOOP (Short & Focused)
    # We don't need 200 updates. 20-50 is enough to adapt.
    total_updates = 30
    steps_per_agent = 500

    env = MultiPersonaEnv(universe) # It will now only pick 'ticker'

    for update in range(1, total_updates + 1):
        memory_states = []
        memory_actions = []
        memory_logprobs = []
        memory_rewards = []

        # We still rotate personalities so it learns how to handle
        # THIS specific stock in Bull, Bear, and Chop conditions.
        for persona in TEAM_PROFILES:
            env.set_personality(persona)
            state, _ = env.reset()

            agent_rewards_raw = []

            for _ in range(steps_per_agent):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

                # Act
                action, log_prob, _ = agent.act(state_tensor)
                next_state, reward, done, _, _ = env.step(action)

                memory_states.append(state_tensor)
                memory_actions.append(torch.tensor(action).to(device))
                memory_logprobs.append(log_prob)
                agent_rewards_raw.append(reward)
                state = next_state
                if done: state, _ = env.reset()

            # Returns Calculation
            returns = []
            R = 0
            gamma = 0.95
            for r in reversed(agent_rewards_raw):
                R = r + gamma * R
                returns.insert(0, R)

            returns_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
            memory_rewards.extend(returns_tensor.tolist())

        # PPO UPDATE
        old_states = torch.cat(memory_states)
        old_actions = torch.stack(memory_actions)
        old_logprobs = torch.stack(memory_logprobs)
        old_rewards = torch.tensor(memory_rewards).to(device)

        # Normalize Targets
        rewards_mean = old_rewards.mean()
        rewards_std = old_rewards.std() + 1e-5
        old_rewards = (old_rewards - rewards_mean) / rewards_std

        dataset = TensorDataset(old_states, old_actions, old_logprobs, old_rewards)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        total_loss = 0
        for _ in range(4):
            for b_states, b_actions, b_old_logprobs, b_rewards in loader:
                logprobs, state_values, dist_entropy, _, _, _, gate_weights = agent.evaluate(b_states, b_actions)
                gate_entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=1).mean()

                advantages = b_rewards - state_values.squeeze()
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                advantages = advantages.detach()

                ratios = torch.exp(logprobs - b_old_logprobs.detach())
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value Clipping
                v_pred = state_values.squeeze()
                v_clip = b_rewards + torch.clamp(v_pred - b_rewards, -0.2, 0.2)
                value_loss = 0.5 * torch.max((v_pred - b_rewards)**2, (v_clip - b_rewards)**2).mean()

                loss = policy_loss + (0.5 * value_loss) - (0.01 * dist_entropy.mean()) + (0.01 * gate_entropy)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                optimizer.step()
                total_loss += loss.item()

        if update % 5 == 0:
            print(f"   🔨 Tuning Update {update}/{total_updates} | Loss: {total_loss:.4f}")

    # 5. SAVE THE SPECIALIST
    filename = f"kan_agent_{ticker}_SPECIALIST.pth"
    torch.save(agent.state_dict(), filename)
    print(f"✅ Specialist Model Saved: {filename}")
    return agent


if __name__ == "__main__":
    # universe = load_universe()
    # final_brain = train_hive_mind(universe)
    # torch.save(final_brain.state_dict(), "kan_hive_mind.pth")
    # --- RUN TESTS ---
    # Test 1: The Bull Market (Does it know how to ride trends?)
    # test_hive_mind("NVDA")

    # Test 2: The Crash (Did it learn from the Conservative Agent?)
    # test_hive_mind("ARKK")

    # specialist_agent = fine_tune_on_single_stock("TSLA")

    # Now TEST it against the Generalist
    print("\n⚔️ COMPARING GENERALIST VS SPECIALIST")
    test_hive_mind("TSLA", model_path="kan_hive_mind.pth")            # The General Doctor
    test_hive_mind("TSLA", model_path="kan_agent_TSLA_SPECIALIST.pth") # The Specialist