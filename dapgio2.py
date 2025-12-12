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
import pandas_ta as ta
import random
import warnings
from torch.utils.data import TensorDataset, DataLoader
import multiprocessing

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

WINDOW_SIZE = 30
SINGLE_STEP_FEATURE_DIM = 8
GAMMA = 0.99
BETA_KLD = 0.001

# Tickers to train on (The "Academy")
TICKERS = ["AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "AMD"]

# Define Agent Profiles
TEAM_PROFILES = {
    "Aggressive": {"risk_aversion": 0.01, "reward_multiplier": 2.0},
    "Conservative": {"risk_aversion": 0.5, "reward_multiplier": 1.0},
    "Standard": {"risk_aversion": 0.1, "reward_multiplier": 1.2},
}
NUM_EXPERTS = len(TEAM_PROFILES)
EXPERT_NAMES = list(TEAM_PROFILES.keys())


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)


# ==============================================================================
# 1. ARCHITECTURE (Must match your saved models)
# ==============================================================================
class ResearchMLPLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x): return self.dropout(self.activation(self.linear(x)))


class VAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder_body = nn.Sequential(ResearchMLPLayer(input_dim * WINDOW_SIZE, 128), ResearchMLPLayer(128, 64))
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        h = self.encoder_body(x)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.decoder_body = nn.Sequential(ResearchMLPLayer(latent_dim, 64), ResearchMLPLayer(64, 128))
        self.fc_out = nn.Linear(128, input_dim * WINDOW_SIZE)

    def forward(self, z):
        h = self.decoder_body(z)
        return self.fc_out(h).view(-1, WINDOW_SIZE, SINGLE_STEP_FEATURE_DIM)


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=8):
        super().__init__()
        self.encoder = VAEEncoder(input_dim, latent_dim)
        self.decoder = VAEDecoder(input_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        logvar = torch.clamp(logvar, min=-10.0, max=2.0)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z


class ResearchAgentBody(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.h0_linear = nn.Linear(latent_dim, hidden_dim)

    def forward(self, x, z_regime):
        batch_size = x.size(0)
        h0 = torch.tanh(self.h0_linear(z_regime)).view(1, batch_size, self.hidden_dim)
        gru_out, _ = self.gru(x, h0)
        return gru_out[:, -1, :]


class KANActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim=64, vae_latent_dim=8):
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.vae = VariationalAutoencoder(SINGLE_STEP_FEATURE_DIM, vae_latent_dim)
        self.expert_body = ResearchAgentBody(SINGLE_STEP_FEATURE_DIM, hidden_dim, vae_latent_dim)
        self.expert_critics = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.num_experts)])
        self.expert_actors = nn.ModuleList([nn.Linear(hidden_dim, action_dim) for _ in range(self.num_experts)])
        self.gate_network = nn.Sequential(nn.Linear(SINGLE_STEP_FEATURE_DIM, 32), nn.GELU(),
                                          nn.Linear(32, self.num_experts))

    def forward(self, x):
        recon_x, mu, logvar, z_regime = self.vae(x)
        gate_weights = F.softmax(self.gate_network(x[:, -1, :]), dim=1)
        body_out = self.expert_body(x, z_regime)
        weighted_logits = sum(
            gate_weights[:, i:i + 1] * self.expert_actors[i](body_out) for i in range(self.num_experts))
        weighted_value = sum(
            gate_weights[:, i:i + 1] * self.expert_critics[i](body_out) for i in range(self.num_experts))
        return weighted_logits, weighted_value.squeeze(1), gate_weights

    def evaluate(self, x):
        recon_x, mu, logvar, z_regime = self.vae(x)
        gate_weights = F.softmax(self.gate_network(x[:, -1, :]), dim=1)
        body_out = self.expert_body(x, z_regime)
        weighted_logits = sum(
            gate_weights[:, i:i + 1] * self.expert_actors[i](body_out) for i in range(self.num_experts))
        weighted_value = sum(
            gate_weights[:, i:i + 1] * self.expert_critics[i](body_out) for i in range(self.num_experts))
        dist = Categorical(logits=weighted_logits)
        return dist.log_prob, weighted_value.squeeze(1), dist.entropy(), gate_weights, recon_x, mu, logvar

    def act(self, x, deterministic=False):
        logits, _, gate_weights = self.forward(x)
        if deterministic:
            action = torch.argmax(logits, dim=1)
            return action.item(), None, gate_weights
        else:
            dist = Categorical(logits=logits)
            action = dist.sample()
            return action.item(), dist.log_prob(action), gate_weights


# ==============================================================================
# 2. DATA LOADING (BATCH DOWNLOAD)
# ==============================================================================
def download_multi_stock(tickers, start_date='2015-01-01', end_date=None):
    if end_date is None: end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

    data_map = {}
    print(f"📥 Batch Downloading {len(tickers)} stocks...")

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)

            if len(data) < 200: continue

            # Feature Engineering
            data['Close_Norm'] = (data['Close'] / data['Close'].shift(1) - 1)
            data['Volume_Norm'] = (data['Volume'] - data['Volume'].rolling(20).mean()) / (
                        data['Volume'].rolling(20).std() + 1e-8)
            data['RSI'] = ta.rsi(data['Close'], length=14)
            data['MACD'] = ta.macd(data['Close'])['MACDh_12_26_9'].fillna(0)
            data['TR'] = ta.true_range(data['High'], data['Low'], data['Close'].shift(1))
            data['ATR'] = ta.atr(data['High'], data['Low'], data['Close'], length=14)
            data['ATR_Norm'] = data['ATR'] / (data['Close'] + 1e-8)
            data['DayOfWeek'] = data.index.dayofweek / 6.0
            data['DayOfMonth'] = data.index.day / 31.0

            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.dropna(inplace=True)

            feature_cols = ['Close_Norm', 'Volume_Norm', 'RSI', 'MACD', 'TR', 'ATR_Norm', 'DayOfWeek', 'DayOfMonth']

            feats = data[feature_cols].values.astype(np.float32)
            prices = data['Close'].values.astype(np.float32)
            dates = data.index

            data_map[ticker] = {'dates': dates, 'features': feats, 'prices': prices}
            print(f"   ✅ {ticker}: {len(feats)} rows")

        except Exception as e:
            print(f"   ❌ {ticker} Failed: {e}")

    return data_map


# ==============================================================================
# 3. MULTI-STOCK ENVIRONMENT
# ==============================================================================
class MultiStockEnv(gym.Env):
    def __init__(self, data_map, window_size, is_validation=False):
        super().__init__()
        self.data_map = data_map
        self.tickers = list(data_map.keys())
        self.window_size = window_size
        self.is_validation = is_validation

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 8), dtype=np.float32)

        self.current_ticker = None
        self._load_random_stock()

    def _load_random_stock(self):
        self.current_ticker = random.choice(self.tickers)
        self.features = self.data_map[self.current_ticker]['features']
        self.prices = self.data_map[self.current_ticker]['prices']
        self.max_step = len(self.prices) - 1

    def reset(self, seed=None):
        self._load_random_stock()
        # Random start point
        self.current_step = random.randint(self.window_size, self.max_step - 50)
        self.balance = 10000.0
        self.shares = 0
        self.buy_price = 0.0
        return self.features[self.current_step - self.window_size: self.current_step], {}

    def step(self, action, expert_profile=None):
        current_price = self.prices[self.current_step]
        self.current_step += 1
        next_price = self.prices[self.current_step]
        done = self.current_step >= self.max_step

        reward = 0.0
        cost = 0.0005  # 0.05%

        # Action Logic
        if action == 1 and self.shares == 0:  # Buy
            self.shares = self.balance / current_price
            self.balance = 0.0
            self.buy_price = current_price
            reward -= cost
        elif action == 2 and self.shares > 0:  # Sell
            self.balance = self.shares * current_price * (1 - cost)
            self.shares = 0
            self.buy_price = 0.0
            reward -= cost

        # Reward Logic
        if self.shares > 0:
            change = (next_price - current_price) / current_price
            reward += change * 50.0  # Base reward

            # Expert Profile adjustments (Only during training)
            if expert_profile:
                reward *= expert_profile['reward_multiplier']
                # Drawdown penalty for conservative
                unrealized_pnl = (next_price - self.buy_price) / self.buy_price
                if unrealized_pnl < -0.05:
                    reward -= 0.5 * expert_profile['risk_aversion']
        else:
            reward -= 0.001  # Opportunity cost

        # For validation, we might just loop until done without random resets
        # But this Env structure is designed for random sampling training

        return self.features[self.current_step - self.window_size: self.current_step], reward, done, False, {}


# ==============================================================================
# 4. TRAINING LOOP (MULTI-STOCK)
# ==============================================================================
def train_multi_stock(agent, train_env, val_env, optimizer, episodes=300, steps_per_ep=500):
    agent.train()
    best_val_return = -9999.0

    print("\n🧠 Hive Mind Training Initiated...")

    for episode in range(1, episodes + 1):
        # --- TRAIN ---
        agent.train()
        obs, _ = train_env.reset()

        memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []

        for _ in range(steps_per_ep):
            state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action_int, log_prob, gate_weights = agent.act(state_t, deterministic=False)

            # Select Expert
            expert_idx = gate_weights.argmax().item()
            profile = TEAM_PROFILES[EXPERT_NAMES[expert_idx]]

            next_obs, reward, done, _, _ = train_env.step(action_int, profile)

            memory_states.append(state_t)
            memory_actions.append(torch.tensor(action_int, dtype=torch.long).to(device))
            memory_logprobs.append(log_prob)
            memory_rewards.append(reward)
            obs = next_obs

            if done: obs, _ = train_env.reset()

        # Update
        old_states = torch.cat(memory_states)
        old_actions = torch.stack(memory_actions)
        old_logprobs = torch.stack(memory_logprobs).detach()

        R = 0
        returns = []
        for r in reversed(memory_rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
        old_rewards = torch.tensor(returns, dtype=torch.float32).to(device)
        old_rewards = (old_rewards - old_rewards.mean()) / (old_rewards.std() + 1e-5)

        dataset = TensorDataset(old_states, old_actions, old_logprobs, old_rewards)
        loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)  # 0 workers for safety

        loss_val = 0
        for _ in range(2):
            for s, a, lp, r in loader:
                new_lp_fn, val, ent, _, _, _, _ = agent.evaluate(s)
                new_lp = new_lp_fn(a)
                adv = r - val.squeeze()
                ratio = torch.exp(new_lp - lp)
                loss = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv).mean() + 0.5 * F.mse_loss(
                    val.squeeze(), r) - 0.01 * ent.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val = loss.item()

        # --- VALIDATE (Every 5 Eps) ---
        if episode % 5 == 0:
            agent.eval()
            total_val_return = 0
            # Test on 3 random stocks from Val set
            for _ in range(3):
                val_obs, _ = val_env.reset()
                val_done = False
                val_bal = 10000.0
                val_shares = 0

                # Run for 200 steps or until done
                for _ in range(200):
                    state_t = torch.tensor(val_obs, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action, _, _ = agent.act(state_t, deterministic=True)

                    val_obs, _, val_done, _, _ = val_env.step(action)
                    if val_done: break

                    # Track dummy pnl for scoring
                    price = val_env.prices[val_env.current_step]
                    curr_val = val_env.balance + (val_env.shares * price)

                final_val = val_env.balance + (val_env.shares * val_env.prices[val_env.current_step])
                total_val_return += ((final_val - 10000.0) / 10000.0) * 100

            avg_val_return = total_val_return / 3
            print(f"   📝 Ep {episode} | Loss: {loss_val:.4f} | Val Return: {avg_val_return:.2f}%")

            if avg_val_return > best_val_return:
                best_val_return = avg_val_return
                torch.save(agent.state_dict(), "refactored_moe_agent.pth")
                print(f"      🏆 New Generalist Saved! ({avg_val_return:.2f}%)")


if __name__ == "__main__":
    # 1. Download All
    full_data = download_multi_stock(TICKERS)

    # 2. Split into Train (2015-2023) and Val (2024+)
    train_map = {}
    val_map = {}

    split_date = "2024-01-01"

    for ticker, data in full_data.items():
        dates = data['dates']
        feats = data['features']
        prices = data['prices']

        # Boolean Mask
        mask_train = dates < split_date
        mask_val = dates >= split_date

        # Only add if enough data
        if np.sum(mask_train) > 200:
            train_map[ticker] = {
                'features': feats[mask_train],
                'prices': prices[mask_train]
            }

        if np.sum(mask_val) > 50:
            val_map[ticker] = {
                'features': feats[mask_val],
                'prices': prices[mask_val]
            }

    print(f"   📊 Train Universe: {len(train_map)} stocks")
    print(f"   📊 Val Universe:   {len(val_map)} stocks")

    # 3. Setup
    train_env = MultiStockEnv(train_map, WINDOW_SIZE, is_validation=False)
    val_env = MultiStockEnv(val_map, WINDOW_SIZE, is_validation=True)

    agent = KANActorCritic((WINDOW_SIZE, SINGLE_STEP_FEATURE_DIM), 3, hidden_dim=64).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=5e-5)

    # 4. Train
    train_multi_stock(agent, train_env, val_env, optimizer)