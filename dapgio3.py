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

WINDOW_SIZE = 31
SINGLE_STEP_FEATURE_DIM = 8
GAMMA = 0.99
BETA_KLD = 0.001

# Tickers to train on (The "Academy")
TICKERS = [
    # --- BIG TECH & GROWTH (Momentum) ---
    "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "TSLA", "META", "AMD", "NFLX", "CRM",

    # --- DEFENSIVE & VALUE (Mean Reversion / Stability) ---
    "JNJ", "PG", "KO", "MCD", "WMT", "COST", "PEP", "LLY", "UNH",

    # --- FINANCIALS & CYCLICALS (Economic Cycles) ---
    "JPM", "BAC", "V", "MA", "CAT", "DE", "XOM", "CVX", "BA",

    # --- INDICES & ETFS (Market Structure) ---
    "SPY", "QQQ", "IWM", "EEM", "TLT", "LQD", "GLD", "SLV", "USO", "UNG",

    # --- CRYPTO (Extreme Volatility) ---
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD", "ADA-USD"
]

# Define Agent Profiles
TEAM_PROFILES = {
    "Aggressive":      {"risk_aversion": 0.01, "reward_multiplier": 2.0},
    "Conservative":    {"risk_aversion": 0.5,  "reward_multiplier": 1.0},
    "Standard":        {"risk_aversion": 0.1,  "reward_multiplier": 1.2},
    "ShortSpecialist": {"risk_aversion": 0.05, "reward_multiplier": 1.5},
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


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(in_channels=64, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = x.permute(0, 2, 1) # (B, F, W)
        features = self.cnn(x)
        global_context = self.global_pool(features).squeeze(-1)
        return features.permute(0, 2, 1), global_context

class ResearchAgentBody(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        cnn_feats, context = self.feature_extractor(x)
        gru_out, _ = self.gru(cnn_feats)
        return gru_out[:, -1, :], context

class KANActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim=64):
        super().__init__()
        self.body = ResearchAgentBody(SINGLE_STEP_FEATURE_DIM, hidden_dim)
        self.expert_critics = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(NUM_EXPERTS)])
        self.expert_actors = nn.ModuleList([nn.Linear(hidden_dim, action_dim) for _ in range(NUM_EXPERTS)])
        self.gate_network = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, NUM_EXPERTS))

    def forward(self, x):
        body_out, context = self.body(x)
        gate_weights = F.softmax(self.gate_network(context), dim=1)
        logits = sum(gate_weights[:, i:i+1] * self.expert_actors[i](body_out) for i in range(NUM_EXPERTS))
        value = sum(gate_weights[:, i:i+1] * self.expert_critics[i](body_out) for i in range(NUM_EXPERTS))
        return logits, value.squeeze(1), gate_weights

    def evaluate(self, x):
        body_out, context = self.body(x)
        gate_weights = F.softmax(self.gate_network(context), dim=1)
        logits = sum(gate_weights[:, i:i+1] * self.expert_actors[i](body_out) for i in range(NUM_EXPERTS))
        value = sum(gate_weights[:, i:i+1] * self.expert_critics[i](body_out) for i in range(NUM_EXPERTS))
        dist = Categorical(logits=logits)
        return dist.log_prob, value.squeeze(1), dist.entropy(), gate_weights, None, None, None

    def act(self, x, deterministic=False):
        logits, _, gate_weights = self.forward(x)
        if deterministic:
            return torch.argmax(logits, dim=1).item(), None, gate_weights
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), gate_weights

# ==============================================================================
# 2. PRODUCTION ENVIRONMENT (Short-Selling & Rebalancing)
# ==============================================================================
class MultiStockEnv(gym.Env):
    def __init__(self, data_map, window_size, is_validation=False):
        super().__init__()
        self.data_map, self.tickers, self.window_size, self.is_validation = data_map, list(data_map.keys()), window_size, is_validation
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, 8), dtype=np.float32)
        self.position, self.portfolio_history, self.max_equity, self.lambda_penalty = 0.0, [10000.0], 10000.0, 1.0

    def reset(self, seed=None):
        self.current_ticker = random.choice(self.tickers)
        self.features, self.prices = self.data_map[self.current_ticker]['features'], self.data_map[self.current_ticker]['prices']
        self.max_step = len(self.prices) - 1
        self.current_step = random.randint(self.window_size, max(self.window_size+1, self.max_step-50))
        self.position, self.portfolio_history, self.max_equity = 0.0, [10000.0], 10000.0
        return self._get_obs(), {}

    def _get_obs(self):
        obs = self.features[max(0, self.current_step - self.window_size) : self.current_step]
        if len(obs) < self.window_size:
            pad = np.zeros((self.window_size - len(obs), SINGLE_STEP_FEATURE_DIM))
            obs = np.vstack([pad, obs])
        return obs.astype(np.float32)

    def step(self, action, expert_profile=None):
        current_price = self.prices[self.current_step]
        self.current_step += 1
        if self.current_step >= len(self.prices): return self._get_obs(), 0.0, True, False, {}

        next_price = self.prices[self.current_step]
        fee, borrow_fee = 0.0005, 0.0001
        prev_val = self.portfolio_history[-1]

        # Rebalancing
        target_pos = 0.0
        if action == 1: target_pos = 1.0
        elif action == 2: target_pos = -1.0

        current_equity = prev_val
        if target_pos != self.position:
            current_equity *= (1 - fee)
            self.position = target_pos

        # Returns
        price_return = (next_price - current_price) / (current_price + 1e-8)
        step_return = self.position * price_return
        if self.position < 0: step_return -= borrow_fee

        new_val = max(10.0, current_equity * (1 + step_return))
        done = self.current_step >= self.max_step or new_val <= 10.0

        # Reward
        self.max_equity = max(self.max_equity, new_val)
        drawdown = (self.max_equity - new_val) / (self.max_equity + 1e-8)
        reward = math.log(np.clip(new_val / (prev_val + 1e-8), 0.5, 2.0)) - (0.2 * drawdown * self.lambda_penalty)
        if self.position < 0 and price_return < -0.01: reward += 0.001 # Short bonus
        if self.position == 0: reward -= 0.0001 # Inflation

        if expert_profile:
            reward *= expert_profile['reward_multiplier']
            if expert_profile['risk_aversion'] > 0.3: reward -= (drawdown * 1.5 * self.lambda_penalty)

        self.portfolio_history.append(new_val)
        return self._get_obs(), float(reward), done, False, {}


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

        # Standard Production Defaults
        self.initial_balance = 10000.0
        self.position = 0.0
        self.max_equity = 10000.0
        self.portfolio_history = [10000.0]

    def reset(self, seed=None):
        self.current_ticker = random.choice(self.tickers)
        self.features = self.data_map[self.current_ticker]['features']
        self.prices = self.data_map[self.current_ticker]['prices']
        self.max_step = len(self.prices) - 1

        # Guard for short windows
        lower_bound = self.window_size
        upper_bound = max(lower_bound + 1, self.max_step - 50)
        self.current_step = random.randint(lower_bound, upper_bound) if upper_bound > lower_bound else lower_bound

        # Reset State
        self.initial_balance = 10000.0
        self.position = 0.0
        self.max_equity = 10000.0
        self.portfolio_history = [10000.0]

        return self.features[self.current_step - self.window_size : self.current_step], {}

    def step(self, action, expert_profile=None):
        current_price = self.prices[self.current_step]
        self.current_step += 1
        next_price = self.prices[self.current_step]
        done = self.current_step >= self.max_step

        fee = 0.0005
        short_borrow_fee = 0.0001
        prev_portfolio_val = self.portfolio_history[-1]

        # 1. Rebalancing Logic
        target_position = 0.0
        if action == 1: target_position = 1.0
        elif action == 2: target_position = -1.0

        current_equity = prev_portfolio_val
        if target_position != self.position:
            current_equity *= (1 - fee)
            self.position = target_position

        # 2. Market Physics (Return-Based)
        # Price return can be positive or negative
        price_return = (next_price - current_price) / (current_price + 1e-8)

        # If position is -1 (Short), a negative price_return becomes a positive gain
        step_return = self.position * price_return

        if self.position < 0:
            step_return -= short_borrow_fee

        new_portfolio_val = current_equity * (1 + step_return)

        # 3. Guardrails
        if new_portfolio_val < 10.0:
            new_portfolio_val = 10.0
            done = True

        # 4. Reward (Log-Utility + Drawdown Penalty)
        self.max_equity = max(self.max_equity, new_portfolio_val)
        drawdown = (self.max_equity - new_portfolio_val) / (self.max_equity + 1e-8)

        # 2. Log-Utility Reward (The main profit driver)
        # Stability clip for the log ratio
        safe_ratio = np.clip(new_portfolio_val / (prev_portfolio_val + 1e-8), 0.5, 2.0)
        reward = math.log(safe_ratio)

        # 3. Symmetric Risk/Drawdown Penalties (The FIX)
        # Uniform penalty for active trading (long or short) to encourage smart risk-taking.
        penalty_weight = 0.2
        cash_penalty = 0.0001 # Small penalty to discourage 'laziness'

        if self.position != 0:
            # Apply Drawdown penalty only for active positions (Long or Short)
            reward -= (penalty_weight * drawdown)
        else:
            # Small penalty for sitting in cash
            reward -= cash_penalty

        # 4. Expert Profile Adjustment (Your original code for experts)
        if expert_profile:
            reward *= expert_profile['reward_multiplier']
            if expert_profile['risk_aversion'] > 0.3: # Conservative Expert
                # Conservative expert gets an extra 1.5x penalty on drawdown
                reward -= (drawdown * 1.5)

        self.portfolio_history.append(new_portfolio_val)
        return self.features[self.current_step - self.window_size : self.current_step], float(reward), done, False, {}

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
                # 1. Unpack Evaluate (7 items)
                # Note: The last 3 are None because we removed VAE
                new_lp_fn, val, ent, gate_weights, _, _, _ = agent.evaluate(s)

                new_lp = new_lp_fn(a)
                adv = r - val.squeeze()

                # 2. PPO Loss
                ratio = torch.exp(new_lp - lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv
                p_loss = -torch.min(surr1, surr2).mean()

                # 3. Value Loss
                v_loss = 0.5 * F.mse_loss(val.squeeze(), r)

                # 4. Entropy Loss (Exploration Bonus)
                entropy_loss = -0.01 * ent.mean()

                # 5. [OPTIONAL] Gate Entropy (Encourage using all experts)
                # Helps prevent the gate from getting stuck on just one expert
                gate_entropy = -torch.sum(gate_weights * torch.log(gate_weights + 1e-8), dim=1).mean()
                gate_loss = -0.01 * gate_entropy

                # 6. TOTAL LOSS (Clean - No VAE)
                loss = p_loss + v_loss + entropy_loss + gate_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                optimizer.step()

                loss_val = loss.item()

        # --- VALIDATE (Every 5 Eps) ---
        if episode % 5 == 0:
            agent.eval()
            total_val_return = 0
            for _ in range(3):
                # 🟢 Reset MUST happen first to create the 'balance' attribute
                val_obs, _ = val_env.reset()
                val_done = False

                while not val_done:
                    state_t = torch.tensor(val_obs, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action, _, _ = agent.act(state_t, deterministic=True)

                    val_obs, _, val_done, _, _ = val_env.step(action)

                    # This now works because balance was created in reset()
                    price = val_env.prices[val_env.current_step-1]
                    # Note: portfolio_history[-1] is the most reliable way to get value
                    curr_val = val_env.portfolio_history[-1]

                final_val = val_env.portfolio_history[-1]
                total_val_return += ((final_val - 10000.0) / 10000.0) * 100

            avg_val_return = total_val_return / 3
            print(f"   📝 Ep {episode} | Loss: {loss_val:.4f} | Val Return: {avg_val_return:.2f}%")

            if avg_val_return > best_val_return:
                best_val_return = avg_val_return
                torch.save(agent.state_dict(), "refactored_moe_agent.pth")
                print(f"      🏆 New Generalist Saved! ({avg_val_return:.2f}%)")

def finetune_and_backtest_asset(ticker, start_date, fine_tune_date, backtest_date):
    print(f"\n\n⚔️ PROTOCOL: Fine-Tune -> Test for {ticker}")
    print(f"   Fine-Tune: {fine_tune_date} -> {backtest_date}")
    print(f"   Backtest:  {backtest_date} -> Now")

    # 1. Data Prep
    full_data = download_multi_stock([ticker], start_date=start_date)
    if not full_data: return
    data = full_data[ticker]
    dates, features, prices = data['dates'], data['features'], data['prices']

    mask_ft = (dates >= fine_tune_date) & (dates < backtest_date)
    mask_bt = (dates >= backtest_date)

    ft_map = {ticker: {'features': features[mask_ft], 'prices': prices[mask_ft]}}
    bt_map = {ticker: {'features': features[mask_bt], 'prices': prices[mask_bt]}}

    if len(features[mask_ft]) < 200: return print("❌ Not enough data.")

    # 2. Load Generalist
    agent = KANActorCritic((WINDOW_SIZE, SINGLE_STEP_FEATURE_DIM), 3, hidden_dim=64).to(device)
    try:
        agent.load_state_dict(torch.load("refactored_moe_agent.pth", map_location=device, weights_only=True))
        print("   ✅ Loaded Generalist (CNN-GRU) Weights.")
    except Exception as e:
        print(f"   ⚠️ Generalist not found or mismatch: {e}. Starting random (NOT RECOMMENDED).")


    # =======================================================
    # 🛑 PHASE 1: TARGETED FINE-TUNING (Target AGGRESSIVE Expert)
    # =======================================================
    print("\n🔨 Phase 1: Targeted Fine-Tuning (Focusing on Aggressive Expert)...")

    # 1. Identify Target Expert Index (Assuming Aggressive is index 0)
    # Check your TEAM_PROFILES definition if this is wrong.
    TARGET_EXPERT_IDX = EXPERT_NAMES.index("Aggressive")
    TARGET_PROFILE = TEAM_PROFILES["Aggressive"]

    # 2. Freeze all non-target components
    for name, param in agent.named_parameters():
        param.requires_grad = False

    # 3. Unfreeze only the components we want to adapt
    # Unfreeze the main body (CNN/GRU) to adapt feature extraction
    for param in agent.body.parameters(): param.requires_grad = True

    # Unfreeze the target expert's actor/critic
    for param in agent.expert_actors[TARGET_EXPERT_IDX].parameters(): param.requires_grad = True
    for param in agent.expert_critics[TARGET_EXPERT_IDX].parameters(): param.requires_grad = True

    # NOTE: We keep the Gate frozen to maintain its global regime classification.

    # 4. Setup Optimizer
    # We only optimize the parameters that are currently requires_grad=True
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, agent.parameters()), lr=5e-6) # Very low LR
    ft_env = MultiStockEnv(ft_map, WINDOW_SIZE, is_validation=False)

    agent.train()
    for epoch in range(50): # Increased epochs for focused training
        obs, _ = ft_env.reset()
        episode_reward = 0.0

        for step in range(50):
            state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            # --- Aggressive Expert Action ---
            # We force the action choice to come from the target expert
            with torch.no_grad():
                body_out, _ = agent.body(state_t) # Get CNN/GRU output

                # Get the logits only from the Aggressive Expert
                logits = agent.expert_actors[TARGET_EXPERT_IDX](body_out)

                # Sample action from this single expert's distribution
                dist = Categorical(logits=logits)
                action_int = dist.sample().item()
                log_prob = dist.log_prob(dist.sample()) # Log prob from the target expert

            # Use the Aggressive Profile's reward
            next_obs, reward, done, _, _ = ft_env.step(action_int, TARGET_PROFILE)
            episode_reward += reward

            # PPO Update using only the Aggressive Critic
            # Note: We must use a dedicated function to get the value from the isolated critic

            # Get Value only from the Aggressive Critic
            body_out_opt, _ = agent.body(state_t)
            val = agent.expert_critics[TARGET_EXPERT_IDX](body_out_opt)

            # PPO Update
            adv = torch.tensor([reward], device=device) - val.squeeze()

            policy_loss = -(log_prob * adv.detach()).mean()
            value_loss = F.mse_loss(val.squeeze(), torch.tensor([reward], device=device))

            # Total Loss
            loss = policy_loss + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            obs = next_obs
            if done: break

        if epoch % 10 == 0:
            print(f"   ⚙️ Ep {epoch}: Avg Reward = {episode_reward / step:.4f}")

    print("   ✅ Targeted Fine-Tuning Complete.")

    # ==========================================
    # 🛑 PHASE 2: BACKTEST (using the whole agent, with the new tuned expert)
    # ==========================================
    print("\n🔮 Backtesting...")
    # NOTE: Set all parameters back to non-training mode for evaluation
    for param in agent.parameters(): param.requires_grad = True # Unfreeze all for generalist use

    bt_env = MultiStockEnv(bt_map, WINDOW_SIZE, is_validation=True)
    obs, _ = bt_env.reset()
    done = False

    actions = []

    agent.eval()
    while not done:
        state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            # Use the full MoE agent here
            action_int, _, _ = agent.act(state_t, deterministic=True)

        obs, _, done, _, _ = bt_env.step(action_int)
        actions.append(action_int)

    initial_val = bt_env.portfolio_history[0]
    final_val = bt_env.portfolio_history[-1]

    ret = ((final_val - initial_val) / initial_val) * 100
    bh = ((bt_env.prices[-1] - bt_env.prices[0]) / bt_env.prices[0]) * 100

    print(f"\n📊 RESULTS: AI {ret:.2f}% vs Buy&Hold {bh:.2f}%")
    print(f"   🧐 Actions: Buys={actions.count(1)} Sells={actions.count(2)} Holds={actions.count(0)}")

    if ret > bh:
        print("   ✅ OUTPERFORMED")
    else:
        print("   ❌ UNDERPERFORMED")

# Global Hyperparameters for Robustness

ENTROPY_BETA = 0.005
VALIDATION_DAYS = 60  # 🛑 Fixed at 60 to prevent IndexError and ensure statistical significance
TOURNAMENT_EPOCHS = 40

def finetune_with_auto_selection(ticker, start_date, fine_tune_date, backtest_date):
    print(f"\n\n⚔️ PROTOCOL: Robust Tournament -> {ticker}")
    full_data = download_multi_stock([ticker], start_date=start_date)
    if not full_data: return
    data = full_data[ticker]
    dates, features, prices = data['dates'], data['features'], data['prices']

    ft_start_dt = pd.to_datetime(fine_tune_date)
    ft_end_dt = pd.to_datetime(backtest_date)
    val_split_dt = ft_end_dt - pd.Timedelta(days=VALIDATION_DAYS)

    m_train = (dates >= ft_start_dt) & (dates < val_split_dt)
    m_val = (dates >= val_split_dt) & (dates < ft_end_dt)
    m_bt = (dates >= ft_end_dt)

    train_map = {ticker: {'features': features[m_train], 'prices': prices[m_train]}}
    val_map = {ticker: {'features': features[m_val], 'prices': prices[m_val]}}
    bt_map = {ticker: {'features': features[m_bt], 'prices': prices[m_bt]}}

    best_oos_score, best_model_state, best_expert_name = -999.0, None, None

    # Test Aggressive vs Conservative vs ShortSpecialist
    for exp_name in ["Aggressive", "Conservative", "ShortSpecialist"]:
        print(f"🥊 Training {exp_name} Candidate...")
        agent = KANActorCritic((WINDOW_SIZE, SINGLE_STEP_FEATURE_DIM), 3, hidden_dim=64).to(device)
        try:
            agent.load_state_dict(torch.load("refactored_moe_agent.pth", map_location=device, weights_only=True))
        except: pass

        idx = EXPERT_NAMES.index(exp_name)
        for p in agent.parameters(): p.requires_grad = False
        for p in agent.body.parameters(): p.requires_grad = True
        for p in agent.expert_actors[idx].parameters(): p.requires_grad = True
        for p in agent.expert_critics[idx].parameters(): p.requires_grad = True

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, agent.parameters()), lr=5e-6)
        env = MultiStockEnv(train_map, WINDOW_SIZE); env.lambda_penalty = 0.3 # 🟢 Lower penalty during schooling

        agent.train()
        for _ in range(40):
            obs, _ = env.reset()
            for _ in range(50):
                st = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                body_out, _ = agent.body(st)
                dist = Categorical(logits=agent.expert_actors[idx](body_out))
                act = dist.sample(); lp = dist.log_prob(act)
                n_obs, r, d, _, _ = env.step(act.item(), TEAM_PROFILES[exp_name])
                val = agent.expert_critics[idx](body_out)

                loss = -(lp * (r - val.detach())).mean() + 0.5 * F.mse_loss(val.view(-1), torch.tensor([r], device=device).view(-1))
                optimizer.zero_grad(); loss.backward(); optimizer.step(); obs = n_obs
                if d: break

        # OOS Validation
        v_env = MultiStockEnv(val_map, WINDOW_SIZE, is_validation=True)
        agent.eval(); obs, _ = v_env.reset(); v_done = False
        while not v_done:
            st = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad(): a, _, _ = agent.act(st, deterministic=True)
            obs, _, v_done, _, _ = v_env.step(a)

        oos_ret = (v_env.portfolio_history[-1] - 10000)/10000
        print(f"   🎯 {exp_name} OOS Result: {oos_ret*100:.2f}%")
        if oos_ret > best_oos_score:
            best_oos_score, best_expert_name, best_model_state = oos_ret, exp_name, agent.state_dict()

    # Final Backtest
    print(f"\n🏆 WINNER: {best_expert_name} | Backtesting on 2025...")
    f_agent = KANActorCritic((WINDOW_SIZE, SINGLE_STEP_FEATURE_DIM), 3, hidden_dim=64).to(device)
    f_agent.load_state_dict(best_model_state); f_agent.eval()
    bt_env = MultiStockEnv(bt_map, WINDOW_SIZE, is_validation=True)
    obs, _ = bt_env.reset(); bt_done, acts = False, []
    while not bt_done:
        st = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad(): a, _, _ = f_agent.act(st, deterministic=True)
        obs, _, bt_done, _, _ = bt_env.step(a); acts.append(a)

    final_ret = (bt_env.portfolio_history[-1] - 10000)/100
    bh_ret = (prices[m_bt][-1] - prices[m_bt][0]) / prices[m_bt][0] * 100

    print("-" * 40 + f"\n📊 LIVE PERFORMANCE: {final_ret:.2f}% (B&H: {bh_ret:.2f}%)\n   🧐 Strategy: {best_expert_name} | Buys: {acts.count(1)} | Shorts: {acts.count(2)}\n" + "-" * 40)

# Example call to run the protocol (outside the function
if __name__ == "__main__":
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
    # finetune_and_backtest_asset(
    #     ticker="TSLA",
    #     start_date="2023-01-01",
    #     fine_tune_date="2024-01-01",
    #     backtest_date="2025-01-01"
    # )
    # finetune_and_backtest_asset(
    #     ticker="ETH-USD",
    #     start_date="2023-01-01",
    #     fine_tune_date="2024-01-01",
    #     backtest_date="2025-01-01"
    # )
    # finetune_and_backtest_asset(
    #     ticker="^IXIC",
    #     start_date="2023-01-01",
    #     fine_tune_date="2024-01-01",
    #     backtest_date="2025-01-01"
    # )
    # Test individual high-momentum stock
    finetune_with_auto_selection("LTC-USD", "2023-01-01", "2024-01-01", "2025-01-01")
    finetune_with_auto_selection("DOGE-USD", "2023-01-01", "2024-01-01", "2025-01-01")
    finetune_with_auto_selection("ETH-USD", "2023-01-01", "2024-01-01", "2025-01-01")
    finetune_with_auto_selection("SOL-USD", "2023-01-01", "2024-01-01", "2025-01-01") # Added: High-Beta Crypto

    # 2. Tech/Momentum Stocks
    finetune_with_auto_selection("NVDA", "2023-01-01", "2024-01-01", "2025-01-01")
    finetune_with_auto_selection("TSLA", "2023-01-01", "2024-01-01", "2025-01-01")
    finetune_with_auto_selection("GOOGL", "2023-01-01", "2024-01-01", "2025-01-01") # Added: Large-Cap Tech

    # 3. Defensive/Value Stocks & Cyclicals
    finetune_with_auto_selection("JNJ", "2023-01-01", "2024-01-01", "2025-01-01") # Added: Defensive Healthcare
    finetune_with_auto_selection("TGT", "2023-01-01", "2024-01-01", "2025-01-01") # Cyclical Retail

    # 4. Market Index
    finetune_with_auto_selection("SPY", "2023-01-01", "2024-01-01", "2025-01-01") # Added: S&P 500 ETF