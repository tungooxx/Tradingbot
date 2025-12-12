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


class FeatureExtractor(nn.Module):
    """
    Replaces the VAE. Uses 1D Convolutions to extract trend and volatility features
    from the 30-day window.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            # Layer 1: Detect local patterns (3-day moves)
            # Input: (Batch, Features, Window) -> (Batch, 8, 30)
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),

            # Layer 2: Detect medium patterns
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            # Layer 3: Compress to a single feature vector per time step
            nn.Conv1d(in_channels=64, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        # We also need a way to summarize the WHOLE window for the Gate
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x shape: (Batch, Window, Features) -> Permute to (Batch, Features, Window)
        x = x.permute(0, 2, 1)

        # Extract features: (Batch, Hidden_Dim, Window)
        features = self.cnn(x)

        # Global Context for Gating: (Batch, Hidden_Dim)
        global_context = self.global_pool(features).squeeze(-1)

        # Permute features back for GRU: (Batch, Window, Hidden_Dim)
        features = features.permute(0, 2, 1)

        return features, global_context

class ResearchAgentBody(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # The Feature Extractor (CNN)
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dim)

        # The Sequence Processor (GRU)
        # Input is now hidden_dim (from CNN) + input_dim (raw price) to preserve signal
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, x):
        # 1. Extract Features (CNN)
        cnn_features, global_context = self.feature_extractor(x)

        # 2. Process Sequence (GRU)
        # We feed the CNN features into the GRU to analyze temporal order
        gru_out, _ = self.gru(cnn_features)

        # Use the last hidden state as the final decision vector
        final_state = gru_out[:, -1, :]

        return final_state, global_context

class KANActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim=64):
        super().__init__()
        self.num_experts = NUM_EXPERTS

        # New Hybrid Body
        self.body = ResearchAgentBody(SINGLE_STEP_FEATURE_DIM, hidden_dim)

        # Experts
        self.expert_critics = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.num_experts)])
        self.expert_actors = nn.ModuleList([nn.Linear(hidden_dim, action_dim) for _ in range(self.num_experts)])

        # 🟢 FIX: Gate uses the CNN's "Global Context"
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, self.num_experts)
        )

    def forward(self, x):
        # 1. Get Body Output & Context
        # body_out: The decision vector from GRU
        # context: The global regime vector from CNN (used for Gating)
        body_out, context = self.body(x)

        # 2. Gating (Decide who is in charge based on Context)
        gate_weights = F.softmax(self.gate_network(context), dim=1)

        # 3. MoE Weighted Sum
        weighted_logits = sum(gate_weights[:, i:i+1] * self.expert_actors[i](body_out) for i in range(self.num_experts))
        weighted_value = sum(gate_weights[:, i:i+1] * self.expert_critics[i](body_out) for i in range(self.num_experts))

        return weighted_logits, weighted_value.squeeze(1), gate_weights

    def evaluate(self, x):
        body_out, context = self.body(x)
        gate_weights = F.softmax(self.gate_network(context), dim=1)

        weighted_logits = sum(gate_weights[:, i:i+1] * self.expert_actors[i](body_out) for i in range(self.num_experts))
        weighted_value = sum(gate_weights[:, i:i+1] * self.expert_critics[i](body_out) for i in range(self.num_experts))

        dist = Categorical(logits=weighted_logits)
        # Note: We return None for VAE outputs since we removed the VAE
        return dist.log_prob, weighted_value.squeeze(1), dist.entropy(), gate_weights, None, None, None

    def act(self, x, deterministic=False):
        logits, _, gate_weights = self.forward(x)
        if deterministic:
            action = torch.argmax(logits, dim=1)
            # 🛑 FIX: Return integer directly
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

        # 🛑 FIX: Dynamically calculate bounds based on actual data size
        self.max_step = len(self.prices) - 1

        # Minimum data required for 1 step is WINDOW_SIZE + 1
        # If we have less, we force the start to a safe index to prevent IndexError
        if self.max_step <= self.window_size:
            self.current_step = self.window_size
        else:
            # We want to start at least at window_size
            # and leave at least 1 step possible (max_step)
            lower_bound = self.window_size
            upper_bound = self.max_step - 1 # Leave at least 1 day to trade

            if upper_bound > lower_bound:
                # Random start for training
                self.current_step = random.randint(lower_bound, upper_bound)
            else:
                self.current_step = lower_bound

        self.balance = 10000.0
        self.shares = 0
        self.buy_price = 0.0
        self.initial_balance = 10000.0
        self.max_equity = 10000.0
        self.portfolio_history = [self.initial_balance]
        self.lambda_penalty = 1.0

        return self.features[self.current_step - self.window_size: self.current_step], {}

    def step(self, action, expert_profile=None):
        current_price = self.prices[self.current_step]
        self.current_step += 1
        next_price = self.prices[self.current_step]
        done = self.current_step >= self.max_step

        # Lower cost to prevent paralysis
        cost = 0.0005

        if action == 1 and self.shares == 0: # Buy
            self.shares = self.balance / current_price
            self.balance = 0.0
            self.buy_price = current_price
            # reward -= cost # (Optional: remove immediate penalty to encourage entry)
        elif action == 2 and self.shares > 0: # Sell
            self.balance = self.shares * current_price * (1 - cost)
            self.shares = 0
            self.buy_price = 0.0

        # --- 🟢 NEW REWARD LOGIC (Log-Utility) ---
        prev_val = self.portfolio_history[-1]
        current_val = self.balance + (self.shares * next_price)
        self.portfolio_history.append(current_val)

        # Update High Water Mark
        self.max_equity = max(self.max_equity, current_val)
        drawdown = (self.max_equity - current_val) / self.max_equity

        if current_val <= 0:
            reward = -10.0 # Bust penalty
        else:
            # Log Return (Geometric Growth)
            log_ret = math.log(current_val / prev_val)

            # Drawdown Penalty (Dynamic Risk Aversion)
            penalty_weight = 0.1 # Lower = More Aggressive
            dd_penalty = penalty_weight * drawdown

            reward = log_ret - dd_penalty

            # Expert Adjustments
            if expert_profile:
                if expert_profile['risk_aversion'] > 0.3: # Conservative
                    reward -= drawdown * 2.0 # Extra pain for drawdown
                reward *= expert_profile['reward_multiplier']

        if done and not self.is_validation:
            self._load_random_stock()
            self.current_step = self.window_size

        return self.features[self.current_step - self.window_size : self.current_step], reward, done, False, {}


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

    final_val = bt_env.balance + (bt_env.shares * bt_env.prices[bt_env.current_step-1])
    ret = ((final_val - bt_env.initial_balance) / bt_env.initial_balance) * 100
    bh = ((bt_env.prices[-1] - bt_env.prices[0]) / bt_env.prices[0]) * 100

    print(f"\n📊 RESULTS: AI {ret:.2f}% vs Buy&Hold {bh:.2f}%")
    print(f"   🧐 Actions: Buys={actions.count(1)} Sells={actions.count(2)} Holds={actions.count(0)}")

    if ret > bh:
        print("   ✅ OUTPERFORMED")
    else:
        print("   ❌ UNDERPERFORMED")

# Global Hyperparameters for Robustness
ENTROPY_BETA = 0.001  # Lowered to allow more conviction
VALIDATION_DAYS = 30  # Shorter OOS window for more relevance
TOURNAMENT_EPOCHS = 50

def finetune_with_auto_selection(ticker, start_date, fine_tune_date, backtest_date):
    print(f"\n\n⚔️ PROTOCOL: Robust Tournament -> Test for {ticker}")
    print(f"   Train Window: {fine_tune_date} -> (Backtest - {VALIDATION_DAYS} days)")
    print(f"   OOS Val Window: (Backtest - {VALIDATION_DAYS} days) -> {backtest_date}")

    # 1. Download Data
    full_data = download_multi_stock([ticker], start_date=start_date)
    if not full_data: return
    data = full_data[ticker]
    dates, features, prices = data['dates'], data['features'], data['prices']

    # 2. Split Fine-Tune Data into Train and OOS Validation
    ft_end_dt = pd.to_datetime(backtest_date)
    ft_start_dt = pd.to_datetime(fine_tune_date)
    val_split_dt = ft_end_dt - pd.Timedelta(days=VALIDATION_DAYS)
    backtest_start_dt = pd.to_datetime(backtest_date)

    mask_train = (dates >= ft_start_dt) & (dates < val_split_dt)
    mask_val = (dates >= val_split_dt) & (dates < ft_end_dt)
    mask_bt = (dates >= backtest_start_dt)

    # 🟢 DEFINING MAPS (Ensures no NameError)
    train_map = {ticker: {'features': features[mask_train], 'prices': prices[mask_train]}}
    val_map = {ticker: {'features': features[mask_val], 'prices': prices[mask_val]}}
    bt_map = {ticker: {'features': features[mask_bt], 'prices': prices[mask_bt]}}

    if np.sum(mask_train) < 100 or np.sum(mask_val) < 10:
        return print(f"❌ Data Error: Split resulted in too few samples for {ticker}")

    # 3. Tournament of Experts
    candidates = ["Aggressive", "Conservative"]
    best_oos_score = -999.0
    best_model_state = None
    best_expert_name = None

    for expert_name in candidates:
        print(f"\n🥊 Training {expert_name} Candidate...")

        # Load Fresh Generalist
        agent = KANActorCritic((WINDOW_SIZE, SINGLE_STEP_FEATURE_DIM), 3, hidden_dim=64).to(device)
        agent.load_state_dict(torch.load("refactored_moe_agent.pth", map_location=device, weights_only=True))

        target_idx = EXPERT_NAMES.index(expert_name)
        target_profile = TEAM_PROFILES[expert_name]

        # Freeze/Unfreeze
        for p in agent.parameters(): p.requires_grad = False
        for p in agent.body.parameters(): p.requires_grad = True
        for p in agent.expert_actors[target_idx].parameters(): p.requires_grad = True
        for p in agent.expert_critics[target_idx].parameters(): p.requires_grad = True

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, agent.parameters()), lr=5e-6)

        # 🟢 Create Training Env
        train_env = MultiStockEnv(train_map, WINDOW_SIZE, is_validation=False)
        # 🟢 LOWER PENALTY during training to encourage exploration
        train_env.lambda_penalty = 0.3

        agent.train()
        for epoch in range(40):
            obs, _ = train_env.reset()
            for _ in range(50):
                state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

                body_out, _ = agent.body(state_t)
                logits = agent.expert_actors[target_idx](body_out)
                dist = Categorical(logits=logits)

                action_int = dist.sample().item()
                log_prob = dist.log_prob(torch.tensor(action_int).to(device))

                next_obs, reward, done, _, _ = train_env.step(action_int, target_profile)

                val = agent.expert_critics[target_idx](body_out)
                adv = torch.tensor([reward], device=device) - val.squeeze()

                policy_loss = -(log_prob * adv.detach()).mean()
                # 🟢 FIXED Broadcasting: view(-1) ensures both input and target are [1]
                value_loss = F.mse_loss(val.view(-1), torch.tensor([reward], device=device).view(-1))
                entropy_loss = dist.entropy().mean()

                loss = policy_loss + 0.5 * value_loss - ENTROPY_BETA * entropy_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                obs = next_obs
                if done: break

        # --- 4. OOS VALIDATION SCORING ---
        val_env = MultiStockEnv(val_map, WINDOW_SIZE, is_validation=True)
        agent.eval()
        obs, _ = val_env.reset()
        val_done = False
        while not val_done:
            state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action, _, _ = agent.act(state_t, deterministic=True)
            obs, _, val_done, _, _ = val_env.step(action)

        oos_return = (val_env.portfolio_history[-1] - 10000) / 10000
        print(f"   🎯 {expert_name} OOS Return (Last 30 Days of 2024): {oos_return*100:.2f}%")

        if oos_return > best_oos_score:
            best_oos_score = oos_return
            best_expert_name = expert_name
            best_model_state = agent.state_dict()

    # --- 5. FINAL TEST ---
    print(f"\n🏆 WINNER: {best_expert_name} | Running on 2025 Data...")
    final_agent = KANActorCritic((WINDOW_SIZE, SINGLE_STEP_FEATURE_DIM), 3, hidden_dim=64).to(device)
    final_agent.load_state_dict(best_model_state)
    final_agent.eval()

    bt_env = MultiStockEnv(bt_map, WINDOW_SIZE, is_validation=True)
    obs, _ = bt_env.reset()
    bt_done = False
    actions = []
    while not bt_done:
        state_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _ = final_agent.act(state_t, deterministic=True)
        obs, _, bt_done, _, _ = bt_env.step(action)
        actions.append(action)

    final_val = bt_env.portfolio_history[-1]
    ret = ((final_val - 10000) / 10000) * 100
    bh = ((prices[mask_bt][-1] - prices[mask_bt][0]) / prices[mask_bt][0]) * 100

    print("-" * 30)
    print(f"📊 PRODUCT PERFORMANCE: {ret:.2f}% (B&H: {bh:.2f}%)")
    print(f"   🧐 Strategy: {best_expert_name} | Buys: {actions.count(1)} | Sells: {actions.count(2)}")
    print("-" * 30)

# Example call to run the protocol (outside the function
if __name__ == "__main__":
    # full_data = download_multi_stock(TICKERS)
    #
    # # 2. Split into Train (2015-2023) and Val (2024+)
    # train_map = {}
    # val_map = {}
    #
    # split_date = "2024-01-01"
    #
    # for ticker, data in full_data.items():
    #     dates = data['dates']
    #     feats = data['features']
    #     prices = data['prices']
    #
    #     # Boolean Mask
    #     mask_train = dates < split_date
    #     mask_val = dates >= split_date
    #
    #     # Only add if enough data
    #     if np.sum(mask_train) > 200:
    #         train_map[ticker] = {
    #             'features': feats[mask_train],
    #             'prices': prices[mask_train]
    #         }
    #
    #     if np.sum(mask_val) > 50:
    #         val_map[ticker] = {
    #             'features': feats[mask_val],
    #             'prices': prices[mask_val]
    #         }
    #
    # print(f"   📊 Train Universe: {len(train_map)} stocks")
    # print(f"   📊 Val Universe:   {len(val_map)} stocks")
    #
    # # 3. Setup
    # train_env = MultiStockEnv(train_map, WINDOW_SIZE, is_validation=False)
    # val_env = MultiStockEnv(val_map, WINDOW_SIZE, is_validation=True)
    #
    # agent = KANActorCritic((WINDOW_SIZE, SINGLE_STEP_FEATURE_DIM), 3, hidden_dim=64).to(device)
    # optimizer = optim.AdamW(agent.parameters(), lr=5e-5)
    #
    # # 4. Train
    # train_multi_stock(agent, train_env, val_env, optimizer)
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
    finetune_with_auto_selection("TSLA", "2023-01-01", "2024-01-01", "2025-01-01")
    finetune_with_auto_selection("TGT",  "2023-01-01", "2024-01-01", "2025-01-01")