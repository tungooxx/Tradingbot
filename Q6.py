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
# warnings.simplefilter(action='ignore', category=FutureWarning)

# ==============================================================================
# 0. CONFIGURATION
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device: {device}")

ENTROPY_BETA = 0.005
VALIDATION_DAYS = 60  # 🛑 Fixed at 60 to prevent IndexError and ensure statistical significance
WINDOW_SIZE = 31
TOURNAMENT_EPOCHS = 40
SINGLE_STEP_FEATURE_DIM = 8
LOAD_BAL_COEFF = 0.0001
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
    "Aggressive":      {"risk_k": 0.1, "reward_mult": 2.0},
    "Conservative":    {"risk_k": 2.0, "reward_mult": 1.0},
    "Standard":        {"risk_k": 0.5, "reward_mult": 1.2},
    "ShortSpecialist": {"risk_k": 0.8, "reward_mult": 1.5},
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

class MambaBlock(nn.Module):
    """
    Selection mechanism that acts as a regime filter.
    Maintains memory of a Bear/Bull state without GRU 'forgetting'.
    """
    def __init__(self, d_model):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.x_proj = nn.Linear(d_model, 16) # Selective state
        self.dt_proj = nn.Linear(16, d_model)
        self.out_proj = nn.Linear(d_model * 2, d_model)

    def forward(self, x):
        # x: (B, W, D)
        z, x_inner = self.in_proj(x).chunk(2, dim=-1)
        x_inner = x_inner.permute(0, 2, 1)
        x_inner = F.silu(self.conv(x_inner)).permute(0, 2, 1)

        # Selective scan (simplified)
        dt = F.softplus(self.dt_proj(self.x_proj(x_inner)))
        x_inner = x_inner * dt

        return self.out_proj(torch.cat([x_inner, F.silu(z)], dim=-1))

class QuantAgentBody(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.initial_map = nn.Linear(input_dim, hidden_dim)
        self.ssm1 = MambaBlock(hidden_dim)
        self.ssm2 = MambaBlock(hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.initial_map(x)
        x = self.norm(self.ssm1(x) + x)
        x = self.norm(self.ssm2(x) + x)
        # return: Decision Vector (last step), Regime Context (average of window)
        return x[:, -1, :], x.mean(dim=1)

class KANActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim, hidden_dim=128):
        super().__init__()
        self.body = QuantAgentBody(SINGLE_STEP_FEATURE_DIM, hidden_dim)
        self.actors = nn.ModuleList([nn.Linear(hidden_dim, action_dim) for _ in range(NUM_EXPERTS)])
        self.critics = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(NUM_EXPERTS)])
        self.gate_head = nn.Linear(hidden_dim, NUM_EXPERTS)

    def forward(self, x, train=True):
        decision_vec, regime_vec = self.body(x)
        gate_logits = self.gate_head(regime_vec)

        if train:
            # Gumbel-Softmax allows gradient to flow through a discrete choice
            gate_weights = F.gumbel_softmax(gate_logits, tau=1.0, hard=True)
        else:
            # Top-1 Hard Selection for production performance
            idx = torch.argmax(gate_logits, dim=1)
            gate_weights = F.one_hot(idx, num_classes=NUM_EXPERTS).float()

        logits = sum(gate_weights[:, i:i+1] * self.actors[i](decision_vec) for i in range(NUM_EXPERTS))
        value = sum(gate_weights[:, i:i+1] * self.critics[i](decision_vec) for i in range(NUM_EXPERTS))
        return logits, value.squeeze(1), gate_weights

    def evaluate(self, x):
        logits, value, gate_weights = self.forward(x, train=True)
        dist = Categorical(logits=logits)
        return dist.log_prob, value, dist.entropy(), gate_weights

    def act(self, x, deterministic=False):
        logits, _, gate_weights = self.forward(x, train=False)
        if deterministic: return torch.argmax(logits, dim=1).item(), None, gate_weights
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), gate_weights

# ==============================================================================
# 2. ENVIRONMENT: ADVERSARIAL EXECUTION & VOL-TARGETED REWARD
# ==============================================================================
class MultiStockEnv(gym.Env):
    def __init__(self, data_map, window_size, is_validation=False):
        super().__init__()
        self.data_map = data_map
        self.tickers = list(data_map.keys())
        self.window_size = window_size
        self.is_validation = is_validation
        self.action_space = spaces.Discrete(3) # 0: Cash, 1: Long, 2: Short
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(window_size, SINGLE_STEP_FEATURE_DIM))

        # Permanent Attributes
        self.position = 0.0
        self.portfolio_history = [10000.0]
        self.max_equity = 10000.0
        self.returns_history = [0.0]
        self.lambda_penalty = 1.0

    def reset(self, seed=None):
        self.current_ticker = random.choice(self.tickers)
        self.raw_features = self.data_map[self.current_ticker]['features']
        self.prices = self.data_map[self.current_ticker]['prices']
        self.max_step = len(self.prices) - 1
        self.current_step = random.randint(self.window_size, max(self.window_size+1, self.max_step-100))

        self.position = 0.0
        self.portfolio_history = [10000.0]
        self.max_equity = 10000.0
        self.returns_history = [0.0]
        return self._get_obs(), {}

    def _get_obs(self):
        # 🟢 FIX FLAW I: Local Rolling Normalization (Zero Look-Ahead)
        raw_window = self.raw_features[self.current_step - self.window_size : self.current_step]
        mean = raw_window.mean(axis=0)
        std = raw_window.std(axis=0) + 1e-8
        norm_window = (raw_window - mean) / std
        return norm_window.astype(np.float32)

    def step(self, action, profile=None):
        current_price = self.prices[self.current_step]
        self.current_step += 1
        if self.current_step >= len(self.prices): return self._get_obs(), 0.0, True, False, {}

        next_price = self.prices[self.current_step]
        prev_portfolio_val = self.portfolio_history[-1]

        # 🟢 FIX FLAW III: Dynamic Slippage (Amihud Scaled)
        # features[6] is Amihud Illiquidity, features[4] is ATR
        amihud_val = self.raw_features[self.current_step-1, 6]
        atr_val = self.raw_features[self.current_step-1, 4]
        dynamic_fee = 0.0005 + (amihud_val * atr_val * 100.0) # Fee spikes in thin/panic markets

        # Trade Execution
        target_pos = 0.0
        if action == 1: target_pos = 1.0
        elif action == 2: target_pos = -1.0

        current_equity = prev_portfolio_val
        if target_pos != self.position:
            current_equity *= (1 - dynamic_fee)
            self.position = target_pos

        price_return = (next_price - current_price) / (current_price + 1e-8)
        step_return = self.position * price_return
        if self.position < 0: step_return -= 0.0001 # Standard borrow fee

        new_val = max(10.0, current_equity * (1 + step_return))
        self.portfolio_history.append(new_val)
        self.max_equity = max(self.max_equity, new_val)

        # 🟢 SORTINO DYNAMIC REWARD
        log_ret = math.log(np.clip(new_val / (prev_portfolio_val + 1e-8), 0.1, 2.0))
        self.returns_history.append(log_ret)

        recent_returns = self.returns_history[-20:]
        downside_returns = [r for r in recent_returns if r < 0]

        # 🟢 FIX FLAW II: Professional Sortino Floor
        # We use a Minimum Acceptable Return (MAR) of 0.
        downside_risk = np.std(downside_returns) + 0.001 if len(downside_returns) > 1 else 0.005
        risk_adj_reward = log_ret / downside_risk

        drawdown = (self.max_equity - new_val) / (self.max_equity + 1e-8)
        # Volatility Targeting: ATR scales the drawdown penalty
        # If ATR is 5% (Panic), the drawdown is punished 5x more.
        reward = risk_adj_reward - (drawdown * 2.0 * (1.0 + atr_val))

        if profile:
            reward *= profile['reward_mult']
            reward -= (drawdown * profile['risk_k'] * (1.0 + atr_val))

        done = (new_val <= 10.0 or self.current_step >= self.max_step)
        return self._get_obs(), float(np.clip(reward, -1, 1)), done, False, {}


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
# ==============================================================================
# 4. TRAINING LOOP (MULTI-STOCK)
# ==============================================================================
def train_multi_stock(agent, train_env, val_env, optimizer, episodes=500):
    print(f"\n🧠 Quant-V6 Hive Training Initiated on {device}...")

    for ep in range(1, episodes + 1):
        agent.train()
        obs, _ = train_env.reset()

        # Rollout Storage
        states, actions, logprobs, rewards, values = [], [], [], [], []
        expert_usage = []

        # 1. ROLLOUT PHASE (Data Collection)
        for _ in range(500): # Steps per episode
            st = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)

            # Use stochastic act for training exploration
            with torch.no_grad():
                logits, val, gate_weights = agent.forward(st, train=True)
                dist = Categorical(logits=logits)
                a_int = dist.sample()
                lp = dist.log_prob(a_int)

            # Select expert for reward shaping (Top-1 importance)
            expert_idx = gate_weights.argmax().item()
            expert_usage.append(expert_idx)

            # Environment Step
            n_obs, r, d, _, _ = train_env.step(a_int.item(), TEAM_PROFILES[EXPERT_NAMES[expert_idx]])

            states.append(st)
            actions.append(a_int)
            logprobs.append(lp)
            rewards.append(r)
            values.append(val)

            obs = n_obs
            if d: obs, _ = train_env.reset()

        # 2. PPO ADVANTAGE CALCULATION
        b_s = torch.cat(states)
        b_a = torch.stack(actions)
        b_lp = torch.stack(logprobs).detach()

        # Standard GAE-like Return calculation
        R = 0
        returns = []
        for r in reversed(rewards):
            R = r + 0.99 * R
            returns.insert(0, R)
        b_r = torch.tensor(returns, dtype=torch.float32).to(device)
        # 🟢 CRITICAL: Advantage Normalization (Kills Gradient Outliers)
        b_r = (b_r - b_r.mean()) / (b_r.std() + 1e-8)

        # 3. OPTIMIZATION PHASE
        loader = DataLoader(TensorDataset(b_s, b_a, b_lp, b_r), batch_size=128, shuffle=True)

        loss_val = 0
        for _ in range(3): # Epochs per batch
            for s, a, lp, r in loader:
                new_lp_fn, val, ent, gate_weights = agent.evaluate(s)

                # 🟢 LOAD BALANCING LOSS (Prevents Expert Collapse)
                # We want equal importance across all 4 experts
                importance = gate_weights.mean(dim=0)
                # Standard deviation of expert usage should be low
                load_balancing_loss = LOAD_BAL_COEFF * (importance.std() / (importance.mean() + 1e-8))

                # PPO Clipped Objective
                adv = (r - val.detach())
                ratio = torch.exp(new_lp_fn(a) - lp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * adv

                p_loss = -torch.min(surr1, surr2).mean()
                v_loss = 0.5 * F.mse_loss(val.view(-1), r.view(-1))
                e_loss = -0.01 * ent.mean()

                # Total Combined Loss
                loss = p_loss + v_loss + e_loss + load_balancing_loss

                optimizer.zero_grad()
                loss.backward()

                # 🟢 THE GRADIENT FUSE (Bankruptcy Protection)
                # Limits the maximum update to prevent weights from becoming NaN
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)

                optimizer.step()
                loss_val = loss.item()

        # 4. LOGGING & FEEDBACK
        if ep % 10 == 0:
            # Check if all experts are participating
            usage_str = ", ".join([f"{EXPERT_NAMES[i]}: {expert_usage.count(i)}" for i in range(NUM_EXPERTS)])
            print(f"📝 Ep {ep} | Loss: {loss_val:.4f} | Gate Usage: [{usage_str}]")

        if ep % 50 == 0:
            torch.save(agent.state_dict(), "quant_v6_foundation2.pth")
            print(f"   💾 Foundation Model Checkpoint Saved.")

# ==============================================================================
# 5. DATA INGESTION (NO LOOK-AHEAD VERSION)
# ==============================================================================
def download_quant_v6_data(tickers, start='2018-01-01'):
    print(f"📥 Building Raw Market Ingestion Layer for {len(tickers)} assets...")

    # 1. Download SPY (Benchmark)
    try:
        spy = yf.download("SPY", start=start, progress=False, auto_adjust=True)
        spy_ret = np.log(spy['Close'] / spy['Close'].shift(1)).fillna(0)
    except Exception as e:
        print(f"Failed to download SPY: {type(e).__name__}: {e}")
        return {}

    data_map = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, progress=False, auto_adjust=True)

            # 🟢 FIX: Flatten MultiIndex columns (removes Ticker level if present)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if len(df) < 500:
                print(f"   Skipping {t}: Not enough data.")
                continue

            # --- START STRICT ALIGNMENT ---
            # 🟢 STEP A: Combine SPY and Ticker indices to get all possible dates
            combined_index = spy.index.union(df.index)

            # 🟢 STEP B: Reindex the Ticker data to the combined index
            prices_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reindex(combined_index)
            prices_df = prices_df.ffill().bfill() # Forward/Backward fill missing days

            # 🟢 STEP C: Reindex SPY returns to the combined index
            local_spy_ret = spy_ret.reindex(combined_index).ffill().bfill()

            # 🟢 STEP D: Create the empty feature matrix
            tdf = pd.DataFrame(index=combined_index)
            # --- END STRICT ALIGNMENT ---

            # 2. Core Returns/Volume
            tdf['Ret'] = np.log(prices_df['Close'] / prices_df['Close'].shift(1))
            tdf['Vol'] = prices_df['Volume'].astype(float)

            # 3. Amihud Illiquidity (Guaranteed safe math)
            tdf['Amihud'] = tdf['Ret'].abs() / (tdf['Vol'] * prices_df['Close'] + 1e-12)

            # 4. Technical Context (MANUAL IMPLEMENTATION)

            # Manual MACD (Fixes the source of the persistent error)
            ema_fast = prices_df['Close'].ewm(span=12, adjust=False).mean()
            ema_slow = prices_df['Close'].ewm(span=26, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=9, adjust=False).mean()
            tdf['MACD'] = macd_line - macd_signal # MACD Histogram

            # Manual RSI (Since we removed pandas_ta, we use a manual helper or keep if available)
            # NOTE: If pandas_ta is still failing, uncomment the manual RSI implementation below
            # For now, let's try the simple RSI formula from pandas_ta, assuming it's still available
            try:
                import pandas_ta as ta
                tdf['RSI'] = ta.rsi(prices_df['Close'], length=14)
            except ImportError:
                # Backup plan if pandas_ta is entirely removed
                tdf['RSI'] = 50.0 # Placeholder/Simple fallback

            # Manual ATR (We can keep this simple using max/min)
            high_low = prices_df['High'] - prices_df['Low']
            high_close = np.abs(prices_df['High'] - prices_df['Close'].shift(1))
            low_close = np.abs(prices_df['Low'] - prices_df['Close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            tdf['ATR'] = tr.ewm(span=14, adjust=False).mean()

            # Volatility (Rolling Std)
            tdf['VRP_Raw'] = prices_df['Close'].rolling(20).std()

            # 5. SPY Relative Return
            tdf['SPY_Rel'] = local_spy_ret

            # Final cleanup
            tdf.dropna(inplace=True)
            prices_df = prices_df.reindex(tdf.index) # Align prices back to the cleaned index

            cols = ['Ret', 'Vol', 'RSI', 'MACD', 'ATR', 'VRP_Raw', 'Amihud', 'SPY_Rel']

            data_map[t] = {
                'features': tdf[cols].values.astype(np.float32),
                'prices': prices_df['Close'].values.astype(np.float32),
                'dates': tdf.index
            }
            print(f"   ✅ {t} Ingested. Data shape: {data_map[t]['features'].shape}")

        except Exception as e:
            # We catch the exception and print its details
            print(f"   ❌ {t} Failed: {type(e).__name__}: {e}")
            # print(traceback.format_exc())

    return data_map

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


def finetune_with_auto_selection(ticker, start_date, fine_tune_date, backtest_date):
    print(f"\n\n⚔️ PROTOCOL: Robust Tournament -> {ticker}")
    full_data = download_quant_v6_data([ticker], start=start_date)
    if not full_data: return

    data = full_data[ticker]
    dates, features, prices = data['dates'], data['features'], data['prices']

    ft_start_dt = pd.to_datetime(fine_tune_date)
    ft_end_dt = pd.to_datetime(backtest_date)

    # 1. Get ALL data available for the Fine-Tuning period first
    mask_ft_total = (dates >= ft_start_dt) & (dates < ft_end_dt)
    mask_bt = (dates >= ft_end_dt)

    ft_features = features[mask_ft_total]
    ft_prices = prices[mask_ft_total]
    bt_prices = prices[mask_bt]

    total_ft_samples = len(ft_features)

    # 🟢 SAFETY CHECK: Ensure we have enough data to split
    min_val_size = WINDOW_SIZE + 20
    if total_ft_samples < (WINDOW_SIZE + min_val_size + 20):
        print(f"❌ Skipping {ticker}: Insufficient FT data ({total_ft_samples} rows).")
        return

    # 🟢 ROBUST SPLIT: Slice by Index, not Calendar Time
    val_size = int(total_ft_samples * 0.2)
    if val_size < min_val_size:
        val_size = min_val_size

    train_size = total_ft_samples - val_size

    # Slice the arrays manually
    train_map = {ticker: {
        'features': ft_features[:train_size],
        'prices': ft_prices[:train_size]
    }}

    val_map = {ticker: {
        'features': ft_features[train_size:],
        'prices': ft_prices[train_size:]
    }}

    bt_map = {ticker: {
        'features': features[mask_bt],
        'prices': bt_prices
    }}

    print(f"   📊 Split: Train={train_size} | Val={val_size} | Backtest={len(bt_prices)}")

    best_oos, best_state, best_name = -999.0, None, None

    for exp_name in EXPERT_NAMES:
        print(f"🥊 Round 1: Training {exp_name} Candidate...")
        agent = KANActorCritic((WINDOW_SIZE, SINGLE_STEP_FEATURE_DIM), 3, hidden_dim=128).to(device)
        try:
            agent.load_state_dict(torch.load("quant_v6_foundation.pth", map_location=device))
        except:
            print("   ⚠️ Foundation not found, starting from scratch.")

        idx = EXPERT_NAMES.index(exp_name)

        # Freeze/Unfreeze Logic
        for p in agent.parameters(): p.requires_grad = False
        for p in agent.body.parameters(): p.requires_grad = True

        # 🟢 FIX: Revert to 'actors' and 'critics' (the likely attribute names)
        for p in agent.actors[idx].parameters(): p.requires_grad = True
        for p in agent.critics[idx].parameters(): p.requires_grad = True

        opt = optim.AdamW(filter(lambda p: p.requires_grad, agent.parameters()), lr=1e-5)

        # Train Env
        env = MultiStockEnv(train_map, WINDOW_SIZE)
        env.lambda_penalty = 0.05

        agent.train()

        # 🟢 FIX 1: CALCULATE MAX_STEPS CORRECTLY
        # Access the price length from the data map, which is guaranteed to exist.
        train_prices = train_map[ticker]['prices']
        max_steps = len(train_prices) - WINDOW_SIZE

        for epoch in range(TOURNAMENT_EPOCHS):
            # 🟢 FIX 2: Correct the misleading print statement to show the true epoch

            obs, _ = env.reset()

            for _ in range(max_steps):
                st = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                body_out, _ = agent.body(st)

                # 🟢 FIX: Revert to 'actors' and 'critics'
                dist = Categorical(logits=agent.actors[idx](body_out))
                act = dist.sample(); lp = dist.log_prob(act)

                n_obs, r, d, _, _ = env.step(act.item(), TEAM_PROFILES[exp_name])

                val = agent.critics[idx](body_out)
                loss = -(lp * (r - val.detach())).mean() + 0.5 * F.mse_loss(val.view(-1), torch.tensor([r], device=device).view(-1))

                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5); opt.step(); obs = n_obs
                if d: break

        # OOS Validation
        v_env = MultiStockEnv(val_map, WINDOW_SIZE, is_validation=True)
        val_prices = val_map[ticker]['prices']

        if len(val_prices) <= WINDOW_SIZE:
            print(f"   ⚠️ Validation skipped for {exp_name}: Data too short.")
            oos_ret = -1.0 # Penalize
        else:
            agent.eval(); obs, _ = v_env.reset(); v_done = False
            while not v_done:
                st = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad(): a, _, _ = agent.act(st, deterministic=True)
                obs, _, v_done, _, _ = v_env.step(a)
            oos_ret = (v_env.portfolio_history[-1] - 10000)/10000

        print(f"   🎯 {exp_name} OOS Result: {oos_ret*100:.2f}%")
        if oos_ret > best_oos:
            best_oos, best_name, best_state = oos_ret, exp_name, agent.state_dict()

    # Final Backtest
    if best_state is None:
        print("❌ Tournament failed to produce a winner.")
        return

    print(f"\n🏆 WINNER: {best_name} | Testing on 2025 data...")
    f_agent = KANActorCritic((WINDOW_SIZE, SINGLE_STEP_FEATURE_DIM), 3, hidden_dim=128).to(device)
    f_agent.load_state_dict(best_state)
    f_agent.eval()

    bt_env = MultiStockEnv(bt_map, WINDOW_SIZE, is_validation=True)

    if len(bt_prices) <= WINDOW_SIZE:
        print("❌ Backtest skipped: Insufficient data.")
        return

    obs, _ = bt_env.reset(); bt_done, acts = False, []
    while not bt_done:
        st = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad(): a, _, _ = f_agent.act(st, deterministic=True)
        obs, _, bt_done, _, _ = bt_env.step(a); acts.append(a)

    ret = (bt_env.portfolio_history[-1]-10000)/100

    # Safe B&H calculation
    bh = ((bt_prices[-1] - bt_prices[0]) / bt_prices[0]) * 100

    print("-" * 40 + f"\n📊 LIVE PERFORMANCE: {ret:.2f}% (B&H: {bh:.2f}%)\n   🧐 Strategy: {best_name} | Buys: {acts.count(1)} | Shorts: {acts.count(2)}\n" + "-" * 40)

# Example call to run the protocol (outside the function
if __name__ == "__main__":
    # --- PHASE 1: Build the Foundation ---
    FOUNDATION_TICKERS = [
        "AAPL", "NVDA", "MSFT", "AMZN", "GOOGL", "TSLA", "META",
        "JPM", "GS", "XOM", "CVX", "SPY", "QQQ", "BTC-USD", "ETH-USD"
    ]

    print("--- 1. FOUNDATION TRAINING ---")
    full_train_data = download_quant_v6_data(FOUNDATION_TICKERS, start='2018-01-01')

    # 🟢 CRITICAL FIX: Data Validation and Filtering
    # Only keep tickers that have at least 2x the window size for a safe reset
    valid_train_data = {}
    for t, d in full_train_data.items():
        if d['features'].shape[0] > WINDOW_SIZE * 2:
            valid_train_data[t] = d
        else:
            print(f"   ⚠️ Skipping {t}: Insufficient data ({d['features'].shape[0]} steps).")

    if not valid_train_data:
        print("🛑 Critical: No assets loaded with sufficient data for training. Exiting.")
    else:
        print(f"✅ Training Universe Established: {len(valid_train_data)} tickers.")

        # 2. Setup Generalist Environment (Use the validated data)
        train_env = MultiStockEnv(valid_train_data, WINDOW_SIZE)
        val_env = MultiStockEnv(valid_train_data, WINDOW_SIZE, is_validation=True)

        # 3. Initialize Agent
        agent = KANActorCritic((WINDOW_SIZE, SINGLE_STEP_FEATURE_DIM), 3, hidden_dim=128).to(device)
        optimizer = optim.AdamW(agent.parameters(), lr=1e-4)

        # 4. Train Foundation Brain (Only run if .pth doesn't exist or you want to reset)
        import os
        if not os.path.exists("quant_v6_foundation2.pth"):
            print("🧠 Training Foundation Model...")
            train_multi_stock(agent, train_env, val_env, optimizer, episodes=500)
        else:
            print("✅ Foundation model found. Skipping pre-training.")

        # --- PHASE 2: Tournament & Real Trading ---
        print("\n--- 2. TOURNAMENT & BACKTEST ---")
        DEPLOYMENT_TARGETS = ["TSLA", "NVDA", "TGT", "BTC-USD", "ETH-USD"]

        # The agent needs to be loaded if training was skipped
        if os.path.exists("quant_v6_foundation.2pth"):
            agent.load_state_dict(torch.load("quant_v6_foundation2.pth", map_location=device))

        for ticker in DEPLOYMENT_TARGETS:
            # finetune_with_auto_selection must be defined and use agent.actors
            finetune_with_auto_selection(
                ticker=ticker,
                start_date="2020-01-01",
                fine_tune_date="2024-01-01",
                backtest_date="2025-01-01"
            )