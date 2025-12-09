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
# ==============================================================================
# 1. YOUR KAN LINEAR IMPLEMENTATION
# ==============================================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # This ensures exact reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) # Call this!
class KANLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
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
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(
                                self.grid_size + self.spline_order,
                                self.in_features,
                                self.out_features,
                                )
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        # Handle flattening for 3D inputs if necessary, though typical usage is 2D here
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1 :] - x)
                            / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )
        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
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
            output = output.view(
                original_shape[0], original_shape[1], self.out_features
            )

        return output

# ==============================================================================
# 2. SHARED MODEL ARCHITECTURE (PRE-TRAIN -> TRANSFER)
# ==============================================================================

class KANBody(nn.Module):
    """
    The 'Brain' that learns market features.
    This is shared between the Predictor and the RL Agent.
    """
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
    """
    Model for Supervised Pre-training.
    Predicts next step's log return.
    """
    def __init__(self, obs_dim, hidden_dim=64):
        super(KANPredictor, self).__init__()
        self.body = KANBody(obs_dim, hidden_dim)
        # Regression head
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.body(x)
        return self.head(features)

class KANActorCritic(nn.Module):
    """
    RL Agent. Inherits the 'body' from the predictor.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=64, pretrained_body=None):
        super(KANActorCritic, self).__init__()

        # If pre-trained body is provided, use it (Transfer Learning)
        if pretrained_body:
            self.body = pretrained_body
            # Freeze the body? Optional. Usually fine-tuning (not freezing) is better for RL.
            # for param in self.body.parameters():
            #     param.requires_grad = False
        else:
            self.body = KANBody(obs_dim, hidden_dim)

        # RL Heads
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
# 3. ENVIRONMENT & UTILS
# ==============================================================================
class TradingLogger:
    def __init__(self, missed_opp_threshold=0.015):
        self.history = []
        self.open_positions = []
        self.missed_opp_threshold = missed_opp_threshold

    def log_step(self, date, open_price, high_price, low_price, real_close, pred_close, action):
        """Log a single timestep.

        action: "buy" | "sell" | "skip"
        pred_close: predicted next close price (or None)
        """
        profit = 0.0
        if action == "buy":
            self.open_positions.append(real_close)
        elif action == "sell":
            if self.open_positions:
                buy_price = self.open_positions.pop(0)
                profit = real_close - buy_price

        self.history.append({
            "Date": date,
            "Open": float(open_price) if open_price is not None else None,
            "High": float(high_price) if high_price is not None else None,
            "Low": float(low_price) if low_price is not None else None,
            "Real_Close": float(real_close) if real_close is not None else None,
            "Pred_Close": float(pred_close) if pred_close is not None else None,
            "Action": action.upper(),
            "Profit": float(profit) if action == "sell" else 0.0,
            "Holding_Count": len(self.open_positions),
        })

    def get_results(self):
        df = pd.DataFrame(self.history)
        if df.empty:
            return df

        # Hindsight: compare to next day's real close
        df["Next_Close"] = df["Real_Close"].shift(-1)
        df["Price_Change_Pct"] = (df["Next_Close"] - df["Real_Close"]) / df["Real_Close"]

        missed = []
        for _, row in df.iterrows():
            if row["Action"] in ["SKIP", "SELL"] and row["Holding_Count"] == 0:
                if row["Price_Change_Pct"] > self.missed_opp_threshold:
                    missed.append("YES")
                else:
                    missed.append("-")
            else:
                missed.append("-")

        df["Miss_Opportunity"] = missed

        final = df[["Date", "Open", "High", "Low", "Real_Close", "Pred_Close", "Action", "Profit", "Miss_Opportunity"]]
        return final

class StockTradingEnv(gym.Env):
    def __init__(self, ticker="SPY", window_size=10):
        super(StockTradingEnv, self).__init__()
        print(f"📥 Downloading {ticker} data...")
        self.df = yf.download(ticker, period="5y", interval="1d", progress=False)

        # Fix MultiIndex (yfinance update)
        if isinstance(self.df.columns, pd.MultiIndex):
            self.df.columns = self.df.columns.get_level_values(0)

        # ==========================================
        # 1. CALCULATE ALL INDICATORS & FEATURES
        # ==========================================
        # A. Basic Features (Returns & Vol)
        self.df["Log_Ret"] = np.log(self.df["Close"] / self.df["Close"].shift(1))
        self.df["Vol_Norm"] = self.df["Volume"] / self.df["Volume"].rolling(20).mean()

        # B. Technical Indicators (pandas_ta)
        self.df.ta.rsi(length=14, append=True)
        self.df.ta.macd(fast=12, slow=26, signal=9, append=True)
        # Note: Bollinger Bands adds 3 columns (Lower, Mid, Upper).
        # We aren't using them in 'self.features' yet, but good to have.
        self.df.ta.bbands(length=20, std=2, append=True)

        # ==========================================
        # 2. CLEAN & NORMALIZE
        # ==========================================
        # Drop NaNs created by rolling windows (Vol_Norm needs 20 days, MACD needs 26)
        self.df.dropna(inplace=True)

        # Normalize Technicals
        # RSI is 0-100 -> Scale to 0-1
        self.df["RSI_14"] = self.df["RSI_14"] / 100.0

        # MACD is unbounded -> Standard Score Normalization
        self.df["MACD_12_26_9"] = (
            self.df["MACD_12_26_9"] - self.df["MACD_12_26_9"].mean()
        ) / (self.df["MACD_12_26_9"].std() + 1e-7)

        # ==========================================
        # 3. DEFINE OBSERVATION SPACE
        # ==========================================
        # Define the 5 columns we want the AI to see
        self.features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9"]

        # Create the data matrix
        self.data = self.df[self.features].values

        # Set limits
        self.window_size = window_size
        self.max_steps = len(self.data) - 1

        # Spaces
        self.action_space = spaces.Discrete(3)

        # Update Shape: Window * Number of Features (10 * 5 = 50)
        self.obs_shape = window_size * len(self.features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        # Step 3: start with $20 balance
        self.balance = 20.0
        self.shares = 0
        self.entry_price = 0
        # Fixed dollar amount to invest per BUY action
        self.invest_amount = 1.0
        return self._get_observation(), {}

    def _get_observation(self, step=None):
        if step is None: step = self.current_step
        window = self.data[step - self.window_size : step]
        return window.flatten().astype(np.float32)

    def step(self, action):
        current_price = self.data[self.current_step][0]
        # Get the Log Return for today (how much the market moved)
        daily_log_ret = self.data[self.current_step][1]

        reward = 0
        done = False

        # --- 1. ACTION LOGIC ---
        if action == 1:  # BUY
            # Fix: Invest ALL available cash, not just $1
            # This mimics "Going Long"
            if self.shares == 0 and self.balance > 0:
                self.shares = self.balance / current_price
                self.balance = 0.0
                self.entry_price = current_price

                # Penalty: Small negative reward for entering to discourage spamming
                reward -= 0.05

        elif action == 2:  # SELL
            if self.shares > 0:
                # 1. Calculate Gross Value
                gross_val = self.shares * current_price

                # 2. Apply 0.1% Fee (Spread/Commissions)
                net_val = gross_val * 0.999

                # 3. Calculate Real Dollar Profit
                # (What we have now) - (What we spent)
                # We need to know 'cost_basis'. Since we go all-in, cost_basis was the entry value.
                cost_basis = self.shares * self.entry_price
                profit_dollars = net_val - cost_basis

                # 4. Reward is % Return on Capital
                reward = (profit_dollars / cost_basis) * 100

                self.balance = net_val
                self.shares = 0
                self.entry_price = 0

        # --- 2. HOLDING REWARD (FIXED) ---
        # Only give holding reward if we actually HOLD shares
        if self.shares > 0:
            # If we hold, we experience the market volatility (good or bad)
            # We add a tiny 'time cost' (-0.001) so it doesn't hold forever if flat
            reward += (daily_log_ret * 10) - 0.001
        else:
            # If we are in cash, we get 0 reward.
            # Safe, but we miss out on potential gains (Opportunity Cost).
            reward += 0.0

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_observation(), reward, done, False, {}


# ==============================================================================
# 4. TRAINING LOGIC
# ==============================================================================

def prepare_pretraining_data(env):
    """Extracts windows (X) and next day Log Returns (y) from environment data"""
    X_list = []
    y_list = []

    # We iterate through the whole history available in env.data
    # data columns: [Close, Log_Ret, Vol_Norm]
    # We want to predict NEXT Log_Ret (index 1)

    for i in range(env.window_size, len(env.data) - 1):
        window = env.data[i - env.window_size : i] # The past
        target = env.data[i][1] # The Log_Ret of "today" (which is next step relative to window)

        X_list.append(window.flatten())
        y_list.append(target)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32).reshape(-1, 1)

def run_pipeline():
    # 1. Init Env
    env = StockTradingEnv(ticker='SPY', window_size=10)
    obs_dim = env.obs_shape
    action_dim = env.action_space.n
    hidden_dim = 32 # Keep small for speed

    # ==========================================
    # PHASE 1: PRE-TRAINING (SUPERVISED)
    # ==========================================
    print("\n🧠 PHASE 1: Pre-training KAN Predictor...")

    # Create dataset
    X_train, y_train = prepare_pretraining_data(env)
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    # Init Predictor
    predictor = KANPredictor(obs_dim, hidden_dim)
    optimizer_pred = optim.Adam(predictor.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train Loop
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer_pred.zero_grad()
            preds = predictor(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer_pred.step()
            total_loss += loss.item()

        if (epoch+1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs} | MSE Loss: {total_loss/len(dataloader):.6f}")

    print("✅ Pre-training complete. Weights primed.")

    # ==========================================
    # PHASE 2: RL TRAINING (PPO)
    # ==========================================
    print("\n🎮 PHASE 2: Transferring Weights & Starting RL...")

    # Pass the TRAINED BODY to the Agent
    agent = KANActorCritic(obs_dim, action_dim, hidden_dim, pretrained_body=predictor.body)
    optimizer_rl = optim.Adam(agent.parameters(), lr=0.0005) # Lower LR for fine-tuning

    # Reset Logger
    logger = TradingLogger()

    # PPO Params
    state, _ = env.reset()
    episode_reward = 0
    memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []

    for step in range(1, 10000):
        # -- Action --
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, _ = agent.act(state_tensor)

        # -- Logging --
        current_idx = env.current_step
        try:
            date = env.df.index[current_idx]
            open_price = float(env.df['Open'].iloc[current_idx]) if 'Open' in env.df.columns else None
            high_price = float(env.df['High'].iloc[current_idx]) if 'High' in env.df.columns else None
            low_price = float(env.df['Low'].iloc[current_idx]) if 'Low' in env.df.columns else None
            real_close = float(env.df['Close'].iloc[current_idx])
        except Exception:
            date, open_price, high_price, low_price, real_close = None, None, None, None, None

        act_str = ["skip", "buy", "sell"][int(action)]

        # Predicted close: predictor predicts next log-return (if available)
        pred_close = None
        try:
            with torch.no_grad():
                pred_logret = predictor(state_tensor)
                # predictor output shape: (1,1)
                pred_close = float(real_close * np.exp(pred_logret.item())) if real_close is not None else None
        except Exception:
            pred_close = None

        logger.log_step(date, open_price, high_price, low_price, real_close, pred_close, act_str)

        # -- Step --
        next_state, reward, done, _, _ = env.step(action)

        # -- Memory --
        memory_states.append(torch.FloatTensor(state))
        memory_actions.append(torch.tensor(action))
        memory_logprobs.append(log_prob)
        memory_rewards.append(reward)

        state = next_state
        episode_reward += reward

        # -- Update (PPO) --
        if step % 500 == 0:
            old_states = torch.stack(memory_states)
            old_actions = torch.stack(memory_actions)
            old_logprobs = torch.stack(memory_logprobs)

            # Rewards
            rewards = []
            discounted_reward = 0
            for r in reversed(memory_rewards):
                discounted_reward = r + 0.99 * discounted_reward
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            # Optimization
            for _ in range(3):
                logprobs, state_values, dist_entropy = agent.evaluate(old_states, old_actions)
                state_values = torch.squeeze(state_values)
                ratios = torch.exp(logprobs - old_logprobs.detach())
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards) - 0.01 * dist_entropy

                optimizer_rl.zero_grad()
                loss.mean().backward()
                optimizer_rl.step()

            memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []

        if done:
            print(f"🏁 Episode Result: Total Reward: {episode_reward:.2f}")
            results = logger.get_results()
            if not results.empty:
                print(f"   Profit: {results['Profit'].sum():.2f}")
                # Print recent trades with detailed columns
                cols = [c for c in ["Date", "Open", "High", "Low", "Real_Close", "Pred_Close", "Action", "Profit", "Miss_Opportunity"] if c in results.columns]
                try:
                    print("\n   Recent Trades:")
                    print(results[cols].tail(10).to_string(index=False))
                except Exception:
                    # Fallback: print raw tail
                    print(results.tail(10))

            # Reset for next episode
            state, _ = env.reset()
            episode_reward = 0
            torch.save(agent.state_dict(), "kan_agent.pth")
            logger = TradingLogger() # CRITICAL: Reset logger to avoid ghost inventory
def run_backtest():
    # 1. SETUP: Define Split Dates
    TICKER = "SPY"
    # We test on data the model (hopefully) hasn't seen
    TEST_START  = "2024-01-01"
    TEST_END    = "2025-12-01" # Or current date

    print(f"⚔️ REALITY CHECK: Backtesting {TICKER}")
    print(f"   Testing Period:  {TEST_START} -> {TEST_END}")

    # 2. LOAD DATA (TEST SET ONLY)
    df = yf.download(TICKER, start=TEST_START, end=TEST_END, progress=False, auto_adjust=True)

    # Fix MultiIndex (yfinance update)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # ======================================================
    # 3. FEATURE ENGINEERING (MUST MATCH TRAINING EXACTLY)
    # ======================================================
    # A. Basic Features
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Vol_Norm'] = df['Volume'] / df['Volume'].rolling(20).mean()

    # B. Technical Indicators
    # Ensure you have pandas_ta installed (pip install pandas_ta)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)

    # C. Clean & Normalize
    df.dropna(inplace=True)

    # RSI Normalization (0-100 -> 0-1)
    df["RSI_14"] = df["RSI_14"] / 100.0

    # MACD Normalization
    df["MACD_12_26_9"] = (df["MACD_12_26_9"] - df["MACD_12_26_9"].mean()) / (df["MACD_12_26_9"].std() + 1e-7)

    # D. Select Features (The exact 5 columns model expects)
    features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9"]

    # ======================================================
    # 4. PREPARE ENV
    # ======================================================
    # We initialize the env, but we force-feed it our processed TEST data
    env = StockTradingEnv(ticker=TICKER)

    # OVERWRITE the environment data with our new dataframe
    env.df = df
    env.data = df[features].values # <--- This creates the (N, 5) shape
    env.features = features
    env.max_steps = len(env.data) - 1

    # 5. LOAD AGENT
    obs_dim = env.obs_shape # Should be 50 (10 window * 5 features)
    action_dim = env.action_space.n

    # Init Agent
    agent = KANActorCritic(obs_dim, action_dim, hidden_dim=32)

    # Load Weights
    try:
        agent.load_state_dict(torch.load("kan_agent.pth"))
        print("✅ Trained Model Loaded Successfully.")
    except FileNotFoundError:
        print("❌ ERROR: No model found. Please Train and torch.save() first!")
        return
    except RuntimeError as e:
        print(f"❌ SIZE MISMATCH: Your saved model has different dimensions than this code.\n{e}")
        return

    agent.eval()

    # 6. RUN SIMULATION
    state, _ = env.reset()
    done = False

    portfolio_history = []
    price_history = []
    dates = []

    # Initial Capital for Buy & Hold comparison
    initial_balance = env.balance
    initial_price = env.data[env.window_size][0] # Close price at start
    buy_hold_shares = initial_balance / initial_price

    print("\n🚀 Starting Backtest Loop...")

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            action, _, _ = agent.act(state_tensor)

        next_state, reward, done, _, _ = env.step(action)

        current_step = env.current_step
        if current_step < len(env.df):
            current_date = env.df.index[current_step]
            current_price = env.data[current_step][0]

            # Agent Value
            agent_value = env.balance + (env.shares * current_price)
            # Buy & Hold Value
            bh_value = buy_hold_shares * current_price

            portfolio_history.append(agent_value)
            price_history.append(bh_value)
            dates.append(current_date)

        state = next_state

    # 7. VISUALIZE
    portfolio_history = np.array(portfolio_history)
    price_history = np.array(price_history)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, portfolio_history, label='KAN Agent', color='blue', linewidth=2)
    plt.plot(dates, price_history, label='Buy & Hold', color='gray', linestyle='--', alpha=0.7)

    plt.title(f"KAN Agent vs. Buy & Hold ({TICKER})\nOut-of-Sample Test (RSI+MACD)")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 8. FINAL METRICS
    if len(portfolio_history) > 0:
        agent_return = (portfolio_history[-1] - initial_balance) / initial_balance * 100
        bh_return = (price_history[-1] - initial_balance) / initial_balance * 100

        print(f"\n📊 FINAL REPORT")
        print(f"   Agent Return:      {agent_return:.2f}%")
        print(f"   Buy & Hold Return: {bh_return:.2f}%")


if __name__ == "__main__":
    # run_pipeline()
    for i in range(10):
        run_backtest()