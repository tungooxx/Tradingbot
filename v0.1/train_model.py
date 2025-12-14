"""
TRAINING SCRIPT USING DAPGIO_IMPROVED
======================================
This script uses the improved classes from dapgio_improved.py to train your model.

Usage:
    python train_model.py

This will:
1. Pre-train the KAN predictor (supervised learning)
2. Train the RL agent (PPO) with transfer learning
3. Save the model to kan_agent_crypto.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from dapgio_improved import (
    StockTradingEnv,
    KANPredictor,
    KANActorCritic,
    TradingConfig,
    setup_logging,
    TradingLogger
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def prepare_pretraining_data(env):
    """Extract windows (X) and next day Log Returns (y) from environment data
    Includes human strategy features if enabled"""
    X_list = []
    y_list = []

    for i in range(env.window_size, len(env.data) - 1):
        # Use _get_observation to include human strategy features
        obs = env._get_observation(i)
        target = env.data[i][1]  # The Log_Ret of "today"

        X_list.append(obs)
        y_list.append(target)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32).reshape(-1, 1)


def train_model(ticker="ETH-USD", mode="crypto", interval="1h", window_size=30, hidden_dim=32, epochs_pretrain=150, steps_rl=100000):
    """
    Complete training pipeline:
    1. Pre-train predictor (supervised)
    2. Train RL agent (PPO) with transfer learning
    
    Args:
        ticker: Symbol to train on (e.g., "ETH-USD", "BTC-USD", "NVDA")
        mode: "crypto" or "stock"
        interval: "1h", "4h" for crypto; "1d" for stock
        window_size: Lookback window size
        hidden_dim: Hidden dimension for model
        epochs_pretrain: Number of epochs for pre-training
        steps_rl: Number of steps for RL training
    """
    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"Mode: {mode.upper()}")
    print(f"Interval: {interval}")
    print(f"Window Size: {window_size}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load config and set mode
    config = TradingConfig()
    config.mode = mode
    config.interval = interval
    config.ticker = ticker
    
    # Adjust window size based on mode
    if mode == "crypto" and interval == "1h":
        # For hourly: 30 hours = ~1.25 days of data
        window_size = window_size
    elif mode == "crypto" and interval == "4h":
        # For 4h: 30 * 4h = 120 hours = 5 days
        window_size = window_size
    else:
        # For daily: 30 days
        window_size = window_size
    
    # Set model path based on mode
    if mode == "crypto":
        config.model_path = "kan_agent_crypto.pth"
    else:
        config.model_path = "kan_agent_stock.pth"
    
    logger = setup_logging(config)

    # 1. Initialize Environment with human-like strategies
    print("\nInitializing environment...")
    env = StockTradingEnv(ticker=ticker, window_size=window_size, config=config, logger=logger, use_human_strategies=True)
    obs_dim = env.obs_shape
    action_dim = env.action_space.n
    
    print(f"   Human-like strategies: ENABLED")
    print(f"   Observation includes {env.human_features_size} human strategy features")

    print(f"   Observation dimension: {obs_dim}")
    print(f"   Action dimension: {action_dim}")
    print(f"   Data points: {len(env.data)}")

    # ==========================================
    # PHASE 1: PRE-TRAINING (SUPERVISED)
    # ==========================================
    print("\n" + "=" * 60)
    print("🧠 PHASE 1: Pre-training KAN Predictor (Supervised Learning)")
    print("=" * 60)

    # Create dataset
    X_train, y_train = prepare_pretraining_data(env)
    print(f"   Training samples: {len(X_train)}")

    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize Predictor
    predictor = KANPredictor(obs_dim, hidden_dim).to(device)
    optimizer_pred = optim.Adam(predictor.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    # Training loop
    print(f"\n   Training for {epochs_pretrain} epochs...")
    for epoch in range(epochs_pretrain):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer_pred.zero_grad()
            preds = predictor(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer_pred.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"   Epoch {epoch + 1:3d}/{epochs_pretrain} | MSE Loss: {avg_loss:.6f}")

    print("Pre-training complete. Weights primed.")

    # ==========================================
    # PHASE 2: RL TRAINING (PPO)
    # ==========================================
    print("\n" + "=" * 60)
    print("🎮 PHASE 2: Transferring Weights & Starting RL Training (PPO)")
    print("=" * 60)

    # Transfer learned body to RL agent
    agent = KANActorCritic(obs_dim, action_dim, hidden_dim, pretrained_body=predictor.body).to(device)
    # Learning rate - increased for faster convergence (was 0.0001, too low)
    optimizer_rl = optim.Adam(agent.parameters(), lr=0.0003)

    # Reset environment and logger
    logger_trading = TradingLogger()
    state, _ = env.reset()
    episode_reward = 0
    memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
    
    # Track confidence statistics for reporting
    confidence_history = []

    print(f"\n   Training for {steps_rl} steps...")
    print("   (Updates every 500 steps)")

    for step in range(1, steps_rl + 1):
        # Get action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob, _ = agent.act(state_tensor, deterministic=False)  # Use sampling for training

        # Logging
        current_idx = env.current_step
        try:
            date = env.df.index[current_idx]
            open_price = float(env.df['Open'].iloc[current_idx]) if 'Open' in env.df.columns else None
            high_price = float(env.df['High'].iloc[current_idx]) if 'High' in env.df.columns else None
            low_price = float(env.df['Low'].iloc[current_idx]) if 'Low' in env.df.columns else None
            real_close = float(env.df['Close'].iloc[current_idx])
        except Exception as e:
            date, open_price, high_price, low_price, real_close = None, None, None, None, None

        act_str = ["skip", "buy", "sell"][int(action)]

        # Get prediction
        pred_close = None
        try:
            with torch.no_grad():
                pred_logret = predictor(state_tensor)
                pred_close = float(real_close * np.exp(pred_logret.item())) if real_close is not None else None
        except Exception:
            pred_close = None

        logger_trading.log_step(date, open_price, high_price, low_price, real_close, pred_close, act_str)

        # Step environment
        next_state, reward, done, _, _ = env.step(action)
        
        # FIXED: Add confidence bonus to encourage high confidence predictions
        # Get confidence from action probabilities
        with torch.no_grad():
            confidence_probs = agent.get_action_confidence(state_tensor)
            confidence = float(np.max(confidence_probs))  # Max probability = confidence
            
            # Extract individual action probabilities
            hold_prob = float(confidence_probs[0]) if len(confidence_probs) > 0 else 0.0
            buy_prob = float(confidence_probs[1]) if len(confidence_probs) > 1 else 0.0
            sell_prob = float(confidence_probs[2]) if len(confidence_probs) > 2 else 0.0
            
            # Calculate confidence bonus first
            if confidence > 0.70:  # High confidence (>70%)
                confidence_bonus = 0.5  # Significant bonus
            elif confidence > 0.60:  # Good confidence (60-70%)
                confidence_bonus = 0.2
            elif confidence > 0.50:  # Decent confidence (50-60%)
                confidence_bonus = 0.1
            elif confidence < 0.35:  # Very low confidence (<35%)
                confidence_bonus = -0.1  # Small penalty
            else:
                confidence_bonus = 0.0  # Neutral for 35-50%
            
            # Get key features from observation for decision-making info
            obs_array = state_tensor.cpu().numpy().flatten()
            human_features = obs_array[env.window_size * 5:] if len(obs_array) > env.window_size * 5 else []
            
            # Extract actual market features from DataFrame (more accurate than observation array)
            try:
                current_idx = env.current_step
                if current_idx < len(env.df):
                    df_row = env.df.iloc[current_idx]
                    last_close = float(df_row['Close']) if 'Close' in df_row else real_close if real_close else 0.0
                    last_logret = float(df_row['Log_Ret']) if 'Log_Ret' in df_row else 0.0
                    last_volnorm = float(df_row['Vol_Norm']) if 'Vol_Norm' in df_row else 0.0
                    last_rsi = float(df_row['RSI_14']) if 'RSI_14' in df_row else 0.0
                    last_macd = float(df_row['MACD_12_26_9']) if 'MACD_12_26_9' in df_row else 0.0
                else:
                    last_close = real_close if real_close else 0.0
                    last_logret = last_volnorm = last_rsi = last_macd = 0.0
            except Exception:
                last_close = real_close if real_close else 0.0
                last_logret = last_volnorm = last_rsi = last_macd = 0.0
            
            # Track confidence for statistics
            confidence_history.append(confidence)
            if len(confidence_history) > 500:
                confidence_history.pop(0)  # Keep only last 500
            
            reward += confidence_bonus

        # Store in memory
        memory_states.append(torch.FloatTensor(state))
        memory_actions.append(torch.tensor(action))
        memory_logprobs.append(log_prob)
        memory_rewards.append(reward)

        state = next_state
        episode_reward += reward

        # PPO Update
        if step % 500 == 0:
            old_states = torch.stack(memory_states).to(device)
            old_actions = torch.stack(memory_actions).to(device)
            old_logprobs = torch.stack(memory_logprobs).to(device)

            # Calculate discounted rewards
            rewards = []
            discounted_reward = 0
            for r in reversed(memory_rewards):
                discounted_reward = r + 0.99 * discounted_reward
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
            
            # FIXED: Don't normalize rewards - let model learn absolute profit
            # Normalization causes training/backtest mismatch
            # Instead, scale rewards to reasonable range (divide by 100 to get percentage-like scale)
            rewards = rewards / 100.0  # Scale to percentage-like values (e.g., +5.0 reward = +5% profit)

            # PPO optimization
            for _ in range(3):
                logprobs, state_values, dist_entropy = agent.evaluate(old_states, old_actions)
                state_values = torch.squeeze(state_values)
                ratios = torch.exp(logprobs - old_logprobs.detach())
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                # FIXED: Reduced entropy coefficient to increase confidence
                # Lower entropy = less exploration = higher confidence in predictions
                # Increased from 0.02 to 0.005 to encourage more confident decisions
                loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards) - 0.005 * dist_entropy

                optimizer_rl.zero_grad()
                loss.mean().backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                optimizer_rl.step()

            # Get last action before clearing memory
            last_action_str = "skip"
            if len(memory_actions) > 0:
                last_action = int(memory_actions[-1].item())
                last_action_str = ["skip", "buy", "sell"][last_action]
            
            # Clear memory
            memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []

            # Calculate confidence statistics from history
            if len(confidence_history) > 0:
                conf_max = np.max(confidence_history)
                conf_mean = np.mean(confidence_history)
                conf_min = np.min(confidence_history)
            else:
                conf_max = conf_mean = conf_min = 0.0
            
            # Get current step's detailed info for display
            with torch.no_grad():
                current_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                current_conf_probs = agent.get_action_confidence(current_state_tensor)
                current_confidence = float(np.max(current_conf_probs))
                current_hold_prob = float(current_conf_probs[0]) if len(current_conf_probs) > 0 else 0.0
                current_buy_prob = float(current_conf_probs[1]) if len(current_conf_probs) > 1 else 0.0
                current_sell_prob = float(current_conf_probs[2]) if len(current_conf_probs) > 2 else 0.0
                
                # Get current market features
                current_obs_array = current_state_tensor.cpu().numpy().flatten()
                current_human_features = current_obs_array[env.window_size * 5:] if len(current_obs_array) > env.window_size * 5 else []
                
                try:
                    current_idx = env.current_step
                    if current_idx < len(env.df):
                        df_row = env.df.iloc[current_idx]
                        current_close = float(df_row['Close']) if 'Close' in df_row else 0.0
                        current_logret = float(df_row['Log_Ret']) if 'Log_Ret' in df_row else 0.0
                        current_volnorm = float(df_row['Vol_Norm']) if 'Vol_Norm' in df_row else 0.0
                        current_rsi = float(df_row['RSI_14']) if 'RSI_14' in df_row else 0.0
                        current_macd = float(df_row['MACD_12_26_9']) if 'MACD_12_26_9' in df_row else 0.0
                    else:
                        current_close = current_logret = current_volnorm = current_rsi = current_macd = 0.0
                except Exception:
                    current_close = current_logret = current_volnorm = current_rsi = current_macd = 0.0
            
            # Print detailed info every 500 steps
            price_display = real_close if real_close else current_close
            
            print(f"\n{'='*80}")
            print(f"Step {step:5d} | Action: {last_action_str.upper():4s} | Price: ${price_display:.2f}")
            print(f"{'-'*80}")
            print(f"CONFIDENCE PROBABILITIES (Current):")
            print(f"  HOLD: {current_hold_prob:.2%} | BUY: {current_buy_prob:.2%} | SELL: {current_sell_prob:.2%} | Max: {current_confidence:.2%}")
            print(f"{'-'*80}")
            print(f"CONFIDENCE STATISTICS (Last 500 steps):")
            print(f"  Max: {conf_max:.2%} | Mean: {conf_mean:.2%} | Min: {conf_min:.2%}")
            print(f"{'-'*80}")
            print(f"MARKET FEATURES (Last Bar):")
            print(f"  Close: ${current_close:.2f} | Log Return: {current_logret:.4f} | Vol Norm: {current_volnorm:.2f}")
            print(f"  RSI: {current_rsi:.2f} | MACD: {current_macd:.4f}")
            if len(current_human_features) > 0:
                current_price_rejection = current_human_features[8] if len(current_human_features) > 8 else 0.0
                print(f"{'-'*80}")
                print(f"HUMAN STRATEGY FEATURES:")
                print(f"  Should Wait: {current_human_features[0]:.0f} | Hesitation: {current_human_features[1]:.0f}")
                print(f"  Rapid Drop: {current_human_features[2]:.0f} | Unusual Vol: {current_human_features[3]:.0f}")
                print(f"  Price Gap: {current_human_features[4]:.0f} | High Volatility: {current_human_features[5]:.0f}")
                print(f"  Flash Crash: {current_human_features[6]:.0f} | Rapid Pump: {current_human_features[7]:.0f}")
                print(f"  Price Rejection: {current_price_rejection:.0f}")
            print(f"{'-'*80}")
            print(f"REWARD: {reward:.4f} | Episode Reward: {episode_reward:.2f}")
            print(f"{'='*80}\n")

        # Episode finished
        if done:
            results = logger_trading.get_results()
            if not results.empty:
                total_profit = results['Profit'].sum()
                print(f"\n   🏁 Episode Complete:")
                print(f"      Total Reward: {episode_reward:.2f}")
                print(f"      Total Profit: ${total_profit:.2f}")

            # Reset for next episode
            state, _ = env.reset()
            episode_reward = 0
            logger_trading = TradingLogger()

    # ==========================================
    # SAVE MODEL
    # ==========================================
    model_path = config.model_path
    torch.save(agent.state_dict(), model_path)
    print(f"\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print("=" * 60)

    return agent, predictor


if __name__ == "__main__":
    # Configuration
    MODE = "crypto"  # "crypto" or "stock"
    INTERVAL = "1h"  # "1h", "4h" for crypto; "1d" for stock
    TICKER = "ETH-USD" if MODE == "crypto" else "QQQ"  # Change to your ticker
    WINDOW_SIZE = 30
    HIDDEN_DIM = 32
    EPOCHS_PRETRAIN = 150  # Increased for better pre-training
    STEPS_RL = 100000  # Increased for better convergence (was 50000)

    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    print(f"Mode: {MODE.upper()}")
    print(f"Interval: {INTERVAL}")
    print(f"Ticker: {TICKER}")
    print(f"Pre-training epochs: {EPOCHS_PRETRAIN}")
    print(f"RL training steps: {STEPS_RL}")
    print("=" * 60 + "\n")

    try:
        agent, predictor = train_model(
            ticker=TICKER,
            mode=MODE,
            interval=INTERVAL,
            window_size=WINDOW_SIZE,
            hidden_dim=HIDDEN_DIM,
            epochs_pretrain=EPOCHS_PRETRAIN,
            steps_rl=STEPS_RL
        )
        print("\nTraining completed successfully!")
        print("Next steps:")
        print("   1. Run predict.py to test predictions")
        print("   2. Use paper_trading.py for manual paper trading")
        print("   3. After validation, integrate with exchange API")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback

        traceback.print_exc()

