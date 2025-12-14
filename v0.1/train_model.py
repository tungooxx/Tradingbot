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
    setup_logging
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
    optimizer_rl = optim.Adam(agent.parameters(), lr=0.0003)

    # Reset environment
    state, _ = env.reset()
    memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []

    for step in range(1, steps_rl + 1):
        # Get action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob, _ = agent.act(state_tensor, deterministic=False)

        # Step environment
        next_state, reward, done, _, _ = env.step(action)

        # Store in memory
        memory_states.append(torch.FloatTensor(state))
        memory_actions.append(torch.tensor(action))
        memory_logprobs.append(log_prob)
        memory_rewards.append(reward)

        state = next_state

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
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            # PPO optimization
            for _ in range(3):
                logprobs, state_values, dist_entropy = agent.evaluate(old_states, old_actions)
                state_values = torch.squeeze(state_values)
                ratios = torch.exp(logprobs - old_logprobs.detach())
                advantages = rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards) - 0.02 * dist_entropy

                optimizer_rl.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                optimizer_rl.step()

            # Clear memory
            memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []

        # Episode finished
        if done:
            state, _ = env.reset()

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

