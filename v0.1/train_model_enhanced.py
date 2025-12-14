"""
ENHANCED TRAINING WITH MULTI-AGENT & IMPROVEMENTS
=================================================

Features:
1. Mixture of Experts (MoE) - 4 specialized agents
2. Enhanced PPO with GAE, gradient clipping
3. Risk-adjusted rewards
4. Regime detection
5. Load balancing for experts

Usage:
    python train_model_enhanced.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from dapgio_improved import (
    StockTradingEnv, 
    KANPredictor,
    TradingConfig,
    setup_logging,
    TradingLogger
)
from dapgio_enhanced import (
    MoEActorCritic,
    EnhancedTradingEnv,
    train_ppo_enhanced
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def prepare_pretraining_data(env):
    """Extract windows (X) and next period Log Returns (y)
    Includes human strategy features if enabled"""
    X_list = []
    y_list = []

    for i in range(env.window_size, len(env.data) - 1):
        # Use _get_observation to include human strategy features
        obs = env._get_observation(i)
        target = env.data[i][1]  # Log_Ret

        X_list.append(obs)
        y_list.append(target)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32).reshape(-1, 1)

def train_enhanced_model(ticker="ETH-USD", mode="crypto", interval="1h", 
                        window_size=30, hidden_dim=32, epochs_pretrain=100, 
                        steps_rl=50000, use_moe=True):
    """
    Enhanced training pipeline with MoE architecture
    """
    print("=" * 60)
    print("ENHANCED TRAINING PIPELINE (MoE Architecture)")
    print("=" * 60)
    print(f"Ticker: {ticker}")
    print(f"Mode: {mode.upper()}")
    print(f"Interval: {interval}")
    print(f"Window Size: {window_size}")
    print(f"Use MoE: {use_moe}")
    print(f"Device: {device}")
    print("=" * 60)
    
    # Load config
    config = TradingConfig()
    config.mode = mode
    config.interval = interval
    config.ticker = ticker
    
    if mode == "crypto":
        config.model_path = "kan_agent_crypto_moe.pth"
    else:
        config.model_path = "kan_agent_stock_moe.pth"
    
    logger = setup_logging(config)
    
    # Initialize Environment
    print("\n[1/2] Initializing environment...")
    if use_moe:
        env = EnhancedTradingEnv(ticker=ticker, window_size=window_size, 
                                 config=config, logger=logger, use_moe=True)
    else:
        env = StockTradingEnv(ticker=ticker, window_size=window_size, 
                              config=config, logger=logger, use_human_strategies=True)
    
    obs_dim = env.obs_shape
    action_dim = env.action_space.n
    
    print(f"   Observation dimension: {obs_dim}")
    print(f"   Action dimension: {action_dim}")
    print(f"   Data points: {len(env.data)}")
    
    # ==========================================
    # PHASE 1: PRE-TRAINING (SUPERVISED)
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 1: Pre-training KAN Predictor")
    print("=" * 60)
    
    X_train, y_train = prepare_pretraining_data(env)
    print(f"   Training samples: {len(X_train)}")
    
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    predictor = KANPredictor(obs_dim, hidden_dim).to(device)
    optimizer_pred = optim.Adam(predictor.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
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

        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"   Epoch {epoch+1:3d}/{epochs_pretrain} | MSE Loss: {avg_loss:.6f}")

    print("[OK] Pre-training complete.")
    
    # ==========================================
    # PHASE 2: RL TRAINING (Enhanced PPO)
    # ==========================================
    print("\n" + "=" * 60)
    print("PHASE 2: Enhanced RL Training (MoE + PPO)")
    print("=" * 60)
    
    # Create agent (MoE or single)
    if use_moe:
        print("   Using Mixture of Experts architecture (4 experts)")
        agent = MoEActorCritic(obs_dim, action_dim, num_experts=4, hidden_dim=hidden_dim).to(device)
        
        # Transfer learned body to experts (optional)
        for expert in agent.experts:
            expert.body = predictor.body  # Share pre-trained body
    else:
        from dapgio_improved import KANActorCritic
        agent = KANActorCritic(obs_dim, action_dim, hidden_dim, pretrained_body=predictor.body).to(device)
    
    optimizer_rl = optim.Adam(agent.parameters(), lr=0.0005)
    
    print(f"\n   Training for {steps_rl} steps...")
    print("   (Updates every 500 steps with GAE)")
    
    # Enhanced PPO training
    train_ppo_enhanced(
        agent=agent,
        env=env,
        optimizer=optimizer_rl,
        steps=steps_rl,
        update_freq=500,
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95
    )
    
    # ==========================================
    # SAVE MODEL
    # ==========================================
    model_path = config.model_path
    torch.save(agent.state_dict(), model_path)
    print(f"\n[OK] Training complete!")
    print(f"[SAVED] Model saved to: {model_path}")
    print("=" * 60)
    
    return agent, predictor

if __name__ == "__main__":
    # Configuration
    MODE = "crypto"
    INTERVAL = "1h"
    TICKER = "ETH-USD"
    WINDOW_SIZE = 30
    HIDDEN_DIM = 32
    EPOCHS_PRETRAIN = 100
    STEPS_RL = 50000
    USE_MOE = True  # Use Mixture of Experts
    
    print("\n" + "=" * 60)
    print("ENHANCED MODEL TRAINING")
    print("=" * 60)
    print(f"Mode: {MODE.upper()}")
    print(f"Interval: {INTERVAL}")
    print(f"Ticker: {TICKER}")
    print(f"Architecture: {'MoE (4 Experts)' if USE_MOE else 'Single Agent'}")
    print(f"Pre-training epochs: {EPOCHS_PRETRAIN}")
    print(f"RL training steps: {STEPS_RL}")
    print("=" * 60 + "\n")
    
    try:
        agent, predictor = train_enhanced_model(
            ticker=TICKER,
            mode=MODE,
            interval=INTERVAL,
            window_size=WINDOW_SIZE,
            hidden_dim=HIDDEN_DIM,
            epochs_pretrain=EPOCHS_PRETRAIN,
            steps_rl=STEPS_RL,
            use_moe=USE_MOE
        )
        print("\n[SUCCESS] Enhanced training completed successfully!")
        print("Next steps:")
        print("   1. Compare performance vs single agent")
        print("   2. Test with predict.py")
        print("   3. Paper trade for validation")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()

