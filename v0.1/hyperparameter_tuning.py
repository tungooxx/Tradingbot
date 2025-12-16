"""
HYPERPARAMETER TUNING SCRIPT
=============================
Trains multiple models with different hyperparameters, backtests each,
and identifies the best configuration.

Usage:
    python hyperparameter_tuning.py --ticker QQQ --mode stock --interval 1d
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import itertools

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dapgio_improved import (
    StockTradingEnv,
    KANPredictor,
    KANActorCritic,
    TradingConfig,
    setup_logging
)
# Import backtest function - add parent directory to path first
sys.path.insert(0, str(Path(__file__).parent))
from backtest_confidence import backtest_confidence_threshold

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
        target = env.data[i][1]  # Log_Ret

        X_list.append(obs)
        y_list.append(target)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32).reshape(-1, 1)


def train_model_with_params(
    ticker: str,
    mode: str,
    interval: str,
    window_size: int,
    hidden_dim: int,
    epochs_pretrain: int,
    steps_rl: int,
    learning_rate: float,
    entropy_coef: float,
    update_freq: int,
    model_name: str
) -> str:
    """
    Train model with specific hyperparameters
    
    Returns:
        Path to saved model
    """
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Entropy Coef: {entropy_coef}")
    print(f"  Hidden Dim: {hidden_dim}")
    print(f"  Steps RL: {steps_rl}")
    print(f"  Update Freq: {update_freq}")
    print(f"{'='*60}")
    
    # Load config
    config = TradingConfig()
    config.mode = mode
    config.interval = interval
    config.ticker = ticker
    config.model_path = model_name
    
    logger = setup_logging(config)
    
    # Initialize environment with human strategies enabled
    env = StockTradingEnv(ticker=ticker, window_size=window_size, config=config, logger=logger, use_human_strategies=True)
    obs_dim = env.obs_shape
    action_dim = env.action_space.n
    
    # 1. Pre-train predictor
    print("\n[1/2] Pre-training predictor...")
    predictor = KANPredictor(obs_dim, hidden_dim).to(device)
    X, y = prepare_pretraining_data(env)
    
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X).to(device),
        torch.FloatTensor(y).to(device)
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    optimizer_pred = optim.Adam(predictor.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs_pretrain):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer_pred.zero_grad()
            pred = predictor(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer_pred.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"   Epoch {epoch + 1:3d}/{epochs_pretrain} | MSE Loss: {avg_loss:.6f}")
    
    # 2. Train RL agent
    print(f"\n[2/2] Training RL agent ({steps_rl} steps)...")
    agent = KANActorCritic(obs_dim, action_dim, hidden_dim, pretrained_body=predictor.body).to(device)
    optimizer_rl = optim.Adam(agent.parameters(), lr=learning_rate)
    
    state, _ = env.reset()
    memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
    episode_reward = 0
    
    for step in range(1, steps_rl + 1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action, log_prob, _ = agent.act(state_tensor, deterministic=False)
        
        next_state, reward, done, truncated, info = env.step(action)
        
        memory_states.append(torch.FloatTensor(state))
        memory_actions.append(torch.tensor(action))
        memory_logprobs.append(log_prob)
        memory_rewards.append(reward)
        
        state = next_state
        episode_reward += reward
        
        # PPO Update
        if step % update_freq == 0:
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
                loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards) - entropy_coef * dist_entropy
                
                optimizer_rl.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer_rl.step()
            
            # Clear memory
            memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
            
            if step % (update_freq * 2) == 0:
                print(f"   Step {step:5d}/{steps_rl} | Episode Reward: {episode_reward:.2f}")
        
        if done or truncated:
            state, _ = env.reset()
            episode_reward = 0
    
    # Save model
    model_path = f"models/{model_name}"
    os.makedirs("models", exist_ok=True)
    torch.save(agent.state_dict(), model_path)
    print(f"\nModel saved: {model_path}")
    
    return model_path


def evaluate_model(
    model_path: str,
    ticker: str,
    mode: str,
    interval: str,
    window_size: int = 30,
    hidden_dim: int = 32
) -> Dict:
    """Backtest model and return performance metrics"""
    
    # Extract hidden_dim from model name if not provided
    if hidden_dim == 32:  # Default, try to extract from filename
        import re
        match = re.search(r'_hd(\d+)\.pth', model_path)
        if match:
            hidden_dim = int(match.group(1))
    
    print(f"\nEvaluating: {model_path} (hidden_dim={hidden_dim})")
    
    # Test multiple confidence thresholds (include lower thresholds for low-confidence models)
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
    best_result = None
    best_threshold = None
    best_return = -float('inf')
    
    for threshold in thresholds:
        result = backtest_confidence_threshold(
            model_path, ticker, mode, interval, window_size, threshold, hidden_dim
        )
        
        if result and result['total_return_pct'] > best_return:
            best_return = result['total_return_pct']
            best_result = result
            best_threshold = threshold
    
    if best_result:
        best_result['best_threshold'] = best_threshold
        return best_result
    else:
        return {
            'total_return_pct': -100.0,
            'win_rate': 0.0,
            'total_trades': 0,
            'signals_executed': 0,
            'best_threshold': None
        }


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning')
    parser.add_argument('--ticker', type=str, default='QQQ', help='Ticker symbol')
    parser.add_argument('--mode', type=str, default='stock', choices=['stock', 'crypto'], help='Trading mode')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval')
    parser.add_argument('--window-size', type=int, default=30, help='Window size')
    parser.add_argument('--max-configs', type=int, default=20, help='Max configurations to test')
    
    args = parser.parse_args()
    
    print("="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    print(f"Ticker: {args.ticker}")
    print(f"Mode: {args.mode}")
    print(f"Interval: {args.interval}")
    print("="*60)
    
    # Define hyperparameter grid
    hyperparams = {
        'learning_rate': [0.0001, 0.0003, 0.0005],
        'entropy_coef': [0.01, 0.03, 0.05],
        'hidden_dim': [32, 64],
        'steps_rl': [50000, 100000],
        'update_freq': [250, 500],
        'epochs_pretrain': [100, 150]
    }
    
    # Generate all combinations
    keys = hyperparams.keys()
    values = hyperparams.values()
    combinations = list(itertools.product(*values))
    
    # Limit number of configs
    if len(combinations) > args.max_configs:
        print(f"\nToo many combinations ({len(combinations)}). Limiting to {args.max_configs}...")
        # Sample evenly
        indices = np.linspace(0, len(combinations)-1, args.max_configs, dtype=int)
        combinations = [combinations[i] for i in indices]
    
    print(f"\nTesting {len(combinations)} configurations...")
    
    results = []
    
    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))
        
        # Create model name
        model_name = f"tuned_{args.ticker}_{idx:02d}_lr{params['learning_rate']}_ent{params['entropy_coef']}_hd{params['hidden_dim']}.pth"
        
        print(f"\n{'='*60}")
        print(f"Configuration {idx}/{len(combinations)}")
        print(f"{'='*60}")
        
        try:
            # Train model
            model_path = train_model_with_params(
                ticker=args.ticker,
                mode=args.mode,
                interval=args.interval,
                window_size=args.window_size,
                hidden_dim=params['hidden_dim'],
                epochs_pretrain=params['epochs_pretrain'],
                steps_rl=params['steps_rl'],
                learning_rate=params['learning_rate'],
                entropy_coef=params['entropy_coef'],
                update_freq=params['update_freq'],
                model_name=model_name
            )
            
            # Evaluate model
            eval_result = evaluate_model(
                model_path, args.ticker, args.mode, args.interval, args.window_size
            )
            
            # Combine results
            result = {
                'config_id': idx,
                'model_path': model_path,
                'model_name': model_name,
                **params,
                **eval_result
            }
            
            results.append(result)
            
            print(f"\nResults:")
            print(f"  Return: {result['total_return_pct']:.2f}%")
            print(f"  Win Rate: {result['win_rate']:.1f}%")
            print(f"  Trades: {result['total_trades']}")
            print(f"  Best Threshold: {result['best_threshold']:.0%}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            continue
    
    # Find best configuration
    if not results:
        print("\nNo results to compare!")
        return
    
    # Sort by total return
    results.sort(key=lambda x: x['total_return_pct'], reverse=True)
    
    # Print summary
    print(f"\n{'='*60}")
    print("HYPERPARAMETER TUNING RESULTS")
    print(f"{'='*60}\n")
    
    print(f"{'Rank':<6} {'Return%':<10} {'Win%':<8} {'Trades':<8} {'LR':<8} {'Ent':<8} {'HD':<6} {'Steps':<8} {'Model':<30}")
    print("-" * 100)
    
    for rank, result in enumerate(results[:10], 1):  # Top 10
        print(f"{rank:<6} "
              f"{result['total_return_pct']:>8.2f}% "
              f"{result['win_rate']:>6.1f}% "
              f"{result['total_trades']:>6} "
              f"{result['learning_rate']:>8.4f} "
              f"{result['entropy_coef']:>8.2f} "
              f"{result['hidden_dim']:>4} "
              f"{result['steps_rl']:>8} "
              f"{result['model_name']:<30}")
    
    # Best configuration
    best = results[0]
    print(f"\n{'='*60}")
    print("BEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {best['model_name']}")
    print(f"Path: {best['model_path']}")
    print(f"\nHyperparameters:")
    print(f"  Learning Rate: {best['learning_rate']}")
    print(f"  Entropy Coef: {best['entropy_coef']}")
    print(f"  Hidden Dim: {best['hidden_dim']}")
    print(f"  Steps RL: {best['steps_rl']}")
    print(f"  Update Freq: {best['update_freq']}")
    print(f"  Epochs Pretrain: {best['epochs_pretrain']}")
    print(f"\nPerformance:")
    print(f"  Total Return: {best['total_return_pct']:.2f}%")
    print(f"  Win Rate: {best['win_rate']:.1f}%")
    print(f"  Total Trades: {best['total_trades']}")
    print(f"  Signals Executed: {best['signals_executed']}")
    print(f"  Best Threshold: {best['best_threshold']:.0%}")
    
    # Save results to JSON
    results_file = f"hyperparameter_results_{args.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Copy best model to main location
    best_model_name = f"kan_agent_{args.mode}_best.pth"
    import shutil
    shutil.copy(best['model_path'], best_model_name)
    print(f"\nBest model copied to: {best_model_name}")
    print(f"\nUse this model for trading:")
    print(f"  python alpaca_paper_trading.py --ticker {args.ticker} --model {best_model_name} --mode {args.mode} --interval {args.interval}")


if __name__ == "__main__":
    main()

