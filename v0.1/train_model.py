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
from trading_constraints_env import create_constrained_env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def prepare_pretraining_data(env):
    """Extract windows (X) and next day Log Returns (y) from environment data"""
    X_list = []
    y_list = []

    for i in range(env.window_size, len(env.data) - 1):
        window = env.data[i - env.window_size: i]  # The past
        target = env.data[i][1]  # The Log_Ret of "today"

        X_list.append(window.flatten())
        y_list.append(target)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32).reshape(-1, 1)


def train_model(ticker="ETH-USD", mode="crypto", interval="1h", window_size=30, hidden_dim=128, 
                epochs_pretrain=200, steps_rl=150000, use_trading_constraints: bool = True,
                min_confidence: float = 0.50, max_positions: int = 3, position_size_pct: float = 0.10,
                use_enhanced_arch: bool = True, confidence_temperature: float = 0.8):
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

    # 1. Initialize Environment
    print("\nInitializing environment...")
    if use_trading_constraints:
        print("   Using trading constraints (matches alpaca_paper_trading.py):")
        print(f"      Min confidence: {min_confidence:.0%}")
        print(f"      Max positions: {max_positions}")
        print(f"      Position size: {position_size_pct:.0%}")
        env = create_constrained_env(
            ticker=ticker,
            window_size=window_size,
            config=config,
            min_confidence=min_confidence,
            max_positions=max_positions,
            position_size_pct=position_size_pct,
            logger=logger
        )
    else:
        env = StockTradingEnv(ticker=ticker, window_size=window_size, config=config, logger=logger)
    
    obs_dim = env.obs_shape
    action_dim = env.action_space.n

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

    # Initialize Predictor with enhanced architecture
    predictor = KANPredictor(obs_dim, hidden_dim, use_enhanced=use_enhanced_arch).to(device)
    
    # Improved optimizer with weight decay for regularization
    optimizer_pred = optim.AdamW(predictor.parameters(), lr=0.0001, weight_decay=1e-5)
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

    # Transfer learned body to RL agent with enhanced architecture
    agent = KANActorCritic(
        obs_dim, 
        action_dim, 
        hidden_dim, 
        pretrained_body=predictor.body,
        use_enhanced=use_enhanced_arch,
        confidence_temperature=confidence_temperature
    ).to(device)
    
    # Improved optimizer with weight decay and learning rate scheduling
    optimizer_rl = optim.AdamW(agent.parameters(), lr=0.0003, weight_decay=1e-5)
    
    # Learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_rl, T_max=steps_rl, eta_min=1e-6)

    # Reset environment and logger
    logger_trading = TradingLogger()
    state, _ = env.reset()
    # Initialize position tracking for stop loss/take profit detection
    if hasattr(env, 'shares'):
        env._prev_shares = env.shares
    # DEBUG: Ensure balance tracking is correct
    episode_start_balance = env.balance  # Use actual balance after reset, not initial_balance
    if abs(episode_start_balance - env.initial_balance) > 1e-6:
        logger.warning(f"Balance mismatch after reset: balance={env.balance}, initial_balance={env.initial_balance}")
    episode_reward = 0
    memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []
    
    # Track reward breakdown for analysis
    reward_breakdown_stats = {
        'hold_count': 0,
        'buy_count': 0,
        'sell_count': 0,
        'hold_reward_total': 0.0,
        'buy_reward_total': 0.0,
        'sell_reward_total': 0.0,
        'holding_reward_total': 0.0,
        'stop_loss_reward_total': 0.0,
        'take_profit_reward_total': 0.0
    }
    
    # DEBUG: Track requested vs executed actions
    debug_stats = {
        'requested_actions': {'HOLD': 0, 'BUY': 0, 'SELL': 0},
        'executed_actions': {'HOLD': 0, 'BUY': 0, 'SELL': 0},
        'blocked_actions': 0,
        'missing_reward_breakdown': 0,
        'logger_actions': {'HOLD': 0, 'BUY': 0, 'SELL': 0},
        'mismatch_count': 0
    }
    
    # Track cumulative totals across all episodes
    total_cumulative_reward = 0.0
    total_cumulative_profit = 0.0

    print(f"\n   Training for {steps_rl} steps...")
    print("   (Updates every 500 steps, cumulative stats every 10000 steps)")

    for step in range(1, steps_rl + 1):
        # Get action and confidence
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Validate state tensor for NaN/Inf values
        if torch.isnan(state_tensor).any() or torch.isinf(state_tensor).any():
            logger.warning(f"Step {step}: Invalid state detected (NaN/Inf), resetting environment")
            state, _ = env.reset()
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            episode_reward = 0
            continue
        
        try:
            action, log_prob, _ = agent.act(state_tensor, deterministic=False)  # Use sampling for training
        except Exception as e:
            logger.error(f"Step {step}: Error in agent.act(): {e}")
            logger.warning(f"Resetting environment due to agent error")
            state, _ = env.reset()
            episode_reward = 0
            continue
        
        # Get action confidence for constrained environment
        action_confidence = None
        if use_trading_constraints:
            with torch.no_grad():
                confidence_probs = agent.get_action_confidence(state_tensor)
                action_confidence = float(np.max(confidence_probs))

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
        requested_action = act_str.upper()

        # DEBUG: Track requested action
        if requested_action == "SKIP":
            debug_stats['requested_actions']['HOLD'] += 1
        elif requested_action == "BUY":
            debug_stats['requested_actions']['BUY'] += 1
        elif requested_action == "SELL":
            debug_stats['requested_actions']['SELL'] += 1

        # Get prediction
        pred_close = None
        try:
            with torch.no_grad():
                pred_logret = predictor(state_tensor)
                pred_close = float(real_close * np.exp(pred_logret.item())) if real_close is not None else None
        except Exception:
            pred_close = None

        # Step environment FIRST to get executed action
        # Step environment (with confidence if using constraints)
        if use_trading_constraints:
            next_state, reward, done, _, info = env.step(action, action_confidence=action_confidence)
            # Log if action was blocked
            if 'action_blocked' in info:
                debug_stats['blocked_actions'] += 1
                logger.debug(f"Step {step}: Action {action} blocked: {info.get('action_blocked')}")
        else:
            next_state, reward, done, _, info = env.step(action)
        
        # DEBUG: Determine executed action from info
        executed_action_str = requested_action  # Default to requested
        if 'reward_breakdown' in info:
            breakdown = info['reward_breakdown']
            executed_action_str = breakdown.get('action_type', requested_action)
        elif 'executed_action' in info:
            executed_action_str = info['executed_action'].upper()
        elif 'action_blocked' in info:
            # If blocked, executed action is likely HOLD
            executed_action_str = "HOLD"
        
        # DEBUG: Check if position was closed by stop loss/take profit
        # This happens when shares go from >0 to 0 but action wasn't SELL
        position_closed_by_sl_tp = False
        if hasattr(env, 'shares'):
            # Check if we had shares before step (we need to track this)
            if not hasattr(env, '_prev_shares'):
                env._prev_shares = env.shares
            # If shares went from >0 to 0, and action wasn't SELL, it was closed by SL/TP
            if env._prev_shares > 0 and env.shares == 0 and executed_action_str != "SELL":
                position_closed_by_sl_tp = True
                executed_action_str = "SELL"  # Log as SELL for logger tracking
                logger.debug(f"Step {step}: Position closed by stop loss/take profit, logging as SELL")
            env._prev_shares = env.shares
        
        # DEBUG: Track executed action
        if executed_action_str == "HOLD":
            debug_stats['executed_actions']['HOLD'] += 1
        elif executed_action_str == "BUY":
            debug_stats['executed_actions']['BUY'] += 1
        elif executed_action_str == "SELL":
            debug_stats['executed_actions']['SELL'] += 1
        
        # DEBUG: Check for mismatch
        if requested_action != executed_action_str and not position_closed_by_sl_tp:
            debug_stats['mismatch_count'] += 1
            if step % 100 == 0:  # Log every 100 steps to avoid spam
                logger.debug(f"Step {step}: Action mismatch - Requested: {requested_action}, Executed: {executed_action_str}")
        
        # NOW log the executed action (not the requested one)
        # If position was closed by SL/TP, log as SELL so logger can match the BUY
        logger_trading.log_step(date, open_price, high_price, low_price, real_close, pred_close, executed_action_str.lower())
        
        # DEBUG: Track what was logged
        logged_action = executed_action_str
        if logged_action == "HOLD":
            debug_stats['logger_actions']['HOLD'] += 1
        elif logged_action == "BUY":
            debug_stats['logger_actions']['BUY'] += 1
        elif logged_action == "SELL":
            debug_stats['logger_actions']['SELL'] += 1
        
        # Track reward breakdown
        if 'reward_breakdown' in info:
            breakdown = info['reward_breakdown']
            action_type = breakdown['action_type']
            if action_type == 'HOLD':
                reward_breakdown_stats['hold_count'] += 1
                reward_breakdown_stats['hold_reward_total'] += breakdown.get('action_reward', 0.0)
            elif action_type == 'BUY':
                reward_breakdown_stats['buy_count'] += 1
                reward_breakdown_stats['buy_reward_total'] += breakdown.get('action_reward', 0.0)
            elif action_type == 'SELL':
                reward_breakdown_stats['sell_count'] += 1
                reward_breakdown_stats['sell_reward_total'] += breakdown.get('action_reward', 0.0)
            
            reward_breakdown_stats['holding_reward_total'] += breakdown.get('holding_reward', 0.0)
            reward_breakdown_stats['stop_loss_reward_total'] += breakdown.get('stop_loss_reward', 0.0)
            reward_breakdown_stats['take_profit_reward_total'] += breakdown.get('take_profit_reward', 0.0)
        else:
            # DEBUG: Track missing reward_breakdown
            debug_stats['missing_reward_breakdown'] += 1
            if step % 100 == 0:  # Log every 100 steps to avoid spam
                logger.warning(f"Step {step}: Missing 'reward_breakdown' in info dict! "
                             f"Info keys: {list(info.keys()) if info else 'None'}")

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
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            # PPO optimization
            for _ in range(3):
                try:
                    logprobs, state_values, dist_entropy = agent.evaluate(old_states, old_actions)
                    
                    # Check for NaN/Inf in model outputs
                    if torch.isnan(logprobs).any() or torch.isnan(state_values).any() or torch.isnan(dist_entropy).any():
                        logger.warning(f"Step {step}: NaN detected in model outputs, skipping update")
                        break
                    
                    state_values = torch.squeeze(state_values)
                    ratios = torch.exp(logprobs - old_logprobs.detach())
                    advantages = rewards - state_values.detach()
                    surr1 = ratios * advantages
                    surr2 = torch.clamp(ratios, 0.8, 1.2) * advantages
                    # Entropy coefficient - slightly increased to encourage exploration (more trading)
                    # But still lower than before to maintain confidence
                    loss = -torch.min(surr1, surr2) + 0.5 * nn.MSELoss()(state_values, rewards) - 0.02 * dist_entropy
                    
                    # Check for NaN in loss
                    if torch.isnan(loss).any() or torch.isinf(loss).any():
                        logger.warning(f"Step {step}: NaN/Inf detected in loss, skipping update")
                        break

                    optimizer_rl.zero_grad()
                    loss.mean().backward()
                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                    
                    # Check for NaN in gradients
                    has_nan_grad = False
                    for param in agent.parameters():
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                has_nan_grad = True
                                break
                    
                    if has_nan_grad:
                        logger.warning(f"Step {step}: NaN/Inf detected in gradients, skipping update")
                        optimizer_rl.zero_grad()  # Clear gradients
                        break
                    
                    optimizer_rl.step()
                    # Update learning rate
                    scheduler.step()
                except Exception as e:
                    logger.error(f"Step {step}: Error during PPO update: {e}")
                    break

            # Clear memory
            memory_states, memory_actions, memory_logprobs, memory_rewards = [], [], [], []

            print(f"   Step {step:5d}/{steps_rl} | Episode Reward: {episode_reward:.2f}")

        # Episode finished
        if done:
            # Calculate profit from environment balance (actual account value)
            # This includes both realized and unrealized P&L
            current_balance = env.balance
            if env.shares > 0:
                # Include unrealized gains/losses from open position
                current_price = env.data[env.current_step][0]  # Close price
                position_value = env.shares * current_price
                total_value = current_balance + position_value
                unrealized_pnl = (current_price - env.entry_price) * env.shares
            else:
                total_value = current_balance
                unrealized_pnl = 0.0
            
            # Calculate profit from episode start balance
            episode_profit = total_value - episode_start_balance
            
            # Also get realized profit from TradingLogger (closed trades only)
            # IMPORTANT: Only count profits from SELL actions in THIS episode
            results = logger_trading.get_results()
            realized_profit = 0.0
            buy_count_logger = 0
            sell_count_logger = 0
            hold_count_logger = 0
            
            if not results.empty:
                # Count actual actions in this episode
                buy_count_logger = len(results[results['Action'] == 'BUY'])
                sell_count_logger = len(results[results['Action'] == 'SELL'])
                hold_count_logger = len(results[results['Action'] == 'SKIP'])
                
                # Filter to only SELL actions to get realized profit
                sell_actions = results[results['Action'] == 'SELL']
                if not sell_actions.empty:
                    realized_profit = sell_actions['Profit'].sum()
                
                # CRITICAL VALIDATION: If all actions were HOLD, profit MUST be 0
                total_logger_actions = buy_count_logger + sell_count_logger + hold_count_logger
                if total_logger_actions > 0 and buy_count_logger == 0 and sell_count_logger == 0:
                    # All actions were HOLD - profit must be 0
                    if abs(realized_profit) > 1e-6:  # Use small epsilon for float comparison
                        logger.warning(f"Step {step}: Logger shows profit ${realized_profit:.2f} but all {hold_count_logger} actions were HOLD! "
                                     f"This is a BUG - forcing profit to 0.0")
                        realized_profit = 0.0
                
                # Additional validation: Check for unmatched positions
                if logger_trading.open_positions:
                    logger.warning(f"Step {step}: Logger has {len(logger_trading.open_positions)} unmatched open positions "
                                 f"at episode end. This may indicate a logging bug.")
            
            # Accumulate cumulative totals (use realized profit for consistency)
            # NOTE: realized_profit has been validated above, so it's safe to add
            total_cumulative_reward += episode_reward
            total_cumulative_profit += realized_profit
            
            # Calculate action counts and reward breakdown
            total_actions = (reward_breakdown_stats['hold_count'] + 
                           reward_breakdown_stats['buy_count'] + 
                           reward_breakdown_stats['sell_count'])
            
            print(f"\n   🏁 Episode Complete:")
            print(f"      Total Reward: {episode_reward:.2f}")
            print(f"      Account Value Change: ${episode_profit:.2f} (from ${episode_start_balance:.2f} to ${total_value:.2f})")
            print(f"      Realized Profit (closed trades): ${realized_profit:.2f}")
            
            # DEBUG: Print comprehensive debug stats
            print(f"\n      🔍 DEBUG STATS:")
            print(f"         Requested actions: HOLD={debug_stats['requested_actions']['HOLD']}, "
                  f"BUY={debug_stats['requested_actions']['BUY']}, SELL={debug_stats['requested_actions']['SELL']}")
            print(f"         Executed actions: HOLD={debug_stats['executed_actions']['HOLD']}, "
                  f"BUY={debug_stats['executed_actions']['BUY']}, SELL={debug_stats['executed_actions']['SELL']}")
            print(f"         Logger actions: HOLD={debug_stats['logger_actions']['HOLD']}, "
                  f"BUY={debug_stats['logger_actions']['BUY']}, SELL={debug_stats['logger_actions']['SELL']}")
            print(f"         Reward breakdown: HOLD={reward_breakdown_stats['hold_count']}, "
                  f"BUY={reward_breakdown_stats['buy_count']}, SELL={reward_breakdown_stats['sell_count']}")
            print(f"         Action mismatches: {debug_stats['mismatch_count']}")
            print(f"         Blocked actions: {debug_stats['blocked_actions']}")
            print(f"         Missing reward_breakdown: {debug_stats['missing_reward_breakdown']}")
            
            # Explain why requested != executed
            requested_total = (debug_stats['requested_actions']['HOLD'] + 
                             debug_stats['requested_actions']['BUY'] + 
                             debug_stats['requested_actions']['SELL'])
            executed_total = (debug_stats['executed_actions']['HOLD'] + 
                            debug_stats['executed_actions']['BUY'] + 
                            debug_stats['executed_actions']['SELL'])
            
            if requested_total > 0:
                blocked_buy = debug_stats['requested_actions']['BUY'] - debug_stats['executed_actions']['BUY']
                blocked_sell = debug_stats['requested_actions']['SELL'] - debug_stats['executed_actions']['SELL']
                if blocked_buy > 0 or blocked_sell > 0:
                    print(f"\n      📊 ACTION BLOCKING ANALYSIS:")
                    print(f"         BUY requests blocked: {blocked_buy} (became HOLD)")
                    print(f"         SELL requests blocked: {blocked_sell} (became HOLD)")
                    print(f"         Total blocked: {blocked_buy + blocked_sell} out of {requested_total} requests")
                    print(f"         Blocking rate: {(blocked_buy + blocked_sell) / requested_total * 100:.1f}%")
                    print(f"         Reason: Trading constraints (max positions, confidence threshold, etc.)")
                    print(f"         This is EXPECTED behavior - constraints prevent risky trades")
            
            # Check for impossible scenario: profit with no trades
            if total_actions == reward_breakdown_stats['hold_count'] and (episode_profit > 0 or abs(realized_profit) > 1e-6):
                print(f"\n      ⚠️  IMPOSSIBLE SCENARIO: Profit with NO TRADES!")
                print(f"         All actions were HOLD - profit should be $0.00")
                print(f"         This indicates a calculation error or carryover from previous episode")
                print(f"         Balance: ${current_balance:.2f}, Start: ${episode_start_balance:.2f}")
                print(f"         Logger action counts: HOLD={hold_count_logger}, BUY={buy_count_logger}, SELL={sell_count_logger}")
                print(f"         Reward breakdown: HOLD={reward_breakdown_stats['hold_count']}, "
                      f"BUY={reward_breakdown_stats['buy_count']}, SELL={reward_breakdown_stats['sell_count']}")
                if env.shares > 0:
                    print(f"         ⚠️  Open position exists but no BUY action recorded!")
                    print(f"         Position: {env.shares} shares @ ${env.entry_price:.2f}")
                else:
                    print(f"         No open position - profit should be $0.00!")
                
                # Force profit to 0 if all actions were HOLD and correct cumulative
                if buy_count_logger == 0 and sell_count_logger == 0:
                    old_realized = realized_profit
                    realized_profit = 0.0
                    # Correct cumulative profit by removing the incorrect value we just added
                    total_cumulative_profit -= old_realized
                    logger.warning(f"Step {step}: Corrected realized_profit from ${old_realized:.2f} to $0.0 "
                                 f"and adjusted cumulative profit")
            
            # DEBUG: Check for mismatches between logger and reward breakdown
            logger_total = debug_stats['logger_actions']['HOLD'] + debug_stats['logger_actions']['BUY'] + debug_stats['logger_actions']['SELL']
            reward_total = (reward_breakdown_stats['hold_count'] + 
                          reward_breakdown_stats['buy_count'] + 
                          reward_breakdown_stats['sell_count'])
            
            if logger_total > 0 and reward_total == 0:
                logger.error(f"Step {step}: CRITICAL MISMATCH - Logger has {logger_total} actions but reward breakdown has 0!")
                logger.error(f"  This suggests 'reward_breakdown' is missing from info dict in {debug_stats['missing_reward_breakdown']} steps")
            
            if (debug_stats['logger_actions']['HOLD'] != reward_breakdown_stats['hold_count'] or
                debug_stats['logger_actions']['BUY'] != reward_breakdown_stats['buy_count'] or
                debug_stats['logger_actions']['SELL'] != reward_breakdown_stats['sell_count']):
                logger.warning(f"Step {step}: Logger vs Reward breakdown mismatch detected!")
                logger.warning(f"  Logger: HOLD={debug_stats['logger_actions']['HOLD']}, "
                             f"BUY={debug_stats['logger_actions']['BUY']}, "
                             f"SELL={debug_stats['logger_actions']['SELL']}")
                logger.warning(f"  Reward: HOLD={reward_breakdown_stats['hold_count']}, "
                             f"BUY={reward_breakdown_stats['buy_count']}, "
                             f"SELL={reward_breakdown_stats['sell_count']}")
            
            if env.shares > 0:
                current_price = env.data[env.current_step][0]
                print(f"      Unrealized P&L (open position): ${unrealized_pnl:.2f} ({env.shares} shares @ ${env.entry_price:.2f} → ${current_price:.2f})")
            else:
                print(f"      No open positions (all trades closed)")
            
            if total_actions > 0:
                print(f"\n   📊 Reward Breakdown:")
                print(f"      Actions: HOLD={reward_breakdown_stats['hold_count']}, "
                      f"BUY={reward_breakdown_stats['buy_count']}, "
                      f"SELL={reward_breakdown_stats['sell_count']}")
                print(f"      Action Rewards: HOLD={reward_breakdown_stats['hold_reward_total']:.2f}, "
                      f"BUY={reward_breakdown_stats['buy_reward_total']:.2f}, "
                      f"SELL={reward_breakdown_stats['sell_reward_total']:.2f}")
                print(f"      Position Rewards: Holding={reward_breakdown_stats['holding_reward_total']:.2f}, "
                      f"Stop Loss={reward_breakdown_stats['stop_loss_reward_total']:.2f}, "
                      f"Take Profit={reward_breakdown_stats['take_profit_reward_total']:.2f}")
                print(f"      Total from actions: {reward_breakdown_stats['hold_reward_total'] + reward_breakdown_stats['buy_reward_total'] + reward_breakdown_stats['sell_reward_total']:.2f}")
                print(f"      Total from positions: {reward_breakdown_stats['holding_reward_total'] + reward_breakdown_stats['stop_loss_reward_total'] + reward_breakdown_stats['take_profit_reward_total']:.2f}")
            
            # Check for misalignment
            if episode_reward < 0 and (episode_profit > 0 or realized_profit > 0):
                print(f"\n      ⚠️  WARNING: Negative reward but positive profit!")
                if episode_profit > 0:
                    print(f"         Reward/Profit ratio: {episode_reward/episode_profit*100:.2f}%")
                if total_actions > 0:
                    print(f"         Avg reward per action: {episode_reward/total_actions:.4f}")
                    if realized_profit > 0:
                        print(f"         Avg realized profit per action: ${realized_profit/total_actions:.2f}")
                
                # Reset breakdown stats for next episode
                reward_breakdown_stats = {
                    'hold_count': 0,
                    'buy_count': 0,
                    'sell_count': 0,
                    'hold_reward_total': 0.0,
                    'buy_reward_total': 0.0,
                    'sell_reward_total': 0.0,
                    'holding_reward_total': 0.0,
                    'stop_loss_reward_total': 0.0,
                    'take_profit_reward_total': 0.0
                }

            # Reset for next episode
            state, _ = env.reset()
            # Reset position tracking
            if hasattr(env, '_prev_shares'):
                env._prev_shares = env.shares
            # DEBUG: Ensure balance tracking is correct
            episode_start_balance = env.balance  # Use actual balance after reset
            if abs(episode_start_balance - env.initial_balance) > 1e-6:
                logger.warning(f"Step {step}: Balance mismatch after reset: balance={env.balance}, initial_balance={env.initial_balance}")
            episode_reward = 0
            # Create fresh logger and verify it's clean
            logger_trading = TradingLogger()
            if not logger_trading.verify_clean():
                logger.error(f"Step {step}: New TradingLogger is not clean! This is a bug.")
                logger_trading.reset()  # Force reset
            # Reset reward breakdown stats
            reward_breakdown_stats = {
                'hold_count': 0,
                'buy_count': 0,
                'sell_count': 0,
                'hold_reward_total': 0.0,
                'buy_reward_total': 0.0,
                'sell_reward_total': 0.0,
                'holding_reward_total': 0.0,
                'stop_loss_reward_total': 0.0,
                'take_profit_reward_total': 0.0
            }
            # Reset debug stats
            debug_stats = {
                'requested_actions': {'HOLD': 0, 'BUY': 0, 'SELL': 0},
                'executed_actions': {'HOLD': 0, 'BUY': 0, 'SELL': 0},
                'blocked_actions': 0,
                'missing_reward_breakdown': 0,
                'logger_actions': {'HOLD': 0, 'BUY': 0, 'SELL': 0},
                'mismatch_count': 0
            }
        
        # Log cumulative totals every 10000 steps
        if step % 10000 == 0:
            # Get current episode realized profit from TradingLogger (closed trades only)
            results = logger_trading.get_results()
            if not results.empty:
                # Filter to only SELL actions to get realized profit
                sell_actions = results[results['Action'] == 'SELL']
                current_realized_profit = sell_actions['Profit'].sum() if not sell_actions.empty else 0.0
            else:
                current_realized_profit = 0.0
            
            # Calculate totals including current episode
            current_total_reward = total_cumulative_reward + episode_reward
            current_total_profit = total_cumulative_profit + current_realized_profit
            
            print(f"\n   📊 Cumulative Stats at Step {step}:")
            print(f"      Total Reward (all episodes): {current_total_reward:.2f}")
            print(f"      Total Profit (all episodes): ${current_total_profit:.2f}")
            if current_total_profit > 0:
                reward_profit_ratio = (current_total_reward / current_total_profit) * 100
                print(f"      Reward/Profit Ratio: {reward_profit_ratio:.2f}%")
                if current_total_reward < 0 and current_total_profit > 0:
                    print(f"      ⚠️  MISALIGNMENT: Negative reward ({current_total_reward:.2f}) but positive profit (${current_total_profit:.2f})")
                    print(f"         This suggests reward function needs adjustment!")
            print()

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
    MODE = "stock"  # "crypto" or "stock"
    INTERVAL = "1h"  # "1h", "4h" for crypto; "1d" for stock
    TICKER = "ETH-USD" if MODE == "crypto" else "TSLA"  # Change to your ticker
    WINDOW_SIZE = 72
    HIDDEN_DIM = 128  # Increased from 32 to 128 for better capacity
    EPOCHS_PRETRAIN = 200  # Increased for better pre-training
    STEPS_RL = 150000  # Increased for better convergence
    USE_ENHANCED_ARCH = True  # Use enhanced architecture with attention
    CONFIDENCE_TEMPERATURE = 0.8  # Temperature scaling for confidence calibration (<1.0 increases confidence)
    
    # Trading constraints (matches alpaca_paper_trading.py)
    USE_TRADING_CONSTRAINTS = True  # Set to True to train with real constraints
    MIN_CONFIDENCE = 0.30  # Minimum confidence to trade
    MAX_POSITIONS = 3  # Maximum concurrent positions
    POSITION_SIZE_PCT = 0.10  # Position size as % of account

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
            steps_rl=STEPS_RL,
            use_trading_constraints=USE_TRADING_CONSTRAINTS,
            min_confidence=MIN_CONFIDENCE,
            max_positions=MAX_POSITIONS,
            position_size_pct=POSITION_SIZE_PCT,
            use_enhanced_arch=USE_ENHANCED_ARCH,
            confidence_temperature=CONFIDENCE_TEMPERATURE
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

