"""
PREDICTION SCRIPT FOR LIVE TRADING
===================================
Uses yfinance to get latest data and generate trading signals.
Run this at 3:30 AM Vietnam time (before market open/close).

This script:
1. Fetches latest data using yfinance
2. Generates trading signals using trained model
3. Saves predictions to file
4. Can send notifications (email/SMS)
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import warnings
import json
import os
from datetime import datetime
import pytz

# Import KAN classes from dapgio_improved to avoid duplication
from dapgio_improved import KANPredictor, KANActorCritic

# Suppress yfinance warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ==============================================================================
# PREDICTION FUNCTION
# ==============================================================================

def get_prediction(ticker="ETH-USD", mode="crypto", interval="1h", model_path=None, window_size=30):
    """
    Get trading prediction for a ticker.
    
    Args:
        ticker: Stock/crypto symbol (e.g., "ETH-USD", "BTC-USD", "NVDA")
        mode: "crypto" or "stock"
        interval: "1h", "4h" for crypto; "1d" for stock
        model_path: Path to trained model file (auto-detected if None)
        window_size: Number of periods for lookback window
    
    Returns:
        dict with prediction results
    """
    # Auto-detect model path if not provided
    if model_path is None:
        model_path = "kan_agent_crypto.pth" if mode == "crypto" else "kan_agent_stock.pth"
    
    print(f"\n🔮 Analyzing {ticker} ({mode.upper()} mode, {interval} interval)...")
    print(f"⏰ Time: {datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    result = {
        "ticker": ticker,
        "mode": mode,
        "interval": interval,
        "timestamp": datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).isoformat(),
        "success": False,
        "action": None,
        "confidence": None,
        "price": None,
        "error": None
    }
    
    try:
        # 1. Fetch Data using yfinance with appropriate interval
        print(f"📥 Fetching data for {ticker}...")
        if mode == "crypto":
            # For crypto: use intraday data
            period = "60d" if interval == "1h" else "120d"
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        else:
            # For stocks: use daily data
            df = yf.download(ticker, period="6mo", interval="1d", progress=False, auto_adjust=True)

        if df.empty:
            raise ValueError(f"No data downloaded for {ticker}")

        # Fix MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # 2. Calculate gap (period change)
        if len(df) > 1:
            prev_close = df['Close'].iloc[-2]
            current_price = float(df['Close'].iloc[-1])
            gap_percent = (current_price - prev_close) / prev_close * 100
            gap_label = f"{interval} change" if mode == "crypto" else "24h change"
        else:
            current_price = float(df['Close'].iloc[-1])
            gap_percent = 0.0
            gap_label = "N/A"

        # 3. Feature Engineering (MUST match training exactly)
        print("🔧 Calculating features...")
        df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
        df['Vol_Norm'] = df['Volume'] / df['Volume'].rolling(20).mean()

        # Technical indicators
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.dropna(inplace=True)

        # Normalize
        df["RSI_14"] = df["RSI_14"] / 100.0
        df["MACD_12_26_9"] = (df["MACD_12_26_9"] - df["MACD_12_26_9"].mean()) / (df["MACD_12_26_9"].std() + 1e-7)

        features = ["Close", "Log_Ret", "Vol_Norm", "RSI_14", "MACD_12_26_9"]

        # Check data length
        if len(df) < window_size:
            raise ValueError(f"Not enough data: {len(df)} rows, need at least {window_size}")

        # 4. Prepare observation
        window = df[features].iloc[-window_size:].values
        obs = window.flatten().astype(np.float32)
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

        # 5. Load Model
        print(f"🤖 Loading model from {model_path}...")
        obs_dim = window_size * len(features)  # 30 * 5 = 150
        action_dim = 3

        agent = KANActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=32)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        agent.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        agent.eval()

        # 6. Get Prediction
        print("🎯 Generating prediction...")
        with torch.no_grad():
            action, log_prob, state_value = agent.act(obs_tensor, deterministic=True)
            # Get confidence probabilities
            features = agent.body(obs_tensor)
            logits = agent.actor_head(features)
            confidence_probs = F.softmax(logits, dim=1).squeeze().detach().cpu().numpy()

        action_names = ["HOLD", "BUY", "SELL"]
        action_name = action_names[action]
        max_confidence = float(confidence_probs.max())

        # 7. Build result
        result.update({
            "success": True,
            "action": action_name,
            "action_id": int(action),
            "confidence": {
                "HOLD": float(confidence_probs[0]),
                "BUY": float(confidence_probs[1]),
                "SELL": float(confidence_probs[2])
            },
            "max_confidence": max_confidence,
            "price": current_price,
            "gap_percent": float(gap_percent),
            "prev_close": float(prev_close) if len(df) > 1 else None,
            "stop_loss": current_price * 0.92 if action == 1 else None,  # -8% for crypto
            "take_profit": current_price * 1.25 if action == 1 else None,  # +25% for crypto
        })

        # 8. Display Results
        print("\n" + "=" * 50)
        print(f"📊 PREDICTION RESULTS FOR {ticker}")
        print("=" * 50)
        print(f"   Current Price:    ${current_price:.2f}")
        print(f"   {gap_label}:       {gap_percent:+.2f}%")
        print(f"   Signal:            {action_name}")
        print(f"   Confidence:        {max_confidence:.1%}")
        print(f"   Confidence Breakdown:")
        print(f"      - HOLD:        {confidence_probs[0]:.1%}")
        print(f"      - BUY:         {confidence_probs[1]:.1%}")
        print(f"      - SELL:        {confidence_probs[2]:.1%}")

        if action == 1:  # BUY
            MAX_CHASE = 3.0
            if gap_percent > MAX_CHASE:
                print(f"\n⚠️  WARNING: GAP TOO LARGE ({gap_percent:.2f}%)")
                print("   RECOMMENDATION: SKIP or Wait for Dip")
            else:
                print(f"\n🚀 TRADE PLAN:")
                print(f"   1. ENTRY:       ${current_price:.2f}")
                print(f"   2. STOP LOSS:   ${result['stop_loss']:.2f} (-8%)")
                print(f"   3. TAKE PROFIT:  ${result['take_profit']:.2f} (+25%)")

        print("=" * 50)

    except Exception as e:
        error_msg = str(e)
        result["error"] = error_msg
        print(f"\n❌ ERROR: {error_msg}")
        import traceback
        traceback.print_exc()

    return result


def save_predictions(predictions, filename="predictions.json"):
    """Save predictions to JSON file"""
    # Load existing predictions if file exists
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            all_predictions = json.load(f)
    else:
        all_predictions = []

    # Add new predictions
    if isinstance(predictions, list):
        all_predictions.extend(predictions)
    else:
        all_predictions.append(predictions)

    # Keep only last 100 predictions
    all_predictions = all_predictions[-100:]

    # Save
    with open(filename, 'w') as f:
        json.dump(all_predictions, f, indent=2)

    print(f"💾 Predictions saved to {filename}")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    # Configuration
    MODE = "stock"  # "crypto" or "stock"
    INTERVAL = "1d"  # "1h", "4h" for crypto; "1d" for stock
    TICKERS = ["^ixic"] if MODE == "stock" else ["ETH-USD"]  # NASDAQ for stocks, ETH-USD for crypto
    WINDOW_SIZE = 30

    print("\n" + "=" * 60)
    print("🤖 TRADING PREDICTION SYSTEM")
    print("=" * 60)
    print(f"Mode: {MODE.upper()}")
    print(f"Interval: {INTERVAL}")
    print(f"⏰ Vietnam Time: {datetime.now(pytz.timezone('Asia/Ho_Chi_Minh')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"📊 Analyzing {len(TICKERS)} ticker(s)")
    print("=" * 60)

    all_predictions = []

    for ticker in TICKERS:
        try:
            prediction = get_prediction(ticker, mode=MODE, interval=INTERVAL, window_size=WINDOW_SIZE)
            all_predictions.append(prediction)
            print()  # Blank line between tickers
        except Exception as e:
            print(f"❌ Failed to predict {ticker}: {e}")

    # Save predictions
    if all_predictions:
        save_predictions(all_predictions)
        print(f"\n✅ Completed predictions for {len(all_predictions)} ticker(s)")

    print("\n" + "=" * 60)

