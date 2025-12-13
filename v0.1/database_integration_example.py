"""
Example: How to integrate database with existing code
=====================================================
This shows how to update paper_trading.py and predict.py to use the database.
"""

from database import TradingDB
from datetime import datetime
import json

# Initialize database
db = TradingDB("trading_bot.db")

# ============================================================================
# EXAMPLE 1: Update PaperTradingAccount to use database
# ============================================================================

class PaperTradingAccountWithDB:
    """Paper trading account with database integration"""
    
    def __init__(self, initial_balance: float = 10000.0, db: TradingDB = None):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.positions = {}
        self.trade_history = []
        self.db = db or TradingDB()
    
    def buy(self, ticker: str, price: float, shares: float) -> bool:
        """Execute buy order and save to database"""
        cost = price * shares
        commission = cost * 0.001
        total_cost = cost + commission
        
        if total_cost > self.balance:
            return False
        
        self.balance -= total_cost
        
        if ticker in self.positions:
            # Add to existing position
            old_shares = self.positions[ticker]["shares"]
            old_cost = self.positions[ticker]["entry_price"] * old_shares
            new_cost = price * shares
            avg_price = (old_cost + new_cost) / (old_shares + shares)
            self.positions[ticker]["shares"] += shares
            self.positions[ticker]["entry_price"] = avg_price
        else:
            self.positions[ticker] = {
                "shares": shares,
                "entry_price": price,
                "entry_time": datetime.now().isoformat()
            }
        
        # Save to database
        trade_id = self.db.add_trade(
            ticker=ticker,
            action="BUY",
            shares=shares,
            price=price,
            commission=commission,
            timestamp=datetime.now()
        )
        
        self.trade_history.append({
            "id": trade_id,
            "ticker": ticker,
            "action": "BUY",
            "shares": shares,
            "price": price,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    
    def sell(self, ticker: str, price: float, shares: float) -> bool:
        """Execute sell order and save to database"""
        if ticker not in self.positions:
            return False
        
        if shares > self.positions[ticker]["shares"]:
            shares = self.positions[ticker]["shares"]
        
        entry_price = self.positions[ticker]["entry_price"]
        gross_val = price * shares
        commission = gross_val * 0.001
        net_val = gross_val - commission
        self.balance += net_val
        
        # Calculate P&L
        cost_basis = entry_price * shares
        pnl = net_val - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis > 0 else 0
        
        # Update position
        self.positions[ticker]["shares"] -= shares
        if self.positions[ticker]["shares"] <= 0:
            del self.positions[ticker]
        
        # Save to database
        trade_id = self.db.add_trade(
            ticker=ticker,
            action="SELL",
            shares=shares,
            price=price,
            commission=commission,
            entry_price=entry_price,
            exit_price=price,
            pnl=pnl,
            pnl_pct=pnl_pct,
            timestamp=datetime.now()
        )
        
        self.trade_history.append({
            "id": trade_id,
            "ticker": ticker,
            "action": "SELL",
            "shares": shares,
            "price": price,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "timestamp": datetime.now().isoformat()
        })
        
        return True
    
    def get_stats_from_db(self, ticker: Optional[str] = None):
        """Get statistics from database"""
        return self.db.get_trade_stats(ticker=ticker)
    
    def close(self):
        """Close database connection"""
        self.db.close()


# ============================================================================
# EXAMPLE 2: Update predict.py to use database
# ============================================================================

def save_predictions_to_db(predictions: Dict, db: TradingDB = None):
    """Save predictions to database instead of JSON"""
    if db is None:
        db = TradingDB()
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    for ticker, pred in predictions.items():
        db.add_prediction(
            ticker=ticker,
            prediction_date=today,
            predicted_price=pred.get('price'),
            predicted_return=pred.get('return'),
            confidence=pred.get('confidence'),
            signal=pred.get('signal'),
            model_version="v0.1"
        )
    
    print(f"Saved {len(predictions)} predictions to database")
    db.close()


# ============================================================================
# EXAMPLE 3: Update training to save metrics
# ============================================================================

def save_training_metrics_to_db(model_name: str, ticker: str, mode: str,
                                interval: str, final_stats: Dict,
                                config: Dict, model_path: str,
                                db: TradingDB = None):
    """Save training metrics to database"""
    if db is None:
        db = TradingDB()
    
    db.add_model_performance(
        model_name=model_name,
        model_version="v0.1",
        training_date=datetime.now().strftime("%Y-%m-%d"),
        ticker=ticker,
        mode=mode,
        interval=interval,
        total_trades=final_stats.get('total_trades'),
        winning_trades=final_stats.get('winning_trades'),
        losing_trades=final_stats.get('losing_trades'),
        win_rate=final_stats.get('win_rate'),
        total_pnl=final_stats.get('total_pnl'),
        final_reward=final_stats.get('final_reward'),
        config=config,
        model_path=model_path
    )
    
    print("Training metrics saved to database")
    db.close()


# ============================================================================
# EXAMPLE 4: Query and analyze data
# ============================================================================

def analyze_performance(db: TradingDB = None):
    """Analyze trading performance from database"""
    if db is None:
        db = TradingDB()
    
    # Get all trades
    trades = db.get_trades()
    
    if len(trades) == 0:
        print("No trades found")
        return
    
    # Overall stats
    stats = db.get_trade_stats()
    print("\n📊 Overall Performance:")
    print(f"  Total Trades: {stats['total_trades']}")
    print(f"  Win Rate: {stats['win_rate']:.2f}%")
    print(f"  Total P&L: ${stats['total_pnl']:.2f}")
    print(f"  Avg P&L: ${stats['avg_pnl']:.2f}")
    print(f"  Stop Losses: {stats['stop_losses']}")
    print(f"  Take Profits: {stats['take_profits']}")
    
    # By ticker
    tickers = trades['ticker'].unique()
    print("\n📈 Performance by Ticker:")
    for ticker in tickers:
        ticker_stats = db.get_trade_stats(ticker=ticker)
        if ticker_stats:
            print(f"\n  {ticker}:")
            print(f"    Trades: {ticker_stats['total_trades']}")
            print(f"    Win Rate: {ticker_stats['win_rate']:.2f}%")
            print(f"    Total P&L: ${ticker_stats['total_pnl']:.2f}")
    
    # Recent trades
    recent = db.get_trades(limit=10)
    print("\n📋 Recent Trades:")
    print(recent[['ticker', 'action', 'shares', 'price', 'pnl', 'timestamp']].to_string())
    
    db.close()


if __name__ == "__main__":
    print("Database Integration Examples")
    print("=" * 50)
    
    # Test database
    db = TradingDB("test_trading_bot.db")
    
    # Add sample trades
    db.add_trade("NVDA", "BUY", 10.0, 150.50, 1.50)
    db.add_trade("NVDA", "SELL", 10.0, 155.00, 1.55,
                 entry_price=150.50, exit_price=155.00,
                 pnl=45.00, pnl_pct=2.99)
    
    # Get stats
    stats = db.get_trade_stats()
    print("\nSample Stats:")
    print(stats)
    
    # Analyze
    analyze_performance(db)
    
    db.close()
    print("\nExamples complete!")

