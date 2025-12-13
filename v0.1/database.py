"""
TRADING BOT DATABASE
====================
SQLite database for storing trading data, predictions, and performance metrics.

Why SQLite?
- ✅ Built into Python (no installation needed)
- ✅ File-based (easy backup, portable)
- ✅ SQL queries (better than JSON for analysis)
- ✅ Free and open source
- ✅ Perfect for single-user applications
- ✅ Can migrate to PostgreSQL later if needed

Usage:
    from database import TradingDB
    
    db = TradingDB()
    db.add_trade(...)
    trades = db.get_trades(date_from='2025-01-01')
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Any
import pandas as pd
from pathlib import Path


class TradingDB:
    """SQLite database for trading bot data"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
        self._create_tables()
    
    def _create_tables(self):
        """Create all necessary tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # 1. TRADES TABLE - All executed trades
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,  -- 'BUY' or 'SELL'
                shares REAL NOT NULL,
                price REAL NOT NULL,
                commission REAL DEFAULT 0.0,
                timestamp DATETIME NOT NULL,
                entry_price REAL,  -- For SELL: original buy price
                exit_price REAL,   -- For SELL: sell price
                pnl REAL,          -- Profit/Loss for completed trades
                pnl_pct REAL,      -- P&L percentage
                stop_loss_triggered BOOLEAN DEFAULT 0,
                take_profit_triggered BOOLEAN DEFAULT 0,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 2. PREDICTIONS TABLE - Model predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                prediction_date DATE NOT NULL,
                predicted_price REAL,
                predicted_return REAL,
                confidence REAL,
                signal TEXT,  -- 'BUY', 'SELL', 'HOLD'
                actual_price REAL,  -- Filled after fact
                actual_return REAL,
                prediction_accuracy REAL,  -- Calculated later
                model_version TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, prediction_date)
            )
        """)
        
        # 3. PORTFOLIO TABLE - Portfolio state over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                balance REAL NOT NULL,
                total_value REAL NOT NULL,
                positions_count INTEGER DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0.0,
                realized_pnl REAL DEFAULT 0.0,
                daily_return REAL,
                cumulative_return REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date)
            )
        """)
        
        # 4. MODEL_PERFORMANCE TABLE - Training and backtest metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT,
                training_date DATE,
                ticker TEXT,
                mode TEXT,  -- 'crypto' or 'stock'
                interval TEXT,  -- '1h', '4h', '1d'
                total_trades INTEGER,
                winning_trades INTEGER,
                losing_trades INTEGER,
                win_rate REAL,
                avg_profit REAL,
                avg_loss REAL,
                total_pnl REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                training_steps INTEGER,
                final_reward REAL,
                config_json TEXT,  -- Store config as JSON
                model_path TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 5. MARKET_DATA_CACHE - Cache historical data to avoid API calls
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                interval TEXT,  -- '1h', '4h', '1d'
                data_json TEXT,  -- Full OHLCV data as JSON
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(ticker, date, interval)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(prediction_date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_portfolio_date ON portfolio(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_data_ticker ON market_data_cache(ticker, date)")
        
        self.conn.commit()
    
    # ============================================================================
    # TRADES METHODS
    # ============================================================================
    
    def add_trade(self, ticker: str, action: str, shares: float, price: float,
                  commission: float = 0.0, entry_price: Optional[float] = None,
                  exit_price: Optional[float] = None, pnl: Optional[float] = None,
                  pnl_pct: Optional[float] = None, stop_loss_triggered: bool = False,
                  take_profit_triggered: bool = False, notes: Optional[str] = None,
                  timestamp: Optional[datetime] = None):
        """Add a trade to the database"""
        if timestamp is None:
            timestamp = datetime.now()
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO trades (ticker, action, shares, price, commission, timestamp,
                              entry_price, exit_price, pnl, pnl_pct, stop_loss_triggered,
                              take_profit_triggered, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (ticker, action.upper(), shares, price, commission, timestamp,
              entry_price, exit_price, pnl, pnl_pct, stop_loss_triggered,
              take_profit_triggered, notes))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_trades(self, ticker: Optional[str] = None, 
                   date_from: Optional[str] = None,
                   date_to: Optional[str] = None,
                   action: Optional[str] = None,
                   limit: Optional[int] = None) -> pd.DataFrame:
        """Get trades with optional filters"""
        query = "SELECT * FROM trades WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        if action:
            query += " AND action = ?"
            params.append(action.upper())
        
        if date_from:
            query += " AND timestamp >= ?"
            params.append(date_from)
        
        if date_to:
            query += " AND timestamp <= ?"
            params.append(date_to)
        
        query += " ORDER BY timestamp DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        df = pd.read_sql_query(query, self.conn, params=params)
        return df
    
    def get_trade_stats(self, ticker: Optional[str] = None,
                       date_from: Optional[str] = None) -> Dict[str, Any]:
        """Get trading statistics"""
        query = """
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) as buy_trades,
                SUM(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END) as sell_trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                AVG(pnl) as avg_pnl,
                SUM(pnl) as total_pnl,
                AVG(pnl_pct) as avg_pnl_pct,
                SUM(CASE WHEN stop_loss_triggered = 1 THEN 1 ELSE 0 END) as stop_losses,
                SUM(CASE WHEN take_profit_triggered = 1 THEN 1 ELSE 0 END) as take_profits
            FROM trades
            WHERE 1=1
        """
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        if date_from:
            query += " AND timestamp >= ?"
            params.append(date_from)
        
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if row and row['total_trades']:
            win_rate = (row['winning_trades'] / row['sell_trades'] * 100) if row['sell_trades'] else 0
            return {
                'total_trades': row['total_trades'],
                'buy_trades': row['buy_trades'],
                'sell_trades': row['sell_trades'],
                'winning_trades': row['winning_trades'],
                'losing_trades': row['losing_trades'],
                'win_rate': win_rate,
                'avg_pnl': row['avg_pnl'],
                'total_pnl': row['total_pnl'],
                'avg_pnl_pct': row['avg_pnl_pct'],
                'stop_losses': row['stop_losses'],
                'take_profits': row['take_profits']
            }
        return {}
    
    # ============================================================================
    # PREDICTIONS METHODS
    # ============================================================================
    
    def add_prediction(self, ticker: str, prediction_date: str,
                      predicted_price: Optional[float] = None,
                      predicted_return: Optional[float] = None,
                      confidence: Optional[float] = None,
                      signal: Optional[str] = None,
                      model_version: Optional[str] = None):
        """Add or update a prediction"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO predictions 
            (ticker, prediction_date, predicted_price, predicted_return, 
             confidence, signal, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (ticker, prediction_date, predicted_price, predicted_return,
              confidence, signal, model_version))
        self.conn.commit()
        return cursor.lastrowid
    
    def update_prediction_actual(self, ticker: str, prediction_date: str,
                                actual_price: float, actual_return: float):
        """Update prediction with actual results"""
        # Calculate accuracy
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT predicted_return FROM predictions
            WHERE ticker = ? AND prediction_date = ?
        """, (ticker, prediction_date))
        row = cursor.fetchone()
        
        accuracy = None
        if row and row['predicted_return'] is not None:
            pred_return = row['predicted_return']
            accuracy = 1.0 - abs(pred_return - actual_return) / (abs(actual_return) + 0.01)
        
        cursor.execute("""
            UPDATE predictions
            SET actual_price = ?, actual_return = ?, prediction_accuracy = ?
            WHERE ticker = ? AND prediction_date = ?
        """, (actual_price, actual_return, accuracy, ticker, prediction_date))
        self.conn.commit()
    
    def get_predictions(self, ticker: Optional[str] = None,
                       date_from: Optional[str] = None,
                       limit: Optional[int] = None) -> pd.DataFrame:
        """Get predictions with optional filters"""
        query = "SELECT * FROM predictions WHERE 1=1"
        params = []
        
        if ticker:
            query += " AND ticker = ?"
            params.append(ticker)
        
        if date_from:
            query += " AND prediction_date >= ?"
            params.append(date_from)
        
        query += " ORDER BY prediction_date DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    # ============================================================================
    # PORTFOLIO METHODS
    # ============================================================================
    
    def add_portfolio_snapshot(self, date: str, balance: float, total_value: float,
                              positions_count: int = 0, unrealized_pnl: float = 0.0,
                              realized_pnl: float = 0.0, daily_return: Optional[float] = None,
                              cumulative_return: Optional[float] = None):
        """Add portfolio snapshot for a date"""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO portfolio
            (date, balance, total_value, positions_count, unrealized_pnl,
             realized_pnl, daily_return, cumulative_return)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (date, balance, total_value, positions_count, unrealized_pnl,
              realized_pnl, daily_return, cumulative_return))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_portfolio_history(self, date_from: Optional[str] = None,
                             limit: Optional[int] = None) -> pd.DataFrame:
        """Get portfolio history"""
        query = "SELECT * FROM portfolio WHERE 1=1"
        params = []
        
        if date_from:
            query += " AND date >= ?"
            params.append(date_from)
        
        query += " ORDER BY date DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    # ============================================================================
    # MODEL PERFORMANCE METHODS
    # ============================================================================
    
    def add_model_performance(self, model_name: str, model_version: Optional[str] = None,
                             training_date: Optional[str] = None, ticker: Optional[str] = None,
                             mode: Optional[str] = None, interval: Optional[str] = None,
                             total_trades: Optional[int] = None, winning_trades: Optional[int] = None,
                             losing_trades: Optional[int] = None, win_rate: Optional[float] = None,
                             avg_profit: Optional[float] = None, avg_loss: Optional[float] = None,
                             total_pnl: Optional[float] = None, sharpe_ratio: Optional[float] = None,
                             max_drawdown: Optional[float] = None, training_steps: Optional[int] = None,
                             final_reward: Optional[float] = None, config: Optional[Dict] = None,
                             model_path: Optional[str] = None):
        """Add model performance metrics"""
        config_json = json.dumps(config) if config else None
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO model_performance
            (model_name, model_version, training_date, ticker, mode, interval,
             total_trades, winning_trades, losing_trades, win_rate, avg_profit,
             avg_loss, total_pnl, sharpe_ratio, max_drawdown, training_steps,
             final_reward, config_json, model_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (model_name, model_version, training_date, ticker, mode, interval,
              total_trades, winning_trades, losing_trades, win_rate, avg_profit,
              avg_loss, total_pnl, sharpe_ratio, max_drawdown, training_steps,
              final_reward, config_json, model_path))
        self.conn.commit()
        return cursor.lastrowid
    
    def get_model_performance(self, model_name: Optional[str] = None,
                             limit: Optional[int] = None) -> pd.DataFrame:
        """Get model performance history"""
        query = "SELECT * FROM model_performance WHERE 1=1"
        params = []
        
        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)
        
        query += " ORDER BY training_date DESC, created_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        return pd.read_sql_query(query, self.conn, params=params)
    
    # ============================================================================
    # MARKET DATA CACHE METHODS
    # ============================================================================
    
    def cache_market_data(self, ticker: str, date: str, data: Dict,
                         interval: str = "1d"):
        """Cache market data to avoid API calls"""
        data_json = json.dumps(data)
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO market_data_cache
            (ticker, date, open, high, low, close, volume, interval, data_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (ticker, date, data.get('Open'), data.get('High'), data.get('Low'),
              data.get('Close'), data.get('Volume'), interval, data_json))
        self.conn.commit()
    
    def get_cached_market_data(self, ticker: str, date: str,
                              interval: str = "1d") -> Optional[Dict]:
        """Get cached market data"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT data_json FROM market_data_cache
            WHERE ticker = ? AND date = ? AND interval = ?
        """, (ticker, date, interval))
        row = cursor.fetchone()
        
        if row:
            return json.loads(row['data_json'])
        return None
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def backup(self, backup_path: str):
        """Create backup of database"""
        import shutil
        shutil.copy2(self.db_path, backup_path)
        print(f"Database backed up to {backup_path}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example usage
    db = TradingDB("trading_bot.db")
    
    # Add a trade
    trade_id = db.add_trade(
        ticker="NVDA",
        action="BUY",
        shares=10.0,
        price=150.50,
        commission=1.50,
        timestamp=datetime.now()
    )
    print(f"Added trade ID: {trade_id}")
    
    # Add a sell trade
    db.add_trade(
        ticker="NVDA",
        action="SELL",
        shares=10.0,
        price=155.00,
        commission=1.55,
        entry_price=150.50,
        exit_price=155.00,
        pnl=45.00,
        pnl_pct=2.99,
        timestamp=datetime.now()
    )
    
    # Get trades
    trades = db.get_trades(ticker="NVDA")
    print("\nTrades:")
    print(trades)
    
    # Get stats
    stats = db.get_trade_stats(ticker="NVDA")
    print("\nStats:")
    print(stats)
    
    # Add prediction
    db.add_prediction(
        ticker="NVDA",
        prediction_date="2025-12-14",
        predicted_price=160.00,
        predicted_return=0.03,
        confidence=0.85,
        signal="BUY",
        model_version="v0.1"
    )
    
    # Get predictions
    predictions = db.get_predictions(ticker="NVDA")
    print("\nPredictions:")
    print(predictions)
    
    db.close()
    print("\nDatabase operations complete!")

