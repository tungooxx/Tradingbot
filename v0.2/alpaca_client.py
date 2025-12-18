"""
ALPACA API CLIENT (v0.2)
========================
Integration with Alpaca Markets API for paper and live trading.

This replaces yfinance for real-time data and enables actual order execution.

Setup:
1. Sign up at https://alpaca.markets/
2. Get API keys from paper trading dashboard
3. Set environment variables:
   - ALPACA_API_KEY
   - ALPACA_SECRET_KEY
4. pip install alpaca-trade-api

Usage:
    from alpaca_client import AlpacaClient
    
    client = AlpacaClient(paper=True)  # Paper trading
    account = client.get_account()
    order = client.place_order('NVDA', 1, 'buy', 'market')
"""

import alpaca_trade_api as tradeapi
from typing import Dict, Optional, List
import os
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("AlpacaClient")


class AlpacaClient:
    """Alpaca API client for trading"""
    
    def __init__(self, paper: bool = True, api_key: Optional[str] = None,
                 secret_key: Optional[str] = None):
        """
        Initialize Alpaca client
        
        Args:
            paper: If True, use paper trading (default: True)
            api_key: API key (or use ALPACA_API_KEY env var)
            secret_key: Secret key (or use ALPACA_SECRET_KEY env var)
        """
        self.paper = paper
        
        # Get API keys
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError(
                "API keys required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
                "environment variables or pass as arguments."
            )
        
        # Initialize client
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        self.api = tradeapi.REST(
            self.api_key,
            self.secret_key,
            base_url,
            api_version='v2'
        )
        
        logger.info(f"Alpaca client initialized ({'PAPER' if paper else 'LIVE'} trading)")
    
    def get_account(self) -> Dict:
        """Get account information"""
        try:
            account = self.api.get_account()
            return {
                'balance': float(account.cash),
                'equity': float(account.equity),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'day_trading_buying_power': float(account.daytrading_buying_power),
                'pattern_day_trader': account.pattern_day_trader,
                'trading_blocked': account.trading_blocked,
                'account_blocked': account.account_blocked,
                'currency': account.currency
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            raise
    
    def get_positions(self) -> Dict[str, Dict]:
        """Get current positions"""
        try:
            positions = self.api.list_positions()
            result = {}
            for pos in positions:
                result[pos.symbol] = {
                    'symbol': pos.symbol,
                    'qty': float(pos.qty),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'unrealized_plpc': float(pos.unrealized_plpc),
                    'side': pos.side
                }
            return result
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def place_order(self, symbol: str, qty: float, side: str,
                   order_type: str = 'market', limit_price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   time_in_force: str = 'day') -> Dict:
        """
        Place an order
        
        Args:
            symbol: Stock symbol (e.g., 'NVDA', 'AAPL')
            qty: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            limit_price: For limit orders
            stop_price: For stop orders
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
        
        Returns:
            Order details dict
        """
        try:
            if order_type == 'market':
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force=time_in_force
                )
            elif order_type == 'limit':
                if not limit_price:
                    raise ValueError("limit_price required for limit orders")
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='limit',
                    limit_price=limit_price,
                    time_in_force=time_in_force
                )
            elif order_type == 'stop':
                if not stop_price:
                    raise ValueError("stop_price required for stop orders")
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='stop',
                    stop_price=stop_price,
                    time_in_force=time_in_force
                )
            elif order_type == 'stop_limit':
                if not limit_price or not stop_price:
                    raise ValueError("limit_price and stop_price required for stop_limit orders")
                order = self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='stop_limit',
                    limit_price=limit_price,
                    stop_price=stop_price,
                    time_in_force=time_in_force
                )
            else:
                raise ValueError(f"Unsupported order_type: {order_type}")
            
            logger.info(f"Order placed: {side.upper()} {qty} {symbol} ({order_type})")
            
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None
            }
        except Exception as e:
            logger.error(f"Order failed: {e}")
            raise Exception(f"Order failed: {e}")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order {order_id} cancelled")
            return True
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False
    
    def cancel_all_orders(self) -> bool:
        """Cancel all open orders"""
        try:
            self.api.cancel_all_orders()
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")
            return False
    
    def get_orders(self, status: str = 'all', limit: int = 50) -> List[Dict]:
        """
        Get order history
        
        Args:
            status: 'all', 'open', 'closed'
            limit: Max number of orders
        """
        try:
            orders = self.api.list_orders(
                status=status,
                limit=limit
            )
            return [{
                'id': o.id,
                'symbol': o.symbol,
                'qty': float(o.qty),
                'side': o.side,
                'type': o.type,
                'status': o.status,
                'filled_qty': float(o.filled_qty) if o.filled_qty else 0,
                'filled_avg_price': float(o.filled_avg_price) if o.filled_avg_price else None,
                'submitted_at': o.submitted_at.isoformat() if o.submitted_at else None,
                'filled_at': o.filled_at.isoformat() if o.filled_at else None
            } for o in orders]
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def get_order(self, order_id: str) -> Optional[Dict]:
        """Get specific order by ID"""
        try:
            order = self.api.get_order(order_id)
            return {
                'id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side,
                'type': order.type,
                'status': order.status,
                'filled_qty': float(order.filled_qty) if order.filled_qty else 0,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'submitted_at': order.submitted_at.isoformat() if order.submitted_at else None,
                'filled_at': order.filled_at.isoformat() if order.filled_at else None
            }
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            return None
    
    def get_latest_bar(self, symbol: str) -> Optional[Dict]:
        """Get latest bar (OHLCV) for symbol"""
        try:
            bars = self.api.get_bars(symbol, '1Min', limit=1)
            if bars:
                bar = bars[0]
                return {
                    'symbol': symbol,
                    'timestamp': bar.t.isoformat(),
                    'open': float(bar.o),
                    'high': float(bar.h),
                    'low': float(bar.l),
                    'close': float(bar.c),
                    'volume': int(bar.v)
                }
        except Exception as e:
            logger.error(f"Error getting bar for {symbol}: {e}")
        return None
    
    def get_historical_bars(self, symbol: str, timeframe: str = '1Day',
                           start: Optional[str] = None,
                           end: Optional[str] = None,
                           limit: int = 100) -> List[Dict]:
        """
        Get historical bars
        
        Args:
            symbol: Stock symbol
            timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'
            start: Start date (YYYY-MM-DD or datetime)
            end: End date (YYYY-MM-DD or datetime)
            limit: Max number of bars
        """
        try:
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start,
                end=end,
                limit=limit
            )
            return [{
                'timestamp': bar.t.isoformat(),
                'open': float(bar.o),
                'high': float(bar.h),
                'low': float(bar.l),
                'close': float(bar.c),
                'volume': int(bar.v)
            } for bar in bars]
        except Exception as e:
            logger.error(f"Error getting historical bars for {symbol}: {e}")
            return []
    
    def close_all_positions(self) -> bool:
        """Close all open positions"""
        try:
            self.api.close_all_positions()
            logger.info("All positions closed")
            return True
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
            return False
    
    def close_position(self, symbol: str) -> bool:
        """Close position for specific symbol"""
        try:
            self.api.close_position(symbol)
            logger.info(f"Position closed for {symbol}")
            return True
        except Exception as e:
            logger.error(f"Error closing position for {symbol}: {e}")
            return False


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check if API keys are set
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_SECRET_KEY'):
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("\nExample (PowerShell):")
        print('  $env:ALPACA_API_KEY = "YOUR_KEY"')
        print('  $env:ALPACA_SECRET_KEY = "YOUR_SECRET"')
        sys.exit(1)
    
    try:
        # Initialize with paper trading
        client = AlpacaClient(paper=True)
        
        # Get account info
        account = client.get_account()
        print(f"\nPaper Trading Account:")
        print(f"  Balance: ${account['balance']:,.2f}")
        print(f"  Portfolio Value: ${account['portfolio_value']:,.2f}")
        print(f"  Buying Power: ${account['buying_power']:,.2f}")
        
        # Get positions
        positions = client.get_positions()
        print(f"\nCurrent Positions: {len(positions)}")
        for symbol, pos in positions.items():
            print(f"  {symbol}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f}")
        
        # Get latest price
        bar = client.get_latest_bar('NVDA')
        if bar:
            print(f"\nNVDA Latest Price: ${bar['close']:.2f}")
        
        # Get recent orders
        orders = client.get_orders(status='all', limit=5)
        print(f"\nRecent Orders: {len(orders)}")
        for order in orders[:3]:
            print(f"  {order['symbol']} {order['side']} {order['qty']} - {order['status']}")
        
        print("\nAlpaca connection successful!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
