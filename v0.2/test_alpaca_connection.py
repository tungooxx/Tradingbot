"""
Test Alpaca Connection (v0.2)
==============================
Quick test to verify Alpaca API connection and account access.

Usage:
    python test_alpaca_connection.py
"""

import os
import sys
from alpaca_client import AlpacaClient

def main():
    """Test Alpaca connection"""
    # Check API keys
    if not os.getenv('ALPACA_API_KEY') or not os.getenv('ALPACA_SECRET_KEY'):
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
        print("\nExample (PowerShell):")
        print('  $env:ALPACA_API_KEY = "YOUR_KEY"')
        print('  $env:ALPACA_SECRET_KEY = "YOUR_SECRET"')
        sys.exit(1)
    
    try:
        # Initialize with paper trading
        print("Connecting to Alpaca Paper Trading...")
        client = AlpacaClient(paper=True)
        
        # Get account info
        print("\n" + "=" * 60)
        print("ACCOUNT INFORMATION")
        print("=" * 60)
        account = client.get_account()
        print(f"Balance: ${account['balance']:,.2f}")
        print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
        print(f"Buying Power: ${account['buying_power']:,.2f}")
        print(f"Day Trading Buying Power: ${account['day_trading_buying_power']:,.2f}")
        print(f"Trading Blocked: {account['trading_blocked']}")
        print(f"Account Blocked: {account['account_blocked']}")
        
        # Get positions
        print("\n" + "=" * 60)
        print("CURRENT POSITIONS")
        print("=" * 60)
        positions = client.get_positions()
        if len(positions) == 0:
            print("No open positions")
        else:
            for symbol, pos in positions.items():
                print(f"\n{symbol}:")
                print(f"  Quantity: {pos['qty']} shares")
                print(f"  Avg Entry Price: ${pos['avg_entry_price']:.2f}")
                print(f"  Current Price: ${pos['current_price']:.2f}")
                print(f"  Market Value: ${pos['market_value']:,.2f}")
                print(f"  Unrealized P&L: ${pos['unrealized_pl']:,.2f} ({pos['unrealized_plpc']:.2f}%)")
        
        # Get recent orders
        print("\n" + "=" * 60)
        print("RECENT ORDERS (Last 5)")
        print("=" * 60)
        orders = client.get_orders(status='all', limit=5)
        if len(orders) == 0:
            print("No recent orders")
        else:
            for order in orders:
                print(f"\n{order['symbol']} {order['side'].upper()}:")
                print(f"  Quantity: {order['qty']} shares")
                print(f"  Type: {order['type']}")
                print(f"  Status: {order['status']}")
                if order['filled_qty'] > 0:
                    print(f"  Filled: {order['filled_qty']} shares @ ${order['filled_avg_price']:.2f}")
                print(f"  Submitted: {order['submitted_at']}")
        
        # Test getting latest price
        print("\n" + "=" * 60)
        print("LATEST PRICE TEST")
        print("=" * 60)
        test_symbol = "TSLA"
        bar = client.get_latest_bar(test_symbol)
        if bar:
            print(f"{test_symbol} Latest Bar:")
            print(f"  Price: ${bar['close']:.2f}")
            print(f"  Volume: {bar['volume']:,}")
            print(f"  Timestamp: {bar['timestamp']}")
        else:
            print(f"Could not get latest bar for {test_symbol} (may need SIP subscription)")
        
        print("\n" + "=" * 60)
        print("✅ Alpaca connection successful!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
