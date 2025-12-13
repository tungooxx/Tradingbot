"""
Test Alpaca Connection
======================
Quick test to verify Alpaca API keys work.
"""

import os
import sys
from alpaca_client import AlpacaClient

# Set API keys (for testing)
os.environ['ALPACA_API_KEY'] = 'PKB7FVE3HR4C4RWAQNW7M7TM6C'
os.environ['ALPACA_SECRET_KEY'] = 'gznmcC7Q9cTPpKyEpiS5m2ti4Ugvojy9WjzovbFGFoT'

def test_connection():
    """Test Alpaca connection"""
    try:
        print("Testing Alpaca connection...")
        print("=" * 50)
        
        # Initialize client
        client = AlpacaClient(paper=True)
        print("[OK] Client initialized")
        
        # Get account
        account = client.get_account()
        print(f"\nPaper Trading Account:")
        print(f"   Balance: ${account['balance']:,.2f}")
        print(f"   Portfolio Value: ${account['portfolio_value']:,.2f}")
        print(f"   Buying Power: ${account['buying_power']:,.2f}")
        print(f"   Trading Blocked: {account['trading_blocked']}")
        
        # Get positions
        positions = client.get_positions()
        print(f"\nCurrent Positions: {len(positions)}")
        for symbol, pos in positions.items():
            print(f"   {symbol}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f}")
        
        # Get latest price
        bar = client.get_latest_bar('NVDA')
        if bar:
            print(f"\nNVDA Latest Price: ${bar['close']:.2f}")
        
        # Get recent orders
        orders = client.get_orders(status='all', limit=5)
        print(f"\nRecent Orders: {len(orders)}")
        for order in orders[:3]:
            print(f"   {order['symbol']} {order['side']} {order['qty']} - {order['status']}")
        
        print("\n" + "=" * 50)
        print("[SUCCESS] Alpaca connection successful!")
        print("[SUCCESS] Ready to start paper trading!")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check API keys are correct")
        print("2. Check internet connection")
        print("3. Verify Alpaca account is active")
        return False

if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)

