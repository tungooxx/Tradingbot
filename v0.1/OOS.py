from alpaca_client import AlpacaClient

# 🚨 REPLACE 'YOUR_API_KEY' and 'YOUR_SECRET_KEY' with your actual keys
API_KEY = "PKB7FVE3HR4C4RWAQNW7M7TM6C"
SECRET_KEY = "gznmcC7Q9cTPpKyEpiS5m2ti4Ugvojy9WjzovbFGFoT"

client = AlpacaClient(
    paper=True, 
    api_key=API_KEY, 
    secret_key=SECRET_KEY
)

account = client.get_account()
print(f"Paper Trading Balance: ${account['balance']:,.2f}")