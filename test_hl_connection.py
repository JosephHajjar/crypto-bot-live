import os
import time
from dotenv import load_dotenv

from eth_account import Account
import eth_account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

def test_connection():
    print("Testing Hyperliquid SDK Connection...")
    load_dotenv()
    
    wallet_address = os.environ.get("HYPERLIQUID_WALLET_ADDRESS").strip() if os.environ.get("HYPERLIQUID_WALLET_ADDRESS") else None
    secret_key = os.environ.get("HYPERLIQUID_API_SECRET").strip() if os.environ.get("HYPERLIQUID_API_SECRET") else None
    
    if not wallet_address or not secret_key:
        print("ERROR: Missing HYPERLIQUID_WALLET_ADDRESS or HYPERLIQUID_API_SECRET in .env file.")
        return
        
    try:
        account = Account.from_key(secret_key)
        print(f"Agent Address configured: {account.address}")
        print(f"Main Wallet Address: {wallet_address}")
    except Exception as e:
        print(f"ERROR: Could not parse private key. Make sure it's valid: {e}")
        return
    
    try:
        # Initialize info endpoint
        info = Info(constants.MAINNET_API_URL, skip_ws=True)
        
        # Test 1: Fetch user clearinghouse state (margin/balances)
        user_state = info.user_state(wallet_address)
        
        margin_summary = user_state.get("marginSummary", {})
        account_value = float(margin_summary.get("accountValue", 0.0))
        total_margin_used = float(margin_summary.get("totalMarginUsed", 0.0))
        
        print("\n--- Hyperliquid Account State ---")
        print(f"Total Account Value: ${account_value:.2f}")
        print(f"Total Margin Used: ${total_margin_used:.2f}")
        print(f"Available Margin: ${account_value - total_margin_used:.2f}")

        # Test Spot
        spot_state = info.spot_user_state(wallet_address)
        print("\n--- Spot State ---")
        for bal in spot_state.get("balances", []):
            print(f"Coin: {bal.get('coin')} | Total: {bal.get('total')}")
        
        # Output any open positions
        positions = user_state.get("assetPositions", [])
        if positions:
            print("\n--- Active Positions ---")
            for pos in positions:
                p = pos.get("position", {})
                coin = p.get("coin")
                size = p.get("szi")
                entry = p.get("entryPx")
                leverage = p.get("leverage", {}).get("value")
                print(f"Coin: {coin} | Size: {size} | Entry Price: {entry} | Leverage: {leverage}x")
        else:
            print("\nNo active positions.")
            
        print("\nConnection test successful!")
        
    except Exception as e:
        print(f"ERROR: Failed to connect to Hyperliquid API: {e}")

if __name__ == "__main__":
    test_connection()
