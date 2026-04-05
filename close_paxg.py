import os
import sys
from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

load_dotenv(r'c:\Users\asdf\.gemini\antigravity\scratch\ml_trading_bot\.env')
wallet_address = os.environ.get("HYPERLIQUID_WALLET_ADDRESS").strip()
secret_key = os.environ.get("HYPERLIQUID_API_SECRET").strip()

account = Account.from_key(secret_key)
info = Info(constants.MAINNET_API_URL, skip_ws=True)
exchange = Exchange(account, constants.MAINNET_API_URL, account_address=wallet_address)

user_state = info.user_state(wallet_address)
positions = user_state.get("assetPositions", [])
for pos in positions:
    p = pos.get("position", {})
    if p.get("coin") == "PAXG" and float(p.get("szi", 0)) != 0.0:
        szi = float(p.get("szi"))
        print(f"Closing PAXG position of {szi}")
        res = exchange.market_close("PAXG", abs(szi))
        print("Response:", res)
