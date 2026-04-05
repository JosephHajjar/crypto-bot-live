import os

def create_trade_gold():
    path = r'c:\Users\asdf\.gemini\antigravity\scratch\ml_trading_bot'

    with open(os.path.join(path, "trade_live.py"), "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace("SYMBOL = 'BTC/USDT'", "SYMBOL = 'PAXG/USDT'")
    content = content.replace("COIN = 'BTC'", "COIN = 'PAXG'")
    content = content.replace("MODEL_LONG_PATH = 'models/holy_grail.pth'", "MODEL_LONG_PATH = 'models_gold_long/holy_grail.pth'")
    content = content.replace("CONFIG_LONG_PATH = 'models/holy_grail_config.json'", "CONFIG_LONG_PATH = 'models_gold_long/holy_grail_config.json'")
    content = content.replace("MODEL_SHORT_PATH = 'models_short/holy_grail_short.pth'", "MODEL_SHORT_PATH = 'models_gold_short/holy_grail_short.pth'")
    content = content.replace("CONFIG_SHORT_PATH = 'models_short/holy_grail_short_config.json'", "CONFIG_SHORT_PATH = 'models_gold_short/holy_grail_short_config.json'")
    content = content.replace("SCALER_PATH = 'data_storage/BTC_USDT_15m_scaler.json'", "SCALER_PATH = 'data_storage/PAXG_USDT_15m_scaler.json'")
    
    # State tracking replacements
    content = content.replace('data_storage/live_state.json', 'data_storage/live_state_gold.json')
    content = content.replace('data_storage/live_trades.json', 'data_storage/live_trades_gold.json')
    
    # Text replacements to avoid confusion in notifications
    content = content.replace("trade_amount_btc", "trade_amount_paxg")
    content = content.replace("BTC Price", "PAXG Price")

    with open(os.path.join(path, "trade_live_gold.py"), "w", encoding="utf-8") as f:
        f.write(content)

    print("Created trade_live_gold.py successfully.")

if __name__ == "__main__":
    create_trade_gold()
