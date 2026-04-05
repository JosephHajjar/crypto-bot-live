import requests
try:
    res = requests.get("http://127.0.0.1:5001/api/bot_signals?symbol=BTCUSDT&interval=15m&limit=10000", timeout=10)
    j = res.json()
    signals = j.get("signals", [])
    buys = len([s for s in signals if s.get("signal") == "BUY"])
    sells = len([s for s in signals if s.get("signal") == "SELL"])
    closes = len([s for s in signals if s.get("signal") == "CLOSE"])
    print(f"API Returned: BUYS: {buys}, SELLS: {sells}, CLOSES: {closes}")
except Exception as e:
    print(e)
