import requests

res = requests.get("http://127.0.0.1:5001/api/bot_signals?limit=10000", timeout=10)
j = res.json()
signals = j.get("signals", [])
buy_sigs = [s for s in signals if s.get("signal") == "BUY"]
sell_sigs = [s for s in signals if s.get("signal") == "SELL"]
print(f"Total: {len(signals)}")
print(f"BUYS: {len(buy_sigs)}")
print(f"SELLS: {len(sell_sigs)}")

if len(sell_sigs) > 0:
    print("SELLS ARE PRESENT.")
else:
    print("NO SELLS FOUND. Printing some probabilities if possible.")
