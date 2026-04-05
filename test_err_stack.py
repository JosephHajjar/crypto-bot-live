import traceback
import sys
sys.path.insert(0, '.')
from dashboard import _load_bot_model, get_bot_signals, app

def test():
    with app.app_context():
        # Inject mock request args to simulate /api/bot_signals?limit=10000
        from flask import request
        with app.test_request_context('/api/bot_signals?limit=10000'):
            try:
                print("Running...")
                res = get_bot_signals()
                j = res.json
                if "error" in j:
                    print("ERROR RETURNED:", j["error"])
                else:
                    signals = j.get("signals", [])
                    buys = len([s for s in signals if s.get("signal") == "BUY"])
                    sells = len([s for s in signals if s.get("signal") == "SELL"])
                    closes = len([s for s in signals if s.get("signal") == "CLOSE"])
                    print(f"SUCCESS, total: {len(signals)}, BUYS: {buys}, SELLS: {sells}, CLOSES: {closes}")
                    print("Last 3 SELL signals:")
                    sell_sigs = [s for s in signals if s.get("signal") == "SELL"]
                    for s in sell_sigs[-3:]:
                        print(s)
            except Exception as e:
                print("CAUGHT!")
                traceback.print_exc()

if __name__ == "__main__":
    test()
