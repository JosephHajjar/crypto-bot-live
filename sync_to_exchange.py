"""
One-time script to sync the ensemble bot's state with the real Hyperliquid position.

This script:
1. Queries Hyperliquid for the current open BTC position
2. Overwrites live_state_ensemble.json with correct exchange data
3. Injects a synthetic trade entry into live_trades_ensemble.json

Run this ONCE, then start the bot normally.
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv
from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.utils import constants

load_dotenv()

COIN = 'BTC'
STATE_FILE = 'data_storage/live_state_ensemble.json'
TRADES_FILE = 'data_storage/live_trades_ensemble.json'

# Load model configs for TP/SL values
with open('models/holy_grail_config.json', 'r') as f:
    cfg_long = json.load(f)
with open('models_short/holy_grail_short_config.json', 'r') as f:
    cfg_short = json.load(f)

LONG_TP = cfg_long.get('take_profit', 0.0125)
LONG_SL = cfg_long.get('stop_loss', 0.0250)
SHORT_TP = cfg_short.get('take_profit', 0.0175)
SHORT_SL = cfg_short.get('stop_loss', 0.0375)


def main():
    wallet_address = os.environ.get("HYPERLIQUID_WALLET_ADDRESS").strip()
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    # ---- 1. Query current exchange state ----
    user_state = info.user_state(wallet_address)
    margin_summary = user_state.get("marginSummary", {})
    account_value = float(margin_summary.get("accountValue", 0.0))

    # Find BTC position
    btc_position = None
    for pos in user_state.get("assetPositions", []):
        if pos['position']['coin'] == COIN:
            btc_position = pos['position']
            break

    if btc_position is None:
        print("ERROR: No open BTC position found on Hyperliquid!")
        print("Cannot sync — there's nothing to adopt.")
        return

    # Extract position details
    szi = float(btc_position['szi'])           # Signed size: positive = long, negative = short
    entry_price = float(btc_position['entryPx'])
    unrealized_pnl = float(btc_position['unrealizedPnl'])
    position_value = float(btc_position['positionValue'])
    margin_used = float(btc_position['marginUsed'])

    direction = 'long' if szi > 0 else 'short'
    size_btc = abs(szi)

    # In Hyperliquid unified account, accountValue IS the portfolio value (includes unrealized PnL)
    # Do NOT add spot USDC — it's the same pool
    total_balance = account_value

    # Compute current price from unrealized PnL
    if direction == 'long':
        current_price = entry_price + (unrealized_pnl / size_btc)
        tp_target = entry_price * (1 + LONG_TP)
        sl_target = entry_price * (1 - LONG_SL)
    else:
        current_price = entry_price - (unrealized_pnl / size_btc)
        tp_target = entry_price * (1 - SHORT_TP)
        sl_target = entry_price * (1 + SHORT_SL)

    # Compute open PnL
    if direction == 'long':
        open_pnl_pct = (current_price - entry_price) / entry_price * 100
        open_pnl_usd = size_btc * (current_price - entry_price)
    else:
        open_pnl_pct = (entry_price - current_price) / entry_price * 100
        open_pnl_usd = size_btc * (entry_price - current_price)

    print("=" * 60)
    print("EXCHANGE POSITION DETECTED:")
    print(f"  Direction:    {direction.upper()}")
    print(f"  Size:         {size_btc} BTC")
    print(f"  Entry Price:  ${entry_price:.2f}")
    print(f"  Current Price:${current_price:.2f}")
    print(f"  Unrealized:   ${unrealized_pnl:.4f}")
    print(f"  Account Value:${account_value:.4f}")
    print(f"  Total Balance:${total_balance:.4f}")
    print(f"  TP Target:    ${tp_target:.2f}")
    print(f"  SL Target:    ${sl_target:.2f}")
    print("=" * 60)

    # ---- 2. Build new state ----
    new_state = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "paper_balance": round(total_balance - open_pnl_usd, 2),  # Bankroll at trade start = portfolio value minus unrealized PnL
        "current_price": round(current_price, 2),
        "bull_prob": 0.0,
        "bear_prob": 0.0,
        "master_control": "ALT",  # ALT mode since this is a high-conviction adopted position
        "in_trade": True,
        "trade_type": direction.upper(),
        "entry_price": entry_price,
        "bars_held": 0,  # Start counting from now
        "open_pnl_pct": round(open_pnl_pct, 4),
        "open_pnl_usd": round(open_pnl_usd, 2),
        "take_profit_target": round(tp_target, 2),
        "stop_loss_target": round(sl_target, 2),
        "trade_amount_btc": size_btc,
        "trade_amount_usd": round(size_btc * current_price, 2),
        "last_error": None,
        "peak_price": round(current_price, 2),
        "trailing_armed": False
    }

    # ---- 3. Build trade log with adoption entry ----
    adoption_record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "trade_type": direction.upper(),
        "action": "ADOPTED",
        "entry_price": entry_price,
        "exit_price": None,
        "return_pct": None,
        "pnl_usd": None,
        "bars_held": 0,
        "reason": f"Position adopted from Hyperliquid (manually opened by user). Entry ${entry_price:.2f}, Size {size_btc} BTC, {direction.upper()}",
        "size_btc": size_btc,
        "adopted": True
    }

    # Load existing trades (should be empty but be safe)
    trades = []
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, "r") as f:
                trades = json.load(f)
        except Exception:
            pass

    trades.append(adoption_record)

    # ---- 4. Write files ----
    with open(STATE_FILE, "w") as f:
        json.dump(new_state, f, indent=2)
    print(f"\n[OK] State written to {STATE_FILE}")

    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, indent=2)
    print(f"[OK] Trade log written to {TRADES_FILE}")

    # ---- 5. Verify ----
    print("\n" + "=" * 60)
    print("VERIFICATION — New State File:")
    with open(STATE_FILE, "r") as f:
        print(json.dumps(json.load(f), indent=2))

    print("\nVERIFICATION — Trade Log:")
    with open(TRADES_FILE, "r") as f:
        print(json.dumps(json.load(f), indent=2))

    print("\n[DONE] SYNC COMPLETE! The bot will now manage this position.")
    print("   Start the bot with: python trade_live_ensemble.py")


if __name__ == "__main__":
    main()
