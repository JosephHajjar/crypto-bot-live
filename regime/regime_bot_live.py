"""
Quant Regime Live Bot — Hyperliquid, 1x Leverage Only.
No ML. No overfitting. Pure math regime detection.

Strategy:
  - BULL/NEUTRAL regime -> LONG BTC (1x)
  - BEAR regime -> SHORT BTC (1x)
  
Checks regime every hour. Only changes position when regime flips.
Writes state to data_storage/regime_state.json for dashboard.
"""
import pandas as pd
import numpy as np
import time
import os
import json
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_regime import score_regime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - REGIME - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('regime_bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ─── CONFIG ───
COIN = 'ETH'  # ETH has $0.23 min order (BTC needs $78). Uses BTC regime signal.
BTC_MIN_NOTIONAL = 78  # Auto-switch to BTC when balance exceeds this
LEVERAGE = 3  # 3x leverage required to meet Hyperliquid $10 minimum order with $4.63
CHECK_INTERVAL = 900  # Check regime every 15 minutes (minimizes execution delay)
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data_storage', 'regime_state.json')

class RegimeBot:
    def __init__(self):
        load_dotenv()
        self.wallet_address = os.environ.get("HYPERLIQUID_WALLET_ADDRESS", "").strip()
        self.secret_key = os.environ.get("HYPERLIQUID_API_SECRET", "").strip()
        if not self.wallet_address or not self.secret_key:
            raise ValueError("Hyperliquid API Keys not found in .env!")

        self.account = Account.from_key(self.secret_key)
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        self.exchange = Exchange(self.account, constants.MAINNET_API_URL, account_address=self.wallet_address)

        self.current_regime = None  # 'BULL', 'NEUTRAL', 'BEAR'
        self.current_position = None  # 'long', 'short', None
        self.entry_price = 0.0
        self.position_size = 0.0
        self.balance = 0.0
        self.last_score = 0.0

        self._sync_balance()
        self._auto_select_coin()
        self._sync_from_exchange()

        logger.info("=" * 60)
        logger.info("  REGIME BOT INITIALIZED — 1x LEVERAGE, NO ML")
        logger.info(f"  Trading: {COIN} (auto-upgrades to BTC at ${BTC_MIN_NOTIONAL}+)")
        logger.info(f"  Balance: ${self.balance:.2f}")
        logger.info(f"  Current position: {self.current_position or 'FLAT'}")
        logger.info(f"  Check interval: {CHECK_INTERVAL}s ({CHECK_INTERVAL//60}m)")
        logger.info("=" * 60)

    def _auto_select_coin(self):
        """Auto-switch to BTC if balance is high enough, otherwise ETH."""
        global COIN
        if self.balance >= BTC_MIN_NOTIONAL:
            COIN = 'BTC'
            logger.info(f"Balance ${self.balance:.2f} >= ${BTC_MIN_NOTIONAL}: trading BTC")
        else:
            COIN = 'ETH'
            logger.info(f"Balance ${self.balance:.2f} < ${BTC_MIN_NOTIONAL}: trading ETH (BTC regime signal)")
        try:
            self.exchange.update_leverage(LEVERAGE, COIN, is_cross=True)
            logger.info(f"Set {COIN} leverage to {LEVERAGE}x (cross)")
        except Exception as e:
            logger.warning(f"Could not set leverage: {e}")

    # ─── EXCHANGE ───

    def _sync_balance(self):
        try:
            user_state = self.info.user_state(self.wallet_address)
            margin = user_state.get("marginSummary", {})
            perps_value = float(margin.get("accountValue", 0.0))

            spot_state = self.info.spot_user_state(self.wallet_address)
            spot_usdc = 0.0
            for bal in spot_state.get("balances", []):
                if bal.get("coin") == "USDC":
                    spot_usdc = float(bal.get("total", 0.0))
                    break

            self.balance = perps_value + spot_usdc
            logger.info(f"Balance: ${self.balance:.2f} (Perps: ${perps_value:.2f}, Spot: ${spot_usdc:.2f})")
        except Exception as e:
            logger.error(f"Balance sync failed: {e}")

    def _sync_from_exchange(self):
        try:
            user_state = self.info.user_state(self.wallet_address)
            exchange_pos = None
            for pos in user_state.get("assetPositions", []):
                if pos['position']['coin'] == COIN:
                    exchange_pos = pos['position']
                    break

            if exchange_pos:
                size = float(exchange_pos['szi'])
                if abs(size) >= 0.00001:
                    self.current_position = 'long' if size > 0 else 'short'
                    self.entry_price = float(exchange_pos['entryPx'])
                    self.position_size = abs(size)
                    logger.info(f"Exchange position: {self.current_position.upper()} {self.position_size} BTC @ ${self.entry_price:.2f}")
                    return

            self.current_position = None
            self.entry_price = 0.0
            self.position_size = 0.0
            logger.info("Exchange position: FLAT")
        except Exception as e:
            logger.error(f"Exchange sync failed: {e}")

    def _ensure_perps_funded(self):
        """Unified margin automatically uses Spot USDC. No transfer needed."""
        pass

    def _execute_position(self, target_position, current_price):
        """
        Ensure exchange position matches target.
        target_position: 'long', 'short', or 'flat'
        """
        self._ensure_perps_funded()
        self._sync_balance()

        # Calculate target size
        # Get coin's size decimals from exchange meta
        try:
            meta = self.info.meta()
            sz_dec = 5  # default
            for asset in meta.get('universe', []):
                if asset['name'] == COIN:
                    sz_dec = asset.get('szDecimals', 5)
                    break
        except:
            sz_dec = 3 if COIN == 'ETH' else 5

        if target_position == 'flat':
            target_size = 0.0
        else:
            # Use coin's own price for sizing, not BTC
            coin_mids = self.info.all_mids()
            coin_price = float(coin_mids.get(COIN, current_price))
            
            # Size = (balance * leverage) / price
            target_notional = self.balance * LEVERAGE
            target_size = round(target_notional / coin_price, sz_dec)
            min_sz = 10 ** (-sz_dec)
            target_size = max(min_sz, target_size)

        target_szi = target_size if target_position == 'long' else (-target_size if target_position == 'short' else 0.0)

        try:
            # Get current exchange position
            user_state = self.info.user_state(self.wallet_address)
            current_szi = 0.0
            for pos in user_state.get("assetPositions", []):
                if pos['position']['coin'] == COIN:
                    current_szi = float(pos['position']['szi'])
                    break

            diff = target_szi - current_szi
            if abs(diff) < 0.00001:
                logger.info(f"Position already correct: {target_position}")
                return True

            is_buy = diff > 0
            trade_size = abs(diff)

            logger.info(f"ORDER: {'BUY' if is_buy else 'SELL'} {trade_size:.6f} BTC (target: {target_position})")
            res = self.exchange.market_open(COIN, is_buy=is_buy, sz=float(trade_size), slippage=0.01)

            if res and res.get('status') == 'ok':
                logger.info(f"ORDER OK: {target_position.upper()} {trade_size:.6f} BTC")
                self.current_position = target_position if target_position != 'flat' else None
                self.entry_price = current_price if target_position != 'flat' else 0.0
                self.position_size = target_size if target_position != 'flat' else 0.0
                return True
            else:
                logger.error(f"ORDER FAILED: {res}")
                return False

        except Exception as e:
            logger.error(f"Execute position failed: {e}")
            return False

    # ─── DATA ───

    def fetch_daily_data(self):
        """Fetch 250 days of daily BTC from Binance (enough for 200 SMA + buffer)."""
        import requests
        all_data = []
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - (300 * 24 * 60 * 60 * 1000)  # 300 days

        url = 'https://api.binance.com/api/v3/klines'
        current = start_ms
        while current < end_ms:
            params = {'symbol': 'BTCUSDT', 'interval': '1d', 'startTime': current, 'limit': 1000}
            try:
                r = requests.get(url, params=params, timeout=15)
                data = r.json()
                if not data or isinstance(data, dict):
                    break
                for row in data:
                    all_data.append({
                        'timestamp': pd.to_datetime(row[0], unit='ms'),
                        'open': float(row[1]), 'high': float(row[2]),
                        'low': float(row[3]), 'close': float(row[4]),
                        'volume': float(row[5])
                    })
                current = data[-1][0] + 86400000
            except Exception as e:
                logger.error(f"Binance fetch error: {e}")
                time.sleep(5)
                break
            time.sleep(0.1)

        df = pd.DataFrame(all_data).set_index('timestamp')
        df = df[~df.index.duplicated()].sort_index()
        return df

    # ─── STATE ───

    def save_state(self, regime_result, current_price):
        pnl_pct = 0.0
        pnl_usd = 0.0
        if self.current_position == 'long' and self.entry_price > 0:
            pnl_pct = (current_price - self.entry_price) / self.entry_price * 100
            pnl_usd = self.position_size * (current_price - self.entry_price)
        elif self.current_position == 'short' and self.entry_price > 0:
            pnl_pct = (self.entry_price - current_price) / self.entry_price * 100
            pnl_usd = self.position_size * (self.entry_price - current_price)

        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "strategy": "QUANT_REGIME",
            "coin": COIN,
            "leverage": LEVERAGE,
            "balance": round(self.balance, 2),
            "current_price": current_price,
            "regime": regime_result['regime'],
            "regime_score": round(regime_result['total_score'], 1),
            "position": self.current_position.upper() if self.current_position else "FLAT",
            "entry_price": self.entry_price,
            "position_size_btc": self.position_size,
            "position_size_usd": round(self.position_size * current_price, 2),
            "open_pnl_pct": round(pnl_pct, 3),
            "open_pnl_usd": round(pnl_usd, 2),
            "breakdown": {k: v['score'] for k, v in regime_result.get('breakdown', {}).items()},
            "last_check": datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        }

        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"State saved: {regime_result['regime']} (score={regime_result['total_score']:+.0f})")

    # ─── MAIN LOGIC ───

    def check_and_trade(self):
        """Core loop: fetch data, score regime, execute if needed."""
        logger.info("=" * 50)
        logger.info("REGIME CHECK")
        logger.info("=" * 50)

        # Fetch daily data (always BTC for regime scoring)
        df = self.fetch_daily_data()
        if len(df) < 201:
            logger.error(f"Not enough daily data: {len(df)} bars (need 201+)")
            return

        current_price = df['close'].iloc[-1]
        logger.info(f"BTC: ${current_price:,.2f} | {len(df)} daily bars")

        # Auto-select coin based on current balance
        self._sync_balance()
        self._auto_select_coin()
        logger.info(f"Trading: {COIN} at 1x leverage")

        # Score regime
        result = score_regime(df)
        new_regime = result['regime']
        self.last_score = result['total_score']

        logger.info(f"Regime: {new_regime} (score: {result['total_score']:+.0f}/100)")
        for name, sig in result.get('breakdown', {}).items():
            logger.info(f"  {name}: {sig['score']:+d}/{sig['max']}")

        # Determine target position
        if new_regime == 'BEAR':
            target = 'short'
        else:  # BULL or NEUTRAL
            target = 'long'

        # Only trade if regime changed or position doesn't match
        self._sync_from_exchange()

        if self.current_position != target:
            old = self.current_position or 'FLAT'
            logger.info(f"REGIME FLIP: {old.upper()} -> {target.upper()}")

            # If we have an existing position, close it first
            if self.current_position is not None:
                logger.info(f"Closing {self.current_position.upper()} before opening {target.upper()}")
                self._execute_position('flat', current_price)
                time.sleep(2)
                self._sync_balance()

            # Open new position
            success = self._execute_position(target, current_price)
            if success:
                logger.info(f"Position opened: {target.upper()} @ ${current_price:,.2f}")
                self._notify(f"REGIME {new_regime}: {target.upper()} BTC @ ${current_price:,.2f} (score={result['total_score']:+.0f})")
            else:
                logger.error(f"Failed to open {target.upper()} position!")
                self._notify(f"REGIME BOT ERROR: Failed to open {target.upper()}")
        else:
            logger.info(f"Position unchanged: {self.current_position.upper()} (regime={new_regime})")

        self.current_regime = new_regime
        self._sync_balance()
        self.save_state(result, current_price)

    def _notify(self, msg):
        """Send push notification."""
        try:
            import urllib.request
            ntfy_topic = 'https://ntfy.sh/TradeBot5234'
            req = urllib.request.Request(ntfy_topic, data=f"[REGIME] {msg}".encode('utf-8'))
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass

    def run_forever(self):
        logger.info("Starting regime bot main loop...")
        logger.info(f"Checking every {CHECK_INTERVAL}s ({CHECK_INTERVAL//3600}h)")

        # Run immediately on start
        self.check_and_trade()

        while True:
            time.sleep(CHECK_INTERVAL)
            try:
                self.check_and_trade()
            except Exception as e:
                logger.error(f"Loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(60)


if __name__ == '__main__':
    try:
        bot = RegimeBot()
        bot.run_forever()
    except Exception as e:
        import traceback
        crash_msg = f"REGIME BOT CRASH: {e}\n{traceback.format_exc()}"
        print(crash_msg)
        try:
            with open('regime_bot_crash.log', 'w') as f:
                f.write(crash_msg)
        except Exception:
            pass
        raise
