"""
ALT-ONLY Live Trading Bot — Pure Threshold Strategy on Hyperliquid.
No PROP commander, no model switcher.

Models:
  - Long Model: Trial 107 (holy_grail.pth) — enters when bull_prob >= 55%
  - Short Model: Trial 270 (holy_grail_short.pth) — enters when bear_prob >= 50%

Exits:
  - Long: 1.9% TP, 1.6% SL, 20 bar max hold (walk-forward optimized, 4/5 overfit tests passed)
  - Short: 1.75% TP, 3.75% SL, 12 bar max hold
  - Catastrophe: 7.5% hard stop on all trades
  - Dynamic hold extension: if edge is still heavy at time barrier, keep holding
"""
import pandas as pd
import numpy as np
import time
import os
import torch
import json
import logging
import urllib.request
from datetime import datetime
from dotenv import load_dotenv

from eth_account import Account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

import sys
sys.path.insert(0, '.')
from data.feature_engineer_btc import compute_live_features, get_feature_cols
from ml.model import AttentionLSTMModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - ALT_BOT - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('alt_bot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ─── CONFIGURATION ───
SYMBOL = 'BTC/USDT'
COIN = 'BTC'
TIMEFRAME = '15m'
LEVERAGE = 15.0
CATASTROPHE_CAP = 0.075  # 7.5% hard stop

MODEL_LONG_PATH = 'models/holy_grail.pth'
CONFIG_LONG_PATH = 'models/holy_grail_config.json'
MODEL_SHORT_PATH = 'models_short/holy_grail_short.pth'
CONFIG_SHORT_PATH = 'models_short/holy_grail_short_config.json'
SCALER_PATH = 'models/BTC_USDT_15m_scaler.json'

NTFY_TOPIC = 'https://ntfy.sh/TradeBot5234'
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(_SCRIPT_DIR, 'data_storage', 'live_state_alt.json')
TRADES_FILE = os.path.join(_SCRIPT_DIR, 'data_storage', 'live_trades_alt.json')


class AltOnlyTrader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load API keys
        load_dotenv()
        self.wallet_address = os.environ.get("HYPERLIQUID_WALLET_ADDRESS", "").strip()
        self.secret_key = os.environ.get("HYPERLIQUID_API_SECRET", "").strip()
        if not self.wallet_address or not self.secret_key:
            raise ValueError("Hyperliquid API Keys not found in .env!")

        # Setup Hyperliquid SDK
        self.account = Account.from_key(self.secret_key)
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        self.exchange = Exchange(self.account, constants.MAINNET_API_URL, account_address=self.wallet_address)

        # Load Long Model (Trial 107 — walk-forward optimized)
        with open(CONFIG_LONG_PATH, 'r') as f:
            cfg_long = json.load(f)
        self.seq_len_long = cfg_long.get('seq_len', 128)
        self.long_tp = cfg_long.get('take_profit', 0.019)
        self.long_sl = cfg_long.get('stop_loss', 0.016)
        self.long_max_hold = cfg_long.get('max_hold_bars', 20)
        self.long_threshold = cfg_long.get('entry_threshold', 0.55)

        self.model_long = AttentionLSTMModel(
            input_dim=cfg_long['input_dim'], hidden_dim=cfg_long['hidden_dim'],
            num_layers=cfg_long['num_layers'], output_dim=2,
            dropout=cfg_long['dropout'], num_heads=cfg_long['num_heads']
        ).to(self.device)
        self.model_long.load_state_dict(torch.load(MODEL_LONG_PATH, map_location=self.device, weights_only=True))
        self.model_long.eval()

        # Load Short Model
        with open(CONFIG_SHORT_PATH, 'r') as f:
            cfg_short = json.load(f)
        self.seq_len_short = cfg_short.get('seq_len', 128)
        self.short_tp = 0.0175
        self.short_sl = 0.0375
        self.short_max_hold = 12

        self.model_short = AttentionLSTMModel(
            input_dim=cfg_short['input_dim'], hidden_dim=cfg_short['hidden_dim'],
            num_layers=cfg_short['num_layers'], output_dim=2,
            dropout=cfg_short['dropout'], num_heads=cfg_short['num_heads']
        ).to(self.device)
        self.model_short.load_state_dict(torch.load(MODEL_SHORT_PATH, map_location=self.device, weights_only=True))
        self.model_short.eval()

        self.position = None  # 'long' or 'short'
        self.entry_price = 0.0
        self.bars_held = 0
        self.active_tp = 0.0
        self.active_sl = 0.0
        self.trade_size_in_btc = 0.0
        self.live_balance = 0.0
        self.last_bull_prob = 0.0
        self.last_bear_prob = 0.0
        self.last_error = None

        # Restore state from disk
        self._load_persisted_state()
        self._sync_from_exchange()
        self._ensure_perps_funded()
        self._sync_balance()

        logger.info(f"ALT-ONLY Bot initialized on {self.device}")
        logger.info(f"  Long Model: TP={self.long_tp*100}%, SL={self.long_sl*100}%, MaxHold={self.long_max_hold}")
        logger.info(f"  Short Model: TP={self.short_tp*100}%, SL={self.short_sl*100}%, MaxHold={self.short_max_hold}")
        logger.info(f"  Leverage: {LEVERAGE}x | Catastrophe Cap: {CATASTROPHE_CAP*100}%")
        logger.info(f"  Current Position: {self.position or 'FLAT'} | Balance: ${self.live_balance:.2f}")

    # ─── STATE PERSISTENCE ───

    def _load_persisted_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    s = json.load(f)
                trade_type = s.get("trade_type")
                if trade_type == "LONG":
                    self.position = "long"
                elif trade_type == "SHORT":
                    self.position = "short"
                self.entry_price = s.get("entry_price", 0.0)
                self.bars_held = s.get("bars_held", 0)
                self.active_tp = s.get("take_profit_target", 0.0)
                self.active_sl = s.get("stop_loss_target", 0.0)
                self.trade_size_in_btc = s.get("trade_amount_btc", 0.0)
                logger.info(f"STATE LOADED: pos={self.position}, entry={self.entry_price}, bars={self.bars_held}, tp={self.active_tp}, sl={self.active_sl}")
            except Exception as e:
                logger.error(f"Failed to load persisted state: {e}")
        else:
            logger.info("No persisted state file found.")

    def save_state(self, current_close, bull_prob=0.0, bear_prob=0.0):
        open_pnl_pct = 0.0
        open_pnl_usd = 0.0
        if self.position == 'long' and self.entry_price > 0:
            open_pnl_pct = (current_close - self.entry_price) / self.entry_price * 100
            open_pnl_usd = self.trade_size_in_btc * (current_close - self.entry_price)
        elif self.position == 'short' and self.entry_price > 0:
            open_pnl_pct = (self.entry_price - current_close) / self.entry_price * 100
            open_pnl_usd = self.trade_size_in_btc * (self.entry_price - current_close)

        state = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "paper_balance": round(self.live_balance, 2),
            "current_price": current_close,
            "bull_prob": round(bull_prob * 100, 6),
            "bear_prob": round(bear_prob * 100, 6),
            "master_control": "ALT",
            "in_trade": self.position is not None,
            "trade_type": self.position.upper() if self.position else None,
            "entry_price": self.entry_price,
            "bars_held": self.bars_held,
            "open_pnl_pct": round(open_pnl_pct, 4),
            "open_pnl_usd": round(open_pnl_usd, 2),
            "take_profit_target": self.active_tp,
            "stop_loss_target": self.active_sl,
            "trade_amount_btc": self.trade_size_in_btc,
            "trade_amount_usd": round(self.trade_size_in_btc * current_close, 2),
            "last_error": self.last_error,
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    # ─── EXCHANGE INTERACTION ───

    def _sync_balance(self):
        """Fetch total balance from Hyperliquid."""
        try:
            user_state = self.info.user_state(self.wallet_address)
            margin_summary = user_state.get("marginSummary", {})
            # In Hyperliquid's unified account, accountValue IS the full portfolio
            # (margin + unrealized PnL). Do NOT add spot USDC — it's the same pool.
            self.live_balance = float(margin_summary.get("accountValue", 0.0))
            logger.info(f"Balance: ${self.live_balance:.2f}")
        except Exception as e:
            logger.error(f"Failed to sync balance: {e}")

    def _ensure_perps_funded(self):
        """Auto-transfer any spot USDC to perps margin so we can trade."""
        try:
            spot_state = self.info.spot_user_state(self.wallet_address)
            spot_usdc = 0.0
            for bal in spot_state.get("balances", []):
                if bal.get("coin") == "USDC":
                    spot_usdc = float(bal.get("total", 0.0))
                    break

            if spot_usdc > 0.10:  # Only transfer if more than 10 cents
                logger.info(f"Transferring ${spot_usdc:.2f} USDC from Spot to Perps...")
                res = self.exchange.usd_class_transfer(spot_usdc, to_perp=True)
                if res and res.get('status') == 'ok':
                    logger.info(f"Transfer OK: ${spot_usdc:.2f} moved to perps margin")
                    self._sync_balance()  # Refresh balance after transfer
                else:
                    logger.warning(f"Transfer response: {res}")
        except Exception as e:
            logger.warning(f"Spot->Perps transfer failed (may already be unified): {e}")

    def _sync_from_exchange(self):
        """On startup, validate bot state against real Hyperliquid position."""
        try:
            user_state = self.info.user_state(self.wallet_address)
            exchange_pos = None
            for pos in user_state.get("assetPositions", []):
                if pos['position']['coin'] == COIN:
                    exchange_pos = pos['position']
                    break

            exchange_size = float(exchange_pos['szi']) if exchange_pos else 0.0
            exchange_entry = float(exchange_pos['entryPx']) if exchange_pos else 0.0
            exchange_direction = 'long' if exchange_size > 0 else ('short' if exchange_size < 0 else None)
            exchange_abs_size = abs(exchange_size)

            # Bot thinks it has a position, but exchange is flat
            if self.position is not None and exchange_abs_size < 0.00001:
                logger.warning(f"SYNC: Bot has {self.position.upper()} but exchange is FLAT. Resetting.")
                self.position = None
                self.entry_price = 0.0
                self.bars_held = 0
                self.active_tp = 0.0
                self.active_sl = 0.0
                self.trade_size_in_btc = 0.0
                return

            if exchange_abs_size >= 0.00001:
                # Bot is flat but exchange has position -> adopt
                if self.position is None:
                    # Preserve bars_held from state file if it was loaded
                    loaded_bars = self.bars_held
                    logger.info(f"SYNC: Adopting exchange {exchange_direction.upper()} | Entry ${exchange_entry:.2f} | Size {exchange_abs_size} BTC | Preserving bars_held={loaded_bars}")
                    self.position = exchange_direction
                    self.entry_price = exchange_entry
                    self.bars_held = max(loaded_bars, 1)  # Keep loaded value, minimum 1
                    self.trade_size_in_btc = exchange_abs_size
                    if exchange_direction == 'long':
                        self.active_tp = exchange_entry * (1 + self.long_tp)
                        self.active_sl = exchange_entry * (1 - self.long_sl)
                    else:
                        self.active_tp = exchange_entry * (1 - self.short_tp)
                        self.active_sl = exchange_entry * (1 + self.short_sl)
                    return

                # Correct size if needed
                if abs(self.trade_size_in_btc - exchange_abs_size) > 0.00001:
                    logger.warning(f"SYNC: Size mismatch! Bot={self.trade_size_in_btc} vs Exchange={exchange_abs_size}. Correcting.")
                    self.trade_size_in_btc = exchange_abs_size

        except Exception as e:
            logger.error(f"Failed to sync from exchange: {e}")

    def _sync_exchange_position(self, current_price, target_position, size_in_btc):
        target_size = size_in_btc if target_position == 'long' else (-size_in_btc if target_position == 'short' else 0.0)

        try:
            user_state = self.info.user_state(self.wallet_address)
            current_pos = 0.0
            for pos in user_state.get("assetPositions", []):
                if pos['position']['coin'] == COIN:
                    current_pos = float(pos['position']['szi'])
                    break

            diff = target_size - current_pos
            if abs(diff) < 0.00001:
                self.last_error = None
                return True

            is_buy = bool(diff > 0)
            size_to_trade = float(abs(diff))

            logger.info(f"ORDER: {target_position.upper()} sz={size_to_trade:.6f} BTC is_buy={is_buy}")
            res = self.exchange.market_open(COIN, is_buy=is_buy, sz=size_to_trade, slippage=0.01)
            if res and res.get('status') == 'ok':
                logger.info(f"ORDER OK: {'BUY' if is_buy else 'SELL'} {size_to_trade} {COIN}")
                self.last_error = None
                return True
            else:
                self.last_error = f"Order failed: {res}"
                logger.error(self.last_error)
                return False
        except Exception as e:
            self.last_error = f"Order exception: {e}"
            logger.error(self.last_error)
            return False

    def _calc_trade_size(self, current_price):
        if self.live_balance < 0.50:
            logger.warning(f"Balance too low: ${self.live_balance:.2f}")
            return 0
        target_notional = self.live_balance * LEVERAGE
        return max(0.0001, round(target_notional / current_price, 5))

    # ─── DATA ───

    def fetch_recent_data(self):
        import requests
        symbol_fmt = SYMBOL.replace('/', '')
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol_fmt}&interval={TIMEFRAME}&limit=1000"
        res = requests.get(url, timeout=15)
        raw_candles = res.json()
        candles = [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])] for c in raw_candles]
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    # ─── TRADE LOGGING ───

    def _record_trade(self, trade_type, entry_price, exit_price, bars_held, reason):
        trade = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trade_type": trade_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "bars_held": bars_held,
            "reason": reason,
            "return_pct": 0.0,
            "pnl_usd": 0.0,
        }
        if trade_type == "LONG":
            trade["return_pct"] = round((exit_price - entry_price) / entry_price * 100, 3)
            trade["pnl_usd"] = round(self.trade_size_in_btc * (exit_price - entry_price), 2)
        else:
            trade["return_pct"] = round((entry_price - exit_price) / entry_price * 100, 3)
            trade["pnl_usd"] = round(self.trade_size_in_btc * (entry_price - exit_price), 2)

        trades = []
        if os.path.exists(TRADES_FILE):
            try:
                with open(TRADES_FILE, "r") as f:
                    trades = json.load(f)
            except Exception:
                pass
        trades.append(trade)
        with open(TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2)
        return trade

    def _notify(self, msg):
        try:
            req = urllib.request.Request(NTFY_TOPIC, data=msg.encode('utf-8'))
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass

    # ─── FAST TP/SL POLLING ───

    def check_tp_sl(self):
        """Polls live price every 5s to catch TP/SL between candle boundaries."""
        if self.position is None:
            return

        try:
            import requests
            res = requests.get(f"https://data-api.binance.vision/api/v3/ticker/price?symbol=BTCUSDT", timeout=5)
            live_price = float(res.json()['price'])
        except Exception:
            return

        try:
            exit_price = None
            reason = None

            # TP/SL check (only if targets are set)
            if self.active_tp > 0 and self.active_sl > 0:
                if self.position == 'long':
                    if live_price <= self.active_sl:
                        exit_price = live_price
                        reason = f"LONG Stop Loss (-{self.long_sl*100}%)"
                    elif live_price >= self.active_tp:
                        exit_price = live_price
                        reason = f"LONG Take Profit (+{self.long_tp*100}%)"
                elif self.position == 'short':
                    if live_price >= self.active_sl:
                        exit_price = live_price
                        reason = f"SHORT Stop Loss (+{self.short_sl*100}%)"
                    elif live_price <= self.active_tp:
                        exit_price = live_price
                        reason = f"SHORT Take Profit (-{self.short_tp*100}%)"

            # CATASTROPHE STOP-LOSS (always active)
            if exit_price is None:
                if self.position == 'long':
                    cat_sl = self.entry_price * (1 - CATASTROPHE_CAP)
                    if live_price <= cat_sl:
                        exit_price = live_price
                        reason = f"CATASTROPHE STOP (-{CATASTROPHE_CAP*100}%)"
                elif self.position == 'short':
                    cat_sl = self.entry_price * (1 + CATASTROPHE_CAP)
                    if live_price >= cat_sl:
                        exit_price = live_price
                        reason = f"CATASTROPHE STOP (-{CATASTROPHE_CAP*100}%)"

            if exit_price is not None:
                logger.info(f"CLOSING {self.position.upper()}: {reason} | ${self.entry_price:.2f} -> ${exit_price:.2f}")
                trade = self._record_trade(self.position.upper(), self.entry_price, exit_price, self.bars_held, reason)
                self._notify(f"CLOSED {self.position.upper()}: {reason} | PnL: {trade['return_pct']:+.2f}% | ${trade['pnl_usd']:+.2f}")
                self._sync_exchange_position(live_price, 'flat', 0.0)
                self._reset_position()
                return

            # TIME BARRIER CHECK (runs every 5s so it can't be missed)
            max_hold = self.long_max_hold if self.position == 'long' else self.short_max_hold
            if self.bars_held >= max_hold:
                is_heavy = (self.position == 'long' and self.last_bull_prob >= self.long_threshold) or \
                           (self.position == 'short' and self.last_bear_prob >= 0.50)

                logger.info(f"TIME BARRIER CHECK: bars={self.bars_held}/{max_hold}, "
                            f"bull={self.last_bull_prob:.4f}, bear={self.last_bear_prob:.4f}, heavy={is_heavy}")

                if not is_heavy:
                    logger.info(f"TIME BARRIER CLOSING {self.position.upper()} @ ${live_price:.2f} after {self.bars_held} bars")
                    trade = self._record_trade(self.position.upper(), self.entry_price, live_price, self.bars_held, "Time Barrier")
                    self._notify(f"CLOSED {self.position.upper()}: Time Barrier ({self.bars_held} bars) | PnL: {trade['return_pct']:+.2f}% | ${trade['pnl_usd']:+.2f}")
                    self._sync_exchange_position(live_price, 'flat', 0.0)
                    self._reset_position()
                    return

                # HARD FAILSAFE: if held 5x max_hold bars, force close no matter what
                if self.bars_held >= max_hold * 5:
                    logger.warning(f"HARD FAILSAFE: Force closing {self.position.upper()} after {self.bars_held} bars")
                    trade = self._record_trade(self.position.upper(), self.entry_price, live_price, self.bars_held, "Hard Failsafe")
                    self._notify(f"FORCE CLOSED {self.position.upper()}: {self.bars_held} bars | PnL: {trade['return_pct']:+.2f}% | ${trade['pnl_usd']:+.2f}")
                    self._sync_exchange_position(live_price, 'flat', 0.0)
                    self._reset_position()

        except Exception as e:
            logger.error(f"check_tp_sl CRASH: {e}")
            import traceback
            traceback.print_exc()
    def step(self, execute_trades=True):
        try:
            df = self.fetch_recent_data()
            current_close = df['close'].iloc[-1]
            current_time = df['timestamp'].iloc[-1]

            if execute_trades:
                self._sync_balance()
                logger.info(f"=== 15m Candle: {current_time} | BTC ${current_close:.2f} | Balance: ${self.live_balance:.2f} | Pos: {self.position or 'FLAT'} ===")

                # Verify exchange position matches bot state
                if self.position is not None:
                    try:
                        user_st = self.info.user_state(self.wallet_address)
                        current_pos_size = 0.0
                        for pos in user_st.get("assetPositions", []):
                            if pos['position']['coin'] == COIN:
                                current_pos_size = float(pos['position']['szi'])
                                break
                        if abs(current_pos_size) < 0.00001:
                            logger.warning("EXTERNAL CLOSE DETECTED! Resetting state.")
                            trade = self._record_trade(self.position.upper(), self.entry_price, current_close, self.bars_held, "Closed Externally")
                            self._notify(f"EXTERNAL CLOSE | PnL: {trade['return_pct']:+.2f}%")
                            self._reset_position()
                    except Exception as e:
                        logger.error(f"Position verification failed: {e}")

            # Compute AI probabilities
            bull_prob = 0.0
            bear_prob = 0.0

            live_df = compute_live_features(df, SCALER_PATH)
            if len(live_df) > 1:
                live_df = live_df.iloc[:-1]  # Drop unclosed candle (or not, if we want fully live tracking, but ML expects closed bar structure. Let's keep dropping to evaluate the latest full features safely)

            max_seq = max(self.seq_len_long, self.seq_len_short)
            if len(live_df) >= max_seq:
                feature_cols = get_feature_cols()
                feat_np = live_df[feature_cols].values.astype(np.float32)
                feat_np = np.nan_to_num(feat_np, nan=0.0, posinf=0.0, neginf=0.0)

                tensor_long = torch.tensor(feat_np[-self.seq_len_long:]).unsqueeze(0).to(self.device)
                tensor_short = torch.tensor(feat_np[-self.seq_len_short:]).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    bull_prob = torch.softmax(self.model_long(tensor_long), dim=1)[0][1].item()
                    bear_prob = torch.softmax(self.model_short(tensor_short), dim=1)[0][1].item()
                    # Safety: clamp NaN outputs to 0 (refuse to trade on garbage)
                    if np.isnan(bull_prob): bull_prob = 0.0
                    if np.isnan(bear_prob): bear_prob = 0.0

                # Cache latest probs so check_tp_sl() can use them for time barrier
                self.last_bull_prob = bull_prob
                self.last_bear_prob = bear_prob

                if execute_trades:
                    logger.info(f"AI -> Bull: {bull_prob*100:.2f}% | Bear: {bear_prob*100:.2f}%")

            # Always save state to update dashboard pseudo-live
            self.save_state(current_close, bull_prob, bear_prob)

            if not execute_trades:
                return

            # ─── ALT TIME BARRIER ───
            if self.position is not None:
                max_hold = self.long_max_hold if self.position == 'long' else self.short_max_hold
                if self.bars_held >= max_hold:
                    # Dynamic hold extension if edge is still heavy
                    is_heavy = (self.position == 'long' and bull_prob >= self.long_threshold) or \
                               (self.position == 'short' and bear_prob >= 0.50)

                    if is_heavy:
                        logger.info(f"Time barrier hit but edge still HEAVY ({bull_prob*100:.1f}%/{bear_prob*100:.1f}%). Extending hold.")
                    else:
                        logger.info(f"CLOSING {self.position.upper()}: Time Barrier ({self.bars_held} bars)")
                        trade = self._record_trade(self.position.upper(), self.entry_price, current_close, self.bars_held, "Time Barrier")
                        self._notify(f"CLOSED {self.position.upper()}: Time Barrier | PnL: {trade['return_pct']:+.2f}% | ${trade['pnl_usd']:+.2f}")
                        self._sync_exchange_position(current_close, 'flat', 0.0)
                        self._reset_position()
                        self._sync_balance()
                        self.save_state(current_close, bull_prob, bear_prob)
                        return

            # ─── ENTRY / FLIP LOGIC ───
            wants_long = bull_prob >= self.long_threshold
            wants_short = bear_prob >= 0.50

            if wants_long:
                if self.position != 'long':
                    # Close existing position if needed
                    if self.position is not None:
                        trade = self._record_trade(self.position.upper(), self.entry_price, current_close, self.bars_held, "Override Flip to LONG")
                        self._notify(f"FLIPPED {self.position.upper()} -> LONG | PnL: {trade['return_pct']:+.2f}%")

                    self._ensure_perps_funded()
                    self._sync_balance()
                    trade_sz = self._calc_trade_size(current_close)
                    if trade_sz > 0 and self._sync_exchange_position(current_close, 'long', trade_sz):
                        self.position = 'long'
                        self.entry_price = current_close
                        self.active_tp = current_close * (1 + self.long_tp)
                        self.active_sl = current_close * (1 - self.long_sl)
                        self.bars_held = 0
                        self.trade_size_in_btc = trade_sz
                        logger.info(f"ENTERED LONG @ ${current_close:.2f} | TP: ${self.active_tp:.2f} | SL: ${self.active_sl:.2f} | Size: {trade_sz} BTC")
                        self._notify(f"LONG ENTRY @ ${current_close:.2f} | TP: ${self.active_tp:.2f} SL: ${self.active_sl:.2f}")
                        self._sync_balance()
                else:
                    self.bars_held += 1

            elif wants_short and not wants_long:
                if self.position != 'short':
                    if self.position is not None:
                        trade = self._record_trade(self.position.upper(), self.entry_price, current_close, self.bars_held, "Override Flip to SHORT")
                        self._notify(f"FLIPPED {self.position.upper()} -> SHORT | PnL: {trade['return_pct']:+.2f}%")

                    self._ensure_perps_funded()
                    self._sync_balance()
                    trade_sz = self._calc_trade_size(current_close)
                    if trade_sz > 0 and self._sync_exchange_position(current_close, 'short', trade_sz):
                        self.position = 'short'
                        self.entry_price = current_close
                        self.active_tp = current_close * (1 - self.short_tp)
                        self.active_sl = current_close * (1 + self.short_sl)
                        self.bars_held = 0
                        self.trade_size_in_btc = trade_sz
                        logger.info(f"ENTERED SHORT @ ${current_close:.2f} | TP: ${self.active_tp:.2f} | SL: ${self.active_sl:.2f} | Size: {trade_sz} BTC")
                        self._notify(f"SHORT ENTRY @ ${current_close:.2f} | TP: ${self.active_tp:.2f} SL: ${self.active_sl:.2f}")
                        self._sync_balance()
                else:
                    self.bars_held += 1
            else:
                # No signal — just hold or stay flat
                if self.position is not None:
                    self.bars_held += 1
                else:
                    logger.info("No edge detected. Staying flat.")

            self.save_state(current_close, bull_prob, bear_prob)

        except Exception as e:
            logger.error(f"Step error: {e}")
            import traceback
            traceback.print_exc()

    def _reset_position(self):
        self.position = None
        self.entry_price = 0.0
        self.bars_held = 0
        self.active_tp = 0.0
        self.active_sl = 0.0
        self.trade_size_in_btc = 0.0

    # ─── MAIN LOOP ───

    def run_forever(self):
        logger.info("=" * 60)
        logger.info("  ALT-ONLY LIVE BOT STARTING")
        logger.info("  Polling every 5s for TP/SL, AI inference on 15m boundaries")
        logger.info("=" * 60)

        last_infer_minute = -1
        last_soft_infer = 0

        while True:
            # Fast TP/SL polling
            self.check_tp_sl()

            # AI inference on 15m boundaries
            now = datetime.utcnow()
            remainder = now.minute % 15

            if remainder == 0:
                if last_infer_minute != now.minute:
                    logger.info(f"15m boundary reached. Running FULL inference...")
                    self.step(execute_trades=True)
                    last_infer_minute = now.minute
                    last_soft_infer = now.timestamp()
            else:
                # Reset for next hour cross or next period
                if remainder >= 1:
                    last_infer_minute = -1
                
                # Soft inference purely for the dashboard visual updates (every 7.5 mins)
                if now.timestamp() - last_soft_infer > 450:
                    self.step(execute_trades=False)
                    last_soft_infer = now.timestamp()

            time.sleep(5)


if __name__ == "__main__":
    try:
        trader = AltOnlyTrader()
        # Run one step immediately
        trader.step()
        # Then loop forever
        trader.run_forever()
    except Exception as e:
        import traceback
        crash_msg = f"FATAL CRASH: {e}\n{traceback.format_exc()}"
        print(crash_msg)
        # Write crash info to a file the debug endpoint can read
        try:
            with open('alt_bot_crash.log', 'w') as f:
                f.write(crash_msg)
        except Exception:
            pass
        raise
