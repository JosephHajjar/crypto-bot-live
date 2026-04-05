import ccxt
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
import eth_account
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants

import sys
sys.path.insert(0, '.')
from data.feature_engineer_btc import compute_live_features, get_feature_cols
from ml.model import AttentionLSTMModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SYMBOL = 'BTC/USDT'
COIN = 'BTC' # Hyperliquid specific ticker
TIMEFRAME = '15m'

MODEL_LONG_PATH = 'models/holy_grail.pth'
CONFIG_LONG_PATH = 'models/holy_grail_config.json'
MODEL_SHORT_PATH = 'models_short/holy_grail_short.pth'
CONFIG_SHORT_PATH = 'models_short/holy_grail_short_config.json'

SCALER_PATH = 'data_storage/BTC_USDT_15m_scaler.json'
NTFY_TOPIC = 'https://ntfy.sh/TradeBot5234'

STATE_FILE = 'data_storage/live_state.json'
TRADES_FILE = 'data_storage/live_trades.json'


class LiveHyperliquidTrader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Env
        load_dotenv()
        self.wallet_address = os.environ.get("HYPERLIQUID_WALLET_ADDRESS").strip() if os.environ.get("HYPERLIQUID_WALLET_ADDRESS") else None
        self.secret_key = os.environ.get("HYPERLIQUID_API_SECRET").strip() if os.environ.get("HYPERLIQUID_API_SECRET") else None
        if not self.wallet_address or not self.secret_key:
             raise ValueError("Hyperliquid API Keys not found in .env!")
        
        # Setup Hyperliquid SDK
        self.account = Account.from_key(self.secret_key)
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)
        self.exchange = Exchange(self.account, constants.MAINNET_API_URL, account_address=self.wallet_address)
        
        # Load Long Model
        with open(CONFIG_LONG_PATH, 'r') as f:
            cfg_long = json.load(f)
        self.seq_len_long = cfg_long.get('seq_len', 128)
            
        self.model_long = AttentionLSTMModel(
            input_dim=cfg_long['input_dim'], hidden_dim=cfg_long['hidden_dim'],
            num_layers=cfg_long['num_layers'], output_dim=2, dropout=cfg_long['dropout'], num_heads=cfg_long['num_heads']
        ).to(self.device)
        self.model_long.load_state_dict(torch.load(MODEL_LONG_PATH, map_location=self.device, weights_only=True))
        self.model_long.eval()

        # Load Short Model
        with open(CONFIG_SHORT_PATH, 'r') as f:
            cfg_short = json.load(f)
        self.seq_len_short = cfg_short.get('seq_len', 128)
        
        self.long_tp = cfg_long.get('take_profit', 0.0125)
        self.long_sl = cfg_long.get('stop_loss', 0.0250)
        self.long_max_hold = cfg_long.get('max_hold_bars', 12)
        
        self.short_tp = cfg_short.get('take_profit', 0.0175)
        self.short_sl = cfg_short.get('stop_loss', 0.0375)
        self.short_max_hold = cfg_short.get('max_hold_bars', 12)

        self.model_short = AttentionLSTMModel(
            input_dim=cfg_short['input_dim'], hidden_dim=cfg_short['hidden_dim'],
            num_layers=cfg_short['num_layers'], output_dim=2, dropout=cfg_short['dropout'], num_heads=cfg_short['num_heads']
        ).to(self.device)
        self.model_short.load_state_dict(torch.load(MODEL_SHORT_PATH, map_location=self.device, weights_only=True))
        self.model_short.eval()
        
        # ============================================================
        # DUAL INDEPENDENT POSITION TRACKING
        # Each bot tracks its own virtual position independently.
        # The exchange net position = long_size - short_size.
        # ============================================================
        self.long_active = False
        self.long_entry = 0.0
        self.long_bars = 0
        self.long_size = 0.0  # BTC amount
        
        self.short_active = False
        self.short_entry = 0.0
        self.short_bars = 0
        self.short_size = 0.0  # BTC amount
        
        self.live_balance = 0.0
        
        # Restore state from disk (survives restarts)
        self._load_persisted_state()
        self._sync_balance()
        
        logger.info(f"Loaded Dual-Independent AI Trader on {self.device}. Long: {self.long_active}, Short: {self.short_active}")

    def _load_persisted_state(self):
        """Restore virtual position state from disk."""
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    s = json.load(f)
                self.long_active = s.get("long_active", False)
                self.long_entry = s.get("long_entry", 0.0)
                self.long_bars = s.get("long_bars", 0)
                self.long_size = s.get("long_size", 0.0)
                self.short_active = s.get("short_active", False)
                self.short_entry = s.get("short_entry", 0.0)
                self.short_bars = s.get("short_bars", 0)
                self.short_size = s.get("short_size", 0.0)
                # Backward compat: old single-position format
                if "in_trade" in s and s.get("in_trade") and not self.long_active and not self.short_active:
                    if s.get("trade_type") == "LONG":
                        self.long_active = True
                        self.long_entry = s.get("entry_price", 0.0)
                        self.long_bars = s.get("bars_held", 0)
                        self.long_size = s.get("trade_amount_btc", 0.0)
                    elif s.get("trade_type") == "SHORT":
                        self.short_active = True
                        self.short_entry = s.get("entry_price", 0.0)
                        self.short_bars = s.get("bars_held", 0)
                        self.short_size = s.get("trade_amount_btc", 0.0)
            except Exception:
                pass

    def _sync_balance(self):
        """Fetch current account balance from Hyperliquid."""
        try:
            user_state = self.info.user_state(self.wallet_address)
            margin_summary = user_state.get("marginSummary", {})
            perp_balance = float(margin_summary.get("accountValue", 0.0))
            
            spot_state = self.info.spot_user_state(self.wallet_address)
            spot_usdc = 0.0
            if "balances" in spot_state:
                for bal in spot_state["balances"]:
                    if bal.get("coin") == "USDC":
                        spot_usdc = float(bal.get("total", 0.0))
                        
            self.live_balance = spot_usdc + perp_balance
        except Exception as e:
            logger.error(f"Failed to sync balance: {e}")

    def fetch_recent_data(self):
        import requests
        symbol_fmt = SYMBOL.replace('/', '')
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol_fmt}&interval={TIMEFRAME}&limit=1000"
        res = requests.get(url, timeout=15)
        raw_candles = res.json()
        
        candles = [
            [int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
            for c in raw_candles
        ]
        
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def _calc_trade_size(self, current_price):
        """Calculate trade size: 5x leverage on half the balance (so both bots can trade)."""
        usable_balance = self.live_balance * 0.5  # Each bot gets half the capital
        target_notional = usable_balance * 5.0
        return max(0.0001, round(target_notional / current_price, 5))

    def _record_trade(self, trade_type, entry_price, exit_price, bars_held, reason):
        """Append a completed trade to the trade log."""
        trade_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trade_type": trade_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": 0.0,
            "pnl_usd": 0.0,
            "bars_held": bars_held,
            "reason": reason
        }
        
        if trade_type == "LONG":
            trade_record["pnl_pct"] = round((exit_price - entry_price) / entry_price * 100, 3)
        else:
            trade_record["pnl_pct"] = round((entry_price - exit_price) / entry_price * 100, 3)
        
        trades = []
        if os.path.exists(TRADES_FILE):
            try:
                with open(TRADES_FILE, "r") as f:
                    trades = json.load(f)
            except Exception:
                pass
        trades.append(trade_record)
        with open(TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2)
        
        return trade_record

    def _send_exchange_order(self, is_buy, size, current_price):
        """Send a market order to Hyperliquid. Returns True on success."""
        try:
            px = current_price * (1.05 if is_buy else 0.95)
            res = self.exchange.market_open(COIN, is_buy=is_buy, sz=size, px=px, slippage=0.01)
            if res and res.get('status') == 'ok':
                logger.info(f"Exchange order OK: {'BUY' if is_buy else 'SELL'} {size} {COIN}")
                return True
            else:
                logger.error(f"Exchange order FAILED: {res}")
                return False
        except Exception as e:
            logger.error(f"Exchange order exception: {e}")
            return False

    def _close_exchange_position(self, size):
        """Close a position on the exchange."""
        try:
            res = self.exchange.market_close(COIN, size)
            logger.info(f"Exchange close: {res}")
            return True
        except Exception as e:
            logger.error(f"Exchange close failed: {e}")
            return False

    def manage_long(self, current_close, completed_high, completed_low):
        """Manage the independent long position."""
        if not self.long_active:
            return
            
        self.long_bars += 1
        tp_price = self.long_entry * (1 + self.long_tp)
        sl_price = self.long_entry * (1 - self.long_sl)
        
        exit_price = None
        reason = None
        
        if completed_low <= sl_price:
            exit_price = sl_price
            reason = f"LONG Stop Loss (-{self.long_sl*100}%)"
        elif completed_high >= tp_price:
            exit_price = tp_price
            reason = f"LONG Take Profit (+{self.long_tp*100}%)"
        elif self.long_bars >= self.long_max_hold:
            exit_price = current_close
            reason = "LONG Time Barrier"
        
        if exit_price is not None:
            logger.info(f"CLOSING LONG: {reason} | Entry ${self.long_entry:.2f} -> Exit ${exit_price:.2f}")
            self._close_exchange_position(self.long_size)
            trade = self._record_trade("LONG", self.long_entry, exit_price, self.long_bars, reason)
            self._notify(f"CLOSED LONG: {reason} | PnL: {trade['pnl_pct']:+.2f}%")
            self._sync_balance()
            
            self.long_active = False
            self.long_entry = 0.0
            self.long_bars = 0
            self.long_size = 0.0

    def manage_short(self, current_close, completed_high, completed_low):
        """Manage the independent short position."""
        if not self.short_active:
            return
            
        self.short_bars += 1
        tp_price = self.short_entry * (1 - self.short_tp)
        sl_price = self.short_entry * (1 + self.short_sl)
        
        exit_price = None
        reason = None
        
        if completed_high >= sl_price:
            exit_price = sl_price
            reason = f"SHORT Stop Loss (+{self.short_sl*100}%)"
        elif completed_low <= tp_price:
            exit_price = tp_price
            reason = f"SHORT Take Profit (-{self.short_tp*100}%)"
        elif self.short_bars >= self.short_max_hold:
            exit_price = current_close
            reason = "SHORT Time Barrier"
        
        if exit_price is not None:
            logger.info(f"CLOSING SHORT: {reason} | Entry ${self.short_entry:.2f} -> Exit ${exit_price:.2f}")
            self._close_exchange_position(self.short_size)
            trade = self._record_trade("SHORT", self.short_entry, exit_price, self.short_bars, reason)
            self._notify(f"CLOSED SHORT: {reason} | PnL: {trade['pnl_pct']:+.2f}%")
            self._sync_balance()
            
            self.short_active = False
            self.short_entry = 0.0
            self.short_bars = 0
            self.short_size = 0.0

    def save_state(self, current_close, bull_prob=0.0, bear_prob=0.0):
        """Persist both virtual positions + metrics to disk."""
        long_pnl_pct = 0.0
        short_pnl_pct = 0.0
        if self.long_active and self.long_entry > 0:
            long_pnl_pct = (current_close - self.long_entry) / self.long_entry * 100
        if self.short_active and self.short_entry > 0:
            short_pnl_pct = (self.short_entry - current_close) / self.short_entry * 100

        state = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "paper_balance": self.live_balance,
            "current_price": current_close,
            "bull_prob": round(bull_prob * 100, 6),
            "bear_prob": round(bear_prob * 100, 6),
            # Dual position state
            "long_active": self.long_active,
            "long_entry": self.long_entry,
            "long_bars": self.long_bars,
            "long_size": self.long_size,
            "short_active": self.short_active,
            "short_entry": self.short_entry,
            "short_bars": self.short_bars,
            "short_size": self.short_size,
            # Dashboard backward-compat fields
            "in_trade": self.long_active or self.short_active,
            "trade_type": "LONG" if self.long_active and not self.short_active else ("SHORT" if self.short_active and not self.long_active else ("HEDGE" if self.long_active and self.short_active else None)),
            "entry_price": self.long_entry if self.long_active else (self.short_entry if self.short_active else 0.0),
            "bars_held": max(self.long_bars, self.short_bars),
            "open_pnl_pct": round(long_pnl_pct + short_pnl_pct, 3),
            "open_pnl_usd": round((self.long_size * self.long_entry * long_pnl_pct / 100) + (self.short_size * self.short_entry * short_pnl_pct / 100), 2),
            "take_profit_target": self.long_entry * (1 + self.long_tp) if self.long_active else (self.short_entry * (1 - self.short_tp) if self.short_active else 0.0),
            "stop_loss_target": self.long_entry * (1 - self.long_sl) if self.long_active else (self.short_entry * (1 + self.short_sl) if self.short_active else 0.0),
            "trade_amount_btc": self.long_size + self.short_size,
            "trade_amount_usd": (self.long_size + self.short_size) * current_close,
        }
        
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def step(self):
        try:
            df = self.fetch_recent_data()
            current_close = df['close'].iloc[-1]
            # Use completed candle for TP/SL
            completed_high = df['high'].iloc[-2] if len(df) >= 2 else df['high'].iloc[-1]
            completed_low  = df['low'].iloc[-2] if len(df) >= 2 else df['low'].iloc[-1]
            current_time = df['timestamp'].iloc[-1]
            
            self._sync_balance()
            
            logger.info(f"Live Market Heartbeat: {current_time} | BTC ${current_close:.2f} | Balance: ${self.live_balance:.2f} | Long: {self.long_active} | Short: {self.short_active}")
            
            # 1. Manage existing positions INDEPENDENTLY
            self.manage_long(current_close, completed_high, completed_low)
            self.manage_short(current_close, completed_high, completed_low)
            
            # 2. Compute AI probabilities
            bull_prob = 0.0
            bear_prob = 0.0
            
            live_df = compute_live_features(df, SCALER_PATH)
            max_seq = max(self.seq_len_long, self.seq_len_short)
            if len(live_df) >= max_seq:
                feature_cols = get_feature_cols()
                feat_np = live_df[feature_cols].values.astype(np.float32)
                
                feat_long = feat_np[-self.seq_len_long:]
                feat_short = feat_np[-self.seq_len_short:]
                
                tensor_long = torch.tensor(feat_long).unsqueeze(0).to(self.device)
                tensor_short = torch.tensor(feat_short).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits_long = self.model_long(tensor_long)
                    bull_prob = torch.softmax(logits_long, dim=1)[0][1].item()
                    
                    logits_short = self.model_short(tensor_short)
                    bear_prob = torch.softmax(logits_short, dim=1)[0][1].item()
                    
                logger.info(f"AI Models -> Bullish Edge: {bull_prob*100:.4f}% | Bearish Edge: {bear_prob*100:.4f}%")

            # 3. INDEPENDENT ENTRY LOGIC — no canceling, each bot acts alone
            
            # --- LONG BOT ---
            if not self.long_active and bull_prob >= 0.60:
                logger.info(f"LONG EDGE DETECTED! Prob: {bull_prob*100:.2f}% @ ${current_close:.2f}")
                trade_sz = self._calc_trade_size(current_close)
                if self._send_exchange_order(is_buy=True, size=trade_sz, current_price=current_close):
                    self.long_active = True
                    self.long_entry = current_close
                    self.long_bars = 0
                    self.long_size = trade_sz
                    self._notify(f"LONG ENTRY {trade_sz} BTC @ ${current_close:.2f} | Prob: {bull_prob*100:.1f}%")
                    self._sync_balance()
            
            # --- SHORT BOT ---
            if not self.short_active and bear_prob >= 0.50:
                logger.info(f"SHORT EDGE DETECTED! Prob: {bear_prob*100:.2f}% @ ${current_close:.2f}")
                trade_sz = self._calc_trade_size(current_close)
                if self._send_exchange_order(is_buy=False, size=trade_sz, current_price=current_close):
                    self.short_active = True
                    self.short_entry = current_close
                    self.short_bars = 0
                    self.short_size = trade_sz
                    self._notify(f"SHORT ENTRY {trade_sz} BTC @ ${current_close:.2f} | Prob: {bear_prob*100:.1f}%")
                    self._sync_balance()
            
            if not self.long_active and not self.short_active and bull_prob < 0.60 and bear_prob < 0.50:
                logger.info("No edge detected. Staying flat.")
                    
            self.save_state(current_close, bull_prob=bull_prob, bear_prob=bear_prob)
                    
        except Exception as e:
            logger.error(f"Execution engine error: {e}")
            import traceback; traceback.print_exc()

    def _notify(self, msg):
        """Send push notification via ntfy."""
        try:
            req = urllib.request.Request(NTFY_TOPIC, data=msg.encode('utf-8'))
            urllib.request.urlopen(req, timeout=5)
            logger.info(f"Notification sent: {msg}")
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")

    def run_forever(self):
        logger.info("Initializing DUAL-INDEPENDENT LIVE EXECUTION Daemon...")
        while True:
            now = datetime.utcnow()
            minutes = now.minute
            seconds = now.second
            
            remainder = minutes % 15
            
            if remainder == 0 and seconds < 60:
                logger.info(f"Execution Barrier Reached (Minute: {minutes}, Second: {seconds}). Running step...")
                self.step()
                # Sleep past the current candle boundary to avoid double-execution
                sleep_secs = max(30, 65 - seconds)
                time.sleep(sleep_secs)
            else:
                # Calculate seconds until next 15m boundary + 5s buffer
                secs_to_next = (15 - remainder) * 60 - seconds + 5
                sleep_time = max(1, min(secs_to_next, 60))
                time.sleep(sleep_time)

if __name__ == "__main__":
    import sys
    trader = LiveHyperliquidTrader()
    # Run once immediately on startup
    trader.step()
    
    # If standard run, begin infinite loop
    if "--cron" not in sys.argv:
        trader.run_forever()
    else:
        logger.info("Cron cycle complete. Exiting.")
