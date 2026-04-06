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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - EnsembleBot - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SYMBOL = 'BTC/USDT'
COIN = 'BTC'
TIMEFRAME = '15m'

MODEL_LONG_PATH = 'models/holy_grail.pth'
CONFIG_LONG_PATH = 'models/holy_grail_config.json'
MODEL_SHORT_PATH = 'models_short/holy_grail_short.pth'
CONFIG_SHORT_PATH = 'models_short/holy_grail_short_config.json'

SCALER_PATH = 'data_storage/BTC_USDT_15m_scaler.json'
NTFY_TOPIC = 'https://ntfy.sh/TradeBot5234'

STATE_FILE = 'data_storage/live_state_ensemble.json'
TRADES_FILE = 'data_storage/live_trades_ensemble.json'

class LiveEnsembleTrader:
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
        self.long_tp = cfg_long.get('take_profit', 0.0125)
        self.long_sl = cfg_long.get('stop_loss', 0.0250)
        self.long_max_hold = cfg_long.get('max_hold_bars', 12)
            
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
        # ENSEMBLE STATE TRACKING
        # ============================================================
        self.master_control = 'PROP' # 'PROP' or 'ALT'
        self.position = None # 'long' or 'short'
        self.entry_price = 0.0
        self.bars_held = 0
        self.active_tp = 0.0
        self.active_sl = 0.0
        self.live_balance = 0.0
        self.trade_size_in_btc = 0.0
        self.last_error = None
        
        self.peak_price = 0.0
        self.trailing_armed = False
        
        self._load_persisted_state()
        logger.info(f"Loaded Ensemble AI Trader on {self.device}. Current Commander: {self.master_control}")

    def _load_persisted_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    s = json.load(f)
                
                self.master_control = s.get("master_control", "PROP")
                trade_type = s.get("trade_type")
                if trade_type == "LONG": self.position = "long"
                elif trade_type == "SHORT": self.position = "short"
                
                self.entry_price = s.get("entry_price", 0.0)
                self.bars_held = s.get("bars_held", 0)
                self.active_tp = s.get("take_profit_target", 0.0)
                self.active_sl = s.get("stop_loss_target", 0.0)
                self.trade_size_in_btc = s.get("trade_amount_btc", 0.0)
                
                self.peak_price = s.get("peak_price", 0.0)
                self.trailing_armed = s.get("trailing_armed", False)
            except Exception:
                pass

    def _sync_balance(self):
        try:
            user_state = self.info.user_state(self.wallet_address)
            margin_summary = user_state.get("marginSummary", {})
            perp_balance = float(margin_summary.get("accountValue", 0.0))
            spot_state = self.info.spot_user_state(self.wallet_address)
            spot_usdc = 0.0
            if "balances" in spot_state:
                for bal in spot_state["balances"]:
                    if bal.get("coin") == "USDC": spot_usdc = float(bal.get("total", 0.0))
            self.live_balance = spot_usdc + perp_balance
        except Exception as e:
            logger.error(f"Failed to sync balance: {e}")

    def _calc_trade_size(self, current_price):
        target_notional = self.live_balance * 10.0
        return max(0.0001, round(target_notional / current_price, 5))

    def _sync_exchange_position(self, current_price, target_position, size_in_btc):
        target_size = size_in_btc if target_position == 'long' else (-size_in_btc if target_position == 'short' else 0.0)
            
        try:
            user_state = self.info.user_state(self.wallet_address)
            asset_positions = user_state.get("assetPositions", [])
            current_pos = 0.0
            for pos in asset_positions:
                if pos['position']['coin'] == COIN:
                    current_pos = float(pos['position']['szi'])
                    break
                    
            diff = target_size - current_pos
            if abs(diff) < 0.00001:
                self.last_error = None
                return True
                
            is_buy = bool(diff > 0)
            size_to_trade = float(abs(diff))
            
            logger.info(f"Attempting order: {target_position} sz={size_to_trade:.6f} BTC is_buy={is_buy} balance=${self.live_balance:.2f}")
            res = self.exchange.market_open(COIN, is_buy=is_buy, sz=size_to_trade, slippage=0.01)
            if res and res.get('status') == 'ok':
                logger.info(f"Target Size: {target_size}. Syncing Exchange Size by {'BUY' if is_buy else 'SELL'} {size_to_trade} {COIN}")
                self.last_error = None
                return True
            else:
                err_msg = f"Sync exchange FAILED: {res}"
                logger.error(err_msg)
                self.last_error = err_msg
                return False
        except Exception as e:
            err_msg = f"Sync exchange exception: {e}"
            logger.error(err_msg)
            self.last_error = err_msg
            return False

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

    def _record_trade(self, trade_type, entry_price, exit_price, bars_held, reason):
        trade_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "trade_type": trade_type,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "return_pct": 0.0,
            "pnl_usd": 0.0,
            "bars_held": bars_held,
            "reason": reason
        }
        
        if trade_type == "LONG":
            trade_record["return_pct"] = round((exit_price - entry_price) / entry_price * 100, 3)
            trade_record["pnl_usd"] = round(self.trade_size_in_btc * (exit_price - entry_price), 2)
        else:
            trade_record["return_pct"] = round((entry_price - exit_price) / entry_price * 100, 3)
            trade_record["pnl_usd"] = round(self.trade_size_in_btc * (entry_price - exit_price), 2)
                
        trades = []
        if os.path.exists(TRADES_FILE):
            try:
                with open(TRADES_FILE, "r") as f:
                    trades = json.load(f)
            except Exception: pass
        trades.append(trade_record)
        with open(TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2)
        return trade_record

    def save_state(self, current_close, bull_prob=0.0, bear_prob=0.0):
        open_pnl_pct = 0.0
        if self.position == 'long' and self.entry_price > 0:
            open_pnl_pct = (current_close - self.entry_price) / self.entry_price * 100
        elif self.position == 'short' and self.entry_price > 0:
            open_pnl_pct = (self.entry_price - current_close) / self.entry_price * 100

        state = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "paper_balance": round(self.live_balance, 2), # Ensemble is native live
            "current_price": current_close,
            "bull_prob": round(bull_prob * 100, 6),
            "bear_prob": round(bear_prob * 100, 6),
            "master_control": self.master_control,
            "in_trade": self.position is not None,
            "trade_type": "LONG" if self.position == "long" else ("SHORT" if self.position == "short" else None),
            "entry_price": self.entry_price,
            "bars_held": self.bars_held,
            "open_pnl_pct": round(open_pnl_pct, 4),
            "open_pnl_usd": round(self.trade_size_in_btc * ((current_close - self.entry_price) if self.position == 'long' else (self.entry_price - current_close)), 2) if self.position else 0.0,
            "take_profit_target": self.active_tp, 
            "stop_loss_target": self.active_sl,
            "trade_amount_btc": self.trade_size_in_btc,
            "trade_amount_usd": round(self.trade_size_in_btc * current_close, 2),
            "last_error": self.last_error,
            "peak_price": self.peak_price,
            "trailing_armed": self.trailing_armed
        }
        with open(STATE_FILE, "w") as f: json.dump(state, f, indent=2)

    def check_tp_sl(self):
        if self.position is None:
            return
            
        try:
            import requests
            symbol_fmt = SYMBOL.replace('/', '')
            res = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol_fmt}", timeout=5)
            live_price = float(res.json()['price'])
        except Exception as e:
            logger.error(f"Failed to fetch fast price: {e}")
            return
            
        exit_price = None
        reason = None
        
        # Check Manual Liquidity Override First
        if os.path.exists('data_storage/manual_override.json'):
            try:
                with open('data_storage/manual_override.json', 'r') as f:
                    override = json.load(f)
                    target = float(override.get('manual_target', 0))
                    # Ensure override is not stale (e.g. > 12 hours old)
                    if target > 0 and (time.time() - override.get('timestamp', 0) < 43200):
                        if self.position == 'long' and live_price >= target:
                            exit_price = live_price
                            reason = f"Human Liquidity Override (Target: {target})"
                        elif self.position == 'short' and live_price <= target:
                            exit_price = live_price
                            reason = f"Human Liquidity Override (Target: {target})"
            except Exception: pass
        
        # Standard ALT TP/SL Logic (Only if no manual override triggered)
        if exit_price is None and self.master_control == 'ALT':
            if self.position == 'long':
                if live_price <= self.active_sl:
                    exit_price = live_price
                    reason = f"ALT LONG Stop Loss (-{self.long_sl*100}%)"
                elif live_price >= self.active_tp:
                    exit_price = live_price
                    reason = f"ALT LONG Take Profit (+{self.long_tp*100}%)"
            elif self.position == 'short':
                 if live_price >= self.active_sl:
                    exit_price = live_price
                    reason = f"ALT SHORT Stop Loss (+{self.short_sl*100}%)"
                 elif live_price <= self.active_tp:
                    exit_price = live_price
                    reason = f"ALT SHORT Take Profit (-{self.short_tp*100}%)"
                
        if exit_price is not None:
             logger.info(f"CLOSING {self.position.upper()} (FAST POLL): {reason} | Entry ${self.entry_price:.2f} -> Exit ${exit_price:.2f}")
             trade = self._record_trade(self.position.upper(), self.entry_price, exit_price, self.bars_held, reason)
             self._notify(f"CLOSED {self.position.upper()}: {reason} | PnL: {trade['return_pct']:+.2f}% | ${trade['pnl_usd']:+.2f}")
             
             # Relinquish command back to proportional, position reset so PROP can safely evaluate edge on next tick
             self.master_control = 'PROP'
             self.position = None
             self.entry_price = 0.0
             self.bars_held = 0
             self.active_tp = 0.0
             self.active_sl = 0.0
             self.trade_size_in_btc = 0.0
             
             self._sync_exchange_position(live_price, 'flat', 0.0)
             self._sync_balance()

    def step(self):
        try:
            df = self.fetch_recent_data()
            current_close = df['close'].iloc[-1]
            current_time = df['timestamp'].iloc[-1]
            
            self._sync_balance()
            
            # --- Dynamic Regime Volatility Analysis ---
            closes = df['close'].values
            returns = (closes[1:] - closes[:-1]) / closes[:-1]
            volatility_monthly = np.std(returns) * np.sqrt(2880) * 100
            
            if self.position is None:
                if volatility_monthly >= 10.50:
                    self.master_control = 'PROP'
                else:
                    self.master_control = 'ALT'
            # ----------------------------------------
            
            logger.info(f"15m Cycle: {current_time} | BTC ${current_close:.2f} | Volatility: {volatility_monthly:.2f}% | Regime: {self.master_control}")
            
            bull_prob = 0.0
            bear_prob = 0.0
            live_df = compute_live_features(df, SCALER_PATH)
            
            if len(live_df) > 1:
                live_df = live_df.iloc[:-1]
                
            max_seq = max(self.seq_len_long, self.seq_len_short)
            if len(live_df) >= max_seq:
                feature_cols = get_feature_cols()
                feat_np = live_df[feature_cols].values.astype(np.float32)
                tensor_long = torch.tensor(feat_np[-self.seq_len_long:]).unsqueeze(0).to(self.device)
                tensor_short = torch.tensor(feat_np[-self.seq_len_short:]).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    logits_long = self.model_long(tensor_long)
                    bull_prob = torch.softmax(logits_long, dim=1)[0][1].item()
                    logits_short = self.model_short(tensor_short)
                    bear_prob = torch.softmax(logits_short, dim=1)[0][1].item()
                    
                logger.info(f"AI Models -> Bullish Edge: {bull_prob*100:.4f}% | Bearish Edge: {bear_prob*100:.4f}%")

            # ALT Time Barrier Check
            if self.master_control == 'ALT' and self.position is not None:
                 max_hold = self.long_max_hold if self.position == 'long' else self.short_max_hold
                 if self.bars_held >= max_hold:
                      logger.info(f"CLOSING {self.position.upper()}: ALT Time Barrier")
                      trade = self._record_trade(self.position.upper(), self.entry_price, current_close, self.bars_held, "ALT Time Barrier")
                      self._notify(f"CLOSED {self.position.upper()}: Time Barrier | PnL: {trade['return_pct']:+.2f}% | ${trade['pnl_usd']:+.2f}")
                      self.master_control = 'PROP'
                      self.position = None
                      self.entry_price = 0.0
                      self.bars_held = 0
                      self.active_tp = 0.0
                      self.active_sl = 0.0
                      self.trade_size_in_btc = 0.0
                      self._sync_exchange_position(current_close, 'flat', 0.0)

            # Evaluate Hierarchy
            alt_wants_long = bull_prob >= 0.60
            alt_wants_short = bear_prob >= 0.50
            
            # --- ALT COMMAND OVERRIDE ---
            if alt_wants_long:
                 logger.info(f"ALT BOT ASSUMING COMMAND -> Wants LONG.")
                 self.master_control = 'ALT'
                 if self.position != 'long':
                      trade_sz = self._calc_trade_size(current_close)
                      if self._sync_exchange_position(current_close, 'long', trade_sz):
                           self.position = 'long'
                           self.entry_price = current_close
                           self.active_tp = current_close * (1 + self.long_tp)
                           self.active_sl = current_close * (1 - self.long_sl)
                           self.bars_held = 0
                           self.trade_size_in_btc = trade_sz
                           logger.info(f"ALT OVERRIDE: Flipped to LONG @ ${current_close:.2f}")
                           self._notify(f"ALT COMMANDER -> OVERRIDE TO LONG @ ${current_close:.2f} | TP: {self.active_tp:.2f} SL: {self.active_sl:.2f}")
                 else:
                      if self.active_tp == 0.0: # Previously PROP, now ALT takes over same direction
                           self.entry_price = current_close # Resetting entry to snap TP/SL properly
                           self.active_tp = current_close * (1 + self.long_tp)
                           self.active_sl = current_close * (1 - self.long_sl)
                           self.bars_held = 0
                           logger.info(f"ALT ASSIMILATED PROP POSITION -> Adjusted LONG TP/SL @ ${current_close:.2f}")
                           self._notify(f"ALT COMMANDER -> TOOK OVER LONG @ ${current_close:.2f} | TP: {self.active_tp:.2f} SL: {self.active_sl:.2f}")
                      else:
                           self.bars_held += 1 # continuing existing ALT trade
            elif alt_wants_short and not alt_wants_long:
                 logger.info(f"ALT BOT ASSUMING COMMAND -> Wants SHORT.")
                 self.master_control = 'ALT'
                 if self.position != 'short':
                      trade_sz = self._calc_trade_size(current_close)
                      if self._sync_exchange_position(current_close, 'short', trade_sz):
                           self.position = 'short'
                           self.entry_price = current_close
                           self.active_tp = current_close * (1 - self.short_tp)
                           self.active_sl = current_close * (1 + self.short_sl)
                           self.bars_held = 0
                           self.trade_size_in_btc = trade_sz
                           logger.info(f"ALT OVERRIDE: Flipped to SHORT @ ${current_close:.2f}")
                           self._notify(f"ALT COMMANDER -> OVERRIDE TO SHORT @ ${current_close:.2f} | TP: {self.active_tp:.2f} SL: {self.active_sl:.2f}")
                 else:
                      if self.active_tp == 0.0:
                           self.entry_price = current_close
                           self.active_tp = current_close * (1 - self.short_tp)
                           self.active_sl = current_close * (1 + self.short_sl)
                           self.bars_held = 0
                           logger.info(f"ALT ASSIMILATED PROP POSITION -> Adjusted SHORT TP/SL @ ${current_close:.2f}")
                           self._notify(f"ALT COMMANDER -> TOOK OVER SHORT @ ${current_close:.2f} | TP: {self.active_tp:.2f} SL: {self.active_sl:.2f}")
                      else:
                           self.bars_held += 1
            else:
                 # --- PROPORTIONAL RUNNING BASELINE ---
                 if self.master_control == 'ALT' and self.position is not None:
                      # ALT is satisfied and is holding existing trade
                      self.bars_held += 1
                 else:
                      # PROP Engine Evaluation
                      self.master_control = 'PROP'
                      # PROP does NOT use active tp/sl
                      self.active_tp = 0.0
                      self.active_sl = 0.0
                      
                      diff_bull = bull_prob - bear_prob
                      diff_bear = bear_prob - bull_prob
                      ENTER_MARGIN = 0.2288
                      FLIP_MARGIN = 0.0008
                      FLAT_MARGIN = -0.0666
                      
                      if self.position is None:
                           if diff_bull > ENTER_MARGIN:
                                trade_sz = self._calc_trade_size(current_close)
                                if self._sync_exchange_position(current_close, 'long', trade_sz):
                                     self.position = 'long'
                                     self.entry_price = current_close
                                     self.bars_held = 0
                                     self.trade_size_in_btc = trade_sz
                                     logger.info(f"PROP BASELINE -> LONG @ ${current_close:.2f}")
                                     self._notify(f"PROP BASELINE -> LONG @ ${current_close:.2f}")
                           elif diff_bear > ENTER_MARGIN:
                                trade_sz = self._calc_trade_size(current_close)
                                if self._sync_exchange_position(current_close, 'short', trade_sz):
                                     self.position = 'short'
                                     self.entry_price = current_close
                                     self.bars_held = 0
                                     self.trade_size_in_btc = trade_sz
                                     logger.info(f"PROP BASELINE -> SHORT @ ${current_close:.2f}")
                                     self._notify(f"PROP BASELINE -> SHORT @ ${current_close:.2f}")
                      else:
                           flipped = False
                           went_flat = False
                           
                           if self.position == 'long':
                                if diff_bear >= FLIP_MARGIN:
                                     target_position = 'short'
                                     flipped = True
                                elif diff_bull < FLAT_MARGIN:
                                     target_position = 'flat'
                                     went_flat = True
                           elif self.position == 'short':
                                if diff_bull >= FLIP_MARGIN:
                                     target_position = 'long'
                                     flipped = True
                                elif diff_bear < FLAT_MARGIN:
                                     target_position = 'flat'
                                     went_flat = True
                                
                           if flipped or went_flat:
                                trade_sz = self._calc_trade_size(current_close) if flipped else 0.0
                                if self._sync_exchange_position(current_close, target_position, trade_sz):
                                     trade_type = "LONG" if self.position == "long" else "SHORT"
                                     if flipped:
                                          logger.info(f"PROP REVERSING! Closing {trade_type} to open {target_position.upper()}.")
                                     else:
                                          logger.info(f"PROP GOING FLAT! Closing {trade_type}.")
                                          
                                     reason = "PROP Reversal Flip" if flipped else "PROP Momentum Dropped (Flat)"
                                     trade = self._record_trade(trade_type, self.entry_price, current_close, self.bars_held, reason)
                                     self._notify(f"{reason} {trade_type} -> {target_position.upper()} | PnL: {trade['return_pct']:+.2f}%")
                                     
                                     self.position = target_position if flipped else None
                                     self.entry_price = current_close if flipped else 0.0
                                     self.bars_held = 0
                                     self.trade_size_in_btc = trade_sz if flipped else 0.0
                           else:
                                self.bars_held += 1
                                
            self.save_state(current_close, bull_prob=bull_prob, bear_prob=bear_prob)
                    
        except Exception as e:
            logger.error(f"Execution engine error: {e}")
            import traceback; traceback.print_exc()

    def _notify(self, msg):
        try:
            req = urllib.request.Request(NTFY_TOPIC, data=msg.encode('utf-8'))
            urllib.request.urlopen(req, timeout=5)
            logger.info(f"Notification sent: {msg}")
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")

    def run_forever(self):
        logger.info("Initializing ENSEMBLE LIVE EXECUTION Daemon...")
        step_ran_this_candle = False
        
        while True:
            self.check_tp_sl()
            now = datetime.utcnow()
            minutes = now.minute
            seconds = now.second
            remainder = minutes % 15
            
            if remainder == 0 and seconds < 10:
                if not step_ran_this_candle:
                    logger.info(f"Barrier Reached. Generating Consensus Inference...")
                    self.step()
                    step_ran_this_candle = True
            elif remainder != 0:
                step_ran_this_candle = False
                
            time.sleep(5)

if __name__ == "__main__":
    import sys
    trader = LiveEnsembleTrader()
    trader.step()
    if "--cron" not in sys.argv:
        trader.run_forever()
    else:
        logger.info("Cron cycle complete. Exiting.")
