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
from data.feature_engineer import compute_live_features, get_feature_cols
from ml.model import AttentionLSTMModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
SYMBOL = 'PAXG/USDT'
COIN = 'PAXG' # Hyperliquid specific ticker
TIMEFRAME = '15m'

MODEL_LONG_PATH = 'models_gold_long/holy_grail.pth'
CONFIG_LONG_PATH = 'models_gold_long/holy_grail_config.json'
MODEL_SHORT_PATH = 'models_gold_short/holy_grail_short.pth'
CONFIG_SHORT_PATH = 'models_gold_short/holy_grail_short_config.json'

SCALER_PATH = 'data_storage/PAXG_USDT_15m_scaler.json'
NTFY_TOPIC = 'https://ntfy.sh/TradeBot5234'


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
        
        self.short_tp = cfg_short.get('take_profit', 0.0150)
        self.short_sl = cfg_short.get('stop_loss', 0.0080)
        self.short_max_hold = cfg_short.get('max_hold_bars', 8)

        self.model_short = AttentionLSTMModel(
            input_dim=cfg_short['input_dim'], hidden_dim=cfg_short['hidden_dim'],
            num_layers=cfg_short['num_layers'], output_dim=2, dropout=cfg_short['dropout'], num_heads=cfg_short['num_heads']
        ).to(self.device)
        self.model_short.load_state_dict(torch.load(MODEL_SHORT_PATH, map_location=self.device, weights_only=True))
        self.model_short.eval()
        
        # Internal State tracking (syncs with exchange)
        self.in_trade = False
        self.trade_type = None # "LONG" or "SHORT"
        self.entry_price = 0.0
        self.entry_time = None
        
        self.bars_held = 0
        if os.path.exists("data_storage/live_state_gold.json"):
            try:
                with open("data_storage/live_state_gold.json", "r") as f:
                    old_state = json.load(f)
                    self.bars_held = old_state.get("bars_held", 0)
            except Exception:
                pass
                
        self.trade_amount_paxg = 0.0
        
        self._sync_state()
        logger.info(f"Loaded Live Hyperliquid AI Trader on {self.device}. Active Trade: {self.in_trade}")

    def _sync_state(self):
        """Synchronizes local state with Hyperliquid DEX."""
        try:
            # 1. Fetch Perp Margin
            user_state = self.info.user_state(self.wallet_address)
            margin_summary = user_state.get("marginSummary", {})
            perp_balance = float(margin_summary.get("accountValue", 0.0))
            
            # 2. Fetch Spot USDC (Unified Margin support)
            spot_state = self.info.spot_user_state(self.wallet_address)
            spot_usdc = 0.0
            if "balances" in spot_state:
                for bal in spot_state["balances"]:
                    if bal.get("coin") == "USDC":
                        spot_usdc = float(bal.get("total", 0.0))
                        
            self.live_balance = spot_usdc + perp_balance
            
            positions = user_state.get("assetPositions", [])
            in_trade_found = False
            for pos in positions:
                p = pos.get("position", {})
                if p.get("coin") == COIN and float(p.get("szi", 0)) != 0.0:
                    szi = float(p.get("szi"))
                    self.in_trade = True
                    self.trade_type = "LONG" if szi > 0 else "SHORT"
                    self.entry_price = float(p.get("entryPx", 0.0))
                    self.trade_amount_paxg = abs(szi)
                    in_trade_found = True
                    break
                    
            if not in_trade_found:
                self.in_trade = False
                self.trade_type = None
                self.bars_held = 0
                self.entry_price = 0.0
                
        except Exception as e:
            logger.error(f"Failed to sync state from Hyperliquid: {e}")

    def fetch_recent_data(self):
        import requests
        symbol_fmt = SYMBOL.replace('/', '')
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol_fmt}&interval={TIMEFRAME}&limit=1000"
        res = requests.get(url, timeout=15)
        raw_candles = res.json()
        
        # CCXT format
        candles = [
            [int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
            for c in raw_candles
        ]
        
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def save_state(self, current_close, bull_prob=0.0, bear_prob=0.0):
        tp_target = 0.0
        sl_target = 0.0
        if self.in_trade:
             if self.trade_type == "LONG":
                 tp_target = self.entry_price * (1 + self.long_tp)
                 sl_target = self.entry_price * (1 - self.long_sl)
             elif self.trade_type == "SHORT":
                 tp_target = self.entry_price * (1 - self.short_tp)
                 sl_target = self.entry_price * (1 + self.short_sl)

        state = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "paper_balance": self.live_balance,
            "in_trade": self.in_trade,
            "trade_type": self.trade_type,
            "entry_price": self.entry_price,
            "current_price": current_close,
            "bars_held": self.bars_held,
            "bull_prob": round(bull_prob * 100, 2),
            "bear_prob": round(bear_prob * 100, 2),
            "open_pnl_usd": 0.0,
            "open_pnl_pct": 0.0,
            "take_profit_target": tp_target,
            "stop_loss_target": sl_target,
            "trade_amount_paxg": self.trade_amount_paxg,
            "trade_amount_usd": self.trade_amount_paxg * current_close if self.trade_amount_paxg else 0.0
        }
        
        if self.in_trade:
             if self.trade_type == "LONG":
                 gross_ret = (current_close - self.entry_price) / self.entry_price
             else:
                 gross_ret = (self.entry_price - current_close) / self.entry_price
             
             state['open_pnl_pct'] = round(gross_ret * 100, 3)
             state['open_pnl_usd'] = round((self.trade_amount_paxg * self.entry_price) * gross_ret, 2)
            
        with open("data_storage/live_state_gold.json", "w") as f:
            json.dump(state, f, indent=2)

    def manage_position(self, current_close, current_high, current_low):
        if not self.in_trade:
            return
            
        self.bars_held += 1
        
        tp_price = 0.0
        sl_price = 0.0
        max_bars = 0
        
        if self.trade_type == "LONG":
            tp_price = self.entry_price * (1 + self.long_tp)
            sl_price = self.entry_price * (1 - self.long_sl)
            max_bars = self.long_max_hold
        else: # SHORT
            tp_price = self.entry_price * (1 - self.short_tp)
            sl_price = self.entry_price * (1 + self.short_sl)
            max_bars = self.short_max_hold

        exit_price = None
        reason = None
        
        if self.trade_type == "LONG":
            if current_low <= sl_price:
                exit_price = sl_price
                reason = f"Stop Loss Hit (-{self.long_sl*100}%)"
            elif current_high >= tp_price:
                exit_price = tp_price
                reason = f"Take Profit Hit (+{self.long_tp*100}%)"
            elif self.bars_held >= max_bars:
                exit_price = current_close
                reason = "Time Barrier Exhausted"
        elif self.trade_type == "SHORT":
            if current_high >= sl_price:
                exit_price = sl_price
                reason = f"Stop Loss Hit (+{self.short_sl*100}% price rise)"
            elif current_low <= tp_price:
                exit_price = tp_price
                reason = f"Take Profit Hit (-{self.short_tp*100}% price drop)"
            elif self.bars_held >= max_bars:
                exit_price = current_close
                reason = "Time Barrier Exhausted"
            
        if exit_price is not None:
            # LIVE CLOSE
            logger.info(f"Triggering EXECUTING LIVE CLOSE for {COIN}... Reason: {reason}")
            try:
                # To close, we send market_close or market_open in opposite direction.
                # The safest using sdk is to run market_close if it exists natively in hyperliquid sdk.
                close_res = self.exchange.market_close(COIN, self.trade_amount_paxg)
                logger.info(f"Hyperliquid Close Response: {close_res}")
                
            except Exception as e:
                logger.error(f"Failed to execute live close order: {e}")
                self._notify(f"🚨 CRITICAL ERROR: Failed to close live position on HL: {e}")
                return

            self._sync_state() # Update balance and active positions
            
            # Record explicit historical trade receipt
            trade_record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "trade_type": self.trade_type,
                "entry_price": self.entry_price,
                "exit_price": current_close,
                "pnl_usd": 0.0, # Handled by exchange PNL natively
                "bars_held": self.bars_held,
                "reason": reason
            }
            
            trades_file = "data_storage/live_trades_gold.json"
            trades = []
            if os.path.exists(trades_file):
                try:
                    with open(trades_file, "r") as f:
                        trades = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read existing live_trades.json: {e}")
            trades.append(trade_record)
            with open(trades_file, "w") as f:
                json.dump(trades, f, indent=2)
            
            logger.info(f"EXIT TRADE [{reason}] @ ${current_close:.2f} | Balance: ${self.live_balance:.2f}")
            self._notify(f"✅ CLOSED LIVE {self.trade_type}: {reason} @ ${current_close:.2f} | Balance: ${self.live_balance:.2f}")
            
            self.in_trade = False
            self.trade_type = None
            self.entry_price = 0.0
            self.bars_held = 0
            self.trade_amount_paxg = 0.0

    def step(self):
        try:
            df = self.fetch_recent_data()
            current_close = df['close'].iloc[-1]
            current_high = df['high'].iloc[-1]
            current_low  = df['low'].iloc[-1]
            current_time = df['timestamp'].iloc[-1]
            
            self._sync_state() # Fetch latest real positions
            
            logger.info(f"Live Market Heartbeat: {current_time} | PAXG Price: ${current_close:.2f} | Balance: ${self.live_balance:.2f}")
            
            # 1. Manage active live positions dynamically
            self.manage_position(current_close, current_high, current_low)
            
            bull_prob = 0.0
            bear_prob = 0.0
            
            # 2. Compute probabilities
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
                    
                logger.info(f"AI Models -> Bullish Edge: {bull_prob*100:.1f}% | Bearish Edge: {bear_prob*100:.1f}%")

            # 3. Enter trade if flat
            if not self.in_trade and len(live_df) >= max(self.seq_len_long, self.seq_len_short):
                if bull_prob >= 0.60 and bear_prob >= 0.50:
                    logger.info("⚡ Conflicting signals detected! High market uncertainty. Remaining flat.")
                elif bull_prob >= 0.60:
                    logger.info(f"🔥🔥 LONG EDGE DETECTED! ATTEMPTING LIVE ENTRY @ ~${current_close:.2f}")
                    try:
                        # Calculate trade size (using roughly 10x leverage on total available balance)
                        # We use max balance or test amount if balance is extremely small.
                        target_notional = max(50.0, self.live_balance * 5.0) 
                        trade_sz = max(0.0001, round(target_notional / current_close, 5))
                        
                        logger.info(f"Submitting LIVE MARKET OPEN for {trade_sz} {COIN} (Notional: ${target_notional:.2f})...")
                        res = self.exchange.market_open(COIN, is_buy=True, sz=trade_sz, px=current_close*1.05, slippage=0.01)
                        if res and res.get('status') == 'ok':
                            self.in_trade = True
                            self.trade_type = "LONG"
                            self.bars_held = 0
                            self._notify(f"🔥 LIVE ENTRY LONG {trade_sz} {COIN} | Balance: ${self.live_balance:.2f} | Prob: {bull_prob*100:.1f}%")
                            self._sync_state() # Lock true entry price via API
                        else:
                            logger.error(f"Failed to open LONG order. API returned: {res}")
                    except Exception as e:
                        logger.error(f"Exception during entry: {e}")
                        
                elif bear_prob >= 0.50:
                    logger.info(f"🩸🩸 SHORT EDGE DETECTED! ATTEMPTING LIVE ENTRY @ ~${current_close:.2f}")
                    try:
                        target_notional = max(50.0, self.live_balance * 5.0)
                        trade_sz = max(0.0001, round(target_notional / current_close, 5))
                        logger.info(f"Submitting LIVE MARKET SHORT for {trade_sz} {COIN} (Notional: ${target_notional:.2f})...")
                        res = self.exchange.market_open(COIN, is_buy=False, sz=trade_sz, px=current_close*0.95, slippage=0.01)
                        if res and res.get('status') == 'ok':
                            self.in_trade = True
                            self.trade_type = "SHORT"
                            self.bars_held = 0
                            self._notify(f"🩸 LIVE ENTRY SHORT {trade_sz} {COIN} | Balance: ${self.live_balance:.2f} | Prob: {bear_prob*100:.1f}%")
                            self._sync_state() # Lock true entry price via API
                        else:
                            logger.error(f"Failed to open SHORT order. API returned: {res}")
                    except Exception as e:
                         logger.error(f"Exception during entry: {e}")
                else:
                    logger.info("No mathematical edge detected. Preserving capital. Staying flat.")
                        
            self.save_state(current_close, bull_prob=bull_prob, bear_prob=bear_prob)
                    
        except Exception as e:
            logger.error(f"Live API or Execution engine crashed on loop: {e}")

    def _notify(self, msg):
        """Send push notification via ntfy."""
        try:
            req = urllib.request.Request(NTFY_TOPIC, data=msg.encode('utf-8'))
            urllib.request.urlopen(req, timeout=5)
            logger.info(f"Notification sent: {msg}")
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")

    def run_forever(self):
        logger.info("Initializing LIVE EXECUTION Background Daemon...")
        while True:
            now = datetime.utcnow()
            minutes = now.minute
            seconds = now.second
            
            next_15m = 15 - (minutes % 15)
            
            if next_15m == 15 and seconds < 10:
                logger.info(f"Execution Barrier Reached (Minute: {minutes}, Second: {seconds}). Running step...")
                self.step()
                time.sleep(20)
            else:
                time.sleep(5)

if __name__ == "__main__":
    import sys
    trader = LiveHyperliquidTrader()
    # Run once immediately on startup
    trader.step()
    
    # If standard run, begin infinite loop on 15m physical candle clock
    if "--cron" not in sys.argv:
        trader.run_forever()
    else:
        logger.info("Cron cycle complete. Exiting stateless runner.")
