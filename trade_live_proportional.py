import pandas as pd
import numpy as np
import time
import os
import torch
import json
import logging
import urllib.request
from datetime import datetime

import sys
sys.path.insert(0, '.')
from data.feature_engineer_btc import compute_live_features, get_feature_cols
from ml.model import AttentionLSTMModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - ProportionalBot - %(levelname)s - %(message)s')
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
NTFY_TOPIC = 'https://ntfy.sh/TradeBot5234' # Using the same topic for alerts

STATE_FILE = 'data_storage/live_state_proportional.json'
TRADES_FILE = 'data_storage/live_trades_proportional.json'


class LiveProportionalTrader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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

        self.model_short = AttentionLSTMModel(
            input_dim=cfg_short['input_dim'], hidden_dim=cfg_short['hidden_dim'],
            num_layers=cfg_short['num_layers'], output_dim=2, dropout=cfg_short['dropout'], num_heads=cfg_short['num_heads']
        ).to(self.device)
        self.model_short.load_state_dict(torch.load(MODEL_SHORT_PATH, map_location=self.device, weights_only=True))
        self.model_short.eval()
        
        # ============================================================
        # PROPORTIONAL PAPER TRACKING
        # ============================================================
        self.position = None # 'long' or 'short'
        self.entry_price = 0.0
        self.bars_held = 0
        self.paper_balance = 1000.0 # Starting with hypothetical $1000 for pure tracking
        
        # Restore state from disk (survives restarts)
        self._load_persisted_state()
        
        logger.info(f"Loaded Proportional Edge AI Trader on {self.device}. Current Position: {self.position}")

    def _load_persisted_state(self):
        """Restore paper position state from disk."""
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
                self.paper_balance = s.get("paper_balance", 1000.0)
            except Exception:
                pass

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

    def _record_trade(self, trade_type, entry_price, exit_price, bars_held, reason):
        """Append a completed trade to the trade log."""
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
            # Rough math based on entire paper balance sizing with 1x leverage, minus simple fees assumption
        else:
            trade_record["return_pct"] = round((entry_price - exit_price) / entry_price * 100, 3)
            
        fees_pct = 0.07 
        net_ret_pct = trade_record["return_pct"] - fees_pct
        
        trade_record["pnl_usd"] = round(self.paper_balance * (net_ret_pct / 100), 2)
        
        self.paper_balance += trade_record["pnl_usd"]
        
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

    def save_state(self, current_close, bull_prob=0.0, bear_prob=0.0):
        """Persist virtual position + metrics to disk."""
        open_pnl_pct = 0.0
        if self.position == 'long' and self.entry_price > 0:
            open_pnl_pct = (current_close - self.entry_price) / self.entry_price * 100
        elif self.position == 'short' and self.entry_price > 0:
            open_pnl_pct = (self.entry_price - current_close) / self.entry_price * 100

        state = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "paper_balance": round(self.paper_balance, 2),
            "current_price": current_close,
            "bull_prob": round(bull_prob * 100, 6),
            "bear_prob": round(bear_prob * 100, 6),
            "in_trade": self.position is not None,
            "trade_type": "LONG" if self.position == "long" else ("SHORT" if self.position == "short" else None),
            "entry_price": self.entry_price,
            "bars_held": self.bars_held,
            "open_pnl_pct": round(open_pnl_pct, 4),
            "open_pnl_usd": round(self.paper_balance * (open_pnl_pct / 100), 2),
            # Unused fields just for dashboard backward compatibility
            "take_profit_target": 0.0, 
            "stop_loss_target": 0.0,
            "trade_amount_btc": round(self.paper_balance / current_close, 5) if current_close else 0,
            "trade_amount_usd": round(self.paper_balance, 2)
        }
        
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)

    def step(self):
        try:
            df = self.fetch_recent_data()
            current_close = df['close'].iloc[-1]
            current_time = df['timestamp'].iloc[-1]
            
            logger.info(f"15m Candle Boundary: {current_time} | BTC ${current_close:.2f} | Paper Balance: ${self.paper_balance:.2f} | Pos: {self.position}")
            
            # 2. Compute AI probabilities
            bull_prob = 0.0
            bear_prob = 0.0
            
            live_df = compute_live_features(df, SCALER_PATH)
            
            # The LSTM must see explicitly fully-closed candles to match its trained distributions.
            if len(live_df) > 1:
                live_df = live_df.iloc[:-1]
                
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

            # 3. CONTINUOUS ENTRY LOGIC
            target_position = 'long' if bull_prob > bear_prob else 'short'
            
            if self.position is None:
                # Initialization run
                self.position = target_position
                self.entry_price = current_close
                self.bars_held = 0
                logger.info(f"STARTING FRESH -> {target_position.upper()} @ ${current_close:.2f}")
                self._notify(f"PROPORTIONAL BOT INITIALIZED -> {target_position.upper()} @ ${current_close:.2f}")
            elif self.position != target_position:
                # Reversal logic!
                trade_type = "LONG" if self.position == "long" else "SHORT"
                logger.info(f"REVERSING EDGE DETECTED! Closing {trade_type} to open {target_position.upper()}.")
                
                trade = self._record_trade(trade_type, self.entry_price, current_close, self.bars_held, "Edge Reversal Flip")
                self._notify(f"PROPORTIONAL CLOSED {trade_type}: Reversal | PnL: {trade['return_pct']:+.2f}% | ${trade['pnl_usd']:+.2f}")
                
                self.position = target_position
                self.entry_price = current_close
                self.bars_held = 0
                self._notify(f"PROPORTIONAL TARGET {target_position.upper()} @ ${current_close:.2f}")
            else:
                # Same position, just increment bars
                self.bars_held += 1
                
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
        logger.info("Initializing PROPORTIONAL PAPER EXECUTION Daemon (with Candle Boundary polling)...")
        step_ran_this_candle = False
        
        while True:
            now = datetime.utcnow()
            minutes = now.minute
            seconds = now.second
            
            remainder = minutes % 15
            
            if remainder == 0 and seconds < 10:
                if not step_ran_this_candle:
                    logger.info(f"Execution Barrier Reached (Minute: {minutes}, Second: {seconds}). Generating Inference...")
                    self.step()
                    step_ran_this_candle = True
            elif remainder != 0:
                step_ran_this_candle = False
                
            # Sleep 5 seconds between polls
            time.sleep(5)

if __name__ == "__main__":
    import sys
    trader = LiveProportionalTrader()
    # Run once immediately on startup
    trader.step()
    
    # If standard run, begin infinite loop
    if "--cron" not in sys.argv:
        trader.run_forever()
    else:
        logger.info("Cron cycle complete. Exiting.")
