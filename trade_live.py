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

import sys
sys.path.insert(0, '.')
from data.feature_engineer import compute_live_features, get_feature_cols
from ml.model import AttentionLSTMModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants (matching Trial 255 config — top-ranked model)
SYMBOL = 'BTC/USDT'
TIMEFRAME = '15m'
MODEL_PATH = 'models/trial_255.pth'
CONFIG_PATH = 'models/trial_255_config.json'
SCALER_PATH = 'data_storage/BTC_USDT_15m_scaler.json'
SEQ_LEN = 128
NTFY_TOPIC = 'https://ntfy.sh/TradeBot5234'
TP_PCT = 0.015
SL_PCT = 0.0075
MAX_HOLD_BARS = 16

class PaperTrader:
    def __init__(self):
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
        })
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        with open(CONFIG_PATH, 'r') as f:
            self.config = json.load(f)
            
        self.model = AttentionLSTMModel(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            output_dim=2,
            dropout=self.config['dropout'],
            num_heads=self.config['num_heads']
        ).to(self.device)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=True))
        self.model.eval()
        
        self.in_trade = False
        self.entry_price = 0.0
        self.entry_time = None
        self.bars_held = 0
        self.paper_balance = 10000.0  # $10,000 starting paper balance
        self.fee = 0.001 # 0.1% binance native fee simulation
        
        # Load state recovery
        if os.path.exists("data_storage/live_state.json"):
            try:
                with open("data_storage/live_state.json", "r") as f:
                    recovery = json.load(f)
                    self.paper_balance = recovery.get('paper_balance', 10000.0)
                    self.in_trade = recovery.get('in_trade', False)
                    self.entry_price = recovery.get('entry_price', 0.0)
                    self.bars_held = recovery.get('bars_held', 0)
                logger.info(f"Recovered State on Boot. Holding Position: {self.in_trade}")
            except Exception as e:
                logger.error(f"Failed to recover state: {e}")
                
        logger.info(f"Loaded AI Trader on {self.device}. Initial Balance: ${self.paper_balance:.2f}")

    def fetch_recent_data(self):
        # We need seq_len + max_hold_bars + buffer for historical indicators (e.g. SMA_50, RealVol_48)
        # 4H indicators need at least 34 bars (34*16 = 544 15m candles). 1000 is perfectly safe.
        candles = self.exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=1000)
        df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    def save_state(self, current_close, bull_prob=0.0):
        state = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "paper_balance": self.paper_balance,
            "in_trade": self.in_trade,
            "entry_price": self.entry_price,
            "current_price": current_close,
            "bars_held": self.bars_held,
            "bull_prob": round(bull_prob * 100, 1),
            "open_pnl_usd": 0.0,
            "open_pnl_pct": 0.0,
            "take_profit_target": self.entry_price * (1 + TP_PCT) if self.in_trade else 0.0,
            "stop_loss_target": self.entry_price * (1 - SL_PCT) if self.in_trade else 0.0,
            "trade_amount_btc": self.paper_balance / self.entry_price if self.in_trade else 0.0,
            "trade_amount_usd": self.paper_balance if self.in_trade else 0.0
        }
        
        if self.in_trade:
            gross_ret = (current_close - self.entry_price) / self.entry_price
            net_ret = gross_ret - (self.fee * 2)
            state['open_pnl_pct'] = round(net_ret * 100, 3)
            state['open_pnl_usd'] = round(self.paper_balance * net_ret, 2)
            
        with open("data_storage/live_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def manage_position(self, current_close, current_high, current_low):
        if not self.in_trade:
            return
            
        self.bars_held += 1
        tp_price = self.entry_price * (1 + TP_PCT)
        sl_price = self.entry_price * (1 - SL_PCT)
        
        exit_price = None
        reason = None
        
        # Check stop loss first (conservative risk management)
        if current_low <= sl_price:
            exit_price = sl_price
            reason = "Stop Loss Hit (-0.75%)"
        # Check take profit
        elif current_high >= tp_price:
            exit_price = tp_price
            reason = "Take Profit Hit (+1.5%)"
        # Check time barrier
        elif self.bars_held >= MAX_HOLD_BARS:
            exit_price = current_close
            reason = "Time Barrier Exhausted (4 Hours Passed)"
            
        if exit_price is not None:
            # Calculate P&L
            gross_ret = (exit_price - self.entry_price) / self.entry_price
            net_ret = gross_ret - (self.fee * 2) # Round trip exchange fee calculation
            profit_usd = self.paper_balance * net_ret
            self.paper_balance += profit_usd
            
            logger.info(f"EXIT TRADE [{reason}] @ ${exit_price:.2f} | P&L: {net_ret*100:.2f}% (${profit_usd:.2f})")
            logger.info(f"New Paper Balance: ${self.paper_balance:.2f}")
            
            # Push notification
            emoji = '✅' if profit_usd > 0 else '❌'
            self._notify(f"{emoji} CLOSED: {reason} @ ${exit_price:.2f} | P&L: {net_ret*100:+.2f}% (${profit_usd:+.2f}) | Balance: ${self.paper_balance:.2f}")
            
            self.in_trade = False
            self.entry_price = 0.0
            self.bars_held = 0

    def step(self):
        try:
            df = self.fetch_recent_data()
            current_close = df['close'].iloc[-1]
            current_high = df['high'].iloc[-1]
            current_low  = df['low'].iloc[-1]
            current_time = df['timestamp'].iloc[-1]
            
            logger.info(f"Live Market Heartbeat: {current_time} | BTC Price: ${current_close:.2f}")
            
            # 1. Manage active paper positions dynamically
            self.manage_position(current_close, current_high, current_low)
            
            # 2. Re-evaluate market for entries if flat
            if not self.in_trade:
                from datetime import datetime
                current_utc_hour = datetime.utcnow().hour
                if 13 <= current_utc_hour <= 16:
                    logger.info("SKIPPING US MARKET OPEN CHOP (5 PM - 9 PM UTC+4). Staying flat.")
                    bull_prob = 0.0
                else:
                    # Compute features on the fly dynamically using historical saved means/stds
                    live_df = compute_live_features(df, SCALER_PATH)
                    
                    if len(live_df) < SEQ_LEN:
                        logger.warning("Waiting for more market stream data to build complete architectural sequence...")
                        return
                    
                # Extract physical trailing 128 sequence with EXACT 41 trained feature columns
                    feature_cols = get_feature_cols()
                    latest_features = live_df[feature_cols].iloc[-SEQ_LEN:].values.astype(np.float32)
                    tensor_input = torch.tensor(latest_features).unsqueeze(0).to(self.device)  # [1, 128, 41]
                    
                    with torch.no_grad():
                        logits = self.model(tensor_input)
                        probabilities = torch.softmax(logits, dim=1)
                        bull_prob = probabilities[0][1].item()
                        prediction = torch.argmax(logits, dim=1).item()
                    
                    logger.info(f"AI Neural Network -> Bullish Edge Probability: {bull_prob*100:.1f}%")
                    
                    if prediction == 1:
                        logger.info(f"🔥🔥 AI DETECTED EDGE! ENTERING PAPER LONG @ ${current_close:.2f}")
                        self.in_trade = True
                        self.entry_price = current_close
                        self.entry_time = current_time
                        self.bars_held = 0
                        tp = current_close * (1 + TP_PCT)
                        sl = current_close * (1 - SL_PCT)
                        self._notify(f"🔥 ENTRY LONG @ ${current_close:.2f} | TP: ${tp:.2f} | SL: ${sl:.2f} | Balance: ${self.paper_balance:.2f} | Prob: {bull_prob*100:.1f}%")
                    else:
                        logger.info("No mathematical edge detected. Preserving capital. Staying flat.")
            else:
                bull_prob = 0.0 # Standardize if we didn't calculate prediction
                
            self.save_state(current_close, bull_prob=bull_prob if 'bull_prob' in locals() else 0.0)
                    
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
        logger.info("Initializing Live Paper Trading Background Daemon...")
        while True:
            # We align it exactly to the real-world 15m clock (0, 15, 30, 45) + 5 seconds for exchange latency
            now = datetime.utcnow()
            minutes = now.minute
            seconds = now.second
            
            # Check exactly when the next physical 15-minute candle closes in reality
            next_15m = 15 - (minutes % 15)
            
            if next_15m == 15 and seconds < 10:
                # Execution Barrier Triggered
                self.step()
                time.sleep(60) # Prevent double firing in the same minute
            else:
                sleep_sec = (next_15m * 60) - seconds + 5
                logger.info(f"Daemon Sleeping for {sleep_sec} seconds... Next Execution Engine Step aligned to exact 15m candle close.")
                
                if not hasattr(self, '_first_run_done'):
                    logger.info("Performing instant baseline structural inspection right now against present timeframe before initiating sleep schedule...")
                    self.step()
                    self._first_run_done = True
                
                time.sleep(sleep_sec)

if __name__ == "__main__":
    trader = PaperTrader()
    trader.run_forever()
