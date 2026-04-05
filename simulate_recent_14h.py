import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from ml.model import AttentionLSTMModel
from data.feature_engineer import get_feature_cols

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def simulate(inject_to_live=True):
    print("Loading data for the recent 14 hour simulation...")
    df = pd.read_csv('data_storage/BTC_USDT_15m_processed.csv')
    df = df.copy()
    
    # We want 14 hours of actionable candles: 14 * 4 = 56 candles.
    # We extract enough back data to fulfill the 128 max sequence length.
    df_eval = df.iloc[-(56 + 128):]
    
    with open('models/holy_grail_config.json', 'r') as f: cfg_l = json.load(f)
    with open('models_short/holy_grail_short_config.json', 'r') as f: cfg_s = json.load(f)
    
    s_long = cfg_l.get('seq_len', 128)
    s_short = cfg_s.get('seq_len', 128)
    max_seq = max(s_long, s_short)
    
    # Load Models
    m_long = AttentionLSTMModel(
        input_dim=cfg_l['input_dim'], hidden_dim=cfg_l['hidden_dim'],
        num_layers=cfg_l['num_layers'], output_dim=2, dropout=cfg_l['dropout'], num_heads=cfg_l['num_heads']
    ).to(device)
    m_long.load_state_dict(torch.load('models/holy_grail.pth', map_location=device, weights_only=True))
    m_long.eval()

    m_short = AttentionLSTMModel(
        input_dim=cfg_s['input_dim'], hidden_dim=cfg_s['hidden_dim'],
        num_layers=cfg_s['num_layers'], output_dim=2, dropout=cfg_s['dropout'], num_heads=cfg_s['num_heads']
    ).to(device)
    m_short.load_state_dict(torch.load('models_short/holy_grail_short.pth', map_location=device, weights_only=True))
    m_short.eval()

    feature_cols = get_feature_cols()
    features_np = df_eval[feature_cols].values.astype(np.float32)
    unix_times = df_eval['timestamp'].values
    close_prices = df_eval['close'].values
    high_prices = df_eval['high'].values
    low_prices = df_eval['low'].values
    
    # Grab TP/SL Limits from configs mapping to what's LIVE
    LONG_TP = cfg_l.get('take_profit', 0.0125)
    LONG_SL = cfg_l.get('stop_loss', 0.025)
    LONG_MAX_BARS = cfg_l.get('max_hold_bars', 12)
    
    SHORT_TP = cfg_s.get('take_profit', 0.0175)
    SHORT_SL = cfg_s.get('stop_loss', 0.0375)
    SHORT_MAX_BARS = cfg_s.get('max_hold_bars', 12)
    
    print("\n================== 14-HOUR 24/7 DUAL-BOT SIMULATION ==================")
    print(f"LONG  -> TP: {LONG_TP*100:.2f}% | SL: {LONG_SL*100:.2f}% | Max: {LONG_MAX_BARS} Bars")
    print(f"SHORT -> TP: {SHORT_TP*100:.2f}% | SL: {SHORT_SL*100:.2f}% | Max: {SHORT_MAX_BARS} Bars")
    print("----------------------------------------------------------------------\n")
    
    position = "flat"
    entry_val = 0.0
    bars_held = 0
    trade_type = ""
    net_pnl = 0.0
    wins = 0
    losses = 0
    
    # Track the live states
    live_trades_log = []
    if os.path.exists('data_storage/live_trades.json'):
        with open('data_storage/live_trades.json', 'r') as f:
            live_trades_log = json.load(f)
            
    live_state = {}
    if os.path.exists('data_storage/live_state.json'):
        with open('data_storage/live_state.json', 'r') as f:
            live_state = json.load(f)

    # Start loop exactly at the 56th candle from the end, which ensures we have max_seq history behind it
    start_idx = len(features_np) - 56
    
    with torch.no_grad():
        for i in range(start_idx, len(features_np)):
            t_val = unix_times[i]
            dt = pd.to_datetime(t_val) if isinstance(t_val, str) else datetime.utcfromtimestamp(t_val)
            c_p, h_p, l_p = close_prices[i], high_prices[i], low_prices[i]
            
            # Predict Probabilities
            seq_l = features_np[i - s_long + 1 : i + 1]
            seq_s = features_np[i - s_short + 1 : i + 1]
            
            tl = torch.tensor(np.array([seq_l])).to(device)
            ts = torch.tensor(np.array([seq_s])).to(device)
            
            prob_bull = torch.softmax(m_long(tl), dim=1)[0, 1].item()
            prob_bear = torch.softmax(m_short(ts), dim=1)[0, 1].item()
            
            if position == "flat":
                if prob_bull >= 0.60 and prob_bear >= 0.50:
                    pass # Conflict
                elif prob_bull >= 0.60:
                    position = "open"
                    trade_type = "LONG"
                    entry_val = c_p
                    bars_held = 0
                    print(f"[{dt} UTC] ⚡ LONG ENTERED @ ${entry_val:.2f} (Bull Prob: {prob_bull*100:.1f}%)")
                elif prob_bear >= 0.50:
                    position = "open"
                    trade_type = "SHORT"
                    entry_val = c_p
                    bars_held = 0
                    print(f"[{dt} UTC] 🛑 SHORT ENTERED @ ${entry_val:.2f} (Bear Prob: {prob_bear*100:.1f}%)")
            else:
                bars_held += 1
                closed = False
                pnl = 0.0
                reason = ""
                
                if trade_type == "LONG":
                    tp_price = entry_val * (1 + LONG_TP)
                    sl_price = entry_val * (1 - LONG_SL)
                    
                    if l_p <= sl_price:
                        pnl = (sl_price - entry_val) / entry_val * 100
                        reason = "HIT STOP LOSS"
                        closed = True
                    elif h_p >= tp_price:
                        pnl = (tp_price - entry_val) / entry_val * 100
                        reason = "HIT TAKE PROFIT"
                        closed = True
                    elif bars_held >= LONG_MAX_BARS:
                        pnl = (c_p - entry_val) / entry_val * 100
                        reason = f"MAX TIME ({LONG_MAX_BARS} bars)"
                        closed = True
                        
                elif trade_type == "SHORT":
                    tp_price = entry_val * (1 - SHORT_TP)
                    sl_price = entry_val * (1 + SHORT_SL)
                    
                    if h_p >= sl_price:
                        pnl = (entry_val - sl_price) / entry_val * 100
                        reason = "HIT STOP LOSS"
                        closed = True
                    elif l_p <= tp_price:
                        pnl = (entry_val - tp_price) / entry_val * 100
                        reason = "HIT TAKE PROFIT"
                        closed = True
                    elif bars_held >= SHORT_MAX_BARS:
                        pnl = (entry_val - c_p) / entry_val * 100
                        reason = f"MAX TIME ({SHORT_MAX_BARS} bars)"
                        closed = True
                        
                if closed:
                    net_pnl += pnl
                    if pnl > 0: wins += 1
                    else: losses += 1
                    marker = "✅" if pnl > 0 else "❌"
                    print(f"[{dt} UTC] {marker} {trade_type} CLOSED: {reason}")
                    print(f"     => Exit Price: ${c_p:.2f} | PnL: {pnl:+.2f}% | Bars in trade: {bars_held}\n")
                    
                    if inject_to_live:
                        net_ret = pnl/100
                        prof_usd = live_state.get('paper_balance', 500.0) * net_ret
                        live_state['paper_balance'] = live_state.get('paper_balance', 500.0) + prof_usd
                        
                        live_trades_log.append({
                            "timestamp": dt.isoformat() + "Z",
                            "trade_type": trade_type,
                            "entry_price": entry_val,
                            "exit_price": c_p,
                            "pnl_usd": round(prof_usd, 2),
                            "return_pct": round(pnl, 2),
                            "bars_held": bars_held,
                            "reason": reason
                        })
                    
                    position = "flat"
                    
    print("====================== SIMULATION COMPLETE ======================")
    if position != "flat":
        print(f"⚠️ Simulation ended with an OPEN {trade_type} trade (Bars held: {bars_held}).")
        if inject_to_live:
            live_state['in_trade'] = True
            live_state['trade_type'] = trade_type
            live_state['entry_price'] = entry_val
            live_state['bars_held'] = bars_held
            live_state['open_pnl_pct'] = 0.0 # Will instantly update when trade_live.py starts
            live_state['open_pnl_usd'] = 0.0
    else:
        if inject_to_live:
            live_state['in_trade'] = False
            live_state['trade_type'] = None
            live_state['entry_price'] = 0.0
            live_state['bars_held'] = 0
            live_state['open_pnl_pct'] = 0.0
            live_state['open_pnl_usd'] = 0.0
            
    if inject_to_live:
        with open('data_storage/live_trades.json', 'w') as f:
            json.dump(live_trades_log, f, indent=2)
        with open('data_storage/live_state.json', 'w') as f:
            json.dump(live_state, f, indent=2)
        print(">> EXPERIMENT SUCCESSFULLY WRITTEN INTO LIVE DASHBOARD STATE!")
        
    print(f"Total Completed Trades : {wins + losses}")
    if (wins+losses) > 0:
        print(f"Overall Win Rate       : {(wins/(wins+losses))*100:.1f}%")
    print(f"Net Cumulative ROI     : {net_pnl:+.2f}%")

if __name__ == '__main__':
    simulate()
