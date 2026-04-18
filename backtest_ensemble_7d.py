"""
Backtest the FINAL DEPLOYED ensemble bot (trade_live_ensemble.py) on the last 7 days.
Faithfully replicates the dual-commander (ALT + PROP) logic with regime switching.
"""
import sys, os, json, time
import torch
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, '.')
from data.feature_engineer_btc import compute_live_features, get_feature_cols
from ml.model import AttentionLSTMModel

# ─── FETCH 7 DAYS OF 15M BTC DATA ───
def fetch_recent_15m(days=10):
    """Fetch extra days for feature warmup, we'll slice to 7 days for trading."""
    all_raw = []
    end_ts = int(time.time() * 1000)
    ims = 900000  # 15m in ms
    candles_needed = days * 96
    pages = (candles_needed + 999) // 1000

    for page in range(pages):
        page_end = end_ts - (page * 1000 * ims)
        page_start = page_end - (1000 * ims)
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=1000&startTime={page_start}&endTime={page_end}"
        res = requests.get(url, timeout=15)
        data = res.json()
        if data and isinstance(data, list):
            all_raw = data + all_raw
        time.sleep(0.1)

    seen = set()
    unique = []
    for r in all_raw:
        if r[0] not in seen:
            seen.add(r[0])
            unique.append(r)
    unique.sort(key=lambda x: x[0])

    df = pd.DataFrame(unique, columns=['timestamp','open','high','low','close','volume',
                                        'close_time','qav','num_trades','tbbav','tbqav','ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    return df


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler_path = 'models/BTC_USDT_15m_scaler.json'
    
    # ─── LOAD MODELS ───
    with open('models/holy_grail_config.json', 'r') as f:
        cfg_long = json.load(f)
    seq_len_long = cfg_long.get('seq_len', 128)
    long_tp = cfg_long.get('take_profit', 0.0125)
    long_sl = cfg_long.get('stop_loss', 0.0250)
    long_max_hold = cfg_long.get('max_hold_bars', 12)
    
    model_long = AttentionLSTMModel(
        input_dim=cfg_long['input_dim'], hidden_dim=cfg_long['hidden_dim'],
        num_layers=cfg_long['num_layers'], output_dim=2, dropout=cfg_long['dropout'], num_heads=cfg_long['num_heads']
    ).to(device)
    model_long.load_state_dict(torch.load('models/holy_grail.pth', map_location=device, weights_only=True))
    model_long.eval()
    
    with open('models_short/holy_grail_short_config.json', 'r') as f:
        cfg_short = json.load(f)
    seq_len_short = cfg_short.get('seq_len', 128)
    short_tp = cfg_short.get('take_profit', 0.0175)
    short_sl = cfg_short.get('stop_loss', 0.0375)
    short_max_hold = cfg_short.get('max_hold_bars', 12)
    
    model_short = AttentionLSTMModel(
        input_dim=cfg_short['input_dim'], hidden_dim=cfg_short['hidden_dim'],
        num_layers=cfg_short['num_layers'], output_dim=2, dropout=cfg_short['dropout'], num_heads=cfg_short['num_heads']
    ).to(device)
    model_short.load_state_dict(torch.load('models_short/holy_grail_short.pth', map_location=device, weights_only=True))
    model_short.eval()
    
    MAX_SEQ = max(seq_len_long, seq_len_short)
    
    # ─── FETCH AND ENGINEER FEATURES ───
    print("Fetching 10 days of 15m BTC data (extra days for feature warmup)...")
    df_raw = fetch_recent_15m(days=10)
    print(f"  Fetched {len(df_raw)} raw candles.")
    
    print("Computing features...")
    df_feat = compute_live_features(df_raw, scaler_path)
    
    feature_cols = get_feature_cols()
    available = [c for c in feature_cols if c in df_feat.columns]
    
    # Slice to last 7.5 days for actual trading (720 candles)
    trade_candles = int(7.5 * 96)
    if len(df_feat) > trade_candles + MAX_SEQ:
        df_trade = df_feat.iloc[-(trade_candles + MAX_SEQ):].copy()
    else:
        df_trade = df_feat.copy()
    
    X = df_trade[available].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    close = df_trade['close'].values
    high = df_trade['high'].values
    low = df_trade['low'].values
    timestamps = df_trade['timestamp'].values if 'timestamp' in df_trade.columns else df_raw['timestamp'].values[-len(X):]
    
    # ─── PRECOMPUTE ALL SIGNALS ───
    print("Running model inference...")
    bull_probs = []
    bear_probs = []
    
    all_seqs_long = []
    all_seqs_short = []
    valid_indices = []
    
    for i in range(MAX_SEQ, len(X)):
        all_seqs_long.append(X[i - seq_len_long : i])
        all_seqs_short.append(X[i - seq_len_short : i])
        valid_indices.append(i)
    
    batch_size = 2048
    with torch.no_grad():
        for start in range(0, len(all_seqs_long), batch_size):
            end = min(start + batch_size, len(all_seqs_long))
            t_long = torch.tensor(np.array(all_seqs_long[start:end])).to(device)
            t_short = torch.tensor(np.array(all_seqs_short[start:end])).to(device)
            
            p_bull = torch.softmax(model_long(t_long), dim=1)[:, 1].cpu().numpy().tolist()
            p_bear = torch.softmax(model_short(t_short), dim=1)[:, 1].cpu().numpy().tolist()
            
            bull_probs.extend(p_bull)
            bear_probs.extend(p_bear)
    
    # Only use last 7 days of signals for trading
    signals_start = max(0, len(bull_probs) - trade_candles)
    
    print(f"  Total signals: {len(bull_probs)}, Trading on last {len(bull_probs) - signals_start}")
    
    # ─── ENSEMBLE BACKTEST ───
    # Faithfully replicate trade_live_ensemble.py logic
    LEVERAGE = 15.0
    CATASTROPHE_CAP = 0.075
    ENTER_MARGIN = 0.2288
    FLIP_MARGIN = 0.0008
    FLAT_MARGIN = -0.0666
    FEE_PCT = 0.00035  # Hyperliquid taker fee per side
    DISABLE_PROP = True  # Toggle: True = ALT-only, False = full ensemble
    
    balance = 100.0  # Start with $100
    initial_balance = balance
    position = None  # 'long' or 'short'
    entry_price = 0.0
    bars_held = 0
    active_tp = 0.0
    active_sl = 0.0
    master_control = 'PROP'
    trades = []
    
    # Track BTC buy-and-hold for comparison
    btc_start_price = close[valid_indices[signals_start]]
    
    for sig_idx in range(signals_start, len(bull_probs)):
        bar_idx = valid_indices[sig_idx]
        bull_prob = bull_probs[sig_idx]
        bear_prob = bear_probs[sig_idx]
        c_p = close[bar_idx]
        h_p = high[bar_idx]
        l_p = low[bar_idx]
        
        # ─── FAST TP/SL CHECK (simulated with bar high/low) ───
        if position is not None:
            exit_price = None
            reason = None
            
            # ALT TP/SL (only check if targets are set — prevents 0.0 ghost triggers)
            if master_control == 'ALT' and active_tp > 0 and active_sl > 0:
                if position == 'long':
                    if l_p <= active_sl:
                        exit_price = active_sl; reason = f"ALT LONG SL (-{long_sl*100}%)"
                    elif h_p >= active_tp:
                        exit_price = active_tp; reason = f"ALT LONG TP (+{long_tp*100}%)"
                elif position == 'short':
                    if h_p >= active_sl:
                        exit_price = active_sl; reason = f"ALT SHORT SL (+{short_sl*100}%)"
                    elif l_p <= active_tp:
                        exit_price = active_tp; reason = f"ALT SHORT TP (-{short_tp*100}%)"
            
            # CATASTROPHE STOP-LOSS (always active)
            if exit_price is None:
                if position == 'long':
                    cat_sl = entry_price * (1 - CATASTROPHE_CAP)
                    if l_p <= cat_sl:
                        exit_price = cat_sl; reason = "CATASTROPHE LONG SL (-7.5%)"
                elif position == 'short':
                    cat_sl = entry_price * (1 + CATASTROPHE_CAP)
                    if h_p >= cat_sl:
                        exit_price = cat_sl; reason = "CATASTROPHE SHORT SL (-7.5%)"
            
            if exit_price is not None:
                if balance <= 0: break
                if position == 'long':
                    ret_pct = (exit_price - entry_price) / entry_price
                else:
                    ret_pct = (entry_price - exit_price) / entry_price
                ret_pct -= FEE_PCT * 2  # entry + exit fee
                leveraged_ret = ret_pct * LEVERAGE
                if leveraged_ret < -1.0: leveraged_ret = -1.0  # Can't lose more than 100%
                pnl_usd = balance * leveraged_ret
                balance += pnl_usd
                if balance < 0: balance = 0
                trades.append({
                    'type': position.upper(), 'entry': entry_price, 'exit': exit_price,
                    'ret_pct': ret_pct * 100, 'pnl_usd': pnl_usd, 'bars': bars_held,
                    'reason': reason, 'commander': master_control
                })
                master_control = 'PROP'
                position = None; entry_price = 0.0; bars_held = 0
                active_tp = 0.0; active_sl = 0.0
                continue
        
        # ─── REGIME DETECTION (only when flat) ───
        if position is None and sig_idx >= 96:  # Need at least ~1 day of returns
            lookback = min(sig_idx, 672)  # ~7 days lookback for monthly vol estimate
            recent_closes = [close[valid_indices[j]] for j in range(sig_idx - lookback, sig_idx)]
            if len(recent_closes) > 1:
                rets = np.diff(recent_closes) / recent_closes[:-1]
                vol_monthly = np.std(rets) * np.sqrt(2880) * 100
                if vol_monthly >= 10.50:
                    master_control = 'PROP'
                else:
                    master_control = 'ALT'
        
        # ─── ALT TIME BARRIER ───
        if master_control == 'ALT' and position is not None:
            max_hold = long_max_hold if position == 'long' else short_max_hold
            if bars_held >= max_hold:
                # Dynamic hold extension if edge is still heavy
                is_heavy_edge = False
                if position == 'long' and bull_prob >= 0.60: is_heavy_edge = True
                elif position == 'short' and bear_prob >= 0.50: is_heavy_edge = True
                
                if not is_heavy_edge:
                    if balance <= 0: break
                    if position == 'long':
                        ret_pct = (c_p - entry_price) / entry_price
                    else:
                        ret_pct = (entry_price - c_p) / entry_price
                    ret_pct -= FEE_PCT * 2
                    leveraged_ret = ret_pct * LEVERAGE
                    if leveraged_ret < -1.0: leveraged_ret = -1.0
                    pnl_usd = balance * leveraged_ret
                    balance += pnl_usd
                    if balance < 0: balance = 0
                    trades.append({
                        'type': position.upper(), 'entry': entry_price, 'exit': c_p,
                        'ret_pct': ret_pct * 100, 'pnl_usd': pnl_usd, 'bars': bars_held,
                        'reason': 'ALT Time Barrier', 'commander': 'ALT'
                    })
                    master_control = 'PROP'
                    position = None; entry_price = 0.0; bars_held = 0
                    active_tp = 0.0; active_sl = 0.0
                    continue
        
        # ─── ENTRY / MANAGEMENT LOGIC ───
        alt_wants_long = bull_prob >= 0.60
        alt_wants_short = bear_prob >= 0.50
        
        if alt_wants_long:
            master_control = 'ALT'
            if position != 'long':
                # Close existing position first
                if position is not None:
                    if balance <= 0: break
                    if position == 'short':
                        ret_pct = (entry_price - c_p) / entry_price - FEE_PCT * 2
                    else:
                        ret_pct = (c_p - entry_price) / entry_price - FEE_PCT * 2
                    leveraged_ret = ret_pct * LEVERAGE
                    if leveraged_ret < -1.0: leveraged_ret = -1.0
                    pnl_usd = balance * leveraged_ret
                    balance += pnl_usd
                    if balance < 0: balance = 0
                    trades.append({
                        'type': position.upper(), 'entry': entry_price, 'exit': c_p,
                        'ret_pct': ret_pct * 100, 'pnl_usd': pnl_usd, 'bars': bars_held,
                        'reason': 'ALT Override Flip', 'commander': master_control
                    })
                
                position = 'long'
                entry_price = c_p
                active_tp = c_p * (1 + long_tp)
                active_sl = c_p * (1 - long_sl)
                bars_held = 0
            else:
                bars_held += 1
                
        elif alt_wants_short and not alt_wants_long:
            master_control = 'ALT'
            if position != 'short':
                if position is not None:
                    if balance <= 0: break
                    if position == 'long':
                        ret_pct = (c_p - entry_price) / entry_price - FEE_PCT * 2
                    else:
                        ret_pct = (entry_price - c_p) / entry_price - FEE_PCT * 2
                    leveraged_ret = ret_pct * LEVERAGE
                    if leveraged_ret < -1.0: leveraged_ret = -1.0
                    pnl_usd = balance * leveraged_ret
                    balance += pnl_usd
                    if balance < 0: balance = 0
                    trades.append({
                        'type': position.upper(), 'entry': entry_price, 'exit': c_p,
                        'ret_pct': ret_pct * 100, 'pnl_usd': pnl_usd, 'bars': bars_held,
                        'reason': 'ALT Override Flip', 'commander': master_control
                    })
                
                position = 'short'
                entry_price = c_p
                active_tp = c_p * (1 - short_tp)
                active_sl = c_p * (1 + short_sl)
                bars_held = 0
            else:
                bars_held += 1
        else:
            # PROP Engine (or just hold if disabled)
            if master_control == 'ALT' and position is not None:
                bars_held += 1  # ALT holding existing trade
            elif DISABLE_PROP:
                # No PROP — just stay flat or hold existing position
                if position is not None:
                    bars_held += 1
            else:
                master_control = 'PROP'
                active_tp = 0.0
                active_sl = 0.0
                
                diff_bull = bull_prob - bear_prob
                diff_bear = bear_prob - bull_prob
                
                if position is None:
                    if diff_bull > ENTER_MARGIN:
                        position = 'long'
                        entry_price = c_p
                        bars_held = 0
                    elif diff_bear > ENTER_MARGIN:
                        position = 'short'
                        entry_price = c_p
                        bars_held = 0
                else:
                    flipped = False
                    went_flat = False
                    target_position = None
                    
                    if position == 'long':
                        if diff_bear >= FLIP_MARGIN:
                            target_position = 'short'; flipped = True
                        elif diff_bull < FLAT_MARGIN:
                            target_position = 'flat'; went_flat = True
                    elif position == 'short':
                        if diff_bull >= FLIP_MARGIN:
                            target_position = 'long'; flipped = True
                        elif diff_bear < FLAT_MARGIN:
                            target_position = 'flat'; went_flat = True
                    
                    if flipped or went_flat:
                        if balance <= 0: break
                        if position == 'long':
                            ret_pct = (c_p - entry_price) / entry_price
                        else:
                            ret_pct = (entry_price - c_p) / entry_price
                        ret_pct -= FEE_PCT * 2
                        leveraged_ret = ret_pct * LEVERAGE
                        if leveraged_ret < -1.0: leveraged_ret = -1.0
                        pnl_usd = balance * leveraged_ret
                        balance += pnl_usd
                        if balance < 0: balance = 0
                        reason = "PROP Reversal Flip" if flipped else "PROP Momentum Flat"
                        trades.append({
                            'type': position.upper(), 'entry': entry_price, 'exit': c_p,
                            'ret_pct': ret_pct * 100, 'pnl_usd': pnl_usd, 'bars': bars_held,
                            'reason': reason, 'commander': 'PROP'
                        })
                        
                        if flipped:
                            position = target_position
                            entry_price = c_p
                            bars_held = 0
                        else:
                            position = None; entry_price = 0.0; bars_held = 0
                    else:
                        bars_held += 1
    
    # ─── CLOSE ANY OPEN POSITION AT END ───
    if position is not None:
        final_close = close[valid_indices[-1]]
        if position == 'long':
            ret_pct = (final_close - entry_price) / entry_price
        else:
            ret_pct = (entry_price - final_close) / entry_price
        ret_pct -= FEE_PCT * 2
        leveraged_ret = ret_pct * LEVERAGE
        if leveraged_ret < -1.0: leveraged_ret = -1.0
        pnl_usd = balance * leveraged_ret
        balance += pnl_usd
        if balance < 0: balance = 0
        trades.append({
            'type': position.upper(), 'entry': entry_price, 'exit': final_close,
            'ret_pct': ret_pct * 100, 'pnl_usd': pnl_usd, 'bars': bars_held,
            'reason': 'End of Backtest', 'commander': master_control
        })
    
    btc_end_price = close[valid_indices[-1]]
    btc_return = (btc_end_price - btc_start_price) / btc_start_price * 100
    
    # ─── RESULTS ───
    print("\n" + "=" * 80)
    print("  ENSEMBLE BOT BACKTEST — LAST 7 DAYS")
    print("=" * 80)
    
    total_roi = (balance - initial_balance) / initial_balance * 100
    wins = [t for t in trades if t['ret_pct'] > 0]
    losses = [t for t in trades if t['ret_pct'] <= 0]
    
    alt_trades = [t for t in trades if t['commander'] == 'ALT']
    prop_trades = [t for t in trades if t['commander'] == 'PROP']
    
    print(f"\n  Starting Balance:  ${initial_balance:.2f}")
    print(f"  Ending Balance:    ${balance:.2f}")
    print(f"  Total ROI:         {total_roi:+.2f}%  (at 15x leverage)")
    print(f"  Total Trades:      {len(trades)}")
    print(f"  Win Rate:          {len(wins)}/{len(trades)} = {len(wins)/len(trades)*100:.1f}%" if trades else "  No trades")
    
    if trades:
        avg_win = np.mean([t['ret_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['ret_pct'] for t in losses]) if losses else 0
        print(f"  Avg Win:           {avg_win:+.3f}%")
        print(f"  Avg Loss:          {avg_loss:+.3f}%")
    
    print(f"\n  ALT Commander:     {len(alt_trades)} trades")
    print(f"  PROP Commander:    {len(prop_trades)} trades")
    
    # ─── BENCHMARK COMPARISON ───
    print("\n" + "-" * 80)
    print("  BENCHMARK COMPARISON")
    print("-" * 80)
    
    # S&P 500 average: ~10.5% annualized → ~0.20% per week
    sp500_annual = 10.5
    sp500_weekly = (1 + sp500_annual / 100) ** (1/52) - 1
    sp500_weekly_pct = sp500_weekly * 100
    
    print(f"\n  BTC Buy & Hold (7d):     {btc_return:+.2f}%")
    print(f"  BTC Price:               ${btc_start_price:.2f} -> ${btc_end_price:.2f}")
    print(f"\n  S&P 500 Avg Weekly:      +{sp500_weekly_pct:.3f}%  (historical ~10.5% annualized)")
    print(f"  S&P 500 Avg 7d Return:   +${initial_balance * sp500_weekly:.2f} on ${initial_balance:.2f}")
    
    print(f"\n  Ensemble Bot (15x lev):  {total_roi:+.2f}%")
    
    if total_roi > sp500_weekly_pct:
        print(f"\n  [PASS] ENSEMBLE BOT BEATS S&P 500 AVERAGE by {total_roi - sp500_weekly_pct:+.2f}%")
    else:
        print(f"\n  [FAIL] ENSEMBLE BOT UNDERPERFORMS S&P 500 by {sp500_weekly_pct - total_roi:+.2f}%")
    
    if total_roi > btc_return:
        print(f"  [PASS] ENSEMBLE BOT BEATS BTC BUY & HOLD by {total_roi - btc_return:+.2f}%")
    else:
        print(f"  [FAIL] ENSEMBLE BOT UNDERPERFORMS BTC by {btc_return - total_roi:+.2f}%")
    
    # ─── TRADE LOG ───
    print("\n" + "-" * 80)
    print("  TRADE LOG")
    print("-" * 80)
    print(f"  {'#':<3} {'Type':<6} {'Cmd':<5} {'Entry':>10} {'Exit':>10} {'Ret%':>8} {'PnL':>9} {'Bars':>5} {'Reason'}")
    print("  " + "-" * 75)
    for i, t in enumerate(trades):
        print(f"  {i+1:<3} {t['type']:<6} {t['commander']:<5} ${t['entry']:>9.2f} ${t['exit']:>9.2f} {t['ret_pct']:>+7.3f}% ${t['pnl_usd']:>+8.2f} {t['bars']:>5} {t['reason']}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
