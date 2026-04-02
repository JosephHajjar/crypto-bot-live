"""
Comprehensive backtest of all qualified models across multiple recent time intervals.
Fetches fresh data from Binance and runs triple-barrier backtests.
"""
import os, sys, json, glob, time
import torch
import numpy as np
import pandas as pd
import requests

from data.feature_engineer import compute_live_features, get_feature_cols
from ml.model import AttentionLSTMModel

# Triple barrier params (match training)
TP = 0.015
SL = 0.0075
MAX_BARS = 16
FEE = 0.001

INTERVALS = {
    '1 day':   1,
    '3 days':  3,
    '7 days':  7,
    '14 days': 14,
    '30 days': 30,
    '60 days': 60,
    '90 days': 90,
}

def fetch_recent_15m(days):
    """Fetch recent 15m candles from Binance."""
    all_raw = []
    end_ts = int(time.time() * 1000)
    ims = 900000  # 15m in ms
    candles_needed = days * 96  # 96 candles per day
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


def backtest_model(model, df_feat, df_raw, seq_len, device):
    """Run triple-barrier backtest on feature-engineered data."""
    feature_cols = get_feature_cols()
    available = [c for c in feature_cols if c in df_feat.columns]
    X = df_feat[available].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Align raw data to feature index
    close = df_feat['close'].values if 'close' in df_feat.columns else df_raw['close'].values[-len(X):]
    high = df_feat['high'].values if 'high' in df_feat.columns else df_raw['high'].values[-len(X):]
    low = df_feat['low'].values if 'low' in df_feat.columns else df_raw['low'].values[-len(X):]

    if len(X) < seq_len + MAX_BARS + 5:
        return None

    all_seqs = np.array([X[i:i+seq_len] for i in range(len(X) - seq_len)])
    seq_tensor = torch.tensor(all_seqs, dtype=torch.float32).to(device)

    signals = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(seq_tensor), 2048):
            batch = seq_tensor[i:i+2048]
            out = model(batch)
            probs = torch.softmax(out, dim=1)
            bull_probs = probs[:, 1].cpu().tolist()
            signals.extend(bull_probs)

    capital = 10000.0
    initial = capital
    max_cap = capital
    max_dd = 0.0
    trades = []
    i = 0

    while i < len(signals):
        idx = i + seq_len
        if idx + MAX_BARS >= len(close):
            break
        if capital > 0:
            prob = signals[i]
            if prob > 0.60:
                # LONG
                entry = close[idx]
                entry_cap = capital
                tp_price = entry * (1 + TP)
                sl_price = entry * (1 - SL)
                exit_p = None
                reason = None
                for j in range(1, MAX_BARS + 1):
                    jdx = idx + j
                    if jdx >= len(close): break
                    if low[jdx] <= sl_price:
                        exit_p = sl_price; reason = 'SL'; break
                    if high[jdx] >= tp_price:
                        exit_p = tp_price; reason = 'TP'; break
                if exit_p is None:
                    exit_p = close[idx + MAX_BARS]; reason = 'TIME'
                ret = (exit_p - entry) / entry - FEE * 2
                capital = entry_cap * (1 + ret)
                if capital > max_cap: max_cap = capital
                dd = (capital - max_cap) / max_cap * 100
                if dd < max_dd: max_dd = dd
                trades.append({'win': ret > 0, 'ret': ret, 'reason': reason, 'type': 'LONG'})
                i += MAX_BARS
            else:
                i += 1
        else:
            i += 1

    roi = (capital - initial) / initial * 100
    wins = sum(1 for t in trades if t['win'])
    losses = len(trades) - wins
    tp_count = sum(1 for t in trades if t['reason'] == 'TP')
    sl_count = sum(1 for t in trades if t['reason'] == 'SL')
    time_count = sum(1 for t in trades if t['reason'] == 'TIME')

    if len(trades) > 1:
        rets = [t['ret'] for t in trades]
        sharpe = np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(len(trades))
    else:
        sharpe = 0

    return {
        'roi': roi, 'sharpe': sharpe, 'max_dd': max_dd,
        'trades': len(trades), 'wins': wins, 'losses': losses,
        'tp': tp_count, 'sl': sl_count, 'time': time_count
    }


def load_model(config_path, model_path, device):
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    model = AttentionLSTMModel(
        input_dim=cfg['input_dim'],
        hidden_dim=cfg['hidden_dim'],
        num_layers=cfg['num_layers'],
        output_dim=2,
        dropout=cfg['dropout'],
        num_heads=cfg['num_heads']
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, cfg


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler_path = 'data_storage/BTC_USDT_15m_scaler.json'

    # Collect all trial configs
    configs = sorted(glob.glob('models/trial_*_config.json'))
    print(f"Found {len(configs)} qualified models to backtest")
    print(f"Device: {device}")

    # Fetch max data needed (90 days + buffer for features)
    max_days = max(INTERVALS.values()) + 30  # extra buffer for indicators
    print(f"\nFetching {max_days} days of fresh 15m BTC data from Binance...")
    df_raw = fetch_recent_15m(max_days)
    print(f"  Got {len(df_raw)} candles ({df_raw['timestamp'].iloc[0]} to {df_raw['timestamp'].iloc[-1]})")

    # Compute features once
    print("Computing features...")
    df_feat = compute_live_features(df_raw, scaler_path)
    print(f"  Features computed: {len(df_feat)} rows, {len(get_feature_cols())} features")

    # For each interval, slice the LAST N days of feature data
    results = {}  # {trial_num: {interval: result_dict}}

    for interval_name, days in INTERVALS.items():
        candles = days * 96
        df_slice = df_feat.iloc[-candles:] if candles < len(df_feat) else df_feat
        print(f"\n{'='*70}")
        print(f"INTERVAL: {interval_name} ({len(df_slice)} candles)")
        print(f"{'='*70}")

        for cfg_path in configs:
            trial_num = cfg_path.split('trial_')[1].split('_config')[0]
            model_path = cfg_path.replace('_config.json', '.pth')
            if not os.path.exists(model_path):
                continue

            try:
                model, cfg = load_model(cfg_path, model_path, device)
                seq_len = cfg['seq_len']
                result = backtest_model(model, df_slice, df_raw, seq_len, device)

                if result is None:
                    continue

                if trial_num not in results:
                    results[trial_num] = {'cfg': cfg}
                results[trial_num][interval_name] = result

                if result['trades'] > 0:
                    print(f"  Trial #{trial_num:>4} | ROI: {result['roi']:>7.2f}% | "
                          f"Sharpe: {result['sharpe']:>5.2f} | DD: {result['max_dd']:>6.2f}% | "
                          f"Trades: {result['trades']:>3} (TP:{result['tp']} SL:{result['sl']} T:{result['time']}) | "
                          f"W/L: {result['wins']}/{result['losses']}")

                del model
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  Trial #{trial_num}: ERROR - {e}")

    # ============================================================
    # FINAL RANKING
    # ============================================================
    print("\n" + "=" * 80)
    print("FINAL RANKING — COMPOSITE SCORE ACROSS ALL INTERVALS")
    print("=" * 80)

    scores = []
    for trial_num, data in results.items():
        cfg = data['cfg']
        interval_rois = []
        interval_sharpes = []
        interval_dds = []
        total_trades = 0
        total_wins = 0
        profitable_intervals = 0

        for interval_name in INTERVALS:
            if interval_name in data:
                r = data[interval_name]
                interval_rois.append(r['roi'])
                interval_sharpes.append(r['sharpe'])
                interval_dds.append(r['max_dd'])
                total_trades += r['trades']
                total_wins += r['wins']
                if r['roi'] > 0:
                    profitable_intervals += 1

        if not interval_rois or total_trades == 0:
            continue

        avg_roi = np.mean(interval_rois)
        avg_sharpe = np.mean(interval_sharpes)
        worst_dd = min(interval_dds) if interval_dds else -100
        win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
        consistency = profitable_intervals / len(interval_rois) * 100

        # Composite score: weighted blend
        composite = (
            avg_roi * 0.3 +
            avg_sharpe * 10 * 0.25 +
            consistency * 0.25 +
            (100 + worst_dd) * 0.2  # penalize drawdown
        )

        scores.append({
            'trial': trial_num,
            'composite': composite,
            'avg_roi': avg_roi,
            'avg_sharpe': avg_sharpe,
            'worst_dd': worst_dd,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'consistency': consistency,
            'cfg': cfg,
            'interval_rois': {k: data[k]['roi'] for k in INTERVALS if k in data}
        })

    scores.sort(key=lambda x: x['composite'], reverse=True)

    print(f"\n{'Rank':<5} {'Trial':<8} {'Score':<8} {'AvgROI':<10} {'Sharpe':<8} {'WinRate':<8} {'MaxDD':<8} {'Consist':<8} {'Trades':<7} {'Arch'}")
    print("-" * 100)

    for i, s in enumerate(scores[:25]):
        cfg = s['cfg']
        arch = f"seq={cfg['seq_len']} h={cfg['hidden_dim']} L={cfg['num_layers']}"
        print(f"#{i+1:<4} T-{s['trial']:<6} {s['composite']:>6.1f}  {s['avg_roi']:>+8.2f}%  {s['avg_sharpe']:>6.2f}  "
              f"{s['win_rate']:>5.1f}%  {s['worst_dd']:>6.2f}%  {s['consistency']:>5.0f}%  {s['total_trades']:>5}  {arch}")

    # Detailed breakdown for top 5
    print("\n" + "=" * 80)
    print("TOP 5 — DETAILED INTERVAL BREAKDOWN")
    print("=" * 80)
    for i, s in enumerate(scores[:5]):
        print(f"\n  #{i+1} Trial #{s['trial']} (Score: {s['composite']:.1f})")
        print(f"      Arch: seq={s['cfg']['seq_len']}, hidden={s['cfg']['hidden_dim']}, layers={s['cfg']['num_layers']}, heads={s['cfg']['num_heads']}")
        print(f"      Original test ROI: {s['cfg'].get('test_roi', '?'):.1f}%, Sharpe: {s['cfg'].get('test_sharpe', '?'):.2f}")
        for interval_name, roi in s['interval_rois'].items():
            r = results[s['trial']][interval_name]
            print(f"      {interval_name:>8}: ROI {roi:>+8.2f}% | Sharpe {r['sharpe']:>5.2f} | DD {r['max_dd']:>6.2f}% | "
                  f"{r['trades']} trades (W:{r['wins']} L:{r['losses']})")

    # Save results
    with open('backtest_results.json', 'w') as f:
        json.dump({'ranking': [{k: v for k, v in s.items() if k != 'cfg'} for s in scores[:25]],
                   'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=2, default=str)
    print("\nResults saved to backtest_results.json")


if __name__ == '__main__':
    main()
