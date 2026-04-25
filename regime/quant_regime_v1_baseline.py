"""
Quant Regime Scorer — Zero ML, Zero Overfitting
Simple rules-based regime detection using proven quant signals.
Outputs: BULL / NEUTRAL / BEAR with transparent scoring.

No training. No GPU. No overfitting. Just math.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')

def score_regime(df, i=None):
    """
    Score a single day's regime. Returns dict with:
      - total_score: -100 to +100
      - regime: BULL / NEUTRAL / BEAR
      - breakdown: individual signal scores
    
    Uses only data available AT that point (no lookahead).
    """
    if i is None:
        i = len(df) - 1
    if i < 200:
        return {'total_score': 0, 'regime': 'NEUTRAL', 'breakdown': {}}
    
    c = df['close'].values
    v = df['volume'].values
    current = c[i]
    
    signals = {}
    
    # === 1. TREND (weight: 30%) ===
    # Price vs key SMAs
    sma_50 = np.mean(c[max(0,i-49):i+1])
    sma_200 = np.mean(c[max(0,i-199):i+1])
    
    # Above both = bullish, below both = bearish
    above_50 = current > sma_50
    above_200 = current > sma_200
    golden_cross = sma_50 > sma_200  # 50 > 200 = bullish structure
    
    trend_score = 0
    if above_50: trend_score += 10
    else: trend_score -= 10
    if above_200: trend_score += 10
    else: trend_score -= 10
    if golden_cross: trend_score += 10
    else: trend_score -= 10
    
    signals['trend'] = {'score': trend_score, 'max': 30,
                        'above_50sma': above_50, 'above_200sma': above_200,
                        'golden_cross': golden_cross}
    
    # === 2. MOMENTUM (weight: 25%) ===
    ret_7 = current / c[i-7] - 1
    ret_30 = current / c[i-30] - 1
    ret_90 = current / c[i-90] - 1
    
    mom_score = 0
    # Short-term momentum
    if ret_7 > 0.03: mom_score += 8
    elif ret_7 > 0: mom_score += 4
    elif ret_7 > -0.03: mom_score -= 4
    else: mom_score -= 8
    
    # Medium-term momentum
    if ret_30 > 0.10: mom_score += 9
    elif ret_30 > 0: mom_score += 4
    elif ret_30 > -0.10: mom_score -= 4
    else: mom_score -= 9
    
    # Long-term momentum (trend confirmation)
    if ret_90 > 0.15: mom_score += 8
    elif ret_90 > 0: mom_score += 4
    elif ret_90 > -0.15: mom_score -= 4
    else: mom_score -= 8
    
    signals['momentum'] = {'score': mom_score, 'max': 25,
                           'ret_7d': f"{ret_7:+.1%}", 'ret_30d': f"{ret_30:+.1%}",
                           'ret_90d': f"{ret_90:+.1%}"}
    
    # === 3. VOLATILITY REGIME (weight: 15%) ===
    daily_rets = np.diff(c[max(0,i-89):i+1]) / c[max(0,i-89):i]
    vol_7 = np.std(daily_rets[-7:]) if len(daily_rets) >= 7 else 0.02
    vol_30 = np.std(daily_rets[-30:]) if len(daily_rets) >= 30 else 0.02
    vol_90 = np.std(daily_rets) if len(daily_rets) >= 60 else 0.02
    
    vol_score = 0
    # Low vol = calm = slightly bullish, high vol = fear = bearish
    vol_ratio = vol_7 / (vol_30 + 1e-10)
    if vol_ratio < 0.7: vol_score += 8   # Vol contracting = bullish
    elif vol_ratio < 1.0: vol_score += 4
    elif vol_ratio < 1.5: vol_score -= 4
    else: vol_score -= 8                  # Vol expanding = bearish
    
    # Absolute vol level
    annual_vol = vol_30 * np.sqrt(365)
    if annual_vol < 0.40: vol_score += 7
    elif annual_vol < 0.60: vol_score += 3
    elif annual_vol < 0.80: vol_score -= 3
    else: vol_score -= 7
    
    signals['volatility'] = {'score': vol_score, 'max': 15,
                             'vol_7d': f"{vol_7*100:.2f}%", 'vol_30d': f"{vol_30*100:.2f}%",
                             'vol_ratio': f"{vol_ratio:.2f}", 'annual_vol': f"{annual_vol:.0%}"}
    
    # === 4. MARKET STRUCTURE (weight: 15%) ===
    struct_score = 0
    
    # Distance from ATH
    ath = np.max(c[:i+1])
    dd_from_ath = current / ath - 1
    
    if dd_from_ath > -0.05: struct_score += 8       # Near ATH = strong
    elif dd_from_ath > -0.15: struct_score += 4
    elif dd_from_ath > -0.30: struct_score -= 2
    elif dd_from_ath > -0.50: struct_score -= 5
    else: struct_score -= 8                          # Deep drawdown = weak
    
    # Higher lows check (last 30 days vs prior 30)
    recent_low = np.min(c[max(0,i-29):i+1])
    prior_low = np.min(c[max(0,i-59):max(0,i-29)])
    higher_low = recent_low > prior_low
    if higher_low: struct_score += 7
    else: struct_score -= 7
    
    signals['structure'] = {'score': struct_score, 'max': 15,
                            'dd_from_ath': f"{dd_from_ath:+.1%}",
                            'higher_low': higher_low}
    
    # === 5. VOLUME CONFIRMATION (weight: 15%) ===
    vol_conf_score = 0
    vol_ma_30 = np.mean(v[max(0,i-29):i+1])
    vol_ma_7 = np.mean(v[max(0,i-6):i+1])
    vol_trend = vol_ma_7 / (vol_ma_30 + 1e-10)
    
    # Rising volume on up moves = bullish confirmation
    if ret_7 > 0 and vol_trend > 1.1: vol_conf_score += 8
    elif ret_7 > 0 and vol_trend > 0.9: vol_conf_score += 4
    elif ret_7 < 0 and vol_trend > 1.1: vol_conf_score -= 8  # Selling pressure
    elif ret_7 < 0 and vol_trend > 0.9: vol_conf_score -= 4
    else: vol_conf_score += 0
    
    # OBV trend (simplified)
    obv_slice = daily_rets[-14:] if len(daily_rets) >= 14 else daily_rets
    vol_slice = v[max(0,i-13):i+1]
    if len(obv_slice) == len(vol_slice):
        obv_direction = np.sum(np.sign(obv_slice) * vol_slice[:len(obv_slice)])
        if obv_direction > 0: vol_conf_score += 7
        else: vol_conf_score -= 7
    
    signals['volume'] = {'score': vol_conf_score, 'max': 15,
                         'vol_trend': f"{vol_trend:.2f}"}
    
    # === TOTAL SCORE ===
    total = sum(s['score'] for s in signals.values())
    max_possible = sum(s['max'] for s in signals.values())
    normalized = total / max_possible * 100  # scale to -100 to +100
    
    # Regime thresholds
    if normalized > 25: regime = 'BULL'
    elif normalized < -25: regime = 'BEAR'
    else: regime = 'NEUTRAL'
    
    return {
        'total_score': normalized,
        'raw_score': total,
        'max_score': max_possible,
        'regime': regime,
        'breakdown': signals
    }


def backtest_regime_scorer(df, start_idx=200, short_in_bear=True):
    """Backtest: long BTC when BULL/NEUTRAL, short BTC when BEAR."""
    results = []
    
    for i in range(start_idx, len(df)):
        r = score_regime(df, i)
        results.append({
            'date': df.index[i],
            'close': df['close'].values[i],
            'score': r['total_score'],
            'regime': r['regime']
        })
    
    res_df = pd.DataFrame(results).set_index('date')
    btc_rets = res_df['close'].pct_change().fillna(0).values
    
    # Strategy: long when BULL/NEUTRAL, short when BEAR
    bear_pos = -1 if short_in_bear else 0
    positions = np.array([1 if r != 'BEAR' else bear_pos for r in res_df['regime']])
    strat_rets = btc_rets * positions
    
    return res_df, btc_rets, strat_rets, positions


def main():
    print("=" * 60)
    print("QUANT REGIME SCORER — NO ML, NO OVERFIT")
    print("=" * 60)
    
    df = pd.read_csv('data_storage/BTC_daily_regime.csv', index_col=0, parse_dates=True)
    print(f"Data: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
    
    # === TODAY'S REGIME ===
    today = score_regime(df)
    print(f"\n{'='*60}")
    print(f"TODAY's REGIME: {today['regime']} (score: {today['total_score']:+.0f}/100)")
    print(f"{'='*60}")
    print(f"\n  {'Signal':<15} {'Score':>6} {'Max':>5}  Details")
    print(f"  {'-'*55}")
    for name, sig in today['breakdown'].items():
        details = ' | '.join(f"{k}={v}" for k, v in sig.items() if k not in ['score', 'max'])
        bar = '+' * max(0, sig['score']) + '-' * max(0, -sig['score'])
        print(f"  {name:<15} {sig['score']:>+5} /{sig['max']:<3}  {details}")
    
    # === LAST 30 DAYS ===
    print(f"\n{'='*60}")
    print("LAST 30 DAYS")
    print(f"{'='*60}")
    print(f"{'Date':<14} {'Price':>10} {'Score':>7} {'Regime':>8} {'Action':>7}")
    print("-" * 50)
    
    for i in range(max(200, len(df)-30), len(df)):
        r = score_regime(df, i)
        date = df.index[i].strftime('%Y-%m-%d')
        price = df['close'].values[i]
        action = 'LONG' if r['regime'] != 'BEAR' else 'SHORT'
        print(f"{date:<14} ${price:>9,.0f} {r['total_score']:>+6.0f}  {r['regime']:>8}  {action:>6}")
    
    # === FULL BACKTEST ===
    print(f"\n{'='*60}")
    print("FULL BACKTEST")
    print(f"{'='*60}")
    
    res_df, btc_rets, strat_rets, positions = backtest_regime_scorer(df)
    
    btc_cum = np.cumprod(1 + btc_rets)
    strat_cum = np.cumprod(1 + strat_rets)
    
    btc_total = (btc_cum[-1] - 1) * 100
    strat_total = (strat_cum[-1] - 1) * 100
    
    n_days = len(res_df)
    n_years = n_days / 365
    btc_annual = ((1 + btc_total/100) ** (1/n_years) - 1) * 100
    strat_annual = ((1 + strat_total/100) ** (1/n_years) - 1) * 100
    
    strat_sharpe = (np.mean(strat_rets) / (np.std(strat_rets) + 1e-10)) * np.sqrt(365)
    btc_sharpe = (np.mean(btc_rets) / (np.std(btc_rets) + 1e-10)) * np.sqrt(365)
    
    dd = (strat_cum / np.maximum.accumulate(strat_cum) - 1).min() * 100
    btc_dd = (btc_cum / np.maximum.accumulate(btc_cum) - 1).min() * 100
    
    bull_days = (res_df['regime'] == 'BULL').sum()
    neutral_days = (res_df['regime'] == 'NEUTRAL').sum()
    bear_days = (res_df['regime'] == 'BEAR').sum()
    
    print(f"Period: {res_df.index[0].date()} to {res_df.index[-1].date()} ({n_days} days)")
    print(f"\n{'':>20} {'BTC Hold':>12} {'Regime Strat':>14}")
    print(f"  {'Total Return':<18} {btc_total:>+11.1f}% {strat_total:>+13.1f}%")
    print(f"  {'Annual Return':<18} {btc_annual:>+11.1f}% {strat_annual:>+13.1f}%")
    print(f"  {'Sharpe Ratio':<18} {btc_sharpe:>11.2f} {strat_sharpe:>13.2f}")
    print(f"  {'Max Drawdown':<18} {btc_dd:>11.1f}% {dd:>13.1f}%")
    print(f"\n  LONG days:    {bull_days + neutral_days} ({(bull_days+neutral_days)/n_days*100:.0f}%) [BULL+NEUTRAL]")
    print(f"  SHORT days:   {bear_days} ({bear_days/n_days*100:.0f}%) [BEAR]")
    
    # === YEARLY BREAKDOWN ===
    print(f"\n{'='*60}")
    print("YEARLY BREAKDOWN")
    print(f"{'='*60}")
    res_df['btc_ret'] = btc_rets
    res_df['strat_ret'] = strat_rets
    
    for year in sorted(res_df.index.year.unique()):
        yr = res_df[res_df.index.year == year]
        btc_yr = (np.prod(1 + yr['btc_ret'].values) - 1) * 100
        strat_yr = (np.prod(1 + yr['strat_ret'].values) - 1) * 100
        bull_yr = (yr['regime'] == 'BULL').sum()
        bear_yr = (yr['regime'] == 'BEAR').sum()
        winner = "STRAT" if strat_yr > btc_yr else "BTC"
        print(f"  {year}: BTC={btc_yr:>+8.1f}%  Strat={strat_yr:>+8.1f}%  Bull={bull_yr:>3}d  Bear={bear_yr:>3}d  [{winner}]")
    
    beats_sp = strat_annual > 15.0
    beats_btc = strat_total > btc_total
    print(f"\n{'[BEATS S&P 500]' if beats_sp else '[Below S&P]'} ({strat_annual:.1f}% vs 15%)")
    print(f"{'[Beats BTC]' if beats_btc else '[BTC buy-hold better]'}")

if __name__ == '__main__':
    main()
