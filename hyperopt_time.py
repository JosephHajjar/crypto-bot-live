import json
import datetime
import collections

try:
    with open('data_storage/precomputed_signals.json', 'r') as f:
        data = json.load(f)
    
    signals = data.get('signals', [])
    
    # Reconstruct trades
    trades = []
    current_entry = None
    
    for s in signals:
        if s['signal'] in ['BUY', 'SELL']:
            current_entry = s
        elif s['signal'] == 'CLOSE' and current_entry is not None:
            trades.append({
                'entry_time': current_entry['time'],
                'signal': current_entry['signal'],
                'pnl': s.get('pnl', 0.0)
            })
            current_entry = None
            
    if not trades:
        print("No completed trades found.")
    else:
        # Group by UTC+4 hour
        stats_by_hour = collections.defaultdict(lambda: {'count': 0, 'wins': 0, 'total_pnl': 0.0})
        
        for t in trades:
            dt = datetime.datetime.utcfromtimestamp(t['entry_time'])
            dt += datetime.timedelta(hours=4)
            hr = dt.hour
            
            stats_by_hour[hr]['count'] += 1
            if t['pnl'] > 0:
                stats_by_hour[hr]['wins'] += 1
            stats_by_hour[hr]['total_pnl'] += t['pnl']
            
        print("--- WORST PERFORMING HOURS (Lowest Win Rate) ---")
        sorted_hours = sorted(range(24), key=lambda h: (stats_by_hour[h]['wins']/stats_by_hour[h]['count']) if stats_by_hour[h]['count'] > 0 else 100.0)
        
        for hr in sorted_hours:
            st = stats_by_hour[hr]
            if st['count'] == 0: continue
            win_rate = (st['wins'] / st['count']) * 100
            avg_pnl = st['total_pnl'] / st['count']
            print(f"{hr:02d}:00 - {(hr+1)%24:02d}:00 | Trades: {st['count']:2d} | Win Rate: {win_rate:5.1f}% | Avg PNL per trade: {avg_pnl:+6.2f}% | Net Total PNL: {st['total_pnl']:+7.2f}%")

except Exception as e:
    print('Failed:', e)
