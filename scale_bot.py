import json, os

ratio = 0.05

try:
    with open('data_storage/live_state.json', 'r') as f:
        state = json.load(f)
    print(f'Old state: {state}')
    state['paper_balance'] = state.get('paper_balance', 10000.0) * ratio
    if 'open_pnl_usd' in state: state['open_pnl_usd'] = state['open_pnl_usd'] * ratio
    if 'trade_amount_usd' in state: state['trade_amount_usd'] = state['trade_amount_usd'] * ratio
    if 'trade_amount_btc' in state: state['trade_amount_btc'] = state['trade_amount_btc'] * ratio
    
    with open('data_storage/live_state.json', 'w') as f:
        json.dump(state, f, indent=2)
    print(f'New state stored.')
except Exception as e:
    print('Failed state:', e)

try:
    if os.path.exists('data_storage/live_trades.json'):
        with open('data_storage/live_trades.json', 'r') as f:
            trades = json.load(f)
        for t in trades:
            t['pnl_usd'] = round(t['pnl_usd'] * ratio, 2)
        with open('data_storage/live_trades.json', 'w') as f:
            json.dump(trades, f, indent=2)
        print('Scaled trades.')
except Exception as e:
    print('Failed trades:', e)
