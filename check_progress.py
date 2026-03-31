import os, json

models_dir = 'models'
configs = sorted([f for f in os.listdir(models_dir) if f.startswith('trial_') and f.endswith('_config.json')])
print(f'Total models saved: {len(configs)}')
print()
for cfg_file in configs:
    with open(os.path.join(models_dir, cfg_file), 'r') as f:
        c = json.load(f)
    trial = c.get('trial', '?')
    val = c.get('val_roi', 0)
    test = c.get('test_roi', 0)
    sharpe = c.get('test_sharpe', 0)
    trades = c.get('test_trades', 0)
    wl = c.get('test_wl', '?')
    seq = c.get('seq_len', '?')
    hid = c.get('hidden_dim', '?')
    print(f'Trial #{trial:>3} | seq={seq:>3} hid={hid:>3} | Val: {val:>7.1f}% | Test: {test:>7.1f}% | Sharpe: {sharpe:>5.2f} | Trades: {trades:>3} | W/L: {wl}')

hg_path = os.path.join(models_dir, 'holy_grail_config.json')
if os.path.exists(hg_path):
    with open(hg_path) as f:
        hg = json.load(f)
    print(f"\nHOLY GRAIL: Trial #{hg.get('trial')} | Test ROI: {hg.get('test_roi',0):.1f}% | Sharpe: {hg.get('test_sharpe',0):.2f}")
