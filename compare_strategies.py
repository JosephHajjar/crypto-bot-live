"""Test: What if LONG and SHORT run independently (can be in both at once)?"""
import torch, json, numpy as np, pandas as pd, requests
from ml.model import AttentionLSTMModel
from data.feature_engineer_btc import compute_live_features, get_feature_cols

print("Fetching data...")
url = 'https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=1000'
df = pd.DataFrame(
    [[int(c[0]),float(c[1]),float(c[2]),float(c[3]),float(c[4]),float(c[5])] for c in requests.get(url).json()],
    columns=['timestamp','open','high','low','close','volume']
)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
live = compute_live_features(df, 'data_storage/BTC_USDT_15m_scaler.json')

ml = AttentionLSTMModel(41,64,1,2,0.22,4)
ml.load_state_dict(torch.load('models/holy_grail.pth', map_location='cpu', weights_only=True)); ml.eval()
ms = AttentionLSTMModel(41,256,3,2,0.42,2)
ms.load_state_dict(torch.load('models_short/holy_grail_short.pth', map_location='cpu', weights_only=True)); ms.eval()

with open('models/holy_grail_config.json') as f: cfg_l = json.load(f)
with open('models_short/holy_grail_short_config.json') as f: cfg_s = json.load(f)

cols = get_feature_cols()
feat = live[cols].values.astype(np.float32)
closes = live['close'].values
highs = live['high'].values
lows = live['low'].values

LTP = cfg_l.get('take_profit', 0.0125); LSL = cfg_l.get('stop_loss', 0.025); LMX = cfg_l.get('max_hold_bars', 12)
STP = cfg_s.get('take_profit', 0.0175); SSL = cfg_s.get('stop_loss', 0.0375); SMX = cfg_s.get('max_hold_bars', 12)

print("Running inference...")
all_bull = []; all_bear = []
with torch.no_grad():
    for i in range(128, len(feat)):
        b = torch.softmax(ml(torch.tensor(feat[i-128:i]).unsqueeze(0)), dim=1)[0][1].item()
        s = torch.softmax(ms(torch.tensor(feat[i-64:i]).unsqueeze(0)), dim=1)[0][1].item()
        all_bull.append(b); all_bear.append(s)

S = 128

def run_independent():
    """Long and Short run as 2 independent bots — can be in both at once."""
    # LONG BOT
    lpos = False; lentry = 0; lbars = 0; ltrades = []
    # SHORT BOT
    spos = False; sentry = 0; sbars = 0; strades = []
    
    for idx in range(len(all_bull)):
        i = S + idx
        pb = all_bull[idx]; ps = all_bear[idx]
        c = closes[i]; h = highs[i]; l = lows[i]
        
        # --- LONG BOT ---
        if lpos:
            lbars += 1
            tp = lentry*(1+LTP); sl = lentry*(1-LSL)
            if l <= sl:
                ltrades.append(-LSL*100); lpos = False
            elif h >= tp:
                ltrades.append(LTP*100); lpos = False
            elif lbars >= LMX:
                ltrades.append((c-lentry)/lentry*100); lpos = False
        elif pb >= 0.60:
            lpos = True; lentry = c; lbars = 0
        
        # --- SHORT BOT ---
        if spos:
            sbars += 1
            tp = sentry*(1-STP); sl = sentry*(1+SSL)
            if h >= sl:
                strades.append(-SSL*100); spos = False
            elif l <= tp:
                strades.append(STP*100); spos = False
            elif sbars >= SMX:
                strades.append((sentry-c)/sentry*100); spos = False
        elif ps >= 0.50:
            spos = True; sentry = c; sbars = 0
    
    return ltrades, strades

def run_single(mode):
    """Single bot, one position at a time."""
    pos = 'flat'; entry = 0; bars = 0; trades = []
    
    for idx in range(len(all_bull)):
        i = S + idx
        pb = all_bull[idx]; ps = all_bear[idx]
        c = closes[i]; h = highs[i]; l = lows[i]
        
        if pos != 'flat':
            bars += 1
            pnl = None
            if pos == 'long':
                if l <= entry*(1-LSL): pnl = -LSL*100
                elif h >= entry*(1+LTP): pnl = LTP*100
                elif bars >= LMX: pnl = (c-entry)/entry*100
            else:
                if h >= entry*(1+SSL): pnl = -SSL*100
                elif l <= entry*(1-STP): pnl = STP*100
                elif bars >= SMX: pnl = (entry-c)/entry*100
            if pnl is not None:
                trades.append({'t': pos, 'pnl': pnl}); pos = 'flat'
        else:
            if mode == 'current':
                if pb >= 0.60 and ps >= 0.50: pass
                elif pb >= 0.60: pos = 'long'; entry = c; bars = 0
                elif ps >= 0.50: pos = 'short'; entry = c; bars = 0
            elif mode == 'long_only':
                if pb >= 0.60: pos = 'long'; entry = c; bars = 0
            elif mode == 'no_cancel':
                if pb >= 0.60: pos = 'long'; entry = c; bars = 0
                elif ps >= 0.50: pos = 'short'; entry = c; bars = 0
    
    return trades

print("\n" + "=" * 75)
print("STRATEGY SHOWDOWN: LAST 10 DAYS")
print("=" * 75)

# Strategy 1: Current
t = run_single('current')
w = sum(1 for x in t if x['pnl'] > 0)
p = sum(x['pnl'] for x in t)
print(f"\n1. CURRENT (Both + Cancel if conflict)")
print(f"   Trades: {len(t)} | Win: {w}/{len(t)} ({w/len(t)*100:.0f}%) | PnL: {p:+.2f}%")

# Strategy 2: Long Only
t = run_single('long_only')
w = sum(1 for x in t if x['pnl'] > 0)
p = sum(x['pnl'] for x in t)
print(f"\n2. LONG ONLY (Remove short model)")
print(f"   Trades: {len(t)} | Win: {w}/{len(t)} ({w/len(t)*100:.0f}%) | PnL: {p:+.2f}%")

# Strategy 3: Both, no cancel, priority to long
t = run_single('no_cancel')
w = sum(1 for x in t if x['pnl'] > 0)
p = sum(x['pnl'] for x in t)
lt = [x for x in t if x['t']=='long']; st = [x for x in t if x['t']=='short']
print(f"\n3. BOTH (No cancel, long priority)")
print(f"   Trades: {len(t)} | Win: {w}/{len(t)} ({w/len(t)*100:.0f}%) | PnL: {p:+.2f}%")
print(f"   Breakdown: {len(lt)} longs ({sum(x['pnl'] for x in lt):+.2f}%), {len(st)} shorts ({sum(x['pnl'] for x in st):+.2f}%)")

# Strategy 4: TWO INDEPENDENT BOTS
lt, st = run_independent()
all_pnl = sum(lt) + sum(st)
lw = sum(1 for x in lt if x > 0); sw = sum(1 for x in st if x > 0)
print(f"\n4. ** TWO INDEPENDENT BOTS (can hold long AND short simultaneously)")
print(f"   Long Bot:  {len(lt)} trades | Win: {lw}/{len(lt)} ({lw/len(lt)*100:.0f}%) | PnL: {sum(lt):+.2f}%")
print(f"   Short Bot: {len(st)} trades | Win: {sw}/{len(st)} ({sw/len(st)*100:.0f}%) | PnL: {sum(st):+.2f}%") if st else print(f"   Short Bot: 0 trades")
print(f"   COMBINED:  {len(lt)+len(st)} trades | PnL: {all_pnl:+.2f}%")

print("\n" + "=" * 75)
winner = max([
    ("CURRENT", sum(x['pnl'] for x in run_single('current'))),
    ("LONG ONLY", sum(x['pnl'] for x in run_single('long_only'))),
    ("BOTH NO CANCEL", sum(x['pnl'] for x in run_single('no_cancel'))),
    ("TWO INDEPENDENT", sum(run_independent()[0]) + sum(run_independent()[1])),
], key=lambda x: x[1])
print(f"WINNER: {winner[0]} with {winner[1]:+.2f}% PnL")
print("=" * 75)
