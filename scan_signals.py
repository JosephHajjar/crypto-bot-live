import torch,json,numpy as np,pandas as pd,requests
from ml.model import AttentionLSTMModel
from data.feature_engineer_btc import compute_live_features,get_feature_cols
from datetime import datetime,timedelta

url='https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=1000'
df=pd.DataFrame([[int(c[0]),float(c[1]),float(c[2]),float(c[3]),float(c[4]),float(c[5])] for c in requests.get(url).json()],columns=['timestamp','open','high','low','close','volume'])
df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms')
live=compute_live_features(df,'data_storage/BTC_USDT_15m_scaler.json')
ml=AttentionLSTMModel(41,64,1,2,0.22,4);ml.load_state_dict(torch.load('models/holy_grail.pth',map_location='cpu',weights_only=True));ml.eval()
ms=AttentionLSTMModel(41,256,3,2,0.42,2);ms.load_state_dict(torch.load('models_short/holy_grail_short.pth',map_location='cpu',weights_only=True));ms.eval()
f=live[get_feature_cols()].values.astype(np.float32)
cutoff=datetime.utcnow()-timedelta(days=7)
longs=[];shorts=[]
with torch.no_grad():
    for i in range(128,len(f)):
        dt=live.index[i]
        if dt<cutoff:continue
        b=torch.softmax(ml(torch.tensor(f[i-128:i]).unsqueeze(0)),dim=1)[0][1].item()
        s=torch.softmax(ms(torch.tensor(f[i-64:i]).unsqueeze(0)),dim=1)[0][1].item()
        if b>=0.60:longs.append((dt,b*100,live.iloc[i]['close']))
        if s>=0.50:shorts.append((dt,s*100,live.iloc[i]['close']))

print("=== PAST 7 DAYS: ALL TRADE SIGNALS ===")
print(f"\nLONG SIGNALS ({len(longs)} total):")
if not longs: print("  None")
for dt,p,px in longs:
    print(f"  {dt.strftime('%b %d %I:%M %p')} UTC | {p:.2f}% | BTC ${px:,.2f}")
print(f"\nSHORT SIGNALS ({len(shorts)} total):")
if not shorts: print("  None")
for dt,p,px in shorts:
    print(f"  {dt.strftime('%b %d %I:%M %p')} UTC | {p:.2f}% | BTC ${px:,.2f}")
