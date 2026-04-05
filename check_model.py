import sys
sys.path.insert(0, '.')
import torch, pandas as pd, numpy as np
from ml.model import AttentionLSTMModel
from data.feature_engineer_btc import get_feature_cols
import json

df = pd.read_csv('data_storage/BTC_USDT_15m_processed.csv')
fcols = get_feature_cols()
X = df[fcols].values.astype('float32')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('models/holy_grail_config.json', 'r') as f: cfg_l = json.load(f)

mod_l = AttentionLSTMModel(
    input_dim=len(fcols), 
    hidden_dim=cfg_l['hidden_dim'], 
    num_layers=cfg_l['num_layers'], 
    output_dim=2, 
    dropout=cfg_l['dropout'], 
    num_heads=cfg_l.get('num_heads', 4)
).to(device)
mod_l.load_state_dict(torch.load('models/holy_grail.pth', map_location=device, weights_only=True))
mod_l.eval()

p_l = []
for i in range(100):
    start = -(128 + i*100)
    end = -(i*100) if i > 0 else None
    t = torch.tensor(X[start:end]).unsqueeze(0).to(device)
    p = torch.softmax(mod_l(t), dim=1)[0,1].item()*100
    p_l.append(p)

print(f'Tested {len(p_l)*100} candles. Max Bullish: {np.max(p_l):.4f}%, StdDev: {np.std(p_l):.6f}')
