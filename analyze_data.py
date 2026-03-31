import pandas as pd
df = pd.read_csv('data_storage/BTC_USDT_15m_processed.csv', index_col=0)
n = len(df)
t = int(n * 0.7)
v = int(n * 0.85)
print(f"Total rows: {n}")
print(f"Train: {t} | Val: {v-t} | Test: {n-v}")
td = df.iloc[:t]["Target"].value_counts().to_dict()
vd = df.iloc[t:v]["Target"].value_counts().to_dict()
testd = df.iloc[v:]["Target"].value_counts().to_dict()
print(f"Target dist (train): {td}")
print(f"Target dist (val): {vd}")
print(f"Target dist (test): {testd}")
print(f"Date range: {df.index[0]} to {df.index[-1]}")
print(f"BTC price range: ${df['close'].min():.0f} - ${df['close'].max():.0f}")
pct_up_train = td.get(1, 0) / (td.get(0, 0) + td.get(1, 0)) * 100
print(f"Pct UP in train: {pct_up_train:.1f}%")
pct_up_val = vd.get(1, 0) / (vd.get(0, 0) + vd.get(1, 0)) * 100
print(f"Pct UP in val: {pct_up_val:.1f}%")
pct_up_test = testd.get(1, 0) / (testd.get(0, 0) + testd.get(1, 0)) * 100
print(f"Pct UP in test: {pct_up_test:.1f}%")
