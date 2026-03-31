import pandas as pd

df = pd.read_csv('data_storage/BTC_USDT_5m_processed.csv')
total = len(df)
ones = df['Target'].sum()
zeros = total - ones

print(f"Total Rows: {total}")
print(f"Class 1 (Profitable Trades): {ones} ({(ones/total)*100:.2f}%)")
print(f"Class 0 (Unprofitable Trades): {zeros} ({(zeros/total)*100:.2f}%)")
