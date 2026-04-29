import pandas as pd
import numpy as np

df = pd.read_csv('test_mixed_scaled(1).csv', index_col=0)
df.index = pd.to_datetime(df.index)
df = df.sort_index()

print('Total rows:', len(df))
print('Anomaly ratio:', df['ground_truth'].mean())
print()

gt = df['ground_truth']
starts = df.index[gt.diff().eq(1)]
stops  = df.index[gt.diff().eq(-1)]

feat_cols = [c for c in df.columns if c != 'ground_truth']

healthy_means = df.loc[gt == 0, feat_cols].mean()
anom_means    = df.loc[gt == 1, feat_cols].mean()
delta = (anom_means - healthy_means).abs().sort_values(ascending=False)

print("Top 10 features with largest healthy vs anomaly mean difference:")
print(delta.head(10).round(4).to_dict())
print()
print("Top 5 features with ZERO discriminative power:")
print(delta.tail(5).round(4).to_dict())
print()
print(f"Number of anomaly windows: {len(starts)}")
for i,(s,e) in enumerate(zip(starts[:5], stops[:5])):
    print(f"  Window {i+1}: {s} -> {e} ({(e-s).seconds}s)")
