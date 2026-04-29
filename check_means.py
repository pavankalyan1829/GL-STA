import pandas as pd
import numpy as np
df = pd.read_csv('test_mixed_scaled(1).csv')
df = df.select_dtypes(include=[np.number])
gt = df['ground_truth']
feat_cols = [c for c in df.columns if c != 'ground_truth']

results = []
for col in feat_cols:
    m_h = df.loc[gt==0, col].mean()
    m_a = df.loc[gt==1, col].mean()
    results.append({'feat': col, 'h_mean': m_h, 'a_mean': m_a, 'diff': m_a - m_h})

res_df = pd.DataFrame(results).sort_values('diff', ascending=False)
print(res_df.to_string(index=False))
