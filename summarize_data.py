import pandas as pd
import numpy as np
import os

files = ["test_mixed_scaled(1).csv", "train_healthy_scaled(1).csv", "metrics_renamed.csv"]

def summarize_file(file_path):
    if not os.path.exists(file_path):
        return f"{file_path} NOT FOUND"
    
    df = pd.read_csv(file_path, index_col=0)
    summary = []
    summary.append(f"### File: {file_path}")
    summary.append(f"- **Rows**: {len(df)}")
    summary.append(f"- **Columns**: {len(df.columns)}")
    
    if 'ground_truth' in df.columns:
        gt_counts = df['ground_truth'].value_counts()
        summary.append(f"- **Labels (ground_truth)**: {gt_counts.get(0, 0)} Healthy, {gt_counts.get(1, 0)} Anomalous")
    
    if 'fault_type' in df.columns:
        ft_counts = df['fault_type'].value_counts()
        summary.append("- **Fault Type Breakdown**:")
        for ft, count in ft_counts.items():
            summary.append(f"  - {ft}: {count}")
            
    feats = df.columns.tolist()
    app_feats = [c for c in feats if c.startswith('app_')]
    db_feats = [c for c in feats if c.startswith('db_')]
    redis_feats = [c for c in feats if c.startswith('redis_')]
    
    summary.append("- **Node Features**:")
    summary.append(f"  - App: {len(app_feats)}")
    summary.append(f"  - DB: {len(db_feats)}")
    summary.append(f"  - Redis: {len(redis_feats)}")
    
    meta = [c for c in feats if not any(c.startswith(p) for p in ['app_', 'db_', 'redis_'])]
    summary.append(f"- **Metadata Columns**: {', '.join(meta)}")
    
    summary.append("\n")
    return "\n".join(summary)

output = ""
for f in files:
    output += summarize_file(f)

print(output)
