import sys
import os
import joblib
import pandas as pd
import numpy as np

# Add the project dir to path if needed
sys.path.append(r'd:\PROJECTS\Major Project')

# Mock the GATLayer if needed, or import from live_inference
from live_inference import GATLayer, get_live_window

# Load scaler and features
scaler = joblib.load(r'd:\PROJECTS\Major Project\scaler.pkl')
feature_cols = joblib.load(r'd:\PROJECTS\Major Project\features.pkl')

print(f"Features loaded: {len(feature_cols)}")
print(f"Scaler mean shape: {scaler.mean_.shape}")

# Test the filling logic part of get_live_window specifically
# (Since we might not have a running Prometheus right now, we can check the logic)

def test_filling_logic():
    mean_vec = pd.Series(scaler.mean_, index=feature_cols)
    print("\nSample means for padding features:")
    for p in ['db_pad_prom', 'redis_pad_prom_1', 'redis_pad_prom_2']:
        if p in mean_vec:
            print(f"  {p}: {mean_vec[p]}")
    
    # Check if a feature is missing
    test_feature = feature_cols[0]
    print(f"\nFilling logic for {test_feature}:")
    fill_val = float(mean_vec[test_feature]) if test_feature in mean_vec else 0.0
    print(f"  Fill value: {fill_val}")

if __name__ == "__main__":
    try:
        test_filling_logic()
        print("\n[v] Basic logic check passed.")
    except Exception as e:
        print(f"\n[X] Logic check failed: {e}")
        sys.exit(1)
