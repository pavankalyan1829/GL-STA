import joblib
import pandas as pd
import numpy as np
from live_inference import get_live_window  # Reusing your existing pipeline

scaler_raw = joblib.load('scaler_raw.pkl')
scaler_combined = joblib.load('scaler_combined.pkl')
features = joblib.load('features.pkl')

print("[*] Checking MySQL Stability (Double-Pass Scaled)...")
combined, raw = get_live_window(features, scaler_raw, scaler_combined)

if combined is not None:
    # Everything is now scaled to mean=0, std=1 including deltas
    # Extract MySQL features specifically
    db_indices = range(7, 14)
    db_delta_indices = [idx + 21 for idx in db_indices]
    
    db_scaled_raw = combined[-1, db_indices]
    db_scaled_delta = combined[-1, db_delta_indices]
    
    # Feature names
    feat_names = [features[i] for i in db_indices]
    
    df = pd.DataFrame({
        "Feature": feat_names,
        "Scaled_Raw": db_scaled_raw,
        "Scaled_Delta": db_scaled_delta
    })
    print("\n--- Current MySQL Signal Snapshot (Double-Normalized) ---")
    print(df.to_string(index=False))
    
    # Check for offsets
    avg_offset = np.abs(db_scaled_raw).mean()
    avg_velocity = np.abs(db_scaled_delta).mean()
    
    print(f"\n[*] Average Raw Z-Score: {avg_offset:.6f}")
    print(f"[*] Average Delta Z-Score: {avg_velocity:.6f}")
    
    if avg_offset < 1.0 and avg_velocity < 1.0:
        print("[v] STABILITY CONFIRMED: Distribution is perfectly centered at zero.")
    else:
        print("[!] WARNING: Significant drift still detected.")
else:
    print("[!] Buffering data... try again in 10 seconds.")
