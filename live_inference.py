import os, time, logging, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import joblib, requests, json
import pandas as pd
import numpy as np
import tensorflow as tf
from collections import deque
from datetime import datetime, timedelta, UTC
from concurrent.futures import ThreadPoolExecutor, as_completed
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import register_keras_serializable

# ─────────────────────────────────────────────────────────────
# 1. CUSTOM LAYER
# ─────────────────────────────────────────────────────────────
@register_keras_serializable(package="Custom")
class GATLayer(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs); self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform', name="gat_w", trainable=True)
        self.a = self.add_weight(shape=(2 * self.units, 1),
                                 initializer='glorot_uniform', name="gat_a", trainable=True)
        super().build(input_shape)

    def call(self, x):
        h = tf.matmul(x, self.W)
        n = tf.shape(h)[1]
        h_i = tf.repeat(h, repeats=n, axis=1)
        h_j = tf.tile(h, [1, n, 1])
        e = tf.nn.leaky_relu(
            tf.reshape(tf.matmul(tf.concat([h_i, h_j], axis=-1), self.a), [-1, n, n])
        )
        return tf.nn.elu(tf.matmul(tf.nn.softmax(e, axis=-1), h))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.units)

    def get_config(self):
        cfg = super().get_config(); cfg.update({"units": self.units}); return cfg

    @classmethod
    def from_config(cls, config): return cls(**config)


# ─────────────────────────────────────────────────────────────
# 2. CONFIGURATION
# ─────────────────────────────────────────────────────────────
PROMETHEUS_URL  = "http://localhost:9090"
WINDOW_SIZE     = "10s"
TIME_STEPS      = 10
N_NODES         = 3
POLL_INTERVAL   = 8          # seconds
WARMUP_ROUNDS   = 5

# Fix B.2 — Smooth last N anomaly scores before deciding
SMOOTH_N        = 3
RECOVERY_SMOOTH_N = 1

RAW_QUERIES = {
    "app_cpu":          'rate(container_cpu_usage_seconds_total{name=~".*app.*"}[30s]) * 100',
    "app_mem":          'container_memory_usage_bytes{name=~".*app.*"}',
    "app_latency":      'max_over_time(probe_duration_seconds{job="http_latency"}[30s]) or vector(0)',
    "app_probe_success":'max_over_time(probe_success{job="http_latency"}[30s]) or vector(1)',
    "app_log_error":    'rate(container_log_errors_total{name=~".*app.*"}[30s]) or vector(0)',
    "app_log_total":    'rate(container_log_lines_total{name=~".*app.*"}[30s]) or vector(0)',
    "app_log_avg_len":  'vector(0)',
    "db_cpu":           'rate(container_cpu_usage_seconds_total{name=~".*db.*"}[30s]) * 100',
    "db_mem":           'container_memory_usage_bytes{name=~".*db.*"}',
    "db_threads":       'mysql_global_status_threads_connected or vector(1)',
    "db_pad_prom":      'vector(1)',
    "db_log_error":     'rate(container_log_errors_total{name=~".*db.*"}[30s]) or vector(0)',
    "db_log_total":     'rate(container_log_lines_total{name=~".*db.*"}[30s]) or vector(0)',
    "db_log_avg_len":   'vector(0)',
    "redis_cpu":        'rate(container_cpu_usage_seconds_total{name=~".*redis.*"}[30s]) * 100',
    "redis_mem":        'container_memory_usage_bytes{name=~".*redis.*"}',
    "redis_pad_prom_1": 'vector(0)',
    "redis_pad_prom_2": 'vector(1)',
    "redis_log_error":  'rate(container_log_errors_total{name=~".*redis.*"}[30s]) or vector(0)',
    "redis_log_total":  'rate(container_log_lines_total{name=~".*redis.*"}[30s]) or vector(0)',
    "redis_log_avg_len":'vector(0)',
}


# ─────────────────────────────────────────────────────────────
# 3. PROMETHEUS HELPERS
# ─────────────────────────────────────────────────────────────
def get_prometheus_time():
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query",
                         params={'query': 'time()'}, timeout=5).json()
        return datetime.fromtimestamp(float(r['data']['result'][0]['value'][1]), tz=UTC)
    except Exception:
        return datetime.now(UTC)


def get_metric_series(query, start, end):
    try:
        r = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range",
                         params={'query': query, 'start': start,
                                 'end': end, 'step': WINDOW_SIZE},
                         timeout=5).json()
        if 'data' not in r or not r['data']['result']:
            return pd.Series(dtype=float)
        df = pd.DataFrame(r['data']['result'][0]['values'], columns=['ts', 'val'])
        df['ts'] = pd.to_datetime(df['ts'], unit='s').dt.floor(WINDOW_SIZE).dt.tz_localize(None)
        return df.set_index('ts')['val'].astype(float)
    except Exception:
        return pd.Series(dtype=float)


def get_live_window(feature_cols, scaler_raw, scaler_combined):
    prom_now = get_prometheus_time() - timedelta(seconds=2)
    start    = prom_now - timedelta(seconds=TIME_STEPS * 10 + 60)
    s_s = start.isoformat().replace('+00:00', 'Z')
    e_s = prom_now.isoformat().replace('+00:00', 'Z')

    master_idx = pd.date_range(start, prom_now, freq=WINDOW_SIZE)\
                   .floor(WINDOW_SIZE).tz_localize(None)
    raw_df = pd.DataFrame(0.0, index=master_idx, columns=list(RAW_QUERIES.keys()))

    # Parallelized Polling for Performance
    # Instead of 21 sequential requests (slow), we fire them all at once.
    with ThreadPoolExecutor(max_workers=len(RAW_QUERIES)) as executor:
        future_map = {
            executor.submit(get_metric_series, query, s_s, e_s): name 
            for name, query in RAW_QUERIES.items()
        }
        
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                series = future.result()
                if not series.empty:
                    raw_df[name] = series.reindex(master_idx, method='ffill').fillna(0)
            except Exception:
                # Keep the sentinel alive even if one metric fetch fails.
                pass

    out_df = pd.DataFrame(0.0, index=master_idx, columns=feature_cols)
    for col in feature_cols:
        if col in raw_df.columns:
            out_df[col] = raw_df[col].values

    window_raw = out_df.tail(TIME_STEPS).values
    if len(window_raw) < TIME_STEPS:
        return None, None

    # Extract only the first 21 features (raw) for the first scaler
    n_raw = 21
    window_raw_21 = window_raw[:, :n_raw]

    # ── DOUBLE-PASS VELOCITY NORMALIZATION ───────────────────────
    # Pass 1: Scale raw window to Z-scores first
    window_raw_z = scaler_raw.transform(window_raw_21)
    window_raw_z = np.clip(window_raw_z, -6.0, 6.0)
    window_raw_z = np.nan_to_num(window_raw_z, nan=0.0)

    # Compute delta on Z-scores
    window_delta_z = np.diff(window_raw_z, axis=0, prepend=window_raw_z[0:1])

    # Stack [Raw_Z, Delta_Z]
    window_unscaled_stack = np.hstack([window_raw_z, window_delta_z])

    # Pass 2: Scale the combined set to unify entropy
    window_combined = scaler_combined.transform(window_unscaled_stack)
    window_combined = np.clip(window_combined, -6.0, 6.0)
    window_combined = np.nan_to_num(window_combined, nan=0.0)
    
    return window_combined, window_raw_21


# ─────────────────────────────────────────────────────────────
# 4. JSON OUTPUT FORMATTERS
# ─────────────────────────────────────────────────────────────
NODE_MAP = {
    'app': 'Nextcloud',
    'db': 'MySQL',
    'redis': 'Redis'
}

FEATURE_MAP = {
    'cpu': 'CPUUtil',
    'mem': 'MemUtil',
    'latency': 'Latency',
    'probe_success': 'HealthProbe',
    'log_error': 'ErrorRate',
    'log_total': 'LogVolume',
    'threads': 'ActiveThreads'
}

FEATURE_HEURISTICS = {
    'CPUUtil': (
        "Infrastructure - CPU Stress",
        "CPU pressure is elevated in {node}",
        "Inspect CPU saturation, hot requests, and container limits for {node}"
    ),
    'MemUtil': (
        "Infrastructure - Memory Stress",
        "Memory usage is elevated in {node}",
        "Inspect memory growth, cache pressure, and container memory limits for {node}"
    ),
    'Latency': (
        "Service - Latency Degradation",
        "Request latency is elevated around {node}",
        "Inspect upstream dependencies, slow queries, and network latency affecting {node}"
    ),
    'HealthProbe': (
        "Service - Health Check Instability",
        "Health probe behavior is unstable for {node}",
        "Inspect readiness/liveness failures and recent restarts for {node}"
    ),
    'ErrorRate': (
        "Application - Error Spike",
        "Error activity has increased in {node}",
        "Inspect recent exceptions, stack traces, and failing requests in {node}"
    ),
    'LogVolume': (
        "Application - Log Surge",
        "Log volume has surged in {node}",
        "Inspect recent deployments, noisy retries, and repeated warnings in {node}"
    ),
    'ActiveThreads': (
        "Database - Connection Pressure",
        "Thread or connection pressure is elevated in {node}",
        "Inspect connection pools, blocked queries, and thread growth in {node}"
    ),
}

def parse_feature_name(col):
    if 'pad' in col: return "Padding", "Pad"
    parts = col.split('_', 1)
    if len(parts) != 2: return "Unknown", col
    node = NODE_MAP.get(parts[0], parts[0].title())
    feat = FEATURE_MAP.get(parts[1], parts[1].title())
    for k, v in FEATURE_MAP.items():
        if parts[1].startswith(k): feat = v
    return node, feat


def get_anomaly_guidance(primary):
    base_feature = primary["feature"].replace(" (Velocity)", "")
    heuristic = FEATURE_HEURISTICS.get(base_feature)

    if heuristic is None:
        return (
            "Service - Behavioral Anomaly",
            f"Abnormal behavior is centered around {primary['node']}",
            f"Inspect recent metric deviations and dependency health for {primary['node']}"
        )

    anomaly_type, insight, action = heuristic
    if "Velocity" in primary["feature"]:
        insight = f"Rapid change detected in {base_feature} for {primary['node']}"
        action = f"Inspect what changed abruptly in {primary['node']} and correlate with recent spikes or drops"

    return (
        anomaly_type,
        insight.format(node=primary["node"]),
        action.format(node=primary["node"])
    )

def formulate_anomaly_json(score, threshold, X_scaled_window, preds_window, feature_cols, scaler_raw, mse_per_node):
    # Step 1: Integrated Root Cause Analysis (42 Features)
    mse_per_feature = np.mean(np.square(X_scaled_window - preds_window), axis=0)
    
    # Calculate global contribution for normalization
    total_raw_mse = np.sum(mse_per_feature)
    
    # We use all 42 features (Raw + Delta) to find the absolute biggest driver
    top_indices = np.argsort(mse_per_feature)[::-1]
    top_contributors = []
    n_raw = 21
    
    # Correct Indexing for 42 Features: [21 Raw | 21 Delta]
    # Node 0 (Nextcloud): raw[0:7]   + delta[21:28]
    # Node 1 (MySQL):     raw[7:14]  + delta[28:35]
    # Node 2 (Redis):     raw[14:21] + delta[35:42]
    
    # Calculate Node Impact based on the full 14-feature node groups
    node_contribution = {}
    if total_raw_mse > 0:
        for i, k in enumerate(["app", "db", "redis"]):
            # Sum up both raw (i*7:(i+1)*7) and delta (21+i*7:21+(i+1)*7)
            s, e = i*7, (i+1)*7
            node_mse = np.sum(np.concatenate([mse_per_feature[s:e], mse_per_feature[21+s:21+e]]))
            node_contribution[k] = float(round((node_mse / total_raw_mse) * 100, 2))
    
    for idx in top_indices:
        col = feature_cols[idx]
        if 'pad' in col:
            continue
            
        node, feat = parse_feature_name(col)
        # Handle Delta labeling
        if idx >= 21:
            feat = f"{feat} (Velocity)"
            
        impact = int((mse_per_feature[idx] / total_raw_mse) * 100) if total_raw_mse > 0 else 0
        
        # Only collect up to 3 valid contributors
        if len(top_contributors) < 3:
            top_contributors.append({
                "node": node,
                "feature": feat,
                "impact": impact
            })

    primary = top_contributors[0] if top_contributors else {"node": "Unknown", "feature": "Unknown", "impact": 100}
    
    anomaly_type, insight, action = get_anomaly_guidance(primary)
        
    confidence = min(1.0, score / (2 * threshold)) if threshold > 0 else 1.0
    
    if score < threshold:
        severity = "NORMAL"
    elif score < 1.5 * threshold:
        severity = "LOW"
    elif score < 2 * threshold:
        severity = "MEDIUM"
    else:
        severity = "HIGH"
    
    # Final report construction with node-level explainability
    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "severity": severity,
        "anomaly_type": anomaly_type,
        "overall_score": float(round(score, 6)),
        "threshold": float(round(threshold, 6)),
        "node_scores": {
            "Nextcloud": float(round(mse_per_node[0], 6)),
            "MySQL":     float(round(mse_per_node[1], 6)),
            "Redis":     float(round(mse_per_node[2], 6))
        },
        "primary_cause": primary,
        "impact_analysis": node_contribution,
        "insight": insight,
        "recommended_action": action
    }
    return json.dumps(report, indent=2)


# ─────────────────────────────────────────────────────────────
# 5. SENTINEL MAIN LOOP (Research Grade Autoencoder)
# ─────────────────────────────────────────────────────────────
def run_sentinel():
    print(f"\n{'='*60}")
    print(f"  SENTINEL LIVE V7.0 — Advanced Distribution Alignment")
    print(f"{'='*60}")

    print("[*] Loading AI assets...")
    try:
        model = tf.keras.models.load_model(
            'gat_lstm_autoencoder.keras',
            custom_objects={'GATLayer': GATLayer},
            compile=False)
            
        scaler_raw      = joblib.load('scaler_raw.pkl')
        scaler_combined = joblib.load('scaler_combined.pkl')
        feature_cols    = joblib.load('features.pkl')
        
        with open('model_thresholds.json') as f:
            thresh_data = json.load(f)
            LIVE_THRESHOLD = thresh_data['gat']['thresh_p85']
            
    except Exception as e:
        print(f"[X] FATAL: Could not load model assets: {e}")
        return

    print(f"[v] Model ready. Double-Pass Scalers loaded.")
    print(f"[v] Research Threshold: {LIVE_THRESHOLD:.4f}")
    print(f"[v] Smoothing: last {SMOOTH_N} SMA")
    print(f"[v] Recovery smoothing: last {RECOVERY_SMOOTH_N} score(s)")
    print(f"[*] Polling Prometheus every {POLL_INTERVAL}s ...\n")

    IDLE_OFFSET = 0.0
    warmup = WARMUP_ROUNDS
    score_hist = deque(maxlen=SMOOTH_N)
    anomaly_active = False

    try:
        while True:
            # Stage 1: Get Double-Normalized Window
            window_combined, window_raw = get_live_window(feature_cols, scaler_raw, scaler_combined)

            if window_combined is None:
                print("[!] Buffering — not enough data yet...")
                time.sleep(POLL_INTERVAL)
                continue

            # Stage 2: Inference
            input_tensor = np.expand_dims(window_combined, axis=0)
            input_tensor = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
            preds = model(input_tensor, training=False).numpy()
            
            # Stage 3: Scoring (14-feat per node)
            mse_per_feature = np.mean(np.square(window_combined - preds[0]), axis=0)
            mse_per_node = []
            for i in range(N_NODES):
                s, e = i*7, (i+1)*7
                node_avg = np.mean(np.concatenate([mse_per_feature[s:e], mse_per_feature[21+s:21+e]]))
                mse_per_node.append(node_avg)
            mse_per_node = np.array(mse_per_node)

            raw_score = 0.7 * np.max(mse_per_node) + 0.3 * np.mean(mse_per_node)
            score_hist.append(raw_score)
            smoothed_score = float(np.mean(score_hist))
            recovery_score = float(np.mean(list(score_hist)[-RECOVERY_SMOOTH_N:]))

            # Stage 4: Calibration & Decision
            if warmup > 0:
                print(f"[*] Calibrating adaptive baseline ({WARMUP_ROUNDS - warmup + 1}/{WARMUP_ROUNDS})...")
                # Record the max 'healthy noise' during warmup
                IDLE_OFFSET = max(IDLE_OFFSET, smoothed_score)
                warmup -= 1
                time.sleep(POLL_INTERVAL)
                continue
            
            # Keep entry stable, but let recovery use a shorter memory so we clear faster.
            detection_score = max(0, smoothed_score - IDLE_OFFSET)
            recovery_corrected_score = max(0, recovery_score - IDLE_OFFSET)
            corrected_score = recovery_corrected_score if anomaly_active else detection_score
            is_anomaly = corrected_score >= LIVE_THRESHOLD
            anomaly_active = is_anomaly

            if is_anomaly:
                json_report = formulate_anomaly_json(
                    score=corrected_score, 
                    threshold=LIVE_THRESHOLD, 
                    X_scaled_window=window_combined, 
                    preds_window=preds[0], 
                    feature_cols=feature_cols, 
                    scaler_raw=scaler_raw,
                    mse_per_node=mse_per_node
                )
                print(f"\n{json_report}\n")
            else:
                ts_str = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts_str}] Score: {corrected_score:.4f} | IDLE_H: {IDLE_OFFSET:.3f} | [ v] HEALTHY")

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        print("\n[*] Sentinel stopped.")

if __name__ == "__main__":
    run_sentinel()
