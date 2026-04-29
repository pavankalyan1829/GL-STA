import pandas as pd
import numpy as np
import os
import re
from datetime import timedelta

# --- CONFIGURATION ---
METRICS_FILE  = os.path.join("nextcloud-ai", "metrics_harvested.csv")  # Named Prometheus columns, real variance

# Load simulator writes per-node files to raw_logs/<node>_raw.log.
# We look first in nextcloud-ai/raw_logs/, then raw_logs/ (current dir).
_RAW_LOGS_CANDIDATES = [
    os.path.join("nextcloud-ai", "raw_logs"),   # when run from project root
    "raw_logs",                                  # when run from nextcloud-ai/
]

RAW_LOGS_DIR  = next((p for p in _RAW_LOGS_CANDIDATES if os.path.isdir(p)), None)

# Injection log written by load_simulator.py as injection_log.csv
_INJ_CANDIDATES = [
    os.path.join("nextcloud-ai", "injection_log.csv"),
    "injection_log.csv",
]
INJECTON_LOG  = next((p for p in _INJ_CANDIDATES if os.path.exists(p)),
                     os.path.join("nextcloud-ai", "injection_log.csv"))

TRAIN_OUTPUT  = "train_healthy_scaled(1).csv"
TEST_OUTPUT   = "test_mixed_scaled(1).csv"
EXPECTED_NODES = ['app', 'db', 'redis']

# Canonical fault-type labels for per-anomaly breakdown
_FAULT_TYPE_PRIORITY = [
    'CPU_SATURATION', 'CPU_RAMP', 'MEM_LEAK', 'IO_SATURATION',
    'DB_LATENCY', 'NET_LOSS', 'NET_DELAY', 'STALL',
    'LOG_STORM', 'FLICKER', 'CACHE_FLICKER',
]

# Prometheus metric feature slots per node (7 each, 21 total).
# These MUST match the order used in live_inference.py SLOT_MAP
# and in preprocess_train.py ordered_feats.
# The 3 log slots per node (_log_error, _log_total, _log_avg_len) are
# zero-filled from Prometheus and then overwritten by raw log counts below.
METRIC_COLS = {
    'app':   ['app_cpu', 'app_mem', 'app_latency', 'app_probe_success',
               'app_log_error', 'app_log_total', 'app_log_avg_len'],
    'db':    ['db_cpu', 'db_mem', 'db_threads', 'db_pad_prom',
               'db_log_error', 'db_log_total', 'db_log_avg_len'],
    'redis': ['redis_cpu', 'redis_mem', 'redis_pad_prom_1', 'redis_pad_prom_2',
               'redis_log_error', 'redis_log_total', 'redis_log_avg_len'],
}

# Keywords that mark a line as an error/anomaly event.
# Includes simulator fault markers (FAULT_START, stress) in addition to
# standard application error vocabulary.
_ERROR_WORDS = frozenset([
    'error', 'fail', 'timeout', 'refused', 'exception', 'fatal', 'crit',
    'fault_start', 'fault_stop', 'stress', 'oom', 'killed', 'panic',
])


# Pattern for load_simulator.py per-line format:
#   [2026-04-06T10:00:00] [APP] message text
_SIM_LOG_RE = re.compile(
    r'\[(202\d-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\]\s+\[(APP|DB|REDIS)\]\s+(.*)',
    re.IGNORECASE
)
# Fallback pattern for older single-file harvested logs:
#   2026-04-06T10:00:00Z  message text   (node inferred from section headers)
_HARVEST_LOG_RE = re.compile(
    r'(202\d-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[^\s]*)\s+(.*)'
)


def _read_file_bytes(filepath):
    """Read a file handling UTF-16 BOM gracefully."""
    with open(filepath, 'rb') as f:
        raw = f.read()
    if raw.startswith(b'\xff\xfe'):
        return raw.decode('utf-16-le', errors='ignore').replace('\x00', '')
    if raw.startswith(b'\xfe\xff'):
        return raw.decode('utf-16-be', errors='ignore').replace('\x00', '')
    return raw.decode('utf-8', errors='ignore').replace('\x00', '')


def parse_raw_logs(log_path):
    """
    Parse raw container logs produced by load_simulator.py.

    Accepts either:
      - A directory path (raw_logs/) containing per-node files
        app_raw.log, db_raw.log, redis_raw.log written by the simulator.
        Each line format: [ISO_TS] [NODE] message
      - A single harvested text file (legacy format).
    """
    log_data = []

    if os.path.isdir(log_path):
        # ── Simulator mode: merge per-node log files ──────────────
        node_files = [
            (os.path.join(log_path, f"{node}_raw.log"), node)
            for node in EXPECTED_NODES
        ]
        found_any = False
        for fpath, node_hint in node_files:
            if not os.path.exists(fpath):
                continue
            found_any = True
            print(f"[*] Parsing simulator log: {fpath} (node={node_hint})...")
            try:
                content = _read_file_bytes(fpath)
            except Exception as e:
                print(f"    [X] Read error {fpath}: {e}")
                continue

            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                m = _SIM_LOG_RE.match(line)
                if m:
                    ts, node_tag, msg = m.group(1), m.group(2).lower(), m.group(3)
                    # node_tag overrides the filename for cross-tag lines
                    log_data.append({'timestamp': ts, 'container': node_tag, 'message': msg})
                    continue
                # Lines without the node tag (e.g., docker log header lines)
                # Fall back to filename-based node hint
                m2 = _HARVEST_LOG_RE.search(line)
                if m2:
                    ts, msg = m2.group(1), m2.group(2)
                    msg = re.sub(r'^(stdout|stderr)\s+[FP]\s+', '', msg)
                    log_data.append({'timestamp': ts, 'container': node_hint, 'message': msg})

        if not found_any:
            print(f"    [!] No *_raw.log files found in {log_path}.")

    else:
        # ── Legacy mode: single harvested text file ────────────────
        print(f"[*] Parsing harvested log file: {log_path}...")
        try:
            content = _read_file_bytes(log_path)
        except Exception as e:
            print(f"    [X] Read Error: {e}")
            return pd.DataFrame()

        current_node = "unknown"
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            # Section header: === APP === / === DB === / === REDIS ===
            if line.startswith("="):
                lu = line.upper()
                if "APP" in lu:               current_node = "app"
                elif "DB" in lu or "MARIADB" in lu: current_node = "db"
                elif "REDIS" in lu:           current_node = "redis"
                continue
            # Try simulator inline format first
            m = _SIM_LOG_RE.match(line)
            if m:
                ts, node_tag, msg = m.group(1), m.group(2).lower(), m.group(3)
                log_data.append({'timestamp': ts, 'container': node_tag, 'message': msg})
                continue
            # Legacy timestamp format
            m2 = _HARVEST_LOG_RE.search(line)
            if m2:
                ts, msg = m2.group(1), m2.group(2)
                msg = re.sub(r'^(stdout|stderr)\s+[FP]\s+', '', msg)
                if current_node == "unknown":
                    ml = msg.lower()
                    if "redis" in ml:                        current_node = "redis"
                    elif any(x in ml for x in ["maria", "mysql"]): current_node = "db"
                    else:                                   current_node = "app"
                log_data.append({'timestamp': ts, 'container': current_node, 'message': msg})

    return pd.DataFrame(log_data)


def preprocess():
    print(f"\n{'='*60}\n   MULTI-MODAL FUSION — LIVE INFERENCE COMPATIBLE\n{'='*60}")

    m_path = (METRICS_FILE if os.path.exists(METRICS_FILE)
               else os.path.join("nextcloud-ai", METRICS_FILE))

    # Log source: prefer simulator's raw_logs/ directory; fall back to
    # legacy single harvested file for backwards compatibility.
    l_path = RAW_LOGS_DIR  # may be None if no directory found
    if l_path is None:
        # Legacy single-file fallback
        _legacy = os.path.join("nextcloud-ai", "raw_logs_harvested.txt")
        l_path = _legacy if os.path.exists(_legacy) else None

    if not os.path.exists(m_path):
        print(f"[X] ERROR: {METRICS_FILE} not found in . or nextcloud-ai/.")
        return

    # ── 1. LOAD RAW PROMETHEUS METRICS ────────────────────────────
    m_df = pd.read_csv(m_path, index_col=0)
    # Handle both tz-aware and naive index formats
    parsed = pd.to_datetime(m_df.index, errors='coerce')
    if parsed.tz is not None:
        parsed = parsed.tz_convert('UTC').tz_localize(None)
    m_df.index = parsed.floor('10s')
    m_df = m_df[~m_df.index.isna()].sort_index()

    # Build lat DataFrame with explicit named columns (matches live inference)
    lat = pd.DataFrame(index=m_df.index)
    for node, cols in METRIC_COLS.items():
        for col in cols:
            if col in m_df.columns:
                lat[col] = m_df[col]
            else:
                lat[col] = 0.0   # pad/log cols not in raw Prometheus metrics

    # ── 2. RECOMPUTE GROUND TRUTH FROM INJECTION LOG ───────────────
    ground_truth = pd.Series(0, index=lat.index, dtype=int)
    if os.path.exists(INJECTON_LOG):
        try:
            # The injection log is a mixed-format CSV:
            #   Old rows (header):  timestamp,node,fault_type,state,intensity,metadata        (6 cols)
            #   New simulator rows: timestamp,node,fault_type,state,intensity,duration_s,metadata (7 cols)
            # Read with the 7-column schema; on_bad_lines='skip' ignores anything wider.
            # Old 6-column rows parse cleanly because duration_s gets NaN and metadata is empty.
            _INJ_COLS = ['timestamp', 'node', 'fault_type', 'state',
                         'intensity', 'duration_s', 'metadata']
            inj = pd.read_csv(
                INJECTON_LOG,
                names=_INJ_COLS,
                skiprows=1,          # skip whichever header line is present
                on_bad_lines='skip', # silently drop truly malformed rows
                engine='python',
            )
            inj['ts'] = pd.to_datetime(inj['timestamp'], errors='coerce')
            inj = inj.dropna(subset=['ts'])

            # Normalise node names: old logs used 'mariadb'/'nextcloud',
            # new simulator uses 'db'/'app'.
            _NODE_MAP = {'mariadb': 'db', 'nextcloud': 'app',
                         'nextcloud_app': 'app', 'nextclouddb': 'db'}
            inj['node'] = inj['node'].str.strip().str.lower().replace(_NODE_MAP)

            # Date-shift alignment (handles Prometheus timeline drift).
            metrics_date = lat.index.min().date()
            inj_date     = inj['ts'].min().date()
            day_delta    = (metrics_date - inj_date).days
            inj['ts_aligned'] = (inj['ts'] + pd.Timedelta(days=day_delta)) \
                                  .dt.tz_localize(None).dt.floor('10s')

            starts = inj[inj['state'].str.upper() == 'START']
            stops  = inj[inj['state'].str.upper() == 'STOP']
            for _, row in starts.iterrows():
                st = row['ts_aligned']
                et_match = stops[
                    (stops['ts_aligned'] >= st) & (stops['node'] == row['node'])
                ]
                if not et_match.empty:
                    et = et_match.iloc[0]['ts_aligned'] + pd.Timedelta(seconds=30)
                elif pd.notna(row.get('duration_s')) and row['duration_s'] > 0:
                    et = st + pd.Timedelta(seconds=float(row['duration_s']) + 30)
                else:
                    et = st + pd.Timedelta(minutes=2)
                ground_truth[(lat.index >= st) & (lat.index <= et)] = 1

            print(f"    [v] Labels: {ground_truth.sum()} anomaly rows "
                  f"({ground_truth.mean():.2%}) from {len(starts)} fault events")
        except Exception as e:
            print(f"    [!] Label realignment failed: {e}")

    healthy_indices = lat.index[ground_truth == 0]
    split_time = healthy_indices[int(len(healthy_indices) * 0.7)]

    # ── 3. POPULATE LOG FEATURE SLOTS FROM RAW LOGS ───────────────
    # The 3 log columns per node in METRIC_COLS are already in 'lat' as
    # zeros (step 1 zero-fills anything not in Prometheus). We overwrite
    # them here with direct per-10s counts from the simulator's raw logs:
    #   {node}_log_total   → lines written to that node's log in the window
    #   {node}_log_error   → lines containing any error/fault keyword
    #   {node}_log_avg_len → mean character length of log messages
    # No NLP/SVD — direct counts avoid the sparse-matrix zero problem.

    if l_path is not None and (os.path.isdir(l_path) or os.path.exists(l_path)):
        logs_df = parse_raw_logs(l_path)
        if not logs_df.empty:
            print("[*] Fusing raw log counts into feature slots...")
            # Normalise timestamps
            clean_ts = (logs_df['timestamp']
                        .str.replace(r'[\[\]]', '', regex=True)
                        .str.extract(r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})')[0]
                        .str.replace('T', ' '))
            logs_df['timestamp'] = (pd.to_datetime(clean_ts, errors='coerce')
                                     .dt.tz_localize(None).dt.floor('10s'))
            logs_df = logs_df.dropna(subset=['timestamp'])
            print(f"    [i] Metric Range: {m_df.index.min()} to {m_df.index.max()}")
            print(f"    [i] Log Range:    {logs_df['timestamp'].min()} "
                  f"to {logs_df['timestamp'].max()}")

            # Derive error flag and message length per line
            logs_df['msg_lower'] = logs_df['message'].str.lower()
            logs_df['is_error']  = logs_df['msg_lower'].apply(
                lambda x: int(any(w in x for w in _ERROR_WORDS)))
            logs_df['msg_len'] = logs_df['message'].str.len()

            filled_nodes = 0
            for node in EXPECTED_NODES:
                node_logs = logs_df[logs_df['container'] == node]
                if node_logs.empty:
                    print(f"    [!] No logs found for node={node}")
                    continue

                g = node_logs.set_index('timestamp').groupby(pd.Grouper(freq='10s'))
                log_total   = g.size().rename(f'{node}_log_total')
                log_error   = g['is_error'].sum().rename(f'{node}_log_error')
                log_avg_len = g['msg_len'].mean().rename(f'{node}_log_avg_len')

                lat[f'{node}_log_total']   = log_total.reindex(lat.index, fill_value=0)
                lat[f'{node}_log_error']   = log_error.reindex(lat.index, fill_value=0)
                lat[f'{node}_log_avg_len'] = log_avg_len.reindex(lat.index, fill_value=0.0)
                filled_nodes += 1

            active_windows = (lat[[f'{n}_log_total' for n in EXPECTED_NODES]]
                              .sum(axis=1) > 0).sum()
            print(f"    [v] Log features filled for {filled_nodes} nodes | "
                  f"{active_windows}/{len(lat)} windows have log activity.")
        else:
            print("    [!] Log files empty — log slots remain zero.")
    else:
        print("    [!] No log source found — log slots remain zero.")

    # ── 4. ASSEMBLE FUSED FEATURE MATRIX ────────────────────────────
    # 'lat' now contains all 21 features: Prometheus metrics + log counts.
    # GAT tensor reshape: [batch, 3 nodes, 7 features per node]
    fused_df = lat.copy()
    fused_df['ground_truth'] = ground_truth

    feat_all = METRIC_COLS['app'] + METRIC_COLS['db'] + METRIC_COLS['redis']
    for col in feat_all:
        if col not in fused_df.columns:
            fused_df[col] = 0.0

    # ── 4b. PER-WINDOW FAULT-TYPE LABEL ─────────────────────────────
    # Assign the dominant active fault type to each window from the
    # injection log. This enables per-anomaly-type recall breakdown.
    fault_type_series = pd.Series('HEALTHY', index=fused_df.index, dtype=str)
    if os.path.exists(INJECTON_LOG):
        try:
            _INJ_COLS2 = ['timestamp', 'node', 'fault_type', 'state',
                          'intensity', 'duration_s', 'metadata']
            inj2 = pd.read_csv(
                INJECTON_LOG,
                names=_INJ_COLS2,
                skiprows=1,
                on_bad_lines='skip',
                engine='python',
            )
            inj2['ts'] = pd.to_datetime(inj2['timestamp'], errors='coerce')
            inj2 = inj2.dropna(subset=['ts'])
            _NODE_MAP2 = {'mariadb': 'db', 'nextcloud': 'app',
                          'nextcloud_app': 'app', 'nextclouddb': 'db'}
            inj2['node'] = inj2['node'].str.strip().str.lower().replace(_NODE_MAP2)
            metrics_date2 = fused_df.index.min().date()
            inj_date2     = inj2['ts'].min().date()
            day_delta2    = (metrics_date2 - inj_date2).days
            inj2['ts_aligned'] = (inj2['ts'] + pd.Timedelta(days=day_delta2)) \
                                   .dt.tz_localize(None).dt.floor('10s')
            starts2 = inj2[inj2['state'].str.upper() == 'START']
            stops2  = inj2[inj2['state'].str.upper() == 'STOP']
            for _, row in starts2.iterrows():
                st = row['ts_aligned']
                ft = str(row['fault_type']).strip().upper()
                et_match = stops2[
                    (stops2['ts_aligned'] >= st) & (stops2['node'] == row['node'])
                ]
                if not et_match.empty:
                    et = et_match.iloc[0]['ts_aligned'] + pd.Timedelta(seconds=30)
                elif pd.notna(row.get('duration_s')) and row['duration_s'] > 0:
                    et = st + pd.Timedelta(seconds=float(row['duration_s']) + 30)
                else:
                    et = st + pd.Timedelta(minutes=2)
                mask = (fused_df.index >= st) & (fused_df.index <= et)
                fault_type_series[mask] = ft
            print(f"    [v] Fault-type labels: "
                  f"{(fault_type_series != 'HEALTHY').sum()} anomaly windows labelled "
                  f"across {fault_type_series[fault_type_series != 'HEALTHY'].nunique()} fault types")
        except Exception as e:
            print(f"    [!] Fault-type labelling failed: {e}")
    fused_df['fault_type'] = fault_type_series

    fused_df = fused_df[feat_all + ['ground_truth', 'fault_type']]

    # Diagnostics: how many columns have real variance?
    prom_cols = [c for c in feat_all if '_log_' not in c]
    log_cols  = [c for c in feat_all if '_log_'     in c]
    prom_active = (fused_df[prom_cols].std() > 0).sum()
    log_active  = (fused_df[log_cols].std()  > 0).sum()

    print(f"\n{'-'*52}\n   FUSED DATASET STATISTICS\n{'-'*52}")
    print(f"Total Windows        : {len(fused_df)}")
    print(f"Total Features       : {len(feat_all)}  (7 per node × 3 nodes)")
    print(f"  Prometheus signals : {len(prom_cols)} cols | {prom_active} with variance")
    print(f"  Log count signals  : {len(log_cols)} cols | {log_active} with variance")
    print(f"Anomaly Ratio        : {ground_truth.mean():.2%}")
    print(f"Fault Types Found    : {sorted(fault_type_series[fault_type_series != 'HEALTHY'].unique())}")
    print(f"Feature order[0:5]   : {feat_all[:5]}")
    print("-" * 52)

    # ── 5. SAVE (Scaling lives in preprocess_train.py) ──────────────
    train_idx = healthy_indices[:int(len(healthy_indices) * 0.7)]
    # Train set: healthy only — drop fault_type from train file for clean AE training
    fused_df.loc[train_idx].drop(columns=['fault_type']).to_csv(TRAIN_OUTPUT)
    fused_df.to_csv(TEST_OUTPUT)
    print(f"[v] Preprocessing complete. "
          f"Fused vector = {len(feat_all)} features "
          f"({log_active}/{len(log_cols)} log slots populated).")


if __name__ == "__main__":
    preprocess()