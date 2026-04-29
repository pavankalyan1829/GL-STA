

import requests
import pandas as pd
import numpy as np
import subprocess
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────
PROMETHEUS_URL  = "http://localhost:9090"
WINDOW_SIZE     = "10s"

INJECTION_LOG   = "injection_log.csv"
OUTPUT_FILE     = "ready_for_training.csv"
METRICS_ARCHIVE = "metrics_harvested.csv"
LOGS_ARCHIVE    = "logs_harvested.csv"
RAW_LOG_TEXT    = "raw_logs_harvested.txt"
RAW_LOG_DIR     = Path("raw_logs")       # Directory written by load_simulator V6

NODES = ['app', 'db', 'redis']

# Error-indicating keywords for log parsing
FAIL_KEYWORDS = [
    'error', 'fail', 'crit', 'fatal', 'exception', 'refused', 'timeout',
    'deadlock', 'reset', 'denied', 'oom', 'panic', 'segfault', 'abort',
    'connection lost', 'too many', 'max retries', 'unavailable'
]

# ─────────────────────────────────────────────────────────────────
#  AUTO DATE DETECTION (reads from injection_log.csv if available)
# ─────────────────────────────────────────────────────────────────
def _detect_date_range():
    """Always returns the last 15 hours relative to the current time."""
    end   = datetime.utcnow()
    start = end - timedelta(hours=13)
    s = start.strftime("%Y-%m-%dT%H:%M:%SZ")
    e = end.strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[✓] Collection window: last 15 hours  ({s}  →  {e})")
    return s, e


# ─────────────────────────────────────────────────────────────────
#  PROMETHEUS QUERY DEFINITIONS — 3 NODES × 4 METRICS
# ─────────────────────────────────────────────────────────────────
def _diagnose_prometheus():
    """Quick check: show what container labels exist in Prometheus right now."""
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/label/__name__/values", timeout=10
        ).json()
        metrics = r.get("data", [])
        cadvisor_metrics = [m for m in metrics if "container" in m]
        print(f"    [diag] Total Prometheus metrics: {len(metrics)}")
        print(f"    [diag] cAdvisor container metrics found: {len(cadvisor_metrics)}")
        if not cadvisor_metrics:
            print("    [diag] ✗ No cAdvisor metrics — is cAdvisor running and scraped by Prometheus?")
        else:
            # Check what label values exist for 'name'
            r2 = requests.get(
                f"{PROMETHEUS_URL}/api/v1/label/name/values", timeout=10
            ).json()
            names = r2.get("data", [])
            print(f"    [diag] Container 'name' label values: {names[:10]}")
    except Exception as e:
        print(f"    [diag] Prometheus unreachable: {e}")


def _build_queries():
    """
    Exactly 12 queries: cpu, mem, latency, network (tx) per node.
    Uses broad OR fallbacks across cAdvisor label schemes.
    """

    def cpu(svc):
        return (
            f'rate(container_cpu_usage_seconds_total'
            f'{{container_label_com_docker_compose_service=~".*{svc}.*"}}[1m]) * 100 or '
            f'rate(container_cpu_usage_seconds_total{{name=~".*{svc}.*"}}[1m]) * 100'
        )

    def mem(svc):
        return (
            f'container_memory_usage_bytes'
            f'{{container_label_com_docker_compose_service=~".*{svc}.*"}} or '
            f'container_memory_usage_bytes{{name=~".*{svc}.*"}}'
        )

    def net_tx(svc):
        return (
            f'rate(container_network_transmit_bytes_total'
            f'{{container_label_com_docker_compose_service=~".*{svc}.*"}}[1m]) or '
            f'rate(container_network_transmit_bytes_total{{name=~".*{svc}.*"}}[1m])'
        )

    def net_rx(svc):
        return (
            f'rate(container_network_receive_bytes_total'
            f'{{container_label_com_docker_compose_service=~".*{svc}.*"}}[1m]) or '
            f'rate(container_network_receive_bytes_total{{name=~".*{svc}.*"}}[1m])'
        )

    # latency: probe duration for app, net latency (round-trip) proxy for db/redis
    return {
        # ── APP (4 metrics) ──────────────────────────────────────
        "app_cpu":      cpu("app"),
        "app_mem":      mem("app"),
        "app_latency":  'max_over_time(probe_duration_seconds{job="http_latency"}[1m]) or '
                        'max_over_time(probe_duration_seconds[1m])',
        "app_network":  net_tx("app"),

        # ── DB  (4 metrics) ──────────────────────────────────────
        "db_cpu":       cpu("db"),
        "db_mem":       mem("db"),
        "db_latency":   'mysql_global_status_slow_queries or '
                        'rate(container_network_receive_bytes_total{name=~".*db.*"}[1m]) or vector(0)',
        "db_network":   net_tx("db"),

        # ── REDIS (4 metrics) ────────────────────────────────────
        "redis_cpu":    cpu("redis"),
        "redis_mem":    mem("redis"),
        "redis_latency":'redis_commands_duration_seconds_total or '
                        'rate(container_network_receive_bytes_total{name=~".*redis.*"}[1m]) or vector(0)',
        "redis_network":net_tx("redis"),
    }


QUERIES = _build_queries()

# Exactly 12 features — the ONLY columns written to ready_for_training.csv
ALL_CORE_FEATURES = list(QUERIES.keys())   # app_cpu, app_mem, app_latency, app_network,
                                           # db_cpu,  db_mem,  db_latency,  db_network,
                                           # redis_cpu,redis_mem,redis_latency,redis_network
NODE_FEATURES = {
    node: [f"{node}_cpu", f"{node}_mem", f"{node}_latency", f"{node}_network"]
    for node in NODES
}


# ─────────────────────────────────────────────────────────────────
#  PROMETHEUS RANGE QUERY HELPER
# ─────────────────────────────────────────────────────────────────
def get_metric_series(query, start, end):
    """Executes PromQL range query; returns a time-indexed Series (max across label sets)."""
    try:
        r = requests.get(
            f"{PROMETHEUS_URL}/api/v1/query_range",
            params={'query': query, 'start': start, 'end': end, 'step': WINDOW_SIZE},
            timeout=30
        ).json()

        if 'data' not in r or not r['data']['result']:
            return pd.Series(dtype=float)

        all_series = []
        for res in r['data']['result']:
            df = pd.DataFrame(res['values'], columns=['ts', 'val'])
            df['ts'] = pd.to_datetime(df['ts'], unit='s').dt.floor(WINDOW_SIZE).dt.tz_localize(None)
            all_series.append(df.set_index('ts')['val'].astype(float))

        if not all_series:
            return pd.Series(dtype=float)

        return pd.concat(all_series, axis=1).max(axis=1)

    except Exception as e:
        print(f"    [!] Query error: {e}")
        return pd.Series(dtype=float)


# ─────────────────────────────────────────────────────────────────
#  LOG EXTRACTION — DOCKER + RAW_LOGS DIRECTORY
# ─────────────────────────────────────────────────────────────────
def _parse_log_lines(lines, node):
    """Extract structured impulses from raw log lines."""
    impulses = []
    for line in lines:
        match = re.search(r'(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})', line)
        if not match:
            continue
        ts_str = match.group(1)
        is_err = 1 if any(k in line.lower() for k in FAIL_KEYWORDS) else 0
        impulses.append({
            'ts': ts_str, 'node': node,
            'is_err': is_err, 'count': 1, 'len': len(line)
        })
    return impulses


def _harvest_docker_logs(start_ts, end_ts):
    """Pull logs directly from running Docker containers."""
    impulses = []

    try:
        raw = subprocess.run('docker ps --format "{{.Names}}"',
                             shell=True, capture_output=True, text=True)
        running_containers = [n.strip() for n in raw.stdout.splitlines() if n.strip()]
    except Exception:
        running_containers = []

    with open(RAW_LOG_TEXT, "a", encoding="utf-8") as f_raw:
        for node in NODES:
            matches = [
                c for c in running_containers
                if node in c.lower()
                and not any(x in c.lower() for x in ['promtail', 'exporter', 'cadvisor', 'loki'])
            ]
            target = matches[0] if matches else node

            print(f"    [→] Docker logs for {node} ({target})...", end=" ", flush=True)
            try:
                cmd = f'docker logs --since "{start_ts}" --until "{end_ts}" --timestamps {target}'
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
                full_output = res.stdout + res.stderr

                if full_output.strip():
                    f_raw.write(f"\n{'='*20} {node.upper()} (docker) {'='*20}\n")
                    f_raw.write(full_output)
                    lines = full_output.splitlines()
                    node_impulses = _parse_log_lines(lines, node)
                    impulses.extend(node_impulses)
                    print(f"{len(node_impulses)} entries.")
                else:
                    print("Empty.")
            except Exception as e:
                print(f"Failed ({e}).")

    return impulses


def _harvest_raw_log_files():
    """Read per-node .log files written by the V6 simulator's log_collector threads."""
    impulses = []

    if not RAW_LOG_DIR.exists():
        return impulses

    with open(RAW_LOG_TEXT, "a", encoding="utf-8") as f_raw:
        for node in NODES:
            log_file = RAW_LOG_DIR / f"{node}_raw.log"
            if not log_file.exists():
                continue

            print(f"    [→] Raw log file {log_file.name}...", end=" ", flush=True)
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as fh:
                    lines = fh.readlines()

                f_raw.write(f"\n{'='*20} {node.upper()} (raw_logs/{node}_raw.log) {'='*20}\n")
                f_raw.writelines(lines)

                node_impulses = _parse_log_lines(lines, node)
                impulses.extend(node_impulses)
                print(f"{len(node_impulses)} entries.")
            except Exception as e:
                print(f"Failed ({e}).")

    return impulses


def archive_and_process_logs(start_ts, end_ts):
    """Merge Docker logs + raw_logs/ files into a single structured log DataFrame."""
    print(f"\n[*] Extracting Logs ({start_ts} → {end_ts})...")

    # Clear / initialise output file
    with open(RAW_LOG_TEXT, "w", encoding="utf-8") as f:
        f.write(f"# Raw Log Archive — harvested {datetime.now().isoformat()}\n")
        f.write(f"# Range: {start_ts} → {end_ts}\n\n")

    impulses = []
    impulses.extend(_harvest_docker_logs(start_ts, end_ts))
    impulses.extend(_harvest_raw_log_files())

    if not impulses:
        return pd.DataFrame()

    ldf = pd.DataFrame(impulses)
    ldf['ts'] = pd.to_datetime(ldf['ts'], errors='coerce').dt.tz_localize(None).dt.floor(WINDOW_SIZE)
    ldf = ldf.dropna(subset=['ts'])

    grouped = ldf.groupby(['ts', 'node']).agg(
        log_error=('is_err', 'sum'),
        log_total=('count', 'sum'),
        log_avg_len=('len', 'mean')
    ).unstack(fill_value=0)

    print(f"    [✓] Log aggregation: {len(grouped)} time windows across all nodes.")
    return grouped


# ─────────────────────────────────────────────────────────────────
#  GROUND TRUTH ALIGNMENT
# ─────────────────────────────────────────────────────────────────
def align_ground_truth(metrics_df, start_dt, end_dt):
    """Label rows in metrics_df using injection_log.csv START/STOP pairs."""
    metrics_df['ground_truth'] = 0

    if not os.path.exists(INJECTION_LOG):
        print("[!] No injection_log.csv found — ground_truth will remain all-zero.")
        return metrics_df

    print("[*] Aligning ground truth from injection_log.csv...")
    try:
        i_df = pd.read_csv(INJECTION_LOG)
        i_df['timestamp'] = pd.to_datetime(i_df['timestamp'], errors='coerce') \
                              .dt.floor(WINDOW_SIZE).dt.tz_localize(None)
        i_df = i_df.dropna(subset=['timestamp'])
        i_df = i_df[(i_df['timestamp'] >= start_dt) & (i_df['timestamp'] <= end_dt)]

        starts = i_df[i_df['state'] == 'START']
        stops  = i_df[i_df['state'] == 'STOP']

        labeled = 0
        for _, row in starts.iterrows():
            st = row['timestamp']
            sp = stops[(stops['timestamp'] >= st) & (stops['node'] == row['node'])]
            if not sp.empty:
                et = sp.iloc[0]['timestamp'] + timedelta(seconds=30)
            else:
                # Fallback: use duration_s column if present, else default 120s
                try:
                    fallback_dur = int(float(row['duration_s']))
                except (KeyError, ValueError, TypeError):
                    fallback_dur = 120
                et = st + timedelta(seconds=fallback_dur)

            mask = (metrics_df.index >= st) & (metrics_df.index <= et)
            metrics_df.loc[mask, 'ground_truth'] = 1
            labeled += mask.sum()

        anomaly_pct = metrics_df['ground_truth'].mean() * 100
        print(f"    [✓] Labeled {labeled} anomalous windows ({anomaly_pct:.1f}% of dataset).")

    except Exception as e:
        print(f"    [!] Ground truth alignment failed: {e}")

    return metrics_df


# ─────────────────────────────────────────────────────────────────
#  MAIN HARVEST PIPELINE
# ─────────────────────────────────────────────────────────────────
def harvest():
    print(f"\n{'='*70}")
    print( "      RESEARCH HARVESTER V13.0 — 3-NODE / 4-METRIC SUITE")
    print(f"{'='*70}")

    TARGET_START, TARGET_END = _detect_date_range()

    start_dt = pd.to_datetime(TARGET_START).tz_localize(None)
    end_dt   = pd.to_datetime(TARGET_END).tz_localize(None)

    master_idx = pd.date_range(start_dt, end_dt, freq=WINDOW_SIZE).floor(WINDOW_SIZE).tz_localize(None)
    metrics_df = pd.DataFrame(index=master_idx)

    # ── 1. HARVEST PROMETHEUS METRICS ────────────────────────────
    print(f"\n[*] Querying Prometheus: {TARGET_START}  →  {TARGET_END}")
    print(f"    Expected windows : {len(master_idx):,}")
    print(f"    Metrics to fetch : {len(QUERIES)} (exactly 12 core features)")

    for feat, query in QUERIES.items():
        series = get_metric_series(query, TARGET_START, TARGET_END)
        metrics_df[feat] = series.reindex(master_idx)
        count = metrics_df[feat].count()
        flag  = "✓" if count > 0 else "✗"
        print(f"  [{flag}] {feat:18}: {count:6,} samples")

    prom_total = sum(metrics_df[f].count() for f in ALL_CORE_FEATURES)

    if prom_total == 0:
        print("\n[!] WARNING: All Prometheus metrics returned 0 — running diagnostics...")
        _diagnose_prometheus()
        print("\n    Possible causes:")
        print("    1. cAdvisor not running  →  docker ps | grep cadvisor")
        print("    2. Container names don't match 'app'/'db'/'redis'")
        print("       Run: curl http://localhost:9090/api/v1/label/name/values")
        print("    3. Prometheus not scraping cAdvisor (check prometheus.yml)")
        print("    4. Blackbox exporter missing for app_latency probe")

    # ── 2. HARVEST LOGS (side output only) ───────────────────────
    logs_fused = archive_and_process_logs(TARGET_START, TARGET_END)
    if not logs_fused.empty:
        logs_fused.columns = [f"{n}_{m}" for m, n in logs_fused.columns]
        logs_fused.to_csv(LOGS_ARCHIVE)
        print(f"\n[✓] Log archive → '{LOGS_ARCHIVE}'  ({len(logs_fused):,} rows)")
        # NOTE: log features are NOT merged into metrics_df — they stay as a
        #       separate file. The training CSV contains ONLY the 12 Prometheus metrics.

    # ── 3. GROUND TRUTH ALIGNMENT ─────────────────────────────────
    metrics_df = align_ground_truth(metrics_df, start_dt, end_dt)

    # ── 4. FINAL SAVE — EXACTLY 12 COLUMNS + ground_truth ─────────
    output_cols = ALL_CORE_FEATURES + ['ground_truth']
    output_cols = [c for c in output_cols if c in metrics_df.columns]

    if prom_total == 0:
        # No Prometheus data at all — abort cleanly
        print(f"\n[✗] ABORT: No Prometheus data recovered. Fix the issue above, then re-run.")
        print(f"    Tip: Start the load simulator first, then run this harvester.")
        return

    # Drop windows where ALL 12 core features are NaN
    valid_data = metrics_df[output_cols].dropna(how='all', subset=ALL_CORE_FEATURES)

    if len(valid_data) == 0:
        print(f"\n[✗] ABORT: No valid windows after filtering. All metric values are NaN.")
        return

    # Fill remaining NaN (partial rows) with 0
    filled = valid_data.fillna(0)
    filled.to_csv(OUTPUT_FILE)
    filled.to_csv(METRICS_ARCHIVE)

    # ── 5. REPORT ─────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  HARVEST COMPLETE")
    print(f"  Total windows  : {len(filled):,}")
    print(f"  Time span      : {start_dt}  →  {end_dt}")
    print(f"  Columns in CSV : {len(filled.columns)}  "
          f"({', '.join(ALL_CORE_FEATURES[:3])} ... ground_truth)")
    print(f"  Anomaly windows: {int(filled['ground_truth'].sum()):,}  "
          f"({filled['ground_truth'].mean()*100:.1f}%)")
    print(f"  Healthy windows: {int((filled['ground_truth']==0).sum()):,}")
    print()
    for node in NODES:
        core  = NODE_FEATURES[node]
        avail = [(c, int(filled[c].astype(bool).sum())) for c in core if c in filled.columns]
        print(f"  {node.upper():5} — " + " | ".join(f"{c.split('_',1)[1]}: {n}" for c, n in avail))
    print()
    print(f"  Output  : {OUTPUT_FILE}")
    print(f"  Archive : {METRICS_ARCHIVE}")
    print(f"  Logs    : {LOGS_ARCHIVE}")
    print(f"  Raw log : {RAW_LOG_TEXT}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    harvest()