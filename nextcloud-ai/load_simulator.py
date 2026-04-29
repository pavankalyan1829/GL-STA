
import requests
import threading
import time
import random
import subprocess
import os
import json
import signal
import sys
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────
URL          = "http://localhost:8080"
ADMIN        = ("admin", "adminpro_123")
USERS        = [("user1", "majorpro_123"), ("user2", "majorpro_123"), ("user3", "majorpro_123")]
NODES        = ["app", "db", "redis"]

# Runtime: 12-14 hours (configurable)
RUN_HOURS    = random.uniform(12, 14)
RUN_SECONDS  = int(RUN_HOURS * 3600)

# Output files
LOG_CSV      = "injection_log.csv"
LOG_DIR      = Path("raw_logs")                   # Per-node live log dumps
SUMMARY_JSON = "session_summary.json"

# Traffic intensities
INTENSITY_MAP = {"LOW": 0.3, "MEDIUM": 0.6, "HIGH": 0.9, "CRITICAL": 1.0}

# Fault categories and weights (controls random selection probability)
FAULT_WEIGHTS = {
    "resource":    35,   # CPU / memory stress
    "availability":20,   # pauses / stalls
    "network":     20,   # latency / loss / throttle
    "io":          10,   # disk saturation
    "security":    10,   # brute-force / log storm
    "cascade":      5,   # compound multi-node faults
}

# ─────────────────────────────────────────────────────────────────
#  FORENSIC CHAOS ENGINE
# ─────────────────────────────────────────────────────────────────
class ForensicChaosEngineV6:

    def __init__(self):
        self.active        = True
        self.lock          = threading.Lock()
        self.start_time    = datetime.now()
        self.fault_counter = defaultdict(int)
        self.session_stats = {
            "start_time":   self.start_time.isoformat(),
            "target_hours": RUN_HOURS,
            "faults":       []
        }

        LOG_DIR.mkdir(exist_ok=True)

        # Initialise CSV header
        if not os.path.exists(LOG_CSV):
            with open(LOG_CSV, "w") as f:
                f.write("timestamp,node,fault_type,state,intensity,duration_s,metadata\n")

        # Open per-node raw log file handles
        self._log_handles = {
            node: open(LOG_DIR / f"{node}_raw.log", "a", encoding="utf-8", buffering=1)
            for node in NODES
        }

        self._hard_reset()

    # ── UTILITIES ─────────────────────────────────────────────────

    def _ts(self):
        return datetime.now().isoformat(timespec="seconds")

    def _elapsed(self):
        return (datetime.now() - self.start_time).total_seconds()

    def _remaining(self):
        return max(0, RUN_SECONDS - self._elapsed())

    def log_event(self, node, fault, state, intensity="HIGH", duration=0, meta=""):
        """Write structured event to CSV and echo to console."""
        with self.lock:
            row = f"{self._ts()},{node},{fault},{state},{intensity},{duration},{meta}\n"
            with open(LOG_CSV, "a") as f:
                f.write(row)

        if state == "START":
            self.fault_counter[fault] += 1

    def _write_raw_log(self, node, message):
        """Append a timestamped line to the node's raw log file."""
        try:
            line = f"[{self._ts()}] [{node.upper()}] {message}\n"
            with self.lock:
                self._log_handles[node].write(line)
        except Exception:
            pass

    def _run(self, cmd, node=None):
        """Execute shell command; capture output and optionally write to raw log."""
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if node:
            if result.stdout.strip():
                self._write_raw_log(node, f"STDOUT: {result.stdout.strip()[:500]}")
            if result.stderr.strip():
                self._write_raw_log(node, f"STDERR: {result.stderr.strip()[:500]}")
        return result

    def _hard_reset(self):
        """Clean up all injected faults from previous runs."""
        print("[!] Resetting environment — purging all latent faults...")
        for node in NODES:
            self._run(f"docker unpause {node}")
            self._run(f"docker exec -u root {node} tc qdisc del dev eth0 root")
        self._run("docker exec app pkill -9 stress-ng",     node="app")
        self._run("docker exec app pkill -9 sha512sum",     node="app")
        self._run("docker exec db  pkill -9 stress-ng",     node="db")
        self._run("docker exec redis pkill -9 stress-ng",   node="redis")

        print("[*] Ensuring toolset is installed in all containers...")
        for node in NODES:
            self._run(
                f"docker exec -u root {node} sh -c "
                f"'apt-get update -qq && apt-get install -y -qq stress-ng iproute2 curl'",
                node=node
            )
        print("[✓] Environment ready.\n")

    # ── RAW LOG COLLECTION ────────────────────────────────────────

    def _log_collector(self, node):
        """
        Background thread: continuously tails a container's stdout/stderr
        and writes to the per-node raw log file.
        """
        self._write_raw_log(node, f"=== Log collection started for node={node} ===")
        last_run = None

        while self.active:
            try:
                since = (datetime.utcnow() - timedelta(seconds=15)).strftime("%Y-%m-%dT%H:%M:%SZ")
                cmd = f'docker logs --since "{since}" --timestamps {node} 2>&1'
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                output = (result.stdout + result.stderr).strip()

                if output and output != last_run:
                    for line in output.splitlines():
                        self._write_raw_log(node, line)
                    last_run = output

            except Exception as e:
                self._write_raw_log(node, f"[collector error] {e}")

            time.sleep(10)

        self._write_raw_log(node, f"=== Log collection stopped for node={node} ===")

    # ── USER TRAFFIC PATTERNS ────────────────────────────────────

    def _traffic_normal(self, user, pwd):
        """Steady-state multi-user DAV operations: uploads, downloads, deletes."""
        session = requests.Session()
        session.auth = (user, pwd)
        self._write_raw_log("app", f"Traffic thread started for user={user} [NORMAL]")

        while self.active:
            try:
                ops = random.randint(2, 7)
                for _ in range(ops):
                    fname = f"file_{random.randint(0, 9999)}.dat"
                    size  = random.choice([1024, 64*1024, 512*1024, 2*1024*1024])
                    rpath = f"{URL}/remote.php/dav/files/{user}/{fname}"

                    session.put(rpath, data=os.urandom(size), timeout=10)
                    time.sleep(random.uniform(0.1, 0.5))

                    if random.random() > 0.3:
                        session.get(rpath, timeout=10)

                    if random.random() > 0.6:
                        session.delete(rpath, timeout=10)

                time.sleep(random.uniform(3, 12))

            except Exception as e:
                self._write_raw_log("app", f"[traffic_normal] user={user} err={e}")
                time.sleep(2)

    def _traffic_heavy(self, user, pwd):
        """High-throughput large-file upload/download pattern."""
        session = requests.Session()
        session.auth = (user, pwd)
        self._write_raw_log("app", f"Traffic thread started for user={user} [HEAVY]")

        while self.active:
            try:
                fname = f"heavy_{random.randint(0, 99)}.bin"
                size  = random.choice([5*1024*1024, 10*1024*1024, 20*1024*1024])
                rpath = f"{URL}/remote.php/dav/files/{user}/{fname}"
                session.put(rpath, data=os.urandom(size), timeout=60)
                time.sleep(random.uniform(5, 20))
                session.get(rpath, timeout=60)
                time.sleep(random.uniform(10, 30))
            except Exception as e:
                self._write_raw_log("app", f"[traffic_heavy] user={user} err={e}")
                time.sleep(5)

    def _traffic_burst(self, user, pwd):
        """Intermittent burst pattern — idle then flood."""
        session = requests.Session()
        session.auth = (user, pwd)

        while self.active:
            try:
                idle = random.uniform(30, 120)
                time.sleep(idle)

                burst = random.randint(20, 50)
                self._write_raw_log("app", f"[traffic_burst] user={user} burst={burst}")
                for _ in range(burst):
                    fname = f"burst_{random.randint(0, 999)}.tmp"
                    rpath = f"{URL}/remote.php/dav/files/{user}/{fname}"
                    session.put(rpath, data=os.urandom(32*1024), timeout=10)
                    time.sleep(0.05)
            except Exception as e:
                self._write_raw_log("app", f"[traffic_burst] err={e}")
                time.sleep(3)

    def _traffic_api_poll(self):
        """Simulates a sync client continuously polling Nextcloud OCS/status APIs."""
        session = requests.Session()
        session.auth = ADMIN
        endpoints = [
            f"{URL}/status.php",
            f"{URL}/ocs/v2.php/apps/files_sharing/api/v1/shares",
            f"{URL}/ocs/v1.php/cloud/users",
            f"{URL}/remote.php/dav/",
        ]

        while self.active:
            try:
                ep = random.choice(endpoints)
                session.get(ep, timeout=5)
                time.sleep(random.uniform(5, 20))
            except Exception:
                time.sleep(5)

    def _traffic_db_stress(self):
        """
        Executes repeated MySQL queries to stress the db node's CPU/memory
        and generate realistic db-layer latency signals.
        """
        while self.active:
            try:
                n = random.randint(1000, 50000)
                cmd = (
                    f'docker exec db mysql -u root -proot nextcloud '
                    f'-e "SELECT COUNT(*) FROM oc_filecache WHERE fileid < {n};" 2>&1'
                )
                res = self._run(cmd, node="db")
                self._write_raw_log("db", f"[db_stress] rows_scanned<{n}: {res.stdout.strip()[:80]}")
                time.sleep(random.uniform(2, 8))
            except Exception as e:
                self._write_raw_log("db", f"[db_stress] err={e}")
                time.sleep(5)

    def _traffic_redis_ops(self):
        """Runs rapid GET/SET/DEL commands against the Redis node."""
        while self.active:
            try:
                key = f"simkey:{random.randint(0, 500)}"
                val = random.randbytes(random.choice([64, 512, 4096])).hex()
                self._run(f"docker exec redis redis-cli SET {key} {val[:200]}", node="redis")
                self._run(f"docker exec redis redis-cli GET {key}", node="redis")
                if random.random() > 0.7:
                    self._run(f"docker exec redis redis-cli DEL {key}", node="redis")
                time.sleep(random.uniform(0.5, 3))
            except Exception as e:
                self._write_raw_log("redis", f"[redis_ops] err={e}")
                time.sleep(2)

    # ── FAULT CATEGORY 1: RESOURCE EXHAUSTION ────────────────────

    def fault_cpu_app_high(self, duration):
        """APP: 2-core CPU saturation at 95%."""
        print(f"  🔥 [CPU] APP — Saturation 95% ({duration}s)")
        self.log_event("app", "CPU_SATURATION", "START", "HIGH", duration)
        self._write_raw_log("app", f"FAULT_START: CPU_SATURATION 95% for {duration}s")
        self._run(f"docker exec -d app stress-ng --cpu 2 --cpu-load 95 -t {duration}s", node="app")
        time.sleep(duration)
        self._write_raw_log("app", "FAULT_STOP: CPU_SATURATION")
        self.log_event("app", "CPU_SATURATION", "STOP", duration=duration)

    def fault_cpu_app_medium(self, duration):
        """APP: Elevated CPU at 60% — mimics sustained workload."""
        print(f"  🌡️  [CPU] APP — Medium Load 60% ({duration}s)")
        self.log_event("app", "CPU_MEDIUM_LOAD", "START", "MEDIUM", duration)
        self._write_raw_log("app", f"FAULT_START: CPU_MEDIUM_LOAD 60% for {duration}s")
        self._run(f"docker exec -d app stress-ng --cpu 1 --cpu-load 60 -t {duration}s", node="app")
        time.sleep(duration)
        self.log_event("app", "CPU_MEDIUM_LOAD", "STOP", duration=duration)

    def fault_cpu_gradual_ramp(self, duration):
        """APP: CPU ramps from 30% → 95% over the duration (gradual degradation)."""
        print(f"  📈 [CPU] APP — Gradual Ramp 30→95% ({duration}s)")
        self.log_event("app", "CPU_GRADUAL_RAMP", "START", "ESCALATING", duration)
        self._write_raw_log("app", f"FAULT_START: CPU_GRADUAL_RAMP over {duration}s")

        steps = 6
        step_dur = duration // steps
        for i, load in enumerate(range(30, 100, (70 // steps))):
            self._write_raw_log("app", f"[cpu_ramp] step={i+1} load={load}%")
            self._run(f"docker exec -d app stress-ng --cpu 1 --cpu-load {load} -t {step_dur}s")
            time.sleep(step_dur)

        self.log_event("app", "CPU_GRADUAL_RAMP", "STOP", duration=duration)

    def fault_cpu_db(self, duration):
        """DB: CPU spike to 80% simulating heavy query processing."""
        print(f"  🔥 [CPU] DB — Query Storm 80% ({duration}s)")
        self.log_event("db", "CPU_SPIKE_DB", "START", "HIGH", duration)
        self._write_raw_log("db", f"FAULT_START: CPU_SPIKE_DB 80% for {duration}s")
        self._run(f"docker exec -d db stress-ng --cpu 1 --cpu-load 80 -t {duration}s", node="db")
        time.sleep(duration)
        self.log_event("db", "CPU_SPIKE_DB", "STOP", duration=duration)

    def fault_mem_app_leak(self, duration):
        """APP: Memory leak simulation — 800 MB VM hog."""
        print(f"  💧 [MEM] APP — Memory Leak 800 MB ({duration}s)")
        self.log_event("app", "MEM_LEAK", "START", "LINEAR", duration)
        self._write_raw_log("app", f"FAULT_START: MEM_LEAK 800M for {duration}s")
        self._run(f"docker exec -d app stress-ng --vm 1 --vm-bytes 800M -t {duration}s", node="app")
        time.sleep(duration)
        self._write_raw_log("app", "FAULT_STOP: MEM_LEAK")
        self.log_event("app", "MEM_LEAK", "STOP", duration=duration)

    def fault_mem_db_pressure(self, duration):
        """DB: Memory pressure — 400 MB to compete with buffer pool."""
        print(f"  💧 [MEM] DB — Memory Pressure 400 MB ({duration}s)")
        self.log_event("db", "MEM_PRESSURE_DB", "START", "HIGH", duration)
        self._write_raw_log("db", f"FAULT_START: MEM_PRESSURE_DB 400M for {duration}s")
        self._run(f"docker exec -d db stress-ng --vm 1 --vm-bytes 400M -t {duration}s", node="db")
        time.sleep(duration)
        self.log_event("db", "MEM_PRESSURE_DB", "STOP", duration=duration)

    def fault_mem_redis_full(self, duration):
        """REDIS: Fill Redis with large keys to simulate memory exhaustion."""
        print(f"  💧 [MEM] REDIS — Key Flood ({duration}s)")
        self.log_event("redis", "MEM_REDIS_FLOOD", "START", "HIGH", duration)
        self._write_raw_log("redis", f"FAULT_START: MEM_REDIS_FLOOD for {duration}s")

        end = time.time() + duration
        i = 0
        while time.time() < end and self.active:
            key = f"flood:{i}"
            val = "x" * 4096
            self._run(f"docker exec redis redis-cli SET {key} '{val}' EX 300", node="redis")
            i += 1
            time.sleep(0.05)

        self._write_raw_log("redis", "FAULT_STOP: MEM_REDIS_FLOOD")
        self.log_event("redis", "MEM_REDIS_FLOOD", "STOP", duration=duration)

    # ── FAULT CATEGORY 2: AVAILABILITY STALLS ────────────────────

    def fault_db_stall(self, duration):
        """DB: Pause the entire container (simulates DB crash/OOM kill)."""
        print(f"  🛑 [AVAIL] DB — Container Pause ({duration}s)")
        self.log_event("db", "DB_STALL", "START", "CRITICAL", duration)
        self._write_raw_log("db", f"FAULT_START: DB_STALL for {duration}s")
        self._run("docker pause db")
        time.sleep(duration)
        self._run("docker unpause db")
        self._write_raw_log("db", "FAULT_STOP: DB_STALL — container resumed")
        self.log_event("db", "DB_STALL", "STOP", duration=duration)

    def fault_redis_flicker(self, duration):
        """REDIS: Intermittent pause/resume (cache unavailability jitter)."""
        print(f"  📡 [AVAIL] REDIS — Cache Flicker ({duration}s)")
        self.log_event("redis", "REDIS_FLICKER", "START", "STOCHASTIC", duration)
        self._write_raw_log("redis", f"FAULT_START: REDIS_FLICKER for {duration}s")

        end = time.time() + duration
        while time.time() < end and self.active:
            pause_t = random.uniform(2, 6)
            resume_t = random.uniform(1, 4)
            self._run("docker pause redis")
            self._write_raw_log("redis", f"[flicker] paused for {pause_t:.1f}s")
            time.sleep(pause_t)
            self._run("docker unpause redis")
            self._write_raw_log("redis", f"[flicker] resumed for {resume_t:.1f}s")
            time.sleep(resume_t)

        self.log_event("redis", "REDIS_FLICKER", "STOP", duration=duration)

    def fault_app_brief_restart(self, duration):
        """APP: Simulate a container restart (pause 10s then unpause)."""
        pause_t = min(10, duration)
        print(f"  🔄 [AVAIL] APP — Brief Restart Simulation ({pause_t}s)")
        self.log_event("app", "APP_RESTART_SIM", "START", "CRITICAL", pause_t)
        self._write_raw_log("app", f"FAULT_START: APP_RESTART_SIM for {pause_t}s")
        self._run("docker pause app")
        time.sleep(pause_t)
        self._run("docker unpause app")
        self._write_raw_log("app", "FAULT_STOP: APP_RESTART_SIM — container resumed")
        self.log_event("app", "APP_RESTART_SIM", "STOP", duration=pause_t)
        time.sleep(duration - pause_t)

    # ── FAULT CATEGORY 3: NETWORK DEGRADATION ────────────────────

    def fault_net_latency_db(self, duration):
        """DB: Inject artificial network latency (300–800 ms jitter)."""
        ms = random.randint(300, 800)
        jitter = ms // 5
        print(f"  ⏳ [NET] DB — Latency {ms}ms ±{jitter}ms ({duration}s)")
        self.log_event("db", "NET_LATENCY_DB", "START", f"{ms}ms", duration, f"jitter={jitter}ms")
        self._write_raw_log("db", f"FAULT_START: NET_LATENCY {ms}ms ±{jitter}ms for {duration}s")
        self._run(f"docker exec -u root db tc qdisc add dev eth0 root netem delay {ms}ms {jitter}ms")
        time.sleep(duration)
        self._run("docker exec -u root db tc qdisc del dev eth0 root")
        self._write_raw_log("db", "FAULT_STOP: NET_LATENCY")
        self.log_event("db", "NET_LATENCY_DB", "STOP", duration=duration)

    def fault_net_latency_app(self, duration):
        """APP: High network latency for upstream connections."""
        ms = random.randint(150, 600)
        print(f"  ⏳ [NET] APP — Latency {ms}ms ({duration}s)")
        self.log_event("app", "NET_LATENCY_APP", "START", f"{ms}ms", duration)
        self._write_raw_log("app", f"FAULT_START: NET_LATENCY_APP {ms}ms for {duration}s")
        self._run(f"docker exec -u root app tc qdisc add dev eth0 root netem delay {ms}ms 30ms")
        time.sleep(duration)
        self._run("docker exec -u root app tc qdisc del dev eth0 root")
        self.log_event("app", "NET_LATENCY_APP", "STOP", duration=duration)

    def fault_net_packet_loss(self, duration):
        """APP: Packet loss between 10%–40% to degrade throughput."""
        pct = random.randint(10, 40)
        print(f"  📉 [NET] APP — Packet Loss {pct}% ({duration}s)")
        self.log_event("app", "NET_PACKET_LOSS", "START", f"{pct}%", duration)
        self._write_raw_log("app", f"FAULT_START: NET_PACKET_LOSS {pct}% for {duration}s")
        self._run(f"docker exec -u root app tc qdisc add dev eth0 root netem loss {pct}%")
        time.sleep(duration)
        self._run("docker exec -u root app tc qdisc del dev eth0 root")
        self.log_event("app", "NET_PACKET_LOSS", "STOP", duration=duration)

    def fault_net_bandwidth_throttle(self, duration):
        """DB: Throttle network bandwidth to 1 Mbit/s — simulates WAN congestion."""
        kbps = random.randint(512, 2048)
        print(f"  🐢 [NET] DB — Bandwidth Throttle {kbps} kbps ({duration}s)")
        self.log_event("db", "NET_THROTTLE_DB", "START", f"{kbps}kbps", duration)
        self._write_raw_log("db", f"FAULT_START: NET_THROTTLE {kbps}kbps for {duration}s")
        self._run(
            f"docker exec -u root db tc qdisc add dev eth0 root tbf "
            f"rate {kbps}kbit burst 32kbit latency 400ms"
        )
        time.sleep(duration)
        self._run("docker exec -u root db tc qdisc del dev eth0 root")
        self.log_event("db", "NET_THROTTLE_DB", "STOP", duration=duration)

    def fault_net_redis_latency(self, duration):
        """REDIS: Network latency to simulate cache miss cascade."""
        ms = random.randint(50, 250)
        print(f"  ⏳ [NET] REDIS — Cache Latency {ms}ms ({duration}s)")
        self.log_event("redis", "NET_LATENCY_REDIS", "START", f"{ms}ms", duration)
        self._write_raw_log("redis", f"FAULT_START: NET_LATENCY_REDIS {ms}ms for {duration}s")
        self._run(f"docker exec -u root redis tc qdisc add dev eth0 root netem delay {ms}ms 20ms")
        time.sleep(duration)
        self._run("docker exec -u root redis tc qdisc del dev eth0 root")
        self.log_event("redis", "NET_LATENCY_REDIS", "STOP", duration=duration)

    # ── FAULT CATEGORY 4: DISK I/O ────────────────────────────────

    def fault_io_saturation(self, duration):
        """APP: Max I/O saturation with multiple workers."""
        print(f"  💾 [IO] APP — Disk I/O Saturation ({duration}s)")
        self.log_event("app", "IO_SATURATION", "START", "HIGH", duration)
        self._write_raw_log("app", f"FAULT_START: IO_SATURATION for {duration}s")
        self._run(f"docker exec -d app stress-ng --io 4 --hdd 2 --hdd-opts direct -t {duration}s", node="app")
        time.sleep(duration)
        self.log_event("app", "IO_SATURATION", "STOP", duration=duration)

    def fault_io_db_writes(self, duration):
        """DB: Simulate heavy write I/O against the database data directory."""
        print(f"  💾 [IO] DB — Heavy Write I/O ({duration}s)")
        self.log_event("db", "IO_DB_WRITES", "START", "HIGH", duration)
        self._write_raw_log("db", f"FAULT_START: IO_DB_WRITES for {duration}s")
        self._run(f"docker exec -d db stress-ng --io 2 --hdd 1 -t {duration}s", node="db")
        time.sleep(duration)
        self.log_event("db", "IO_DB_WRITES", "STOP", duration=duration)

    # ── FAULT CATEGORY 5: SECURITY / LOG ANOMALIES ───────────────

    def fault_auth_brute_force(self, duration):
        """Brute-force login attempts → auth error log storm on APP node."""
        print(f"  🔐 [SEC] APP — Auth Brute Force ({duration}s)")
        self.log_event("app", "AUTH_BRUTE_FORCE", "START", "HIGH", duration)
        self._write_raw_log("app", f"FAULT_START: AUTH_BRUTE_FORCE for {duration}s")

        end = time.time() + duration
        attempt = 0
        while time.time() < end and self.active:
            try:
                user = random.choice(["root", "admin", "nextcloud", "user1", "test"])
                pwd  = "".join(random.choices("abcdefghijklmnop0123456789", k=8))
                requests.get(f"{URL}/login", auth=(user, pwd), timeout=2)
                attempt += 1
            except Exception:
                pass
            time.sleep(random.uniform(0.05, 0.2))

        self._write_raw_log("app", f"FAULT_STOP: AUTH_BRUTE_FORCE total_attempts={attempt}")
        self.log_event("app", "AUTH_BRUTE_FORCE", "STOP", duration=duration, meta=f"attempts={attempt}")

    def fault_log_flood(self, duration):
        """APP: Synthetic log flooding via rapid HTTP error generation."""
        print(f"  ⛈️  [LOG] APP — HTTP 404/500 Log Flood ({duration}s)")
        self.log_event("app", "LOG_FLOOD", "START", "HIGH", duration)
        self._write_raw_log("app", f"FAULT_START: LOG_FLOOD for {duration}s")

        end = time.time() + duration
        while time.time() < end and self.active:
            try:
                bad_path = f"/nonexistent_{random.randint(0, 9999)}"
                requests.get(f"{URL}{bad_path}", timeout=2)
            except Exception:
                pass
            time.sleep(0.08)

        self.log_event("app", "LOG_FLOOD", "STOP", duration=duration)

    # ── FAULT CATEGORY 6: CASCADE / COMPOUND FAULTS ──────────────

    def fault_cascade_cpu_net(self, duration):
        """
        APP+DB compound fault: CPU stress on APP while DB network is degraded.
        Models a realistic overload scenario during a spike.
        """
        print(f"  ☄️  [CASCADE] APP CPU + DB NET Degradation ({duration}s)")
        self.log_event("app", "CASCADE_CPU_NET", "START", "CRITICAL", duration, "compound=APP_CPU+DB_NET")
        self._write_raw_log("app", f"FAULT_START: CASCADE_CPU_NET for {duration}s")
        self._write_raw_log("db",  f"FAULT_START: CASCADE_CPU_NET for {duration}s")

        ms = random.randint(200, 500)
        self._run(f"docker exec -d app stress-ng --cpu 2 --cpu-load 85 -t {duration}s")
        self._run(f"docker exec -u root db tc qdisc add dev eth0 root netem delay {ms}ms 40ms")
        time.sleep(duration)
        self._run("docker exec -u root db tc qdisc del dev eth0 root")
        self.log_event("app", "CASCADE_CPU_NET", "STOP", duration=duration)

    def fault_cascade_mem_cache(self, duration):
        """
        APP memory pressure while Redis cache flickers — tests graceful degradation.
        """
        print(f"  ☄️  [CASCADE] APP MEM Pressure + REDIS Flicker ({duration}s)")
        self.log_event("app", "CASCADE_MEM_CACHE", "START", "CRITICAL", duration, "compound=APP_MEM+REDIS")
        self._write_raw_log("app",   f"FAULT_START: CASCADE_MEM_CACHE for {duration}s")
        self._write_raw_log("redis", f"FAULT_START: CASCADE_MEM_CACHE for {duration}s")

        half = duration // 2
        self._run(f"docker exec -d app stress-ng --vm 1 --vm-bytes 600M -t {duration}s")

        end = time.time() + half
        while time.time() < end and self.active:
            self._run("docker pause redis")
            time.sleep(random.uniform(3, 8))
            self._run("docker unpause redis")
            time.sleep(random.uniform(2, 5))

        time.sleep(duration - half)
        self.log_event("app", "CASCADE_MEM_CACHE", "STOP", duration=duration)

    def fault_cascade_three_node(self, duration):
        """
        All three nodes simultaneously degraded — worst-case scenario.
        APP: CPU spike; DB: Stall; REDIS: Network latency.
        """
        stall_t  = min(30, duration // 3)
        cpu_t    = duration
        redis_ms = random.randint(100, 300)

        print(f"  🌋 [CASCADE] THREE-NODE COMPOUND FAULT ({duration}s)")
        self.log_event("app",   "CASCADE_3NODE", "START", "CRITICAL", duration, "phase=all_nodes")
        self.log_event("db",    "CASCADE_3NODE", "START", "CRITICAL", duration)
        self.log_event("redis", "CASCADE_3NODE", "START", "CRITICAL", duration)
        for n in NODES:
            self._write_raw_log(n, f"FAULT_START: CASCADE_3NODE for {duration}s")

        # APP: CPU stress for full duration
        self._run(f"docker exec -d app stress-ng --cpu 2 --cpu-load 90 -t {cpu_t}s")
        # REDIS: network latency for full duration
        self._run(f"docker exec -u root redis tc qdisc add dev eth0 root netem delay {redis_ms}ms 20ms")
        # DB: brief stall then throttled
        self._run("docker pause db")
        time.sleep(stall_t)
        self._run("docker unpause db")
        self._run("docker exec -u root db tc qdisc add dev eth0 root netem delay 200ms 50ms")

        time.sleep(duration - stall_t)
        self._run("docker exec -u root db tc qdisc del dev eth0 root")
        self._run("docker exec -u root redis tc qdisc del dev eth0 root")
        for n in NODES:
            self.log_event(n, "CASCADE_3NODE", "STOP", duration=duration)

    # ── FAULT SELECTOR ────────────────────────────────────────────

    def _weighted_random_fault(self):
        """
        Selects a fault using weighted categories.
        Returns a (function, base_duration) tuple.
        """
        # Category → available faults
        catalogue = {
            "resource": [
                (self.fault_cpu_app_high,     (90,  240)),
                (self.fault_cpu_app_medium,   (120, 360)),
                (self.fault_cpu_gradual_ramp, (180, 360)),
                (self.fault_cpu_db,           (90,  180)),
                (self.fault_mem_app_leak,     (120, 300)),
                (self.fault_mem_db_pressure,  (90,  240)),
                (self.fault_mem_redis_full,   (60,  180)),
            ],
            "availability": [
                (self.fault_db_stall,          (15, 60)),
                (self.fault_redis_flicker,     (60, 180)),
                (self.fault_app_brief_restart, (30, 90)),
            ],
            "network": [
                (self.fault_net_latency_db,        (90,  240)),
                (self.fault_net_latency_app,       (90,  180)),
                (self.fault_net_packet_loss,       (60,  150)),
                (self.fault_net_bandwidth_throttle,(90,  240)),
                (self.fault_net_redis_latency,     (60,  120)),
            ],
            "io": [
                (self.fault_io_saturation,  (90, 240)),
                (self.fault_io_db_writes,   (90, 180)),
            ],
            "security": [
                (self.fault_auth_brute_force, (60, 180)),
                (self.fault_log_flood,        (60, 120)),
            ],
            "cascade": [
                (self.fault_cascade_cpu_net,    (120, 300)),
                (self.fault_cascade_mem_cache,  (120, 300)),
                (self.fault_cascade_three_node, (120, 240)),
            ],
        }

        population = []
        weights    = []
        for cat, w in FAULT_WEIGHTS.items():
            for fn, dur_range in catalogue[cat]:
                population.append((fn, dur_range, cat))
                weights.append(w / len(catalogue[cat]))

        chosen_fn, dur_range, cat = random.choices(population, weights=weights, k=1)[0]
        duration = random.randint(*dur_range)
        return chosen_fn, duration, cat

    # ── SCHEDULER PHASES ──────────────────────────────────────────

    def _healthy_baseline_phase(self, seconds):
        """Sleep for a baseline window with status prints every 2 minutes."""
        print(f"\n  [✓] NOMINAL — Healthy baseline for {seconds//60:.0f} min...")
        end = time.time() + seconds
        while time.time() < end and self.active:
            remaining_h = self._remaining() / 3600
            elapsed_h   = self._elapsed() / 3600
            print(f"      ⟳ Elapsed: {elapsed_h:.2f}h | Remaining: {remaining_h:.2f}h | "
                  f"Faults injected: {sum(self.fault_counter.values())}")
            time.sleep(min(120, max(0, end - time.time())))

    def _recovery_phase(self, seconds):
        """Post-fault cooldown — let metrics return to baseline."""
        print(f"  [⟳] RECOVERY — Settling for {seconds}s...")
        time.sleep(seconds)

    # ── MAIN ORCHESTRATOR ─────────────────────────────────────────

    def run(self):
        banner = (
            f"\n{'='*70}\n"
            f"  FORENSIC CHAOS ENGINE V6.0 — 12-14H RESEARCH SUITE\n"
            f"  Start   : {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"  Runtime : {RUN_HOURS:.2f} hours ({RUN_SECONDS} seconds)\n"
            f"  Nodes   : {', '.join(NODES)}\n"
            f"  Faults  : 18 types across 6 categories\n"
            f"  Log Dir : {LOG_DIR.resolve()}\n"
            f"{'='*70}\n"
        )
        print(banner)

        # ── START BACKGROUND THREADS ──────────────────────────────

        # Raw log collectors for each node
        for node in NODES:
            threading.Thread(target=self._log_collector, args=(node,), daemon=True).start()

        # User traffic threads (diverse patterns)
        for u, p in USERS:
            pattern = random.choice([self._traffic_normal, self._traffic_heavy, self._traffic_burst])
            threading.Thread(target=pattern, args=(u, p), daemon=True).start()

        # API poller (sync client simulation)
        threading.Thread(target=self._traffic_api_poll, daemon=True).start()

        # DB-level query stress
        threading.Thread(target=self._traffic_db_stress, daemon=True).start()

        # Redis-level ops stress
        threading.Thread(target=self._traffic_redis_ops, daemon=True).start()

        # ── MAIN FAULT INJECTION LOOP ─────────────────────────────

        try:
            phase_num = 0
            while self._remaining() > 60:
                phase_num += 1
                elapsed_pct = self._elapsed() / RUN_SECONDS

                # Adaptive baseline duration:
                #  - Early hours: longer stable windows (richer healthy data)
                #  - Mid session: balanced
                #  - Late session: shorter baselines (more anomaly density)
                if elapsed_pct < 0.3:
                    baseline_secs = random.randint(480, 720)   # 8-12 min
                elif elapsed_pct < 0.7:
                    baseline_secs = random.randint(360, 600)   # 6-10 min
                else:
                    baseline_secs = random.randint(240, 420)   # 4-7 min

                baseline_secs = min(baseline_secs, int(self._remaining() * 0.5))
                if baseline_secs > 30:
                    self._healthy_baseline_phase(baseline_secs)

                if not self.active or self._remaining() < 120:
                    break

                # Occasionally chain 2 faults back-to-back (compound stress)
                n_faults = random.choices([1, 2, 3], weights=[60, 30, 10])[0]
                for _ in range(n_faults):
                    if self._remaining() < 90:
                        break

                    fn, duration, category = self._weighted_random_fault()
                    # Cap duration so we don't exceed run window
                    duration = min(duration, max(60, int(self._remaining() - 120)))

                    print(f"\n  ── Phase {phase_num} │ Category: {category.upper()} "
                          f"│ Duration: {duration}s ──")

                    self.session_stats["faults"].append({
                        "phase":    phase_num,
                        "category": category,
                        "function": fn.__name__,
                        "duration": duration,
                        "elapsed":  f"{self._elapsed()/3600:.2f}h"
                    })

                    fn(duration)

                # Recovery window after fault(s)
                recovery = random.randint(90, 240)
                recovery = min(recovery, max(30, int(self._remaining() - 60)))
                self._recovery_phase(recovery)

        except KeyboardInterrupt:
            print("\n\n[!] KeyboardInterrupt received — shutting down cleanly...")

        finally:
            self._shutdown()

    def _shutdown(self):
        print("\n[!] Simulator shutting down...")
        self.active = False
        self._hard_reset()

        # Close raw log file handles
        for node, fh in self._log_handles.items():
            try:
                self._write_raw_log(node, "=== Simulator shutdown ===")
                fh.close()
            except Exception:
                pass

        # Write session summary JSON
        self.session_stats["end_time"]       = datetime.now().isoformat()
        self.session_stats["elapsed_hours"]  = round(self._elapsed() / 3600, 3)
        self.session_stats["fault_counts"]   = dict(self.fault_counter)
        self.session_stats["total_faults"]   = sum(self.fault_counter.values())

        with open(SUMMARY_JSON, "w") as f:
            json.dump(self.session_stats, f, indent=2)

        print(f"\n{'='*70}")
        print(f"  SESSION COMPLETE")
        print(f"  Runtime : {self._elapsed()/3600:.2f} hours")
        print(f"  Faults  : {sum(self.fault_counter.values())} injected")
        print(f"  Logs    : {LOG_DIR.resolve()}/")
        print(f"  Summary : {SUMMARY_JSON}")
        print(f"  CSV     : {LOG_CSV}")
        print(f"{'='*70}\n")


# ─────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = ForensicChaosEngineV6()

    # Handle SIGTERM gracefully (e.g. docker stop, task scheduler)
    def _sigterm_handler(sig, frame):
        engine.active = False
        engine._shutdown()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sigterm_handler)

    engine.run()