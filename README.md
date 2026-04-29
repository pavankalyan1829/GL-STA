# GL-STA

GL-STA is a graph-aware spatio-temporal anomaly detection pipeline for a three-node Nextcloud testbed. The project combines Prometheus metrics, Loki/container logs, sliding-window feature engineering, a GAT-LSTM autoencoder, baseline recurrent autoencoders, and explainable root cause analysis utilities.

## Repository Contents

The current codebase is organized around these components:

- `nextcloud-ai/` - Docker Compose testbed, Prometheus/Loki/Grafana configuration, data harvesting scripts, and fault injection workload simulator.
- `data_preprocessing.py` - Fuses harvested metrics and raw logs, labels windows from the injection log, and writes train/test CSV files.
- `preprocess_train.py` - Builds sliding windows, trains/evaluates RNN, GRU, LSTM, and GAT-LSTM autoencoders, and saves evaluation artifacts.
- `live_inference.py` - Polls Prometheus in real time, builds online windows, loads the trained model/scalers, and reports anomalies with root-cause signals.
- `research_evaluation.py` - Produces threshold sweeps, baseline comparisons, and research metrics from saved evaluation artifacts.
- `test_inference_logic.py`, `check_means.py`, `ckeck_stability.py`, `summarize_data.py`, `nextcloud-ai/diagnose_labels.py`, and `nextcloud-ai/debug_inference.py` - Utility and diagnostic scripts.

Generated datasets, trained model files, local Docker volumes, and harvested runtime logs are ignored by Git. Store large artifacts separately through GitHub Releases, cloud storage, or an artifact registry if they are needed for reproducibility.

## Testbed

The Docker stack in `nextcloud-ai/docker-compose.yml` launches:

- Nextcloud application container
- MariaDB database
- Redis cache
- Prometheus
- cAdvisor
- node-exporter
- mysqld-exporter
- redis-exporter
- blackbox-exporter
- Loki
- Promtail
- Grafana

Start the stack from the testbed directory:

```powershell
cd nextcloud-ai
docker compose up -d
```

Useful local endpoints:

- Nextcloud: `http://localhost:8080`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`
- Loki: `http://localhost:3100`

The sample compose file contains development credentials for a local experiment. Change them before using the stack outside a controlled lab.

## Installation

Create a Python environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Docker Desktop must be running before starting the testbed.

## Workflow

1. Start the Nextcloud observability stack.

```powershell
cd nextcloud-ai
docker compose up -d
```

2. Generate workload and inject faults.

```powershell
python load_simulator.py
```

3. Harvest metrics and logs.

```powershell
python data_harvest.py
```

4. Build the train/test datasets from the project root.

```powershell
cd ..
python data_preprocessing.py
```

5. Train and evaluate the anomaly detection models.

```powershell
python preprocess_train.py
```

6. Run research evaluation summaries and plots.

```powershell
python research_evaluation.py
```

7. Run live inference against the active Prometheus endpoint.

```powershell
python live_inference.py
```

## Outputs

The pipeline can generate the following local artifacts:

- `train_healthy_scaled(1).csv`
- `test_mixed_scaled(1).csv`
- `gat_lstm_autoencoder.keras`
- `rnn_autoencoder.keras`
- `gru_autoencoder.keras`
- `lstm_autoencoder.keras`
- `scaler*.pkl`
- `features.pkl`
- `eval_artifacts.npz`
- `model_thresholds.json`
- `anomaly_stats.json`
- `model_comparison.png`

These files are ignored by Git because they are generated or large. Include them in a release bundle only when required.

## Suggested Paper Module Mapping

If your paper describes a modular repository layout, this implementation maps as follows:

- `testbed/` -> `nextcloud-ai/docker-compose.yml`, `prometheus.yml`, `loki-config.yml`, `promtail-config.yml`, `load_simulator.py`
- `collection/` -> `nextcloud-ai/data_harvest.py`
- `preprocessing/` -> `data_preprocessing.py`
- `model/` -> `preprocess_train.py`
- `xrca/` -> root-cause logic in `live_inference.py`
- `evaluation/` -> `research_evaluation.py` and utility scripts

## Citation

If this repository accompanies a paper, add the BibTeX citation here after publication.
