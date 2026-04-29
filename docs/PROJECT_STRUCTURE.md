# Project Structure

This repository keeps the research pipeline and the local Nextcloud testbed together while excluding generated runtime state.

## Testbed And Fault Injection

- `nextcloud-ai/docker-compose.yml` starts the Nextcloud, MariaDB, Redis, Prometheus, Loki, Promtail, Grafana, and exporter stack.
- `nextcloud-ai/prometheus.yml` defines metric scrape targets.
- `nextcloud-ai/loki-config.yml` configures local Loki storage.
- `nextcloud-ai/promtail-config.yml` ships Docker container logs to Loki.
- `nextcloud-ai/load_simulator.py` generates user activity and injects resource, network, availability, I/O, security, and cascade faults.
- `nextcloud-ai/reprovision_users.ps1` helps recreate local Nextcloud test users.

## Collection

- `nextcloud-ai/data_harvest.py` queries Prometheus, reads log streams/raw logs, and writes harvested datasets for preprocessing.
- `nextcloud-ai/diagnose_labels.py` and `nextcloud-ai/debug_inference.py` are diagnostic helpers for Prometheus labels and inference behavior.

## Preprocessing

- `data_preprocessing.py` combines metrics, logs, and injection labels into healthy training data and mixed test data.
- `summarize_data.py`, `check_means.py`, and `ckeck_stability.py` provide quick sanity checks over generated datasets and model behavior.

## Model Training

- `preprocess_train.py` defines the custom graph attention layer, prepares sliding windows, trains the proposed GAT-LSTM autoencoder, trains recurrent baselines, computes thresholds, and writes evaluation artifacts.

## Live Inference And XRCA

- `live_inference.py` polls Prometheus, maintains recent windows, scores anomalies with the trained model, smooths decisions, and reports root-cause indicators by service and feature group.

## Evaluation

- `research_evaluation.py` reads saved evaluation artifacts and produces threshold sensitivity, baseline comparison, and fault-type metrics.
- `test_inference_logic.py` contains focused inference logic checks.

## Ignored Runtime State

The following are intentionally excluded from Git:

- Docker database and Nextcloud runtime directories.
- Harvested logs and metrics.
- Generated train/test CSV files.
- Trained model files.
- Pickled scalers/vectorizers.
- Plot and evaluation artifact outputs.

This keeps the GitHub repository small, reviewable, and source-focused.
