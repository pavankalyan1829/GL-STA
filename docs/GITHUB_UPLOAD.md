# GitHub Upload Guide

This project is prepared for GitHub with:

- `.gitignore` to exclude generated datasets, trained models, local Docker volumes, caches, and logs.
- `README.md` with setup, workflow, and module documentation.
- `requirements.txt` for Python dependencies.

## Upload With Git

From the project root:

```powershell
git init
git add README.md requirements.txt .gitignore docs nextcloud-ai/docker-compose.yml nextcloud-ai/prometheus.yml nextcloud-ai/loki-config.yml nextcloud-ai/promtail-config.yml nextcloud-ai/data_harvest.py nextcloud-ai/load_simulator.py nextcloud-ai/diagnose_labels.py nextcloud-ai/debug_inference.py *.py
git commit -m "Initial GL-STA project release"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/GL-STA.git
git push -u origin main
```

If the repository already exists locally, skip `git init`. If the remote already exists, use:

```powershell
git remote set-url origin https://github.com/YOUR-USERNAME/GL-STA.git
```

## Create The GitHub Repository

1. Open GitHub and create a new repository named `GL-STA`.
2. Do not initialize it with a README, `.gitignore`, or license because this project already has local files.
3. Copy the repository URL and use it as the `origin` remote.

## Large Artifacts

Do not commit trained models, CSV datasets, `.npz` evaluation bundles, Docker database folders, or Nextcloud runtime files. Add those as GitHub Release assets or store them in an external artifact location.
