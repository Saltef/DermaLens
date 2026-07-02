# GitHub Release Checklist

## Commit These

- `api/`
- `frontend/`
- `scripts/`
- `models/skin_classifier.onnx`
- `models/label_map.json`
- `models/prior_profiles.json`
- `models/training_prior_combined.json`
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`
- `.env.example`
- `README.md`
- `DATASETS.md`
- `PORTFOLIO_WRITEUP.md`
- `FINAL_TECHNICAL_EXPERIMENTS.md`
- `DISCLAIMER.md`
- `LICENSE`

## Do Not Commit These

- `data/raw/`
- `data/processed/`
- `private-data/`
- `models/experiments/`
- uploaded user photos
- downloaded third-party dataset archives

## Suggested Repository Description

DermaLens Local: privacy-preserving facial skin screening with FastAPI, ONNX Runtime, Docker, EXIF stripping, and critical ML evaluation.

## Test Commands Before Push

```powershell
python -m py_compile api\*.py scripts\*.py
docker compose config
docker compose up --build
```

Open:

```text
http://127.0.0.1:8765
```

Health check:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8765/health
```

## Suggested README Badge Text

```text
Local-first | Docker | FastAPI | ONNX Runtime | No upload retention by default
```

## Portfolio Caveat

Do not describe the project as a diagnostic app. Use:

```text
screening-style visual observations
```

Avoid:

```text
diagnoses skin disease
```

Preferred portfolio framing:

```text
An applied ML case study showing local inference, privacy-first image handling, model comparison, calibration, holdout validation, leakage-aware data splitting, and data-quality limitation analysis.
```
