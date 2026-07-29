# DermaLens Guide

This guide is for someone reviewing DermaLens for the first time. It explains how to run the project, what to inspect, and how to interpret the machine-learning results without reading every script first.

## 1. What This Project Is

DermaLens is a local-first facial skin screening prototype. It accepts a face photo, strips metadata, runs local ONNX inference, and returns cautious screening-style findings.

It is best understood as an applied ML portfolio project, not as a medical product. The important work is the full system: privacy handling, local deployment, dataset preparation, model comparison, holdout testing, and honest documentation of limitations.

## 2. Quick Demo

Requirements:

- Docker Desktop
- Git

Run:

```powershell
docker compose up --build
```

Open:

```text
http://127.0.0.1:8765
```

Try the app with a clear, front-facing image. The app does not save uploads by default. It re-encodes the image before inference, which strips EXIF/GPS metadata.

Health check:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8765/health
```

Expected response:

```json
{"status":"ok","privacy":{"save_uploads":false,"offline_hf":true}}
```

## 3. Quick Test Path

Run the automated tests:

```powershell
python -m pytest -q
```

What the tests cover:

- API upload rejection for non-images, corrupt images, and large files
- EXIF stripping
- safe storage-name handling
- detector-backed facial-region bounds
- model probability helpers
- grouped split leakage checks

## 4. Suggested Reading Order

For a 10-minute review:

1. [README.md](README.md): project overview and run instructions.
2. [PORTFOLIO_WRITEUP.md](PORTFOLIO_WRITEUP.md): the employer-facing story.
3. [MODEL_CARD.md](MODEL_CARD.md): model scope, intended use, and limitations.
4. [DATA_CARD.md](DATA_CARD.md): data policy, split protocol, and data risks.
5. [BENCHMARKS.md](BENCHMARKS.md): which benchmarks were run, which are diagnostics, and which are external-validity candidates.

For a deeper technical review:

1. [FINAL_TECHNICAL_EXPERIMENTS.md](FINAL_TECHNICAL_EXPERIMENTS.md): experiment history and why the 80% result was rejected.
2. [scripts/prepare_imagefolder.py](scripts/prepare_imagefolder.py): grouped case-level split and leakage audit.
3. [api/main.py](api/main.py): FastAPI entry point and upload hardening.
4. [api/pipeline.py](api/pipeline.py): image loading, metadata stripping, and analysis orchestration.
5. [api/model_adapter.py](api/model_adapter.py): ONNX inference and optional prior correction.

## 5. How To Interpret The Results

The deployed model is intentionally modest:

- MobileNetV3-Small ONNX
- local CPU-friendly inference
- conservative prior calibration
- fair grouped retrain baseline: 44.8% +/- 9.9 accuracy and 29.1% +/- 3.9 macro recall across 12 matched seeds

The earlier combined validation split reported 69.4% accuracy and 48.4% macro recall, but that number is kept mainly as experiment history because the preparation path had image-level leakage risk. The later grouped SCIN result fixed fold-level case overlap, but a second audit found that it is still not a clean model holdout because the fixed deployed ONNX model had already been trained on SCIN-derived head/neck images.

The repo includes the fair-baseline runner:

```powershell
python scripts/run_grouped_mobilenet_baseline.py `
  --manifest data/raw/scin/face_skin_manifest.csv `
  --image-root data/raw/scin `
  --summary-output models/grouped_scin_mobilenet_retrained_baseline_12seed_metrics.json `
  --seeds 42 7 13 21 84 101 202 404 707 808 909 1001 `
  --epochs 8
```

The completed 12-seed run reached 44.8% +/- 9.9 accuracy and 29.1% +/- 3.9 macro recall. This lower result is the fair baseline; the old 86.2% fixed-model number stays in the repo only as a contaminated diagnostic.

The repo also includes a subgroup workflow by Fitzpatrick and Monk tone metadata in `models/grouped_scin_subgroup_metrics.json`. Read it as an example of fairness-aware evaluation mechanics, not proof that the model is fair; the darkest Monk bucket is too small, and SCIN's available tone labels are retrospective image estimates rather than controlled clinical subgroup labels.

There is also one grouped-split decoupled-head artifact: `models/grouped_scin_decoupled_logit_head_metrics.json`. It freezes the deployed ONNX image model and retrains only a class-balanced head over its logits, with C selected on a nested grouped calibration split. The refreshed head reached 75.3% +/- 1.7 accuracy and 70.7% +/- 11.4 macro recall. Read it as a fixed-encoder operating-point experiment, not as a clean model benchmark, because the frozen encoder may have seen SCIN-derived cases during original fine-tuning.

The high-leverage foundation-model path is implemented in `scripts/evaluate_derm_foundation_embeddings.py`. It extracts `google/derm-foundation` embeddings, trains a class-balanced linear probe, selects C on nested calibration data, and evaluates on the held-out grouped fold. The completed 12-seed result is 69.8% +/- 5.2 accuracy and 34.7% +/- 5.0 macro recall. Compared with the fair MobileNet baseline, Derm Foundation has a robust paired accuracy lift and a smaller positive macro-recall lift after expanding beyond the fragile five-seed test. Both learned models beat the majority-class floor on macro recall; neither solves the low-support tail classes.

The research pipeline found stronger experimental results:

- ConvNeXt frozen embeddings improved macro recall.
- Neural heads and ensembles reached higher original-validation performance.
- A validation-tuned class-bias ensemble reached 81.4%, but fresh holdout testing did not reproduce it.

The project therefore does not claim a validated 80% model. The honest conclusion is that the fair baseline is now known, Derm Foundation gives the strongest clean representation lift, and the next evidence-producing step is better data: cleaner labels, more face-specific examples, and enough tail support for meaningful per-class recall.

## 6. What To Look For In The Code

Privacy and safety:

- `api/main.py` limits upload size and rejects invalid uploads.
- `api/pipeline.py` guards image decoding and strips metadata through re-encoding.
- `docker-compose.yml` binds the app to `127.0.0.1` and disables upload retention by default.

Model serving:

- `api/model_adapter.py` loads the ONNX classifier and label map.
- `models/skin_classifier.onnx` is the deployable model.
- `models/label_map.json` defines the class labels and problem type.

Data validity:

- `scripts/prepare_imagefolder.py` uses grouped splitting by `case_id`.
- Every prepared dataset can emit `split_audit.json`.
- `models/label_mapping_rules_v2.json` documents the heuristic label mapping.

## 7. Responsible Caveats

DermaLens is not a diagnostic tool. It is not clinically validated. Its labels are broad and overlap visually, especially acne-like texture, folliculitis-like bumps, and dermatitis-like irritation. It should be evaluated as a portfolio system that demonstrates applied ML judgment, not as a medical-grade dermatology classifier.

## 8. Good Interview Discussion Points

- Why macro recall matters more than raw accuracy for imbalanced screening labels.
- Why image-level splits can leak in medical datasets with multiple photos per case.
- Why the 81.4% result and the later 86.2% fixed-model diagnostic were rejected as final headlines.
- Why low validation support can make tail-class macro recall effectively undefined.
- Why local inference changes model-size and deployment choices.
- Why more data can hurt if it comes from the wrong distribution.
