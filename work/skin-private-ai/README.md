# DermaLens

Private facial skin screening on local hardware. DermaLens is a portfolio ML project that explores how far a privacy-preserving dermatology-vision workflow can go with small, noisy, imbalanced public datasets.

This is not a diagnostic medical device. Outputs are screening-style observations and should be reviewed by a qualified clinician for medical decisions.

## Portfolio Snapshot

- **Stack:** FastAPI, ONNX Runtime, Docker, static HTML/CSS/JS.
- **Privacy:** EXIF stripping, localhost binding, no upload retention by default.
- **Deployable model:** MobileNetV3-Small ONNX classifier with optional prior calibration.
- **Research models:** ConvNeXt frozen embeddings, neural classifier heads, long-tail supervised contrastive tests, targeted augmentation, probability ensembles, and calibration sweeps.
- **Best deployable ONNX result:** 69.4% accuracy and 48.4% macro recall after conservative calibration.
- **Best untuned experimental validation result:** 79.2% accuracy and 71.0% macro recall with a mixed ConvNeXt ensemble.
- **Critical limitation:** fresh holdout testing did not confirm the validation-tuned 81.4% result. A later review also identified image-level split leakage risk in multi-photo cases, so future headline metrics should use the grouped split protocol now built into `scripts/prepare_imagefolder.py`.
- **Write-up:** See [PORTFOLIO_WRITEUP.md](PORTFOLIO_WRITEUP.md).

## What The App Does

- Runs a local web UI on `127.0.0.1`.
- Accepts a face photo upload.
- Strips EXIF/GPS metadata by re-encoding the image.
- Processes the image in memory by default.
- Computes basic image quality and facial skin-region signals.
- Returns cautious, non-diagnostic findings.
- Provides a clean model adapter path for ONNX or PyTorch classifiers.

## Architecture

```text
Browser UI
  -> FastAPI API
    -> privacy layer
       - file type validation
       - EXIF stripping
       - no retention by default
    -> preprocessing
       - resize
       - simple facial-region crop
       - quality checks
    -> classifier adapter
       - ONNX deployed model
       - heuristic fallback if model files are missing
    -> response
       - possible findings
       - confidence
       - limitations
       - clinician-review flags
```

## Local Run With Docker

```powershell
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

## Privacy Defaults

- API binds only to `127.0.0.1`.
- No photos are saved unless `SAVE_UPLOADS=true`.
- EXIF metadata is stripped before inference.
- Inference can run with no external network after dependencies/model weights are available.
- Telemetry-related environment variables are disabled in `docker-compose.yml`.

## Optional Prior Calibration

The ONNX runtime can apply log-prior correction after inference:

```text
adjusted_logit = model_logit + PRIOR_ALPHA * (log(target_prior) - log(training_prior))
```

The runtime code defaults to no correction unless environment variables are set. For the current combined broad model, the best accuracy setting tested was:

```powershell
PRIOR_PROFILE=conservative_population_like
PRIOR_ALPHA=0.4
TRAINING_PRIOR_PATH=/app/models/training_prior_combined.json
PRIOR_PROFILE_PATH=/app/models/prior_profiles.json
```

On the combined validation split, this moved accuracy from `68.3%` to `69.4%` and macro recall from `44.4%` to `48.4%`. A stronger `portfolio_upload` profile at `PRIOR_ALPHA=1.0` improved macro recall to `54.1%` but lowered accuracy to `67.2%`.

The Docker Compose demo enables the conservative calibrated setting by default. Set `PRIOR_PROFILE=` and `PRIOR_ALPHA=0.0` to run raw model outputs.

## Modeling Conclusion

The strongest experimental approach used ConvNeXt-Tiny embeddings with lightweight downstream classifiers. Model tuning alone did not produce a stable, fresh-holdout 80% result. The main bottleneck is data quality:

- not enough face-specific examples for weak classes
- broad labels that overlap visually, especially acne-like texture, folliculitis-like bumps, and dermatitis-like irritation
- auxiliary augmentation data that increased class counts but introduced distribution shift
- small holdout sets with high variance

The next real improvement should come from label audit, clearer class definitions, and additional face-aligned data rather than another small classifier-head tweak.

Target labels for the current prototype:

- acne-like texture
- rosacea-like facial redness
- dermatitis-like irritation
- hyperpigmentation / melasma-like uneven pigmentation
- folliculitis-like bumps
- clinician-review / uncertain

## Adding A Trained ONNX Classifier

Place these files in `models/`:

```text
models/skin_classifier.onnx
models/label_map.json
```

Use `models/label_map.example.json` as the starting schema. When both files exist, the app automatically uses ONNX inference. Without them, it uses the heuristic fallback.

The runtime expects:

```text
input: float32 tensor shaped [1, 3, 224, 224]
normalization: ImageNet mean/std
output: one logit per label
```

For multi-label outputs, set:

```json
{ "problem_type": "multilabel" }
```

For single-class softmax outputs, set:

```json
{ "problem_type": "multiclass" }
```

## Dataset Prep

We do not have a dataset checked into this project. See [DATASETS.md](DATASETS.md) for the recommended dataset plan and the local folder layout.

The preparation script now defaults to patient/case-aware splitting when the manifest has a `case_id` column:

```powershell
python scripts/prepare_imagefolder.py `
  --manifest data/raw/scin/face_skin_manifest.csv `
  --image-root data/raw/scin `
  --output data/processed/scin_grouped_v1
```

Each run writes `split_audit.json` with image counts, group counts, and a leakage check. For medical image datasets, avoid `--allow-image-level-split` unless there is truly no case or patient identifier.

For stricter SCIN labels, rebuild the manifest with:

```powershell
python scripts/scin_build_manifest.py `
  --min-label-confidence 0.45 `
  --exclude-mixed-labels `
  --mixed-label-margin 0.15
```

The mapping rationale is versioned in `models/label_mapping_rules_v2.json`.

## GPU Training

Docker GPU passthrough can be tested with:

```powershell
docker compose -f docker-compose.train-gpu.yml build
docker compose -f docker-compose.train-gpu.yml run --rm trainer-gpu
```

Example CUDA training run:

```powershell
docker compose -f docker-compose.train-gpu.yml run --rm trainer-gpu `
  python scripts/train_export_onnx.py `
    --data-dir data/processed/scin_headneck_plus_fitzpatrick_v1 `
    --output-dir models/experiments/efficientnet_gpu_combined_20e `
    --model efficientnet_b0 `
    --epochs 20 `
    --batch-size 12 `
    --num-workers 2 `
    --lr 0.00005 `
    --class-weights none
```

On this Windows Docker setup, image loading from the bind-mounted project folder can dominate training time. ACNE04 measured about 3.4 seconds for a single 16-image batch with `num-workers=0`. For longer runs, prefer a WSL-native project path or copy prepared ImageFolder data into a Docker volume before training. The GPU compose file also keeps a `torch-cache` volume so pretrained weights do not download on every run.
