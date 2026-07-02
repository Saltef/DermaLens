# DermaLens

Private facial skin screening on local hardware. DermaLens is a portfolio ML project that explores how far a privacy-preserving dermatology-vision workflow can go with small, noisy, imbalanced public datasets.

This is not a diagnostic medical device. Outputs are screening-style observations and should be reviewed by a qualified clinician for medical decisions.

## Reader Guide

If you are reviewing this project for a portfolio or interview, start here:

1. **Run the app:** use the Docker commands in [Local Run With Docker](#local-run-with-docker), then open `http://127.0.0.1:8765`.
2. **Try one image:** upload a clear face photo. The app runs locally, strips metadata, and does not retain the upload by default.
3. **Check the engineering path:** read [What The App Does](#what-the-app-does), [Architecture](#architecture), and [Privacy Defaults](#privacy-defaults).
4. **Understand the ML story:** read [Portfolio Snapshot](#portfolio-snapshot), [Modeling Conclusion](#modeling-conclusion), and then [PORTFOLIO_WRITEUP.md](PORTFOLIO_WRITEUP.md).
5. **Review the critical methodology:** read [MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md), and the opening section of [FINAL_TECHNICAL_EXPERIMENTS.md](FINAL_TECHNICAL_EXPERIMENTS.md).

For a more explicit walkthrough, see [GUIDE.md](GUIDE.md).

## Portfolio Snapshot

- **Stack:** FastAPI, ONNX Runtime, Docker, static HTML/CSS/JS.
- **Privacy:** EXIF stripping, localhost binding, no upload retention by default.
- **Deployable model:** MobileNetV3-Small ONNX classifier with optional prior calibration.
- **Research models:** ConvNeXt frozen embeddings, neural classifier heads, long-tail supervised contrastive tests, targeted augmentation, probability ensembles, and calibration sweeps.
- **Runtime polish:** ONNX inference runs off the async event loop, uploads have a decompression-bomb guard, and facial region summaries now use an OpenCV face detector with a geometry fallback.
- **Best deployable ONNX result:** 69.4% accuracy and 48.4% macro recall after conservative calibration.
- **Clean grouped SCIN check:** fixed deployed ONNX model with conservative prior calibration reached 86.2% +/- 1.2 accuracy and 63.1% +/- 10.1 macro recall across 5 grouped case-level split seeds. See `models/grouped_scin_clean_split_metrics.json`.
- **Grouped modeling win, pre-nested C selection:** a decoupled balanced head over frozen ONNX logits lifted macro recall to 73.1% +/- 10.1, a +9.9 point gain over the deployed operating point, with an accuracy trade-off to 75.1% +/- 2.0. The script now nests C-selection inside the training fold; rerun with local SCIN data before treating this as the final refreshed number.
- **Derm Foundation result:** the high-leverage dermatology-specific embedding probe completed, but it was not a Pareto win: 66.8% +/- 6.9 accuracy and 33.8% +/- 5.9 macro recall across the same grouped split seeds.
- **Subgroup audit:** Fitzpatrick/Monk subgroup metrics are now reported in `models/grouped_scin_subgroup_metrics.json`; darker Monk buckets are too small for fairness claims.
- **Best untuned experimental validation result:** 79.2% accuracy and 71.0% macro recall with a mixed ConvNeXt ensemble.
- **Critical limitation:** fresh holdout testing did not confirm the validation-tuned 81.4% result. The grouped SCIN check gives a clean post-correction baseline, but tail-class estimates remain fragile because some validation classes have only 2-7 images per split.
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
       - OpenCV face-detected region crop with fallback geometry
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

Run the test suite:

```powershell
python -m pytest -q
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

After the grouped-split correction, I evaluated the fixed deployed ONNX model on SCIN-only case-level splits across five seeds. With the same conservative prior calibration used by Docker, the model reached:

```text
accuracy:     86.2% +/- 1.2
macro recall: 63.1% +/- 10.1
```

This is the cleanest post-correction deployed-model result in the repo. It is not a final clinical claim: the SCIN-only split is smaller than the earlier merged benchmark, and rare labels such as hyperpigmentation and clinician-review have very small validation counts. The result is useful because it converts the old "TBD on clean split" caveat into a measured baseline.

The subgroup audit now reports Fitzpatrick and Monk tone buckets across the same five grouped split seeds. It is useful as a fairness workflow demonstration, but it is not strong enough for a fairness claim because some darker-tone buckets have only a handful of validation images.

I also tested a decoupled cRT-style head under the same grouped protocol. The image model stayed frozen; I used its ONNX logits as a compact representation and retrained only a class-balanced logistic head on each grouped training split. This created a tail-sensitive operating point:

```text
deployed prior baseline:   86.2% +/- 1.2 accuracy, 63.1% +/- 10.1 macro recall
decoupled balanced head:   75.1% +/- 2.0 accuracy, 73.1% +/- 10.1 macro recall
```

The win is in tail recall, not raw accuracy. Folliculitis recall rose from 44.4% to 70.1%, clinician-review from 45.0% to 68.0%, hyperpigmentation from 40.0% to 73.3%, and rosacea from 68.6% to 77.1%. This is not the default app setting, but it demonstrates that the grouped protocol can measure a real modeling improvement. A later review found that C was selected on the evaluation fold in this artifact; the script has been corrected to select C on a nested grouped calibration split and should be rerun when the local data is restored.

## Foundation Embedding Experiment

The repo now includes a direct Derm Foundation embedding experiment:

```powershell
python scripts/evaluate_derm_foundation_embeddings.py `
  --manifest data/raw/scin/face_skin_manifest.csv `
  --image-root data/raw/scin/images `
  --output models/grouped_scin_derm_foundation_embedding_metrics.json
```

This uses Google's `google/derm-foundation` embedding model as a frozen representation, trains a class-balanced linear probe, selects C on a nested grouped calibration split, and evaluates once on the held-out grouped fold.

The completed result did **not** improve the model:

```text
deployed grouped baseline: 86.2% +/- 1.2 accuracy, 63.1% +/- 10.1 macro recall
Derm Foundation probe:     66.8% +/- 6.9 accuracy, 33.8% +/- 5.9 macro recall
```

The failure mode was the tail: hyperpigmentation recall stayed at 0.0 and folliculitis/rosacea recall remained weak. I also sanity-checked logistic-probe variants with and without class weighting; none recovered the deployed grouped baseline. This is a useful negative result because it shows that a dermatology foundation representation alone does not solve noisy task-specific label mapping.

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
