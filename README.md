# DermaLens

Private facial skin screening on local hardware. DermaLens is a portfolio ML project that explores how far a privacy-preserving dermatology-vision workflow can go with small, noisy, imbalanced public datasets.

This is not a diagnostic medical device. Outputs are screening-style observations and should be reviewed by a qualified clinician for medical decisions.

## Reader Guide

If you are reviewing this project for a portfolio or interview, start here:

1. **Run the app:** use the Docker commands in [Local Run With Docker](#local-run-with-docker), then open `http://127.0.0.1:8765`.
2. **Try one image:** upload a clear face photo. The app runs locally, strips metadata, and does not retain the upload by default.
3. **Check the engineering path:** read [What The App Does](#what-the-app-does), [Architecture](#architecture), and [Privacy Defaults](#privacy-defaults).
4. **Understand the ML story:** read [Portfolio Snapshot](#portfolio-snapshot), [Modeling Conclusion](#modeling-conclusion), and then [PORTFOLIO_WRITEUP.md](PORTFOLIO_WRITEUP.md).
5. **Review the critical methodology:** read [MODEL_CARD.md](MODEL_CARD.md), [DATA_CARD.md](DATA_CARD.md), [BENCHMARKS.md](BENCHMARKS.md), and the opening section of [FINAL_TECHNICAL_EXPERIMENTS.md](FINAL_TECHNICAL_EXPERIMENTS.md).

For a more explicit walkthrough, see [GUIDE.md](GUIDE.md).

## Portfolio Snapshot

- **Stack:** FastAPI, ONNX Runtime, Docker, static HTML/CSS/JS.
- **Privacy:** EXIF stripping, localhost binding, no upload retention by default.
- **Deployable model:** MobileNetV3-Small ONNX classifier with optional prior calibration.
- **Research models:** ConvNeXt frozen embeddings, neural classifier heads, long-tail supervised contrastive tests, targeted augmentation, probability ensembles, and calibration sweeps.
- **Runtime polish:** ONNX inference runs off the async event loop, uploads have a decompression-bomb guard, and facial region summaries now use an OpenCV face detector with a geometry fallback.
- **Fair grouped MobileNet baseline:** retraining MobileNetV3-Small separately on 12 grouped SCIN folds reached 44.8% +/- 9.9 accuracy and 29.1% +/- 3.9 macro recall. See `models/grouped_scin_mobilenet_retrained_baseline_12seed_metrics.json`.
- **MobileNet checkpoint-selection diagnostic:** if the same MobileNet training histories are reselected post hoc by validation accuracy instead of macro recall, MobileNet reaches 52.4% +/- 5.2 accuracy and 25.1% +/- 3.3 macro recall. This is optimistic because it chooses from the evaluation-fold history. The conservative envelope is therefore >= +16.1 accuracy points and >= +4.8 macro-recall points for Derm Foundation against the best MobileNet policy for each metric separately. See `models/grouped_scin_mobilenet_checkpoint_selection_diagnostic.json`.
- **Majority-class floor:** always predicting `dermatitis_like_irritation` reaches 63.7% +/- 1.6 accuracy and 16.7% macro recall on the 12 fair grouped validation folds.
- **Fair Derm Foundation comparison:** the local SavedModel Derm Foundation linear probe reached 68.6% +/- 5.0 accuracy and 33.9% +/- 3.7 macro recall over the same 12 grouped folds. Paired seeds show an accuracy lift over fair MobileNet (+23.8 points, 95% CI +16.1 to +31.4, exact sign p=0.00049) and a smaller macro-recall lift (+4.8 points, 95% CI +2.6 to +7.1, exact sign p=0.00635). These p-values are repeated-split stability diagnostics, not independent external-test-set inference, because the 12 folds resample the same SCIN case universe. See `models/grouped_scin_fair_model_comparison_metrics.json` and `models/grouped_scin_derm_foundation_embedding_12seed_local_model_metrics.json`.
- **Architecture-matched ImageNet control:** BiT-M R101x3 ImageNet embeddings under the same grouped/nested probe protocol reached 62.7% +/- 4.9 accuracy and 29.4% +/- 6.2 macro recall. Derm Foundation remains ahead by +5.9 accuracy points and +4.6 macro-recall points, narrowing the claim from "bigger model wins" to a SCIN downstream benefit for dermatology-specific pretraining. See `models/grouped_scin_bit_m_r101x3_embedding_12seed_metrics.json`.
- **Generic frozen-encoder control:** ConvNeXt-Tiny ImageNet embeddings under the same grouped/nested probe protocol reached 53.1% +/- 4.1 accuracy and 23.1% +/- 3.3 macro recall. Derm Foundation remains ahead by +15.5 accuracy points and +10.8 macro-recall points, but the cleaner attribution control is now BiT-M R101x3. See `models/grouped_scin_convnext_tiny_embedding_12seed_metrics.json`.
- **Legacy deployable ONNX result:** 69.4% accuracy and 48.4% macro recall after conservative calibration on the earlier combined validation path. This is retained as experiment history because the original split path had leakage risk.
- **Fixed-model SCIN diagnostic, not a clean held-out headline:** the deployed ONNX model reached 86.2% +/- 1.2 accuracy and 63.1% +/- 10.1 macro recall on grouped SCIN folds, but the model was previously fine-tuned on SCIN-derived head/neck data. Grouping prevents overlap inside the new folds; it does not prove the fixed model had never seen those validation cases. See `models/grouped_scin_clean_split_metrics.json`.
- **Grouped modeling experiments:** Derm Foundation is the clean representation comparison. The decoupled head is a refreshed fixed-encoder operating-point experiment because it uses frozen logits from the previously SCIN-trained deployed ONNX model.
- **Derm Foundation result:** the dermatology-specific embedding probe improves over the fair MobileNet baseline under 10- and 12-seed sensitivity checks, but it does not solve the tail-label problem and class-level gains remain uneven.
- **Subgroup workflow demo:** Fitzpatrick/Monk subgroup metrics are reported in `models/grouped_scin_subgroup_metrics.json`, but the buckets are too small and the tone labels too noisy for fairness claims.
- **Best untuned experimental validation result:** 79.2% accuracy and 71.0% macro recall with a mixed ConvNeXt ensemble.
- **Critical limitation:** fresh holdout testing did not confirm the validation-tuned 81.4% result, and the later fixed-model grouped SCIN check is not a clean model holdout because original training-case exclusion was not available. Some tail classes have fewer than 10 validation images per split, so per-class recall and macro recall are underpowered diagnostics rather than stable evidence.
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

After the grouped-split correction, I first evaluated the fixed deployed ONNX model on SCIN-only case-level splits across five seeds. With the same conservative prior calibration used by Docker, the model reached:

```text
accuracy:     86.2% +/- 1.2
macro recall: 63.1% +/- 10.1
```

This is now treated as a **fixed-model diagnostic**, not a clean held-out model result. The grouped split prevents case overlap between the newly constructed train and validation folds, but the deployed model had already been fine-tuned on SCIN-derived head/neck data. Without the original model-training case list, these validation folds cannot exclude cases the model may have seen.

The corrected fair-baseline path now exists:

```powershell
python scripts/run_grouped_mobilenet_baseline.py `
  --manifest data/raw/scin/face_skin_manifest.csv `
  --image-root data/raw/scin `
  --summary-output models/grouped_scin_mobilenet_retrained_baseline_12seed_metrics.json `
  --seeds 42 7 13 21 84 101 202 404 707 808 909 1001 `
  --epochs 8 `
  --batch-size 16 `
  --num-workers 0
```

The completed 12-seed fair baseline reached:

```text
majority-class baseline: 63.7% +/- 1.6 accuracy, 16.7% macro recall
fair MobileNetV3 retrain: 44.8% +/- 9.9 accuracy, 29.1% +/- 3.9 macro recall
```

This is much lower than the contaminated fixed-model diagnostic, which confirms that the audit found a real validity problem rather than a documentation nuance. It also makes the foundation-model comparison fair.

The subgroup workflow now reports Fitzpatrick and Monk tone buckets across the same five grouped split seeds. It is useful as a fairness-aware reporting demonstration, but it is not strong enough for a fairness claim because some buckets have only a handful of validation images and SCIN's own documentation notes that Fitzpatrick and Monk scales were not intended for retrospective estimation from images.

I also tested a decoupled cRT-style head under the same grouped protocol. The image model stayed frozen; I used its ONNX logits as a compact representation and retrained only a class-balanced logistic head on each grouped training split. C is now selected on a nested grouped calibration split carved from training data only. This created a tail-sensitive operating point:

```text
fixed ONNX diagnostic:     86.2% +/- 1.2 accuracy, 63.1% +/- 10.1 macro recall
nested decoupled head:     75.3% +/- 1.7 accuracy, 70.7% +/- 11.4 macro recall
```

This is a valid operating-point result for the fixed deployed encoder, not a clean representation benchmark. The frozen encoder may have seen SCIN-derived training cases during original fine-tuning, so this result is excluded from the fair MobileNet versus Derm Foundation comparison.

## Foundation Embedding Experiment

The repo now includes a direct Derm Foundation embedding experiment:

```powershell
python scripts/evaluate_derm_foundation_embeddings.py `
  --manifest data/raw/scin/face_skin_manifest.csv `
  --image-root data/raw/scin/images `
  --output models/grouped_scin_derm_foundation_embedding_metrics.json
```

This uses Google's `google/derm-foundation` embedding model as a frozen representation, trains a class-balanced linear probe, selects C on a nested grouped calibration split, and evaluates once on the held-out grouped fold. The flagship 12-seed artifact now uses `--embedding-source local-model`, which runs the local SavedModel instead of Google's precomputed SCIN embedding file. Three native Windows seeds failed before evaluation (`303`, `505`, `606`) and were excluded by failure status rather than by metric; seed `707` was recovered in the completed 12-seed run.

The completed Derm Foundation probe result, after expanding the matched-seed comparison, was:

```text
fixed ONNX diagnostic: 86.2% +/- 1.2 accuracy, 63.1% +/- 10.1 macro recall
majority baseline:     63.7% +/- 1.6 accuracy, 16.7% macro recall
Derm Foundation probe, local model: 68.6% +/- 5.0 accuracy, 33.9% +/- 3.7 macro recall
fair MobileNet retrain: 44.8% +/- 9.9 accuracy, 29.1% +/- 3.9 macro recall
accuracy-selected MobileNet diagnostic: 52.4% +/- 5.2 accuracy, 25.1% +/- 3.3 macro recall
BiT-M R101x3 ImageNet probe: 62.7% +/- 4.9 accuracy, 29.4% +/- 6.2 macro recall
ConvNeXt-Tiny frozen probe: 53.1% +/- 4.1 accuracy, 23.1% +/- 3.3 macro recall
```

The honest comparison is still mixed, but stronger than the five-seed result. At five seeds, exact paired tests could not cross p<0.05 even with all accuracy deltas positive, so the original t-test was too fragile to carry alone. After expanding to 10 and 12 matched completed seeds, Derm Foundation gives a large accuracy lift over fair MobileNet and a smaller positive macro-recall lift. At 12 seeds, the local-model paired deltas are +23.8 points accuracy and +4.8 points macro recall. Because these seeds are repeated grouped resamples of the same dataset, the paired tests should be read as robustness checks rather than independent clinical evidence. Class-level recall still drops for rosacea and hyperpigmentation and improves mainly on acne, dermatitis, and clinician-review, so this is not a solved tail-label model.

I also checked whether the accuracy lift was mainly caused by MobileNet's macro-recall checkpoint selection. The diagnostic answer is no, though the gap shrinks: reselecting MobileNet epochs post hoc for validation accuracy raises MobileNet to 52.4% accuracy. To avoid mixing MobileNet policies, the conservative statement is: Derm Foundation is at least +16.1 points better on accuracy and at least +4.8 points better on macro recall against the most favorable MobileNet policy for each metric separately. This diagnostic is not a clean estimate because it uses the evaluation history for epoch choice; it is included only to make the comparison harder to overstate.

I then ran generic and architecture-matched frozen-encoder controls with the same nested grouped linear-probe protocol. ConvNeXt-Tiny reached 53.1% accuracy and 23.1% macro recall. The stronger control, BiT-M R101x3 ImageNet at the same 448px scale, reached 62.7% accuracy and 29.4% macro recall. Derm Foundation still leads BiT-M by +5.9 accuracy points and +4.6 macro-recall points, so the attribution claim is narrower and stronger: dermatology-specific pretraining appears to help on this SCIN downstream task beyond architecture and input resolution, but this is still not external validation. Google's own documentation presents SCIN as a public downstream linear-classifier use case for Derm Foundation, so this remains a vendor-home downstream benchmark.

## Current Validity Status

The current repo now distinguishes between validated machinery and unresolved model evidence:

1. Implemented: grouped ImageFolder preparation with case-level leakage audits.
2. Implemented: support-aware ONNX evaluation with low-support label flags.
3. Completed: fair grouped MobileNet retrain/evaluate runner across 12 matched seeds.
4. Completed: Derm Foundation comparison and 5/10/12 seed-count sensitivity against that fair MobileNet baseline.
5. Still needed for stronger science: more face-specific data or a label audit for the low-support/overlapping tail classes.
6. Any metric involving labels with fewer than about 10 validation examples should be described as underpowered, not as a stable tail-class result.

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
