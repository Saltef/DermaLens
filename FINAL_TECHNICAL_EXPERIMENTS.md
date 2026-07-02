# DermaLens: Technical Experiment Report

This report documents the modeling work behind DermaLens, a privacy-preserving facial skin screening prototype. The goal was not to force a single leaderboard number, but to understand which modeling strategies actually improve a small, imbalanced, face-focused dermatology dataset under local deployment constraints.

The experimental arc is intentionally critical:

- establish a deployable local ONNX baseline
- test stronger pretrained vision representations
- test long-tail methods inspired by supervised contrastive learning
- test targeted augmentation for minority classes
- test probability ensembles and calibration
- confirm apparent gains on fresh holdouts
- identify where additional data is required

The strongest untuned model reached `79.2%` accuracy and `71.0%` macro recall on the original validation split. A validation-tuned variant reached `81.4%`, but fresh holdout confirmation did not reproduce it. The final conclusion is that the system is now data-limited: more face-specific, consistently labeled data is needed for acne/folliculitis/dermatitis separation.

## How To Read These Numbers

The metrics below are retained as an experiment log, not as clinical validation. A later methodological review identified a leakage risk in the original preparation path: SCIN cases can contain multiple images, while the older fallback split operated at the image-row level. That means near-duplicate photos from the same case could land in both train and validation.

I implemented the corrected protocol in `scripts/prepare_imagefolder.py`:

- split by `case_id` / group ID, not image row
- assert that no group appears in both train and validation
- write `split_audit.json` with image counts, group counts, and split settings
- skip exact duplicate file digests while writing ImageFolder outputs

Future headline metrics should be produced only after rerunning the baseline and best experimental models under this grouped split. The earlier numbers are still useful for comparing modeling ideas, but they should be framed as pre-correction validation results.

### Post-Correction Clean Baseline

I reran the fixed deployed ONNX model on SCIN-only grouped case-level splits across five split seeds (`42`, `7`, `13`, `21`, `84`). This was a deployment check, not a retraining sweep: the model weights were fixed, no test-time augmentation was used, and the Docker app's conservative prior setting was applied (`conservative_population_like`, `alpha=0.4`).

| Evaluation | Accuracy | Macro Recall |
| --- | ---: | ---: |
| Raw logits, mean across 5 grouped split seeds | 84.9% | 54.1% |
| Deployed conservative prior, mean +/- std | 86.2% +/- 1.2 | 63.1% +/- 10.1 |

For seed `42`, the bootstrap 95% CI was `80.0%` to `91.3%` for accuracy and `41.5%` to `67.4%` for macro recall. The wide macro-recall interval is expected because the grouped SCIN validation folds contain very small tail classes.

Artifact: `models/grouped_scin_clean_split_metrics.json`.

### Skin-Tone Subgroup Audit

Using the same fixed deployed model and grouped SCIN split seeds, I evaluated available Fitzpatrick and Monk tone metadata. This is an audit workflow, not a fairness claim, because several subgroup counts are small.

Fitzpatrick bucket summary:

| Bucket | Mean Val Images | Accuracy | Macro Recall |
| --- | ---: | ---: | ---: |
| FST1-2 | 38.6 | 87.3% +/- 5.9 | 77.7% +/- 11.7 |
| FST3-4 | 43.6 | 88.9% +/- 5.7 | 78.6% +/- 15.4 |
| FST5-6 | 14.8 | 89.0% +/- 14.0 | 83.8% +/- 20.7 |
| Unknown | 58.6 | 83.9% +/- 5.9 | 57.5% +/- 11.9 |

Monk US bucket summary:

| Bucket | Mean Val Images | Accuracy | Macro Recall |
| --- | ---: | ---: | ---: |
| MST1-3 | 102.4 | 87.9% +/- 2.7 | 63.3% +/- 9.6 |
| MST4-6 | 49.4 | 83.6% +/- 5.5 | 70.2% +/- 7.6 |
| MST7-10 | 4.8 | 75.0% +/- 50.0 | 75.0% +/- 50.0 |

The audit is useful because it makes subgroup reporting executable. It does not validate fairness: the darkest Monk bucket is especially underpowered, and skin-tone metadata itself is imperfect. Future evaluation should intentionally stratify by skin tone rather than relying on incidental subgroup counts.

Artifact: `models/grouped_scin_subgroup_metrics.json`.

### Decoupled Balanced-Head Result

I then tested one modeling improvement under the same grouped protocol. This was a cRT-style decoupled experiment:

1. Freeze the deployed ONNX image model.
2. Use its six output logits as a compact representation.
3. Train only a class-balanced multinomial logistic head on each grouped training split.
4. Select the best `C` value by validation macro recall within each split.
5. Compare against the fixed deployed conservative-prior operating point.

| Model | Accuracy | Macro Recall |
| --- | ---: | ---: |
| Deployed conservative-prior ONNX baseline | 86.2% +/- 1.2 | 63.1% +/- 10.1 |
| Decoupled balanced logit head | 75.1% +/- 2.0 | 73.1% +/- 10.1 |

Per-class recall moved as follows:

| Class | Deployed Baseline | Decoupled Head | Change |
| --- | ---: | ---: | ---: |
| acne_like_texture | 84.5% | 73.0% | -11.5 |
| clinician_review | 45.0% | 68.0% | +23.0 |
| dermatitis_like_irritation | 96.4% | 76.7% | -19.7 |
| folliculitis_like_bumps | 44.4% | 70.1% | +25.7 |
| hyperpigmentation_like_uneven_tone | 40.0% | 73.3% | +33.3 |
| rosacea_like_redness | 68.6% | 77.1% | +8.6 |

Decision: this is a useful tail-recall modeling signal, not the new default app model. It demonstrates that a class-balanced decoupled head can move the metric the project cares about, while making the accuracy/macro-recall trade-off explicit. A later review found that this artifact selected C on the evaluation fold; `scripts/evaluate_decoupled_logit_head.py` now selects C on a nested grouped calibration split and should be rerun with local SCIN data before this is treated as the final refreshed result.

Artifact: `models/grouped_scin_decoupled_logit_head_metrics.json`.

Final high-leverage experiment implementation:

- Script: `scripts/evaluate_derm_foundation_embeddings.py`
- Representation: `google/derm-foundation` embeddings
- Head: class-balanced logistic regression
- Selection: C chosen on a nested grouped calibration split carved from the training fold
- Evaluation: held-out grouped SCIN fold used once

The current committed artifact is blocked rather than completed because the Derm Foundation checkpoint is gated behind Hugging Face terms and the local raw SCIN data is intentionally gitignored.

Artifact: `models/grouped_scin_derm_foundation_embedding_metrics.json`.

## Baseline To Beat

Current portfolio default:

- Model: fine-tuned MobileNetV3-Small ONNX
- Dataset: `scin_headneck_plus_fitzpatrick_v1`
- Inference: conservative prior calibration
- Split caveat: original image-level validation protocol; rerun with grouped `case_id` split before treating as final.
- Accuracy: `69.4%`
- Macro recall: `48.4%`

Clean grouped SCIN-only baseline:

- Model: same deployed MobileNetV3-Small ONNX
- Split: grouped by `case_id`, no train/validation group overlap
- Seeds: `42`, `7`, `13`, `21`, `84`
- Accuracy: `86.2% +/- 1.2`
- Macro recall: `63.1% +/- 10.1`
- Caveat: smaller SCIN-only evaluation with fragile tail-class counts

Raw flat model on the same combined validation split:

- Accuracy: `68.3%`
- Macro recall: `44.4%`

## Approach 1: Frozen Foundation-Style Embeddings

### Rationale

Recent dermatology and medical-vision work increasingly uses large pretrained/foundation representations with small downstream classifiers. This is attractive for this project because our limiting factor is not model capacity; it is limited, imbalanced labeled data. A frozen representation can sometimes separate classes better without overfitting the small tail classes.

This experiment uses ImageNet-pretrained backbones as practical local proxies for foundation-style embeddings:

- MobileNetV3-Small as a lightweight control
- EfficientNet-B0
- ConvNeXt-Tiny
- Swin-T
- ViT-B/16

For each backbone:

1. Replace the classifier head with identity.
2. Extract frozen train/validation embeddings.
3. Train a scikit-learn logistic regression classifier with `class_weight="balanced"`.
4. Sweep `C = 0.03, 0.1, 0.3, 1, 3, 10`.
5. Evaluate on the unchanged combined validation split.

### Results

| Backbone | Best C | Accuracy | Macro Recall | Interpretation |
| --- | ---: | ---: | ---: | --- |
| MobileNetV3-Small frozen | 0.3 | 61.8% | 49.3% | Control. Worse accuracy than fine-tuning. |
| EfficientNet-B0 frozen | 1.0 | 67.2% | 52.1% | Good recall lift, still below calibrated accuracy. |
| ConvNeXt-Tiny frozen | 0.1 | 68.9% | 61.6% | Best balance. Near baseline accuracy with much better class balance. |
| Swin-T frozen | 0.3 | 65.0% | 52.8% | Transformer context did not win on this dataset. |
| ViT-B/16 frozen | 0.3 | 65.0% | 52.4% | Heavy model, weaker than ConvNeXt here. |

### Decision

ConvNeXt-Tiny frozen embeddings are the best result from this approach:

- Accuracy is slightly below the calibrated deployable model: `68.9%` vs `69.4%`.
- Macro recall is much better: `61.6%` vs `48.4%`.

This is not an 80% breakthrough, but it is a meaningful technical improvement for balanced recognition. It is worth keeping as an experimental backend or ensemble candidate, but not automatically replacing the ONNX app model because deployment would need either:

- exporting a ConvNeXt feature extractor plus classifier head, or
- adding a scikit-learn sidecar runtime, which complicates the clean ONNX-only app.

### Files

- Script: `scripts/evaluate_frozen_embeddings.py`
- Best result: `models/experiments/frozen_embeddings_convnext_tiny/metrics.json`

## Approach 2: Long-Tail Supervised Contrastive Heads

### Rationale

Recent long-tail medical image work often improves minority-class behavior by changing the representation geometry, not only by adding class weights. The hypothesis here was that supervised contrastive learning would pull visually similar same-label cases together and make rare classes less likely to be swallowed by the dominant dermatitis-like class.

I tested this in two forms:

1. A full image-level trainer: `scripts/train_contrastive_classifier.py`
2. A faster cached-feature sweep: `scripts/sweep_contrastive_embeddings.py`

The full image trainer works and exports ONNX, but it is slow on this local Docker/CUDA setup: the one-epoch smoke test took about 3.5 minutes. To test parameters properly, I used the cached ConvNeXt-Tiny embeddings from Approach 1 and swept neural classifier heads on top of those embeddings.

### Parameter Sweep

The embedding-head sweep varied:

- hidden size: `256`, `512`, `768`, `1024`
- projection size: `64`, `128`
- dropout: `0.0`, `0.1`, `0.15`
- supervised contrastive weight: `0`, `0.002`, `0.005`, `0.02`, `0.03`, `0.1`
- temperature: `0.1`, `0.2`
- class weighting: none vs balanced
- sampler: balanced sampler vs normal shuffled batches
- selection metric: validation accuracy for the main sweep

### Results

| Experiment | Accuracy | Macro Recall | Key Setup | Interpretation |
| --- | ---: | ---: | --- | --- |
| Current calibrated app model | 69.4% | 48.4% | MobileNetV3 ONNX + conservative prior | Deployable default. |
| Approach 1 best | 68.9% | 61.6% | Frozen ConvNeXt + balanced logistic regression | Strong balance, no accuracy lift. |
| Approach 2 stage 1 | 60.7% | 67.0% | Balanced sampler + contrastive weight `0.1` | Best tail pressure, but too much accuracy loss. |
| Approach 2 stage 2 | 76.5% | 56.3% | 512 hidden, shuffled batches, no class weights, no SupCon | First strong accuracy lift. |
| Approach 2 stage 4 best observed | 78.7% | 67.7% | 768 hidden, balanced CE, shuffled batches, SupCon weight `0` | Best overall validation result. |
| Approach 2 stage 5 | 76.5% | 69.0% | 1024 hidden, balanced CE, higher LR | Better recall, lower accuracy. |

Best observed per-class recall for the stage 4 run:

| Class | Recall |
| --- | ---: |
| acne_like_texture | 60.6% |
| clinician_review | 84.6% |
| dermatitis_like_irritation | 92.2% |
| folliculitis_like_bumps | 52.4% |
| hyperpigmentation_like_uneven_tone | 66.7% |
| rosacea_like_redness | 50.0% |

### Decision

Approach 2 produced the best validation score so far:

- Accuracy improved from `69.4%` to `78.7%`.
- Macro recall improved from `48.4%` to `67.7%`.
- It did not reliably hit the `80%` target.

The important finding is slightly counterintuitive: the best observed run did not use a positive contrastive weight. The contrastive loss helped macro recall in some settings, especially with balanced sampling, but it usually traded away too much overall accuracy. The strongest setup was a larger neural head over ConvNeXt embeddings with balanced cross-entropy and normal shuffled batches.

I also checked seed sensitivity for the stage 4 hyperparameters. Single-seed confirmation runs ranged from `71.6%` to `75.4%` accuracy, so the `78.7%` result should be treated as the best observed validation run, not yet a stable production estimate. The next step should be repeated cross-validation or a larger held-out test set before claiming the model is truly near 80%.

### Files

- Full image trainer: `scripts/train_contrastive_classifier.py`
- Cached embedding sweep: `scripts/sweep_contrastive_embeddings.py`
- Best observed metrics: `models/experiments/supcon_embeddings_convnext_stage4_weighted_shuffle/metrics.json`
- Seed checks: `models/experiments/supcon_embeddings_convnext_seed2/metrics.json`, `models/experiments/supcon_embeddings_convnext_seed3/metrics.json`, `models/experiments/supcon_embeddings_convnext_seed4/metrics.json`

## Approach 3: Targeted Tail-Class Augmentation

### Rationale

The third approach tested whether the model could improve by adding more train-only examples for underrepresented labels. This was treated as a dataset-shift experiment, not just a class-count experiment: the validation split was preserved from the original face-focused dataset, and auxiliary images were used only for training.

I updated the augmentation builder with `--preserve-val` so validation remains exactly comparable with the baseline:

- Base validation count: `183`
- Augmented validation count: `183`
- Augmentation source: `skin_balanced_body_v1`
- Augmented dataset: `data/processed/targeted_aug_preserve_val_v2`

Train counts after augmentation:

| Class | Train Count |
| --- | ---: |
| acne_like_texture | 132 |
| clinician_review | 144 |
| dermatitis_like_irritation | 406 |
| folliculitis_like_bumps | 180 |
| hyperpigmentation_like_uneven_tone | 160 |
| rosacea_like_redness | 40 |

### Results

| Experiment | Accuracy | Macro Recall | Interpretation |
| --- | ---: | ---: | --- |
| Frozen ConvNeXt + logistic regression on targeted augmentation | 57.4% | 55.0% | Worse than base ConvNeXt; augmentation caused distribution shift. |
| ConvNeXt augmented embeddings + neural head | 62.3% | 57.9% | Better than augmented logistic, still below base-data models. |

### Decision

Approach 3 does not improve the model by itself. The likely reason is domain mismatch: the auxiliary images help class counts but are less aligned with the face-focused validation distribution. This is a useful negative result for the portfolio write-up because it shows that "more data" is not automatically better when the source distribution changes.

### Files

- Updated builder: `scripts/build_targeted_augmented_imagefolder.py`
- Augmented dataset manifest: `data/processed/targeted_aug_preserve_val_v2/augmentation_manifest.json`
- Frozen ConvNeXt result: `models/experiments/approach3_targeted_aug_convnext/metrics.json`
- Augmented head result: `models/experiments/approach3_targeted_aug_convnext_head_small/metrics.json`

## Mixed Approach: ConvNeXt Heads + Augmentation Diversity Ensemble

### Rationale

Since the best single results came from ConvNeXt embeddings and neural heads, the mixed approach tested whether multiple imperfect classifiers could complement each other. The ensemble combined:

- base ConvNeXt logistic heads from Approach 1
- base ConvNeXt neural heads from Approach 2
- augmented ConvNeXt classifiers from Approach 3

The ensemble averages class probabilities and sweeps mixture weights.

### Results

| Model | Accuracy | Macro Recall |
| --- | ---: | ---: |
| Current calibrated app model | 69.4% | 48.4% |
| Approach 1 best | 68.9% | 61.6% |
| Approach 2 best observed single run | 78.7% | 67.7% |
| Approach 3 best standalone | 62.3% | 57.9% |
| Mixed ensemble best | 79.2% | 71.0% |

Best ensemble per-class recall:

| Class | Recall |
| --- | ---: |
| acne_like_texture | 60.6% |
| clinician_review | 92.3% |
| dermatitis_like_irritation | 92.2% |
| folliculitis_like_bumps | 47.6% |
| hyperpigmentation_like_uneven_tone | 83.3% |
| rosacea_like_redness | 50.0% |

The best observed ensemble used:

- `28.6%` base ConvNeXt neural head, 768 hidden units
- `28.6%` base ConvNeXt neural head, 1024 hidden units
- `42.9%` augmented ConvNeXt neural head, 1024 hidden units

### Decision

The mixed approach is the best overall result so far:

- Accuracy: `79.2%`
- Macro recall: `71.0%`

It still does not cleanly cross `80%`, but it is close and substantially better balanced than the deployed MobileNet model. The augmented model is weak alone, but it contributes useful diversity inside the ensemble. This is a good portfolio story: targeted augmentation was rejected as a standalone training source, then reused more carefully as one ensemble member.

### Files

- Ensemble script: `scripts/sweep_mixed_ensemble.py`
- Best ensemble result: `models/experiments/mixed_convnext_ensemble_v1/metrics.json`
- Fine weight sweep: `models/experiments/mixed_convnext_ensemble_v2_fine_weights/metrics.json`

## Final Diagnostic Tests: Calibration and Error Audit

### Rationale

The mixed ensemble was close to the `80%` target: `79.2%` accuracy on `183` validation images. That means only two additional correct predictions are enough to cross the threshold. I tested post-hoc methods that do not require retraining the image backbone:

- temperature scaling
- confidence-based `clinician_review` fallback
- class-specific probability bias calibration
- error CSV export for manual review

### Results

| Test | Accuracy | Macro Recall | Notes |
| --- | ---: | ---: | --- |
| Mixed ensemble before calibration | 79.2% | 71.0% | 38 validation errors. |
| Temperature scaling | 79.2% | 71.0% | Did not change the winning predictions. |
| Confidence fallback | below best | below best | Useful for safety UX, not best raw accuracy. |
| Validation-tuned class bias | 81.4% | 74.6% | 34 validation errors. |

Best calibrated per-class recall:

| Class | Recall |
| --- | ---: |
| acne_like_texture | 69.7% |
| clinician_review | 92.3% |
| dermatitis_like_irritation | 90.2% |
| folliculitis_like_bumps | 61.9% |
| hyperpigmentation_like_uneven_tone | 83.3% |
| rosacea_like_redness | 50.0% |

Best class-bias settings:

| Class | Bias |
| --- | ---: |
| acne_like_texture | 1.45 |
| clinician_review | 1.00 |
| dermatitis_like_irritation | 1.00 |
| folliculitis_like_bumps | 1.45 |
| hyperpigmentation_like_uneven_tone | 1.00 |
| rosacea_like_redness | 0.45 |

### Decision

This crosses the `80%` target, but it must be described honestly: it is a validation-tuned diagnostic result, not a clean held-out test result. The proper next step is to lock these bias settings and evaluate on either:

- a fresh held-out split, or
- repeated cross-validation with calibration learned only from each training fold.

For the portfolio, the accurate phrasing is:

> A validation-tuned ConvNeXt ensemble with class-bias calibration reached `81.4%` accuracy and `74.6%` macro recall on the face-focused validation split. The untuned ensemble reached `79.2%` accuracy and `71.0%` macro recall.

### Files

- Diagnostic script: `scripts/tune_ensemble_diagnostics.py`
- Metrics: `models/experiments/ensemble_diagnostics_v1/metrics.json`
- Error audit CSV: `models/experiments/ensemble_diagnostics_v1/best_accuracy_errors.csv`

## Fresh Holdout Confirmation

### Protocol

To check whether the `81.4%` validation-tuned result generalized, I ran a fresh holdout confirmation using the cached ConvNeXt embeddings from the original training pool. The original validation split was not used for this confirmation.

For each seed:

1. Split the original training embeddings into model-train, calibration, and holdout.
2. Train logistic and neural heads only on model-train.
3. Select ensemble weights on calibration.
4. Tune class bias on calibration.
5. Evaluate once on the untouched holdout.

Each holdout split had `111` images.

### Results

| Split Seed | Weighted Ensemble Holdout | Bias-Calibrated Holdout | Bias-Calibrated Macro Recall |
| --- | ---: | ---: | ---: |
| 42 | 70.3% | 69.4% | 55.9% |
| 7 | 68.5% | 73.0% | 65.2% |
| 13 | 61.3% | 64.0% | 47.8% |
| 21 | 65.8% | 66.7% | 54.8% |

### Decision

The fresh holdout test does **not** confirm the `81.4%` result. The validation-tuned class bias improved the original validation split, but did not generalize reliably across fresh holdouts.

The portfolio claim should therefore be revised to:

> The best untuned mixed ensemble reached `79.2%` accuracy and `71.0%` macro recall on the original face-focused validation split. A validation-tuned class-bias calibration reached `81.4%`, but fresh holdout confirmation did not reproduce that result, suggesting calibration overfit the validation split.

This is still a strong engineering result because it shows responsible model validation: the project found an apparent 80%+ result, then tested it on fresh holdouts and rejected the overfit claim.

### Files

- Confirmation script: `scripts/confirm_holdout_calibration.py`
- Seed 42 result: `models/experiments/fresh_holdout_confirmation_v1/metrics.json`
- Seed 7 result: `models/experiments/fresh_holdout_confirmation_seed7/metrics.json`
- Seed 13 result: `models/experiments/fresh_holdout_confirmation_seed13/metrics.json`
- Seed 21 result: `models/experiments/fresh_holdout_confirmation_seed21/metrics.json`

## Approach 2 Revisit: Holdout-Selected Neural Heads

### Protocol

After the fresh holdout test rejected the validation-tuned `81.4%` result, I revisited Approach 2 directly. The goal was to see whether the ConvNeXt neural-head solution could be improved without relying on the original validation split.

I tested:

- hidden size: focused on `1024`
- dropout: `0.0`, `0.15`
- label smoothing: `0.0`, `0.03`
- supervised contrastive weight: `0.0`, `0.005`, `0.02`
- feature normalization on/off
- shuffle vs balanced sampling in the larger profile
- seed ensembling across five neural-head initializations

All model choices were selected on a calibration split, then scored on a fresh holdout.

### Results

Seed `7` holdout sweep:

| Test | Calibration Selection | Holdout Accuracy | Holdout Macro Recall |
| --- | --- | ---: | ---: |
| Best single neural-head config | `h1024_d0_ls0.03_sup0_shuffle_norm0` | 67.6% | 59.6% |
| Best diagnostic holdout config | `h1024_d0_ls0_sup0_shuffle_norm0` | 69.4% | 62.2% |
| Best seed ensemble selected on calibration | `h768_plain` | 69.4% | 62.2% |

### Decision

The revisit did not find a generalizable improvement. The original high Approach 2 score on the face-focused validation split appears to depend heavily on the specific split and seed. Light supervised contrastive regularization sometimes improves macro recall, but it did not produce a reliable holdout accuracy gain.

Current conclusion for Approach 2:

- Useful as an experimental backend on the original validation split.
- Not yet strong enough to be the validated default model.
- Most promising future work is not more head tuning; it is better data splitting, label audit, and cleaner face-aligned training data.

### Files

- Holdout sweep script: `scripts/sweep_approach2_holdout.py`
- Seed ensemble script: `scripts/approach2_seed_ensemble_holdout.py`
- Holdout sweep result: `models/experiments/approach2_holdout_quick_seed7/metrics.json`
- Seed ensemble result: `models/experiments/approach2_seed_ensemble_seed7/metrics.json`

## Error Audit Workflow

### Rationale

After the model-tuning attempts plateaued, the next practical path is label and data-quality review. The strongest recurring failure mode is not a generic model-capacity issue; it is class overlap among broad facial-skin labels.

I generated a contact-sheet review workflow from the best diagnostic ensemble errors.

### Error Clusters

The current review queue contains `34` misclassified validation images.

| Actual Label | Error Count |
| --- | ---: |
| acne_like_texture | 10 |
| clinician_review | 1 |
| dermatitis_like_irritation | 10 |
| folliculitis_like_bumps | 8 |
| hyperpigmentation_like_uneven_tone | 1 |
| rosacea_like_redness | 4 |

Largest actual/predicted confusion pairs:

| Actual | Predicted | Count |
| --- | --- | ---: |
| acne_like_texture | dermatitis_like_irritation | 9 |
| folliculitis_like_bumps | dermatitis_like_irritation | 7 |
| dermatitis_like_irritation | folliculitis_like_bumps | 6 |
| dermatitis_like_irritation | acne_like_texture | 3 |
| rosacea_like_redness | acne_like_texture | 2 |

### Review Actions

The editable review CSV includes these intended actions:

- `keep_label`
- `relabel`
- `exclude_ambiguous`
- `exclude_low_quality`
- `exclude_non_face_or_ood`

### Decision

This is the most realistic next route toward a true `80%` result. If the audit finds mislabeled or ambiguous examples in the high-confusion groups, the corrected dataset should be rebuilt and the ConvNeXt embedding/ensemble tests rerun. The target is not to tune around noisy labels, but to make the label space cleaner and more learnable.

### Files

- Review builder: `scripts/build_error_review_sheet.py`
- Review HTML: `models/experiments/error_review_validation_best/index.html`
- Editable review CSV: `models/experiments/error_review_validation_best/review_queue.csv`
- Error summary: `models/experiments/error_review_validation_best/summary.json`
