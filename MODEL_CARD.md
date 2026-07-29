# Model Card: DermaLens

## Intended Use

DermaLens is a portfolio research prototype for private, local-first facial skin image screening. It is designed to demonstrate computer vision workflow design, privacy-preserving inference, model evaluation, and critical reporting.

It is not a medical device, diagnostic system, triage system, or replacement for clinician review.

## Model

- Deployed path: MobileNetV3-Small exported to ONNX.
- Runtime: ONNX Runtime inside a local FastAPI/Docker app.
- Inputs: one RGB face photo, resized to 224 by 224 with ImageNet normalization.
- Outputs: class probabilities over broad screening-style labels.

## Current Target Labels

- acne-like texture
- rosacea-like facial redness
- dermatitis-like irritation
- hyperpigmentation / melasma-like uneven tone
- folliculitis-like bumps
- clinician-review / uncertain

## Evaluation Status

The current deployable ONNX model reached 69.4% accuracy and 48.4% macro recall on the earlier combined validation split after conservative prior calibration. Experimental ConvNeXt ensembles reached higher validation results, but fresh holdout testing did not confirm the tuned 81.4% result.

A later methodological review identified a split-leakage risk: SCIN can contribute multiple photos per case, and older preparation code split at the image level. The corrected protocol now requires grouped train/validation splits by `case_id` and writes a `split_audit.json` artifact.

Under the corrected grouped SCIN-only protocol, the fixed deployed ONNX model with conservative prior calibration reached 86.2% +/- 1.2 accuracy and 63.1% +/- 10.1 macro recall across five split seeds. This is now classified as a contaminated fixed-model diagnostic, not a clean held-out baseline. The grouped split prevents overlap between the newly constructed folds, but the fixed ONNX model was previously fine-tuned on SCIN-derived head/neck data. Without the original model-training case list, this run cannot prove that validation cases were unseen by the deployed model.

The fair comparison has now been run with `scripts/run_grouped_mobilenet_baseline.py`: each seed prepares a grouped split, trains MobileNet only on that seed's training cases, exports ONNX, evaluates once on untouched validation cases, and aggregates support-aware metrics. The 12-seed fair MobileNet baseline reached 44.8% +/- 9.9 accuracy and 29.1% +/- 3.9 macro recall. A majority-class dermatitis baseline reached 63.7% +/- 1.6 accuracy and 16.7% macro recall.

I also added an optimistic checkpoint-selection diagnostic for MobileNet. Reselecting the best MobileNet epoch by validation accuracy from the existing training histories raises MobileNet to 52.4% +/- 5.2 accuracy and 25.1% +/- 3.3 macro recall. This is not a clean held-out estimate because it chooses from evaluation-fold history, but it checks whether the Derm Foundation lift disappears under a harder-to-beat MobileNet comparator. It does not.

### Skin-Tone Subgroup Audit

I also evaluated the same grouped SCIN splits by available Fitzpatrick and Monk tone metadata. These are workflow-demo metrics, not fairness validation: several buckets are small, SCIN tone labels are retrospective image metadata rather than controlled clinical subgroup labels, and the fixed-model evaluation itself is not a clean model holdout.

Fitzpatrick bucket summary across five grouped split seeds:

| Bucket | Mean Val Images | Accuracy | Macro Recall |
| --- | ---: | ---: | ---: |
| FST1-2 | 38.6 | 87.3% +/- 5.9 | 77.7% +/- 11.7 |
| FST3-4 | 43.6 | 88.9% +/- 5.7 | 78.6% +/- 15.4 |
| FST5-6 | 14.8 | 89.0% +/- 14.0 | 83.8% +/- 20.7 |
| Unknown | 58.6 | 83.9% +/- 5.9 | 57.5% +/- 11.9 |

Monk US bucket summary across five grouped split seeds:

| Bucket | Mean Val Images | Accuracy | Macro Recall |
| --- | ---: | ---: | ---: |
| MST1-3 | 102.4 | 87.9% +/- 2.7 | 63.3% +/- 9.6 |
| MST4-6 | 49.4 | 83.6% +/- 5.5 | 70.2% +/- 7.6 |
| MST7-10 | 4.8 | 75.0% +/- 50.0 | 75.0% +/- 50.0 |

The subgroup workflow does not show an obvious aggregate drop for darker Fitzpatrick buckets in this small SCIN-only sample, but the darkest Monk bucket is too underpowered to interpret. The right next step is not to claim fairness; it is to expand and stratify the evaluation set.

Artifact: `models/grouped_scin_subgroup_metrics.json`.

### Tail-Sensitive Head

I tested a decoupled balanced head under the same grouped SCIN protocol. The deployed ONNX image model was frozen, and only a class-balanced logistic head over the frozen logits was retrained. The refreshed artifact selects C on a nested grouped calibration split and reports 75.3% +/- 1.7 accuracy and 70.7% +/- 11.4 macro recall. This should be read as a fixed-encoder operating-point result, not as a clean representation benchmark, because the frozen deployed encoder may include information from original SCIN-derived fine-tuning cases.

Artifact: `models/grouped_scin_decoupled_logit_head_metrics.json`.

I also ran a Derm Foundation embedding evaluation using `google/derm-foundation` as the frozen representation with the same grouped/nested protocol. After the reviewer caveat about five-seed inference, I expanded the matched comparison to 10 and 12 completed seeds. The 12-seed class-balanced linear probe reached 69.8% +/- 5.2 accuracy and 34.7% +/- 5.0 macro recall. Compared with the fair fold-retrained MobileNet baseline, paired seeds support a large accuracy gain: +25.0 points, 95% CI +17.8 to +32.2, exact sign p=0.00049. The macro-recall gain is smaller but positive in the expanded run: +5.6 points, 95% CI +2.6 to +8.6, exact sign p=0.00635. These p-values are repeated-split stability diagnostics rather than independent external-test-set inference. Derm Foundation still loses mean recall on rosacea and hyperpigmentation, so this is not a solved tail-label classifier.

Artifacts: `models/grouped_scin_derm_foundation_embedding_12seed_metrics.json` and `models/grouped_scin_seed_count_sensitivity_metrics.json`.

Additional diagnostic artifact: `models/grouped_scin_mobilenet_checkpoint_selection_diagnostic.json`.

## Known Limitations

- Broad labels overlap visually, especially acne, folliculitis, and dermatitis-like irritation.
- Public datasets are noisy and not fully face-specific.
- Performance has not been clinically validated.
- The reported 86.2% grouped SCIN fixed-model check is contaminated as a model-holdout estimate unless the original ONNX training cases can be excluded or the model is retrained per grouped fold.
- Several tail labels have validation support too small for meaningful mean +/- std recall. Metrics for labels with fewer than about 10 validation images should be treated as undefined/underpowered diagnostics.
- Performance may vary by lighting, camera processing, makeup, filters, and skin tone. The current subgroup workflow is underpowered for the darkest Monk bucket.
- Region summaries use an OpenCV frontal-face detector with a geometry fallback; this is better than the original fixed crop but still not a landmark-grade facial analysis pipeline.

## Safety Behavior

The UI and API present outputs as non-diagnostic screening observations. The app strips EXIF metadata, binds to localhost by default, and does not retain uploaded photos unless `SAVE_UPLOADS=true`.

## Recommended Next Evaluation

1. Rebuild manifests with strict label confidence settings.
2. Prepare ImageFolder data with grouped `case_id` splitting.
3. Treat the 12-seed fair fold-retrained MobileNetV3 result, not the fixed-model diagnostic, as the baseline for future modeling comparisons.
4. Rerun decoupled-head probes against fair representations or mark them as fixed-encoder experiment logs.
5. Rerun Derm Foundation with `--embedding-source local-model` when compute allows, so the SCIN downstream evaluation does not depend on Google's shipped SCIN precomputed embedding file.
6. Add enough validated face-specific tail examples before treating macro recall as stable evidence.
