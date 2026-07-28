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

The missing fair comparison is now implemented in `scripts/run_grouped_mobilenet_baseline.py`: each seed prepares a grouped split, trains MobileNet only on that seed's training cases, exports ONNX, evaluates once on untouched validation cases, and aggregates support-aware metrics. A one-seed, one-epoch CPU smoke run completed at 44.7% accuracy and 30.7% macro recall. This is a harness-validation artifact, not a deployable performance claim.

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

I tested a decoupled balanced head under the same grouped SCIN protocol. The deployed ONNX image model was frozen, and only a class-balanced logistic head over the frozen logits was retrained. The artifact reports 75.1% +/- 2.0 accuracy and 73.1% +/- 10.1 macro recall, but it should not be compared as a clean improvement over the 86.2% fixed-model diagnostic because that diagnostic may include cases seen during original model training. A later review also found that this artifact selected C on the evaluation fold; the script now performs C-selection on a nested grouped calibration split and should be rerun before this is treated as a final score.

Artifact: `models/grouped_scin_decoupled_logit_head_metrics.json`.

I also ran a Derm Foundation embedding evaluation using `google/derm-foundation` as the frozen representation with the same grouped/nested protocol. The class-balanced linear probe reached 66.8% +/- 6.9 accuracy and 33.8% +/- 5.9 macro recall. This is a completed experiment, but it does not support the stronger claim that Derm Foundation is worse than a fair MobileNet baseline, because the fair fold-retrained MobileNet baseline has only been smoke-tested so far. The narrower conclusion is that a simple linear probe over Derm Foundation embeddings did not solve the current mapped tail-label problem.

Artifact: `models/grouped_scin_derm_foundation_embedding_metrics.json`.

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
3. Run `scripts/run_grouped_mobilenet_baseline.py` for the full 5-seed MobileNetV3 retrain.
4. Compare Derm Foundation and decoupled-head probes against that fold-retrained baseline.
5. Report accuracy, macro recall, per-class recall, seed variance, confidence intervals, and per-class support.
