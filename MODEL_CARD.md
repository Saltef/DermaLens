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

Under the corrected grouped SCIN-only protocol, the fixed deployed ONNX model with conservative prior calibration reached 86.2% +/- 1.2 accuracy and 63.1% +/- 10.1 macro recall across five split seeds. This is the cleanest deployed-model baseline currently reported, but it is still limited by small tail-class validation counts.

### Skin-Tone Subgroup Audit

I also evaluated the same grouped SCIN splits by available Fitzpatrick and Monk tone metadata. These are audit metrics, not fairness validation: several buckets are small and SCIN tone labels are image/dataset metadata rather than controlled clinical subgroup labels.

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

The subgroup audit does not show an obvious aggregate drop for darker Fitzpatrick buckets in this small SCIN-only sample, but the darkest Monk bucket is too underpowered to interpret. The right next step is not to claim fairness; it is to expand and stratify the evaluation set.

Artifact: `models/grouped_scin_subgroup_metrics.json`.

### Tail-Sensitive Head

I tested a decoupled balanced head under the same grouped SCIN protocol. The deployed ONNX image model was frozen, and only a class-balanced logistic head over the frozen logits was retrained. This improved macro recall from 63.1% +/- 10.1 to 73.1% +/- 10.1 across five grouped split seeds, mainly by lifting clinician-review, folliculitis, hyperpigmentation, and rosacea recall. Accuracy dropped from 86.2% +/- 1.2 to 75.1% +/- 2.0, so this is documented as a tail-sensitive operating point rather than the default app model. A later review found that this artifact selected C on the evaluation fold; the script now performs C-selection on a nested grouped calibration split and should be rerun before this is treated as the final refreshed score.

Artifact: `models/grouped_scin_decoupled_logit_head_metrics.json`.

I also ran a Derm Foundation embedding evaluation using `google/derm-foundation` as the frozen representation with the same grouped/nested protocol. This did not produce a Pareto improvement. The class-balanced linear probe reached 66.8% +/- 6.9 accuracy and 33.8% +/- 5.9 macro recall, below the deployed grouped baseline at 86.2% +/- 1.2 accuracy and 63.1% +/- 10.1 macro recall. The main failure was tail recall: hyperpigmentation stayed at 0.0 mean recall, and folliculitis/rosacea remained weak.

Artifact: `models/grouped_scin_derm_foundation_embedding_metrics.json`.

## Known Limitations

- Broad labels overlap visually, especially acne, folliculitis, and dermatitis-like irritation.
- Public datasets are noisy and not fully face-specific.
- Performance has not been clinically validated.
- Performance may vary by lighting, camera processing, makeup, filters, and skin tone. The current subgroup audit is underpowered for the darkest Monk bucket.
- Region summaries use an OpenCV frontal-face detector with a geometry fallback; this is better than the original fixed crop but still not a landmark-grade facial analysis pipeline.

## Safety Behavior

The UI and API present outputs as non-diagnostic screening observations. The app strips EXIF metadata, binds to localhost by default, and does not retain uploaded photos unless `SAVE_UPLOADS=true`.

## Recommended Next Evaluation

1. Rebuild manifests with strict label confidence settings.
2. Prepare ImageFolder data with grouped `case_id` splitting.
3. Rerun baseline ONNX and frozen-embedding experiments.
4. Report accuracy, macro recall, per-class recall, seed variance, and confidence intervals.
5. Add subgroup metrics by available Fitzpatrick or Monk skin-tone metadata.
