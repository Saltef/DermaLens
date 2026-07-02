# Model Card: DermaLens Local

## Intended Use

DermaLens Local is a portfolio research prototype for private, local-first facial skin image screening. It is designed to demonstrate computer vision workflow design, privacy-preserving inference, model evaluation, and critical reporting.

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

A later methodological review identified a split-leakage risk: SCIN can contribute multiple photos per case, and older preparation code split at the image level. The corrected protocol now requires grouped train/validation splits by `case_id` and writes a `split_audit.json` artifact. New headline metrics should be reported only after rerunning under this grouped protocol.

## Known Limitations

- Broad labels overlap visually, especially acne, folliculitis, and dermatitis-like irritation.
- Public datasets are noisy and not fully face-specific.
- Performance has not been clinically validated.
- Performance may vary by lighting, camera processing, makeup, filters, and skin tone.
- Region summaries are approximate and do not use a landmark-based face detector yet.

## Safety Behavior

The UI and API present outputs as non-diagnostic screening observations. The app strips EXIF metadata, binds to localhost by default, and does not retain uploaded photos unless `SAVE_UPLOADS=true`.

## Recommended Next Evaluation

1. Rebuild manifests with strict label confidence settings.
2. Prepare ImageFolder data with grouped `case_id` splitting.
3. Rerun baseline ONNX and frozen-embedding experiments.
4. Report accuracy, macro recall, per-class recall, seed variance, and confidence intervals.
5. Add subgroup metrics by available Fitzpatrick or Monk skin-tone metadata.
