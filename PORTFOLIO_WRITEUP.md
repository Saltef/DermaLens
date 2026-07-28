# DermaLens: Private Facial Skin Screening with Local Vision Models

## Executive Summary

DermaLens is a portfolio machine-learning project that asks a practical question: how far can a privacy-preserving facial skin screening system go when inference runs locally and the available labeled data is small, imbalanced, and noisy?

The final application is a Dockerized local web app. A user uploads a facial image, the system strips metadata, runs local inference, and returns cautious screening-style signals for broad facial skin patterns such as acne-like texture, dermatitis-like irritation, folliculitis-like bumps, rosacea-like redness, hyperpigmentation-like uneven tone, and clinician-review cases.

The strongest untuned experimental model reached **79.2% accuracy and 71.0% macro recall** on the original face-focused validation split. A validation-tuned calibration pass reached **81.4% accuracy**, but fresh holdout testing did not reproduce that result. I therefore rejected the 81.4% number as an overfit diagnostic result rather than presenting it as the project outcome.

After a later technical review, I also found a more fundamental validity risk: the original dataset preparation could split multiple photos from the same SCIN case across train and validation. I fixed this by making `case_id` grouped splitting the default protocol and by writing an auditable `split_audit.json` for every prepared ImageFolder dataset.

I then ran a post-correction check on SCIN-only grouped splits. The fixed deployed ONNX model, using the same conservative prior calibration as the Docker app, reached **86.2% +/- 1.2 accuracy** and **63.1% +/- 10.1 macro recall** across five grouped split seeds. A later audit found that this is not a clean held-out model result: the deployed model had already been fine-tuned on SCIN-derived head/neck images, and the grouped split only prevents overlap inside the new folds. Without the original training-case list, this check cannot exclude cases the model may have seen.

I also added a skin-tone subgroup workflow using SCIN Fitzpatrick and Monk metadata. The workflow did not show an obvious aggregate drop across Fitzpatrick buckets in the small SCIN-only sample, but the darkest Monk bucket was too small to interpret. I treat that as a process win rather than a fairness claim: the project now has the machinery to report subgroup performance and the discipline to say when the subgroup data is underpowered.

Finally, I ran one modeling improvement under the corrected grouped split: a decoupled cRT-style head. I froze the deployed ONNX image model, used its logits as a compact representation, and retrained only a class-balanced logistic head on each grouped split. The artifact reports **75.1% accuracy** and **73.1% macro recall**, but I no longer frame it as a clean lift over the fixed ONNX result because that fixed ONNX result is contaminated as a model-holdout baseline. A later review also found that this artifact selected C on the evaluation fold, so the code has been corrected to use a nested grouped calibration split and the number should be refreshed before being treated as final.

That decision changed the project story: the strongest remaining gap was a fair fold-retrained MobileNetV3 baseline. I ran that baseline across five grouped SCIN seeds. It reached **48.0% +/- 3.2 accuracy** and **30.2% +/- 5.3 macro recall**, far below the contaminated fixed-model diagnostic. Against that fair baseline, the Derm Foundation probe became a clean modeling result: **66.8% +/- 6.9 accuracy** and **33.8% +/- 5.9 macro recall**, a Pareto lift of **+18.8 accuracy points** and **+3.7 macro-recall points**. The model is still not clinically strong, but the experiment now demonstrates both scientific correction and measurable improvement.

This is not a medical device and does not provide diagnosis.

## What I Built

- A local-first facial skin screening app using FastAPI, Docker, ONNX Runtime, and a static frontend.
- A privacy layer that re-encodes uploaded images to strip EXIF/GPS metadata and avoids photo retention by default.
- A deployable compact ONNX baseline using MobileNetV3-Small.
- A research pipeline for comparing frozen vision embeddings, neural classifier heads, supervised contrastive learning, targeted augmentation, prior calibration, and ensemble methods.
- A validation and error-audit workflow that surfaces where the model fails and what data would be needed next.
- A corrected data-splitting protocol that prevents case-level leakage and records split metadata for review.

## Why This Problem Is Hard

Facial skin images are sensitive, and many consumer-facing dermatology demos assume cloud upload. A local-first design solves a privacy problem, but it creates an engineering constraint: the model must be small enough to run on consumer hardware and easy to deploy inside Docker.

The data problem is more difficult. The labels are broad and visually overlapping. Acne-like texture, folliculitis-like bumps, and dermatitis-like irritation are not cleanly separable in many consumer photos, especially when lighting, camera quality, facial crop, and symptom severity vary. A model can appear to improve by learning the majority class, but that is not useful for a screening tool. For that reason, I tracked both accuracy and macro recall throughout the project.

## Experimental Strategy

I treated this as an applied ML study rather than a single training run.

### 1. Deployable Baseline

The first deployable model was a MobileNetV3-Small classifier exported to ONNX. It is fast, portable, and appropriate for local inference.

| Model | Accuracy | Macro Recall |
| --- | ---: | ---: |
| Raw MobileNetV3 ONNX | 68.3% | 44.4% |
| Conservative prior-calibrated MobileNetV3 ONNX | 69.4% | 48.4% |
| Conservative MobileNetV3 ONNX on grouped SCIN-only splits, 5 split seeds | 86.2% +/- 1.2 | 63.1% +/- 10.1 | Fixed-model diagnostic; not clean model holdout |
| Fair grouped MobileNetV3 retrain, 5 split seeds | 48.0% +/- 3.2 | 30.2% +/- 5.3 | Clean baseline |
| Derm Foundation linear probe, 5 grouped split seeds | 66.8% +/- 6.9 | 33.8% +/- 5.9 | Fair Pareto lift over retrained MobileNet |
| Decoupled balanced logit head on grouped SCIN-only splits, 5 split seeds | 75.1% +/- 2.0 | 73.1% +/- 10.1 | Pre-nested C-selection artifact; not final |

The grouped SCIN-only fixed-model result avoids case overlap inside each newly constructed split, but it does not answer the key validity question because the deployed model was already trained on SCIN-derived data. The fair comparison is the fold-retrained MobileNetV3 result: train a new model on each grouped training fold and evaluate each fold's untouched validation cases.

The decoupled head may move tail behavior in the right direction, but the artifact predates nested C-selection and its comparison baseline is not clean. I treat it as a strong experimental lead rather than a validated improvement.

I also ran the most important next experiment: a direct Derm Foundation embedding probe. It used `google/derm-foundation` as the frozen representation, trained a class-balanced linear classifier, selected C on nested grouped calibration data, and evaluated once on the held-out grouped fold. The probe reached **66.8% +/- 6.9 accuracy** and **33.8% +/- 5.9 macro recall**. Compared against the fair MobileNet baseline, Derm Foundation improved both accuracy and macro recall.

The narrower result is still important: a simple linear probe over Derm Foundation embeddings improved the representation, but it did not solve the current mapped tail-label problem. Hyperpigmentation remained at 0.0 mean recall and the validation support for several tail classes is too small for stable claims.

### 2. Frozen Foundation-Style Embeddings

Recent medical-vision work often uses strong pretrained representations with smaller downstream classifiers. I tested this locally using ImageNet-pretrained backbones as practical proxies for foundation-style embeddings.

I compared MobileNetV3, EfficientNet-B0, ConvNeXt-Tiny, Swin-T, and ViT-B/16 by freezing the image encoder and training balanced logistic regression on the extracted embeddings.

| Backbone | Accuracy | Macro Recall | Interpretation |
| --- | ---: | ---: | --- |
| MobileNetV3-Small | 61.8% | 49.3% | Lightweight control. |
| EfficientNet-B0 | 67.2% | 52.1% | Better class balance. |
| ConvNeXt-Tiny | 68.9% | 61.6% | Best balanced representation. |
| Swin-T | 65.0% | 52.8% | Did not win here. |
| ViT-B/16 | 65.0% | 52.4% | Larger but not better on this data. |

ConvNeXt-Tiny became the best feature extractor for later experiments because it improved macro recall without simply leaning harder into the majority class.

### 3. Long-Tail Neural Heads and Supervised Contrastive Learning

Because the dataset is imbalanced, I tested long-tail classification techniques inspired by supervised contrastive learning and representation regularization. The hypothesis was that same-class clustering would help rare labels resist being absorbed into the dominant dermatitis-like class.

I tested:

- balanced cross-entropy
- balanced sampling
- supervised contrastive loss
- projection heads
- dropout
- label smoothing
- seed ensembling
- calibration/holdout selection

The strongest observed validation result came from a ConvNeXt embedding head rather than a positive supervised-contrastive term:

| Experiment | Accuracy | Macro Recall |
| --- | ---: | ---: |
| MobileNetV3 calibrated baseline | 69.4% | 48.4% |
| ConvNeXt frozen logistic head | 68.9% | 61.6% |
| ConvNeXt neural head, best observed validation run | 78.7% | 67.7% |

The negative finding matters: supervised contrastive loss sometimes improved macro recall, but it usually reduced overall accuracy. The best setup was a larger neural classifier head over ConvNeXt embeddings with balanced cross-entropy and normal shuffled batches.

When I retested this family on fresh calibration/holdout splits, the results dropped into the high 60s to low 70s. That suggests the original validation result was useful but not stable enough to claim as the validated model.

### 4. Targeted Tail-Class Augmentation

I then tested whether adding more minority-class examples would help. This was not treated as a simple class-count problem. I preserved the original validation split and used auxiliary data only for training.

| Experiment | Accuracy | Macro Recall | Interpretation |
| --- | ---: | ---: | --- |
| Targeted augmentation + ConvNeXt logistic head | 57.4% | 55.0% | More data hurt because the source distribution shifted. |
| Targeted augmentation + neural head | 62.3% | 57.9% | Better, but still below base-data models. |

This was an important failed experiment. It showed that broad or body-oriented augmentation data is not a substitute for face-aligned examples. More images are not automatically better if they move the training distribution away from the deployment setting.

### 5. Mixed Ensemble and Calibration

The best original-validation result came from combining ConvNeXt-based classifiers:

- frozen ConvNeXt logistic heads
- ConvNeXt neural heads
- one weak but diverse augmented-data neural head

| Model | Accuracy | Macro Recall |
| --- | ---: | ---: |
| Calibrated MobileNetV3 app model | 69.4% | 48.4% |
| Best single ConvNeXt neural head | 78.7% | 67.7% |
| Mixed ConvNeXt ensemble | 79.2% | 71.0% |
| Validation-tuned class-bias ensemble | 81.4% | 74.6% |

At first, the 81.4% result looked like a breakthrough. I then ran fresh holdout confirmation where ensemble weights and class bias were selected on calibration splits and evaluated on untouched holdout splits.

| Fresh Split Seed | Bias-Calibrated Holdout Accuracy | Macro Recall |
| --- | ---: | ---: |
| 42 | 69.4% | 55.9% |
| 7 | 73.0% | 65.2% |
| 13 | 64.0% | 47.8% |
| 21 | 66.7% | 54.8% |

The 81.4% result did not generalize. I kept it in the report as a diagnostic upper bound and explicitly rejected it as a validated performance claim.

## Main Limitation: The Model Needs Better Data

The strongest error pattern is acne/folliculitis/dermatitis confusion. In the error audit, the largest confusion pairs were:

| Actual | Predicted | Count |
| --- | --- | ---: |
| acne_like_texture | dermatitis_like_irritation | 9 |
| folliculitis_like_bumps | dermatitis_like_irritation | 7 |
| dermatitis_like_irritation | folliculitis_like_bumps | 6 |

This is where additional data is needed. Specifically, the project needs:

- more face-specific examples for acne-like texture, folliculitis-like bumps, and rosacea-like redness
- cleaner labeling guidelines for overlapping inflammatory presentations
- a normal/low-concern class to reduce forced classification
- label review by someone with dermatology expertise
- a larger independent test set stratified by lighting, skin tone, camera quality, and condition severity

The current model is not mainly limited by whether the classifier head is linear, neural, contrastive, or ensembled. It is limited by the ambiguity and sparsity of the supervised signal.

An important methodological limitation was also discovered after the first round of experiments: SCIN can include several images per case, and the initial fallback split operated at the image level. That can inflate validation scores in medical imaging because same-case photos may be visually near-duplicate. I corrected the preparation code to split by case/group ID, added an overlap assertion, and added split audit metadata. A later audit found a second limitation: fixed-model evaluation on newly grouped SCIN folds is still not clean if the fixed model had already been trained on the same case universe. The next correction is fold-level retraining.

## What This Project Demonstrates

This project demonstrates the full applied ML loop:

- privacy-aware product architecture
- local inference deployment with Docker and ONNX Runtime
- dataset construction and class mapping from public dermatology sources
- evaluation with both accuracy and macro recall
- model comparison across compact CNNs, pretrained embedding backbones, transformer-style backbones, neural heads, and ensemble methods
- literature-informed experiments such as long-tail supervised contrastive learning and frozen foundation-style representations
- calibration, holdout confirmation, and rejection of an overfit result
- discovery of both image-level split leakage and fixed-model evaluation contamination
- subgroup evaluation workflow by available skin-tone metadata, with underpowered buckets explicitly demoted
- decoupled balanced-head and Derm Foundation experiments that are now framed as leads pending a fair fold-retrained baseline
- error analysis that turns model failure into a concrete data acquisition plan

The most important outcome is not just a score. It is a defensible process: when a stronger critique found the headline was contaminated, the repo demoted the claim instead of defending it. The next result that matters is a fair grouped retrain/evaluate baseline.

## Local Demo

```powershell
docker compose up --build
```

Open:

```text
http://127.0.0.1:8765
```

The app runs locally, strips uploaded image metadata, performs inference without cloud upload, and returns cautious non-diagnostic findings.
