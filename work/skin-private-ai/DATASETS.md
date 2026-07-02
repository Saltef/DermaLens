# Dataset Plan

We do not keep training data in this repository. Public dermatology datasets contain sensitive medical images, so datasets should live outside source control under `data/raw/` and `data/processed/`.

## Recommended First Dataset

### SCIN

Best first choice for this project.

- Source: https://github.com/google-research-datasets/scin
- Size: 10,000+ dermatology images from 5,000+ volunteer contributions.
- Strengths: real-world consumer-style images, dermatologist labels, symptom/history metadata, estimated Fitzpatrick Skin Type, estimated Monk Skin Tone.
- Fit for us: closer to phone-upload skin photos than dermoscopy datasets.
- Caution: confirm the SCIN Data Use License before training or redistribution.

Use SCIN first for:

- acne-like/inflammatory eruptions
- dermatitis/eczema-like conditions
- infectious/inflammatory common conditions
- skin-tone stratified evaluation

## Recommended Evaluation Dataset

### DDI

Use for fairness and robustness evaluation, not as the main training set.

- Source: https://ddi-dataset.github.io/
- Size: 656 images from 570 patients.
- Strengths: diverse skin tones, biopsy-proven labels, expert curation.
- Restriction: personal, non-commercial research use only; do not redistribute.

## Optional Training Datasets

## Deep Search Findings

Ranked by usefulness for this private, face-photo-oriented portfolio app:

| Dataset | Best Use | Fit For Current Broad Model | Access / License Notes |
| --- | --- | --- | --- |
| SCIN | Primary broad training backbone | High | Public Google dataset with real-world contributed skin-condition photos, dermatologist labels, eFST/eMST metadata. Keep checking the SCIN Data Use License before redistribution. |
| Fitzpatrick17k | Supplemental broad training and skin-tone robustness | Medium | Clinical images and FST labels; CC BY-NC-SA 3.0. Many source URLs are broken and labels need careful mapping. |
| DDI | Fairness and robustness evaluation | Medium for eval, low for training | Diverse, biopsy-proven, 656 images. Strong evaluation set; access/use is restricted and non-commercial/research oriented. |
| PAD-UFES-20 | `clinician_review` / lesion-risk branch | Low for broad facial issues, high for lesion branch | Smartphone clinical lesion images, CC BY 4.0. Do not mix into acne/rosacea/dermatitis classes. |
| ACNE04 | Acne severity branch | Low for broad model, high for acne branch | Facial acne severity/lesion-count data. We already downloaded and prepared it locally. |
| SkinCon | Concept labels/explainability | Indirect | Dense dermatologist concept labels over Fitzpatrick17k and DDI, useful for concepts like scale, plaque, erosion, papules. |
| SkinCAP | Caption/explainer layer | Indirect | Dermatology images with rich captions; not a replacement for classifier labels. |
| DermaSynth | VLM/explainer pretraining | Indirect | Synthetic image-text pairs built from open dermatology datasets. Better for local VLM/explanation than the primary classifier. |
| MM-Skin | VLM/explainer research | Indirect | Large multimodal dermatology dataset from textbooks and public datasets. Useful research direction, but not ideal for our compact ONNX classifier. |
| Derm1M / DermLIP | Future foundation-model direction | Indirect | Large-scale image-text dermatology work; useful if weights/data become practically accessible. |
| HAM10000 / ISIC / BCN20000 / DERM12345 / HIBA | Lesion/dermoscopy branch | Low for face/common-condition model | Mostly dermoscopy and pigmented lesion focused. Useful for lesion triage only, not acne/rosacea/dermatitis. |
| FFHQ/CelebA-style face datasets | Possible `normal_or_low_concern` negatives | Risky | Non-medical face datasets can bias the model toward celebrity/photo-quality artifacts and have consent/licensing concerns. Use only after careful license review. |

The biggest missing dataset for this project is still a clean, licensed set of face photos labeled for rosacea, dermatitis, hyperpigmentation/melasma, folliculitis, healthy/low-concern, and acne. Public datasets do not currently cover that combination well.

### ACNE04

Best next dataset to investigate for the face-focused version of this project.

- Source: referenced by the acne LDL implementation: https://github.com/openface-io/acne-lds
- Size: 1,457 facial acne images with lesion-count annotations and severity labels, plus 18,983 lesion boxes reported in the ISBI 2024 paper.
- Strengths: directly aligned with facial acne grading from normal photos.
- Caution: access may route through the original ACNE04 release rather than the LDS repository; confirm license/terms before downloading or committing derived artifacts.

Use ACNE04 for:

- a dedicated acne severity branch
- lesion-count-aware training
- testing label distribution smoothing or ordinal severity losses

Do not collapse ACNE04 directly into the broad `acne_like_texture` class if we want severity. Keep severity/count metadata so the model can learn more than "acne vs not acne."

Expected local layout after downloading and unpacking the official archive:

```text
data/raw/acne04/
  VOCdevkit2007/
    VOC2007/
      JPEGImages_300/
      ImageSets/
        Main/
          NNEW_trainval_0.txt
          NNEW_test_0.txt
```

Build ACNE04 manifests:

```powershell
python scripts/acne04_build_manifest.py `
  --raw-dir data/raw/acne04 `
  --output-dir data/raw/acne04 `
  --fold 0
```

Prepare a severity ImageFolder split:

```powershell
python scripts/prepare_imagefolder.py `
  --manifest data/raw/acne04/acne04_manifest.csv `
  --image-root data/raw/acne04 `
  --output data/processed/acne04_severity_fold0_v1 `
  --label-column label `
  --path-column image_path `
  --split-column split
```

Train an acne severity model:

```powershell
docker compose -f docker-compose.train-gpu.yml run --rm trainer-gpu `
  python scripts/train_export_onnx.py `
    --data-dir data/processed/acne04_severity_fold0_v1 `
    --output-dir models/experiments/acne04_mobilenet_v1 `
    --model mobilenet_v3_small `
    --epochs 30 `
    --batch-size 16 `
    --lr 0.0001 `
    --class-weights balanced `
    --label-smoothing 0.05
```

### AcneSCU / ACNE-DET

Potentially useful for acne lesion detection rather than whole-image classification.

- AcneSCU source reference: https://github.com/pingguokiller/acnedetection
- Strengths: high-resolution acne lesion annotations and detection-oriented research baseline.
- Caution: verify dataset availability, release conditions, and redistribution limits before use.

Use these only if we add a second model head for localization/counting. They are less useful for the current ImageFolder classifier unless converted carefully.

### Fitzpatrick17k

Useful for broad dermatology coverage and skin-type metadata.

- Source: https://github.com/mattgroh/fitzpatrick17k
- Size: 16,577 clinical images, 114 skin conditions.
- License: Creative Commons Attribution-NonCommercial-ShareAlike 3.0.
- Caution: not face-specific, links may be broken, label quality varies by source.

Use Fitzpatrick17k for:

- improving broad condition recognition
- better skin-type metadata coverage
- testing whether the classifier overfits SCIN label language

Do not mix it blindly with SCIN labels. Build a mapping table first because Fitzpatrick17k uses many more specific diagnoses.

### PAD-UFES-20

Useful because it contains smartphone clinical images rather than dermoscopy-only images.

- Source: https://data.mendeley.com/datasets/zr7vgbcyr2/1
- Size: 2,298 images from 1,641 skin lesions and 1,373 patients.
- Strengths: smartphone images and clinical metadata.
- Caution: lesion/cancer-oriented, not face-condition oriented.

Use PAD-UFES-20 only for a separate `clinician_review` or lesion-risk branch. Do not use it to improve acne/rosacea/dermatitis labels.

## Optional Explanation Datasets

### SkinCon

Useful for morphology/concept labels, not primary diagnosis.

- Source: https://github.com/Imageomics/SkinCon
- Contains dermatology concepts such as scale, plaque, erosion, crust, papule, and color descriptors.
- Fit for us: better explanations and UI descriptions.

### SkinCAP

Useful for caption-style explanations.

- Source: https://huggingface.co/datasets/joshuachou/SkinCAP
- Contains 4,000 dermatology images with rich medical captions.
- Fit for us: training/evaluating an explainer, not replacing the classifier.

## Datasets To Treat Carefully

### DermNet / DermAtlas

Useful medical references, but do not scrape images unless their terms explicitly permit the exact use. These are better as references or manually reviewed examples than as bulk training data.

## Avoid For This MVP

### ISIC / HAM10000

Excellent datasets, but mostly dermoscopy/pigmented lesion focused. They are less aligned with consumer face photos and common facial issues like acne, rosacea-like redness, dermatitis, or hyperpigmentation.

## Local Folder Layout

```text
data/
  raw/
    scin/
    ddi/
    fitzpatrick17k/
  processed/
    face_skin_v1/
      train/
        acne_like_texture/
        rosacea_like_redness/
        dermatitis_like_irritation/
        hyperpigmentation_like_uneven_tone/
        folliculitis_like_bumps/
        clinician_review/
      val/
        acne_like_texture/
        rosacea_like_redness/
        dermatitis_like_irritation/
        hyperpigmentation_like_uneven_tone/
        folliculitis_like_bumps/
        clinician_review/
```

## Label Strategy

For v1, keep labels broad and screening-oriented:

- `acne_like_texture`
- `rosacea_like_redness`
- `dermatitis_like_irritation`
- `hyperpigmentation_like_uneven_tone`
- `folliculitis_like_bumps`
- `clinician_review`

Do not train the model to output precise medical diagnoses until we have enough verified labels and validation.

## Minimum Bar Before Training

Aim for at least:

- 300+ images per class for a rough first model.
- 1,000+ images per class for something more stable.
- Validation split stratified by label and skin tone when possible.
- No duplicate images across train and validation.
- Evaluation broken down by skin tone, image quality, and body location.

## Commands

Prepare ImageFolder data after collecting a CSV manifest:

```powershell
python scripts/scin_build_manifest.py `
  --download-metadata `
  --download-images `
  --max-per-label 500

python scripts/prepare_imagefolder.py `
  --manifest data/raw/scin/face_skin_manifest.csv `
  --image-root data/raw/scin `
  --output data/processed/face_skin_v1 `
  --label-column label `
  --path-column image_path
```

Train and export ONNX:

```powershell
python scripts/train_export_onnx.py `
  --data-dir data/processed/face_skin_v1 `
  --output-dir models `
  --model mobilenet_v3_small `
  --epochs 8
```

Build a capped Fitzpatrick17k manifest:

```powershell
python scripts/fitzpatrick_build_manifest.py `
  --download-metadata `
  --download-images `
  --max-per-label 150

python scripts/prepare_imagefolder.py `
  --manifest data/raw/fitzpatrick17k/manifest.csv `
  --image-root data/raw/fitzpatrick17k `
  --output data/processed/fitzpatrick_mapped_v1 `
  --label-column label `
  --path-column image_path
```

Build a targeted augmented broad dataset:

This keeps validation face-focused from `scin_headneck_plus_fitzpatrick_v1`, but adds train-only minority examples from the larger SCIN all-body pool. It is meant to improve weak classes without inflating validation.

```powershell
python scripts/build_targeted_augmented_imagefolder.py `
  --base data/processed/scin_headneck_plus_fitzpatrick_v1 `
  --augment data/processed/skin_balanced_body_v1 `
  --output data/processed/broad_targeted_aug_v1
```

Train the targeted broad model:

```powershell
docker compose -f docker-compose.train-gpu.yml run --rm trainer-gpu `
  python -u scripts/train_export_onnx.py `
    --data-dir data/processed/broad_targeted_aug_v1 `
    --output-dir models/experiments/mobilenet_broad_targeted_aug_10e `
    --model mobilenet_v3_small `
    --epochs 10 `
    --batch-size 16 `
    --num-workers 0 `
    --lr 0.00008 `
    --class-weights balanced `
    --label-smoothing 0.05 `
    --random-erasing 0.05 `
    --select-metric val_accuracy
```

Evaluate it against the unchanged face-focused validation split:

```powershell
docker compose -f docker-compose.train-gpu.yml run --rm trainer-gpu `
  python scripts/evaluate_onnx.py `
    --model models/experiments/mobilenet_broad_targeted_aug_10e/skin_classifier.onnx `
    --label-map models/experiments/mobilenet_broad_targeted_aug_10e/label_map.json `
    --data-dir data/processed/broad_targeted_aug_v1/val `
    --output models/experiments/mobilenet_broad_targeted_aug_10e/eval_targeted_aug_val.json
```
