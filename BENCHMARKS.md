# Benchmark Ledger

DermaLens uses benchmarks in two different ways:

1. **Executed internal benchmarks** for the face-focused SCIN task.
2. **External dermatology benchmarks** used to position future evaluation, not to inflate the current facial-condition result.

## Executed Benchmarks

| Benchmark | Status | Result | Validity |
| --- | --- | ---: | --- |
| Fair grouped MobileNetV3 retrain on SCIN | Completed | 48.0% +/- 3.2 accuracy, 30.2% +/- 5.3 macro recall | Clean internal baseline. Each seed trains on grouped training cases and evaluates untouched validation cases. |
| Derm Foundation linear probe on grouped SCIN | Completed | 66.8% +/- 6.9 accuracy, 33.8% +/- 5.9 macro recall | Clean representation comparison. Pareto lift over fair MobileNet, but tail labels remain underpowered. |
| Nested decoupled logit head on grouped SCIN | Completed | 75.3% +/- 1.7 accuracy, 70.7% +/- 11.4 macro recall | Fixed-encoder operating point. Not a clean representation benchmark because the frozen deployed encoder was previously trained on SCIN-derived data. |
| Fixed deployed ONNX on grouped SCIN | Diagnostic only | 86.2% +/- 1.2 accuracy, 63.1% +/- 10.1 macro recall | Contaminated as model-holdout evidence; grouped folds do not exclude original model-training cases. |

Primary artifacts:

- `models/grouped_scin_mobilenet_retrained_baseline_metrics.json`
- `models/grouped_scin_derm_foundation_embedding_metrics.json`
- `models/grouped_scin_fair_model_comparison_metrics.json`
- `models/grouped_scin_decoupled_logit_head_metrics.json`
- `models/grouped_scin_clean_split_metrics.json`

## External Benchmark Positioning

| Benchmark | Why It Matters | Why It Is Not The Current Headline |
| --- | --- | --- |
| [ISIC Challenge / ISIC Archive](https://challenge.isic-archive.com/data/) | Standard melanoma and skin-lesion benchmark family with public challenge splits and broad community usage. | Mostly lesion/dermoscopy oriented. Useful for a future lesion-risk branch, not for claiming improvement on facial acne/rosacea/dermatitis labels. |
| [HAM10000](https://pmc.ncbi.nlm.nih.gov/articles/PMC6091241/) | Widely used pigmented-lesion dataset with 10,015 dermoscopic images and standard comparison value. | Dermoscopy/pigmented lesion task, not consumer-style facial inflammatory/pigmentary concerns. |
| [DDI](https://ddi-dataset.github.io/index.html) | Diverse, biopsy-proven clinical image benchmark designed for skin-tone performance evaluation. | Strong external fairness/robustness benchmark, but restricted to personal non-commercial research and lesion-focused labels. |
| [PAD-UFES-20](https://pmc.ncbi.nlm.nih.gov/articles/PMC7479321/) | Smartphone skin-lesion dataset with patient metadata and 2,298 images from 1,641 lesions. | More aligned with phone images than dermoscopy datasets, but still lesion/cancer oriented rather than face-condition oriented. |
| [SCIN](https://github.com/google-research-datasets/scin) | Real-world contributed dermatology images with dermatologist weighted differential labels and skin-tone metadata. | Best current fit for DermaLens, but the tail classes are too small after strict face-focused filtering. |

## Interpretation Rule

A benchmark is only a **headline benchmark** if it tests the same target distribution, label space, and leakage controls as the model claim. ISIC, HAM10000, DDI, and PAD-UFES-20 are important dermatology benchmarks, but they should be reported as external validity or branch-specific evaluations unless the project adds a lesion-risk model with matching labels.

For the current portfolio claim, the cleanest result is:

```text
Derm Foundation linear probe > fair grouped MobileNetV3 retrain
66.8% vs 48.0% accuracy
33.8% vs 30.2% macro recall
```

This is a representation lift, not clinical validation.
