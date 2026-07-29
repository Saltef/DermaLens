# Benchmark Ledger

DermaLens uses benchmarks in two different ways:

1. **Executed internal benchmarks** for the face-focused SCIN task.
2. **External dermatology benchmarks** used to position future evaluation, not to inflate the current facial-condition result.

## Executed Benchmarks

| Benchmark | Status | Result | Validity |
| --- | --- | ---: | --- |
| Majority-class dermatitis baseline on grouped SCIN | Completed | 63.7% +/- 1.6 accuracy, 16.7% macro recall | Required imbalanced-class floor across the 12-seed fair comparison. High accuracy because dermatitis dominates validation support; low macro recall because five labels are never predicted. |
| Fair grouped MobileNetV3 retrain on SCIN | Completed | 44.8% +/- 9.9 accuracy, 29.1% +/- 3.9 macro recall | Clean internal baseline across 12 matched seeds. Each seed trains on grouped training cases and evaluates untouched validation cases. |
| Accuracy-selected MobileNet checkpoint diagnostic | Diagnostic only | 52.4% +/- 5.2 accuracy, 25.1% +/- 3.3 macro recall | Optimistic sensitivity check. It reselects the best MobileNet epoch from the evaluation-fold history, so it is not a clean held-out estimate. |
| ConvNeXt-Tiny frozen ImageNet embedding control | Completed | 53.1% +/- 4.1 accuracy, 23.1% +/- 3.3 macro recall | Generic frozen-encoder control under the same grouped/nested linear-probe protocol. It reduces but does not remove the capacity/input-resolution confound. |
| Derm Foundation linear probe on grouped SCIN | Completed | 69.8% +/- 5.2 accuracy, 34.7% +/- 5.0 macro recall | Clean representation comparison across the same 12 matched seeds. Large paired accuracy lift over fair MobileNet and a smaller positive macro-recall lift; tail classes remain weak. |
| Nested decoupled logit head on grouped SCIN | Completed | 75.3% +/- 1.7 accuracy, 70.7% +/- 11.4 macro recall | Fixed-encoder operating point. Not a clean representation benchmark because the frozen deployed encoder was previously trained on SCIN-derived data. |
| Fixed deployed ONNX on grouped SCIN | Diagnostic only | 86.2% +/- 1.2 accuracy, 63.1% +/- 10.1 macro recall | Contaminated as model-holdout evidence; grouped folds do not exclude original model-training cases. |

Primary artifacts:

- `models/grouped_scin_mobilenet_retrained_baseline_metrics.json`
- `models/grouped_scin_mobilenet_retrained_baseline_10seed_metrics.json`
- `models/grouped_scin_mobilenet_retrained_baseline_12seed_metrics.json`
- `models/grouped_scin_mobilenet_checkpoint_selection_diagnostic.json`
- `models/grouped_scin_convnext_tiny_embedding_12seed_metrics.json`
- `models/grouped_scin_derm_foundation_embedding_metrics.json`
- `models/grouped_scin_derm_foundation_embedding_10seed_metrics.json`
- `models/grouped_scin_derm_foundation_embedding_12seed_metrics.json`
- `models/grouped_scin_fair_model_comparison_metrics.json`
- `models/grouped_scin_seed_count_sensitivity_metrics.json`
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
Majority-class dermatitis floor: 63.7% accuracy, 16.7% macro recall
Fair grouped MobileNetV3 retrain: 44.8% accuracy, 29.1% macro recall
Accuracy-selected MobileNet diagnostic: 52.4% accuracy, 25.1% macro recall
ConvNeXt-Tiny frozen embedding control: 53.1% accuracy, 23.1% macro recall
Derm Foundation linear probe: 69.8% accuracy, 34.7% macro recall
```

The defensible statistical claim changed after expanding the matched seeds. With only five seeds, no exact paired non-parametric test can reach p<0.05; the five-seed accuracy p-value rested on an untestable t-test normality assumption. At 12 matched completed seeds, Derm Foundation gives a large paired accuracy gain over fair MobileNet (+25.0 points, 95% CI +17.8 to +32.2, exact sign p=0.00049) and a smaller macro-recall gain (+5.6 points, 95% CI +2.6 to +8.6, exact sign p=0.00635). These p-values are repeated-split stability diagnostics, not independent external-test-set inference, because all seeds resample the same SCIN case universe. At the class level, Derm Foundation still loses mean recall on rosacea and hyperpigmentation and gains mainly on acne, dermatitis, and clinician-review.

Honest headline:

```text
Derm Foundation embeddings give a large, robust accuracy improvement over fold-retrained MobileNetV3.
They also show a smaller positive macro-recall lift over that fair baseline, while both learned models beat the majority-class floor on macro recall.
Absolute performance remains far from usable, and tail classes are underpowered.
```

Robustness check: if MobileNet is given an optimistic post-hoc accuracy-selected checkpoint from the same training histories, Derm Foundation still leads on accuracy, but the checkpoint policies cross: accuracy-selected MobileNet is better on accuracy and worse on macro recall than the exported macro-selected checkpoint. The conservative envelope is therefore >= +17.3 accuracy points and >= +5.6 macro-recall points against the most favorable MobileNet policy for each metric separately. This is not promoted to the main baseline because it selects from evaluation-fold history.

Attribution check: ConvNeXt-Tiny ImageNet embeddings under the same grouped/nested probe protocol reached 53.1% accuracy and 23.1% macro recall. Derm Foundation remains ahead by +16.7 accuracy points and +11.6 macro-recall points, but the attribution is still not pure "dermatology pretraining." Derm Foundation is a much larger BiT-101x3-style 448px representation, while the deployed MobileNet is compact and 224px. Google's [Derm Foundation documentation](https://developers.google.com/health-ai-developer-foundations/derm-foundation) points to SCIN as a public linear-classifier example, and the [model card](https://developers.google.com/health-ai-developer-foundations/derm-foundation/model-card) identifies the model architecture and 6144-dimensional embeddings. This result is therefore a SCIN downstream benchmark on the vendor home dataset, not external validation.
