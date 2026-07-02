# Data Card: DermaLens Local

## Data Policy

No training images are committed to this repository. Dataset folders under `data/raw/` and `data/processed/` are ignored by Git. Error-review images, cached embeddings, and experiment artifacts under `models/experiments/` are also ignored.

## Primary Data Source

The current workflow was designed around SCIN-style public dermatology data, filtered toward head/neck cases when available. SCIN cases may contain multiple photos per case, so splitting must be done by `case_id`, not individual image row.

## Label Construction

DermaLens uses broad portfolio labels mapped from dataset metadata. The mapping is intentionally documented as heuristic rather than clinical truth. See `models/label_mapping_rules_v2.json`.

The SCIN manifest builder supports stricter filtering:

- `--min-label-confidence`
- `--exclude-mixed-labels`
- `--mixed-label-margin`

These flags reduce noisy/mixed labels at the cost of fewer training examples.

## Split Protocol

The accepted protocol is grouped train/validation splitting:

- group key: `case_id` when available
- leakage assertion: no group ID may appear in both train and validation
- audit artifact: `split_audit.json`
- exact duplicate protection: duplicate file digests are skipped during output writing

Image-level splitting is allowed only with the explicit `--allow-image-level-split` flag for datasets with no case or patient identifier.

## Known Data Risks

- Label noise and multi-label clinical reality collapsed into one broad target.
- Skin-tone imbalance and potential underperformance on darker skin.
- Face-photo distribution shift across datasets.
- Near-duplicate public images may exist even across nominal dataset boundaries.
- Auxiliary datasets can increase class counts while hurting generalization if their capture setting differs from phone face photos.

## Recommended Reporting

Every reported metric should include:

- dataset source and version
- split type and seed
- number of images and unique groups per class
- whether calibration was learned on a separate calibration fold
- per-class recall and macro recall
- subgroup metrics when skin-tone metadata is available
