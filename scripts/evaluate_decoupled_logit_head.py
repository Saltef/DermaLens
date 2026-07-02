from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train decoupled balanced heads on frozen ONNX logits for grouped SCIN splits.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--model", default="models/skin_classifier.onnx")
    parser.add_argument("--label-map", default="models/label_map.json")
    parser.add_argument("--prior-profiles", default="models/prior_profiles.json")
    parser.add_argument("--profile", default="conservative_population_like")
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--seeds", default="42,7,13,21,84")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = _read_labels(Path(args.label_map))
    rows = _read_manifest(Path(args.manifest))
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    target_prior = _read_profile(Path(args.prior_profiles), args.profile, labels)

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    logit_cache = _logit_cache(session, input_name, rows, Path(args.image_root))

    split_results = []
    for seed in seeds:
        train_rows, val_rows = _grouped_split(rows, val_ratio=args.val_ratio, seed=seed)
        train_rows, val_rows = _dedup_like_imagefolder(train_rows, val_rows, Path(args.image_root))
        train_rows = [row for row in train_rows if row["image_path"] in logit_cache]
        val_rows = [row for row in val_rows if row["image_path"] in logit_cache]

        x_train = np.vstack([logit_cache[row["image_path"]] for row in train_rows])
        y_train = np.array([labels.index(row["label"]) for row in train_rows])
        x_val = np.vstack([logit_cache[row["image_path"]] for row in val_rows])
        y_val = np.array([labels.index(row["label"]) for row in val_rows])

        train_prior = _training_prior(train_rows, labels)
        correction = _prior_correction(labels, train_prior, target_prior) * args.alpha

        deployed_preds = np.argmax(x_val + correction, axis=1)
        deployed_metrics = _metrics_from_indices(y_val, deployed_preds, labels)

        best_head = None
        head_results = []
        for c_value in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            )
            model.fit(x_train, y_train)
            pred = model.predict(x_val)
            metrics = _metrics_from_indices(y_val, pred, labels)
            metrics["c"] = c_value
            head_results.append(metrics)
            if best_head is None or (metrics["macro_recall"], metrics["accuracy"]) > (
                best_head["macro_recall"],
                best_head["accuracy"],
            ):
                best_head = metrics

        assert best_head is not None
        split_results.append(
            {
                "seed": seed,
                "train_images": len(train_rows),
                "val_images": len(val_rows),
                "deployed_conservative_prior": deployed_metrics,
                "best_decoupled_balanced_logit_head": best_head,
                "head_sweep": head_results,
            }
        )
        print(
            f"seed={seed} deployed_acc={deployed_metrics['accuracy']:.4f} "
            f"deployed_macro={deployed_metrics['macro_recall']:.4f} "
            f"head_acc={best_head['accuracy']:.4f} head_macro={best_head['macro_recall']:.4f} c={best_head['c']}"
        )

    payload = {
        "protocol": (
            "Decoupled cRT-style experiment: freeze the deployed ONNX image model, use its logits as a compact representation, "
            "and retrain only a class-balanced logistic head on each grouped SCIN split."
        ),
        "model": args.model,
        "manifest": "data/raw/scin/face_skin_manifest.csv",
        "group_key": "case_id",
        "seeds": seeds,
        "baseline": "deployed_conservative_prior",
        "candidate": "best_decoupled_balanced_logit_head",
        "summary": {
            "deployed_conservative_prior": _summarize(split_results, "deployed_conservative_prior", labels),
            "best_decoupled_balanced_logit_head": _summarize(split_results, "best_decoupled_balanced_logit_head", labels),
        },
        "split_results": split_results,
        "interpretation": (
            "This is a cheap decoupled-head baseline. It tests whether class-balanced retraining on frozen deployed logits can "
            "increase tail macro recall under grouped evaluation without changing the image encoder."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {output}")


def _read_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [item["code"] for item in payload["labels"]]


def _read_profile(path: Path, profile_name: str, labels: list[str]) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = payload[profile_name]
    values = {label: max(float(profile.get(label, 0.0)), 1e-8) for label in labels}
    total = sum(values.values())
    return {label: value / total for label, value in values.items()}


def _read_manifest(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row.get("image_path") and row.get("label") and row.get("case_id")]


def _logit_cache(session: ort.InferenceSession, input_name: str, rows: list[dict], image_root: Path) -> dict[str, np.ndarray]:
    cache = {}
    for row in rows:
        image_path = row["image_path"]
        if image_path in cache:
            continue
        path = image_root / image_path
        if not path.exists():
            continue
        cache[image_path] = np.asarray(session.run(None, {input_name: _preprocess(Image.open(path))})[0], dtype=np.float32).reshape(-1)
    return cache


def _grouped_split(rows: list[dict], *, val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    train_rows = []
    val_rows = []
    for _, label_rows in _by(rows, "label").items():
        groups = _by(label_rows, "case_id")
        group_ids = list(groups)
        rng.shuffle(group_ids)
        val_count = max(1, round(len(group_ids) * val_ratio))
        if val_count >= len(group_ids):
            val_count = len(group_ids) - 1
        val_group_ids = set(group_ids[:val_count])
        train_rows.extend(row for group_id in group_ids if group_id not in val_group_ids for row in groups[group_id])
        val_rows.extend(row for group_id in group_ids if group_id in val_group_ids for row in groups[group_id])
    overlap = {row["case_id"] for row in train_rows} & {row["case_id"] for row in val_rows}
    if overlap:
        raise ValueError(f"Group leakage detected for seed {seed}: {sorted(overlap)[:5]}")
    return train_rows, val_rows


def _dedup_like_imagefolder(train_rows: list[dict], val_rows: list[dict], image_root: Path) -> tuple[list[dict], list[dict]]:
    kept_train = []
    kept_val = []
    seen_by_label: dict[str, set[str]] = defaultdict(set)
    for row in train_rows:
        digest = _row_digest(row, image_root)
        if not digest or digest in seen_by_label[row["label"]]:
            continue
        seen_by_label[row["label"]].add(digest)
        kept_train.append(row)
    for row in val_rows:
        digest = _row_digest(row, image_root)
        if not digest or digest in seen_by_label[row["label"]]:
            continue
        seen_by_label[row["label"]].add(digest)
        kept_val.append(row)
    return kept_train, kept_val


def _row_digest(row: dict, image_root: Path) -> str:
    path = image_root / row["image_path"]
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _training_prior(rows: list[dict], labels: list[str], smoothing: float = 1.0) -> dict[str, float]:
    counts = {label: smoothing for label in labels}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], smoothing) + 1
    total = sum(counts.values())
    return {label: counts[label] / total for label in labels}


def _prior_correction(labels: list[str], train_prior: dict[str, float], target_prior: dict[str, float]) -> np.ndarray:
    return np.array([math.log(target_prior[label]) - math.log(train_prior[label]) for label in labels], dtype=np.float32)


def _metrics_from_indices(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict:
    per_class_recall = {}
    confusion = {actual: {predicted: 0 for predicted in labels} for actual in labels}
    for actual_idx, pred_idx in zip(y_true.tolist(), y_pred.tolist()):
        actual = labels[actual_idx]
        predicted = labels[pred_idx]
        confusion[actual][predicted] += 1
    for idx, label in enumerate(labels):
        mask = y_true == idx
        if int(mask.sum()) > 0:
            per_class_recall[label] = float((y_pred[mask] == idx).sum() / mask.sum())
    return {
        "accuracy": float((y_true == y_pred).sum() / len(y_true)),
        "macro_recall": float(sum(per_class_recall.values()) / len(per_class_recall)),
        "per_class_recall": per_class_recall,
        "confusion": confusion,
    }


def _summarize(split_results: list[dict], key: str, labels: list[str]) -> dict:
    metrics = [result[key] for result in split_results]
    per_class = {}
    for label in labels:
        values = [item["per_class_recall"].get(label, 0.0) for item in metrics]
        per_class[label] = _mean_std(values)
    return {
        "accuracy": _mean_std([item["accuracy"] for item in metrics]),
        "macro_recall": _mean_std([item["macro_recall"] for item in metrics]),
        "per_class_recall": per_class,
    }


def _mean_std(values: list[float]) -> dict:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, :, :, :].astype(np.float32)


def _by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "")].append(row)
    return grouped


if __name__ == "__main__":
    main()
