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

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ONNX performance by SCIN skin-tone metadata.")
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
    target_prior = _read_profile(Path(args.prior_profiles), args.profile, labels)
    rows = _read_manifest(Path(args.manifest))
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    split_results = []
    for seed in seeds:
        train_rows, val_rows = _grouped_split(rows, val_ratio=args.val_ratio, seed=seed)
        train_rows, val_rows = _dedup_like_imagefolder(train_rows, val_rows, Path(args.image_root))
        train_prior = _training_prior(train_rows, labels)
        correction = _prior_correction(labels, train_prior, target_prior) * args.alpha
        evaluated = _evaluate_rows(session, input_name, val_rows, Path(args.image_root), labels, correction)
        split_results.append(
            {
                "seed": seed,
                "train_images": len(train_rows),
                "val_images": len(evaluated),
                "overall": _metrics(evaluated, labels),
                "subgroups": {
                    "fitzpatrick_bucket": _subgroup_metrics(evaluated, labels, "fitzpatrick_bucket"),
                    "monk_us_bucket": _subgroup_metrics(evaluated, labels, "monk_us_bucket"),
                },
            }
        )
        print(
            f"seed={seed} val={len(evaluated)} "
            f"accuracy={split_results[-1]['overall']['accuracy']:.4f} "
            f"macro_recall={split_results[-1]['overall']['macro_recall']:.4f}"
        )

    payload = {
        "protocol": "SCIN-only grouped case-level subgroup evaluation using fixed deployed ONNX model and deployed conservative prior correction.",
        "model": args.model,
        "manifest": "data/raw/scin/face_skin_manifest.csv",
        "group_key": "case_id",
        "profile": args.profile,
        "alpha": args.alpha,
        "seeds": seeds,
        "overall_summary": _summarize_overall(split_results),
        "subgroup_summary": {
            "fitzpatrick_bucket": _summarize_subgroups(split_results, "fitzpatrick_bucket"),
            "monk_us_bucket": _summarize_subgroups(split_results, "monk_us_bucket"),
        },
        "split_results": split_results,
        "interpretation": (
            "Subgroup estimates are directionally useful but not definitive. Several buckets and labels have small "
            "validation counts, so worst-group metrics should be treated as an audit signal rather than a clinical claim."
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


def _grouped_split(rows: list[dict], *, val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    train_rows = []
    val_rows = []
    for label, label_rows in _by(rows, "label").items():
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


def _evaluate_rows(
    session: ort.InferenceSession,
    input_name: str,
    rows: list[dict],
    image_root: Path,
    labels: list[str],
    correction: np.ndarray,
) -> list[dict]:
    evaluated = []
    for row in rows:
        path = image_root / row["image_path"]
        if not path.exists():
            continue
        logits = np.asarray(session.run(None, {input_name: _preprocess(Image.open(path))})[0], dtype=np.float32).reshape(-1)
        predicted = labels[int(np.argmax(logits + correction))]
        evaluated.append(
            {
                "actual": row["label"],
                "predicted": predicted,
                "fitzpatrick_bucket": _fitzpatrick_bucket(row.get("fitzpatrick_skin_type", "")),
                "monk_us_bucket": _monk_bucket(row.get("monk_skin_tone_us", "")),
            }
        )
    return evaluated


def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, :, :, :].astype(np.float32)


def _fitzpatrick_bucket(value: str) -> str:
    normalized = value.strip().upper()
    if normalized in {"FST1", "FST2"}:
        return "FST1-2"
    if normalized in {"FST3", "FST4"}:
        return "FST3-4"
    if normalized in {"FST5", "FST6"}:
        return "FST5-6"
    return "unknown"


def _monk_bucket(value: str) -> str:
    try:
        tone = int(float(value))
    except ValueError:
        return "unknown"
    if 1 <= tone <= 3:
        return "MST1-3"
    if 4 <= tone <= 6:
        return "MST4-6"
    if 7 <= tone <= 10:
        return "MST7-10"
    return "unknown"


def _metrics(rows: list[dict], labels: list[str]) -> dict:
    if not rows:
        return {"count": 0, "accuracy": None, "macro_recall": None, "per_class_recall": {}}
    correct = sum(1 for row in rows if row["actual"] == row["predicted"])
    per_class = {}
    for label in labels:
        label_rows = [row for row in rows if row["actual"] == label]
        if label_rows:
            per_class[label] = sum(1 for row in label_rows if row["predicted"] == label) / len(label_rows)
    return {
        "count": len(rows),
        "accuracy": correct / len(rows),
        "macro_recall": sum(per_class.values()) / len(per_class) if per_class else None,
        "per_class_recall": per_class,
    }


def _subgroup_metrics(rows: list[dict], labels: list[str], key: str) -> dict[str, dict]:
    return {group: _metrics(group_rows, labels) for group, group_rows in sorted(_by(rows, key).items())}


def _summarize_overall(split_results: list[dict]) -> dict:
    return {
        "accuracy": _mean_std([result["overall"]["accuracy"] for result in split_results]),
        "macro_recall": _mean_std([result["overall"]["macro_recall"] for result in split_results]),
        "val_images": _mean_std([result["val_images"] for result in split_results]),
    }


def _summarize_subgroups(split_results: list[dict], key: str) -> dict[str, dict]:
    groups = sorted({group for result in split_results for group in result["subgroups"][key]})
    summary = {}
    for group in groups:
        group_rows = [result["subgroups"][key][group] for result in split_results if group in result["subgroups"][key]]
        summary[group] = {
            "count": _mean_std([item["count"] for item in group_rows]),
            "accuracy": _mean_std([item["accuracy"] for item in group_rows if item["accuracy"] is not None]),
            "macro_recall": _mean_std([item["macro_recall"] for item in group_rows if item["macro_recall"] is not None]),
        }
    return summary


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "values": []}
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def _by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get(key, "")].append(row)
    return grouped


if __name__ == "__main__":
    main()
