from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import datasets

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEFAULT_PROFILES = {
    "portfolio_upload": {
        "acne_like_texture": 0.20,
        "clinician_review": 0.12,
        "dermatitis_like_irritation": 0.38,
        "folliculitis_like_bumps": 0.10,
        "hyperpigmentation_like_uneven_tone": 0.08,
        "rosacea_like_redness": 0.12,
    },
    "conservative_population_like": {
        "acne_like_texture": 0.24,
        "clinician_review": 0.08,
        "dermatitis_like_irritation": 0.46,
        "folliculitis_like_bumps": 0.07,
        "hyperpigmentation_like_uneven_tone": 0.08,
        "rosacea_like_redness": 0.07,
    },
    "minority_sensitive_upload": {
        "acne_like_texture": 0.18,
        "clinician_review": 0.16,
        "dermatitis_like_irritation": 0.30,
        "folliculitis_like_bumps": 0.12,
        "hyperpigmentation_like_uneven_tone": 0.10,
        "rosacea_like_redness": 0.14,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prior-corrected ONNX multiclass predictions.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--data-dir", required=True, help="ImageFolder split to evaluate, usually .../val.")
    parser.add_argument("--train-dir", required=True, help="ImageFolder train split used to estimate training priors.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--alphas",
        default="0,0.1,0.2,0.3,0.4,0.5,0.75,1.0",
        help="Comma-separated prior correction strengths.",
    )
    parser.add_argument("--profiles", default=None, help="Optional JSON file of named target-prior profiles.")
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--tta-flip", action="store_true", help="Average logits from original and horizontal flip.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = _read_labels(Path(args.label_map))
    dataset = datasets.ImageFolder(args.data_dir)
    train_priors = _folder_priors(Path(args.train_dir), labels, smoothing=args.smoothing)
    profiles = _read_profiles(Path(args.profiles)) if args.profiles else DEFAULT_PROFILES
    alphas = [float(item.strip()) for item in args.alphas.split(",") if item.strip()]

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    samples = []
    for image_path, class_idx in dataset.samples:
        image = Image.open(image_path)
        logits = _predict_logits(session, input_name, image, tta_flip=args.tta_flip)
        samples.append({"actual": dataset.classes[class_idx], "logits": logits})

    results = []
    raw = _evaluate(samples, labels, labels, None)
    raw["profile"] = "raw"
    raw["alpha"] = 0.0
    results.append(raw)
    print(f"profile=raw alpha=0.00 accuracy={raw['accuracy']:.4f} macro_recall={raw['macro_recall']:.4f}")

    for profile_name, target_prior in profiles.items():
        normalized_target = _normalize_profile(target_prior, labels)
        correction = _prior_correction(labels, train_priors, normalized_target)
        for alpha in alphas:
            metrics = _evaluate(samples, labels, dataset.classes, correction * alpha)
            metrics["profile"] = profile_name
            metrics["alpha"] = alpha
            metrics["target_prior"] = normalized_target
            results.append(metrics)
            print(
                f"profile={profile_name} alpha={alpha:.2f} "
                f"accuracy={metrics['accuracy']:.4f} macro_recall={metrics['macro_recall']:.4f}"
            )

    best_accuracy = max(results, key=lambda item: (item["accuracy"], item["macro_recall"]))
    best_macro = max(results, key=lambda item: (item["macro_recall"], item["accuracy"]))
    payload = {
        "model": args.model,
        "data_dir": args.data_dir,
        "train_dir": args.train_dir,
        "training_prior": train_priors,
        "best_accuracy": best_accuracy,
        "best_macro_recall": best_macro,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"best_accuracy profile={best_accuracy['profile']} alpha={best_accuracy['alpha']:.2f} "
        f"accuracy={best_accuracy['accuracy']:.4f} macro_recall={best_accuracy['macro_recall']:.4f}"
    )
    print(
        f"best_macro profile={best_macro['profile']} alpha={best_macro['alpha']:.2f} "
        f"accuracy={best_macro['accuracy']:.4f} macro_recall={best_macro['macro_recall']:.4f}"
    )
    print(f"wrote {output}")


def _evaluate(
    samples: list[dict],
    labels: list[str],
    dataset_classes: list[str],
    correction: np.ndarray | None,
) -> dict:
    correct = 0
    total = 0
    per_class = {class_name: {"correct": 0, "total": 0} for class_name in dataset_classes}
    confusion = {
        actual: {predicted: 0 for predicted in labels}
        for actual in dataset_classes
    }
    for sample in samples:
        actual = sample["actual"]
        logits = sample["logits"] if correction is None else sample["logits"] + correction
        predicted = labels[int(np.argmax(logits))]
        confusion.setdefault(actual, {})
        confusion[actual][predicted] = confusion[actual].get(predicted, 0) + 1
        is_correct = actual == predicted
        correct += int(is_correct)
        total += 1
        per_class.setdefault(actual, {"correct": 0, "total": 0})
        per_class[actual]["total"] += 1
        per_class[actual]["correct"] += int(is_correct)

    per_class_recall = {
        class_name: values["correct"] / max(1, values["total"])
        for class_name, values in per_class.items()
        if values["total"] > 0
    }
    return {
        "total": total,
        "accuracy": correct / max(1, total),
        "macro_recall": sum(per_class_recall.values()) / max(1, len(per_class_recall)),
        "per_class_recall": per_class_recall,
        "confusion": confusion,
    }


def _read_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [item["code"] for item in payload["labels"]]


def _read_profiles(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Prior profile file must be a JSON object.")
    return payload


def _folder_priors(train_dir: Path, labels: list[str], *, smoothing: float) -> dict[str, float]:
    counts = {}
    for label in labels:
        label_dir = train_dir / label
        counts[label] = len([path for path in label_dir.iterdir() if path.is_file()]) if label_dir.exists() else 0
    total = sum(counts.values()) + smoothing * len(labels)
    return {label: (counts[label] + smoothing) / total for label in labels}


def _normalize_profile(profile: dict[str, float], labels: list[str]) -> dict[str, float]:
    values = {label: max(float(profile.get(label, 0.0)), 1e-8) for label in labels}
    total = sum(values.values())
    return {label: values[label] / total for label in labels}


def _prior_correction(
    labels: list[str],
    train_prior: dict[str, float],
    target_prior: dict[str, float],
) -> np.ndarray:
    return np.array(
        [math.log(target_prior[label]) - math.log(train_prior[label]) for label in labels],
        dtype=np.float32,
    )


def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, :, :, :].astype(np.float32)


def _predict_logits(session: ort.InferenceSession, input_name: str, image: Image.Image, *, tta_flip: bool) -> np.ndarray:
    tensors = [_preprocess(image)]
    if tta_flip:
        tensors.append(_preprocess(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
    outputs = [
        np.asarray(session.run(None, {input_name: tensor})[0], dtype=np.float32).reshape(-1)
        for tensor in tensors
    ]
    return np.mean(outputs, axis=0)


if __name__ == "__main__":
    main()
