from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import datasets

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate flat ONNX with optional branch refiners.")
    parser.add_argument("--flat-model-dir", required=True)
    parser.add_argument("--hierarchy", required=True)
    parser.add_argument("--branch-root", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.25,0.30")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hierarchy = json.loads(Path(args.hierarchy).read_text(encoding="utf-8"))
    label_to_group = {
        label: group
        for group, labels in hierarchy["groups"].items()
        for label in labels
    }
    flat_model = OnnxModel(Path(args.flat_model_dir), scores=True)
    branch_models = {
        group: OnnxModel(Path(args.branch_root) / group, scores=False)
        for group in hierarchy["groups"]
        if (Path(args.branch_root) / group / "skin_classifier.onnx").exists()
    }
    dataset = datasets.ImageFolder(args.data_dir)
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    results = []

    for threshold in thresholds:
        metrics = _evaluate(dataset, flat_model, branch_models, label_to_group, threshold)
        results.append(metrics)
        print(
            f"threshold={threshold:.2f} accuracy={metrics['accuracy']:.4f} "
            f"macro_recall={metrics['macro_recall']:.4f} refinements={metrics['refinements']}"
        )

    best = max(results, key=lambda item: (item["accuracy"], item["macro_recall"]))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "flat_model_dir": str(args.flat_model_dir),
                "branch_root": str(args.branch_root),
                "data_dir": str(args.data_dir),
                "best": best,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"best_threshold={best['threshold']:.2f} accuracy={best['accuracy']:.4f} "
        f"macro_recall={best['macro_recall']:.4f}"
    )
    print(f"wrote {output}")


def _evaluate(
    dataset: datasets.ImageFolder,
    flat_model: "OnnxModel",
    branch_models: dict[str, "OnnxModel"],
    label_to_group: dict[str, str],
    threshold: float,
) -> dict:
    correct = 0
    total = 0
    refinements = 0
    per_class = {label: {"correct": 0, "total": 0} for label in dataset.classes}
    confusion = {
        actual: {predicted: 0 for predicted in dataset.classes}
        for actual in dataset.classes
    }

    for image_path, actual_idx in dataset.samples:
        actual = dataset.classes[actual_idx]
        tensor = _preprocess(Image.open(image_path))
        flat_scores = flat_model.predict_scores(tensor)
        ranked = sorted(flat_scores.items(), key=lambda item: item[1], reverse=True)
        predicted = ranked[0][0]
        top_group = label_to_group.get(ranked[0][0])
        second_group = label_to_group.get(ranked[1][0]) if len(ranked) > 1 else None
        margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else 1.0

        if top_group and top_group == second_group and margin <= threshold and top_group in branch_models:
            predicted = branch_models[top_group].predict_label(tensor)
            refinements += 1

        confusion[actual][predicted] = confusion[actual].get(predicted, 0) + 1
        is_correct = predicted == actual
        correct += int(is_correct)
        total += 1
        per_class[actual]["total"] += 1
        per_class[actual]["correct"] += int(is_correct)

    per_class_recall = {
        label: values["correct"] / max(1, values["total"])
        for label, values in per_class.items()
        if values["total"] > 0
    }
    return {
        "threshold": threshold,
        "accuracy": correct / max(1, total),
        "macro_recall": sum(per_class_recall.values()) / max(1, len(per_class_recall)),
        "total": total,
        "refinements": refinements,
        "per_class_recall": per_class_recall,
        "confusion": confusion,
    }


class OnnxModel:
    def __init__(self, model_dir: Path, *, scores: bool) -> None:
        self.labels = _read_labels(model_dir / "label_map.json")
        self.problem_type = _read_problem_type(model_dir / "label_map.json")
        self.return_scores = scores
        self.session = ort.InferenceSession(str(model_dir / "skin_classifier.onnx"), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict_scores(self, tensor: np.ndarray) -> dict[str, float]:
        logits = np.asarray(self.session.run(None, {self.input_name: tensor})[0], dtype=np.float32).reshape(-1)
        values = _scores_from_logits(logits, self.problem_type)
        return {label: float(score) for label, score in zip(self.labels, values)}

    def predict_label(self, tensor: np.ndarray) -> str:
        scores = self.predict_scores(tensor)
        return max(scores.items(), key=lambda item: item[1])[0]


def _read_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [item["code"] for item in payload["labels"]]


def _read_problem_type(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("problem_type", "multiclass")


def _scores_from_logits(logits: np.ndarray, problem_type: str) -> np.ndarray:
    if problem_type == "multiclass":
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)
    return 1.0 / (1.0 + np.exp(-logits))


def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, :, :, :].astype(np.float32)


if __name__ == "__main__":
    main()
