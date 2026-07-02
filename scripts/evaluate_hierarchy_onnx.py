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
    parser = argparse.ArgumentParser(description="Evaluate a hierarchical ONNX classifier.")
    parser.add_argument("--hierarchy", required=True, help="hierarchy.json from build_hierarchy_imagefolders.py.")
    parser.add_argument("--group-model-dir", required=True)
    parser.add_argument("--branch-root", required=True, help="Folder containing one subfolder per branch model.")
    parser.add_argument("--data-dir", required=True, help="Original flat ImageFolder split, usually .../val.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hierarchy = json.loads(Path(args.hierarchy).read_text(encoding="utf-8"))
    group_model = _load_model(Path(args.group_model_dir))
    branch_models = {
        group: _load_model(Path(args.branch_root) / group)
        for group in hierarchy["groups"]
        if (Path(args.branch_root) / group / "skin_classifier.onnx").exists()
    }
    dataset = datasets.ImageFolder(args.data_dir)
    all_labels = dataset.classes

    correct = 0
    total = 0
    per_class = {label: {"correct": 0, "total": 0} for label in all_labels}
    confusion = {
        actual: {predicted: 0 for predicted in all_labels}
        for actual in all_labels
    }

    for image_path, actual_idx in dataset.samples:
        actual = dataset.classes[actual_idx]
        tensor = _preprocess(Image.open(image_path))
        group = group_model.predict(tensor)
        branch = branch_models.get(group)
        predicted = branch.predict(tensor) if branch else hierarchy["groups"][group][0]
        if predicted not in confusion[actual]:
            confusion[actual][predicted] = 0
        confusion[actual][predicted] += 1
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
    metrics = {
        "data_dir": str(args.data_dir),
        "total": total,
        "accuracy": correct / max(1, total),
        "macro_recall": sum(per_class_recall.values()) / max(1, len(per_class_recall)),
        "per_class_recall": per_class_recall,
        "confusion": confusion,
        "group_model_dir": str(args.group_model_dir),
        "branch_root": str(args.branch_root),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"accuracy={metrics['accuracy']:.4f} macro_recall={metrics['macro_recall']:.4f} total={total}")
    print(f"wrote {output}")


class OnnxModel:
    def __init__(self, model_dir: Path) -> None:
        self.labels = _read_labels(model_dir / "label_map.json")
        self.session = ort.InferenceSession(str(model_dir / "skin_classifier.onnx"), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, tensor: np.ndarray) -> str:
        logits = np.asarray(self.session.run(None, {self.input_name: tensor})[0]).reshape(-1)
        return self.labels[int(np.argmax(logits))]


def _load_model(model_dir: Path) -> OnnxModel:
    if not (model_dir / "skin_classifier.onnx").exists():
        raise FileNotFoundError(model_dir / "skin_classifier.onnx")
    if not (model_dir / "label_map.json").exists():
        raise FileNotFoundError(model_dir / "label_map.json")
    return OnnxModel(model_dir)


def _read_labels(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [item["code"] for item in payload["labels"]]


def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, :, :, :].astype(np.float32)


if __name__ == "__main__":
    main()
