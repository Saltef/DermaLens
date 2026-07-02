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
    parser = argparse.ArgumentParser(description="Evaluate an ONNX classifier on an ImageFolder split.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--data-dir", required=True, help="ImageFolder split directory, usually .../val.")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = _read_labels(Path(args.label_map))
    dataset = datasets.ImageFolder(args.data_dir)
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    correct = 0
    total = 0
    per_class = {class_name: {"correct": 0, "total": 0} for class_name in dataset.classes}
    confusion = {
        actual: {predicted: 0 for predicted in dataset.classes}
        for actual in dataset.classes
    }

    for image_path, class_idx in dataset.samples:
        actual = dataset.classes[class_idx]
        tensor = _preprocess(Image.open(image_path))
        logits = np.asarray(session.run(None, {input_name: tensor})[0]).reshape(-1)
        pred_idx = int(np.argmax(logits))
        predicted = labels[pred_idx] if pred_idx < len(labels) else f"unknown_{pred_idx}"
        if predicted not in per_class:
            per_class[predicted] = {"correct": 0, "total": 0}
        if actual not in confusion:
            confusion[actual] = {}
        confusion[actual][predicted] = confusion[actual].get(predicted, 0) + 1

        is_correct = actual == predicted
        correct += int(is_correct)
        total += 1
        per_class[actual]["total"] += 1
        per_class[actual]["correct"] += int(is_correct)

    per_class_recall = {
        class_name: values["correct"] / max(1, values["total"])
        for class_name, values in per_class.items()
        if values["total"] > 0
    }
    metrics = {
        "model": str(args.model),
        "data_dir": str(args.data_dir),
        "total": total,
        "accuracy": correct / max(1, total),
        "macro_recall": sum(per_class_recall.values()) / max(1, len(per_class_recall)),
        "per_class_recall": per_class_recall,
        "confusion": confusion,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"accuracy={metrics['accuracy']:.4f} macro_recall={metrics['macro_recall']:.4f} total={total}")
    print(f"wrote {output}")


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

