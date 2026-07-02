from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models


BACKBONES = ["mobilenet_v3_small", "efficientnet_b0", "convnext_tiny", "swin_t", "vit_b_16"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen pretrained vision embeddings with a small balanced classifier."
    )
    parser.add_argument("--data-dir", required=True, help="ImageFolder root with train/ and val/.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--c-values", default="0.03,0.1,0.3,1,3,10")
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_model, transform = build_feature_model(args.backbone)
    feature_model = feature_model.to(device).eval()

    train_ds = datasets.ImageFolder(data_dir / "train", transform=transform)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=transform)
    if train_ds.classes != val_ds.classes:
        raise ValueError("Train and validation classes differ.")

    train_cache = output_dir / f"{args.backbone}_train_embeddings.npz"
    val_cache = output_dir / f"{args.backbone}_val_embeddings.npz"
    if args.cache and train_cache.exists() and val_cache.exists():
        train_npz = np.load(train_cache)
        val_npz = np.load(val_cache)
        x_train, y_train = train_npz["x"], train_npz["y"]
        x_val, y_val = val_npz["x"], val_npz["y"]
    else:
        x_train, y_train = extract_embeddings(train_ds, feature_model, device, args.batch_size, args.num_workers)
        x_val, y_val = extract_embeddings(val_ds, feature_model, device, args.batch_size, args.num_workers)
        if args.cache:
            np.savez_compressed(train_cache, x=x_train, y=y_train)
            np.savez_compressed(val_cache, x=x_val, y=y_val)

    c_values = [float(item.strip()) for item in args.c_values.split(",") if item.strip()]
    results = []
    best = None
    for c_value in c_values:
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=args.max_iter,
                solver="lbfgs",
            ),
        )
        classifier.fit(x_train, y_train)
        preds = classifier.predict(x_val)
        metrics = build_metrics(preds, y_val, train_ds.classes)
        metrics["c"] = c_value
        results.append(metrics)
        print(
            f"backbone={args.backbone} C={c_value:g} "
            f"accuracy={metrics['accuracy']:.4f} macro_recall={metrics['macro_recall']:.4f}"
        )
        if best is None or (metrics["accuracy"], metrics["macro_recall"]) > (best["accuracy"], best["macro_recall"]):
            best = metrics

    payload = {
        "approach": "frozen_pretrained_embeddings_logistic_regression",
        "rationale": (
            "Use a frozen pretrained vision representation and train only a small balanced classifier. "
            "This tests whether stronger generic/foundation-style features compensate for limited labels."
        ),
        "backbone": args.backbone,
        "device": str(device),
        "data_dir": str(data_dir),
        "classes": train_ds.classes,
        "train_count": len(train_ds),
        "val_count": len(val_ds),
        "best": best,
        "results": results,
    }
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"best backbone={args.backbone} C={best['c']:g} "
        f"accuracy={best['accuracy']:.4f} macro_recall={best['macro_recall']:.4f}"
    )
    print(f"wrote {output_path}")


@torch.no_grad()
def extract_embeddings(
    dataset: datasets.ImageFolder,
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    features = []
    labels = []
    for images, batch_labels in loader:
        output = model(images.to(device))
        if output.ndim > 2:
            output = torch.flatten(output, start_dim=1)
        features.append(output.detach().cpu().numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def build_feature_model(name: str) -> tuple[nn.Module, object]:
    if name == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Identity()
        return model, weights.transforms()
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Identity()
        return model, weights.transforms()
    if name == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        model.classifier[-1] = nn.Identity()
        return model, weights.transforms()
    if name == "swin_t":
        weights = models.Swin_T_Weights.IMAGENET1K_V1
        model = models.swin_t(weights=weights)
        model.head = nn.Identity()
        return model, weights.transforms()
    if name == "vit_b_16":
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        model = models.vit_b_16(weights=weights)
        model.heads = nn.Identity()
        return model, weights.transforms()
    raise ValueError(name)


def build_metrics(preds: np.ndarray, actual: np.ndarray, classes: list[str]) -> dict:
    per_class = {}
    for idx, class_name in enumerate(classes):
        mask = actual == idx
        per_class[class_name] = float(np.mean(preds[mask] == actual[mask])) if np.any(mask) else 0.0
    labels = list(range(len(classes)))
    matrix = confusion_matrix(actual, preds, labels=labels)
    return {
        "accuracy": float(accuracy_score(actual, preds)),
        "macro_recall": float(recall_score(actual, preds, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": per_class,
        "confusion": {
            actual_name: {
                pred_name: int(matrix[actual_idx, pred_idx])
                for pred_idx, pred_name in enumerate(classes)
            }
            for actual_idx, actual_name in enumerate(classes)
        },
    }


if __name__ == "__main__":
    main()
