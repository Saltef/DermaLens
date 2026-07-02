from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact facial skin classifier and export ONNX.")
    parser.add_argument("--data-dir", required=True, help="Folder with train/ and val/ ImageFolder subfolders.")
    parser.add_argument("--output-dir", default="models", help="Where to write skin_classifier.onnx and label_map.json.")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--model", choices=["mobilenet_v3_small", "efficientnet_b0"], default="mobilenet_v3_small")
    parser.add_argument("--no-pretrained", action="store_true", help="Skip ImageNet weights.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Train only the classifier head.")
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--select-metric", choices=["val_accuracy", "macro_recall"], default="val_accuracy")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)

    train_steps = [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if args.random_erasing > 0:
        train_steps.append(transforms.RandomErasing(p=args.random_erasing, scale=(0.02, 0.12), ratio=(0.3, 3.3)))
    transform_train = transforms.Compose(train_steps)
    transform_val = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(data_dir / "train", transform=transform_train)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=transform_val)
    sampler = _balanced_sampler(train_ds) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(
        args.model,
        num_classes=len(train_ds.classes),
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    class_weights = _class_weights(train_ds).to(device) if args.class_weights == "balanced" else None
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_score = -1.0
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(labels.numel())
        scheduler.step()

        metrics = evaluate(model, val_loader, device, train_ds.classes)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = running_loss / max(1, len(train_ds))
        history.append(metrics)
        print(
            f"epoch={epoch + 1} train_loss={metrics['train_loss']:.4f} "
            f"val_acc={metrics['val_accuracy']:.4f} macro_recall={metrics['macro_recall']:.4f}"
        )
        score = metrics[args.select_metric]
        if score >= best_score:
            best_score = score
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    export_onnx(model.cpu(), output_dir / "skin_classifier.onnx")
    write_label_map(train_ds.classes, output_dir / "label_map.json")
    (output_dir / "training_metrics.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "pretrained": not args.no_pretrained,
                "freeze_backbone": args.freeze_backbone,
                "class_weights": args.class_weights,
                "balanced_sampler": args.balanced_sampler,
                "label_smoothing": args.label_smoothing,
                "random_erasing": args.random_erasing,
                "select_metric": args.select_metric,
                "seed": args.seed,
                "validation_preprocessing": "Resize((224, 224), bicubic) + ImageNet normalization, matching api.model_adapter._preprocess_for_classifier.",
                "epochs": args.epochs,
                "classes": train_ds.classes,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'skin_classifier.onnx'}")
    print(f"wrote {output_dir / 'label_map.json'}")
    print(f"wrote {output_dir / 'training_metrics.json'}")


def build_model(name: str, num_classes: int, *, pretrained: bool, freeze_backbone: bool) -> nn.Module:
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, classes: list[str]) -> dict:
    model.eval()
    correct = 0
    total = 0
    per_class = {class_name: {"correct": 0, "total": 0} for class_name in classes}
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())
        for pred, label in zip(preds.detach().cpu().tolist(), labels.detach().cpu().tolist()):
            class_name = classes[label]
            per_class[class_name]["total"] += 1
            per_class[class_name]["correct"] += int(pred == label)

    per_class_recall = {
        class_name: values["correct"] / max(1, values["total"])
        for class_name, values in per_class.items()
    }
    return {
        "val_accuracy": correct / max(1, total),
        "macro_recall": sum(per_class_recall.values()) / max(1, len(per_class_recall)),
        "per_class_recall": per_class_recall,
    }


def _class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = torch.zeros(len(dataset.classes), dtype=torch.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = counts.sum() / torch.clamp(counts, min=1.0)
    return weights / weights.mean()


def _balanced_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    counts = torch.zeros(len(dataset.classes), dtype=torch.float32)
    labels = []
    for _, label in dataset.samples:
        labels.append(label)
        counts[label] += 1
    sample_weights = [float(1.0 / max(1.0, counts[label].item())) for label in labels]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def export_onnx(model: nn.Module, path: Path) -> None:
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def write_label_map(classes: list[str], path: Path) -> None:
    labels = [
        {
            "code": class_name,
            "label": class_name.replace("_", " ").title(),
            "rationale": "The trained classifier found visual patterns associated with this class.",
        }
        for class_name in classes
    ]
    path.write_text(json.dumps({"problem_type": "multiclass", "labels": labels}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
