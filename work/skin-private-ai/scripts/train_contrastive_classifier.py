from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classifier with supervised contrastive regularization.")
    parser.add_argument("--data-dir", required=True, help="Folder with train/ and val/ ImageFolder subfolders.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", choices=["mobilenet_v3_small", "efficientnet_b0"], default="mobilenet_v3_small")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--random-erasing", type=float, default=0.0)
    parser.add_argument("--supcon-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--select-metric", choices=["val_accuracy", "macro_recall"], default="val_accuracy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=_train_transform(args.random_erasing))
    val_ds = datasets.ImageFolder(data_dir / "val", transform=_val_transform())
    sampler = _balanced_sampler(train_ds) if args.balanced_sampler else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} train_images={len(train_ds)} val_images={len(val_ds)} classes={len(train_ds.classes)}")
    model = ContrastiveClassifier(
        args.model,
        num_classes=len(train_ds.classes),
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
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
        running_ce = 0.0
        running_supcon = 0.0
        seen = 0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, projected = model(images, return_projection=True)
            ce_loss = criterion(logits, labels)
            supcon = supervised_contrastive_loss(projected, labels, temperature=args.temperature)
            loss = ce_loss + args.supcon_weight * supcon
            loss.backward()
            optimizer.step()

            batch_size = int(labels.numel())
            running_loss += float(loss.item()) * batch_size
            running_ce += float(ce_loss.item()) * batch_size
            running_supcon += float(supcon.item()) * batch_size
            seen += batch_size
            if batch_idx == 1 or batch_idx % 10 == 0:
                print(
                    f"epoch={epoch + 1} batch={batch_idx}/{len(train_loader)} "
                    f"loss={float(loss.item()):.4f}",
                    flush=True,
                )
        scheduler.step()

        metrics = evaluate(model, val_loader, device, train_ds.classes)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = running_loss / max(1, seen)
        metrics["ce_loss"] = running_ce / max(1, seen)
        metrics["supcon_loss"] = running_supcon / max(1, seen)
        history.append(metrics)
        print(
            f"epoch={epoch + 1} train_loss={metrics['train_loss']:.4f} "
            f"ce={metrics['ce_loss']:.4f} supcon={metrics['supcon_loss']:.4f} "
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
                "approach": "supervised_contrastive_regularization",
                "model": args.model,
                "pretrained": not args.no_pretrained,
                "freeze_backbone": args.freeze_backbone,
                "class_weights": args.class_weights,
                "balanced_sampler": args.balanced_sampler,
                "label_smoothing": args.label_smoothing,
                "random_erasing": args.random_erasing,
                "supcon_weight": args.supcon_weight,
                "temperature": args.temperature,
                "projection_dim": args.projection_dim,
                "dropout": args.dropout,
                "select_metric": args.select_metric,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
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


class ContrastiveClassifier(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        *,
        num_classes: int,
        pretrained: bool,
        freeze_backbone: bool,
        projection_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        if backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.efficientnet_b0(weights=weights)
            self.features = base.features
            self.avgpool = base.avgpool
            feature_dim = base.classifier[-1].in_features
        else:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            self.features = base.features
            self.avgpool = base.avgpool
            feature_dim = base.classifier[0].in_features

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        hidden_dim = min(1024, max(256, feature_dim))
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, images: torch.Tensor, return_projection: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.avgpool(self.features(images))
        features = torch.flatten(features, 1)
        logits = self.classifier(features)
        if return_projection:
            return logits, self.projection(features)
        return logits


def supervised_contrastive_loss(projected: torch.Tensor, labels: torch.Tensor, *, temperature: float) -> torch.Tensor:
    features = F.normalize(projected, dim=1)
    logits = features @ features.T / max(temperature, 1e-6)
    batch_size = labels.shape[0]
    self_mask = torch.eye(batch_size, dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    logits = logits.masked_fill(self_mask, -torch.finfo(logits.dtype).max)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not torch.any(valid):
        return projected.new_tensor(0.0)
    mean_log_prob_pos = (log_prob * positive_mask).sum(dim=1)[valid] / positive_count[valid]
    return -mean_log_prob_pos.mean()


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


def _train_transform(random_erasing: float) -> transforms.Compose:
    steps = [
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if random_erasing > 0:
        steps.append(transforms.RandomErasing(p=random_erasing, scale=(0.02, 0.12), ratio=(0.3, 3.3)))
    return transforms.Compose(steps)


def _val_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


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
