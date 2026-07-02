from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep supervised-contrastive MLP heads over cached embeddings.")
    parser.add_argument("--train-embeddings", required=True)
    parser.add_argument("--val-embeddings", required=True)
    parser.add_argument("--classes", required=True, help="JSON list or label_map.json.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dims", default="256,512")
    parser.add_argument("--projection-dims", default="64,128")
    parser.add_argument("--dropouts", default="0.1,0.25")
    parser.add_argument("--supcon-weights", default="0.03,0.07,0.1,0.2")
    parser.add_argument("--temperatures", default="0.1,0.2,0.3")
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--class-weights", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--sampler", choices=["shuffle", "balanced"], default="balanced")
    parser.add_argument("--select-metric", choices=["accuracy", "macro_recall"], default="accuracy")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    classes = _read_classes(Path(args.classes))
    train_npz = np.load(args.train_embeddings)
    val_npz = np.load(args.val_embeddings)
    x_train = train_npz["x"].astype(np.float32)
    y_train = train_npz["y"].astype(np.int64)
    x_val = val_npz["x"].astype(np.float32)
    y_val = val_npz["y"].astype(np.int64)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    grid = list(
        itertools.product(
            _ints(args.hidden_dims),
            _ints(args.projection_dims),
            _floats(args.dropouts),
            _floats(args.supcon_weights),
            _floats(args.temperatures),
        )
    )
    print(f"device={device} train={len(y_train)} val={len(y_val)} runs={len(grid)}")

    best = None
    results = []
    for run_idx, (hidden_dim, projection_dim, dropout, supcon_weight, temperature) in enumerate(grid, start=1):
        model = EmbeddingHead(
            input_dim=x_train.shape[1],
            hidden_dim=hidden_dim,
            projection_dim=projection_dim,
            num_classes=len(classes),
            dropout=dropout,
        ).to(device)
        loader = _loader(x_train, y_train, args.batch_size, sampler_mode=args.sampler)
        class_weights = _class_weights(y_train, len(classes)).to(device) if args.class_weights == "balanced" else None
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

        best_run = None
        for epoch in range(args.epochs):
            train_loss = _train_epoch(
                model,
                loader,
                criterion,
                optimizer,
                device,
                supcon_weight=supcon_weight,
                temperature=temperature,
            )
            scheduler.step()
            if epoch == args.epochs - 1 or (epoch + 1) % 25 == 0:
                metrics = _evaluate(model, x_val_t, y_val_t, classes)
                metrics["epoch"] = epoch + 1
                metrics["train_loss"] = train_loss
                if best_run is None or _score(metrics, args.select_metric) >= _score(best_run, args.select_metric):
                    best_run = metrics

        result = {
            "run": run_idx,
            "hidden_dim": hidden_dim,
            "projection_dim": projection_dim,
            "dropout": dropout,
            "supcon_weight": supcon_weight,
            "temperature": temperature,
            "best": best_run,
        }
        results.append(result)
        print(
            f"run={run_idx}/{len(grid)} hidden={hidden_dim} proj={projection_dim} dropout={dropout:g} "
            f"w={supcon_weight:g} temp={temperature:g} "
            f"accuracy={best_run['accuracy']:.4f} macro_recall={best_run['macro_recall']:.4f}"
        )
        if best is None or _score(best_run, args.select_metric) >= _score(best["best"], args.select_metric):
            best = result

    payload = {
        "approach": "supervised_contrastive_embedding_head",
        "rationale": (
            "Train a small neural head on cached frozen image embeddings with cross-entropy plus "
            "supervised contrastive loss. This tests whether same-class clustering improves long-tail recognition "
            "without repeatedly fine-tuning the image backbone."
        ),
        "train_embeddings": str(args.train_embeddings),
        "val_embeddings": str(args.val_embeddings),
        "classes": classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "label_smoothing": args.label_smoothing,
        "class_weights": args.class_weights,
        "sampler": args.sampler,
        "select_metric": args.select_metric,
        "best": best,
        "results": results,
    }
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"best run={best['run']} accuracy={best['best']['accuracy']:.4f} "
        f"macro_recall={best['best']['macro_recall']:.4f}"
    )
    print(f"wrote {output_path}")


class EmbeddingHead(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, projection_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.classifier(features), self.projection(features)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    supcon_weight: float,
    temperature: float,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, projected = model(x_batch)
        ce_loss = criterion(logits, y_batch)
        contrastive = supervised_contrastive_loss(projected, y_batch, temperature=temperature)
        loss = ce_loss + supcon_weight * contrastive
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(y_batch.numel())
        total += int(y_batch.numel())
    return total_loss / max(1, total)


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
def _evaluate(model: nn.Module, x_val: torch.Tensor, y_val: torch.Tensor, classes: list[str]) -> dict:
    model.eval()
    logits, _ = model(x_val)
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    actual = y_val.detach().cpu().numpy()
    labels = list(range(len(classes)))
    matrix = confusion_matrix(actual, preds, labels=labels)
    per_class_recall = {}
    for idx, class_name in enumerate(classes):
        mask = actual == idx
        per_class_recall[class_name] = float(np.mean(preds[mask] == actual[mask])) if np.any(mask) else 0.0
    return {
        "accuracy": float(accuracy_score(actual, preds)),
        "macro_recall": float(recall_score(actual, preds, labels=labels, average="macro", zero_division=0)),
        "per_class_recall": per_class_recall,
        "confusion": {
            actual_name: {
                pred_name: int(matrix[actual_idx, pred_idx])
                for pred_idx, pred_name in enumerate(classes)
            }
            for actual_idx, actual_name in enumerate(classes)
        },
    }


def _loader(x_train: np.ndarray, y_train: np.ndarray, batch_size: int, *, sampler_mode: str) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    if sampler_mode == "shuffle":
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    counts = np.bincount(y_train, minlength=int(y_train.max()) + 1).astype(np.float32)
    weights = np.asarray([1.0 / max(1.0, counts[label]) for label in y_train], dtype=np.float32)
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.clip(counts, 1.0, None)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _read_classes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return [str(item["code"]) for item in payload["labels"]]


def _floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _score(metrics: dict, metric_name: str) -> tuple[float, float]:
    secondary = "macro_recall" if metric_name == "accuracy" else "accuracy"
    return float(metrics[metric_name]), float(metrics[secondary])


if __name__ == "__main__":
    main()
