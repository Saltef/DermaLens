from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approach 2 holdout sweep for ConvNeXt embedding heads.")
    parser.add_argument("--train-embeddings", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calib-size", type=float, default=0.15)
    parser.add_argument("--holdout-size", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = _read_classes(Path(args.classes))
    npz = np.load(args.train_embeddings)
    x = npz["x"].astype(np.float32)
    y = npz["y"].astype(np.int64)

    train_idx, temp_idx = train_test_split(
        np.arange(len(y)),
        test_size=args.calib_size + args.holdout_size,
        stratify=y,
        random_state=args.seed,
    )
    holdout_relative = args.holdout_size / (args.calib_size + args.holdout_size)
    calib_idx, holdout_idx = train_test_split(
        temp_idx,
        test_size=holdout_relative,
        stratify=y[temp_idx],
        random_state=args.seed,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx]).astype(np.float32)
    x_calib = scaler.transform(x[calib_idx]).astype(np.float32)
    x_holdout = scaler.transform(x[holdout_idx]).astype(np.float32)
    y_train = y[train_idx]
    y_calib = y[calib_idx]
    y_holdout = y[holdout_idx]

    configs = _configs(args.profile)
    results = []
    for run_idx, config in enumerate(configs, start=1):
        result = _run_config(
            config,
            x_train,
            y_train,
            x_calib,
            y_calib,
            x_holdout,
            y_holdout,
            classes,
            args,
        )
        result["run"] = run_idx
        results.append(result)
        print(
            f"run={run_idx}/{len(configs)} {config['name']} "
            f"calib_acc={result['calibration']['accuracy']:.4f} "
            f"holdout_acc={result['holdout']['accuracy']:.4f} "
            f"holdout_macro={result['holdout']['macro_recall']:.4f}"
        )

    best_by_calib = max(results, key=lambda item: (item["calibration"]["accuracy"], item["calibration"]["macro_recall"]))
    best_oracle_holdout = max(results, key=lambda item: (item["holdout"]["accuracy"], item["holdout"]["macro_recall"]))
    payload = {
        "approach": "approach2_holdout_sweep",
        "source_embeddings": args.train_embeddings,
        "classes": classes,
        "seed": args.seed,
        "split_counts": {
            "train": _counts(y_train, classes),
            "calibration": _counts(y_calib, classes),
            "holdout": _counts(y_holdout, classes),
        },
        "results": results,
        "best_by_calibration": best_by_calib,
        "best_oracle_holdout": best_oracle_holdout,
        "note": "best_by_calibration is the honest selection. best_oracle_holdout is diagnostic only.",
    }
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "best_by_calibration "
        f"name={best_by_calib['config']['name']} "
        f"holdout_acc={best_by_calib['holdout']['accuracy']:.4f} "
        f"holdout_macro={best_by_calib['holdout']['macro_recall']:.4f}"
    )
    print(
        "best_oracle_holdout "
        f"name={best_oracle_holdout['config']['name']} "
        f"holdout_acc={best_oracle_holdout['holdout']['accuracy']:.4f} "
        f"holdout_macro={best_oracle_holdout['holdout']['macro_recall']:.4f}"
    )
    print(f"wrote {output_path}")


class EmbeddingHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, projection_dim: int, num_classes: int, dropout: float) -> None:
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


def _configs(profile: str) -> list[dict]:
    configs = []
    if profile == "quick":
        raw = itertools.product(
            [1024],
            [0.0, 0.15],
            [0.0, 0.03],
            [0.0, 0.005, 0.02],
            ["shuffle"],
            [False, True],
        )
    else:
        raw = itertools.product(
            [768, 1024],
            [0.0, 0.15],
            [0.0, 0.03],
            [0.0, 0.005, 0.02],
            ["shuffle", "balanced"],
            [False, True],
        )
    for hidden_dim, dropout, label_smoothing, supcon_weight, sampler, normalize in raw:
        if sampler == "balanced" and supcon_weight == 0.0 and not normalize:
            continue
        name = (
            f"h{hidden_dim}_d{dropout:g}_ls{label_smoothing:g}_"
            f"sup{supcon_weight:g}_{sampler}_norm{int(normalize)}"
        )
        configs.append(
            {
                "name": name,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "label_smoothing": label_smoothing,
                "supcon_weight": supcon_weight,
                "temperature": 0.2,
                "sampler": sampler,
                "normalize": normalize,
                "class_weights": "balanced",
            }
        )
    return configs


def _run_config(
    config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_calib: np.ndarray,
    y_calib: np.ndarray,
    x_holdout: np.ndarray,
    y_holdout: np.ndarray,
    classes: list[str],
    args: argparse.Namespace,
) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if config["normalize"]:
        x_train = _l2_normalize(x_train)
        x_calib = _l2_normalize(x_calib)
        x_holdout = _l2_normalize(x_holdout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmbeddingHead(
        input_dim=x_train.shape[1],
        hidden_dim=config["hidden_dim"],
        projection_dim=64,
        num_classes=len(classes),
        dropout=config["dropout"],
    ).to(device)
    loader = _loader(x_train, y_train, args.batch_size, config["sampler"])
    criterion = nn.CrossEntropyLoss(
        weight=_class_weights(y_train, len(classes)).to(device),
        label_smoothing=config["label_smoothing"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_state = None
    best_calib = None
    x_calib_t = torch.from_numpy(x_calib).to(device)
    y_calib_t = torch.from_numpy(y_calib).to(device)
    for epoch in range(args.epochs):
        _train_epoch(model, loader, criterion, optimizer, device, config)
        scheduler.step()
        if epoch == args.epochs - 1 or (epoch + 1) % 20 == 0:
            calib_metrics = _evaluate(model, x_calib_t, y_calib_t, classes)
            if best_calib is None or (calib_metrics["accuracy"], calib_metrics["macro_recall"]) >= (
                best_calib["accuracy"],
                best_calib["macro_recall"],
            ):
                best_calib = calib_metrics
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    holdout = _evaluate(model, torch.from_numpy(x_holdout).to(device), torch.from_numpy(y_holdout).to(device), classes)
    return {"config": config, "calibration": best_calib, "holdout": holdout}


def _train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device, config: dict) -> None:
    model.train()
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, projected = model(x_batch)
        ce_loss = criterion(logits, y_batch)
        loss = ce_loss + config["supcon_weight"] * _supcon(projected, y_batch, config["temperature"])
        loss.backward()
        optimizer.step()


def _supcon(projected: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
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
    return -((log_prob * positive_mask).sum(dim=1)[valid] / positive_count[valid]).mean()


@torch.no_grad()
def _evaluate(model: nn.Module, x_eval: torch.Tensor, y_eval: torch.Tensor, classes: list[str]) -> dict:
    model.eval()
    logits, _ = model(x_eval)
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    actual = y_eval.detach().cpu().numpy()
    return _metrics(preds, actual, classes)


def _metrics(preds: np.ndarray, actual: np.ndarray, classes: list[str]) -> dict:
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


def _loader(x_train: np.ndarray, y_train: np.ndarray, batch_size: int, sampler_mode: str) -> DataLoader:
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


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norm, 1e-8, None)).astype(np.float32)


def _counts(labels: np.ndarray, classes: list[str]) -> dict[str, int]:
    counts = np.bincount(labels, minlength=len(classes))
    return {class_name: int(counts[idx]) for idx, class_name in enumerate(classes)}


def _read_classes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(item["code"]) for item in payload["labels"]]


if __name__ == "__main__":
    main()
