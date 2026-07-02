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
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train several embedding classifiers and sweep probability ensembles.")
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--base-val", required=True)
    parser.add_argument("--aug-train", required=True)
    parser.add_argument("--aug-val", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-step", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = _read_classes(Path(args.classes))

    base = _load_scaled(args.base_train, args.base_val)
    aug = _load_scaled(args.aug_train, args.aug_val)
    if not np.array_equal(base["y_val"], aug["y_val"]):
        raise ValueError("Validation labels differ; ensemble predictions would not align.")
    y_val = base["y_val"]

    members = []
    members.append(_fit_logreg("base_logreg_c01", base, classes, c_value=0.1))
    members.append(_fit_logreg("base_logreg_c03", base, classes, c_value=0.3))
    members.append(_fit_logreg("aug_logreg_c10", aug, classes, c_value=10.0))

    head_specs = [
        ("base_head_768_seed2", base, 768, 0.0004, 90, 2),
        ("base_head_768_seed7", base, 768, 0.0004, 90, 7),
        ("base_head_1024_seed7", base, 1024, 0.0006, 110, 7),
        ("aug_head_1024_seed7", aug, 1024, 0.0004, 70, 7),
    ]
    for name, data, hidden_dim, lr, epochs, seed in head_specs:
        members.append(
            _fit_head(
                name,
                data,
                classes,
                hidden_dim=hidden_dim,
                lr=lr,
                epochs=min(epochs, args.epochs),
                batch_size=args.batch_size,
                seed=seed,
            )
        )

    individual = []
    for member in members:
        metrics = _metrics(member["probs"].argmax(axis=1), y_val, classes)
        metrics["name"] = member["name"]
        individual.append(metrics)
        print(f"{member['name']} accuracy={metrics['accuracy']:.4f} macro_recall={metrics['macro_recall']:.4f}")

    ensembles = []
    units = max(1, int(round(1.0 / args.weight_step)))
    for weight_units in _simplex_units(len(members), units):
        normalized = np.asarray(weight_units, dtype=np.float32) / float(units)
        probs = sum(float(weight) * member["probs"] for weight, member in zip(normalized, members))
        preds = probs.argmax(axis=1)
        metrics = _metrics(preds, y_val, classes)
        metrics["weights"] = {
            member["name"]: float(weight)
            for member, weight in zip(members, normalized)
            if weight > 0
        }
        ensembles.append(metrics)

    best_accuracy = max(ensembles, key=lambda item: (item["accuracy"], item["macro_recall"]))
    best_macro = max(ensembles, key=lambda item: (item["macro_recall"], item["accuracy"]))
    payload = {
        "approach": "mixed_embedding_classifier_ensemble",
        "classes": classes,
        "individual": individual,
        "best_accuracy": best_accuracy,
        "best_macro_recall": best_macro,
        "weight_step": args.weight_step,
        "member_names": [member["name"] for member in members],
    }
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"best_accuracy accuracy={best_accuracy['accuracy']:.4f} macro_recall={best_accuracy['macro_recall']:.4f}")
    print(f"best_macro accuracy={best_macro['accuracy']:.4f} macro_recall={best_macro['macro_recall']:.4f}")
    print(f"wrote {output_path}")


class Head(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_logreg(name: str, data: dict, classes: list[str], *, c_value: float) -> dict:
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000, solver="lbfgs"),
    )
    model.fit(data["x_train"], data["y_train"])
    probs = model.predict_proba(data["x_val"])
    return {"name": name, "probs": _align_probs(probs, model[-1].classes_, len(classes))}


def _fit_head(
    name: str,
    data: dict,
    classes: list[str],
    *,
    hidden_dim: int,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Head(data["x_train"].shape[1], hidden_dim, len(classes)).to(device)
    x_train = torch.from_numpy(data["x_train"])
    y_train = torch.from_numpy(data["y_train"])
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    weights = _class_weights(data["y_train"], len(classes)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

    for _ in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(data["x_val"]).to(device))
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return {"name": name, "probs": probs}


def _load_scaled(train_path: str, val_path: str) -> dict:
    train_npz = np.load(train_path)
    val_npz = np.load(val_path)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_npz["x"].astype(np.float32)).astype(np.float32)
    x_val = scaler.transform(val_npz["x"].astype(np.float32)).astype(np.float32)
    return {
        "x_train": x_train,
        "y_train": train_npz["y"].astype(np.int64),
        "x_val": x_val,
        "y_val": val_npz["y"].astype(np.int64),
    }


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


def _align_probs(probs: np.ndarray, class_indices: np.ndarray, num_classes: int) -> np.ndarray:
    aligned = np.zeros((probs.shape[0], num_classes), dtype=np.float32)
    for source_idx, class_idx in enumerate(class_indices):
        aligned[:, int(class_idx)] = probs[:, source_idx]
    return aligned


def _class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.clip(counts, 1.0, None)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _read_classes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(item["code"]) for item in payload["labels"]]


def _simplex_units(size: int, total: int) -> list[tuple[int, ...]]:
    if size == 1:
        return [(total,)]
    weights = []
    for value in range(total + 1):
        for suffix in _simplex_units(size - 1, total - value):
            weights.append((value, *suffix))
    return weights


if __name__ == "__main__":
    main()
