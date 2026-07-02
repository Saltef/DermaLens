from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Confirm ensemble calibration on a fresh stratified holdout split.")
    parser.add_argument("--train-embeddings", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calib-size", type=float, default=0.15)
    parser.add_argument("--holdout-size", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-step", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
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
    relative_holdout = args.holdout_size / (args.calib_size + args.holdout_size)
    calib_idx, holdout_idx = train_test_split(
        temp_idx,
        test_size=relative_holdout,
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

    members = _train_members(x_train, y_train, x_calib, x_holdout, classes, args)
    calib_probs = [member["calib_probs"] for member in members]
    holdout_probs = [member["holdout_probs"] for member in members]

    best_weighted = _best_weighted_ensemble(calib_probs, holdout_probs, y_calib, y_holdout, classes, args.weight_step)
    best_biased = _tune_bias_on_calib(
        best_weighted["calib_probs"],
        best_weighted["holdout_probs"],
        y_calib,
        y_holdout,
        classes,
    )

    individual = []
    for member in members:
        individual.append(
            {
                "name": member["name"],
                "calib": _metrics(member["calib_probs"].argmax(axis=1), y_calib, classes),
                "holdout": _metrics(member["holdout_probs"].argmax(axis=1), y_holdout, classes),
            }
        )

    payload = {
        "approach": "fresh_holdout_confirmation",
        "source_embeddings": args.train_embeddings,
        "classes": classes,
        "seed": args.seed,
        "split_counts": {
            "train": _counts(y_train, classes),
            "calibration": _counts(y_calib, classes),
            "holdout": _counts(y_holdout, classes),
        },
        "individual": individual,
        "best_weighted_ensemble": _without_probs(best_weighted),
        "best_bias_calibrated_ensemble": _without_probs(best_biased),
        "note": (
            "Weights and class bias are selected only on the calibration split. "
            "The holdout split is not used for tuning."
        ),
    }
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        "best_weighted_holdout "
        f"accuracy={best_weighted['holdout']['accuracy']:.4f} "
        f"macro_recall={best_weighted['holdout']['macro_recall']:.4f}"
    )
    print(
        "best_bias_holdout "
        f"accuracy={best_biased['holdout']['accuracy']:.4f} "
        f"macro_recall={best_biased['holdout']['macro_recall']:.4f}"
    )
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


def _train_members(x_train: np.ndarray, y_train: np.ndarray, x_calib: np.ndarray, x_holdout: np.ndarray, classes: list[str], args: argparse.Namespace) -> list[dict]:
    members = [
        _fit_logreg("logreg_c01", x_train, y_train, x_calib, x_holdout, classes, c_value=0.1),
        _fit_logreg("logreg_c03", x_train, y_train, x_calib, x_holdout, classes, c_value=0.3),
        _fit_head("head_768_seed2", x_train, y_train, x_calib, x_holdout, classes, 768, 0.0004, args.epochs, args.batch_size, 2),
        _fit_head("head_768_seed7", x_train, y_train, x_calib, x_holdout, classes, 768, 0.0004, args.epochs, args.batch_size, 7),
        _fit_head("head_1024_seed7", x_train, y_train, x_calib, x_holdout, classes, 1024, 0.0006, args.epochs, args.batch_size, 7),
    ]
    return members


def _fit_logreg(name: str, x_train: np.ndarray, y_train: np.ndarray, x_calib: np.ndarray, x_holdout: np.ndarray, classes: list[str], *, c_value: float) -> dict:
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(C=c_value, class_weight="balanced", max_iter=3000, solver="lbfgs"),
    )
    model.fit(x_train, y_train)
    return {
        "name": name,
        "calib_probs": _align_probs(model.predict_proba(x_calib), model[-1].classes_, len(classes)),
        "holdout_probs": _align_probs(model.predict_proba(x_holdout), model[-1].classes_, len(classes)),
    }


def _fit_head(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_calib: np.ndarray,
    x_holdout: np.ndarray,
    classes: list[str],
    hidden_dim: int,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Head(x_train.shape[1], hidden_dim, len(classes)).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
    )
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_train, len(classes)).to(device))
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
        calib_probs = torch.softmax(model(torch.from_numpy(x_calib).to(device)), dim=1).detach().cpu().numpy()
        holdout_probs = torch.softmax(model(torch.from_numpy(x_holdout).to(device)), dim=1).detach().cpu().numpy()
    return {"name": name, "calib_probs": calib_probs, "holdout_probs": holdout_probs}


def _best_weighted_ensemble(
    calib_probs: list[np.ndarray],
    holdout_probs: list[np.ndarray],
    y_calib: np.ndarray,
    y_holdout: np.ndarray,
    classes: list[str],
    weight_step: float,
) -> dict:
    units = max(1, int(round(1.0 / weight_step)))
    best = None
    for weight_units in _simplex_units(len(calib_probs), units):
        weights = np.asarray(weight_units, dtype=np.float32) / float(units)
        calib = sum(float(weight) * probs for weight, probs in zip(weights, calib_probs))
        calib_metrics = _metrics(calib.argmax(axis=1), y_calib, classes)
        if best is None or (calib_metrics["accuracy"], calib_metrics["macro_recall"]) > (
            best["calibration"]["accuracy"],
            best["calibration"]["macro_recall"],
        ):
            holdout = sum(float(weight) * probs for weight, probs in zip(weights, holdout_probs))
            best = {
                "weights": [float(weight) for weight in weights],
                "calib_probs": calib,
                "holdout_probs": holdout,
                "calibration": calib_metrics,
                "holdout": _metrics(holdout.argmax(axis=1), y_holdout, classes),
            }
    return best


def _tune_bias_on_calib(
    calib_probs: np.ndarray,
    holdout_probs: np.ndarray,
    y_calib: np.ndarray,
    y_holdout: np.ndarray,
    classes: list[str],
) -> dict:
    factors = np.asarray([0.45, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.45, 1.7, 2.0], dtype=np.float32)
    bias = np.ones(len(classes), dtype=np.float32)
    best_calib = _metrics((calib_probs * bias).argmax(axis=1), y_calib, classes)
    for _ in range(4):
        improved = False
        for class_idx in range(len(classes)):
            current = bias[class_idx]
            for factor in factors:
                candidate = bias.copy()
                candidate[class_idx] = current * factor
                metrics = _metrics((calib_probs * candidate).argmax(axis=1), y_calib, classes)
                if (metrics["accuracy"], metrics["macro_recall"]) > (
                    best_calib["accuracy"],
                    best_calib["macro_recall"],
                ):
                    bias = candidate
                    best_calib = metrics
                    improved = True
        if not improved:
            break
    holdout_adjusted = holdout_probs * bias
    return {
        "class_bias": {class_name: float(value) for class_name, value in zip(classes, bias)},
        "calibration": best_calib,
        "holdout": _metrics(holdout_adjusted.argmax(axis=1), y_holdout, classes),
        "calib_probs": calib_probs * bias,
        "holdout_probs": holdout_adjusted,
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


def _simplex_units(size: int, total: int) -> list[tuple[int, ...]]:
    if size == 1:
        return [(total,)]
    weights = []
    for value in range(total + 1):
        for suffix in _simplex_units(size - 1, total - value):
            weights.append((value, *suffix))
    return weights


def _counts(labels: np.ndarray, classes: list[str]) -> dict[str, int]:
    counts = np.bincount(labels, minlength=len(classes))
    return {class_name: int(counts[idx]) for idx, class_name in enumerate(classes)}


def _without_probs(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if not key.endswith("_probs")}


def _read_classes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(item["code"]) for item in payload["labels"]]


if __name__ == "__main__":
    main()
