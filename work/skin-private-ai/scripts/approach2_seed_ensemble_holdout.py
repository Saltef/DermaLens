from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test approach-2 seed ensembles on a fresh holdout split.")
    parser.add_argument("--train-embeddings", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--calib-size", type=float, default=0.15)
    parser.add_argument("--holdout-size", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=128)
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
    calib_idx, holdout_idx = train_test_split(temp_idx, test_size=holdout_relative, stratify=y[temp_idx], random_state=args.seed)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx]).astype(np.float32)
    x_calib = scaler.transform(x[calib_idx]).astype(np.float32)
    x_holdout = scaler.transform(x[holdout_idx]).astype(np.float32)
    y_train = y[train_idx]
    y_calib = y[calib_idx]
    y_holdout = y[holdout_idx]

    configs = [
        {"name": "h1024_plain", "hidden_dim": 1024, "dropout": 0.0, "label_smoothing": 0.0, "lr": 4e-4},
        {"name": "h1024_smooth", "hidden_dim": 1024, "dropout": 0.0, "label_smoothing": 0.03, "lr": 4e-4},
        {"name": "h1024_dropout", "hidden_dim": 1024, "dropout": 0.15, "label_smoothing": 0.0, "lr": 4e-4},
        {"name": "h768_plain", "hidden_dim": 768, "dropout": 0.0, "label_smoothing": 0.0, "lr": 4e-4},
    ]
    member_seeds = [1, 2, 3, 5, 7]
    results = []
    for config in configs:
        calib_probs = []
        holdout_probs = []
        members = []
        for member_seed in member_seeds:
            probs = _fit_head(config, member_seed, x_train, y_train, x_calib, x_holdout, classes, args)
            calib_probs.append(probs["calib"])
            holdout_probs.append(probs["holdout"])
            members.append(
                {
                    "seed": member_seed,
                    "calibration": _metrics(probs["calib"].argmax(axis=1), y_calib, classes),
                    "holdout": _metrics(probs["holdout"].argmax(axis=1), y_holdout, classes),
                }
            )
        ensemble_calib = np.mean(calib_probs, axis=0)
        ensemble_holdout = np.mean(holdout_probs, axis=0)
        result = {
            "config": config,
            "member_seeds": member_seeds,
            "members": members,
            "ensemble_calibration": _metrics(ensemble_calib.argmax(axis=1), y_calib, classes),
            "ensemble_holdout": _metrics(ensemble_holdout.argmax(axis=1), y_holdout, classes),
        }
        results.append(result)
        print(
            f"{config['name']} calib_acc={result['ensemble_calibration']['accuracy']:.4f} "
            f"holdout_acc={result['ensemble_holdout']['accuracy']:.4f} "
            f"holdout_macro={result['ensemble_holdout']['macro_recall']:.4f}"
        )

    best = max(results, key=lambda item: (item["ensemble_calibration"]["accuracy"], item["ensemble_calibration"]["macro_recall"]))
    oracle = max(results, key=lambda item: (item["ensemble_holdout"]["accuracy"], item["ensemble_holdout"]["macro_recall"]))
    payload = {
        "approach": "approach2_seed_ensemble_holdout",
        "classes": classes,
        "seed": args.seed,
        "results": results,
        "best_by_calibration": best,
        "best_oracle_holdout": oracle,
        "note": "Seed ensembles are approach-2 neural heads only. best_oracle_holdout is diagnostic only.",
    }
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"best_by_calibration={best['config']['name']} "
        f"holdout_acc={best['ensemble_holdout']['accuracy']:.4f} "
        f"holdout_macro={best['ensemble_holdout']['macro_recall']:.4f}"
    )
    print(f"wrote {output_path}")


class Head(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_head(config: dict, seed: int, x_train: np.ndarray, y_train: np.ndarray, x_calib: np.ndarray, x_holdout: np.ndarray, classes: list[str], args: argparse.Namespace) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Head(x_train.shape[1], config["hidden_dim"], len(classes), config["dropout"]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)), batch_size=args.batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss(weight=_class_weights(y_train, len(classes)).to(device), label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    for _ in range(args.epochs):
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
        calib = torch.softmax(model(torch.from_numpy(x_calib).to(device)), dim=1).detach().cpu().numpy()
        holdout = torch.softmax(model(torch.from_numpy(x_holdout).to(device)), dim=1).detach().cpu().numpy()
    return {"calib": calib, "holdout": holdout}


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


def _class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.clip(counts, 1.0, None)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def _read_classes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(item["code"]) for item in payload["labels"]]


if __name__ == "__main__":
    main()
