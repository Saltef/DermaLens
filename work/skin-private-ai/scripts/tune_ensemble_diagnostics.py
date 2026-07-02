from __future__ import annotations

import argparse
import csv
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
from torchvision import datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run calibration and error-audit tests on the mixed ConvNeXt ensemble.")
    parser.add_argument("--base-train", required=True)
    parser.add_argument("--base-val", required=True)
    parser.add_argument("--aug-train", required=True)
    parser.add_argument("--aug-val", required=True)
    parser.add_argument("--classes", required=True)
    parser.add_argument("--val-imagefolder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=90)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    classes = _read_classes(Path(args.classes))
    clinician_idx = classes.index("clinician_review") if "clinician_review" in classes else None

    base = _load_scaled(args.base_train, args.base_val)
    aug = _load_scaled(args.aug_train, args.aug_val)
    if not np.array_equal(base["y_val"], aug["y_val"]):
        raise ValueError("Validation labels differ; cannot compare predictions.")
    y_val = base["y_val"]

    members = _train_members(base, aug, classes, args.batch_size, args.epochs)
    member_by_name = {member["name"]: member for member in members}
    v1_probs = (
        (2.0 / 7.0) * member_by_name["base_head_768_seed7"]["probs"]
        + (2.0 / 7.0) * member_by_name["base_head_1024_seed7"]["probs"]
        + (3.0 / 7.0) * member_by_name["aug_head_1024_seed7"]["probs"]
    )
    v2_probs = (
        0.2 * member_by_name["aug_logreg_c10"]["probs"]
        + 0.4 * member_by_name["base_head_768_seed2"]["probs"]
        + 0.2 * member_by_name["base_head_768_seed7"]["probs"]
        + 0.2 * member_by_name["aug_head_1024_seed7"]["probs"]
    )

    trials = []
    for name, probs in [("mixed_v1_fixed_weights", v1_probs), ("mixed_v2_fixed_weights", v2_probs)]:
        trials.append(_trial(name, probs, y_val, classes))
        trials.extend(_temperature_trials(name, probs, y_val, classes))
        trials.extend(_confidence_fallback_trials(name, probs, y_val, classes, clinician_idx))
        trials.append(_bias_tuned_trial(name, probs, y_val, classes))

    best_accuracy = max(trials, key=lambda item: (item["accuracy"], item["macro_recall"]))
    best_macro = max(trials, key=lambda item: (item["macro_recall"], item["accuracy"]))
    val_paths = _val_paths(Path(args.val_imagefolder))
    _write_error_csv(output_dir / "best_accuracy_errors.csv", best_accuracy, y_val, val_paths, classes)
    _write_error_csv(output_dir / "mixed_v1_errors.csv", _trial("mixed_v1_fixed_weights", v1_probs, y_val, classes), y_val, val_paths, classes)

    payload = {
        "approach": "ensemble_calibration_and_error_audit",
        "classes": classes,
        "trials": [_without_probs(item) for item in sorted(trials, key=lambda item: (item["accuracy"], item["macro_recall"]), reverse=True)],
        "best_accuracy": _without_probs(best_accuracy),
        "best_macro_recall": _without_probs(best_macro),
        "note": (
            "Bias and threshold sweeps are tuned on the validation split. Treat these as diagnostic upper-bound "
            "tests unless repeated with cross-validation or a separate held-out test set."
        ),
    }
    output_path = output_dir / "metrics.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"best_accuracy={best_accuracy['name']} accuracy={best_accuracy['accuracy']:.4f} macro_recall={best_accuracy['macro_recall']:.4f}")
    print(f"best_macro={best_macro['name']} accuracy={best_macro['accuracy']:.4f} macro_recall={best_macro['macro_recall']:.4f}")
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


def _train_members(base: dict, aug: dict, classes: list[str], batch_size: int, epochs: int) -> list[dict]:
    members = [
        _fit_logreg("base_logreg_c01", base, classes, c_value=0.1),
        _fit_logreg("base_logreg_c03", base, classes, c_value=0.3),
        _fit_logreg("aug_logreg_c10", aug, classes, c_value=10.0),
    ]
    specs = [
        ("base_head_768_seed2", base, 768, 0.0004, 90, 2),
        ("base_head_768_seed7", base, 768, 0.0004, 90, 7),
        ("base_head_1024_seed7", base, 1024, 0.0006, 110, 7),
        ("aug_head_1024_seed7", aug, 1024, 0.0004, 70, 7),
    ]
    for name, data, hidden_dim, lr, spec_epochs, seed in specs:
        members.append(
            _fit_head(
                name,
                data,
                classes,
                hidden_dim=hidden_dim,
                lr=lr,
                epochs=min(spec_epochs, epochs),
                batch_size=batch_size,
                seed=seed,
            )
        )
    for member in members:
        metrics = _metrics(member["probs"].argmax(axis=1), base["y_val"], classes)
        print(f"{member['name']} accuracy={metrics['accuracy']:.4f} macro_recall={metrics['macro_recall']:.4f}")
    return members


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
    loader = DataLoader(
        TensorDataset(torch.from_numpy(data["x_train"]), torch.from_numpy(data["y_train"])),
        batch_size=batch_size,
        shuffle=True,
    )
    criterion = nn.CrossEntropyLoss(weight=_class_weights(data["y_train"], len(classes)).to(device))
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


def _temperature_trials(prefix: str, probs: np.ndarray, y_val: np.ndarray, classes: list[str]) -> list[dict]:
    trials = []
    for temperature in [0.5, 0.7, 0.85, 1.15, 1.3, 1.6, 2.0]:
        adjusted = np.power(np.clip(probs, 1e-8, 1.0), 1.0 / temperature)
        adjusted = adjusted / adjusted.sum(axis=1, keepdims=True)
        trial = _trial(f"{prefix}_temperature_{temperature:g}", adjusted, y_val, classes)
        trial["temperature"] = temperature
        trials.append(trial)
    return trials


def _confidence_fallback_trials(
    prefix: str,
    probs: np.ndarray,
    y_val: np.ndarray,
    classes: list[str],
    clinician_idx: int | None,
) -> list[dict]:
    if clinician_idx is None:
        return []
    trials = []
    for threshold in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
        preds = probs.argmax(axis=1)
        preds = np.where(probs.max(axis=1) < threshold, clinician_idx, preds)
        trial = _metrics(preds, y_val, classes)
        trial["name"] = f"{prefix}_clinician_fallback_{threshold:g}"
        trial["threshold"] = threshold
        trials.append(trial)
    return trials


def _bias_tuned_trial(prefix: str, probs: np.ndarray, y_val: np.ndarray, classes: list[str]) -> dict:
    bias = np.ones(len(classes), dtype=np.float32)
    factors = np.asarray([0.45, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.45, 1.7, 2.0], dtype=np.float32)
    best = _trial(f"{prefix}_bias_tuned", probs * bias, y_val, classes)
    for _ in range(4):
        improved = False
        for class_idx in range(len(classes)):
            class_best = best
            class_bias = bias[class_idx]
            for factor in factors:
                candidate_bias = bias.copy()
                candidate_bias[class_idx] = class_bias * factor
                candidate = _trial(f"{prefix}_bias_tuned", probs * candidate_bias, y_val, classes)
                if (candidate["accuracy"], candidate["macro_recall"]) > (class_best["accuracy"], class_best["macro_recall"]):
                    class_best = candidate
                    bias = candidate_bias
                    improved = True
            best = class_best
        if not improved:
            break
    best["class_bias"] = {class_name: float(value) for class_name, value in zip(classes, bias)}
    return best


def _trial(name: str, probs: np.ndarray, y_val: np.ndarray, classes: list[str]) -> dict:
    adjusted = probs / probs.sum(axis=1, keepdims=True)
    metrics = _metrics(adjusted.argmax(axis=1), y_val, classes)
    metrics["name"] = name
    metrics["probs"] = adjusted
    return metrics


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


def _write_error_csv(path: Path, trial: dict, actual: np.ndarray, val_paths: list[Path], classes: list[str]) -> None:
    probs = trial.get("probs")
    if probs is None:
        return
    preds = probs.argmax(axis=1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "actual", "predicted", "confidence", "second_label", "second_confidence"],
        )
        writer.writeheader()
        for idx, (pred, label) in enumerate(zip(preds, actual)):
            if int(pred) == int(label):
                continue
            order = np.argsort(probs[idx])[::-1]
            writer.writerow(
                {
                    "path": str(val_paths[idx]) if idx < len(val_paths) else "",
                    "actual": classes[int(label)],
                    "predicted": classes[int(pred)],
                    "confidence": f"{float(probs[idx, pred]):.6f}",
                    "second_label": classes[int(order[1])],
                    "second_confidence": f"{float(probs[idx, order[1]]):.6f}",
                }
            )


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


def _val_paths(path: Path) -> list[Path]:
    return [Path(sample_path) for sample_path, _ in datasets.ImageFolder(path).samples]


def _read_classes(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(item["code"]) for item in payload["labels"]]


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _without_probs(trial: dict) -> dict:
    return {key: value for key, value in trial.items() if key != "probs"}


if __name__ == "__main__":
    main()
