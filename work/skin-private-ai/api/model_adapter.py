from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/models/skin_classifier.onnx"))
DEFAULT_LABEL_MAP_PATH = Path(os.getenv("LABEL_MAP_PATH", "/app/models/label_map.json"))
DEFAULT_PRIOR_PROFILE_PATH = Path(os.getenv("PRIOR_PROFILE_PATH", "/app/models/prior_profiles.json"))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class ModelLabel:
    code: str
    label: str
    rationale: str


@dataclass(frozen=True)
class OnnxPrediction:
    code: str
    label: str
    score: float
    rationale: str


class OnnxSkinClassifier:
    def __init__(self, model_path: Path, label_map_path: Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is not installed in the runtime image.") from exc

        if not model_path.exists():
            raise FileNotFoundError(model_path)
        if not label_map_path.exists():
            raise FileNotFoundError(label_map_path)

        payload = json.loads(label_map_path.read_text(encoding="utf-8"))
        self.problem_type = payload.get("problem_type", "multilabel")
        self.labels = [
            ModelLabel(
                code=item["code"],
                label=item["label"],
                rationale=item.get("rationale", "The trained classifier found a matching visual pattern."),
            )
            for item in payload["labels"]
        ]

        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.model_path = model_path
        self.calibration = calibration_metadata()

    def predict(self, image: Image.Image) -> list[OnnxPrediction]:
        tensor = _preprocess_for_classifier(image)
        outputs = self.session.run(None, {self.input_name: tensor})
        logits = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        logits = _apply_prior_correction(logits, self.labels)
        scores = _scores_from_logits(logits, self.problem_type)

        if len(scores) != len(self.labels):
            raise ValueError(
                f"Model output has {len(scores)} scores but label map has {len(self.labels)} labels."
            )

        predictions = [
            OnnxPrediction(
                code=label.code,
                label=label.label,
                score=float(score),
                rationale=label.rationale,
            )
            for label, score in zip(self.labels, scores)
        ]
        return sorted(predictions, key=lambda item: item.score, reverse=True)


@lru_cache(maxsize=1)
def get_onnx_classifier() -> OnnxSkinClassifier | None:
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    label_map_path = Path(os.getenv("LABEL_MAP_PATH", str(DEFAULT_LABEL_MAP_PATH)))
    if not model_path.exists() or not label_map_path.exists():
        return None
    return OnnxSkinClassifier(model_path, label_map_path)


def _preprocess_for_classifier(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))
    return arr[np.newaxis, :, :, :].astype(np.float32)


def _scores_from_logits(logits: np.ndarray, problem_type: str) -> np.ndarray:
    if problem_type == "multiclass":
        shifted = logits - np.max(logits)
        exp = np.exp(shifted)
        return exp / np.sum(exp)
    return 1.0 / (1.0 + np.exp(-logits))


def _apply_prior_correction(logits: np.ndarray, labels: list[ModelLabel]) -> np.ndarray:
    profile_name = os.getenv("PRIOR_PROFILE", "").strip()
    if not profile_name:
        return logits
    alpha = float(os.getenv("PRIOR_ALPHA", "0.0"))
    if alpha <= 0:
        return logits
    train_prior = _load_prior_mapping(os.getenv("TRAINING_PRIOR_PATH", ""))
    target_prior = _load_target_prior(profile_name)
    if not train_prior or not target_prior:
        return logits
    adjusted = logits.copy()
    for idx, label in enumerate(labels):
        source = max(float(train_prior.get(label.code, 0.0)), 1e-8)
        target = max(float(target_prior.get(label.code, 0.0)), 1e-8)
        adjusted[idx] += alpha * (np.log(target) - np.log(source))
    return adjusted


def calibration_metadata() -> dict:
    profile_name = os.getenv("PRIOR_PROFILE", "").strip()
    alpha = float(os.getenv("PRIOR_ALPHA", "0.0"))
    return {
        "enabled": bool(profile_name) and alpha > 0,
        "profile": profile_name,
        "alpha": alpha,
    }


def _load_target_prior(profile_name: str) -> dict[str, float]:
    path = Path(os.getenv("PRIOR_PROFILE_PATH", str(DEFAULT_PRIOR_PROFILE_PATH)))
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = payload.get(profile_name, {})
    return _normalize_prior(profile)


def _load_prior_mapping(path_value: str) -> dict[str, float]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.exists():
        return {}
    return _normalize_prior(json.loads(path.read_text(encoding="utf-8")))


def _normalize_prior(values: dict) -> dict[str, float]:
    cleaned = {str(key): max(float(value), 1e-8) for key, value in values.items()}
    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in cleaned.items()}
