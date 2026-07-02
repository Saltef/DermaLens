from __future__ import annotations

import numpy as np

from api.model_adapter import ModelLabel, _apply_prior_correction, _scores_from_logits


def test_scores_from_logits_multiclass_softmax_sums_to_one() -> None:
    scores = _scores_from_logits(np.array([1.0, 2.0, 3.0], dtype=np.float32), "multiclass")

    assert np.isclose(scores.sum(), 1.0)
    assert scores.argmax() == 2


def test_scores_from_logits_multilabel_uses_sigmoid() -> None:
    scores = _scores_from_logits(np.array([0.0], dtype=np.float32), "multilabel")

    assert np.isclose(scores[0], 0.5)


def test_prior_correction_disabled_without_profile(monkeypatch) -> None:
    monkeypatch.delenv("PRIOR_PROFILE", raising=False)
    labels = [ModelLabel(code="a", label="A", rationale="")]
    logits = np.array([0.25], dtype=np.float32)

    adjusted = _apply_prior_correction(logits, labels)

    assert np.array_equal(adjusted, logits)
