from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_shape() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "save_uploads" in payload["privacy"]


def test_non_image_upload_rejected() -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415


def test_corrupt_image_returns_400_not_500() -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("broken.jpg", b"not really an image", "image/jpeg")},
    )

    assert response.status_code == 400


def test_large_upload_rejected() -> None:
    response = client.post(
        "/api/analyze",
        files={"file": ("large.jpg", b"0" * (12 * 1024 * 1024 + 1), "image/jpeg")},
    )

    assert response.status_code == 413
