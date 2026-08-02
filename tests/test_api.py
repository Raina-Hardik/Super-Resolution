from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app
from core.auth import generate_token

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text


@patch("api.routes.process_job.apply_async")
def test_submit_job(mock_apply_async):
    mock_apply_async.return_value = MagicMock(id="test_job_123")

    # Create dummy image
    file_data = b"dummy_image_data"
    files = {"file": ("test.png", file_data, "image/png")}

    response = client.post("/api/v1/jobs", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "PENDING"
    mock_apply_async.assert_called_once()


@patch("api.routes.AsyncResult")
def test_job_status(mock_async_result):
    # Mock pending status
    mock_task = MagicMock()
    mock_task.status = "PROCESSING"
    mock_task.info = {"progress": 45}
    mock_async_result.return_value = mock_task

    response = client.get("/api/v1/jobs/test_job_123")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "PROCESSING"
    assert data["progress"] == 45


def test_cache_bust_requires_auth():
    # Without auth
    response = client.post("/api/v1/cache/bust")
    assert response.status_code in [401, 403]

    # With guest auth (no admin role)
    from core.auth import generate_token
    guest_token = generate_token({"role": "guest"})
    response = client.post("/api/v1/cache/bust", headers={"Authorization": f"Bearer {guest_token}"})
    assert response.status_code == 403


@patch("api.routes.redis_client.keys")
@patch("api.routes.redis_client.delete")
def test_cache_bust_with_admin_auth(mock_delete, mock_keys):
    mock_keys.return_value = ["tile_cache:1", "tile_cache:2"]

    admin_token = generate_token({"role": "admin"})
    response = client.post("/api/v1/cache/bust", headers={"Authorization": f"Bearer {admin_token}"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["cleared_keys"] == 2
    mock_delete.assert_called_once_with("tile_cache:1", "tile_cache:2")


@patch("api.routes.process_job.apply_async")
def test_rate_limiting(mock_apply_async):
    mock_apply_async.return_value = MagicMock(id="test_job_rate_limit")

    # Reset rate limit store for this test
    from api.dependencies import _rate_limit_store
    _rate_limit_store.clear()

    # Send 11 requests, 11th should fail
    # Since TestClient uses "testclient" as IP, it will be rate limited
    file_data = b"dummy_image_data"

    for _i in range(10):
        response = client.post("/api/v1/jobs", files={"file": ("test.png", file_data, "image/png")})
        assert response.status_code == 200

    response = client.post("/api/v1/jobs", files={"file": ("test.png", file_data, "image/png")})
    assert response.status_code == 429
    assert response.json()["detail"] == "Too Many Requests"
