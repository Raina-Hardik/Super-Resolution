from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert "version" in response.json()


def test_api_info():
    response = client.get("/api/v1/info")
    assert response.status_code == 200
    assert "name" in response.json()
    assert "endpoints" in response.json()


def test_docs():
    response = client.get("/docs")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
