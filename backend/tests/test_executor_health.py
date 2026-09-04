from fastapi.testclient import TestClient

from executor.main import app

client = TestClient(app)


def test_executor_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "executor"}
