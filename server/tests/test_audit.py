from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_audit_events_endpoint_requires_auth():
    response = client.get("/api/v1/audit/events")
    assert response.status_code in (401, 403)


def test_audit_events_endpoint_with_auth():
    # Login as operator
    login_res = client.post(
        "/api/v1/auth/login",
        json={"email": "operator@reconpilot.local", "password": "OperatorSecret2026!"},
    )
    assert login_res.status_code == 200
    token = login_res.json()["data"]["access_token"]

    response = client.get(
        "/api/v1/audit/events",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert "items" in data["data"]
    assert "meta" in data["data"]
    assert data["data"]["meta"]["total_items"] >= 0
    assert isinstance(data["data"]["items"], list)
