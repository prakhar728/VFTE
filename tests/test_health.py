from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def test_health_ok():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["service"] == "fpm"
    assert "version" in body
