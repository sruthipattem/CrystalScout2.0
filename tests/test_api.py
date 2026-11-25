"""FastAPI layer. Skipped automatically if fastapi / httpx are not installed."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

from fastapi.testclient import TestClient  # noqa: E402

from crystalscout.config import OBELIX_DIR  # noqa: E402

USER, PASSWORD = "scientist", "test-only-scientist-password"


@pytest.fixture(scope="module")
def client():
    from webapp import auth
    from api.main import app

    auth._failures.clear()
    return TestClient(app)


@pytest.fixture(scope="module")
def headers(client):
    r = client.post("/auth/login", json={"username": USER, "password": PASSWORD})
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_health_is_public(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_login_rules(client):
    from webapp import auth

    assert client.post("/auth/login", json={"username": USER, "password": "wrong"}).status_code == 401
    assert client.post("/auth/login", json={"username": "admin", "password": PASSWORD}).status_code == 401
    auth._failures.clear()


def test_protected_routes_need_token(client):
    for path in ["/models", "/features", "/runs", "/auth/me"]:
        assert client.get(path).status_code == 401
    assert client.get("/models", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_models_and_features(client, headers):
    m = client.get("/models", headers=headers).json()
    assert set(m["models"]) == {"sendek2019", "logreg", "random_forest", "gboost"}
    assert "LLB" in client.get("/features", headers=headers).json()


def test_predict_cif(client, headers):
    cif = (OBELIX_DIR / "cifs" / "jqc.cif").read_bytes()
    r = client.post("/predict/cif", headers=headers, files={"file": ("jqc.cif", cif, "chemical/x-cif")})
    assert r.status_code == 200
    body = r.json()
    assert body["formula"] == "Li7BiO6" and 0 <= body["probability"] <= 1
    bad = client.post("/predict/cif", headers=headers, files={"file": ("x.cif", b"not a cif", "text/plain")})
    assert bad.status_code == 400


def test_predict_descriptors(client, headers):
    good = {"LLB": 5.0, "SBI": 0.6, "AFC": 9.0, "LASD": 2.1, "LLSD": 2.4}
    r = client.post("/predict/descriptors", headers=headers, json=good)
    assert r.status_code == 200 and "probabilities" in r.json()
    assert client.post("/predict/descriptors", headers=headers, json={"LLB": 1}).status_code == 422


def test_screen_and_runs(client, headers):
    r = client.post("/screen", headers=headers, json={"band_gap_min": 2.0, "top": 5})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 5 and body["results"][0]["rank"] == 1
    run_id = body["summary"]["run_id"]
    assert client.get(f"/runs/{run_id}?top=3", headers=headers).status_code == 200
    assert client.get("/runs/does-not-exist", headers=headers).status_code == 404
    assert client.post("/screen", headers=headers, json={"rank_model": "nope"}).status_code == 400
