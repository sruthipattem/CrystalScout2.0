"""API logic and API sign-in, tested without a web server (no fastapi needed)."""
import json
import math

import pytest

from api import security, service
from crystalscout.config import OBELIX_DIR


def test_health_and_models():
    assert service.health()["status"] == "ok"
    m = service.models_info()
    assert set(m["models"]) == {"sendek2019", "logreg", "random_forest", "gboost"}
    json.dumps(m)


def test_predict_cif_and_descriptors_agree():
    cif = (OBELIX_DIR / "cifs" / "jqc.cif").read_text()
    r = service.predict_cif(cif, "jqc.cif")
    assert r["formula"] == "Li7BiO6" and 0 <= r["probability"] <= 1
    assert r["verdict"] in {"promising", "borderline", "unlikely"}
    d = service.predict_descriptors(r["descriptors"])
    assert math.isclose(d["probability"], r["probability"])
    json.dumps(r)


@pytest.mark.parametrize("bad", [{}, {"LLB": "x", "SBI": 1, "AFC": 1, "LASD": 1, "LLSD": 1},
                                 {"LLB": float("nan"), "SBI": 1, "AFC": 1, "LASD": 1, "LLSD": 1}])
def test_bad_descriptors(bad):
    with pytest.raises(service.InputError):
        service.predict_descriptors(bad)


@pytest.mark.parametrize("bad", ["", "not a cif"])
def test_bad_cif(bad):
    with pytest.raises(service.InputError):
        service.predict_cif(bad)


def test_screen_and_run_detail():
    s = service.screen(band_gap_min=2.0, top=5, run_by="test")
    assert len(s["results"]) == 5 and [r["rank"] for r in s["results"]] == [1, 2, 3, 4, 5]
    json.dumps(s)
    assert service.run_detail(s["summary"]["run_id"], 3)["summary"]["run_id"] == s["summary"]["run_id"]
    with pytest.raises(service.NotFound):
        service.run_detail("does-not-exist")
    with pytest.raises(service.InputError):
        service.screen(rank_model="nope")


def test_api_tokens(monkeypatch):
    from webapp import auth

    auth._failures.clear()
    with pytest.raises(security.LoginError):
        security.login("scientist", "wrong")
    tok = security.login("scientist", "test-only-scientist-password")["access_token"]
    assert security.verify(tok) == "scientist"
    for bad in [None, "abc", tok + "x"]:
        with pytest.raises(security.LoginError):
            security.verify(bad)
    monkeypatch.setitem(auth.USERS["scientist"], "ver", "changed")
    with pytest.raises(security.LoginError):
        security.verify(tok)
    auth._failures.clear()
