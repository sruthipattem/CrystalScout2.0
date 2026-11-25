"""Web app: every page renders and every action works."""
import io
import re
import time

import pytest

from crystalscout.config import OBELIX_DIR


@pytest.fixture(scope="module")
def client():
    from webapp.app import app

    app.config["TESTING"] = True
    c = app.test_client()
    r = c.post("/login", data={"username": "scientist", "password": "test-only-scientist-password"})
    assert r.status_code == 302
    return c


@pytest.mark.parametrize("url", ["/", "/screen", "/shortlist", "/predict", "/explainer", "/models", "/data",
                                 "/methods", "/material/mp-532413", "/data?q=garnet", "/simple", "/developers",
                                 "/account", "/predict?formula=Li6PS5Cl%0ALi3InCl6+C2/m", "/shortlist?model=sigma&family=sulfide",
                                 "/shortlist?model=crystalscout&no_risk=1&no_tox=1&struct=1&domain=in"])
def test_pages_render(client, url):
    r = client.get(url)
    assert r.status_code == 200, url
    assert b"CrystalScout" in r.data


def test_compare_and_estimate(client):
    from webapp.app import STATE

    ids = STATE.default_run().shortlist["material_id"].head(3).tolist()
    r = client.get("/compare?ids=" + ",".join(ids))
    assert r.status_code == 200 and b"Side-by-side" in r.data
    assert client.get("/compare?ids=" + ids[0]).status_code == 400
    d = client.get("/api/estimate?formula=Li6PS5Cl").get_json()
    assert d["tier"] and 0 <= d["crystalscout_score"] <= 1
    assert client.get("/api/estimate?formula=XyZ").status_code == 400


def test_material_without_structure(client):
    from crystalscout.data.materials_project import load_snapshot
    from crystalscout.featurize import load_pool_features

    pool = set(load_pool_features()["material_id"])
    snap = load_snapshot()
    mid = snap[~snap["material_id"].isin(pool)]["material_id"].iloc[0]
    r = client.get(f"/material/{mid}")
    assert r.status_code == 200 and b"composition only" in r.data


def test_formula_predict_post(client):
    r = client.post("/predict", data={"kind": "formula", "formulas": "Li6PS5Cl\nLi3PO4\nBadThing"})
    assert r.status_code == 200 and b"Quick estimates" in r.data and b"must contain Li" in r.data


def test_404(client):
    assert client.get("/material/mp-does-not-exist").status_code == 404


def test_cif_upload_and_detail(client):
    cif = (OBELIX_DIR / "cifs" / "jqc.cif").read_bytes()
    r = client.post("/predict", data={"cifs": [(io.BytesIO(cif), "a.cif"), (io.BytesIO(b"junk"), "b.cif")]},
                    content_type="multipart/form-data")
    assert r.status_code == 200 and b"Could not read" in r.data
    m = re.search(rb"/predict/([0-9a-f]+)/0", r.data)
    assert client.get(m.group(0).decode()).status_code == 200
    assert client.get(f"/predict/{m.group(1).decode()}.csv").status_code == 200


def test_csv_scoring(client):
    csv = b"formula,LLB,SBI,AFC,LASD,LLSD\nA,1,0.1,8,2.5,2.4\n"
    r = client.post("/predict", data={"kind": "csv", "csv": (io.BytesIO(csv), "x.csv")}, content_type="multipart/form-data")
    assert r.status_code == 200
    r = client.post("/predict", data={"kind": "csv", "csv": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")}, content_type="multipart/form-data")
    assert r.status_code == 400


def test_empty_predict(client):
    assert client.post("/predict", data={}).status_code == 400


def test_screen_job_and_exports(client):
    r = client.post("/screen", data={"source": "snapshot", "band_gap_min": "2", "ehull_max": "0.01", "exclude_tm": "on",
                                     "exclude_elements": "Ge", "max_nsites": "0", "rank_model": "crystalscout"})
    assert r.status_code == 302
    jid = r.location.rsplit("/", 1)[1]
    for _ in range(120):
        j = client.get(f"/api/jobs/{jid}").get_json()
        if j["status"] != "running":
            break
        time.sleep(0.25)
    assert j["status"] == "done", j
    rid = j["run_id"]
    assert client.get(f"/runs/{rid}?model=random_forest&domain=in").status_code == 200
    assert client.get(f"/runs/{rid}/rejected?reason=band gap too small").status_code == 200
    pdf = client.get(f"/runs/{rid}/report.pdf")
    assert pdf.status_code == 200 and pdf.data[:4] == b"%PDF"
    assert client.get(f"/runs/{rid}/candidates.csv").status_code == 200


def test_snapshot_range_guard(client):
    assert client.post("/screen", data={"source": "snapshot", "band_gap_min": "0.1", "ehull_max": "0"}).status_code == 400


def test_api_predict(client):
    r = client.post("/api/predict", json={"LLB": 1, "SBI": .1, "AFC": 8, "LASD": 2.5, "LLSD": 2.4})
    d = r.get_json()
    assert set(d["probabilities"]) == {"logreg", "random_forest", "gboost", "sendek2019"}
    assert client.post("/api/predict", json={"LLB": 1}).status_code == 400
