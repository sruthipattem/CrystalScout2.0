"""Existence check, feasibility, CIF generation and the held-out validation."""
import pytest

from crystalscout.core.cif import parse_cif
from crystalscout.ml.design import assign_charges, generate_variants
from crystalscout.ml.regression import get_regressor, parse_formula
from crystalscout.ml.structgen import assess, comp_key, generate, lookup, size_fit


@pytest.fixture(scope="module")
def reg():
    return get_regressor(verbose=False)


@pytest.fixture(scope="module")
def bundle():
    from crystalscout.ml.models import get_bundle

    return get_bundle()


def test_existence_lookup():
    assert lookup(parse_formula("Li6PS5Cl"))["status"] == "measured"
    assert lookup(parse_formula("Cl1Li6P1S5"))["status"] == "measured"        # order and 1s do not matter
    k = lookup(parse_formula("Li2S"))
    assert k["status"] == "known" and k["best_mp"]["e_hull"] == 0
    assert lookup(parse_formula("Li6PS5Cl0.3Br0.7"))["status"] == "hypothetical"
    assert comp_key({"Li": 2, "S": 1}) == comp_key({"S": 0.5, "Li": 1})


def test_size_fit():
    assert size_fit("Br", -1, "Cl", -1)["level"] == "good"
    assert size_fit("Ge", 4, "P", 5)["level"] in ("good", "ok")
    assert size_fit("Ba", 2, "P", 5)["level"] == "misfit"


def test_variants_pass_hard_filters():
    rej = {}
    vs = generate_variants(parse_formula("Li7La3Zr2O12"), allow_tm=True, rejected=rej)
    assert rej["size"] > 0                                                   # misfits removed before prediction
    for v in vs:
        assert assign_charges(v["comp"])["_imbalance"] <= 0.05 and v["fit"]["level"] != "misfit"


def test_cif_for_known_and_new_formulas():
    real = generate("Li6PS5Cl", space_group="F-43m")
    assert real["ok"] and real["exact"] and "Real" not in real["cif"][:0]
    new = generate("Li6PS5Cl0.3Br0.7", base="Li6PS5Cl")
    assert new["ok"] and not new["exact"] and "NOT RELAXED" in new["cif"]
    st = parse_cif(new["cif"])                                               # the CIF reads back
    comp = st.composition()
    ratio = comp["Br"] / comp["P"]
    assert abs(ratio - 0.7) < 0.02 and abs(comp["Li"] / comp["P"] - 6) < 0.05
    bad = generate("LiCl5")
    assert not bad["ok"]


def test_assess_uses_structure_path(reg, bundle):
    a = assess(reg, "Li6PS5Cl0.3Br0.7", "F-43m", "Li6PS5Cl", None, bundle)
    assert a["sigma"]["has_structure"] and a["p_logreg"] is not None
    assert a["feasibility"]["key"] == "hypo_plausible" and a["existence"]["status"] == "hypothetical"
    assert "not a measured" in a["honesty"]
    u = assess(reg, "LiCl5", None, None, None, bundle)
    assert u["feasibility"]["key"] == "unlikely" and not u["feasibility"]["charge_balanced"]
    m = assess(reg, "Li7La3Zr2O12", "Ia-3d", None, None, bundle)
    assert m["feasibility"]["key"] == "measured" and m["existence"]["sigma_max"] > m["existence"]["sigma_min"]


def test_pages_show_reality_check(client):
    r = client.post("/simple", data={"action": "formula", "formula": "Li6PS5Cl0.3Br0.7"})
    assert r.status_code == 200 and b"Does it exist?" in r.data and b"Download CIF" in r.data
    j = client.get("/api/lab?formula=Li6PS5Cl0.3Br0.7&sg=F-43m&base=Li6PS5Cl").get_json()
    assert j["existence"]["status"] == "hypothetical" and j["structure"]["viewer"]["atoms"] and j["cif_url"]
    c = client.get(j["cif_url"])
    assert c.status_code == 200 and b"_cell_length_a" in c.data and b"Br" in c.data
    assert client.get("/design/cif?formula=LiCl5").status_code == 400
    assert b"Reality check" in client.get("/recipe?formula=Li6PS5Cl0.3Br0.7&batch=5").data


def test_holdout_validation_file():
    from crystalscout.ml.validation import load_validation

    v = load_validation()
    assert v and len(v["families"]) >= 5
    garnet = next(c for c in v["cases"] if c["family"] == "garnet")
    assert garnet["good_in_top_quarter"] >= 2                                   # Ta-LLZO found without seeing any garnet


@pytest.fixture(scope="module")
def client():
    from webapp.app import app

    app.config["TESTING"] = True
    c = app.test_client()
    assert c.post("/login", data={"username": "expo", "password": "test-only-expo-password"}).status_code == 302
    return c


def test_api_service_design():
    import json

    from api import service

    c = service.design_check("Li6PS5Cl0.3Br0.7", "F-43m", "Li6PS5Cl")
    assert c["existence"]["status"] == "hypothetical" and c["structure"] and "viewer" not in c["structure"]
    d = service.design_cif("Li6PS5Cl0.3Br0.7", None, "Li6PS5Cl")
    assert "_cell_length_a" in d["cif"] and not d["exact"]
    a = service.design_advise("Li6PS5Cl", "F-43m", 3)
    assert len(a["suggestions"]) == 3 and a["best"] and json.dumps(a)
    with pytest.raises(service.InputError):
        service.design_cif("NaCl")
