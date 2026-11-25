"""Formula Lab, Dopant Advisor and Synthesis Planner."""
import pytest

from crystalscout.ml.design import advise, assign_charges, generate_variants, lab_evaluate, synthesis_plan, to_formula
from crystalscout.ml.insight import FormulaError, parse_checked_formula
from crystalscout.ml.regression import get_regressor, parse_formula


@pytest.fixture(scope="module")
def client():
    from webapp.app import app

    app.config["TESTING"] = True
    c = app.test_client()
    assert c.post("/login", data={"username": "expo", "password": "test-only-expo-password"}).status_code == 302
    return c


@pytest.fixture(scope="module")
def reg():
    return get_regressor(verbose=False)


@pytest.mark.parametrize("f,expect", [("Li6PS5Cl", {"P": 5, "S": -2, "Cl": -1}), ("Li7La3Zr2O12", {"La": 3, "Zr": 4}),
                                      ("Li3InCl6", {"In": 3}), ("Li10GeP2S12", {"Ge": 4, "P": 5})])
def test_oxidation_states(f, expect):
    ch = assign_charges(parse_formula(f))
    assert ch["_imbalance"] == 0 and all(ch[k] == v for k, v in expect.items())


def test_variants_are_charge_neutral():
    comp = parse_formula("Li6PS5Cl")
    vs = generate_variants(comp)
    assert len(vs) > 20
    for v in vs:
        ch = assign_charges(v["comp"])
        assert ch is not None and ch["_imbalance"] < 1e-6, v["formula"]
    assert to_formula(parse_formula("Li6PS5Cl")) == "Li6PS5Cl"


def test_advisor_finds_known_improvements(reg):
    from crystalscout.ml.models import get_bundle

    a = advise(reg, "Li6PS5Cl", "F-43m", top=5, bundle=get_bundle())
    assert a["best"]["gain"] > 1 and "Br" in a["best"]["formula"]          # halogen-rich / mixed-halide argyrodite
    assert any(k["formula"] == "Li6PS5Cl0.5Br0.5" and k["measured"] for k in a["known"])  # known one kept apart
    assert all(s["badge"]["key"] in ("hypo_plausible", "hypo_strained") for s in a["suggestions"])
    assert all(s["structure_scored"] for s in a["suggestions"])
    b = advise(reg, "Li7La3Zr2O12", "Ia-3d", top=5, allow_tm=True, bundle=get_bundle())
    assert b["best"]["dopant"] in ("Ta", "Nb", "Hf", "Ga", "Al") and b["best"]["gain"] > 1   # Zr-site dopants
    assert any("Ta" in k["formula"] for k in b["known"])                    # Ta-LLZO is a benchmark, not a "discovery"
    assert all(not s["flags"]["toxic"] for s in a["suggestions"])


def test_recipe_balances(reg):
    p = synthesis_plan("Li6PS5Cl", 1.0)
    got = {r["precursor"]: r["moles_per_fu"] for r in p["rows"]}
    assert p["balanced"] and got == {"Li2S": 2.5, "P2S5": 0.5, "LiCl": 1.0}
    assert abs(sum(r["grams"] for r in p["rows"]) - 1.0) < 1e-3
    q = synthesis_plan("Li7La3Zr2O12", 10)
    assert q["balanced"] and {"Li2O", "La2O3", "ZrO2"} == {r["precursor"] for r in q["rows"]}


def test_lab_explains_change(reg):
    r = lab_evaluate(reg, "Li6PO5Cl", "F-43m", base="Li6PS5Cl")
    assert r["delta_log"] < -1 and r["drivers"]
    with pytest.raises(FormulaError):
        parse_checked_formula("NaCl")


def test_design_pages(client):
    assert client.get("/lab").status_code == 200
    j = client.get("/api/lab?formula=Li6PS5Br&sg=F-43m&base=Li6PS5Cl").get_json()
    assert j["base"]["formula"] == "Li6PS5Cl" and j["gain"] > 0 and dict(j["elements"])["Br"] == 1 and "Cl" in j["swaps"]["Br"]
    assert client.get("/api/lab?formula=XyZ").status_code == 400
    r = client.get("/advisor?formula=Li6PS5Cl&sg=F-43m")
    assert r.status_code == 200 and b"Best new candidate" in r.data and b"Already known in this doping space" in r.data
    assert client.get("/advisor").status_code == 200
    assert client.get("/advisor?formula=NaCl").status_code == 400
    r = client.get("/recipe?formula=Li6PS5Cl0.5Br0.5&batch=5")
    assert r.status_code == 200 and b"Element balance exact" in r.data
    a = client.get("/api/advise?formula=Li6PS5Cl&sg=F-43m&top=3").get_json()
    assert len(a["suggestions"]) == 3
    assert client.get("/api/recipe?formula=Li3InCl6&batch=2").get_json()["balanced"]
