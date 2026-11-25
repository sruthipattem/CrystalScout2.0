"""Data, Stage A, models, screening and benchmark tests (use bundled data)."""
import numpy as np
import pandas as pd
import pytest

from crystalscout.core.descriptors import FEATURES
from crystalscout.data.materials_project import MaterialsProjectClient, MPError, load_snapshot
from crystalscout.data.obelix import build_features, load_obelix, training_table
from crystalscout.data.stage_a import StageACriteria, apply_stage_a
from crystalscout.ml.benchmark import run_benchmark
from crystalscout.ml.explain import applicability, logreg_contributions
from crystalscout.ml.models import get_bundle as load_bundle
from crystalscout.screening import norm_formula, run_screening


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def test_obelix_labels_and_split():
    df = load_obelix()
    assert len(df) == 599 and df["has_cif"].sum() == 321
    assert set(df["split"]) == {"train", "test"}
    assert df.loc[df["sigma"] >= 1e-4, "label"].eq(1).all()
    assert df["sigma_upper_bound"].sum() > 0


def test_obelix_split_has_no_composition_leakage():
    t = training_table()
    train = set(t[t.split == "train"].formula)
    assert not t[t.split == "test"].formula.isin(train).any()


def test_obelix_features_exclusions():
    f = build_features()
    ok = f[f.excluded_reason == ""]
    assert len(ok) >= 300
    assert ok[FEATURES].notna().all().all()
    assert (f.excluded_reason == "CIF contains no Li sites").sum() == 7


def test_stage_a_default_funnel():
    snap = load_snapshot()
    passed, rejected, stats = apply_stage_a(snap, StageACriteria())
    assert stats["n_input"] == len(snap) == 19806
    assert stats["n_passed"] == 505
    assert (passed["band_gap"] > 1).all() and (passed["energy_above_hull"] <= 1e-6).all()
    assert rejected["rejection_reasons"].str.len().gt(0).all()
    assert [n for _, n in stats["funnel"]] == sorted([n for _, n in stats["funnel"]], reverse=True)


def test_stage_a_records_every_failed_rule():
    df = pd.DataFrame([{"material_id": "x", "formula": "LiFeO2", "band_gap": 0.2, "energy_above_hull": 0.3,
                        "elements": "Fe Li O", "nsites": 4}])
    _, rej, _ = apply_stage_a(df, StageACriteria())
    r = rej.iloc[0]["rejection_reasons"]
    assert "band gap" in r and "hull" in r and "transition metal" in r


def test_models_reproduce_report(bundle):
    t = training_table()
    te = t[t.split == "test"]
    p = bundle.proba(te[FEATURES].to_numpy(float))
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(te.label, p) == pytest.approx(bundle.report["models"]["logreg"]["test"]["roc_auc"], abs=1e-6)
    for m in ("logreg", "random_forest", "gboost", "sendek2019"):
        assert 0.5 < bundle.report["models"][m]["cv_all_metrics"]["roc_auc"] <= 1.0


def test_retrained_signs_mostly_agree_with_published(bundle):
    eq = bundle.report["logreg_equation"]
    same = sum((eq["coef_raw"][k] > 0) == (eq["sendek2019"][k] > 0) for k in FEATURES)
    assert same >= 4


def test_linear_explanation_is_exact(bundle):
    x = np.array([1.0, 0.1, 8.0, 2.5, 2.4])
    e = logreg_contributions(bundle, x)
    assert e["probability"] == pytest.approx(bundle.proba(x[None])[0], abs=1e-9)
    assert e["baseline_logit"] + sum(t["contribution"] for t in e["terms"]) == pytest.approx(e["logit"])


def test_applicability_flags_outliers(bundle):
    assert not applicability(bundle, np.array([30.0, 0.1, 8.0, 2.5, 2.4]))["in_domain"]


def test_screening_snapshot(bundle):
    run = run_screening(bundle, save=False)
    s = run.shortlist
    assert len(s) == 505 and s["p_logreg"].notna().all() and s["log10_sigma_pred"].notna().all()
    assert s["cs_score"].is_monotonic_decreasing and s["cs_score"].between(0, 1).all()
    assert (s["feature_error"] == "").all()
    assert run.summary()["n_stage_a"] == 505


def test_dftmd_benchmark(bundle):
    b = run_benchmark(bundle)
    assert b["n"] == 38 and b["n_pos"] == 10
    assert b["models"]["logreg"]["roc_auc"] > 0.7


def test_formula_normalisation():
    assert norm_formula("Li3InCl6") == norm_formula("Cl6InLi3")
    assert norm_formula("Li3(PO4)") == norm_formula("Li6P2O8")


def test_live_client_requires_key(monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    monkeypatch.delenv("PMG_MAPI_KEY", raising=False)
    with pytest.raises(MPError):
        MaterialsProjectClient()


def test_live_client_parses_pages(monkeypatch):
    from crystalscout.data import materials_project as mp
    from tests.test_core import rocksalt

    pages = [{"data": [{"material_id": "mp-1", "formula_pretty": "LiCl", "band_gap": 6.0, "energy_above_hull": 0.0,
                        "formation_energy_per_atom": -1.9, "symmetry": {"symbol": "Fm-3m"}, "nsites": 8,
                        "elements": ["Cl", "Li"], "structure": rocksalt().as_dict()}], "meta": {"total_doc": 1}}]
    monkeypatch.setattr(mp.MaterialsProjectClient, "_get", lambda self, params: pages[0])
    monkeypatch.setitem(__import__("sys").modules, "mp_api", None)  # force the REST path
    recs = mp.MaterialsProjectClient("k").stage_a_query()
    assert recs[0]["material_id"] == "mp-1" and recs[0]["space_group"] == "Fm-3m"
    assert recs[0]["structure"]["sites"]
