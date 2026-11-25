"""σ-Regressor, CrystalScout Score and industry read-outs."""
import numpy as np
import pytest

from crystalscout.ml.consensus import cs_score
from crystalscout.ml.insight import FormulaError, chem_family, parse_checked_formula, sigma_readout, tier
from crystalscout.ml.regression import crystal_system, get_regressor, parse_formula


def test_parse_formula():
    assert parse_formula("Li10Ge(PS6)2") == {"Li": 10, "Ge": 1, "P": 2, "S": 12}
    assert parse_formula("Li3(PS4)0.5Cl") == {"Li": 3, "P": 0.5, "S": 2, "Cl": 1}


def test_crystal_system():
    assert crystal_system("Fm-3m")["crystal_system"] == 6
    assert crystal_system("P2_1/c")["crystal_system"] == 1


def test_formula_validation():
    with pytest.raises(FormulaError):
        parse_checked_formula("NaCl")
    with pytest.raises(FormulaError):
        parse_checked_formula("LiQq2")
    assert parse_checked_formula("Li6PS5Cl")["S"] == 5


def test_regressor_report_and_prediction():
    reg = get_regressor(verbose=False)
    rep = reg.report
    assert rep["protocol"]["n_total"] == 562 and rep["protocol"]["n_test"] == 113
    et = rep["models"]["extra_trees"]
    assert et["cv"]["r2"] > 0.6 and et["test"]["mae"] < 1.6 and et["test"]["superionic_auc"] > 0.8
    assert 0.7 < rep["interval"]["test_coverage"] < 0.95
    fast = sigma_readout(reg, parse_checked_formula("Li6PS5Cl"), "F-43m", np.nan, None, "Li6PS5Cl")
    slow = sigma_readout(reg, parse_checked_formula("Li3PO4"), "Pmn2_1", np.nan, None, "Li3PO4")
    assert fast["log10_sigma"] > slow["log10_sigma"] + 2
    assert fast["log10_lo"] <= fast["log10_sigma"] <= fast["log10_hi"]


def test_cs_score_and_tiers():
    s = cs_score([-3, -5, -8], [0.9, np.nan, 0.1])
    assert s[0] == s.max() and np.all((s >= 0) & (s <= 1))
    assert tier(-2.5)["name"] == "Cell-ready" and tier(-3.5)["name"] == "Superionic" and tier(-9)["name"] == "Poor"


def test_chem_family():
    assert chem_family(parse_formula("Li6PS5Cl")) == "sulfide"
    assert chem_family(parse_formula("Li3InCl6")) == "halide"
    assert chem_family(parse_formula("Li7La3Zr2O12")) == "oxide"
