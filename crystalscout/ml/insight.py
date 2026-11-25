"""Decision-ready read-outs for one material: predicted σ, tier, CrystalScout Score.

Used by the web app and the REST API so both say exactly the same thing.

Conductivity tiers (room temperature, S/cm), chosen to match how cell
engineers talk about solid electrolytes:

    ≥ 1e-2   Exceptional    on par with liquid electrolytes (LGPS-class)
    ≥ 1e-3   Cell-ready     ≥ 1 mS/cm, the usual target for practical cells
    ≥ 1e-4   Superionic     the screening threshold used to train the models
    ≥ 1e-6   Moderate       useful only in thin films or composites
    <  1e-6  Poor           not a practical Li-ion conductor
"""
from __future__ import annotations

import math
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..core.descriptors import FEATURES
from ..core.elements import Z_OF
from .consensus import reference_score

TIERS = [
    (-2.0, "Exceptional", "on par with liquid electrolytes", "violet"),
    (-3.0, "Cell-ready", "≥ 1 mS/cm, the target for practical solid-state cells", "good"),
    (-4.0, "Superionic", "above the 10⁻⁴ S/cm screening threshold", "teal"),
    (-6.0, "Moderate", "useful mainly in thin films or composites", "warn"),
    (-99, "Poor", "not a practical Li-ion conductor", "bad"),
]


class FormulaError(ValueError):
    pass


def tier(log_sigma: Optional[float]) -> Dict:
    if log_sigma is None or log_sigma != log_sigma:
        return {"name": "Unknown", "note": "", "tone": "muted"}
    for cut, name, note, tone in TIERS:
        if log_sigma >= cut:
            return {"name": name, "note": note, "tone": tone}
    return {"name": "Poor", "note": TIERS[-1][2], "tone": "bad"}


def fmt_sigma(log_sigma: float) -> str:
    """'3.2 × 10⁻⁴ S/cm' style text for a log10 conductivity."""
    if log_sigma is None or log_sigma != log_sigma:
        return "—"
    v = 10 ** log_sigma
    m, e = f"{v:.1e}".split("e")
    sup = str(int(e)).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m} × 10{sup} S/cm"


def parse_checked_formula(formula: str) -> Dict[str, float]:
    from .regression import parse_formula

    f = (formula or "").strip()
    if not f:
        raise FormulaError("Enter a chemical formula, for example Li6PS5Cl.")
    if len(f) > 80 or not re.fullmatch(r"[A-Za-z0-9().\[\]{}\s]+", f):
        raise FormulaError("Use element symbols, numbers and brackets only, for example Li3InCl6.")
    comp = parse_formula(f)
    unknown = [e for e in comp if e not in Z_OF]
    if unknown:
        raise FormulaError("Unknown element symbol: " + ", ".join(unknown) + ". Symbols are case-sensitive (Cl, not CL).")
    if "Li" not in comp:
        raise FormulaError("CrystalScout screens lithium conductors: the formula must contain Li.")
    if len(comp) < 2:
        raise FormulaError("Add at least one other element besides Li.")
    return comp


def sigma_readout(reg, comp: Dict[str, float], space_group=None, volume_per_atom=np.nan,
                  descriptors: Optional[Dict[str, float]] = None, formula: str = "",
                  p_logreg: Optional[float] = None, ref: Optional[pd.DataFrame] = None) -> Dict:
    """Everything the UI shows about predicted conductivity for one material."""
    d = None
    if descriptors and all(descriptors.get(k) is not None and descriptors.get(k) == descriptors.get(k) for k in FEATURES):
        d = {k: float(descriptors[k]) for k in FEATURES}
    r = reg.predict_one(comp, space_group, volume_per_atom, d, formula)
    ls, lo, hi = float(r["log10_sigma_pred"]), float(r["log10_sigma_lo"]), float(r["log10_sigma_hi"])
    out = {
        "log10_sigma": ls, "log10_lo": lo, "log10_hi": hi, "spread": float(r["log10_sigma_spread"]),
        "sigma_S_cm": 10 ** ls, "sigma_mS_cm": float(r["sigma_pred_mScm"]),
        "sigma_text": fmt_sigma(ls), "lo_text": fmt_sigma(lo), "hi_text": fmt_sigma(hi),
        "p_trees_superionic": float(r["p_superionic_trees"]), "tier": tier(ls),
        "known_composition": bool(r["known_composition"]), "out_of_range": list(r["out_of_range"]),
        "has_structure": bool(d), "interval": int(round(reg.report.get("interval", {}).get("nominal", 0.8) * 100)),
    }
    # where the value sits on a −10…−1 bar (for the gauge)
    out["gauge_pct"] = max(0.0, min(100.0, (ls + 10) / 9 * 100))
    out["gauge_lo"] = max(0.0, min(100.0, (lo + 10) / 9 * 100))
    out["gauge_hi"] = max(0.0, min(100.0, (hi + 10) / 9 * 100))
    if ref is not None and len(ref):
        rs = reference_score(ls, p_logreg, ref)
        out["cs_score"] = rs["score"]
        out["cs_beats"] = rs["beats"]
        out["cs_n_ref"] = rs["n_ref"]
        rs_sig = ref["log10_sigma_pred"].dropna().to_numpy(float)
        out["sigma_pct"] = float((rs_sig < ls).mean()) if len(rs_sig) else None
    return out


def verdict(p_logreg: Optional[float], log_sigma: Optional[float], sigma_pct: Optional[float] = None) -> Dict:
    """Plain-language recommendation that combines both model families.

    The σ side counts as positive when the prediction clears 10⁻⁴ S/cm or sits in the top quarter of
    the standard screen (predictions for idealised MP crystals are compressed towards the mean, so the
    relative position is the more reliable signal)."""
    s_ok = log_sigma is not None and log_sigma == log_sigma and (
        log_sigma >= -4 or (sigma_pct is not None and sigma_pct >= 0.75))
    p_ok = p_logreg is not None and p_logreg == p_logreg and p_logreg >= 0.5
    if p_logreg is None or p_logreg != p_logreg:
        if s_ok:
            return {"label": "Promising from composition", "tone": "teal",
                    "action": "Get or compute a crystal structure to confirm with the structure model."}
        return {"label": "Low priority", "tone": "warn",
                "action": "Composition alone suggests modest conductivity."}
    if s_ok and p_ok:
        return {"label": "Strong candidate", "tone": "good",
                "action": "Both model families agree. Prioritise for DFT-MD or synthesis."}
    if s_ok or p_ok:
        return {"label": "Worth a closer look", "tone": "teal",
                "action": "The models disagree; check the drivers below before committing lab time."}
    return {"label": "Low priority", "tone": "warn",
            "action": "Neither model expects fast Li⁺ transport."}


def safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------ industry flags
SUPPLY_RISK = {"Ge", "Ga", "In", "Sc", "Te", "Hf", "Ta", "Nb", "Y", "La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb",
               "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Ag", "Au", "Pt", "Pd", "Rh", "Ir", "Ru", "Re", "Os", "Cs", "Rb"}
TOXIC = {"Pb", "Cd", "Hg", "Tl", "As", "Be", "Se", "Sb", "Ba"}
FAMILIES = [("sulfide", {"S"}), ("selenide/telluride", {"Se", "Te"}), ("halide", {"F", "Cl", "Br", "I"}),
            ("nitride", {"N"}), ("hydride", {"H"}), ("oxide", {"O"}), ("phosphide", {"P"}), ("boride", {"B"})]
FAMILY_LABELS = {"sulfide": "Sulfide", "selenide/telluride": "Selenide / telluride", "halide": "Halide",
                 "nitride": "Nitride", "hydride": "Hydride", "oxide": "Oxide", "phosphide": "Phosphide",
                 "boride": "Boride", "other": "Other"}


def chem_family(comp: Dict[str, float]) -> str:
    """Anion family: oxygen-containing thio/halo compounds go to the softer anion (e.g. Li6PS5Cl -> sulfide)."""
    els = set(comp)
    if "S" in els and not ({"O"} & els and comp.get("O", 0) > comp.get("S", 0)):
        return "sulfide"
    for name, anions in FAMILIES:
        if els & anions:
            if name == "halide" and "O" in els and sum(comp.get(a, 0) for a in anions) < comp.get("O", 0):
                return "oxide"
            return name
    return "other"


def industry_flags(comp: Dict[str, float]) -> Dict:
    risk = sorted(set(comp) & SUPPLY_RISK)
    tox = sorted(set(comp) & TOXIC)
    return {"supply_risk": risk, "toxic": tox, "family": chem_family(comp),
            "family_label": FAMILY_LABELS[chem_family(comp)],
            "moisture_sensitive": chem_family(comp) in ("sulfide", "selenide/telluride", "nitride", "hydride", "phosphide")}


def add_industry_columns(df: pd.DataFrame) -> pd.DataFrame:
    from .regression import parse_formula

    df = df.copy()
    fl = [industry_flags(parse_formula(f)) for f in df["formula"].astype(str)]
    df["family"] = [f["family"] for f in fl]
    df["supply_risk"] = [" ".join(f["supply_risk"]) for f in fl]
    df["toxic"] = [" ".join(f["toxic"]) for f in fl]
    df["moisture_sensitive"] = [f["moisture_sensitive"] for f in fl]
    return df
