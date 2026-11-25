"""Stage A: rule-based property filters (Sendek et al. 2017 prerequisites).

Each material is tested against every rule and *all* failing rules are
recorded, so the rejection log explains exactly why a material was dropped.

Default criteria reproduce the base paper's population, minus the
oxidation-potential filter (> 4 V), which needs grand-potential phase-diagram
calculations and is documented as future work (docs/METHODS.md).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from ..core.chemistry import TRANSITION_METALS


@dataclass
class StageACriteria:
    band_gap_min: float = 1.0          # eV, strictly greater than (paper: > 1 eV)
    ehull_max: float = 0.0             # eV/atom, on the convex hull
    ehull_tol: float = 1e-6            # numerical tolerance for "= 0"
    exclude_transition_metals: bool = True
    require_li: bool = True
    exclude_elements: Tuple[str, ...] = ()
    max_nsites: int = 0                # 0 = no limit

    def as_dict(self) -> Dict:
        return asdict(self)

    def describe(self) -> List[str]:
        out = ["contains Li"] if self.require_li else []
        out.append(f"band gap > {self.band_gap_min:g} eV")
        out.append("on the convex hull (E_hull = 0)" if self.ehull_max <= 0 else f"E_hull ≤ {self.ehull_max:g} eV/atom")
        if self.exclude_transition_metals:
            out.append("no transition metals")
        if self.exclude_elements:
            out.append("excludes " + ", ".join(self.exclude_elements))
        if self.max_nsites:
            out.append(f"≤ {self.max_nsites} sites per cell")
        return out


RULES = ["li", "band_gap", "ehull", "tm", "excluded", "nsites"]
RULE_LABEL = {
    "li": "no Li",
    "band_gap": "band gap too small",
    "ehull": "above convex hull",
    "tm": "contains transition metal",
    "excluded": "contains excluded element",
    "nsites": "cell too large",
}


def apply_stage_a(df: pd.DataFrame, c: StageACriteria) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Return (passed, rejected, stats). ``df`` needs columns material_id,
    formula, band_gap, energy_above_hull, elements (space-separated)."""
    els = df["elements"].fillna("").str.split()
    fails = {
        "li": ~els.map(lambda e: "Li" in e) if c.require_li else pd.Series(False, index=df.index),
        "band_gap": ~(df["band_gap"].astype(float) > c.band_gap_min),
        "ehull": ~(df["energy_above_hull"].astype(float) <= c.ehull_max + c.ehull_tol),
        "tm": els.map(lambda e: bool(set(e) & TRANSITION_METALS)) if c.exclude_transition_metals else pd.Series(False, index=df.index),
        "excluded": els.map(lambda e: bool(set(e) & set(c.exclude_elements))),
        "nsites": (df["nsites"].astype(float) > c.max_nsites) if c.max_nsites else pd.Series(False, index=df.index),
    }
    fail_df = pd.DataFrame(fails)
    reasons = fail_df.apply(lambda r: "; ".join(RULE_LABEL[k] for k in RULES if r[k]), axis=1)
    passed_mask = ~fail_df.any(axis=1)
    out = df.copy()
    out["rejection_reasons"] = reasons
    # sequential funnel (paper order) for the dashboard
    funnel = [("Li-containing structures", int(len(df)))]
    m = pd.Series(True, index=df.index)
    for key, label in (("ehull", "on convex hull" if c.ehull_max <= 0 else f"E_hull ≤ {c.ehull_max:g}"),
                       ("band_gap", f"band gap > {c.band_gap_min:g} eV"),
                       ("tm", "no transition metals"), ("excluded", "excluded elements"), ("nsites", "cell size")):
        if not fail_df[key].any() and key in ("excluded", "nsites"):
            continue
        m &= ~fail_df[key]
        funnel.append((label, int(m.sum())))
    stats = {
        "n_input": int(len(df)),
        "n_passed": int(passed_mask.sum()),
        "n_rejected": int((~passed_mask).sum()),
        "fail_counts": {RULE_LABEL[k]: int(v) for k, v in fail_df.sum().items() if v},
        "funnel": funnel,
        "criteria": c.as_dict(),
        "criteria_text": c.describe(),
    }
    return out[passed_mask].copy(), out[~passed_mask].copy(), stats
