"""Held-out validation of the design tools (no family the model is tested on is seen in training).

Why this exists: the Dopant Advisor's headline picks (Ta-doped LLZO, mixed-halide argyrodites) are
in OBELiX, so a model trained on all of OBELiX has already seen them. That shows the ranking is
sensible, not that it can discover anything. Here we remove a whole crystal family from training
and ask two questions:

1. Family hold-out: trained without any garnet (or argyrodite, NASICON, ...), how well does the
   sigma-Regressor rank the measured members of that unseen family? (Spearman rho, MAE, and how
   many of its top picks are really superionic compared with picking at random.)
2. Advisor case study: trained without the family, the advisor dopes the parent compound (e.g.
   Li7La3Zr2O12). Every generated variant that was actually made and measured (OBELiX) is located
   in the advisor's ranking: are the good ones near the top?

Run:  python -m crystalscout.cli validate-advisor   (writes data/models/advisor_validation.json)
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import MODELS_DIR
from .regression import LOG_THRESHOLD, _models, get_regressor, obelix_regression_table

VALIDATION_FILE = MODELS_DIR / "advisor_validation.json"
FAMILIES = ("garnet", "argyrodites", "NASICON", "LGPS", "perovskites", "thio-LISICON", "LISICON")
CASES = [
    {"family": "garnet", "base": "Li7La3Zr2O12", "space_group": "Ia-3d", "allow_tm": True,
     "story": "Garnet LLZO: Ta, Nb, Ga and Al doping are the best-known routes to cubic, fast-conducting LLZO."},
    {"family": "argyrodites", "base": "Li6PS5Cl", "space_group": "F-43m", "allow_tm": False,
     "story": "Argyrodite Li6PS5Cl: halide mixing (Br) and Li-deficient halogen-rich variants are known to help."},
]


def _heldout_regressor(reg, table: pd.DataFrame, family: str):
    tr = table[table["family"] != family]
    pipe = _models()["extra_trees"].fit(tr[reg.feature_names], tr["log10_sigma"].to_numpy())
    return replace(reg, model=pipe, eval_model=pipe, train_groups=sorted(set(tr["group"]))), tr


def _spearman(a, b) -> Optional[float]:
    if len(a) < 3:
        return None
    v = pd.Series(list(a)).corr(pd.Series(list(b)), method="spearman")
    return None if v != v else float(v)


def family_holdout(reg=None, table: Optional[pd.DataFrame] = None, families=FAMILIES) -> List[Dict]:
    reg = reg or get_regressor(verbose=False)
    table = obelix_regression_table() if table is None else table
    out = []
    for fam in families:
        te = table[table["family"] == fam]
        if len(te) < 10:
            continue
        hreg, _ = _heldout_regressor(reg, table, fam)
        p = hreg.predict_frame(te[reg.feature_names])["log10_sigma_pred"].to_numpy()
        y = te["log10_sigma"].to_numpy()
        k = max(1, int(round(len(te) * 0.25)))
        top = np.argsort(-p)[:k]
        base_rate = float((y >= LOG_THRESHOLD).mean())
        out.append({"family": fam, "n": int(len(te)), "spearman": _spearman(p, y),
                    "mae": float(np.mean(np.abs(p - y))), "within_1_log": float(np.mean(np.abs(p - y) <= 1.0)),
                    "top_quarter_superionic": float((y[top] >= LOG_THRESHOLD).mean()), "base_rate_superionic": base_rate,
                    "top_quarter_k": k})
    return out


def advisor_case(case: Dict, reg=None, table: Optional[pd.DataFrame] = None, bundle=None) -> Dict:
    """Advisor run with the family held out; where do the variants that were really made land?"""
    from .design import assign_charges, generate_variants
    from .insight import parse_checked_formula
    from .structgen import assess, lookup, template_by_id

    reg = reg or get_regressor(verbose=False)
    table = obelix_regression_table() if table is None else table
    hreg, tr = _heldout_regressor(reg, table, case["family"])
    comp = parse_checked_formula(case["base"])
    base_a = assess(hreg, case["base"], case["space_group"], None, None, bundle)
    host = template_by_id(base_a["structure"]["host_id"]) if base_a["structure"] and base_a["structure"].get("ok") else None
    rejected: Dict[str, int] = {}
    variants = generate_variants(comp, allow_tm=case["allow_tm"], avoid_toxic=True, rejected=rejected)
    rows = []
    for v in variants:
        a = assess(hreg, v["formula"], case["space_group"], case["base"], None, bundle, host_template=host)
        if a["feasibility"]["key"] == "unlikely":
            continue
        ex = lookup(v["comp"])
        meas = [o["sigma"] for o in ex["obelix"] if o.get("sigma")]
        rows.append({"formula": v["formula"], "dopant": v["dopant"], "host": v["host"], "fraction": v["fraction"],
                     "pred": a["sigma"]["log10_sigma"], "measured": float(np.log10(max(meas))) if meas else None})
    rows.sort(key=lambda r: -r["pred"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    made = [r for r in rows if r["measured"] is not None]
    n = len(rows)
    good = [r for r in made if r["measured"] >= LOG_THRESHOLD]
    return {
        **{k: case[k] for k in ("family", "base", "space_group", "story")},
        "n_train": int(len(tr)), "n_heldout": int((table["family"] == case["family"]).sum()),
        "n_variants": n, "n_made": len(made), "base_pred": base_a["sigma"]["log10_sigma"],
        "spearman_made": _spearman([r["pred"] for r in made], [r["measured"] for r in made]),
        "made": [{k: r[k] for k in ("formula", "dopant", "fraction", "rank", "pred", "measured")} for r in made],
        "good_in_top_quarter": int(sum(1 for r in good if r["rank"] <= max(1, n // 4))), "n_good": len(good),
        "median_rank_good_pct": float(np.median([r["rank"] / n for r in good])) if good and n else None,
        "top5": [{k: r[k] for k in ("formula", "dopant", "fraction", "rank", "pred", "measured")} for r in rows[:5]],
    }


def run_all(save: bool = True, verbose: bool = True) -> Dict:
    from .models import get_bundle

    t0 = time.time()
    reg = get_regressor(verbose=False)
    table = obelix_regression_table()
    bundle = get_bundle()
    fam = family_holdout(reg, table)
    if verbose:
        for f in fam:
            print(f"  {f['family']:13s} n={f['n']:3d}  rho={f['spearman']:.2f}  MAE={f['mae']:.2f}  "
                  f"top-quarter superionic {f['top_quarter_superionic']:.0%} vs {f['base_rate_superionic']:.0%} at random")
    cases = [advisor_case(c, reg, table, bundle) for c in CASES]
    if verbose:
        for c in cases:
            print(f"  {c['base']}: {c['n_variants']} variants, {c['n_made']} were made; rho={c['spearman_made']}; "
                  f"{c['good_in_top_quarter']}/{c['n_good']} superionic ones in the top quarter")
    out = {"families": fam, "cases": cases, "seconds": round(time.time() - t0, 1),
           "protocol": "Extra Trees (same settings as the deployed model) retrained with every OBELiX entry of the family removed; "
                       "tested only on that family. The advisor case study scores every generated variant on the base's crystal."}
    if save:
        VALIDATION_FILE.write_text(json.dumps(out, indent=1))
    return out


def load_validation() -> Optional[Dict]:
    try:
        return json.loads(VALIDATION_FILE.read_text())
    except (OSError, ValueError):
        return None
