"""Framework-free API logic.

Every FastAPI route in ``api/main.py`` is a thin wrapper around one function
here, so the same code can be called from Python, tested without a web
server, and reused by other front ends. All return values are plain
JSON-safe Python (dict / list / str / float / int / bool / None).
"""
from __future__ import annotations

import json
import math
import threading
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from crystalscout.core.cif import CIFError
from crystalscout.core.descriptors import FEATURE_INFO, FEATURES, DescriptorError
from crystalscout.data.stage_a import StageACriteria
from crystalscout.featurize import featurize_cif_text
from crystalscout.ml.explain import applicability, explain_sentence, logreg_contributions
from crystalscout.ml.models import MODEL_LABELS, PRIMARY_MODEL, get_bundle
from crystalscout.ml.insight import FormulaError, industry_flags, parse_checked_formula, sigma_readout, verdict
from crystalscout.ml.regression import get_regressor, structure_volume_per_atom
from crystalscout.screening import (DEFAULT_RANK, RANK_LABELS, list_runs, load_run, obelix_lookup, run_screening,
                                    save_run)

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
SUPERIONIC_CUTOFF = 0.5  # probability at which a material is called "promising"

SHORTLIST_COLUMNS = ["rank", "material_id", "formula", "band_gap", "energy_above_hull", "space_group", "cs_score",
                     "log10_sigma_pred", "log10_sigma_lo", "log10_sigma_hi", "sigma_pred_mScm", "p_logreg", "p_random_forest",
                     "p_gboost", "p_sendek2019", "consensus", "n_models_positive", "why", "top_driver",
                     "in_domain", "domain_note", "obelix_sigma", "obelix_id", *FEATURES]


class InputError(ValueError):
    """Bad input from the client (becomes HTTP 400)."""


class NotFound(LookupError):
    """Requested item does not exist (becomes HTTP 404)."""


# ----------------------------------------------------------------- model cache
_lock = threading.Lock()
_bundle = None


def bundle():
    """Load the trained models once and reuse them for every request."""
    global _bundle
    with _lock:
        if _bundle is None:
            _bundle = get_bundle()
    return _bundle


def _clean(obj):
    """Turn numpy / pandas values into JSON-safe Python (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if math.isnan(f) or math.isinf(f) else f
    return obj


def _verdict(p: float) -> str:
    if p >= SUPERIONIC_CUTOFF:
        return "promising"
    if p >= 0.3:
        return "borderline"
    return "unlikely"


# ----------------------------------------------------------------- endpoints
_ref = None


def reference():
    """The standard screen, used to turn one material's predictions into a CrystalScout Score."""
    global _ref
    with _lock:
        if _ref is None:
            from crystalscout.data.stage_a import StageACriteria

            for r in list_runs():
                if r["source"] == "snapshot" and r["criteria_text"] == StageACriteria().describe() \
                        and r.get("rank_model") == DEFAULT_RANK:
                    run = load_run(r["run_id"])
                    if run is not None:
                        _ref = run.shortlist
                        break
    if _ref is None:
        _ref = run_screening(bundle(), save=True).shortlist
    return _ref


def _sigma_json(rd: Dict) -> Dict:
    return {"log10_sigma": rd["log10_sigma"], "sigma_S_cm": rd["sigma_S_cm"], "sigma_mS_cm": rd["sigma_mS_cm"],
            "interval_80_S_cm": [10 ** rd["log10_lo"], 10 ** rd["log10_hi"]], "tier": rd["tier"]["name"],
            "trees_voting_superionic": rd["p_trees_superionic"], "crystalscout_score": rd.get("cs_score"),
            "beats_share_of_standard_screen": rd.get("cs_beats"), "used_structure": rd["has_structure"],
            "known_composition": rd["known_composition"], "out_of_range": rd["out_of_range"]}


def health() -> Dict:
    b = bundle()
    return {"status": "ok", "ranking_model": DEFAULT_RANK, "rank_options": list(RANK_LABELS),
            "classifiers": list(MODEL_LABELS), "regressor": "extra_trees", "features": FEATURES}


def models_info() -> Dict:
    """Labels and held-out test metrics for every model."""
    rep = bundle().report
    keep = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "enrichment"]
    out = {}
    for key, label in MODEL_LABELS.items():
        test = rep["models"].get(key, {}).get("test", {})
        out[key] = {"label": label, "test": {k: test.get(k) for k in keep}}
    reg = get_regressor(verbose=False).report
    regs = {k: {"label": m["label"], "cv": m["cv"], "test": m["test"]} for k, m in reg["models"].items()}
    return _clean({"ranking_model": DEFAULT_RANK, "protocol": rep.get("protocol", {}), "models": out,
                   "sigma_regressor": {"protocol": reg["protocol"], "models": regs, "interval": reg["interval"]}})


def _score(x: np.ndarray) -> Dict:
    b = bundle()
    probs = {m: float(b.models[m].predict_proba(x[None, :])[0, 1]) for m in MODEL_LABELS}
    expl = logreg_contributions(b, x)
    dom = applicability(b, x)
    p = probs[b.primary]
    return {
        "probability": p,
        "verdict": _verdict(p),
        "superionic": p >= SUPERIONIC_CUTOFF,
        "probabilities": probs,
        "why": explain_sentence(expl),
        "contributions": [{"feature": t["feature"], "name": t["name"], "value": t["value"],
                           "contribution": t["contribution"], "direction": t["direction"]} for t in expl["terms"]],
        "applicability": dom,
    }


def predict_descriptors(values: Dict[str, float]) -> Dict:
    """Score a material from its 5 descriptors."""
    missing = [k for k in FEATURES if k not in values]
    if missing:
        raise InputError("missing descriptors: " + ", ".join(missing))
    try:
        x = np.array([float(values[k]) for k in FEATURES])
    except (TypeError, ValueError):
        raise InputError("descriptors must be numbers")
    if not np.isfinite(x).all():
        raise InputError("descriptors must be finite numbers")
    return _clean({"descriptors": dict(zip(FEATURES, x.tolist())), **_score(x)})


def predict_formula(formula: str, space_group: Optional[str] = None) -> Dict:
    """Composition-only conductivity estimate (no crystal structure needed)."""
    try:
        comp = parse_checked_formula(formula)
    except FormulaError as exc:
        raise InputError(str(exc))
    rd = sigma_readout(get_regressor(verbose=False), comp, space_group or None, np.nan, None, formula, None, reference())
    look = obelix_lookup(formula)
    return _clean({"formula": formula, "space_group": space_group, "sigma": _sigma_json(rd),
                   "verdict": verdict(None, rd["log10_sigma"], rd.get("sigma_pct")), "flags": industry_flags(comp),
                   "obelix_measured_sigma": look["sigma"] if look else None})


def predict_cif(text: str, name: str = "upload.cif") -> Dict:
    """Read a CIF, compute its 5 descriptors and score it."""
    if not text or not text.strip():
        raise InputError("the CIF file is empty")
    if len(text.encode("utf-8", errors="ignore")) > MAX_UPLOAD_BYTES:
        raise InputError("the CIF file is larger than 8 MB")
    try:
        st, feats = featurize_cif_text(text, name=name)
    except (CIFError, DescriptorError, ValueError) as exc:
        raise InputError(f"could not read this crystal structure: {exc}")
    x = np.array([feats[k] for k in FEATURES], dtype=float)
    formula = st.formula()
    look = obelix_lookup(formula)
    scored = _score(x)
    rd = sigma_readout(get_regressor(verbose=False), st.composition(), st.meta.get("space_group"),
                       structure_volume_per_atom(st), {k: feats[k] for k in FEATURES}, formula, scored["probability"],
                       reference())
    return _clean({
        "name": name,
        "formula": formula,
        "descriptors": {k: feats[k] for k in FEATURES},
        "structure": {"n_sites": feats.get("n_sites"), "n_li_sites": feats.get("n_li_sites"),
                      "anions": feats.get("anions"), "disordered": feats.get("disordered"),
                      "warnings": feats.get("descriptor_warnings") or ""},
        **scored,
        "sigma": _sigma_json(rd),
        "recommendation": verdict(scored["probability"], rd["log10_sigma"], rd.get("sigma_pct")),
        "flags": industry_flags(st.composition()),
        "obelix_measured_sigma": look["sigma"] if look else None,
    })


def _rows(df: pd.DataFrame, top: int) -> List[Dict]:
    cols = [c for c in SHORTLIST_COLUMNS if c in df.columns]
    return json.loads(df.head(top)[cols].to_json(orient="records"))


def screen(band_gap_min: float = 1.0, ehull_max: float = 0.0, exclude_transition_metals: bool = True,
           exclude_elements: Optional[List[str]] = None, max_nsites: int = 0, rank_model: str = DEFAULT_RANK,
           top: int = 20, source: str = "snapshot", api_key: Optional[str] = None, run_by: str = "") -> Dict:
    """Run Stage A + scoring and return the ranked shortlist."""
    if rank_model not in RANK_LABELS:
        raise InputError("rank_model must be one of: " + ", ".join(RANK_LABELS))
    if source not in ("snapshot", "live"):
        raise InputError("source must be 'snapshot' or 'live'")
    if not (0 <= band_gap_min <= 10):
        raise InputError("band_gap_min must be between 0 and 10 eV")
    if not (0 <= ehull_max <= 1):
        raise InputError("ehull_max must be between 0 and 1 eV/atom")
    top = max(1, min(int(top), 500))
    elements = tuple(sorted({e.strip().capitalize() for e in (exclude_elements or []) if e.strip()}))
    c = StageACriteria(band_gap_min=band_gap_min, ehull_max=ehull_max,
                       exclude_transition_metals=exclude_transition_metals, exclude_elements=elements,
                       max_nsites=max(0, int(max_nsites)))
    try:
        run = run_screening(bundle(), c, source=source, api_key=api_key, rank_model=rank_model, save=False)
    except RuntimeError as exc:
        raise InputError(str(exc))
    run.stats["run_by"] = run_by
    save_run(run)
    return _clean({"summary": run.summary(), "results": _rows(run.shortlist, top)})


def runs() -> List[Dict]:
    return _clean(list_runs())


def run_detail(run_id: str, top: int = 20) -> Dict:
    run = load_run(run_id)
    if run is None:
        raise NotFound(f"no run with id {run_id}")
    top = max(1, min(int(top), 500))
    return _clean({"summary": run.summary(), "results": _rows(run.shortlist, top)})


def feature_info() -> Dict:
    return {k: dict(FEATURE_INFO[k]) for k in FEATURES}


# ---------------------------------------------------------------- design tools
def _reality_json(a: Dict) -> Dict:
    from crystalscout.ml.structgen import structure_summary

    st = structure_summary(a["structure"])
    if st:
        st.pop("viewer", None)
    return {"existence": a["existence"], "feasibility": a["feasibility"], "confidence": a["confidence"],
            "structure": st, "honesty": a["honesty"]}


def design_check(formula: str, space_group: Optional[str] = None, base: Optional[str] = None) -> Dict:
    """Reality check for any formula: exists?, feasible?, crystal structure, structure-path prediction."""
    from crystalscout.ml.structgen import assess

    try:
        a = assess(get_regressor(verbose=False), formula, space_group or None, base or None, reference(), bundle())
    except FormulaError as exc:
        raise InputError(str(exc))
    return _clean({"formula": formula, "sigma": _sigma_json(a["sigma"]), "p_superionic_logreg": a["p_logreg"],
                   "verdict": verdict(a["p_logreg"], a["sigma"]["log10_sigma"], a["sigma"].get("sigma_pct")),
                   **_reality_json(a)})


def design_cif(formula: str, space_group: Optional[str] = None, base: Optional[str] = None) -> Dict:
    """CIF text for any formula (the real structure when known, else a substitution model)."""
    from crystalscout.ml.structgen import generate

    try:
        parse_checked_formula(formula)
    except FormulaError as exc:
        raise InputError(str(exc))
    gen = generate(formula, base or None, space_group=space_group or None)
    if not gen.get("ok"):
        raise InputError("No crystal structure could be built: " + gen.get("reason", ""))
    return {"formula": formula, "exact": gen["exact"], "host": gen["host_id"], "host_formula": gen["host_formula"],
            "relaxed": gen["relaxed"], "cif": gen["cif"]}


def design_lab(formula: str, space_group: Optional[str] = None, base: Optional[str] = None) -> Dict:
    from crystalscout.ml.design import lab_evaluate

    try:
        r = lab_evaluate(get_regressor(verbose=False), formula, space_group or None, base or None, reference(), bundle())
    except FormulaError as exc:
        raise InputError(str(exc))
    s = r["sigma"]
    return _clean({"formula": r["formula"], "sigma": _sigma_json(s), "verdict": r["verdict"], "flags": r["flags"],
                   "charge_balanced": r["charge_balanced"], "oxidation_states": r["oxidation"],
                   "baseline": r.get("base"), "gain_vs_baseline": r.get("gain"), "compared_on": r.get("compare_path"),
                   "drivers": r.get("drivers", []), "descriptor_change": r.get("descriptor_change"),
                   **_reality_json({**r, "honesty": r["honesty"]})})


def design_advise(formula: str, space_group: Optional[str] = None, top: int = 10, allow_tm: bool = False,
                  avoid_toxic: bool = True, avoid_supply_risk: bool = False) -> Dict:
    from crystalscout.ml.design import advise

    if not 1 <= int(top) <= 100:
        raise InputError("top must be between 1 and 100")
    try:
        r = advise(get_regressor(verbose=False), formula, space_group or None, int(top), allow_tm, avoid_toxic,
                   avoid_supply_risk, reference(), bundle())
    except FormulaError as exc:
        raise InputError(str(exc))
    r.pop("all", None)
    r["base"] = _sigma_json(r["base"])
    for sgg in r["suggestions"]:
        sgg.pop("pretty", None)
    for k in r.get("known", []):
        k.pop("pretty", None)
    r.pop("pretty", None)
    if r.get("best"):
        r["best"] = {k: v for k, v in r["best"].items() if k != "pretty"}
    r["honesty"] = "Predicted conductivities are screening priorities, not measurements."
    return _clean(r)


def design_recipe(formula: str, batch_g: float = 1.0) -> Dict:
    from crystalscout.ml.design import synthesis_plan

    if not 0.01 <= float(batch_g) <= 10000:
        raise InputError("batch_g must be between 0.01 and 10000")
    try:
        r = synthesis_plan(formula, float(batch_g))
    except FormulaError as exc:
        raise InputError(str(exc))
    for row in r.get("rows", []):
        row.pop("pretty", None)
    r.pop("pretty", None)
    return _clean(r)
