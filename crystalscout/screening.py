"""End-to-end screening: Stage A -> shared feature extractor -> trained model
-> ranked, explained shortlist. Works on the bundled snapshot or on a live
Materials Project pull."""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import RUNS_DIR
from .core.descriptors import FEATURES, sendek_probability
from .data.materials_project import load_snapshot
from .data.obelix import load_obelix
from .data.stage_a import StageACriteria, apply_stage_a
from .featurize import featurize_records, load_pool_features
from .ml.explain import applicability, explain_sentence, logreg_contributions
from .ml.models import MODEL_LABELS

MODEL_KEYS = ["logreg", "random_forest", "gboost", "sendek2019"]
DEFAULT_RANK = "crystalscout"
RANK_LABELS = {"crystalscout": "CrystalScout Score (σ-Regressor + structure model)",
               "sigma": "σ-Regressor: predicted conductivity (Extra Trees)", **MODEL_LABELS}
RANK_SHORT = {"crystalscout": "CrystalScout Score", "sigma": "Predicted σ", "logreg": "Logistic regression",
              "random_forest": "Random forest", "gboost": "Gradient boosting", "sendek2019": "Sendek 2019 equation"}
SIGMA_COLS = ["log10_sigma_pred", "log10_sigma_lo", "log10_sigma_hi", "log10_sigma_spread", "sigma_pred_mScm",
              "p_superionic_trees"]


def rank_column(model: str) -> str:
    """Column a shortlist is sorted by for a ranking choice."""
    if model == "crystalscout":
        return "cs_score"
    return "log10_sigma_pred" if model == "sigma" else f"p_{model}"


def add_sigma(df: pd.DataFrame, structures: Optional[Dict] = None) -> pd.DataFrame:
    """Add σ-Regressor predictions (all rows, structure descriptors used when present)."""
    from .ml.regression import get_regressor, mp_feature_frame

    df = df.copy()
    if not len(df):
        for c in SIGMA_COLS:
            df[c] = pd.Series(dtype=float)
        return df
    reg = get_regressor(verbose=False)
    rec = df[["material_id", "formula"]].copy()
    rec["space_group"] = df["space_group"] if "space_group" in df.columns else None
    desc = df[["material_id", *FEATURES]] if all(k in df.columns for k in FEATURES) else None
    F = mp_feature_frame(rec, desc, structures)
    pred = reg.predict_frame(F)
    for c in SIGMA_COLS:
        df[c] = pred[c].to_numpy()
    from .ml.consensus import cs_score

    lr = df["p_logreg"].to_numpy(float) if "p_logreg" in df.columns else np.full(len(df), np.nan)
    df["cs_score"] = cs_score(df["log10_sigma_pred"], lr)
    return df


# ------------------------------------------------------------ formula match
def norm_formula(f: str) -> str:
    """Order-independent reduced formula key, e.g. 'Cl6Li3In' -> 'Cl6InLi3'."""
    f = str(f).replace(" ", "")
    # expand one level of parentheses: Li3(PO4)2 -> Li3P2O8
    def expand(m):
        inner, mult = m.group(1), float(m.group(2) or 1)
        return "".join(f"{el}{(float(n) if n else 1.0) * mult:g}" for el, n in re.findall(r"([A-Z][a-z]?)([\d.]*)", inner))
    while "(" in f:
        nf = re.sub(r"\(([^()]*)\)([\d.]*)", expand, f)
        if nf == f:
            break
        f = nf
    comp: Dict[str, float] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)([\d.]*)", f):
        comp[el] = comp.get(el, 0.0) + (float(n) if n else 1.0)
    if not comp:
        return ""
    vals = list(comp.values())
    if all(abs(v - round(v)) < 1e-3 for v in vals):
        g = 0
        for v in vals:
            g = math.gcd(g, int(round(v)))
        g = g or 1
    else:
        g = min(vals)
    return "".join(f"{el}{comp[el] / g:g}" for el in sorted(comp))


_OBX_INDEX: Optional[Dict[str, Dict]] = None


def obelix_lookup(formula: str) -> Optional[Dict]:
    """Measured OBELiX conductivity for the same composition, if any."""
    global _OBX_INDEX
    if _OBX_INDEX is None:
        df = load_obelix()
        _OBX_INDEX = {}
        for _, r in df.iterrows():
            k = norm_formula(r["formula"])
            if k and (k not in _OBX_INDEX or r["sigma"] > _OBX_INDEX[k]["sigma"]):
                _OBX_INDEX[k] = {"ID": r["ID"], "sigma": float(r["sigma"]), "family": r["family"], "doi": r["DOI"],
                                 "split": r["split"]}
    return _OBX_INDEX.get(norm_formula(formula))


# ----------------------------------------------------------------- scoring
def score_frame(bundle, feats: pd.DataFrame) -> pd.DataFrame:
    """Add probability columns, consensus, explanation and domain flags."""
    df = feats.copy().reset_index(drop=True)
    ok = df[FEATURES].notna().all(axis=1).to_numpy()
    X = df.loc[ok, FEATURES].to_numpy(float)
    idx = df.index[ok]

    def put(col, values, fill=np.nan):
        arr = pd.Series([fill] * len(df), index=df.index, dtype=object)
        if len(idx):
            arr.loc[idx] = list(values)
        df[col] = arr.infer_objects()

    trained = ["logreg", "random_forest", "gboost"]
    probs = {m: (bundle.models[m].predict_proba(X)[:, 1] if len(X) else np.array([])) for m in MODEL_KEYS}
    for m in MODEL_KEYS:
        put(f"p_{m}", probs[m])
    if len(X):
        ranks = pd.DataFrame({m: pd.Series(probs[m]).rank(pct=True) for m in trained})
        consensus = ranks.mean(axis=1).to_numpy()
        npos = (np.column_stack([probs[m] for m in trained]) >= 0.5).sum(axis=1)
    else:
        consensus, npos = [], []
    put("consensus", consensus)
    put("n_models_positive", npos)
    expl = [logreg_contributions(bundle, x) for x in X]
    dom = [applicability(bundle, x) for x in X]
    put("why", [explain_sentence(e) for e in expl], "")
    put("top_driver", [e["terms"][0]["feature"] + (" ↑" if e["terms"][0]["contribution"] > 0 else " ↓") for e in expl], "")
    put("in_domain", [d["in_domain"] for d in dom], None)
    put("domain_note", [d["note"] for d in dom], "")
    return df


@dataclass
class ScreeningRun:
    run_id: str
    source: str
    criteria: StageACriteria
    stats: Dict
    shortlist: pd.DataFrame
    rejected: pd.DataFrame
    created: str
    rank_model: str = DEFAULT_RANK

    def top(self, n: int = 20) -> pd.DataFrame:
        return self.shortlist.head(n)

    def _n_superionic(self) -> int:
        s = self.shortlist
        if self.rank_model == "crystalscout":
            return int((s["p_logreg"] >= 0.5).sum()) if "p_logreg" in s else 0
        if self.rank_model == "sigma":
            return int((s["log10_sigma_pred"] >= -4).sum()) if "log10_sigma_pred" in s else 0
        return int((s[f"p_{self.rank_model}"] >= 0.5).sum())

    def summary(self) -> Dict:
        s = self.shortlist
        return {
            "run_id": self.run_id, "source": self.source, "created": self.created, "rank_model": self.rank_model,
            "criteria_text": self.criteria.describe(), "funnel": self.stats["funnel"],
            "n_input": self.stats["n_input"], "n_stage_a": self.stats["n_passed"],
            "n_scored": int(s[rank_column(self.rank_model)].notna().sum()) if len(s) else 0,
            "n_structure_scored": int(s["p_logreg"].notna().sum()) if len(s) else 0,
            "n_predicted_superionic": int(self._n_superionic()) if len(s) else 0,
            "n_above_1mScm": int((s["sigma_pred_mScm"] >= 1.0).sum()) if len(s) and "sigma_pred_mScm" in s else 0,
            "n_new_candidates": int(s["obelix_sigma"].isna().sum()) if len(s) and "obelix_sigma" in s else 0,
            "n_all_models_agree": int((s["n_models_positive"] == 3).sum()) if len(s) else 0,
            "n_feature_errors": int((s["feature_error"] != "").sum()) if len(s) else 0,
            "run_by": self.stats.get("run_by", ""),
        }


def run_screening(bundle, criteria: Optional[StageACriteria] = None, source: str = "snapshot",
                  api_key: Optional[str] = None, rank_model: str = DEFAULT_RANK,
                  progress: Optional[Callable[[str, int, int], None]] = None, save: bool = True) -> ScreeningRun:
    criteria = criteria or StageACriteria()
    t0 = time.time()
    if source == "live":
        from .data.materials_project import MaterialsProjectClient

        client = MaterialsProjectClient(api_key)
        recs = client.stage_a_query(criteria.band_gap_min, criteria.ehull_max, criteria.exclude_transition_metals,
                                    progress=(lambda a, b: progress("download", a, b)) if progress else None)
        base = pd.DataFrame([{k: v for k, v in r.items() if k != "structure"} for r in recs])
        if base.empty:
            raise RuntimeError("The Materials Project returned no materials for these criteria.")
        passed, rejected, stats = apply_stage_a(base, criteria)  # re-check locally (extra rules)
        structs = {r["material_id"]: r["structure"] for r in recs}
        feats = featurize_records([(m, structs[m]) for m in passed["material_id"]],
                                  progress=(lambda a, b: progress("features", a, b)) if progress else None)
        stats["note"] = "Live Materials Project query; rules applied server-side, so the funnel starts at survivors."
        sigma_structs = structs
    else:
        snap = load_snapshot()
        passed, rejected, stats = apply_stage_a(snap, criteria)
        pool = load_pool_features()
        feats = pool[pool["material_id"].isin(passed["material_id"])]
        missing = set(passed["material_id"]) - set(feats["material_id"])
        if missing:
            stats["note"] = (f"{len(missing)} survivors are outside the bundled structure pool "
                             "(gap ≥ 0.5 eV, E_hull ≤ 0.1): they get a composition-based σ estimate but no "
                             "structure-descriptor scores. Use the live Materials Project mode for full scoring.")
        from .data.materials_project import load_pool_structures

        sigma_structs = load_pool_structures()
    merged = passed.merge(feats, on="material_id", how="left")
    no_struct = merged["LLB"].isna() & merged["feature_error"].isna()
    merged["feature_error"] = merged["feature_error"].fillna("")
    merged.loc[no_struct, "feature_error"] = "structure not available offline"
    scored = add_sigma(score_frame(bundle, merged), sigma_structs)
    key = rank_column(rank_model)
    second = "log10_sigma_pred" if rank_model in ("crystalscout", "logreg", "random_forest", "gboost", "sendek2019") \
        else "cs_score"
    scored = scored.sort_values([key, second], ascending=False, na_position="last").reset_index(drop=True)
    scored["rank"] = np.arange(1, len(scored) + 1)
    # experimental cross-check against OBELiX
    look = [obelix_lookup(f) for f in scored["formula"]]
    scored["obelix_sigma"] = [l["sigma"] if l else np.nan for l in look]
    scored["obelix_id"] = [l["ID"] if l else "" for l in look]
    run_id = time.strftime("%Y%m%d-%H%M%S") + ("-live" if source == "live" else "-snapshot")
    run = ScreeningRun(run_id=run_id, source=source, criteria=criteria, stats=stats, shortlist=scored,
                       rejected=rejected, created=time.strftime("%Y-%m-%d %H:%M:%S"), rank_model=rank_model)
    run.stats["seconds"] = round(time.time() - t0, 2)
    if save:
        save_run(run)
    return run


def save_run(run: ScreeningRun) -> None:
    d = RUNS_DIR / run.run_id
    d.mkdir(parents=True, exist_ok=True)
    run.shortlist.to_csv(d / "ranked_candidates.csv", index=False)
    run.rejected[["material_id", "formula", "band_gap", "energy_above_hull", "elements", "rejection_reasons"]].to_csv(
        d / "stage_a_rejections.csv.gz", index=False, compression="gzip")
    with open(d / "run.json", "w") as fh:
        json.dump({"summary": run.summary(), "stats": run.stats, "criteria": run.criteria.as_dict()}, fh, indent=1,
                  default=str)


def list_runs() -> List[Dict]:
    out = []
    for d in sorted(RUNS_DIR.glob("*"), reverse=True):
        f = d / "run.json"
        if f.exists():
            try:
                out.append(json.load(open(f))["summary"])
            except Exception:
                pass
    return out


def load_run(run_id: str) -> Optional[ScreeningRun]:
    d = RUNS_DIR / run_id
    if not (d / "run.json").exists():
        return None
    meta = json.load(open(d / "run.json"))
    sl = pd.read_csv(d / "ranked_candidates.csv", keep_default_na=False, na_values=[""])
    rej = pd.read_csv(d / "stage_a_rejections.csv.gz", keep_default_na=False, na_values=[""])
    crit = StageACriteria(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["criteria"].items()})
    s = meta["summary"]
    if "log10_sigma_pred" not in sl.columns or "cs_score" not in sl.columns:  # run saved before the σ-Regressor existed
        from .data.materials_project import load_pool_structures

        sl = add_sigma(sl, load_pool_structures() if s["source"] == "snapshot" else None)
    return ScreeningRun(run_id=run_id, source=s["source"], criteria=crit, stats=meta["stats"], shortlist=sl,
                        rejected=rej, created=s["created"], rank_model=s.get("rank_model", "logreg"))
