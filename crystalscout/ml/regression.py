"""CrystalScout σ-Regressor: predict room-temperature log10 σ (S/cm), with uncertainty.

Why a regressor next to the Sendek classifiers
----------------------------------------------
The base-paper model answers yes/no ("superionic?"). Industry users need a
number they can rank by and compare with a target (e.g. "> 1 mS/cm"), plus an
honest error bar. This module follows the team's project plan (Extra Trees on
log10 σ, composition + structure features, formula-grouped validation) and
adds a calibrated prediction interval and a training-overlap flag.

Protocol (fixed before looking at test scores)
----------------------------------------------
* Data      OBELiX, all 599 entries; the 37 values reported as "<1E-10" are
            upper bounds, not measurements, and are dropped -> 562 rows.
* Split     OBELiX official train/test split. The test rows are used once.
* Tuning    5-fold GroupKFold on the training rows, grouped by reduced
            formula (repeated compositions never straddle folds).
* Primary   Extra Trees, 500 trees (the team's model choice). Mean baseline,
            ridge, random forest and gradient boosting are reported for
            comparison only.
* Interval  80 % interval = prediction ± k·(spread across trees), with k
            calibrated on the out-of-fold residuals of the training rows;
            the test-set coverage is reported, not assumed.

Features (computed offline, identically for OBELiX, Materials Project and uploads)
---------------------------------------------------------------------------------
* Composition (Magpie-style): 11 element properties x 6 statistics
  (fraction-weighted mean, average deviation, min, max, range, value of the
  most abundant element) + stoichiometry norms + Li and anion fractions.
* Structure: crystal system and lattice centring (from the space-group
  symbol), volume per atom, and the five Sendek descriptors when a crystal
  structure is available (missing values are median-imputed with indicator
  columns, so composition-only candidates can still be scored).

Cell edge lengths a, b, c are deliberately NOT used: the Materials Project
stores primitive cells while OBELiX reports conventional cells, so the same
crystal would get different values. Volume per atom is the same either way.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import MODELS_DIR, PROCESSED_DIR, RANDOM_SEED, SUPERIONIC_THRESHOLD
from ..core.chemistry import CONVENTIONAL_ANIONS
from ..core.descriptors import FEATURES as SENDEK
from ..core.elements import PROPS, element_props

REGRESSOR_FILE = MODELS_DIR / "sigma_regressor.joblib"
REPORT_FILE = MODELS_DIR / "regression_report.json"
TABLE_CACHE = PROCESSED_DIR / "obelix_regression_table.csv"
PRIMARY = "extra_trees"
INTERVAL = 0.80
LOG_THRESHOLD = math.log10(SUPERIONIC_THRESHOLD)  # -4

REG_LABELS = {
    "baseline": "Mean baseline",
    "ridge": "Ridge regression",
    "random_forest": "Random forest (500 trees)",
    "gboost": "Gradient boosting",
    "extra_trees": "Extra Trees (500 trees) — CrystalScout σ-Regressor",
}
ANION_FRAC = ["O", "S", "Se", "Te", "N", "F", "Cl", "Br", "I", "P", "B", "H"]
SYSTEMS = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]


# --------------------------------------------------------------------------- parsing
def parse_formula(formula: str) -> Dict[str, float]:
    """Element amounts from a formula, with nested parentheses and decimals: 'Li3(PS4)0.5Cl'."""
    s = re.sub(r"\s+", "", str(formula))
    stack: List[Dict[str, float]] = [{}]
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "([{":
            stack.append({})
            i += 1
        elif ch in ")]}":
            i += 1
            m = re.match(r"\d*\.?\d+", s[i:])
            mult = float(m.group()) if m else 1.0
            i += len(m.group()) if m else 0
            group = stack.pop()
            for el, n in group.items():
                stack[-1][el] = stack[-1].get(el, 0.0) + n * mult
        else:
            m = re.match(r"([A-Z][a-z]?)(\d*\.?\d*)", s[i:])
            if not m or not m.group(1):
                i += 1
                continue
            el, n = m.group(1), m.group(2)
            stack[-1][el] = stack[-1].get(el, 0.0) + (float(n) if n not in ("", ".") else 1.0)
            i += len(m.group())
    out = stack[0]
    while len(stack) > 1:  # unbalanced brackets: fold what we have
        for el, n in stack.pop().items():
            out[el] = out.get(el, 0.0) + n
    return {el: n for el, n in out.items() if n > 0}


def crystal_system(symbol) -> Dict[str, float]:
    """Crystal system (0-6) and centring from a Hermann-Mauguin symbol such as 'P -1', 'C2/m', 'Fm-3m'."""
    s = re.sub(r"\s+", "", str(symbol or ""))
    if not s or s.lower() == "nan":
        return {"crystal_system": np.nan, "centring": np.nan}
    lat, body = s[0].upper(), s[1:].replace("_", "")
    if "6" in body:
        sysn = 5
    elif body.startswith("3") or body.startswith("-3"):
        sysn = 4
    elif "3" in body:
        sysn = 6
    elif "4" in body:
        sysn = 3
    elif body in ("1", "-1"):
        sysn = 0
    elif re.fullmatch(r"(21|2|m|c|a|n|b)(/(m|c|a|n|b))?", body):
        sysn = 1
    else:
        sysn = 2
    if lat == "R":
        sysn = 4
    centring = {"P": 0, "A": 1, "B": 1, "C": 1, "I": 2, "F": 3, "R": 4}.get(lat, np.nan)
    return {"crystal_system": float(sysn), "centring": float(centring)}


def volume_from_params(a, b, c, alpha, beta, gamma) -> float:
    try:
        ca, cb, cg = (math.cos(math.radians(float(x))) for x in (alpha, beta, gamma))
        v = float(a) * float(b) * float(c) * math.sqrt(max(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg, 0))
        return v if v > 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


# --------------------------------------------------------------------------- features
def composition_features(comp: Dict[str, float]) -> Dict[str, float]:
    tot = sum(comp.values())
    out: Dict[str, float] = {}
    if tot <= 0:
        return out
    els = list(comp)
    frac = np.array([comp[e] / tot for e in els])
    props = [element_props(e) for e in els]
    main = int(np.argmax(frac))
    for p in PROPS:
        v = np.array([pr[p] for pr in props])
        mean = float((frac * v).sum())
        out[f"mean_{p}"] = mean
        out[f"avgdev_{p}"] = float((frac * np.abs(v - mean)).sum())
        out[f"min_{p}"] = float(v.min())
        out[f"max_{p}"] = float(v.max())
        out[f"range_{p}"] = float(v.max() - v.min())
        out[f"mode_{p}"] = float(v[main])
    out["n_elements"] = float(len(els))
    for k in (2, 3, 5, 7, 10):
        out[f"stoich_L{k}"] = float((frac ** k).sum() ** (1.0 / k))
    for el in ["Li"] + ANION_FRAC:
        out[f"frac_{el}"] = float(comp.get(el, 0.0) / tot)
    an = [e for e in els if e in CONVENTIONAL_ANIONS]
    en = {e: element_props(e)["electronegativity"] for e in els}
    out["frac_anions"] = float(sum(comp[e] for e in an) / tot)
    out["en_li_anion"] = float(np.mean([en[e] - en["Li"] for e in an])) if an and "Li" in en else 0.0
    out["n_anion_types"] = float(len(an))
    return out


def feature_row(comp: Dict[str, float], space_group=None, volume_per_atom=np.nan,
                descriptors: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    row = composition_features(comp)
    row.update(crystal_system(space_group))
    row["volume_per_atom"] = float(volume_per_atom) if volume_per_atom is not None else np.nan
    for k in SENDEK:
        v = (descriptors or {}).get(k, np.nan)
        row[k] = float(v) if v is not None and v == v else np.nan
    return row


def obelix_regression_table(use_cache: bool = True) -> pd.DataFrame:
    """One row per usable OBELiX entry: identifiers, target and features."""
    if use_cache and TABLE_CACHE.exists():
        return pd.read_csv(TABLE_CACHE)
    from ..data.obelix import build_features, load_obelix

    df = load_obelix()
    df = df[~df["sigma_upper_bound"] & df["sigma"].notna() & (df["sigma"] > 0)].copy()
    desc = build_features()
    desc = desc[desc["excluded_reason"] == ""].set_index("ID")
    rows = []
    for _, r in df.iterrows():
        comp = parse_formula(r["True Composition"]) or parse_formula(r["Reduced Composition"])
        natoms = sum(comp.values())
        vol = volume_from_params(r["a"], r["b"], r["c"], r["alpha"], r["beta"], r["gamma"])
        d = desc.loc[r["ID"], SENDEK].to_dict() if r["ID"] in desc.index else None
        feats = feature_row(comp, r["Space group"], vol / natoms if natoms else np.nan, d)
        rows.append({"ID": r["ID"], "formula": r["Reduced Composition"], "family": r["family"], "split": r["split"],
                     "log10_sigma": float(np.log10(r["sigma"])), "group": norm_formula(r["Reduced Composition"]),
                     **feats})
    out = pd.DataFrame(rows)
    out.to_csv(TABLE_CACHE, index=False)
    return out


def norm_formula(f: str) -> str:
    comp = parse_formula(f)
    if not comp:
        return str(f)
    g = min(comp.values())
    return "".join(f"{el}{round(comp[el] / g, 3):g}" for el in sorted(comp))


# --------------------------------------------------------------------------- models
def _models() -> Dict[str, object]:
    imp = lambda: SimpleImputer(strategy="median", add_indicator=True)  # noqa: E731
    return {
        "baseline": Pipeline([("impute", imp()), ("model", DummyRegressor())]),
        "ridge": Pipeline([("impute", imp()), ("scale", StandardScaler()),
                           ("model", RidgeCV(alphas=np.logspace(-2, 3, 20)))]),
        "random_forest": Pipeline([("impute", imp()), ("model", RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=RANDOM_SEED))]),
        "gboost": Pipeline([("impute", imp()), ("model", HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=8, l2_regularization=1.0,
            random_state=RANDOM_SEED))]),
        "extra_trees": Pipeline([("impute", imp()), ("model", ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=1, max_features=0.6, n_jobs=-1, random_state=RANDOM_SEED))]),
    }


def _tree_spread(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    Xt = pipe.named_steps["impute"].transform(X)
    preds = np.stack([t.predict(Xt) for t in pipe.named_steps["model"].estimators_])
    return preds.std(axis=0)


def _metrics(y, p) -> Dict[str, float]:
    y, p = np.asarray(y, float), np.asarray(p, float)
    out = {"n": int(len(y)), "mae": float(mean_absolute_error(y, p)),
           "rmse": float(math.sqrt(mean_squared_error(y, p))), "r2": float(r2_score(y, p)),
           "spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman")),
           "within_1_log": float((np.abs(y - p) <= 1.0).mean())}
    lab = (y >= LOG_THRESHOLD).astype(int)
    if 0 < lab.sum() < len(lab):
        out["superionic_auc"] = float(roc_auc_score(lab, p))
    return out


@dataclass
class SigmaRegressor:
    model: Pipeline
    feature_names: List[str]
    k_interval: float
    train_groups: List[str]
    train_ranges: Dict[str, List[float]]
    report: Dict = field(default_factory=dict)
    eval_model: Optional[Pipeline] = None  # fitted on the training split only (for honest evaluation)

    def predict_frame(self, F: pd.DataFrame, evaluation: bool = False) -> pd.DataFrame:
        """Predictions from the deployed model (all OBELiX rows), or with ``evaluation=True`` from the
        model that never saw the OBELiX test split."""
        model = self.eval_model if (evaluation and self.eval_model is not None) else self.model
        X = F.reindex(columns=self.feature_names)
        # one pass over the 500 trees gives the mean (= the forest prediction), the spread and the votes
        Xt = np.asarray(model.named_steps["impute"].transform(X), dtype=np.float32)
        trees = np.stack([t.predict(Xt, check_input=False) for t in model.named_steps["model"].estimators_])
        mu = trees.mean(axis=0)
        sd = trees.std(axis=0)
        half = self.k_interval * sd
        out = pd.DataFrame({"log10_sigma_pred": mu, "log10_sigma_lo": mu - half, "log10_sigma_hi": mu + half,
                            "log10_sigma_spread": sd}, index=F.index)
        out["sigma_pred_mScm"] = 10 ** (mu + 3)
        # fraction of trees that put the material above the superionic threshold
        out["p_superionic_trees"] = (trees >= LOG_THRESHOLD).mean(axis=0)
        return out

    def predict_one(self, comp: Dict[str, float], space_group=None, volume_per_atom=np.nan,
                    descriptors: Optional[Dict[str, float]] = None, formula: str = "") -> Dict:
        row = feature_row(comp, space_group, volume_per_atom, descriptors)
        r = self.predict_frame(pd.DataFrame([row])).iloc[0].to_dict()
        r["known_composition"] = bool(formula and norm_formula(formula) in set(self.train_groups))
        r["out_of_range"] = [k for k in ("volume_per_atom", "mean_electronegativity", "frac_Li", "mean_Z")
                             if k in row and row[k] == row[k] and not
                             (self.train_ranges[k][0] <= row[k] <= self.train_ranges[k][1])]
        r["has_structure_descriptors"] = bool(descriptors)
        return r


def train_regressor(table: Optional[pd.DataFrame] = None, verbose: bool = False) -> SigmaRegressor:
    t0 = time.time()
    table = obelix_regression_table() if table is None else table
    meta = {"ID", "formula", "family", "split", "log10_sigma", "group"}
    feats = [c for c in table.columns if c not in meta]
    tr = table[table["split"] == "train"].reset_index(drop=True)
    te = table[table["split"] == "test"].reset_index(drop=True)
    Xtr, ytr, gtr = tr[feats], tr["log10_sigma"].to_numpy(), tr["group"].to_numpy()
    Xte, yte = te[feats], te["log10_sigma"].to_numpy()
    cv = GroupKFold(n_splits=5)
    report: Dict = {
        "protocol": {
            "target": "log10 of room-temperature ionic conductivity (S/cm)",
            "data": "OBELiX; '<1E-10' upper-bound values removed",
            "n_total": int(len(table)), "n_train": int(len(tr)), "n_test": int(len(te)),
            "n_features": len(feats), "split": "OBELiX official train/test split",
            "cv": "5-fold GroupKFold on training rows, grouped by reduced formula",
            "n_train_with_structure": int(tr["LLB"].notna().sum()),
            "n_test_with_structure": int(te["LLB"].notna().sum()),
            "interval": f"{int(INTERVAL * 100)}% interval from tree spread, calibrated on out-of-fold residuals",
            "primary": PRIMARY,
        },
        "models": {},
    }
    fitted = {}
    oof_primary = np.zeros(len(tr))
    oof_spread = np.zeros(len(tr))
    for name, pipe in _models().items():
        oof = np.zeros(len(tr))
        for a, b in cv.split(Xtr, ytr, gtr):
            m = _models()[name].fit(Xtr.iloc[a], ytr[a])
            oof[b] = m.predict(Xtr.iloc[b])
            if name == PRIMARY:
                oof_spread[b] = _tree_spread(m, Xtr.iloc[b])
        if name == PRIMARY:
            oof_primary = oof
        pipe.fit(Xtr, ytr)
        fitted[name] = pipe
        pte = pipe.predict(Xte)
        report["models"][name] = {"label": REG_LABELS[name], "cv": _metrics(ytr, oof), "test": _metrics(yte, pte)}
        # structure-available subset of the test set (comparable to the 5-descriptor classifiers)
        mask = te["LLB"].notna().to_numpy()
        if mask.sum() >= 10:
            report["models"][name]["test_with_structure"] = _metrics(yte[mask], pte[mask])
        if verbose:
            m = report["models"][name]
            print(f"  {name:14s} CV MAE {m['cv']['mae']:.3f} R2 {m['cv']['r2']:.3f} | "
                  f"test MAE {m['test']['mae']:.3f} RMSE {m['test']['rmse']:.3f} R2 {m['test']['r2']:.3f}")
    ratio = np.abs(ytr - oof_primary) / np.maximum(oof_spread, 1e-6)
    k = float(np.quantile(ratio, INTERVAL))
    best = fitted[PRIMARY]
    sd_te = _tree_spread(best, Xte)
    mu_te = best.predict(Xte)
    report["interval"] = {"k": k, "nominal": INTERVAL,
                          "cv_coverage": float((ratio <= k).mean()),
                          "test_coverage": float((np.abs(yte - mu_te) <= k * sd_te).mean()),
                          "test_mean_halfwidth_log": float((k * sd_te).mean())}
    imp = best.named_steps["model"].feature_importances_[:len(feats)]
    order = np.argsort(-imp)[:15]
    report["top_features"] = [{"feature": feats[i], "importance": float(imp[i])} for i in order]
    report["test_predictions"] = [{"formula": f, "family": fam, "true": float(y), "pred": float(p)}
                                  for f, fam, y, p in zip(te["formula"], te["family"], yte, mu_te)]
    # Production model: same settings, refit on every OBELiX row once evaluation is finished
    # (standard practice; all metrics above come from the train-only model).
    deployed = _models()[PRIMARY].fit(table[feats], table["log10_sigma"].to_numpy())
    report["deployment"] = {"model": REG_LABELS[PRIMARY], "n_rows": int(len(table)),
                            "note": "After evaluation the same model was refit on all OBELiX rows for use in the "
                                    "app; every reported metric comes from the model trained on the training split only."}
    report["seconds"] = round(time.time() - t0, 1)
    ranges = {c: [float(np.nanmin(table[c])), float(np.nanmax(table[c]))]
              for c in ("volume_per_atom", "mean_electronegativity", "frac_Li", "mean_Z")}
    reg = SigmaRegressor(model=deployed, feature_names=feats, k_interval=k,
                         train_groups=sorted(set(table["group"])), train_ranges=ranges, report=report,
                         eval_model=best)
    return reg


def save_regressor(reg: SigmaRegressor) -> None:
    import sklearn

    joblib.dump({"reg": reg, "sklearn_version": sklearn.__version__}, REGRESSOR_FILE, compress=3)
    REPORT_FILE.write_text(json.dumps(reg.report, indent=1))


_REG: Optional[SigmaRegressor] = None


def get_regressor(verbose: bool = True) -> SigmaRegressor:
    """Load the saved regressor; retrain (a few seconds) if missing or built by another scikit-learn."""
    global _REG
    if _REG is not None:
        return _REG
    import sklearn

    try:
        if REGRESSOR_FILE.exists():
            d = joblib.load(REGRESSOR_FILE)
            if d.get("sklearn_version") == sklearn.__version__ and getattr(d["reg"], "eval_model", None) is not None:
                _REG = d["reg"]
                return _REG
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"Could not load the σ-regressor ({exc}); retraining.")
    if verbose:
        print("Training the CrystalScout σ-Regressor (one time, a few seconds) ...")
    _REG = train_regressor()
    save_regressor(_REG)
    return _REG


# --------------------------------------------------------------------------- candidates
def volume_per_atom_from_dict(st: Dict) -> float:
    """Volume per atom of a pymatgen-style structure dict (occupancy-weighted atom count)."""
    try:
        vol = abs(float(np.linalg.det(np.array(st["lattice"]["matrix"], float))))
        n = sum(sum(sp.get("occu", 1.0) for sp in site["species"]) for site in st["sites"])
        return vol / n if n > 0 else np.nan
    except (KeyError, TypeError, ValueError):
        return np.nan


def structure_volume_per_atom(st) -> float:
    """Volume per atom of a crystalscout.core.structure.Structure."""
    try:
        n = sum(st.composition().values())
        return float(st.volume) / n if n > 0 else np.nan
    except Exception:  # noqa: BLE001
        return np.nan


def mp_feature_frame(records: pd.DataFrame, pool_features: Optional[pd.DataFrame] = None,
                     structures: Optional[Dict] = None) -> pd.DataFrame:
    """Features for Materials Project rows (formula, space_group, nsites, material_id)."""
    desc = pool_features.set_index("material_id") if pool_features is not None and len(pool_features) else None
    rows = []
    for _, r in records.iterrows():
        comp = parse_formula(r["formula"])
        vpa = np.nan
        st = (structures or {}).get(r["material_id"]) if structures is not None else None
        if st is not None:
            vpa = volume_per_atom_from_dict(st)
        d = None
        if desc is not None and r["material_id"] in desc.index:
            dd = desc.loc[r["material_id"]]
            if all(dd.get(k) == dd.get(k) for k in SENDEK):
                d = {k: float(dd[k]) for k in SENDEK}
        rows.append(feature_row(comp, r.get("space_group"), vpa, d))
    return pd.DataFrame(rows, index=records.index)
