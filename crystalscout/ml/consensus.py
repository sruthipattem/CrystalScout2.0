"""CrystalScout Score: the platform's default ranking.

Two model families see different evidence and fail in different places:

* the σ-Regressor (composition + structure, trained on measured OBELiX
  conductivities) ranks real laboratory materials well, but transfers less
  well to idealised Materials Project crystals;
* the retrained Sendek logistic regression (five structural descriptors)
  transfers well to Materials Project crystals (DFT-MD benchmark) but ranks
  OBELiX laboratory materials poorly.

The CrystalScout Score is the plain average of the two percentile ranks,
with equal weights fixed in advance (no weights are fitted, so nothing can
over-fit). When a candidate has no crystal structure, only the σ-Regressor
percentile is available and is used on its own.

``evaluate`` reports the score on both independent checks next to its two
ingredients, so the choice can be audited.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

W_SIGMA = 0.5


def percentile(x) -> np.ndarray:
    return pd.Series(np.asarray(x, float)).rank(pct=True).to_numpy()


def cs_score(log_sigma, p_lr) -> np.ndarray:
    """Within-set CrystalScout Score in [0, 1] (1 = best in the set)."""
    s = percentile(log_sigma)
    lr = np.asarray(p_lr, float)
    has = ~np.isnan(lr)
    out = s.copy()
    if has.any():
        l = np.full(len(lr), np.nan)
        l[has] = percentile(lr[has])
        out[has] = W_SIGMA * s[has] + (1 - W_SIGMA) * l[has]
    return out


def reference_score(log_sigma: float, p_lr: Optional[float], ref: pd.DataFrame) -> Dict:
    """Score one material against a reference set (e.g. the standard screen).

    Returns the score and the share of reference candidates it beats."""
    rs = np.sort(ref["log10_sigma_pred"].dropna().to_numpy(float))
    s = float(np.searchsorted(rs, log_sigma, side="right") / max(len(rs), 1))
    if p_lr is not None and p_lr == p_lr:
        rl = np.sort(ref["p_logreg"].dropna().to_numpy(float))
        l = float(np.searchsorted(rl, p_lr, side="right") / max(len(rl), 1))
        score = W_SIGMA * s + (1 - W_SIGMA) * l
    else:
        score = s
    if "cs_score" in ref:
        beats = float((ref["cs_score"].dropna().to_numpy(float) < score).mean())
    else:
        beats = score
    return {"score": score, "beats": beats, "n_ref": int(len(rs))}


def evaluate(bundle, regressor) -> Dict:
    """AUC of σ-Regressor, logistic regression and the CrystalScout Score on both checks."""
    from sklearn.metrics import roc_auc_score

    from ..core.descriptors import FEATURES
    from ..data.materials_project import load_pool_structures, load_snapshot
    from ..screening import add_sigma
    from .benchmark import run_benchmark
    from .regression import obelix_regression_table

    out: Dict = {"weights": {"sigma_regressor": W_SIGMA, "logistic_regression": 1 - W_SIGMA}}
    t = obelix_regression_table()
    te = t[(t["split"] == "test") & t["LLB"].notna()].copy()
    sig = regressor.predict_frame(te, evaluation=True)["log10_sigma_pred"].to_numpy()
    lr = bundle.models["logreg"].predict_proba(te[FEATURES].to_numpy(float))[:, 1]
    y = (te["log10_sigma"].to_numpy() >= -4).astype(int)
    out["obelix_test"] = {"n": int(len(te)), "n_pos": int(y.sum()),
                          "sigma_regressor": float(roc_auc_score(y, sig)),
                          "logistic_regression": float(roc_auc_score(y, lr)),
                          "crystalscout_score": float(roc_auc_score(y, cs_score(sig, lr)))}
    bench = pd.DataFrame(run_benchmark(bundle)["table"])
    snap = load_snapshot().set_index("material_id")
    bench["space_group"] = [snap.loc[m, "space_group"] if m in snap.index else None for m in bench["material_id"]]
    bench = add_sigma(bench, load_pool_structures())
    yb = bench["superionic_dftmd"].astype(int).to_numpy()
    out["dftmd_benchmark"] = {"n": int(len(bench)), "n_pos": int(yb.sum()),
                              "sigma_regressor": float(roc_auc_score(yb, bench["log10_sigma_pred"])),
                              "logistic_regression": float(roc_auc_score(yb, bench["p_logreg"])),
                              "crystalscout_score": float(roc_auc_score(yb, cs_score(bench["log10_sigma_pred"],
                                                                                     bench["p_logreg"])))}
    return out
