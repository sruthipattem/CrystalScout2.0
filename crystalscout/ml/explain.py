"""Per-prediction explanations.

For the logistic-regression ranking model the explanation is *exact*: the
log-odds is a sum of per-feature terms, so

    contribution_i = theta_i * (x_i - mean_i) / std_i     (log-odds units)

measured against the average training material. This equals the SHAP value
of a linear model with an interventional (training-mean) baseline, so no
sampling approximation is needed. Positive = pushes towards "superionic".

These are statements about the *model*, not physical causation (Student
Guide, step 11).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..core.descriptors import FEATURE_INFO, FEATURES


def logreg_contributions(bundle, x: np.ndarray) -> Dict:
    lr = bundle.models["logreg"]
    sc, clf = lr.named_steps["scale"], lr.named_steps["clf"]
    z = (np.asarray(x, float) - sc.mean_) / sc.scale_
    contrib = clf.coef_[0] * z
    base = float(clf.intercept_[0])  # log-odds of the average training material
    logit = base + float(contrib.sum())
    items: List[Dict] = []
    for k, c, v, m in zip(FEATURES, contrib, x, sc.mean_):
        items.append({"feature": k, "name": FEATURE_INFO[k]["name"], "value": float(v), "train_mean": float(m),
                      "contribution": float(c), "direction": "up" if c > 0 else "down"})
    items.sort(key=lambda d: -abs(d["contribution"]))
    return {"baseline_logit": base, "logit": logit, "probability": float(1 / (1 + np.exp(-logit))), "terms": items}


def explain_sentence(expl: Dict, top: int = 2) -> str:
    ups = [t for t in expl["terms"] if t["contribution"] > 0][:top]
    downs = [t for t in expl["terms"] if t["contribution"] < 0][:top]
    parts = []
    if ups:
        parts.append(" and ".join(f"{t['feature']} ({t['value']:.2f})" for t in ups) + " pushed the score up")
    if downs:
        parts.append(" and ".join(f"{t['feature']} ({t['value']:.2f})" for t in downs) + " pulled it down")
    return "; ".join(parts) + "." if parts else "All descriptors are close to the training average."


def applicability(bundle, x: np.ndarray) -> Dict:
    """Is the candidate inside the region the model was trained on?

    Flags any descriptor outside the 1st-99th percentile of the training set,
    and reports a Mahalanobis distance percentile against training data.
    """
    stats = bundle.feature_stats
    out_of_range = [k for k, v in zip(FEATURES, x) if not (stats[k]["p01"] <= v <= stats[k]["p99"])]
    note = ""
    pct = None
    if bundle.train_X is not None and len(bundle.train_X) > 10:
        X = bundle.train_X
        mu = X.mean(0)
        cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        inv = np.linalg.inv(cov)
        d_train = np.sqrt(np.einsum("ij,jk,ik->i", X - mu, inv, X - mu))
        d = float(np.sqrt((x - mu) @ inv @ (x - mu)))
        pct = float((d_train < d).mean() * 100)
    in_domain = not out_of_range and (pct is None or pct <= 97.5)
    if out_of_range:
        note = "outside training range: " + ", ".join(out_of_range)
    elif pct is not None and pct > 97.5:
        note = "unusual combination of descriptors (Mahalanobis > 97.5th pct of training)"
    return {"in_domain": in_domain, "out_of_range": out_of_range, "mahalanobis_pct": pct, "note": note}
