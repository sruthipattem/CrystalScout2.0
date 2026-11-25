"""External benchmark: Sendek et al. 2019 DFT-MD verdicts (Tables 1-2).

40 Materials Project structures were simulated with DFT-MD in the base paper
(20 chosen by their ML model, 21 at random; LiSO3F appears in both). They give
an independent, physics-based label for materials our OBELiX-trained models
never saw. 38 of them are in the bundled structure pool (mp-569782 lies outside
it; Li6UO6 is not in the snapshot).

Caveat recorded in the UI: the published Sendek model *chose* the Table-1
materials, so its own score on this set is biased upward; the retrained
CrystalScout models are independent of the selection.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..config import DATA_DIR
from ..core.descriptors import FEATURES
from ..featurize import load_pool_features
from .models import MODEL_LABELS

BENCH_FILE = DATA_DIR / "benchmarks" / "sendek2019_dftmd.csv"


def load_benchmark() -> pd.DataFrame:
    return pd.read_csv(BENCH_FILE)


def run_benchmark(bundle) -> Dict:
    b = load_benchmark()
    pool = load_pool_features()
    df = b.merge(pool, on="material_id", how="inner")
    df = df[df[FEATURES].notna().all(axis=1)].reset_index(drop=True)
    X = df[FEATURES].to_numpy(float)
    y = df["superionic_dftmd"].to_numpy(int)
    out: Dict = {"n": int(len(df)), "n_pos": int(y.sum()), "n_missing": int(len(b) - len(df)), "models": {}}
    for m in ["logreg", "random_forest", "gboost", "sendek2019"]:
        p = bundle.models[m].predict_proba(X)[:, 1]
        df[f"p_{m}"] = p
        order = np.argsort(-p, kind="stable")
        k = int(y.sum())
        out["models"][m] = {
            "label": MODEL_LABELS[m],
            "roc_auc": float(roc_auc_score(y, p)),
            "hits_in_top_k": int(y[order[:k]].sum()), "k": k,
            "precision_at_k": float(y[order[:k]].mean()),
            "enrichment": float(y[order[:k]].mean() / y.mean()),
        }
    out["table"] = df[["material_id", "formula", "selection", "rt_sigma_mScm", "superionic_dftmd", "marginal"]
                      + [f"p_{m}" for m in out["models"]]].round(4).to_dict(orient="records")
    return out
