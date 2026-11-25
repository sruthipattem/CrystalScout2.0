"""Model training, validation and inference.

Protocol (docs/METHODS.md):

* Labels   OBELiX RT conductivity >= 1e-4 S/cm -> superionic (1).
* Split    OBELiX's official train/test split, which has no composition shared
           between train and test. The test set is touched exactly once, for
           the final report.
* Tuning   hyper-parameters chosen by grouped 5-fold CV on the training split
           only, with folds grouped by crystal *family* so that dopant variants
           of one family never sit on both sides of a fold.
* Models   1. Sendek 2019 published equation (no training, zero-shot benchmark)
           2. Logistic regression -- the base-paper model family, retrained
           3. Random forest           (enhancement, Student Guide step 8)
           4. Gradient boosting       (XGBoost if installed, else sklearn HGB)
* Metrics  precision, recall, F1, accuracy, ROC-AUC, PR-AUC, confusion matrix,
           bootstrap 95 % CI for ROC-AUC, plus leave-family-out CV AUC.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import MODELS_DIR, RANDOM_SEED
from ..core.descriptors import FEATURES, SENDEK_2019_COEFS, SENDEK_2019_INTERCEPT

MODEL_FILE = MODELS_DIR / "crystalscout_models.joblib"
REPORT_FILE = MODELS_DIR / "evaluation_report.json"

MODEL_LABELS = {
    "sendek2019": "Sendek 2019 (published eq. 1)",
    "logreg": "Logistic regression (retrained on OBELiX)",
    "random_forest": "Random forest",
    "gboost": "Gradient boosting",
}
PRIMARY_MODEL = "logreg"


class SendekPublished(BaseEstimator, ClassifierMixin):
    """The published Sendek 2019 equation as a fixed scikit-learn classifier."""

    classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def decision_function(self, X):
        X = np.asarray(X, float)
        theta = np.array([SENDEK_2019_COEFS[k] for k in FEATURES])
        return X @ theta + SENDEK_2019_INTERCEPT

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _gboost():
    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.9, eval_metric="logloss", random_state=RANDOM_SEED), "XGBoost"
    except Exception:
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, l2_regularization=1.0,
                                              class_weight="balanced", random_state=RANDOM_SEED), "HistGradientBoosting"


def candidate_models() -> Dict[str, Dict]:
    gb, gb_name = _gboost()
    return {
        "logreg": {
            "estimator": Pipeline([("scale", StandardScaler()),
                                   ("clf", LogisticRegression(class_weight="balanced", max_iter=5000))]),
            "grid": {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]},
        },
        "random_forest": {
            "estimator": RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1),
            "grid": {"min_samples_leaf": [2, 4, 8], "max_depth": [None, 6]},
        },
        "gboost": {"estimator": gb, "grid": {}, "impl": gb_name},
    }


def _metrics(y, p, thr=0.5) -> Dict:
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y, p)
    prec, rec, _ = precision_recall_curve(y, p)
    return {
        "n": int(len(y)), "n_pos": int(np.sum(y)),
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, p)) if len(set(y)) > 1 else float("nan"),
        "confusion": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "roc_curve": {"fpr": [round(float(v), 4) for v in fpr], "tpr": [round(float(v), 4) for v in tpr]},
        "pr_curve": {"precision": [round(float(v), 4) for v in prec], "recall": [round(float(v), 4) for v in rec]},
        **_enrichment(y, p),
    }


def _enrichment(y, p, frac: float = 0.2) -> Dict:
    """Discovery-rate view used by Sendek et al.: how much more often the
    model's top picks are superionic than a random pick from the same pool."""
    y, p = np.asarray(y), np.asarray(p)
    k = max(1, int(round(frac * len(y))))
    top = np.argsort(-p, kind="stable")[:k]
    base = float(np.mean(y)) if len(y) else float("nan")
    prec_k = float(np.mean(y[top]))
    return {"top_frac": frac, "top_k": k, "precision_at_k": prec_k, "base_rate": base,
            "enrichment": prec_k / base if base > 0 else float("nan")}


def _bootstrap_auc(y, p, n=1000, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], p[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def _grouped_cv_auc(est, X, y, groups, n_splits=5) -> Dict:
    """Out-of-fold AUC with family-grouped folds."""
    n_groups = len(set(groups))
    k = min(n_splits, n_groups)
    try:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED)
        splits = list(cv.split(X, y, groups))
    except Exception:
        splits = list(GroupKFold(n_splits=k).split(X, y, groups))
    oof = np.full(len(y), np.nan)
    fold_auc = []
    for tr, te in splits:
        m = clone(est).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        if len(set(y[te])) > 1:
            fold_auc.append(roc_auc_score(y[te], oof[te]))
    return {"oof_auc": float(roc_auc_score(y, oof)), "fold_auc_mean": float(np.mean(fold_auc)),
            "fold_auc_std": float(np.std(fold_auc)), "n_folds": len(splits), "oof": oof}


def _tune(spec, X, y, groups) -> (object, Dict):
    est, grid = spec["estimator"], spec["grid"]
    if not grid:
        return est, {}
    import itertools

    keys = list(grid)
    best, best_params, best_auc = None, None, -1
    for combo in itertools.product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        cand = clone(est).set_params(**params)
        auc = _grouped_cv_auc(cand, X, y, groups)["oof_auc"]
        if auc > best_auc:
            best, best_params, best_auc = cand, params, auc
    return best, best_params


@dataclass
class TrainedBundle:
    models: Dict[str, object]
    feature_stats: Dict[str, Dict[str, float]]
    report: Dict
    primary: str = PRIMARY_MODEL
    train_X: Optional[np.ndarray] = None

    def proba(self, X: np.ndarray, model: Optional[str] = None) -> np.ndarray:
        return self.models[model or self.primary].predict_proba(np.asarray(X, float))[:, 1]


def train_all(table: pd.DataFrame, verbose: bool = False) -> TrainedBundle:
    tr = table[table["split"] == "train"].reset_index(drop=True)
    te = table[table["split"] == "test"].reset_index(drop=True)
    Xtr, ytr, gtr = tr[FEATURES].to_numpy(float), tr["label"].to_numpy(int), tr["family"].to_numpy()
    Xte, yte = te[FEATURES].to_numpy(float), te["label"].to_numpy(int)
    Xall, yall, gall = table[FEATURES].to_numpy(float), table["label"].to_numpy(int), table["family"].to_numpy()

    report: Dict = {
        "protocol": {
            "label": "superionic if RT conductivity >= 1e-4 S/cm",
            "split": "OBELiX official train/test split (no composition shared)",
            "tuning": "grouped 5-fold CV on train split, grouped by crystal family",
            "n_train": int(len(tr)), "n_train_pos": int(ytr.sum()),
            "n_test": int(len(te)), "n_test_pos": int(yte.sum()),
            "n_families": int(len(set(gall))),
        },
        "models": {},
    }
    fitted: Dict[str, object] = {}
    specs = candidate_models()
    order = ["sendek2019", "logreg", "random_forest", "gboost"]
    for name in order:
        if verbose:
            print(f"  training {name} ...")
        if name == "sendek2019":
            model, params, impl = SendekPublished(), {}, "fixed coefficients"
        else:
            spec = specs[name]
            model, params = _tune(spec, Xtr, ytr, gtr)
            model = clone(model).fit(Xtr, ytr)
            impl = spec.get("impl", type(model[-1] if isinstance(model, Pipeline) else model).__name__)
        fitted[name] = model
        p_te = model.predict_proba(Xte)[:, 1]
        p_tr = model.predict_proba(Xtr)[:, 1]
        cv_train = _grouped_cv_auc(model if name != "sendek2019" else SendekPublished(), Xtr, ytr, gtr)
        lofo_all = _grouped_cv_auc(model if name != "sendek2019" else SendekPublished(), Xall, yall, gall, n_splits=10)
        entry = {
            "label": MODEL_LABELS[name], "implementation": impl, "params": params,
            "test": _metrics(yte, p_te), "train": {k: v for k, v in _metrics(ytr, p_tr).items() if k in ("roc_auc", "f1", "accuracy")},
            "test_auc_ci95": _bootstrap_auc(yte, p_te),
            "cv_train_family_grouped_auc": cv_train["oof_auc"],
            "cv_all_leave_families_out_auc": lofo_all["oof_auc"],
            "cv_all_fold_auc": [lofo_all["fold_auc_mean"], lofo_all["fold_auc_std"]],
            "cv_all_metrics": {k: v for k, v in _metrics(yall, lofo_all["oof"]).items()
                               if k not in ("roc_curve", "pr_curve")},
            "cv_all_roc_curve": _metrics(yall, lofo_all["oof"])["roc_curve"],
            "test_predictions": [{"ID": i, "formula": f, "y": int(y), "p": round(float(p), 4)}
                                 for i, f, y, p in zip(te["ID"], te["formula"], yte, p_te)],
        }
        report["models"][name] = entry

    # interpretability of the retrained logistic model, in raw feature units so
    # it can be compared term-by-term with Sendek's published equation
    lr = fitted["logreg"]
    sc, clf = lr.named_steps["scale"], lr.named_steps["clf"]
    coef_std = clf.coef_[0]
    coef_raw = coef_std / sc.scale_
    intercept_raw = float(clf.intercept_[0] - np.sum(coef_std * sc.mean_ / sc.scale_))
    report["logreg_equation"] = {
        "coef_standardized": dict(zip(FEATURES, map(float, coef_std))),
        "coef_raw": dict(zip(FEATURES, map(float, coef_raw))),
        "intercept_raw": intercept_raw,
        "sendek2019": {**SENDEK_2019_COEFS, "intercept": SENDEK_2019_INTERCEPT},
    }
    # global importance for the tree models (impurity) + permutation for all
    from sklearn.inspection import permutation_importance

    imp = {}
    for name in ("logreg", "random_forest", "gboost"):
        r = permutation_importance(fitted[name], Xte, yte, scoring="roc_auc", n_repeats=30, random_state=RANDOM_SEED)
        imp[name] = {k: [float(m), float(s)] for k, m, s in zip(FEATURES, r.importances_mean, r.importances_std)}
    report["permutation_importance_test_auc"] = imp

    stats = {k: {"mean": float(tr[k].mean()), "std": float(tr[k].std()), "min": float(tr[k].min()),
                 "max": float(tr[k].max()), "p01": float(tr[k].quantile(0.01)), "p99": float(tr[k].quantile(0.99))}
             for k in FEATURES}
    # rank models by grouped CV on train (the test set is NOT used to choose)
    report["selection_note"] = ("The ranking model is logistic regression by design (base-paper model family, "
                                "interpretable). Test metrics of all models are reported for comparison only.")
    return TrainedBundle(models=fitted, feature_stats=stats, report=report, train_X=Xtr)


def _sk_version() -> str:
    import sklearn

    return sklearn.__version__


def save_bundle(b: TrainedBundle, path: Path = MODEL_FILE) -> None:
    joblib.dump({"models": b.models, "feature_stats": b.feature_stats, "primary": b.primary, "train_X": b.train_X,
                 "sklearn_version": _sk_version()}, path)
    with open(REPORT_FILE, "w") as fh:
        json.dump(b.report, fh, indent=1)


def get_bundle(verbose: bool = True) -> TrainedBundle:
    """Load the saved models; retrain from cached descriptors when they are
    missing or were pickled by a different scikit-learn version."""
    try:
        if MODEL_FILE.exists():
            d = joblib.load(MODEL_FILE)
            if d.get("sklearn_version") == _sk_version():
                return load_bundle()
            if verbose:
                print(f"Saved models were built with scikit-learn {d.get('sklearn_version')}; "
                      f"retraining for {_sk_version()} (about a minute, once).")
    except Exception as exc:  # noqa: BLE001 - unpickling across versions can fail in many ways
        if verbose:
            print(f"Could not load saved models ({exc}); retraining (about a minute, once).")
    from ..data.obelix import training_table

    b = train_all(training_table())
    save_bundle(b)
    return load_bundle()


def load_bundle(path: Path = MODEL_FILE) -> TrainedBundle:
    d = joblib.load(path)
    with open(REPORT_FILE) as fh:
        report = json.load(fh)
    return TrainedBundle(models=d["models"], feature_stats=d["feature_stats"], report=report,
                         primary=d.get("primary", PRIMARY_MODEL), train_X=d.get("train_X"))
