"""OBELiX integration (Stage B training labels).

OBELiX: F. Therrien et al., "OBELiX: A Curated Dataset of Crystal Structures
and Experimentally Measured Ionic Conductivities for Lithium Solid-State
Electrolytes", arXiv:2502.14234 (2025). CC BY 4.0.
https://github.com/NRC-Mila/OBELiX

599 synthesized Li solid electrolytes with room-temperature ionic
conductivity; 321 have a CIF. The dataset ships an official train/test split
(479 / 120) that we keep as the held-out test set.

Documented caveats, each turned into an explicit per-entry flag here:

* ``sigma_upper_bound``  conductivity reported as "<1E-10" (an upper bound).
* ``close_match``        CIF taken from a closely related compound, not the
                          measured one.
* ``disordered``         CIF has partial occupancy / mixed sites.
* ``li_mismatch``        Li content of the CIF differs from the reported
                          composition by more than 10 % (doped / off-
                          stoichiometric phases described by a parent CIF).
* ``no_li_in_cif``       CIF has no Li sites at all -> descriptors impossible,
                          entry excluded from training with that reason.
* coordinates were randomly perturbed by the dataset authors to respect ICSD
  licensing; descriptors are geometric averages and are robust to this, but
  it is recorded in docs/DATA.md.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import OBELIX_DIR, PROCESSED_DIR, SUPERIONIC_THRESHOLD
from ..core.cif import read_cif
from ..core.descriptors import FEATURES, DescriptorError, compute_descriptors

FEATURE_CACHE = PROCESSED_DIR / "obelix_features.csv"


def _parse_sigma(v) -> Tuple[float, bool]:
    s = str(v).strip()
    upper = s.startswith("<")
    try:
        return float(s.lstrip("<")), upper
    except ValueError:
        return float("nan"), upper


def _composition(formula: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for el, n in re.findall(r"([A-Z][a-z]?)([\d.]*)", str(formula)):
        out[el] = out.get(el, 0.0) + (float(n) if n else 1.0)
    return out


def load_obelix(obelix_dir: Path = OBELIX_DIR) -> pd.DataFrame:
    """The full table with labels, split and CIF path, one row per entry."""
    df = pd.read_csv(obelix_dir / "all.csv")
    test_ids = set(pd.read_csv(obelix_dir / "test.csv")["ID"])
    sig = df["Ionic conductivity (S cm-1)"].map(_parse_sigma)
    df["sigma"] = [s for s, _ in sig]
    df["sigma_upper_bound"] = [u for _, u in sig]
    df["log10_sigma"] = np.log10(df["sigma"].clip(lower=1e-14))
    df["label"] = (df["sigma"] >= SUPERIONIC_THRESHOLD).astype(int)
    df["split"] = np.where(df["ID"].isin(test_ids), "test", "train")
    cif_dir = obelix_dir / "cifs"
    df["cif_path"] = [str(cif_dir / f"{i}.cif") if (cif_dir / f"{i}.cif").exists() else "" for i in df["ID"]]
    df["has_cif"] = df["cif_path"] != ""
    df["close_match"] = df["close match"].astype(str).str.lower().eq("yes")
    df["formula"] = df["Reduced Composition"]
    df["family"] = df["Family"].fillna("unknown")
    return df


def build_features(df: Optional[pd.DataFrame] = None, use_cache: bool = True, verbose: bool = False) -> pd.DataFrame:
    """Compute the 5 descriptors for every OBELiX entry that has a CIF.

    Returns the OBELiX table restricted to CIF entries, with descriptor
    columns plus caveat flags and an ``excluded_reason`` column ("" when the
    entry is usable for training).
    """
    if df is None:
        df = load_obelix()
    if use_cache and FEATURE_CACHE.exists():
        cached = pd.read_csv(FEATURE_CACHE, keep_default_na=False, na_values=[""])
        cached["excluded_reason"] = cached["excluded_reason"].fillna("")
        return cached
    rows = []
    for _, r in df[df["has_cif"]].iterrows():
        rec = {"ID": r["ID"]}
        try:
            st = read_cif(r["cif_path"], name=r["ID"])
            comp = st.composition()
            rep = _composition(r["True Composition"])
            li_cif, li_rep = comp.get("Li", 0.0), rep.get("Li", 0.0)
            # compare Li fraction (robust to different Z)
            f_cif = li_cif / max(sum(comp.values()), 1e-9)
            f_rep = li_rep / max(sum(rep.values()), 1e-9)
            rec["li_mismatch"] = bool(f_rep > 0 and abs(f_cif - f_rep) / f_rep > 0.10)
            rec["no_li_in_cif"] = li_cif <= 1e-6
            rec["disordered"] = not st.is_ordered
            rec["n_sites"] = len(st)
            if rec["no_li_in_cif"]:
                rec["excluded_reason"] = "CIF contains no Li sites"
            else:
                d = compute_descriptors(st)
                for k in FEATURES:
                    rec[k] = getattr(d, k)
                rec["anions"] = " ".join(d.anions)
                rec["excluded_reason"] = ""
        except (DescriptorError, Exception) as exc:  # noqa: BLE001 - record any parse failure
            rec["excluded_reason"] = f"descriptor failure: {exc}"
        rows.append(rec)
        if verbose and len(rows) % 50 == 0:
            print(f"  {len(rows)} CIFs processed")
    feats = pd.DataFrame(rows)
    keep = ["ID", "formula", "family", "sigma", "sigma_upper_bound", "log10_sigma", "label", "split", "close_match",
            "Space group", "DOI"]
    out = df[keep].merge(feats, on="ID", how="inner")
    out["excluded_reason"] = out["excluded_reason"].fillna("")
    out.to_csv(FEATURE_CACHE, index=False)
    return out


def training_table(feats: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    if feats is None:
        feats = build_features()
    ok = feats[feats["excluded_reason"] == ""].copy()
    return ok.dropna(subset=FEATURES)
