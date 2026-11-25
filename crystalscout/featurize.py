"""Shared feature extractor: the *same code path* for Materials Project
candidates, OBELiX training CIFs and user uploads (deck slide 8)."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .config import PROCESSED_DIR
from .core.cif import CIFError, parse_cif
from .core.descriptors import FEATURES, DescriptorError, compute_descriptors
from .core.structure import Structure

POOL_FEATURES = PROCESSED_DIR / "mp_pool_features.csv"


def featurize_structure(st: Structure) -> Dict:
    d = compute_descriptors(st)
    out = {k: getattr(d, k) for k in FEATURES}
    out.update(n_sites=d.n_sites, n_li_sites=d.n_li_sites, anions=" ".join(d.anions),
               framework_cations=" ".join(d.framework_cations), disordered=d.disordered,
               descriptor_warnings="; ".join(d.warnings))
    return out


def featurize_records(records: Iterable[Tuple[str, Dict]], progress=None) -> pd.DataFrame:
    """records: iterable of (material_id, pymatgen-structure-dict)."""
    rows = []
    records = list(records)
    for k, (mid, sd) in enumerate(records):
        row = {"material_id": mid}
        try:
            row.update(featurize_structure(Structure.from_pymatgen_dict(sd, name=mid)))
            row["feature_error"] = ""
        except (DescriptorError, Exception) as exc:  # noqa: BLE001
            row["feature_error"] = str(exc)
        rows.append(row)
        if progress:
            progress(k + 1, len(records))
    return pd.DataFrame(rows)


def featurize_cif_text(text: str, name: str = "upload") -> Tuple[Structure, Dict]:
    st = parse_cif(text, name=name)
    return st, featurize_structure(st)


def load_pool_features(rebuild: bool = False) -> pd.DataFrame:
    if POOL_FEATURES.exists() and not rebuild:
        df = pd.read_csv(POOL_FEATURES, keep_default_na=False, na_values=[""])
        for c in ("feature_error", "descriptor_warnings", "anions", "framework_cations"):
            if c in df:
                df[c] = df[c].fillna("")
        return df
    from .data.materials_project import load_pool_structures

    pool = load_pool_structures()
    df = featurize_records(sorted(pool.items()))
    df["feature_error"] = df["feature_error"].fillna("")
    df.to_csv(POOL_FEATURES, index=False)
    return df
