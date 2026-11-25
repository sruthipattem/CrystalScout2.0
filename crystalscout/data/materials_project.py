"""Materials Project access (Stage A candidate pool).

Two sources, same output schema:

1. **Live API** -- ``MaterialsProjectClient`` queries
   ``https://api.materialsproject.org/materials/summary/`` with your API key
   (free at https://next-gen.materialsproject.org/api). If the official
   ``mp-api`` package is installed it is used; otherwise a plain-HTTPS client
   with no extra dependency is used. Filters are pushed to the server so only
   Stage-A survivors (plus their structures) are downloaded.

2. **Bundled offline snapshot** -- 19,806 Li-containing Materials Project
   entries (MP API extraction of 2026-07-20, redistributed under CC BY 4.0 via
   the Scandium-Dataset compilation). It lets the whole pipeline, the tests
   and the web app run without a key or network. Structures are bundled for
   the 1,648 transition-metal-free entries with band gap >= 0.5 eV and
   E_hull <= 0.1 eV/atom, which covers every Stage-A threshold the UI allows.
"""
from __future__ import annotations

import gzip
import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import pandas as pd

from ..config import MP_DIR
from ..core.chemistry import TRANSITION_METALS

SNAPSHOT_CSV = MP_DIR / "mp_li_snapshot.csv.gz"
POOL_STRUCTURES = MP_DIR / "mp_pool_structures.json.gz"
SNAPSHOT_INFO = {
    "source": "Materials Project (via Scandium-Dataset v1.0.0 compilation, CC BY 4.0)",
    "extraction_date": "2026-07-20",
    "n_entries": None,
}
API_URL = "https://api.materialsproject.org/materials/summary/"
FIELDS = ["material_id", "formula_pretty", "band_gap", "energy_above_hull", "formation_energy_per_atom",
          "symmetry", "nsites", "elements", "structure", "is_stable", "theoretical"]


class MPError(RuntimeError):
    pass


# --------------------------------------------------------------- snapshot
def load_snapshot() -> pd.DataFrame:
    df = pd.read_csv(SNAPSHOT_CSV)
    df["elements"] = df["elements"].fillna("")
    SNAPSHOT_INFO["n_entries"] = len(df)
    return df


_POOL_CACHE: Dict[str, Dict] = {}


def load_pool_structures() -> Dict[str, Dict]:
    if not _POOL_CACHE:
        with gzip.open(POOL_STRUCTURES, "rt") as fh:
            _POOL_CACHE.update(json.load(fh))
    return _POOL_CACHE


# ------------------------------------------------------------------- live
def get_api_key(explicit: Optional[str] = None) -> Optional[str]:
    return explicit or os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")


class MaterialsProjectClient:
    def __init__(self, api_key: Optional[str] = None, timeout: float = 60.0):
        self.api_key = get_api_key(api_key)
        if not self.api_key:
            raise MPError("No Materials Project API key. Set MP_API_KEY or pass api_key=...")
        self.timeout = timeout

    def _get(self, params: Dict) -> Dict:
        url = API_URL + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"X-API-KEY": self.api_key, "accept": "application/json",
                                                   "user-agent": "CrystalScout/2.0"})
        ctx = ssl.create_default_context()
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout, context=ctx) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
                if e.code in (401, 403):
                    raise MPError("Materials Project rejected the API key (HTTP %d)." % e.code) from e
                if attempt == 2:
                    raise MPError(f"Materials Project API error: HTTP {e.code}") from e
            except urllib.error.URLError as e:  # type: ignore[attr-defined]
                if attempt == 2:
                    raise MPError(f"Could not reach the Materials Project API: {e.reason}") from e
            time.sleep(1.5 * (attempt + 1))
        raise MPError("unreachable")

    def stage_a_query(self, band_gap_min: float = 1.0, ehull_max: float = 0.0, exclude_tm: bool = True,
                      limit: Optional[int] = None, progress: Optional[Callable[[int, int], None]] = None) -> List[Dict]:
        """Server-side Stage A. Returns list of dicts with the snapshot schema
        plus ``structure`` (pymatgen dict)."""
        try:  # official client when available
            from mp_api.client import MPRester  # type: ignore

            with MPRester(self.api_key) as mpr:
                docs = mpr.materials.summary.search(
                    elements=["Li"], band_gap=(band_gap_min, None), energy_above_hull=(0, ehull_max),
                    exclude_elements=sorted(TRANSITION_METALS) if exclude_tm else None,
                    fields=FIELDS,
                )
            out = [self._normalise(d.model_dump() if hasattr(d, "model_dump") else dict(d)) for d in docs]
            return out[:limit] if limit else out
        except ImportError:
            pass
        params = {
            "elements": "Li",
            "band_gap_min": band_gap_min,
            "energy_above_hull_min": 0,
            "energy_above_hull_max": ehull_max + 1e-9,
            "deprecated": "false",
            "_fields": ",".join(FIELDS),
            "_limit": 500,
        }
        if exclude_tm:
            params["exclude_elements"] = ",".join(sorted(TRANSITION_METALS))
        out: List[Dict] = []
        skip = 0
        total = None
        while True:
            params["_skip"] = skip
            page = self._get(params)
            data = page.get("data", [])
            total = page.get("meta", {}).get("total_doc", total)
            out.extend(self._normalise(d) for d in data)
            skip += len(data)
            if progress:
                progress(len(out), total or len(out))
            if not data or (total is not None and skip >= total) or (limit and len(out) >= limit):
                break
        return out[:limit] if limit else out

    @staticmethod
    def _normalise(d: Dict) -> Dict:
        sym = d.get("symmetry") or {}
        if hasattr(sym, "model_dump"):
            sym = sym.model_dump()
        st = d.get("structure")
        if hasattr(st, "as_dict"):
            st = st.as_dict()
        els = d.get("elements") or []
        els = [str(getattr(e, "value", e)) for e in els]
        return {
            "material_id": str(d.get("material_id")),
            "formula": d.get("formula_pretty"),
            "band_gap": d.get("band_gap"),
            "energy_above_hull": d.get("energy_above_hull"),
            "formation_energy_per_atom": d.get("formation_energy_per_atom"),
            "space_group": (sym or {}).get("symbol"),
            "nsites": d.get("nsites"),
            "elements": " ".join(els),
            "theoretical": d.get("theoretical"),
            "structure": st,
        }


def snapshot_record_structure(material_id: str) -> Optional[Dict]:
    return load_pool_structures().get(material_id)
