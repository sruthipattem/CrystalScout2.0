"""Rebuild the bundled Materials Project snapshot.

Input: a JSON-lines file of Li-containing MP entries (fields: source_id,
formula, band_gap, energy_above_hull, formation_energy_per_atom,
space_group_symbol, nsites, structure_json). We produced it from the
Scandium-Dataset v1.0.0 parquet partition ``source=mp`` (MP extraction
2026-07-20). You normally never need to run this: use ``crystalscout stage-a
--live`` to pull fresh data straight from the Materials Project API instead.

Usage:  python scripts/build_mp_snapshot.py path/to/mp_li.jsonl
"""
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crystalscout.config import MP_DIR  # noqa: E402
from crystalscout.core.chemistry import TRANSITION_METALS  # noqa: E402

POOL_GAP_MIN, POOL_EHULL_MAX = 0.5, 0.1


def main(src: str) -> None:
    MP_DIR.mkdir(parents=True, exist_ok=True)
    rows, pool = [], {}
    with open(src) as fh:
        for line in fh:
            r = json.loads(line)
            st = json.loads(r["structure_json"])
            els = sorted({sp["element"] for s in st["sites"] for sp in s["species"]})
            if "Li" not in els:
                continue
            rows.append({
                "material_id": r["source_id"], "formula": r["formula"], "band_gap": round(r["band_gap"], 4),
                "energy_above_hull": round(r["energy_above_hull"], 6),
                "formation_energy_per_atom": round(r["formation_energy_per_atom"], 5),
                "space_group": r.get("space_group_symbol"), "nsites": r["nsites"], "elements": " ".join(els),
            })
            if (not set(els) & TRANSITION_METALS) and r["band_gap"] >= POOL_GAP_MIN and r["energy_above_hull"] <= POOL_EHULL_MAX:
                st.pop("properties", None)
                for s in st["sites"]:
                    s.pop("properties", None)
                    s.pop("xyz", None)
                    s["abc"] = [round(x, 6) for x in s["abc"]]
                st["lattice"] = {"matrix": [[round(x, 6) for x in v] for v in st["lattice"]["matrix"]]}
                pool[r["source_id"]] = st
    df = pd.DataFrame(rows).sort_values("material_id")
    df.to_csv(MP_DIR / "mp_li_snapshot.csv.gz", index=False, compression="gzip")
    with gzip.open(MP_DIR / "mp_pool_structures.json.gz", "wt") as fh:
        json.dump(pool, fh, separators=(",", ":"))
    print(f"snapshot: {len(df)} Li-containing entries; structure pool: {len(pool)}")


if __name__ == "__main__":
    main(sys.argv[1])
