# Data sources and provenance

## OBELiX (training labels)

- **Source:** github.com/NRC-Mila/OBELiX (`data/downloads/all.csv`, `train.csv`, `test.csv`, `all_cifs.tar.gz`). Licence CC BY 4.0, included as `data/obelix/LICENSE_CC-BY-4.0.txt`.
- **Contents:** 599 entries with room-temperature conductivity; 321 have CIFs.

**Caveats, each turned into a per-entry flag:**

| Caveat | Entries | How CrystalScout handles it |
|---|---|---|
| Conductivity reported as an upper bound ("<1E-10") | 35 | Label 0 |
| CIF taken from a closely matching compound (`close match = Yes`) | 61 | Flagged |
| Disordered or partially occupied CIF | 243 of 314 usable | Occupancy-weighted descriptors |
| CIF Li content differs from the reported composition by more than 10 % | 13 | Flagged |
| CIF contains no Li sites | 7 | Excluded from training |
| CIF coordinates randomly perturbed by the authors (ICSD licensing) | All CIFs | Descriptors are bond-averaged and robust to this |

## Materials Project snapshot (candidate pool)

- **Source:** Materials Project API extraction of 2026-07-20, redistributed in the Scandium-Dataset v1.0.0 compilation (github.com/ScandiumLabs-in/Scandium-Dataset, partition `source=mp`). MP data is CC BY 4.0.
- **`mp_li_snapshot.csv.gz`:** all 19,806 Li-containing entries, with material_id, formula, band gap, E_hull, formation energy, space group, nsites and elements.
- **`mp_pool_structures.json.gz`:** structures for the 1,648 transition-metal-free entries with band gap ≥ 0.5 eV and E_hull ≤ 0.1 eV/atom.
- **Rebuild:** `scripts/build_mp_snapshot.py`.
- **Spot checks against known MP values all match:**

| Material | MP ID | Band gap | E_hull |
|---|---|---|---|
| LiF | mp-1138 | 8.72 eV | 0 |
| Li₃PS₄ | mp-985583 | 2.81 eV | 0 |
| Li₅B₇S₁₃ | mp-532413 | 3.55 eV | 0 |

For the newest data, use the live mode. It queries `api.materialsproject.org/materials/summary` with your key.

## DFT-MD benchmark

`data/benchmarks/sendek2019_dftmd.csv` is transcribed from Sendek et al., *Chem. Mater.* 2019, Tables 1 and 2 (40 materials). Verdicts follow the paper's own classification: superionic, marginal or slow.
