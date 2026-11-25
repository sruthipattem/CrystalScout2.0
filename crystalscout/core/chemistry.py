"""Element data used by the screening pipeline.

Kept as plain tables so the package runs without pymatgen. Values:

* Pauling electronegativities -- standard tabulation (CRC Handbook).
* Transition metals -- d-block groups 3-12 (incl. Y, La is treated as a
  lanthanide, not a transition metal, matching Materials Project usage).
"""
from __future__ import annotations

from typing import Dict, Iterable, Set

PAULING_EN: Dict[str, float] = {
    "H": 2.20, "He": 0.0, "Li": 0.98, "Be": 1.57, "B": 2.04, "C": 2.55, "N": 3.04,
    "O": 3.44, "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19,
    "S": 2.58, "Cl": 3.16, "K": 0.82, "Ca": 1.00, "Sc": 1.36, "Ti": 1.54, "V": 1.63,
    "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91, "Cu": 1.90, "Zn": 1.65,
    "Ga": 1.81, "Ge": 2.01, "As": 2.18, "Se": 2.55, "Br": 2.96, "Kr": 3.00, "Rb": 0.82,
    "Sr": 0.95, "Y": 1.22, "Zr": 1.33, "Nb": 1.60, "Mo": 2.16, "Tc": 1.90, "Ru": 2.20,
    "Rh": 2.28, "Pd": 2.20, "Ag": 1.93, "Cd": 1.69, "In": 1.78, "Sn": 1.96, "Sb": 2.05,
    "Te": 2.10, "I": 2.66, "Xe": 2.60, "Cs": 0.79, "Ba": 0.89, "La": 1.10, "Ce": 1.12,
    "Pr": 1.13, "Nd": 1.14, "Pm": 1.13, "Sm": 1.17, "Eu": 1.20, "Gd": 1.20, "Tb": 1.10,
    "Dy": 1.22, "Ho": 1.23, "Er": 1.24, "Tm": 1.25, "Yb": 1.10, "Lu": 1.27, "Hf": 1.30,
    "Ta": 1.50, "W": 2.36, "Re": 1.90, "Os": 2.20, "Ir": 2.20, "Pt": 2.28, "Au": 2.54,
    "Hg": 2.00, "Tl": 1.62, "Pb": 2.33, "Bi": 2.02, "Po": 2.00, "At": 2.20, "Fr": 0.70,
    "Ra": 0.90, "Ac": 1.10, "Th": 1.30, "Pa": 1.50, "U": 1.38, "Np": 1.36, "Pu": 1.28,
}

TRANSITION_METALS: Set[str] = set(
    "Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Y Zr Nb Mo Tc Ru Rh Pd Ag Cd "
    "Hf Ta W Re Os Ir Pt Au Hg "
    "Rf Db Sg Bh Hs Mt Ds Rg Cn".split()
)

# Elements that act as anions in essentially every Li solid electrolyte.
CONVENTIONAL_ANIONS: Set[str] = set("N O F S Cl Se Br Te I".split())

ALL_ELEMENTS: Set[str] = set(PAULING_EN)


def electronegativity(symbol: str) -> float:
    return PAULING_EN.get(symbol, 1.8)


def bond_ionicity(a: str, b: str) -> float:
    """Pauling fractional ionic character: 1 - exp(-(dchi)^2 / 4)."""
    import math

    d = electronegativity(a) - electronegativity(b)
    return 1.0 - math.exp(-0.25 * d * d)


def identify_anions(elements: Iterable[str]) -> Set[str]:
    """Return the set of anion species for a Li compound.

    Rule (documented in docs/METHODS.md):
      1. Any conventional anion present (N, O, F, S, Cl, Se, Br, Te, I) is an
         anion. H is then treated as a cation (hydroxides, amides, ammine
         ligands).
      2. If none is present (e.g. LiBH4, Li3P, Li3Sb), the most electronegative
         non-Li element is the anion.
    """
    els = {e for e in elements}
    found = els & CONVENTIONAL_ANIONS
    if found:
        return found
    others = [e for e in els if e != "Li"]
    if not others:
        return set()
    return {max(others, key=electronegativity)}


def has_transition_metal(elements: Iterable[str]) -> bool:
    return bool(set(elements) & TRANSITION_METALS)
