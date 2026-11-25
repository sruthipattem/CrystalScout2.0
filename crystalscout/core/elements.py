"""Periodic-table data for composition features (no external packages).

Atomic numbers and standard atomic weights (IUPAC) are tabulated; period,
group, block and valence-electron counts are derived from the Madelung
(aufbau) filling order, which is exact for the quantities used here except a
handful of well-known d/f-shell exceptions (Cr, Cu, ...). Those exceptions
change a valence count by one electron and are deliberately not special-cased
so that features stay a smooth function of Z.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

from .chemistry import PAULING_EN

SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
    "Fr Ra Ac Th Pa U Np Pu"
).split()

MASSES = [
    1.008, 4.0026, 6.94, 9.0122, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180, 22.990, 24.305, 26.982,
    28.085, 30.974, 32.06, 35.45, 39.948, 39.098, 40.078, 44.956, 47.867, 50.942, 51.996, 54.938, 55.845,
    58.933, 58.693, 63.546, 65.38, 69.723, 72.630, 74.922, 78.971, 79.904, 83.798, 85.468, 87.62, 88.906,
    91.224, 92.906, 95.95, 98.0, 101.07, 102.91, 106.42, 107.87, 112.41, 114.82, 118.71, 121.76, 127.60,
    126.90, 131.29, 132.91, 137.33, 138.91, 140.12, 140.91, 144.24, 145.0, 150.36, 151.96, 157.25, 158.93,
    162.50, 164.93, 167.26, 168.93, 173.05, 174.97, 178.49, 180.95, 183.84, 186.21, 190.23, 192.22, 195.08,
    196.97, 200.59, 204.38, 207.2, 208.98, 209.0, 210.0, 222.0, 223.0, 226.0, 227.0, 232.04, 231.04, 238.03,
    237.0, 244.0,
]

Z_OF: Dict[str, int] = {s: i + 1 for i, s in enumerate(SYMBOLS)}
MASS: Dict[str, float] = dict(zip(SYMBOLS, MASSES))

_ORDER = [(1, "s"), (2, "s"), (2, "p"), (3, "s"), (3, "p"), (4, "s"), (3, "d"), (4, "p"), (5, "s"), (4, "d"),
          (5, "p"), (6, "s"), (4, "f"), (5, "d"), (6, "p"), (7, "s"), (5, "f"), (6, "d"), (7, "p")]
_CAP = {"s": 2, "p": 6, "d": 10, "f": 14}


@lru_cache(maxsize=None)
def _config(z: int) -> Tuple[Tuple[int, str, int], ...]:
    left, out = z, []
    for n, l in _ORDER:
        if left <= 0:
            break
        k = min(_CAP[l], left)
        out.append((n, l, k))
        left -= k
    return tuple(out)


@lru_cache(maxsize=None)
def element_props(symbol: str) -> Dict[str, float]:
    """Derived properties of one element (unknown symbols fall back to Li-like zeros)."""
    z = Z_OF.get(symbol)
    if z is None:
        return {k: 0.0 for k in PROPS}
    cfg = _config(z)
    period = max(n for n, l, _ in cfg if l in "sp")
    ns = sum(k for n, l, k in cfg if l == "s" and n == period)
    np_ = sum(k for n, l, k in cfg if l == "p" and n == period)
    nd = sum(k for n, l, k in cfg if l == "d" and n == period - 1)
    nf = sum(k for n, l, k in cfg if l == "f" and n == period - 2)
    last_l = cfg[-1][1]
    if z == 2:
        group = 18
    elif last_l == "s":
        group = ns
    elif last_l == "p":
        group = 10 + ns + np_
    elif last_l == "d":
        group = ns + nd
    else:  # lanthanides / actinides
        group = 3
    block = {"s": 1, "p": 2, "d": 3, "f": 4}[last_l if z != 2 else "s"]
    return {
        "Z": float(z), "period": float(period), "group": float(group), "mass": MASS[symbol],
        "electronegativity": PAULING_EN.get(symbol, 1.8), "ns": float(ns), "np": float(np_), "nd": float(nd),
        "nf": float(nf), "nvalence": float(ns + np_ + nd + nf), "block": float(block),
    }


PROPS = ["Z", "period", "group", "mass", "electronegativity", "ns", "np", "nd", "nf", "nvalence", "block"]
