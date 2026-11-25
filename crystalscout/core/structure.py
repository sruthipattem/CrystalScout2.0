"""A small, dependency-light periodic crystal structure.

Why not pymatgen? pymatgen is excellent but heavy (compiled extensions,
~200 MB of dependencies). CrystalScout only needs: a lattice, sites that may be
partially occupied, and periodic neighbor search. Implementing that directly
keeps the package installable anywhere (``pip install numpy pandas scikit-learn
flask``) and makes every geometric step inspectable.

Conventions
-----------
* ``lattice`` is a 3x3 matrix whose *rows* are the a, b, c vectors (Angstrom),
  identical to pymatgen's ``Lattice.matrix``.
* Each site carries ``species``: a dict ``{element: occupancy}``. A fully
  ordered site is ``{"Li": 1.0}``; a disordered site may be
  ``{"Li": 0.5}`` (vacancy-rich) or ``{"S": 0.6, "Cl": 0.4}`` (mixed).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def lattice_from_parameters(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Standard crystallographic setting: a along x, b in the xy plane."""
    al, be, ga = (math.radians(x) for x in (alpha, beta, gamma))
    ax = np.array([a, 0.0, 0.0])
    bx = np.array([b * math.cos(ga), b * math.sin(ga), 0.0])
    cx_ = c * math.cos(be)
    cy_ = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz2 = c * c - cx_ * cx_ - cy_ * cy_
    cz_ = math.sqrt(max(cz2, 1e-12))
    return np.array([ax, bx, [cx_, cy_, cz_]])


@dataclass
class Site:
    frac: np.ndarray
    species: Dict[str, float]
    label: str = ""

    @property
    def occupancy(self) -> float:
        return float(sum(self.species.values()))

    def occ(self, element: str) -> float:
        return float(self.species.get(element, 0.0))

    @property
    def major(self) -> str:
        return max(self.species, key=self.species.get)


@dataclass
class Structure:
    lattice: np.ndarray
    sites: List[Site]
    name: str = ""
    meta: Dict = field(default_factory=dict)

    # ------------------------------------------------------------------ basics
    def __len__(self) -> int:
        return len(self.sites)

    def __iter__(self):
        return iter(self.sites)

    @property
    def frac_coords(self) -> np.ndarray:
        return np.array([s.frac for s in self.sites], dtype=float).reshape(-1, 3)

    @property
    def cart_coords(self) -> np.ndarray:
        return self.frac_coords @ self.lattice

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    @property
    def lattice_parameters(self) -> Tuple[float, float, float, float, float, float]:
        a, b, c = (np.linalg.norm(v) for v in self.lattice)
        def ang(u, v):
            return math.degrees(math.acos(np.clip(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
        L = self.lattice
        return (a, b, c, ang(L[1], L[2]), ang(L[0], L[2]), ang(L[0], L[1]))

    @property
    def elements(self) -> List[str]:
        out = []
        for s in self.sites:
            for e, o in s.species.items():
                if o > 0 and e not in out:
                    out.append(e)
        return out

    def composition(self) -> Dict[str, float]:
        comp: Dict[str, float] = {}
        for s in self.sites:
            for e, o in s.species.items():
                comp[e] = comp.get(e, 0.0) + o
        return comp

    def formula(self) -> str:
        comp = self.composition()
        # order: Li first, then by electronegativity (anions last), like MP
        from .chemistry import electronegativity

        keys = sorted(comp, key=lambda e: (e != "Li", electronegativity(e), e))
        vals = np.array([comp[k] for k in keys])
        # reduce by gcd when integral
        if np.allclose(vals, np.round(vals), atol=1e-3):
            g = 0
            for v in np.round(vals).astype(int):
                g = math.gcd(g, int(v))
            if g > 1:
                vals = vals / g
        parts = []
        for k, v in zip(keys, vals):
            if abs(v - 1) < 1e-3:
                parts.append(k)
            elif abs(v - round(v)) < 1e-3:
                parts.append(f"{k}{int(round(v))}")
            else:
                parts.append(f"{k}{v:.3g}")
        return "".join(parts)

    @property
    def is_ordered(self) -> bool:
        return all(len(s.species) == 1 and abs(s.occupancy - 1) < 1e-6 for s in self.sites)

    # --------------------------------------------------------- neighbor search
    def _image_offsets(self, cutoff: float) -> np.ndarray:
        inv = np.linalg.inv(self.lattice)
        # spacing between lattice planes = 1 / |reciprocal vector|
        recip_norms = np.linalg.norm(inv, axis=0)
        n = [int(math.ceil(cutoff * rn)) + 1 for rn in recip_norms]
        rng = [np.arange(-k, k + 1) for k in n]
        return np.array(np.meshgrid(*rng, indexing="ij")).reshape(3, -1).T.astype(float)

    def neighbors(self, cutoff: float = 6.0) -> List[Tuple[np.ndarray, np.ndarray]]:
        """For every site i return ``(j_indices, distances)`` of all periodic
        images of all sites within ``cutoff`` (excluding i's own zero image).
        Cached per cutoff."""
        cache = self.__dict__.setdefault("_nn_cache", {})
        key = round(cutoff, 4)
        if key in cache:
            return cache[key]
        f = self.frac_coords
        L = self.lattice
        offsets = self._image_offsets(cutoff)
        n = len(f)
        out = []
        for i in range(n):
            d = f - f[i]
            d -= np.round(d)  # minimum-image start point
            # (n, n_off, 3)
            shifted = d[:, None, :] + offsets[None, :, :]
            cart = shifted @ L
            dist = np.sqrt(np.einsum("ijk,ijk->ij", cart, cart))
            mask = dist <= cutoff
            mask[i] &= dist[i] > 1e-6
            jj, _ = np.nonzero(mask)
            out.append((jj, dist[mask]))
        cache[key] = out
        return out

    # -------------------------------------------------------------- converters
    @classmethod
    def from_pymatgen_dict(cls, d, name: str = "") -> "Structure":
        """Build from a pymatgen ``Structure.as_dict()`` payload (this is what
        the Materials Project API returns in the ``structure`` field)."""
        if isinstance(d, str):
            d = json.loads(d)
        lat = np.array(d["lattice"]["matrix"], dtype=float)
        sites = []
        for s in d["sites"]:
            sp = {}
            for x in s["species"]:
                el = x.get("element")
                if el is None:  # e.g. {"element": "Li", "oxidation_state": 1}
                    continue
                sp[el] = sp.get(el, 0.0) + float(x.get("occu", 1.0))
            sites.append(Site(np.array(s["abc"], dtype=float) % 1.0, sp, s.get("label", "")))
        return cls(lat, sites, name=name)

    def as_dict(self) -> Dict:
        return {
            "@module": "pymatgen.core.structure",
            "@class": "Structure",
            "lattice": {"matrix": self.lattice.tolist()},
            "sites": [
                {
                    "species": [{"element": e, "occu": o} for e, o in s.species.items()],
                    "abc": [float(x) for x in s.frac],
                    "label": s.label or s.major,
                }
                for s in self.sites
            ],
        }

    def to_cif(self) -> str:
        """Write a P1 CIF (all sites explicit, occupancies preserved)."""
        a, b, c, al, be, ga = self.lattice_parameters
        lines = [
            "# written by CrystalScout",
            f"data_{self.formula() or 'structure'}",
            "_symmetry_space_group_name_H-M   'P 1'",
            f"_cell_length_a   {a:.6f}", f"_cell_length_b   {b:.6f}", f"_cell_length_c   {c:.6f}",
            f"_cell_angle_alpha   {al:.6f}", f"_cell_angle_beta   {be:.6f}", f"_cell_angle_gamma   {ga:.6f}",
            "_symmetry_Int_Tables_number   1",
            "loop_", " _symmetry_equiv_pos_site_id", " _symmetry_equiv_pos_as_xyz", "  1  'x, y, z'",
            "loop_", " _atom_site_type_symbol", " _atom_site_label",
            " _atom_site_fract_x", " _atom_site_fract_y", " _atom_site_fract_z", " _atom_site_occupancy",
        ]
        k = 0
        for s in self.sites:
            for e, o in s.species.items():
                lines.append(f"  {e}  {e}{k}  {s.frac[0]:.6f}  {s.frac[1]:.6f}  {s.frac[2]:.6f}  {o:.4f}")
                k += 1
        return "\n".join(lines) + "\n"

    def viewer_payload(self, max_atoms: int = 600) -> Dict:
        """Compact JSON for the in-browser 3D viewer."""
        cart = self.cart_coords
        atoms = []
        for s, xyz in zip(self.sites[:max_atoms], cart[:max_atoms]):
            atoms.append({"el": s.major, "occ": round(s.occupancy, 3), "xyz": [round(float(v), 3) for v in xyz]})
        return {"lattice": self.lattice.round(4).tolist(), "atoms": atoms, "truncated": len(self.sites) > max_atoms,
                "elements": sorted({e for s in self.sites for e, o in s.species.items() if o > 1e-3})}
