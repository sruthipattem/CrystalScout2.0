"""Minimal, robust CIF reader.

Handles what real-world Li-electrolyte CIFs (ICSD exports, pymatgen output,
the OBELiX dataset, Materials Project downloads) actually contain:

* quoted values and ``loop_`` blocks
* symmetry operators under ``_symmetry_equiv_pos_as_xyz`` *or*
  ``_space_group_symop_operation_xyz``
* coordinates with standard uncertainties, e.g. ``0.1234(5)``
* partial occupancy and mixed sites (several rows on the same position)
* type symbols with charges (``Li1+``, ``O2-``) or labels only (``Li1``)

The asymmetric unit is expanded with every symmetry operator and positions
closer than ``merge_tol`` Angstrom are merged into one site whose species
dictionary holds each element's occupancy.
"""
from __future__ import annotations

import re
import shlex
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

from .chemistry import ALL_ELEMENTS
from .structure import Site, Structure, lattice_from_parameters


class CIFError(ValueError):
    pass


_NUM_RE = re.compile(r"^([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _num(tok: str) -> Optional[float]:
    if tok in (".", "?"):
        return None
    m = _NUM_RE.match(tok.strip())
    return float(m.group(1)) if m else None


def _tokenize(line: str) -> List[str]:
    try:
        return shlex.split(line, posix=True)
    except ValueError:
        return line.split()


def _parse_blocks(text: str) -> Tuple[Dict[str, str], List[Dict[str, List[str]]]]:
    """Return (scalar tags, list of loops) for the FIRST data block."""
    tags: Dict[str, str] = {}
    loops: List[Dict[str, List[str]]] = []
    lines = text.replace("\r", "").split("\n")
    i, n = 0, len(lines)
    seen_data = False
    while i < n:
        raw = lines[i]
        line = raw.strip()
        if not line or line.startswith("#"):
            i += 1
            continue
        if line.lower().startswith("data_"):
            if seen_data:
                break  # only the first structure
            seen_data = True
            i += 1
            continue
        if line.startswith(";"):  # multi-line text field -> skip
            i += 1
            while i < n and not lines[i].startswith(";"):
                i += 1
            i += 1
            continue
        if line.lower() == "loop_":
            i += 1
            keys = []
            while i < n and lines[i].strip().startswith("_"):
                keys.append(lines[i].strip().split()[0].lower())
                i += 1
            values: List[str] = []
            while i < n:
                s = lines[i].strip()
                if not s or s.startswith("#"):
                    i += 1
                    if not s and values:
                        # blank line may end a loop in sloppy files; peek
                        continue
                    continue
                if s.startswith("_") or s.lower() == "loop_" or s.lower().startswith("data_"):
                    break
                if s.startswith(";"):
                    i += 1
                    buf = []
                    while i < n and not lines[i].startswith(";"):
                        buf.append(lines[i])
                        i += 1
                    values.append(" ".join(buf))
                    i += 1
                    continue
                values.extend(_tokenize(s))
                i += 1
            if keys:
                cols: Dict[str, List[str]] = {k: [] for k in keys}
                for r in range(len(values) // len(keys)):
                    for c, k in enumerate(keys):
                        cols[k].append(values[r * len(keys) + c])
                loops.append(cols)
            continue
        if line.startswith("_"):
            parts = line.split(None, 1)
            key = parts[0].lower()
            if len(parts) > 1:
                toks = _tokenize(parts[1])
                tags[key] = toks[0] if toks else parts[1]
            else:  # value on next line(s)
                i += 1
                if i < n and lines[i].startswith(";"):
                    buf = []
                    i += 1
                    while i < n and not lines[i].startswith(";"):
                        buf.append(lines[i])
                        i += 1
                    tags[key] = " ".join(buf)
                elif i < n:
                    toks = _tokenize(lines[i].strip())
                    tags[key] = toks[0] if toks else ""
            i += 1
            continue
        i += 1
    return tags, loops


def parse_symop(op: str) -> Tuple[np.ndarray, np.ndarray]:
    """'-x+1/2, y, z+1/4' -> (3x3 rotation, translation)."""
    op = op.replace(" ", "").lower().strip("'\"")
    parts = op.split(",")
    if len(parts) != 3:
        raise CIFError(f"bad symmetry operator: {op!r}")
    R = np.zeros((3, 3))
    t = np.zeros(3)
    for row, expr in enumerate(parts):
        expr = expr.replace("-", "+-")
        for term in filter(None, expr.split("+")):
            sign = -1.0 if term.startswith("-") else 1.0
            term = term.lstrip("-")
            hit = False
            for col, var in enumerate("xyz"):
                if var in term:
                    coef = term.replace(var, "").replace("*", "")
                    R[row, col] += sign * (float(Fraction(coef)) if coef else 1.0)
                    hit = True
                    break
            if not hit:
                t[row] += sign * float(Fraction(term))
    return R, t


_EL_RE = re.compile(r"^([A-Z][a-z]?)")


def _element(sym: str) -> Optional[str]:
    sym = sym.strip()
    m = _EL_RE.match(sym)
    if not m:
        m = _EL_RE.match(sym.capitalize())
    if not m:
        return None
    el = m.group(1)
    if el not in ALL_ELEMENTS and len(el) == 2 and el[0] in ALL_ELEMENTS:
        el = el[0]
    if el == "D" or el == "T":  # deuterium / tritium
        return "H"
    return el if el in ALL_ELEMENTS else None


def parse_cif(text: str, name: str = "", merge_tol: float = 0.05) -> Structure:
    tags, loops = _parse_blocks(text)

    def tag(*keys):
        for k in keys:
            if k in tags:
                return _num(tags[k])
        return None

    params = [tag("_cell_length_a"), tag("_cell_length_b"), tag("_cell_length_c"),
              tag("_cell_angle_alpha"), tag("_cell_angle_beta"), tag("_cell_angle_gamma")]
    if any(p is None for p in params):
        raise CIFError("missing unit-cell parameters")
    lattice = lattice_from_parameters(*params)

    ops: List[Tuple[np.ndarray, np.ndarray]] = []
    atom_loop = None
    for lp in loops:
        for k in ("_symmetry_equiv_pos_as_xyz", "_space_group_symop_operation_xyz", "_space_group_symop.operation_xyz"):
            if k in lp:
                ops = [parse_symop(o) for o in lp[k]]
        if "_atom_site_fract_x" in lp:
            atom_loop = lp
    warnings = []
    if not ops:
        ops = [(np.eye(3), np.zeros(3))]
        warnings.append("no symmetry operators found; assuming P1")
    if atom_loop is None:
        raise CIFError("no _atom_site_fract_x loop")

    n_rows = len(atom_loop["_atom_site_fract_x"])
    syms = atom_loop.get("_atom_site_type_symbol") or atom_loop.get("_atom_site_label")
    if syms is None:
        raise CIFError("atom sites have neither type_symbol nor label")
    occs = atom_loop.get("_atom_site_occupancy", ["1"] * n_rows)

    cart_tol2 = merge_tol ** 2
    positions: List[np.ndarray] = []
    species: List[Dict[str, float]] = []
    labels: List[str] = []
    L = lattice
    pos_arr = np.zeros((0, 3))

    def find(frac):
        if not len(pos_arr):
            return -1
        d = pos_arr - frac
        d -= np.round(d)
        c = d @ L
        d2 = np.einsum("ij,ij->i", c, c)
        k = int(np.argmin(d2))
        return k if d2[k] < cart_tol2 else -1

    for r in range(n_rows):
        el = _element(syms[r])
        if el is None:
            warnings.append(f"skipped unknown species {syms[r]!r}")
            continue
        xyz = [_num(atom_loop[k][r]) for k in ("_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z")]
        if any(v is None for v in xyz):
            continue
        occ = _num(occs[r]) if r < len(occs) else 1.0
        occ = 1.0 if occ is None else occ
        if occ <= 0:
            continue
        base = np.array(xyz)
        row_positions: List[int] = []
        for R, t in ops:
            f = (R @ base + t) % 1.0
            k = find(f)
            if k >= 0:
                if k in row_positions:
                    continue  # symmetry copy of this same row
                row_positions.append(k)
                species[k][el] = min(1.0, species[k].get(el, 0.0) + occ)
            else:
                positions.append(f)
                pos_arr = np.vstack([pos_arr, f])
                species.append({el: occ})
                labels.append(syms[r])
                row_positions.append(len(positions) - 1)

    if not positions:
        raise CIFError("no atoms parsed")
    sites = [Site(p, s, l) for p, s, l in zip(positions, species, labels)]
    st = Structure(lattice, sites, name=name)
    st.meta["warnings"] = warnings
    sg = tags.get("_symmetry_space_group_name_h-m") or tags.get("_space_group_name_h-m_alt")
    if sg:
        st.meta["space_group"] = sg
    return st


def read_cif(path, name: str = "") -> Structure:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return parse_cif(fh.read(), name=name or str(path))
