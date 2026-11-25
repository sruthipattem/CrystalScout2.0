"""Existence check, feasibility check and crystal-structure (CIF) generation for any formula.

The composition models can score any formula, including ones that cannot exist. This module
adds the checks a materials chemist would make before spending lab time, and builds a crystal
structure for formulas that have none, so the structure model can score them too.

1. Existence (``lookup``)
   The formula is matched (composition-normalised) against the bundled Materials Project
   snapshot (19,806 Li compounds with DFT energy above hull) and OBELiX (599 measured
   electrolytes, with ICSD numbers where the dataset gives them).

2. Charge balance (``design.assign_charges``)
   A formula whose oxidation states cannot sum to zero (imbalance > 0.05 e per formula unit)
   is rejected as "unlikely to exist".

3. Structure generation (``generate``)
   A host crystal is chosen from 1,969 real templates (321 OBELiX CIFs + 1,648 Materials
   Project structures): the one whose Li / cation / anion site counts and species fit the
   target best (or the parent compound the user started from). Target species are placed on
   the host sites by site substitution with fractional occupancies (the standard
   "disordered / virtual crystal" description used in ICSD and OBELiX CIFs for doped phases),
   the Li sites are filled or emptied to the target Li content, and the cell is scaled by
   the change in ionic volume (Vegard-type scaling with Shannon radii). The structure is NOT
   relaxed: it is a starting model for DFT or a machine-learned potential, and the CIF says so.

4. Size fit (``size_fit``)
   Every substitution is checked with Shannon ionic radii (CN 6). Mismatch above 60 % is
   rejected; 35-60 % is flagged as strained.

5. Stability
   For known compounds the Materials Project energy above hull is reported. For hypothetical
   compounds it is NOT computed (that needs DFT or an ML interatomic potential such as CHGNet
   or M3GNet); the parent's stability and the size fit are shown as proxies instead.

6. Structure-path scoring (``assess``)
   The generated CIF is read back through the same ``featurize_cif_text`` path as uploads, so
   the real Sendek descriptors and the volume per atom are used instead of missing values, and
   the logistic-regression model can score it too.

Every number remains a screening priority, not a measured conductivity.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.structure import Site, Structure

STABLE_EHULL = 0.05  # eV/atom: "on or near the hull" (common screening tolerance)

# ------------------------------------------------------------------ Shannon ionic radii (Å, CN 6)
RADII: Dict[Tuple[str, int], float] = {
    ("Li", 1): 0.76, ("Na", 1): 1.02, ("K", 1): 1.38, ("Rb", 1): 1.52, ("Cs", 1): 1.67, ("Ag", 1): 1.15,
    ("Cu", 1): 0.77, ("Tl", 1): 1.50,
    ("Be", 2): 0.45, ("Mg", 2): 0.72, ("Ca", 2): 1.00, ("Sr", 2): 1.18, ("Ba", 2): 1.35, ("Zn", 2): 0.74,
    ("Cd", 2): 0.95, ("Pb", 2): 1.19, ("Sn", 2): 1.18, ("Ge", 2): 0.73, ("Mn", 2): 0.83, ("Fe", 2): 0.78,
    ("Co", 2): 0.745, ("Ni", 2): 0.69, ("Cu", 2): 0.73, ("Hg", 2): 1.02, ("Eu", 2): 1.17,
    ("B", 3): 0.27, ("Al", 3): 0.535, ("Ga", 3): 0.62, ("In", 3): 0.80, ("Sc", 3): 0.745, ("Y", 3): 0.90,
    ("La", 3): 1.032, ("Ce", 3): 1.01, ("Pr", 3): 0.99, ("Nd", 3): 0.983, ("Sm", 3): 0.958, ("Eu", 3): 0.947,
    ("Gd", 3): 0.938, ("Tb", 3): 0.923, ("Dy", 3): 0.912, ("Ho", 3): 0.901, ("Er", 3): 0.89, ("Tm", 3): 0.88,
    ("Yb", 3): 0.868, ("Lu", 3): 0.861, ("Bi", 3): 1.03, ("Sb", 3): 0.76, ("As", 3): 0.58, ("Fe", 3): 0.645,
    ("Cr", 3): 0.615, ("Ti", 3): 0.67, ("V", 3): 0.64, ("P", 3): 0.44, ("In", 1): 1.40,
    ("C", 4): 0.16, ("Si", 4): 0.40, ("Ge", 4): 0.53, ("Sn", 4): 0.69, ("Ti", 4): 0.605, ("Zr", 4): 0.72,
    ("Hf", 4): 0.71, ("Ce", 4): 0.87, ("Pb", 4): 0.775, ("Mn", 4): 0.53, ("Te", 4): 0.97, ("Se", 4): 0.50,
    ("S", 4): 0.37, ("V", 4): 0.58, ("Nb", 4): 0.68, ("Mo", 4): 0.65, ("W", 4): 0.66,
    ("P", 5): 0.38, ("As", 5): 0.46, ("Sb", 5): 0.60, ("Nb", 5): 0.64, ("Ta", 5): 0.64, ("V", 5): 0.54,
    ("Bi", 5): 0.76,
    ("W", 6): 0.60, ("Mo", 6): 0.59, ("S", 6): 0.29, ("Se", 6): 0.42, ("Te", 6): 0.56, ("Cr", 6): 0.44,
    ("F", -1): 1.33, ("Cl", -1): 1.81, ("Br", -1): 1.96, ("I", -1): 2.20, ("H", -1): 1.40,
    ("O", -2): 1.40, ("S", -2): 1.84, ("Se", -2): 1.98, ("Te", -2): 2.21,
    ("N", -3): 1.46, ("P", -3): 2.12, ("As", -3): 2.22, ("Sb", -3): 2.45, ("B", -3): 1.90, ("C", -4): 2.60,
}

FIT_LEVELS = [(0.15, "good", "good fit"), (0.25, "ok", "acceptable"), (0.60, "strained", "strained"),
              (9e9, "misfit", "size misfit")]


def ionic_radius(el: str, q: Optional[int]) -> Optional[float]:
    if q is not None and (el, q) in RADII:
        return RADII[(el, q)]
    same_sign = [r for (e, c), r in RADII.items() if e == el and (q is None or (c > 0) == (q > 0))]
    return float(np.mean(same_sign)) if same_sign else None


def size_fit(dopant: str, qd: Optional[int], host: str, qh: Optional[int]) -> Dict:
    """Relative Shannon-radius mismatch of a dopant on a host site, with a verdict."""
    rd, rh = ionic_radius(dopant, qd), ionic_radius(host, qh)
    if rd is None or rh is None:
        return {"mismatch": None, "level": "unknown", "label": "size unknown", "r_dopant": rd, "r_host": rh}
    mis = abs(rd - rh) / rh
    if abs(rd - rh) <= 0.20:  # small high-valence cations (P5+, Si4+, Ge4+...) share tetrahedral sites
        mis = min(mis, 0.20)
    level, label = next((lv, lb) for cut, lv, lb in FIT_LEVELS if mis <= cut)
    return {"mismatch": round(mis, 3), "level": level, "label": label, "r_dopant": rd, "r_host": rh}


# ------------------------------------------------------------------ composition keys
def comp_key(comp: Dict[str, float]) -> str:
    tot = sum(v for v in comp.values() if v > 0)
    if tot <= 0:
        return ""
    return "|".join(f"{e}:{comp[e] / tot:.3f}" for e in sorted(comp) if comp[e] > 1e-9)


def _parse(formula: str) -> Dict[str, float]:
    from .regression import parse_formula

    return parse_formula(str(formula))


@lru_cache(maxsize=1)
def _existence_index() -> Dict[str, Dict[str, List[Dict]]]:
    """composition key -> {"mp": [...], "obelix": [...]}"""
    import pandas as pd

    from ..data.materials_project import load_snapshot
    from ..data.obelix import load_obelix

    idx: Dict[str, Dict[str, List[Dict]]] = {}
    snap = load_snapshot()
    for r in snap.itertuples(index=False):
        k = comp_key(_parse(r.formula))
        if k:
            idx.setdefault(k, {"mp": [], "obelix": []})["mp"].append({
                "id": r.material_id, "formula": r.formula, "e_hull": float(r.energy_above_hull),
                "band_gap": float(r.band_gap), "space_group": str(r.space_group)})
    obx = load_obelix()
    for _, r in obx.iterrows():
        k = comp_key(_parse(r["formula"]))
        if not k:
            continue
        icsd = r.get("ICSD ID")
        icsd = None if icsd is None or (isinstance(icsd, float) and math.isnan(icsd)) or str(icsd).strip() in ("", "nan") else str(icsd).split(".")[0]
        sig = r.get("sigma")
        idx.setdefault(k, {"mp": [], "obelix": []})["obelix"].append({
            "id": r["ID"], "formula": r["formula"], "sigma": float(sig) if sig == sig else None,
            "upper_bound": bool(r.get("sigma_upper_bound", False)), "family": r.get("family", ""),
            "space_group": str(r.get("Space group", "") or ""), "icsd": icsd, "doi": str(r.get("DOI", "") or ""),
            "split": r.get("split", "")})
    for v in idx.values():
        v["mp"].sort(key=lambda m: m["e_hull"])
        v["obelix"].sort(key=lambda o: -(o["sigma"] or 0))
    return idx


def lookup(comp: Dict[str, float]) -> Dict:
    """Is this exact composition known? Materials Project (DFT) and OBELiX (measured)."""
    hit = _existence_index().get(comp_key(comp), {"mp": [], "obelix": []})
    mp, ob = hit["mp"], hit["obelix"]
    if ob:
        status, label = "measured", "Exists · measured in the lab"
    elif mp:
        status, label = "known", "Exists · in Materials Project (DFT)"
    else:
        status, label = "hypothetical", "Not found · hypothetical compound"
    best_mp = mp[0] if mp else None
    icsd = next((o["icsd"] for o in ob if o.get("icsd")), None)
    sig = [o["sigma"] for o in ob if o.get("sigma")]
    return {"status": status, "label": label, "mp": mp[:5], "obelix": ob[:5], "n_mp": len(mp), "n_obelix": len(ob),
            "sigma_min": min(sig) if sig else None, "sigma_max": max(sig) if sig else None,
            "best_mp": best_mp, "icsd": icsd,
            "searched": "Materials Project snapshot (19,806 Li compounds) and OBELiX (599 measured electrolytes)"}


# ------------------------------------------------------------------ host templates
@dataclass
class Template:
    id: str
    source: str
    formula: str
    space_group: str
    cell: Dict[str, float]
    charges: Dict[str, int]
    e_hull: Optional[float] = None
    sigma: Optional[float] = None
    n_sites: int = 0
    penalty: float = 0.0
    key: str = ""
    structure: Optional[Structure] = field(default=None, repr=False)


def _charges(comp: Dict[str, float]) -> Optional[Dict[str, int]]:
    from .design import assign_charges

    if len([e for e in comp if e != "Li"]) > 7:
        return None
    ch = assign_charges(comp)
    if not ch or ch.get("_imbalance", 1) > 0.05:
        return None
    return {k: v for k, v in ch.items() if not k.startswith("_")}


@lru_cache(maxsize=1)
def templates() -> Tuple[Template, ...]:
    """Every real crystal structure the generator can start from."""
    from ..core.cif import parse_cif
    from ..data.materials_project import load_pool_structures, load_snapshot
    from ..data.obelix import load_obelix

    out: List[Template] = []
    obx = load_obelix()
    for _, r in obx[obx["has_cif"]].iterrows():
        try:
            st = parse_cif(open(r["cif_path"], encoding="utf-8", errors="replace").read(), name=r["ID"])
        except Exception:  # noqa: BLE001
            continue
        cell = st.composition()
        ch = _charges(cell)
        if not ch or "Li" not in cell:
            continue
        sig = r.get("sigma")
        out.append(Template(f"OBELiX {r['ID']}", "OBELiX (experimental CIF)", r["formula"], str(r.get("Space group", "") or ""),
                            cell, ch, None, float(sig) if sig == sig else None, len(st.sites),
                            0.15 if bool(r.get("close_match", False)) else 0.0, comp_key(cell), st))
    snap = load_snapshot().set_index("material_id")
    for mid, sd in load_pool_structures().items():
        try:
            st = Structure.from_pymatgen_dict(sd, name=mid)
        except Exception:  # noqa: BLE001
            continue
        cell = st.composition()
        ch = _charges(cell)
        if not ch or "Li" not in cell:
            continue
        row = snap.loc[mid] if mid in snap.index else None
        eh = float(row["energy_above_hull"]) if row is not None else None
        sg = str(row["space_group"]) if row is not None else ""
        out.append(Template(mid, "Materials Project (DFT structure)", str(row["formula"]) if row is not None else st.formula(),
                            sg, cell, ch, eh, None, len(st.sites), 0.05 + 2.0 * (eh or 0.0), comp_key(cell), st))
    return tuple(out)


def _role(q: int, el: str) -> str:
    return "li" if el == "Li" else ("cat" if q > 0 else "an")


def _pair_cost(t: str, qt: int, h: str, qh: int) -> float:
    if t == h:
        return 0.0
    fit = size_fit(t, qt, h, qh)
    mis = fit["mismatch"] if fit["mismatch"] is not None else 0.5
    return 0.25 + 0.4 * abs(qt - qh) + mis


def _allocate(target: Dict[str, float], host: Dict[str, float], tq: Dict[str, int], hq: Dict[str, int]):
    """Greedy placement of target species on host site classes of the same role."""
    cap = dict(host)
    rem = dict(target)
    alloc: Dict[Tuple[str, str], float] = {}
    for e in list(rem):
        if e in cap:
            x = min(rem[e], cap[e])
            if x > 0:
                alloc[(e, e)] = x
                rem[e] -= x
                cap[e] -= x
    cost = 0.0
    for t in sorted(rem, key=lambda e: -rem[e]):
        if rem[t] <= 1e-9:
            continue
        for h in sorted([h for h in cap if cap[h] > 1e-9], key=lambda h: _pair_cost(t, tq[t], h, hq[h])):
            x = min(rem[t], cap[h])
            alloc[(t, h)] = alloc.get((t, h), 0.0) + x
            cost += x * _pair_cost(t, tq[t], h, hq[h])
            rem[t] -= x
            cap[h] -= x
            if rem[t] <= 1e-9:
                break
    overflow = sum(max(v, 0.0) for v in rem.values())
    vacancy = sum(max(v, 0.0) for v in cap.values())
    return alloc, overflow, vacancy, cost


def _split(comp: Dict[str, float], ch: Dict[str, int]) -> Dict[str, Dict[str, float]]:
    out = {"li": {}, "cat": {}, "an": {}}
    for e, n in comp.items():
        out[_role(ch.get(e, 0), e)][e] = n
    return out


def _match(target: Dict[str, float], tch: Dict[str, int], tpl: Template) -> Optional[Tuple[float, float]]:
    """(score, scale) of fitting ``target`` onto template ``tpl``; lower score = better host."""
    T, H = _split(target, tch), _split(tpl.cell, tpl.charges)
    an_t, an_h = sum(T["an"].values()), sum(H["an"].values())
    if an_t <= 0 or an_h <= 0 or not H["li"]:
        return None
    if bool(T["cat"]) != bool(H["cat"]):
        return None
    f = an_h / an_t
    li_mis = abs(f * sum(T["li"].values()) - sum(H["li"].values())) / an_h
    cat_mis = abs(f * sum(T["cat"].values()) - sum(H["cat"].values())) / an_h
    if li_mis > 0.6 or cat_mis > 0.35:
        return None
    cost = 0.0
    for role in ("cat", "an"):
        tt = {e: f * n for e, n in T[role].items()}
        _, over, _, c = _allocate(tt, H[role], tch, tpl.charges)
        cost += c + 2.0 * over
    cost /= max(f * sum(target.values()), 1e-9) / 4.0
    score = 4.0 * (li_mis + cat_mis) + cost + tpl.penalty + tpl.n_sites / 2000.0
    return score, f


def _sg_norm(sg: Optional[str]) -> str:
    return "".join(ch for ch in str(sg or "") if ch not in " _").lower()


def find_host(comp: Dict[str, float], charges: Dict[str, int], base: Optional[Dict[str, float]] = None,
              space_group: Optional[str] = None) -> Optional[Dict]:
    """Best real crystal to build ``comp`` from (the exact compound when it exists; the given space group preferred)."""
    key, bkey = comp_key(comp), comp_key(base) if base else None
    sgn = _sg_norm(space_group)
    best = None
    for tpl in templates():
        m = _match(comp, charges, tpl)
        if m is None:
            continue
        score, f = m
        same_sg = bool(sgn) and _sg_norm(tpl.space_group) == sgn
        if tpl.key == key:
            # the compound itself: use its real structure (unless the user asked for another polymorph)
            score -= 10.0 if (not sgn or same_sg) else 0.8
        elif bkey and tpl.key == bkey:
            score -= 1.0   # the parent the user started from
        if same_sg:
            score -= 1.2   # same polymorph as the user asked for
        if best is None or score < best[0]:
            best = (score, f, tpl)
    if best is None:
        return None
    return {"template": best[2], "scale": best[1], "score": best[0], "exact": best[2].key == key}


# ------------------------------------------------------------------ substitution
def _substitute(tpl: Template, target: Dict[str, float], tch: Dict[str, int], f: float) -> Tuple[Structure, Dict]:
    st = tpl.structure
    T, H = _split(target, tch), _split(tpl.cell, tpl.charges)
    alloc: Dict[Tuple[str, str], float] = {}
    notes: List[str] = []
    fits: List[Dict] = []
    overflow = vacancy = 0.0
    for role in ("cat", "an"):
        tt = {e: f * n for e, n in T[role].items()}
        a, ov, va, _ = _allocate(tt, H[role], tch, tpl.charges)
        alloc.update(a)
        overflow += ov
        vacancy += va
    for (t, h), x in alloc.items():
        if t != h and x > 1e-6:
            fit = size_fit(t, tch.get(t), h, tpl.charges.get(h))
            fits.append({"dopant": t, "host": h, "amount": round(x / f, 4), **fit})
    # Li content
    li_t = f * T["li"].get("Li", 0.0)
    li_h = H["li"].get("Li", 0.0)
    new_sites: List[Site] = []
    li_extra = max(li_t - li_h, 0.0)
    li_ratio = li_t / li_h if li_h > 0 and li_t <= li_h else 1.0
    spare_tot = 0.0
    if li_extra > 0:
        spare_tot = sum(max(1.0 - s.occupancy, 0.0) for s in st.sites if s.occ("Li") > 0)
    li_fill = min(li_extra / spare_tot, 1.0) if spare_tot > 0 else 0.0
    li_short = max(li_extra - spare_tot, 0.0)
    for s in st.sites:
        sp: Dict[str, float] = {}
        for h, occ in s.species.items():
            if h == "Li":
                v = occ * li_ratio
                if li_extra > 0:
                    v += max(1.0 - s.occupancy, 0.0) * li_fill
                sp["Li"] = sp.get("Li", 0.0) + v
                continue
            ch = tpl.cell.get(h, 0.0)
            if ch <= 0:
                continue
            for (t, hh), x in alloc.items():
                if hh == h and x > 1e-9:
                    sp[t] = sp.get(t, 0.0) + occ * x / ch
        sp = {e: round(min(o, 1.0), 5) for e, o in sp.items() if o > 1e-5}
        tot = sum(sp.values())
        if tot > 1.0:
            sp = {e: round(o / tot, 5) for e, o in sp.items()}
        if sp:
            new_sites.append(Site(np.array(s.frac, float), sp, s.label))
    # Vegard-type cell scaling from the change in ionic volume
    def ion_vol(cell: Dict[str, float], ch: Dict[str, int]) -> float:
        v = 0.0
        for e, n in cell.items():
            r = ionic_radius(e, ch.get(e))
            v += n * (r or 1.0) ** 3
        return v

    placed: Dict[str, float] = {}
    for (t, h), x in alloc.items():
        placed[t] = placed.get(t, 0.0) + x
    placed["Li"] = li_t - li_short
    ratio = ion_vol(placed, tch) / max(ion_vol(tpl.cell, tpl.charges), 1e-9)
    s_lin = float(np.clip(ratio ** (1 / 3), 0.9, 1.1))
    new = Structure(np.array(st.lattice, float) * s_lin, new_sites, name="", meta={})
    got = new.composition()
    want = {e: f * n for e, n in target.items()}
    dev = max(abs(got.get(e, 0.0) - want.get(e, 0.0)) for e in set(got) | set(want)) / max(sum(want.values()) / len(want), 1e-9)
    if overflow > 1e-3:
        notes.append(f"{overflow / f:.2f} atoms per formula unit did not fit on the host sites.")
    if vacancy > 1e-3:
        notes.append(f"{vacancy / f:.2f} host sites per formula unit left vacant.")
    if li_short > 1e-3:
        notes.append(f"{li_short / f:.2f} Li per formula unit could not be placed (host Li sites full).")
    return new, {"fits": fits, "scale": round(s_lin, 4), "composition_error": round(float(dev), 4), "notes": notes,
                 "li_short": li_short / f if f else 0.0}


def _cif_text(st: Structure, formula: str, host: Template, info: Dict, exact: bool, sg: str) -> str:
    head = [
        f"# CrystalScout structure model for {formula}",
        "# Generated by CrystalScout (Team Crystal Scout, Neil Gogte Institute of Technology)",
    ]
    if exact:
        head += [f"# Source: real structure of this compound, {host.id} ({host.source}).",
                 f"# Space group of the source: {sg or 'n/a'}. Written in P1 with all sites explicit."]
    else:
        head += [f"# Method: site substitution on host {host.id} ({host.formula}, {host.source}, space group {host.space_group or 'n/a'}),",
                 f"#   fractional occupancies for mixed sites, Li sites re-filled to the target Li content,",
                 f"#   cell scaled x{info['scale']} from Shannon ionic volumes (Vegard-type).",
                 "# NOT RELAXED: a starting model. Relax with DFT or an ML potential (CHGNet, M3GNet, MACE) before use.",
                 "# Order the partial occupancies (e.g. supercell + enumeration) for DFT."]
    head.append("# Predicted properties are screening priorities, not measurements.")
    body = st.to_cif().split("\n", 1)[1]
    name = "".join(ch for ch in formula if ch.isalnum() or ch == ".").replace(".", "p")
    return "\n".join(head) + f"\ndata_{name}\n" + body.split("\n", 1)[1]


def template_by_id(tid: str) -> Optional[Template]:
    return next((t for t in templates() if t.id == tid), None)


def generate(formula: str, base: Optional[str] = None, host_template: Optional[Template] = None,
             space_group: Optional[str] = None) -> Dict:
    """Crystal structure + CIF for ``formula`` (real one when known, else a substitution model).

    ``host_template`` forces the parent crystal (the Dopant Advisor builds every variant on the base's crystal)."""
    from .design import assign_charges

    comp = _parse(formula)
    ch_full = assign_charges(comp)
    if not ch_full or ch_full.get("_imbalance", 1) > 0.05:
        return {"ok": False, "reason": "The formula is not charge-balanced, so no sensible crystal can be built."}
    ch = {k: v for k, v in ch_full.items() if not k.startswith("_")}
    host = None
    if host_template is not None:
        m = _match(comp, ch, host_template)
        if m is not None:
            host = {"template": host_template, "scale": m[1], "score": m[0], "exact": host_template.key == comp_key(comp)}
    if host is None:
        bcomp = _parse(base) if base else None
        host = find_host(comp, ch, bcomp, space_group)
    if host is None:
        return {"ok": False, "reason": "No crystal template with compatible Li, cation and anion sites was found."}
    tpl, f, exact = host["template"], host["scale"], host["exact"]
    if exact:
        st = Structure(np.array(tpl.structure.lattice, float), [Site(np.array(s.frac), dict(s.species), s.label) for s in tpl.structure.sites])
        info = {"fits": [], "scale": 1.0, "composition_error": 0.0, "notes": [], "li_short": 0.0}
    else:
        st, info = _substitute(tpl, comp, ch, f)
    st.name = formula
    cif = _cif_text(st, formula, tpl, info, exact, tpl.space_group)
    worst = max((x for x in info["fits"] if x["mismatch"] is not None), key=lambda x: x["mismatch"], default=None)
    return {"ok": True, "structure": st, "cif": cif, "exact": exact, "host_id": tpl.id, "host_formula": tpl.formula,
            "host_source": tpl.source, "host_space_group": tpl.space_group, "host_e_hull": tpl.e_hull,
            "host_sigma": tpl.sigma, "fits": info["fits"], "worst_fit": worst, "scale": info["scale"],
            "composition_error": info["composition_error"], "notes": info["notes"], "charges": ch,
            "n_sites": len(st.sites), "relaxed": exact, "volume_per_atom": None}


# ------------------------------------------------------------------ the full check
BADGES = {
    "measured": ("Measured · real material", "good", "Synthesised and measured (OBELiX). The structure is the experimental one."),
    "known_stable": ("Known · DFT-stable", "good", "In Materials Project on or near the convex hull."),
    "known_meta": ("Known · metastable", "warn", "In Materials Project above the hull; may need disorder or high-temperature synthesis."),
    "hypo_plausible": ("Hypothetical · plausible", "teal", "Not in any database. Charge-balanced, sizes fit, built from a real host crystal."),
    "hypo_strained": ("Hypothetical · strained", "warn", "Not in any database. Charge-balanced but the dopant size or site fit is poor."),
    "unlikely": ("Unlikely to exist", "bad", "Fails a basic chemistry check (charge balance, ion size or site fit)."),
}


def _badge(key: str) -> Dict:
    label, tone, note = BADGES[key]
    return {"key": key, "label": label, "tone": tone, "note": note}


def feasibility(comp: Dict[str, float], exist: Dict, gen: Optional[Dict], charges: Optional[Dict]) -> Dict:
    reasons: List[str] = []
    balanced = bool(charges and charges.get("_imbalance", 1) <= 0.05)
    if exist["status"] == "measured":
        b = _badge("measured")
    elif not balanced:
        b = _badge("unlikely")
        reasons.append("Oxidation states cannot sum to zero" + (f" (off by {charges['_imbalance']:g} e per formula unit)." if charges else "."))
    elif exist["status"] == "known":
        eh = exist["best_mp"]["e_hull"]
        b = _badge("known_stable" if eh <= STABLE_EHULL else "known_meta")
        reasons.append(f"Materials Project {exist['best_mp']['id']}: E_hull {eh:.3f} eV/atom.")
    elif not gen or not gen.get("ok"):
        b = _badge("unlikely")
        reasons.append(gen.get("reason") if gen else "No structure could be built.")
    else:
        worst = gen.get("worst_fit")
        lvl = worst["level"] if worst else "good"
        bad_site = gen["composition_error"] > 0.05 or any("could not" in n or "did not fit" in n for n in gen["notes"])
        if lvl == "misfit":
            b = _badge("unlikely")
            reasons.append(f"{worst['dopant']} is {worst['mismatch'] * 100:.0f} % different in size from {worst['host']} (Shannon radii).")
        elif gen["composition_error"] > 0.15:
            b = _badge("unlikely")
            reasons.append("No known crystal has room for this composition: " + " ".join(gen["notes"]))
        elif lvl in ("strained", "unknown") or bad_site:
            b = _badge("hypo_strained")
            if worst and lvl in ("strained", "unknown"):
                reasons.append(f"{worst['dopant']} on the {worst['host']} site: {worst['label']}"
                               + (f" ({worst['mismatch'] * 100:.0f} % size mismatch)." if worst["mismatch"] is not None else "."))
            reasons += gen["notes"]
        else:
            b = _badge("hypo_plausible")
            if worst:
                reasons.append(f"Largest size mismatch {worst['mismatch'] * 100:.0f} % ({worst['dopant']} on {worst['host']}): {worst['label']}.")
            if gen.get("host_e_hull") is not None:
                reasons.append(f"Parent structure {gen['host_id']} has E_hull {gen['host_e_hull']:.3f} eV/atom.")
            elif "OBELiX" in gen.get("host_id", ""):
                reasons.append(f"Parent structure {gen['host_id']} is a synthesised material.")
    if exist["status"] == "hypothetical":
        stability = "Not computed for hypothetical compounds (needs DFT or an ML potential such as CHGNet). Parent stability and size fit are shown as proxies."
    elif exist["best_mp"]:
        stability = f"E_hull {exist['best_mp']['e_hull']:.3f} eV/atom (Materials Project {exist['best_mp']['id']})."
    else:
        stability = "Synthesised material (measured in OBELiX); not in the Materials Project snapshot."
    return {**b, "reasons": reasons, "charge_balanced": balanced, "stability": stability,
            "rankable": b["key"] in ("hypo_plausible", "hypo_strained")}


def confidence(rd: Dict) -> Dict:
    width = rd["log10_hi"] - rd["log10_lo"]
    oor = rd.get("out_of_range") or []
    if oor or width > 3.5:
        lvl, tone = "Low", "bad"
    elif width > 2.5 or not rd.get("has_structure"):
        lvl, tone = "Medium", "warn"
    else:
        lvl, tone = "High", "good"
    notes = []
    if oor:
        notes.append("Outside the training range for: " + ", ".join(x.replace("_", " ") for x in oor) + ".")
    if width > 3.5:
        notes.append(f"The trees disagree widely (80 % range spans {width:.1f} orders of magnitude).")
    if not rd.get("has_structure"):
        notes.append("Scored from composition only (no structure descriptors).")
    if rd.get("known_composition"):
        notes.append("This composition is in the training data, so the prediction mostly reflects its measured value.")
    return {"level": lvl, "tone": tone, "width": round(width, 2), "notes": notes}


def logreg_probability(bundle, descriptors: Optional[Dict[str, float]]) -> Optional[float]:
    from ..core.descriptors import FEATURES

    if bundle is None or not descriptors or any(descriptors.get(k) is None or descriptors[k] != descriptors[k] for k in FEATURES):
        return None
    try:
        return float(bundle.models["logreg"].predict_proba(np.array([[descriptors[k] for k in FEATURES]], float))[:, 1][0])
    except Exception:  # noqa: BLE001
        return None


def assess(reg, formula: str, space_group: Optional[str] = None, base: Optional[str] = None, ref=None,
           bundle=None, want_structure: bool = True, host_template: Optional[Template] = None,
           gen: Optional[Dict] = None) -> Dict:
    """Existence + feasibility + structure model + structure-path prediction for one formula."""
    from ..featurize import featurize_cif_text
    from .design import assign_charges
    from .insight import parse_checked_formula, sigma_readout
    from .regression import structure_volume_per_atom

    comp = parse_checked_formula(formula)
    charges = assign_charges(comp)
    exist = lookup(comp)
    gen = (generate(formula, base, host_template, space_group) if want_structure else None) if gen is None else gen
    desc, vpa, sg_used = None, np.nan, space_group or None
    if gen and gen.get("ok"):
        try:
            st2, feats = featurize_cif_text(gen["cif"], name=formula)
            from ..core.descriptors import FEATURES

            if all(feats.get(k) == feats.get(k) and feats.get(k) is not None for k in FEATURES):
                desc = {k: float(feats[k]) for k in FEATURES}
            vpa = structure_volume_per_atom(st2)
            gen["volume_per_atom"] = round(float(vpa), 3) if vpa == vpa else None
            gen["descriptors"] = desc
            sg_used = space_group or gen["host_space_group"] or None
        except Exception as exc:  # noqa: BLE001
            gen["descriptor_error"] = str(exc)
    p_lr = logreg_probability(bundle, desc)
    rd = sigma_readout(reg, comp, sg_used, vpa, desc, formula, p_lr, ref)
    rd_comp = sigma_readout(reg, comp, space_group or None, np.nan, None, formula, None, None) if desc else None
    feas = feasibility(comp, exist, gen, charges)
    return {"formula": formula, "comp": comp, "existence": exist, "structure": gen, "feasibility": feas,
            "sigma": rd, "sigma_composition_only": rd_comp, "p_logreg": p_lr, "confidence": confidence(rd),
            "space_group_used": sg_used or "",
            "honesty": "Screening priority from the model, not a measured conductivity."}


def structure_summary(gen: Optional[Dict], max_atoms: int = 400) -> Optional[Dict]:
    """JSON-safe description of a generated structure (for pages and the API)."""
    if not gen or not gen.get("ok"):
        return None
    st: Structure = gen["structure"]
    a, b, c, al, be, ga = st.lattice_parameters
    return {"exact": gen["exact"], "host_id": gen["host_id"], "host_formula": gen["host_formula"],
            "host_source": gen["host_source"], "host_space_group": gen["host_space_group"],
            "host_e_hull": gen["host_e_hull"], "n_sites": gen["n_sites"], "scale": gen["scale"],
            "volume_per_atom": gen.get("volume_per_atom"), "relaxed": gen["relaxed"],
            "lattice": {"a": round(a, 3), "b": round(b, 3), "c": round(c, 3), "alpha": round(al, 2), "beta": round(be, 2),
                        "gamma": round(ga, 2), "volume": round(float(st.volume), 2)},
            "fits": gen["fits"], "notes": gen["notes"], "descriptors": gen.get("descriptors"),
            "viewer": st.viewer_payload(max_atoms)}
