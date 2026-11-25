"""Materials design tools: Formula Lab, Dopant Advisor and Synthesis Planner.

Formula Lab
    Predict any edited formula instantly and explain what the edit changed
    (which model features moved, weighted by how much the model relies on them).

Dopant Advisor
    Given a base Li compound, generate chemically sensible doped variants
    (substitution on every non-Li site, isovalent and aliovalent, with lithium
    adjusted to keep the formula charge-neutral), predict all of them with the
    sigma-Regressor in one batch and return the best as ready-to-make formulas.

Synthesis Planner
    Turns any target formula into a solid-state recipe: the binary precursors
    (e.g. Li2S + P2S5 + LiCl for Li6PS5Cl), their molar ratio from a
    non-negative least-squares element balance, and the grams needed for a
    chosen batch size.

All numbers are model predictions that rank candidates for the lab; they are
not measurements.
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from ..core.chemistry import TRANSITION_METALS
from ..core.elements import MASS, Z_OF
from .insight import SUPPLY_RISK, TOXIC, chem_family, fmt_sigma, industry_flags, tier

# ---------------------------------------------------------------- oxidation states
ANION_CHARGE = {"O": -2, "S": -2, "Se": -2, "Te": -2, "N": -3, "F": -1, "Cl": -1, "Br": -1, "I": -1}
# common positive oxidation states, most likely first
CATION_STATES: Dict[str, Tuple[int, ...]] = {
    "Li": (1,), "Na": (1,), "K": (1,), "Rb": (1,), "Cs": (1,), "Ag": (1,), "H": (1,),
    "Be": (2,), "Mg": (2,), "Ca": (2,), "Sr": (2,), "Ba": (2,), "Zn": (2,), "Cd": (2,),
    "B": (3,), "Al": (3,), "Ga": (3,), "In": (3, 1), "Sc": (3,), "Y": (3,), "La": (3,), "Ce": (3, 4), "Nd": (3,),
    "Gd": (3,), "Er": (3,), "Yb": (3,), "Lu": (3,), "Bi": (3, 5), "Sb": (5, 3), "As": (5, 3),
    "Si": (4,), "Ge": (4, 2), "Sn": (4, 2), "Ti": (4,), "Zr": (4,), "Hf": (4,), "Pb": (2, 4), "C": (4,),
    "P": (5, 3), "Nb": (5,), "Ta": (5,), "V": (5,), "W": (6,), "Mo": (6,),
}

# dopant candidates by oxidation state (the advisor tries every one near the host's charge)
DOPANTS: Dict[int, Tuple[str, ...]] = {
    1: ("Na", "Ag"),
    2: ("Mg", "Ca", "Sr", "Zn", "Ba"),
    3: ("B", "Al", "Ga", "In", "Sc", "Y", "La"),
    4: ("Si", "Ge", "Sn", "Ti", "Zr", "Hf"),
    5: ("P", "As", "Sb", "Nb", "Ta"),
    6: ("W", "Mo"),
}
ANION_DOPANTS: Dict[int, Tuple[str, ...]] = {-1: ("F", "Cl", "Br", "I"), -2: ("O", "S", "Se"), -3: ("N",)}
FRACTIONS = (0.1, 0.25, 0.5)


def assign_charges(comp: Dict[str, float]) -> Optional[Dict[str, int]]:
    """Oxidation state per element that makes the formula neutral (most common states first)."""
    anions = {e: ANION_CHARGE[e] for e in comp if e in ANION_CHARGE}
    if not anions:  # e.g. Li3P, Li3Sb: most electronegative non-Li element is the anion
        from ..core.chemistry import electronegativity

        others = [e for e in comp if e != "Li"]
        if not others:
            return None
        a = max(others, key=electronegativity)
        anions = {a: -3 if a in ("P", "As", "Sb", "N", "B") else -2}
    cations = [e for e in comp if e not in anions]
    choices = [CATION_STATES.get(e, (2, 3, 4)) for e in cations]
    neg = sum(comp[a] * q for a, q in anions.items())
    best, best_err = None, 1e9
    for combo in itertools.product(*choices):
        pos = sum(comp[e] * q for e, q in zip(cations, combo))
        err = abs(pos + neg)
        if err < best_err - 1e-9:
            best, best_err = dict(zip(cations, combo)), err
        if err < 1e-6:
            break
    if best is None:
        return None
    best.update(anions)
    best["_imbalance"] = round(best_err, 4)
    return best


def fmt_amount(v: float) -> str:
    if abs(v - round(v)) < 1e-6:
        n = int(round(v))
        return "" if n == 1 else str(n)
    return f"{v:.3f}".rstrip("0").rstrip(".")


def order_elements(comp: Dict[str, float]) -> List[str]:
    """Li first, then cations by electronegativity, anions last (by electronegativity)."""
    from ..core.chemistry import electronegativity

    an = [e for e in comp if e in ANION_CHARGE]
    cat = [e for e in comp if e not in ANION_CHARGE and e != "Li"]
    return (["Li"] if "Li" in comp else []) + sorted(cat, key=electronegativity) + sorted(an, key=electronegativity)


def to_formula(comp: Dict[str, float], order: Optional[List[str]] = None) -> str:
    order = [e for e in (order or order_elements(comp)) if comp.get(e, 0) > 1e-9]
    order += [e for e in order_elements(comp) if e not in order and comp.get(e, 0) > 1e-9]
    return "".join(f"{e}{fmt_amount(comp[e])}" for e in order)


def pretty(formula: str) -> str:
    """HTML with subscripts."""
    import re

    return re.sub(r"(\d+(?:\.\d+)?)", r"<sub>\1</sub>", formula)


# ---------------------------------------------------------------- batch prediction
def predict_many(reg, comps: List[Dict[str, float]], space_group=None) -> pd.DataFrame:
    """Predict a list of compositions in one pass through the 500 trees."""
    from .regression import feature_row

    F = pd.DataFrame([feature_row(c, space_group, np.nan, None) for c in comps])
    return reg.predict_frame(F).reset_index(drop=True)


# ---------------------------------------------------------------- Formula Lab
@lru_cache(maxsize=1)
def _importances(key: int = 0) -> Dict[str, float]:
    from .regression import get_regressor

    reg = get_regressor(verbose=False)
    m = reg.model.named_steps["model"]
    names = reg.feature_names
    return dict(zip(names, m.feature_importances_[: len(names)]))


FEATURE_WORDS = {
    "crystal_system": "crystal system", "centring": "lattice centring", "n_elements": "number of elements",
    "frac_Li": "lithium fraction", "frac_S": "sulfur fraction", "frac_O": "oxygen fraction", "frac_P": "phosphorus fraction",
    "frac_Se": "selenium fraction", "frac_Cl": "chlorine fraction", "frac_Br": "bromine fraction", "frac_I": "iodine fraction",
    "frac_F": "fluorine fraction", "frac_N": "nitrogen fraction", "frac_anions": "anion share", "en_li_anion": "Li–anion electronegativity gap",
    "volume_per_atom": "volume per atom", "n_anion_types": "number of anion types",
}


def _word(f: str) -> str:
    if f in FEATURE_WORDS:
        return FEATURE_WORDS[f]
    for stat, w in (("mean_", "average "), ("avgdev_", "spread of "), ("max_", "largest "), ("min_", "smallest "),
                    ("range_", "range of "), ("mode_", "typical ")):
        if f.startswith(stat):
            prop = f[len(stat):].replace("nvalence", "valence electrons").replace("electronegativity", "electronegativity") \
                .replace("Z", "atomic number").replace("ns", "s electrons").replace("np", "p electrons") \
                .replace("nd", "d electrons").replace("nf", "f electrons")
            return w + prop
    return f.replace("_", " ")


def explain_change(base: Dict[str, float], new: Dict[str, float], space_group=None, top: int = 4) -> List[Dict]:
    """Which features moved most between two compositions, weighted by model importance."""
    from .regression import feature_row

    a = feature_row(base, space_group, np.nan, None)
    b = feature_row(new, space_group, np.nan, None)
    imp = _importances()
    rows = []
    skip_props = ("_ns", "_np", "_nd", "_nf", "_block", "_period", "_group")
    for f, w in imp.items():
        if f.startswith(("min_", "max_", "range_", "mode_")) or f.endswith(skip_props) or f in ("LLB", "SBI", "AFC", "LASD", "LLSD"):
            continue
        va, vb = a.get(f, np.nan), b.get(f, np.nan)
        if va is None or vb is None or (isinstance(va, float) and math.isnan(va)) or (isinstance(vb, float) and math.isnan(vb)):
            continue
        scale = max(abs(va), abs(vb), 1e-9)
        rel = (vb - va) / scale
        if abs(rel) < 1e-6:
            continue
        rows.append({"feature": f, "label": _word(f), "before": float(va), "after": float(vb), "weight": float(w), "score": abs(rel) * w,
                     "direction": "up" if vb > va else "down"})
    rows.sort(key=lambda r: -r["score"])
    return rows[:top]


_BASE_CACHE: Dict[Tuple, Dict] = {}


def _assess_cached(reg, formula: str, space_group, ref, bundle) -> Dict:
    from .structgen import assess

    key = (formula, space_group or "", id(reg), id(bundle), id(ref))
    if key not in _BASE_CACHE:
        if len(_BASE_CACHE) > 64:
            _BASE_CACHE.clear()
        _BASE_CACHE[key] = assess(reg, formula, space_group, None, ref, bundle)
    return _BASE_CACHE[key]


def lab_evaluate(reg, formula: str, space_group=None, base: Optional[str] = None, ref=None, bundle=None) -> Dict:
    """Everything the Formula Lab shows for one (possibly edited) formula: existence, feasibility,
    a crystal structure (CIF) and the structure-path prediction."""
    from .insight import verdict
    from .structgen import assess

    a = assess(reg, formula, space_group, base if base and base != formula else None, ref, bundle)
    comp, rd = a["comp"], a["sigma"]
    charges = assign_charges(comp)
    out = {"formula": formula, "pretty": pretty(formula), "space_group": space_group or a["space_group_used"], "sigma": rd,
           "verdict": verdict(a["p_logreg"], rd["log10_sigma"], rd.get("sigma_pct")), "flags": industry_flags(comp),
           "charge_balanced": bool(charges and charges.get("_imbalance", 1) <= 0.05),
           "imbalance": charges.get("_imbalance") if charges else None,
           "oxidation": {k: v for k, v in (charges or {}).items() if not k.startswith("_")},
           "existence": a["existence"], "feasibility": a["feasibility"], "structure": a["structure"],
           "confidence": a["confidence"], "p_logreg": a["p_logreg"], "sigma_composition_only": a["sigma_composition_only"],
           "honesty": a["honesty"]}
    if base and base != formula:
        from .insight import parse_checked_formula

        bcomp = parse_checked_formula(base)
        ba = _assess_cached(reg, base, space_group, ref, bundle)
        both = rd.get("has_structure") and ba["sigma"].get("has_structure")
        b_ls = ba["sigma"]["log10_sigma"] if both else (ba["sigma_composition_only"] or ba["sigma"])["log10_sigma"]
        n_ls = rd["log10_sigma"] if both else (a["sigma_composition_only"] or rd)["log10_sigma"]
        out["base"] = {"formula": base, "pretty": pretty(base), "log10_sigma": b_ls, "sigma_text": fmt_sigma(b_ls),
                       "feasibility": ba["feasibility"]["label"]}
        out["compare_path"] = "crystal structure" if both else "composition"
        out["delta_log"] = n_ls - b_ls
        out["gain"] = 10 ** out["delta_log"]
        out["drivers"] = explain_change(bcomp, comp, space_group)
        bd = (ba["structure"] or {}).get("descriptors") if ba["structure"] else None
        nd = (a["structure"] or {}).get("descriptors") if a["structure"] else None
        if bd and nd:
            out["descriptor_change"] = [{"name": k, "before": round(bd[k], 3), "after": round(nd[k], 3)} for k in ("LLB", "SBI", "AFC", "LASD", "LLSD")]
    return out


# ---------------------------------------------------------------- Dopant Advisor
def generate_variants(comp: Dict[str, float], allow_tm: bool = False, avoid_toxic: bool = True,
                      avoid_supply_risk: bool = False, fractions=FRACTIONS, rejected: Optional[Dict[str, int]] = None) -> List[Dict]:
    """Substitutional dopings of every non-Li site, charge-balanced through the Li content.

    Hard filters (counted in ``rejected``): the variant must be charge-neutral (imbalance <= 0.05 e per
    formula unit) and the dopant must fit the host site (Shannon-radius mismatch <= 60 %)."""
    from .structgen import size_fit

    rejected = rejected if rejected is not None else {}
    for k in ("charge", "size", "lithium"):
        rejected.setdefault(k, 0)
    charges = assign_charges(comp)
    if not charges or charges.get("_imbalance", 1) > 0.05:
        return []
    li0 = comp.get("Li", 0.0)
    out, seen = [], set()
    for host, n in comp.items():
        if host == "Li":
            continue
        q = charges.get(host)
        if q is None:
            continue
        if q > 0:
            pool = [(d, qd) for qd in (q - 1, q, q + 1) for d in DOPANTS.get(qd, ()) if d != host]
        else:
            pool = [(d, qd) for qd in (q - 1, q, q + 1) for d in ANION_DOPANTS.get(qd, ()) if d != host]
        for d, qd in pool:
            if d in comp and d != host:
                same_site = False
            if not allow_tm and d in TRANSITION_METALS:
                continue
            if avoid_toxic and d in TOXIC:
                continue
            if avoid_supply_risk and d in SUPPLY_RISK:
                continue
            fit = size_fit(d, qd, host, q)
            if fit["level"] == "misfit":
                rejected["size"] += len(fractions)
                continue
            for f in fractions:
                x = round(f * n, 4)
                new = dict(comp)
                new[host] = round(n - x, 4)
                new[d] = round(new.get(d, 0.0) + x, 4)
                # charge compensation through lithium: Li change = -(qd - q) * x  (anions: q, qd negative)
                dli = -(qd - q) * x
                new["Li"] = round(li0 + dli, 4)
                if new["Li"] <= 0.2 or abs(dli) > 0.6 * max(li0, 1e-9):
                    rejected["lithium"] += 1
                    continue
                if new[host] <= 1e-6:
                    del new[host]
                chk = assign_charges(new)
                if not chk or chk.get("_imbalance", 1) > 0.05:
                    rejected["charge"] += 1
                    continue
                key = tuple(sorted((k, round(v, 3)) for k, v in new.items()))
                if key in seen:
                    continue
                seen.add(key)
                kind = "isovalent" if qd == q else ("aliovalent, adds Li⁺" if dli > 0 else "aliovalent, removes Li⁺")
                order = order_elements(comp)
                if d not in order:
                    order.insert(order.index(host) + 1, d)
                out.append({"comp": new, "formula": to_formula(new, order), "host": host, "dopant": d, "fraction": f, "x": x,
                            "site": "anion" if q < 0 else "cation", "kind": kind, "li_change": round(dli, 4),
                            "host_charge": q, "dopant_charge": qd, "fit": fit})
    return out


def advise(reg, formula: str, space_group=None, top: int = 10, allow_tm: bool = False, avoid_toxic: bool = True,
           avoid_supply_risk: bool = False, ref=None, bundle=None, n_struct: int = 20) -> Dict:
    """Rank every sensible doping of ``formula``.

    1. Generate charge-neutral substitutions (hard charge and ion-size filters).
    2. Look every variant up in Materials Project and OBELiX: known compounds are listed separately as
       benchmarks and never ranked against hypothetical ones.
    3. Predict all hypothetical variants from composition, then build a crystal structure for the best
       ``n_struct`` on the base compound's real crystal and re-score them through the structure path
       (real descriptors, volume per atom, logistic regression).
    4. Give each a feasibility badge; "unlikely to exist" variants are dropped from the ranking."""
    from .insight import FormulaError, parse_checked_formula, verdict
    from .structgen import assess, lookup, template_by_id

    comp = parse_checked_formula(formula)
    charges = assign_charges(comp)
    if not charges or charges.get("_imbalance", 1) > 0.05:
        raise FormulaError("The base formula is not charge-balanced, so it cannot be doped sensibly. Check the formula.")
    base_a = assess(reg, formula, space_group, None, ref, bundle)
    base = base_a["sigma"]
    gen_b = base_a["structure"]
    host = template_by_id(gen_b["host_id"]) if gen_b and gen_b.get("ok") else None
    rejected: Dict[str, int] = {}
    variants = generate_variants(comp, allow_tm, avoid_toxic, avoid_supply_risk, rejected=rejected)
    result = {"formula": formula, "pretty": pretty(formula), "space_group": space_group or base_a["space_group_used"], "base": base,
              "base_assessment": {"existence": base_a["existence"], "feasibility": base_a["feasibility"],
                                  "confidence": base_a["confidence"], "structure_ok": bool(gen_b and gen_b.get("ok")),
                                  "host_id": gen_b.get("host_id") if gen_b else None},
              "base_family": chem_family(comp), "n_variants": len(variants), "charges": charges, "suggestions": [], "known": [],
              "rejected": rejected, "host_id": host.id if host else None, "n_struct": 0}
    if not variants:
        return result
    pred = predict_many(reg, [v["comp"] for v in variants], space_group or base_a["space_group_used"] or None)
    for v, (_, p) in zip(variants, pred.iterrows()):
        v["log10_sigma_comp"] = float(p["log10_sigma_pred"])
        v["exist"] = lookup(v["comp"])
    known = [v for v in variants if v["exist"]["status"] != "hypothetical"]
    hypo = sorted([v for v in variants if v["exist"]["status"] == "hypothetical"], key=lambda v: -v["log10_sigma_comp"])
    sg_used = space_group or base_a["space_group_used"] or None
    scored, unlikely = [], 0
    limit = n_struct if host is not None else min(n_struct, 8)
    for v in hypo[:limit]:
        a = assess(reg, v["formula"], sg_used, formula, ref, bundle, host_template=host)
        if a["feasibility"]["key"] == "unlikely":
            unlikely += 1
            continue
        scored.append((v, a))
    rejected["site"] = unlikely
    result["n_struct"] = len(scored)
    # plausible before strained, then by structure-path conductivity
    scored.sort(key=lambda t: (t[1]["feasibility"]["key"] != "hypo_plausible", -t[1]["sigma"]["log10_sigma"], t[0]["fraction"]))
    b_ls = base["log10_sigma"]
    for rank, (v, a) in enumerate(scored[:top], 1):
        s = a["sigma"]
        result["suggestions"].append({
            "rank": rank, "formula": v["formula"], "pretty": pretty(v["formula"]), "host": v["host"], "dopant": v["dopant"],
            "fraction": v["fraction"], "x": v["x"], "site": v["site"], "kind": v["kind"], "li_change": v["li_change"],
            "log10_sigma": s["log10_sigma"], "sigma_text": s["sigma_text"], "sigma_mS": 10 ** s["log10_sigma"] * 1000,
            "lo_text": s["lo_text"], "hi_text": s["hi_text"], "p_trees": s["p_trees_superionic"], "p_logreg": a["p_logreg"],
            "log10_sigma_comp": v["log10_sigma_comp"], "structure_scored": bool(s.get("has_structure")),
            "gain": 10 ** (s["log10_sigma"] - b_ls), "tier": tier(s["log10_sigma"]), "flags": industry_flags(v["comp"]),
            "fit": v["fit"], "badge": a["feasibility"], "confidence": a["confidence"],
            "verdict": verdict(a["p_logreg"], s["log10_sigma"], s.get("sigma_pct")), "measured": None,
        })
    for v in sorted(known, key=lambda v: -v["log10_sigma_comp"])[:8]:
        a = assess(reg, v["formula"], sg_used, None, ref, bundle)
        ob = v["exist"]["obelix"]
        result["known"].append({"formula": v["formula"], "pretty": pretty(v["formula"]), "dopant": v["dopant"], "host": v["host"],
                                "fraction": v["fraction"], "badge": a["feasibility"], "sigma_text": a["sigma"]["sigma_text"],
                                "log10_sigma": a["sigma"]["log10_sigma"],
                                "measured": ob[0]["sigma"] if ob else None, "obelix_id": ob[0]["id"] if ob else None,
                                "mp_id": v["exist"]["best_mp"]["id"] if v["exist"]["best_mp"] else None,
                                "e_hull": v["exist"]["best_mp"]["e_hull"] if v["exist"]["best_mp"] else None})
    if result["suggestions"]:
        plaus = [x for x in result["suggestions"] if x["badge"]["key"] == "hypo_plausible"]
        result["best"] = (plaus or result["suggestions"])[0]
        result["n_better"] = int(sum(1 for _, a in scored if a["sigma"]["log10_sigma"] > b_ls))
    result["n_hypothetical"] = len(hypo)
    result["all"] = [{"formula": x["formula"], "dopant": x["dopant"], "host": x["host"], "fraction": x["fraction"],
                      "log10_sigma": x["log10_sigma"]} for x in result["suggestions"]]
    return result


# ---------------------------------------------------------------- Synthesis Planner
def binary(cation: str, q: int, anion: str, a: int) -> Tuple[str, Dict[str, float]]:
    """Charge-balanced binary compound, e.g. (P, 5, S, -2) -> P2S5."""
    a = abs(a)
    g = math.gcd(q, a)
    nc, na = a // g, q // g
    comp = {cation: float(nc), anion: float(na)}
    return f"{cation}{'' if nc == 1 else nc}{anion}{'' if na == 1 else na}", comp


def molar_mass(comp: Dict[str, float]) -> float:
    return sum(MASS.get(e, 0.0) * n for e, n in comp.items())


def synthesis_plan(formula: str, batch_g: float = 1.0) -> Dict:
    """Binary precursors, molar ratio and grams for a solid-state synthesis of ``formula``."""
    from .insight import parse_checked_formula

    comp = parse_checked_formula(formula)
    charges = assign_charges(comp)
    if not charges:
        return {"ok": False, "reason": "Could not assign oxidation states to this formula."}
    anions = [e for e in comp if charges.get(e, 0) < 0]
    if not anions:
        return {"ok": False, "reason": "No anion found."}
    main = max(anions, key=lambda a: comp[a] * abs(charges[a]))
    cands: Dict[str, Dict[str, float]] = {}
    for e in comp:
        q = charges.get(e, 0)
        if q <= 0:
            continue
        partners = anions if e == "Li" else [main]
        for an in partners:
            name, c = binary(e, q, an, charges[an])
            cands[name] = c
    # every minor anion also needs a source: if no Li-X exists, pair it with the first cation
    for an in anions:
        if not any(an in c for c in cands.values()):
            e = next(x for x in comp if charges.get(x, 0) > 0)
            name, c = binary(e, charges[e], an, charges[an])
            cands[name] = c
    names = list(cands)
    els = list(comp)
    A = np.array([[cands[n].get(e, 0.0) for n in names] for e in els])
    b = np.array([comp[e] for e in els])
    x, res = nnls(A, b)
    rel = res / max(np.linalg.norm(b), 1e-9)
    used = [(n, float(v)) for n, v in zip(names, x) if v > 1e-6]
    mass_total = sum(v * molar_mass(cands[n]) for n, v in used)
    target_mass = molar_mass(comp)
    rows = []
    for n, v in used:
        mm = molar_mass(cands[n])
        grams = batch_g * v * mm / target_mass
        rows.append({"precursor": n, "pretty": pretty(n), "moles_per_fu": round(v, 4), "molar_mass": round(mm, 2),
                     "grams": round(grams, 4), "mass_pct": round(100 * v * mm / mass_total, 1)})
    rows.sort(key=lambda r: -r["grams"])
    eq = " + ".join(f"{fmt_amount(r['moles_per_fu']) or '1'} {r['precursor']}" for r in rows) + f"  →  {formula}"
    fam = chem_family(comp)
    route = {"sulfide": "Ball-mill the precursors under argon (e.g. 500 rpm, 10–20 h), then anneal at 450–550 °C in a sealed quartz tube.",
             "selenide/telluride": "Mill under argon and anneal in a sealed, evacuated quartz ampoule.",
             "halide": "Mill under argon, then anneal at 250–550 °C in a sealed ampoule; handle in a dry room.",
             "oxide": "Mix and calcine in air (e.g. 900 °C), then sinter the pellet (1,100–1,230 °C for garnets) with a little excess Li to offset Li loss.",
             "nitride": "Handle under argon; react the nitride precursors at moderate temperature in a sealed container."}.get(
        fam, "Mix stoichiometrically under an inert atmosphere and anneal; confirm the phase by XRD.")
    return {"ok": bool(rel < 1e-6), "formula": formula, "pretty": pretty(formula), "batch_g": batch_g, "rows": rows, "equation": eq,
            "balanced": bool(rel < 1e-6), "residual": float(rel), "family": fam, "route": route,
            "molar_mass": round(target_mass, 2),
            "moisture_sensitive": fam in ("sulfide", "selenide/telluride", "nitride", "halide", "phosphide")}
