"""The five Sendek et al. structural descriptors, computed from a crystal
structure with periodic boundary conditions and partial occupancy.

Reference: A. D. Sendek et al., Energy Environ. Sci. 10, 306 (2017) and
Chem. Mater. 31, 342 (2019), eq. 1:

    P(superionic) = 1 / (1 + exp(-(0.18 LLB - 4.01 SBI - 0.47 AFC
                                   + 8.70 LASD - 2.17 LLSD - 6.56)))

The papers define the descriptors in words; they do not publish code. The
operational definitions below are ours and are written down precisely so
every number in CrystalScout is reproducible (see docs/METHODS.md):

LLB   Li-Li bond number. Expected number of Li neighbours of a Li site within
      ``LL_BOND_CUTOFF`` (3.0 A), averaged over Li sites.
SBI   Sublattice bond ionicity. Mean Pauling ionicity 1 - exp(-dchi^2/4) of
      the framework (non-Li cation -> anion) bonds. For Li binaries with no
      framework cation the Li-anion bonds are used.
AFC   Anion framework coordination. Mean number of anion neighbours of an
      anion inside the anion sublattice, counted up to ``AFC_SHELL`` (1.12) x
      the mean of its 4 shortest anion-anion contacts (bcc -> 8,
      fcc/hcp -> 12; robust to one anomalously short contact).
LASD  Li-anion separation distance. Mean Li-anion bond length over each Li
      site's first coordination shell (anions within ``SHELL_TOL`` x the
      nearest Li-anion distance).
LLSD  Li-Li separation distance. Mean distance from each Li site to its
      nearest Li neighbour.

Partial occupancy is handled as an *expectation*: every site-level quantity
is weighted by the occupancy of the relevant element on that site, and
neighbour counts add the neighbour's occupancy rather than 1. Pairs of sites
closer than ``SPLIT_SITE_MIN`` (1.6 A for Li-Li, 1.5 A otherwise) are
alternative positions of one atom (split sites) and are never counted as
neighbours.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .chemistry import bond_ionicity, identify_anions
from .structure import Structure

FEATURES = ["LLB", "SBI", "AFC", "LASD", "LLSD"]

FEATURE_INFO = {
    "LLB": {"name": "Li-Li bond number", "unit": "", "desc": "How many Li neighbours each Li has within 3 Å — connectivity of the Li sublattice."},
    "SBI": {"name": "Sublattice bond ionicity", "unit": "", "desc": "Average ionic character of the framework (non-Li) bonds; softer, more covalent frameworks tend to help Li hop."},
    "AFC": {"name": "Anion framework coordination", "unit": "", "desc": "How many anion neighbours each anion has; bcc-like frameworks (8) are favoured over close-packed ones (12)."},
    "LASD": {"name": "Li-anion separation distance", "unit": "Å", "desc": "Average Li–anion bond length; longer, weaker bonds make Li easier to move."},
    "LLSD": {"name": "Li-Li separation distance", "unit": "Å", "desc": "Distance from each Li to its nearest Li; shorter hops favour fast transport."},
}

LL_BOND_CUTOFF = 3.0
AFC_SHELL = 1.12
AFC_REF_K = 4
SHELL_TOL = 1.25
SPLIT_SITE_MIN = 1.5
LI_SPLIT_MIN = 1.6
SEARCH_CUTOFF = 7.0
MIN_BOND = 1.2  # shorter cation-anion contacts are split-site artefacts


@dataclass
class DescriptorResult:
    LLB: float
    SBI: float
    AFC: float
    LASD: float
    LLSD: float
    n_sites: int
    n_li_sites: int
    anions: List[str]
    framework_cations: List[str]
    disordered: bool
    warnings: List[str] = field(default_factory=list)

    def vector(self) -> np.ndarray:
        return np.array([self.LLB, self.SBI, self.AFC, self.LASD, self.LLSD], dtype=float)

    def as_dict(self) -> Dict:
        return asdict(self)


class DescriptorError(ValueError):
    pass


def _wmean(values, weights) -> Optional[float]:
    values = np.asarray(values, float)
    weights = np.asarray(weights, float)
    if values.size == 0 or weights.sum() <= 0:
        return None
    return float(np.sum(values * weights) / np.sum(weights))


def compute_descriptors(st: Structure) -> DescriptorResult:
    warnings: List[str] = []
    elements = st.elements
    if "Li" not in elements:
        raise DescriptorError("structure contains no Li")
    anions = identify_anions(elements)
    if not anions:
        raise DescriptorError("could not identify an anion species")
    framework = [e for e in elements if e != "Li" and e not in anions]

    li_occ = np.array([s.occ("Li") for s in st.sites])
    an_occ = np.array([sum(s.occ(a) for a in anions) for s in st.sites])
    fw_occ = np.array([sum(s.occ(c) for c in framework) for s in st.sites])
    li_idx = np.nonzero(li_occ > 1e-6)[0]
    an_idx = np.nonzero(an_occ > 1e-6)[0]
    fw_idx = np.nonzero(fw_occ > 1e-6)[0]
    if len(an_idx) == 0:
        raise DescriptorError("no anion sites")

    nn = st.neighbors(SEARCH_CUTOFF)

    # ---- Li-centred quantities: LLB, LLSD, LASD
    llb, llsd, lasd, w_llsd, w_lasd = [], [], [], [], []
    for i in li_idx:
        j, d = nn[i]
        # Li neighbours (ignore split partners)
        lm = (li_occ[j] > 1e-6) & (d >= LI_SPLIT_MIN)
        llb.append(float(np.sum(li_occ[j][lm & (d <= LL_BOND_CUTOFF)])))
        if lm.any():
            llsd.append(float(d[lm].min()))
            w_llsd.append(li_occ[i])
        am = (an_occ[j] > 1e-6) & (d >= MIN_BOND)
        if am.any():
            dmin = d[am].min()
            shell = am & (d <= SHELL_TOL * dmin)
            lasd.append(_wmean(d[shell], an_occ[j][shell]))
            w_lasd.append(li_occ[i])
    LLB = _wmean(llb, li_occ[li_idx])
    LLSD = _wmean(llsd, w_llsd)
    LASD = _wmean(lasd, w_lasd)
    if LLSD is None:
        LLSD = SEARCH_CUTOFF
        warnings.append(f"no Li-Li pair within {SEARCH_CUTOFF} Å; LLSD capped")
    if LASD is None:
        raise DescriptorError("no Li-anion bonds found")

    # ---- SBI: framework cation -> anion bond ionicity
    vals, wts = [], []
    centres = fw_idx if len(fw_idx) else li_idx
    if not len(fw_idx):
        warnings.append("no framework cation; SBI uses Li-anion bonds")
    for i in centres:
        j, d = nn[i]
        am = (an_occ[j] > 1e-6) & (d >= MIN_BOND * 0.8)
        if not am.any():
            continue
        dmin = d[am].min()
        shell = am & (d <= SHELL_TOL * dmin)
        s_i = st.sites[i]
        cations = framework if len(fw_idx) else ["Li"]
        for jj in j[shell]:
            s_j = st.sites[jj]
            for c in cations:
                oc = s_i.occ(c)
                if oc <= 0:
                    continue
                for a in anions:
                    oa = s_j.occ(a)
                    if oa > 0:
                        vals.append(bond_ionicity(c, a))
                        wts.append(oc * oa)
    SBI = _wmean(vals, wts)
    if SBI is None:
        raise DescriptorError("no framework bonds found")

    # ---- AFC: anion-anion coordination inside the anion sublattice
    afc, w_afc = [], []
    for i in an_idx:
        j, d = nn[i]
        am = (an_occ[j] > 1e-6) & (d >= SPLIT_SITE_MIN)
        if not am.any():
            continue
        # robust shell anchor: mean of the AFC_REF_K shortest anion-anion
        # contacts, so a single anomalously short contact (a common artefact
        # of disordered / perturbed CIFs) cannot collapse the shell.
        dref = float(np.mean(np.sort(d[am])[:AFC_REF_K]))
        shell = am & (d <= AFC_SHELL * dref)
        afc.append(float(np.sum(an_occ[j][shell])))
        w_afc.append(an_occ[i])
    AFC = _wmean(afc, w_afc)
    if AFC is None:
        AFC = 0.0
        warnings.append("isolated anions; AFC set to 0")

    return DescriptorResult(
        LLB=round(LLB, 4), SBI=round(SBI, 4), AFC=round(AFC, 4), LASD=round(LASD, 4), LLSD=round(LLSD, 4),
        n_sites=len(st), n_li_sites=int(len(li_idx)), anions=sorted(anions), framework_cations=sorted(framework),
        disordered=not st.is_ordered, warnings=warnings,
    )


# ----------------------------------------------------------------- published
SENDEK_2019_COEFS = {"LLB": 0.18, "SBI": -4.01, "AFC": -0.47, "LASD": 8.70, "LLSD": -2.17}
SENDEK_2019_INTERCEPT = -6.56


def sendek_logit(x: Dict[str, float]) -> float:
    return SENDEK_2019_INTERCEPT + sum(SENDEK_2019_COEFS[k] * float(x[k]) for k in FEATURES)


def sendek_probability(x: Dict[str, float]) -> float:
    z = sendek_logit(x)
    return float(1.0 / (1.0 + np.exp(-z)))
