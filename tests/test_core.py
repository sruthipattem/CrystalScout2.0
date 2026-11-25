"""Science-core tests: CIF parsing, structure geometry, descriptors."""
import math

import numpy as np
import pytest

from crystalscout.config import OBELIX_DIR
from crystalscout.core.chemistry import bond_ionicity, has_transition_metal, identify_anions
from crystalscout.core.cif import CIFError, parse_cif, parse_symop, read_cif
from crystalscout.core.descriptors import (FEATURES, DescriptorError, compute_descriptors, sendek_logit,
                                           sendek_probability)
from crystalscout.core.structure import Site, Structure, lattice_from_parameters


def rocksalt(a=5.14):
    fcc = [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]
    sites = [Site(np.array(p, float), {"Cl": 1.0}) for p in fcc]
    sites += [Site((np.array(p, float) + [.5, 0, 0]) % 1, {"Li": 1.0}) for p in fcc]
    return Structure(np.eye(3) * a, sites)


def test_lattice_from_parameters_volume():
    L = lattice_from_parameters(5, 6, 7, 90, 100, 90)
    assert math.isclose(abs(np.linalg.det(L)), 5 * 6 * 7 * math.sin(math.radians(100)), rel_tol=1e-9)


def test_symop_parsing():
    R, t = parse_symop("-x+1/2, y, z+1/4")
    assert np.allclose(R, np.diag([-1, 1, 1])) and np.allclose(t, [.5, 0, .25])
    R, t = parse_symop("x-y, x, z+1/6")
    assert np.allclose(R[0], [1, -1, 0]) and np.isclose(t[2], 1 / 6)


def test_rocksalt_descriptors_match_geometry():
    d = compute_descriptors(rocksalt(5.14))
    assert d.AFC == pytest.approx(12.0)           # fcc anion lattice
    assert d.LASD == pytest.approx(5.14 / 2, abs=1e-3)
    assert d.LLSD == pytest.approx(5.14 / math.sqrt(2), abs=1e-3)
    assert d.LLB == 0.0                            # Li-Li 3.63 A > 3.0 A cutoff
    assert d.SBI == pytest.approx(bond_ionicity("Li", "Cl"), abs=1e-3)


def test_bcc_anion_framework_gives_afc_8():
    sites = [Site(np.array([0, 0, 0.]), {"S": 1.0}), Site(np.array([.5, .5, .5]), {"S": 1.0}),
             Site(np.array([.5, .25, 0.]), {"Li": 0.5})]
    assert compute_descriptors(Structure(np.eye(3) * 4.2, sites)).AFC == pytest.approx(8.0)


def test_partial_occupancy_weights_neighbours():
    full = compute_descriptors(rocksalt())
    st = rocksalt()
    for s in st.sites:
        if "Li" in s.species:
            s.species = {"Li": 0.5}
    half = compute_descriptors(st)
    assert half.LASD == pytest.approx(full.LASD)   # geometry unchanged
    assert half.disordered and not full.disordered


def test_no_li_raises():
    st = Structure(np.eye(3) * 4, [Site(np.zeros(3), {"Na": 1.0}), Site(np.array([.5, .5, .5]), {"Cl": 1.0})])
    with pytest.raises(DescriptorError):
        compute_descriptors(st)


def test_anion_rules():
    assert identify_anions(["Li", "P", "S"]) == {"S"}
    assert identify_anions(["Li", "B", "H"]) == {"H"}          # borohydride
    assert identify_anions(["Li", "N", "H"]) == {"N"}          # amide: H is not an anion
    assert has_transition_metal(["Li", "Zr", "O"]) and not has_transition_metal(["Li", "La", "O"])


def test_sendek_equation():
    x = dict(LLB=1, SBI=0.5, AFC=6, LASD=2.2, LLSD=2.8)
    z = 0.18 - 4.01 * 0.5 - 0.47 * 6 + 8.70 * 2.2 - 2.17 * 2.8 - 6.56
    assert sendek_logit(x) == pytest.approx(z)
    assert sendek_probability(x) == pytest.approx(1 / (1 + math.exp(-z)))


def test_cif_round_trip_and_obelix_file():
    st = read_cif(OBELIX_DIR / "cifs" / "jqc.cif")       # Li7BiO6 from OBELiX
    comp = st.composition()
    assert comp["Li"] / comp["O"] == pytest.approx(7 / 6, rel=1e-3)
    again = parse_cif(st.to_cif())
    assert len(again) == len(st)
    a, b = compute_descriptors(st), compute_descriptors(again)
    for k in FEATURES:
        assert getattr(a, k) == pytest.approx(getattr(b, k), abs=1e-3)


def test_cif_with_uncertainties_and_symmetry():
    text = """data_test
_cell_length_a 4.0(1)
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_space_group_symop_operation_xyz
'x, y, z'
'x+1/2, y+1/2, z+1/2'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
S1 S2- 0.0 0.0 0.0(2) 1
Li1 Li1+ 0.5 0.25 0.0 0.5
"""
    st = parse_cif(text)
    assert st.composition()["S"] == pytest.approx(2)
    assert st.composition()["Li"] == pytest.approx(1.0)  # 2 positions x 0.5


def test_bad_cif():
    with pytest.raises(CIFError):
        parse_cif("data_x\n_cell_length_a 3\n")


def test_pymatgen_dict_roundtrip():
    st = rocksalt()
    st2 = Structure.from_pymatgen_dict(st.as_dict())
    assert compute_descriptors(st2).LASD == pytest.approx(compute_descriptors(st).LASD)
