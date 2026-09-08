"""Unit tests for openmm_localcwld.metadata.build_particle_parameters().

These tests build minimal synthetic OpenMM System/Topology pairs by hand
(no PDBFixer, no force field XML, no PDB file) so they can run anywhere
OpenMM itself is installed. They do NOT reproduce the full LCWLD-020
completion definition from the plan (section 24 / DEC-020), which requires
comparing against test_lips_vs_pmeV2.6.py's build_phase_cwld_metadata() on
the actual 1CKK system -- that comparison needs PDBFixer, the Amber19SB/
TIP3P force field files, and 1CKK.pdb, none of which this test module
depends on. See openmm-localcwld/docs/decisions/DEC-001-toolchain.md for
why that cross-check has to run in the real `openmm_dev` environment
rather than here.

Run with:
    python -m pytest openmm-localcwld/python/tests/test_metadata.py -v
"""

import math

import numpy as np
import pytest

mm = pytest.importorskip("openmm")
from openmm import app, unit  # noqa: E402

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from openmm_localcwld import metadata as md  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to build minimal synthetic System/Topology pairs.
# ---------------------------------------------------------------------------

def _element(symbol):
    # app.Element is the usual public alias; fall back to the element
    # submodule in case a given OpenMM version doesn't re-export it at the
    # app package's top level.
    try:
        return app.Element.getBySymbol(symbol)
    except AttributeError:
        from openmm.app import element as _element_module
        return _element_module.Element.getBySymbol(symbol)


def make_system(residue_specs):
    """residue_specs: list of (resname, [(atom_name, element_symbol, qbase), ...])

    Returns (system, topology, index_by_atom_name) where index_by_atom_name
    maps 'RESNAME_INDEX.ATOMNAME' -> global particle index (residue index is
    included in the key so repeated residue names/atom names don't collide).
    """
    topology = app.Topology()
    chain = topology.addChain()
    system = mm.System()
    nb = mm.NonbondedForce()
    system.addForce(nb)

    index_by_key = {}
    for res_idx, (resname, atoms) in enumerate(residue_specs):
        residue = topology.addResidue(resname, chain)
        for atom_name, element_symbol, qbase in atoms:
            elem = _element(element_symbol)
            atom = topology.addAtom(atom_name, elem, residue)
            idx = system.addParticle(elem.mass)
            nb.addParticle(qbase * unit.elementary_charge, 0.3 * unit.nanometer, 0.5 * unit.kilojoule_per_mole)
            assert atom.index == idx
            index_by_key[f"{resname}_{res_idx}.{atom_name}"] = idx

    assert system.getNumParticles() == topology.getNumAtoms()
    return system, topology, index_by_key


# ---------------------------------------------------------------------------
# Water
# ---------------------------------------------------------------------------

def _tip3p_water_specs(n=1, o_charge=-0.834, h_charge=0.417):
    specs = []
    for _ in range(n):
        specs.append(("HOH", [("O", "O", o_charge), ("H1", "H", h_charge), ("H2", "H", h_charge)]))
    return specs


def test_water_response_on():
    system, topology, idx = make_system(_tip3p_water_specs(1))
    params = md.build_particle_parameters(system, topology, enable_water_response=True)

    o = idx["HOH_0.O"]
    h1 = idx["HOH_0.H1"]
    h2 = idx["HOH_0.H2"]

    assert params.dens_source[o] == 1.0
    assert params.source_class_weight[o] == md.WATER_SOURCE_WEIGHT
    assert params.dens_source[h1] == 0.0 and params.dens_source[h2] == 0.0

    assert params.dpolar[o] == pytest.approx(-0.15)
    assert params.dpolar[h1] == pytest.approx(0.075)
    assert params.dpolar[h2] == pytest.approx(0.075)
    for i in (o, h1, h2):
        assert params.is_polar[i] == 1.0
        assert params.dens_sink[i] == 1.0
        assert params.static_phase[i] == 1.0


def test_water_response_off_still_source():
    system, topology, idx = make_system(_tip3p_water_specs(1))
    params = md.build_particle_parameters(system, topology, enable_water_response=False)

    o = idx["HOH_0.O"]
    h1 = idx["HOH_0.H1"]

    # dens_source/source_class_weight for water O are set unconditionally in
    # v2.6 (not gated by enable_water_response) -- only the response fields
    # (dpolar/is_polar/dens_sink/static_phase) are gated.
    assert params.dens_source[o] == 1.0
    assert params.source_class_weight[o] == md.WATER_SOURCE_WEIGHT
    assert params.dpolar[o] == 0.0
    assert params.is_polar[o] == 0.0
    assert params.dens_sink[o] == 0.0
    assert params.static_phase[o] == 0.0
    assert params.dens_source[h1] == 0.0


def test_malformed_water_missing_hydrogen_is_not_a_source():
    # Only one H atom in the residue -> v2.6's `len(h_atoms) == 2` gate fails,
    # so the O atom never gets dens_source=1 even though it is a water O.
    system, topology, idx = make_system([("HOH", [("O", "O", -0.834), ("H1", "H", 0.417)])])
    params = md.build_particle_parameters(system, topology)
    o = idx["HOH_0.O"]
    assert params.dens_source[o] == 0.0
    assert params.source_class_weight[o] == 0.0


# ---------------------------------------------------------------------------
# Ions
# ---------------------------------------------------------------------------

def test_ion_classification_and_alias_resolution_without_mutation():
    system, topology, idx = make_system([
        ("NA", [("NA", "Na", 1.0)]),
        ("CL", [("CL", "Cl", -1.0)]),
        ("CA", [("CA", "Ca", 2.0)]),
    ])
    original_names = [r.name for r in topology.residues()]

    params = md.build_particle_parameters(system, topology, ca_source_weight=1.5)

    na = idx["NA_0.NA"]
    cl = idx["CL_1.CL"]
    ca = idx["CA_2.CA"]

    assert params.dens_source[na] == 1.0
    assert params.source_class_weight[na] == md.ION_SOURCE_WEIGHT
    assert params.dens_source[cl] == 1.0
    assert params.source_class_weight[cl] == md.ION_SOURCE_WEIGHT
    assert params.dens_source[ca] == 1.0
    assert params.source_class_weight[ca] == 1.5  # overridden ca_source_weight, not the 2.0 default

    # ions are never response sinks / polar in this model
    for i in (na, cl, ca):
        assert params.dens_sink[i] == 0.0
        assert params.is_polar[i] == 0.0

    # the caller's Topology must be untouched (non-mutating alias resolution)
    assert [r.name for r in topology.residues()] == original_names == ["NA", "CL", "CA"]


def test_ca_source_weight_default_matches_ion_source_weight():
    system, topology, idx = make_system([("CA", [("CA", "Ca", 2.0)])])
    params = md.build_particle_parameters(system, topology)
    ca = idx["CA_0.CA"]
    assert params.source_class_weight[ca] == md.DEFAULT_CA_SOURCE_WEIGHT == 2.0


# ---------------------------------------------------------------------------
# Protein solute: ASP charged sidechain vs. plain backbone, FAST_ACTIVE_DENSITY
# ---------------------------------------------------------------------------

def _asp_specs():
    return [(
        "ASP",
        [
            ("N", "N", -0.4),
            ("CA", "C", 0.0),
            ("C", "C", 0.6),
            ("O", "O", -0.6),
            ("CB", "C", -0.2),
            ("OD1", "O", -0.7),
            ("OD2", "O", -0.7),
        ],
    )]


def test_asp_charged_sidechain_gets_charged_weight():
    system, topology, idx = make_system(_asp_specs())
    params = md.build_particle_parameters(system, topology)

    od1 = idx["ASP_0.OD1"]
    od2 = idx["ASP_0.OD2"]
    backbone_o = idx["ASP_0.O"]
    backbone_n = idx["ASP_0.N"]

    for i in (od1, od2):
        assert params.source_class_weight[i] == md.CHARGED_SOURCE_WEIGHT
        assert params.dpolar[i] == pytest.approx(md.SOLUTE_DPOLAR_BY_ELEMENT["O"])
        assert params.is_polar[i] == 1.0
        assert params.dens_sink[i] == 1.0
        assert params.static_phase[i] == 1.0
        assert params.dens_source[i] == 1.0

    # backbone O is a heavy O but not a named charged site -> plain polar weight
    assert params.source_class_weight[backbone_o] == md.POLAR_SOURCE_WEIGHT
    assert params.dpolar[backbone_o] == pytest.approx(md.SOLUTE_DPOLAR_BY_ELEMENT["O"])

    # backbone N is O/N/S -> included, plain polar weight
    assert params.source_class_weight[backbone_n] == md.POLAR_SOURCE_WEIGHT
    assert params.dpolar[backbone_n] == pytest.approx(md.SOLUTE_DPOLAR_BY_ELEMENT["N"])


def test_fast_active_density_default_excludes_carbon():
    system, topology, idx = make_system(_asp_specs())
    ca_atom = idx["ASP_0.CA"]
    cb_atom = idx["ASP_0.CB"]

    params_default = md.build_particle_parameters(system, topology)  # fast_active_density=True
    for i in (ca_atom, cb_atom):
        assert params_default.dens_source[i] == 0.0
        assert params_default.dpolar[i] == 0.0
        assert params_default.is_polar[i] == 0.0
        assert params_default.source_class_weight[i] == 0.0

    params_full = md.build_particle_parameters(system, topology, fast_active_density=False)
    for i in (ca_atom, cb_atom):
        # carbon becomes a (non-response) density source once the gate is lifted
        assert params_full.dens_source[i] == 1.0
        assert params_full.source_class_weight[i] == md.POLAR_SOURCE_WEIGHT
        assert params_full.dpolar[i] == 0.0  # SOLUTE_DPOLAR_BY_ELEMENT["C"] == 0.0
        assert params_full.is_polar[i] == 0.0
        assert params_full.dens_sink[i] == 0.0  # is_polar is 0 -> never a sink


def test_enable_solute_polarization_false_zeroes_solute_source_too():
    """Documents a real discrepancy between the plan's prose (section 17.3:
    "enable_solute_polarization=False 时，溶质可以继续是 source") and the
    actual v2.6 reference code, where the entire per-atom solute block
    (including dens_source assignment) is nested inside
    `if enable_solute_polarization:`. This module reproduces the v2.6 CODE
    exactly (per plan section 5's "reproduce, not reinvent" mandate), so with
    enable_solute_polarization=False a solute heavy atom is NOT a density
    source either. See PLAN-ERRATA appended to
    PLAN_LocalCWLDForce_OpenMM_Plugin.md section 54 -- this is flagged, not
    silently resolved."""
    system, topology, idx = make_system(_asp_specs())
    od1 = idx["ASP_0.OD1"]
    params = md.build_particle_parameters(system, topology, enable_solute_polarization=False)
    assert params.dens_source[od1] == 0.0
    assert params.dpolar[od1] == 0.0
    assert params.is_polar[od1] == 0.0
    assert params.dens_sink[od1] == 0.0
    assert params.source_class_weight[od1] == 0.0


# ---------------------------------------------------------------------------
# Lipid headgroup
# ---------------------------------------------------------------------------

def test_lipid_headgroup_source_and_tail_inert():
    system, topology, idx = make_system([(
        "POPC",
        [
            ("P", "P", 1.0),
            ("O11", "O", -0.5),
            ("N", "N", 0.3),
            ("C1", "C", -0.1),
            ("C2", "C", -0.1),
        ],
    )])
    params = md.build_particle_parameters(system, topology)

    p = idx["POPC_0.P"]
    o = idx["POPC_0.O11"]
    n = idx["POPC_0.N"]
    c1 = idx["POPC_0.C1"]

    for i in (p, o, n):
        assert params.dens_source[i] == 1.0
        assert params.source_class_weight[i] == md.LIPID_HEADGROUP_SOURCE_WEIGHT

    # tails and, in general, lipids never become sinks/polar in this model
    assert params.dens_source[c1] == 0.0
    for i in (p, o, n, c1):
        assert params.dens_sink[i] == 0.0
        assert params.is_polar[i] == 0.0


# ---------------------------------------------------------------------------
# charge_mod / qref2
# ---------------------------------------------------------------------------

def test_charge_mod_qref2_uses_water_oxygen_only():
    o_charge = -0.834
    h_charge = 0.417
    system, topology, idx = make_system(
        _tip3p_water_specs(2, o_charge=o_charge, h_charge=h_charge)
        + [("NA", [("NA", "Na", 1.0)])]
    )
    a_q2 = 0.5
    params = md.build_particle_parameters(system, topology, a_q2=a_q2)

    qref2_expected = o_charge ** 2  # both waters have identical O charge here
    o0 = idx["HOH_0.O"]
    expected_charge_mod_o = np.clip(1.0 + a_q2 * (o_charge ** 2 - qref2_expected), md.CHARGE_MOD_MIN, md.CHARGE_MOD_MAX)
    assert params.charge_mod[o0] == pytest.approx(expected_charge_mod_o)

    na = idx["NA_2.NA"]
    expected_charge_mod_na = np.clip(1.0 + a_q2 * (1.0 ** 2 - qref2_expected), md.CHARGE_MOD_MIN, md.CHARGE_MOD_MAX)
    assert params.charge_mod[na] == pytest.approx(expected_charge_mod_na)


def test_charge_mod_clamped_to_bounds():
    # Large a_q2 and a wide charge spread should hit both clamp bounds.
    system, topology, idx = make_system(
        _tip3p_water_specs(1, o_charge=-0.834, h_charge=0.417)
        + [("NA", [("NA", "Na", 5.0)])]  # unrealistically large charge, on purpose
    )
    params = md.build_particle_parameters(system, topology, a_q2=5.0)
    na = idx["NA_1.NA"]
    assert params.charge_mod[na] == md.CHARGE_MOD_MAX
    assert np.all(params.charge_mod >= md.CHARGE_MOD_MIN)
    assert np.all(params.charge_mod <= md.CHARGE_MOD_MAX)


# ---------------------------------------------------------------------------
# residue_id and general invariants
# ---------------------------------------------------------------------------

def test_residue_id_is_residue_index():
    system, topology, idx = make_system(_tip3p_water_specs(3))
    params = md.build_particle_parameters(system, topology)
    for res_idx, residue in enumerate(topology.residues()):
        for atom in residue.atoms():
            assert params.residue_id[atom.index] == res_idx


def test_ligand_charged_weight_branch(monkeypatch):
    """CHARGED_SIDECHAIN_POLAR_ATOMS only has ARG/LYS/ASP/GLU/HIS as keys, all
    of which are also in AMINO_ACIDS, so the `else: LIGAND_CHARGED_SOURCE_WEIGHT`
    branch in build_particle_parameters is unreachable with the frozen
    constants as shipped. Exercise it directly via monkeypatch so the branch
    itself is still covered by a test."""
    monkeypatch.setitem(md.CHARGED_SIDECHAIN_POLAR_ATOMS, "LIG", {"OX1"})
    system, topology, idx = make_system([("LIG", [("OX1", "O", -0.5), ("CX", "C", 0.1)])])
    params = md.build_particle_parameters(system, topology)
    ox1 = idx["LIG_0.OX1"]
    assert params.source_class_weight[ox1] == md.LIGAND_CHARGED_SOURCE_WEIGHT


def test_no_mutation_of_inputs():
    system, topology, idx = make_system(_tip3p_water_specs(1) + [("NA", [("NA", "Na", 1.0)])])
    n_particles_before = system.getNumParticles()
    names_before = [r.name for r in topology.residues()]
    md.build_particle_parameters(system, topology)
    assert system.getNumParticles() == n_particles_before
    assert [r.name for r in topology.residues()] == names_before


def test_particle_parameters_shape_validation_and_tuple_order():
    system, topology, idx = make_system(_tip3p_water_specs(1))
    params = md.build_particle_parameters(system, topology)
    assert params.num_particles == 3

    tuples = list(params.as_particle_tuples())
    assert len(tuples) == 3
    for t in tuples:
        assert len(t) == 9
        # last field (residue_id) must be int, everything else float
        assert isinstance(t[-1], int)
        for v in t[:-1]:
            assert isinstance(v, float)

    with pytest.raises(ValueError):
        md.ParticleParameters(
            qbase=np.zeros(3),
            charge_mod=np.zeros(2),  # wrong length on purpose
            dpolar=np.zeros(3),
            is_polar=np.zeros(3),
            dens_source=np.zeros(3),
            dens_sink=np.zeros(3),
            source_class_weight=np.zeros(3),
            static_phase=np.zeros(3),
            residue_id=np.zeros(3, dtype=np.int64),
        )
