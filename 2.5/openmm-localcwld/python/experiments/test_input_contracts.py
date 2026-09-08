"""Desired input contracts: known failures, deliberately NOT xfailed.

Run separately from numerical experiments to reproduce existing validation
gaps. These tests require a fix to production validation before they pass;
passing algebra experiments does not close these issues or any release gate.
"""

import numpy as np
import openmm as mm
from openmm import app
import pytest

from localcwld_reference import Globals, ParticleArrays, compute_local_cwld
from openmm_localcwld import metadata as md


def _system(num_particles=2, num_nonbonded=1):
    system = mm.System()
    for _ in range(num_particles):
        system.addParticle(1.0)
    for _ in range(num_nonbonded):
        force = mm.NonbondedForce()
        for _ in range(num_particles):
            force.addParticle(1.0, 0.3, 0.0)
        system.addForce(force)
    return system


def _fixed_particles():
    return ParticleArrays(
        qbase=np.array([1.0, -1.0]), charge_mod=np.ones(2),
        dpolar=np.zeros(2), is_polar=np.zeros(2), dens_source=np.zeros(2),
        dens_sink=np.zeros(2), source_class_weight=np.zeros(2),
        static_phase=np.zeros(2), residue_id=np.arange(2),
    )


def test_metadata_rejects_topology_particle_count_mismatch():
    topology = app.Topology()
    residue = topology.addResidue("NA", topology.addChain())
    topology.addAtom("NA", app.Element.getBySymbol("Na"), residue)
    with pytest.raises(ValueError):
        md.build_particle_parameters(_system(2), topology)


@pytest.mark.parametrize("num_nonbonded", [0, 2], ids=["missing", "ambiguous"])
def test_metadata_requires_exactly_one_nonbonded_force(num_nonbonded):
    with pytest.raises(ValueError):
        md.extract_nonbonded_charges(_system(num_nonbonded=num_nonbonded))


def test_metadata_rejects_two_dimensional_arrays():
    arrays = {name: np.ones((2, 1)) for name in md.PARTICLE_PARAMETER_ORDER[:-1]}
    arrays["residue_id"] = np.arange(2)
    with pytest.raises(ValueError):
        md.ParticleParameters(**arrays)


@pytest.mark.parametrize("name", ["k_polar", "one_4pi_eps0", "q_penalty_strength"])
def test_globals_reject_nonfinite_constants(name):
    with pytest.raises(ValueError):
        Globals(r_env=0.35, rc=1.2, **{name: float("nan")})


def test_reference_rejects_nonfinite_positions():
    positions = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    with pytest.raises(ValueError):
        compute_local_cwld(positions, None, _fixed_particles(), [], Globals(r_env=0.35, rc=1.2))


@pytest.mark.parametrize("pair", [(-1, 0), (0, 2)], ids=["negative", "out-of-range"])
def test_reference_rejects_invalid_exclusion_indices(pair):
    positions = np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]])
    with pytest.raises(ValueError):
        compute_local_cwld(positions, None, _fixed_particles(), [pair], Globals(r_env=0.35, rc=1.2))
