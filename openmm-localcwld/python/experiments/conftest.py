"""Small deterministic systems: no Context, integration, or golden writes."""

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localcwld_reference import Globals, ParticleArrays  # noqa: E402


@dataclass
class ExperimentCase:
    positions: np.ndarray
    box: np.ndarray
    particles: ParticleArrays
    exclusions: list
    globals: Globals


@pytest.fixture(params=[1, 2, 3], ids=lambda order: f"ell{order}")
def order(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["no-penalty", "penalty"])
def penalty(request):
    return request.param


@pytest.fixture
def case(order, penalty):
    rng = np.random.default_rng(240831)
    box = np.array([[3.0, 0.0, 0.0], [0.5, 2.8, 0.0], [-0.3, 0.4, 2.7]])
    positions = rng.uniform(-0.22, 0.22, (8, 3))
    positions += rng.integers(-1, 2, (8, 3)) @ box
    particles = ParticleArrays(
        qbase=rng.uniform(-1.0, 1.0, 8),
        charge_mod=rng.uniform(0.5, 2.0, 8),
        dpolar=np.array([-0.15, 0.075, 0.0, 0.012, -0.075, 0.0, 0.015, 0.02]),
        is_polar=np.array([1, 1, 0, 1, 1, 0, 1, 1], dtype=float),
        dens_source=np.array([0, 1, 1, 0, 1, 1, 0, 1], dtype=float),
        dens_sink=np.array([1, 1, 0, 1, 1, 0, 1, 1], dtype=float),
        source_class_weight=rng.uniform(0.3, 2.0, 8),
        static_phase=np.ones(8),
        residue_id=np.arange(8) // 2,
    )
    return ExperimentCase(
        positions, box, particles, [(3, 0), (4, 7)],
        Globals(r_env=0.35, rc=1.2, zmm_order=order, use_q_penalty=penalty),
    )
