"""openmm_localcwld.metadata
============================

Standalone particle-classification / metadata builder for ``LocalCWLDForce``.

This module is a pure extraction of the particle-classification logic in
``test_lips_vs_pmeV2.6.py`` (``build_phase_cwld_metadata()``, roughly lines
242-379, plus its helpers ``extract_nonbonded_charges``,
``active_water_source_qref2`` and ``clamp_charge_mod``). Its job is to
*reproduce* v2.6's exact per-particle numbers for a given ``(System,
Topology)`` pair, not to redesign the classification. See
``PLAN_LocalCWLDForce_OpenMM_Plugin.md`` sections 17-18 for the frozen
semantics this module must not deviate from.

Frozen facts this module preserves (do not "fix" any of these without going
through a Decision Record / Errata, per the plan's section 15/54 protocol):

- ``residue_id`` is ``residue.index`` (v2.6 calls it ``mol_id``, but the
  actual value is the residue index, not a real molecule id).
- Water H atoms are never a density source. Water O only becomes a source
  (and, if enabled, a response sink) when the residue has exactly one atom
  with element O *and* exactly two atoms with element H. A malformed or
  virtual-site water residue that fails this shape check silently
  contributes nothing as a source -- this gate is implicit in v2.6 and is
  reproduced here exactly.
- ``qref2`` (used for ``charge_mod``) is the mean of ``qbase**2`` over
  *active water source* atoms (water O with ``dens_source>0``); if there are
  none, it falls back to all water atoms, then to the whole system.
- ``FAST_ACTIVE_DENSITY`` (default ``True`` in v2.6) restricts protein/ligand
  solute response *and* solute density-source designation to O/N/S heavy
  atoms plus the explicit charged-sidechain atom list; carbon and hydrogen
  never become density sources under the default. This gate is not named in
  the plan's frozen-semantics tables (sections 17.1/17.3), but it changes
  which solute atoms carry ``dens_source=1`` in the exact reference model.
  It is treated here as a frozen builder-time default matching v2.6, and is
  flagged in ``docs/decisions`` as something the plan's own Errata process
  should record explicitly rather than have an implementer "discover" and
  silently reinterpret.
- Ion residue classification uses the *normalized* resname
  (``ION_RESNAME_ALIASES``). v2.6 achieves this by mutating
  ``residue.name`` in place via ``normalize_ion_resnames()`` before calling
  the metadata builder. This module gets the identical classification
  result by resolving the alias locally instead, so callers do not need to
  pre-normalize the topology and the input ``Topology``/``System`` objects
  are never mutated. This is a non-mutating *implementation* detail, not a
  semantics change -- the resulting nine arrays are the same either way.

Deliberately NOT reproduced here (per plan section 17.3, these are analysis-
only / legacy-fast-path fields that do not enter ``LocalCWLDForce``):
``phase_ref_protein_ligand``, ``phase_ref_ion``, ``phase_ref_headgroup``,
``q_driver``, ``water_triplets``, and friends.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

try:
    import openmm as mm
    from openmm import unit
except ImportError as exc:  # pragma: no cover - exercised only without OpenMM installed
    raise ImportError(
        "openmm_localcwld.metadata requires an OpenMM installation "
        "(the 'openmm' Python package). This module only reads System/"
        "Topology objects and does not run any simulation."
    ) from exc


# ---------------------------------------------------------------------------
# Frozen constants, copied verbatim from test_lips_vs_pmeV2.6.py.
# Do not "tune" these here; changing a default value is a scientific-model
# change per plan section 44 and must go through a Decision Record.
# ---------------------------------------------------------------------------

DEFAULT_A_Q2 = 0.5
DEFAULT_CA_SOURCE_WEIGHT = 2.0
DEFAULT_DPOLAR_O = -0.15

CHARGE_MOD_MIN = 0.25
CHARGE_MOD_MAX = 2.5

WATER_SOURCE_WEIGHT = 0.5
ION_SOURCE_WEIGHT = 2.0
LIPID_HEADGROUP_SOURCE_WEIGHT = 0.3
POLAR_SOURCE_WEIGHT = 1.0
CHARGED_SOURCE_WEIGHT = 1.5
LIGAND_CHARGED_SOURCE_WEIGHT = 1.5

WATER_RESNAMES = {"HOH", "WAT", "SOL", "TP3"}

AMINO_ACIDS = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}

ION_RESNAME_ALIASES = {
    "NA": "Na+", "Na": "Na+", "Na+": "Na+",
    "CL": "Cl-", "Cl": "Cl-", "Cl-": "Cl-",
    "K": "K+", "K+": "K+",
    "CA": "Ca2+", "Ca": "Ca2+", "CAL": "Ca2+", "Ca2+": "Ca2+",
    "MG": "Mg2+", "Mg": "Mg2+", "Mg2+": "Mg2+",
    "ZN": "Zn2+", "Zn": "Zn2+", "Zn2+": "Zn2+",
}
DIVALENT_ION_RESNAMES = {"Ca2+", "Mg2+", "Zn2+"}

LIPID_RESNAMES = {
    "POPC", "POPE", "POPG", "POPS", "POPA", "DPPC", "DOPC", "DOPE", "DOPS", "DOPG",
    "DLPC", "DMPC", "DSPC", "PIP", "PIP2", "PIP3", "PI", "PSM", "SM", "CER",
    "CHL", "CHOL", "CLR",
}

SOLUTE_DPOLAR_BY_ELEMENT = {
    "O": 0.012,
    "N": 0.012,
    "C": 0.0,
    "S": 0.015,
    "H": 0.0,
}
SOLUTE_IS_POLAR_BY_ELEMENT = {
    "O": 1.0,
    "N": 1.0,
    "C": 0.0,
    "S": 1.0,
    "H": 0.0,
}
CHARGED_SIDECHAIN_POLAR_ATOMS = {
    "ARG": {"NE", "NH1", "NH2"},
    "LYS": {"NZ"},
    "ASP": {"OD1", "OD2"},
    "GLU": {"OE1", "OE2"},
    "HIS": {"ND1", "NE2"},
}

#: Per-particle parameter order frozen by the plan section 17.2 / the C++
#: ``LocalCWLDForce::addParticle`` signature.
PARTICLE_PARAMETER_ORDER = (
    "qbase",
    "charge_mod",
    "dpolar",
    "is_polar",
    "dens_source",
    "dens_sink",
    "source_class_weight",
    "static_phase",
    "residue_id",
)


@dataclass(frozen=True)
class ParticleParameters:
    """The nine frozen per-particle arrays LocalCWLDForce consumes.

    Array order matches ``PARTICLE_PARAMETER_ORDER`` / plan section 17.2.
    All arrays except ``residue_id`` are ``float64``; ``residue_id`` is
    ``int64``. All arrays have length ``system.getNumParticles()``.
    """

    qbase: np.ndarray
    charge_mod: np.ndarray
    dpolar: np.ndarray
    is_polar: np.ndarray
    dens_source: np.ndarray
    dens_sink: np.ndarray
    source_class_weight: np.ndarray
    static_phase: np.ndarray
    residue_id: np.ndarray

    def __post_init__(self) -> None:
        n = len(self.qbase)
        for name in fields(self):
            arr = getattr(self, name.name)
            if len(arr) != n:
                raise ValueError(
                    f"ParticleParameters.{name.name} has length {len(arr)}, "
                    f"expected {n} (from qbase)"
                )
            if name.name == "residue_id":
                if not np.issubdtype(arr.dtype, np.integer):
                    raise ValueError("residue_id must be an integer array")
            else:
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"ParticleParameters.{name.name} contains non-finite values")

    @property
    def num_particles(self) -> int:
        return len(self.qbase)

    def as_particle_tuples(self):
        """Yield ``(qbase, charge_mod, dpolar, is_polar, dens_source,
        dens_sink, source_class_weight, static_phase, residue_id)`` tuples
        in ``PARTICLE_PARAMETER_ORDER``, suitable for repeated calls to
        ``LocalCWLDForce.addParticle(*params)``."""
        for i in range(self.num_particles):
            yield (
                float(self.qbase[i]),
                float(self.charge_mod[i]),
                float(self.dpolar[i]),
                float(self.is_polar[i]),
                float(self.dens_source[i]),
                float(self.dens_sink[i]),
                float(self.source_class_weight[i]),
                float(self.static_phase[i]),
                int(self.residue_id[i]),
            )


def extract_nonbonded_charges(system: "mm.System") -> np.ndarray:
    """Return ``qbase`` (elementary charge) for every particle, read from the
    System's unique ``NonbondedForce``. Verbatim port of v2.6's
    ``extract_nonbonded_charges``."""
    nb_force = next(f for f in system.getForces() if isinstance(f, mm.NonbondedForce))
    charges = np.zeros(system.getNumParticles())
    for i in range(system.getNumParticles()):
        q, _, _ = nb_force.getParticleParameters(i)
        charges[i] = q.value_in_unit(unit.elementary_charge)
    return charges


def active_water_source_qref2(qbase: np.ndarray, water_mask: np.ndarray, dens_source: np.ndarray):
    """Verbatim port of v2.6's ``active_water_source_qref2``: mean qbase^2 over
    active water-source atoms, falling back to all water, then to everything."""
    source_mask = water_mask & (dens_source > 0.0)
    if np.any(source_mask):
        return float(np.mean(qbase[source_mask] ** 2)), int(np.sum(source_mask))
    if np.any(water_mask):
        return float(np.mean(qbase[water_mask] ** 2)), int(np.sum(water_mask))
    return float(np.mean(qbase ** 2)), int(len(qbase))


def clamp_charge_mod(charge_mod: np.ndarray) -> np.ndarray:
    """Verbatim port of v2.6's ``clamp_charge_mod``."""
    return np.clip(charge_mod, CHARGE_MOD_MIN, CHARGE_MOD_MAX)


def _normalized_residue_name(residue_name: str) -> str:
    """Non-mutating equivalent of v2.6's ``normalize_ion_resnames()``: resolve
    the ion alias for classification purposes without touching the caller's
    Topology object."""
    return ION_RESNAME_ALIASES.get(residue_name, residue_name)


def _is_lipid_headgroup_atom(atom, normalized_residue_name: str) -> bool:
    elem = atom.element.symbol if atom.element is not None else ""
    if normalized_residue_name not in LIPID_RESNAMES:
        return False
    return elem in {"O", "N", "P", "S"}


def build_particle_parameters(
    system: "mm.System",
    topology,
    *,
    a_q2: float = DEFAULT_A_Q2,
    ca_source_weight: float = DEFAULT_CA_SOURCE_WEIGHT,
    enable_water_response: bool = True,
    enable_solute_polarization: bool = True,
    fast_active_density: bool = True,
    dpolar_o: float = DEFAULT_DPOLAR_O,
) -> ParticleParameters:
    """Build the nine frozen per-particle arrays for ``LocalCWLDForce``.

    This is a pure function: it does not mutate ``system`` or ``topology``,
    does not create an Integrator/Context, and does not run any simulation.

    Parameters mirror ``build_phase_cwld_metadata()`` in
    ``test_lips_vs_pmeV2.6.py``. ``fast_active_density`` is new here (v2.6
    hardcodes the module-level ``FAST_ACTIVE_DENSITY = True``); default
    ``True`` reproduces v2.6 exactly. Do not flip this default casually --
    it changes which solute heavy atoms get ``dens_source=1`` (see module
    docstring).
    """
    qbase = extract_nonbonded_charges(system)
    n_atoms = len(qbase)

    dpolar = np.zeros(n_atoms)
    is_polar = np.zeros(n_atoms)
    dens_source = np.zeros(n_atoms)
    dens_sink = np.zeros(n_atoms)
    source_class_weight = np.zeros(n_atoms)
    static_phase = np.zeros(n_atoms)
    residue_id = -np.ones(n_atoms, dtype=np.int64)
    water_mask = np.zeros(n_atoms, dtype=bool)

    for residue in topology.residues():
        atom_indices = [atom.index for atom in residue.atoms()]
        residue_id[atom_indices] = residue.index
        residue_name = _normalized_residue_name(residue.name)

        if residue_name in WATER_RESNAMES:
            water_mask[atom_indices] = True
            o_atom = next(
                (a for a in residue.atoms() if a.element is not None and a.element.symbol == "O"),
                None,
            )
            h_atoms = [a for a in residue.atoms() if a.element is not None and a.element.symbol == "H"]
            if o_atom is not None and len(h_atoms) == 2:
                dens_source[o_atom.index] = 1.0
                source_class_weight[o_atom.index] = WATER_SOURCE_WEIGHT
                if enable_water_response:
                    dpolar[o_atom.index] = dpolar_o
                    dpolar[h_atoms[0].index] = -0.5 * dpolar_o
                    dpolar[h_atoms[1].index] = -0.5 * dpolar_o
                    idx = [o_atom.index, h_atoms[0].index, h_atoms[1].index]
                    is_polar[idx] = 1.0
                    dens_sink[idx] = 1.0
                    static_phase[idx] = 1.0

        elif residue_name in ION_RESNAME_ALIASES.values():
            dens_source[atom_indices] = 1.0
            source_class_weight[atom_indices] = (
                ca_source_weight if residue_name in DIVALENT_ION_RESNAMES else ION_SOURCE_WEIGHT
            )

        elif residue_name in LIPID_RESNAMES:
            for atom in residue.atoms():
                if _is_lipid_headgroup_atom(atom, residue_name):
                    dens_source[atom.index] = 1.0
                    source_class_weight[atom.index] = LIPID_HEADGROUP_SOURCE_WEIGHT
                # lipid tails contribute nothing (no source, no sink, no polar)

        else:
            is_protein = residue_name in AMINO_ACIDS
            if enable_solute_polarization:
                for atom in residue.atoms():
                    elem = atom.element.symbol if atom.element is not None else "C"
                    is_charged_site = atom.name in CHARGED_SIDECHAIN_POLAR_ATOMS.get(residue_name, set())
                    is_active_heavy = elem in ("O", "N", "S") or is_charged_site
                    if fast_active_density and not is_active_heavy:
                        continue
                    dpolar[atom.index] = SOLUTE_DPOLAR_BY_ELEMENT.get(elem, 0.0)
                    is_polar[atom.index] = SOLUTE_IS_POLAR_BY_ELEMENT.get(elem, 0.0)
                    dens_source[atom.index] = 1.0 if elem != "H" else 0.0
                    if dens_source[atom.index] > 0.0:
                        if is_charged_site:
                            source_class_weight[atom.index] = (
                                CHARGED_SOURCE_WEIGHT if is_protein else LIGAND_CHARGED_SOURCE_WEIGHT
                            )
                        else:
                            source_class_weight[atom.index] = POLAR_SOURCE_WEIGHT
                    dens_sink[atom.index] = 1.0 if is_polar[atom.index] > 0.0 else 0.0
                    if is_polar[atom.index] > 0.0:
                        static_phase[atom.index] = 1.0

    qref2, _ = active_water_source_qref2(qbase, water_mask, dens_source)
    charge_mod = clamp_charge_mod(1.0 + a_q2 * (qbase ** 2 - qref2))

    return ParticleParameters(
        qbase=qbase,
        charge_mod=charge_mod,
        dpolar=dpolar,
        is_polar=is_polar,
        dens_source=dens_source,
        dens_sink=dens_sink,
        source_class_weight=source_class_weight,
        static_phase=static_phase,
        residue_id=residue_id,
    )
