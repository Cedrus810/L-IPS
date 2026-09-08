#ifndef OPENMM_LOCALCWLDFORCEINFO_H_
#define OPENMM_LOCALCWLDFORCEINFO_H_

#include "openmm/LocalCWLDForce.h"
#include "openmm/common/ComputeForceInfo.h"

#include <cstdio>
#include <cstdlib>

namespace LocalCWLDPlugin {

/**
 * Tells the platform which particles are interchangeable, so it can reorder
 * atoms for locality without changing results (ticket LCWLD-090).
 *
 * Getting this wrong is silent: the platform reorders anyway, and the only
 * symptom is that results drift when the reordering happens to differ. So the
 * comparison below is deliberately strict -- two particles are identical only
 * if every one of the nine frozen parameters matches, plus their exclusions.
 *
 * `residueId` is deliberately NOT compared -- see LCWLD-161.
 *
 * It was, and it cost 3.07 rc traversals (1412 -> 480 us/step, end-to-end
 * 1.55x -> 3.2x). Comparing it makes every residue distinct, so no two water
 * molecules are interchangeable, so OpenMM's atom reordering -- which only
 * permutes molecules WITHIN a set of identical ones -- has nothing it is
 * allowed to move. The shared neighbour list then loses its spatial locality
 * and every kernel that walks it gets 2-5x slower, ours included.
 *
 * Why dropping it is safe: the kernels never read the VALUE of residueId, only
 * `residueId[atom2] == residueId[atom1]` (localCWLD.cc:264,297). The physics
 * depends on the residue PARTITION, not on the labels. OpenMM permutes whole
 * molecules, and the exclusions we declare below put every atom of a residue
 * in the same molecule, so a permutation carries a residue's atoms together
 * and the partition survives even though the labels stay with the slots.
 *
 * That safety argument has one assumption -- no residue straddles two
 * molecules -- and `CommonCalcLocalCWLDForceKernel::initialize` now checks it
 * and throws rather than letting it be silent.
 */
class LocalCWLDForceInfo : public OpenMM::ComputeForceInfo {
public:
    /**
     * @param residuesAreMoleculeLocal  true when every residue's atoms are
     *        connected to each other by exclusions, so OpenMM -- which permutes
     *        whole molecules -- can never move part of a residue without the
     *        rest. Only then may residueId be left out of the comparison.
     *        `CommonCalcLocalCWLDForceKernel::initialize` computes it.
     *
     * `LOCALCWLD_PROBE_STRICT_FORCEINFO=1` forces the strict path regardless,
     * so the before/after of LCWLD-161 can be measured in ONE binary with ONE
     * variable changed. Cross-build ratios are how several earlier numbers in
     * this project went wrong. Read once: areParticlesIdentical is called
     * O(N^2) during molecule grouping and getenv() in there is not free.
     */
    LocalCWLDForceInfo(const LocalCWLDForce& force, bool residuesAreMoleculeLocal)
        : force(force),
          strictResidue(!residuesAreMoleculeLocal ||
                        std::getenv("LOCALCWLD_PROBE_STRICT_FORCEINFO") != nullptr) {
        if (strictResidue)
            std::printf("[localcwld] areParticlesIdentical compares residueId "
                        "(%s). Atom reordering is restricted; expect the shared "
                        "neighbour list to be slower -- see LCWLD-161.\n",
                        residuesAreMoleculeLocal ? "probe G"
                                                 : "a residue spans two molecules");
    }

    bool areParticlesIdentical(int particle1, int particle2) override {
        double q1, m1, d1, p1, s1, k1, w1, ph1;
        double q2, m2, d2, p2, s2, k2, w2, ph2;
        int r1, r2;
        force.getParticleParameters(particle1, q1, m1, d1, p1, s1, k1, w1, ph1, r1);
        force.getParticleParameters(particle2, q2, m2, d2, p2, s2, k2, w2, ph2, r2);
        if (strictResidue && r1 != r2)
            return false;       // probe G only; see the constructor
        return (q1 == q2 && m1 == m2 && d1 == d2 && p1 == p2 && s1 == s2 &&
                k1 == k2 && w1 == w2 && ph1 == ph2);
    }

    int getNumParticleGroups() override {
        return force.getNumExclusions();
    }

    void getParticlesInGroup(int index, std::vector<int>& particles) override {
        int p1, p2;
        force.getExclusionParticles(index, p1, p2);
        particles.resize(2);
        particles[0] = p1;
        particles[1] = p2;
    }

    bool areGroupsIdentical(int group1, int group2) override {
        return true;   // all exclusions are the same kind of thing
    }

private:
    const LocalCWLDForce& force;
    const bool strictResidue;
};

} // namespace LocalCWLDPlugin

#endif /* OPENMM_LOCALCWLDFORCEINFO_H_ */
