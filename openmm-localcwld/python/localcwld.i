/*
 * LCWLD-080: SWIG wrapper for LocalCWLDForce.
 *
 * Only the public API from openmmapi/include/openmm/LocalCWLDForce.h is
 * declared here, and it is declared BY HAND -- SWIG does not read the real
 * header, it reads this file.
 *
 * That sounds more dangerous than it is, and it is worth being precise because
 * the first version of this comment overstated it. SWIG emits a real C++ call
 * against the real header, so a declaration that drifts in arity or type fails
 * to compile. A drift in parameter NAMES cannot matter -- the call is
 * positional and the names here are labels. What no test on this side can see
 * is the header itself being permuted: the serialization proxy calls
 * addParticle positionally too, so both Python routes would be wrong together.
 * That case is caught by TestCudaLocalCWLDAcceptance, which compares
 * per-particle forces against the production CustomGBForce.
 *
 * The parameter order is the frozen plan section 17.2 order, the same one used
 * by the C++ API, the serialization proxy, the fixtures and the device arrays.
 * Permuting it here would be silent.
 */
%module localcwld

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include <std_string.i>
%include <typemaps.i>

%{
/*
 * The %factory block in OpenMM's own SWIG input enumerates every Force class
 * OpenMM ships, including the Amoeba, Drude and RPMD ones, and the generated
 * wrapper dynamic_casts to all of them. Leaving these out compiles into a wall
 * of "'AmoebaMultipoleForce' is not a member of 'OpenMM'" that points at
 * generated code and says nothing about the cause.
 */
#include "openmm/LocalCWLDForce.h"
#include "OpenMM.h"
#include "OpenMMAmoeba.h"
#include "OpenMMDrude.h"
#include "openmm/RPMDIntegrator.h"
#include "openmm/RPMDMonteCarloBarostat.h"
#include "openmm/RPMDUpdater.h"
#include <sstream>
%}

%pythoncode %{
import openmm as mm
import openmm.unit as unit
%}

/*
 * Translate C++ exceptions into Python ones.
 *
 * Not optional. Every setter on this class validates its argument and throws
 * OpenMMException on bad input -- that is deliberate, it is how a builder bug
 * gets caught early. Without this block the exception unwinds through the C
 * wrapper with no handler and calls std::terminate: `force.addExclusion(2, 2)`
 * kills the interpreter instead of raising. The validation is worse than
 * useless from Python until this is here.
 */
%exception {
    try {
        $action
    } catch (std::exception& e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        SWIG_fail;
    }
}

namespace LocalCWLDPlugin {

class LocalCWLDForce : public OpenMM::Force {
public:
    enum NonbondedMethod {
        CutoffPeriodic = 0
    };

    LocalCWLDForce();

    int addParticle(double qbase, double chargeMod, double dpolar, double isPolar,
                    double densSource, double densSink, double sourceClassWeight,
                    double staticPhase, int residueId);
    int getNumParticles() const;

    /* Returned to Python as a 9-tuple, in the section 17.2 order. */
    %apply double& OUTPUT {double& qbase};
    %apply double& OUTPUT {double& chargeMod};
    %apply double& OUTPUT {double& dpolar};
    %apply double& OUTPUT {double& isPolar};
    %apply double& OUTPUT {double& densSource};
    %apply double& OUTPUT {double& densSink};
    %apply double& OUTPUT {double& sourceClassWeight};
    %apply double& OUTPUT {double& staticPhase};
    %apply int& OUTPUT {int& residueId};
    void getParticleParameters(int index, double& qbase, double& chargeMod,
                               double& dpolar, double& isPolar, double& densSource,
                               double& densSink, double& sourceClassWeight,
                               double& staticPhase, int& residueId) const;
    %clear double& qbase;
    %clear double& chargeMod;
    %clear double& dpolar;
    %clear double& isPolar;
    %clear double& densSource;
    %clear double& densSink;
    %clear double& sourceClassWeight;
    %clear double& staticPhase;
    %clear int& residueId;

    void setParticleParameters(int index, double qbase, double chargeMod,
                               double dpolar, double isPolar, double densSource,
                               double densSink, double sourceClassWeight,
                               double staticPhase, int residueId);

    int addExclusion(int particle1, int particle2);
    int getNumExclusions() const;

    %apply int& OUTPUT {int& particle1};
    %apply int& OUTPUT {int& particle2};
    void getExclusionParticles(int index, int& particle1, int& particle2) const;
    %clear int& particle1;
    %clear int& particle2;

    void setExclusionParticles(int index, int particle1, int particle2);

    double getEnvironmentCutoff() const;
    void setEnvironmentCutoff(double distance);
    double getCutoffDistance() const;
    void setCutoffDistance(double distance);
    int getZMMOrder() const;
    void setZMMOrder(int order);
    double getRho0() const;
    void setRho0(double value);
    double getKPolar() const;
    void setKPolar(double value);
    double getChargeDeltaClamp() const;
    void setChargeDeltaClamp(double value);
    bool getUseQPenalty() const;
    void setUseQPenalty(bool enabled);
    double getQPenaltyStrength() const;
    void setQPenaltyStrength(double value);
    double getOne4PiEps0() const;
    void setOne4PiEps0(double value);
    NonbondedMethod getNonbondedMethod() const;
    void setNonbondedMethod(NonbondedMethod method);

    void updateParametersInContext(OpenMM::Context& context);
    bool usesPeriodicBoundaryConditions() const;

    /*
     * System::getForce() hands back an OpenMM::Force, and the %factory in
     * OpenMM's own SWIG input cannot know about a plugin type. Without these
     * there is no way to get from a deserialized System back to the typed
     * object -- which is exactly what the LCWLD-060 XML route produces.
     */
    %extend {
        static LocalCWLDPlugin::LocalCWLDForce& cast(OpenMM::Force& force) {
            return dynamic_cast<LocalCWLDPlugin::LocalCWLDForce&>(force);
        }
        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<LocalCWLDPlugin::LocalCWLDForce*>(&force) != NULL);
        }
    }
};

}
