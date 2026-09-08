/**
 * Register the LocalCWLD kernels with the CUDA platform (ticket LCWLD-110).
 *
 * `registerKernelFactories()` is the symbol OpenMM looks up in every shared
 * library it finds in its plugin directory. Everything else here is plumbing.
 *
 * Note there is no nvcc and no .cu file anywhere in this target: the device
 * code in platforms/common/kernels/localCWLD.cc is a string, compiled at run
 * time by whichever NVRTC libOpenMMCUDA.so is linked against (12.9 on this
 * installation). See docs/decisions/DEC-001-toolchain.md.
 */
#include <exception>
#include <vector>

#include "CommonLocalCWLDKernels.h"
#include "openmm/LocalCWLDKernels.h"
#include "openmm/KernelFactory.h"
#include "openmm/OpenMMException.h"
#include "openmm/Platform.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/internal/windowsExport.h"

using namespace LocalCWLDPlugin;
using namespace OpenMM;

class CudaLocalCWLDKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform,
                                 ContextImpl& context) const override {
        CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(
            context.getPlatformData())->contexts[0];
        if (name == CalcLocalCWLDForceKernel::Name())
            return new CommonCalcLocalCWLDForceKernel(name, platform, cu);
        throw OpenMMException(
            (std::string("Tried to create kernel with illegal kernel name '") + name + "'").c_str());
    }
};

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaLocalCWLDKernelFactory* factory = new CudaLocalCWLDKernelFactory();
        platform.registerKernelFactory(CalcLocalCWLDForceKernel::Name(), factory);
    } catch (std::exception& ex) {
        // The CUDA platform may legitimately be absent (no driver, CPU-only
        // host). That is not an error for this plugin -- it simply does not
        // register. Throwing here would break OpenMM's whole plugin scan.
    }
}

extern "C" OPENMM_EXPORT void registerLocalCWLDCudaKernelFactories() {
    try {
        Platform::getPlatformByName("CUDA");
    } catch (...) {
        Platform::registerPlatform(new CudaPlatform());
    }
    registerKernelFactories();
}
