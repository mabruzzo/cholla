#include <string>

#include "chemistry.h"
#include "../io/ParameterMap.h"
#include "../grid/grid3D.h"
#include "../cooling/cooling_cuda.h"  // provides Cooling_Update
#include "../cooling/load_cloudy_texture.h"  // provides Load_Cuda_Textures and Free_Cuda_Textures

class TabulatedCoolingFunctor{
  bool cloudy_;
public:
  TabulatedCoolingFunctor(bool cloudy)
    : cloudy_(cloudy)
  {
    if (cloudy_) Load_Cuda_Textures();
  }

  ~TabulatedCoolingFunctor() { if (cloudy_) Free_Cuda_Textures(); }

  void operator()(Grid3D& grid) {
    Header& H = grid.H;
    Cooling_Update(grid.C.device, H.nx, H.ny, H.nz, H.n_ghost, H.n_fields, H.dt, gama, this->cloudy_);
  }

};

std::function<void(Grid3D&)> configure_chemistry_callback(ParameterMap& pmap) {

  // we use the traditional macros to set default-chemistry kinds to avoid
  // breaking older setups.
  // -> in the future, I think we can do away with this...
  std::string default_kind = "none";
#if defined(COOLING_GPU) && defined(CLOUDY_COOL)
  default_kind = "tabulated-cloudy";
#elif defined(COOLING_GPU)
  default_kind = "piecewise-cie";
#elif defined(CHEMISTRY_GPU)
  default_kind = "chemistry-gpu";
#elif defined(COOLING_GRACKLE)
  default_kind = "grackle";
#endif

  std::string chemistry_kind = pmap.value_or("chemistry.kind", default_kind);

#if defined(CHEMISTRY_GPU) || defined(COOLING_GRACKLE)
  CHOLLA_ASSERT(
    chemistry_kind == default_kind,
    "based on the defined macros, it is currently an error to pass a value to the "
    "chemistry.kind parameter other than \"%s\" (even \"none\" is invalid) This is "
    "because the \"%s\" functionality is invoked outside of the chemistry_callback "
    "machinery (this will be fixed in the near future)",
    default_kind.c_str(), default_kind.c_str());
  return {};
#else
  if (chemistry_kind == "none") {
    return {};
  } else if (chemistry_kind == "tabulated-cloudy"){
    CHOLLA_ASSERT(chemistry_kind == default_kind, "NOT IMPLEMENTED YET");
    TabulatedCoolingFunctor fn(true);
    return {fn};
  } else if (chemistry_kind == "piecewise-cie"){
    CHOLLA_ASSERT(chemistry_kind == default_kind, "NOT IMPLEMENTED YET");
    TabulatedCoolingFunctor fn(false);
    return {fn};
  } else if (chemistry_kind == "chemistry-gpu" or chemistry_kind == "grackle"){
    CHOLLA_ERROR("chemistry.kind doesn't support %s yet (unless certain macros are defined)", chemistry_kind.c_str());
  } else {
    CHOLLA_ERROR("\"%s\" is not a supported chemistry.kind parameter value.", chemistry_kind.c_str());
  }
#endif  // defined(CHEMISTRY_GPU) || defined(COOLING_GRACKLE)
}