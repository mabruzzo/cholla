#pragma once

  #include <functional>

  #include "../analysis/feedback_analysis.h"
  #include "../global/global.h"

namespace feedinfoLUT {
  // this enum acts like a lookup table (LUT). It maps the names of analysis statistics to
  // contiguous indices. LEN specfies the number of named analysis statistics
  enum {
    countSN = 0,
    countResolved,
    countUnresolved,
    totalEnergy,
    totalMomentum,
    totalUnresEnergy,
    totalWindMomentum,
    totalWindEnergy,
    // make sure the following is always the last entry so that it reflects the number of entries
    LEN
  };
}

namespace feedback
{
static const Real ENERGY_PER_SN = 1e51 / MASS_UNIT * TIME_UNIT * TIME_UNIT / LENGTH_UNIT / LENGTH_UNIT;
// 10 solarMasses per SN
static const Real MASS_PER_SN = 10.0;
// 2.8e5 M_s km/s * n_0^{-0.17} -> eq.(34) Kim & Ostriker (2015)
static const Real FINAL_MOMENTUM = 2.8e5 / LENGTH_UNIT * 1e5 * TIME_UNIT;
// 30.2 pc * n_0^{-0.46} -> eq.(31) Kim & Ostriker (2015)
static const Real R_SH = 0.0302;

/* construct the feedback function (or not based on the specified parameters & compilation mode)
 *
 * \note
 * we could probably define the following function regardless of the defined compiler flags */
std::function<void(Grid3D&)> configure_feedback_callback(struct parameters& P,
                                                         FeedbackAnalysis& analysis);

}  // namespace feedback