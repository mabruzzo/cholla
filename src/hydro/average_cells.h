/*! \file average_cells.h
 *  \brief Definitions of functions and classes that implement logic related to averaging cells with
 *         neighbors. */

#ifndef AVERAGE_CELLS_H
#define AVERAGE_CELLS_H

#include <math.h>

#include "../global/global.h"
#include "hydro_cuda.h"

/*! \brief Object that checks whether a given cell meets the conditions for slow-cell averaging.
*          The main motivation for creating this class is reducing ifdef statements (and allow to modify the
*          actual slow-cell-condition. */
struct SlowCellConditionChecker {

  // omit data-members if they aren't used for anything
  #ifdef AVERAGE_SLOW_CELLS
  Real max_dti_slow, dx, dy, dz;
  #endif

  __host__ __device__ SlowCellConditionChecker(Real max_dti_slow, Real dx, Real dy, Real dz)
  #ifdef AVERAGE_SLOW_CELLS
    : max_dti_slow{max_dti_slow}, dx{dx}, dy{dy}, dz{dz}
  #endif
  {
  }

  /*! \brief Returns whether the cell meets the condition for being considered a slow cell that must
   *  be averaged.
   */
  template<bool verbose = false>
  __device__ bool is_slow(Real E, Real d, Real d_inv, Real vx, Real vy, Real vz, Real gamma) const
  {
    return this->max_dti_if_slow(E, d, d_inv, vx, vy, vz, gamma) >= 0.0;
  }

  /*! \brief Returns the max inverse timestep of the specified cell, if it meets the criteria for being
   *  a slow cell. If it doesn't, return a negative value instead.
   */
  __device__ Real max_dti_if_slow(Real E, Real d, Real d_inv, Real vx, Real vy, Real vz, Real gamma) const
  {
  #ifndef AVERAGE_SLOW_CELLS
    return -1.0;
  #else
    Real max_dti = hydroInverseCrossingTime(E, d, d_inv, vx, vy, vz, dx, dy, dz, gamma);
    return (max_dti > max_dti_slow) ? max_dti : -1.0;
  #endif
  }

};


#ifdef AVERAGE_SLOW_CELLS

void Average_Slow_Cells(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields,
                        Real gamma, SlowCellConditionChecker slow_check,
                        Real xbound, Real ybound, Real zbound, int nx_offset, int ny_offset, int nz_offset);

__global__ void Average_Slow_Cells_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields,
                                      Real gamma, SlowCellConditionChecker slow_check,
                                      Real xbound, Real ybound, Real zbound, int nx_offset, int ny_offset, int nz_offset);
#endif

#endif /* AVERAGE_CELLS_H */