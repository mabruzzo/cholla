/*! \file hydro_cuda.cu
 *  \brief Definitions of functions used in all cuda integration algorithms. */

#include <float.h>
#include <math.h>
#include <stdio.h>

#include <limits>

#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../gravity/static_grav.h"
#include "../hydro/average_cells.h"
#include "../hydro/hydro_cuda.h"
#include "../utils/DeviceVector.h"
#include "../utils/cuda_utilities.h"
#include "../utils/gpu.hpp"
#include "../utils/hydro_utilities.h"
#include "../utils/reduction_utilities.h"

__global__ void Update_Conserved_Variables_1D(Real *dev_conserved, Real *dev_F, int n_cells, int x_off, int n_ghost,
                                              Real dx, Real xbound, Real dt, Real gamma, int n_fields, int custom_grav)
{
  int id;
#ifdef STATIC_GRAV
  Real d, d_inv, vx;
  Real gx, d_n, d_inv_n, vx_n;
  gx = 0.0;
#endif

  Real dtodx = dt / dx;

  // get a global thread ID
  id = threadIdx.x + blockIdx.x * blockDim.x;

  // threads corresponding to real cells do the calculation
  if (id > n_ghost - 1 && id < n_cells - n_ghost) {
#ifdef STATIC_GRAV
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
#endif

    // update the conserved variable array
    dev_conserved[id] += dtodx * (dev_F[id - 1] - dev_F[id]);
    dev_conserved[n_cells + id] += dtodx * (dev_F[n_cells + id - 1] - dev_F[n_cells + id]);
    dev_conserved[2 * n_cells + id] += dtodx * (dev_F[2 * n_cells + id - 1] - dev_F[2 * n_cells + id]);
    dev_conserved[3 * n_cells + id] += dtodx * (dev_F[3 * n_cells + id - 1] - dev_F[3 * n_cells + id]);
    dev_conserved[4 * n_cells + id] += dtodx * (dev_F[4 * n_cells + id - 1] - dev_F[4 * n_cells + id]);
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      dev_conserved[(5 + i) * n_cells + id] +=
          dtodx * (dev_F[(5 + i) * n_cells + id - 1] - dev_F[(5 + i) * n_cells + id]);
    }
#endif
#ifdef DE
    dev_conserved[(n_fields - 1) * n_cells + id] +=
        dtodx * (dev_F[(n_fields - 1) * n_cells + id - 1] - dev_F[(n_fields - 1) * n_cells + id]);
#endif
#ifdef STATIC_GRAV  // add gravitational source terms, time averaged from n to
                    // n+1
    calc_g_1D(id, x_off, n_ghost, custom_grav, dx, xbound, &gx);
    d_n     = dev_conserved[id];
    d_inv_n = 1.0 / d_n;
    vx_n    = dev_conserved[1 * n_cells + id] * d_inv_n;
    dev_conserved[n_cells + id] += 0.5 * dt * gx * (d + d_n);
    dev_conserved[4 * n_cells + id] += 0.25 * dt * gx * (d + d_n) * (vx + vx_n);
#endif
    if (dev_conserved[id] != dev_conserved[id]) {
      printf("%3d Thread crashed in final update. %f\n", id, dev_conserved[id]);
    }
    /*
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) *
    (gamma - 1.0); if (P < 0.0) printf("%d Negative pressure after final
    update.\n", id);
    */
  }
}

__global__ void Update_Conserved_Variables_2D(Real *dev_conserved, Real *dev_F_x, Real *dev_F_y, int nx, int ny,
                                              int x_off, int y_off, int n_ghost, Real dx, Real dy, Real xbound,
                                              Real ybound, Real dt, Real gamma, int n_fields, int custom_grav)
{
  int id, xid, yid, n_cells;
  int imo, jmo;

#ifdef STATIC_GRAV
  Real d, d_inv, vx, vy;
  Real gx, gy, d_n, d_inv_n, vx_n, vy_n;
  gx = 0.0;
  gy = 0.0;
#endif

  Real dtodx = dt / dx;
  Real dtody = dt / dy;

  n_cells = nx * ny;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  id          = threadIdx.x + blockId * blockDim.x;
  yid         = id / nx;
  xid         = id - yid * nx;
  imo         = xid - 1 + yid * nx;
  jmo         = xid + (yid - 1) * nx;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost) {
#ifdef STATIC_GRAV
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
    vy    = dev_conserved[2 * n_cells + id] * d_inv;
#endif
    // update the conserved variable array
    dev_conserved[id] += dtodx * (dev_F_x[imo] - dev_F_x[id]) + dtody * (dev_F_y[jmo] - dev_F_y[id]);
    dev_conserved[n_cells + id] += dtodx * (dev_F_x[n_cells + imo] - dev_F_x[n_cells + id]) +
                                   dtody * (dev_F_y[n_cells + jmo] - dev_F_y[n_cells + id]);
    dev_conserved[2 * n_cells + id] += dtodx * (dev_F_x[2 * n_cells + imo] - dev_F_x[2 * n_cells + id]) +
                                       dtody * (dev_F_y[2 * n_cells + jmo] - dev_F_y[2 * n_cells + id]);
    dev_conserved[3 * n_cells + id] += dtodx * (dev_F_x[3 * n_cells + imo] - dev_F_x[3 * n_cells + id]) +
                                       dtody * (dev_F_y[3 * n_cells + jmo] - dev_F_y[3 * n_cells + id]);
    dev_conserved[4 * n_cells + id] += dtodx * (dev_F_x[4 * n_cells + imo] - dev_F_x[4 * n_cells + id]) +
                                       dtody * (dev_F_y[4 * n_cells + jmo] - dev_F_y[4 * n_cells + id]);
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      dev_conserved[(5 + i) * n_cells + id] +=
          dtodx * (dev_F_x[(5 + i) * n_cells + imo] - dev_F_x[(5 + i) * n_cells + id]) +
          dtody * (dev_F_y[(5 + i) * n_cells + jmo] - dev_F_y[(5 + i) * n_cells + id]);
    }
#endif
#ifdef DE
    dev_conserved[(n_fields - 1) * n_cells + id] +=
        dtodx * (dev_F_x[(n_fields - 1) * n_cells + imo] - dev_F_x[(n_fields - 1) * n_cells + id]) +
        dtody * (dev_F_y[(n_fields - 1) * n_cells + jmo] - dev_F_y[(n_fields - 1) * n_cells + id]);
#endif
#ifdef STATIC_GRAV
    // calculate the gravitational acceleration as a function of x & y position
    calc_g_2D(xid, yid, x_off, y_off, n_ghost, custom_grav, dx, dy, xbound, ybound, &gx, &gy);
    // add gravitational source terms, time averaged from n to n+1
    d_n     = dev_conserved[id];
    d_inv_n = 1.0 / d_n;
    vx_n    = dev_conserved[1 * n_cells + id] * d_inv_n;
    vy_n    = dev_conserved[2 * n_cells + id] * d_inv_n;
    dev_conserved[n_cells + id] += 0.5 * dt * gx * (d + d_n);
    dev_conserved[2 * n_cells + id] += 0.5 * dt * gy * (d + d_n);
    dev_conserved[4 * n_cells + id] +=
        0.25 * dt * gx * (d + d_n) * (vx + vx_n) + 0.25 * dt * gy * (d + d_n) * (vy + vy_n);
#endif
    if (dev_conserved[id] < 0.0 || dev_conserved[id] != dev_conserved[id]) {
      printf("%3d %3d Thread crashed in final update. %f %f %f\n", xid, yid, dtodx * (dev_F_x[imo] - dev_F_x[id]),
             dtody * (dev_F_y[jmo] - dev_F_y[id]), dev_conserved[id]);
    }
    /*
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) *
    (gamma - 1.0); if (P < 0.0) printf("%3d %3d Negative pressure after final
    update. %f %f %f %f\n", xid, yid, dev_conserved[4*n_cells + id],
    0.5*d*vx*vx, 0.5*d*vy*vy, P);
    */
  }
}

__global__ void Update_Conserved_Variables_3D(Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, Real *Q_Ly, Real *Q_Ry,
                                              Real *Q_Lz, Real *Q_Rz, Real *dev_F_x, Real *dev_F_y, Real *dev_F_z,
                                              int nx, int ny, int nz, int x_off, int y_off, int z_off, int n_ghost,
                                              Real dx, Real dy, Real dz, Real xbound, Real ybound, Real zbound, Real dt,
                                              Real gamma, int n_fields, int custom_grav, Real density_floor,
                                              Real *dev_potential)
{
  int id, xid, yid, zid, n_cells;
  int imo, jmo, kmo;

#ifdef STATIC_GRAV
  Real d, d_inv, vx, vy, vz;
  Real gx, gy, gz, d_n, d_inv_n, vx_n, vy_n, vz_n;
  gx = 0.0;
  gy = 0.0;
  gz = 0.0;
#endif

#ifdef DENSITY_FLOOR
  Real dens_0;
#endif

#ifdef GRAVITY
  Real d, d_inv, vx, vy, vz;
  Real gx, gy, gz, d_n, d_inv_n, vx_n, vy_n, vz_n;
  Real pot_l, pot_r;
  int id_l, id_r;
  gx = 0.0;
  gy = 0.0;
  gz = 0.0;

  #ifdef GRAVITY_5_POINTS_GRADIENT
  int id_ll, id_rr;
  Real pot_ll, pot_rr;
  #endif

#endif  // GRAVITY

  Real dtodx = dt / dx;
  Real dtody = dt / dy;
  Real dtodz = dt / dz;
  n_cells    = nx * ny * nz;

  // get a global thread ID
  id  = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx * ny);
  yid = (id - zid * nx * ny) / nx;
  xid = id - zid * nx * ny - yid * nx;
  imo = xid - 1 + yid * nx + zid * nx * ny;
  jmo = xid + (yid - 1) * nx + zid * nx * ny;
  kmo = xid + yid * nx + (zid - 1) * nx * ny;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
      zid < nz - n_ghost) {
#if defined(STATIC_GRAV) || defined(GRAVITY)
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
    vy    = dev_conserved[2 * n_cells + id] * d_inv;
    vz    = dev_conserved[3 * n_cells + id] * d_inv;
#endif

    // update the conserved variable array
    dev_conserved[id] += dtodx * (dev_F_x[imo] - dev_F_x[id]) + dtody * (dev_F_y[jmo] - dev_F_y[id]) +
                         dtodz * (dev_F_z[kmo] - dev_F_z[id]);
    dev_conserved[n_cells + id] += dtodx * (dev_F_x[n_cells + imo] - dev_F_x[n_cells + id]) +
                                   dtody * (dev_F_y[n_cells + jmo] - dev_F_y[n_cells + id]) +
                                   dtodz * (dev_F_z[n_cells + kmo] - dev_F_z[n_cells + id]);
    dev_conserved[2 * n_cells + id] += dtodx * (dev_F_x[2 * n_cells + imo] - dev_F_x[2 * n_cells + id]) +
                                       dtody * (dev_F_y[2 * n_cells + jmo] - dev_F_y[2 * n_cells + id]) +
                                       dtodz * (dev_F_z[2 * n_cells + kmo] - dev_F_z[2 * n_cells + id]);
    dev_conserved[3 * n_cells + id] += dtodx * (dev_F_x[3 * n_cells + imo] - dev_F_x[3 * n_cells + id]) +
                                       dtody * (dev_F_y[3 * n_cells + jmo] - dev_F_y[3 * n_cells + id]) +
                                       dtodz * (dev_F_z[3 * n_cells + kmo] - dev_F_z[3 * n_cells + id]);
    dev_conserved[4 * n_cells + id] += dtodx * (dev_F_x[4 * n_cells + imo] - dev_F_x[4 * n_cells + id]) +
                                       dtody * (dev_F_y[4 * n_cells + jmo] - dev_F_y[4 * n_cells + id]) +
                                       dtodz * (dev_F_z[4 * n_cells + kmo] - dev_F_z[4 * n_cells + id]);
#ifdef SCALAR
    for (int i = 0; i < NSCALARS; i++) {
      dev_conserved[(5 + i) * n_cells + id] +=
          dtodx * (dev_F_x[(5 + i) * n_cells + imo] - dev_F_x[(5 + i) * n_cells + id]) +
          dtody * (dev_F_y[(5 + i) * n_cells + jmo] - dev_F_y[(5 + i) * n_cells + id]) +
          dtodz * (dev_F_z[(5 + i) * n_cells + kmo] - dev_F_z[(5 + i) * n_cells + id]);
  #ifdef COOLING_GRACKLE
      // If the updated value is negative, then revert to the value before the
      // update
      if (dev_conserved[(5 + i) * n_cells + id] < 0) {
        dev_conserved[(5 + i) * n_cells + id] -=
            dtodx * (dev_F_x[(5 + i) * n_cells + imo] - dev_F_x[(5 + i) * n_cells + id]) +
            dtody * (dev_F_y[(5 + i) * n_cells + jmo] - dev_F_y[(5 + i) * n_cells + id]) +
            dtodz * (dev_F_z[(5 + i) * n_cells + kmo] - dev_F_z[(5 + i) * n_cells + id]);
      }
  #endif
    }
#endif
#ifdef DE
    dev_conserved[(n_fields - 1) * n_cells + id] +=
        dtodx * (dev_F_x[(n_fields - 1) * n_cells + imo] - dev_F_x[(n_fields - 1) * n_cells + id]) +
        dtody * (dev_F_y[(n_fields - 1) * n_cells + jmo] - dev_F_y[(n_fields - 1) * n_cells + id]) +
        dtodz * (dev_F_z[(n_fields - 1) * n_cells + kmo] - dev_F_z[(n_fields - 1) * n_cells + id]);
    // +  0.5*P*(dtodx*(vx_imo-vx_ipo) + dtody*(vy_jmo-vy_jpo) +
    // dtodz*(vz_kmo-vz_kpo));
    // Note: this term is added in a separate kernel to avoid synchronization
    // issues
#endif

#ifdef DENSITY_FLOOR
    if (dev_conserved[id] < density_floor) {
      if (dev_conserved[id] > 0) {
        dens_0 = dev_conserved[id];
        // Set the density to the density floor
        dev_conserved[id] = density_floor;
        // Scale the conserved values to the new density
        dev_conserved[1 * n_cells + id] *= (density_floor / dens_0);
        dev_conserved[2 * n_cells + id] *= (density_floor / dens_0);
        dev_conserved[3 * n_cells + id] *= (density_floor / dens_0);
        dev_conserved[4 * n_cells + id] *= (density_floor / dens_0);
  #ifdef DE
        dev_conserved[(n_fields - 1) * n_cells + id] *= (density_floor / dens_0);
  #endif
      } else {
        // If the density is negative: average the density on that cell
        dens_0 = dev_conserved[id];
        Average_Cell_Single_Field(0, xid, yid, zid, nx, ny, nz, n_cells, dev_conserved);
      }
    }
#endif  // DENSITY_FLOOR

#ifdef STATIC_GRAV
    calc_g_3D(xid, yid, zid, x_off, y_off, z_off, n_ghost, custom_grav, dx, dy, dz, xbound, ybound, zbound, &gx, &gy,
              &gz);
    d_n     = dev_conserved[id];
    d_inv_n = 1.0 / d_n;
    vx_n    = dev_conserved[1 * n_cells + id] * d_inv_n;
    vy_n    = dev_conserved[2 * n_cells + id] * d_inv_n;
    vz_n    = dev_conserved[3 * n_cells + id] * d_inv_n;
    dev_conserved[n_cells + id] += 0.5 * dt * gx * (d + d_n);
    dev_conserved[2 * n_cells + id] += 0.5 * dt * gy * (d + d_n);
    dev_conserved[3 * n_cells + id] += 0.5 * dt * gz * (d + d_n);
    dev_conserved[4 * n_cells + id] += 0.25 * dt * gx * (d + d_n) * (vx + vx_n) +
                                       0.25 * dt * gy * (d + d_n) * (vy + vy_n) +
                                       0.25 * dt * gz * (d + d_n) * (vz + vz_n);
#endif

#ifdef GRAVITY
    d_n     = dev_conserved[id];
    d_inv_n = 1.0 / d_n;
    vx_n    = dev_conserved[1 * n_cells + id] * d_inv_n;
    vy_n    = dev_conserved[2 * n_cells + id] * d_inv_n;
    vz_n    = dev_conserved[3 * n_cells + id] * d_inv_n;

    // Calculate the -gradient of potential
    // Get X componet of gravity field
    id_l  = (xid - 1) + (yid)*nx + (zid)*nx * ny;
    id_r  = (xid + 1) + (yid)*nx + (zid)*nx * ny;
    pot_l = dev_potential[id_l];
    pot_r = dev_potential[id_r];
  #ifdef GRAVITY_5_POINTS_GRADIENT
    id_ll  = (xid - 2) + (yid)*nx + (zid)*nx * ny;
    id_rr  = (xid + 2) + (yid)*nx + (zid)*nx * ny;
    pot_ll = dev_potential[id_ll];
    pot_rr = dev_potential[id_rr];
    gx     = -1 * (-pot_rr + 8 * pot_r - 8 * pot_l + pot_ll) / (12 * dx);
  #else
    gx = -0.5 * (pot_r - pot_l) / dx;
  #endif

    // Get Y componet of gravity field
    id_l  = (xid) + (yid - 1) * nx + (zid)*nx * ny;
    id_r  = (xid) + (yid + 1) * nx + (zid)*nx * ny;
    pot_l = dev_potential[id_l];
    pot_r = dev_potential[id_r];
  #ifdef GRAVITY_5_POINTS_GRADIENT
    id_ll  = (xid) + (yid - 2) * nx + (zid)*nx * ny;
    id_rr  = (xid) + (yid + 2) * nx + (zid)*nx * ny;
    pot_ll = dev_potential[id_ll];
    pot_rr = dev_potential[id_rr];
    gy     = -1 * (-pot_rr + 8 * pot_r - 8 * pot_l + pot_ll) / (12 * dx);
  #else
    gy = -0.5 * (pot_r - pot_l) / dy;
  #endif
    // Get Z componet of gravity field
    id_l  = (xid) + (yid)*nx + (zid - 1) * nx * ny;
    id_r  = (xid) + (yid)*nx + (zid + 1) * nx * ny;
    pot_l = dev_potential[id_l];
    pot_r = dev_potential[id_r];
  #ifdef GRAVITY_5_POINTS_GRADIENT
    id_ll  = (xid) + (yid)*nx + (zid - 2) * nx * ny;
    id_rr  = (xid) + (yid)*nx + (zid + 2) * nx * ny;
    pot_ll = dev_potential[id_ll];
    pot_rr = dev_potential[id_rr];
    gz     = -1 * (-pot_rr + 8 * pot_r - 8 * pot_l + pot_ll) / (12 * dx);
  #else
    gz = -0.5 * (pot_r - pot_l) / dz;
  #endif

    // Add gravity term to Momentum
    dev_conserved[n_cells + id] += 0.5 * dt * gx * (d + d_n);
    dev_conserved[2 * n_cells + id] += 0.5 * dt * gy * (d + d_n);
    dev_conserved[3 * n_cells + id] += 0.5 * dt * gz * (d + d_n);

    // Add gravity term to Total Energy
    // Add the work done by the gravitational force
    dev_conserved[4 * n_cells + id] +=
        0.5 * dt * (gx * (d * vx + d_n * vx_n) + gy * (d * vy + d_n * vy_n) + gz * (d * vz + d_n * vz_n));

#endif  // GRAVITY
  }
}

/*! Returns whether a cell has crashed
 *
 *  \note
 *  It probably won't come up, but it's unclear why we don't consider a density of 0 or
 *  energy of 0 to be crashed... (I'm just keeping the logic consistent with what it used to be)
 */
__device__ bool Cell_Is_Crashed(Real density, Real Etot_density)
{
  return (density < 0.0) || (density != density) || (Etot_density < 0.0) || (Etot_density != Etot_density);
}

__global__ void PostUpdate_Conserved_Correct_Crashed_3D(Real *dev_conserved, int nx, int ny, int nz, int x_off,
                                                        int y_off, int z_off, int n_ghost, Real gamma, int n_fields,
                                                        SlowCellConditionChecker slow_check, int *any_error)
{
  int n_cells = nx * ny * nz;

  // get a global thread ID
  int id  = threadIdx.x + blockIdx.x * blockDim.x;
  int zid = id / (nx * ny);
  int yid = (id - zid * nx * ny) / nx;
  int xid = id - zid * nx * ny - yid * nx;

  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
      zid < nz - n_ghost) {
    // threads corresponding to real cells do the calculation

    // this logic get's skipped if we apply both a density floor and temperature floor
    // (we apply the logic if we only use one kind of floor or we use neither kind)
#if !(defined(DENSITY_FLOOR) && defined(TEMPERATURE_FLOOR))
    if (Cell_Is_Crashed(dev_conserved[grid_enum::density * n_cells + id],
                        dev_conserved[grid_enum::Energy * n_cells + id])) {
      printf("%3d %3d %3d Thread crashed in final update. %e - - - %e\n", xid + x_off, yid + y_off, zid + z_off,
             dev_conserved[grid_enum::density * n_cells + id], dev_conserved[grid_enum::Energy * n_cells + id]);
      bool success = Average_Cell_All_Fields(xid, yid, zid, nx, ny, nz, n_cells, n_fields, gamma, dev_conserved,
                                             n_ghost, slow_check);
      if (!success) {
        printf("%3d %3d %3d there was an issue with averaging the neighboring cells\n", xid + x_off, yid + y_off,
               zid + z_off);
        *any_error = 1;
      }
    }
#endif  // DENSITY_FLOOR
    /*
    d  =  dev_conserved[            id];
    d_inv = 1.0 / d;
    vx =  dev_conserved[1*n_cells + id] * d_inv;
    vy =  dev_conserved[2*n_cells + id] * d_inv;
    vz =  dev_conserved[3*n_cells + id] * d_inv;
    P  = (dev_conserved[4*n_cells + id] - 0.5*d*(vx*vx + vy*vy + vz*vz)) *
    (gamma - 1.0); if (P < 0.0) printf("%3d %3d %3d Negative pressure after
    final update. %f %f %f %f %f\n", xid, yid, zid, dev_conserved[4*n_cells +
    id], 0.5*d*vx*vx, 0.5*d*vy*vy, 0.5*d*vz*vz, P);
    */
  }
}
__device__ __host__ Real hydroInverseCrossingTime(Real const &E, Real const &d, Real const &d_inv, Real const &vx,
                                                  Real const &vy, Real const &vz, Real const &dx, Real const &dy,
                                                  Real const &dz, Real const &gamma)
{
  // Compute pressure and sound speed
  Real P  = (E - 0.5 * d * (vx * vx + vy * vy + vz * vz)) * (gamma - 1.0);
  Real cs = sqrt(d_inv * gamma * P);

  // Find maximum inverse crossing time in the cell (i.e. minimum crossing time)
  Real cellMaxInverseDt = fmax((fabs(vx) + cs) / dx, (fabs(vy) + cs) / dy);
  cellMaxInverseDt      = fmax(cellMaxInverseDt, (fabs(vz) + cs) / dz);
  cellMaxInverseDt      = fmax(cellMaxInverseDt, 0.0);

  return cellMaxInverseDt;
}

__device__ __host__ Real mhdInverseCrossingTime(Real const &E, Real const &d, Real const &d_inv, Real const &vx,
                                                Real const &vy, Real const &vz, Real const &avgBx, Real const &avgBy,
                                                Real const &avgBz, Real const &dx, Real const &dy, Real const &dz,
                                                Real const &gamma)
{
  // Compute the gas pressure and fast magnetosonic speed
  Real gasP = hydro_utilities::Calc_Pressure_Primitive(E, d, vx, vy, vz, gamma, avgBx, avgBy, avgBz);
  Real cf   = mhd::utils::fastMagnetosonicSpeed(d, gasP, avgBx, avgBy, avgBz, gamma);

  // Find maximum inverse crossing time in the cell (i.e. minimum crossing time)
  Real cellMaxInverseDt = fmax((fabs(vx) + cf) / dx, (fabs(vy) + cf) / dy);
  cellMaxInverseDt      = fmax(cellMaxInverseDt, (fabs(vz) + cf) / dz);
  cellMaxInverseDt      = fmax(cellMaxInverseDt, 0.0);

  return cellMaxInverseDt;
}

__global__ void Calc_dt_1D(Real *dev_conserved, Real *dev_dti, Real gamma, int n_ghost, int nx, Real dx)
{
  Real max_dti = -DBL_MAX;

  Real d, d_inv, vx, vy, vz, P, cs;
  int n_cells = nx;

  // Grid stride loop to perform as much of the reduction as possible. The
  // fact that `id` has type `size_t` is important. I'm not totally sure why
  // but setting it to int results in some kind of silent over/underflow issue
  // even though we're not hitting those kinds of numbers. Setting it to type
  // uint or size_t fixes them
  for (size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x) {
    // threads corresponding to real cells do the calculation
    if (id > n_ghost - 1 && id < n_cells - n_ghost) {
      // start timestep calculation here
      // every thread collects the conserved variables it needs from global
      // memory
      d     = dev_conserved[id];
      d_inv = 1.0 / d;
      vx    = dev_conserved[1 * n_cells + id] * d_inv;
      vy    = dev_conserved[2 * n_cells + id] * d_inv;
      vz    = dev_conserved[3 * n_cells + id] * d_inv;
      P     = (dev_conserved[4 * n_cells + id] - 0.5 * d * (vx * vx + vy * vy + vz * vz)) * (gamma - 1.0);
      P     = fmax(P, (Real)TINY_NUMBER);
      // find the max wavespeed in that cell, use it to calculate the inverse
      // timestep
      cs      = sqrt(d_inv * gamma * P);
      max_dti = fmax(max_dti, (fabs(vx) + cs) / dx);
    }
  }

  // do the grid wide reduction (find the max inverse timestep in the grid)
  reduction_utilities::gridReduceMax(max_dti, dev_dti);
}

__global__ void Calc_dt_2D(Real *dev_conserved, Real *dev_dti, Real gamma, int n_ghost, int nx, int ny, Real dx,
                           Real dy)
{
  Real max_dti = -DBL_MAX;

  Real d, d_inv, vx, vy, vz, P, cs;
  int xid, yid, n_cells;
  n_cells = nx * ny;

  // Grid stride loop to perform as much of the reduction as possible. The
  // fact that `id` has type `size_t` is important. I'm not totally sure why
  // but setting it to int results in some kind of silent over/underflow issue
  // even though we're not hitting those kinds of numbers. Setting it to type
  // uint or size_t fixes them
  for (size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x) {
    // get a global thread ID
    yid = id / nx;
    xid = id - yid * nx;

    // threads corresponding to real cells do the calculation
    if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost) {
      // every thread collects the conserved variables it needs from global
      // memory
      d     = dev_conserved[id];
      d_inv = 1.0 / d;
      vx    = dev_conserved[1 * n_cells + id] * d_inv;
      vy    = dev_conserved[2 * n_cells + id] * d_inv;
      vz    = dev_conserved[3 * n_cells + id] * d_inv;
      P     = (dev_conserved[4 * n_cells + id] - 0.5 * d * (vx * vx + vy * vy + vz * vz)) * (gamma - 1.0);
      P     = fmax(P, (Real)1.0e-20);
      // find the max wavespeed in that cell, use it to calculate the inverse
      // timestep
      cs      = sqrt(d_inv * gamma * P);
      max_dti = fmax(max_dti, fmax((fabs(vx) + cs) / dx, (fabs(vy) + cs) / dy));
    }
  }

  // do the grid wide reduction (find the max inverse timestep in the grid)
  reduction_utilities::gridReduceMax(max_dti, dev_dti);
}

__global__ void Calc_dt_3D(Real *dev_conserved, Real *dev_dti, Real gamma, int n_ghost, int n_fields, int nx, int ny,
                           int nz, Real dx, Real dy, Real dz)
{
  Real max_dti = -DBL_MAX;

  Real d, d_inv, vx, vy, vz, E;
  int xid, yid, zid, n_cells;

  n_cells = nx * ny * nz;

  // Grid stride loop to perform as much of the reduction as possible. The
  // fact that `id` has type `size_t` is important. I'm not totally sure why
  // but setting it to int results in some kind of silent over/underflow issue
  // even though we're not hitting those kinds of numbers. Setting it to type
  // uint or size_t fixes them
  for (size_t id = threadIdx.x + blockIdx.x * blockDim.x; id < n_cells; id += blockDim.x * gridDim.x) {
    // get a global thread ID
    cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);

    // threads corresponding to real cells do the calculation
    if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
        zid < nz - n_ghost) {
      // every thread collects the conserved variables it needs from global
      // memory
      d     = dev_conserved[id];
      d_inv = 1.0 / d;
      vx    = dev_conserved[1 * n_cells + id] * d_inv;
      vy    = dev_conserved[2 * n_cells + id] * d_inv;
      vz    = dev_conserved[3 * n_cells + id] * d_inv;
      E     = dev_conserved[4 * n_cells + id];

// Compute the maximum inverse crossing time in the cell
#ifdef MHD
      // Compute the cell centered magnetic field using a straight average of
      // the faces
      auto const magnetic_centered =
          mhd::utils::cellCenteredMagneticFields(dev_conserved, id, xid, yid, zid, n_cells, nx, ny);
      max_dti = fmax(max_dti, mhdInverseCrossingTime(E, d, d_inv, vx, vy, vz, magnetic_centered.x(),
                                                     magnetic_centered.y(), magnetic_centered.z(), dx, dy, dz, gamma));
#else   // not MHD
      max_dti = fmax(max_dti, hydroInverseCrossingTime(E, d, d_inv, vx, vy, vz, dx, dy, dz, gamma));
#endif  // MHD
    }
  }

  // do the grid wide reduction (find the max inverse timestep in the grid)
  reduction_utilities::gridReduceMax(max_dti, dev_dti);
}

Real Calc_dt_GPU(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real dx, Real dy, Real dz,
                 Real gamma)
{
  // Allocate the device memory
  cuda_utilities::DeviceVector<Real> static dev_dti(1);

  // Set the device side inverse time step to the smallest possible double so
  // that the reduction isn't using the maximum value of the previous iteration
  dev_dti.assign(std::numeric_limits<double>::lowest());

  // compute dt and store in dev_dti
  if (nx > 1 && ny == 1 && nz == 1)  // 1D
  {
    // set launch parameters for GPU kernels.
    cuda_utilities::AutomaticLaunchParams static const launchParams(Calc_dt_1D);
    hipLaunchKernelGGL(Calc_dt_1D, launchParams.get_numBlocks(), launchParams.get_threadsPerBlock(), 0, 0,
                       dev_conserved, dev_dti.data(), gamma, n_ghost, nx, dx);
  } else if (nx > 1 && ny > 1 && nz == 1)  // 2D
  {
    // set launch parameters for GPU kernels.
    cuda_utilities::AutomaticLaunchParams static const launchParams(Calc_dt_2D);
    hipLaunchKernelGGL(Calc_dt_2D, launchParams.get_numBlocks(), launchParams.get_threadsPerBlock(), 0, 0,
                       dev_conserved, dev_dti.data(), gamma, n_ghost, nx, ny, dx, dy);
  } else if (nx > 1 && ny > 1 && nz > 1)  // 3D
  {
    // set launch parameters for GPU kernels.
    cuda_utilities::AutomaticLaunchParams static const launchParams(Calc_dt_3D);
    hipLaunchKernelGGL(Calc_dt_3D, launchParams.get_numBlocks(), launchParams.get_threadsPerBlock(), 0, 0,
                       dev_conserved, dev_dti.data(), gamma, n_ghost, n_fields, nx, ny, nz, dx, dy, dz);
  }
  GPU_Error_Check();

  // Note: dev_dti[0] is DeviceVector syntactic sugar for returning a value via
  // cudaMemcpy
  return dev_dti[0];
}

__global__ void Temperature_Ceiling_Kernel(Real *conserved, int nx, int ny, int nz, int n_ghost, int n_fields,
                                           Real gamma, Real T_ceiling, int *counter)
{
  const int id      = threadIdx.x + blockIdx.x * blockDim.x;
  const int n_cells = nx * ny * nz;
  int xid, yid, zid;
  cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);
  const bool real_cell = (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost &&
                          zid > n_ghost - 1 && zid < nz - n_ghost);
  if (!real_cell) return;

  const Real d  = conserved[grid_enum::density * n_cells + id];
  const Real mx = conserved[grid_enum::momentum_x * n_cells + id];
  const Real my = conserved[grid_enum::momentum_y * n_cells + id];
  const Real mz = conserved[grid_enum::momentum_z * n_cells + id];
  const Real E  = conserved[grid_enum::Energy * n_cells + id];

  // compute 1/density (we take some care to avoid a source of NANs)
  const Real d_inv = 1.0 / (d + TINY_NUMBER * (d == 0.0));

  // calculate local kinetic energy
  const Real KE = 0.5 * d_inv * ((mx * mx) + ((my * my) + (mz * mz)));

  // convert T_ceiling to specific_eint_ceiling
  // -> keep in mind, that specific internal energy has units of velocity^2
  const Real particle_mass          = 0.6 * MP;
  const Real specific_eint_ceil_CGS = KB * T_ceiling / (particle_mass * (gamma - 1));
  const Real specific_eint_ceil     = specific_eint_ceil_CGS * (VELOCITY_UNIT * VELOCITY_UNIT);

  const Real local_eint_ceil = d * specific_eint_ceil;
  const Real local_etot_ceil = local_eint_ceil + KE;

  bool applied_ceiling = false;

  if (E > local_etot_ceil) {
    conserved[grid_enum::Energy * n_cells + id] = local_etot_ceil;
    applied_ceiling                             = true;
  }

#ifdef DE
  if (conserved[grid_enum::GasEnergy * n_cells + id] > local_eint_ceil) {
    conserved[grid_enum::GasEnergy * n_cells + id] = local_eint_ceil;
    applied_ceiling                                = true;
  }
#endif  // DE

  if (applied_ceiling) atomicAdd(counter, 1);
}

void Temperature_Ceiling(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real gamma,
                         Real T_ceiling)
{
  int n_cells = nx * ny * nz;
  int ngrid   = (n_cells + TPB - 1) / TPB;
  dim3 dim1dGrid(ngrid, 1, 1);
  dim3 dim1dBlock(TPB, 1, 1);

  cuda_utilities::DeviceVector<int> counter(1, true);
  int *dev_counter = counter.data();

  if (nx > 1 && ny > 1 && nz > 1) {  // 3D
    hipLaunchKernelGGL(Temperature_Ceiling_Kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost,
                       n_fields, gamma, T_ceiling, dev_counter);
  }
  int host_counter = counter[0];
  if (host_counter > 0) {
    printf("HYDRO WARNING: Temperature Ceiling applied to num_cells: %d \n", host_counter);
  }
}

__device__ Real SlowCellConditionChecker::max_dti_if_slow(Real E, Real d, Real d_inv, Real vx, Real vy, Real vz,
                                                          Real gamma) const
{
#ifndef AVERAGE_SLOW_CELLS
  return -1.0;
#else
  Real max_dti = hydroInverseCrossingTime(E, d, d_inv, vx, vy, vz, dx, dy, dz, gamma);
  return (max_dti > max_dti_slow) ? max_dti : -1.0;
#endif
}

#ifdef AVERAGE_SLOW_CELLS

void Average_Slow_Cells(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real gamma,
                        SlowCellConditionChecker slow_check, Real xbound, Real ybound, Real zbound, int nx_offset,
                        int ny_offset, int nz_offset)
{
  // set values for GPU kernels
  int n_cells = nx * ny * nz;
  int ngrid   = (n_cells + TPB - 1) / TPB;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB, 1, 1);

  if (nx > 1 && ny > 1 && nz > 1) {  // 3D
    hipLaunchKernelGGL(Average_Slow_Cells_3D, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, n_fields,
                       gamma, slow_check, xbound, ybound, zbound, nx_offset, ny_offset, nz_offset);
  }
}

__global__ void Average_Slow_Cells_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields,
                                      Real gamma, SlowCellConditionChecker slow_check, Real xbound, Real ybound,
                                      Real zbound, int nx_offset, int ny_offset, int nz_offset)
{
  int id, xid, yid, zid, n_cells;
  Real d, d_inv, vx, vy, vz, E, max_dti;
  Real speed, temp, P, cs;

  // get a global thread ID
  id      = threadIdx.x + blockIdx.x * blockDim.x;
  n_cells = nx * ny * nz;

  cuda_utilities::compute3DIndices(id, nx, ny, xid, yid, zid);

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
      zid < nz - n_ghost) {
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
    vy    = dev_conserved[2 * n_cells + id] * d_inv;
    vz    = dev_conserved[3 * n_cells + id] * d_inv;
    E     = dev_conserved[4 * n_cells + id];

    // retrieve the max inverse crossing time in the cell if the cell meets the threshold for being a slow-cell.
    // (if the cell doesn't meet the threshold, a negative value is returned instead)
    max_dti = slow_check.max_dti_if_slow(E, d, d_inv, vx, vy, vz, gamma);

    if (max_dti >= 0) {
      speed  = sqrt(vx * vx + vy * vy + vz * vz);
      temp   = (gamma - 1) * (E - 0.5 * (speed * speed) * d) * ENERGY_UNIT / (d * DENSITY_UNIT / 0.6 / MP) / KB;
      P      = (E - 0.5 * d * (vx * vx + vy * vy + vz * vz)) * (gamma - 1.0);
      cs     = sqrt(d_inv * gamma * P) * VELOCITY_UNIT * 1e-5;
      Real x = xbound + (nx_offset + xid - n_ghost + 0.5) * slow_check.dx;
      Real y = ybound + (ny_offset + yid - n_ghost + 0.5) * slow_check.dy;
      Real z = zbound + (nz_offset + zid - n_ghost + 0.5) * slow_check.dz;
      // Average this cell
      kernel_printf(
          " Average Slow Cell [ %.5e %.5e %.5e ] -> dt_cell=%f    dt_min=%f, n=%.3e, "
          "T=%.3e, v=%.3e (%.3e, %.3e, %.3e), cs=%.3e\n",
          x, y, z, 1. / max_dti, 1. / slow_check.max_dti_slow, dev_conserved[id] * DENSITY_UNIT / 0.6 / MP, temp,
          speed * VELOCITY_UNIT * 1e-5, vx * VELOCITY_UNIT * 1e-5, vy * VELOCITY_UNIT * 1e-5, vz * VELOCITY_UNIT * 1e-5,
          cs);
      Average_Cell_All_Fields(xid, yid, zid, nx, ny, nz, n_cells, n_fields, gamma, dev_conserved, n_ghost, slow_check);
    }
  }
}
#endif  // AVERAGE_SLOW_CELLS

#ifdef DE
__global__ void Partial_Update_Advected_Internal_Energy_1D(Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, int nx,
                                                           int n_ghost, Real dx, Real dt, Real gamma, int n_fields)
{
  int id, xid, n_cells;
  int imo, ipo;
  Real d, d_inv, vx, vy, vz;
  Real vx_imo, vx_ipo;
  Real P, E, E_kin, GE;

  Real dtodx = dt / dx;
  n_cells    = nx;

  // get a global thread ID
  id  = threadIdx.x + blockIdx.x * blockDim.x;
  xid = id;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost) {
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
    vy    = dev_conserved[2 * n_cells + id] * d_inv;
    vz    = dev_conserved[3 * n_cells + id] * d_inv;
    // PRESSURE_DE
    E     = dev_conserved[4 * n_cells + id];
    GE    = dev_conserved[(n_fields - 1) * n_cells + id];
    E_kin = 0.5 * d * (vx * vx + vy * vy + vz * vz);
    P     = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, GE, gamma);
    P     = fmax(P, (Real)TINY_NUMBER);

    imo = xid - 1;
    ipo = xid + 1;

    vx_imo = dev_conserved[1 * n_cells + imo] / dev_conserved[imo];
    vx_ipo = dev_conserved[1 * n_cells + ipo] / dev_conserved[ipo];

    // Use center values of neighbor cells for the divergence of velocity
    dev_conserved[(n_fields - 1) * n_cells + id] += 0.5 * P * (dtodx * (vx_imo - vx_ipo));
  }
}

__global__ void Partial_Update_Advected_Internal_Energy_2D(Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, Real *Q_Ly,
                                                           Real *Q_Ry, int nx, int ny, int n_ghost, Real dx, Real dy,
                                                           Real dt, Real gamma, int n_fields)
{
  int id, xid, yid, n_cells;
  int imo, jmo;
  int ipo, jpo;
  Real d, d_inv, vx, vy, vz;
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo;
  Real P, E, E_kin, GE;

  Real dtodx = dt / dx;
  Real dtody = dt / dy;
  n_cells    = nx * ny;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  id          = threadIdx.x + blockId * blockDim.x;
  yid         = id / nx;
  xid         = id - yid * nx;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost) {
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
    vy    = dev_conserved[2 * n_cells + id] * d_inv;
    vz    = dev_conserved[3 * n_cells + id] * d_inv;
    // PRESSURE_DE
    E     = dev_conserved[4 * n_cells + id];
    GE    = dev_conserved[(n_fields - 1) * n_cells + id];
    E_kin = 0.5 * d * (vx * vx + vy * vy + vz * vz);
    P     = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, GE, gamma);
    P     = fmax(P, (Real)TINY_NUMBER);

    imo = xid - 1 + yid * nx;
    ipo = xid + 1 + yid * nx;
    jmo = xid + (yid - 1) * nx;
    jpo = xid + (yid + 1) * nx;

    vx_imo = dev_conserved[1 * n_cells + imo] / dev_conserved[imo];
    vx_ipo = dev_conserved[1 * n_cells + ipo] / dev_conserved[ipo];
    vy_jmo = dev_conserved[2 * n_cells + jmo] / dev_conserved[jmo];
    vy_jpo = dev_conserved[2 * n_cells + jpo] / dev_conserved[jpo];

    // Use center values of neighbor cells for the divergence of velocity
    dev_conserved[(n_fields - 1) * n_cells + id] += 0.5 * P * (dtodx * (vx_imo - vx_ipo) + dtody * (vy_jmo - vy_jpo));
  }
}

__global__ void Partial_Update_Advected_Internal_Energy_3D(Real *dev_conserved, Real *Q_Lx, Real *Q_Rx, Real *Q_Ly,
                                                           Real *Q_Ry, Real *Q_Lz, Real *Q_Rz, int nx, int ny, int nz,
                                                           int n_ghost, Real dx, Real dy, Real dz, Real dt, Real gamma,
                                                           int n_fields)
{
  int id, xid, yid, zid, n_cells;
  int imo, jmo, kmo;
  int ipo, jpo, kpo;
  Real d, d_inv, vx, vy, vz;
  Real vx_imo, vx_ipo, vy_jmo, vy_jpo, vz_kmo, vz_kpo;
  Real P, E, E_kin, GE;
  // Real vx_L, vx_R, vy_L, vy_R, vz_L, vz_R;

  Real dtodx = dt / dx;
  Real dtody = dt / dy;
  Real dtodz = dt / dz;
  n_cells    = nx * ny * nz;

  // get a global thread ID
  id  = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx * ny);
  yid = (id - zid * nx * ny) / nx;
  xid = id - zid * nx * ny - yid * nx;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
      zid < nz - n_ghost) {
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
    vy    = dev_conserved[2 * n_cells + id] * d_inv;
    vz    = dev_conserved[3 * n_cells + id] * d_inv;
    // PRESSURE_DE
    E     = dev_conserved[4 * n_cells + id];
    GE    = dev_conserved[(n_fields - 1) * n_cells + id];
    E_kin = hydro_utilities::Calc_Kinetic_Energy_From_Velocity(d, vx, vy, vz);
  #ifdef MHD
    // Add the magnetic energy
    auto magnetic_centered = mhd::utils::cellCenteredMagneticFields(dev_conserved, id, xid, yid, zid, n_cells, nx, ny);
    E_kin += mhd::utils::computeMagneticEnergy(magnetic_centered.x(), magnetic_centered.y(), magnetic_centered.z());
  #endif  // MHD
    P = hydro_utilities::Get_Pressure_From_DE(E, E - E_kin, GE, gamma);
    P = fmax(P, (Real)TINY_NUMBER);

    imo = xid - 1 + yid * nx + zid * nx * ny;
    jmo = xid + (yid - 1) * nx + zid * nx * ny;
    kmo = xid + yid * nx + (zid - 1) * nx * ny;

    ipo = xid + 1 + yid * nx + zid * nx * ny;
    jpo = xid + (yid + 1) * nx + zid * nx * ny;
    kpo = xid + yid * nx + (zid + 1) * nx * ny;

    vx_imo = dev_conserved[1 * n_cells + imo] / dev_conserved[imo];
    vx_ipo = dev_conserved[1 * n_cells + ipo] / dev_conserved[ipo];
    vy_jmo = dev_conserved[2 * n_cells + jmo] / dev_conserved[jmo];
    vy_jpo = dev_conserved[2 * n_cells + jpo] / dev_conserved[jpo];
    vz_kmo = dev_conserved[3 * n_cells + kmo] / dev_conserved[kmo];
    vz_kpo = dev_conserved[3 * n_cells + kpo] / dev_conserved[kpo];

    // Use center values of neighbor cells for the divergence of velocity
    dev_conserved[(n_fields - 1) * n_cells + id] +=
        0.5 * P * (dtodx * (vx_imo - vx_ipo) + dtody * (vy_jmo - vy_jpo) + dtodz * (vz_kmo - vz_kpo));

    // OPTION 2: Use the reconstructed velocities to compute the velocity
    // gradient
    // Use the reconstructed Velocities instead of neighbor cells centered
    // values
    // vx_R = Q_Lx[1*n_cells + id]  / Q_Lx[id];
    // vx_L = Q_Rx[1*n_cells + imo] / Q_Rx[imo];
    // vy_R = Q_Ly[2*n_cells + id]  / Q_Ly[id];
    // vy_L = Q_Ry[2*n_cells + jmo] / Q_Ry[jmo];
    // vz_R = Q_Lz[3*n_cells + id]  / Q_Lz[id];
    // vz_L = Q_Rz[3*n_cells + kmo] / Q_Rz[kmo];

    // Use the reconstructed Velocities instead of neighbor cells centered
    // values
    //  dev_conserved[(n_fields-1)*n_cells + id] +=  P * ( dtodx * ( vx_L - vx_R
    //  ) + dtody * ( vy_L - vy_R ) + dtodz * ( vz_L - vz_R ) );
  }
}

/*! The folliowing function is used to retrieve the total energy density if the cell isn't crashed.
 *  If the cell is returned, 0 is returned
 *
 *  \note
 *  This is useful for implementing `Select_Internal_Energy_ND`
 */
__device__ Real E_If_Not_Crashed(Real *dev_conserved, int n_cells, int spatial_idx)
{
  Real d = dev_conserved[grid_enum::density * n_cells + spatial_idx];
  Real E = dev_conserved[grid_enum::Energy * n_cells + spatial_idx];
  // NOTE: don't try to get clever here (we explicitly want to return 0 in cases where E is NaN)
  return Cell_Is_Crashed(d, E) ? 0 : E;
}

__global__ void Select_Internal_Energy_1D(Real *dev_conserved, int nx, int n_ghost, int n_fields)
{
  int id, xid, n_cells;
  Real d, d_inv, vx, vy, vz, E, U_total, U_advected, U, Emax;
  int imo, ipo;
  n_cells = nx;

  Real eta_1 = DE_ETA_1;
  Real eta_2 = DE_ETA_2;

  // get a global thread ID
  id  = threadIdx.x + blockIdx.x * blockDim.x;
  xid = id;

  imo = max(xid - 1, n_ghost);
  ipo = min(xid + 1, nx - n_ghost - 1);

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost) {
    // every thread collects the conserved variables it needs from global memory
    d          = dev_conserved[id];
    d_inv      = 1.0 / d;
    vx         = dev_conserved[1 * n_cells + id] * d_inv;
    vy         = dev_conserved[2 * n_cells + id] * d_inv;
    vz         = dev_conserved[3 * n_cells + id] * d_inv;
    E          = dev_conserved[4 * n_cells + id];
    U_advected = dev_conserved[(n_fields - 1) * n_cells + id];
    U_total    = E - 0.5 * d * (vx * vx + vy * vy + vz * vz);

    // We will deal with this crashed cell later in a different kernel... (at the time of writing, a 1D
    // simulation always uses floors for this purpose)
    if (Cell_Is_Crashed(d, E)) return;

    // find the max nearby total energy (from the local cell and any uncrashed neighbors)
    // -> we take the stance that "crashed" neighbors are unreliable, even if total energy looks ok
    // -> to effectively ignore a crashed neighbor, the `E_If_Not_Crashed` function provides a value of 0,
    //    instead of the neighboring cell's E if the neighbor crashed
    Emax = E;
    Emax = fmax(Emax, E_If_Not_Crashed(dev_conserved, n_cells, imo));
    Emax = fmax(Emax, E_If_Not_Crashed(dev_conserved, n_cells, ipo));

    // Ordinarily, we only use the "advected" internal energy if both:
    // - the thermal energy divided by total energy is a small fraction (smaller than eta_1)
    // - AND we aren't masking shock heating (details controlled by Emax & eta_2)
    // We ALSO explicitly use the "advected" internal energy if the total energy is positive but doesn't
    // exceed kinetic energy (i.e. U_total <= 0).
    bool prefer_U_total = (U_total > E * eta_1) or (U_total > Emax * eta_2);
    if (prefer_U_total and (U_total > 0)) {
      U = U_total;
    } else {
      U = U_advected;
    }

    // Optional: Avoid Negative Internal  Energies
    U = fmax(U, (Real)TINY_NUMBER);

    // Write Selected internal energy to the GasEnergy array ONLY
    // to avoid mixing updated and non-updated values of E
    // since the Dual Energy condition depends on the neighbor cells
    dev_conserved[(n_fields - 1) * n_cells + id] = U;
  }
}

__global__ void Select_Internal_Energy_2D(Real *dev_conserved, int nx, int ny, int n_ghost, int n_fields)
{
  int id, xid, yid, n_cells;
  Real d, d_inv, vx, vy, vz, E, U_total, U_advected, U, Emax;
  int imo, ipo, jmo, jpo;
  n_cells = nx * ny;

  Real eta_1 = DE_ETA_1;
  Real eta_2 = DE_ETA_2;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  id          = threadIdx.x + blockId * blockDim.x;
  yid         = id / nx;
  xid         = id - yid * nx;

  imo = max(xid - 1, n_ghost);
  imo = imo + yid * nx;
  ipo = min(xid + 1, nx - n_ghost - 1);
  ipo = ipo + yid * nx;
  jmo = max(yid - 1, n_ghost);
  jmo = xid + jmo * nx;
  jpo = min(yid + 1, ny - n_ghost - 1);
  jpo = xid + jpo * nx;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost) {
    // every thread collects the conserved variables it needs from global memory
    d          = dev_conserved[id];
    d_inv      = 1.0 / d;
    vx         = dev_conserved[1 * n_cells + id] * d_inv;
    vy         = dev_conserved[2 * n_cells + id] * d_inv;
    vz         = dev_conserved[3 * n_cells + id] * d_inv;
    E          = dev_conserved[4 * n_cells + id];
    U_advected = dev_conserved[(n_fields - 1) * n_cells + id];
    U_total    = E - 0.5 * d * (vx * vx + vy * vy + vz * vz);

    // We will deal with this crashed cell later in a different kernel... (at the time of writing, a 1D
    // simulation always uses floors for this purpose)
    if (Cell_Is_Crashed(d, E)) return;

    // find the max nearby total energy (from the local cell and any uncrashed neighbors)
    // -> we take the stance that "crashed" neighbors are unreliable, even if total energy looks ok
    // -> to effectively ignore a crashed neighbor, the `E_If_Not_Crashed` function provides a value of 0,
    //    instead of the neighboring cell's E if the neighbor crashed
    Emax = E;
    Emax = fmax(Emax, E_If_Not_Crashed(dev_conserved, n_cells, imo));
    Emax = fmax(Emax, E_If_Not_Crashed(dev_conserved, n_cells, ipo));
    Emax = fmax(Emax, E_If_Not_Crashed(dev_conserved, n_cells, jmo));
    Emax = fmax(Emax, E_If_Not_Crashed(dev_conserved, n_cells, jpo));

    // We only use the "advected" internal energy if the following 3 conditions are satisfied:
    // - the thermal energy divided by total energy is a small fraction (smaller than eta_1)
    // - AND we aren't masking shock heating (details controlled by Emax & eta_2)
    // We ALSO explicitly use the "advected" internal energy if the total energy is positive but doesn't
    // exceed kinetic energy (i.e. U_total <= 0).
    bool prefer_U_total = (U_total > E * eta_1) or (U_total > Emax * eta_2);
    if (prefer_U_total and (U_total > 0)) {
      U = U_total;
    } else {
      U = U_advected;
    }

    // Optional: Avoid Negative Internal  Energies
    U = fmax(U, (Real)TINY_NUMBER);

    // Write Selected internal energy to the GasEnergy array ONLY
    // to avoid mixing updated and non-updated values of E
    // since the Dual Energy condition depends on the neighbour cells
    dev_conserved[(n_fields - 1) * n_cells + id] = U;
  }
}

__global__ void Select_Internal_Energy_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields)
{
  // The scenario where E > 0 and E doesn't exceed the kinetic energy (i.e. U_total <= 0),
  // comes up with some frequency when we include particle feedback.
  // -> in that scenario, we explicitly use the value held by the U_advected field
  // -> the separate `Sync_Energies_3D` kernel, we then override the E field with KE + U_advected
  // -> one might argue we should actually override the E field with KE + U_advected before this kernel
  //    (we leave that for future consideration)
  // -> regardless, it is **REALLY IMPORTANT** that the E field is **NOT** modified in this kernel
  //    (modifying it will produce race conditions!!!)

  int id, xid, yid, zid, n_cells;
  Real d, d_inv, vx, vy, vz, E, U_total, U_advected, U, Emax;
  int imo, ipo, jmo, jpo, kmo, kpo;
  n_cells = nx * ny * nz;

  Real eta_1 = DE_ETA_1;
  Real eta_2 = DE_ETA_2;

  // get a global thread ID
  id  = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx * ny);
  yid = (id - zid * nx * ny) / nx;
  xid = id - zid * nx * ny - yid * nx;

  imo = max(xid - 1, n_ghost);
  imo = imo + yid * nx + zid * nx * ny;
  ipo = min(xid + 1, nx - n_ghost - 1);
  ipo = ipo + yid * nx + zid * nx * ny;
  jmo = max(yid - 1, n_ghost);
  jmo = xid + jmo * nx + zid * nx * ny;
  jpo = min(yid + 1, ny - n_ghost - 1);
  jpo = xid + jpo * nx + zid * nx * ny;
  kmo = max(zid - 1, n_ghost);
  kmo = xid + yid * nx + kmo * nx * ny;
  kpo = min(zid + 1, nz - n_ghost - 1);
  kpo = xid + yid * nx + kpo * nx * ny;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
      zid < nz - n_ghost) {
    // every thread collects the conserved variables it needs from global memory
    d          = dev_conserved[id];
    d_inv      = 1.0 / d;
    vx         = dev_conserved[1 * n_cells + id] * d_inv;
    vy         = dev_conserved[2 * n_cells + id] * d_inv;
    vz         = dev_conserved[3 * n_cells + id] * d_inv;
    E          = dev_conserved[4 * n_cells + id];
    U_advected = dev_conserved[(n_fields - 1) * n_cells + id];
    U_total    = E - 0.5 * d * (vx * vx + vy * vy + vz * vz);

    // We will deal with this crashed cell later in a different kernel... (at the time of writing, a 1D
    // simulation always uses floors for this purpose)
    if (Cell_Is_Crashed(d, E)) return;

    // find the max nearby total energy (from the local cell and any uncrashed neighbors)
    // -> we take the stance that "crashed" neighbors are unreliable, even if total energy looks ok
    // -> to effectively ignore a crashed neighbor, the `E_If_Not_Crashed` function provides a value of 0,
    //    instead of the neighboring cell's E if the neighbor crashed
    Emax = E;
    Emax = fmax(Emax, E_If_Not_Crashed(dev_conserved, n_cells, imo));
    Emax = fmax(Emax, E_If_Not_Crashed(dev_conserved, n_cells, ipo));
    Emax = fmax(Emax, E_If_Not_Crashed(dev_conserved, n_cells, jmo));
    Emax = fmax(Emax, E_If_Not_Crashed(dev_conserved, n_cells, jpo));
    Emax = fmax(Emax, E_If_Not_Crashed(dev_conserved, n_cells, kmo));
    Emax = fmax(Emax, E_If_Not_Crashed(dev_conserved, n_cells, kpo));

    // Ordinarily, we only use the "advected" internal energy if both:
    // - the thermal energy divided by total energy is a small fraction (smaller than eta_1)
    // - AND we aren't masking shock heating (details controlled by Emax & eta_2)
    // We also explicitly use the "advected" internal energy if the total energy is positive but doesn't
    // exceed kinetic energy (i.e. U_total <= 0). This scenario comes up in simulations with particle-based
    // feedback.
    bool prefer_U_total = (U_total > E * eta_1) or (U_total > Emax * eta_2);
    if (prefer_U_total and (U_total > 0)) {
      U = U_total;
    } else {
      U = U_advected;
    }

    // Optional: Avoid Negative Internal  Energies
    U = fmax(U, (Real)TINY_NUMBER);

    // Write Selected internal energy to the GasEnergy array ONLY
    // to avoid mixing updated and non-updated values of E
    // since the Dual Energy condition depends on the neighbour cells
    dev_conserved[(n_fields - 1) * n_cells + id] = U;
  }
}

__global__ void Sync_Energies_1D(Real *dev_conserved, int nx, int n_ghost, Real gamma, int n_fields)
{
  int id, xid, n_cells;
  Real d, d_inv, vx, vy, vz, U;
  n_cells = nx;

  // get a global thread ID
  id  = threadIdx.x + blockIdx.x * blockDim.x;
  xid = id;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost) {
    // every thread collects the conserved variables it needs from global memory
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
    vy    = dev_conserved[2 * n_cells + id] * d_inv;
    vz    = dev_conserved[3 * n_cells + id] * d_inv;
    U     = dev_conserved[(n_fields - 1) * n_cells + id];

    // Use the previously selected Internal Energy to update the total energy
    dev_conserved[4 * n_cells + id] = 0.5 * d * (vx * vx + vy * vy + vz * vz) + U;
  }
}

__global__ void Sync_Energies_2D(Real *dev_conserved, int nx, int ny, int n_ghost, Real gamma, int n_fields)
{
  int id, xid, yid, n_cells;
  Real d, d_inv, vx, vy, vz, U;
  n_cells = nx * ny;

  // get a global thread ID
  int blockId = blockIdx.x + blockIdx.y * gridDim.x;
  id          = threadIdx.x + blockId * blockDim.x;
  yid         = id / nx;
  xid         = id - yid * nx;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost) {
    // every thread collects the conserved variables it needs from global memory
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
    vy    = dev_conserved[2 * n_cells + id] * d_inv;
    vz    = dev_conserved[3 * n_cells + id] * d_inv;
    U     = dev_conserved[(n_fields - 1) * n_cells + id];

    // Use the previously selected Internal Energy to update the total energy
    dev_conserved[4 * n_cells + id] = 0.5 * d * (vx * vx + vy * vy + vz * vz) + U;
  }
}

__global__ void Sync_Energies_3D(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, Real gamma, int n_fields)
{
  // Called in a separate kernel to avoid interfering with energy selection in
  // Select_Internal_Energy

  int id, xid, yid, zid, n_cells;
  Real d, d_inv, vx, vy, vz, U;
  n_cells = nx * ny * nz;

  // get a global thread ID
  id  = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx * ny);
  yid = (id - zid * nx * ny) / nx;
  xid = id - zid * nx * ny - yid * nx;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
      zid < nz - n_ghost) {
    // every thread collects the conserved variables it needs from global memory
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
    vy    = dev_conserved[2 * n_cells + id] * d_inv;
    vz    = dev_conserved[3 * n_cells + id] * d_inv;
    U     = dev_conserved[(n_fields - 1) * n_cells + id];

    // Use the previously selected Internal Energy to update the total energy
    dev_conserved[4 * n_cells + id] = 0.5 * d * (vx * vx + vy * vy + vz * vz) + U;
  }
}

#endif  // DE

void Apply_Temperature_Floor(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields, Real U_floor)
{
  // set values for GPU kernels
  int n_cells = nx * ny * nz;
  int ngrid   = (n_cells + TPB - 1) / TPB;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB, 1, 1);

  hipLaunchKernelGGL(Temperature_Floor_Kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost,
                     n_fields, U_floor);
}

__global__ void Temperature_Floor_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int n_fields,
                                         Real U_floor)
{
  int id, xid, yid, zid, n_cells;
  Real d, d_inv, vx, vy, vz, E, Ekin, Udens;
  n_cells = nx * ny * nz;

  // get a global thread ID
  id  = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx * ny);
  yid = (id - zid * nx * ny) / nx;
  xid = id - zid * nx * ny - yid * nx;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
      zid < nz - n_ghost) {
    d     = dev_conserved[id];
    d_inv = 1.0 / d;
    vx    = dev_conserved[1 * n_cells + id] * d_inv;
    vy    = dev_conserved[2 * n_cells + id] * d_inv;
    vz    = dev_conserved[3 * n_cells + id] * d_inv;
    E     = dev_conserved[4 * n_cells + id];
    Ekin  = 0.5 * d * (vx * vx + vy * vy + vz * vz);

    int num_applications = 0;

    Udens = (E - Ekin);
    if (Udens / d < U_floor) {
      num_applications++;
      dev_conserved[4 * n_cells + id] = Ekin + d * U_floor;
    }

#ifdef DE
    Udens = dev_conserved[(n_fields - 1) * n_cells + id];
    if (Udens / d < U_floor) {
      num_applications++;
      dev_conserved[(n_fields - 1) * n_cells + id] = d * U_floor;
    }
#else
    Udens = -123456789;  // set to a dumb-looking number so that it's clear that it's not real when printing it
#endif

    if (num_applications > 0) {
      printf("T_Floor %3d %3d %3d d: %e spec_Ekin:%e spec_eint_floor: %e BC: E_dens:%e GasEnergy_dens:%e\n", xid, yid,
             zid, d, Ekin, U_floor, E, Udens);
    }
  }
}

__device__ Real Average_Cell_Single_Field(int field_indx, int i, int j, int k, int nx, int ny, int nz, int ncells,
                                          Real *conserved)
{
  Real v_l, v_r, v_d, v_u, v_b, v_t, v_avrg;
  int id;

  id                                  = (i - 1) + (j)*nx + (k)*nx * ny;
  v_l                                 = conserved[field_indx * ncells + id];
  id                                  = (i + 1) + (j)*nx + (k)*nx * ny;
  v_r                                 = conserved[field_indx * ncells + id];
  id                                  = (i) + (j - 1) * nx + (k)*nx * ny;
  v_d                                 = conserved[field_indx * ncells + id];
  id                                  = (i) + (j + 1) * nx + (k)*nx * ny;
  v_u                                 = conserved[field_indx * ncells + id];
  id                                  = (i) + (j)*nx + (k - 1) * nx * ny;
  v_b                                 = conserved[field_indx * ncells + id];
  id                                  = (i) + (j)*nx + (k + 1) * nx * ny;
  v_t                                 = conserved[field_indx * ncells + id];
  v_avrg                              = (v_l + v_r + v_d + v_u + v_b + v_t) / 6;
  id                                  = (i) + (j)*nx + (k)*nx * ny;
  conserved[field_indx * ncells + id] = v_avrg;
  return v_avrg;
}

__device__ bool Average_Cell_All_Fields(int i, int j, int k, int nx, int ny, int nz, int ncells, int n_fields,
                                        Real gamma, Real *conserved, int stale_depth,
                                        SlowCellConditionChecker slow_check)
{
  int id = i + (j)*nx + (k)*nx * ny;

  // print out the the values stored in the cell before the correction
  // -> we put this logic (and variable declarations in its own scope) to force us to redeclare variables later
  //    (this lets the compiler help us avoid silly errors where we forget to load a variable)
  {
    Real d, mx, my, mz, E, P, Udens;
    d  = conserved[grid_enum::density * ncells + id];
    mx = conserved[grid_enum::momentum_x * ncells + id];
    my = conserved[grid_enum::momentum_y * ncells + id];
    mz = conserved[grid_enum::momentum_z * ncells + id];
    E  = conserved[grid_enum::Energy * ncells + id];
    P  = (E - (0.5 / d) * (mx * mx + my * my + mz * mz)) * (gamma - 1.0);

#ifdef DE
    Udens               = conserved[grid_enum::GasEnergy * ncells + id];
    Real kinetic_energy = hydro_utilities::Calc_Kinetic_Energy_From_Momentum(d, mx, my, mz);
    P                   = hydro_utilities::Get_Pressure_From_DE(E, E - kinetic_energy, Udens, gamma);
#else
    Udens = -123456789;  // set to a dumb-looking number so that it's clear that it's not real when printing it
    P     = hydro_utilities::Calc_Pressure_Conserved(E, d, mx, my, mz, gamma);
#endif
    printf("%3d %3d %3d BC: d: %e  E:%e  P:%e  vx:%e  vy:%e  vz:%e  Uadv:%e\n", i, j, k, d, E, P, mx / d, my / d,
           mz / d, Udens);
  }

  // initialize the variables that we use to accumulate information about neighboring values
  int N = 0;
  Real d_av, vx_av, vy_av, vz_av, P_av;
  d_av = vx_av = vy_av = vz_av = P_av = 0.0;
#ifdef SCALAR
  Real scalar[NSCALARS], scalar_av[NSCALARS];
  for (int n = 0; n < NSCALARS; n++) {  // NOLINT
    scalar_av[n] = 0.0;
  }
#endif

  for (int kk = k - 1; kk <= k + 1; kk++) {
    for (int jj = j - 1; jj <= j + 1; jj++) {
      for (int ii = i - 1; ii <= i + 1; ii++) {
        if (ii <= stale_depth - 1 || ii >= nx - stale_depth || jj <= stale_depth - 1 || jj >= ny - stale_depth ||
            kk <= stale_depth - 1 || kk >= nz - stale_depth) {
          continue;
        }

        int idn = ii + jj * nx + kk * nx * ny;
        Real d  = conserved[grid_enum::density * ncells + idn];
        Real mx = conserved[grid_enum::momentum_x * ncells + idn];
        Real my = conserved[grid_enum::momentum_y * ncells + idn];
        Real mz = conserved[grid_enum::momentum_z * ncells + idn];
        Real E  = conserved[grid_enum::Energy * ncells + idn];

        // this function CAN use the "advected internal energy" field to compute pressure when the dual energy
        // formalism. This is because this function has an explicit pre-condition that this function can only
        // be applied when the "advected internal energy" field and the "total energy" fields are properly reconciled
#ifdef DE
        Real P = hydro_utilities::Get_Pressure_From_DE(
            E, E - hydro_utilities::Calc_Kinetic_Energy_From_Momentum(d, mx, my, mz),
            conserved[grid_enum::GasEnergy * ncells + id], gamma);
#else
        Real P = hydro_utilities::Calc_Pressure_Conserved(E, d, mx, my, mz, gamma);
#endif

#ifdef SCALAR
        for (int n = 0; n < NSCALARS; n++) {  // NOLINT
          scalar[n] = conserved[grid_enum::scalar * ncells + idn];
        }
#endif
        Real d_inv = 1.0 / d;
        if (d > 0.0 && P > 0.0 && not slow_check.is_slow(E, d, d_inv, mx * d_inv, my * d_inv, mz * d_inv, gamma)) {
          d_av += d;
          vx_av += mx;
          vy_av += my;
          vz_av += mz;
          P_av += P / (gamma - 1.0);
#ifdef SCALAR
          for (int n = 0; n < NSCALARS; n++) {  // NOLINT
            scalar_av[n] += scalar[n];
          }
#endif
          N++;
        } else if ((ii != i) || (jj != j) || (kk != k)) {
#ifdef DE
          Real Udens = conserved[grid_enum::GasEnergy * ncells + idn];
#else
          Real Udens =
              -123456789;  // set to a dumb-looking number so that it's clear that it's not real when printing it
#endif
          printf("%3d %3d %3d skipped-neighbor: d: %e  E:%e  P:%e  vx:%e  vy:%e  vz:%e  Uadv:%e\n", ii, jj, kk, d, E, P,
                 mx / d, my / d, mz / d, Udens);
        }
      }
    }
  }

  // update the accumulator variables so that they now hold averages
  P_av  = P_av / N;
  vx_av = vx_av / d_av;
  vy_av = vy_av / d_av;
  vz_av = vz_av / d_av;
#ifdef SCALAR
  for (int n = 0; n < NSCALARS; n++) {  // NOLINT
    scalar_av[n] = scalar_av[n] / d_av;
  }
#endif
  d_av          = d_av / N;
  Real Udens_av = P_av / (gamma - 1.0);
  Real E_av     = Udens_av + 0.5 * d_av * math_utils::SquareMagnitude(vx_av, vy_av, vz_av);

  // replace cell values with new averaged values
  conserved[id + ncells * grid_enum::density]    = d_av;
  conserved[id + ncells * grid_enum::momentum_x] = d_av * vx_av;
  conserved[id + ncells * grid_enum::momentum_y] = d_av * vy_av;
  conserved[id + ncells * grid_enum::momentum_z] = d_av * vz_av;
  conserved[id + ncells * grid_enum::Energy]     = E_av;
#ifdef DE
  conserved[id + ncells * grid_enum::GasEnergy] = Udens_av;
#endif
#ifdef SCALAR
  for (int n = 0; n < NSCALARS; n++) {  // NOLINT
    conserved[id + ncells * grid_enum::scalar] = d_av * scalar_av[n];
  }
#endif

  // print out the values now that they have been replaced
  printf("%3d %3d %3d FC: d: %e  E:%e  P:%e  vx:%e  vy:%e  vz:%e  Udens:%e\n", i, j, k, d_av, E_av, P_av, vx_av, vy_av,
         vz_av, Udens_av);

  // success requires that at least 2 cells contributed to the average
  return N > 1;
}

void Apply_Scalar_Floor(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int field_num, Real scalar_floor)
{
  // set values for GPU kernels
  int n_cells = nx * ny * nz;
  int ngrid   = (n_cells + TPB - 1) / TPB;
  // number of blocks per 1D grid
  dim3 dim1dGrid(ngrid, 1, 1);
  //  number of threads per 1D block
  dim3 dim1dBlock(TPB, 1, 1);

  hipLaunchKernelGGL(Scalar_Floor_Kernel, dim1dGrid, dim1dBlock, 0, 0, dev_conserved, nx, ny, nz, n_ghost, field_num,
                     scalar_floor);
}

__global__ void Scalar_Floor_Kernel(Real *dev_conserved, int nx, int ny, int nz, int n_ghost, int field_num,
                                    Real scalar_floor)
{
  int id, xid, yid, zid, n_cells;
  Real scalar;  // variable to store the value of the scalar before a floor is applied
  n_cells = nx * ny * nz;

  // get a global thread ID
  id  = threadIdx.x + blockIdx.x * blockDim.x;
  zid = id / (nx * ny);
  yid = (id - zid * nx * ny) / nx;
  xid = id - zid * nx * ny - yid * nx;

  // threads corresponding to real cells do the calculation
  if (xid > n_ghost - 1 && xid < nx - n_ghost && yid > n_ghost - 1 && yid < ny - n_ghost && zid > n_ghost - 1 &&
      zid < nz - n_ghost) {
    scalar = dev_conserved[id + n_cells * field_num];

    if (scalar < scalar_floor) {
      // printf("###Thread scalar change  %f -> %f \n", scalar, scalar_floor);
      dev_conserved[id + n_cells * field_num] = scalar_floor;
    }
  }
}
