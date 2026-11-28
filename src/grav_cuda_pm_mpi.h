#ifndef GRAV_CUDA_PM_MPI_H
#define GRAV_CUDA_PM_MPI_H

#include "grav_mpi_comm.h"

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <vector>

/* Multi-node / multi-rank 1-rank-per-GPU distributed 3D PM solver.
 * Mirrors the exact 10-stage algorithm of GravDistPm across MPI ranks:
 *   1. Local deposit into per-rank active X-plane buffer
 *   2. Sparse MPI reduce-scatter of X-planes to canonical contiguous X-slabs
 *   3. 2D cuFFT D2Z (Y, Z) per rank
 *   4. MPI Alltoallv transpose (X-slabs <-> Y-slabs)
 *   5. 1D cuFFT Z2Z forward -> Green's/CIC filter -> 1D cuFFT Z2Z inverse
 *   6. MPI Alltoallv transpose (Y-slabs <-> X-slabs)
 *   7. 2D cuFFT Z2D (Y, Z) -> potential
 *   8. 2-plane periodic X halo exchange with (rank-1) and (rank+1)
 *   9. 4th-order finite-difference gradient + potential packing
 *  10. Sparse MPI gather from canonical X-slabs to per-rank held X-planes
 *
 * Supports arbitrary world_size >= 1 (G=1 takes local memcpy fast paths). */
class GravMpiPm {
 public:
  GravMpiPm();
  ~GravMpiPm();

  GravMpiPm(const GravMpiPm&) = delete;
  GravMpiPm& operator=(const GravMpiPm&) = delete;

  void init(const grav_mpi::GravMpiCtx* mpi_ctx,
            cudaStream_t stream,
            int side,
            const std::vector<int>& local_x_planes);

  void reconfigure_x_planes(const std::vector<int>& local_x_planes);

  size_t device_bytes() const;
  double last_comm_ms() const;

  void begin_deposit();
  void deposit(const void* d_x_double3, const float* d_mass_or_null,
               float uniform_mass, size_t count, double box);

  void solve_and_gather(double box, double split);

  void interpolate(const void* d_x_double3, size_t count, double box,
                   float grav_G, void* d_acc_float3, float* d_pot_or_null);

  /* Copies canonical local slab field [nx_my, side, side] double4 to host. */
  void debug_copy_local_slab_field_to_host(double4* h_local_slab, int& out_x0, int& out_nx) const;

  void destroy();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

#endif
