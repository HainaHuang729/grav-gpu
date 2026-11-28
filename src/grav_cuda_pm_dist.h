#ifndef GRAV_CUDA_PM_DIST_H
#define GRAV_CUDA_PM_DIST_H

#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <vector>

/* Single-process multi-GPU all-device PM solver using cuFFT + can_peer-aware
 * CUDA P2P (NVLink cudaMemcpyPeerAsync + double-buffered pinned host staging
 * for non-NVLink pairs, avoiding unsafe cross-socket PCIe P2P).
 * Holds only ~O(N^3 / G) mesh memory per GPU with zero CPU computation and
 * supports G >= 2 GPUs with arbitrary Stage-4B per-GPU X-plane sets. */
class GravDistPm {
 public:
  GravDistPm();
  ~GravDistPm();

  GravDistPm(const GravDistPm&) = delete;
  GravDistPm& operator=(const GravDistPm&) = delete;

  void init(const std::vector<int>& devs,
            const std::vector<cudaStream_t>& streams,
            const std::vector<std::vector<int>>& can_peer,
            int side,
            const std::vector<std::vector<int>>& gpu_x_planes);

  void reconfigure_x_planes(const std::vector<std::vector<int>>& gpu_x_planes);

  size_t device_bytes(int g) const;

  void begin_deposit(int g);
  void deposit(int g, const void* d_x_double3, const float* d_mass_or_null,
               float uniform_mass, size_t count, double box);
  void deposit_shift(int g, const void* d_x_double3, const float* d_mass_or_null,
                     float uniform_mass, size_t count, double box, double shift);

  /* Must be called by a single host coordinator thread after all GPUs finish
   * deposit and synchronize on a barrier. Executes reduce-scatter, 3D FFT
   * (2D D2Z -> X<->Y -> 1D Z2Z), 4th-order FD or exact -i*k gradient,
   * and gather into per-GPU d_local_field (slot=0) or d_local_field_shift (slot=1). */
  void solve_and_gather(double box, double split, int slot = 0);

  void interpolate(int g, const void* d_x_double3, size_t count, double box,
                   float grav_G, void* d_acc_float3, float* d_pot_or_null);
  void interpolate_shift(int g, const void* d_x_double3, size_t count, double box,
                         float grav_G, void* d_acc_float3, float* d_pot_or_null,
                         int slot, double shift, float acc_scale, int accumulate);

  /* Test helper: copies canonical slab field [N, N, N] double4 to host. */
  void debug_copy_slab_field_to_host(double4* h_full_field) const;

  void destroy();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

#endif
