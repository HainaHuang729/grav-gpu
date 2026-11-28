#include "config.h"
#include "grav_cuda_pm_mpi.h"
#include "grav_pm_config.cuh"

#ifndef SWIFT_KERNEL_LONG_GRAVITY_H
#ifndef restrict
#define restrict __restrict__
#define GRAV_PM_MPI_LOCAL_RESTRICT
#endif
#include "kernel_long_gravity.h"
#ifdef GRAV_PM_MPI_LOCAL_RESTRICT
#undef restrict
#undef GRAV_PM_MPI_LOCAL_RESTRICT
#endif
#endif

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int mpi_pm_block = 256;
constexpr double mpi_pm_pi = 3.1415926535897932384626433832795;

void check_cuda_pm(cudaError_t st, int line) {
  if (st != cudaSuccess) {
    std::fprintf(stderr, "SWIFT MPI PM CUDA error line %d: %s\n",
                 line, cudaGetErrorString(st));
    throw std::runtime_error(cudaGetErrorString(st));
  }
}

void check_cufft_pm(cufftResult st, int line) {
  if (st != CUFFT_SUCCESS) {
    std::fprintf(stderr, "SWIFT MPI PM cuFFT error %d line %d\n", int(st), line);
    throw std::runtime_error("cuFFT failure in GravMpiPm");
  }
}

#define CHECK_CUDA_PM(x) check_cuda_pm((x), __LINE__)
#define CHECK_CUFFT_PM(x) check_cufft_pm((x), __LINE__)

unsigned int blocks_for(size_t n) {
  return static_cast<unsigned int>((n + mpi_pm_block - 1) / mpi_pm_block);
}

struct MpiPlaneRun {
  int holder_lx0;
  int owner_lx0;
  int count;
};

__device__ __forceinline__ int wrap_idx(int x, int n) {
  const int r = x % n;
  return r < 0 ? r + n : r;
}

template <int Order>
__global__ void mpi_deposit_kernel_t(
    const double3* __restrict__ x,
    const float* __restrict__ mass_or_null,
    float uniform_mass,
    size_t count,
    double* __restrict__ local_dens,
    const int* __restrict__ global_to_local_x,
    int local_nx,
    int* __restrict__ oob_flag,
    int n,
    double box,
    double shift) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const double mp = double(mass_or_null ? mass_or_null[i] : uniform_mass);
  if (mp == 0.0) return;
  const double3 pos = x[i];
  const PmStencil<Order> st(pos.x, pos.y, pos.z, n, box, shift);
  #pragma unroll
  for (int ix = 0; ix < Order; ++ix) {
    const int gx = wrap_idx(st.base[0] + ix, n);
    const int lx = global_to_local_x[gx];
    if (lx < 0 || lx >= local_nx) {
      atomicExch(oob_flag, gx + 1);
      continue;
    }
    const double mx = mp * st.weight[0][ix];
    const size_t x_off = size_t(lx) * n * n;
    #pragma unroll
    for (int iy = 0; iy < Order; ++iy) {
      const int gy = wrap_idx(st.base[1] + iy, n);
      const double mxy = mx * st.weight[1][iy];
      const size_t xy_off = x_off + size_t(gy) * n;
      #pragma unroll
      for (int iz = 0; iz < Order; ++iz) {
        const int gz = wrap_idx(st.base[2] + iz, n);
        atomicAdd(&local_dens[xy_off + size_t(gz)], mxy * st.weight[2][iz]);
      }
    }
  }
}

template <int Order>
__global__ void mpi_interpolate_kernel_t(
    const double3* __restrict__ x,
    size_t count,
    const double4* __restrict__ local_field,
    const int* __restrict__ global_to_local_x,
    int local_nx,
    int* __restrict__ oob_flag,
    int n,
    double box,
    float grav_G,
    float3* __restrict__ acc_out,
    float* __restrict__ pot_out_or_null,
    double shift,
    float acc_scale,
    int accumulate) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const double3 pos = x[i];
  const PmStencil<Order> st(pos.x, pos.y, pos.z, n, box, shift);
  double4 answer{};
  #pragma unroll
  for (int ix = 0; ix < Order; ++ix) {
    const int gx = wrap_idx(st.base[0] + ix, n);
    const int lx = global_to_local_x[gx];
    if (lx < 0 || lx >= local_nx) {
      atomicExch(oob_flag, gx + 1);
      continue;
    }
    const size_t x_off = size_t(lx) * n * n;
    #pragma unroll
    for (int iy = 0; iy < Order; ++iy) {
      const int gy = wrap_idx(st.base[1] + iy, n);
      const double wxy = st.weight[0][ix] * st.weight[1][iy];
      const size_t xy_off = x_off + size_t(gy) * n;
      #pragma unroll
      for (int iz = 0; iz < Order; ++iz) {
        const int gz = wrap_idx(st.base[2] + iz, n);
        const double4 c = local_field[xy_off + size_t(gz)];
        const double w = wxy * st.weight[2][iz];
        answer.x += w * c.x;
        answer.y += w * c.y;
        answer.z += w * c.z;
        answer.w += w * c.w;
      }
    }
  }
  const float s = grav_G * acc_scale;
  const float ax = s * float(answer.x);
  const float ay = s * float(answer.y);
  const float az = s * float(answer.z);
  if (accumulate) {
    acc_out[i].x += ax;
    acc_out[i].y += ay;
    acc_out[i].z += az;
    if (pot_out_or_null) pot_out_or_null[i] += s * float(answer.w);
  } else {
    acc_out[i] = {ax, ay, az};
    if (pot_out_or_null) pot_out_or_null[i] = s * float(answer.w);
  }
}

__global__ void mpi_add_doubles_kernel(double* __restrict__ dst,
                                       const double* __restrict__ src,
                                       size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) dst[i] += src[i];
}

__device__ __forceinline__ int find_owner_segment(int idx, const int* bounds, int G) {
  for (int s = 0; s < G; ++s) {
    if (idx < bounds[s + 1]) return s;
  }
  return G - 1;
}

__global__ void mpi_pack_xy_kernel(
    const cufftDoubleComplex* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int nx_g,
    int n,
    int nz_c,
    const int* __restrict__ y0_bounds,
    int G) {
  const size_t total = size_t(nx_g) * n * nz_c;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int kz = int(i % nz_c);
  const size_t tmp = i / nz_c;
  const int ky = int(tmp % n);
  const int lx = int(tmp / n);
  const int dst = find_owner_segment(ky, y0_bounds, G);
  const int ly = ky - y0_bounds[dst];
  const size_t out_off =
      size_t(nx_g) * size_t(y0_bounds[dst]) * nz_c +
      (size_t(ly) * nz_c + size_t(kz)) * size_t(nx_g) + size_t(lx);
  out[out_off] = in[i];
}

__global__ void mpi_unpack_xy_kernel(
    const cufftDoubleComplex* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int ny_g,
    int n,
    int nz_c,
    const int* __restrict__ x0_bounds,
    int G) {
  const size_t total = size_t(ny_g) * nz_c * n;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int x = int(i % n);
  const size_t tmp = i / n;
  const int kz = int(tmp % nz_c);
  const int ly = int(tmp / nz_c);
  const int src = find_owner_segment(x, x0_bounds, G);
  const int nx_src = x0_bounds[src + 1] - x0_bounds[src];
  const int lx = x - x0_bounds[src];
  const size_t in_off =
      size_t(x0_bounds[src]) * size_t(ny_g) * nz_c +
      (size_t(ly) * nz_c + size_t(kz)) * size_t(nx_src) + size_t(lx);
  out[i] = in[in_off];
}

__global__ void mpi_pack_yx_kernel(
    const cufftDoubleComplex* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int ny_g,
    int n,
    int nz_c,
    const int* __restrict__ x0_bounds,
    int G) {
  const size_t total = size_t(ny_g) * nz_c * n;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int x = int(i % n);
  const size_t tmp = i / n;
  const int kz = int(tmp % nz_c);
  const int ly = int(tmp / nz_c);
  const int dst = find_owner_segment(x, x0_bounds, G);
  const int lx = x - x0_bounds[dst];
  const size_t out_off =
      size_t(x0_bounds[dst]) * size_t(ny_g) * nz_c +
      (size_t(lx) * size_t(ny_g) + size_t(ly)) * nz_c + size_t(kz);
  out[out_off] = in[i];
}

__global__ void mpi_unpack_yx_kernel(
    const cufftDoubleComplex* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int nx_g,
    int n,
    int nz_c,
    const int* __restrict__ y0_bounds,
    int G) {
  const size_t total = size_t(nx_g) * n * nz_c;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int kz = int(i % nz_c);
  const size_t tmp = i / nz_c;
  const int y = int(tmp % n);
  const int lx = int(tmp / n);
  const int src = find_owner_segment(y, y0_bounds, G);
  const int ny_src = y0_bounds[src + 1] - y0_bounds[src];
  const int ly = y - y0_bounds[src];
  const size_t in_off =
      size_t(nx_g) * size_t(y0_bounds[src]) * nz_c +
      (size_t(lx) * size_t(ny_src) + size_t(ly)) * nz_c + size_t(kz);
  out[i] = in[in_off];
}

__device__ __forceinline__ double mpi_inverse_sinc(int mode, int n) {
  if (mode == 0) return 1.0;
  const double x = (mpi_pm_pi / n) * mode;
  return x / sin(x);
}

__global__ void mpi_filter_kernel(
    cufftDoubleComplex* __restrict__ modes,
    int y0_g,
    int ny_g,
    int n,
    int nz_c,
    double box,
    double split,
    int order,
    GravNuStepParams nu,
    GravFrStepParams fr) {
  const size_t total = size_t(ny_g) * nz_c * n;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int half = n / 2;
  int kx = int(i % n);
  const size_t tmp = i / n;
  const int kz = int(tmp % nz_c);
  const int ly = int(tmp / nz_c);
  int ky = y0_g + ly;
  if (kx > half) kx -= n;
  if (ky > half) ky -= n;
  if (kx == 0 && ky == 0 && kz == 0) {
    modes[i] = {0.0, 0.0};
    return;
  }
  const double k2 = double(kx) * kx + double(ky) * ky + double(kz) * kz;
  const double u2 = k2 * (4.0 * mpi_pm_pi * mpi_pm_pi * split * split / (box * box));
#ifdef GADGET2_LONG_RANGE_CORRECTION
  const double window = exp(-u2);
#else
  const double arg = 0.5 * mpi_pm_pi * sqrt(u2);
  const double window = arg / (sinh(arg) + FLT_MIN);
#endif
  const double cic = mpi_inverse_sinc(kx, n) * mpi_inverse_sinc(ky, n) * mpi_inverse_sinc(kz, n);
  const double deconv = grav_pm_deconv_weight(cic, order);
  double weight = (-1.0 / (mpi_pm_pi * box)) * window / (k2 + FLT_MIN) * deconv;
  if (nu.enabled || fr.enabled) {
    const double k_phys = (2.0 * mpi_pm_pi / box) * sqrt(k2);
    if (nu.enabled) weight *= grav_nu_pm_boost(k_phys, nu);
    if (fr.enabled) weight *= grav_fr_pm_boost(k_phys, fr);
  }
  modes[i].x *= weight;
  modes[i].y *= weight;
}

__global__ void mpi_spectral_grad_y_kernel(
    const cufftDoubleComplex* __restrict__ phi_y,
    cufftDoubleComplex* __restrict__ out_y,
    int y0_g,
    int ny_g,
    int n,
    int nz_c,
    double two_pi_over_box,
    int axis) {
  const size_t total = size_t(ny_g) * nz_c * n;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int half = n / 2;
  int kx = int(i % n);
  const size_t tmp = i / n;
  const int kz = int(tmp % nz_c);
  const int ly = int(tmp / nz_c);
  int ky = y0_g + ly;
  if (kx > half) kx -= n;
  if (ky > half) ky -= n;
  int k_int = (axis == 0) ? kx : ((axis == 1) ? ky : kz);
  if (k_int == half || k_int == -half || k_int == 0) {
    out_y[i] = {0.0, 0.0};
    return;
  }
  const double k_phys = two_pi_over_box * double(k_int);
  const cufftDoubleComplex z = phi_y[i];
  out_y[i] = {k_phys * z.y, -k_phys * z.x};
}

__global__ void mpi_pack_slab_comp_kernel(
    const double* __restrict__ src,
    double4* __restrict__ dst,
    size_t total,
    int axis) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const double v = src[i];
  if (axis == 0) dst[i].x = v;
  else if (axis == 1) dst[i].y = v;
  else if (axis == 2) dst[i].z = v;
  else dst[i].w = v;
}

__device__ __forceinline__ double mpi_stencil(double lo2, double lo1, double hi1, double hi2) {
  return (1.0 / 12.0) * hi2 - (2.0 / 3.0) * hi1 + (2.0 / 3.0) * lo1 - (1.0 / 12.0) * lo2;
}

__global__ void mpi_differentiate_slab_kernel(
    const double* __restrict__ pot_padded,
    double4* __restrict__ slab_field,
    int nx_g,
    int n,
    double fac) {
  const size_t plane = size_t(n) * n;
  const size_t total = size_t(nx_g) * plane;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int lx = int(i / plane);
  const size_t yz = i - size_t(lx) * plane;
  const int y = int(yz / n);
  const int z = int(yz - size_t(y) * n);
  const int px = lx + 2;
  const size_t row = size_t(px) * plane + yz;

  const size_t xm2 = size_t(px - 2) * plane + yz;
  const size_t xm1 = size_t(px - 1) * plane + yz;
  const size_t xp1 = size_t(px + 1) * plane + yz;
  const size_t xp2 = size_t(px + 2) * plane + yz;

  const size_t plane_base = size_t(px) * plane;
  const size_t ym2 = plane_base + size_t(wrap_idx(y - 2, n)) * n + size_t(z);
  const size_t ym1 = plane_base + size_t(wrap_idx(y - 1, n)) * n + size_t(z);
  const size_t yp1 = plane_base + size_t(wrap_idx(y + 1, n)) * n + size_t(z);
  const size_t yp2 = plane_base + size_t(wrap_idx(y + 2, n)) * n + size_t(z);

  const size_t xy_base = plane_base + size_t(y) * n;
  const size_t zm2 = xy_base + size_t(wrap_idx(z - 2, n));
  const size_t zm1 = xy_base + size_t(wrap_idx(z - 1, n));
  const size_t zp1 = xy_base + size_t(wrap_idx(z + 1, n));
  const size_t zp2 = xy_base + size_t(wrap_idx(z + 2, n));

  slab_field[i] = {
      fac * mpi_stencil(pot_padded[xm2], pot_padded[xm1], pot_padded[xp1], pot_padded[xp2]),
      fac * mpi_stencil(pot_padded[ym2], pot_padded[ym1], pot_padded[yp1], pot_padded[yp2]),
      fac * mpi_stencil(pot_padded[zm2], pot_padded[zm1], pot_padded[zp1], pot_padded[zp2]),
      pot_padded[row]};
}

std::vector<int> sanitize_planes(std::vector<int> planes, int side) {
  std::vector<int> clean;
  clean.reserve(planes.size());
  for (int gx : planes) {
    int r = gx % side;
    if (r < 0) r += side;
    clean.push_back(r);
  }
  std::sort(clean.begin(), clean.end());
  clean.erase(std::unique(clean.begin(), clean.end()), clean.end());
  return clean;
}

}  // namespace

struct GravMpiPm::Impl {
  bool active = false;
  const grav_mpi::GravMpiCtx* mpi = nullptr;
  int G = 1;
  int rank = 0;
  int side = 0;
  int nz_c = 0;
  cudaStream_t stream = nullptr;

  std::vector<int> x0, y0, nx, ny;
  int* d_x0 = nullptr;
  int* d_y0 = nullptr;

  cufftHandle plan_2d_d2z = 0;
  cufftHandle plan_2d_z2d = 0;
  cufftHandle plan_1d_z2z = 0;
  cufftDoubleComplex* buf0 = nullptr;
  cufftDoubleComplex* buf1 = nullptr;
  cufftDoubleComplex* buf_phi = nullptr;
  double* d_slab_real_padded = nullptr;
  double4* d_slab_field = nullptr;

  // Dynamic MPI device staging scratch for Step 2 / Step 4 / Step 6 / Step 10
  char* d_xfer_send = nullptr;
  char* d_xfer_recv = nullptr;
  size_t xfer_send_cap = 0;
  size_t xfer_recv_cap = 0;
  std::vector<char> h_xfer_send, h_xfer_recv;
  std::vector<std::vector<char>> h_halo_send_scratch, h_halo_recv_scratch;

  std::vector<int> local_planes;
  size_t local_dens_cap_planes = 0;
  size_t local_field_cap_planes = 0;
  double* d_local_dens = nullptr;
  double4* d_local_field = nullptr;
  double4* d_local_field_shift = nullptr;
  bool has_shift_field = false;
  int* d_global_to_local_x = nullptr;
  int* d_oob = nullptr;
  size_t fixed_bytes = 0;
  size_t local_bytes = 0;
  double last_comm_ms_val = 0.0;

  // runs[holder_rank][owner_rank] across all ranks (reconstructed via compact bitmask Allgather)
  std::vector<std::vector<std::vector<MpiPlaneRun>>> runs;

  void ensure_xfer_caps(size_t need_send, size_t need_recv) {
    if (need_send > xfer_send_cap) {
      if (d_xfer_send) {
        CHECK_CUDA_PM(cudaFree(d_xfer_send));
        d_xfer_send = nullptr;
      }
      xfer_send_cap = std::max(need_send, need_send * 5 / 4);
      CHECK_CUDA_PM(cudaMalloc(&d_xfer_send, std::max<size_t>(1, xfer_send_cap)));
    }
    if (need_recv > xfer_recv_cap) {
      if (d_xfer_recv) {
        CHECK_CUDA_PM(cudaFree(d_xfer_recv));
        d_xfer_recv = nullptr;
      }
      xfer_recv_cap = std::max(need_recv, need_recv * 5 / 4);
      CHECK_CUDA_PM(cudaMalloc(&d_xfer_recv, std::max<size_t>(1, xfer_recv_cap)));
    }
  }

  void build_runs() {
    std::vector<unsigned char> my_have(size_t(side), 0);
    for (int gx : local_planes) my_have[gx] = 1;
    std::vector<unsigned char> all_have(size_t(G) * size_t(side), 0);
    if (G == 1) {
      all_have = my_have;
    } else {
      GRAV_MPI_CHECK(MPI_Allgather(my_have.data(), side, MPI_UNSIGNED_CHAR,
                                   all_have.data(), side, MPI_UNSIGNED_CHAR, mpi->world));
    }

    runs.assign(G, std::vector<std::vector<MpiPlaneRun>>(G));
    for (int hg = 0; hg < G; ++hg) {
      std::vector<int> pl;
      pl.reserve(side);
      for (int gx = 0; gx < side; ++gx) {
        if (all_have[size_t(hg) * size_t(side) + size_t(gx)]) pl.push_back(gx);
      }
      for (int og = 0; og < G; ++og) {
        const int ox0 = x0[og];
        const int ox1 = x0[og + 1];
        auto& out_runs = runs[hg][og];
        for (int lx = 0; lx < int(pl.size()); ++lx) {
          const int gx = pl[lx];
          if (gx < ox0 || gx >= ox1) continue;
          const int olx = gx - ox0;
          if (!out_runs.empty() &&
              out_runs.back().holder_lx0 + out_runs.back().count == lx &&
              out_runs.back().owner_lx0 + out_runs.back().count == olx) {
            ++out_runs.back().count;
          } else {
            out_runs.push_back({lx, olx, 1});
          }
        }
      }
    }
  }

  void apply_planes(const std::vector<int>& my_x_planes, bool initial_alloc) {
    CHECK_CUDA_PM(cudaSetDevice(mpi->cuda_dev));
    const size_t plane = size_t(side) * side;
    local_planes = sanitize_planes(my_x_planes, side);
    const size_t lnx = local_planes.size();

    std::vector<int> h_map(side, -1);
    for (size_t lx = 0; lx < lnx; ++lx) {
      h_map[local_planes[lx]] = int(lx);
    }
    CHECK_CUDA_PM(cudaMemcpyAsync(d_global_to_local_x, h_map.data(),
                                  size_t(side) * sizeof(int),
                                  cudaMemcpyHostToDevice, stream));

    if (initial_alloc || lnx > local_dens_cap_planes) {
      if (d_local_dens) {
        CHECK_CUDA_PM(cudaFree(d_local_dens));
        d_local_dens = nullptr;
      }
      local_dens_cap_planes = lnx;
      if (lnx > 0) {
        CHECK_CUDA_PM(cudaMalloc(&d_local_dens, lnx * plane * sizeof(double)));
      }
    }
    if (initial_alloc || lnx > local_field_cap_planes) {
      if (d_local_field) {
        CHECK_CUDA_PM(cudaFree(d_local_field));
        d_local_field = nullptr;
      }
      if (d_local_field_shift) {
        CHECK_CUDA_PM(cudaFree(d_local_field_shift));
        d_local_field_shift = nullptr;
      }
      local_field_cap_planes = lnx;
      if (lnx > 0) {
        CHECK_CUDA_PM(cudaMalloc(&d_local_field, lnx * plane * sizeof(double4)));
        if (has_shift_field) {
          CHECK_CUDA_PM(cudaMalloc(&d_local_field_shift, lnx * plane * sizeof(double4)));
        }
      }
    }
    local_bytes = local_dens_cap_planes * plane * sizeof(double) +
                  local_field_cap_planes * plane * sizeof(double4) * (has_shift_field ? 2 : 1);
    CHECK_CUDA_PM(cudaStreamSynchronize(stream));
    build_runs();
  }

  void check_oob() {
    int oob = 0;
    CHECK_CUDA_PM(cudaMemcpyAsync(&oob, d_oob, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA_PM(cudaStreamSynchronize(stream));
    if (oob != 0) {
      throw std::runtime_error("GravMpiPm: particle touched non-local X-plane " +
                               std::to_string(oob - 1) + " on rank " + std::to_string(rank));
    }
  }
};

GravMpiPm::GravMpiPm() : impl_(std::make_unique<Impl>()) {}

GravMpiPm::~GravMpiPm() {
  try {
    destroy();
  } catch (...) {
  }
}

void GravMpiPm::init(const grav_mpi::GravMpiCtx* mpi_ctx,
                     cudaStream_t stream,
                     int side,
                     const std::vector<int>& local_x_planes) {
  destroy();
  if (!mpi_ctx) throw std::runtime_error("GravMpiPm::init requires non-null GravMpiCtx");
  const int G = mpi_ctx->world_size;
  if (G < 1) throw std::runtime_error("GravMpiPm: invalid world_size");
  if ((side & 1) != 0 || (G > 1 && side < 2 * G)) {
    throw std::runtime_error("GravMpiPm requires even side >= 2 * G");
  }

  auto& im = *impl_;
  im.mpi = mpi_ctx;
  im.G = G;
  im.rank = mpi_ctx->world_rank;
  im.side = side;
  im.nz_c = side / 2 + 1;
  im.stream = stream;

  im.x0.resize(G + 1);
  im.y0.resize(G + 1);
  im.nx.resize(G);
  im.ny.resize(G);
  for (int g = 0; g <= G; ++g) {
    im.x0[g] = (g * side) / G;
    im.y0[g] = (g * side) / G;
  }
  for (int g = 0; g < G; ++g) {
    im.nx[g] = im.x0[g + 1] - im.x0[g];
    im.ny[g] = im.y0[g + 1] - im.y0[g];
    if ((G > 1 && im.nx[g] < 2) || im.ny[g] < 1) {
      throw std::runtime_error("GravMpiPm: slab width too narrow for 4th-order stencil");
    }
  }

  CHECK_CUDA_PM(cudaSetDevice(im.mpi->cuda_dev));
  im.active = true;

  try {
    const int r = im.rank;
    const size_t plane = size_t(side) * side;
    const int m_max = std::max(im.nx[r], im.ny[r]);
    const size_t cpx_elems = size_t(m_max) * size_t(side) * size_t(im.nz_c);
    const size_t cpx_bytes = cpx_elems * sizeof(cufftDoubleComplex);
    const size_t pad_bytes = size_t(im.nx[r] + 4) * plane * sizeof(double);
    const size_t fld_bytes = size_t(im.nx[r]) * plane * sizeof(double4);
    const size_t map_bytes = size_t(side) * sizeof(int);
    const size_t bnd_bytes = size_t(G + 1) * sizeof(int);

    CHECK_CUDA_PM(cudaMalloc(&im.d_x0, bnd_bytes));
    CHECK_CUDA_PM(cudaMalloc(&im.d_y0, bnd_bytes));
    CHECK_CUDA_PM(cudaMemcpyAsync(im.d_x0, im.x0.data(), bnd_bytes, cudaMemcpyHostToDevice, im.stream));
    CHECK_CUDA_PM(cudaMemcpyAsync(im.d_y0, im.y0.data(), bnd_bytes, cudaMemcpyHostToDevice, im.stream));

    CHECK_CUDA_PM(cudaMalloc(&im.buf0, cpx_bytes));
    CHECK_CUDA_PM(cudaMalloc(&im.buf1, cpx_bytes));
    CHECK_CUDA_PM(cudaMalloc(&im.d_slab_real_padded, pad_bytes));
    CHECK_CUDA_PM(cudaMalloc(&im.d_slab_field, fld_bytes));
    CHECK_CUDA_PM(cudaMalloc(&im.d_global_to_local_x, map_bytes));
    CHECK_CUDA_PM(cudaMalloc(&im.d_oob, sizeof(int)));
    CHECK_CUDA_PM(cudaMemsetAsync(im.d_oob, 0, sizeof(int), im.stream));
    im.fixed_bytes = 2 * cpx_bytes + pad_bytes + fld_bytes + map_bytes + 2 * bnd_bytes + sizeof(int);

    int n2d[2] = {side, side};
    int inembed_r[2] = {side, side};
    int onembed_c[2] = {side, im.nz_c};
    CHECK_CUFFT_PM(cufftPlanMany(&im.plan_2d_d2z, 2, n2d,
                                 inembed_r, 1, int(plane),
                                 onembed_c, 1, side * im.nz_c,
                                 CUFFT_D2Z, im.nx[r]));
    CHECK_CUFFT_PM(cufftSetStream(im.plan_2d_d2z, im.stream));

    CHECK_CUFFT_PM(cufftPlanMany(&im.plan_2d_z2d, 2, n2d,
                                 onembed_c, 1, side * im.nz_c,
                                 inembed_r, 1, int(plane),
                                 CUFFT_Z2D, im.nx[r]));
    CHECK_CUFFT_PM(cufftSetStream(im.plan_2d_z2d, im.stream));

    int n1d[1] = {side};
    CHECK_CUFFT_PM(cufftPlanMany(&im.plan_1d_z2z, 1, n1d,
                                 n1d, 1, side,
                                 n1d, 1, side,
                                 CUFFT_Z2Z, im.ny[r] * im.nz_c));
    CHECK_CUFFT_PM(cufftSetStream(im.plan_1d_z2z, im.stream));

    im.apply_planes(local_x_planes, true);
  } catch (...) {
    destroy();
    throw;
  }
}

void GravMpiPm::reconfigure_x_planes(const std::vector<int>& local_x_planes) {
  if (!impl_->active) throw std::runtime_error("GravMpiPm not initialized");
  impl_->apply_planes(local_x_planes, false);
}

size_t GravMpiPm::device_bytes() const {
  if (!impl_->active) return 0;
  return impl_->fixed_bytes + impl_->local_bytes + impl_->xfer_send_cap + impl_->xfer_recv_cap;
}

double GravMpiPm::last_comm_ms() const {
  return impl_->last_comm_ms_val;
}

void GravMpiPm::begin_deposit() {
  auto& im = *impl_;
  CHECK_CUDA_PM(cudaSetDevice(im.mpi->cuda_dev));
  CHECK_CUDA_PM(cudaMemsetAsync(im.d_oob, 0, sizeof(int), im.stream));
  const size_t lnx = im.local_planes.size();
  if (lnx > 0) {
    const size_t bytes = lnx * size_t(im.side) * size_t(im.side) * sizeof(double);
    CHECK_CUDA_PM(cudaMemsetAsync(im.d_local_dens, 0, bytes, im.stream));
  }
}

void GravMpiPm::deposit_shift(const void* d_x_double3, const float* d_mass_or_null,
                              float uniform_mass, size_t count, double box, double shift) {
  if (!count) return;
  auto& im = *impl_;
  const auto cfg = grav_pm_get_config();
  const int lnx = int(im.local_planes.size());
  if (cfg.order == 3) {
    mpi_deposit_kernel_t<3><<<blocks_for(count), mpi_pm_block, 0, im.stream>>>(
        static_cast<const double3*>(d_x_double3),
        d_mass_or_null, uniform_mass, count,
        im.d_local_dens, im.d_global_to_local_x, lnx, im.d_oob,
        im.side, box, shift);
  } else if (cfg.order == 4) {
    mpi_deposit_kernel_t<4><<<blocks_for(count), mpi_pm_block, 0, im.stream>>>(
        static_cast<const double3*>(d_x_double3),
        d_mass_or_null, uniform_mass, count,
        im.d_local_dens, im.d_global_to_local_x, lnx, im.d_oob,
        im.side, box, shift);
  } else {
    mpi_deposit_kernel_t<2><<<blocks_for(count), mpi_pm_block, 0, im.stream>>>(
        static_cast<const double3*>(d_x_double3),
        d_mass_or_null, uniform_mass, count,
        im.d_local_dens, im.d_global_to_local_x, lnx, im.d_oob,
        im.side, box, shift);
  }
  CHECK_CUDA_PM(cudaGetLastError());
}

void GravMpiPm::deposit(const void* d_x_double3, const float* d_mass_or_null,
                        float uniform_mass, size_t count, double box) {
  deposit_shift(d_x_double3, d_mass_or_null, uniform_mass, count, box, 0.0);
}

void GravMpiPm::solve_and_gather(double box, double split, int slot) {
  auto& im = *impl_;
  if (!im.active) throw std::runtime_error("GravMpiPm not initialized");
  const auto cfg = grav_pm_get_config();
  CHECK_CUDA_PM(cudaSetDevice(im.mpi->cuda_dev));

  const int G = im.G;
  const int r = im.rank;
  const int n = im.side;
  const int nz_c = im.nz_c;
  const int nx_my = im.nx[r];
  const int ny_my = im.ny[r];
  const size_t P = size_t(n) * size_t(n);
  double comm_sec = 0.0;

  if (slot == 1 && !im.has_shift_field) {
    im.has_shift_field = true;
    if (im.local_field_cap_planes > 0) {
      CHECK_CUDA_PM(cudaMalloc(&im.d_local_field_shift, im.local_field_cap_planes * P * sizeof(double4)));
    }
    im.local_bytes = im.local_dens_cap_planes * P * sizeof(double) +
                     im.local_field_cap_planes * P * sizeof(double4) * 2;
  }
  if (cfg.spectral_grad && !im.buf_phi) {
    const int m_max = std::max(nx_my, ny_my);
    const size_t cpx_bytes = size_t(m_max) * size_t(n) * size_t(nz_c) * sizeof(cufftDoubleComplex);
    CHECK_CUDA_PM(cudaMalloc(&im.buf_phi, cpx_bytes));
    im.fixed_bytes += cpx_bytes;
  }

  // Step 2: Reduce-scatter of held X-planes -> canonical X-slab interior
  double* interior = im.d_slab_real_padded + 2 * P;
  CHECK_CUDA_PM(cudaMemsetAsync(interior, 0, size_t(nx_my) * P * sizeof(double), im.stream));
  for (const auto& rn : im.runs[r][r]) {
    const size_t cnt = size_t(rn.count) * P;
    mpi_add_doubles_kernel<<<blocks_for(cnt), mpi_pm_block, 0, im.stream>>>(
        interior + size_t(rn.owner_lx0) * P,
        im.d_local_dens + size_t(rn.holder_lx0) * P,
        cnt);
    CHECK_CUDA_PM(cudaGetLastError());
  }

  if (G > 1) {
    std::vector<int> send_bytes(G, 0);
    size_t tot_send_planes = 0, tot_recv_planes = 0;
    for (int dst = 0; dst < G; ++dst) {
      if (dst == r) continue;
      size_t pl_cnt = 0;
      for (const auto& rn : im.runs[r][dst]) pl_cnt += size_t(rn.count);
      send_bytes[dst] = int(pl_cnt * P * sizeof(double));
      tot_send_planes += pl_cnt;
    }
    for (int src = 0; src < G; ++src) {
      if (src == r) continue;
      for (const auto& rn : im.runs[src][r]) tot_recv_planes += size_t(rn.count);
    }
    im.ensure_xfer_caps(tot_send_planes * P * sizeof(double),
                        tot_recv_planes * P * sizeof(double));
    size_t off_p = 0;
    for (int dst = 0; dst < G; ++dst) {
      if (dst == r) continue;
      for (const auto& rn : im.runs[r][dst]) {
        const size_t bytes = size_t(rn.count) * P * sizeof(double);
        CHECK_CUDA_PM(cudaMemcpyAsync(
            im.d_xfer_send + off_p * P * sizeof(double),
            im.d_local_dens + size_t(rn.holder_lx0) * P,
            bytes, cudaMemcpyDeviceToDevice, im.stream));
        off_p += size_t(rn.count);
      }
    }
    std::vector<int> recv_bytes, recv_displs;
    const double t0 = MPI_Wtime();
    im.mpi->alltoallv_bytes_dev(im.d_xfer_send, send_bytes,
                                im.d_xfer_recv, recv_bytes, recv_displs,
                                im.stream, im.h_xfer_send, im.h_xfer_recv);
    comm_sec += MPI_Wtime() - t0;

    for (int src = 0; src < G; ++src) {
      if (src == r || im.runs[src][r].empty()) continue;
      const double* recv_buf = reinterpret_cast<const double*>(im.d_xfer_recv + recv_displs[src]);
      size_t off = 0;
      for (const auto& rn : im.runs[src][r]) {
        const size_t cnt = size_t(rn.count) * P;
        mpi_add_doubles_kernel<<<blocks_for(cnt), mpi_pm_block, 0, im.stream>>>(
            interior + size_t(rn.owner_lx0) * P,
            recv_buf + off * P,
            cnt);
        CHECK_CUDA_PM(cudaGetLastError());
        off += size_t(rn.count);
      }
    }
  }

  // Step 3: 2D D2Z cuFFT
  CHECK_CUFFT_PM(cufftExecD2Z(im.plan_2d_d2z, interior, im.buf0));

  // Step 4: Pack X->Y, MPI Alltoallv, Unpack X->Y
  {
    const size_t total = size_t(nx_my) * n * nz_c;
    mpi_pack_xy_kernel<<<blocks_for(total), mpi_pm_block, 0, im.stream>>>(
        im.buf0, im.buf1, nx_my, n, nz_c, im.d_y0, G);
    CHECK_CUDA_PM(cudaGetLastError());
  }
  if (G == 1) {
    const size_t bytes = size_t(nx_my) * size_t(n) * size_t(nz_c) * sizeof(cufftDoubleComplex);
    CHECK_CUDA_PM(cudaMemcpyAsync(im.buf0, im.buf1, bytes, cudaMemcpyDeviceToDevice, im.stream));
  } else {
    std::vector<int> send_bytes(G, 0), recv_bytes, recv_displs;
    for (int p = 0; p < G; ++p) {
      send_bytes[p] = int(size_t(nx_my) * size_t(im.ny[p]) * size_t(nz_c) * sizeof(cufftDoubleComplex));
    }
    const double t0 = MPI_Wtime();
    im.mpi->alltoallv_bytes_dev(im.buf1, send_bytes,
                                im.buf0, recv_bytes, recv_displs,
                                im.stream, im.h_xfer_send, im.h_xfer_recv);
    comm_sec += MPI_Wtime() - t0;
  }
  {
    const size_t total = size_t(ny_my) * nz_c * n;
    mpi_unpack_xy_kernel<<<blocks_for(total), mpi_pm_block, 0, im.stream>>>(
        im.buf0, im.buf1, ny_my, n, nz_c, im.d_x0, G);
    CHECK_CUDA_PM(cudaGetLastError());
  }

  // Step 5: 1D Z2Z Forward -> Filter
  CHECK_CUFFT_PM(cufftExecZ2Z(im.plan_1d_z2z, im.buf1, im.buf1, CUFFT_FORWARD));
  {
    const size_t total = size_t(ny_my) * nz_c * n;
    const auto nu_sp = grav_nu_get_step_params();
    const auto fr_sp = grav_fr_get_step_params();
    mpi_filter_kernel<<<blocks_for(total), mpi_pm_block, 0, im.stream>>>(
        im.buf1, im.y0[r], ny_my, n, nz_c, box, split, cfg.order, nu_sp, fr_sp);
    CHECK_CUDA_PM(cudaGetLastError());
  }

  auto inverse_y_to_x_real = [&]() {
    CHECK_CUFFT_PM(cufftExecZ2Z(im.plan_1d_z2z, im.buf1, im.buf1, CUFFT_INVERSE));
    {
      const size_t total = size_t(ny_my) * nz_c * n;
      mpi_pack_yx_kernel<<<blocks_for(total), mpi_pm_block, 0, im.stream>>>(
          im.buf1, im.buf0, ny_my, n, nz_c, im.d_x0, G);
      CHECK_CUDA_PM(cudaGetLastError());
    }
    if (G == 1) {
      const size_t bytes = size_t(ny_my) * size_t(n) * size_t(nz_c) * sizeof(cufftDoubleComplex);
      CHECK_CUDA_PM(cudaMemcpyAsync(im.buf1, im.buf0, bytes, cudaMemcpyDeviceToDevice, im.stream));
    } else {
      std::vector<int> send_bytes(G, 0), recv_bytes, recv_displs;
      for (int p = 0; p < G; ++p) {
        send_bytes[p] = int(size_t(im.nx[p]) * size_t(ny_my) * size_t(nz_c) * sizeof(cufftDoubleComplex));
      }
      const double t0 = MPI_Wtime();
      im.mpi->alltoallv_bytes_dev(im.buf0, send_bytes,
                                  im.buf1, recv_bytes, recv_displs,
                                  im.stream, im.h_xfer_send, im.h_xfer_recv);
      comm_sec += MPI_Wtime() - t0;
    }
    {
      const size_t total = size_t(nx_my) * n * nz_c;
      mpi_unpack_yx_kernel<<<blocks_for(total), mpi_pm_block, 0, im.stream>>>(
          im.buf1, im.buf0, nx_my, n, nz_c, im.d_y0, G);
      CHECK_CUDA_PM(cudaGetLastError());
    }
    CHECK_CUFFT_PM(cufftExecZ2D(im.plan_2d_z2d, im.buf0, interior));
  };

  if (cfg.spectral_grad) {
    const size_t bytes = size_t(ny_my) * nz_c * n * sizeof(cufftDoubleComplex);
    CHECK_CUDA_PM(cudaMemcpyAsync(im.buf_phi, im.buf1, bytes, cudaMemcpyDeviceToDevice, im.stream));
    const double two_pi_over_box = 2.0 * mpi_pm_pi / box;
    const int axes[4] = {3, 0, 1, 2};
    for (int pass = 0; pass < 4; ++pass) {
      const int axis = axes[pass];
      if (pass > 0) {
        const size_t total = size_t(ny_my) * nz_c * n;
        mpi_spectral_grad_y_kernel<<<blocks_for(total), mpi_pm_block, 0, im.stream>>>(
            im.buf_phi, im.buf1, im.y0[r], ny_my, n, nz_c, two_pi_over_box, axis);
        CHECK_CUDA_PM(cudaGetLastError());
      }
      inverse_y_to_x_real();
      const size_t total = size_t(nx_my) * P;
      mpi_pack_slab_comp_kernel<<<blocks_for(total), mpi_pm_block, 0, im.stream>>>(
          interior, im.d_slab_field, total, axis);
      CHECK_CUDA_PM(cudaGetLastError());
    }
  } else {
    inverse_y_to_x_real();

    // Step 8: 2-plane periodic X halo exchange
    const size_t halo_bytes = 2 * P * sizeof(double);
    if (G == 1) {
      CHECK_CUDA_PM(cudaMemcpyAsync(im.d_slab_real_padded + size_t(nx_my + 2) * P,
                                    im.d_slab_real_padded + 2 * P,
                                    halo_bytes, cudaMemcpyDeviceToDevice, im.stream));
      CHECK_CUDA_PM(cudaMemcpyAsync(im.d_slab_real_padded,
                                    im.d_slab_real_padded + size_t(nx_my) * P,
                                    halo_bytes, cudaMemcpyDeviceToDevice, im.stream));
    } else {
      const int left_r = (r - 1 + G) % G;
      const int right_r = (r + 1) % G;
      const double t0 = MPI_Wtime();
      im.mpi->exchange_sparse_dev(
          {left_r}, {im.d_slab_real_padded + 2 * P}, {int(halo_bytes)}, im.h_halo_send_scratch,
          {right_r}, {im.d_slab_real_padded + size_t(nx_my + 2) * P}, {int(halo_bytes)}, im.h_halo_recv_scratch,
          41, im.stream);
      im.mpi->exchange_sparse_dev(
          {right_r}, {im.d_slab_real_padded + size_t(nx_my) * P}, {int(halo_bytes)}, im.h_halo_send_scratch,
          {left_r}, {im.d_slab_real_padded}, {int(halo_bytes)}, im.h_halo_recv_scratch,
          42, im.stream);
      comm_sec += MPI_Wtime() - t0;
    }

    // Step 9: 4th-order differentiation
    {
      const double fac = double(n) / box;
      const size_t total = size_t(nx_my) * P;
      mpi_differentiate_slab_kernel<<<blocks_for(total), mpi_pm_block, 0, im.stream>>>(
          im.d_slab_real_padded, im.d_slab_field, nx_my, n, fac);
      CHECK_CUDA_PM(cudaGetLastError());
    }
  }

  // Step 10: Gather d_slab_field -> d_local_field (or d_local_field_shift)
  double4* dst_local_field = (slot == 1) ? im.d_local_field_shift : im.d_local_field;
  for (const auto& rn : im.runs[r][r]) {
    const size_t bytes = size_t(rn.count) * P * sizeof(double4);
    CHECK_CUDA_PM(cudaMemcpyAsync(
        dst_local_field + size_t(rn.holder_lx0) * P,
        im.d_slab_field + size_t(rn.owner_lx0) * P,
        bytes, cudaMemcpyDeviceToDevice, im.stream));
  }
  if (G > 1) {
    std::vector<int> send_bytes(G, 0);
    size_t tot_send_planes = 0, tot_recv_planes = 0;
    for (int holder = 0; holder < G; ++holder) {
      if (holder == r) continue;
      size_t pl_cnt = 0;
      for (const auto& rn : im.runs[holder][r]) pl_cnt += size_t(rn.count);
      send_bytes[holder] = int(pl_cnt * P * sizeof(double4));
      tot_send_planes += pl_cnt;
    }
    for (int src = 0; src < G; ++src) {
      if (src == r) continue;
      for (const auto& rn : im.runs[r][src]) tot_recv_planes += size_t(rn.count);
    }
    im.ensure_xfer_caps(tot_send_planes * P * sizeof(double4),
                        tot_recv_planes * P * sizeof(double4));
    size_t off_p = 0;
    for (int holder = 0; holder < G; ++holder) {
      if (holder == r) continue;
      for (const auto& rn : im.runs[holder][r]) {
        const size_t bytes = size_t(rn.count) * P * sizeof(double4);
        CHECK_CUDA_PM(cudaMemcpyAsync(
            im.d_xfer_send + off_p * P * sizeof(double4),
            im.d_slab_field + size_t(rn.owner_lx0) * P,
            bytes, cudaMemcpyDeviceToDevice, im.stream));
        off_p += size_t(rn.count);
      }
    }
    std::vector<int> recv_bytes, recv_displs;
    const double t0 = MPI_Wtime();
    im.mpi->alltoallv_bytes_dev(im.d_xfer_send, send_bytes,
                                im.d_xfer_recv, recv_bytes, recv_displs,
                                im.stream, im.h_xfer_send, im.h_xfer_recv);
    comm_sec += MPI_Wtime() - t0;

    for (int src = 0; src < G; ++src) {
      if (src == r || im.runs[r][src].empty()) continue;
      const double4* recv_buf = reinterpret_cast<const double4*>(im.d_xfer_recv + recv_displs[src]);
      size_t off = 0;
      for (const auto& rn : im.runs[r][src]) {
        const size_t bytes = size_t(rn.count) * P * sizeof(double4);
        CHECK_CUDA_PM(cudaMemcpyAsync(
            dst_local_field + size_t(rn.holder_lx0) * P,
            recv_buf + off * P,
            bytes, cudaMemcpyDeviceToDevice, im.stream));
        off += size_t(rn.count);
      }
    }
  }

  im.last_comm_ms_val = comm_sec * 1e3;
  im.check_oob();
}

void GravMpiPm::interpolate_shift(const void* d_x_double3, size_t count,
                                  double box, float grav_G, void* d_acc_float3,
                                  float* d_pot_or_null, int slot, double shift,
                                  float acc_scale, int accumulate) {
  if (!count) return;
  auto& im = *impl_;
  const auto cfg = grav_pm_get_config();
  const int lnx = int(im.local_planes.size());
  const double4* field = (slot == 1 && im.has_shift_field) ? im.d_local_field_shift : im.d_local_field;
  if (cfg.order == 3) {
    mpi_interpolate_kernel_t<3><<<blocks_for(count), mpi_pm_block, 0, im.stream>>>(
        static_cast<const double3*>(d_x_double3), count,
        field, im.d_global_to_local_x, lnx, im.d_oob,
        im.side, box, grav_G, static_cast<float3*>(d_acc_float3), d_pot_or_null,
        shift, acc_scale, accumulate);
  } else if (cfg.order == 4) {
    mpi_interpolate_kernel_t<4><<<blocks_for(count), mpi_pm_block, 0, im.stream>>>(
        static_cast<const double3*>(d_x_double3), count,
        field, im.d_global_to_local_x, lnx, im.d_oob,
        im.side, box, grav_G, static_cast<float3*>(d_acc_float3), d_pot_or_null,
        shift, acc_scale, accumulate);
  } else {
    mpi_interpolate_kernel_t<2><<<blocks_for(count), mpi_pm_block, 0, im.stream>>>(
        static_cast<const double3*>(d_x_double3), count,
        field, im.d_global_to_local_x, lnx, im.d_oob,
        im.side, box, grav_G, static_cast<float3*>(d_acc_float3), d_pot_or_null,
        shift, acc_scale, accumulate);
  }
  CHECK_CUDA_PM(cudaGetLastError());
}

void GravMpiPm::interpolate(const void* d_x_double3, size_t count,
                            double box, float grav_G, void* d_acc_float3,
                            float* d_pot_or_null) {
  interpolate_shift(d_x_double3, count, box, grav_G, d_acc_float3, d_pot_or_null,
                    0, 0.0, 1.0f, 0);
}

void GravMpiPm::debug_copy_local_slab_field_to_host(double4* h_local_slab, int& out_x0, int& out_nx) const {
  auto& im = *impl_;
  out_x0 = im.x0[im.rank];
  out_nx = im.nx[im.rank];
  if (!im.active || !h_local_slab) return;
  const size_t P = size_t(im.side) * size_t(im.side);
  CHECK_CUDA_PM(cudaSetDevice(im.mpi->cuda_dev));
  CHECK_CUDA_PM(cudaMemcpyAsync(
      h_local_slab,
      im.d_slab_field,
      size_t(out_nx) * P * sizeof(double4),
      cudaMemcpyDeviceToHost, im.stream));
  CHECK_CUDA_PM(cudaStreamSynchronize(im.stream));
}

void GravMpiPm::destroy() {
  auto& im = *impl_;
  if (!im.active) return;
  if (im.mpi) cudaSetDevice(im.mpi->cuda_dev);
  if (im.stream) cudaStreamSynchronize(im.stream);
  if (im.plan_2d_d2z) cufftDestroy(im.plan_2d_d2z);
  if (im.plan_2d_z2d) cufftDestroy(im.plan_2d_z2d);
  if (im.plan_1d_z2z) cufftDestroy(im.plan_1d_z2z);
  if (im.d_x0) cudaFree(im.d_x0);
  if (im.d_y0) cudaFree(im.d_y0);
  if (im.buf0) cudaFree(im.buf0);
  if (im.buf1) cudaFree(im.buf1);
  if (im.buf_phi) cudaFree(im.buf_phi);
  if (im.d_slab_real_padded) cudaFree(im.d_slab_real_padded);
  if (im.d_slab_field) cudaFree(im.d_slab_field);
  if (im.d_xfer_send) cudaFree(im.d_xfer_send);
  if (im.d_xfer_recv) cudaFree(im.d_xfer_recv);
  if (im.d_local_dens) cudaFree(im.d_local_dens);
  if (im.d_local_field) cudaFree(im.d_local_field);
  if (im.d_local_field_shift) cudaFree(im.d_local_field_shift);
  if (im.d_global_to_local_x) cudaFree(im.d_global_to_local_x);
  if (im.d_oob) cudaFree(im.d_oob);
  im.plan_2d_d2z = im.plan_2d_z2d = im.plan_1d_z2z = 0;
  im.d_x0 = im.d_y0 = nullptr;
  im.buf0 = im.buf1 = im.buf_phi = nullptr;
  im.d_slab_real_padded = nullptr;
  im.d_slab_field = nullptr;
  im.d_xfer_send = im.d_xfer_recv = nullptr;
  im.d_local_dens = nullptr;
  im.d_local_field = im.d_local_field_shift = nullptr;
  im.has_shift_field = false;
  im.d_global_to_local_x = im.d_oob = nullptr;
  im.xfer_send_cap = im.xfer_recv_cap = 0;
  im.local_dens_cap_planes = im.local_field_cap_planes = 0;
  im.fixed_bytes = im.local_bytes = 0;
  im.runs.clear();
  im.local_planes.clear();
  im.x0.clear();
  im.y0.clear();
  im.nx.clear();
  im.ny.clear();
  im.active = false;
}
