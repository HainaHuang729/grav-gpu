#include "grav_pm_compat.h"
#include "grav_cuda_pm_dist.h"
#include "grav_pm_config.cuh"

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int dist_pm_block = 256;
constexpr int dist_pm_max_gpus = 16;
constexpr double dist_pm_pi = 3.1415926535897932384626433832795;

void check_cuda(cudaError_t st, int line) {
  if (st != cudaSuccess) {
    std::fprintf(stderr, "SWIFT DIST PM CUDA error line %d: %s\n",
                 line, cudaGetErrorString(st));
    throw std::runtime_error(cudaGetErrorString(st));
  }
}

void check_cufft(cufftResult st, int line) {
  if (st != CUFFT_SUCCESS) {
    std::fprintf(stderr, "SWIFT DIST PM cuFFT error %d line %d\n", int(st), line);
    throw std::runtime_error("cuFFT failure in GravDistPm");
  }
}

#define CHECK_CUDA(x) check_cuda((x), __LINE__)
#define CHECK_CUFFT(x) check_cufft((x), __LINE__)

unsigned int blocks_for(size_t n) {
  return static_cast<unsigned int>((n + dist_pm_block - 1) / dist_pm_block);
}

struct DistPmSplits {
  int x0[dist_pm_max_gpus + 1];
  int y0[dist_pm_max_gpus + 1];
  int G;
};

struct PlaneRun {
  int holder_lx0;
  int owner_lx0;
  int count;
};

__device__ __forceinline__ int wrap_idx(int x, int n) {
  const int r = x % n;
  return r < 0 ? r + n : r;
}

template <int Order>
__global__ void dist_deposit_kernel_t(
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
__global__ void dist_interpolate_kernel_t(
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

__global__ void add_doubles_kernel(double* __restrict__ dst,
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

__global__ void pack_xy_kernel(
    const cufftDoubleComplex* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int nx_g,
    int n,
    int nz_c,
    DistPmSplits sp) {
  const size_t total = size_t(nx_g) * n * nz_c;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int kz = int(i % nz_c);
  const size_t tmp = i / nz_c;
  const int ky = int(tmp % n);
  const int lx = int(tmp / n);
  const int dst = find_owner_segment(ky, sp.y0, sp.G);
  const int ly = ky - sp.y0[dst];
  const size_t out_off =
      size_t(nx_g) * size_t(sp.y0[dst]) * nz_c +
      (size_t(ly) * nz_c + size_t(kz)) * size_t(nx_g) + size_t(lx);
  out[out_off] = in[i];
}

__global__ void unpack_xy_kernel(
    const cufftDoubleComplex* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int ny_g,
    int n,
    int nz_c,
    DistPmSplits sp) {
  const size_t total = size_t(ny_g) * nz_c * n;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int x = int(i % n);
  const size_t tmp = i / n;
  const int kz = int(tmp % nz_c);
  const int ly = int(tmp / nz_c);
  const int src = find_owner_segment(x, sp.x0, sp.G);
  const int nx_src = sp.x0[src + 1] - sp.x0[src];
  const int lx = x - sp.x0[src];
  const size_t in_off =
      size_t(sp.x0[src]) * size_t(ny_g) * nz_c +
      (size_t(ly) * nz_c + size_t(kz)) * size_t(nx_src) + size_t(lx);
  out[i] = in[in_off];
}

__global__ void pack_yx_kernel(
    const cufftDoubleComplex* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int ny_g,
    int n,
    int nz_c,
    DistPmSplits sp) {
  const size_t total = size_t(ny_g) * nz_c * n;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int x = int(i % n);
  const size_t tmp = i / n;
  const int kz = int(tmp % nz_c);
  const int ly = int(tmp / nz_c);
  const int dst = find_owner_segment(x, sp.x0, sp.G);
  const int lx = x - sp.x0[dst];
  const size_t out_off =
      size_t(sp.x0[dst]) * size_t(ny_g) * nz_c +
      (size_t(lx) * size_t(ny_g) + size_t(ly)) * nz_c + size_t(kz);
  out[out_off] = in[i];
}

__global__ void unpack_yx_kernel(
    const cufftDoubleComplex* __restrict__ in,
    cufftDoubleComplex* __restrict__ out,
    int nx_g,
    int n,
    int nz_c,
    DistPmSplits sp) {
  const size_t total = size_t(nx_g) * n * nz_c;
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int kz = int(i % nz_c);
  const size_t tmp = i / nz_c;
  const int y = int(tmp % n);
  const int lx = int(tmp / n);
  const int src = find_owner_segment(y, sp.y0, sp.G);
  const int ny_src = sp.y0[src + 1] - sp.y0[src];
  const int ly = y - sp.y0[src];
  const size_t in_off =
      size_t(nx_g) * size_t(sp.y0[src]) * nz_c +
      (size_t(lx) * size_t(ny_src) + size_t(ly)) * nz_c + size_t(kz);
  out[i] = in[in_off];
}

__device__ __forceinline__ double dist_inverse_sinc(int mode, int n) {
  if (mode == 0) return 1.0;
  const double x = (dist_pm_pi / n) * mode;
  return x / sin(x);
}

__global__ void dist_filter_kernel(
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
  const double u2 = k2 * (4.0 * dist_pm_pi * dist_pm_pi * split * split / (box * box));
#ifdef GADGET2_LONG_RANGE_CORRECTION
  const double window = exp(-u2);
#else
  const double arg = 0.5 * dist_pm_pi * sqrt(u2);
  const double window = arg / (sinh(arg) + FLT_MIN);
#endif
  const double cic = dist_inverse_sinc(kx, n) * dist_inverse_sinc(ky, n) * dist_inverse_sinc(kz, n);
  const double deconv = grav_pm_deconv_weight(cic, order);
  double weight = (-1.0 / (dist_pm_pi * box)) * window / (k2 + FLT_MIN) * deconv;
  if (nu.enabled || fr.enabled) {
    const double k_phys = (2.0 * dist_pm_pi / box) * sqrt(k2);
    if (nu.enabled) weight *= grav_nu_pm_boost(k_phys, nu);
    if (fr.enabled) weight *= grav_fr_pm_boost(k_phys, fr);
  }
  modes[i].x *= weight;
  modes[i].y *= weight;
}

__global__ void dist_spectral_grad_y_kernel(
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

__global__ void dist_pack_slab_comp_kernel(
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

__device__ __forceinline__ double dist_stencil(double lo2, double lo1, double hi1, double hi2) {
  return (1.0 / 12.0) * hi2 - (2.0 / 3.0) * hi1 + (2.0 / 3.0) * lo1 - (1.0 / 12.0) * lo2;
}

__global__ void dist_differentiate_slab_kernel(
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
      fac * dist_stencil(pot_padded[xm2], pot_padded[xm1], pot_padded[xp1], pot_padded[xp2]),
      fac * dist_stencil(pot_padded[ym2], pot_padded[ym1], pot_padded[yp1], pot_padded[yp2]),
      fac * dist_stencil(pot_padded[zm2], pot_padded[zm1], pot_padded[zp1], pot_padded[zp2]),
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

struct GravDistPm::Impl {
  bool active = false;
  int G = 0;
  int side = 0;
  int nz_c = 0;
  DistPmSplits splits{};

  std::vector<int> devs;
  std::vector<cudaStream_t> streams;
  std::vector<std::vector<int>> can_peer;
  size_t stage_slot_bytes = 0;

  std::vector<int> nx, ny;
  std::vector<cufftHandle> plan_2d_d2z, plan_2d_z2d, plan_1d_z2z;
  std::vector<cufftDoubleComplex*> buf0, buf1, buf_phi;
  std::vector<double*> d_slab_real_padded;
  std::vector<double4*> d_slab_field;

  // Transport synchronization & staging
  std::vector<cudaEvent_t> ev_prod;
  std::vector<std::vector<cudaEvent_t>> ev_done;
  std::vector<std::vector<int>> slot_cursor;
  std::vector<std::vector<void*>> h_stage0, h_stage1;
  std::vector<std::vector<cudaEvent_t>> ev_d2h0, ev_d2h1, ev_h2d0, ev_h2d1;

  std::vector<std::vector<int>> local_planes;
  std::vector<size_t> local_dens_cap_planes;
  std::vector<size_t> local_field_cap_planes;
  std::vector<double*> d_local_dens;
  std::vector<double4*> d_local_field;
  std::vector<double4*> d_local_field_shift;
  bool has_shift_field = false;
  std::vector<int*> d_global_to_local_x;
  std::vector<int*> d_oob;
  std::vector<size_t> fixed_bytes;
  std::vector<size_t> local_bytes;

  // runs[holder_g][owner_g]
  std::vector<std::vector<std::vector<PlaneRun>>> runs;

  void build_runs() {
    runs.assign(G, std::vector<std::vector<PlaneRun>>(G));
    for (int hg = 0; hg < G; ++hg) {
      const auto& pl = local_planes[hg];
      for (int og = 0; og < G; ++og) {
        const int ox0 = splits.x0[og];
        const int ox1 = splits.x0[og + 1];
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

  void apply_planes(const std::vector<std::vector<int>>& gpu_x_planes, bool initial_alloc) {
    if (int(gpu_x_planes.size()) != G) {
      throw std::runtime_error("GravDistPm: gpu_x_planes size != G");
    }
    int prev_dev = 0;
    CHECK_CUDA(cudaGetDevice(&prev_dev));
    const size_t plane = size_t(side) * side;

    for (int g = 0; g < G; ++g) {
      CHECK_CUDA(cudaSetDevice(devs[g]));
      local_planes[g] = sanitize_planes(gpu_x_planes[g], side);
      const size_t lnx = local_planes[g].size();

      std::vector<int> h_map(side, -1);
      for (size_t lx = 0; lx < lnx; ++lx) {
        h_map[local_planes[g][lx]] = int(lx);
      }
      CHECK_CUDA(cudaMemcpyAsync(d_global_to_local_x[g], h_map.data(),
                                 size_t(side) * sizeof(int),
                                 cudaMemcpyHostToDevice, streams[g]));

      if (initial_alloc || lnx > local_dens_cap_planes[g]) {
        if (d_local_dens[g]) {
          CHECK_CUDA(cudaFree(d_local_dens[g]));
          d_local_dens[g] = nullptr;
        }
        local_dens_cap_planes[g] = lnx;
        if (lnx > 0) {
          CHECK_CUDA(cudaMalloc(&d_local_dens[g], lnx * plane * sizeof(double)));
        }
      }
      if (initial_alloc || lnx > local_field_cap_planes[g]) {
        if (d_local_field[g]) {
          CHECK_CUDA(cudaFree(d_local_field[g]));
          d_local_field[g] = nullptr;
        }
        if (d_local_field_shift[g]) {
          CHECK_CUDA(cudaFree(d_local_field_shift[g]));
          d_local_field_shift[g] = nullptr;
        }
        local_field_cap_planes[g] = lnx;
        if (lnx > 0) {
          CHECK_CUDA(cudaMalloc(&d_local_field[g], lnx * plane * sizeof(double4)));
          if (has_shift_field) {
            CHECK_CUDA(cudaMalloc(&d_local_field_shift[g], lnx * plane * sizeof(double4)));
          }
        }
      }
      local_bytes[g] = local_dens_cap_planes[g] * plane * sizeof(double) +
                       local_field_cap_planes[g] * plane * sizeof(double4) * (has_shift_field ? 2 : 1);
      CHECK_CUDA(cudaStreamSynchronize(streams[g]));
    }
    build_runs();
    CHECK_CUDA(cudaSetDevice(prev_dev));
  }

  void record_prod() {
    for (int g = 0; g < G; ++g) {
      CHECK_CUDA(cudaSetDevice(devs[g]));
      CHECK_CUDA(cudaEventRecord(ev_prod[g], streams[g]));
    }
  }

  void enqueue_xfer(int src, int dst, const void* d_src, void* d_dst, size_t bytes,
                    std::vector<std::vector<int>>& used) {
    if (!bytes) return;
    if (src == dst) {
      CHECK_CUDA(cudaSetDevice(devs[src]));
      CHECK_CUDA(cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, streams[src]));
      return;
    }
    used[src][dst] = 1;
    if (can_peer[dst][src]) {
      CHECK_CUDA(cudaSetDevice(devs[dst]));
      CHECK_CUDA(cudaStreamWaitEvent(streams[dst], ev_prod[src], 0));
      CHECK_CUDA(cudaMemcpyPeerAsync(d_dst, devs[dst], d_src, devs[src], bytes, streams[dst]));
      CHECK_CUDA(cudaEventRecord(ev_done[src][dst], streams[dst]));
      return;
    }
    const char* sp = static_cast<const char*>(d_src);
    char* dp = static_cast<char*>(d_dst);
    for (size_t off = 0; off < bytes;) {
      const size_t chunk = std::min(stage_slot_bytes, bytes - off);
      const int sl = (slot_cursor[src][dst]++) & 1;
      void* hb = sl ? h_stage1[src][dst] : h_stage0[src][dst];
      cudaEvent_t ed = sl ? ev_d2h1[src][dst] : ev_d2h0[src][dst];
      cudaEvent_t eh = sl ? ev_h2d1[src][dst] : ev_h2d0[src][dst];
      CHECK_CUDA(cudaSetDevice(devs[src]));
      CHECK_CUDA(cudaStreamWaitEvent(streams[src], eh, 0));
      CHECK_CUDA(cudaMemcpyAsync(hb, sp + off, chunk, cudaMemcpyDeviceToHost, streams[src]));
      CHECK_CUDA(cudaEventRecord(ed, streams[src]));
      CHECK_CUDA(cudaSetDevice(devs[dst]));
      CHECK_CUDA(cudaStreamWaitEvent(streams[dst], ed, 0));
      CHECK_CUDA(cudaMemcpyAsync(dp + off, hb, chunk, cudaMemcpyHostToDevice, streams[dst]));
      CHECK_CUDA(cudaEventRecord(eh, streams[dst]));
      off += chunk;
    }
    CHECK_CUDA(cudaEventRecord(ev_done[src][dst], streams[dst]));
  }

  void wait_done(const std::vector<std::vector<int>>& used) {
    for (int src = 0; src < G; ++src) {
      CHECK_CUDA(cudaSetDevice(devs[src]));
      for (int dst = 0; dst < G; ++dst) {
        if (src != dst && used[src][dst]) {
          CHECK_CUDA(cudaStreamWaitEvent(streams[src], ev_done[src][dst], 0));
        }
      }
    }
  }

  void check_oob() {
    int prev_dev = 0;
    CHECK_CUDA(cudaGetDevice(&prev_dev));
    for (int g = 0; g < G; ++g) {
      CHECK_CUDA(cudaSetDevice(devs[g]));
      int oob = 0;
      CHECK_CUDA(cudaMemcpyAsync(&oob, d_oob[g], sizeof(int), cudaMemcpyDeviceToHost, streams[g]));
      CHECK_CUDA(cudaStreamSynchronize(streams[g]));
      if (oob != 0) {
        CHECK_CUDA(cudaSetDevice(prev_dev));
        throw std::runtime_error("GravDistPm: particle touched non-local X-plane " +
                                 std::to_string(oob - 1) + " on GPU " + std::to_string(g));
      }
    }
    CHECK_CUDA(cudaSetDevice(prev_dev));
  }
};

GravDistPm::GravDistPm() : impl_(std::make_unique<Impl>()) {}

GravDistPm::~GravDistPm() {
  try {
    destroy();
  } catch (...) {
  }
}

void GravDistPm::init(const std::vector<int>& devs,
                       const std::vector<cudaStream_t>& streams,
                       const std::vector<std::vector<int>>& can_peer,
                       int side,
                       const std::vector<std::vector<int>>& gpu_x_planes) {
  destroy();
  const int G = int(devs.size());
  if (G < 2 || G > dist_pm_max_gpus) {
    throw std::runtime_error("GravDistPm requires 2 <= G <= 16, got " + std::to_string(G));
  }
  if (int(streams.size()) != G) {
    throw std::runtime_error("GravDistPm: streams size != devs size");
  }
  if (int(can_peer.size()) != G) {
    throw std::runtime_error("GravDistPm: can_peer rows != G");
  }
  for (int g = 0; g < G; ++g) {
    if (int(can_peer[g].size()) != G) {
      throw std::runtime_error("GravDistPm: can_peer cols != G");
    }
  }
  if (side < 2 * G || (side & 1) != 0) {
    throw std::runtime_error("GravDistPm requires even side >= 2 * G");
  }

  auto& im = *impl_;
  im.G = G;
  im.side = side;
  im.nz_c = side / 2 + 1;
  im.devs = devs;
  im.streams = streams;
  im.can_peer = can_peer;
  im.splits.G = G;
  im.nx.resize(G);
  im.ny.resize(G);
  int max_nx_ny = 0;
  for (int g = 0; g <= G; ++g) {
    im.splits.x0[g] = (g * side) / G;
    im.splits.y0[g] = (g * side) / G;
  }
  for (int g = 0; g < G; ++g) {
    im.nx[g] = im.splits.x0[g + 1] - im.splits.x0[g];
    im.ny[g] = im.splits.y0[g + 1] - im.splits.y0[g];
    max_nx_ny = std::max(max_nx_ny, std::max(im.nx[g], im.ny[g]));
    if (im.nx[g] < 2 || im.ny[g] < 1) {
      throw std::runtime_error("GravDistPm: slab width too narrow for 4th-order stencil");
    }
  }
  const size_t a2a_max_bytes =
      size_t(max_nx_ny) * size_t(max_nx_ny) * size_t(im.nz_c) * sizeof(cufftDoubleComplex);
  im.stage_slot_bytes = std::min<size_t>(std::max<size_t>(a2a_max_bytes, 4ULL << 20), 16ULL << 20);

  im.plan_2d_d2z.assign(G, 0);
  im.plan_2d_z2d.assign(G, 0);
  im.plan_1d_z2z.assign(G, 0);
  im.buf0.assign(G, nullptr);
  im.buf1.assign(G, nullptr);
  im.buf_phi.assign(G, nullptr);
  im.d_slab_real_padded.assign(G, nullptr);
  im.d_slab_field.assign(G, nullptr);
  im.ev_prod.assign(G, nullptr);
  im.ev_done.assign(G, std::vector<cudaEvent_t>(G, nullptr));
  im.slot_cursor.assign(G, std::vector<int>(G, 0));
  im.h_stage0.assign(G, std::vector<void*>(G, nullptr));
  im.h_stage1.assign(G, std::vector<void*>(G, nullptr));
  im.ev_d2h0.assign(G, std::vector<cudaEvent_t>(G, nullptr));
  im.ev_d2h1.assign(G, std::vector<cudaEvent_t>(G, nullptr));
  im.ev_h2d0.assign(G, std::vector<cudaEvent_t>(G, nullptr));
  im.ev_h2d1.assign(G, std::vector<cudaEvent_t>(G, nullptr));

  im.local_planes.resize(G);
  im.local_dens_cap_planes.assign(G, 0);
  im.local_field_cap_planes.assign(G, 0);
  im.d_local_dens.assign(G, nullptr);
  im.d_local_field.assign(G, nullptr);
  im.d_local_field_shift.assign(G, nullptr);
  im.has_shift_field = false;
  im.d_global_to_local_x.assign(G, nullptr);
  im.d_oob.assign(G, nullptr);
  im.fixed_bytes.assign(G, 0);
  im.local_bytes.assign(G, 0);

  int prev_dev = 0;
  CHECK_CUDA(cudaGetDevice(&prev_dev));
  im.active = true;

  try {
    const size_t plane = size_t(side) * side;
    for (int g = 0; g < G; ++g) {
      CHECK_CUDA(cudaSetDevice(im.devs[g]));
      CHECK_CUDA(cudaEventCreateWithFlags(&im.ev_prod[g], cudaEventDisableTiming));
      const int m_max = std::max(im.nx[g], im.ny[g]);
      const size_t cpx_elems = size_t(m_max) * size_t(side) * size_t(im.nz_c);
      const size_t cpx_bytes = cpx_elems * sizeof(cufftDoubleComplex);
      const size_t pad_bytes = size_t(im.nx[g] + 4) * plane * sizeof(double);
      const size_t fld_bytes = size_t(im.nx[g]) * plane * sizeof(double4);
      const size_t map_bytes = size_t(side) * sizeof(int);

      CHECK_CUDA(cudaMalloc(&im.buf0[g], cpx_bytes));
      CHECK_CUDA(cudaMalloc(&im.buf1[g], cpx_bytes));
      CHECK_CUDA(cudaMalloc(&im.d_slab_real_padded[g], pad_bytes));
      CHECK_CUDA(cudaMalloc(&im.d_slab_field[g], fld_bytes));
      CHECK_CUDA(cudaMalloc(&im.d_global_to_local_x[g], map_bytes));
      CHECK_CUDA(cudaMalloc(&im.d_oob[g], sizeof(int)));
      CHECK_CUDA(cudaMemsetAsync(im.d_oob[g], 0, sizeof(int), im.streams[g]));
      im.fixed_bytes[g] = 2 * cpx_bytes + pad_bytes + fld_bytes + map_bytes + sizeof(int);

      int n2d[2] = {side, side};
      int inembed_r[2] = {side, side};
      int onembed_c[2] = {side, im.nz_c};
      CHECK_CUFFT(cufftPlanMany(&im.plan_2d_d2z[g], 2, n2d,
                                inembed_r, 1, int(plane),
                                onembed_c, 1, side * im.nz_c,
                                CUFFT_D2Z, im.nx[g]));
      CHECK_CUFFT(cufftSetStream(im.plan_2d_d2z[g], im.streams[g]));

      CHECK_CUFFT(cufftPlanMany(&im.plan_2d_z2d[g], 2, n2d,
                                onembed_c, 1, side * im.nz_c,
                                inembed_r, 1, int(plane),
                                CUFFT_Z2D, im.nx[g]));
      CHECK_CUFFT(cufftSetStream(im.plan_2d_z2d[g], im.streams[g]));

      int n1d[1] = {side};
      CHECK_CUFFT(cufftPlanMany(&im.plan_1d_z2z[g], 1, n1d,
                                n1d, 1, side,
                                n1d, 1, side,
                                CUFFT_Z2Z, im.ny[g] * im.nz_c));
      CHECK_CUFFT(cufftSetStream(im.plan_1d_z2z[g], im.streams[g]));
    }

    for (int src = 0; src < G; ++src) {
      for (int dst = 0; dst < G; ++dst) {
        if (src == dst) continue;
        CHECK_CUDA(cudaSetDevice(im.devs[dst]));
        CHECK_CUDA(cudaEventCreateWithFlags(&im.ev_done[src][dst], cudaEventDisableTiming));
        if (!im.can_peer[dst][src]) {
          CHECK_CUDA(cudaHostAlloc(&im.h_stage0[src][dst], im.stage_slot_bytes, cudaHostAllocPortable));
          CHECK_CUDA(cudaHostAlloc(&im.h_stage1[src][dst], im.stage_slot_bytes, cudaHostAllocPortable));
          CHECK_CUDA(cudaEventCreateWithFlags(&im.ev_h2d0[src][dst], cudaEventDisableTiming));
          CHECK_CUDA(cudaEventCreateWithFlags(&im.ev_h2d1[src][dst], cudaEventDisableTiming));
          CHECK_CUDA(cudaSetDevice(im.devs[src]));
          CHECK_CUDA(cudaEventCreateWithFlags(&im.ev_d2h0[src][dst], cudaEventDisableTiming));
          CHECK_CUDA(cudaEventCreateWithFlags(&im.ev_d2h1[src][dst], cudaEventDisableTiming));
        }
      }
    }

    im.apply_planes(gpu_x_planes, true);
    CHECK_CUDA(cudaSetDevice(prev_dev));
  } catch (...) {
    CHECK_CUDA(cudaSetDevice(prev_dev));
    destroy();
    throw;
  }
}

void GravDistPm::reconfigure_x_planes(const std::vector<std::vector<int>>& gpu_x_planes) {
  if (!impl_->active) throw std::runtime_error("GravDistPm not initialized");
  impl_->apply_planes(gpu_x_planes, false);
}

size_t GravDistPm::device_bytes(int g) const {
  if (!impl_->active || g < 0 || g >= impl_->G) return 0;
  return impl_->fixed_bytes[g] + impl_->local_bytes[g];
}

void GravDistPm::begin_deposit(int g) {
  auto& im = *impl_;
  CHECK_CUDA(cudaMemsetAsync(im.d_oob[g], 0, sizeof(int), im.streams[g]));
  const size_t lnx = im.local_planes[g].size();
  if (lnx > 0) {
    const size_t bytes = lnx * size_t(im.side) * size_t(im.side) * sizeof(double);
    CHECK_CUDA(cudaMemsetAsync(im.d_local_dens[g], 0, bytes, im.streams[g]));
  }
}

void GravDistPm::deposit_shift(int g, const void* d_x_double3, const float* d_mass_or_null,
                               float uniform_mass, size_t count, double box, double shift) {
  if (!count) return;
  auto& im = *impl_;
  const auto cfg = grav_pm_get_config();
  const int lnx = int(im.local_planes[g].size());
  if (cfg.order == 3) {
    dist_deposit_kernel_t<3><<<blocks_for(count), dist_pm_block, 0, im.streams[g]>>>(
        static_cast<const double3*>(d_x_double3),
        d_mass_or_null, uniform_mass, count,
        im.d_local_dens[g], im.d_global_to_local_x[g], lnx, im.d_oob[g],
        im.side, box, shift);
  } else if (cfg.order == 4) {
    dist_deposit_kernel_t<4><<<blocks_for(count), dist_pm_block, 0, im.streams[g]>>>(
        static_cast<const double3*>(d_x_double3),
        d_mass_or_null, uniform_mass, count,
        im.d_local_dens[g], im.d_global_to_local_x[g], lnx, im.d_oob[g],
        im.side, box, shift);
  } else {
    dist_deposit_kernel_t<2><<<blocks_for(count), dist_pm_block, 0, im.streams[g]>>>(
        static_cast<const double3*>(d_x_double3),
        d_mass_or_null, uniform_mass, count,
        im.d_local_dens[g], im.d_global_to_local_x[g], lnx, im.d_oob[g],
        im.side, box, shift);
  }
  CHECK_CUDA(cudaGetLastError());
}

void GravDistPm::deposit(int g, const void* d_x_double3, const float* d_mass_or_null,
                          float uniform_mass, size_t count, double box) {
  deposit_shift(g, d_x_double3, d_mass_or_null, uniform_mass, count, box, 0.0);
}

void GravDistPm::solve_and_gather(double box, double split, int slot) {
  auto& im = *impl_;
  if (!im.active) throw std::runtime_error("GravDistPm not initialized");
  const auto cfg = grav_pm_get_config();
  int prev_dev = 0;
  CHECK_CUDA(cudaGetDevice(&prev_dev));

  const int G = im.G;
  const int n = im.side;
  const int nz_c = im.nz_c;
  const size_t P = size_t(n) * size_t(n);

  try {
    if (slot == 1 && !im.has_shift_field) {
      im.has_shift_field = true;
      for (int g = 0; g < G; ++g) {
        CHECK_CUDA(cudaSetDevice(im.devs[g]));
        const size_t cap = im.local_field_cap_planes[g];
        if (cap > 0) {
          CHECK_CUDA(cudaMalloc(&im.d_local_field_shift[g], cap * P * sizeof(double4)));
        }
        im.local_bytes[g] = im.local_dens_cap_planes[g] * P * sizeof(double) +
                            cap * P * sizeof(double4) * 2;
      }
    }
    if (cfg.spectral_grad && !im.buf_phi[0]) {
      for (int g = 0; g < G; ++g) {
        CHECK_CUDA(cudaSetDevice(im.devs[g]));
        const int m_max = std::max(im.nx[g], im.ny[g]);
        const size_t cpx_bytes = size_t(m_max) * size_t(n) * size_t(nz_c) * sizeof(cufftDoubleComplex);
        CHECK_CUDA(cudaMalloc(&im.buf_phi[g], cpx_bytes));
        im.fixed_bytes[g] += cpx_bytes;
      }
    }
    // Step 2: Zero slab interior + local reduce-scatter contributions -> d_slab_real_padded + 2*P
    for (int g = 0; g < G; ++g) {
      CHECK_CUDA(cudaSetDevice(im.devs[g]));
      double* interior = im.d_slab_real_padded[g] + 2 * P;
      CHECK_CUDA(cudaMemsetAsync(interior, 0, size_t(im.nx[g]) * P * sizeof(double), im.streams[g]));
      for (const auto& r : im.runs[g][g]) {
        const size_t cnt = size_t(r.count) * P;
        add_doubles_kernel<<<blocks_for(cnt), dist_pm_block, 0, im.streams[g]>>>(
            interior + size_t(r.owner_lx0) * P,
            im.d_local_dens[g] + size_t(r.holder_lx0) * P,
            cnt);
        CHECK_CUDA(cudaGetLastError());
      }
    }

    // Ring rounds 1..G-1 for remote plane reduce-scatter into buf0[dst]
    for (int step = 1; step < G; ++step) {
      std::vector<size_t> recv_planes(G, 0);
      std::vector<std::vector<int>> used(G, std::vector<int>(G, 0));
      im.record_prod();
      for (int g = 0; g < G; ++g) {
        const int dst = (g + step) % G;
        double* dst_scratch = reinterpret_cast<double*>(im.buf0[dst]);
        for (const auto& r : im.runs[g][dst]) {
          im.enqueue_xfer(
              g, dst,
              im.d_local_dens[g] + size_t(r.holder_lx0) * P,
              dst_scratch + recv_planes[dst] * P,
              size_t(r.count) * P * sizeof(double),
              used);
          recv_planes[dst] += size_t(r.count);
        }
      }
      im.wait_done(used);

      for (int g = 0; g < G; ++g) {
        const int src = (g - step + G) % G;
        if (im.runs[src][g].empty()) continue;
        CHECK_CUDA(cudaSetDevice(im.devs[g]));
        double* interior = im.d_slab_real_padded[g] + 2 * P;
        const double* recv_buf = reinterpret_cast<const double*>(im.buf0[g]);
        size_t off = 0;
        for (const auto& r : im.runs[src][g]) {
          const size_t cnt = size_t(r.count) * P;
          add_doubles_kernel<<<blocks_for(cnt), dist_pm_block, 0, im.streams[g]>>>(
              interior + size_t(r.owner_lx0) * P,
              recv_buf + off * P,
              cnt);
          CHECK_CUDA(cudaGetLastError());
          off += size_t(r.count);
        }
      }
    }

    // Step 3: 2D D2Z cuFFT
    for (int g = 0; g < G; ++g) {
      CHECK_CUDA(cudaSetDevice(im.devs[g]));
      CHECK_CUFFT(cufftExecD2Z(im.plan_2d_d2z[g], im.d_slab_real_padded[g] + 2 * P, im.buf0[g]));
    }

    // Step 4: Pack X->Y, can_peer A2A, Unpack X->Y
    for (int g = 0; g < G; ++g) {
      CHECK_CUDA(cudaSetDevice(im.devs[g]));
      const size_t total = size_t(im.nx[g]) * n * nz_c;
      pack_xy_kernel<<<blocks_for(total), dist_pm_block, 0, im.streams[g]>>>(
          im.buf0[g], im.buf1[g], im.nx[g], n, nz_c, im.splits);
      CHECK_CUDA(cudaGetLastError());
    }

    {
      std::vector<std::vector<int>> used(G, std::vector<int>(G, 0));
      im.record_prod();
      for (int g = 0; g < G; ++g) {
        for (int p = 0; p < G; ++p) {
          const size_t bytes =
              size_t(im.nx[g]) * size_t(im.ny[p]) * nz_c * sizeof(cufftDoubleComplex);
          im.enqueue_xfer(
              g, p,
              im.buf1[g] + size_t(im.nx[g]) * size_t(im.splits.y0[p]) * nz_c,
              im.buf0[p] + size_t(im.splits.x0[g]) * size_t(im.ny[p]) * nz_c,
              bytes,
              used);
        }
      }
      im.wait_done(used);
    }

    for (int g = 0; g < G; ++g) {
      CHECK_CUDA(cudaSetDevice(im.devs[g]));
      const size_t total = size_t(im.ny[g]) * nz_c * n;
      unpack_xy_kernel<<<blocks_for(total), dist_pm_block, 0, im.streams[g]>>>(
          im.buf0[g], im.buf1[g], im.ny[g], n, nz_c, im.splits);
      CHECK_CUDA(cudaGetLastError());
    }

    // Step 5: 1D Z2Z Forward -> Filter
    for (int g = 0; g < G; ++g) {
      CHECK_CUDA(cudaSetDevice(im.devs[g]));
      CHECK_CUFFT(cufftExecZ2Z(im.plan_1d_z2z[g], im.buf1[g], im.buf1[g], CUFFT_FORWARD));
      const size_t total = size_t(im.ny[g]) * nz_c * n;
      const auto nu_sp = grav_nu_get_step_params();
      const auto fr_sp = grav_fr_get_step_params();
      dist_filter_kernel<<<blocks_for(total), dist_pm_block, 0, im.streams[g]>>>(
          im.buf1[g], im.splits.y0[g], im.ny[g], n, nz_c, box, split, cfg.order, nu_sp, fr_sp);
      CHECK_CUDA(cudaGetLastError());
    }

    auto inverse_y_to_x_real = [&]() {
      for (int g = 0; g < G; ++g) {
        CHECK_CUDA(cudaSetDevice(im.devs[g]));
        CHECK_CUFFT(cufftExecZ2Z(im.plan_1d_z2z[g], im.buf1[g], im.buf1[g], CUFFT_INVERSE));
        const size_t total = size_t(im.ny[g]) * nz_c * n;
        pack_yx_kernel<<<blocks_for(total), dist_pm_block, 0, im.streams[g]>>>(
            im.buf1[g], im.buf0[g], im.ny[g], n, nz_c, im.splits);
        CHECK_CUDA(cudaGetLastError());
      }
      {
        std::vector<std::vector<int>> used(G, std::vector<int>(G, 0));
        im.record_prod();
        for (int g = 0; g < G; ++g) {
          for (int p = 0; p < G; ++p) {
            const size_t bytes =
                size_t(im.nx[p]) * size_t(im.ny[g]) * nz_c * sizeof(cufftDoubleComplex);
            im.enqueue_xfer(
                g, p,
                im.buf0[g] + size_t(im.splits.x0[p]) * size_t(im.ny[g]) * nz_c,
                im.buf1[p] + size_t(im.nx[p]) * size_t(im.splits.y0[g]) * nz_c,
                bytes,
                used);
          }
        }
        im.wait_done(used);
      }
      for (int g = 0; g < G; ++g) {
        CHECK_CUDA(cudaSetDevice(im.devs[g]));
        const size_t total = size_t(im.nx[g]) * n * nz_c;
        unpack_yx_kernel<<<blocks_for(total), dist_pm_block, 0, im.streams[g]>>>(
            im.buf1[g], im.buf0[g], im.nx[g], n, nz_c, im.splits);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUFFT(cufftExecZ2D(im.plan_2d_z2d[g], im.buf0[g], im.d_slab_real_padded[g] + 2 * P));
      }
    };

    if (cfg.spectral_grad) {
      // Save filtered Phi(k) in buf_phi, then inverse-FFT Phi and -i*k_{x,y,z}*Phi
      for (int g = 0; g < G; ++g) {
        CHECK_CUDA(cudaSetDevice(im.devs[g]));
        const size_t bytes = size_t(im.ny[g]) * nz_c * n * sizeof(cufftDoubleComplex);
        CHECK_CUDA(cudaMemcpyAsync(im.buf_phi[g], im.buf1[g], bytes,
                                   cudaMemcpyDeviceToDevice, im.streams[g]));
      }
      const double two_pi_over_box = 2.0 * dist_pm_pi / box;
      const int axes[4] = {3, 0, 1, 2};
      for (int pass = 0; pass < 4; ++pass) {
        const int axis = axes[pass];
        if (pass > 0) {
          for (int g = 0; g < G; ++g) {
            CHECK_CUDA(cudaSetDevice(im.devs[g]));
            const size_t total = size_t(im.ny[g]) * nz_c * n;
            dist_spectral_grad_y_kernel<<<blocks_for(total), dist_pm_block, 0, im.streams[g]>>>(
                im.buf_phi[g], im.buf1[g], im.splits.y0[g], im.ny[g], n, nz_c,
                two_pi_over_box, axis);
            CHECK_CUDA(cudaGetLastError());
          }
        }
        inverse_y_to_x_real();
        for (int g = 0; g < G; ++g) {
          CHECK_CUDA(cudaSetDevice(im.devs[g]));
          const size_t total = size_t(im.nx[g]) * P;
          dist_pack_slab_comp_kernel<<<blocks_for(total), dist_pm_block, 0, im.streams[g]>>>(
              im.d_slab_real_padded[g] + 2 * P, im.d_slab_field[g], total, axis);
          CHECK_CUDA(cudaGetLastError());
        }
      }
    } else {
      inverse_y_to_x_real();

      // Step 8: 2-plane periodic halo exchange
      {
        std::vector<std::vector<int>> used(G, std::vector<int>(G, 0));
        im.record_prod();
        for (int g = 0; g < G; ++g) {
          const int left_g = (g - 1 + G) % G;
          const int right_g = (g + 1) % G;
          im.enqueue_xfer(
              g, left_g,
              im.d_slab_real_padded[g] + 2 * P,
              im.d_slab_real_padded[left_g] + size_t(im.nx[left_g] + 2) * P,
              2 * P * sizeof(double),
              used);
          im.enqueue_xfer(
              g, right_g,
              im.d_slab_real_padded[g] + size_t(im.nx[g]) * P,
              im.d_slab_real_padded[right_g],
              2 * P * sizeof(double),
              used);
        }
        im.wait_done(used);
      }

      // Step 9: 4th-order differentiation
      const double fac = double(n) / box;
      for (int g = 0; g < G; ++g) {
        CHECK_CUDA(cudaSetDevice(im.devs[g]));
        const size_t total = size_t(im.nx[g]) * P;
        dist_differentiate_slab_kernel<<<blocks_for(total), dist_pm_block, 0, im.streams[g]>>>(
            im.d_slab_real_padded[g], im.d_slab_field[g], im.nx[g], n, fac);
        CHECK_CUDA(cudaGetLastError());
      }
    }

    // Step 10: Direct plane gather d_slab_field -> d_local_field (or d_local_field_shift)
    auto& dst_fields = (slot == 1) ? im.d_local_field_shift : im.d_local_field;
    {
      std::vector<std::vector<int>> used(G, std::vector<int>(G, 0));
      im.record_prod();
      for (int owner = 0; owner < G; ++owner) {
        for (int holder = 0; holder < G; ++holder) {
          for (const auto& r : im.runs[holder][owner]) {
            im.enqueue_xfer(
                owner, holder,
                im.d_slab_field[owner] + size_t(r.owner_lx0) * P,
                dst_fields[holder] + size_t(r.holder_lx0) * P,
                size_t(r.count) * P * sizeof(double4),
                used);
          }
        }
      }
      im.wait_done(used);
    }

    im.check_oob();
    CHECK_CUDA(cudaSetDevice(prev_dev));
  } catch (...) {
    CHECK_CUDA(cudaSetDevice(prev_dev));
    throw;
  }
}

void GravDistPm::interpolate_shift(int g, const void* d_x_double3, size_t count,
                                   double box, float grav_G, void* d_acc_float3,
                                   float* d_pot_or_null, int slot, double shift,
                                   float acc_scale, int accumulate) {
  if (!count) return;
  auto& im = *impl_;
  const auto cfg = grav_pm_get_config();
  const int lnx = int(im.local_planes[g].size());
  const double4* field = (slot == 1 && im.has_shift_field) ? im.d_local_field_shift[g] : im.d_local_field[g];
  if (cfg.order == 3) {
    dist_interpolate_kernel_t<3><<<blocks_for(count), dist_pm_block, 0, im.streams[g]>>>(
        static_cast<const double3*>(d_x_double3), count,
        field, im.d_global_to_local_x[g], lnx, im.d_oob[g],
        im.side, box, grav_G, static_cast<float3*>(d_acc_float3), d_pot_or_null,
        shift, acc_scale, accumulate);
  } else if (cfg.order == 4) {
    dist_interpolate_kernel_t<4><<<blocks_for(count), dist_pm_block, 0, im.streams[g]>>>(
        static_cast<const double3*>(d_x_double3), count,
        field, im.d_global_to_local_x[g], lnx, im.d_oob[g],
        im.side, box, grav_G, static_cast<float3*>(d_acc_float3), d_pot_or_null,
        shift, acc_scale, accumulate);
  } else {
    dist_interpolate_kernel_t<2><<<blocks_for(count), dist_pm_block, 0, im.streams[g]>>>(
        static_cast<const double3*>(d_x_double3), count,
        field, im.d_global_to_local_x[g], lnx, im.d_oob[g],
        im.side, box, grav_G, static_cast<float3*>(d_acc_float3), d_pot_or_null,
        shift, acc_scale, accumulate);
  }
  CHECK_CUDA(cudaGetLastError());
}

void GravDistPm::interpolate(int g, const void* d_x_double3, size_t count,
                              double box, float grav_G, void* d_acc_float3,
                              float* d_pot_or_null) {
  interpolate_shift(g, d_x_double3, count, box, grav_G, d_acc_float3, d_pot_or_null,
                    0, 0.0, 1.0f, 0);
}

void GravDistPm::debug_copy_slab_field_to_host(double4* h_full_field) const {
  auto& im = *impl_;
  if (!im.active || !h_full_field) return;
  int prev_dev = 0;
  CHECK_CUDA(cudaGetDevice(&prev_dev));
  const size_t P = size_t(im.side) * size_t(im.side);
  for (int g = 0; g < im.G; ++g) {
    CHECK_CUDA(cudaSetDevice(im.devs[g]));
    CHECK_CUDA(cudaMemcpyAsync(
        h_full_field + size_t(im.splits.x0[g]) * P,
        im.d_slab_field[g],
        size_t(im.nx[g]) * P * sizeof(double4),
        cudaMemcpyDeviceToHost, im.streams[g]));
    CHECK_CUDA(cudaStreamSynchronize(im.streams[g]));
  }
  CHECK_CUDA(cudaSetDevice(prev_dev));
}

void GravDistPm::destroy() {
  auto& im = *impl_;
  if (!im.active) return;
  int prev_dev = 0;
  const bool have_prev = (cudaGetDevice(&prev_dev) == cudaSuccess);
  for (int g = 0; g < im.G; ++g) {
    cudaSetDevice(im.devs[g]);
    if (g < int(im.streams.size()) && im.streams[g]) cudaStreamSynchronize(im.streams[g]);
    if (im.plan_2d_d2z[g]) cufftDestroy(im.plan_2d_d2z[g]);
    if (im.plan_2d_z2d[g]) cufftDestroy(im.plan_2d_z2d[g]);
    if (im.plan_1d_z2z[g]) cufftDestroy(im.plan_1d_z2z[g]);
    if (im.buf0[g]) cudaFree(im.buf0[g]);
    if (im.buf1[g]) cudaFree(im.buf1[g]);
    if (im.buf_phi[g]) cudaFree(im.buf_phi[g]);
    if (im.d_slab_real_padded[g]) cudaFree(im.d_slab_real_padded[g]);
    if (im.d_slab_field[g]) cudaFree(im.d_slab_field[g]);
    if (im.d_local_dens[g]) cudaFree(im.d_local_dens[g]);
    if (im.d_local_field[g]) cudaFree(im.d_local_field[g]);
    if (im.d_local_field_shift[g]) cudaFree(im.d_local_field_shift[g]);
    if (im.d_global_to_local_x[g]) cudaFree(im.d_global_to_local_x[g]);
    if (im.d_oob[g]) cudaFree(im.d_oob[g]);
    if (im.ev_prod[g]) cudaEventDestroy(im.ev_prod[g]);
  }
  for (int src = 0; src < im.G; ++src) {
    for (int dst = 0; dst < im.G; ++dst) {
      if (im.ev_done[src][dst]) cudaEventDestroy(im.ev_done[src][dst]);
      if (im.ev_d2h0[src][dst]) cudaEventDestroy(im.ev_d2h0[src][dst]);
      if (im.ev_d2h1[src][dst]) cudaEventDestroy(im.ev_d2h1[src][dst]);
      if (im.ev_h2d0[src][dst]) cudaEventDestroy(im.ev_h2d0[src][dst]);
      if (im.ev_h2d1[src][dst]) cudaEventDestroy(im.ev_h2d1[src][dst]);
      if (im.h_stage0[src][dst]) cudaFreeHost(im.h_stage0[src][dst]);
      if (im.h_stage1[src][dst]) cudaFreeHost(im.h_stage1[src][dst]);
    }
  }
  im.ev_prod.clear();
  im.ev_done.clear();
  im.slot_cursor.clear();
  im.h_stage0.clear();
  im.h_stage1.clear();
  im.ev_d2h0.clear();
  im.ev_d2h1.clear();
  im.ev_h2d0.clear();
  im.ev_h2d1.clear();
  im.can_peer.clear();
  im.plan_2d_d2z.clear();
  im.plan_2d_z2d.clear();
  im.plan_1d_z2z.clear();
  im.buf0.clear();
  im.buf1.clear();
  im.buf_phi.clear();
  im.d_slab_real_padded.clear();
  im.d_slab_field.clear();
  im.local_planes.clear();
  im.local_dens_cap_planes.clear();
  im.local_field_cap_planes.clear();
  im.d_local_dens.clear();
  im.d_local_field.clear();
  im.d_local_field_shift.clear();
  im.has_shift_field = false;
  im.d_global_to_local_x.clear();
  im.d_oob.clear();
  im.fixed_bytes.clear();
  im.local_bytes.clear();
  im.runs.clear();
  im.devs.clear();
  im.streams.clear();
  im.nx.clear();
  im.ny.clear();
  im.active = false;
  im.G = 0;
  im.side = 0;
  if (have_prev) cudaSetDevice(prev_dev);
}
