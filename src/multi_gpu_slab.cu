#include <algorithm>
#include <array>
#include <cerrno>
#include <cfloat>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <omp.h>

#include "swift_cuda_pm.h"
#include "swift_cuda_pm_dist.h"
#include "swift_cuda_fof.h"
#include "swift_domain_3d.h"
#include "validated_device_tree.cuh"
#include "gpu_gravity_functions.cuh"
#ifdef SWIFT_FAST_M2P_EXP
#define expf __expf
#endif
#include "m2p_native_math.cuh"
#ifdef SWIFT_FAST_M2P_EXP
#undef expf
#endif
#undef min
#undef max

using Tick = unsigned long long;
constexpr Tick timeline = 1ULL << 57;
constexpr int num_slabs = 12;
constexpr int slab_roots_count = 12 * 12;
constexpr int tree_levels = 14;
constexpr double hubble0_default = 67.36;
constexpr double omega_m_default = 0.3153, omega_de_default = 0.6847, omega_cdm_default = 0.266;
constexpr float dt_displacement_default = 1.121728e-4f;
constexpr double dt_max_default = 1e-2, max_dt_RMS_factor_default = 0.25;
constexpr float grav_G_default = 43.0092f;
constexpr float grav_eta_default = 0.025f, tolerance = 0.001f, theta_default = 0.7f;
constexpr int team = 128;
constexpr int scan_team = 256;
constexpr int scan_grid_max = 256;
constexpr int max_steps_guard = 65536;

struct StateInput {
  uint64_t n;
  double box, initial_scale, plummer_softening;
};
struct Footer {
  uint64_t n;
  double box, scale;
};
struct Meta {
  Tick next;
  unsigned int active;
  unsigned int invalid;
  unsigned int nodes;
};

struct Physics {
  double a_begin, log_span;
  float box, split, cutoff, G;
  Tick pm_span, short_span_cap;
  float short_step_factor, mac_tolerance;
  double dt_max, rms_length, max_dt_RMS_factor;
  bool dynamic_pm;
  double hubble0, omega_m, omega_de, omega_r, omega_k, w0, wa;
  float theta, grav_eta;
  bool periodic, comoving;
  float ext_M, ext_a, ext_cx, ext_cy, ext_cz;
  __host__ __device__ double scale(Tick ti) const {
    return comoving ? (a_begin * exp(log_span * (double(ti) / timeline))) : 1.0;
  }
  __host__ __device__ double time_at(Tick ti) const {
    return comoving ? scale(ti) : (a_begin + log_span * (double(ti) / timeline));
  }
  __host__ __device__ double H(double a) const {
    if (!comoving) return 1.0;
    const double inv_a = 1.0 / a;
    const double a2 = inv_a * inv_a;
    const double a3 = a2 * inv_a;
    const double a4 = a2 * a2;
    double de_fac = 1.0;
    if (w0 != -1.0 || wa != 0.0) {
      de_fac = pow(a, -3.0 * (1.0 + w0 + wa)) * exp(-3.0 * wa * (1.0 - a));
    }
    return hubble0 * sqrt(omega_r * a4 + omega_m * a3 + omega_k * a2 + omega_de * de_fac);
  }
  __host__ __device__ double factors(Tick lo, Tick hi, bool drift) const {
    if (lo == hi) return 0.0;
    if (!comoving) return log_span * (double(hi - lo) / timeline);
    const double a = scale(lo), b = scale(hi), m = (a + b) * 0.5;
    const double da = drift ? a * a * a : a * a;
    const double dm = drift ? m * m * m : m * m;
    const double db = drift ? b * b * b : b * b;
    return (b - a) / 6.0 * (1.0 / (da * H(a)) + 4.0 / (dm * H(m)) + 1.0 / (db * H(b)));
  }
};

struct ForceNode {
  float4 center;
  multipole m;
  int n;
};
static_assert(sizeof(grav_tensor) >= 35 * sizeof(float), "grav_tensor layout");
static_assert(offsetof(grav_tensor, F_112) == 34 * sizeof(float), "F_112 offset");
#include "multipoles_fast.cuh"

struct SlabTreeView {
  const Geometry* geometry;
  const ForceNode* sources;
  const int* escape;
  const int* first_child;
  const float4* packed_sources;
  const float* packed_soft;
  const uint8_t* home_cell;
  const int* order;
};

struct AllTreesView {
  SlabTreeView t[num_slabs];
};

static_assert(SELF_GRAVITY_MULTIPOLE_ORDER == 4, "CompactMultipole requires SELF_GRAVITY_MULTIPOLE_ORDER == 4");

struct alignas(16) NodeMeta {
  float4 center;        // 16B: x, y, z, radius
  float max_softening;  //  4B
  float power2;         //  4B: m.power[2]
  float M_000;          //  4B: m.M_000
  int n_flags;          //  4B: bits 0..30 = n, bit 31 = (first_child >= 0)
};
static_assert(sizeof(NodeMeta) == 32, "NodeMeta must be 32 bytes (1 cache sector)");

struct alignas(16) CompactMultipole {
  float M_000;
  float M_200, M_020, M_002, M_110, M_101, M_011;
  float M_300, M_030, M_003, M_210, M_201, M_120, M_021, M_102, M_012, M_111;
  float M_400, M_040, M_004, M_310, M_301, M_130, M_031, M_103, M_013,
        M_220, M_202, M_022, M_211, M_121, M_112;
};
static_assert(sizeof(CompactMultipole) == 128, "CompactMultipole must be 128 bytes (1 cache line)");

struct SplitTravTreeView {
  const Geometry* __restrict__ geometry;
  const NodeMeta* __restrict__ node_meta;
  const CompactMultipole* __restrict__ compact_m;
};

struct SlabForceInput {
  SlabTreeView base;
  SplitTravTreeView split;
};

struct AllSlabsData {
  const double3* x[num_slabs];
  const float* mass[num_slabs];
  const float* soft[num_slabs];
  int ns[num_slabs];
};

struct EwaldMode {
  double kx, ky, kz, weight;
};

struct LightconeHeader {
  char magic[8];
  uint64_t count;
  double obs[3];
  double a_begin, a_end, a_obs, r_max;
  double box;
};
static_assert(sizeof(LightconeHeader) == 80, "LightconeHeader layout");

struct LightconeHit {
  double x[3];
  uint64_t orig_idx;
  float v[3];
  float a_cross;
  float mass;
  float r_comoving;
};
static_assert(sizeof(LightconeHit) == 56, "LightconeHit layout");

template <class V>
__host__ __device__ __forceinline__ V permute_in(V v, int axis) {
  if (axis == 1) return {v.y, v.z, v.x};
  if (axis == 2) return {v.z, v.x, v.y};
  return v;
}
template <class V>
__host__ __device__ __forceinline__ V permute_out(V v, int axis) {
  if (axis == 1) return {v.z, v.x, v.y};
  if (axis == 2) return {v.y, v.z, v.x};
  return v;
}

__host__ __device__ __forceinline__ double wrapped_coord(double x, double box) {
  if (x < 0.0 || x >= box) {
    x -= floor(x / box) * box;
    if (x >= box) x = 0.0;
  }
  return x;
}

__host__ __device__ __forceinline__ Tick bin_size(unsigned char b) { return 1ULL << (b + 1); }
__host__ __device__ __forceinline__ bool is_active(Tick ti, unsigned char b) {
  return !(ti & (bin_size(b) - 1));
}
__device__ __forceinline__ int wrap_cell(int x) { return (x + 12) % 12; }

__device__ __forceinline__ float minimum_image_float(float dx, float box) {
  const float half = 0.5f * box;
  return dx > half ? dx - box : (dx < -half ? dx + box : dx);
}
__device__ __forceinline__ float3 periodic_delta_float(float3 a, float3 b, float box) {
  return {minimum_image_float(a.x - b.x, box), minimum_image_float(a.y - b.y, box),
          minimum_image_float(a.z - b.z, box)};
}
__device__ __forceinline__ float norm2(float3 x) { return x.x * x.x + x.y * x.y + x.z * x.z; }
__device__ __forceinline__ float3 periodic_delta(float3 a, float3 b, float L) {
  return {nearestf1(a.x - b.x, L), nearestf1(a.y - b.y, L), nearestf1(a.z - b.z, L)};
}
#include "grouped_force.cuh"

static void check_loc(cudaError_t status, int line) {
  if (status != cudaSuccess) {
    int dev = -1;
    cudaGetDevice(&dev);
    std::fprintf(stderr, "CUDA_ERR dev=%d line=%d err=%d(%s)\n",
                 dev, line, int(status), cudaGetErrorString(status));
    std::fflush(stderr);
    throw std::runtime_error(cudaGetErrorString(status));
  }
}
#ifdef check
#undef check
#endif
#define check(x) check_loc((x), __LINE__)

__device__ __forceinline__ void long_grav_eval_fast(
    const float r_over_r_s, float* corr_f, float* corr_pot) {
  const float two_over_sqrt_pi = float(M_2_SQRTPI);
  const float u = 0.5f * r_over_r_s;
  const float u2 = u * u;
  const float exp_u2 = __expf(-u2);
  const float t = __frcp_rn(1.f + 0.3275911f * u);
  float a = 1.061405429f * t - 1.453152027f;
  a = a * t + 1.421413741f;
  a = a * t - 0.284496736f;
  a = a * t + 0.254829592f;
  a = a * t;
  const float erfc_u = a * exp_u2;
  *corr_pot = erfc_u;
  *corr_f = erfc_u + two_over_sqrt_pi * u * exp_u2;
}

__device__ __forceinline__ void iact_grav_pp_truncated_fast(
    const float r2, const float h2, const float h_inv, const float h_inv3,
    const float mass, const float r_s_inv, float* f_ij, float* pot_ij) {
  const float r_inv = rsqrtf(r2 + FLT_MIN);
  const float r = r2 * r_inv;
  if (r2 >= h2) {
    *f_ij = mass * r_inv * r_inv * r_inv;
    *pot_ij = -mass * r_inv;
  } else {
    const float ui = r * h_inv;
    *f_ij = mass * h_inv3 * grav_force_eval(ui);
    *pot_ij = mass * h_inv * grav_pot_eval(ui);
  }
  float corr_f, corr_pot;
  long_grav_eval_fast(r * r_s_inv, &corr_f, &corr_pot);
  *f_ij *= corr_f;
  *pot_ij *= corr_pot;
}

__global__ void zero_empty_roots(ForceNode* force_nodes, int* escape, int* first_child, int nr) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nr) {
    force_nodes[i] = ForceNode{};
    escape[i] = -1;
    first_child[i] = -1;
  }
}

__global__ void tree_offsets_begin(int* levels, int* cursor, int nr) {
  levels[0] = 0;
  levels[1] = nr;
  *cursor = nr;
}
__global__ void tree_offsets_end(int* levels, int* cursor, int level, int capacity) {
  if (*cursor > capacity) *cursor = capacity;
  levels[level + 2] = *cursor;
}
__global__ void check_tree_limit(const Geometry* nodes, const int* levels, int level,
                                 int leaf_limit, int* invalid) {
  const int i = levels[level] + blockIdx.x * blockDim.x + threadIdx.x;
  if (i < levels[level + 1] && nodes[i].end - nodes[i].start > leaf_limit) atomicAdd(invalid, 1);
}
__global__ void tree_links(const Geometry* nodes, int* escape, const int* levels,
                           int level, int* first_child) {
  const int i = levels[level] + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= levels[level + 1]) return;
  if (level == 0) escape[i] = -1;
  int next = escape[i];
  for (int k = 7; k >= 0; --k) {
    const int child = nodes[i].children[k];
    if (child >= 0) {
      escape[child] = next;
      next = child;
    }
  }
  first_child[i] = (next == escape[i] ? -1 : next);
}

__global__ void gather_sources_slab(const double3* x, const float* mass_or_null,
                                    float uniform_mass, const float* soft_or_null,
                                    const int* order, float4* packed,
                                    float* packed_soft, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int pid = order[i];
  const double3 pos = x[pid];
  const float mp = mass_or_null ? mass_or_null[pid] : uniform_mass;
  packed[i] = {float(pos.x), float(pos.y), float(pos.z), mp};
  if (packed_soft && soft_or_null) packed_soft[i] = soft_or_null[pid];
}

__global__ void extract_home_cells(const Key* keys, uint8_t* home_cell, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) home_cell[i] = uint8_t(keys[i] >> octree_shift);
}

__global__ void mark_active_slab(const int* order, const unsigned char* bins,
                                 unsigned char* flags, int n, Tick ti) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) flags[i] = is_active(ti, bins[order[i]]);
}

template <bool use_m2l, bool fast_split>
__global__ void __launch_bounds__(team, 4) tree_force_warp_slab(
    const double3* target_x, const float* target_old_acc, float uniform_soft,
    const float* target_soft_arr,
    const SlabTreeView src0, const SlabTreeView src1, const SlabTreeView src2,
    float4* force, int n, Physics p, bool initial, const int* tree_invalid,
    const int* active_sorted, const int* active_count, bool compact_targets) {
  const int targets = active_count ? *active_count : n;
  const int warp_base = (blockIdx.x * blockDim.x + threadIdx.x) & ~31;
  if (warp_base >= targets || *tree_invalid) return;
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool lane_valid = (warp_base + lane) < targets;
  const uint32_t valid_mask = __ballot_sync(0xffffffff, lane_valid);

  constexpr int warps_per_block = team / 32;
  constexpr int warp_stack_size = 104;
  __shared__ int s_node[warps_per_block][warp_stack_size];
  __shared__ uint32_t s_mask[warps_per_block][warp_stack_size];
  __shared__ float4 s_leaf[warps_per_block][64];
  __shared__ float s_leaf_soft[warps_per_block][64];
  __shared__ grav_tensor s_field[warps_per_block];

  const SlabTreeView srcs[3] = {src0, src1, src2};

  int sorted_i = 0, i = 0, home_cell = -1;
  float3 x{};
  float old = 1.f;
  if (lane_valid) {
    const int tidx = warp_base + lane;
    sorted_i = active_sorted ? active_sorted[tidx] : tidx;
    i = compact_targets ? tidx : src1.order[sorted_i];
    const double3 px = target_x[i];
    x = {float(px.x), float(px.y), float(px.z)};
    old = target_old_acc[i];
    home_cell = int(src1.home_cell[sorted_i]);
  }
  const float inv_split = 1.f / p.split;
  const float target_soft = (lane_valid && target_soft_arr) ? target_soft_arr[i] : uniform_soft;
  const float h_inv = 1.f / target_soft;
  const float h2 = target_soft * target_soft;
  const float h_inv3 = h_inv * h_inv * h_inv;

  const int root0 = __shfl_sync(0xffffffff, home_cell, 0);
  const bool same_home = __all_sync(0xffffffff, !lane_valid || home_cell == root0);
  if (!same_home) {
    if (!lane_valid) return;
    const int home_y = home_cell / 12, home_z = home_cell % 12;
    float4 result{};
    for (int dx_i = 0; dx_i < 3; ++dx_i) {
      const auto& st = srcs[dx_i];
      for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
        int node = wrap_cell(home_y + dy) * 12 + wrap_cell(home_z + dz);
        while (node >= 0) {
          const auto& source = st.sources[node];
          const float4 c = source.center;
          const float3 d = periodic_delta_float({c.x, c.y, c.z}, x, p.box);
          const float r2 = norm2(d);
          const float reach = c.w + p.cutoff;
          if (!source.n || r2 > reach * reach) { node = st.escape[node]; continue; }
          const float soft = fmaxf(target_soft, source.m.max_softening);
          const float min_r2 = soft * soft;
          const float rad2 = c.w * c.w;
          const bool accept = source.n > 1 && r2 > min_r2 && (initial ?
              rad2 < p.theta * p.theta * r2 :
              rad2 < r2 && 8.f * source.m.power[2] < p.mac_tolerance * old * r2 * r2);
          if (accept) {
            reduced_grav_tensor contribution{};
            swift_cuda_gravity_M2P(&source.m, d.x, d.y, d.z, r2, soft, 1, inv_split, &contribution);
            result.x += contribution.F_100; result.y += contribution.F_010;
            result.z += contribution.F_001; result.w += contribution.F_000;
            node = st.escape[node];
            continue;
          }
          if (st.first_child[node] >= 0) { node = st.first_child[node]; continue; }
          const int start = st.geometry[node].start, end = st.geometry[node].end;
          for (int j = start; j < end; ++j) {
            if (dx_i == 1 && sorted_i == j) continue;
            const auto a = st.packed_sources[j];
            const float3 r = periodic_delta_float({a.x, a.y, a.z}, x, p.box);
            const float distance = norm2(r);
            float accel, pot;
            if (!target_soft_arr && !st.packed_soft) {
              if constexpr (fast_split) {
                iact_grav_pp_truncated_fast(distance, h2, h_inv, h_inv3, a.w, inv_split, &accel, &pot);
              } else {
                iact_grav_pp_truncated(distance, h2, h_inv, h_inv3, a.w, inv_split, &accel, &pot);
              }
            } else {
              const float sh = st.packed_soft ? st.packed_soft[j] : uniform_soft;
              const float hij = fmaxf(target_soft, sh);
              const float hi = 1.f / hij;
              if constexpr (fast_split) {
                iact_grav_pp_truncated_fast(distance, hij * hij, hi, hi * hi * hi, a.w, inv_split, &accel, &pot);
              } else {
                iact_grav_pp_truncated(distance, hij * hij, hi, hi * hi * hi, a.w, inv_split, &accel, &pot);
              }
            }
            result.x += r.x * accel; result.y += r.y * accel; result.z += r.z * accel; result.w += pot;
          }
          node = st.escape[node];
        }
      }
    }
    force[i] = result;
    return;
  }

  const float3 x0{
    __shfl_sync(0xffffffff, x.x, 0),
    __shfl_sync(0xffffffff, x.y, 0),
    __shfl_sync(0xffffffff, x.z, 0)
  };
  const float3 rel = lane_valid ? periodic_delta_float(x, x0, p.box) : float3{0.f, 0.f, 0.f};
  float min_rx = rel.x, max_rx = rel.x, min_ry = rel.y, max_ry = rel.y, min_rz = rel.z, max_rz = rel.z;
  float old_min = lane_valid ? old : FLT_MAX;
  float warp_soft_acc = lane_valid ? target_soft : 0.f;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    min_rx = fminf(min_rx, __shfl_xor_sync(0xffffffff, min_rx, off));
    max_rx = fmaxf(max_rx, __shfl_xor_sync(0xffffffff, max_rx, off));
    min_ry = fminf(min_ry, __shfl_xor_sync(0xffffffff, min_ry, off));
    max_ry = fmaxf(max_ry, __shfl_xor_sync(0xffffffff, max_ry, off));
    min_rz = fminf(min_rz, __shfl_xor_sync(0xffffffff, min_rz, off));
    max_rz = fmaxf(max_rz, __shfl_xor_sync(0xffffffff, max_rz, off));
    old_min = fminf(old_min, __shfl_xor_sync(0xffffffff, old_min, off));
    warp_soft_acc = fmaxf(warp_soft_acc, __shfl_xor_sync(0xffffffff, warp_soft_acc, off));
  }
  auto wrap_f = [](float v, float box) {
    return v < 0.f ? v + box : (v >= box ? v - box : v);
  };
  const float3 warp_center{
    wrap_f(x0.x + 0.5f * (min_rx + max_rx), p.box),
    wrap_f(x0.y + 0.5f * (min_ry + max_ry), p.box),
    wrap_f(x0.z + 0.5f * (min_rz + max_rz), p.box)
  };
  const float3 d_wc = lane_valid ? periodic_delta_float(x, warp_center, p.box) : float3{0.f, 0.f, 0.f};
  float r2_max = norm2(d_wc);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    r2_max = fmaxf(r2_max, __shfl_xor_sync(0xffffffff, r2_max, off));
  const float warp_radius = sqrtf(r2_max) + 1e-6f;
  const float warp_soft = warp_soft_acc;

  const bool can_m2l = use_m2l && (__popc(valid_mask) >= 8) && (warp_radius < 1.5f * p.split);
  if constexpr (use_m2l) {
    if (can_m2l) {
      float* sf = reinterpret_cast<float*>(&s_field[warp_id]);
      if (lane < 35) sf[lane] = 0.f;
      if (lane + 32 < 35) sf[lane + 32] = 0.f;
    }
  }
  __syncwarp();

  int my_m2l_slab = -1, my_m2l_node = -1, m2l_count = 0;
  bool any_m2l = false;
  auto flush_m2l = [&]() {
    if (!m2l_count) return;
    grav_tensor lf{};
    if (lane < m2l_count) {
      const auto& src = srcs[my_m2l_slab].sources[my_m2l_node];
      const float4 sc = src.center;
      const float3 dm = periodic_delta_float(warp_center, {sc.x, sc.y, sc.z}, p.box);
      const float r2m = norm2(dm);
      const float softm = fmaxf(warp_soft, src.m.max_softening);
      potential_derivatives_M2L deriv;
      swift_cuda_potential_derivatives_compute_M2L(
          dm.x, dm.y, dm.z, r2m, rsqrtf(r2m), softm, 1, inv_split, &deriv);
      swift_cuda_gravity_M2L_apply(&lf, &src.m, &deriv);
    }
    float* lfp = reinterpret_cast<float*>(&lf);
    float* sfp = reinterpret_cast<float*>(&s_field[warp_id]);
#pragma unroll
    for (int k = 0; k < 35; ++k) {
      float val = lfp[k];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
      if (lane == 0) sfp[k] += val;
    }
    __syncwarp();
    m2l_count = 0;
    my_m2l_slab = -1;
    my_m2l_node = -1;
  };

  float4 result{};
  const int home_y = root0 / 12, home_z = root0 % 12;
  for (int dx_i = 0; dx_i < 3; ++dx_i) {
    const auto& st = srcs[dx_i];
    for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
      const int root_u = wrap_cell(home_y + dy) * 12 + wrap_cell(home_z + dz);
      if (lane == 0) {
        s_node[warp_id][0] = root_u;
        s_mask[warp_id][0] = valid_mask;
      }
      int sp = 1;
      __syncwarp();
      while (sp > 0) {
        --sp;
        const int u = s_node[warp_id][sp];
        const uint32_t cur_mask = s_mask[warp_id][sp];
        const auto& source = st.sources[u];
        const float4 c = source.center;
        const int src_n = source.n;
        const float3 dw = periodic_delta_float({c.x, c.y, c.z}, warp_center, p.box);
        const float w_reach = c.w + p.cutoff + warp_radius;
        if (!src_n || norm2(dw) > w_reach * w_reach) continue;

        if constexpr (use_m2l) {
          if (can_m2l && cur_mask == valid_mask && src_n > 1) {
            const float3 dm = periodic_delta_float(warp_center, {c.x, c.y, c.z}, p.box);
            const float r2m = norm2(dm);
            const float rho_sum = warp_radius + c.w;
            const float soft_w = fmaxf(warp_soft, source.m.max_softening);
            if (rho_sum * rho_sum < 0.4225f * r2m) {
              const float dist = sqrtf(r2m);
              const float closest = dist - warp_radius;
              if (closest > soft_w && dist + warp_radius <= c.w + p.cutoff) {
                const float closest2 = closest * closest;
                const float rhoA2 = warp_radius * warp_radius;
                const float r6m = r2m * r2m * r2m;
                const bool m_ok = 8.f * source.m.power[2] < p.mac_tolerance * old_min * closest2 * closest2;
                const bool l_ok = 5.f * source.m.M_000 * rhoA2 * rhoA2 < p.mac_tolerance * old_min * r6m;
                if (initial || (m_ok && l_ok)) {
                  if (lane == m2l_count) {
                    my_m2l_slab = dx_i;
                    my_m2l_node = u;
                  }
                  ++m2l_count;
                  any_m2l = true;
                  if (m2l_count == 32) flush_m2l();
                  continue;
                }
              }
            }
          }
        }

        const bool lane_active = ((cur_mask >> lane) & 1u) != 0;
        const float3 d = periodic_delta_float({c.x, c.y, c.z}, x, p.box);
        const float r2 = norm2(d);
        const float reach = c.w + p.cutoff;
        const bool in_cut = lane_active && (r2 <= reach * reach);
        const float soft = fmaxf(target_soft, source.m.max_softening);
        const float rad2 = c.w * c.w;
        const bool accept = in_cut && (src_n > 1) && (r2 > soft * soft) && (initial ?
            rad2 < p.theta * p.theta * r2 :
            rad2 < r2 && 8.f * source.m.power[2] < p.mac_tolerance * old * r2 * r2);
        if (accept) {
          reduced_grav_tensor contribution{};
          swift_cuda_gravity_M2P(&source.m, d.x, d.y, d.z, r2, soft, 1, inv_split, &contribution);
          result.x += contribution.F_100; result.y += contribution.F_010;
          result.z += contribution.F_001; result.w += contribution.F_000;
        }
        const uint32_t need_open = __ballot_sync(0xffffffff, in_cut && !accept);
        if (!need_open) continue;

        if (st.first_child[u] >= 0) {
          const int ch = (lane < 8) ? st.geometry[u].children[lane] : -1;
          const uint32_t valid_ch = __ballot_sync(0xffffffff, lane < 8 && ch >= 0);
          const int nch = __popc(valid_ch);
          if (lane < 8 && ch >= 0) {
            const int pos = sp + __popc(valid_ch & ((1u << lane) - 1u));
            s_node[warp_id][pos] = ch;
            s_mask[warp_id][pos] = need_open;
          }
          sp += nch;
          __syncwarp();
        } else {
          const int start = st.geometry[u].start, end = st.geometry[u].end;
          for (int base = start; base < end; base += 64) {
            const int chunk = min(64, end - base);
            if (lane < chunk) s_leaf[warp_id][lane] = st.packed_sources[base + lane];
            if (lane + 32 < chunk) s_leaf[warp_id][lane + 32] = st.packed_sources[base + lane + 32];
            if (st.packed_soft) {
              if (lane < chunk) s_leaf_soft[warp_id][lane] = st.packed_soft[base + lane];
              if (lane + 32 < chunk) s_leaf_soft[warp_id][lane + 32] = st.packed_soft[base + lane + 32];
            }
            __syncwarp();
            if ((need_open >> lane) & 1u) {
              for (int k = 0; k < chunk; ++k) {
                if (dx_i == 1 && sorted_i == base + k) continue;
                const float4 a = s_leaf[warp_id][k];
                const float3 r = periodic_delta_float({a.x, a.y, a.z}, x, p.box);
                const float distance = norm2(r);
                float accel, pot;
                if (!target_soft_arr && !st.packed_soft) {
                  if constexpr (fast_split) {
                    iact_grav_pp_truncated_fast(distance, h2, h_inv, h_inv3, a.w, inv_split, &accel, &pot);
                  } else {
                    iact_grav_pp_truncated(distance, h2, h_inv, h_inv3, a.w, inv_split, &accel, &pot);
                  }
                } else {
                  const float sh = st.packed_soft ? s_leaf_soft[warp_id][k] : uniform_soft;
                  const float hij = fmaxf(target_soft, sh);
                  const float hi = 1.f / hij;
                  if constexpr (fast_split) {
                    iact_grav_pp_truncated_fast(distance, hij * hij, hi, hi * hi * hi, a.w, inv_split, &accel, &pot);
                  } else {
                    iact_grav_pp_truncated(distance, hij * hij, hi, hi * hi * hi, a.w, inv_split, &accel, &pot);
                  }
                }
                result.x += r.x * accel; result.y += r.y * accel; result.z += r.z * accel; result.w += pot;
              }
            }
            __syncwarp();
          }
        }
      }
    }
  }
  if constexpr (use_m2l) {
    if (any_m2l) {
      flush_m2l();
      if (lane_valid) {
        const float4 far = evaluate_tile(s_field[warp_id], periodic_delta_float(x, warp_center, p.box));
        result.x += far.x; result.y += far.y; result.z += far.z; result.w += far.w;
      }
    }
  }
  if (lane_valid) force[i] = result;
}

constexpr int split_p2p_chunk_sz = 16;
constexpr int split_m2l_chunk_sz = 32;

struct SplitWarpSlice {
  int start;
  uint16_t count;
  uint16_t home_cell;
};

struct SplitWarpMeta {
  float3 warp_center;
  float warp_soft;
  uint32_t valid_mask;
  uint32_t warp_flags;  // bit 0: has_m2l, bit 2: fast_rel
  int start;
  int count;
};

struct SplitP2PChunk {
  int next;
  int count;
  uint32_t masks[split_p2p_chunk_sz];
  int starts[split_p2p_chunk_sz];
  uint32_t meta[split_p2p_chunk_sz];  // low 28 bits: len, high 4 bits: dx_i
};

struct SplitM2LChunk {
  int next;
  int count;
  uint32_t nodes[split_m2l_chunk_sz];  // low 30 bits: u, high 2 bits: dx_i
};

__global__ void build_home_warp_slices_kernel(
    const uint8_t* __restrict__ home_cell,
    const int* __restrict__ active_sorted,
    const int* __restrict__ active_count,
    int n, const int* __restrict__ tree_invalid,
    SplitWarpSlice* __restrict__ out_slices,
    int* __restrict__ out_total_warps) {
  const int tid = threadIdx.x;
  const int targets = active_count ? *active_count : n;
  if (*tree_invalid || targets <= 0) {
    if (tid == 0) *out_total_warps = 0;
    return;
  }
  __shared__ int s_begin[slab_roots_count + 1];
  __shared__ int s_warp_off[slab_roots_count + 1];
  if (tid <= slab_roots_count) {
    if (tid == 0) {
      s_begin[0] = 0;
    } else if (tid == slab_roots_count) {
      s_begin[slab_roots_count] = targets;
    } else {
      int lo = 0, hi = targets;
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        const int sid = active_sorted ? active_sorted[mid] : mid;
        if (int(home_cell[sid]) < tid) lo = mid + 1;
        else hi = mid;
      }
      s_begin[tid] = lo;
    }
  }
  __syncthreads();
  if (tid == 0) {
    int acc = 0;
    for (int c = 0; c < slab_roots_count; ++c) {
      s_warp_off[c] = acc;
      const int b = s_begin[c];
      const int e = s_begin[c + 1];
      if (b < e) {
        const int next32 = (b + 31) & ~31;
        if (b < next32) ++acc;
        if (next32 < e) acc += (e - next32 + 31) >> 5;
      }
    }
    s_warp_off[slab_roots_count] = acc;
    *out_total_warps = acc;
  }
  __syncthreads();
  if (tid < slab_roots_count) {
    const int b = s_begin[tid];
    const int e = s_begin[tid + 1];
    if (b < e) {
      int w = s_warp_off[tid];
      const int next32 = (b + 31) & ~31;
      if (b < next32) {
        // Unaligned prefix belonged to a mixed-home warp in monolithic -> disable M2L
        const int cnt = min(e, next32) - b;
        out_slices[w++] = {b, uint16_t(cnt), uint16_t(tid)};
      }
      for (int pos = next32; pos < e; pos += 32, ++w) {
        const int cnt = min(32, e - pos);
        // Full 32-aligned warp or final tail warp of active list matches monolithic same_home == true
        const uint16_t m2l_bit = (cnt == 32 || pos + cnt == targets) ? uint16_t(0x8000u) : uint16_t(0u);
        out_slices[w] = {pos, uint16_t(cnt), uint16_t(uint16_t(tid) | m2l_bit)};
      }
    }
  }
}

__global__ void pack_split_nodes_kernel(
    const ForceNode* __restrict__ force_nodes,
    const int* __restrict__ first_child,
    NodeMeta* __restrict__ node_meta,
    CompactMultipole* __restrict__ compact_m,
    int num_nodes) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= num_nodes) return;
  const ForceNode& fn = force_nodes[u];
  const int n = fn.n;
  const int fc = first_child[u];
  NodeMeta nm;
  nm.center = fn.center;
  nm.max_softening = fn.m.max_softening;
  nm.power2 = fn.m.power[2];
  nm.M_000 = fn.m.M_000;
  nm.n_flags = (n & 0x7FFFFFFF) | ((fc >= 0) ? int(0x80000000u) : 0);
  node_meta[u] = nm;
  if (n > 1) {
    const multipole& m = fn.m;
    CompactMultipole cm;
    cm.M_000 = m.M_000;
    cm.M_200 = m.M_200; cm.M_020 = m.M_020; cm.M_002 = m.M_002;
    cm.M_110 = m.M_110; cm.M_101 = m.M_101; cm.M_011 = m.M_011;
    cm.M_300 = m.M_300; cm.M_030 = m.M_030; cm.M_003 = m.M_003;
    cm.M_210 = m.M_210; cm.M_201 = m.M_201; cm.M_120 = m.M_120;
    cm.M_021 = m.M_021; cm.M_102 = m.M_102; cm.M_012 = m.M_012;
    cm.M_111 = m.M_111;
    cm.M_400 = m.M_400; cm.M_040 = m.M_040; cm.M_004 = m.M_004;
    cm.M_310 = m.M_310; cm.M_301 = m.M_301; cm.M_130 = m.M_130;
    cm.M_031 = m.M_031; cm.M_103 = m.M_103; cm.M_013 = m.M_013;
    cm.M_220 = m.M_220; cm.M_202 = m.M_202; cm.M_022 = m.M_022;
    cm.M_211 = m.M_211; cm.M_121 = m.M_121; cm.M_112 = m.M_112;
    compact_m[u] = cm;
  }
}

__device__ __forceinline__ void gravity_M2P_periodic_fast(
    const CompactMultipole* __restrict__ cm,
    float r_x, float r_y, float r_z, float r2, float inv_split,
    float4& acc) {
  const float r_inv = rsqrtf(r2);
  const float r = r2 * r_inv;
  struct chi_derivatives derivs;
  swift_cuda_kernel_long_grav_derivatives(r, inv_split, &derivs);

  const float c0r = derivs.chi_0 * r_inv;
  const float Dt_1 = c0r;
  float Dt_2 = (derivs.chi_1 - c0r) * r_inv;
  float Dt_3 = (((c0r - derivs.chi_1) * 3.f) * r_inv + derivs.chi_2) * r_inv;
  float Dt_4 = ((((-c0r + derivs.chi_1) * 15.f) * r_inv - 6.f * derivs.chi_2) * r_inv + derivs.chi_3) * r_inv;
  float Dt_5 = (((((c0r - derivs.chi_1) * 105.f) * r_inv + 45.f * derivs.chi_2) * r_inv - 10.f * derivs.chi_3) * r_inv + derivs.chi_4) * r_inv;
  const float Dt_6 = ((((((-c0r + derivs.chi_1) * 945.f) * r_inv - 420.f * derivs.chi_2) * r_inv + 105.f * derivs.chi_3) * r_inv - 15.f * derivs.chi_4) * r_inv + derivs.chi_5) * r_inv;

  const float rx = r_x * r_inv;
  const float ry = r_y * r_inv;
  const float rz = r_z * r_inv;

  // Order 0 + 1
  {
    const float M0 = __ldg(&cm->M_000);
    acc.w -= M0 * Dt_1;
    const float m0_dt2 = M0 * Dt_2;
    acc.x -= m0_dt2 * rx;
    acc.y -= m0_dt2 * ry;
    acc.z -= m0_dt2 * rz;
  }

  const float rx2 = rx * rx;
  const float ry2 = ry * ry;
  const float rz2 = rz * rz;

  // Order 2 (dipole = 0 -> potential only)
  Dt_2 *= r_inv;
  {
    acc.w -= __ldg(&cm->M_200) * (rx2 * Dt_3 + Dt_2);
    acc.w -= __ldg(&cm->M_020) * (ry2 * Dt_3 + Dt_2);
    acc.w -= __ldg(&cm->M_002) * (rz2 * Dt_3 + Dt_2);
    acc.w -= __ldg(&cm->M_110) * (rx * ry * Dt_3);
    acc.w -= __ldg(&cm->M_101) * (rx * rz * Dt_3);
    acc.w -= __ldg(&cm->M_011) * (ry * rz * Dt_3);
  }

  // Order 3: D_3 -> acc.w += M_3 * d, acc.xyz -= M_2 * d
  Dt_3 *= r_inv;
  {
    float d;
    d = (rx2 * Dt_4 + 3.f * Dt_3) * rx;
    acc.w += __ldg(&cm->M_300) * d; acc.x -= __ldg(&cm->M_200) * d;

    d = (ry2 * Dt_4 + 3.f * Dt_3) * ry;
    acc.w += __ldg(&cm->M_030) * d; acc.y -= __ldg(&cm->M_020) * d;

    d = (rz2 * Dt_4 + 3.f * Dt_3) * rz;
    acc.w += __ldg(&cm->M_003) * d; acc.z -= __ldg(&cm->M_002) * d;

    d = (rx2 * Dt_4 + Dt_3) * ry;
    acc.w += __ldg(&cm->M_210) * d; acc.x -= __ldg(&cm->M_110) * d; acc.y -= __ldg(&cm->M_200) * d;

    d = (rx2 * Dt_4 + Dt_3) * rz;
    acc.w += __ldg(&cm->M_201) * d; acc.x -= __ldg(&cm->M_101) * d; acc.z -= __ldg(&cm->M_200) * d;

    d = (ry2 * Dt_4 + Dt_3) * rx;
    acc.w += __ldg(&cm->M_120) * d; acc.x -= __ldg(&cm->M_020) * d; acc.y -= __ldg(&cm->M_110) * d;

    d = (ry2 * Dt_4 + Dt_3) * rz;
    acc.w += __ldg(&cm->M_021) * d; acc.y -= __ldg(&cm->M_011) * d; acc.z -= __ldg(&cm->M_020) * d;

    d = (rz2 * Dt_4 + Dt_3) * rx;
    acc.w += __ldg(&cm->M_102) * d; acc.x -= __ldg(&cm->M_002) * d; acc.z -= __ldg(&cm->M_101) * d;

    d = (rz2 * Dt_4 + Dt_3) * ry;
    acc.w += __ldg(&cm->M_012) * d; acc.y -= __ldg(&cm->M_002) * d; acc.z -= __ldg(&cm->M_011) * d;

    d = rx * ry * rz * Dt_4;
    acc.w += __ldg(&cm->M_111) * d;
    acc.x -= __ldg(&cm->M_011) * d; acc.y -= __ldg(&cm->M_101) * d; acc.z -= __ldg(&cm->M_110) * d;
  }

  // Order 4: D_4 -> acc.w -= M_4 * d, acc.xyz += M_3 * d
  Dt_3 *= r_inv;
  Dt_4 *= r_inv;
  {
    float d;
    d = (rx2 * Dt_5 + 6.f * Dt_4) * rx2 + 3.f * Dt_3;
    acc.w -= __ldg(&cm->M_400) * d; acc.x += __ldg(&cm->M_300) * d;

    d = (ry2 * Dt_5 + 6.f * Dt_4) * ry2 + 3.f * Dt_3;
    acc.w -= __ldg(&cm->M_040) * d; acc.y += __ldg(&cm->M_030) * d;

    d = (rz2 * Dt_5 + 6.f * Dt_4) * rz2 + 3.f * Dt_3;
    acc.w -= __ldg(&cm->M_004) * d; acc.z += __ldg(&cm->M_003) * d;

    const float a4x = (rx2 * Dt_5 + 3.f * Dt_4) * rx;
    d = a4x * ry;
    acc.w -= __ldg(&cm->M_310) * d; acc.x += __ldg(&cm->M_210) * d; acc.y += __ldg(&cm->M_300) * d;
    d = a4x * rz;
    acc.w -= __ldg(&cm->M_301) * d; acc.x += __ldg(&cm->M_201) * d; acc.z += __ldg(&cm->M_300) * d;

    const float a4y = (ry2 * Dt_5 + 3.f * Dt_4) * ry;
    d = a4y * rx;
    acc.w -= __ldg(&cm->M_130) * d; acc.x += __ldg(&cm->M_030) * d; acc.y += __ldg(&cm->M_120) * d;
    d = a4y * rz;
    acc.w -= __ldg(&cm->M_031) * d; acc.y += __ldg(&cm->M_021) * d; acc.z += __ldg(&cm->M_030) * d;

    const float a4z = (rz2 * Dt_5 + 3.f * Dt_4) * rz;
    d = a4z * rx;
    acc.w -= __ldg(&cm->M_103) * d; acc.x += __ldg(&cm->M_003) * d; acc.z += __ldg(&cm->M_102) * d;
    d = a4z * ry;
    acc.w -= __ldg(&cm->M_013) * d; acc.y += __ldg(&cm->M_003) * d; acc.z += __ldg(&cm->M_012) * d;

    d = rx2 * ry2 * Dt_5 + (rx2 + ry2) * Dt_4 + Dt_3;
    acc.w -= __ldg(&cm->M_220) * d; acc.x += __ldg(&cm->M_120) * d; acc.y += __ldg(&cm->M_210) * d;

    d = rx2 * rz2 * Dt_5 + (rx2 + rz2) * Dt_4 + Dt_3;
    acc.w -= __ldg(&cm->M_202) * d; acc.x += __ldg(&cm->M_102) * d; acc.z += __ldg(&cm->M_201) * d;

    d = ry2 * rz2 * Dt_5 + (ry2 + rz2) * Dt_4 + Dt_3;
    acc.w -= __ldg(&cm->M_022) * d; acc.y += __ldg(&cm->M_012) * d; acc.z += __ldg(&cm->M_021) * d;

    d = (rx2 * Dt_5 + Dt_4) * (ry * rz);
    acc.w -= __ldg(&cm->M_211) * d; acc.x += __ldg(&cm->M_111) * d; acc.y += __ldg(&cm->M_201) * d; acc.z += __ldg(&cm->M_210) * d;

    d = (ry2 * Dt_5 + Dt_4) * (rx * rz);
    acc.w -= __ldg(&cm->M_121) * d; acc.x += __ldg(&cm->M_021) * d; acc.y += __ldg(&cm->M_111) * d; acc.z += __ldg(&cm->M_120) * d;

    d = (rz2 * Dt_5 + Dt_4) * (rx * ry);
    acc.w -= __ldg(&cm->M_112) * d; acc.x += __ldg(&cm->M_012) * d; acc.y += __ldg(&cm->M_102) * d; acc.z += __ldg(&cm->M_111) * d;
  }

  // Order 5: D_5 -> acc.xyz -= M_4 * d
  Dt_4 *= r_inv;
  Dt_5 *= r_inv;
  {
    float d;
    d = ((rx2 * Dt_6 + 10.f * Dt_5) * rx2 + 15.f * Dt_4) * rx;
    acc.x -= __ldg(&cm->M_400) * d;

    d = ((ry2 * Dt_6 + 10.f * Dt_5) * ry2 + 15.f * Dt_4) * ry;
    acc.y -= __ldg(&cm->M_040) * d;

    d = ((rz2 * Dt_6 + 10.f * Dt_5) * rz2 + 15.f * Dt_4) * rz;
    acc.z -= __ldg(&cm->M_004) * d;

    const float c4x = (rx2 * Dt_6 + 6.f * Dt_5) * rx2 + 3.f * Dt_4;
    d = c4x * ry;
    acc.x -= __ldg(&cm->M_310) * d; acc.y -= __ldg(&cm->M_400) * d;
    d = c4x * rz;
    acc.x -= __ldg(&cm->M_301) * d; acc.z -= __ldg(&cm->M_400) * d;

    const float c4y = (ry2 * Dt_6 + 6.f * Dt_5) * ry2 + 3.f * Dt_4;
    d = c4y * rx;
    acc.x -= __ldg(&cm->M_040) * d; acc.y -= __ldg(&cm->M_130) * d;
    d = c4y * rz;
    acc.y -= __ldg(&cm->M_031) * d; acc.z -= __ldg(&cm->M_040) * d;

    const float c4z = (rz2 * Dt_6 + 6.f * Dt_5) * rz2 + 3.f * Dt_4;
    d = c4z * rx;
    acc.x -= __ldg(&cm->M_004) * d; acc.z -= __ldg(&cm->M_103) * d;
    d = c4z * ry;
    acc.y -= __ldg(&cm->M_004) * d; acc.z -= __ldg(&cm->M_013) * d;

    d = (rx2 * ry2 * Dt_6 + (rx2 + 3.f * ry2) * Dt_5 + 3.f * Dt_4) * rx;
    acc.x -= __ldg(&cm->M_220) * d; acc.y -= __ldg(&cm->M_310) * d;

    d = (rx2 * rz2 * Dt_6 + (rx2 + 3.f * rz2) * Dt_5 + 3.f * Dt_4) * rx;
    acc.x -= __ldg(&cm->M_202) * d; acc.z -= __ldg(&cm->M_301) * d;

    d = (ry2 * rx2 * Dt_6 + (ry2 + 3.f * rx2) * Dt_5 + 3.f * Dt_4) * ry;
    acc.x -= __ldg(&cm->M_130) * d; acc.y -= __ldg(&cm->M_220) * d;

    d = (ry2 * rz2 * Dt_6 + (ry2 + 3.f * rz2) * Dt_5 + 3.f * Dt_4) * ry;
    acc.y -= __ldg(&cm->M_022) * d; acc.z -= __ldg(&cm->M_031) * d;

    d = (rz2 * rx2 * Dt_6 + (rz2 + 3.f * rx2) * Dt_5 + 3.f * Dt_4) * rz;
    acc.x -= __ldg(&cm->M_103) * d; acc.z -= __ldg(&cm->M_202) * d;

    d = (rz2 * ry2 * Dt_6 + (rz2 + 3.f * ry2) * Dt_5 + 3.f * Dt_4) * rz;
    acc.y -= __ldg(&cm->M_013) * d; acc.z -= __ldg(&cm->M_022) * d;

    const float rxyz = rx * ry * rz;
    d = (rx2 * Dt_6 + 3.f * Dt_5) * rxyz;
    acc.x -= __ldg(&cm->M_211) * d; acc.y -= __ldg(&cm->M_301) * d; acc.z -= __ldg(&cm->M_310) * d;

    d = (ry2 * Dt_6 + 3.f * Dt_5) * rxyz;
    acc.x -= __ldg(&cm->M_031) * d; acc.y -= __ldg(&cm->M_121) * d; acc.z -= __ldg(&cm->M_130) * d;

    d = (rz2 * Dt_6 + 3.f * Dt_5) * rxyz;
    acc.x -= __ldg(&cm->M_013) * d; acc.y -= __ldg(&cm->M_103) * d; acc.z -= __ldg(&cm->M_112) * d;

    d = (ry2 * rz2 * Dt_6 + (ry2 + rz2) * Dt_5 + Dt_4) * rx;
    acc.x -= __ldg(&cm->M_022) * d; acc.y -= __ldg(&cm->M_112) * d; acc.z -= __ldg(&cm->M_121) * d;

    d = (rx2 * rz2 * Dt_6 + (rx2 + rz2) * Dt_5 + Dt_4) * ry;
    acc.x -= __ldg(&cm->M_112) * d; acc.y -= __ldg(&cm->M_202) * d; acc.z -= __ldg(&cm->M_211) * d;

    d = (rx2 * ry2 * Dt_6 + (rx2 + ry2) * Dt_5 + Dt_4) * rz;
    acc.x -= __ldg(&cm->M_121) * d; acc.y -= __ldg(&cm->M_211) * d; acc.z -= __ldg(&cm->M_220) * d;
  }
}

template <bool use_m2l>
__global__ void __launch_bounds__(team, 5) traverse_m2p_split_kernel(
    const double3* __restrict__ target_x, const float* __restrict__ target_old_acc,
    float uniform_soft, const float* __restrict__ target_soft_arr,
    const SplitTravTreeView src0, const SplitTravTreeView src1, const SplitTravTreeView src2,
    const int* __restrict__ target_order,
    const SplitWarpSlice* __restrict__ slices,
    int warp_offset, int n_batch_warps,
    SplitWarpMeta* __restrict__ metas,
    int* __restrict__ head_p2p, int* __restrict__ head_m2l,
    SplitP2PChunk* __restrict__ p2p_chunks, int p2p_cap,
    SplitM2LChunk* __restrict__ m2l_chunks, int m2l_cap,
    int* __restrict__ p2p_warps, int* __restrict__ m2l_warps,
    int* __restrict__ split_counts,
    float4* __restrict__ force, int n, Physics p, bool initial,
    const int* __restrict__ tree_invalid,
    const int* __restrict__ active_sorted,
    bool compact_targets) {
  if (*tree_invalid) return;
  const int local_warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (local_warp >= n_batch_warps) return;
  const SplitWarpSlice sl = slices[warp_offset + local_warp];
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool lane_valid = lane < int(sl.count);
  const int home_cell = int(sl.home_cell & 0x7FFFu);
  const bool slice_allow_m2l = (sl.home_cell & 0x8000u) != 0;
  const int home_y = home_cell / 12;
  const int home_z = home_cell % 12;
  const uint32_t valid_mask = __ballot_sync(0xffffffff, lane_valid);
  if (!valid_mask) return;

  constexpr int warps_per_block = team / 32;
  constexpr int warp_stack_size = 104;
  struct SplitWarpLocalState {
    float warp_soft;
    float old_min_tol;
    float max_leaf_r;
    int p2p_head;
    int m2l_head;
    int sl_start;
    int sl_count;
    uint32_t valid_mask;
    int any_m2l;
  };
  __shared__ int s_node[warps_per_block][warp_stack_size];
  __shared__ uint32_t s_mask[warps_per_block][warp_stack_size];
  __shared__ uint32_t s_m2l[warps_per_block][split_m2l_chunk_sz];
  __shared__ uint32_t s_p2p_mask[warps_per_block][split_p2p_chunk_sz];
  __shared__ int s_p2p_start[warps_per_block][split_p2p_chunk_sz];
  __shared__ uint32_t s_p2p_meta[warps_per_block][split_p2p_chunk_sz];
  __shared__ SplitWarpLocalState s_ws[warps_per_block];

  int i = 0;
  float3 x{};
  float target_soft = uniform_soft;
  float old_tol = 0.f;
  float3 warp_center{};
  float warp_radius = 0.f;
  bool can_m2l = false;

  {
    const int tidx = sl.start + lane;
    float old = 1.f;
    if (lane_valid) {
      const int sorted_i = active_sorted ? active_sorted[tidx] : tidx;
      i = compact_targets ? tidx : target_order[sorted_i];
      const double3 px = target_x[i];
      x = {float(px.x), float(px.y), float(px.z)};
      old = target_old_acc[i];
      if (target_soft_arr) target_soft = target_soft_arr[i];
    }
    old_tol = p.mac_tolerance * old;

    const float3 x0{
        __shfl_sync(0xffffffff, x.x, 0),
        __shfl_sync(0xffffffff, x.y, 0),
        __shfl_sync(0xffffffff, x.z, 0)
    };
    const float3 rel = lane_valid ? periodic_delta_float(x, x0, p.box) : float3{0.f, 0.f, 0.f};
    float min_rx = rel.x, max_rx = rel.x, min_ry = rel.y, max_ry = rel.y, min_rz = rel.z, max_rz = rel.z;
    float old_min = lane_valid ? old : FLT_MAX;
    float soft_max = lane_valid ? target_soft : 0.f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      min_rx = fminf(min_rx, __shfl_xor_sync(0xffffffff, min_rx, off));
      max_rx = fmaxf(max_rx, __shfl_xor_sync(0xffffffff, max_rx, off));
      min_ry = fminf(min_ry, __shfl_xor_sync(0xffffffff, min_ry, off));
      max_ry = fmaxf(max_ry, __shfl_xor_sync(0xffffffff, max_ry, off));
      min_rz = fminf(min_rz, __shfl_xor_sync(0xffffffff, min_rz, off));
      max_rz = fmaxf(max_rz, __shfl_xor_sync(0xffffffff, max_rz, off));
      old_min = fminf(old_min, __shfl_xor_sync(0xffffffff, old_min, off));
      if (target_soft_arr) soft_max = fmaxf(soft_max, __shfl_xor_sync(0xffffffff, soft_max, off));
    }
    auto wrap_f = [](float v, float box) {
      return v < 0.f ? v + box : (v >= box ? v - box : v);
    };
    warp_center = {
        wrap_f(x0.x + 0.5f * (min_rx + max_rx), p.box),
        wrap_f(x0.y + 0.5f * (min_ry + max_ry), p.box),
        wrap_f(x0.z + 0.5f * (min_rz + max_rz), p.box)
    };
    const float3 d_wc = lane_valid ? periodic_delta_float(x, warp_center, p.box) : float3{0.f, 0.f, 0.f};
    float r2_max = norm2(d_wc);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      r2_max = fmaxf(r2_max, __shfl_xor_sync(0xffffffff, r2_max, off));
    warp_radius = sqrtf(r2_max) + 1e-6f;
    can_m2l = use_m2l && slice_allow_m2l && (__popc(valid_mask) >= 8) && (warp_radius < 1.5f * p.split);
    if (lane == 0) {
      s_ws[warp_id] = {
          target_soft_arr ? soft_max : uniform_soft,
          p.mac_tolerance * old_min,
          0.f, -1, -1, sl.start, int(sl.count), valid_mask, 0};
    }
    __syncwarp();
  }

  const float inv_split = 1.f / p.split;
  int p2p_cnt = 0;
  int m2l_cnt = 0;

  auto flush_p2p = [&]() {
    if (!p2p_cnt) return;
    __syncwarp();
    int cid = -1;
    if (lane == 0) {
      cid = atomicAdd(&split_counts[0], 1);
      if (cid >= p2p_cap) atomicOr(&split_counts[4], 1);
    }
    cid = __shfl_sync(0xffffffff, cid, 0);
    if (cid >= 0 && cid < p2p_cap) {
      if (lane == 0) {
        p2p_chunks[cid].next = s_ws[warp_id].p2p_head;
        p2p_chunks[cid].count = p2p_cnt;
        s_ws[warp_id].p2p_head = cid;
      }
      if (lane < p2p_cnt) {
        p2p_chunks[cid].masks[lane] = s_p2p_mask[warp_id][lane];
        p2p_chunks[cid].starts[lane] = s_p2p_start[warp_id][lane];
        p2p_chunks[cid].meta[lane] = s_p2p_meta[warp_id][lane];
      }
    }
    p2p_cnt = 0;
    __syncwarp();
  };

  auto flush_m2l = [&]() {
    if (!m2l_cnt) return;
    __syncwarp();
    int cid = -1;
    if (lane == 0) {
      cid = atomicAdd(&split_counts[1], 1);
      if (cid >= m2l_cap) atomicOr(&split_counts[4], 1);
    }
    cid = __shfl_sync(0xffffffff, cid, 0);
    if (cid >= 0 && cid < m2l_cap) {
      if (lane == 0) {
        m2l_chunks[cid].next = s_ws[warp_id].m2l_head;
        m2l_chunks[cid].count = m2l_cnt;
        s_ws[warp_id].m2l_head = cid;
      }
      if (lane < m2l_cnt) {
        m2l_chunks[cid].nodes[lane] = s_m2l[warp_id][lane];
      }
    }
    m2l_cnt = 0;
    __syncwarp();
  };

  float4 result{};
  for (int dx_i = 0; dx_i < 3; ++dx_i) {
    const Geometry* __restrict__ st_geom =
        (dx_i == 0) ? src0.geometry : ((dx_i == 1) ? src1.geometry : src2.geometry);
    const NodeMeta* __restrict__ st_meta =
        (dx_i == 0) ? src0.node_meta : ((dx_i == 1) ? src1.node_meta : src2.node_meta);
    const CompactMultipole* __restrict__ st_cm =
        (dx_i == 0) ? src0.compact_m : ((dx_i == 1) ? src1.compact_m : src2.compact_m);
    for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
      const int root_u = wrap_cell(home_y + dy) * 12 + wrap_cell(home_z + dz);
      if (lane == 0) {
        s_node[warp_id][0] = root_u;
        s_mask[warp_id][0] = s_ws[warp_id].valid_mask;
      }
      int sp = 1;
      __syncwarp();
      while (sp > 0) {
        --sp;
        const int u = s_node[warp_id][sp];
        const uint32_t cur_mask = s_mask[warp_id][sp];
        const NodeMeta nm = st_meta[u];
        const float4 c = nm.center;
        const int src_n = nm.n_flags & 0x7FFFFFFF;
        const bool is_internal = (uint32_t(nm.n_flags) >> 31) != 0;
        const float3 dw = periodic_delta_float({c.x, c.y, c.z}, warp_center, p.box);
        const float r2w = norm2(dw);
        const float w_reach = c.w + p.cutoff + warp_radius;
        if (!src_n || r2w > w_reach * w_reach) continue;

        if constexpr (use_m2l) {
          if (can_m2l && cur_mask == s_ws[warp_id].valid_mask && src_n > 1) {
            const float rho_sum = warp_radius + c.w;
            const float soft_w = fmaxf(s_ws[warp_id].warp_soft, nm.max_softening);
            if (rho_sum * rho_sum < 0.4225f * r2w) {
              const float dist = sqrtf(r2w);
              const float closest = dist - warp_radius;
              if (closest > soft_w && dist + warp_radius <= c.w + p.cutoff) {
                const float closest2 = closest * closest;
                const float rhoA2 = warp_radius * warp_radius;
                const float r6m = r2w * r2w * r2w;
                const float old_min_tol = s_ws[warp_id].old_min_tol;
                const bool m_ok = 8.f * nm.power2 < old_min_tol * closest2 * closest2;
                const bool l_ok = 5.f * nm.M_000 * rhoA2 * rhoA2 < old_min_tol * r6m;
                if (initial || (m_ok && l_ok)) {
                  if (lane == 0) {
                    s_m2l[warp_id][m2l_cnt] = (uint32_t(dx_i) << 30) | uint32_t(u);
                    s_ws[warp_id].any_m2l = 1;
                  }
                  ++m2l_cnt;
                  if (m2l_cnt == split_m2l_chunk_sz) flush_m2l();
                  continue;
                }
              }
            }
          }
        }

        const bool lane_active = ((cur_mask >> lane) & 1u) != 0;
        const float3 d = periodic_delta_float({c.x, c.y, c.z}, x, p.box);
        const float r2 = norm2(d);
        const float reach = c.w + p.cutoff;
        const bool in_cut = lane_active && (r2 <= reach * reach);
        const float soft = fmaxf(target_soft, nm.max_softening);
        const float rad2 = c.w * c.w;
        const bool accept = in_cut && (src_n > 1) && (r2 > soft * soft) && (initial ?
            rad2 < (p.theta * p.theta) * r2 :
            rad2 < r2 && 8.f * nm.power2 < old_tol * r2 * r2);
        if (accept) {
          gravity_M2P_periodic_fast(st_cm + u, d.x, d.y, d.z, r2, inv_split, result);
        }
        const uint32_t need_open = __ballot_sync(0xffffffff, in_cut && !accept);
        if (!need_open) continue;

        if (is_internal) {
          const int ch = (lane < 8) ? st_geom[u].children[lane] : -1;
          const uint32_t valid_ch = __ballot_sync(0xffffffff, lane < 8 && ch >= 0);
          const int nch = __popc(valid_ch);
          if (lane < 8 && ch >= 0) {
            const int pos = sp + __popc(valid_ch & ((1u << lane) - 1u));
            s_node[warp_id][pos] = ch;
            s_mask[warp_id][pos] = need_open;
          }
          sp += nch;
          __syncwarp();
        } else {
          const Geometry geom = st_geom[u];
          const int len = geom.end - geom.start;
          if (len > 0) {
            if (lane == 0) {
              s_ws[warp_id].max_leaf_r = fmaxf(s_ws[warp_id].max_leaf_r, c.w);
              s_p2p_mask[warp_id][p2p_cnt] = need_open;
              s_p2p_start[warp_id][p2p_cnt] = geom.start;
              s_p2p_meta[warp_id][p2p_cnt] = (uint32_t(dx_i) << 28) | uint32_t(len);
            }
            ++p2p_cnt;
            if (p2p_cnt == split_p2p_chunk_sz) flush_p2p();
          }
        }
      }
    }
  }

  flush_p2p();
  if constexpr (use_m2l) flush_m2l();
  if (lane_valid) force[i] = result;

  if (lane == 0) {
    const int p2p_h = s_ws[warp_id].p2p_head;
    const int m2l_h = s_ws[warp_id].m2l_head;
    if (p2p_h >= 0 || m2l_h >= 0) {
      head_p2p[local_warp] = p2p_h;
      head_m2l[local_warp] = m2l_h;
      const bool fast_rel = (p.cutoff + 2.f * warp_radius + 2.f * s_ws[warp_id].max_leaf_r) < 0.48f * p.box;
      metas[local_warp] = {
          warp_center, s_ws[warp_id].warp_soft, s_ws[warp_id].valid_mask,
          (s_ws[warp_id].any_m2l ? 1u : 0u) | (fast_rel ? 4u : 0u),
          s_ws[warp_id].sl_start, s_ws[warp_id].sl_count};
      if (p2p_h >= 0) {
        const int pw = atomicAdd(&split_counts[2], 1);
        if (pw < n_batch_warps) p2p_warps[pw] = local_warp;
      }
      if (m2l_h >= 0) {
        const int mw = atomicAdd(&split_counts[3], 1);
        if (mw < n_batch_warps) m2l_warps[mw] = local_warp;
      }
    }
  }
}

__device__ __forceinline__ void gravity_M2L_apply_compact(
    grav_tensor* __restrict__ l_b,
    const CompactMultipole* __restrict__ m_a,
    const potential_derivatives_M2L* __restrict__ pot) {
  static_assert(SELF_GRAVITY_MULTIPOLE_ORDER == 4,
                "gravity_M2L_apply_compact requires SELF_GRAVITY_MULTIPOLE_ORDER == 4");

  const float M_000 = __ldg(&m_a->M_000);
  const float D_000 = pot->D_000;

  /* 0th order term */
  l_b->F_000 += M_000 * D_000;

  /* 1st order terms (dipole is zero in CoM frame) */
  const float D_100 = pot->D_100;
  const float D_010 = pot->D_010;
  const float D_001 = pot->D_001;
  l_b->F_100 += M_000 * D_100;
  l_b->F_010 += M_000 * D_010;
  l_b->F_001 += M_000 * D_001;

  /* 2nd order terms */
  const float M_200 = __ldg(&m_a->M_200);
  const float M_020 = __ldg(&m_a->M_020);
  const float M_002 = __ldg(&m_a->M_002);
  const float M_110 = __ldg(&m_a->M_110);
  const float M_101 = __ldg(&m_a->M_101);
  const float M_011 = __ldg(&m_a->M_011);

  const float D_200 = pot->D_200;
  const float D_020 = pot->D_020;
  const float D_002 = pot->D_002;
  const float D_110 = pot->D_110;
  const float D_101 = pot->D_101;
  const float D_011 = pot->D_011;

  l_b->F_000 += M_200 * D_200 + M_020 * D_020 + M_002 * D_002;
  l_b->F_000 += M_110 * D_110 + M_101 * D_101 + M_011 * D_011;

  l_b->F_200 += M_000 * D_200;
  l_b->F_020 += M_000 * D_020;
  l_b->F_002 += M_000 * D_002;
  l_b->F_110 += M_000 * D_110;
  l_b->F_101 += M_000 * D_101;
  l_b->F_011 += M_000 * D_011;

  /* 3rd order terms */
  const float M_300 = __ldg(&m_a->M_300);
  const float M_030 = __ldg(&m_a->M_030);
  const float M_003 = __ldg(&m_a->M_003);
  const float M_210 = __ldg(&m_a->M_210);
  const float M_201 = __ldg(&m_a->M_201);
  const float M_021 = __ldg(&m_a->M_021);
  const float M_120 = __ldg(&m_a->M_120);
  const float M_012 = __ldg(&m_a->M_012);
  const float M_102 = __ldg(&m_a->M_102);
  const float M_111 = __ldg(&m_a->M_111);

  const float D_300 = pot->D_300;
  const float D_030 = pot->D_030;
  const float D_003 = pot->D_003;
  const float D_210 = pot->D_210;
  const float D_201 = pot->D_201;
  const float D_021 = pot->D_021;
  const float D_120 = pot->D_120;
  const float D_012 = pot->D_012;
  const float D_102 = pot->D_102;
  const float D_111 = pot->D_111;

  l_b->F_000 += M_300 * D_300 + M_030 * D_030 + M_003 * D_003;
  l_b->F_000 += M_210 * D_210 + M_201 * D_201 + M_120 * D_120;
  l_b->F_000 += M_021 * D_021 + M_102 * D_102 + M_012 * D_012;
  l_b->F_000 += M_111 * D_111;

  l_b->F_100 += M_200 * D_300 + M_020 * D_120 + M_002 * D_102;
  l_b->F_100 += M_110 * D_210 + M_101 * D_201 + M_011 * D_111;
  l_b->F_010 += M_200 * D_210 + M_020 * D_030 + M_002 * D_012;
  l_b->F_010 += M_110 * D_120 + M_101 * D_111 + M_011 * D_021;
  l_b->F_001 += M_200 * D_201 + M_020 * D_021 + M_002 * D_003;
  l_b->F_001 += M_110 * D_111 + M_101 * D_102 + M_011 * D_012;

  l_b->F_300 += M_000 * D_300;
  l_b->F_030 += M_000 * D_030;
  l_b->F_003 += M_000 * D_003;
  l_b->F_210 += M_000 * D_210;
  l_b->F_201 += M_000 * D_201;
  l_b->F_120 += M_000 * D_120;
  l_b->F_021 += M_000 * D_021;
  l_b->F_102 += M_000 * D_102;
  l_b->F_012 += M_000 * D_012;
  l_b->F_111 += M_000 * D_111;

  /* 4th order terms */
  const float M_400 = __ldg(&m_a->M_400);
  const float M_040 = __ldg(&m_a->M_040);
  const float M_004 = __ldg(&m_a->M_004);
  const float M_310 = __ldg(&m_a->M_310);
  const float M_301 = __ldg(&m_a->M_301);
  const float M_031 = __ldg(&m_a->M_031);
  const float M_130 = __ldg(&m_a->M_130);
  const float M_013 = __ldg(&m_a->M_013);
  const float M_103 = __ldg(&m_a->M_103);
  const float M_220 = __ldg(&m_a->M_220);
  const float M_202 = __ldg(&m_a->M_202);
  const float M_022 = __ldg(&m_a->M_022);
  const float M_211 = __ldg(&m_a->M_211);
  const float M_121 = __ldg(&m_a->M_121);
  const float M_112 = __ldg(&m_a->M_112);

  const float D_400 = pot->D_400;
  const float D_040 = pot->D_040;
  const float D_004 = pot->D_004;
  const float D_310 = pot->D_310;
  const float D_301 = pot->D_301;
  const float D_031 = pot->D_031;
  const float D_130 = pot->D_130;
  const float D_013 = pot->D_013;
  const float D_103 = pot->D_103;
  const float D_220 = pot->D_220;
  const float D_202 = pot->D_202;
  const float D_022 = pot->D_022;
  const float D_211 = pot->D_211;
  const float D_121 = pot->D_121;
  const float D_112 = pot->D_112;

  l_b->F_000 += M_004 * D_004 + M_013 * D_013 + M_022 * D_022 + M_031 * D_031 +
                M_040 * D_040 + M_103 * D_103 + M_112 * D_112 + M_121 * D_121 +
                M_130 * D_130 + M_202 * D_202 + M_211 * D_211 + M_220 * D_220 +
                M_301 * D_301 + M_310 * D_310 + M_400 * D_400;

  l_b->F_001 += M_003 * D_004 + M_012 * D_013 + M_021 * D_022 + M_030 * D_031 +
                M_102 * D_103 + M_111 * D_112 + M_120 * D_121 + M_201 * D_202 +
                M_210 * D_211 + M_300 * D_301;
  l_b->F_010 += M_003 * D_013 + M_012 * D_022 + M_021 * D_031 + M_030 * D_040 +
                M_102 * D_112 + M_111 * D_121 + M_120 * D_130 + M_201 * D_211 +
                M_210 * D_220 + M_300 * D_310;
  l_b->F_100 += M_003 * D_103 + M_012 * D_112 + M_021 * D_121 + M_030 * D_130 +
                M_102 * D_202 + M_111 * D_211 + M_120 * D_220 + M_201 * D_301 +
                M_210 * D_310 + M_300 * D_400;

  l_b->F_002 += M_002 * D_004 + M_011 * D_013 + M_020 * D_022 + M_101 * D_103 +
                M_110 * D_112 + M_200 * D_202;
  l_b->F_011 += M_002 * D_013 + M_011 * D_022 + M_020 * D_031 + M_101 * D_112 +
                M_110 * D_121 + M_200 * D_211;
  l_b->F_020 += M_002 * D_022 + M_011 * D_031 + M_020 * D_040 + M_101 * D_121 +
                M_110 * D_130 + M_200 * D_220;
  l_b->F_101 += M_002 * D_103 + M_011 * D_112 + M_020 * D_121 + M_101 * D_202 +
                M_110 * D_211 + M_200 * D_301;
  l_b->F_110 += M_002 * D_112 + M_011 * D_121 + M_020 * D_130 + M_101 * D_211 +
                M_110 * D_220 + M_200 * D_310;
  l_b->F_200 += M_002 * D_202 + M_011 * D_211 + M_020 * D_220 + M_101 * D_301 +
                M_110 * D_310 + M_200 * D_400;

  l_b->F_004 += M_000 * D_004;
  l_b->F_013 += M_000 * D_013;
  l_b->F_022 += M_000 * D_022;
  l_b->F_031 += M_000 * D_031;
  l_b->F_040 += M_000 * D_040;
  l_b->F_103 += M_000 * D_103;
  l_b->F_112 += M_000 * D_112;
  l_b->F_121 += M_000 * D_121;
  l_b->F_130 += M_000 * D_130;
  l_b->F_202 += M_000 * D_202;
  l_b->F_211 += M_000 * D_211;
  l_b->F_220 += M_000 * D_220;
  l_b->F_301 += M_000 * D_301;
  l_b->F_310 += M_000 * D_310;
  l_b->F_400 += M_000 * D_400;
}

__global__ void __launch_bounds__(team, 4) eval_m2l_split_kernel(
    const double3* __restrict__ target_x,
    const SplitTravTreeView src0, const SplitTravTreeView src1, const SplitTravTreeView src2,
    const int* __restrict__ target_order,
    const SplitWarpMeta* __restrict__ metas,
    const int* __restrict__ head_m2l,
    const SplitM2LChunk* __restrict__ m2l_chunks,
    const int* __restrict__ m2l_warps, int n_m2l_warps,
    float4* __restrict__ force, int n, Physics p, const int* __restrict__ tree_invalid,
    const int* __restrict__ active_sorted, bool compact_targets) {
  if (*tree_invalid) return;
  const int mw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (mw >= n_m2l_warps) return;
  const int local_warp = m2l_warps[mw];
  const SplitWarpMeta meta = metas[local_warp];
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;

  constexpr int warps_per_block = team / 32;
  __shared__ grav_tensor s_field[warps_per_block];

  const float inv_split = 1.f / p.split;
  grav_tensor lf{};
  int cid = head_m2l[local_warp];
  while (cid >= 0) {
    const int count = m2l_chunks[cid].count;
    if (lane < count) {
      const uint32_t packed = m2l_chunks[cid].nodes[lane];
      const int dx_i = int(packed >> 30);
      const int u = int(packed & 0x3FFFFFFFu);
      const NodeMeta* __restrict__ st_meta =
          (dx_i == 0) ? src0.node_meta : ((dx_i == 1) ? src1.node_meta : src2.node_meta);
      const CompactMultipole* __restrict__ st_cm =
          (dx_i == 0) ? src0.compact_m : ((dx_i == 1) ? src1.compact_m : src2.compact_m);
      const NodeMeta nm = st_meta[u];
      const float4 sc = nm.center;
      const float3 dm = periodic_delta_float(meta.warp_center, {sc.x, sc.y, sc.z}, p.box);
      const float r2m = norm2(dm);
      const float softm = fmaxf(meta.warp_soft, nm.max_softening);
      potential_derivatives_M2L deriv;
      swift_cuda_potential_derivatives_compute_M2L(
          dm.x, dm.y, dm.z, r2m, rsqrtf(r2m), softm, 1, inv_split, &deriv);
      gravity_M2L_apply_compact(&lf, st_cm + u, &deriv);
    }
    cid = m2l_chunks[cid].next;
  }

  const float* lfp = reinterpret_cast<const float*>(&lf);
  float* sfp = reinterpret_cast<float*>(&s_field[warp_id]);
#pragma unroll
  for (int k = 0; k < 35; ++k) {
    float val = lfp[k];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      val += __shfl_down_sync(0xffffffff, val, off);
    if (lane == 0) sfp[k] = val;
  }
  __syncwarp();

  if (lane < meta.count) {
    const int tidx = meta.start + lane;
    const int sorted_i = active_sorted ? active_sorted[tidx] : tidx;
    const int i = compact_targets ? tidx : target_order[sorted_i];
    const double3 px = target_x[i];
    const float3 x{float(px.x), float(px.y), float(px.z)};
    const float4 far = evaluate_tile(s_field[warp_id], periodic_delta_float(x, meta.warp_center, p.box));
    float4 cur = force[i];
    cur.x += far.x;
    cur.y += far.y;
    cur.z += far.z;
    cur.w += far.w;
    force[i] = cur;
  }
}

template <bool fast_split, bool uniform_soft>
__global__ void __launch_bounds__(team, uniform_soft ? 8 : 6) eval_p2p_split_kernel(
    const double3* __restrict__ target_x, float uniform_soft_val,
    const float* __restrict__ target_soft_arr,
    const SlabTreeView src0, const SlabTreeView src1, const SlabTreeView src2,
    const SplitWarpMeta* __restrict__ metas,
    const int* __restrict__ head_p2p,
    const SplitP2PChunk* __restrict__ p2p_chunks,
    const int* __restrict__ p2p_warps, int n_p2p_warps,
    float4* __restrict__ force, int n, Physics p, const int* __restrict__ tree_invalid,
    const int* __restrict__ active_sorted, bool compact_targets) {
  if (*tree_invalid) return;
  const int pw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (pw >= n_p2p_warps) return;
  const int local_warp = p2p_warps[pw];
  const SplitWarpMeta meta = metas[local_warp];

  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int tidx = meta.start + lane;
  const bool lane_valid = lane < meta.count;
  const bool fast_rel = (meta.warp_flags & 4u) != 0;

  constexpr int warps_per_block = team / 32;
  __shared__ float4 s_leaf[warps_per_block][32];
  __shared__ float s_leaf_soft[warps_per_block][uniform_soft ? 1 : 32];

  int sorted_i = -1, i = 0;
  float3 x{};
  float target_soft = uniform_soft_val;
  if (lane_valid) {
    sorted_i = active_sorted ? active_sorted[tidx] : tidx;
    i = compact_targets ? tidx : src1.order[sorted_i];
    const double3 px = target_x[i];
    x = {float(px.x), float(px.y), float(px.z)};
    if (fast_rel) x = periodic_delta_float(x, meta.warp_center, p.box);
    if constexpr (!uniform_soft) {
      if (target_soft_arr) target_soft = target_soft_arr[i];
    }
  }
  const float inv_split = 1.f / p.split;
  const float h_inv = 1.f / target_soft;
  const float h2 = target_soft * target_soft;
  const float h_inv3 = h_inv * h_inv * h_inv;

  float4 result{};
  int cid = head_p2p[local_warp];
  while (cid >= 0) {
    const int count = p2p_chunks[cid].count;
    for (int e = 0; e < count; ++e) {
      const uint32_t mask = p2p_chunks[cid].masks[e];
      const int lstart = p2p_chunks[cid].starts[e];
      const uint32_t lmeta = p2p_chunks[cid].meta[e];
      const int len = int(lmeta & 0x0FFFFFFFu);
      const int dx_i = int(lmeta >> 28);
      const float4* src_packed = (dx_i == 0) ? src0.packed_sources : ((dx_i == 1) ? src1.packed_sources : src2.packed_sources);
      const float* src_soft = (dx_i == 0) ? src0.packed_soft : ((dx_i == 1) ? src1.packed_soft : src2.packed_soft);
      const bool me_active = ((mask >> lane) & 1u) != 0;
      for (int base = 0; base < len; base += 32) {
        const int chunk = min(32, len - base);
        if (lane < chunk) {
          float4 a = src_packed[lstart + base + lane];
          if (fast_rel) {
            const float3 rel_a = periodic_delta_float({a.x, a.y, a.z}, meta.warp_center, p.box);
            a.x = rel_a.x;
            a.y = rel_a.y;
            a.z = rel_a.z;
          }
          s_leaf[warp_id][lane] = a;
          if constexpr (!uniform_soft) {
            s_leaf_soft[warp_id][lane] = src_soft ? src_soft[lstart + base + lane] : uniform_soft_val;
          }
        }
        __syncwarp();
        const bool has_self = (dx_i == 1) && __any_sync(0xffffffff, lane_valid && sorted_i >= lstart + base && sorted_i < lstart + base + chunk);
        if (me_active) {
          if (fast_rel && !has_self) {
#pragma unroll 4
            for (int k = 0; k < chunk; ++k) {
              const float4 a = s_leaf[warp_id][k];
              const float3 r{a.x - x.x, a.y - x.y, a.z - x.z};
              const float dist2 = norm2(r);
              float accel, pot;
              if constexpr (uniform_soft) {
                if constexpr (fast_split) {
                  iact_grav_pp_truncated_fast(dist2, h2, h_inv, h_inv3, a.w, inv_split, &accel, &pot);
                } else {
                  iact_grav_pp_truncated(dist2, h2, h_inv, h_inv3, a.w, inv_split, &accel, &pot);
                }
              } else {
                const float hij = fmaxf(target_soft, s_leaf_soft[warp_id][k]);
                const float hi = 1.f / hij;
                if constexpr (fast_split) {
                  iact_grav_pp_truncated_fast(dist2, hij * hij, hi, hi * hi * hi, a.w, inv_split, &accel, &pot);
                } else {
                  iact_grav_pp_truncated(dist2, hij * hij, hi, hi * hi * hi, a.w, inv_split, &accel, &pot);
                }
              }
              result.x += r.x * accel;
              result.y += r.y * accel;
              result.z += r.z * accel;
              result.w += pot;
            }
          } else {
            for (int k = 0; k < chunk; ++k) {
              if (dx_i == 1 && sorted_i == lstart + base + k) continue;
              const float4 a = s_leaf[warp_id][k];
              const float3 r = fast_rel
                  ? float3{a.x - x.x, a.y - x.y, a.z - x.z}
                  : periodic_delta_float({a.x, a.y, a.z}, x, p.box);
              const float dist2 = norm2(r);
              float accel, pot;
              if constexpr (uniform_soft) {
                if constexpr (fast_split) {
                  iact_grav_pp_truncated_fast(dist2, h2, h_inv, h_inv3, a.w, inv_split, &accel, &pot);
                } else {
                  iact_grav_pp_truncated(dist2, h2, h_inv, h_inv3, a.w, inv_split, &accel, &pot);
                }
              } else {
                const float hij = fmaxf(target_soft, s_leaf_soft[warp_id][k]);
                const float hi = 1.f / hij;
                if constexpr (fast_split) {
                  iact_grav_pp_truncated_fast(dist2, hij * hij, hi, hi * hi * hi, a.w, inv_split, &accel, &pot);
                } else {
                  iact_grav_pp_truncated(dist2, hij * hij, hi, hi * hi * hi, a.w, inv_split, &accel, &pot);
                }
              }
              result.x += r.x * accel;
              result.y += r.y * accel;
              result.z += r.z * accel;
              result.w += pot;
            }
          }
        }
        __syncwarp();
      }
    }
    cid = p2p_chunks[cid].next;
  }
  if (lane_valid) {
    float4 cur = force[i];
    cur.x += result.x;
    cur.y += result.y;
    cur.z += result.z;
    cur.w += result.w;
    force[i] = cur;
  }
}

__device__ unsigned char next_bin(float3 raw, const float3 mesh, float eps,
    Tick ti, unsigned char previous, Physics p, double a, double H, bool initial,
    float dt_rms, Tick span_cap) {
  const double a_inv2 = p.comoving ? (1.0 / (a * a)) : 1.0;
  float ax = raw.x * p.G * a_inv2, ay = raw.y * p.G * a_inv2, az = raw.z * p.G * a_inv2;
  ax += mesh.x * a_inv2; ay += mesh.y * a_inv2; az += mesh.z * a_inv2;
  const float ac2 = ax * ax + ay * ay + az * az;
  const float inv = ac2 > 0.f ? 1.f / sqrtf(ac2) : FLT_MAX;
  float dt;
  if (p.comoving) {
    dt = sqrtf(2.f * kernel_gravity_softening_plummer_equivalent_inv * float(a) * p.grav_eta * eps * inv);
    dt = fminf(dt, dt_rms);
    dt *= float(H) * p.short_step_factor;
    if (p.dynamic_pm) dt = fminf(dt, float(p.dt_max));
  } else {
    dt = sqrtf(2.f * kernel_gravity_softening_plummer_equivalent_inv * p.grav_eta * eps * inv) * p.short_step_factor;
    if (p.dt_max > 0.0) dt = fminf(dt, float(p.dt_max));
  }
  const double fractional_ticks = double(dt) * timeline / p.log_span;
  const Tick trial = Tick(fmin(double(span_cap), fmax(4.0, fractional_ticks)));
  Tick span = 1ULL << (63 - __clzll(trial));
  if (!initial) {
    const Tick old = bin_size(previous);
    if (span > 2 * old) span = 2 * old;
    if (span > old && (ti & (span - 1))) span = old;
  }
  return 62 - __clzll(span);
}

__global__ void build_kick_table(double* finish, double* begin, Physics p, Tick ti) {
  const int b = threadIdx.x;
  if (b == 0 || b > 56) return;
  const Tick width = bin_size(b);
  finish[b] = p.factors(ti >= width / 2 ? ti - width / 2 : 0, ti, false);
  begin[b] = p.factors(ti, ti + width / 2, false);
}

__global__ void init_meta(Meta* m, const int* tree_invalid, unsigned int nodes) {
  *m = {timeline, 0, unsigned(*tree_invalid), nodes};
}

__global__ void integrate_slab(float3* velocity, float* old_acc_arr, unsigned char* bins,
    const float4* raw, const float3* mesh_acc, const int* active_ids,
    const int* src_order, int active_n, float uniform_soft, const float* soft_arr,
    Physics p, Tick ti, Meta* stats, const double* finish, const double* begin,
    double mesh_kick, double a, double H, float dt_rms, Tick span_cap,
    const int* tree_invalid, float3* synchronized_v, double closing_pm_kick) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  Tick next = timeline;
  unsigned int changed = 0, bad = 0;
  if (idx < active_n && !*tree_invalid) {
    const int i = active_ids ? src_order[active_ids[idx]] : idx;
    const unsigned char old = ti ? bins[i] : 1;
    const auto f = raw[i];
    const float3 am = mesh_acc[i];
    float3 v = velocity[i];
    if (ti) {
      v.x += p.G * f.x * finish[old];
      v.y += p.G * f.y * finish[old];
      v.z += p.G * f.z * finish[old];
    }
    if (synchronized_v) {
      float3 synced = v;
      synced.x += am.x * closing_pm_kick;
      synced.y += am.y * closing_pm_kick;
      synced.z += am.z * closing_pm_kick;
      synchronized_v[i] = synced;
    }
    if (mesh_kick) {
      v.x += am.x * mesh_kick;
      v.y += am.y * mesh_kick;
      v.z += am.z * mesh_kick;
    }
    const float3 total = {f.x + am.x / p.G, f.y + am.y / p.G, f.z + am.z / p.G};
    const float oa = sqrtf(norm2(total));
    old_acc_arr[i] = oa;
    if (ti != timeline) {
      const float part_soft = soft_arr ? soft_arr[i] : uniform_soft;
      const unsigned char b = next_bin({f.x, f.y, f.z}, am, part_soft, ti, old, p, a, H, !ti, dt_rms, span_cap);
      bins[i] = b;
      next = ti + bin_size(b);
      v.x += p.G * f.x * begin[b];
      v.y += p.G * f.y * begin[b];
      v.z += p.G * f.z * begin[b];
    }
    velocity[i] = v;
    changed = 1;
    bad = !(isfinite(v.x) && isfinite(v.y) && isfinite(v.z) && isfinite(oa));
  }
  for (int off = 16; off > 0; off /= 2) {
    const auto t = __shfl_down_sync(0xffffffff, next, off);
    if (t < next) next = t;
    changed += __shfl_down_sync(0xffffffff, changed, off);
    bad += __shfl_down_sync(0xffffffff, bad, off);
  }
  __shared__ Tick warp_next[team / 32];
  __shared__ unsigned int warp_changed[team / 32], warp_bad[team / 32];
  if (!(threadIdx.x & 31)) {
    warp_next[threadIdx.x / 32] = next;
    warp_changed[threadIdx.x / 32] = changed;
    warp_bad[threadIdx.x / 32] = bad;
  }
  __syncthreads();
  if (!threadIdx.x) {
    for (int w = 1; w < team / 32; ++w) {
      if (warp_next[w] < next) next = warp_next[w];
      changed += warp_changed[w];
      bad += warp_bad[w];
    }
    if (next != timeline) atomicMin(&stats->next, next);
    if (changed) atomicAdd(&stats->active, changed);
    if (bad) atomicAdd(&stats->invalid, bad);
  }
}

__global__ void reduce_energy_mom_kernel(
    const double3* __restrict__ pos,
    const float3* __restrict__ vel,
    const float4* __restrict__ raw,
    const float3* __restrict__ mesh_acc,
    const float* __restrict__ pm_pot,
    const unsigned char* __restrict__ bins,
    const float* __restrict__ mass_arr,
    float uniform_mass,
    int n,
    Tick ti,
    const double* __restrict__ finish,
    double closing_pm_kick,
    double grav_G,
    double pot_norm,
    double inv_a,
    double* __restrict__ out_accum) {
  double v[10] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    const double m = mass_arr ? double(mass_arr[i]) : double(uniform_mass);
    const float4 rf = raw[i];
    const float3 am = mesh_acc ? mesh_acc[i] : float3{0.f, 0.f, 0.f};
    const double pp = pm_pot ? double(pm_pot[i]) : 0.0;
    double vx = double(vel[i].x);
    double vy = double(vel[i].y);
    double vz = double(vel[i].z);
    if (ti) {
      const unsigned char old = bins[i];
      const double kf = finish[old];
      vx += grav_G * double(rf.x) * kf;
      vy += grav_G * double(rf.y) * kf;
      vz += grav_G * double(rf.z) * kf;
    }
    if (closing_pm_kick != 0.0) {
      vx += double(am.x) * closing_pm_kick;
      vy += double(am.y) * closing_pm_kick;
      vz += double(am.z) * closing_pm_kick;
    }
    vx *= inv_a;
    vy *= inv_a;
    vz *= inv_a;
    const double v2 = vx * vx + vy * vy + vz * vz;
    const double phi = (double(rf.w) + pot_norm) * grav_G + pp;
    const double3 x = pos[i];
    v[0] += 0.5 * m * v2;
    v[1] += 0.5 * m * phi;
    v[2] += m * vx;
    v[3] += m * vy;
    v[4] += m * vz;
    v[5] += m * (x.y * vz - x.z * vy);
    v[6] += m * (x.z * vx - x.x * vz);
    v[7] += m * (x.x * vy - x.y * vx);
    v[8] += m;
    v[9] += v2;
  }
  __shared__ double smem[8][10];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  for (int k = 0; k < 10; ++k) {
    double val = v[k];
    for (int off = 16; off > 0; off >>= 1) val += __shfl_down_sync(0xffffffff, val, off);
    if (lane == 0) smem[wid][k] = val;
  }
  __syncthreads();
  if (threadIdx.x < 10) {
    double sum = 0.0;
    const int nwarps = blockDim.x >> 5;
    for (int w = 0; w < nwarps; ++w) sum += smem[w][threadIdx.x];
    atomicAdd(&out_accum[threadIdx.x], sum);
  }
}

struct DevMigrant {
  double3 x;
  float3 v, mesh_acc;
  float old_acc, mass, soft;
  uint32_t orig_idx;
  int dst_slab;
  unsigned char bin;
  unsigned char pad[3];
};

struct ScanPartial {
  uint64_t active;
  Tick min_nx;
  double v2_sum;
  float max_v;
};

__global__ void drift_only_kernel(
    double3* d_x, const float3* d_v, int ns, double drift_fac, double box, bool periodic) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= ns) return;
  const float3 vi = d_v[k];
  double3 px = d_x[k];
  px.x += drift_fac * double(vi.x);
  px.y += drift_fac * double(vi.y);
  px.z += drift_fac * double(vi.z);
  if (periodic) {
    px.x = wrapped_coord(px.x, box);
    px.y = wrapped_coord(px.y, box);
    px.z = wrapped_coord(px.z, box);
  }
  d_x[k] = px;
}

__global__ void compute_node_vel_slab_kernel(
    const Geometry* nodes, const ForceNode* force_nodes, const float3* v_in,
    const float* mass_or_null, float uniform_mass, const int* sorted,
    int depth, const int* levels, float4* node_vel_max) {
  const int i = levels[depth] + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= levels[depth + 1]) return;
  if (!force_nodes[i].n) {
    node_vel_max[i] = {0.f, 0.f, 0.f, 0.f};
    return;
  }
  const Geometry geom = nodes[i];
  bool leaf = true;
  for (int k = 0; k < 8; ++k) leaf &= (geom.children[k] < 0);

  if (leaf) {
    double vcx = 0.0, vcy = 0.0, vcz = 0.0, msum = 0.0;
    for (int j = geom.start; j < geom.end; ++j) {
      const int pid = sorted[j];
      const float3 pv = v_in[pid];
      const float mp = mass_or_null ? mass_or_null[pid] : uniform_mass;
      msum += double(mp);
      vcx += double(mp) * double(pv.x);
      vcy += double(mp) * double(pv.y);
      vcz += double(mp) * double(pv.z);
    }
    const double inv_m = (msum > 0.0) ? (1.0 / msum) : 0.0;
    const float cx = float(vcx * inv_m);
    const float cy = float(vcy * inv_m);
    const float cz = float(vcz * inv_m);
    float vmax = 0.f;
    for (int j = geom.start; j < geom.end; ++j) {
      const int pid = sorted[j];
      const float3 pv = v_in[pid];
      const float dvx = pv.x - cx, dvy = pv.y - cy, dvz = pv.z - cz;
      vmax = fmaxf(vmax, sqrtf(dvx * dvx + dvy * dvy + dvz * dvz));
    }
    node_vel_max[i] = {cx, cy, cz, vmax};
  } else {
    double vcx = 0.0, vcy = 0.0, vcz = 0.0, msum = 0.0;
    for (int k = 0; k < 8; ++k) {
      const int ch = geom.children[k];
      if (ch < 0 || !force_nodes[ch].n) continue;
      const float cm = force_nodes[ch].m.M_000;
      const float4 cvm = node_vel_max[ch];
      msum += double(cm);
      vcx += double(cm) * double(cvm.x);
      vcy += double(cm) * double(cvm.y);
      vcz += double(cm) * double(cvm.z);
    }
    const double inv_m = (msum > 0.0) ? (1.0 / msum) : 0.0;
    const float cx = float(vcx * inv_m);
    const float cy = float(vcy * inv_m);
    const float cz = float(vcz * inv_m);
    float vmax = 0.f;
    for (int k = 0; k < 8; ++k) {
      const int ch = geom.children[k];
      if (ch < 0 || !force_nodes[ch].n) continue;
      const float4 cvm = node_vel_max[ch];
      const float dvx = cvm.x - cx, dvy = cvm.y - cy, dvz = cvm.z - cz;
      vmax = fmaxf(vmax, cvm.w + sqrtf(dvx * dvx + dvy * dvy + dvz * dvz));
    }
    node_vel_max[i] = {cx, cy, cz, vmax};
  }
}

__global__ void drift_tree_nodes_kernel(
    ForceNode* force_nodes, const float4* node_vel_max,
    NodeMeta* __restrict__ node_meta,
    int num_nodes, float drift_fac_f, float box) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_nodes) return;
  ForceNode fn = force_nodes[i];
  if (!fn.n) return;
  const float4 vm = node_vel_max[i];
  float cx = fn.center.x + drift_fac_f * vm.x;
  float cy = fn.center.y + drift_fac_f * vm.y;
  float cz = fn.center.z + drift_fac_f * vm.z;
  if (cx < 0.f || cx >= box) { cx -= floorf(cx / box) * box; if (cx >= box) cx = 0.f; }
  if (cy < 0.f || cy >= box) { cy -= floorf(cy / box) * box; if (cy >= box) cy = 0.f; }
  if (cz < 0.f || cz >= box) { cz -= floorf(cz / box) * box; if (cz >= box) cz = 0.f; }
  fn.center.x = cx;
  fn.center.y = cy;
  fn.center.z = cz;
  fn.center.w += 1.25f * drift_fac_f * vm.w;
  force_nodes[i] = fn;
  if (node_meta) node_meta[i].center = fn.center;
}

__global__ void mark_tail_survivors_kernel(
    const unsigned char* d_flags, int keep_ns, int m_cnt, unsigned char* tail_keep_flags) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < m_cnt) tail_keep_flags[j] = (d_flags[keep_ns + j] == 0) ? 1 : 0;
}

__global__ void compact_migrants_parallel_kernel(
    double3* d_x, float3* d_v, float3* d_mesh_acc, float* d_old_acc,
    unsigned char* d_bins, uint32_t* d_orig_idx, float* d_mass, float* d_soft,
    const int* mig_ids, const int* tail_keep, const int* d_K) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= *d_K) return;
  const int dst = mig_ids[t];
  const int src = tail_keep[t];
  d_x[dst] = d_x[src];
  d_v[dst] = d_v[src];
  d_mesh_acc[dst] = d_mesh_acc[src];
  d_old_acc[dst] = d_old_acc[src];
  d_bins[dst] = d_bins[src];
  d_orig_idx[dst] = d_orig_idx[src];
  if (d_mass) d_mass[dst] = d_mass[src];
  if (d_soft) d_soft[dst] = d_soft[src];
}

__global__ void drift_flag_migrants_kernel(
    double3* d_x, const float3* d_v, unsigned char* d_flags, int ns,
    int cur_slab, double drift_fac, double box, double inv_wx, bool periodic) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= ns) return;
  const float3 vi = d_v[k];
  double3 px = d_x[k];
  px.x += drift_fac * double(vi.x);
  px.y += drift_fac * double(vi.y);
  px.z += drift_fac * double(vi.z);
  if (periodic) {
    px.x = wrapped_coord(px.x, box);
    px.y = wrapped_coord(px.y, box);
    px.z = wrapped_coord(px.z, box);
  }
  d_x[k] = px;
  const int dest = min(max(int(px.x * inv_wx), 0), num_slabs - 1);
  d_flags[k] = (dest != cur_slab) ? 1 : 0;
}

template <bool use_m2l>
__global__ void __launch_bounds__(team, 4) tree_force_warp_isolated(
    const double3* target_x, const float* target_old_acc, float uniform_soft,
    const float* target_soft_arr, int target_slab, AllTreesView all_trees,
    float4* force, int n, Physics p, bool initial, const int* tree_invalid,
    const int* active_sorted, const int* active_count) {
  const int targets = active_count ? *active_count : n;
  const int warp_base = (blockIdx.x * blockDim.x + threadIdx.x) & ~31;
  if (warp_base >= targets || *tree_invalid) return;
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const bool lane_valid = (warp_base + lane) < targets;
  const uint32_t valid_mask = __ballot_sync(0xffffffff, lane_valid);

  constexpr int warps_per_block = team / 32;
  constexpr int warp_stack_size = 104;
  __shared__ int s_node[warps_per_block][warp_stack_size];
  __shared__ uint32_t s_mask[warps_per_block][warp_stack_size];
  __shared__ float4 s_leaf[warps_per_block][64];
  __shared__ float s_leaf_soft[warps_per_block][64];
  __shared__ grav_tensor s_field[warps_per_block];

  const auto& src_t = all_trees.t[target_slab];
  int sorted_i = 0, i = 0, home_cell = -1;
  float3 x{};
  float old = 1.f;
  if (lane_valid) {
    const int tidx = warp_base + lane;
    sorted_i = active_sorted ? active_sorted[tidx] : tidx;
    i = src_t.order[sorted_i];
    const double3 px = target_x[i];
    x = {float(px.x), float(px.y), float(px.z)};
    old = target_old_acc[i];
    home_cell = int(src_t.home_cell[sorted_i]);
  }
  const float target_soft = (lane_valid && target_soft_arr) ? target_soft_arr[i] : uniform_soft;
  const float h_inv = 1.f / target_soft;
  const float h2 = target_soft * target_soft;
  const float h_inv3 = h_inv * h_inv * h_inv;

  auto add_ext_plummer = [&](float4& res) {
    if (p.ext_M > 0.f) {
      const float3 dr{p.ext_cx - x.x, p.ext_cy - x.y, p.ext_cz - x.z};
      const float r2e = norm2(dr) + p.ext_a * p.ext_a;
      const float re_inv = rsqrtf(r2e);
      const float ae = p.ext_M * re_inv * re_inv * re_inv;
      res.x += dr.x * ae;
      res.y += dr.y * ae;
      res.z += dr.z * ae;
      res.w += -p.ext_M * re_inv;
    }
  };

  const int root0 = __shfl_sync(0xffffffff, home_cell, 0);
  const bool same_home = __all_sync(0xffffffff, !lane_valid || home_cell == root0);
  if (!same_home) {
    if (!lane_valid) return;
    float4 result{};
    for (int src_s = 0; src_s < num_slabs; ++src_s) {
      const auto& st = all_trees.t[src_s];
      for (int root_u = 0; root_u < slab_roots_count; ++root_u) {
        if (!st.sources[root_u].n) continue;
        int node = root_u;
        while (node >= 0) {
          const auto& source = st.sources[node];
          const float4 c = source.center;
          if (!source.n) { node = st.escape[node]; continue; }
          const float3 d{c.x - x.x, c.y - x.y, c.z - x.z};
          const float r2 = norm2(d);
          const float soft = fmaxf(target_soft, source.m.max_softening);
          const float min_r2 = soft * soft;
          const float rad2 = c.w * c.w;
          const bool accept = source.n > 1 && r2 > min_r2 && (initial ?
              rad2 < p.theta * p.theta * r2 :
              rad2 < r2 && 8.f * source.m.power[2] < p.mac_tolerance * old * r2 * r2);
          if (accept) {
            reduced_grav_tensor contribution{};
            swift_cuda_gravity_M2P(&source.m, d.x, d.y, d.z, r2, soft, 0, 0.f, &contribution);
            result.x += contribution.F_100; result.y += contribution.F_010;
            result.z += contribution.F_001; result.w += contribution.F_000;
            node = st.escape[node];
            continue;
          }
          if (st.first_child[node] >= 0) { node = st.first_child[node]; continue; }
          const int start = st.geometry[node].start, end = st.geometry[node].end;
          for (int j = start; j < end; ++j) {
            if (src_s == target_slab && sorted_i == j) continue;
            const auto a = st.packed_sources[j];
            const float3 r{a.x - x.x, a.y - x.y, a.z - x.z};
            const float distance = norm2(r);
            float accel, pot;
            if (!target_soft_arr && !st.packed_soft) {
              iact_grav_pp_full(distance, h2, h_inv, h_inv3, a.w, &accel, &pot);
            } else {
              const float sh = st.packed_soft ? st.packed_soft[j] : uniform_soft;
              const float hij = fmaxf(target_soft, sh);
              const float hi = 1.f / hij;
              iact_grav_pp_full(distance, hij * hij, hi, hi * hi * hi, a.w, &accel, &pot);
            }
            result.x += r.x * accel; result.y += r.y * accel; result.z += r.z * accel; result.w += pot;
          }
          node = st.escape[node];
        }
      }
    }
    add_ext_plummer(result);
    force[i] = result;
    return;
  }

  float min_x = lane_valid ? x.x : FLT_MAX, max_x = lane_valid ? x.x : -FLT_MAX;
  float min_y = lane_valid ? x.y : FLT_MAX, max_y = lane_valid ? x.y : -FLT_MAX;
  float min_z = lane_valid ? x.z : FLT_MAX, max_z = lane_valid ? x.z : -FLT_MAX;
  float old_min = lane_valid ? old : FLT_MAX;
  float warp_soft_acc = lane_valid ? target_soft : 0.f;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    min_x = fminf(min_x, __shfl_xor_sync(0xffffffff, min_x, off));
    max_x = fmaxf(max_x, __shfl_xor_sync(0xffffffff, max_x, off));
    min_y = fminf(min_y, __shfl_xor_sync(0xffffffff, min_y, off));
    max_y = fmaxf(max_y, __shfl_xor_sync(0xffffffff, max_y, off));
    min_z = fminf(min_z, __shfl_xor_sync(0xffffffff, min_z, off));
    max_z = fmaxf(max_z, __shfl_xor_sync(0xffffffff, max_z, off));
    old_min = fminf(old_min, __shfl_xor_sync(0xffffffff, old_min, off));
    warp_soft_acc = fmaxf(warp_soft_acc, __shfl_xor_sync(0xffffffff, warp_soft_acc, off));
  }
  const float3 warp_center{0.5f * (min_x + max_x), 0.5f * (min_y + max_y), 0.5f * (min_z + max_z)};
  const float3 d_wc = lane_valid ? float3{x.x - warp_center.x, x.y - warp_center.y, x.z - warp_center.z}
                                 : float3{0.f, 0.f, 0.f};
  float r2_max = norm2(d_wc);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    r2_max = fmaxf(r2_max, __shfl_xor_sync(0xffffffff, r2_max, off));
  const float warp_radius = sqrtf(r2_max) + 1e-6f;
  const float warp_soft = warp_soft_acc;

  const bool can_m2l = use_m2l && (__popc(valid_mask) >= 8);
  if constexpr (use_m2l) {
    if (can_m2l) {
      float* sf = reinterpret_cast<float*>(&s_field[warp_id]);
      if (lane < 35) sf[lane] = 0.f;
      if (lane + 32 < 35) sf[lane + 32] = 0.f;
    }
  }
  __syncwarp();

  int my_m2l_slab = -1, my_m2l_node = -1, m2l_count = 0;
  bool any_m2l = false;
  auto flush_m2l = [&]() {
    if (!m2l_count) return;
    grav_tensor lf{};
    if (lane < m2l_count) {
      const auto& src = all_trees.t[my_m2l_slab].sources[my_m2l_node];
      const float4 sc = src.center;
      const float3 dm{warp_center.x - sc.x, warp_center.y - sc.y, warp_center.z - sc.z};
      const float r2m = norm2(dm);
      const float softm = fmaxf(warp_soft, src.m.max_softening);
      potential_derivatives_M2L deriv;
      swift_cuda_potential_derivatives_compute_M2L(
          dm.x, dm.y, dm.z, r2m, rsqrtf(r2m), softm, 0, 0.f, &deriv);
      swift_cuda_gravity_M2L_apply(&lf, &src.m, &deriv);
    }
    float* lfp = reinterpret_cast<float*>(&lf);
    float* sfp = reinterpret_cast<float*>(&s_field[warp_id]);
#pragma unroll
    for (int k = 0; k < 35; ++k) {
      float val = lfp[k];
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
      if (lane == 0) sfp[k] += val;
    }
    __syncwarp();
    m2l_count = 0;
    my_m2l_slab = -1;
    my_m2l_node = -1;
  };

  float4 result{};
  for (int src_s = 0; src_s < num_slabs; ++src_s) {
    const auto& st = all_trees.t[src_s];
    for (int root_u = 0; root_u < slab_roots_count; ++root_u) {
      if (!st.sources[root_u].n) continue;
      if (lane == 0) {
        s_node[warp_id][0] = root_u;
        s_mask[warp_id][0] = valid_mask;
      }
      int sp = 1;
      __syncwarp();
      while (sp > 0) {
        --sp;
        const int u = s_node[warp_id][sp];
        const uint32_t cur_mask = s_mask[warp_id][sp];
        const auto& source = st.sources[u];
        const float4 c = source.center;
        const int src_n = source.n;
        if (!src_n) continue;

        if constexpr (use_m2l) {
          if (can_m2l && cur_mask == valid_mask && src_n > 1) {
            const float3 dm{warp_center.x - c.x, warp_center.y - c.y, warp_center.z - c.z};
            const float r2m = norm2(dm);
            const float rho_sum = warp_radius + c.w;
            const float soft_w = fmaxf(warp_soft, source.m.max_softening);
            if (rho_sum * rho_sum < 0.4225f * r2m) {
              const float dist = sqrtf(r2m);
              const float closest = dist - warp_radius;
              if (closest > soft_w) {
                const float closest2 = closest * closest;
                const float rhoA2 = warp_radius * warp_radius;
                const float r6m = r2m * r2m * r2m;
                const bool m_ok = 8.f * source.m.power[2] < p.mac_tolerance * old_min * closest2 * closest2;
                const bool l_ok = 5.f * source.m.M_000 * rhoA2 * rhoA2 < p.mac_tolerance * old_min * r6m;
                if (initial || (m_ok && l_ok)) {
                  if (lane == m2l_count) {
                    my_m2l_slab = src_s;
                    my_m2l_node = u;
                  }
                  ++m2l_count;
                  any_m2l = true;
                  if (m2l_count == 32) flush_m2l();
                  continue;
                }
              }
            }
          }
        }

        const bool lane_active = ((cur_mask >> lane) & 1u) != 0;
        const float3 d{c.x - x.x, c.y - x.y, c.z - x.z};
        const float r2 = norm2(d);
        const float soft = fmaxf(target_soft, source.m.max_softening);
        const float rad2 = c.w * c.w;
        const bool accept = lane_active && (src_n > 1) && (r2 > soft * soft) && (initial ?
            rad2 < p.theta * p.theta * r2 :
            rad2 < r2 && 8.f * source.m.power[2] < p.mac_tolerance * old * r2 * r2);
        if (accept) {
          reduced_grav_tensor contribution{};
          swift_cuda_gravity_M2P(&source.m, d.x, d.y, d.z, r2, soft, 0, 0.f, &contribution);
          result.x += contribution.F_100; result.y += contribution.F_010;
          result.z += contribution.F_001; result.w += contribution.F_000;
        }
        const uint32_t need_open = __ballot_sync(0xffffffff, lane_active && !accept);
        if (!need_open) continue;

        if (st.first_child[u] >= 0) {
          const int ch = (lane < 8) ? st.geometry[u].children[lane] : -1;
          const uint32_t valid_ch = __ballot_sync(0xffffffff, lane < 8 && ch >= 0);
          const int nch = __popc(valid_ch);
          if (lane < 8 && ch >= 0) {
            const int pos = sp + __popc(valid_ch & ((1u << lane) - 1u));
            s_node[warp_id][pos] = ch;
            s_mask[warp_id][pos] = need_open;
          }
          sp += nch;
          __syncwarp();
        } else {
          const int start = st.geometry[u].start, end = st.geometry[u].end;
          for (int base = start; base < end; base += 64) {
            const int chunk = min(64, end - base);
            if (lane < chunk) s_leaf[warp_id][lane] = st.packed_sources[base + lane];
            if (lane + 32 < chunk) s_leaf[warp_id][lane + 32] = st.packed_sources[base + lane + 32];
            if (st.packed_soft) {
              if (lane < chunk) s_leaf_soft[warp_id][lane] = st.packed_soft[base + lane];
              if (lane + 32 < chunk) s_leaf_soft[warp_id][lane + 32] = st.packed_soft[base + lane + 32];
            }
            __syncwarp();
            if ((need_open >> lane) & 1u) {
              for (int k = 0; k < chunk; ++k) {
                if (src_s == target_slab && sorted_i == base + k) continue;
                const float4 a = s_leaf[warp_id][k];
                const float3 r{a.x - x.x, a.y - x.y, a.z - x.z};
                const float distance = norm2(r);
                float accel, pot;
                if (!target_soft_arr && !st.packed_soft) {
                  iact_grav_pp_full(distance, h2, h_inv, h_inv3, a.w, &accel, &pot);
                } else {
                  const float sh = st.packed_soft ? s_leaf_soft[warp_id][k] : uniform_soft;
                  const float hij = fmaxf(target_soft, sh);
                  const float hi = 1.f / hij;
                  iact_grav_pp_full(distance, hij * hij, hi, hi * hi * hi, a.w, &accel, &pot);
                }
                result.x += r.x * accel; result.y += r.y * accel; result.z += r.z * accel; result.w += pot;
              }
            }
            __syncwarp();
          }
        }
      }
    }
  }
  if constexpr (use_m2l) {
    if (any_m2l) {
      flush_m2l();
      if (lane_valid) {
        const float4 far = evaluate_tile(s_field[warp_id], float3{x.x - warp_center.x, x.y - warp_center.y, x.z - warp_center.z});
        result.x += far.x; result.y += far.y; result.z += far.z; result.w += far.w;
      }
    }
  }
  if (lane_valid) {
    add_ext_plummer(result);
    force[i] = result;
  }
}

__device__ __forceinline__ double minimum_image_double(double dx, double box) {
  const double half = 0.5 * box;
  return dx > half ? dx - box : (dx < -half ? dx + box : dx);
}

__device__ __forceinline__ double wendland_c2_force_d(double u) {
  double w = 21.0 * u - 90.0;
  w = w * u + 140.0;
  w = w * u - 84.0;
  w = w * u;
  w = w * u + 14.0;
  return w;
}

__global__ void ewald_kspace_accum_kernel(
    AllSlabsData slabs, float first_mass, const EwaldMode* modes,
    int n_modes, double* out_C, double* out_S) {
  const int m = blockIdx.x;
  if (m >= n_modes) return;
  const EwaldMode md = modes[m];
  double c_sum = 0.0, s_sum = 0.0;
  for (int s = 0; s < num_slabs; ++s) {
    const int ns = slabs.ns[s];
    const double3* px = slabs.x[s];
    const float* pm = slabs.mass[s];
    for (int j = threadIdx.x; j < ns; j += blockDim.x) {
      const double3 pos = px[j];
      const double mp = pm ? double(pm[j]) : double(first_mass);
      const double ph = md.kx * pos.x + md.ky * pos.y + md.kz * pos.z;
      double sv, cv;
      sincos(ph, &sv, &cv);
      c_sum += mp * cv;
      s_sum += mp * sv;
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    c_sum += __shfl_down_sync(0xffffffff, c_sum, off);
    s_sum += __shfl_down_sync(0xffffffff, s_sum, off);
  }
  __shared__ double wc[8], ws[8];
  const int wid = threadIdx.x >> 5;
  if (!(threadIdx.x & 31)) {
    wc[wid] = c_sum;
    ws[wid] = s_sum;
  }
  __syncthreads();
  if (!threadIdx.x) {
    double fc = 0.0, fs = 0.0;
    const int nw = blockDim.x >> 5;
    for (int w = 0; w < nw; ++w) {
      fc += wc[w];
      fs += ws[w];
    }
    out_C[m] = fc;
    out_S[m] = fs;
  }
}

__global__ void ewald_target_force_kernel(
    AllSlabsData slabs, float first_mass, float first_soft,
    const double3* tgt_x, const float* tgt_h, const int* tgt_slab, const int* tgt_lid,
    int n_tgt, double box, double r_s, double grav_G,
    const EwaldMode* modes, const double* C_k, const double* S_k, int n_modes,
    double3* out_short, double3* out_total) {
  const int t = blockIdx.x;
  if (t >= n_tgt) return;
  const double3 xt = tgt_x[t];
  const double ht = double(tgt_h[t]);
  const int t_s = tgt_slab[t];
  const int t_l = tgt_lid[t];
  const double alpha = 4.0 / box;
  const double a2 = alpha * alpha;
  const double r_cut = 1.05 * box;
  const double r_cut2 = r_cut * r_cut;
  const double short_cut = 7.0 * r_s;
  const double short_cut2 = short_cut * short_cut;
  const double inv_rs = 1.0 / r_s;
  const double two_sqrt_pi = M_2_SQRTPI;
  const double taylor_pref = 2.0 * two_sqrt_pi * alpha * a2;

  double sx = 0.0, sy = 0.0, sz = 0.0;
  double fx = 0.0, fy = 0.0, fz = 0.0;

  for (int s = 0; s < num_slabs; ++s) {
    const int ns = slabs.ns[s];
    const double3* px = slabs.x[s];
    const float* pm = slabs.mass[s];
    const float* ps = slabs.soft[s];
    for (int j = threadIdx.x; j < ns; j += blockDim.x) {
      const bool is_self = (s == t_s && j == t_l);
      const double3 xj = px[j];
      const double mp = pm ? double(pm[j]) : double(first_mass);
      const double hs = ps ? double(ps[j]) : double(first_soft);
      const double hij = fmax(ht, hs);
      const double hij2 = hij * hij;
      const double hij_inv = 1.0 / hij;
      const double hij_inv3 = hij_inv * hij_inv * hij_inv;

      const double dxb = minimum_image_double(xj.x - xt.x, box);
      const double dyb = minimum_image_double(xj.y - xt.y, box);
      const double dzb = minimum_image_double(xj.z - xt.z, box);

      for (int nx = -1; nx <= 1; ++nx) {
        const double rx = dxb + double(nx) * box;
        const double rx2 = rx * rx;
        if (rx2 >= r_cut2) continue;
        for (int ny = -1; ny <= 1; ++ny) {
          const double ry = dyb + double(ny) * box;
          const double rxy2 = rx2 + ry * ry;
          if (rxy2 >= r_cut2) continue;
          for (int nz = -1; nz <= 1; ++nz) {
            const bool zero_sh = (nx == 0 && ny == 0 && nz == 0);
            if (zero_sh && is_self) continue;
            const double rz = dzb + double(nz) * box;
            const double R2 = rxy2 + rz * rz;
            if (R2 <= 0.0 || R2 >= r_cut2) continue;
            const double R = sqrt(R2);
            const double u = alpha * R;
            const double exp_u2 = exp(-u * u);

            double k_bare;
            if (zero_sh && R2 < hij2) {
              k_bare = hij_inv3 * wendland_c2_force_d(R * hij_inv);
            } else {
              k_bare = 1.0 / (R2 * R);
            }

            if (zero_sh && R2 <= short_cut2) {
              const double us = 0.5 * R * inv_rs;
              const double corr_s = erfc(us) + two_sqrt_pi * us * exp(-us * us);
              const double fs_val = mp * k_bare * corr_s;
              sx += rx * fs_val;
              sy += ry * fs_val;
              sz += rz * fs_val;
            }

            double f_ew;
            if (zero_sh && R2 < hij2) {
              double tail;
              if (u < 1e-3) {
                const double u2 = u * u;
                tail = taylor_pref * (1.0 / 3.0 - u2 / 5.0 + u2 * u2 / 14.0 - u2 * u2 * u2 / 54.0);
              } else {
                tail = (erf(u) - two_sqrt_pi * u * exp_u2) / (R2 * R);
              }
              f_ew = k_bare - tail;
            } else {
              f_ew = k_bare * (erfc(u) + two_sqrt_pi * u * exp_u2);
            }
            const double ft = mp * f_ew;
            fx += rx * ft;
            fy += ry * ft;
            fz += rz * ft;
          }
        }
      }
    }
  }

  for (int m = threadIdx.x; m < n_modes; m += blockDim.x) {
    const EwaldMode md = modes[m];
    const double ph = md.kx * xt.x + md.ky * xt.y + md.kz * xt.z;
    double sv, cv;
    sincos(ph, &sv, &cv);
    const double fac = md.weight * (cv * S_k[m] - sv * C_k[m]);
    fx += md.kx * fac;
    fy += md.ky * fac;
    fz += md.kz * fac;
  }

  for (int off = 16; off > 0; off >>= 1) {
    sx += __shfl_down_sync(0xffffffff, sx, off);
    sy += __shfl_down_sync(0xffffffff, sy, off);
    sz += __shfl_down_sync(0xffffffff, sz, off);
    fx += __shfl_down_sync(0xffffffff, fx, off);
    fy += __shfl_down_sync(0xffffffff, fy, off);
    fz += __shfl_down_sync(0xffffffff, fz, off);
  }
  __shared__ double sh_s[8][3], sh_f[8][3];
  const int wid = threadIdx.x >> 5;
  if (!(threadIdx.x & 31)) {
    sh_s[wid][0] = sx; sh_s[wid][1] = sy; sh_s[wid][2] = sz;
    sh_f[wid][0] = fx; sh_f[wid][1] = fy; sh_f[wid][2] = fz;
  }
  __syncthreads();
  if (!threadIdx.x) {
    double asx = 0.0, asy = 0.0, asz = 0.0, afx = 0.0, afy = 0.0, afz = 0.0;
    const int nw = blockDim.x >> 5;
    for (int w = 0; w < nw; ++w) {
      asx += sh_s[w][0]; asy += sh_s[w][1]; asz += sh_s[w][2];
      afx += sh_f[w][0]; afy += sh_f[w][1]; afz += sh_f[w][2];
    }
    out_short[t] = {asx * grav_G, asy * grav_G, asz * grav_G};
    out_total[t] = {afx * grav_G, afy * grav_G, afz * grav_G};
  }
}

__global__ void exact_direct_isolated_kernel(
    AllSlabsData slabs, float first_mass, float first_soft,
    const double3* tgt_x, const float* tgt_h, const int* tgt_slab, const int* tgt_lid,
    int n_tgt, Physics p, double3* out_total) {
  const int t = blockIdx.x;
  if (t >= n_tgt) return;
  const double3 xt = tgt_x[t];
  const double ht = double(tgt_h[t]);
  const int t_s = tgt_slab[t];
  const int t_l = tgt_lid[t];
  double fx = 0.0, fy = 0.0, fz = 0.0;
  for (int s = 0; s < num_slabs; ++s) {
    const int ns = slabs.ns[s];
    const double3* px = slabs.x[s];
    const float* pm = slabs.mass[s];
    const float* ps = slabs.soft[s];
    for (int j = threadIdx.x; j < ns; j += blockDim.x) {
      if (s == t_s && j == t_l) continue;
      const double3 xj = px[j];
      const double rx = xj.x - xt.x, ry = xj.y - xt.y, rz = xj.z - xt.z;
      const double R2 = rx * rx + ry * ry + rz * rz;
      if (R2 <= 0.0) continue;
      const double mp = pm ? double(pm[j]) : double(first_mass);
      const double hs = ps ? double(ps[j]) : double(first_soft);
      const double hij = fmax(ht, hs);
      const double hij2 = hij * hij;
      double k_bare;
      if (R2 < hij2) {
        const double hinv = 1.0 / hij;
        const double R = sqrt(R2);
        k_bare = hinv * hinv * hinv * wendland_c2_force_d(R * hinv);
      } else {
        const double R = sqrt(R2);
        k_bare = 1.0 / (R2 * R);
      }
      const double fac = mp * k_bare;
      fx += rx * fac; fy += ry * fac; fz += rz * fac;
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    fx += __shfl_down_sync(0xffffffff, fx, off);
    fy += __shfl_down_sync(0xffffffff, fy, off);
    fz += __shfl_down_sync(0xffffffff, fz, off);
  }
  __shared__ double sh_f[8][3];
  const int wid = threadIdx.x >> 5;
  if (!(threadIdx.x & 31)) {
    sh_f[wid][0] = fx; sh_f[wid][1] = fy; sh_f[wid][2] = fz;
  }
  __syncthreads();
  if (!threadIdx.x) {
    double afx = 0.0, afy = 0.0, afz = 0.0;
    const int nw = blockDim.x >> 5;
    for (int w = 0; w < nw; ++w) {
      afx += sh_f[w][0]; afy += sh_f[w][1]; afz += sh_f[w][2];
    }
    if (p.ext_M > 0.f) {
      const double dx = double(p.ext_cx) - xt.x;
      const double dy = double(p.ext_cy) - xt.y;
      const double dz = double(p.ext_cz) - xt.z;
      const double r2e = dx * dx + dy * dy + dz * dz + double(p.ext_a) * double(p.ext_a);
      const double re = sqrt(r2e);
      const double ae = double(p.ext_M) / (r2e * re);
      afx += dx * ae; afy += dy * ae; afz += dz * ae;
    }
    out_total[t] = {afx * double(p.G), afy * double(p.G), afz * double(p.G)};
  }
}

__global__ void lightcone_detect_kernel(
    const double3* d_x, const float3* d_v, const uint32_t* d_orig_idx,
    const float* d_mass, float first_mass, int ns,
    double drift_fac, double a_prev, double a_cur,
    double chi_0, double chi_1, double lc_r_max,
    double3 obs_perm, const double3* shifts, int n_shifts, int slab_axis,
    LightconeHit* out_hits, unsigned int* out_count, unsigned int hit_cap) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= ns) return;
  const double3 x0 = d_x[k];
  const float3 v = d_v[k];
  const uint64_t oid = uint64_t(d_orig_idx[k]);
  const float mp = d_mass ? d_mass[k] : first_mass;
  const double denom_shift = chi_0 - chi_1;

  for (int si = 0; si < n_shifts; ++si) {
    const double3 sh = shifts[si];
    const double rx0 = x0.x + sh.x - obs_perm.x;
    const double ry0 = x0.y + sh.y - obs_perm.y;
    const double rz0 = x0.z + sh.z - obs_perm.z;
    const double d0 = sqrt(rx0 * rx0 + ry0 * ry0 + rz0 * rz0);
    if (d0 >= chi_0) continue;

    const double drx = drift_fac * double(v.x);
    const double dry = drift_fac * double(v.y);
    const double drz = drift_fac * double(v.z);
    const double rx1 = rx0 + drx;
    const double ry1 = ry0 + dry;
    const double rz1 = rz0 + drz;
    const double d1 = sqrt(rx1 * rx1 + ry1 * ry1 + rz1 * rz1);
    if (d1 < chi_1) continue;

    const double denom = (d1 - d0) + denom_shift;
    double tau = (denom > 0.0) ? ((chi_0 - d0) / denom) : 0.5;
    tau = fmin(1.0, fmax(0.0, tau));
    const double cx = rx0 + tau * drx;
    const double cy = ry0 + tau * dry;
    const double cz = rz0 + tau * drz;
    const double rc = sqrt(cx * cx + cy * cy + cz * cz);
    if (rc > lc_r_max) continue;

    const unsigned int slot = atomicAdd(out_count, 1u);
    if (slot < hit_cap) {
      const double a_cross = a_prev + tau * (a_cur - a_prev);
      const float inv_a = float(1.0 / a_cross);
      const double3 cout_v = permute_out(double3{cx, cy, cz}, slab_axis);
      const float3 vout_v = permute_out(float3{v.x * inv_a, v.y * inv_a, v.z * inv_a}, slab_axis);
      LightconeHit hit{};
      hit.x[0] = cout_v.x; hit.x[1] = cout_v.y; hit.x[2] = cout_v.z;
      hit.orig_idx = oid;
      hit.v[0] = vout_v.x; hit.v[1] = vout_v.y; hit.v[2] = vout_v.z;
      hit.a_cross = float(a_cross);
      hit.mass = mp;
      hit.r_comoving = float(rc);
      out_hits[slot] = hit;
    }
  }
}

__global__ void pack_migrants_kernel(
    const double3* d_x, const float3* d_v, const float3* d_mesh_acc,
    const float* d_old_acc, const unsigned char* d_bins, const uint32_t* d_orig_idx,
    const float* d_mass, float first_mass, const float* d_soft, float first_soft,
    const int* mig_ids, int m_cnt, double inv_wx, DevMigrant* out) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= m_cnt) return;
  const int k = mig_ids[j];
  const double3 px = d_x[k];
  const int dest = min(max(int(px.x * inv_wx), 0), num_slabs - 1);
  DevMigrant dm{};
  dm.x = px;
  dm.v = d_v[k];
  dm.mesh_acc = d_mesh_acc[k];
  dm.old_acc = d_old_acc[k];
  dm.mass = d_mass ? d_mass[k] : first_mass;
  dm.soft = d_soft ? d_soft[k] : first_soft;
  dm.orig_idx = d_orig_idx[k];
  dm.dst_slab = dest;
  dm.bin = d_bins[k];
  out[j] = dm;
}

__global__ void remove_migrants_backwards_kernel(
    double3* d_x, float3* d_v, float3* d_mesh_acc, float* d_old_acc,
    unsigned char* d_bins, uint32_t* d_orig_idx, float* d_mass, float* d_soft,
    const int* mig_ids, int m_cnt, int ns) {
  int cur_ns = ns;
  for (int j = m_cnt - 1; j >= 0; --j) {
    const int idx = mig_ids[j];
    const int back = --cur_ns;
    if (idx != back) {
      d_x[idx] = d_x[back];
      d_v[idx] = d_v[back];
      d_mesh_acc[idx] = d_mesh_acc[back];
      d_old_acc[idx] = d_old_acc[back];
      d_bins[idx] = d_bins[back];
      d_orig_idx[idx] = d_orig_idx[back];
      if (d_mass) d_mass[idx] = d_mass[back];
      if (d_soft) d_soft[idx] = d_soft[back];
    }
  }
}

__global__ void append_migrants_kernel(
    double3* d_x, float3* d_v, float3* d_mesh_acc, float* d_old_acc,
    unsigned char* d_bins, uint32_t* d_orig_idx, float* d_mass, float* d_soft,
    const DevMigrant* in_mig, int base_ns, int in_cnt) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= in_cnt) return;
  const int k = base_ns + j;
  const DevMigrant m = in_mig[j];
  d_x[k] = m.x;
  d_v[k] = m.v;
  d_mesh_acc[k] = m.mesh_acc;
  d_old_acc[k] = m.old_acc;
  d_bins[k] = m.bin;
  d_orig_idx[k] = m.orig_idx;
  if (d_mass) d_mass[k] = m.mass;
  if (d_soft) d_soft[k] = m.soft;
}

__global__ void scan_slab_active_v2_kernel(
    const unsigned char* d_bins, const float3* d_v, int ns, Tick ti,
    bool force_all, bool need_vel_sum, bool track_max_v, ScanPartial* partials) {
  uint64_t act = 0;
  Tick min_nx = timeline;
  double v2 = 0.0;
  float mv = 0.f;
  const int stride = gridDim.x * blockDim.x;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < ns; k += stride) {
    const unsigned char b = d_bins[k];
    if (force_all || is_active(ti, b)) {
      ++act;
    } else {
      const Tick w = bin_size(b);
      const Tick nx = (ti & ~(w - 1ULL)) + w;
      if (nx < min_nx) min_nx = nx;
    }
    if (need_vel_sum || track_max_v) {
      const float3 vi = d_v[k];
      if (need_vel_sum) v2 += double(vi.x) * vi.x + double(vi.y) * vi.y + double(vi.z) * vi.z;
      if (track_max_v) mv = fmaxf(mv, fmaxf(fabsf(vi.x), fmaxf(fabsf(vi.y), fabsf(vi.z))));
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    act += __shfl_down_sync(0xffffffff, act, off);
    const Tick t = __shfl_down_sync(0xffffffff, min_nx, off);
    if (t < min_nx) min_nx = t;
    v2 += __shfl_down_sync(0xffffffff, v2, off);
    mv = fmaxf(mv, __shfl_down_sync(0xffffffff, mv, off));
  }
  __shared__ uint64_t w_act[scan_team / 32];
  __shared__ Tick w_nx[scan_team / 32];
  __shared__ double w_v2[scan_team / 32];
  __shared__ float w_mv[scan_team / 32];
  const int warp = threadIdx.x >> 5;
  if (!(threadIdx.x & 31)) {
    w_act[warp] = act;
    w_nx[warp] = min_nx;
    w_v2[warp] = v2;
    w_mv[warp] = mv;
  }
  __syncthreads();
  if (!threadIdx.x) {
    uint64_t ta = 0;
    Tick tnx = timeline;
    double tv2 = 0.0;
    float tmv = 0.f;
    for (int w = 0; w < scan_team / 32; ++w) {
      ta += w_act[w];
      if (w_nx[w] < tnx) tnx = w_nx[w];
      tv2 += w_v2[w];
      tmv = fmaxf(tmv, w_mv[w]);
    }
    partials[blockIdx.x] = {ta, tnx, tv2, tmv};
  }
}

template <class T>
struct DevPtr {
  T* p = nullptr;
  size_t bytes = 0;
  void alloc(size_t count, size_t& tracker) {
    free(tracker);
    if (!count) return;
    bytes = count * sizeof(T);
    check(cudaMalloc(&p, bytes));
    tracker += bytes;
  }
  void free(size_t& tracker) {
    if (p) {
      cudaFree(p);
      tracker -= bytes;
      p = nullptr;
      bytes = 0;
    }
  }
};

struct DevSlab {
  DevPtr<double3> d_x;
  DevPtr<float3> d_v, d_mesh_acc;
  DevPtr<float> d_old_acc, d_mass, d_soft;
  DevPtr<unsigned char> d_bins;
  DevPtr<uint32_t> d_orig_idx;
  int global_id = -1;
  size_t ns = 0, cap = 0;

  void alloc(size_t new_cap, bool uniform_mass, bool uniform_soft, size_t& tracker) {
    cap = new_cap;
    d_x.alloc(cap, tracker);
    d_v.alloc(cap, tracker);
    d_mesh_acc.alloc(cap, tracker);
    d_old_acc.alloc(cap, tracker);
    d_bins.alloc(cap, tracker);
    d_orig_idx.alloc(cap, tracker);
    if (!uniform_mass) d_mass.alloc(cap, tracker);
    if (!uniform_soft) d_soft.alloc(cap, tracker);
  }
  void free_all(size_t& tracker) {
    d_x.free(tracker);
    d_v.free(tracker);
    d_mesh_acc.free(tracker);
    d_old_acc.free(tracker);
    d_bins.free(tracker);
    d_orig_idx.free(tracker);
    d_mass.free(tracker);
    d_soft.free(tracker);
    cap = 0;
  }
};

struct SlabTreeSlot {
  DevPtr<float4> packed_sources;
  DevPtr<float> packed_soft;
  DevPtr<Geometry> geometry;
  DevPtr<ForceNode> force_nodes;
  DevPtr<NodeMeta> node_meta;
  DevPtr<CompactMultipole> compact_m;
  DevPtr<float4> node_vel_max;
  DevPtr<int> escape, first_child, order, node_count, levels;
  DevPtr<uint8_t> home_cell;
  size_t part_cap = 0, node_cap = 0;
  unsigned int nodes = 0;

  void alloc(size_t pcap, size_t ncap, bool uniform_soft, bool need_node_vel, bool need_split_nodes, size_t& tracker) {
    part_cap = pcap;
    node_cap = ncap;
    packed_sources.alloc(pcap, tracker);
    if (!uniform_soft) packed_soft.alloc(pcap, tracker);
    geometry.alloc(ncap, tracker);
    force_nodes.alloc(ncap, tracker);
    if (need_split_nodes) {
      node_meta.alloc(ncap, tracker);
      compact_m.alloc(ncap, tracker);
    }
    if (need_node_vel) node_vel_max.alloc(ncap, tracker);
    escape.alloc(ncap, tracker);
    first_child.alloc(ncap, tracker);
    order.alloc(pcap, tracker);
    home_cell.alloc(pcap, tracker);
    node_count.alloc(1, tracker);
    levels.alloc(tree_levels + 2, tracker);
    nodes = 0;
  }
  void free_all(size_t& tracker) {
    packed_sources.free(tracker);
    packed_soft.free(tracker);
    geometry.free(tracker);
    force_nodes.free(tracker);
    node_meta.free(tracker);
    compact_m.free(tracker);
    node_vel_max.free(tracker);
    escape.free(tracker);
    first_child.free(tracker);
    order.free(tracker);
    home_cell.free(tracker);
    node_count.free(tracker);
    levels.free(tracker);
    part_cap = 0;
    node_cap = 0;
    nodes = 0;
  }
  SlabTreeView view() const {
    return {geometry.p, force_nodes.p, escape.p, first_child.p, packed_sources.p, packed_soft.p, home_cell.p, order.p};
  }
  SplitTravTreeView split_view() const {
    return {geometry.p, node_meta.p, compact_m.p};
  }
  SlabForceInput full_view() const {
    return {view(), split_view()};
  }
};

struct GhostTreeView {
  float4* packed_sources = nullptr;
  float* packed_soft = nullptr;
  Geometry* geometry = nullptr;
  ForceNode* force_nodes = nullptr;
  NodeMeta* node_meta = nullptr;
  CompactMultipole* compact_m = nullptr;
  int* escape = nullptr;
  int* first_child = nullptr;
  SlabTreeView view() const {
    return {geometry, force_nodes, escape, first_child, packed_sources, packed_soft, nullptr, nullptr};
  }
  SplitTravTreeView split_view() const {
    return {geometry, node_meta, compact_m};
  }
  SlabForceInput full_view() const {
    return {view(), split_view()};
  }
};

struct HostSlabStaging {
  std::vector<double3> x;
  std::vector<float3> v;
  std::vector<float> mass;
  std::vector<float> soft;
  std::vector<uint32_t> orig_idx;
};

struct PinnedHostBuf {
  void* p = nullptr;
  size_t bytes = 0;
  void alloc(size_t b, unsigned int flags = cudaHostAllocDefault) {
    if (bytes >= b) return;
    if (p) check(cudaFreeHost(p));
    p = nullptr;
    size_t target = b;
    if (b > (64ULL << 20)) {
      const size_t grown = std::max<size_t>(b + (b >> 2), b + size_t(64ULL << 20));
      constexpr size_t align = 2ULL << 20;
      target = ((grown + align - 1ULL) / align) * align;
    }
    bytes = target;
    check(cudaHostAlloc(&p, bytes, flags));
  }
  void free() {
    if (p) { cudaFreeHost(p); p = nullptr; bytes = 0; }
  }
  ~PinnedHostBuf() { free(); }
};

struct GpuCtx {
  static constexpr int bounce_slots = 4;
  static constexpr size_t bounce_chunk_bytes = 64ULL << 20;

  int dev = 0;
  int slabs_per_gpu = 0;
  bool need_two_ghosts = false;
  bool need_soft = false;
  bool need_split_nodes = false;
  size_t device_bytes = 0;
  cudaStream_t compute_stream{};
  cudaStream_t copy_stream{};
  cudaStream_t d2h_left_stream{};
  cudaStream_t d2h_right_stream{};
  cudaEvent_t ev_left_start{}, ev_left_ready{}, ev_s0_done{}, ev_right_start{}, ev_right_ready{};
  cudaEvent_t ev_force_start{}, ev_force_end{}, ev_prof0{}, ev_prof1{};
  cudaEvent_t ev_d2h_left[bounce_slots]{}, ev_d2h_right[bounce_slots]{}, ev_h2d_done[bounce_slots]{};

  std::vector<DevSlab> slabs;
  std::vector<SlabTreeSlot> trees;

  // Phase pool (aliased Build Phase vs Force Phase)
  DevPtr<unsigned char> d_phase_pool;
  size_t pool_local_cap = 0, pool_local_nodes = 0;
  size_t pool_ghost_cap = 0, pool_ghost_nodes = 0;
  size_t sort_bytes = 0, select_bytes = 0;

  // Build views
  Key *d_key_tmp = nullptr, *d_keys_sorted = nullptr;
  int* d_ord_tmp = nullptr;
  double3* d_node_com = nullptr;
  unsigned char* d_sort_storage = nullptr;

  // Force views
  GhostTreeView ghost, ghost_right;
  float4* d_force = nullptr;
  unsigned char* d_flags = nullptr;
  int* d_active_ids = nullptr;
  float* d_pm_pot = nullptr;

  // Persistent small buffers
  DevPtr<unsigned char> d_select_storage;
  DevPtr<int> d_active_count, d_tree_invalid;
  DevPtr<Meta> d_stats;
  DevPtr<double> d_kick_finish, d_kick_begin;
  DevPtr<double> d_pm_density;
  DevPtr<double4> d_pm_field;
  DevPtr<ScanPartial> d_scan_tmp;
  DevPtr<DevMigrant> d_mig_out, d_mig_in;
  DevPtr<unsigned char> d_tail_flags;
  DevPtr<int> d_tail_keep, d_tail_count;
  size_t mig_out_cap = 0, mig_in_cap = 0, tail_cap = 0;

  // Priority 3: 3-kernel zero-spill home-cell-aligned hybrid split workspace
  DevPtr<SplitWarpSlice> d_split_slices;
  DevPtr<SplitWarpMeta> d_split_meta;
  DevPtr<int> d_split_total_warps, d_split_head_p2p, d_split_head_m2l;
  DevPtr<int> d_split_p2p_warps, d_split_m2l_warps, d_split_counts;
  DevPtr<SplitP2PChunk> d_split_p2p;
  DevPtr<SplitM2LChunk> d_split_m2l;
  size_t split_slice_cap = 0, split_meta_cap = 0, split_p2p_cap = 0, split_m2l_cap = 0;
  int split_chunk_scale = 0;

  void free_split_bufs() {
    check(cudaSetDevice(dev));
    if (split_slice_cap > 0) {
      d_split_slices.free(device_bytes);
      d_split_total_warps.free(device_bytes);
      split_slice_cap = 0;
    }
    if (split_meta_cap > 0) {
      d_split_meta.free(device_bytes);
      d_split_head_p2p.free(device_bytes);
      d_split_head_m2l.free(device_bytes);
      d_split_p2p_warps.free(device_bytes);
      d_split_m2l_warps.free(device_bytes);
      d_split_counts.free(device_bytes);
      d_split_p2p.free(device_bytes);
      d_split_m2l.free(device_bytes);
      split_meta_cap = split_p2p_cap = split_m2l_cap = 0;
      split_chunk_scale = 0;
    }
  }

  void ensure_split_slices(size_t need_warps) {
    if (need_warps <= split_slice_cap) return;
    check(cudaSetDevice(dev));
    check(cudaStreamSynchronize(compute_stream));
    if (split_slice_cap > 0) {
      d_split_slices.free(device_bytes);
      d_split_total_warps.free(device_bytes);
    }
    split_slice_cap = need_warps + 256;
    d_split_slices.alloc(split_slice_cap, device_bytes);
    d_split_total_warps.alloc(1, device_bytes);
  }

  void ensure_split_bufs(size_t batch_warps, int scale = 1) {
    const size_t req_p2p = batch_warps * 32ULL * size_t(scale);
    const size_t req_m2l = batch_warps * 28ULL * size_t(scale);
    if (batch_warps <= split_meta_cap && req_p2p <= split_p2p_cap && req_m2l <= split_m2l_cap) {
      return;
    }
    check(cudaSetDevice(dev));
    check(cudaStreamSynchronize(compute_stream));
    if (split_meta_cap > 0) {
      d_split_meta.free(device_bytes);
      d_split_head_p2p.free(device_bytes);
      d_split_head_m2l.free(device_bytes);
      d_split_p2p_warps.free(device_bytes);
      d_split_m2l_warps.free(device_bytes);
      d_split_counts.free(device_bytes);
      d_split_p2p.free(device_bytes);
      d_split_m2l.free(device_bytes);
    }
    split_meta_cap = batch_warps;
    split_p2p_cap = req_p2p;
    split_m2l_cap = req_m2l;
    split_chunk_scale = scale;
    d_split_meta.alloc(split_meta_cap, device_bytes);
    d_split_head_p2p.alloc(split_meta_cap, device_bytes);
    d_split_head_m2l.alloc(split_meta_cap, device_bytes);
    d_split_p2p_warps.alloc(split_meta_cap, device_bytes);
    d_split_m2l_warps.alloc(split_meta_cap, device_bytes);
    d_split_counts.alloc(5, device_bytes);
    d_split_p2p.alloc(split_p2p_cap, device_bytes);
    d_split_m2l.alloc(split_m2l_cap, device_bytes);
  }

  PinnedHostBuf h_bounce;
  std::vector<ScanPartial> h_scan;

  void ensure_select_cap(size_t need_cap) {
    size_t selb = 0;
    cub::CountingInputIterator<int> seq(0);
    check(cub::DeviceSelect::Flagged(
        nullptr, selb, seq, static_cast<unsigned char*>(nullptr),
        static_cast<int*>(nullptr), d_active_count.p, int(need_cap)));
    if (selb > select_bytes) {
      select_bytes = selb;
      d_select_storage.alloc(std::max<size_t>(select_bytes, 1), device_bytes);
    }
  }

  void ensure_phase_pool(size_t loc_cap, size_t loc_nodes, size_t gh_cap, size_t gh_nodes) {
    const size_t new_loc_c = std::max(pool_local_cap, loc_cap);
    const size_t new_loc_n = std::max(pool_local_nodes, loc_nodes);
    const size_t new_gh_c = std::max(pool_ghost_cap, gh_cap);
    const size_t new_gh_n = std::max(pool_ghost_nodes, gh_nodes);

    if (new_loc_c <= pool_local_cap && new_loc_n <= pool_local_nodes &&
        new_gh_c <= pool_ghost_cap && new_gh_n <= pool_ghost_nodes && d_phase_pool.p) {
      return;
    }

    d_phase_pool.free(device_bytes);  // Free first to avoid 2x spike
    pool_local_cap = new_loc_c;
    pool_local_nodes = new_loc_n;
    pool_ghost_cap = std::max<size_t>(new_gh_c, 1);
    pool_ghost_nodes = std::max<size_t>(new_gh_n, size_t(slab_roots_count));

    ensure_select_cap(pool_local_cap);
    size_t sb = 0;
    check(cub::DeviceRadixSort::SortPairs(
        nullptr, sb, static_cast<Key*>(nullptr), static_cast<Key*>(nullptr),
        static_cast<int*>(nullptr), static_cast<int*>(nullptr), int(pool_local_cap), 0, 59));
    sort_bytes = sb;

    auto align_up = [](size_t v) { return (v + 255ULL) & ~255ULL; };
    size_t build_bytes = 0;
    auto push_b = [&](size_t bytes) {
      const size_t off = build_bytes;
      build_bytes += align_up(std::max<size_t>(bytes, 1));
      return off;
    };
    const size_t off_key_tmp = push_b(pool_local_cap * sizeof(Key));
    const size_t off_keys_sorted = push_b(pool_local_cap * sizeof(Key));
    const size_t off_ord_tmp = push_b(pool_local_cap * sizeof(int));
    const size_t off_node_com = push_b(pool_local_nodes * sizeof(double3));
    const size_t off_sort_storage = push_b(sort_bytes);

    size_t force_bytes = 0;
    auto push_f = [&](size_t bytes) {
      const size_t off = force_bytes;
      force_bytes += align_up(std::max<size_t>(bytes, 1));
      return off;
    };
    const size_t off_gh_packed = push_f(pool_ghost_cap * sizeof(float4));
    const size_t off_gh_geom = push_f(pool_ghost_nodes * sizeof(Geometry));
    size_t off_gh_fn = 0, off_gh_esc = 0, off_gh_fc = 0;
    size_t off_gh_nmeta = 0, off_gh_cm = 0;
    if (need_split_nodes) {
      off_gh_nmeta = push_f(pool_ghost_nodes * sizeof(NodeMeta));
      off_gh_cm = push_f(pool_ghost_nodes * sizeof(CompactMultipole));
    } else {
      off_gh_fn = push_f(pool_ghost_nodes * sizeof(ForceNode));
      off_gh_esc = push_f(pool_ghost_nodes * sizeof(int));
      off_gh_fc = push_f(pool_ghost_nodes * sizeof(int));
    }
    size_t off_gh_soft = 0, off_ghr_soft = 0;
    if (need_soft) off_gh_soft = push_f(pool_ghost_cap * sizeof(float));
    size_t off_ghr_packed = 0, off_ghr_geom = 0, off_ghr_fn = 0, off_ghr_esc = 0, off_ghr_fc = 0;
    size_t off_ghr_nmeta = 0, off_ghr_cm = 0;
    if (need_two_ghosts) {
      off_ghr_packed = push_f(pool_ghost_cap * sizeof(float4));
      off_ghr_geom = push_f(pool_ghost_nodes * sizeof(Geometry));
      if (need_split_nodes) {
        off_ghr_nmeta = push_f(pool_ghost_nodes * sizeof(NodeMeta));
        off_ghr_cm = push_f(pool_ghost_nodes * sizeof(CompactMultipole));
      } else {
        off_ghr_fn = push_f(pool_ghost_nodes * sizeof(ForceNode));
        off_ghr_esc = push_f(pool_ghost_nodes * sizeof(int));
        off_ghr_fc = push_f(pool_ghost_nodes * sizeof(int));
      }
      if (need_soft) off_ghr_soft = push_f(pool_ghost_cap * sizeof(float));
    }
    const size_t off_force = push_f(pool_local_cap * sizeof(float4));
    // Alias d_flags+d_active_ids (only used when !is_final) with d_pm_pot (only used when is_final)
    const size_t off_act_or_pot = force_bytes;
    const size_t act_pair_bytes = align_up(pool_local_cap * sizeof(unsigned char)) +
                                  align_up(pool_local_cap * sizeof(int));
    const size_t pot_bytes = align_up(pool_local_cap * sizeof(float));
    force_bytes += std::max(act_pair_bytes, pot_bytes);

    d_phase_pool.alloc(std::max(build_bytes, force_bytes), device_bytes);
    auto at = [&](size_t off) -> unsigned char* { return d_phase_pool.p + off; };
    d_key_tmp = reinterpret_cast<Key*>(at(off_key_tmp));
    d_keys_sorted = reinterpret_cast<Key*>(at(off_keys_sorted));
    d_ord_tmp = reinterpret_cast<int*>(at(off_ord_tmp));
    d_node_com = reinterpret_cast<double3*>(at(off_node_com));
    d_sort_storage = at(off_sort_storage);

    ghost.packed_sources = reinterpret_cast<float4*>(at(off_gh_packed));
    ghost.packed_soft = need_soft ? reinterpret_cast<float*>(at(off_gh_soft)) : nullptr;
    ghost.geometry = reinterpret_cast<Geometry*>(at(off_gh_geom));
    ghost.force_nodes = need_split_nodes ? nullptr : reinterpret_cast<ForceNode*>(at(off_gh_fn));
    ghost.escape = need_split_nodes ? nullptr : reinterpret_cast<int*>(at(off_gh_esc));
    ghost.first_child = need_split_nodes ? nullptr : reinterpret_cast<int*>(at(off_gh_fc));
    ghost.node_meta = need_split_nodes ? reinterpret_cast<NodeMeta*>(at(off_gh_nmeta)) : nullptr;
    ghost.compact_m = need_split_nodes ? reinterpret_cast<CompactMultipole*>(at(off_gh_cm)) : nullptr;
    if (need_two_ghosts) {
      ghost_right.packed_sources = reinterpret_cast<float4*>(at(off_ghr_packed));
      ghost_right.packed_soft = need_soft ? reinterpret_cast<float*>(at(off_ghr_soft)) : nullptr;
      ghost_right.geometry = reinterpret_cast<Geometry*>(at(off_ghr_geom));
      ghost_right.force_nodes = need_split_nodes ? nullptr : reinterpret_cast<ForceNode*>(at(off_ghr_fn));
      ghost_right.escape = need_split_nodes ? nullptr : reinterpret_cast<int*>(at(off_ghr_esc));
      ghost_right.first_child = need_split_nodes ? nullptr : reinterpret_cast<int*>(at(off_ghr_fc));
      ghost_right.node_meta = need_split_nodes ? reinterpret_cast<NodeMeta*>(at(off_ghr_nmeta)) : nullptr;
      ghost_right.compact_m = need_split_nodes ? reinterpret_cast<CompactMultipole*>(at(off_ghr_cm)) : nullptr;
    } else {
      ghost_right = GhostTreeView{};
    }
    d_force = reinterpret_cast<float4*>(at(off_force));

    d_flags = at(off_act_or_pot);
    d_active_ids = reinterpret_cast<int*>(at(off_act_or_pot + align_up(pool_local_cap * sizeof(unsigned char))));
    d_pm_pot = reinterpret_cast<float*>(at(off_act_or_pot));
  }

  void ensure_mig_out(size_t need) {
    if (need <= mig_out_cap) return;
    mig_out_cap = std::max<size_t>(need, std::max<size_t>(mig_out_cap * 2, 262144));
    d_mig_out.alloc(mig_out_cap, device_bytes);
  }
  void ensure_mig_in(size_t need) {
    if (need <= mig_in_cap) return;
    mig_in_cap = std::max<size_t>(need, std::max<size_t>(mig_in_cap * 2, 262144));
    d_mig_in.alloc(mig_in_cap, device_bytes);
  }
  void ensure_tail_cap(size_t need) {
    if (!d_tail_count.p) d_tail_count.alloc(1, device_bytes);
    if (need > tail_cap) {
      tail_cap = std::max<size_t>(need, std::max<size_t>(tail_cap * 2, 262144));
      d_tail_flags.alloc(tail_cap, device_bytes);
      d_tail_keep.alloc(tail_cap, device_bytes);
    }
    ensure_select_cap(need);
  }

  void cleanup() {
    check(cudaSetDevice(dev));
    for (auto& sl : slabs) sl.free_all(device_bytes);
    for (auto& tr : trees) tr.free_all(device_bytes);
    d_phase_pool.free(device_bytes);
    d_select_storage.free(device_bytes);
    d_active_count.free(device_bytes);
    d_tree_invalid.free(device_bytes);
    d_stats.free(device_bytes);
    d_kick_finish.free(device_bytes);
    d_kick_begin.free(device_bytes);
    d_pm_density.free(device_bytes);
    d_pm_field.free(device_bytes);
    d_scan_tmp.free(device_bytes);
    d_mig_out.free(device_bytes);
    d_mig_in.free(device_bytes);
    d_tail_flags.free(device_bytes);
    d_tail_keep.free(device_bytes);
    d_tail_count.free(device_bytes);
    h_bounce.free();
    if (ev_left_start) { cudaEventDestroy(ev_left_start); ev_left_start = nullptr; }
    if (ev_left_ready) { cudaEventDestroy(ev_left_ready); ev_left_ready = nullptr; }
    if (ev_s0_done) { cudaEventDestroy(ev_s0_done); ev_s0_done = nullptr; }
    if (ev_right_start) { cudaEventDestroy(ev_right_start); ev_right_start = nullptr; }
    if (ev_right_ready) { cudaEventDestroy(ev_right_ready); ev_right_ready = nullptr; }
    if (ev_force_start) { cudaEventDestroy(ev_force_start); ev_force_start = nullptr; }
    if (ev_force_end) { cudaEventDestroy(ev_force_end); ev_force_end = nullptr; }
    if (ev_prof0) { cudaEventDestroy(ev_prof0); ev_prof0 = nullptr; }
    if (ev_prof1) { cudaEventDestroy(ev_prof1); ev_prof1 = nullptr; }
    for (int i = 0; i < bounce_slots; ++i) {
      if (ev_d2h_left[i]) { cudaEventDestroy(ev_d2h_left[i]); ev_d2h_left[i] = nullptr; }
      if (ev_d2h_right[i]) { cudaEventDestroy(ev_d2h_right[i]); ev_d2h_right[i] = nullptr; }
      if (ev_h2d_done[i]) { cudaEventDestroy(ev_h2d_done[i]); ev_h2d_done[i] = nullptr; }
    }
    if (compute_stream) { cudaStreamDestroy(compute_stream); compute_stream = nullptr; }
    if (copy_stream) { cudaStreamDestroy(copy_stream); copy_stream = nullptr; }
    if (d2h_left_stream) { cudaStreamDestroy(d2h_left_stream); d2h_left_stream = nullptr; }
    free_split_bufs();
    if (d2h_right_stream) { cudaStreamDestroy(d2h_right_stream); d2h_right_stream = nullptr; }
  }
};

double wall() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
template <class T>
void read_bin(FILE* file, T* data, size_t n) {
  if (fread(data, sizeof(T), n, file) != n) throw std::runtime_error("Incomplete particle input");
}
template <class T>
void write_bin(FILE* file, const T* data, size_t n) {
  const auto* ptr = reinterpret_cast<const char*>(data);
  size_t rem = n * sizeof(T);
  constexpr size_t chunk = 64ULL << 20;
  while (rem > 0) {
    const size_t step = std::min(chunk, rem);
    const size_t wrote = fwrite(ptr, 1, step, file);
    if (wrote != step) {
      throw std::runtime_error(std::string("Incomplete particle output: ") + std::strerror(errno));
    }
    ptr += wrote;
    rem -= wrote;
  }
}

struct AsyncLightconeWriter {
  FILE* fp = nullptr;
  PinnedHostBuf slot_buf[2];
  size_t slot_cnt[2]{};
  bool slot_busy[2]{};
  std::queue<int> ready_q;
  std::mutex mtx;
  std::condition_variable cv;
  std::thread worker;
  std::exception_ptr err = nullptr;
  bool stop = false;
  int next_slot = 0;

  void start(FILE* out_fp) {
    fp = out_fp;
    stop = false;
    err = nullptr;
    slot_busy[0] = slot_busy[1] = false;
    slot_cnt[0] = slot_cnt[1] = 0;
    next_slot = 0;
    worker = std::thread([this]() {
      try {
        while (true) {
          int s = -1;
          size_t cnt = 0;
          {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [this]() { return !ready_q.empty() || stop; });
            if (ready_q.empty() && stop) break;
            s = ready_q.front();
            ready_q.pop();
            cnt = slot_cnt[s];
          }
          if (cnt > 0 && fp) {
            write_bin(fp, reinterpret_cast<const LightconeHit*>(slot_buf[s].p), cnt);
          }
          {
            std::lock_guard<std::mutex> lk(mtx);
            slot_busy[s] = false;
          }
          cv.notify_all();
        }
      } catch (...) {
        std::lock_guard<std::mutex> lk(mtx);
        err = std::current_exception();
        stop = true;
        slot_busy[0] = slot_busy[1] = false;
        cv.notify_all();
      }
    });
  }

  std::pair<int, LightconeHit*> acquire_slot(size_t count) {
    int s = next_slot;
    next_slot ^= 1;
    {
      std::unique_lock<std::mutex> lk(mtx);
      cv.wait(lk, [this, s]() { return !slot_busy[s] || err != nullptr; });
      if (err) std::rethrow_exception(err);
      slot_busy[s] = true;
    }
    slot_buf[s].alloc(count * sizeof(LightconeHit), cudaHostAllocPortable);
    return {s, reinterpret_cast<LightconeHit*>(slot_buf[s].p)};
  }

  void submit_slot(int s, size_t count) {
    std::lock_guard<std::mutex> lk(mtx);
    if (err) std::rethrow_exception(err);
    slot_cnt[s] = count;
    ready_q.push(s);
    cv.notify_all();
  }

  void finish() {
    if (!worker.joinable()) return;
    {
      std::lock_guard<std::mutex> lk(mtx);
      stop = true;
    }
    cv.notify_all();
    worker.join();
    if (err) {
      std::exception_ptr e = err;
      err = nullptr;
      std::rethrow_exception(e);
    }
  }

  ~AsyncLightconeWriter() {
    if (worker.joinable()) {
      {
        std::lock_guard<std::mutex> lk(mtx);
        stop = true;
      }
      cv.notify_all();
      worker.join();
    }
  }
};

struct FofHeader {
  uint64_t n;
  uint64_t num_groups;
  uint64_t largest_group;
  uint64_t min_group_size;
  double box;
  double scale;
  double link;
};

void write_fof_bin(const std::string& path, uint64_t n, uint64_t num_groups,
                   uint64_t largest_group, uint64_t min_group_size,
                   double box, double scale, double link, const int64_t* ids) {
  FILE* out = fopen(path.c_str(), "wb");
  if (!out) throw std::runtime_error("Cannot open FoF output: " + path);
  const FofHeader hdr{n, num_groups, largest_group, min_group_size, box, scale, link};
  write_bin(out, &hdr, 1);
  write_bin(out, ids, n);
  if (fclose(out)) throw std::runtime_error("Cannot close FoF output: " + path);
}

size_t compute_slab_cap(size_t n) {
  const double slack = (n > 50000000ULL) ? 1.04 : ((n > 10000000ULL) ? 1.07 : 1.15);
  return std::max<size_t>(4096, size_t(double(n) * slack) + 32768);
}
size_t compute_node_cap(size_t cap, int leaf_limit) {
  return std::max<size_t>(32768, (11ULL * cap) / (2ULL * size_t(leaf_limit)));
}

double comoving_distance_mpc(double a_lo, double a_hi, const Physics& p) {
  if (!p.comoving || a_lo >= a_hi) return 0.0;
  constexpr double c_km_s = 299792.458;
  constexpr int n_panels = 64;
  const double h = (a_hi - a_lo) / double(n_panels);
  auto f = [&](double a) {
    return c_km_s / (a * a * p.H(a));
  };
  double sum = f(a_lo) + f(a_hi);
  for (int i = 1; i < n_panels; ++i) {
    sum += ((i & 1) ? 4.0 : 2.0) * f(a_lo + double(i) * h);
  }
  return sum * (h / 3.0);
}

std::vector<EwaldMode> build_ewald_modes(double box) {
  constexpr int n_max = 7;
  constexpr int n2_max = 42;
  const double alpha = 4.0 / box;
  const double inv4a2 = 1.0 / (4.0 * alpha * alpha);
  const double dk = 2.0 * acos(-1.0) / box;
  const double vol_pref = 4.0 * acos(-1.0) / (box * box * box);
  std::vector<EwaldMode> modes;
  modes.reserve(600);
  for (int nx = 0; nx <= n_max; ++nx) {
    for (int ny = (nx == 0 ? 0 : -n_max); ny <= n_max; ++ny) {
      for (int nz = (nx == 0 && ny == 0 ? 1 : -n_max); nz <= n_max; ++nz) {
        const int n2 = nx * nx + ny * ny + nz * nz;
        if (n2 == 0 || n2 > n2_max) continue;
        const double kx = dk * double(nx);
        const double ky = dk * double(ny);
        const double kz = dk * double(nz);
        const double k2 = kx * kx + ky * ky + kz * kz;
        const double w = 2.0 * vol_pref * exp(-k2 * inv4a2) / k2;
        modes.push_back({kx, ky, kz, w});
      }
    }
  }
  return modes;
}

__global__ void gather_fc_samples_kernel(
    const double3* d_x, const float* d_soft, float first_soft,
    const float4* d_force, const float3* d_mesh_acc,
    const int* d_lids, int q_n, int dst_off,
    double3* out_x, float* out_h, float4* out_short, float3* out_mesh) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= q_n) return;
  const int lid = d_lids[k];
  out_x[dst_off + k] = d_x[lid];
  out_h[dst_off + k] = d_soft ? d_soft[lid] : first_soft;
  out_short[dst_off + k] = d_force[lid];
  if (out_mesh && d_mesh_acc) out_mesh[dst_off + k] = d_mesh_acc[lid];
}

#include "swift_domain_3d_engine.cuh"

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <input.bin> <output.bin> <a_end>\n";
    return 2;
  }
  try {
    const double started = wall();

    int dev_count = 0;
    check(cudaGetDeviceCount(&dev_count));
    const char* ngpu_env = std::getenv("SWIFT_NUM_GPUS");
    const int num_gpus = ngpu_env ? std::stoi(ngpu_env) : dev_count;
    if (num_gpus <= 0 || num_gpus > dev_count || num_gpus > num_slabs) {
      throw std::runtime_error("Invalid SWIFT_NUM_GPUS=" + std::to_string(num_gpus) +
                               " (dev_count=" + std::to_string(dev_count) + ")");
    }

    const char* periodic_env = std::getenv("SWIFT_PERIODIC");
    const char* comoving_env = std::getenv("SWIFT_COMOVING");
    const bool periodic = !periodic_env || std::string(periodic_env) != "0";
    const bool comoving = comoving_env ? (std::string(comoving_env) != "0") : periodic;
    if (!periodic && num_gpus > 1) {
      throw std::runtime_error("SWIFT_PERIODIC=0 (isolated mode) currently requires SWIFT_NUM_GPUS=1");
    }

    FILE* input = fopen(argv[1], "rb");
    if (!input) return 3;
    StateInput h{};
    read_bin(input, &h, 1);
    const uint64_t N = h.n;
    if (N == 0 || h.box <= 0.0 || h.plummer_softening <= 0.0) return 3;
    if (comoving ? (h.initial_scale <= 0.0) : (h.initial_scale < 0.0)) return 3;
    const double a_end = std::stod(argv[3]);
    if (!(a_end > h.initial_scale)) return 3;

    const bool dynamic_pm = periodic && (std::getenv("SWIFT_DYNAMIC_PM") != nullptr);
    const char* requested_pm = std::getenv("SWIFT_PM_INTERVALS");
    const int pm_intervals = requested_pm ? std::stoi(requested_pm) : (dynamic_pm ? 16 : 1);
    if (pm_intervals <= 0 || pm_intervals > 4096 || (pm_intervals & (pm_intervals - 1)))
      throw std::runtime_error("SWIFT_PM_INTERVALS must be a power of two between 1 and 4096");
    const char* requested_short = std::getenv("SWIFT_SHORT_CAP_INTERVALS");
    const int short_intervals = requested_short ? std::stoi(requested_short) : pm_intervals;
    if (short_intervals < pm_intervals || short_intervals > 4096 ||
        (short_intervals & (short_intervals - 1)))
      throw std::runtime_error("Invalid SWIFT_SHORT_CAP_INTERVALS");

    const char* requested_step = std::getenv("SWIFT_SHORT_STEP_FACTOR");
    const char* requested_mac = std::getenv("SWIFT_MAC_TOLERANCE");
    const char* requested_g = std::getenv("SWIFT_GRAV_G");
    const char* requested_dt_max = std::getenv("SWIFT_DT_MAX");
    const char* requested_rms_fac = std::getenv("SWIFT_MAX_DT_RMS_FACTOR");
    const char* leaf_override = std::getenv("SWIFT_LEAF_LIMIT");
    const char* m2l_env = std::getenv("SWIFT_WARP_M2L");
    const char* fsplit_env = std::getenv("SWIFT_FAST_SPLIT");
    const char* lazy_env = std::getenv("SWIFT_LAZY_TREE");
    const char* split_force_env = std::getenv("SWIFT_SPLIT_FORCE");
    const char* split_batch_env = std::getenv("SWIFT_SPLIT_BATCH_WARPS");
    const char* pm_dist_env = std::getenv("SWIFT_PM_DIST");
    const char* domain_mode_env = std::getenv("SWIFT_DOMAIN_MODE");
    const char* rebal_interval_env = std::getenv("SWIFT_REBALANCE_INTERVAL");
    const char* rebal_min_gain_env = std::getenv("SWIFT_REBALANCE_MIN_GAIN");
    const char* force0_out_path = std::getenv("SWIFT_FORCE0_OUT");
    const char* lazy_max_env = std::getenv("SWIFT_LAZY_TREE_MAX_STEPS");
    const char* lazy_disp_env = std::getenv("SWIFT_LAZY_TREE_DISP_FRAC");
    const char* lazy_act_env = std::getenv("SWIFT_LAZY_TREE_ACTIVE_FRAC");
    const char* skip_out_env = std::getenv("SWIFT_SKIP_OUTPUT_WRITE");
    const char* force_comp_path = std::getenv("SWIFT_FORCE_COMPONENTS");
    const char* force3_out_path = std::getenv("SWIFT_FORCE3_OUT");
    const char* slab_axis_env = std::getenv("SWIFT_SLAB_AXIS");
    const char* slab_offset_env = std::getenv("SWIFT_SLAB_OFFSET");
    const char* p2p_env = std::getenv("SWIFT_P2P_MODE");
    const char* balance_env = std::getenv("SWIFT_SLAB_BALANCE");
    const char* slab_counts_env = std::getenv("SWIFT_SLAB_COUNTS");
    const char* out_scales_env = std::getenv("SWIFT_OUTPUT_SCALES");
    const char* out_prefix_env = std::getenv("SWIFT_OUTPUT_PREFIX");
    const char* midpoint_path = std::getenv("SWIFT_RESTART_MIDPOINT_OUT");
    const char* run_fof_env = std::getenv("SWIFT_RUN_FOF");
    const char* fof_ratio_env = std::getenv("SWIFT_FOF_LINK_RATIO");
    const char* fof_min_env = std::getenv("SWIFT_FOF_MIN_SIZE");
    const char* fof_slabs_env = std::getenv("SWIFT_FOF_SLABS");
    const char* pm_grid_env = std::getenv("SWIFT_PM_GRID");
    const char* soft_in_env = std::getenv("SWIFT_SOFT_IN");
    const char* h0_env = std::getenv("SWIFT_COSMO_H0");
    const char* om_m_env = std::getenv("SWIFT_COSMO_OMEGA_M");
    const char* om_de_env = std::getenv("SWIFT_COSMO_OMEGA_DE");
    const char* om_cdm_env = std::getenv("SWIFT_COSMO_OMEGA_CDM");
    const char* om_r_env = std::getenv("SWIFT_COSMO_OMEGA_R");
    const char* om_k_env = std::getenv("SWIFT_COSMO_OMEGA_K");
    const char* w0_env = std::getenv("SWIFT_COSMO_W0");
    const char* wa_env = std::getenv("SWIFT_COSMO_WA");
    const char* theta_env = std::getenv("SWIFT_THETA");
    const char* grav_eta_env = std::getenv("SWIFT_GRAV_ETA");
    const char* ext_pot_env = std::getenv("SWIFT_EXT_POT_PLUMMER");
    const char* fc_n_env = std::getenv("SWIFT_FORCE_CHECK_N");
    const char* energy_log_env = std::getenv("SWIFT_ENERGY_LOG");
    const bool energy_log = energy_log_env && std::string(energy_log_env) != "0";
    const char* lc_out_env = std::getenv("SWIFT_LIGHTCONE_OUT");
    const char* lc_obs_env = std::getenv("SWIFT_LIGHTCONE_OBS");
    const char* lc_aobs_env = std::getenv("SWIFT_LIGHTCONE_AOBS");
    const char* lc_rmax_env = std::getenv("SWIFT_LIGHTCONE_RMAX");
    const char* lc_rep_env = std::getenv("SWIFT_LIGHTCONE_REPLICATE");
    const int pm_side = pm_grid_env ? std::stoi(pm_grid_env) : 128;
    if (pm_side < 32 || pm_side > 1024 || (pm_side & (pm_side - 1))) {
      throw std::runtime_error("SWIFT_PM_GRID must be a power of two between 32 and 1024");
    }
    const bool has_soft_in = soft_in_env && *soft_in_env;
    const int requested_fof_slabs = fof_slabs_env ? std::stoi(fof_slabs_env) : 0;
    if (requested_fof_slabs < 0 || requested_fof_slabs > 1024) {
      throw std::runtime_error("SWIFT_FOF_SLABS must be between 0 and 1024");
    }
    const int force_check_n = fc_n_env ? std::stoi(fc_n_env) : 0;
    if (force_check_n < 0 || (force_check_n > 0 && num_gpus > 1)) {
      throw std::runtime_error("SWIFT_FORCE_CHECK_N must be >= 0 and requires SWIFT_NUM_GPUS=1");
    }
    const bool lc_enabled = lc_out_env && *lc_out_env;
    if (lc_enabled && (!periodic || !comoving || num_gpus > 1)) {
      throw std::runtime_error("SWIFT_LIGHTCONE_OUT requires SWIFT_PERIODIC=1, SWIFT_COMOVING=1, and SWIFT_NUM_GPUS=1");
    }
    const bool lc_replicate = !lc_rep_env || std::string(lc_rep_env) != "0";
    double3 lc_obs{0.5 * h.box, 0.5 * h.box, 0.5 * h.box};
    const double lc_a_obs = lc_aobs_env ? std::stod(lc_aobs_env) : a_end;
    double lc_r_max_req = lc_rmax_env ? std::stod(lc_rmax_env) : 0.0;
    if (lc_enabled) {
      if (!std::isfinite(lc_a_obs) || lc_a_obs <= 0.0 || !std::isfinite(lc_r_max_req) || lc_r_max_req < 0.0) {
        throw std::runtime_error("Invalid SWIFT_LIGHTCONE_AOBS or SWIFT_LIGHTCONE_RMAX");
      }
      if (lc_obs_env && *lc_obs_env) {
        std::stringstream ss(lc_obs_env);
        std::string tok;
        double coords[3]{};
        int ci = 0;
        while (std::getline(ss, tok, ',')) {
          if (ci >= 3) throw std::runtime_error("SWIFT_LIGHTCONE_OBS must be x,y,z");
          coords[ci++] = std::stod(tok);
        }
        if (ci != 3 || !std::isfinite(coords[0]) || !std::isfinite(coords[1]) || !std::isfinite(coords[2])) {
          throw std::runtime_error("SWIFT_LIGHTCONE_OBS must be 3 finite numbers");
        }
        lc_obs = {coords[0], coords[1], coords[2]};
      }
    }
    float ext_M_val = 0.f, ext_a_val = 1.f;
    double3 ext_c_orig{0.5 * h.box, 0.5 * h.box, 0.5 * h.box};
    if (ext_pot_env && *ext_pot_env) {
      if (periodic) {
        throw std::runtime_error("SWIFT_EXT_POT_PLUMMER requires SWIFT_PERIODIC=0");
      }
      std::stringstream ss(ext_pot_env);
      std::string tok;
      double vals[5]{};
      int vi = 0;
      while (std::getline(ss, tok, ',')) {
        if (vi >= 5) throw std::runtime_error("SWIFT_EXT_POT_PLUMMER must be M,a_s,cx,cy,cz");
        vals[vi++] = std::stod(tok);
      }
      if (vi != 5 || !std::isfinite(vals[0]) || vals[0] <= 0.0 ||
          !std::isfinite(vals[1]) || vals[1] <= 0.0 ||
          !std::isfinite(vals[2]) || !std::isfinite(vals[3]) || !std::isfinite(vals[4])) {
        throw std::runtime_error("Invalid SWIFT_EXT_POT_PLUMMER=M,a_s,cx,cy,cz");
      }
      ext_M_val = float(vals[0]);
      ext_a_val = float(vals[1]);
      ext_c_orig = {vals[2], vals[3], vals[4]};
    }
    const bool run_fof = run_fof_env && std::string(run_fof_env) == "1";
    const double fof_link_ratio = fof_ratio_env ? std::stod(fof_ratio_env) : 0.2;
    const int fof_min_size = fof_min_env ? std::stoi(fof_min_env) : 32;
    if (run_fof) {
      if (!std::isfinite(fof_link_ratio) || fof_link_ratio <= 0.0 || fof_min_size < 2) {
        throw std::runtime_error("Invalid SWIFT_FOF_LINK_RATIO or SWIFT_FOF_MIN_SIZE");
      }
      if (N > (1ULL << 30)) {
        throw std::runtime_error("SWIFT_RUN_FOF supports N <= 2^30 particles");
      }
    }
    std::string p2p_mode = p2p_env ? p2p_env : "nvlink";
    if (p2p_mode != "nvlink" && p2p_mode != "all" && p2p_mode != "off")
      throw std::runtime_error("SWIFT_P2P_MODE must be nvlink, all, or off");

    auto trim_str = [](const std::string& s) -> std::string {
      const size_t lo = s.find_first_not_of(" \t\r\n");
      if (lo == std::string::npos) return "";
      const size_t hi = s.find_last_not_of(" \t\r\n");
      return s.substr(lo, hi - lo + 1);
    };

    std::string balance_mode = balance_env ? balance_env : "auto";
    if (balance_mode == "1") balance_mode = "auto";
    else if (balance_mode == "0") balance_mode = "uniform";
    if (balance_mode != "auto" && balance_mode != "uniform")
      throw std::runtime_error("SWIFT_SLAB_BALANCE must be auto, uniform, 1, or 0");

    std::vector<int> fixed_counts;
    if (slab_counts_env && *slab_counts_env) {
      std::stringstream ss(slab_counts_env);
      std::string tok;
      int sum_cnt = 0;
      while (std::getline(ss, tok, ',')) {
        const std::string t = trim_str(tok);
        if (t.empty()) throw std::runtime_error("Empty token in SWIFT_SLAB_COUNTS");
        size_t pos = 0;
        const int c = std::stoi(t, &pos);
        if (pos != t.size() || c <= 0)
          throw std::runtime_error("Invalid count in SWIFT_SLAB_COUNTS: " + t);
        fixed_counts.push_back(c);
        sum_cnt += c;
      }
      if (int(fixed_counts.size()) != num_gpus || sum_cnt != num_slabs) {
        throw std::runtime_error("SWIFT_SLAB_COUNTS must have G positive integers summing to 12");
      }
      balance_mode = "fixed";
    } else if (balance_mode == "uniform") {
      if (num_slabs % num_gpus != 0) {
        throw std::runtime_error("Uniform slab balance requires 12 divisible by SWIFT_NUM_GPUS");
      }
      fixed_counts.assign(num_gpus, num_slabs / num_gpus);
    }

    std::vector<double> snap_scales;
    if (out_scales_env && *out_scales_env) {
      if (midpoint_path && *midpoint_path) {
        throw std::runtime_error("Cannot combine SWIFT_OUTPUT_SCALES and SWIFT_RESTART_MIDPOINT_OUT");
      }
      std::stringstream ss(out_scales_env);
      std::string tok;
      while (std::getline(ss, tok, ',')) {
        const std::string t = trim_str(tok);
        if (t.empty()) throw std::runtime_error("Empty token in SWIFT_OUTPUT_SCALES");
        size_t pos = 0;
        const double val = std::stod(t, &pos);
        if (pos != t.size() || !std::isfinite(val))
          throw std::runtime_error("Invalid scale factor in SWIFT_OUTPUT_SCALES: " + t);
        if (!(val > h.initial_scale && val < a_end))
          throw std::runtime_error("SWIFT_OUTPUT_SCALES must satisfy a_begin < a_k < a_end");
        if (!snap_scales.empty() && !(val > snap_scales.back()))
          throw std::runtime_error("SWIFT_OUTPUT_SCALES must be strictly increasing");
        snap_scales.push_back(val);
      }
    }
    std::string out_prefix;
    if (out_prefix_env && *out_prefix_env) {
      out_prefix = out_prefix_env;
    } else {
      out_prefix = argv[2];
      if (out_prefix.size() >= 4 && out_prefix.substr(out_prefix.size() - 4) == ".bin") {
        out_prefix.erase(out_prefix.size() - 4);
      }
      out_prefix += "_snap";
    }

    int requested_axis = -1;
    if (slab_axis_env && std::string(slab_axis_env) != "auto") {
      requested_axis = std::stoi(slab_axis_env);
      if (requested_axis < 0 || requested_axis > 2)
        throw std::runtime_error("SWIFT_SLAB_AXIS must be auto, 0, 1, or 2");
    }
    int requested_offset = -1;
    if (slab_offset_env && std::string(slab_offset_env) != "auto") {
      requested_offset = std::stoi(slab_offset_env);
      if (requested_offset < 0 || requested_offset >= num_slabs)
        throw std::runtime_error("SWIFT_SLAB_OFFSET must be auto or 0..11");
    }

    const float step_factor = requested_step ? std::stof(requested_step) : 1.f;
    const float mac = requested_mac ? std::stof(requested_mac) : tolerance;
    const float grav_G = requested_g ? std::stof(requested_g) : grav_G_default;
    const double dt_max_val = requested_dt_max ? std::stod(requested_dt_max) : (comoving ? dt_max_default : 0.0);
    const double max_rms_fac = requested_rms_fac ? std::stod(requested_rms_fac) : max_dt_RMS_factor_default;
    const double cosmo_h0 = h0_env ? std::stod(h0_env) : hubble0_default;
    const double cosmo_om = om_m_env ? std::stod(om_m_env) : omega_m_default;
    const double cosmo_ode = om_de_env ? std::stod(om_de_env) : omega_de_default;
    const double cosmo_ocdm = om_cdm_env ? std::stod(om_cdm_env) : (om_m_env ? cosmo_om : omega_cdm_default);
    const double cosmo_or = om_r_env ? std::stod(om_r_env) : 0.0;
    const double cosmo_ok = om_k_env ? std::stod(om_k_env) : 0.0;
    const double cosmo_w0 = w0_env ? std::stod(w0_env) : -1.0;
    const double cosmo_wa = wa_env ? std::stod(wa_env) : 0.0;
    const float theta_val = theta_env ? std::stof(theta_env) : theta_default;
    const float grav_eta_val = grav_eta_env ? std::stof(grav_eta_env) : grav_eta_default;
    const int leaf_limit = leaf_override ? std::stoi(leaf_override) : 64;
    const bool warp_m2l = !m2l_env || std::string(m2l_env) != "0";
    const bool fast_split = !fsplit_env || std::string(fsplit_env) != "0";
    const bool lazy_tree_enabled = periodic && (!lazy_env || std::string(lazy_env) != "0");
    const bool split_force_enabled = periodic && (!split_force_env || std::string(split_force_env) != "0");
    const bool pm_dist_enabled = periodic && (num_gpus > 1) && (!pm_dist_env || std::string(pm_dist_env) != "0");
    const int rebalance_interval = rebal_interval_env ? std::stoi(rebal_interval_env) : 8;
    const double rebalance_min_gain = rebal_min_gain_env ? std::stod(rebal_min_gain_env) : 0.02;
    if (rebalance_interval < 0 || !std::isfinite(rebalance_min_gain) || rebalance_min_gain < 0.0) {
      throw std::runtime_error("Invalid SWIFT_REBALANCE_INTERVAL or SWIFT_REBALANCE_MIN_GAIN");
    }
    swift3d::DomainMode domain_mode = swift3d::parse_domain_mode(domain_mode_env, num_gpus);
    if (domain_mode != swift3d::DomainMode::Slab1D && (!periodic || !split_force_enabled)) {
      if (domain_mode_env && std::string(domain_mode_env) != "auto" &&
          std::string(domain_mode_env) != "1d_slab") {
        throw std::runtime_error("3D domain mode requires SWIFT_PERIODIC=1 and SWIFT_SPLIT_FORCE=1");
      }
      domain_mode = swift3d::DomainMode::Slab1D;
    }
    const bool split_profile = split_force_enabled && std::getenv("SWIFT_SPLIT_PROFILE") &&
                               std::string(std::getenv("SWIFT_SPLIT_PROFILE")) == "1";
    const int split_batch_warps = split_batch_env ? std::stoi(split_batch_env) : 16384;
    if (split_batch_warps < 256 || split_batch_warps > (1 << 20)) {
      throw std::runtime_error("SWIFT_SPLIT_BATCH_WARPS must be between 256 and 1048576");
    }
    if (force0_out_path && num_gpus > 1 && domain_mode == swift3d::DomainMode::Slab1D) {
      throw std::runtime_error("SWIFT_FORCE0_OUT on 1d_slab requires SWIFT_NUM_GPUS=1");
    }
    const int lazy_max_steps = lazy_max_env ? std::stoi(lazy_max_env) : 16;
    const double lazy_disp_frac = lazy_disp_env ? std::stod(lazy_disp_env) : 0.15;
    const double lazy_active_frac = lazy_act_env ? std::stod(lazy_act_env) : 0.50;
    const bool skip_output_write = skip_out_env && std::string(skip_out_env) == "1";

    if (leaf_limit <= 0 || !std::isfinite(step_factor) || step_factor <= 0.f ||
        lazy_max_steps < 1 || !std::isfinite(lazy_disp_frac) || lazy_disp_frac <= 0.0 ||
        !std::isfinite(lazy_active_frac) || lazy_active_frac <= 0.0 || lazy_active_frac > 1.0 ||
        !std::isfinite(mac) || mac <= 0.f || !std::isfinite(grav_G) || grav_G <= 0.f ||
        !std::isfinite(dt_max_val) || dt_max_val < 0.0 || (comoving && dt_max_val <= 0.0) ||
        !std::isfinite(cosmo_h0) || cosmo_h0 <= 0.0 ||
        !std::isfinite(cosmo_om) || cosmo_om <= 0.0 ||
        !std::isfinite(cosmo_ocdm) || cosmo_ocdm <= 0.0 ||
        !std::isfinite(cosmo_ode) || cosmo_ode < 0.0 ||
        !std::isfinite(cosmo_or) || cosmo_or < 0.0 ||
        !std::isfinite(cosmo_ok) || !std::isfinite(cosmo_w0) || !std::isfinite(cosmo_wa) ||
        !std::isfinite(theta_val) || theta_val <= 0.f ||
        !std::isfinite(grav_eta_val) || grav_eta_val <= 0.f)
      throw std::runtime_error("Invalid physics or cosmology parameter");

    // Initialize GPUs & P2P matrix (slab vectors sized after Pass 1 load balancing)
    std::vector<GpuCtx> gpus(num_gpus);
    std::vector<std::vector<int>> can_peer(num_gpus, std::vector<int>(num_gpus, 0));
    for (int g = 0; g < num_gpus; ++g) {
      gpus[g].dev = g;
      check(cudaSetDevice(g));
      check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
      check(cudaStreamCreateWithFlags(&gpus[g].compute_stream, cudaStreamNonBlocking));
      check(cudaStreamCreateWithFlags(&gpus[g].copy_stream, cudaStreamNonBlocking));
      check(cudaStreamCreateWithFlags(&gpus[g].d2h_left_stream, cudaStreamNonBlocking));
      check(cudaStreamCreateWithFlags(&gpus[g].d2h_right_stream, cudaStreamNonBlocking));
      check(cudaEventCreateWithFlags(&gpus[g].ev_left_start, 0));
      check(cudaEventCreateWithFlags(&gpus[g].ev_left_ready, 0));
      check(cudaEventCreateWithFlags(&gpus[g].ev_s0_done, cudaEventDisableTiming));
      check(cudaEventCreateWithFlags(&gpus[g].ev_right_start, 0));
      check(cudaEventCreateWithFlags(&gpus[g].ev_right_ready, 0));
      check(cudaEventCreateWithFlags(&gpus[g].ev_force_start, 0));
      check(cudaEventCreateWithFlags(&gpus[g].ev_force_end, 0));
      check(cudaEventCreateWithFlags(&gpus[g].ev_prof0, 0));
      check(cudaEventCreateWithFlags(&gpus[g].ev_prof1, 0));
      for (int i = 0; i < GpuCtx::bounce_slots; ++i) {
        check(cudaEventCreateWithFlags(&gpus[g].ev_d2h_left[i], cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&gpus[g].ev_d2h_right[i], cudaEventDisableTiming));
        check(cudaEventCreateWithFlags(&gpus[g].ev_h2d_done[i], cudaEventDisableTiming));
      }
      for (int s = 0; s < num_gpus; ++s) {
        if (s == g) { can_peer[g][s] = 1; continue; }
        int acc = 0, nat = 0;
        check(cudaDeviceGetP2PAttribute(&acc, cudaDevP2PAttrAccessSupported, g, s));
        check(cudaDeviceGetP2PAttribute(&nat, cudaDevP2PAttrNativeAtomicSupported, g, s));
        const int cp = (p2p_mode == "off") ? 0 : ((p2p_mode == "all") ? acc : (acc && nat));
        can_peer[g][s] = cp;
        if (cp) {
          cudaError_t perr = cudaDeviceEnablePeerAccess(s, 0);
          if (perr != cudaSuccess && perr != cudaErrorPeerAccessAlreadyEnabled) check(perr);
          cudaGetLastError();
        }
      }
    }
    check(cudaSetDevice(0));

    // Pass 1: Scan input.bin for mass/softening uniformity & per-axis slab histograms
    const double import_start = wall();
    const uint64_t pos_offset = sizeof(StateInput);
    const uint64_t vel_offset = pos_offset + N * sizeof(double3);
    const uint64_t mass_offset = vel_offset + N * sizeof(float3);

    const uint64_t chunk = std::min<uint64_t>(N, 16ULL << 20);
    std::vector<double3> rx(chunk);
    std::vector<float3> rv(chunk);
    std::vector<float> rm(chunk), rs;
    if (has_soft_in) rs.resize(chunk);

    std::array<std::array<size_t, num_slabs>, 3> axis_counts{};
    std::array<uint64_t, swift3d::N_TOP_CELLS> init_cell_counts{};
    double total_mass = 0.0, min_mass = DBL_MAX;
    float first_mass = 0.f;
    float first_soft_plummer = float(h.plummer_softening);
    bool mass_uniform = true;
    bool soft_uniform = true;
    const double inv_wx = double(num_slabs) / h.box;

    FILE* fx = fopen(argv[1], "rb");
    FILE* fv = fopen(argv[1], "rb");
    FILE* fm = fopen(argv[1], "rb");
    FILE* fs = has_soft_in ? fopen(soft_in_env, "rb") : nullptr;
    if (!fx || !fv || !fm || (has_soft_in && !fs))
      throw std::runtime_error("Failed reopening input or SWIFT_SOFT_IN for streaming");
    if (fs) {
      if (fseeko(fs, 0, SEEK_END) != 0) throw std::runtime_error("Failed seeking SWIFT_SOFT_IN");
      const off_t fsz = ftello(fs);
      if (fsz != off_t(N * sizeof(float))) {
        throw std::runtime_error("SWIFT_SOFT_IN size mismatch: got " + std::to_string(fsz) +
                                 " expected " + std::to_string(N * sizeof(float)));
      }
      fseeko(fs, 0, SEEK_SET);
    }
    fseeko(fx, off_t(pos_offset), SEEK_SET);
    fseeko(fv, off_t(vel_offset), SEEK_SET);
    fseeko(fm, off_t(mass_offset), SEEK_SET);

    for (uint64_t base = 0; base < N; base += chunk) {
      const size_t cnt = size_t(std::min<uint64_t>(chunk, N - base));
      read_bin(fx, rx.data(), cnt);
      read_bin(fm, rm.data(), cnt);
      if (fs) read_bin(fs, rs.data(), cnt);
      if (base == 0) {
        first_mass = rm[0];
        if (fs) first_soft_plummer = rs[0];
      }
      for (size_t k = 0; k < cnt; ++k) {
        const double px = periodic ? wrapped_coord(rx[k].x, h.box) : rx[k].x;
        const double py = periodic ? wrapped_coord(rx[k].y, h.box) : rx[k].y;
        const double pz = periodic ? wrapped_coord(rx[k].z, h.box) : rx[k].z;
        const int sx = std::min(std::max(int(px * inv_wx), 0), num_slabs - 1);
        const int sy = std::min(std::max(int(py * inv_wx), 0), num_slabs - 1);
        const int sz = std::min(std::max(int(pz * inv_wx), 0), num_slabs - 1);
        ++axis_counts[0][sx];
        ++axis_counts[1][sy];
        ++axis_counts[2][sz];
        if (domain_mode != swift3d::DomainMode::Slab1D) {
          ++init_cell_counts[(sx * num_slabs + sy) * num_slabs + sz];
        }
        const float mp = rm[k];
        if (mp != first_mass) mass_uniform = false;
        total_mass += double(mp);
        if (mp > 0.f && double(mp) < min_mass) min_mass = double(mp);
        if (fs) {
          const float sp = rs[k];
          if (!std::isfinite(sp) || sp <= 0.f) throw std::runtime_error("Invalid softening in SWIFT_SOFT_IN");
          if (sp != first_soft_plummer) soft_uniform = false;
        }
      }
    }

    if (domain_mode != swift3d::DomainMode::Slab1D) {
      fclose(fx);
      fclose(fv);
      fclose(fm);
      if (fs) fclose(fs);
      fclose(input);
      {
        std::vector<double3>().swap(rx);
        std::vector<float3>().swap(rv);
        std::vector<float>().swap(rm);
        std::vector<float>().swap(rs);
      }
      for (auto& G : gpus) G.cleanup();
      check(cudaSetDevice(0));

      swift3d::Domain3DRunConfig d3cfg{};
      d3cfg.input_path = argv[1];
      d3cfg.output_path = argv[2];
      d3cfg.started_wall = started;
      d3cfg.import_start = import_start;
      d3cfg.h = h;
      d3cfg.N = N;
      d3cfg.a_end = a_end;
      d3cfg.domain_mode = domain_mode;
      d3cfg.rebalance_interval = rebalance_interval;
      d3cfg.rebalance_min_gain = rebalance_min_gain;
      d3cfg.init_cell_counts = init_cell_counts;
      d3cfg.total_mass = total_mass;
      d3cfg.first_mass_in = double(first_mass);
      d3cfg.first_soft_in = double(first_soft_plummer);
      d3cfg.min_mass = min_mass;
      d3cfg.mass_uniform = mass_uniform;
      d3cfg.soft_uniform = soft_uniform;
      d3cfg.has_soft_in = has_soft_in;
      d3cfg.soft_in_env = soft_in_env;
      d3cfg.num_gpus = num_gpus;
      d3cfg.can_peer = can_peer;
      d3cfg.p2p_mode = p2p_mode;
      d3cfg.periodic = periodic;
      d3cfg.comoving = comoving;
      d3cfg.dynamic_pm = dynamic_pm;
      d3cfg.pm_intervals = pm_intervals;
      d3cfg.short_intervals = short_intervals;
      d3cfg.pm_side = pm_side;
      d3cfg.step_factor = step_factor;
      d3cfg.mac = mac;
      d3cfg.grav_G = grav_G;
      d3cfg.dt_max_val = dt_max_val;
      d3cfg.max_rms_fac = max_rms_fac;
      d3cfg.cosmo_h0 = cosmo_h0;
      d3cfg.cosmo_om = cosmo_om;
      d3cfg.cosmo_ode = cosmo_ode;
      d3cfg.cosmo_ocdm = cosmo_ocdm;
      d3cfg.cosmo_or = cosmo_or;
      d3cfg.cosmo_ok = cosmo_ok;
      d3cfg.cosmo_w0 = cosmo_w0;
      d3cfg.cosmo_wa = cosmo_wa;
      d3cfg.theta_val = theta_val;
      d3cfg.grav_eta_val = grav_eta_val;
      d3cfg.leaf_limit = leaf_limit;
      d3cfg.warp_m2l = warp_m2l;
      d3cfg.fast_split = fast_split;
      d3cfg.lazy_tree_enabled = lazy_tree_enabled;
      d3cfg.split_force_enabled = split_force_enabled;
      d3cfg.pm_dist_enabled = pm_dist_enabled;
      d3cfg.split_profile = split_profile;
      d3cfg.split_batch_warps = split_batch_warps;
      d3cfg.lazy_max_steps = lazy_max_steps;
      d3cfg.lazy_disp_frac = lazy_disp_frac;
      d3cfg.lazy_active_frac = lazy_active_frac;
      d3cfg.skip_output_write = skip_output_write;
      d3cfg.energy_log = energy_log;
      d3cfg.force0_out_path = force0_out_path;
      d3cfg.force_comp_path = force_comp_path;
      d3cfg.force3_out_path = force3_out_path;
      d3cfg.midpoint_path = midpoint_path;
      d3cfg.snap_scales = snap_scales;
      d3cfg.out_prefix = out_prefix;
      d3cfg.run_fof = run_fof;
      d3cfg.fof_link_ratio = fof_link_ratio;
      d3cfg.fof_min_size = fof_min_size;
      d3cfg.requested_fof_slabs = requested_fof_slabs;
      d3cfg.force_check_n = force_check_n;
      d3cfg.lc_enabled = lc_enabled;
      d3cfg.lc_replicate = lc_replicate;
      d3cfg.lc_obs = lc_obs;
      d3cfg.lc_a_obs = lc_a_obs;
      d3cfg.lc_r_max_req = lc_r_max_req;
      d3cfg.lc_out_env = lc_out_env;
      return swift3d::run_domain_3d_evolution(d3cfg, compute_slab_cap, compute_node_cap);
    }

    // Optimal cyclic load balance (axis, offset, contiguous per-GPU slab counts) WITHOUT shifting coordinates
    int slab_axis = 0, slab_offset = 0;
    std::vector<int> gpu_slab_start(num_gpus, 0), gpu_slab_count(num_gpus, 0);
    {
      struct PartitionPlan {
        size_t gpu_peak = ~size_t(0);
        int has_single = 1;
        size_t cross_bridge = ~size_t(0);
        long double sum_sq = std::numeric_limits<long double>::infinity();
        size_t slab_peak = ~size_t(0);
        int axis = 0;
        int offset = 0;
        std::vector<int> counts;
      };
      auto is_better = [](const PartitionPlan& a, const PartitionPlan& b) -> bool {
        if (a.gpu_peak != b.gpu_peak) return a.gpu_peak < b.gpu_peak;
        if (a.has_single != b.has_single) return a.has_single < b.has_single;
        if (a.cross_bridge != b.cross_bridge) return a.cross_bridge < b.cross_bridge;
        if (a.sum_sq != b.sum_sq) return a.sum_sq < b.sum_sq;
        return a.slab_peak < b.slab_peak;
      };
      auto eval_counts = [&](const std::array<size_t, num_slabs>& w,
                             const std::array<size_t, num_slabs + 1>& pref,
                             int ax, int off, size_t slab_peak,
                             const std::vector<int>& cnts) -> PartitionPlan {
        PartitionPlan pl;
        pl.axis = ax;
        pl.offset = off;
        pl.slab_peak = slab_peak;
        pl.counts = cnts;
        pl.gpu_peak = 0;
        pl.has_single = 0;
        pl.cross_bridge = 0;
        pl.sum_sq = 0.0L;
        std::vector<int> seg_s(num_gpus), seg_e(num_gpus);
        int pos = 0;
        for (int g = 0; g < num_gpus; ++g) {
          seg_s[g] = pos;
          seg_e[g] = pos + cnts[g] - 1;
          pos += cnts[g];
          const size_t ssum = pref[pos] - pref[seg_s[g]];
          pl.gpu_peak = std::max(pl.gpu_peak, ssum);
          if (num_gpus > 1 && num_gpus <= 6 && cnts[g] < 2) pl.has_single = 1;
          pl.sum_sq += (long double)(ssum) * (long double)(ssum);
        }
        if (num_gpus > 1) {
          for (int g = 0; g < num_gpus; ++g) {
            const int ng = (g + 1) % num_gpus;
            if (!can_peer[g][ng]) {
              pl.cross_bridge += w[seg_e[g]] + w[seg_s[ng]];
            }
          }
        }
        return pl;
      };
      auto solve_dp = [&](const std::array<size_t, num_slabs>& w,
                          const std::array<size_t, num_slabs + 1>& pref,
                          size_t cap, int min_len, std::vector<int>& out_cnts) -> bool {
        const size_t INF_CROSS = ~size_t(0);
        std::array<std::array<bool, num_slabs + 1>, num_slabs + 1> ok{};
        std::array<std::array<size_t, num_slabs + 1>, num_slabs + 1> dp_cross{};
        std::array<std::array<long double, num_slabs + 1>, num_slabs + 1> dp_sq{};
        std::array<std::array<int, num_slabs + 1>, num_slabs + 1> dp_prev{};
        for (int p = min_len; p <= num_slabs - (num_gpus - 1) * min_len; ++p) {
          const size_t sw = pref[p] - pref[0];
          if (sw <= cap) {
            ok[1][p] = true;
            dp_cross[1][p] = 0;
            dp_sq[1][p] = (long double)(sw) * (long double)(sw);
            dp_prev[1][p] = 0;
          }
        }
        const size_t wrap_cross = (!can_peer[num_gpus - 1][0]) ? (w[num_slabs - 1] + w[0]) : 0;
        for (int g = 2; g <= num_gpus; ++g) {
          const int p_lo = (g == num_gpus) ? num_slabs : (g * min_len);
          const int p_hi = (g == num_gpus) ? num_slabs : (num_slabs - (num_gpus - g) * min_len);
          for (int p = p_lo; p <= p_hi; ++p) {
            dp_cross[g][p] = INF_CROSS;
            for (int prev = (g - 1) * min_len; prev <= p - min_len; ++prev) {
              if (!ok[g - 1][prev]) continue;
              const size_t sw = pref[p] - pref[prev];
              if (sw > cap) continue;
              const size_t edge = (!can_peer[g - 2][g - 1]) ? (w[prev - 1] + w[prev]) : 0;
              const size_t cand_c = dp_cross[g - 1][prev] + edge + ((g == num_gpus) ? wrap_cross : 0);
              const long double cand_sq = dp_sq[g - 1][prev] + (long double)(sw) * (long double)(sw);
              if (!ok[g][p] || cand_c < dp_cross[g][p] ||
                  (cand_c == dp_cross[g][p] && cand_sq < dp_sq[g][p])) {
                ok[g][p] = true;
                dp_cross[g][p] = cand_c;
                dp_sq[g][p] = cand_sq;
                dp_prev[g][p] = prev;
              }
            }
          }
        }
        if (!ok[num_gpus][num_slabs]) return false;
        out_cnts.resize(num_gpus);
        int cur = num_slabs;
        for (int g = num_gpus; g >= 1; --g) {
          const int prev = dp_prev[g][cur];
          out_cnts[g - 1] = cur - prev;
          cur = prev;
        }
        return true;
      };

      PartitionPlan best;
      const int ax_lo = (requested_axis >= 0) ? requested_axis : 0;
      const int ax_hi = (requested_axis >= 0) ? requested_axis : 2;
      const int off_lo = (requested_offset >= 0) ? requested_offset : 0;
      const int off_hi = (requested_offset >= 0) ? requested_offset : ((num_gpus == 1) ? 0 : num_slabs - 1);
      for (int ax = ax_lo; ax <= ax_hi; ++ax) {
        size_t slab_peak = 0;
        for (int s = 0; s < num_slabs; ++s) slab_peak = std::max(slab_peak, axis_counts[ax][s]);
        for (int off = off_lo; off <= off_hi; ++off) {
          std::array<size_t, num_slabs> w{};
          std::array<size_t, num_slabs + 1> pref{};
          for (int i = 0; i < num_slabs; ++i) {
            w[i] = axis_counts[ax][(off + i) % num_slabs];
            pref[i + 1] = pref[i] + w[i];
          }
          PartitionPlan cand;
          if (!fixed_counts.empty()) {
            cand = eval_counts(w, pref, ax, off, slab_peak, fixed_counts);
          } else if (num_gpus == 1) {
            cand = eval_counts(w, pref, ax, off, slab_peak, std::vector<int>{num_slabs});
          } else {
            std::vector<size_t> cands;
            cands.reserve((num_slabs * (num_slabs + 1)) / 2);
            for (int a = 0; a < num_slabs; ++a)
              for (int b = a + 1; b <= num_slabs; ++b)
                cands.push_back(pref[b] - pref[a]);
            std::sort(cands.begin(), cands.end());
            cands.erase(std::unique(cands.begin(), cands.end()), cands.end());
            size_t lo = 0, hi = cands.size() - 1;
            std::vector<int> opt_cnts;
            while (lo < hi) {
              const size_t mid = lo + (hi - lo) / 2;
              if (solve_dp(w, pref, cands[mid], 1, opt_cnts)) hi = mid;
              else lo = mid + 1;
            }
            const size_t p_best = cands[lo];
            if (!(num_gpus <= 6 && solve_dp(w, pref, p_best, 2, opt_cnts))) {
              if (!solve_dp(w, pref, p_best, 1, opt_cnts))
                throw std::runtime_error("Internal DP partition error");
            }
            cand = eval_counts(w, pref, ax, off, slab_peak, opt_cnts);
          }
          if (best.counts.empty() || is_better(cand, best)) {
            best = std::move(cand);
          }
        }
      }
      slab_axis = best.axis;
      slab_offset = best.offset;
      gpu_slab_count = std::move(best.counts);
      int start = 0;
      for (int g = 0; g < num_gpus; ++g) {
        gpu_slab_start[g] = start;
        start += gpu_slab_count[g];
      }
    }

    std::array<std::pair<int, int>, num_slabs> slab_owner{};
    auto owned_slab = [&](int g, int ls) -> int {
      return (slab_offset + gpu_slab_start[g] + ls) % num_slabs;
    };
    for (int g = 0; g < num_gpus; ++g) {
      gpus[g].slabs_per_gpu = gpu_slab_count[g];
      gpus[g].need_two_ghosts = (num_gpus > 1 && gpu_slab_count[g] == 1);
      gpus[g].slabs.resize(gpu_slab_count[g]);
      gpus[g].trees.resize(gpu_slab_count[g]);
      for (int ls = 0; ls < gpu_slab_count[g]; ++ls) {
        slab_owner[owned_slab(g, ls)] = {g, ls};
      }
    }
    auto owner_of_slab = [&](int s) -> std::pair<int, int> {
      return slab_owner[s];
    };

    const std::array<size_t, num_slabs>& slab_counts = axis_counts[slab_axis];
    std::array<HostSlabStaging, num_slabs> h_stage;
    size_t max_initial_slab_n = 0;
    for (int s = 0; s < num_slabs; ++s) {
      const size_t sc = slab_counts[s];
      max_initial_slab_n = std::max(max_initial_slab_n, sc);
      h_stage[s].x.reserve(sc);
      h_stage[s].v.reserve(sc);
      h_stage[s].orig_idx.reserve(sc);
      if (!mass_uniform) h_stage[s].mass.reserve(sc);
      if (!soft_uniform) h_stage[s].soft.reserve(sc);
    }

    // Pass 2: Stream into per-slab host buckets, upload to owner GPUs, then free host buckets
    fseeko(fx, off_t(pos_offset), SEEK_SET);
    fseeko(fv, off_t(vel_offset), SEEK_SET);
    fseeko(fm, off_t(mass_offset), SEEK_SET);
    if (fs && !soft_uniform) fseeko(fs, 0, SEEK_SET);
    const float a0_f = comoving ? float(h.initial_scale) : 1.f;
    for (uint64_t base = 0; base < N; base += chunk) {
      const size_t cnt = size_t(std::min<uint64_t>(chunk, N - base));
      read_bin(fx, rx.data(), cnt);
      read_bin(fv, rv.data(), cnt);
      if (!mass_uniform) read_bin(fm, rm.data(), cnt);
      if (fs && !soft_uniform) read_bin(fs, rs.data(), cnt);
      for (size_t k = 0; k < cnt; ++k) {
        double3 pos = rx[k];
        if (periodic) {
          pos.x = wrapped_coord(pos.x, h.box);
          pos.y = wrapped_coord(pos.y, h.box);
          pos.z = wrapped_coord(pos.z, h.box);
        }
        pos = permute_in(pos, slab_axis);
        const int s = std::min(std::max(int(pos.x * inv_wx), 0), num_slabs - 1);
        auto& HS = h_stage[s];
        HS.x.push_back(pos);
        const float3 vi = permute_in(rv[k], slab_axis);
        HS.v.push_back(float3{vi.x * a0_f, vi.y * a0_f, vi.z * a0_f});
        HS.orig_idx.push_back(uint32_t(base + k));
        if (!mass_uniform) HS.mass.push_back(rm[k]);
        if (!soft_uniform) HS.soft.push_back(float(double(rs[k]) * kernel_gravity_softening_plummer_equivalent));
      }
    }
    fclose(fx);
    fclose(fv);
    fclose(fm);
    if (fs) fclose(fs);
    fclose(input);
    { std::vector<double3>().swap(rx); std::vector<float3>().swap(rv); std::vector<float>().swap(rm); std::vector<float>().swap(rs); }
    if (const char* unlink_env = std::getenv("SWIFT_UNLINK_INPUT_AFTER_LOAD")) {
      if (std::string(unlink_env) == "1") std::remove(argv[1]);
    }

    // Allocate GPU resident buffers & upload each slab to its owner GPU
    const size_t pm_points = periodic ? (size_t(pm_side) * size_t(pm_side) * size_t(pm_side)) : 0ULL;
#pragma omp parallel num_threads(num_gpus)
    {
      const int g = omp_get_thread_num();
      auto& G = gpus[g];
      check(cudaSetDevice(G.dev));
      G.need_soft = !soft_uniform;
      G.need_split_nodes = split_force_enabled;

      G.d_active_count.alloc(1, G.device_bytes);
      G.d_tree_invalid.alloc(1, G.device_bytes);
      G.d_stats.alloc(1, G.device_bytes);
      G.d_kick_finish.alloc(57, G.device_bytes);
      G.d_kick_begin.alloc(57, G.device_bytes);
      if (periodic && !pm_dist_enabled) {
        G.d_pm_density.alloc(pm_points, G.device_bytes);
        G.d_pm_field.alloc(pm_points, G.device_bytes);
      }
      G.d_scan_tmp.alloc(scan_grid_max, G.device_bytes);
      G.h_scan.resize(scan_grid_max);
      G.ensure_mig_out(262144);
      G.ensure_mig_in(262144);

      size_t loc_c = 0, loc_n = 0;
      for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
        const int s = owned_slab(g, ls);
        const auto& HS = h_stage[s];
        const size_t sc = HS.x.size();
        const size_t cap = compute_slab_cap(sc);
        const size_t ncap = compute_node_cap(cap, leaf_limit);
        auto& sl = G.slabs[ls];
        sl.global_id = s;
        sl.ns = sc;
        sl.alloc(cap, mass_uniform, soft_uniform, G.device_bytes);
        G.trees[ls].alloc(cap, ncap, soft_uniform, lazy_tree_enabled, split_force_enabled, G.device_bytes);
        loc_c = std::max(loc_c, cap);
        loc_n = std::max(loc_n, ncap);

        if (sc > 0) {
          check(cudaMemcpyAsync(sl.d_x.p, HS.x.data(), sc * sizeof(double3), cudaMemcpyHostToDevice, G.compute_stream));
          check(cudaMemcpyAsync(sl.d_v.p, HS.v.data(), sc * sizeof(float3), cudaMemcpyHostToDevice, G.compute_stream));
          check(cudaMemcpyAsync(sl.d_orig_idx.p, HS.orig_idx.data(), sc * sizeof(uint32_t), cudaMemcpyHostToDevice, G.compute_stream));
          if (!mass_uniform)
            check(cudaMemcpyAsync(sl.d_mass.p, HS.mass.data(), sc * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
          if (!soft_uniform)
            check(cudaMemcpyAsync(sl.d_soft.p, HS.soft.data(), sc * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
        }
        check(cudaMemsetAsync(sl.d_mesh_acc.p, 0, cap * sizeof(float3), G.compute_stream));
        check(cudaMemsetAsync(sl.d_old_acc.p, 0, cap * sizeof(float), G.compute_stream));
        check(cudaMemsetAsync(sl.d_bins.p, 1, cap * sizeof(unsigned char), G.compute_stream));
      }
      const int left_s = (owned_slab(g, 0) + num_slabs - 1) % num_slabs;
      const int right_s = (owned_slab(g, G.slabs_per_gpu - 1) + 1) % num_slabs;
      const size_t lc = compute_slab_cap(slab_counts[left_s]);
      const size_t rc = compute_slab_cap(slab_counts[right_s]);
      const size_t gh_c = (num_gpus == 1) ? 1 : std::max(lc, rc);
      const size_t gh_n = (num_gpus == 1) ? size_t(slab_roots_count) : compute_node_cap(gh_c, leaf_limit);
#pragma omp critical(phase_pool_alloc)
      {
        G.ensure_phase_pool(loc_c, loc_n, gh_c, gh_n);
        if (split_force_enabled) {
          G.ensure_split_bufs(size_t(split_batch_warps), 1);
          G.ensure_split_slices((loc_c + 31) / 32 + 2 * slab_roots_count);
        }
      }
      check(cudaStreamSynchronize(G.compute_stream));
    }
    // Free host staging immediately!
    for (int s = 0; s < num_slabs; ++s) {
      std::vector<double3>().swap(h_stage[s].x);
      std::vector<float3>().swap(h_stage[s].v);
      std::vector<float>().swap(h_stage[s].mass);
      std::vector<float>().swap(h_stage[s].soft);
      std::vector<uint32_t>().swap(h_stage[s].orig_idx);
    }
    check(cudaSetDevice(0));
    const double load_s = wall() - import_start;

    SwiftDistPm dist_pm;
    if (pm_dist_enabled) {
      std::vector<int> devs(num_gpus);
      std::vector<cudaStream_t> streams(num_gpus);
      std::vector<std::vector<int>> gpu_x_planes(num_gpus);
      for (int g = 0; g < num_gpus; ++g) {
        devs[g] = gpus[g].dev;
        streams[g] = gpus[g].compute_stream;
        for (int ls = 0; ls < gpus[g].slabs_per_gpu; ++ls) {
          const int s = owned_slab(g, ls);
          const int px_lo = int(std::floor(double(s) * pm_side / double(num_slabs))) - 1;
          const int px_hi = int(std::ceil(double(s + 1) * pm_side / double(num_slabs))) + 1;
          for (int px = px_lo; px <= px_hi; ++px) {
            int gx = px % pm_side;
            if (gx < 0) gx += pm_side;
            gpu_x_planes[g].push_back(gx);
          }
        }
      }
      dist_pm.init(devs, streams, can_peer, pm_side, gpu_x_planes);
      for (int g = 0; g < num_gpus; ++g) {
        gpus[g].device_bytes += dist_pm.device_bytes(g);
      }
    }

    std::vector<double> knots;
    knots.push_back(h.initial_scale);
    for (double sa : snap_scales) knots.push_back(sa);
    knots.push_back(a_end);

    const double log_span = comoving ? log(knots[1] / knots[0]) : (knots[1] - knots[0]);
    const float soft = has_soft_in
        ? float(double(first_soft_plummer) * kernel_gravity_softening_plummer_equivalent)
        : float(h.plummer_softening * kernel_gravity_softening_plummer_equivalent);
    const double rho_crit0 = 3.0 * cosmo_h0 * cosmo_h0 / (8.0 * acos(-1.0) * double(grav_G));
    const double d_dm = cbrt(min_mass / (cosmo_ocdm * rho_crit0));
    const double r_s = 1.25 * h.box / double(pm_side);
    const double3 ext_c_perm = permute_in(ext_c_orig, slab_axis);
    Physics p{knots[0], log_span,
        float(h.box), periodic ? float(r_s) : 1e30f, periodic ? float(4.5 * r_s) : 1e30f, grav_G,
        timeline / pm_intervals, timeline / short_intervals, step_factor, mac,
        dt_max_val, std::min(r_s, d_dm), max_rms_fac, dynamic_pm,
        cosmo_h0, cosmo_om, cosmo_ode, cosmo_or, cosmo_ok, cosmo_w0, cosmo_wa,
        theta_val, grav_eta_val,
        periodic, comoving,
        ext_M_val, ext_a_val, float(ext_c_perm.x), float(ext_c_perm.y), float(ext_c_perm.z)};
    if (comoving) {
      const double Ha0 = p.H(h.initial_scale);
      const double Ha1 = p.H(a_end);
      if (!std::isfinite(Ha0) || Ha0 <= 0.0 || !std::isfinite(Ha1) || Ha1 <= 0.0) {
        throw std::runtime_error("Invalid cosmology H(a) <= 0 or NaN");
      }
    }

    probe_header gh{};
    gh.top[0] = gh.top[1] = gh.top[2] = 12;
    for (int axis = 0; axis < 3; ++axis) {
      gh.width[axis] = h.box / 12.0;
      gh.inv_width[axis] = 1.0 / gh.width[axis];
    }
    gh.particles = N;
    gh.leaf_limit = leaf_limit;

    // Report MEMORY_PLAN & GPU_INPUT
    std::vector<size_t> gpu_part_counts(num_gpus, 0), gpu_dev_bytes(num_gpus, 0), gpu_free_vram(num_gpus, 0);
    size_t total_vram_0 = 0, sum_dev_bytes = 0, min_free_vram = ~size_t(0);
    cudaDeviceProp prop0{};
    for (int g = 0; g < num_gpus; ++g) {
      check(cudaSetDevice(g));
      size_t fv_b = 0, tv_b = 0;
      check(cudaMemGetInfo(&fv_b, &tv_b));
      if (g == 0) {
        total_vram_0 = tv_b;
        check(cudaGetDeviceProperties(&prop0, 0));
      }
      for (int ls = 0; ls < gpus[g].slabs_per_gpu; ++ls) gpu_part_counts[g] += gpus[g].slabs[ls].ns;
      gpu_dev_bytes[g] = gpus[g].device_bytes;
      gpu_free_vram[g] = fv_b;
      sum_dev_bytes += gpus[g].device_bytes;
      min_free_vram = std::min(min_free_vram, fv_b);
    }
    check(cudaSetDevice(0));

    auto print_vec = [&](const auto& vec) {
      std::cout << "[";
      for (size_t i = 0; i < vec.size(); ++i) std::cout << (i ? "," : "") << vec[i];
      std::cout << "]";
    };
    std::cout << "MEMORY_PLAN {\"mode\":\"multi_gpu_slab_resident\",\"num_gpus\":" << num_gpus
              << ",\"slabs\":" << num_slabs << ",\"slab_axis\":" << slab_axis
              << ",\"slab_offset\":" << slab_offset
              << ",\"balance_mode\":\"" << balance_mode << "\",\"particles\":" << N
              << ",\"max_initial_slab_n\":" << max_initial_slab_n
              << ",\"per_gpu_slabs\":";
    print_vec(gpu_slab_count);
    std::cout << ",\"per_gpu_particles\":";
    print_vec(gpu_part_counts);
    std::cout << ",\"per_gpu_device_bytes\":";
    print_vec(gpu_dev_bytes);
    std::cout << ",\"per_gpu_free_vram_bytes\":";
    print_vec(gpu_free_vram);
    std::cout << ",\"can_peer\":[";
    for (int i = 0; i < num_gpus; ++i) {
      if (i) std::cout << ",";
      print_vec(can_peer[i]);
    }
    std::cout << "],\"p2p_mode\":\"" << p2p_mode
              << "\",\"uniform_mass\":" << (mass_uniform ? "true" : "false")
              << ",\"uniform_soft\":" << (soft_uniform ? "true" : "false")
              << ",\"device_buffer_bytes\":" << sum_dev_bytes
              << ",\"free_vram_bytes\":" << min_free_vram
              << ",\"total_vram_bytes\":" << total_vram_0 << "}" << std::endl;

    std::cout << std::setprecision(12)
              << "GPU_INPUT {\"particles\":" << N << ",\"num_gpus\":" << num_gpus
              << ",\"periodic\":" << (periodic ? "true" : "false")
              << ",\"comoving\":" << (comoving ? "true" : "false")
              << ",\"pm_grid\":" << (periodic ? pm_side : 0)
              << ",\"dynamic_pm\":" << (dynamic_pm ? "true" : "false")
              << ",\"grav_G\":" << p.G << ",\"pm_intervals\":" << pm_intervals
              << ",\"short_cap_intervals\":" << short_intervals
              << ",\"short_step_factor\":" << step_factor
              << ",\"mac_tolerance\":" << mac
              << ",\"theta\":" << p.theta << ",\"grav_eta\":" << p.grav_eta
              << ",\"hubble0\":" << p.hubble0 << ",\"omega_m\":" << p.omega_m
              << ",\"omega_de\":" << p.omega_de << ",\"omega_r\":" << p.omega_r
              << ",\"omega_k\":" << p.omega_k << ",\"w0\":" << p.w0 << ",\"wa\":" << p.wa
              << ",\"force_check_n\":" << force_check_n
              << ",\"lightcone\":" << (lc_enabled ? "true" : "false")
              << ",\"ext_pot_M\":" << p.ext_M
              << ",\"a_begin\":" << h.initial_scale << ",\"a_end\":" << a_end
              << ",\"warp_m2l\":" << (warp_m2l ? "true" : "false")
              << ",\"fast_split\":" << (fast_split ? "true" : "false")
              << ",\"split_force\":" << (split_force_enabled ? "true" : "false")
              << ",\"pm_dist\":" << (pm_dist_enabled ? "true" : "false")
              << ",\"domain_mode\":\"1d_slab\""
              << ",\"split_batch_warps\":" << split_batch_warps
              << ",\"tree_leaf_limit\":" << gh.leaf_limit
              << ",\"softening_internal_mpc\":" << soft << ",\"box_mpc\":" << h.box
              << ",\"prepare_and_load_s\":" << load_s
              << ",\"device\":\"" << prop0.name << "\"}" << std::endl;

    // Pinned host PM reduction buffers (legacy fallback only when num_gpus > 1 and !pm_dist_enabled)
    std::vector<PinnedHostBuf> h_pm_dens(num_gpus);
    PinnedHostBuf h_pm_field;
    if (periodic && num_gpus > 1 && !pm_dist_enabled) {
      for (int g = 0; g < num_gpus; ++g) h_pm_dens[g].alloc(pm_points * sizeof(double));
      h_pm_field.alloc(pm_points * sizeof(double4));
    }
    const size_t bounce_alloc_bytes =
        (num_gpus > 1) ? (GpuCtx::bounce_slots * GpuCtx::bounce_chunk_bytes) : (80ULL << 20);
    for (int g = 0; g < num_gpus; ++g) {
      gpus[g].h_bounce.alloc(bounce_alloc_bytes, cudaHostAllocPortable);
    }

    // Optional built-in GPU force check (Ewald periodic or direct isolated) & lightcone state (1 GPU)
    size_t aux_dev_bytes = 0;
    const int fc_target_cap = int(std::min<uint64_t>(uint64_t(force_check_n), N));
    std::vector<EwaldMode> h_ewald_modes;
    DevPtr<EwaldMode> d_ewald_modes;
    DevPtr<double> d_ewald_C, d_ewald_S;
    DevPtr<double3> d_fc_tgt_x, d_fc_short_ref, d_fc_total_ref;
    DevPtr<float> d_fc_tgt_h;
    DevPtr<int> d_fc_lids, d_fc_tgt_slab, d_fc_tgt_lid;
    DevPtr<float4> d_fc_gpu_short;
    DevPtr<float3> d_fc_gpu_mesh;
    if (fc_target_cap > 0) {
      check(cudaSetDevice(0));
      d_fc_tgt_x.alloc(fc_target_cap, aux_dev_bytes);
      d_fc_tgt_h.alloc(fc_target_cap, aux_dev_bytes);
      d_fc_lids.alloc(fc_target_cap, aux_dev_bytes);
      d_fc_tgt_slab.alloc(fc_target_cap, aux_dev_bytes);
      d_fc_tgt_lid.alloc(fc_target_cap, aux_dev_bytes);
      d_fc_gpu_short.alloc(fc_target_cap, aux_dev_bytes);
      d_fc_gpu_mesh.alloc(fc_target_cap, aux_dev_bytes);
      d_fc_short_ref.alloc(fc_target_cap, aux_dev_bytes);
      d_fc_total_ref.alloc(fc_target_cap, aux_dev_bytes);
      if (periodic) {
        h_ewald_modes = build_ewald_modes(h.box);
        d_ewald_modes.alloc(h_ewald_modes.size(), aux_dev_bytes);
        d_ewald_C.alloc(h_ewald_modes.size(), aux_dev_bytes);
        d_ewald_S.alloc(h_ewald_modes.size(), aux_dev_bytes);
        check(cudaMemcpy(d_ewald_modes.p, h_ewald_modes.data(),
                         h_ewald_modes.size() * sizeof(EwaldMode), cudaMemcpyHostToDevice));
      }
    }

    FILE* lc_fp = nullptr;
    uint64_t lc_total_hits = 0;
    double lc_r_max = 0.0;
    const double3 lc_obs_perm = permute_in(lc_obs, slab_axis);
    DevPtr<LightconeHit> d_lc_hits;
    DevPtr<unsigned int> d_lc_count;
    DevPtr<double3> d_lc_shifts;
    size_t lc_hit_cap = 0, lc_shift_cap = 0;
    AsyncLightconeWriter lc_writer;
    if (lc_enabled) {
      check(cudaSetDevice(0));
      const double chi_max_evo = comoving_distance_mpc(h.initial_scale, lc_a_obs, p);
      lc_r_max = (lc_r_max_req > 0.0) ? lc_r_max_req : chi_max_evo;
      lc_fp = fopen(lc_out_env, "wb");
      if (!lc_fp) throw std::runtime_error(std::string("Cannot open SWIFT_LIGHTCONE_OUT: ") + lc_out_env);
      const LightconeHeader blank_hdr{{'S','W','L','C','0','0','0','1'}, 0,
                                      {lc_obs.x, lc_obs.y, lc_obs.z},
                                      h.initial_scale, a_end, lc_a_obs, lc_r_max, h.box};
      write_bin(lc_fp, &blank_hdr, 1);
      lc_writer.start(lc_fp);
      lc_hit_cap = 262144;
      d_lc_hits.alloc(lc_hit_cap, aux_dev_bytes);
      d_lc_count.alloc(1, aux_dev_bytes);
    }

    // Shared host coordination state per step
    std::array<std::vector<DevMigrant>, num_slabs> h_mig_out, h_mig_in;
    std::array<size_t, num_slabs> slab_active{}, slab_migrants{};
    std::array<Tick, num_slabs> slab_inactive_min{}, slab_active_next{};
    std::array<double, num_slabs> slab_v2_sum{};
    std::array<float, num_slabs> slab_max_v{};
    std::array<unsigned int, num_slabs> slab_nodes{};
    std::array<unsigned int, num_slabs> slab_invalid{};
    std::array<int, num_slabs> slab_builds{}, slab_packed_stale{}, slab_ns_changed{};
    std::array<double, num_slabs> slab_force_ms{}, slab_kick_ms{}, slab_ghost_ms{}, slab_xfer_ms{};
    std::vector<int> phase_pool_need_grow(num_gpus, 0);
    bool all_slabs_built = false;
    double cum_max_disp = 0.0;
    int lazy_steps_since_rebuild = 0;
    uint64_t total_lazy_steps = 0;
    uint64_t total_full_rebuilds = 0;
    uint64_t split_batches_run = 0, split_overflow_retries = 0;
    std::vector<float4> out_force0_f4;
    if (force0_out_path) out_force0_f4.resize(N);

    // Final export host outputs (allocated only on is_final)
    std::vector<double3> out_x;
    std::vector<float3> out_v, out_short_f3, out_long_f3;
    std::vector<float> out_pot;
    std::vector<float4> out_short_f4, out_long_f4;
    const double potential_norm = periodic
        ? (4.0 * acos(-1.0) * total_mass * p.split * p.split / (h.box * h.box * h.box))
        : 0.0;

    size_t sanity_bad = 0;
    double sanity_v2_sum = 0.0, sanity_pot_sum = 0.0;
    float sanity_pot_min = FLT_MAX, sanity_pot_max = -FLT_MAX;
    std::vector<double*> d_energy_accum(num_gpus, nullptr);
    std::vector<std::array<double, 10>> gpu_energy(num_gpus);
    if (energy_log) {
      for (int g = 0; g < num_gpus; ++g) {
        check(cudaSetDevice(g));
        check(cudaMalloc(&d_energy_accum[g], 10 * sizeof(double)));
      }
      check(cudaSetDevice(0));
    }

    Tick previous = 0, ti = 0;
    Tick next_pm_boundary = 0;
    Tick prev_pm_span = 0, current_pm_span = p.pm_span, next_pm_span = p.pm_span;
    Tick short_span_cap = p.short_span_cap;
    float dt_max_rms = dt_displacement_default;
    double pm_kick = 0.0;
    unsigned int pm_updates = 0;
    double last_pm_span_scale = log_span / double(pm_intervals);
    float last_dt_max_rms = dt_displacement_default;
    size_t peak_slab_seen = max_initial_slab_n;
    bool midpoint_written = false;

    struct PendingSnap {
      bool active = false;
      size_t index = 0;
      double a = 0.0;
      std::string path;
      double write_s = 0.0;
    };
    std::thread snap_writer;
    std::exception_ptr snap_writer_err = nullptr;
    PendingSnap pending_snap;
    auto join_snap_writer = [&]() {
      if (snap_writer.joinable()) snap_writer.join();
      if (snap_writer_err) {
        std::exception_ptr err = snap_writer_err;
        snap_writer_err = nullptr;
        std::rethrow_exception(err);
      }
      if (pending_snap.active) {
        std::cout << std::setprecision(12)
                  << "SNAPSHOT_WRITTEN {\"index\":" << pending_snap.index
                  << ",\"a\":" << pending_snap.a
                  << ",\"path\":\"" << pending_snap.path
                  << "\",\"write_s\":" << pending_snap.write_s << "}" << std::endl;
        pending_snap.active = false;
      }
    };
    struct WriterJoinGuard {
      std::thread& t;
      ~WriterJoinGuard() {
        if (t.joinable()) t.join();
      }
    } writer_guard{snap_writer};

    double evo_start = wall();

    // Helper: grow slab cleanly via temporary host staging (prevents 2x VRAM spike)
    auto grow_resident_slab = [&](GpuCtx& G, int ls, size_t need_n) {
      auto& S = G.slabs[ls];
      auto& T = G.trees[ls];
      const size_t new_cap = compute_slab_cap(need_n);
      const size_t new_ncap = compute_node_cap(new_cap, leaf_limit);
      std::vector<double3> sx(S.ns);
      std::vector<float3> sv(S.ns), sm(S.ns);
      std::vector<float> so(S.ns), smass, ssoft;
      std::vector<unsigned char> sb(S.ns);
      std::vector<uint32_t> sid(S.ns);
      if (S.ns > 0) {
        check(cudaMemcpyAsync(sx.data(), S.d_x.p, S.ns * sizeof(double3), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaMemcpyAsync(sv.data(), S.d_v.p, S.ns * sizeof(float3), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaMemcpyAsync(sm.data(), S.d_mesh_acc.p, S.ns * sizeof(float3), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaMemcpyAsync(so.data(), S.d_old_acc.p, S.ns * sizeof(float), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaMemcpyAsync(sb.data(), S.d_bins.p, S.ns * sizeof(unsigned char), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaMemcpyAsync(sid.data(), S.d_orig_idx.p, S.ns * sizeof(uint32_t), cudaMemcpyDeviceToHost, G.compute_stream));
        if (!mass_uniform) {
          smass.resize(S.ns);
          check(cudaMemcpyAsync(smass.data(), S.d_mass.p, S.ns * sizeof(float), cudaMemcpyDeviceToHost, G.compute_stream));
        }
        if (!soft_uniform) {
          ssoft.resize(S.ns);
          check(cudaMemcpyAsync(ssoft.data(), S.d_soft.p, S.ns * sizeof(float), cudaMemcpyDeviceToHost, G.compute_stream));
        }
        check(cudaStreamSynchronize(G.compute_stream));
      }
      S.free_all(G.device_bytes);
      T.free_all(G.device_bytes);
      S.alloc(new_cap, mass_uniform, soft_uniform, G.device_bytes);
      T.alloc(new_cap, new_ncap, soft_uniform, lazy_tree_enabled, split_force_enabled, G.device_bytes);
      if (S.ns > 0) {
        check(cudaMemcpyAsync(S.d_x.p, sx.data(), S.ns * sizeof(double3), cudaMemcpyHostToDevice, G.compute_stream));
        check(cudaMemcpyAsync(S.d_v.p, sv.data(), S.ns * sizeof(float3), cudaMemcpyHostToDevice, G.compute_stream));
        check(cudaMemcpyAsync(S.d_mesh_acc.p, sm.data(), S.ns * sizeof(float3), cudaMemcpyHostToDevice, G.compute_stream));
        check(cudaMemcpyAsync(S.d_old_acc.p, so.data(), S.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
        check(cudaMemcpyAsync(S.d_bins.p, sb.data(), S.ns * sizeof(unsigned char), cudaMemcpyHostToDevice, G.compute_stream));
        check(cudaMemcpyAsync(S.d_orig_idx.p, sid.data(), S.ns * sizeof(uint32_t), cudaMemcpyHostToDevice, G.compute_stream));
        if (!mass_uniform)
          check(cudaMemcpyAsync(S.d_mass.p, smass.data(), S.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
        if (!soft_uniform)
          check(cudaMemcpyAsync(S.d_soft.p, ssoft.data(), S.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
        check(cudaStreamSynchronize(G.compute_stream));
      }
    };

    auto grow_tree_nodes = [&](GpuCtx& G, int ls, size_t min_ncap) {
      auto& T = G.trees[ls];
      const size_t pcap = T.part_cap;
      T.free_all(G.device_bytes);
      T.alloc(pcap, min_ncap, soft_uniform, lazy_tree_enabled, split_force_enabled, G.device_bytes);
    };

    auto build_slab_tree = [&](GpuCtx& G, int ls, bool compute_nvel) {
      auto& S = G.slabs[ls];
      auto& T = G.trees[ls];
      const int k = S.global_id;
      const int nk = int(S.ns);
      if (nk == 0) {
        zero_empty_roots<<<blocks(slab_roots_count), team, 0, G.compute_stream>>>(
            T.force_nodes.p, T.escape.p, T.first_child.p, slab_roots_count);
        if (split_force_enabled) {
          pack_split_nodes_kernel<<<blocks(slab_roots_count), team, 0, G.compute_stream>>>(
              T.force_nodes.p, T.first_child.p, T.node_meta.p, T.compact_m.p, slab_roots_count);
        }
        T.nodes = slab_roots_count;
        return;
      }
      for (int attempt = 0; attempt < 4; ++attempt) {
#pragma omp critical(phase_pool_alloc)
        G.ensure_phase_pool(std::max(G.pool_local_cap, S.cap), std::max(G.pool_local_nodes, T.node_cap),
                            G.pool_ghost_cap, G.pool_ghost_nodes);
        check(cudaMemsetAsync(G.d_tree_invalid.p, 0, sizeof(int), G.compute_stream));
        const int p_grid = blocks(nk);
        key_particles_fast_slab<<<p_grid, team, 0, G.compute_stream>>>(
            S.d_x.p, nk, k, gh, G.d_key_tmp, G.d_ord_tmp);
        size_t sb = G.sort_bytes;
        check(cub::DeviceRadixSort::SortPairs(
            G.d_sort_storage, sb, G.d_key_tmp, G.d_keys_sorted,
            G.d_ord_tmp, T.order.p, nk, 0, 59, G.compute_stream));
        extract_home_cells<<<p_grid, team, 0, G.compute_stream>>>(G.d_keys_sorted, T.home_cell.p, nk);
        roots<<<blocks(slab_roots_count), team, 0, G.compute_stream>>>(
            G.d_keys_sorted, nk, T.geometry.p, slab_roots_count);
        tree_offsets_begin<<<1, 1, 0, G.compute_stream>>>(T.levels.p, T.node_count.p, slab_roots_count);

        const int n_grid = blocks(int(T.node_cap));
        for (int depth = 0; depth < tree_levels; ++depth) {
          split<<<n_grid, team, 0, G.compute_stream>>>(
              G.d_keys_sorted, T.geometry.p, 0, 0, depth, gh.leaf_limit,
              T.node_count.p, int(T.node_cap), G.d_tree_invalid.p, T.levels.p);
          tree_offsets_end<<<1, 1, 0, G.compute_stream>>>(T.levels.p, T.node_count.p, depth, int(T.node_cap));
          tree_links<<<n_grid, team, 0, G.compute_stream>>>(
              T.geometry.p, T.escape.p, T.levels.p, depth, T.first_child.p);
        }
        unsigned int nc = 0;
        int inv_cap = 0;
        check(cudaMemcpyAsync(&nc, T.node_count.p, sizeof(unsigned int), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaMemcpyAsync(&inv_cap, G.d_tree_invalid.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaStreamSynchronize(G.compute_stream));
        if (inv_cap != 0 || nc >= T.node_cap) {
          if (attempt < 3) {
            const size_t next_nc = std::max(T.node_cap * 3 / 2, size_t(nc) + 131072);
            grow_tree_nodes(G, ls, next_nc);
            continue;
          }
          throw std::runtime_error("Slab tree split overflow on slab " + std::to_string(k));
        }

        check_tree_limit<<<n_grid, team, 0, G.compute_stream>>>(
            T.geometry.p, T.levels.p, tree_levels, gh.leaf_limit, G.d_tree_invalid.p);
        tree_links<<<n_grid, team, 0, G.compute_stream>>>(
            T.geometry.p, T.escape.p, T.levels.p, tree_levels, T.first_child.p);
        for (int depth = tree_levels; depth >= 0; --depth) {
          multipoles_fast_slab<<<n_grid, team, 0, G.compute_stream>>>(
              T.geometry.p, T.force_nodes.p, G.d_node_com, S.d_x.p,
              mass_uniform ? nullptr : S.d_mass.p, first_mass, soft,
              T.order.p, depth, k, gh, T.levels.p,
              soft_uniform ? nullptr : S.d_soft.p);
        }
        gather_sources_slab<<<p_grid, team, 0, G.compute_stream>>>(
            S.d_x.p, mass_uniform ? nullptr : S.d_mass.p, first_mass,
            soft_uniform ? nullptr : S.d_soft.p,
            T.order.p, T.packed_sources.p, T.packed_soft.p, nk);
        if (compute_nvel) {
          for (int depth = tree_levels; depth >= 0; --depth) {
            compute_node_vel_slab_kernel<<<n_grid, team, 0, G.compute_stream>>>(
                T.geometry.p, T.force_nodes.p, S.d_v.p,
                mass_uniform ? nullptr : S.d_mass.p, first_mass,
                T.order.p, depth, T.levels.p, T.node_vel_max.p);
          }
        }
        if (split_force_enabled) {
          pack_split_nodes_kernel<<<blocks(int(nc)), team, 0, G.compute_stream>>>(
              T.force_nodes.p, T.first_child.p, T.node_meta.p, T.compact_m.p, int(nc));
        }
        check(cudaGetLastError());
        int inv_leaf = 0;
        check(cudaMemcpyAsync(&inv_leaf, G.d_tree_invalid.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaStreamSynchronize(G.compute_stream));
        if (inv_leaf != 0) {
          throw std::runtime_error("Slab tree depth>14 leaf limit failed: slab=" + std::to_string(k) +
                                   " nodes=" + std::to_string(nc));
        }
        T.nodes = nc;
        return;
      }
    };

    auto copy_ghost_tree = [&](GpuCtx& dst, GhostTreeView& dst_ghost,
                               const GpuCtx& src, int src_ls, bool use_src_left_stream) {
      const size_t nk = src.slabs[src_ls].ns;
      const size_t nc = src.trees[src_ls].nodes;
      const auto& st = src.trees[src_ls];
      struct CopySpan {
        void* dst;
        const void* src;
        size_t bytes;
      };
      const CopySpan spans[8] = {
          {dst_ghost.packed_sources, st.packed_sources.p, nk * sizeof(float4)},
          {dst_ghost.geometry, st.geometry.p, nc * sizeof(Geometry)},
          {dst_ghost.force_nodes, st.force_nodes.p, split_force_enabled ? 0 : nc * sizeof(ForceNode)},
          {dst_ghost.escape, st.escape.p, split_force_enabled ? 0 : nc * sizeof(int)},
          {dst_ghost.first_child, st.first_child.p, split_force_enabled ? 0 : nc * sizeof(int)},
          {dst_ghost.packed_soft, st.packed_soft.p, !soft_uniform ? nk * sizeof(float) : 0},
          {dst_ghost.node_meta, st.node_meta.p, split_force_enabled ? nc * sizeof(NodeMeta) : 0},
          {dst_ghost.compact_m, st.compact_m.p, split_force_enabled ? nc * sizeof(CompactMultipole) : 0},
      };
      if (can_peer[dst.dev][src.dev]) {
        check(cudaSetDevice(dst.dev));
        for (const auto& sp : spans) {
          if (sp.bytes > 0) {
            check(cudaMemcpyPeerAsync(sp.dst, dst.dev, sp.src, src.dev, sp.bytes, dst.copy_stream));
          }
        }
        return;
      }

      const size_t ring_bytes = GpuCtx::bounce_slots * GpuCtx::bounce_chunk_bytes;
      dst.h_bounce.alloc(ring_bytes, cudaHostAllocPortable);
      cudaStream_t s_stream = use_src_left_stream ? src.d2h_left_stream : src.d2h_right_stream;
      const cudaEvent_t* s_evs = use_src_left_stream ? src.ev_d2h_left : src.ev_d2h_right;
      int slot = 0;
      for (const auto& sp : spans) {
        if (sp.bytes == 0) continue;
        const char* s_bytes = static_cast<const char*>(sp.src);
        char* d_bytes = static_cast<char*>(sp.dst);
        for (size_t off = 0; off < sp.bytes;) {
          const size_t chunk = std::min(GpuCtx::bounce_chunk_bytes, sp.bytes - off);
          char* h_slot = static_cast<char*>(dst.h_bounce.p) + size_t(slot) * GpuCtx::bounce_chunk_bytes;
          check(cudaSetDevice(src.dev));
          check(cudaStreamWaitEvent(s_stream, dst.ev_h2d_done[slot], 0));
          check(cudaMemcpyAsync(h_slot, s_bytes + off, chunk, cudaMemcpyDeviceToHost, s_stream));
          check(cudaEventRecord(s_evs[slot], s_stream));
          check(cudaSetDevice(dst.dev));
          check(cudaStreamWaitEvent(dst.copy_stream, s_evs[slot], 0));
          check(cudaMemcpyAsync(d_bytes + off, h_slot, chunk, cudaMemcpyHostToDevice, dst.copy_stream));
          check(cudaEventRecord(dst.ev_h2d_done[slot], dst.copy_stream));
          off += chunk;
          slot = (slot + 1) % GpuCtx::bounce_slots;
        }
      }
      check(cudaSetDevice(dst.dev));
    };

    int global_iteration = 0;
    for (size_t seg = 0; seg + 1 < knots.size(); ++seg) {
      const double seg_a_begin = knots[seg];
      const double seg_a_end = knots[seg + 1];
      const bool is_last_seg = (seg + 1 == knots.size() - 1);
      const double seg_log_span = comoving ? log(seg_a_end / seg_a_begin) : (seg_a_end - seg_a_begin);
      p.a_begin = seg_a_begin;
      p.log_span = seg_log_span;
      previous = 0;
      ti = 0;
      next_pm_boundary = 0;
      prev_pm_span = 0;
      current_pm_span = p.pm_span;
      next_pm_span = p.pm_span;
      short_span_cap = p.short_span_cap;
      dt_max_rms = dt_displacement_default;
      pm_kick = 0.0;
      last_pm_span_scale = seg_log_span / double(pm_intervals);
      last_dt_max_rms = dt_displacement_default;
      all_slabs_built = false;
      cum_max_disp = 0.0;
      lazy_steps_since_rebuild = 0;
      slab_packed_stale.fill(0);
      slab_ns_changed.fill(0);

      if (seg > 0) {
#pragma omp parallel num_threads(num_gpus)
        {
          const int g = omp_get_thread_num();
          auto& G = gpus[g];
          check(cudaSetDevice(G.dev));
          for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
            auto& sl = G.slabs[ls];
            check(cudaMemsetAsync(sl.d_mesh_acc.p, 0, sl.cap * sizeof(float3), G.compute_stream));
            check(cudaMemsetAsync(sl.d_old_acc.p, 0, sl.cap * sizeof(float), G.compute_stream));
            check(cudaMemsetAsync(sl.d_bins.p, 1, sl.cap * sizeof(unsigned char), G.compute_stream));
          }
          check(cudaStreamSynchronize(G.compute_stream));
        }
        check(cudaSetDevice(0));
      }

      for (int iteration = 0;; ++iteration, ++global_iteration) {
        if (iteration > max_steps_guard)
          throw std::runtime_error("Exceeded max_steps_guard");

        const double step_start = wall();
        const bool is_pm_step = periodic ? (ti == 0 || ti == next_pm_boundary) : (ti == 0);
        const bool is_final = (ti == timeline);
        const bool is_run_final = is_final && is_last_seg;
        const bool midpoint = midpoint_path && ti == timeline / 2;
        if (midpoint && !is_pm_step) throw std::runtime_error("Midpoint is not a PM boundary");
        const double closing_pm_kick = (periodic && midpoint) ? p.factors(ti - current_pm_span / 2, ti, false) : 0.0;
        const double drift_fac = (ti > 0) ? p.factors(previous, ti, true) : 0.0;
        const double a = p.scale(ti);
        const double H_cur = p.H(a);
        const bool need_vel_sum = (periodic && is_pm_step && !is_final && dynamic_pm);
        const bool do_fc_now = (fc_target_cap > 0) && ((ti == 0 && seg == 0) || is_run_final);
        std::array<int, num_slabs> fc_counts{}, fc_offsets{};
        int n_fc_tgt = 0;

        const double wx = h.box / double(num_slabs);
        const double max_safe_disp = periodic
            ? std::min(lazy_disp_frac * r_s, 0.35 * std::max(0.0, wx - double(p.cutoff)))
            : 0.0;
        const bool eligible_lazy = lazy_tree_enabled && periodic && (ti > 0) && !is_pm_step &&
            !is_final && !midpoint && !do_fc_now && all_slabs_built &&
            (lazy_steps_since_rebuild < lazy_max_steps) && (max_safe_disp > 0.0);
        bool do_lazy_step = false;
        bool force_rebuild_all = is_pm_step || (ti == 0);
        double step_max_disp = 0.0;
        double step_slice_ms = 0.0, step_phase1_ms = 0.0, step_m2l_ms = 0.0, step_p2p_ms = 0.0;
        uint64_t step_p2p_chunks = 0, step_m2l_chunks = 0, step_p2p_warps = 0, step_m2l_warps = 0, step_tot_warps = 0;
        int any_skipped_build = 0;
        slab_ns_changed.fill(0);

        if (is_final || midpoint) {
          join_snap_writer();
          if (!skip_output_write) {
            out_x.resize(N);
            out_v.resize(N);
            out_pot.resize(N);
          }
          if (is_run_final && force_comp_path) {
            out_short_f4.resize(N);
            out_long_f4.resize(N);
          }
          if (is_run_final && force3_out_path) {
            out_short_f3.resize(N);
            out_long_f3.resize(N);
          }
        }

        double drift_ms = 0.0, pm_ms = 0.0, tree_ms = 0.0, force_ms = 0.0;
        double kick_ms = 0.0, ghost_ms = 0.0, export_ms = 0.0;
        size_t total_migrants = 0;

        // Single OpenMP parallel team across GPUs for Steps A, B, C, D
#pragma omp parallel num_threads(num_gpus)
        {
          const int g = omp_get_thread_num();
          auto& G = gpus[g];
          check(cudaSetDevice(G.dev));

          // ================= STEP A: Drift + Migrate + Active/v2 scan =================
          const double t_a0 = wall();
          if (ti > 0) {
            if (lc_enabled && g == 0) {
              const double a_prev = p.scale(previous);
              if (a_prev < lc_a_obs) {
                const double a_step_end = std::min(a, lc_a_obs);
                const double tau_scale = (a > a_prev) ? std::min(1.0, (a_step_end - a_prev) / (a - a_prev)) : 1.0;
                const double step_drift_fac = drift_fac * tau_scale;
                const double chi_0 = comoving_distance_mpc(a_prev, lc_a_obs, p);
                const double chi_1 = comoving_distance_mpc(a_step_end, lc_a_obs, p);
                const double r_hi = std::min(chi_0, lc_r_max);
                if (chi_0 > chi_1 && chi_1 <= lc_r_max) {
                  std::vector<double3> shifts;
                  if (!lc_replicate) {
                    shifts.push_back({0.0, 0.0, 0.0});
                  } else {
                    const double L = h.box;
                    const double m_box = 0.5 * L;
                    auto axis_rng = [&](double obs_q) -> std::pair<int, int> {
                      return {int(std::floor((obs_q - r_hi - m_box) / L)) - 1,
                              int(std::ceil((obs_q + r_hi + m_box) / L))};
                    };
                    const auto [nx0, nx1] = axis_rng(lc_obs_perm.x);
                    const auto [ny0, ny1] = axis_rng(lc_obs_perm.y);
                    const auto [nz0, nz1] = axis_rng(lc_obs_perm.z);
                    for (int nx = nx0; nx <= nx1; ++nx) {
                      const double bx0 = double(nx) * L - m_box, bx1 = double(nx + 1) * L + m_box;
                      const double dx_min = (lc_obs_perm.x < bx0) ? (bx0 - lc_obs_perm.x) : ((lc_obs_perm.x > bx1) ? (lc_obs_perm.x - bx1) : 0.0);
                      const double dx_max = std::max(std::abs(lc_obs_perm.x - bx0), std::abs(lc_obs_perm.x - bx1));
                      for (int ny = ny0; ny <= ny1; ++ny) {
                        const double by0 = double(ny) * L - m_box, by1 = double(ny + 1) * L + m_box;
                        const double dy_min = (lc_obs_perm.y < by0) ? (by0 - lc_obs_perm.y) : ((lc_obs_perm.y > by1) ? (lc_obs_perm.y - by1) : 0.0);
                        const double dy_max = std::max(std::abs(lc_obs_perm.y - by0), std::abs(lc_obs_perm.y - by1));
                        for (int nz = nz0; nz <= nz1; ++nz) {
                          const double bz0 = double(nz) * L - m_box, bz1 = double(nz + 1) * L + m_box;
                          const double dz_min = (lc_obs_perm.z < bz0) ? (bz0 - lc_obs_perm.z) : ((lc_obs_perm.z > bz1) ? (lc_obs_perm.z - bz1) : 0.0);
                          const double dz_max = std::max(std::abs(lc_obs_perm.z - bz0), std::abs(lc_obs_perm.z - bz1));
                          const double d_min = std::sqrt(dx_min * dx_min + dy_min * dy_min + dz_min * dz_min);
                          const double d_max = std::sqrt(dx_max * dx_max + dy_max * dy_max + dz_max * dz_max);
                          if (d_min <= r_hi && d_max >= chi_1) {
                            shifts.push_back({double(nx) * L, double(ny) * L, double(nz) * L});
                          }
                        }
                      }
                    }
                  }
                  if (!shifts.empty()) {
                    if (shifts.size() > lc_shift_cap) {
                      lc_shift_cap = shifts.size() + 64;
                      d_lc_shifts.alloc(lc_shift_cap, aux_dev_bytes);
                    }
                    check(cudaMemcpyAsync(d_lc_shifts.p, shifts.data(),
                                          shifts.size() * sizeof(double3),
                                          cudaMemcpyHostToDevice, G.compute_stream));
                    for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
                      const auto& S = G.slabs[ls];
                      if (S.ns == 0) continue;
                      unsigned int hit_cnt = 0;
                      for (int retry = 0; retry < 2; ++retry) {
                        check(cudaMemsetAsync(d_lc_count.p, 0, sizeof(unsigned int), G.compute_stream));
                        lightcone_detect_kernel<<<blocks(int(S.ns)), team, 0, G.compute_stream>>>(
                            S.d_x.p, S.d_v.p, S.d_orig_idx.p,
                            mass_uniform ? nullptr : S.d_mass.p, first_mass, int(S.ns),
                            step_drift_fac, a_prev, a_step_end, chi_0, chi_1, lc_r_max,
                            lc_obs_perm, d_lc_shifts.p, int(shifts.size()), slab_axis,
                            d_lc_hits.p, d_lc_count.p, unsigned(lc_hit_cap));
                        check(cudaMemcpyAsync(&hit_cnt, d_lc_count.p, sizeof(unsigned int),
                                              cudaMemcpyDeviceToHost, G.compute_stream));
                        check(cudaStreamSynchronize(G.compute_stream));
                        if (size_t(hit_cnt) <= lc_hit_cap) break;
                        lc_hit_cap = std::max(size_t(hit_cnt) + 65536ULL, lc_hit_cap * 2ULL);
                        d_lc_hits.alloc(lc_hit_cap, aux_dev_bytes);
                      }
                      if (size_t(hit_cnt) > lc_hit_cap) {
                        throw std::runtime_error("Lightcone hit buffer overflow");
                      }
                      if (hit_cnt > 0) {
                        auto [slot, lc_ptr] = lc_writer.acquire_slot(hit_cnt);
                        check(cudaMemcpyAsync(lc_ptr, d_lc_hits.p,
                                              size_t(hit_cnt) * sizeof(LightconeHit),
                                              cudaMemcpyDeviceToHost, G.compute_stream));
                        check(cudaStreamSynchronize(G.compute_stream));
                        lc_writer.submit_slot(slot, hit_cnt);
                        lc_total_hits += uint64_t(hit_cnt);
                      }
                    }
                  }
                }
              }
            }
          }

          auto scan_local_slabs = [&](bool track_mv) {
            for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
              const auto& S = G.slabs[ls];
              const int s = S.global_id;
              if (S.ns == 0) {
                slab_active[s] = 0;
                slab_inactive_min[s] = timeline;
                slab_v2_sum[s] = 0.0;
                slab_max_v[s] = 0.f;
                continue;
              }
              const int g_blk = std::min(scan_grid_max, std::max(1, int((S.ns + scan_team - 1) / scan_team)));
              scan_slab_active_v2_kernel<<<g_blk, scan_team, 0, G.compute_stream>>>(
                  S.d_bins.p, S.d_v.p, int(S.ns), ti, (ti == 0 || is_final),
                  need_vel_sum, track_mv, G.d_scan_tmp.p);
              check(cudaMemcpyAsync(G.h_scan.data(), G.d_scan_tmp.p,
                                    size_t(g_blk) * sizeof(ScanPartial),
                                    cudaMemcpyDeviceToHost, G.compute_stream));
              check(cudaStreamSynchronize(G.compute_stream));
              uint64_t act = 0;
              Tick nx = timeline;
              double v2 = 0.0;
              float mv = 0.f;
              for (int b = 0; b < g_blk; ++b) {
                act += G.h_scan[b].active;
                if (G.h_scan[b].min_nx < nx) nx = G.h_scan[b].min_nx;
                v2 += G.h_scan[b].v2_sum;
                if (G.h_scan[b].max_v > mv) mv = G.h_scan[b].max_v;
              }
              slab_active[s] = size_t(act);
              slab_inactive_min[s] = nx;
              slab_v2_sum[s] = v2;
              slab_max_v[s] = mv;
            }
          };

          if (ti > 0) {
            if (eligible_lazy) {
              scan_local_slabs(true);
#pragma omp barrier
              if (g == 0) {
                uint64_t tot_act = 0;
                float gmv = 0.f;
                for (int s = 0; s < num_slabs; ++s) {
                  tot_act += slab_active[s];
                  gmv = std::max(gmv, slab_max_v[s]);
                }
                step_max_disp = drift_fac * double(gmv);
                do_lazy_step = (tot_act < uint64_t(double(N) * lazy_active_frac)) &&
                               ((cum_max_disp + step_max_disp) < max_safe_disp);
                if (!do_lazy_step && lazy_steps_since_rebuild > 0) force_rebuild_all = true;
              }
#pragma omp barrier
            }

            if (do_lazy_step) {
              for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
                const auto& S = G.slabs[ls];
                const int s = S.global_id;
                h_mig_out[s].clear();
                slab_migrants[s] = 0;
                if (S.ns > 0) {
                  drift_only_kernel<<<blocks(int(S.ns)), team, 0, G.compute_stream>>>(
                      S.d_x.p, S.d_v.p, int(S.ns), drift_fac, h.box, periodic);
                }
              }
              check(cudaStreamSynchronize(G.compute_stream));
              if (g == 0) {
                cum_max_disp += step_max_disp;
                ++lazy_steps_since_rebuild;
                ++total_lazy_steps;
              }
            } else {
              for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
                auto& S = G.slabs[ls];
                const int s = S.global_id;
                h_mig_out[s].clear();
                if (S.ns == 0) { slab_migrants[s] = 0; continue; }
                drift_flag_migrants_kernel<<<blocks(int(S.ns)), team, 0, G.compute_stream>>>(
                    S.d_x.p, S.d_v.p, G.d_flags, int(S.ns), s, drift_fac, h.box, inv_wx, periodic);
                cub::CountingInputIterator<int> seq(0);
                size_t selb = G.select_bytes;
                check(cub::DeviceSelect::Flagged(
                    G.d_select_storage.p, selb, seq, G.d_flags,
                    G.d_active_ids, G.d_active_count.p, int(S.ns), G.compute_stream));
                int m_cnt = 0;
                check(cudaMemcpyAsync(&m_cnt, G.d_active_count.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
                check(cudaStreamSynchronize(G.compute_stream));
                slab_migrants[s] = size_t(m_cnt);
                if (m_cnt > 0) {
                  slab_ns_changed[s] = 1;
                  G.ensure_mig_out(size_t(m_cnt));
                  pack_migrants_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                      S.d_x.p, S.d_v.p, S.d_mesh_acc.p, S.d_old_acc.p, S.d_bins.p,
                      S.d_orig_idx.p, mass_uniform ? nullptr : S.d_mass.p, first_mass,
                      soft_uniform ? nullptr : S.d_soft.p, soft,
                      G.d_active_ids, m_cnt, inv_wx, G.d_mig_out.p);
                  h_mig_out[s].resize(size_t(m_cnt));
                  check(cudaMemcpyAsync(h_mig_out[s].data(), G.d_mig_out.p,
                                        size_t(m_cnt) * sizeof(DevMigrant),
                                        cudaMemcpyDeviceToHost, G.compute_stream));
                  const int keep_ns = int(S.ns) - m_cnt;
                  G.ensure_tail_cap(size_t(m_cnt));
                  mark_tail_survivors_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                      G.d_flags, keep_ns, m_cnt, G.d_tail_flags.p);
                  cub::CountingInputIterator<int> tail_seq(keep_ns);
                  size_t selb_t = G.select_bytes;
                  check(cub::DeviceSelect::Flagged(
                      G.d_select_storage.p, selb_t, tail_seq, G.d_tail_flags.p,
                      G.d_tail_keep.p, G.d_tail_count.p, m_cnt, G.compute_stream));
                  compact_migrants_parallel_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                      S.d_x.p, S.d_v.p, S.d_mesh_acc.p, S.d_old_acc.p, S.d_bins.p,
                      S.d_orig_idx.p, mass_uniform ? nullptr : S.d_mass.p,
                      soft_uniform ? nullptr : S.d_soft.p,
                      G.d_active_ids, G.d_tail_keep.p, G.d_tail_count.p);
                  S.ns -= size_t(m_cnt);
                  check(cudaStreamSynchronize(G.compute_stream));
                }
              }
#pragma omp barrier
              if (g == 0) {
                for (int d = 0; d < num_slabs; ++d) h_mig_in[d].clear();
                for (int s = 0; s < num_slabs; ++s) {
                  total_migrants += h_mig_out[s].size();
                  for (const auto& m : h_mig_out[s]) h_mig_in[m.dst_slab].push_back(m);
                }
                if (lazy_steps_since_rebuild > 0) force_rebuild_all = true;
              }
#pragma omp barrier
              for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
                auto& S = G.slabs[ls];
                const int s = S.global_id;
                const auto& in_vec = h_mig_in[s];
                const size_t in_cnt = in_vec.size();
                if (in_cnt > 0) {
                  slab_ns_changed[s] = 1;
                  if (S.ns + in_cnt > S.cap) grow_resident_slab(G, ls, S.ns + in_cnt);
                  G.ensure_mig_in(in_cnt);
                  check(cudaMemcpyAsync(G.d_mig_in.p, in_vec.data(),
                                        in_cnt * sizeof(DevMigrant),
                                        cudaMemcpyHostToDevice, G.compute_stream));
                  append_migrants_kernel<<<blocks(int(in_cnt)), team, 0, G.compute_stream>>>(
                      S.d_x.p, S.d_v.p, S.d_mesh_acc.p, S.d_old_acc.p, S.d_bins.p,
                      S.d_orig_idx.p, mass_uniform ? nullptr : S.d_mass.p,
                      soft_uniform ? nullptr : S.d_soft.p,
                      G.d_mig_in.p, int(S.ns), int(in_cnt));
                  S.ns += in_cnt;
                }
              }
              if (!eligible_lazy || total_migrants > 0) {
                scan_local_slabs(false);
              }
            }
          } else {
            scan_local_slabs(false);
          }
#pragma omp barrier
          if (g == 0) {
            if (!do_lazy_step && force_rebuild_all) {
              cum_max_disp = 0.0;
              lazy_steps_since_rebuild = 0;
            }
            drift_ms = (wall() - t_a0) * 1000.0;
          for (int gg = 0; gg < num_gpus; ++gg)
            for (int ls = 0; ls < gpus[gg].slabs_per_gpu; ++ls)
              peak_slab_seen = std::max(peak_slab_seen, gpus[gg].slabs[ls].ns);

          if (periodic && is_pm_step) {
            ++pm_updates;
            if (is_final) {
              pm_kick = p.factors(timeline - current_pm_span / 2, timeline, false);
            } else {
              if (dynamic_pm) {
                double vel_norm_sum = 0.0;
                for (int s = 0; s < num_slabs; ++s) vel_norm_sum += slab_v2_sum[s];
                const double rms_vel_dm = vel_norm_sum / double(N);
                const double dt_dm = rms_vel_dm > 0.0 ? (a * a * p.rms_length / sqrt(rms_vel_dm)) : 1e30;
                const float dt_rms = float(dt_dm * p.max_dt_RMS_factor);
                const double dt_mesh = fmin(double(dt_rms) * H_cur, p.dt_max);
                const double raw_ticks = dt_mesh * (double(timeline) / p.log_span);
                Tick trial = raw_ticks >= double(timeline) ? timeline : (raw_ticks < 4.0 ? 4ULL : Tick(raw_ticks));
                Tick new_dti = 1ULL << (63 - __builtin_clzll(trial));
                if (ti > 0) {
                  const Tick old_dti = next_pm_span;
                  if (new_dti > 2 * old_dti) new_dti = 2 * old_dti;
                  if (new_dti > old_dti && (ti & (new_dti - 1ULL)) != 0) new_dti = old_dti;
                  prev_pm_span = old_dti;
                } else {
                  prev_pm_span = 0;
                }
                next_pm_span = new_dti;
                short_span_cap = new_dti;
                dt_max_rms = dt_rms;
              } else {
                prev_pm_span = (ti == 0) ? 0 : p.pm_span;
                next_pm_span = p.pm_span;
                short_span_cap = p.short_span_cap;
                dt_max_rms = dt_displacement_default;
              }
              current_pm_span = next_pm_span;
              last_pm_span_scale = double(current_pm_span) / double(timeline) * p.log_span;
              last_dt_max_rms = dt_max_rms;
              const Tick lo = (ti == 0) ? 0 : (ti - prev_pm_span / 2);
              const Tick hi = ti + next_pm_span / 2;
              pm_kick = p.factors(lo, hi, false);
              next_pm_boundary = ti + next_pm_span;
            }
          } else if (is_pm_step) {
            pm_kick = 0.0;
            next_pm_boundary = timeline;
            short_span_cap = p.short_span_cap;
          }

          if (do_fc_now) {
            std::vector<int> h_lids, h_slabs, h_s_lids;
            h_lids.reserve(fc_target_cap);
            h_slabs.reserve(fc_target_cap);
            h_s_lids.reserve(fc_target_cap);
            int rem = fc_target_cap;
            uint64_t rem_parts = N;
            for (int ls = 0; ls < gpus[0].slabs_per_gpu; ++ls) {
              const auto& sl = gpus[0].slabs[ls];
              int q = 0;
              if (sl.ns > 0 && rem > 0) {
                if (ls == gpus[0].slabs_per_gpu - 1 || sl.ns >= rem_parts) {
                  q = std::min<int>(int(sl.ns), rem);
                } else {
                  q = int(std::min<uint64_t>(sl.ns, (uint64_t(rem) * sl.ns + rem_parts - 1) / rem_parts));
                }
              }
              fc_counts[ls] = q;
              fc_offsets[ls] = n_fc_tgt;
              const int stride = (q > 0) ? std::max(1, int(sl.ns) / q) : 1;
              for (int k = 0; k < q; ++k) {
                const int lid = k * stride;
                h_lids.push_back(lid);
                h_slabs.push_back(sl.global_id);
                h_s_lids.push_back(lid);
              }
              n_fc_tgt += q;
              rem -= q;
              rem_parts -= sl.ns;
            }
            if (n_fc_tgt > 0) {
              check(cudaMemcpyAsync(d_fc_lids.p, h_lids.data(), size_t(n_fc_tgt) * sizeof(int),
                                    cudaMemcpyHostToDevice, gpus[0].compute_stream));
              check(cudaMemcpyAsync(d_fc_tgt_slab.p, h_slabs.data(), size_t(n_fc_tgt) * sizeof(int),
                                    cudaMemcpyHostToDevice, gpus[0].compute_stream));
              check(cudaMemcpyAsync(d_fc_tgt_lid.p, h_s_lids.data(), size_t(n_fc_tgt) * sizeof(int),
                                    cudaMemcpyHostToDevice, gpus[0].compute_stream));
            }
          }
        }
#pragma omp barrier

        // ================= STEP B: Multi-GPU Resident PM =================
        if (periodic && is_pm_step) {
          const double t_pm0 = wall();
          if (pm_dist_enabled) {
            dist_pm.begin_deposit(g);
            for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
              const auto& S = G.slabs[ls];
              if (S.ns > 0) {
                dist_pm.deposit(g, S.d_x.p, mass_uniform ? nullptr : S.d_mass.p,
                                first_mass, S.ns, h.box);
              }
            }
            check(cudaStreamSynchronize(G.compute_stream));
#pragma omp barrier
            if (g == 0) {
              dist_pm.solve_and_gather(h.box, r_s);
            }
#pragma omp barrier
            for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
              const auto& S = G.slabs[ls];
              if (S.ns > 0) {
                dist_pm.interpolate(g, S.d_x.p, S.ns, h.box, p.G,
                                    S.d_mesh_acc.p, nullptr);
              }
            }
            check(cudaStreamSynchronize(G.compute_stream));
#pragma omp barrier
          } else if (num_gpus == 1) {
            check(cudaMemsetAsync(G.d_pm_density.p, 0, pm_points * sizeof(double), G.compute_stream));
            for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
              const auto& S = G.slabs[ls];
              if (S.ns > 0) {
                swift_pm_cuda_deposit_buf(
                    S.d_x.p, mass_uniform ? nullptr : S.d_mass.p, first_mass,
                    S.ns, G.d_pm_density.p, pm_side, h.box, G.compute_stream);
              }
            }
            swift_pm_cuda_solve_on_device(G.d_pm_density.p, G.d_pm_field.p, pm_side, h.box, r_s, G.compute_stream);
            for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
              const auto& S = G.slabs[ls];
              if (S.ns > 0) {
                swift_pm_cuda_interpolate_buf(
                    S.d_x.p, S.ns, G.d_pm_field.p, pm_side, h.box, p.G,
                    S.d_mesh_acc.p, nullptr, G.compute_stream);
              }
            }
            check(cudaStreamSynchronize(G.compute_stream));
          } else {
            check(cudaMemsetAsync(G.d_pm_density.p, 0, pm_points * sizeof(double), G.compute_stream));
            for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
              const auto& S = G.slabs[ls];
              if (S.ns > 0) {
                swift_pm_cuda_deposit_buf(
                    S.d_x.p, mass_uniform ? nullptr : S.d_mass.p, first_mass,
                    S.ns, G.d_pm_density.p, pm_side, h.box, G.compute_stream);
              }
            }
            check(cudaMemcpyAsync(h_pm_dens[g].p, G.d_pm_density.p,
                                  pm_points * sizeof(double),
                                  cudaMemcpyDeviceToHost, G.compute_stream));
            check(cudaStreamSynchronize(G.compute_stream));
#pragma omp barrier
            if (g == 0) {
              double* d0 = static_cast<double*>(h_pm_dens[0].p);
              for (int gg = 1; gg < num_gpus; ++gg) {
                const double* dg = static_cast<const double*>(h_pm_dens[gg].p);
                for (size_t i = 0; i < pm_points; ++i) d0[i] += dg[i];
              }
              check(cudaMemcpyAsync(G.d_pm_density.p, d0, pm_points * sizeof(double),
                                    cudaMemcpyHostToDevice, G.compute_stream));
              swift_pm_cuda_solve_on_device(G.d_pm_density.p, G.d_pm_field.p, pm_side, h.box, r_s, G.compute_stream);
              check(cudaMemcpyAsync(h_pm_field.p, G.d_pm_field.p,
                                    pm_points * sizeof(double4),
                                    cudaMemcpyDeviceToHost, G.compute_stream));
              check(cudaStreamSynchronize(G.compute_stream));
            }
#pragma omp barrier
            if (g > 0) {
              check(cudaMemcpyAsync(G.d_pm_field.p, h_pm_field.p,
                                    pm_points * sizeof(double4),
                                    cudaMemcpyHostToDevice, G.compute_stream));
            }
            for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
              const auto& S = G.slabs[ls];
              if (S.ns > 0) {
                swift_pm_cuda_interpolate_buf(
                    S.d_x.p, S.ns, G.d_pm_field.p, pm_side, h.box, p.G,
                    S.d_mesh_acc.p, nullptr, G.compute_stream);
              }
            }
            check(cudaStreamSynchronize(G.compute_stream));
#pragma omp barrier
          }
          if (g == 0) pm_ms = (wall() - t_pm0) * 1000.0;
        }

        // ================= STEP C: Parallel Local Slab Tree Build =================
        const double t_c0 = wall();
        bool step_any_active = false;
        for (int q = 0; q < num_slabs; ++q) {
          if (slab_active[q] > 0) step_any_active = true;
        }
        bool local_skipped_build = false;
        if (do_lazy_step) {
          for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
            const auto& S = G.slabs[ls];
            auto& T = G.trees[ls];
            const int s = S.global_id;
            const bool need_sources = (slab_active[s] > 0) ||
                (slab_active[(s + num_slabs - 1) % num_slabs] > 0) ||
                (slab_active[(s + 1) % num_slabs] > 0);
            if (S.ns > 0) {
              drift_tree_nodes_kernel<<<blocks(int(T.nodes)), team, 0, G.compute_stream>>>(
                  T.force_nodes.p, T.node_vel_max.p,
                  split_force_enabled ? T.node_meta.p : nullptr,
                  int(T.nodes), float(drift_fac), float(h.box));
              slab_packed_stale[s] = 1;
              if (need_sources) {
                gather_sources_slab<<<blocks(int(S.ns)), team, 0, G.compute_stream>>>(
                    S.d_x.p, mass_uniform ? nullptr : S.d_mass.p, first_mass,
                    nullptr, T.order.p, T.packed_sources.p, nullptr, int(S.ns));
                slab_packed_stale[s] = 0;
              }
            }
            slab_builds[s] = 0;
            slab_nodes[s] = T.nodes;
          }
        } else {
          for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
            const auto& S = G.slabs[ls];
            const int s = S.global_id;
            const bool need_build = is_pm_step ||
                (!periodic && step_any_active) ||
                (slab_active[s] > 0) ||
                (slab_active[(s + num_slabs - 1) % num_slabs] > 0) ||
                (slab_active[(s + 1) % num_slabs] > 0);
            const bool slab_needs_full = force_rebuild_all || need_build ||
                (slab_ns_changed[s] != 0) || (slab_packed_stale[s] != 0);
            if (slab_needs_full) {
              const bool will_integrate = is_pm_step || (slab_active[s] > 0);
              build_slab_tree(G, ls, lazy_tree_enabled && !is_final && !will_integrate);
              slab_packed_stale[s] = 0;
              slab_builds[s] = 1;
              slab_nodes[s] = G.trees[ls].nodes;
            } else {
              if (ti > 0 && S.ns > 0) {
                slab_packed_stale[s] = 1;
                local_skipped_build = true;
              }
              slab_builds[s] = 0;
              slab_nodes[s] = 0;
            }
          }
        }
        check(cudaStreamSynchronize(G.compute_stream));
        if (local_skipped_build) {
#pragma omp atomic write
          any_skipped_build = 1;
        }
#pragma omp barrier
        if (g == 0) {
          tree_ms = (wall() - t_c0) * 1000.0;
          if (!do_lazy_step) {
            if (force_rebuild_all) {
              all_slabs_built = true;
              ++total_full_rebuilds;
            } else if (any_skipped_build) {
              all_slabs_built = false;
            }
          }
        }

        // Reconcile phase_pool sizes cleanly before Step D Force/Ghost P2P
        size_t req_loc_c = G.pool_local_cap, req_loc_n = G.pool_local_nodes;
        size_t req_gh_c = G.pool_ghost_cap, req_gh_n = G.pool_ghost_nodes;
        for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
          req_loc_c = std::max(req_loc_c, G.slabs[ls].cap);
          req_loc_n = std::max(req_loc_n, G.trees[ls].node_cap);
        }
        if (num_gpus > 1) {
          const int left_global = (owned_slab(g, 0) + num_slabs - 1) % num_slabs;
          const int right_global = (owned_slab(g, G.slabs_per_gpu - 1) + 1) % num_slabs;
          const auto [lg, lls] = owner_of_slab(left_global);
          const auto [rg, rls] = owner_of_slab(right_global);
          req_gh_c = std::max({req_gh_c, gpus[lg].slabs[lls].ns, gpus[rg].slabs[rls].ns});
          req_gh_n = std::max({req_gh_n, size_t(gpus[lg].trees[lls].nodes), size_t(gpus[rg].trees[rls].nodes)});
        }
#pragma omp critical(phase_pool_alloc)
        G.ensure_phase_pool(req_loc_c, req_loc_n, req_gh_c, req_gh_n);
#pragma omp barrier

        // ================= STEP D: Overlapped Ghost P2P + Force + Integrate =================
        build_kick_table<<<1, 64, 0, G.compute_stream>>>(G.d_kick_finish.p, G.d_kick_begin.p, p, ti);

        AllTreesView iso_trees{};
        if (!periodic) {
          for (int lls = 0; lls < G.slabs_per_gpu; ++lls) {
            iso_trees.t[G.slabs[lls].global_id] = G.trees[lls].view();
          }
        }

        auto run_local_slab = [&](int ls, const SlabForceInput& in_m, const SlabForceInput& in_c, const SlabForceInput& in_p) {
          const SlabTreeView& v_m = in_m.base;
          const SlabTreeView& v_c = in_c.base;
          const SlabTreeView& v_p = in_p.base;
          const auto& S = G.slabs[ls];
          const int s = S.global_id;
          const int ns = int(S.ns);
          if (ns == 0 || (!is_pm_step && slab_active[s] == 0)) {
            slab_active_next[s] = timeline;
            slab_invalid[s] = 0;
            slab_force_ms[s] = 0.0;
            slab_kick_ms[s] = 0.0;
            slab_xfer_ms[s] = 0.0;
            return;
          }
          const int force_count = is_pm_step ? ns : int(slab_active[s]);
          const int* active_ptr = nullptr;
          if (!is_pm_step) {
            mark_active_slab<<<blocks(ns), team, 0, G.compute_stream>>>(
                G.trees[ls].order.p, S.d_bins.p, G.d_flags, ns, ti);
            cub::CountingInputIterator<int> seq(0);
            size_t selb = G.select_bytes;
            check(cub::DeviceSelect::Flagged(
                G.d_select_storage.p, selb, seq, G.d_flags,
                G.d_active_ids, G.d_active_count.p, ns, G.compute_stream));
            active_ptr = G.d_active_ids;
          }

          check(cudaEventRecord(G.ev_force_start, G.compute_stream));
          check(cudaMemsetAsync(G.d_tree_invalid.p, 0, sizeof(int), G.compute_stream));
          if (periodic && split_force_enabled) {
            G.ensure_split_slices(size_t((force_count + 31) / 32 + 2 * slab_roots_count));
            uint64_t loc_batches = 0, loc_retries = 0;
            double l_slice_ms = 0.0, l_phase1_ms = 0.0, l_m2l_ms = 0.0, l_p2p_ms = 0.0;
            uint64_t l_p2p_ch = 0, l_m2l_ch = 0, l_p2p_w = 0, l_m2l_w = 0, l_tot_w = 0;
            if (split_profile) check(cudaEventRecord(G.ev_prof0, G.compute_stream));
            build_home_warp_slices_kernel<<<1, 256, 0, G.compute_stream>>>(
                v_c.home_cell, active_ptr, active_ptr ? G.d_active_count.p : nullptr,
                ns, G.d_tree_invalid.p, G.d_split_slices.p, G.d_split_total_warps.p);
            int total_warps = 0;
            check(cudaMemcpyAsync(&total_warps, G.d_split_total_warps.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
            if (split_profile) check(cudaEventRecord(G.ev_prof1, G.compute_stream));
            check(cudaStreamSynchronize(G.compute_stream));
            if (split_profile) {
              float tms = 0.f;
              check(cudaEventElapsedTime(&tms, G.ev_prof0, G.ev_prof1));
              l_slice_ms += double(tms);
            }

            int warp_off = 0;
            int cur_batch = split_batch_warps;
            while (warp_off < total_warps) {
              const int n_warps = std::min(cur_batch, total_warps - warp_off);
              check(cudaMemsetAsync(G.d_split_counts.p, 0, 5 * sizeof(int), G.compute_stream));
              if (split_profile) check(cudaEventRecord(G.ev_prof0, G.compute_stream));
              if (warp_m2l) {
                traverse_m2p_split_kernel<true><<<blocks(n_warps * 32), team, 0, G.compute_stream>>>(
                    S.d_x.p, S.d_old_acc.p, soft,
                    soft_uniform ? nullptr : S.d_soft.p,
                    in_m.split, in_c.split, in_p.split, v_c.order,
                    G.d_split_slices.p, warp_off, n_warps,
                    G.d_split_meta.p, G.d_split_head_p2p.p, G.d_split_head_m2l.p,
                    G.d_split_p2p.p, int(G.split_p2p_cap),
                    G.d_split_m2l.p, int(G.split_m2l_cap),
                    G.d_split_p2p_warps.p, G.d_split_m2l_warps.p, G.d_split_counts.p,
                    G.d_force, ns, p, ti == 0, G.d_tree_invalid.p, active_ptr, false);
              } else {
                traverse_m2p_split_kernel<false><<<blocks(n_warps * 32), team, 0, G.compute_stream>>>(
                    S.d_x.p, S.d_old_acc.p, soft,
                    soft_uniform ? nullptr : S.d_soft.p,
                    in_m.split, in_c.split, in_p.split, v_c.order,
                    G.d_split_slices.p, warp_off, n_warps,
                    G.d_split_meta.p, G.d_split_head_p2p.p, G.d_split_head_m2l.p,
                    G.d_split_p2p.p, int(G.split_p2p_cap),
                    G.d_split_m2l.p, int(G.split_m2l_cap),
                    G.d_split_p2p_warps.p, G.d_split_m2l_warps.p, G.d_split_counts.p,
                    G.d_force, ns, p, ti == 0, G.d_tree_invalid.p, active_ptr, false);
              }
              if (split_profile) check(cudaEventRecord(G.ev_prof1, G.compute_stream));
              int h_counts[5]{};
              check(cudaMemcpyAsync(h_counts, G.d_split_counts.p, 5 * sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
              check(cudaStreamSynchronize(G.compute_stream));
              if (split_profile) {
                float tms = 0.f;
                check(cudaEventElapsedTime(&tms, G.ev_prof0, G.ev_prof1));
                l_phase1_ms += double(tms);
              }
              if (h_counts[4] != 0) {
                ++loc_retries;
                if (G.split_chunk_scale < 4 && cur_batch == split_batch_warps) {
                  G.ensure_split_bufs(size_t(split_batch_warps), G.split_chunk_scale * 2);
                  continue;
                }
                if (cur_batch > 256) {
                  cur_batch = std::max(256, cur_batch / 2);
                  continue;
                }
                throw std::runtime_error("Split traversal chunk overflow at minimum batch");
              }
              if (h_counts[3] > 0) {
                if (split_profile) check(cudaEventRecord(G.ev_prof0, G.compute_stream));
                eval_m2l_split_kernel<<<blocks(h_counts[3] * 32), team, 0, G.compute_stream>>>(
                    S.d_x.p, in_m.split, in_c.split, in_p.split, v_c.order,
                    G.d_split_meta.p, G.d_split_head_m2l.p, G.d_split_m2l.p,
                    G.d_split_m2l_warps.p, h_counts[3],
                    G.d_force, ns, p, G.d_tree_invalid.p, active_ptr, false);
                if (split_profile) {
                  check(cudaEventRecord(G.ev_prof1, G.compute_stream));
                  check(cudaEventSynchronize(G.ev_prof1));
                  float tms = 0.f;
                  check(cudaEventElapsedTime(&tms, G.ev_prof0, G.ev_prof1));
                  l_m2l_ms += double(tms);
                }
              }
              if (h_counts[2] > 0) {
                if (split_profile) check(cudaEventRecord(G.ev_prof0, G.compute_stream));
                const auto launch_p2p = [&](auto fsp_t, auto us_t) {
                  eval_p2p_split_kernel<decltype(fsp_t)::value, decltype(us_t)::value>
                      <<<blocks(h_counts[2] * 32), team, 0, G.compute_stream>>>(
                          S.d_x.p, soft, soft_uniform ? nullptr : S.d_soft.p,
                          v_m, v_c, v_p, G.d_split_meta.p, G.d_split_head_p2p.p, G.d_split_p2p.p,
                          G.d_split_p2p_warps.p, h_counts[2],
                          G.d_force, ns, p, G.d_tree_invalid.p, active_ptr, false);
                };
                if (fast_split) {
                  if (soft_uniform) launch_p2p(std::true_type{}, std::true_type{});
                  else launch_p2p(std::true_type{}, std::false_type{});
                } else {
                  if (soft_uniform) launch_p2p(std::false_type{}, std::true_type{});
                  else launch_p2p(std::false_type{}, std::false_type{});
                }
                if (split_profile) {
                  check(cudaEventRecord(G.ev_prof1, G.compute_stream));
                  check(cudaEventSynchronize(G.ev_prof1));
                  float tms = 0.f;
                  check(cudaEventElapsedTime(&tms, G.ev_prof0, G.ev_prof1));
                  l_p2p_ms += double(tms);
                }
              }
              if (split_profile) {
                l_p2p_ch += uint64_t(h_counts[0]);
                l_m2l_ch += uint64_t(h_counts[1]);
                l_p2p_w += uint64_t(h_counts[2]);
                l_m2l_w += uint64_t(h_counts[3]);
                l_tot_w += uint64_t(n_warps);
              }
              ++loc_batches;
              warp_off += n_warps;
              if (cur_batch < split_batch_warps) cur_batch = split_batch_warps;
            }
            if (loc_batches || loc_retries || split_profile) {
#pragma omp critical(split_stats)
              {
                split_batches_run += loc_batches;
                split_overflow_retries += loc_retries;
                if (split_profile) {
                  step_slice_ms += l_slice_ms;
                  step_phase1_ms += l_phase1_ms;
                  step_m2l_ms += l_m2l_ms;
                  step_p2p_ms += l_p2p_ms;
                  step_p2p_chunks += l_p2p_ch;
                  step_m2l_chunks += l_m2l_ch;
                  step_p2p_warps += l_p2p_w;
                  step_m2l_warps += l_m2l_w;
                  step_tot_warps += l_tot_w;
                }
              }
            }
          } else if (periodic) {
            const auto launch_w = [&](auto m2l_t, auto fsp_t) {
              tree_force_warp_slab<decltype(m2l_t)::value, decltype(fsp_t)::value>
                  <<<blocks(force_count), team, 0, G.compute_stream>>>(
                      S.d_x.p, S.d_old_acc.p, soft,
                      soft_uniform ? nullptr : S.d_soft.p,
                      v_m, v_c, v_p,
                      G.d_force, ns, p, ti == 0, G.d_tree_invalid.p,
                      active_ptr, active_ptr ? G.d_active_count.p : nullptr,
                      false);
            };
            if (warp_m2l) {
              if (fast_split) launch_w(std::true_type{}, std::true_type{});
              else launch_w(std::true_type{}, std::false_type{});
            } else {
              if (fast_split) launch_w(std::false_type{}, std::true_type{});
              else launch_w(std::false_type{}, std::false_type{});
            }
          } else {
            if (warp_m2l) {
              tree_force_warp_isolated<true><<<blocks(force_count), team, 0, G.compute_stream>>>(
                  S.d_x.p, S.d_old_acc.p, soft,
                  soft_uniform ? nullptr : S.d_soft.p,
                  s, iso_trees, G.d_force, ns, p, ti == 0, G.d_tree_invalid.p,
                  active_ptr, active_ptr ? G.d_active_count.p : nullptr);
            } else {
              tree_force_warp_isolated<false><<<blocks(force_count), team, 0, G.compute_stream>>>(
                  S.d_x.p, S.d_old_acc.p, soft,
                  soft_uniform ? nullptr : S.d_soft.p,
                  s, iso_trees, G.d_force, ns, p, ti == 0, G.d_tree_invalid.p,
                  active_ptr, active_ptr ? G.d_active_count.p : nullptr);
            }
          }
          check(cudaEventRecord(G.ev_force_end, G.compute_stream));
          check(cudaGetLastError());
          check(cudaStreamSynchronize(G.compute_stream));
          float force_elapsed_ms = 0.f;
          check(cudaEventElapsedTime(&force_elapsed_ms, G.ev_force_start, G.ev_force_end));
          slab_force_ms[s] = double(force_elapsed_ms);

          if (force0_out_path && global_iteration == 0 && num_gpus == 1) {
            std::vector<float4> hf(ns);
            std::vector<uint32_t> hidx(ns);
            check(cudaMemcpy(hf.data(), G.d_force, size_t(ns) * sizeof(float4), cudaMemcpyDeviceToHost));
            check(cudaMemcpy(hidx.data(), S.d_orig_idx.p, size_t(ns) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            for (int k = 0; k < ns; ++k) {
              const float4 rf = hf[k];
              const float3 sf3 = permute_out(float3{rf.x, rf.y, rf.z}, slab_axis);
              out_force0_f4[hidx[k]] = float4{sf3.x, sf3.y, sf3.z, rf.w};
            }
          }

          if (do_fc_now && fc_counts[ls] > 0) {
            gather_fc_samples_kernel<<<blocks(fc_counts[ls]), team, 0, G.compute_stream>>>(
                S.d_x.p, soft_uniform ? nullptr : S.d_soft.p, soft,
                G.d_force, S.d_mesh_acc.p,
                d_fc_lids.p + fc_offsets[ls], fc_counts[ls], fc_offsets[ls],
                d_fc_tgt_x.p, d_fc_tgt_h.p, d_fc_gpu_short.p,
                (periodic && is_pm_step) ? d_fc_gpu_mesh.p : nullptr);
            check(cudaGetLastError());
          }

          const bool is_energy_step = energy_log && (is_pm_step || is_final || (!periodic && (ti % p.short_span_cap == 0)));
          if (is_energy_step && ns > 0) {
            if (periodic) {
              if (pm_dist_enabled) {
                dist_pm.interpolate(g, S.d_x.p, size_t(ns), h.box, p.G,
                                    S.d_mesh_acc.p, G.d_pm_pot);
              } else {
                swift_pm_cuda_interpolate_buf(
                    S.d_x.p, size_t(ns), G.d_pm_field.p, pm_side, h.box, p.G,
                    S.d_mesh_acc.p, G.d_pm_pot, G.compute_stream);
              }
            }
            const double closing_pm_e =
                (!periodic || ti == 0) ? 0.0
                : (is_pm_step ? (is_final ? pm_kick : p.factors(ti - prev_pm_span / 2, ti, false))
                              : p.factors(ti - current_pm_span / 2, ti, false));
            const int eblocks = std::min(256, (ns + 255) / 256);
            reduce_energy_mom_kernel<<<eblocks, 256, 0, G.compute_stream>>>(
                S.d_x.p, S.d_v.p, G.d_force, S.d_mesh_acc.p,
                periodic ? G.d_pm_pot : nullptr, S.d_bins.p,
                mass_uniform ? nullptr : S.d_mass.p, first_mass,
                ns, ti, G.d_kick_finish.p, closing_pm_e,
                double(p.G), potential_norm, comoving ? (1.0 / a) : 1.0,
                d_energy_accum[g]);
            check(cudaGetLastError());
          }

          const double tk0 = wall();
          init_meta<<<1, 1, 0, G.compute_stream>>>(G.d_stats.p, G.d_tree_invalid.p, G.trees[ls].nodes);
          float3* synced_v = nullptr;
          if (midpoint) check(cudaMalloc(&synced_v, size_t(ns) * sizeof(float3)));
          integrate_slab<<<blocks(force_count), team, 0, G.compute_stream>>>(
              S.d_v.p, S.d_old_acc.p, S.d_bins.p, G.d_force, S.d_mesh_acc.p,
              active_ptr, G.trees[ls].order.p, force_count, soft,
              soft_uniform ? nullptr : S.d_soft.p, p, ti,
              G.d_stats.p, G.d_kick_finish.p, G.d_kick_begin.p,
              is_pm_step ? pm_kick : 0.0, a, H_cur, dt_max_rms, short_span_cap,
              G.d_tree_invalid.p, synced_v, closing_pm_kick);
          Meta slab_stats{};
          check(cudaMemcpyAsync(&slab_stats, G.d_stats.p, sizeof(Meta), cudaMemcpyDeviceToHost, G.compute_stream));
          check(cudaStreamSynchronize(G.compute_stream));
          slab_kick_ms[s] = (wall() - tk0) * 1000.0;

          if (slab_stats.active != unsigned(force_count)) {
            throw std::runtime_error("Active count mismatch on slab " + std::to_string(s) +
                                     ": integrate=" + std::to_string(slab_stats.active) +
                                     " expected=" + std::to_string(force_count));
          }
          slab_active_next[s] = slab_stats.next;
          slab_invalid[s] = slab_stats.invalid;
          slab_xfer_ms[s] = 0.0;
          if (lazy_tree_enabled && !is_final && ns > 0) {
            const int n_grid_v = blocks(int(G.trees[ls].node_cap));
            for (int depth = tree_levels; depth >= 0; --depth) {
              compute_node_vel_slab_kernel<<<n_grid_v, team, 0, G.compute_stream>>>(
                  G.trees[ls].geometry.p, G.trees[ls].force_nodes.p, S.d_v.p,
                  mass_uniform ? nullptr : S.d_mass.p, first_mass,
                  G.trees[ls].order.p, depth, G.trees[ls].levels.p, G.trees[ls].node_vel_max.p);
            }
          }

          if (is_final || midpoint) {
            check(cudaStreamSynchronize(G.copy_stream));
            const double te0 = wall();
            if (periodic) {
              if (pm_dist_enabled) {
                dist_pm.interpolate(g, S.d_x.p, size_t(ns), h.box, p.G,
                                    S.d_mesh_acc.p, G.d_pm_pot);
              } else {
                swift_pm_cuda_interpolate_buf(
                    S.d_x.p, size_t(ns), G.d_pm_field.p, pm_side, h.box, p.G,
                    S.d_mesh_acc.p, G.d_pm_pot, G.compute_stream);
              }
            } else {
              check(cudaMemsetAsync(G.d_pm_pot, 0, size_t(ns) * sizeof(float), G.compute_stream));
            }

            constexpr int exp_chunk = 1 << 20;
            const size_t chunk_n = size_t(std::min(ns, exp_chunk));
            const bool need_mesh = is_run_final && (force_comp_path || force3_out_path);
            auto align256 = [](size_t v) { return (v + 255ULL) & ~255ULL; };
            size_t h_off = 0;
            const size_t off_hx = h_off; h_off += align256(chunk_n * sizeof(double3));
            const size_t off_hv = h_off; h_off += align256(chunk_n * sizeof(float3));
            const size_t off_hraw = h_off; h_off += align256(chunk_n * sizeof(float4));
            const size_t off_hpot = h_off; h_off += align256(chunk_n * sizeof(float));
            const size_t off_hidx = h_off; h_off += align256(chunk_n * sizeof(uint32_t));
            const size_t off_hmesh = h_off;
            if (need_mesh) h_off += align256(chunk_n * sizeof(float3));
            G.h_bounce.alloc(h_off, cudaHostAllocPortable);

            char* h_base = static_cast<char*>(G.h_bounce.p);
            double3* hx = reinterpret_cast<double3*>(h_base + off_hx);
            float3* hv = reinterpret_cast<float3*>(h_base + off_hv);
            float4* hraw = reinterpret_cast<float4*>(h_base + off_hraw);
            float* hpot = reinterpret_cast<float*>(h_base + off_hpot);
            uint32_t* hidx = reinterpret_cast<uint32_t*>(h_base + off_hidx);
            float3* hmesh = need_mesh ? reinterpret_cast<float3*>(h_base + off_hmesh) : nullptr;

            const float inv_a = float(1.0 / a);
            size_t loc_bad = 0;
            double loc_v2 = 0.0, loc_psum = 0.0;
            float loc_pmin = FLT_MAX, loc_pmax = -FLT_MAX;
            double d2h_ms_acc = 0.0;
            const float3* v_src = midpoint ? synced_v : S.d_v.p;

            for (int base = 0; base < ns; base += exp_chunk) {
              const int cnt = std::min(exp_chunk, ns - base);
              const double tc0 = (base == 0) ? te0 : wall();
              check(cudaMemcpyAsync(hx, S.d_x.p + base, size_t(cnt) * sizeof(double3), cudaMemcpyDeviceToHost, G.compute_stream));
              check(cudaMemcpyAsync(hv, v_src + base, size_t(cnt) * sizeof(float3), cudaMemcpyDeviceToHost, G.compute_stream));
              check(cudaMemcpyAsync(hraw, G.d_force + base, size_t(cnt) * sizeof(float4), cudaMemcpyDeviceToHost, G.compute_stream));
              check(cudaMemcpyAsync(hpot, G.d_pm_pot + base, size_t(cnt) * sizeof(float), cudaMemcpyDeviceToHost, G.compute_stream));
              check(cudaMemcpyAsync(hidx, S.d_orig_idx.p + base, size_t(cnt) * sizeof(uint32_t), cudaMemcpyDeviceToHost, G.compute_stream));
              if (need_mesh) {
                check(cudaMemcpyAsync(hmesh, S.d_mesh_acc.p + base, size_t(cnt) * sizeof(float3), cudaMemcpyDeviceToHost, G.compute_stream));
              }
              check(cudaStreamSynchronize(G.compute_stream));
              d2h_ms_acc += (wall() - tc0) * 1000.0;

              for (int k = 0; k < cnt; ++k) {
                const float4 rf = hraw[k];
                const float pp = hpot[k];
                const float pot_val = float(double(rf.w) + potential_norm) * p.G + pp;
                const double3 xk = hx[k];
                const float3 vk = hv[k];
                if (!std::isfinite(xk.x) || !std::isfinite(xk.y) || !std::isfinite(xk.z) ||
                    (periodic && (xk.x < 0.0 || xk.x >= h.box || xk.y < 0.0 || xk.y >= h.box || xk.z < 0.0 || xk.z >= h.box)) ||
                    !std::isfinite(vk.x) || !std::isfinite(vk.y) || !std::isfinite(vk.z) || !std::isfinite(pot_val)) {
                  ++loc_bad;
                } else {
                  const float3 vf{vk.x * inv_a, vk.y * inv_a, vk.z * inv_a};
                  loc_v2 += double(vf.x) * vf.x + double(vf.y) * vf.y + double(vf.z) * vf.z;
                  loc_psum += double(pot_val);
                  if (pot_val < loc_pmin) loc_pmin = pot_val;
                  if (pot_val > loc_pmax) loc_pmax = pot_val;
                }
                const uint32_t gid = hidx[k];
                if (!skip_output_write) {
                  out_x[gid] = permute_out(xk, slab_axis);
                  out_v[gid] = permute_out(float3{vk.x * inv_a, vk.y * inv_a, vk.z * inv_a}, slab_axis);
                  out_pot[gid] = pot_val;
                }
                if (is_run_final && force_comp_path) {
                  const float3 sf3 = permute_out(float3{rf.x, rf.y, rf.z}, slab_axis);
                  out_short_f4[gid] = float4{sf3.x, sf3.y, sf3.z, rf.w};
                  const float3 ma = permute_out(hmesh[k], slab_axis);
                  out_long_f4[gid] = float4{ma.x, ma.y, ma.z, pp};
                }
                if (is_run_final && force3_out_path) {
                  out_short_f3[gid] = permute_out(float3{rf.x, rf.y, rf.z}, slab_axis);
                  out_long_f3[gid] = permute_out(hmesh[k], slab_axis);
                }
              }
            }
            slab_xfer_ms[s] = d2h_ms_acc;
            if (synced_v) check(cudaFree(synced_v));
#pragma omp critical
            {
              sanity_bad += loc_bad;
              sanity_v2_sum += loc_v2;
              sanity_pot_sum += loc_psum;
              if (loc_pmin < sanity_pot_min) sanity_pot_min = loc_pmin;
              if (loc_pmax > sanity_pot_max) sanity_pot_max = loc_pmax;
            }
          }
        };

        const bool is_energy_step_gpu = energy_log && (is_pm_step || is_final || (!periodic && (ti % p.short_span_cap == 0)));
        if (is_energy_step_gpu) {
          check(cudaMemsetAsync(d_energy_accum[g], 0, 10 * sizeof(double), G.compute_stream));
        }

        if (num_gpus == 1) {
          for (int ls = 0; ls < G.slabs_per_gpu; ++ls) {
            slab_ghost_ms[owned_slab(0, ls)] = 0.0;
            run_local_slab(ls, G.trees[(ls + 11) % 12].full_view(), G.trees[ls].full_view(), G.trees[(ls + 1) % 12].full_view());
          }
        } else {
          const int left_global = (owned_slab(g, 0) + num_slabs - 1) % num_slabs;
          const int right_global = (owned_slab(g, G.slabs_per_gpu - 1) + 1) % num_slabs;
          const auto [lg, lls] = owner_of_slab(left_global);
          const auto [rg, rls] = owner_of_slab(right_global);

          const int s0 = owned_slab(g, 0);
          const int slast = owned_slab(g, G.slabs_per_gpu - 1);
          const bool need_s0 = (G.slabs[0].ns > 0) && (is_pm_step || slab_active[s0] > 0);
          const bool need_last = (G.slabs_per_gpu > 1) && (G.slabs[G.slabs_per_gpu - 1].ns > 0) && (is_pm_step || slab_active[slast] > 0);
          const bool hide_left = (G.slabs_per_gpu > 2) && (!can_peer[g][lg] && (can_peer[g][rg] || (g % 2 == 0)));

          if (G.slabs_per_gpu == 1) {
            if (need_s0) {
              check(cudaEventRecord(G.ev_left_start, G.copy_stream));
              copy_ghost_tree(G, G.ghost, gpus[lg], lls, false);
              check(cudaEventRecord(G.ev_left_ready, G.copy_stream));
              check(cudaEventRecord(G.ev_right_start, G.copy_stream));
              copy_ghost_tree(G, G.ghost_right, gpus[rg], rls, true);
              check(cudaEventRecord(G.ev_right_ready, G.copy_stream));
              check(cudaStreamWaitEvent(G.compute_stream, G.ev_right_ready, 0));
              run_local_slab(0, G.ghost.full_view(), G.trees[0].full_view(), G.ghost_right.full_view());
            } else {
              slab_ghost_ms[s0] = 0.0;
              run_local_slab(0, G.ghost.full_view(), G.trees[0].full_view(), G.ghost_right.full_view());
            }
            check(cudaStreamSynchronize(G.copy_stream));
            check(cudaStreamSynchronize(G.compute_stream));
            if (need_s0) {
              float gms_l = 0.f, gms_r = 0.f;
              check(cudaEventElapsedTime(&gms_l, G.ev_left_start, G.ev_left_ready));
              check(cudaEventElapsedTime(&gms_r, G.ev_right_start, G.ev_right_ready));
              slab_ghost_ms[s0] = double(gms_l) + double(gms_r);
            }
          } else {
            if (hide_left) {
              if (need_s0) {
                check(cudaEventRecord(G.ev_left_start, G.copy_stream));
                copy_ghost_tree(G, G.ghost, gpus[lg], lls, false);
                check(cudaEventRecord(G.ev_left_ready, G.copy_stream));
              } else {
                slab_ghost_ms[s0] = 0.0;
              }
              for (int ls = 1; ls < G.slabs_per_gpu - 1; ++ls) {
                slab_ghost_ms[owned_slab(g, ls)] = 0.0;
                run_local_slab(ls, G.trees[ls - 1].full_view(), G.trees[ls].full_view(), G.trees[ls + 1].full_view());
              }
              if (need_s0) {
                check(cudaStreamWaitEvent(G.compute_stream, G.ev_left_ready, 0));
                run_local_slab(0, G.ghost.full_view(), G.trees[0].full_view(), G.trees[1].full_view());
                check(cudaEventRecord(G.ev_s0_done, G.compute_stream));
              } else {
                run_local_slab(0, G.ghost.full_view(), G.trees[0].full_view(), G.trees[1].full_view());
              }
              if (need_last) {
                if (need_s0) check(cudaStreamWaitEvent(G.copy_stream, G.ev_s0_done, 0));
                check(cudaEventRecord(G.ev_right_start, G.copy_stream));
                copy_ghost_tree(G, G.ghost, gpus[rg], rls, true);
                check(cudaEventRecord(G.ev_right_ready, G.copy_stream));
                check(cudaStreamWaitEvent(G.compute_stream, G.ev_right_ready, 0));
              } else {
                slab_ghost_ms[slast] = 0.0;
              }
              run_local_slab(G.slabs_per_gpu - 1,
                             G.trees[G.slabs_per_gpu - 2].full_view(),
                             G.trees[G.slabs_per_gpu - 1].full_view(),
                             G.ghost.full_view());
            } else {
              if (need_s0) {
                check(cudaEventRecord(G.ev_left_start, G.copy_stream));
                copy_ghost_tree(G, G.ghost, gpus[lg], lls, false);
                check(cudaEventRecord(G.ev_left_ready, G.copy_stream));
                check(cudaStreamWaitEvent(G.compute_stream, G.ev_left_ready, 0));
                run_local_slab(0, G.ghost.full_view(), G.trees[0].full_view(), G.trees[1].full_view());
                check(cudaEventRecord(G.ev_s0_done, G.compute_stream));
              } else {
                slab_ghost_ms[s0] = 0.0;
                run_local_slab(0, G.ghost.full_view(), G.trees[0].full_view(), G.trees[1].full_view());
              }

              if (need_last) {
                if (need_s0) check(cudaStreamWaitEvent(G.copy_stream, G.ev_s0_done, 0));
                check(cudaEventRecord(G.ev_right_start, G.copy_stream));
                copy_ghost_tree(G, G.ghost, gpus[rg], rls, true);
                check(cudaEventRecord(G.ev_right_ready, G.copy_stream));
              } else if (G.slabs_per_gpu > 1) {
                slab_ghost_ms[slast] = 0.0;
              }

              for (int ls = 1; ls < G.slabs_per_gpu - 1; ++ls) {
                slab_ghost_ms[owned_slab(g, ls)] = 0.0;
                run_local_slab(ls, G.trees[ls - 1].full_view(), G.trees[ls].full_view(), G.trees[ls + 1].full_view());
              }

              if (G.slabs_per_gpu > 1) {
                if (need_last) check(cudaStreamWaitEvent(G.compute_stream, G.ev_right_ready, 0));
                run_local_slab(G.slabs_per_gpu - 1,
                               G.trees[G.slabs_per_gpu - 2].full_view(),
                               G.trees[G.slabs_per_gpu - 1].full_view(),
                               G.ghost.full_view());
              }
            }
            check(cudaStreamSynchronize(G.copy_stream));
            check(cudaStreamSynchronize(G.compute_stream));
            if (need_s0) {
              float gms = 0.f;
              check(cudaEventElapsedTime(&gms, G.ev_left_start, G.ev_left_ready));
              slab_ghost_ms[s0] = double(gms);
            }
            if (need_last) {
              float gms = 0.f;
              check(cudaEventElapsedTime(&gms, G.ev_right_start, G.ev_right_ready));
              slab_ghost_ms[slast] = double(gms);
            }
          }
        }
        if (is_energy_step_gpu) {
          check(cudaMemcpyAsync(gpu_energy[g].data(), d_energy_accum[g], 10 * sizeof(double),
                                cudaMemcpyDeviceToHost, G.compute_stream));
          check(cudaStreamSynchronize(G.compute_stream));
        }
#pragma omp barrier
        }  // end omp parallel

        Tick step_next = timeline;
        uint64_t step_active = 0, step_nodes = 0;
        unsigned int step_invalid = 0;
        int step_builds = 0;
        double step_ghost_ms = 0.0, step_xfer_ms = 0.0;
        double step_force_ms = 0.0, step_kick_ms = 0.0;
        if (num_gpus == 1) {
          for (int s = 0; s < num_slabs; ++s) {
            step_force_ms += slab_force_ms[s];
            step_kick_ms += slab_kick_ms[s];
            step_ghost_ms += slab_ghost_ms[s];
            step_xfer_ms += slab_xfer_ms[s];
          }
        } else {
          // Across G GPUs running concurrently, track critical-path GPU max per step
          for (int g = 0; g < num_gpus; ++g) {
            double gf = 0.0, gk = 0.0, gg = 0.0, gx = 0.0;
            for (int ls = 0; ls < gpus[g].slabs_per_gpu; ++ls) {
              const int s = owned_slab(g, ls);
              gf += slab_force_ms[s];
              gk += slab_kick_ms[s];
              gg += slab_ghost_ms[s];
              gx += slab_xfer_ms[s];
            }
            step_force_ms = std::max(step_force_ms, gf);
            step_kick_ms = std::max(step_kick_ms, gk);
            step_ghost_ms = std::max(step_ghost_ms, gg);
            step_xfer_ms = std::max(step_xfer_ms, gx);
          }
        }
        for (int s = 0; s < num_slabs; ++s) {
          if (slab_inactive_min[s] < step_next) step_next = slab_inactive_min[s];
          if (slab_active_next[s] < step_next) step_next = slab_active_next[s];
          step_active += slab_active[s];
          step_nodes += slab_nodes[s];
          step_invalid |= slab_invalid[s];
          step_builds += slab_builds[s];
        }
        force_ms = step_force_ms;
        kick_ms = step_kick_ms;
        ghost_ms = step_ghost_ms;
        export_ms = step_xfer_ms;

        if (step_invalid || !step_active || (!is_final && (step_next <= ti || step_next > timeline)))
          throw std::runtime_error("Invalid multi-GPU slab state at step " + std::to_string(global_iteration) +
                                   " invalid=" + std::to_string(step_invalid) +
                                   " active=" + std::to_string(step_active));

        const double step_wall = wall() - step_start;
        std::cout << "STEP {\"step\":" << global_iteration << ",\"a\":" << a
                  << ",\"integer_time\":" << ti << ",\"active\":" << step_active
                  << ",\"nodes\":" << step_nodes << ",\"wall_s\":" << step_wall
                  << ",\"drift_ms\":" << drift_ms << ",\"key_sort_ms\":0"
                  << ",\"tree_multipoles_ms\":" << tree_ms << ",\"force_ms\":" << force_ms
                  << ",\"kick_bins_ms\":" << kick_ms
                  << ",\"h2d_d2h_ms\":" << (ghost_ms + export_ms)
                  << ",\"ghost_ms\":" << ghost_ms
                  << ",\"slab_builds\":" << step_builds << ",\"migrants\":" << total_migrants
                  << ",\"pm\":" << (is_pm_step ? "true" : "false")
                  << ",\"lazy\":" << (do_lazy_step ? "true" : "false")
                  << ",\"pm_device_ms\":" << pm_ms << "}" << std::endl;
        if (split_profile) {
          std::cout << "SPLIT_PROF {\"step\":" << global_iteration
                    << ",\"active\":" << step_active
                    << ",\"pm\":" << (is_pm_step ? "true" : "false")
                    << ",\"tot_warps\":" << step_tot_warps
                    << ",\"slice_ms\":" << step_slice_ms
                    << ",\"phase1_ms\":" << step_phase1_ms
                    << ",\"m2l_ms\":" << step_m2l_ms
                    << ",\"p2p_ms\":" << step_p2p_ms
                    << ",\"p2p_chunks\":" << step_p2p_chunks
                    << ",\"m2l_chunks\":" << step_m2l_chunks
                    << ",\"p2p_warps\":" << step_p2p_warps
                    << ",\"m2l_warps\":" << step_m2l_warps << "}" << std::endl;
        }

        const bool is_energy_step_out = energy_log && (is_pm_step || is_final || (!periodic && (ti % p.short_span_cap == 0)));
        if (is_energy_step_out) {
          double tot[10] = {0};
          for (int g = 0; g < num_gpus; ++g)
            for (int k = 0; k < 10; ++k) tot[k] += gpu_energy[g][k];
          const double inv_a_e = comoving ? (1.0 / a) : 1.0;
          const double U_phys = tot[1] * inv_a_e;
          std::cout << std::setprecision(12)
                    << "ENERGY {\"step\":" << global_iteration << ",\"a\":" << a
                    << ",\"T\":" << tot[0] << ",\"U\":" << U_phys
                    << ",\"W_com\":" << tot[1] << ",\"E\":" << (tot[0] + U_phys)
                    << ",\"Px\":" << tot[2] << ",\"Py\":" << tot[3] << ",\"Pz\":" << tot[4]
                    << ",\"Lx\":" << tot[5] << ",\"Ly\":" << tot[6] << ",\"Lz\":" << tot[7]
                    << ",\"M\":" << tot[8] << ",\"v_rms\":" << sqrt(tot[9] / double(N)) << "}" << std::endl;
        }

        if (is_pm_step) {
          std::cout << "PROGRESS {\"pm_updates\":" << pm_updates
                    << ",\"z\":" << 1.0 / a - 1.0
                    << ",\"dt_max_rms\":" << dt_max_rms
                    << ",\"next_pm_span\":" << next_pm_span
                    << ",\"steps\":" << global_iteration
                    << ",\"elapsed_s\":" << (wall() - evo_start) << "}" << std::endl;
        }

        if (do_fc_now && n_fc_tgt > 0) {
          const double fc_t0 = wall();
          auto& G0 = gpus[0];
          check(cudaSetDevice(G0.dev));
          AllSlabsData all_slabs{};
          for (int ls = 0; ls < G0.slabs_per_gpu; ++ls) {
            const auto& sl = G0.slabs[ls];
            const int sid = sl.global_id;
            all_slabs.x[sid] = sl.d_x.p;
            all_slabs.mass[sid] = mass_uniform ? nullptr : sl.d_mass.p;
            all_slabs.soft[sid] = soft_uniform ? nullptr : sl.d_soft.p;
            all_slabs.ns[sid] = int(sl.ns);
          }
          if (periodic) {
            if (!is_pm_step) {
              check(cudaMemsetAsync(G0.d_pm_density.p, 0, pm_points * sizeof(double), G0.compute_stream));
              for (int ls = 0; ls < G0.slabs_per_gpu; ++ls) {
                const auto& sl = G0.slabs[ls];
                if (sl.ns > 0) {
                  swift_pm_cuda_deposit_buf(
                      sl.d_x.p, mass_uniform ? nullptr : sl.d_mass.p, first_mass,
                      sl.ns, G0.d_pm_density.p, pm_side, h.box, G0.compute_stream);
                }
              }
              swift_pm_cuda_solve_on_device(G0.d_pm_density.p, G0.d_pm_field.p, pm_side, h.box, r_s, G0.compute_stream);
              swift_pm_cuda_interpolate_buf(
                  d_fc_tgt_x.p, size_t(n_fc_tgt), G0.d_pm_field.p, pm_side, h.box, p.G,
                  d_fc_gpu_mesh.p, nullptr, G0.compute_stream);
            }
            const int n_modes = int(h_ewald_modes.size());
            ewald_kspace_accum_kernel<<<n_modes, 256, 0, G0.compute_stream>>>(
                all_slabs, first_mass, d_ewald_modes.p, n_modes, d_ewald_C.p, d_ewald_S.p);
            ewald_target_force_kernel<<<n_fc_tgt, 256, 0, G0.compute_stream>>>(
                all_slabs, first_mass, soft, d_fc_tgt_x.p, d_fc_tgt_h.p,
                d_fc_tgt_slab.p, d_fc_tgt_lid.p, n_fc_tgt, h.box, r_s, double(p.G),
                d_ewald_modes.p, d_ewald_C.p, d_ewald_S.p, n_modes,
                d_fc_short_ref.p, d_fc_total_ref.p);
          } else {
            check(cudaMemsetAsync(d_fc_gpu_mesh.p, 0, size_t(n_fc_tgt) * sizeof(float3), G0.compute_stream));
            exact_direct_isolated_kernel<<<n_fc_tgt, 256, 0, G0.compute_stream>>>(
                all_slabs, first_mass, soft, d_fc_tgt_x.p, d_fc_tgt_h.p,
                d_fc_tgt_slab.p, d_fc_tgt_lid.p, n_fc_tgt, p, d_fc_total_ref.p);
          }
          check(cudaGetLastError());
          std::vector<float4> h_gs(n_fc_tgt);
          std::vector<float3> h_gm(n_fc_tgt);
          std::vector<double3> h_rs(n_fc_tgt), h_rt(n_fc_tgt);
          check(cudaMemcpyAsync(h_gs.data(), d_fc_gpu_short.p, size_t(n_fc_tgt) * sizeof(float4),
                                cudaMemcpyDeviceToHost, G0.compute_stream));
          check(cudaMemcpyAsync(h_gm.data(), d_fc_gpu_mesh.p, size_t(n_fc_tgt) * sizeof(float3),
                                cudaMemcpyDeviceToHost, G0.compute_stream));
          if (periodic) {
            check(cudaMemcpyAsync(h_rs.data(), d_fc_short_ref.p, size_t(n_fc_tgt) * sizeof(double3),
                                  cudaMemcpyDeviceToHost, G0.compute_stream));
          }
          check(cudaMemcpyAsync(h_rt.data(), d_fc_total_ref.p, size_t(n_fc_tgt) * sizeof(double3),
                                cudaMemcpyDeviceToHost, G0.compute_stream));
          check(cudaStreamSynchronize(G0.compute_stream));

          std::vector<double> rel_s, rel_t;
          rel_s.reserve(n_fc_tgt);
          rel_t.reserve(n_fc_tgt);
          for (int q = 0; q < n_fc_tgt; ++q) {
            const double gx = double(h_gs[q].x) * double(p.G);
            const double gy = double(h_gs[q].y) * double(p.G);
            const double gz = double(h_gs[q].z) * double(p.G);
            if (periodic) {
              const double smag = std::sqrt(h_rs[q].x * h_rs[q].x + h_rs[q].y * h_rs[q].y + h_rs[q].z * h_rs[q].z);
              if (smag > 1e-18) {
                const double ds = std::sqrt((gx - h_rs[q].x) * (gx - h_rs[q].x) +
                                            (gy - h_rs[q].y) * (gy - h_rs[q].y) +
                                            (gz - h_rs[q].z) * (gz - h_rs[q].z));
                rel_s.push_back(ds / smag);
              }
            }
            const double tx = gx + double(h_gm[q].x);
            const double ty = gy + double(h_gm[q].y);
            const double tz = gz + double(h_gm[q].z);
            const double tmag = std::sqrt(h_rt[q].x * h_rt[q].x + h_rt[q].y * h_rt[q].y + h_rt[q].z * h_rt[q].z);
            if (tmag > 1e-18) {
              const double dt = std::sqrt((tx - h_rt[q].x) * (tx - h_rt[q].x) +
                                          (ty - h_rt[q].y) * (ty - h_rt[q].y) +
                                          (tz - h_rt[q].z) * (tz - h_rt[q].z));
              rel_t.push_back(dt / tmag);
            }
          }
          auto stats_of = [](std::vector<double>& v) -> std::array<double, 3> {
            if (v.empty()) return {0.0, 0.0, 0.0};
            std::sort(v.begin(), v.end());
            const double med = (v.size() & 1) ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
            const double p95 = v[size_t(std::min<size_t>(v.size() - 1, size_t(0.95 * double(v.size() - 1))))];
            return {med, p95, v.back()};
          };
          const auto st_s = stats_of(rel_s);
          const auto st_t = stats_of(rel_t);
          std::cout << std::setprecision(12)
                    << "FORCE_CHECK {\"step\":" << global_iteration
                    << ",\"a\":" << p.time_at(ti)
                    << ",\"n\":" << n_fc_tgt
                    << ",\"mode\":\"" << (periodic ? "ewald" : "direct_isolated") << "\""
                    << ",\"short_med_rel\":" << st_s[0]
                    << ",\"short_p95_rel\":" << st_s[1]
                    << ",\"short_max_rel\":" << st_s[2]
                    << ",\"total_med_rel\":" << st_t[0]
                    << ",\"total_p95_rel\":" << st_t[1]
                    << ",\"total_max_rel\":" << st_t[2]
                    << ",\"wall_ms\":" << ((wall() - fc_t0) * 1000.0) << "}" << std::endl;
        }

        if (midpoint) {
          if (step_active != N || sanity_bad) throw std::runtime_error("Invalid midpoint state");
          FILE* out = fopen(midpoint_path, "wb");
          if (!out) throw std::runtime_error("Cannot open midpoint");
          const Footer header{N, h.box, a};
          write_bin(out, &header, 1);
          write_bin(out, out_x.data(), N);
          write_bin(out, out_v.data(), N);
          write_bin(out, out_pot.data(), N);
          if (fclose(out)) throw std::runtime_error("Cannot close midpoint");
          midpoint_written = true;
          sanity_bad = 0;
          sanity_v2_sum = sanity_pot_sum = 0.0;
          sanity_pot_min = FLT_MAX;
          sanity_pot_max = -FLT_MAX;
          std::vector<double3>().swap(out_x);
          std::vector<float3>().swap(out_v);
          std::vector<float>().swap(out_pot);
        }
        if (is_final) {
          if (!is_last_seg) {
            if (step_active != N || sanity_bad) throw std::runtime_error("Invalid intermediate snapshot state");
            char buf[32];
            std::snprintf(buf, sizeof(buf), "_%04zu.bin", seg);
            const std::string snap_path = out_prefix + buf;

            std::vector<int64_t> fof_ids;
            std::string fof_path;
            uint64_t fof_ngrp = 0, fof_ngrouped = 0, fof_largest = 0;
            const double fof_link = fof_link_ratio * h.box / std::cbrt(double(N));
            if (run_fof && !skip_output_write) {
              check(cudaSetDevice(0));
              const double fof_t0 = wall();
              double fof_el[9]{};
              fof_ids.resize(N);
              const int fof_slabs = requested_fof_slabs > 0 ? requested_fof_slabs : ((N <= (128ULL << 20)) ? 1 : 16);
              const int fst = (fof_slabs <= 1)
                  ? whole_fof_swift(
                        reinterpret_cast<const double*>(out_x.data()), int(N),
                        h.box, h.box, h.box, fof_link, fof_min_size,
                        fof_ids.data(), &fof_ngrp, &fof_ngrouped, &fof_largest, fof_el, nullptr)
                  : whole_fof_slabbed_swift(
                        reinterpret_cast<const double*>(out_x.data()), N,
                        h.box, h.box, h.box, fof_link, fof_min_size, fof_slabs,
                        fof_ids.data(), &fof_ngrp, &fof_ngrouped, &fof_largest, fof_el, nullptr);
              if (fst != 0) throw std::runtime_error("FoF failed at intermediate snapshot");
              fof_path = snap_path + ".fof.bin";
              std::cout << std::setprecision(12)
                        << "FOF_RESULT {\"index\":" << seg
                        << ",\"a\":" << seg_a_end
                        << ",\"path\":\"" << fof_path
                        << "\",\"fof_slabs\":" << fof_slabs
                        << ",\"groups\":" << fof_ngrp
                        << ",\"grouped_particles\":" << fof_ngrouped
                        << ",\"largest\":" << fof_largest
                        << ",\"min_size\":" << fof_min_size
                        << ",\"link_mpc\":" << fof_link
                        << ",\"fof_s\":" << (wall() - fof_t0)
                        << ",\"neighbors_union_s\":" << fof_el[6] << "}" << std::endl;
            }

            std::cout << std::setprecision(12)
                      << "SNAPSHOT_QUEUED {\"index\":" << seg
                      << ",\"a\":" << seg_a_end
                      << ",\"path\":\"" << snap_path
                      << "\",\"d2h_ms\":" << export_ms << "}" << std::endl;
            if (!skip_output_write) {
              pending_snap = {true, seg, seg_a_end, snap_path, 0.0};
              snap_writer = std::thread([
                  wx = std::move(out_x), wv = std::move(out_v), wpot = std::move(out_pot),
                  wfof = std::move(fof_ids), fof_path, fof_ngrp, fof_largest,
                  fof_min_u64 = uint64_t(fof_min_size), fof_link,
                  N, box = h.box, seg_a_end, snap_path,
                  &pending_snap, &snap_writer_err]() {
                try {
                  const double tw0 = wall();
                  FILE* out = fopen(snap_path.c_str(), "wb");
                  if (!out) throw std::runtime_error("Cannot open intermediate snapshot: " + snap_path);
                  const Footer hdr{N, box, seg_a_end};
                  write_bin(out, &hdr, 1);
                  write_bin(out, wx.data(), N);
                  write_bin(out, wv.data(), N);
                  write_bin(out, wpot.data(), N);
                  if (fclose(out)) throw std::runtime_error("Cannot close intermediate snapshot: " + snap_path);
                  if (!fof_path.empty()) {
                    write_fof_bin(fof_path, N, fof_ngrp, fof_largest, fof_min_u64,
                                  box, seg_a_end, fof_link, wfof.data());
                  }
                  pending_snap.write_s = wall() - tw0;
                } catch (...) {
                  snap_writer_err = std::current_exception();
                }
              });
            }
            sanity_bad = 0;
            sanity_v2_sum = sanity_pot_sum = 0.0;
            sanity_pot_min = FLT_MAX;
            sanity_pot_max = -FLT_MAX;
            break;
          }
          if (midpoint_path && !midpoint_written) throw std::runtime_error("Missed midpoint");
          break;
        }
        previous = ti;
        ti = step_next;
      }
    }

    join_snap_writer();

    if (force0_out_path) {
      FILE* fp = fopen(force0_out_path, "wb");
      if (!fp) throw std::runtime_error("Could not open SWIFT_FORCE0_OUT output");
      write_bin(fp, out_force0_f4.data(), out_force0_f4.size());
      if (fclose(fp)) throw std::runtime_error("Could not close SWIFT_FORCE0_OUT");
    }
    if (force_comp_path) {
      FILE* fp = fopen(force_comp_path, "wb");
      if (!fp) throw std::runtime_error("Could not open SWIFT_FORCE_COMPONENTS output");
      write_bin(fp, out_short_f4.data(), out_short_f4.size());
      write_bin(fp, out_long_f4.data(), out_long_f4.size());
      if (fclose(fp)) throw std::runtime_error("Could not close SWIFT_FORCE_COMPONENTS");
    }
    if (force3_out_path) {
      FILE* fp = fopen(force3_out_path, "wb");
      if (!fp) throw std::runtime_error("Could not open SWIFT_FORCE3_OUT output");
      write_bin(fp, out_short_f3.data(), out_short_f3.size());
      write_bin(fp, out_long_f3.data(), out_long_f3.size());
      if (fclose(fp)) throw std::runtime_error("Could not close SWIFT_FORCE3_OUT");
    }

    const double start_output = wall();
    size_t output_bytes = 0;
    if (!skip_output_write) {
      FILE* output = fopen(argv[2], "wb");
      if (!output) return 4;
      const Footer footer{N, h.box, a_end};
      write_bin(output, &footer, 1);
      write_bin(output, out_x.data(), N);
      write_bin(output, out_v.data(), N);
      write_bin(output, out_pot.data(), N);
      if (fclose(output)) return 4;
      output_bytes = N * (sizeof(double3) + sizeof(float3) + sizeof(float));
    }
    const double export_s = wall() - start_output;
    const double evo_wall = wall() - evo_start;

    std::cout << std::setprecision(12)
              << "SANITY {\"particles\":" << N << ",\"bad_count\":" << sanity_bad
              << ",\"v_rms_kms_internal\":" << sqrt(sanity_v2_sum / double(N))
              << ",\"pot_mean\":" << (sanity_pot_sum / double(N))
              << ",\"pot_min\":" << sanity_pot_min << ",\"pot_max\":" << sanity_pot_max << "}" << std::endl;

    std::cout << std::setprecision(12)
              << "RESULT {\"evolution_wall_s\":" << evo_wall
              << ",\"whole_program_s\":" << (wall() - started)
              << ",\"pm_updates_including_initial\":" << pm_updates
              << ",\"fast_tree\":true,\"warp_force\":true"
              << ",\"warp_m2l\":" << (warp_m2l ? "true" : "false")
              << ",\"fast_split\":" << (fast_split ? "true" : "false")
              << ",\"split_force\":" << (split_force_enabled ? "true" : "false")
              << ",\"pm_dist\":" << (pm_dist_enabled ? "true" : "false")
              << ",\"domain_mode\":\"1d_slab\""
              << ",\"split_batch_warps\":" << split_batch_warps
              << ",\"lazy_tree\":" << (lazy_tree_enabled ? "true" : "false")
              << ",\"split_batches\":" << split_batches_run
              << ",\"split_overflow_retries\":" << split_overflow_retries
              << ",\"lazy_steps\":" << total_lazy_steps
              << ",\"full_rebuilds\":" << total_full_rebuilds
              << ",\"dynamic_pm\":" << (dynamic_pm ? "true" : "false")
              << ",\"last_pm_span_scale\":" << last_pm_span_scale
              << ",\"last_dt_max_rms\":" << last_dt_max_rms
              << ",\"peak_slab_particles\":" << peak_slab_seen
              << ",\"device_buffer_bytes\":" << sum_dev_bytes
              << ",\"memory_mode\":\"multi_gpu_slab_resident\",\"num_gpus\":" << num_gpus
              << ",\"final_particle_D2H_bytes\":" << output_bytes
              << ",\"export_s\":" << export_s << "}" << std::endl;

    if (pm_dist_enabled) dist_pm.destroy();
    for (int g = 0; g < num_gpus; ++g) {
      if (d_energy_accum[g]) {
        check(cudaSetDevice(g));
        cudaFree(d_energy_accum[g]);
      }
    }
    check(cudaSetDevice(0));
    for (auto& G : gpus) G.cleanup();

    if (fc_target_cap > 0 || lc_enabled) {
      check(cudaSetDevice(0));
      d_ewald_modes.free(aux_dev_bytes);
      d_ewald_C.free(aux_dev_bytes);
      d_ewald_S.free(aux_dev_bytes);
      d_fc_tgt_x.free(aux_dev_bytes);
      d_fc_tgt_h.free(aux_dev_bytes);
      d_fc_lids.free(aux_dev_bytes);
      d_fc_tgt_slab.free(aux_dev_bytes);
      d_fc_tgt_lid.free(aux_dev_bytes);
      d_fc_gpu_short.free(aux_dev_bytes);
      d_fc_gpu_mesh.free(aux_dev_bytes);
      d_fc_short_ref.free(aux_dev_bytes);
      d_fc_total_ref.free(aux_dev_bytes);
      d_lc_hits.free(aux_dev_bytes);
      d_lc_count.free(aux_dev_bytes);
      d_lc_shifts.free(aux_dev_bytes);
    }

    if (lc_enabled && lc_fp) {
      lc_writer.finish();
      if (fseeko(lc_fp, 0, SEEK_SET) != 0) throw std::runtime_error("Cannot seek lightcone header");
      const LightconeHeader final_hdr{{'S','W','L','C','0','0','0','1'}, lc_total_hits,
                                      {lc_obs.x, lc_obs.y, lc_obs.z},
                                      h.initial_scale, a_end, lc_a_obs, lc_r_max, h.box};
      write_bin(lc_fp, &final_hdr, 1);
      if (fclose(lc_fp)) throw std::runtime_error("Cannot close lightcone output");
      lc_fp = nullptr;
      std::cout << std::setprecision(12)
                << "LIGHTCONE_RESULT {\"path\":\"" << lc_out_env
                << "\",\"hits\":" << lc_total_hits
                << ",\"r_max\":" << lc_r_max
                << ",\"a_obs\":" << lc_a_obs
                << ",\"replicate\":" << (lc_replicate ? "true" : "false") << "}" << std::endl;
    }

    if (run_fof && !skip_output_write) {
      check(cudaSetDevice(0));
      const double fof_t0 = wall();
      const double fof_link = fof_link_ratio * h.box / std::cbrt(double(N));
      const int fof_slabs = requested_fof_slabs > 0 ? requested_fof_slabs : ((N <= (128ULL << 20)) ? 1 : 16);
      uint64_t ngrp = 0, ngrouped = 0, largest = 0;
      double fof_el[9]{};
      std::vector<int64_t> fof_ids(N);
      const int fst = (fof_slabs <= 1)
          ? whole_fof_swift(
                reinterpret_cast<const double*>(out_x.data()), int(N),
                h.box, h.box, h.box, fof_link, fof_min_size,
                fof_ids.data(), &ngrp, &ngrouped, &largest, fof_el, nullptr)
          : whole_fof_slabbed_swift(
                reinterpret_cast<const double*>(out_x.data()), N,
                h.box, h.box, h.box, fof_link, fof_min_size, fof_slabs,
                fof_ids.data(), &ngrp, &ngrouped, &largest, fof_el, nullptr);
      if (fst != 0) throw std::runtime_error("FoF failed at final snapshot");
      const std::string fof_path = std::string(argv[2]) + ".fof.bin";
      write_fof_bin(fof_path, N, ngrp, largest, uint64_t(fof_min_size),
                    h.box, a_end, fof_link, fof_ids.data());
      std::cout << std::setprecision(12)
                << "FOF_RESULT {\"index\":\"final\",\"a\":" << a_end
                << ",\"path\":\"" << fof_path
                << "\",\"fof_slabs\":" << fof_slabs
                << ",\"groups\":" << ngrp
                << ",\"grouped_particles\":" << ngrouped
                << ",\"largest\":" << largest
                << ",\"min_size\":" << fof_min_size
                << ",\"link_mpc\":" << fof_link
                << ",\"fof_s\":" << (wall() - fof_t0)
                << ",\"neighbors_union_s\":" << fof_el[6] << "}" << std::endl;
    }

    return sanity_bad ? 6 : 0;
  } catch (const std::exception& e) {
    std::cerr << "Multi-GPU slab resident evolution failed: " << e.what() << std::endl;
    return 5;
  }
}
