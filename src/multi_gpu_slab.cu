#include <algorithm>
#include <array>
#include <cerrno>
#include <cfloat>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <omp.h>

#include "swift_cuda_pm.h"
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
constexpr double hubble0 = 67.36;
constexpr double omega_m = 0.3153, omega_de = 0.6847, omega_cdm = 0.266;
constexpr float dt_displacement_default = 1.121728e-4f;
constexpr double dt_max_default = 1e-2, max_dt_RMS_factor_default = 0.25;
constexpr float grav_G_default = 43.0092f;
constexpr float grav_eta = 0.025f, tolerance = 0.001f, theta = 0.7f;
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
  __host__ __device__ double scale(Tick ti) const {
    return a_begin * exp(log_span * (double(ti) / timeline));
  }
  __host__ __device__ double H(double a) const {
    return hubble0 * sqrt(omega_m / (a * a * a) + omega_de);
  }
  __host__ __device__ double factors(Tick lo, Tick hi, bool drift) const {
    if (lo == hi) return 0.0;
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
  const uint8_t* home_cell;
  const int* order;
};

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
                                    float uniform_mass, const int* order,
                                    float4* packed, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int pid = order[i];
  const double3 pos = x[pid];
  const float mp = mass_or_null ? mass_or_null[pid] : uniform_mass;
  packed[i] = {float(pos.x), float(pos.y), float(pos.z), mp};
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
  const float target_soft = uniform_soft;
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
              rad2 < theta * theta * r2 :
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
            if constexpr (fast_split) {
              iact_grav_pp_truncated_fast(distance, h2, h_inv, h_inv3, a.w, inv_split, &accel, &pot);
            } else {
              iact_grav_pp_truncated(distance, h2, h_inv, h_inv3, a.w, inv_split, &accel, &pot);
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
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    min_rx = fminf(min_rx, __shfl_xor_sync(0xffffffff, min_rx, off));
    max_rx = fmaxf(max_rx, __shfl_xor_sync(0xffffffff, max_rx, off));
    min_ry = fminf(min_ry, __shfl_xor_sync(0xffffffff, min_ry, off));
    max_ry = fmaxf(max_ry, __shfl_xor_sync(0xffffffff, max_ry, off));
    min_rz = fminf(min_rz, __shfl_xor_sync(0xffffffff, min_rz, off));
    max_rz = fmaxf(max_rz, __shfl_xor_sync(0xffffffff, max_rz, off));
    old_min = fminf(old_min, __shfl_xor_sync(0xffffffff, old_min, off));
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
  const float warp_soft = target_soft;

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
            rad2 < theta * theta * r2 :
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
            __syncwarp();
            if ((need_open >> lane) & 1u) {
              for (int k = 0; k < chunk; ++k) {
                if (dx_i == 1 && sorted_i == base + k) continue;
                const float4 a = s_leaf[warp_id][k];
                const float3 r = periodic_delta_float({a.x, a.y, a.z}, x, p.box);
                const float distance = norm2(r);
                float accel, pot;
                if constexpr (fast_split) {
                  iact_grav_pp_truncated_fast(distance, h2, h_inv, h_inv3, a.w, inv_split, &accel, &pot);
                } else {
                  iact_grav_pp_truncated(distance, h2, h_inv, h_inv3, a.w, inv_split, &accel, &pot);
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

__device__ unsigned char next_bin(float3 raw, const float3 mesh, float eps,
    Tick ti, unsigned char previous, Physics p, double a, double H, bool initial,
    float dt_rms, Tick span_cap) {
  const double a_inv2 = 1.0 / (a * a);
  float ax = raw.x * p.G * a_inv2, ay = raw.y * p.G * a_inv2, az = raw.z * p.G * a_inv2;
  ax += mesh.x * a_inv2; ay += mesh.y * a_inv2; az += mesh.z * a_inv2;
  const float ac2 = ax * ax + ay * ay + az * az;
  const float inv = ac2 > 0.f ? 1.f / sqrtf(ac2) : FLT_MAX;
  float dt = sqrtf(2.f * kernel_gravity_softening_plummer_equivalent_inv * float(a) * grav_eta * eps * inv);
  dt = fminf(dt, dt_rms);
  dt *= float(H) * p.short_step_factor;
  if (p.dynamic_pm) dt = fminf(dt, float(p.dt_max));
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
    const int* src_order, int active_n, float uniform_soft, Physics p, Tick ti,
    Meta* stats, const double* finish, const double* begin, double mesh_kick,
    double a, double H, float dt_rms, Tick span_cap, const int* tree_invalid) {
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
    if (mesh_kick) {
      v.x += am.x * mesh_kick;
      v.y += am.y * mesh_kick;
      v.z += am.z * mesh_kick;
    }
    const float3 total = {f.x + am.x / p.G, f.y + am.y / p.G, f.z + am.z / p.G};
    const float oa = sqrtf(norm2(total));
    old_acc_arr[i] = oa;
    if (ti != timeline) {
      const unsigned char b = next_bin({f.x, f.y, f.z}, am, uniform_soft, ti, old, p, a, H, !ti, dt_rms, span_cap);
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

struct DevMigrant {
  double3 x;
  float3 v, mesh_acc;
  float old_acc, mass;
  uint32_t orig_idx;
  int dst_slab;
  unsigned char bin;
  unsigned char pad[7];
};

struct ScanPartial {
  uint64_t active;
  Tick min_nx;
  double v2_sum;
};

__global__ void drift_flag_migrants_kernel(
    double3* d_x, const float3* d_v, unsigned char* d_flags, int ns,
    int cur_slab, double drift_fac, double box, double inv_wx) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= ns) return;
  const float3 vi = d_v[k];
  double3 px = d_x[k];
  px.x = wrapped_coord(px.x + drift_fac * double(vi.x), box);
  px.y = wrapped_coord(px.y + drift_fac * double(vi.y), box);
  px.z = wrapped_coord(px.z + drift_fac * double(vi.z), box);
  d_x[k] = px;
  const int dest = min(max(int(px.x * inv_wx), 0), num_slabs - 1);
  d_flags[k] = (dest != cur_slab) ? 1 : 0;
}

__global__ void pack_migrants_kernel(
    const double3* d_x, const float3* d_v, const float3* d_mesh_acc,
    const float* d_old_acc, const unsigned char* d_bins, const uint32_t* d_orig_idx,
    const float* d_mass, float first_mass, const int* mig_ids, int m_cnt,
    double inv_wx, DevMigrant* out) {
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
  dm.orig_idx = d_orig_idx[k];
  dm.dst_slab = dest;
  dm.bin = d_bins[k];
  out[j] = dm;
}

__global__ void remove_migrants_backwards_kernel(
    double3* d_x, float3* d_v, float3* d_mesh_acc, float* d_old_acc,
    unsigned char* d_bins, uint32_t* d_orig_idx, float* d_mass,
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
    }
  }
}

__global__ void append_migrants_kernel(
    double3* d_x, float3* d_v, float3* d_mesh_acc, float* d_old_acc,
    unsigned char* d_bins, uint32_t* d_orig_idx, float* d_mass,
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
}

__global__ void scan_slab_active_v2_kernel(
    const unsigned char* d_bins, const float3* d_v, int ns, Tick ti,
    bool force_all, bool need_vel_sum, ScanPartial* partials) {
  uint64_t act = 0;
  Tick min_nx = timeline;
  double v2 = 0.0;
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
    if (need_vel_sum) {
      const float3 vi = d_v[k];
      v2 += double(vi.x) * vi.x + double(vi.y) * vi.y + double(vi.z) * vi.z;
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    act += __shfl_down_sync(0xffffffff, act, off);
    const Tick t = __shfl_down_sync(0xffffffff, min_nx, off);
    if (t < min_nx) min_nx = t;
    v2 += __shfl_down_sync(0xffffffff, v2, off);
  }
  __shared__ uint64_t w_act[scan_team / 32];
  __shared__ Tick w_nx[scan_team / 32];
  __shared__ double w_v2[scan_team / 32];
  const int warp = threadIdx.x >> 5;
  if (!(threadIdx.x & 31)) {
    w_act[warp] = act;
    w_nx[warp] = min_nx;
    w_v2[warp] = v2;
  }
  __syncthreads();
  if (!threadIdx.x) {
    uint64_t ta = 0;
    Tick tnx = timeline;
    double tv2 = 0.0;
    for (int w = 0; w < scan_team / 32; ++w) {
      ta += w_act[w];
      if (w_nx[w] < tnx) tnx = w_nx[w];
      tv2 += w_v2[w];
    }
    partials[blockIdx.x] = {ta, tnx, tv2};
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
  DevPtr<float> d_old_acc, d_mass;
  DevPtr<unsigned char> d_bins;
  DevPtr<uint32_t> d_orig_idx;
  int global_id = -1;
  size_t ns = 0, cap = 0;

  void alloc(size_t new_cap, bool uniform_mass, size_t& tracker) {
    cap = new_cap;
    d_x.alloc(cap, tracker);
    d_v.alloc(cap, tracker);
    d_mesh_acc.alloc(cap, tracker);
    d_old_acc.alloc(cap, tracker);
    d_bins.alloc(cap, tracker);
    d_orig_idx.alloc(cap, tracker);
    if (!uniform_mass) d_mass.alloc(cap, tracker);
  }
  void free_all(size_t& tracker) {
    d_x.free(tracker);
    d_v.free(tracker);
    d_mesh_acc.free(tracker);
    d_old_acc.free(tracker);
    d_bins.free(tracker);
    d_orig_idx.free(tracker);
    d_mass.free(tracker);
    cap = 0;
  }
};

struct SlabTreeSlot {
  DevPtr<float4> packed_sources;
  DevPtr<Geometry> geometry;
  DevPtr<ForceNode> force_nodes;
  DevPtr<int> escape, first_child, order, node_count, levels;
  DevPtr<uint8_t> home_cell;
  size_t part_cap = 0, node_cap = 0;
  unsigned int nodes = 0;

  void alloc(size_t pcap, size_t ncap, size_t& tracker) {
    part_cap = pcap;
    node_cap = ncap;
    packed_sources.alloc(pcap, tracker);
    geometry.alloc(ncap, tracker);
    force_nodes.alloc(ncap, tracker);
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
    geometry.free(tracker);
    force_nodes.free(tracker);
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
    return {geometry.p, force_nodes.p, escape.p, first_child.p, packed_sources.p, home_cell.p, order.p};
  }
};

struct GhostTreeView {
  float4* packed_sources = nullptr;
  Geometry* geometry = nullptr;
  ForceNode* force_nodes = nullptr;
  int* escape = nullptr;
  int* first_child = nullptr;
  SlabTreeView view() const {
    return {geometry, force_nodes, escape, first_child, packed_sources, nullptr, nullptr};
  }
};

struct HostSlabStaging {
  std::vector<double3> x;
  std::vector<float3> v;
  std::vector<float> mass;
  std::vector<uint32_t> orig_idx;
};

struct PinnedHostBuf {
  void* p = nullptr;
  size_t bytes = 0;
  void alloc(size_t b) {
    if (bytes >= b) return;
    if (p) cudaFreeHost(p);
    bytes = b;
    check(cudaHostAlloc(&p, bytes, cudaHostAllocDefault));
  }
  void free() {
    if (p) { cudaFreeHost(p); p = nullptr; bytes = 0; }
  }
  ~PinnedHostBuf() { free(); }
};

struct GpuCtx {
  int dev = 0;
  int slabs_per_gpu = 0;
  size_t device_bytes = 0;
  cudaStream_t compute_stream{};
  cudaStream_t copy_stream{};
  cudaStream_t d2h_left_stream{};
  cudaStream_t d2h_right_stream{};
  cudaEvent_t ev_left_ready{}, ev_s0_done{}, ev_right_ready{};

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
  GhostTreeView ghost;
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
  size_t mig_out_cap = 0, mig_in_cap = 0;

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
    const size_t off_gh_fn = push_f(pool_ghost_nodes * sizeof(ForceNode));
    const size_t off_gh_esc = push_f(pool_ghost_nodes * sizeof(int));
    const size_t off_gh_fc = push_f(pool_ghost_nodes * sizeof(int));
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
    ghost.geometry = reinterpret_cast<Geometry*>(at(off_gh_geom));
    ghost.force_nodes = reinterpret_cast<ForceNode*>(at(off_gh_fn));
    ghost.escape = reinterpret_cast<int*>(at(off_gh_esc));
    ghost.first_child = reinterpret_cast<int*>(at(off_gh_fc));
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
    h_bounce.free();
    if (ev_left_ready) { cudaEventDestroy(ev_left_ready); ev_left_ready = nullptr; }
    if (ev_s0_done) { cudaEventDestroy(ev_s0_done); ev_s0_done = nullptr; }
    if (ev_right_ready) { cudaEventDestroy(ev_right_ready); ev_right_ready = nullptr; }
    if (compute_stream) { cudaStreamDestroy(compute_stream); compute_stream = nullptr; }
    if (copy_stream) { cudaStreamDestroy(copy_stream); copy_stream = nullptr; }
    if (d2h_left_stream) { cudaStreamDestroy(d2h_left_stream); d2h_left_stream = nullptr; }
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

size_t compute_slab_cap(size_t n) {
  const double slack = (n > 50000000ULL) ? 1.04 : ((n > 10000000ULL) ? 1.07 : 1.15);
  return std::max<size_t>(4096, size_t(double(n) * slack) + 32768);
}
size_t compute_node_cap(size_t cap, int leaf_limit) {
  return std::max<size_t>(32768, (11ULL * cap) / (2ULL * size_t(leaf_limit)));
}

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
    if (num_gpus <= 0 || num_gpus > dev_count || (num_slabs % num_gpus != 0)) {
      throw std::runtime_error("Invalid SWIFT_NUM_GPUS=" + std::to_string(num_gpus) +
                               " (dev_count=" + std::to_string(dev_count) + ")");
    }
    const int slabs_per_gpu = num_slabs / num_gpus;

    FILE* input = fopen(argv[1], "rb");
    if (!input) return 3;
    StateInput h{};
    read_bin(input, &h, 1);
    const uint64_t N = h.n;
    if (N == 0 || h.box <= 0.0 || h.initial_scale <= 0.0 || h.plummer_softening <= 0.0) return 3;
    const double a_end = std::stod(argv[3]);
    if (!(a_end > h.initial_scale)) return 3;

    const bool dynamic_pm = std::getenv("SWIFT_DYNAMIC_PM") != nullptr;
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
    const char* skip_out_env = std::getenv("SWIFT_SKIP_OUTPUT_WRITE");
    const char* force_comp_path = std::getenv("SWIFT_FORCE_COMPONENTS");
    const char* force3_out_path = std::getenv("SWIFT_FORCE3_OUT");
    const char* slab_axis_env = std::getenv("SWIFT_SLAB_AXIS");
    const char* slab_offset_env = std::getenv("SWIFT_SLAB_OFFSET");
    const char* p2p_env = std::getenv("SWIFT_P2P_MODE");
    std::string p2p_mode = p2p_env ? p2p_env : "nvlink";
    if (p2p_mode != "nvlink" && p2p_mode != "all" && p2p_mode != "off")
      throw std::runtime_error("SWIFT_P2P_MODE must be nvlink, all, or off");

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
    const double dt_max_val = requested_dt_max ? std::stod(requested_dt_max) : dt_max_default;
    const double max_rms_fac = requested_rms_fac ? std::stod(requested_rms_fac) : max_dt_RMS_factor_default;
    const int leaf_limit = leaf_override ? std::stoi(leaf_override) : 64;
    const bool warp_m2l = !m2l_env || std::string(m2l_env) != "0";
    const bool fast_split = !fsplit_env || std::string(fsplit_env) != "0";
    const bool skip_output_write = skip_out_env && std::string(skip_out_env) == "1";

    if (leaf_limit <= 0 || !std::isfinite(step_factor) || step_factor <= 0.f ||
        !std::isfinite(mac) || mac <= 0.f || !std::isfinite(grav_G) || grav_G <= 0.f)
      throw std::runtime_error("Invalid physics parameter");

    // Initialize GPUs & P2P matrix
    std::vector<GpuCtx> gpus(num_gpus);
    std::vector<std::vector<int>> can_peer(num_gpus, std::vector<int>(num_gpus, 0));
    for (int g = 0; g < num_gpus; ++g) {
      gpus[g].dev = g;
      gpus[g].slabs_per_gpu = slabs_per_gpu;
      gpus[g].slabs.resize(slabs_per_gpu);
      gpus[g].trees.resize(slabs_per_gpu);
      check(cudaSetDevice(g));
      check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
      check(cudaStreamCreateWithFlags(&gpus[g].compute_stream, cudaStreamNonBlocking));
      check(cudaStreamCreateWithFlags(&gpus[g].copy_stream, cudaStreamNonBlocking));
      check(cudaStreamCreateWithFlags(&gpus[g].d2h_left_stream, cudaStreamNonBlocking));
      check(cudaStreamCreateWithFlags(&gpus[g].d2h_right_stream, cudaStreamNonBlocking));
      check(cudaEventCreateWithFlags(&gpus[g].ev_left_ready, cudaEventDisableTiming));
      check(cudaEventCreateWithFlags(&gpus[g].ev_s0_done, cudaEventDisableTiming));
      check(cudaEventCreateWithFlags(&gpus[g].ev_right_ready, cudaEventDisableTiming));
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

    // Pass 1: Scan input.bin for mass uniformity & per-axis slab histograms
    const double import_start = wall();
    const uint64_t pos_offset = sizeof(StateInput);
    const uint64_t vel_offset = pos_offset + N * sizeof(double3);
    const uint64_t mass_offset = vel_offset + N * sizeof(float3);

    const uint64_t chunk = std::min<uint64_t>(N, 16ULL << 20);
    std::vector<double3> rx(chunk);
    std::vector<float3> rv(chunk);
    std::vector<float> rm(chunk);

    std::array<std::array<size_t, num_slabs>, 3> axis_counts{};
    double total_mass = 0.0, min_mass = DBL_MAX;
    float first_mass = 0.f;
    bool mass_uniform = true;
    const double inv_wx = double(num_slabs) / h.box;

    FILE* fx = fopen(argv[1], "rb");
    FILE* fv = fopen(argv[1], "rb");
    FILE* fm = fopen(argv[1], "rb");
    if (!fx || !fv || !fm) throw std::runtime_error("Failed reopening input for streaming");
    fseeko(fx, off_t(pos_offset), SEEK_SET);
    fseeko(fv, off_t(vel_offset), SEEK_SET);
    fseeko(fm, off_t(mass_offset), SEEK_SET);

    for (uint64_t base = 0; base < N; base += chunk) {
      const size_t cnt = size_t(std::min<uint64_t>(chunk, N - base));
      read_bin(fx, rx.data(), cnt);
      read_bin(fm, rm.data(), cnt);
      if (base == 0) first_mass = rm[0];
      for (size_t k = 0; k < cnt; ++k) {
        const int sx = std::min(std::max(int(wrapped_coord(rx[k].x, h.box) * inv_wx), 0), num_slabs - 1);
        const int sy = std::min(std::max(int(wrapped_coord(rx[k].y, h.box) * inv_wx), 0), num_slabs - 1);
        const int sz = std::min(std::max(int(wrapped_coord(rx[k].z, h.box) * inv_wx), 0), num_slabs - 1);
        ++axis_counts[0][sx];
        ++axis_counts[1][sy];
        ++axis_counts[2][sz];
        const float mp = rm[k];
        if (mp != first_mass) mass_uniform = false;
        total_mass += double(mp);
        if (mp > 0.f && double(mp) < min_mass) min_mass = double(mp);
      }
    }

    // Optimal cyclic load balance (axis, offset) WITHOUT shifting coordinates
    int slab_axis = 0, slab_offset = 0;
    {
      size_t best_gpu_peak = ~size_t(0), best_slab_peak = ~size_t(0);
      const int ax_lo = (requested_axis >= 0) ? requested_axis : 0;
      const int ax_hi = (requested_axis >= 0) ? requested_axis : 2;
      const int off_lo = (requested_offset >= 0) ? requested_offset : 0;
      const int off_hi = (requested_offset >= 0) ? requested_offset : ((num_gpus == 1) ? 0 : num_slabs - 1);
      for (int ax = ax_lo; ax <= ax_hi; ++ax) {
        size_t slab_peak = 0;
        for (int s = 0; s < num_slabs; ++s) slab_peak = std::max(slab_peak, axis_counts[ax][s]);
        for (int off = off_lo; off <= off_hi; ++off) {
          size_t gpu_peak = 0;
          for (int g = 0; g < num_gpus; ++g) {
            size_t gsum = 0;
            for (int ls = 0; ls < slabs_per_gpu; ++ls)
              gsum += axis_counts[ax][(off + g * slabs_per_gpu + ls) % num_slabs];
            gpu_peak = std::max(gpu_peak, gsum);
          }
          if (gpu_peak < best_gpu_peak || (gpu_peak == best_gpu_peak && slab_peak < best_slab_peak)) {
            best_gpu_peak = gpu_peak;
            best_slab_peak = slab_peak;
            slab_axis = ax;
            slab_offset = off;
          }
        }
      }
    }

    auto owned_slab = [&](int g, int ls) -> int {
      return (slab_offset + g * slabs_per_gpu + ls) % num_slabs;
    };
    auto owner_of_slab = [&](int s) -> std::pair<int, int> {
      const int rel = (s - slab_offset + num_slabs) % num_slabs;
      return {rel / slabs_per_gpu, rel % slabs_per_gpu};
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
    }

    // Pass 2: Stream into per-slab host buckets, upload to owner GPUs, then free host buckets
    fseeko(fx, off_t(pos_offset), SEEK_SET);
    fseeko(fv, off_t(vel_offset), SEEK_SET);
    fseeko(fm, off_t(mass_offset), SEEK_SET);
    const float a0_f = float(h.initial_scale);
    for (uint64_t base = 0; base < N; base += chunk) {
      const size_t cnt = size_t(std::min<uint64_t>(chunk, N - base));
      read_bin(fx, rx.data(), cnt);
      read_bin(fv, rv.data(), cnt);
      if (!mass_uniform) read_bin(fm, rm.data(), cnt);
      for (size_t k = 0; k < cnt; ++k) {
        double3 pos = rx[k];
        pos.x = wrapped_coord(pos.x, h.box);
        pos.y = wrapped_coord(pos.y, h.box);
        pos.z = wrapped_coord(pos.z, h.box);
        pos = permute_in(pos, slab_axis);
        const int s = std::min(std::max(int(pos.x * inv_wx), 0), num_slabs - 1);
        auto& HS = h_stage[s];
        HS.x.push_back(pos);
        const float3 vi = permute_in(rv[k], slab_axis);
        HS.v.push_back(float3{vi.x * a0_f, vi.y * a0_f, vi.z * a0_f});
        HS.orig_idx.push_back(uint32_t(base + k));
        if (!mass_uniform) HS.mass.push_back(rm[k]);
      }
    }
    fclose(fx);
    fclose(fv);
    fclose(fm);
    fclose(input);
    { std::vector<double3>().swap(rx); std::vector<float3>().swap(rv); std::vector<float>().swap(rm); }
    if (const char* unlink_env = std::getenv("SWIFT_UNLINK_INPUT_AFTER_LOAD")) {
      if (std::string(unlink_env) == "1") std::remove(argv[1]);
    }

    // Allocate GPU resident buffers & upload each slab to its owner GPU
    constexpr size_t pm_points = 128ULL * 128ULL * 128ULL;
#pragma omp parallel num_threads(num_gpus)
    {
      const int g = omp_get_thread_num();
      auto& G = gpus[g];
      check(cudaSetDevice(G.dev));

      G.d_active_count.alloc(1, G.device_bytes);
      G.d_tree_invalid.alloc(1, G.device_bytes);
      G.d_stats.alloc(1, G.device_bytes);
      G.d_kick_finish.alloc(57, G.device_bytes);
      G.d_kick_begin.alloc(57, G.device_bytes);
      G.d_pm_density.alloc(pm_points, G.device_bytes);
      G.d_pm_field.alloc(pm_points, G.device_bytes);
      G.d_scan_tmp.alloc(scan_grid_max, G.device_bytes);
      G.h_scan.resize(scan_grid_max);
      G.ensure_mig_out(262144);
      G.ensure_mig_in(262144);

      size_t loc_c = 0, loc_n = 0;
      for (int ls = 0; ls < slabs_per_gpu; ++ls) {
        const int s = owned_slab(g, ls);
        const auto& HS = h_stage[s];
        const size_t sc = HS.x.size();
        const size_t cap = compute_slab_cap(sc);
        const size_t ncap = compute_node_cap(cap, leaf_limit);
        auto& sl = G.slabs[ls];
        sl.global_id = s;
        sl.ns = sc;
        sl.alloc(cap, mass_uniform, G.device_bytes);
        G.trees[ls].alloc(cap, ncap, G.device_bytes);
        loc_c = std::max(loc_c, cap);
        loc_n = std::max(loc_n, ncap);

        if (sc > 0) {
          check(cudaMemcpyAsync(sl.d_x.p, HS.x.data(), sc * sizeof(double3), cudaMemcpyHostToDevice, G.compute_stream));
          check(cudaMemcpyAsync(sl.d_v.p, HS.v.data(), sc * sizeof(float3), cudaMemcpyHostToDevice, G.compute_stream));
          check(cudaMemcpyAsync(sl.d_orig_idx.p, HS.orig_idx.data(), sc * sizeof(uint32_t), cudaMemcpyHostToDevice, G.compute_stream));
          if (!mass_uniform)
            check(cudaMemcpyAsync(sl.d_mass.p, HS.mass.data(), sc * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
        }
        check(cudaMemsetAsync(sl.d_mesh_acc.p, 0, cap * sizeof(float3), G.compute_stream));
        check(cudaMemsetAsync(sl.d_old_acc.p, 0, cap * sizeof(float), G.compute_stream));
        check(cudaMemsetAsync(sl.d_bins.p, 1, cap * sizeof(unsigned char), G.compute_stream));
      }
      const int left_s = (owned_slab(g, 0) + num_slabs - 1) % num_slabs;
      const int right_s = (owned_slab(g, slabs_per_gpu - 1) + 1) % num_slabs;
      const size_t lc = compute_slab_cap(slab_counts[left_s]);
      const size_t rc = compute_slab_cap(slab_counts[right_s]);
      const size_t gh_c = (num_gpus == 1) ? 1 : std::max(lc, rc);
      const size_t gh_n = (num_gpus == 1) ? size_t(slab_roots_count) : compute_node_cap(gh_c, leaf_limit);
#pragma omp critical(phase_pool_alloc)
      G.ensure_phase_pool(loc_c, loc_n, gh_c, gh_n);
      check(cudaStreamSynchronize(G.compute_stream));
    }
    // Free host staging immediately!
    for (int s = 0; s < num_slabs; ++s) {
      std::vector<double3>().swap(h_stage[s].x);
      std::vector<float3>().swap(h_stage[s].v);
      std::vector<float>().swap(h_stage[s].mass);
      std::vector<uint32_t>().swap(h_stage[s].orig_idx);
    }
    check(cudaSetDevice(0));
    const double load_s = wall() - import_start;

    const double log_span = log(a_end / h.initial_scale);
    const float soft = float(h.plummer_softening * kernel_gravity_softening_plummer_equivalent);
    const double rho_crit0 = 3.0 * hubble0 * hubble0 / (8.0 * acos(-1.0) * double(grav_G));
    const double d_dm = cbrt(min_mass / (omega_cdm * rho_crit0));
    const double r_s = 1.25 * h.box / 128.0;
    const Physics p{h.initial_scale, log_span,
        float(h.box), float(r_s), float(4.5 * r_s), grav_G,
        timeline / pm_intervals, timeline / short_intervals, step_factor, mac,
        dt_max_val, std::min(r_s, d_dm), max_rms_fac, dynamic_pm};

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
      for (int ls = 0; ls < slabs_per_gpu; ++ls) gpu_part_counts[g] += gpus[g].slabs[ls].ns;
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
              << ",\"slab_offset\":" << slab_offset << ",\"particles\":" << N
              << ",\"max_initial_slab_n\":" << max_initial_slab_n
              << ",\"per_gpu_particles\":";
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
              << ",\"device_buffer_bytes\":" << sum_dev_bytes
              << ",\"free_vram_bytes\":" << min_free_vram
              << ",\"total_vram_bytes\":" << total_vram_0 << "}" << std::endl;

    std::cout << std::setprecision(12)
              << "GPU_INPUT {\"particles\":" << N << ",\"num_gpus\":" << num_gpus
              << ",\"dynamic_pm\":" << (dynamic_pm ? "true" : "false")
              << ",\"grav_G\":" << p.G << ",\"pm_intervals\":" << pm_intervals
              << ",\"short_cap_intervals\":" << short_intervals
              << ",\"short_step_factor\":" << step_factor
              << ",\"mac_tolerance\":" << mac
              << ",\"a_begin\":" << h.initial_scale << ",\"a_end\":" << a_end
              << ",\"warp_m2l\":" << (warp_m2l ? "true" : "false")
              << ",\"fast_split\":" << (fast_split ? "true" : "false")
              << ",\"tree_leaf_limit\":" << gh.leaf_limit
              << ",\"softening_internal_mpc\":" << soft << ",\"box_mpc\":" << h.box
              << ",\"prepare_and_load_s\":" << load_s
              << ",\"device\":\"" << prop0.name << "\"}" << std::endl;

    // Pinned host PM reduction buffers (16 MiB * G + 64 MiB)
    std::vector<PinnedHostBuf> h_pm_dens(num_gpus);
    for (int g = 0; g < num_gpus; ++g) h_pm_dens[g].alloc(pm_points * sizeof(double));
    PinnedHostBuf h_pm_field;
    h_pm_field.alloc(pm_points * sizeof(double4));

    // Shared host coordination state per step
    std::array<std::vector<DevMigrant>, num_slabs> h_mig_out, h_mig_in;
    std::array<size_t, num_slabs> slab_active{}, slab_migrants{};
    std::array<Tick, num_slabs> slab_inactive_min{}, slab_active_next{};
    std::array<double, num_slabs> slab_v2_sum{};
    std::array<unsigned int, num_slabs> slab_nodes{};
    std::array<unsigned int, num_slabs> slab_invalid{};
    std::array<int, num_slabs> slab_builds{};
    std::array<double, num_slabs> slab_force_ms{}, slab_kick_ms{}, slab_ghost_ms{}, slab_xfer_ms{};
    std::vector<int> phase_pool_need_grow(num_gpus, 0);

    // Final export host outputs (allocated only on is_final)
    std::vector<double3> out_x;
    std::vector<float3> out_v, out_short_f3, out_long_f3;
    std::vector<float> out_pot;
    std::vector<float4> out_short_f4, out_long_f4;
    const double potential_norm = 4.0 * acos(-1.0) * total_mass * p.split * p.split / (h.box * h.box * h.box);

    size_t sanity_bad = 0;
    double sanity_v2_sum = 0.0, sanity_pot_sum = 0.0;
    float sanity_pot_min = FLT_MAX, sanity_pot_max = -FLT_MAX;

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
    double evo_start = wall();

    // Helper: grow slab cleanly via temporary host staging (prevents 2x VRAM spike)
    auto grow_resident_slab = [&](GpuCtx& G, int ls, size_t need_n) {
      auto& S = G.slabs[ls];
      auto& T = G.trees[ls];
      const size_t new_cap = compute_slab_cap(need_n);
      const size_t new_ncap = compute_node_cap(new_cap, leaf_limit);
      std::vector<double3> sx(S.ns);
      std::vector<float3> sv(S.ns), sm(S.ns);
      std::vector<float> so(S.ns), smass;
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
        check(cudaStreamSynchronize(G.compute_stream));
      }
      S.free_all(G.device_bytes);
      T.free_all(G.device_bytes);
      S.alloc(new_cap, mass_uniform, G.device_bytes);
      T.alloc(new_cap, new_ncap, G.device_bytes);
      if (S.ns > 0) {
        check(cudaMemcpyAsync(S.d_x.p, sx.data(), S.ns * sizeof(double3), cudaMemcpyHostToDevice, G.compute_stream));
        check(cudaMemcpyAsync(S.d_v.p, sv.data(), S.ns * sizeof(float3), cudaMemcpyHostToDevice, G.compute_stream));
        check(cudaMemcpyAsync(S.d_mesh_acc.p, sm.data(), S.ns * sizeof(float3), cudaMemcpyHostToDevice, G.compute_stream));
        check(cudaMemcpyAsync(S.d_old_acc.p, so.data(), S.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
        check(cudaMemcpyAsync(S.d_bins.p, sb.data(), S.ns * sizeof(unsigned char), cudaMemcpyHostToDevice, G.compute_stream));
        check(cudaMemcpyAsync(S.d_orig_idx.p, sid.data(), S.ns * sizeof(uint32_t), cudaMemcpyHostToDevice, G.compute_stream));
        if (!mass_uniform)
          check(cudaMemcpyAsync(S.d_mass.p, smass.data(), S.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
        check(cudaStreamSynchronize(G.compute_stream));
      }
    };

    auto grow_tree_nodes = [&](GpuCtx& G, int ls, size_t min_ncap) {
      auto& T = G.trees[ls];
      const size_t pcap = T.part_cap;
      T.free_all(G.device_bytes);
      T.alloc(pcap, min_ncap, G.device_bytes);
    };

    auto build_slab_tree = [&](GpuCtx& G, int ls) {
      auto& S = G.slabs[ls];
      auto& T = G.trees[ls];
      const int k = S.global_id;
      const int nk = int(S.ns);
      if (nk == 0) {
        zero_empty_roots<<<blocks(slab_roots_count), team, 0, G.compute_stream>>>(
            T.force_nodes.p, T.escape.p, T.first_child.p, slab_roots_count);
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
              T.order.p, depth, k, gh, T.levels.p);
        }
        gather_sources_slab<<<p_grid, team, 0, G.compute_stream>>>(
            S.d_x.p, mass_uniform ? nullptr : S.d_mass.p, first_mass,
            T.order.p, T.packed_sources.p, nk);
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

    auto copy_ghost_tree = [&](GpuCtx& dst, const GpuCtx& src, int src_ls) {
      const size_t nk = src.slabs[src_ls].ns;
      const size_t nc = src.trees[src_ls].nodes;
      const auto& st = src.trees[src_ls];
      const double tg0 = wall();
      auto transfer_arr = [&](void* d_ptr, const void* s_ptr, size_t bytes) {
        if (!bytes) return;
        if (can_peer[dst.dev][src.dev]) {
          check(cudaMemcpyPeerAsync(d_ptr, dst.dev, s_ptr, src.dev, bytes, dst.copy_stream));
        } else {
          dst.h_bounce.alloc(bytes);
          cudaStream_t s_stream = (src_ls == 0) ? src.d2h_left_stream : src.d2h_right_stream;
          check(cudaSetDevice(src.dev));
          check(cudaMemcpyAsync(dst.h_bounce.p, s_ptr, bytes, cudaMemcpyDeviceToHost, s_stream));
          check(cudaStreamSynchronize(s_stream));
          check(cudaSetDevice(dst.dev));
          check(cudaMemcpyAsync(d_ptr, dst.h_bounce.p, bytes, cudaMemcpyHostToDevice, dst.copy_stream));
          check(cudaStreamSynchronize(dst.copy_stream));
        }
      };
      if (nk > 0) transfer_arr(dst.ghost.packed_sources, st.packed_sources.p, nk * sizeof(float4));
      if (nc > 0) {
        transfer_arr(dst.ghost.geometry, st.geometry.p, nc * sizeof(Geometry));
        transfer_arr(dst.ghost.force_nodes, st.force_nodes.p, nc * sizeof(ForceNode));
        transfer_arr(dst.ghost.escape, st.escape.p, nc * sizeof(int));
        transfer_arr(dst.ghost.first_child, st.first_child.p, nc * sizeof(int));
      }
      return (wall() - tg0) * 1000.0;
    };

    for (int iteration = 0;; ++iteration) {
      if (iteration > max_steps_guard)
        throw std::runtime_error("Exceeded max_steps_guard");

      const double step_start = wall();
      const bool is_pm_step = (ti == 0 || ti == next_pm_boundary);
      const bool is_final = (ti == timeline);
      const double drift_fac = (ti > 0) ? p.factors(previous, ti, true) : 0.0;
      const double a = p.scale(ti);
      const double H_cur = p.H(a);
      const bool need_vel_sum = (is_pm_step && !is_final && dynamic_pm);

      if (is_final) {
        if (!skip_output_write) {
          out_x.resize(N);
          out_v.resize(N);
          out_pot.resize(N);
        }
        if (force_comp_path) {
          out_short_f4.resize(N);
          out_long_f4.resize(N);
        }
        if (force3_out_path) {
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
          for (int ls = 0; ls < slabs_per_gpu; ++ls) {
            auto& S = G.slabs[ls];
            const int s = S.global_id;
            h_mig_out[s].clear();
            if (S.ns == 0) { slab_migrants[s] = 0; continue; }
            drift_flag_migrants_kernel<<<blocks(int(S.ns)), team, 0, G.compute_stream>>>(
                S.d_x.p, S.d_v.p, G.d_flags, int(S.ns), s, drift_fac, h.box, inv_wx);
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
              G.ensure_mig_out(size_t(m_cnt));
              pack_migrants_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                  S.d_x.p, S.d_v.p, S.d_mesh_acc.p, S.d_old_acc.p, S.d_bins.p,
                  S.d_orig_idx.p, mass_uniform ? nullptr : S.d_mass.p, first_mass,
                  G.d_active_ids, m_cnt, inv_wx, G.d_mig_out.p);
              h_mig_out[s].resize(size_t(m_cnt));
              check(cudaMemcpyAsync(h_mig_out[s].data(), G.d_mig_out.p,
                                    size_t(m_cnt) * sizeof(DevMigrant),
                                    cudaMemcpyDeviceToHost, G.compute_stream));
              remove_migrants_backwards_kernel<<<1, 1, 0, G.compute_stream>>>(
                  S.d_x.p, S.d_v.p, S.d_mesh_acc.p, S.d_old_acc.p, S.d_bins.p,
                  S.d_orig_idx.p, mass_uniform ? nullptr : S.d_mass.p,
                  G.d_active_ids, m_cnt, int(S.ns));
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
          }
#pragma omp barrier
          for (int ls = 0; ls < slabs_per_gpu; ++ls) {
            auto& S = G.slabs[ls];
            const int s = S.global_id;
            const auto& in_vec = h_mig_in[s];
            const size_t in_cnt = in_vec.size();
            if (in_cnt > 0) {
              if (S.ns + in_cnt > S.cap) grow_resident_slab(G, ls, S.ns + in_cnt);
              G.ensure_mig_in(in_cnt);
              check(cudaMemcpyAsync(G.d_mig_in.p, in_vec.data(),
                                    in_cnt * sizeof(DevMigrant),
                                    cudaMemcpyHostToDevice, G.compute_stream));
              append_migrants_kernel<<<blocks(int(in_cnt)), team, 0, G.compute_stream>>>(
                  S.d_x.p, S.d_v.p, S.d_mesh_acc.p, S.d_old_acc.p, S.d_bins.p,
                  S.d_orig_idx.p, mass_uniform ? nullptr : S.d_mass.p,
                  G.d_mig_in.p, int(S.ns), int(in_cnt));
              S.ns += in_cnt;
            }
          }
        }

        // Scan active count, inactive min tick, and optional v2_sum
        for (int ls = 0; ls < slabs_per_gpu; ++ls) {
          const auto& S = G.slabs[ls];
          const int s = S.global_id;
          if (S.ns == 0) {
            slab_active[s] = 0;
            slab_inactive_min[s] = timeline;
            slab_v2_sum[s] = 0.0;
            continue;
          }
          const int g_blk = std::min(scan_grid_max, std::max(1, int((S.ns + scan_team - 1) / scan_team)));
          scan_slab_active_v2_kernel<<<g_blk, scan_team, 0, G.compute_stream>>>(
              S.d_bins.p, S.d_v.p, int(S.ns), ti, (ti == 0 || is_final),
              need_vel_sum, G.d_scan_tmp.p);
          check(cudaMemcpyAsync(G.h_scan.data(), G.d_scan_tmp.p,
                                size_t(g_blk) * sizeof(ScanPartial),
                                cudaMemcpyDeviceToHost, G.compute_stream));
          check(cudaStreamSynchronize(G.compute_stream));
          uint64_t act = 0;
          Tick nx = timeline;
          double v2 = 0.0;
          for (int b = 0; b < g_blk; ++b) {
            act += G.h_scan[b].active;
            if (G.h_scan[b].min_nx < nx) nx = G.h_scan[b].min_nx;
            v2 += G.h_scan[b].v2_sum;
          }
          slab_active[s] = size_t(act);
          slab_inactive_min[s] = nx;
          slab_v2_sum[s] = v2;
        }
#pragma omp barrier
        if (g == 0) {
          drift_ms = (wall() - t_a0) * 1000.0;
          for (int gg = 0; gg < num_gpus; ++gg)
            for (int ls = 0; ls < slabs_per_gpu; ++ls)
              peak_slab_seen = std::max(peak_slab_seen, gpus[gg].slabs[ls].ns);

          if (is_pm_step) {
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
              last_pm_span_scale = double(current_pm_span) / double(timeline) * log_span;
              last_dt_max_rms = dt_max_rms;
              const Tick lo = (ti == 0) ? 0 : (ti - prev_pm_span / 2);
              const Tick hi = ti + next_pm_span / 2;
              pm_kick = p.factors(lo, hi, false);
              next_pm_boundary = ti + next_pm_span;
            }
          }
        }
#pragma omp barrier

        // ================= STEP B: Multi-GPU Resident PM =================
        if (is_pm_step) {
          const double t_pm0 = wall();
          check(cudaMemsetAsync(G.d_pm_density.p, 0, pm_points * sizeof(double), G.compute_stream));
          for (int ls = 0; ls < slabs_per_gpu; ++ls) {
            const auto& S = G.slabs[ls];
            if (S.ns > 0) {
              swift_pm_cuda_deposit_buf(
                  S.d_x.p, mass_uniform ? nullptr : S.d_mass.p, first_mass,
                  S.ns, G.d_pm_density.p, 128, h.box, G.compute_stream);
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
            swift_pm_cuda_solve_on_device(G.d_pm_density.p, G.d_pm_field.p, 128, h.box, r_s, G.compute_stream);
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
          for (int ls = 0; ls < slabs_per_gpu; ++ls) {
            const auto& S = G.slabs[ls];
            if (S.ns > 0) {
              swift_pm_cuda_interpolate_buf(
                  S.d_x.p, S.ns, G.d_pm_field.p, 128, h.box, p.G,
                  S.d_mesh_acc.p, nullptr, G.compute_stream);
            }
          }
          check(cudaStreamSynchronize(G.compute_stream));
#pragma omp barrier
          if (g == 0) pm_ms = (wall() - t_pm0) * 1000.0;
        }

        // ================= STEP C: Parallel Local Slab Tree Build =================
        const double t_c0 = wall();
        for (int ls = 0; ls < slabs_per_gpu; ++ls) {
          const auto& S = G.slabs[ls];
          const int s = S.global_id;
          const bool need_build = is_pm_step ||
              (slab_active[s] > 0) ||
              (slab_active[(s + num_slabs - 1) % num_slabs] > 0) ||
              (slab_active[(s + 1) % num_slabs] > 0);
          if (need_build) {
            build_slab_tree(G, ls);
            slab_builds[s] = 1;
            slab_nodes[s] = G.trees[ls].nodes;
          } else {
            slab_builds[s] = 0;
            slab_nodes[s] = 0;
          }
        }
        check(cudaStreamSynchronize(G.compute_stream));
#pragma omp barrier
        if (g == 0) tree_ms = (wall() - t_c0) * 1000.0;

        // Reconcile phase_pool sizes cleanly before Step D Force/Ghost P2P
        size_t req_loc_c = G.pool_local_cap, req_loc_n = G.pool_local_nodes;
        size_t req_gh_c = G.pool_ghost_cap, req_gh_n = G.pool_ghost_nodes;
        for (int ls = 0; ls < slabs_per_gpu; ++ls) {
          req_loc_c = std::max(req_loc_c, G.slabs[ls].cap);
          req_loc_n = std::max(req_loc_n, G.trees[ls].node_cap);
        }
        if (num_gpus > 1) {
          const int left_global = (owned_slab(g, 0) + num_slabs - 1) % num_slabs;
          const int right_global = (owned_slab(g, slabs_per_gpu - 1) + 1) % num_slabs;
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

        auto run_local_slab = [&](int ls, const SlabTreeView& v_m, const SlabTreeView& v_c, const SlabTreeView& v_p) {
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

          const double tf0 = wall();
          check(cudaMemsetAsync(G.d_tree_invalid.p, 0, sizeof(int), G.compute_stream));
          const auto launch_w = [&](auto m2l_t, auto fsp_t) {
            tree_force_warp_slab<decltype(m2l_t)::value, decltype(fsp_t)::value>
                <<<blocks(force_count), team, 0, G.compute_stream>>>(
                    S.d_x.p, S.d_old_acc.p, soft,
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
          check(cudaGetLastError());
          check(cudaStreamSynchronize(G.compute_stream));
          slab_force_ms[s] = (wall() - tf0) * 1000.0;

          const double tk0 = wall();
          init_meta<<<1, 1, 0, G.compute_stream>>>(G.d_stats.p, G.d_tree_invalid.p, G.trees[ls].nodes);
          integrate_slab<<<blocks(force_count), team, 0, G.compute_stream>>>(
              S.d_v.p, S.d_old_acc.p, S.d_bins.p, G.d_force, S.d_mesh_acc.p,
              active_ptr, G.trees[ls].order.p, force_count, soft, p, ti,
              G.d_stats.p, G.d_kick_finish.p, G.d_kick_begin.p,
              is_pm_step ? pm_kick : 0.0, a, H_cur, dt_max_rms, short_span_cap,
              G.d_tree_invalid.p);
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

          if (is_final) {
            const double te0 = wall();
            swift_pm_cuda_interpolate_buf(
                S.d_x.p, size_t(ns), G.d_pm_field.p, 128, h.box, p.G,
                S.d_mesh_acc.p, G.d_pm_pot, G.compute_stream);

            std::vector<double3> hx(ns);
            std::vector<float3> hv(ns), hmesh;
            std::vector<float4> hraw(ns);
            std::vector<float> hpot(ns);
            std::vector<uint32_t> hidx(ns);
            check(cudaMemcpyAsync(hx.data(), S.d_x.p, size_t(ns) * sizeof(double3), cudaMemcpyDeviceToHost, G.compute_stream));
            check(cudaMemcpyAsync(hv.data(), S.d_v.p, size_t(ns) * sizeof(float3), cudaMemcpyDeviceToHost, G.compute_stream));
            check(cudaMemcpyAsync(hraw.data(), G.d_force, size_t(ns) * sizeof(float4), cudaMemcpyDeviceToHost, G.compute_stream));
            check(cudaMemcpyAsync(hpot.data(), G.d_pm_pot, size_t(ns) * sizeof(float), cudaMemcpyDeviceToHost, G.compute_stream));
            check(cudaMemcpyAsync(hidx.data(), S.d_orig_idx.p, size_t(ns) * sizeof(uint32_t), cudaMemcpyDeviceToHost, G.compute_stream));
            if (force_comp_path || force3_out_path) {
              hmesh.resize(ns);
              check(cudaMemcpyAsync(hmesh.data(), S.d_mesh_acc.p, size_t(ns) * sizeof(float3), cudaMemcpyDeviceToHost, G.compute_stream));
            }
            check(cudaStreamSynchronize(G.compute_stream));
            slab_xfer_ms[s] = (wall() - te0) * 1000.0;

            const float inv_a = float(1.0 / a_end);
            size_t loc_bad = 0;
            double loc_v2 = 0.0, loc_psum = 0.0;
            float loc_pmin = FLT_MAX, loc_pmax = -FLT_MAX;
            for (int k = 0; k < ns; ++k) {
              const float4 rf = hraw[k];
              const float pp = hpot[k];
              const float pot_val = float(double(rf.w) + potential_norm) * p.G + pp;
              const double3 xk = hx[k];
              const float3 vk = hv[k];
              if (!std::isfinite(xk.x) || !std::isfinite(xk.y) || !std::isfinite(xk.z) ||
                  xk.x < 0.0 || xk.x >= h.box || xk.y < 0.0 || xk.y >= h.box || xk.z < 0.0 || xk.z >= h.box ||
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
              if (force_comp_path) {
                const float3 sf3 = permute_out(float3{rf.x, rf.y, rf.z}, slab_axis);
                out_short_f4[gid] = float4{sf3.x, sf3.y, sf3.z, rf.w};
                const float3 ma = permute_out(hmesh[k], slab_axis);
                out_long_f4[gid] = float4{ma.x, ma.y, ma.z, pp};
              }
              if (force3_out_path) {
                out_short_f3[gid] = permute_out(float3{rf.x, rf.y, rf.z}, slab_axis);
                out_long_f3[gid] = permute_out(hmesh[k], slab_axis);
              }
            }
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

        if (num_gpus == 1) {
          for (int ls = 0; ls < slabs_per_gpu; ++ls) {
            slab_ghost_ms[ls] = 0.0;
            run_local_slab(ls, G.trees[(ls + 11) % 12].view(), G.trees[ls].view(), G.trees[(ls + 1) % 12].view());
          }
        } else {
          const int left_global = (owned_slab(g, 0) + num_slabs - 1) % num_slabs;
          const int right_global = (owned_slab(g, slabs_per_gpu - 1) + 1) % num_slabs;
          const auto [lg, lls] = owner_of_slab(left_global);
          const auto [rg, rls] = owner_of_slab(right_global);

          const int s0 = owned_slab(g, 0);
          const int slast = owned_slab(g, slabs_per_gpu - 1);
          const bool need_s0 = (G.slabs[0].ns > 0) && (is_pm_step || slab_active[s0] > 0);
          const bool need_last = (slabs_per_gpu > 1) && (G.slabs[slabs_per_gpu - 1].ns > 0) && (is_pm_step || slab_active[slast] > 0);

          if (need_s0) {
            slab_ghost_ms[s0] = copy_ghost_tree(G, gpus[lg], lls);
            check(cudaEventRecord(G.ev_left_ready, G.copy_stream));
            check(cudaStreamWaitEvent(G.compute_stream, G.ev_left_ready, 0));
            run_local_slab(0, G.ghost.view(), G.trees[0].view(), G.trees[1].view());
            check(cudaEventRecord(G.ev_s0_done, G.compute_stream));
          } else {
            slab_ghost_ms[s0] = 0.0;
            run_local_slab(0, G.ghost.view(), G.trees[0].view(), G.trees[1].view());
          }

          if (need_last) {
            if (need_s0) check(cudaStreamWaitEvent(G.copy_stream, G.ev_s0_done, 0));
            slab_ghost_ms[slast] = copy_ghost_tree(G, gpus[rg], rls);
            check(cudaEventRecord(G.ev_right_ready, G.copy_stream));
          } else if (slabs_per_gpu > 1) {
            slab_ghost_ms[slast] = 0.0;
          }

          for (int ls = 1; ls < slabs_per_gpu - 1; ++ls) {
            slab_ghost_ms[owned_slab(g, ls)] = 0.0;
            run_local_slab(ls, G.trees[ls - 1].view(), G.trees[ls].view(), G.trees[ls + 1].view());
          }

          if (slabs_per_gpu > 1) {
            if (need_last) check(cudaStreamWaitEvent(G.compute_stream, G.ev_right_ready, 0));
            run_local_slab(slabs_per_gpu - 1,
                           G.trees[slabs_per_gpu - 2].view(),
                           G.trees[slabs_per_gpu - 1].view(),
                           G.ghost.view());
          }
          check(cudaStreamSynchronize(G.copy_stream));
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
          for (int ls = 0; ls < slabs_per_gpu; ++ls) {
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
        throw std::runtime_error("Invalid multi-GPU slab state at step " + std::to_string(iteration) +
                                 " invalid=" + std::to_string(step_invalid) +
                                 " active=" + std::to_string(step_active));

      const double step_wall = wall() - step_start;
      std::cout << "STEP {\"step\":" << iteration << ",\"a\":" << a
                << ",\"integer_time\":" << ti << ",\"active\":" << step_active
                << ",\"nodes\":" << step_nodes << ",\"wall_s\":" << step_wall
                << ",\"drift_ms\":" << drift_ms << ",\"key_sort_ms\":0"
                << ",\"tree_multipoles_ms\":" << tree_ms << ",\"force_ms\":" << force_ms
                << ",\"kick_bins_ms\":" << kick_ms
                << ",\"h2d_d2h_ms\":" << (ghost_ms + export_ms)
                << ",\"ghost_ms\":" << ghost_ms
                << ",\"slab_builds\":" << step_builds << ",\"migrants\":" << total_migrants
                << ",\"pm\":" << (is_pm_step ? "true" : "false")
                << ",\"pm_device_ms\":" << pm_ms << "}" << std::endl;

      if (is_pm_step) {
        std::cout << "PROGRESS {\"pm_updates\":" << pm_updates
                  << ",\"z\":" << 1.0 / a - 1.0
                  << ",\"dt_max_rms\":" << dt_max_rms
                  << ",\"next_pm_span\":" << next_pm_span
                  << ",\"steps\":" << iteration
                  << ",\"elapsed_s\":" << (wall() - evo_start) << "}" << std::endl;
      }

      if (is_final) break;
      previous = ti;
      ti = step_next;
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
              << ",\"dynamic_pm\":" << (dynamic_pm ? "true" : "false")
              << ",\"last_pm_span_scale\":" << last_pm_span_scale
              << ",\"last_dt_max_rms\":" << last_dt_max_rms
              << ",\"peak_slab_particles\":" << peak_slab_seen
              << ",\"device_buffer_bytes\":" << sum_dev_bytes
              << ",\"memory_mode\":\"multi_gpu_slab_resident\",\"num_gpus\":" << num_gpus
              << ",\"final_particle_D2H_bytes\":" << output_bytes
              << ",\"export_s\":" << export_s << "}" << std::endl;

    for (auto& G : gpus) G.cleanup();
    return sanity_bad ? 6 : 0;
  } catch (const std::exception& e) {
    std::cerr << "Multi-GPU slab resident evolution failed: " << e.what() << std::endl;
    return 5;
  }
}
