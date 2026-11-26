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
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <omp.h>

#include "validated_device_tree.cuh"
#include "gpu_gravity_functions.cuh"
#ifdef GRAV_FAST_M2P_EXP
#define expf __expf
#endif
#include "m2p_native_math.cuh"
#ifdef GRAV_FAST_M2P_EXP
#undef expf
#endif
#undef min
#undef max

using Tick = unsigned long long;
constexpr Tick timeline = 1ULL << 57;
constexpr int num_slabs = 12;
constexpr int slab_roots_count = 12 * 12;  // 144 top-level cells per slab
constexpr int tree_levels = 14;            // up to 14 octree levels (max DFS stack 1+7*14=99 < 104)
constexpr double hubble0 = 67.36;
constexpr double omega_m = 0.3153, omega_de = 0.6847, omega_cdm = 0.266;
constexpr float dt_displacement_default = 1.121728e-4f;
constexpr double dt_max_default = 1e-2, max_dt_RMS_factor_default = 0.25;
constexpr float grav_G_default = 43.0092f;
constexpr float grav_eta = 0.025f, tolerance = 0.001f, theta = 0.7f;
constexpr int team = 128;
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

__global__ void map_active_unsorted(const int* order, const int* active_sorted,
                                    int* active_unsorted, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) active_unsorted[idx] = order[active_sorted[idx]];
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
            grav_cuda_gravity_M2P(&source.m, d.x, d.y, d.z, r2, soft, 1, inv_split, &contribution);
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
      grav_cuda_potential_derivatives_compute_M2L(
          dm.x, dm.y, dm.z, r2m, rsqrtf(r2m), softm, 1, inv_split, &deriv);
      grav_cuda_gravity_M2L_apply(&lf, &src.m, &deriv);
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
          grav_cuda_gravity_M2P(&source.m, d.x, d.y, d.z, r2, soft, 1, inv_split, &contribution);
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

struct PinnedStager {
  static constexpr size_t chunk_bytes = 24ULL << 20;  // 24 MiB (multiple of 24, 16, 12, 8, 4, 1)
  void* buf[2]{};
  cudaEvent_t ev[2]{};
  bool used[2]{false, false};
  int cursor = 0;

  PinnedStager() {
    for (int i = 0; i < 2; ++i) {
      check(cudaHostAlloc(&buf[i], chunk_bytes, cudaHostAllocDefault));
      check(cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming));
    }
  }
  ~PinnedStager() {
    for (int i = 0; i < 2; ++i) {
      if (used[i]) cudaEventSynchronize(ev[i]);
      cudaEventDestroy(ev[i]);
      cudaFreeHost(buf[i]);
    }
  }
  PinnedStager(const PinnedStager&) = delete;
  PinnedStager& operator=(const PinnedStager&) = delete;

  static void parallel_memcpy(void* dst, const void* src, size_t len) {
    if (len < (4ULL << 20)) {
      std::memcpy(dst, src, len);
      return;
    }
    auto* d = static_cast<char*>(dst);
    const auto* s = static_cast<const char*>(src);
    constexpr int nthreads = 8;
    const size_t stripe = (len + nthreads - 1) / nthreads;
#pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int t = 0; t < nthreads; ++t) {
      const size_t begin = size_t(t) * stripe;
      if (begin < len) {
        const size_t cnt = std::min(stripe, len - begin);
        std::memcpy(d + begin, s + begin, cnt);
      }
    }
  }

  void upload(void* d_dst, const void* h_src, size_t bytes, cudaStream_t stream) {
    if (!bytes) return;
    const auto* src = static_cast<const char*>(h_src);
    auto* dst = static_cast<char*>(d_dst);
    for (size_t off = 0; off < bytes; off += chunk_bytes) {
      const size_t len = std::min(chunk_bytes, bytes - off);
      const int slot = (cursor++) & 1;
      if (used[slot]) check(cudaEventSynchronize(ev[slot]));
      parallel_memcpy(buf[slot], src + off, len);
      check(cudaMemcpyAsync(dst + off, buf[slot], len, cudaMemcpyHostToDevice, stream));
      check(cudaEventRecord(ev[slot], stream));
      used[slot] = true;
    }
  }

  void download(void* h_dst, const void* d_src, size_t bytes, cudaStream_t stream) {
    if (!bytes) return;
    auto* dst = static_cast<char*>(h_dst);
    const auto* src = static_cast<const char*>(d_src);
    const size_t nchunks = (bytes + chunk_bytes - 1) / chunk_bytes;
    int first_slot = 0;
    for (size_t idx = 0; idx < nchunks; ++idx) {
      const size_t off = idx * chunk_bytes;
      const size_t len = std::min(chunk_bytes, bytes - off);
      const int slot = (idx == 0) ? ((cursor++) & 1) : ((first_slot + int(idx)) & 1);
      if (idx == 0) {
        first_slot = slot;
        if (used[slot]) check(cudaEventSynchronize(ev[slot]));
        check(cudaMemcpyAsync(buf[slot], src + off, len, cudaMemcpyDeviceToHost, stream));
        check(cudaEventRecord(ev[slot], stream));
        used[slot] = true;
      }
      if (idx + 1 < nchunks) {
        const size_t noff = (idx + 1) * chunk_bytes;
        const size_t nlen = std::min(chunk_bytes, bytes - noff);
        const int nslot = (cursor++) & 1;
        if (used[nslot]) check(cudaEventSynchronize(ev[nslot]));
        check(cudaMemcpyAsync(buf[nslot], src + noff, nlen, cudaMemcpyDeviceToHost, stream));
        check(cudaEventRecord(ev[nslot], stream));
        used[nslot] = true;
      }
      check(cudaEventSynchronize(ev[slot]));
      parallel_memcpy(dst + off, buf[slot], len);
    }
  }
};

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
  ~DevPtr() {
    if (p) cudaFree(p);
  }
};

struct SlabTreeSlot {
  DevPtr<float4> packed_sources;
  DevPtr<Geometry> geometry;
  DevPtr<ForceNode> force_nodes;
  DevPtr<int> escape, first_child, order, node_count, levels;
  DevPtr<uint8_t> home_cell;
  int slab_id = -1;
  unsigned int nodes = 0;

  void alloc(size_t slab_cap, size_t node_cap, size_t& tracker) {
    slab_id = -1;
    packed_sources.alloc(slab_cap, tracker);
    geometry.alloc(node_cap, tracker);
    force_nodes.alloc(node_cap, tracker);
    escape.alloc(node_cap, tracker);
    first_child.alloc(node_cap, tracker);
    order.alloc(slab_cap, tracker);
    home_cell.alloc(slab_cap, tracker);
    node_count.alloc(1, tracker);
    levels.alloc(tree_levels + 2, tracker);
  }
  SlabTreeView view() const {
    return {geometry.p, force_nodes.p, escape.p, first_child.p, packed_sources.p, home_cell.p, order.p};
  }
};

struct SlabHost {
  std::vector<double3> x;
  std::vector<float3> v;
  std::vector<float3> mesh_acc;
  std::vector<float> old_acc;
  std::vector<unsigned char> bins;
  std::vector<uint32_t> orig_idx;
  std::vector<float> mass;  // non-empty only if masses are non-uniform
};

struct Migrant {
  double3 x;
  float3 v, mesh_acc;
  float old_acc;
  unsigned char bin;
  uint32_t orig_idx;
  float mass;
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

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " <input.bin> <output.bin> <a_end>\n";
    return 2;
  }
  try {
    const double started = wall();
    check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

    FILE* input = fopen(argv[1], "rb");
    if (!input) return 3;
    StateInput h{};
    read_bin(input, &h, 1);
    const uint64_t N = h.n;
    if (N == 0 || h.box <= 0.0 || h.initial_scale <= 0.0 || h.plummer_softening <= 0.0) return 3;
    const double a_end = std::stod(argv[3]);
    if (!(a_end > h.initial_scale)) return 3;

    const bool dynamic_pm = std::getenv("GRAV_DYNAMIC_PM") != nullptr;
    const char* requested_pm = std::getenv("GRAV_PM_INTERVALS");
    const int pm_intervals = requested_pm ? std::stoi(requested_pm) : (dynamic_pm ? 16 : 1);
    if (pm_intervals <= 0 || pm_intervals > 4096 || (pm_intervals & (pm_intervals - 1)))
      throw std::runtime_error("GRAV_PM_INTERVALS must be a power of two between 1 and 4096");
    const char* requested_short = std::getenv("GRAV_SHORT_CAP_INTERVALS");
    const int short_intervals = requested_short ? std::stoi(requested_short) : pm_intervals;
    if (short_intervals < pm_intervals || short_intervals > 4096 ||
        (short_intervals & (short_intervals - 1)))
      throw std::runtime_error("Invalid GRAV_SHORT_CAP_INTERVALS");

    const char* requested_step = std::getenv("GRAV_SHORT_STEP_FACTOR");
    const char* requested_mac = std::getenv("GRAV_MAC_TOLERANCE");
    const char* requested_g = std::getenv("GRAV_G");
    const char* requested_dt_max = std::getenv("GRAV_DT_MAX");
    const char* requested_rms_fac = std::getenv("GRAV_MAX_DT_RMS_FACTOR");
    const char* leaf_override = std::getenv("GRAV_LEAF_LIMIT");
    const char* m2l_env = std::getenv("GRAV_WARP_M2L");
    const char* fsplit_env = std::getenv("GRAV_FAST_SPLIT");
    const char* skip_out_env = std::getenv("GRAV_SKIP_OUTPUT_WRITE");
    const char* force_comp_path = std::getenv("GRAV_FORCE_COMPONENTS");
    const char* slab_axis_env = std::getenv("GRAV_SLAB_AXIS");
    int requested_axis = -1;
    if (slab_axis_env && std::string(slab_axis_env) != "auto") {
      requested_axis = std::stoi(slab_axis_env);
      if (requested_axis < 0 || requested_axis > 2)
        throw std::runtime_error("GRAV_SLAB_AXIS must be auto, 0, 1, or 2");
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

    // Chunked streaming read from input.bin (double3[N], float3[N], float[N])
    const double import_start = wall();
    const uint64_t pos_offset = sizeof(StateInput);
    const uint64_t vel_offset = pos_offset + N * sizeof(double3);
    const uint64_t mass_offset = vel_offset + N * sizeof(float3);

    const uint64_t chunk = std::min<uint64_t>(N, 16ULL << 20);
    std::vector<double3> rx(chunk);
    std::vector<float3> rv(chunk);
    std::vector<float> rm(chunk);

    // First pass: check mass uniformity & count particles per slab across all 3 axes
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

    int slab_axis = requested_axis;
    if (slab_axis < 0) {
      size_t best_peak = ~size_t(0);
      for (int ax = 0; ax < 3; ++ax) {
        size_t ax_max = 0;
        for (int s = 0; s < num_slabs; ++s) ax_max = std::max(ax_max, axis_counts[ax][s]);
        if (ax_max < best_peak) {
          best_peak = ax_max;
          slab_axis = ax;
        }
      }
    }
    const std::array<size_t, num_slabs>& slab_counts = axis_counts[slab_axis];

    std::array<SlabHost, num_slabs> slabs;
    size_t max_slab_n = 0;
    for (int s = 0; s < num_slabs; ++s) {
      const size_t sc = slab_counts[s];
      max_slab_n = std::max(max_slab_n, sc);
      const size_t res = size_t(double(sc) * 1.05) + 1024;
      slabs[s].x.reserve(res);
      slabs[s].v.reserve(res);
      slabs[s].mesh_acc.resize(sc, float3{0.f, 0.f, 0.f});
      slabs[s].old_acc.resize(sc, 0.f);
      slabs[s].bins.resize(sc, 1);
      slabs[s].orig_idx.reserve(res);
      if (!mass_uniform) slabs[s].mass.reserve(res);
    }

    // Second pass: stream positions, velocities, masses into slab buckets (permuted to slab_axis)
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
        auto& S = slabs[s];
        S.x.push_back(pos);
        const float3 vi = permute_in(rv[k], slab_axis);
        S.v.push_back(float3{vi.x * a0_f, vi.y * a0_f, vi.z * a0_f});
        S.orig_idx.push_back(uint32_t(base + k));
        if (!mass_uniform) S.mass.push_back(rm[k]);
      }
    }
    fclose(fx);
    fclose(fv);
    fclose(fm);
    fclose(input);
    { std::vector<double3>().swap(rx); std::vector<float3>().swap(rv); std::vector<float>().swap(rm); }
    const char* unlink_env = std::getenv("GRAV_UNLINK_INPUT_AFTER_LOAD");
    if (unlink_env && std::string(unlink_env) == "1") {
      std::remove(argv[1]);
    }
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

    cudaStream_t stream{};
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    PinnedStager stager;

    size_t device_bytes = 0;
    size_t slab_cap = 0, node_cap = 0, sort_bytes = 0, select_bytes = 0;
    std::array<SlabTreeSlot, 3> slots;
    DevPtr<double3> d_x;
    DevPtr<float> d_mass;
    DevPtr<int> d_active_count, d_tree_invalid;
    DevPtr<Meta> d_stats;
    DevPtr<double> d_kick_finish, d_kick_begin;
    DevPtr<unsigned char> d_select_storage, d_work_pool;

    // Pooled device pointers aliased across disjoint build and target phases:
    Key *d_key_tmp = nullptr, *d_keys_sorted = nullptr;
    int* d_ord_tmp = nullptr;
    double3* d_node_com = nullptr;
    unsigned char* d_sort_storage = nullptr;

    float3 *d_v = nullptr, *d_mesh_acc = nullptr;
    float *d_old_acc = nullptr, *d_pm_pot = nullptr;
    unsigned char *d_bins = nullptr, *d_flags = nullptr;
    float4* d_force = nullptr;
    int* d_active_ids = nullptr;

    d_active_count.alloc(1, device_bytes);
    d_tree_invalid.alloc(1, device_bytes);
    d_stats.alloc(1, device_bytes);
    d_kick_finish.alloc(57, device_bytes);
    d_kick_begin.alloc(57, device_bytes);

    auto ensure_slab_cap = [&](size_t need) {
      if (need <= slab_cap) return;
      const double slack = (need > 100000000ULL) ? 1.06 : 1.15;
      const size_t next_cap = std::max<size_t>(need, size_t(double(need) * slack) + 65536);
      const size_t next_nodes = std::max<size_t>(32768, (6ULL * next_cap) / size_t(leaf_limit));
      slab_cap = next_cap;
      node_cap = next_nodes;
      for (auto& sl : slots) sl.alloc(slab_cap, node_cap, device_bytes);
      d_x.alloc(slab_cap, device_bytes);
      if (!mass_uniform) d_mass.alloc(slab_cap, device_bytes);

      size_t sb = 0, selb = 0;
      check(cub::DeviceRadixSort::SortPairs(
          nullptr, sb, static_cast<Key*>(nullptr), static_cast<Key*>(nullptr),
          static_cast<int*>(nullptr), static_cast<int*>(nullptr), int(slab_cap), 0, 59));
      cub::CountingInputIterator<int> seq(0);
      check(cub::DeviceSelect::Flagged(
          nullptr, selb, seq, static_cast<unsigned char*>(nullptr),
          static_cast<int*>(nullptr), d_active_count.p, int(slab_cap)));
      sort_bytes = sb;
      select_bytes = selb;
      d_select_storage.alloc(std::max<size_t>(select_bytes, 1), device_bytes);

      auto align_up = [](size_t v) { return (v + 255ULL) & ~255ULL; };
      size_t build_bytes = 0;
      auto push_build = [&](size_t bytes) {
        const size_t off = build_bytes;
        build_bytes += align_up(std::max<size_t>(bytes, 1));
        return off;
      };
      const size_t off_key_tmp = push_build(slab_cap * sizeof(Key));
      const size_t off_keys_sorted = push_build(slab_cap * sizeof(Key));
      const size_t off_ord_tmp = push_build(slab_cap * sizeof(int));
      const size_t off_node_com = push_build(node_cap * sizeof(double3));
      const size_t off_sort_storage = push_build(sort_bytes);

      size_t target_bytes = 0;
      auto push_target = [&](size_t bytes) {
        const size_t off = target_bytes;
        target_bytes += align_up(std::max<size_t>(bytes, 1));
        return off;
      };
      const size_t off_v = push_target(slab_cap * sizeof(float3));
      const size_t off_mesh_acc = push_target(slab_cap * sizeof(float3));
      const size_t off_old_acc = push_target(slab_cap * sizeof(float));
      const size_t off_bins = push_target(slab_cap * sizeof(unsigned char));
      const size_t off_force = push_target(slab_cap * sizeof(float4));
      const size_t off_pm_pot = push_target(slab_cap * sizeof(float));
      const size_t off_flags = push_target(slab_cap * sizeof(unsigned char));
      const size_t off_active_ids = push_target(slab_cap * sizeof(int));

      d_work_pool.alloc(std::max(build_bytes, target_bytes), device_bytes);
      auto at = [&](size_t off) -> unsigned char* { return d_work_pool.p + off; };
      d_key_tmp = reinterpret_cast<Key*>(at(off_key_tmp));
      d_keys_sorted = reinterpret_cast<Key*>(at(off_keys_sorted));
      d_ord_tmp = reinterpret_cast<int*>(at(off_ord_tmp));
      d_node_com = reinterpret_cast<double3*>(at(off_node_com));
      d_sort_storage = at(off_sort_storage);

      d_v = reinterpret_cast<float3*>(at(off_v));
      d_mesh_acc = reinterpret_cast<float3*>(at(off_mesh_acc));
      d_old_acc = reinterpret_cast<float*>(at(off_old_acc));
      d_bins = at(off_bins);
      d_force = reinterpret_cast<float4*>(at(off_force));
      d_pm_pot = reinterpret_cast<float*>(at(off_pm_pot));
      d_flags = at(off_flags);
      d_active_ids = reinterpret_cast<int*>(at(off_active_ids));
    };
    ensure_slab_cap(max_slab_n);

    // Initialize PM buffers upfront so MEMORY_PLAN includes PM VRAM.
    grav_pm_cuda_begin(128, stream);
    check(cudaStreamSynchronize(stream));
    size_t free_vram = 0, total_vram = 0;
    check(cudaMemGetInfo(&free_vram, &total_vram));
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0));

    std::cout << "MEMORY_PLAN {\"mode\":\"slab_device\",\"slabs\":" << num_slabs
              << ",\"slab_axis\":" << slab_axis
              << ",\"particles\":" << N << ",\"max_initial_slab_n\":" << max_slab_n
              << ",\"slab_cap\":" << slab_cap << ",\"node_cap\":" << node_cap
              << ",\"uniform_mass\":" << (mass_uniform ? "true" : "false")
              << ",\"device_buffer_bytes\":" << device_bytes
              << ",\"work_pool_bytes\":" << d_work_pool.bytes
              << ",\"sort_scratch_bytes\":" << sort_bytes
              << ",\"free_vram_bytes\":" << free_vram
              << ",\"total_vram_bytes\":" << total_vram << "}" << std::endl;

    std::cout << std::setprecision(12)
              << "GPU_INPUT {\"particles\":" << N
              << ",\"dynamic_pm\":" << (dynamic_pm ? "true" : "false")
              << ",\"grav_G\":" << p.G
              << ",\"pm_intervals\":" << pm_intervals
              << ",\"short_cap_intervals\":" << short_intervals
              << ",\"short_step_factor\":" << step_factor
              << ",\"mac_tolerance\":" << mac
              << ",\"a_begin\":" << h.initial_scale << ",\"a_end\":" << a_end
              << ",\"warp_m2l\":" << (warp_m2l ? "true" : "false")
              << ",\"fast_split\":" << (fast_split ? "true" : "false")
              << ",\"tree_leaf_limit\":" << gh.leaf_limit
              << ",\"softening_internal_mpc\":" << soft << ",\"box_mpc\":" << h.box
              << ",\"prepare_and_load_s\":" << load_s
              << ",\"device\":\"" << prop.name << "\"}" << std::endl;

    // Deferred final output scatter buffers on host (allocated only on is_final to keep evolution RSS low)
    std::vector<double3> out_x;
    std::vector<float3> out_v;
    std::vector<float> out_pot;
    std::vector<float4> out_short_f, out_long_f;
    // Lazy-grown sparse active-target staging buffers on host
    std::vector<int> h_act_unsorted;
    std::vector<double3> h_act_x;
    std::vector<float3> h_act_v, h_act_mesh;
    std::vector<float> h_act_old;
    std::vector<unsigned char> h_act_bins;
    const double potential_norm = 4.0 * acos(-1.0) * total_mass * p.split * p.split / (h.box * h.box * h.box);

    Tick previous = 0, ti = 0;
    Tick next_pm_boundary = 0;
    Tick prev_pm_span = 0, current_pm_span = p.pm_span, next_pm_span = p.pm_span;
    Tick short_span_cap = p.short_span_cap;
    float dt_max_rms = dt_displacement_default;
    double pm_kick = 0.0;
    unsigned int pm_updates = 0;
    double last_pm_span_scale = log_span / double(pm_intervals);
    float last_dt_max_rms = dt_displacement_default;
    size_t peak_slab_seen = max_slab_n;
    double total_wall_evo = 0.0;

    // Host sanity summary on final step
    size_t sanity_bad = 0;
    double sanity_v2_sum = 0.0, sanity_pot_sum = 0.0;
    float sanity_pot_min = FLT_MAX, sanity_pot_max = -FLT_MAX;

    std::vector<float> h_pot_buf;
    std::vector<float4> h_raw_buf;
    if (!skip_output_write || force_comp_path) {
      h_pot_buf.resize(slab_cap);
      h_raw_buf.resize(slab_cap);
    }
    // Also small host chunk buffer for sanity stats even if skip_output_write=1
    std::vector<float> sanity_pm_pot;
    if (skip_output_write && !force_comp_path) {
      sanity_pm_pot.resize(slab_cap);
      h_raw_buf.resize(slab_cap);
    }

    auto ensure_slab_tree = [&](int k, double& tree_ms_acc, int& builds) {
      auto& sl = slots[k % 3];
      if (sl.slab_id == k) return;
      const double t0 = wall();
      const auto& S = slabs[k];
      const int nk = int(S.x.size());
      if (nk == 0) {
        sl.slab_id = k;
        sl.nodes = 0;
        check(cudaMemsetAsync(d_tree_invalid.p, 0, sizeof(int), stream));
        return;
      }
      stager.upload(d_x.p, S.x.data(), size_t(nk) * sizeof(double3), stream);
      if (!mass_uniform)
        stager.upload(d_mass.p, S.mass.data(), size_t(nk) * sizeof(float), stream);

      check(cudaMemsetAsync(d_tree_invalid.p, 0, sizeof(int), stream));
      const int p_grid = blocks(nk);
      key_particles_fast_slab<<<p_grid, team, 0, stream>>>(
          d_x.p, nk, k, gh, d_key_tmp, d_ord_tmp);
      check(cub::DeviceRadixSort::SortPairs(
          d_sort_storage, sort_bytes, d_key_tmp, d_keys_sorted,
          d_ord_tmp, sl.order.p, nk, 0, 59, stream));
      extract_home_cells<<<p_grid, team, 0, stream>>>(d_keys_sorted, sl.home_cell.p, nk);
      roots<<<blocks(slab_roots_count), team, 0, stream>>>(
          d_keys_sorted, nk, sl.geometry.p, slab_roots_count);
      tree_offsets_begin<<<1, 1, 0, stream>>>(sl.levels.p, sl.node_count.p, slab_roots_count);

      const int n_grid = blocks(int(node_cap));
      for (int depth = 0; depth < tree_levels; ++depth) {
        split<<<n_grid, team, 0, stream>>>(
            d_keys_sorted, sl.geometry.p, 0, 0, depth, gh.leaf_limit,
            sl.node_count.p, int(node_cap), d_tree_invalid.p, sl.levels.p);
        tree_offsets_end<<<1, 1, 0, stream>>>(sl.levels.p, sl.node_count.p, depth, int(node_cap));
        tree_links<<<n_grid, team, 0, stream>>>(
            sl.geometry.p, sl.escape.p, sl.levels.p, depth, sl.first_child.p);
      }
      check_tree_limit<<<n_grid, team, 0, stream>>>(
          sl.geometry.p, sl.levels.p, tree_levels, gh.leaf_limit, d_tree_invalid.p);
      tree_links<<<n_grid, team, 0, stream>>>(
          sl.geometry.p, sl.escape.p, sl.levels.p, tree_levels, sl.first_child.p);

      for (int depth = tree_levels; depth >= 0; --depth) {
        multipoles_fast_slab<<<n_grid, team, 0, stream>>>(
            sl.geometry.p, sl.force_nodes.p, d_node_com, d_x.p,
            mass_uniform ? nullptr : d_mass.p, first_mass, soft,
            sl.order.p, depth, k, gh, sl.levels.p);
      }
      gather_sources_slab<<<p_grid, team, 0, stream>>>(
          d_x.p, mass_uniform ? nullptr : d_mass.p, first_mass,
          sl.order.p, sl.packed_sources.p, nk);
      unsigned int nc = 0;
      int inv = 0;
      check(cudaMemcpyAsync(&nc, sl.node_count.p, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
      check(cudaMemcpyAsync(&inv, d_tree_invalid.p, sizeof(int), cudaMemcpyDeviceToHost, stream));
      check(cudaStreamSynchronize(stream));
      if (inv != 0 || nc >= node_cap) {
        throw std::runtime_error("Slab tree build failed: slab=" + std::to_string(k) +
                                 " nodes=" + std::to_string(nc) +
                                 " node_cap=" + std::to_string(node_cap) +
                                 " invalid=" + std::to_string(inv));
      }
      sl.nodes = nc;
      sl.slab_id = k;
      ++builds;
      tree_ms_acc += (wall() - t0) * 1000.0;
    };

    const double evo_start = wall();
    for (int iteration = 0;; ++iteration) {
      if (iteration > max_steps_guard)
        throw std::runtime_error("Exceeded max_steps_guard");

      const double step_start = wall();
      const bool is_pm_step = (ti == 0 || ti == next_pm_boundary);
      const bool is_final = (ti == timeline);
      const double drift_fac = (ti > 0) ? p.factors(previous, ti, true) : 0.0;
      const double a = p.scale(ti);
      const double H_cur = p.H(a);

      // Pass 1: Parallel host drift + slab migration + active scan + vel^2 sum
      const double t_p1_0 = wall();
      size_t total_migrants = 0;
      if (ti > 0) {
        std::array<std::vector<Migrant>, num_slabs> global_mig;
        std::array<std::array<std::vector<Migrant>, num_slabs>, num_slabs> local_mig;
#pragma omp parallel for schedule(dynamic, 1)
        for (int s = 0; s < num_slabs; ++s) {
          auto& S = slabs[s];
          auto& out_m = local_mig[s];
          size_t k = 0;
          while (k < S.x.size()) {
            const float3 vi = S.v[k];
            double3 px = S.x[k];
            px.x = wrapped_coord(px.x + drift_fac * double(vi.x), h.box);
            px.y = wrapped_coord(px.y + drift_fac * double(vi.y), h.box);
            px.z = wrapped_coord(px.z + drift_fac * double(vi.z), h.box);
            S.x[k] = px;
            const int dest = std::min(std::max(int(px.x * inv_wx), 0), num_slabs - 1);
            if (dest != s) {
              out_m[dest].push_back({px, vi, S.mesh_acc[k], S.old_acc[k], S.bins[k],
                                     S.orig_idx[k], mass_uniform ? first_mass : S.mass[k]});
              const size_t back = S.x.size() - 1;
              if (k != back) {
                S.x[k] = S.x[back];
                S.v[k] = S.v[back];
                S.mesh_acc[k] = S.mesh_acc[back];
                S.old_acc[k] = S.old_acc[back];
                S.bins[k] = S.bins[back];
                S.orig_idx[k] = S.orig_idx[back];
                if (!mass_uniform) S.mass[k] = S.mass[back];
              }
              S.x.pop_back(); S.v.pop_back(); S.mesh_acc.pop_back();
              S.old_acc.pop_back(); S.bins.pop_back(); S.orig_idx.pop_back();
              if (!mass_uniform) S.mass.pop_back();
              continue;
            }
            ++k;
          }
        }
        std::array<size_t, num_slabs> dst_mig_cnt{};
        for (int src = 0; src < num_slabs; ++src)
          for (int dst = 0; dst < num_slabs; ++dst)
            dst_mig_cnt[dst] += local_mig[src][dst].size();
        for (int dst = 0; dst < num_slabs; ++dst) {
          total_migrants += dst_mig_cnt[dst];
          global_mig[dst].reserve(dst_mig_cnt[dst]);
        }
        for (int src = 0; src < num_slabs; ++src) {
          for (int dst = 0; dst < num_slabs; ++dst) {
            if (!local_mig[src][dst].empty()) {
              global_mig[dst].insert(global_mig[dst].end(),
                                     local_mig[src][dst].begin(), local_mig[src][dst].end());
            }
          }
        }
#pragma omp parallel for schedule(dynamic, 1)
        for (int dst = 0; dst < num_slabs; ++dst) {
          if (global_mig[dst].empty()) continue;
          auto& S = slabs[dst];
          for (const auto& m : global_mig[dst]) {
            S.x.push_back(m.x);
            S.v.push_back(m.v);
            S.mesh_acc.push_back(m.mesh_acc);
            S.old_acc.push_back(m.old_acc);
            S.bins.push_back(m.bin);
            S.orig_idx.push_back(m.orig_idx);
            if (!mass_uniform) S.mass.push_back(m.mass);
          }
        }
      }

      std::array<size_t, num_slabs> slab_active{};
      std::array<Tick, num_slabs> slab_inactive_min;
      slab_inactive_min.fill(timeline);
      double vel_norm_sum = 0.0;
      const bool need_vel_sum = (is_pm_step && !is_final && dynamic_pm);
#pragma omp parallel for schedule(dynamic, 1) reduction(+:vel_norm_sum)
      for (int s = 0; s < num_slabs; ++s) {
        const auto& S = slabs[s];
        const size_t ns = S.x.size();
        size_t act = 0;
        Tick min_nx = timeline;
        double v2 = 0.0;
        for (size_t k = 0; k < ns; ++k) {
          const unsigned char b = S.bins[k];
          if (ti == 0 || is_final || is_active(ti, b)) {
            ++act;
          } else {
            const Tick w = bin_size(b);
            const Tick nx = (ti & ~(w - 1ULL)) + w;
            if (nx < min_nx) min_nx = nx;
          }
          if (need_vel_sum) {
            const float3 vi = S.v[k];
            v2 += double(vi.x) * vi.x + double(vi.y) * vi.y + double(vi.z) * vi.z;
          }
        }
        slab_active[s] = act;
        slab_inactive_min[s] = min_nx;
        vel_norm_sum += v2;
      }
      const double drift_ms = (wall() - t_p1_0) * 1000.0;

      // Ensure device slab capacity accommodates max slab after migration
      size_t cur_max_slab = 0;
      for (int s = 0; s < num_slabs; ++s) cur_max_slab = std::max(cur_max_slab, slabs[s].x.size());
      peak_slab_seen = std::max(peak_slab_seen, cur_max_slab);
      ensure_slab_cap(cur_max_slab);
      if (is_final) {
        h_act_unsorted.clear(); h_act_unsorted.shrink_to_fit();
        h_act_x.clear(); h_act_x.shrink_to_fit();
        h_act_v.clear(); h_act_v.shrink_to_fit();
        h_act_old.clear(); h_act_old.shrink_to_fit();
        h_act_bins.clear(); h_act_bins.shrink_to_fit();
        h_act_mesh.clear(); h_act_mesh.shrink_to_fit();
        if (!skip_output_write) {
          out_x.resize(N);
          out_v.resize(N);
          out_pot.resize(N);
        }
        if (force_comp_path) {
          out_short_f.resize(N);
          out_long_f.resize(N);
        }
        if (!skip_output_write || force_comp_path) {
          if (h_pot_buf.size() < cur_max_slab) h_pot_buf.resize(cur_max_slab);
          if (h_raw_buf.size() < cur_max_slab) h_raw_buf.resize(cur_max_slab);
        } else {
          if (sanity_pm_pot.size() < cur_max_slab) sanity_pm_pot.resize(cur_max_slab);
          if (h_raw_buf.size() < cur_max_slab) h_raw_buf.resize(cur_max_slab);
        }
      }

      // PM cadence update & Streaming PM deposit + FFT solve
      double pm_ms = 0.0;
      if (is_pm_step) {
        ++pm_updates;
        if (is_final) {
          pm_kick = p.factors(timeline - current_pm_span / 2, timeline, false);
        } else {
          if (dynamic_pm) {
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

        const double t_pm0 = wall();
        grav_pm_cuda_begin(128, stream);
        for (int s = 0; s < num_slabs; ++s) {
          const auto& S = slabs[s];
          const size_t ns = S.x.size();
          if (!ns) continue;
          stager.upload(d_x.p, S.x.data(), ns * sizeof(double3), stream);
          if (!mass_uniform)
            stager.upload(d_mass.p, S.mass.data(), ns * sizeof(float), stream);
          grav_pm_cuda_deposit_double3(
              d_x.p, mass_uniform ? nullptr : d_mass.p, first_mass, ns, 128, h.box, stream);
        }
        grav_pm_cuda_solve(128, h.box, r_s, stream);
        check(cudaStreamSynchronize(stream));
        pm_ms = (wall() - t_pm0) * 1000.0;
      }

      // Pass 2: 12-slab sliding window tree + force + integrate
      for (auto& sl : slots) sl.slab_id = -1;
      build_kick_table<<<1, 64, 0, stream>>>(d_kick_finish.p, d_kick_begin.p, p, ti);

      double tree_ms = 0.0, force_ms = 0.0, kick_ms = 0.0, transfer_ms = 0.0;
      int slab_builds = 0;
      Tick step_next = timeline;
      uint64_t step_active = 0, step_nodes = 0;
      unsigned int step_invalid = 0;

      for (int s = 0; s < num_slabs; ++s) {
        if (slab_inactive_min[s] < step_next) step_next = slab_inactive_min[s];
        const auto& S = slabs[s];
        const int ns = int(S.x.size());
        if (!is_pm_step && slab_active[s] == 0) continue;
        if (ns == 0) continue;

        ensure_slab_tree((s + num_slabs - 1) % num_slabs, tree_ms, slab_builds);
        ensure_slab_tree(s, tree_ms, slab_builds);
        ensure_slab_tree((s + 1) % num_slabs, tree_ms, slab_builds);

        const auto& slot_m = slots[(s + num_slabs - 1) % num_slabs % 3];
        const auto& slot_c = slots[s % 3];
        const auto& slot_p = slots[(s + 1) % num_slabs % 3];
        step_nodes += slot_c.nodes;

        const bool use_compact = (!is_pm_step && !is_final && slab_active[s] * 2 < size_t(ns));
        int force_count = ns;
        const int* active_ptr = nullptr;

        if (use_compact) {
          const double t_up0 = wall();
          stager.upload(d_bins, S.bins.data(), size_t(ns) * sizeof(unsigned char), stream);
          mark_active_slab<<<blocks(ns), team, 0, stream>>>(
              slot_c.order.p, d_bins, d_flags, ns, ti);
          cub::CountingInputIterator<int> seq(0);
          check(cub::DeviceSelect::Flagged(
              d_select_storage.p, select_bytes, seq, d_flags,
              d_active_ids, d_active_count.p, ns, stream));
          check(cudaMemcpyAsync(&force_count, d_active_count.p, sizeof(int), cudaMemcpyDeviceToHost, stream));
          check(cudaStreamSynchronize(stream));
          if (size_t(force_count) != slab_active[s]) {
            throw std::runtime_error("Compact active count mismatch on slab " + std::to_string(s) +
                                     ": gpu=" + std::to_string(force_count) +
                                     " cpu=" + std::to_string(slab_active[s]));
          }
          active_ptr = d_active_ids;
          if (force_count > 0) {
            int* d_act_unsorted = reinterpret_cast<int*>(d_force);
            map_active_unsorted<<<blocks(force_count), team, 0, stream>>>(
                slot_c.order.p, d_active_ids, d_act_unsorted, force_count);
            if (h_act_unsorted.size() < size_t(force_count)) {
              const size_t cap_act = size_t(force_count);
              h_act_unsorted.resize(cap_act);
              h_act_x.resize(cap_act);
              h_act_v.resize(cap_act);
              h_act_old.resize(cap_act);
              h_act_bins.resize(cap_act);
              h_act_mesh.resize(cap_act);
            }
            stager.download(h_act_unsorted.data(), d_act_unsorted, size_t(force_count) * sizeof(int), stream);
            const auto* sx = S.x.data();
            const auto* sv = S.v.data();
            const auto* so = S.old_acc.data();
            const auto* sb = S.bins.data();
            const auto* sm = S.mesh_acc.data();
            const int* map_u = h_act_unsorted.data();
            auto* ax = h_act_x.data();
            auto* av = h_act_v.data();
            auto* ao = h_act_old.data();
            auto* ab = h_act_bins.data();
            auto* am = h_act_mesh.data();
#pragma omp parallel for schedule(static) if(force_count >= 16384)
            for (int idx = 0; idx < force_count; ++idx) {
              const int k = map_u[idx];
              ax[idx] = sx[k];
              av[idx] = sv[k];
              ao[idx] = so[k];
              ab[idx] = sb[k];
              am[idx] = sm[k];
            }
            stager.upload(d_x.p, ax, size_t(force_count) * sizeof(double3), stream);
            stager.upload(d_v, av, size_t(force_count) * sizeof(float3), stream);
            stager.upload(d_old_acc, ao, size_t(force_count) * sizeof(float), stream);
            stager.upload(d_bins, ab, size_t(force_count) * sizeof(unsigned char), stream);
            stager.upload(d_mesh_acc, am, size_t(force_count) * sizeof(float3), stream);
            check(cudaStreamSynchronize(stream));
          }
          transfer_ms += (wall() - t_up0) * 1000.0;
        } else {
          const double t_up0 = wall();
          stager.upload(d_x.p, S.x.data(), size_t(ns) * sizeof(double3), stream);
          stager.upload(d_v, S.v.data(), size_t(ns) * sizeof(float3), stream);
          stager.upload(d_old_acc, S.old_acc.data(), size_t(ns) * sizeof(float), stream);
          stager.upload(d_bins, S.bins.data(), size_t(ns) * sizeof(unsigned char), stream);
          if (!is_pm_step)
            stager.upload(d_mesh_acc, S.mesh_acc.data(), size_t(ns) * sizeof(float3), stream);
          check(cudaStreamSynchronize(stream));
          transfer_ms += (wall() - t_up0) * 1000.0;

          if (is_pm_step) {
            const double t_interp = wall();
            float* pot_ptr = (is_final && (d_pm_pot != nullptr)) ? d_pm_pot : nullptr;
            grav_pm_cuda_interpolate_double3(
                d_x.p, size_t(ns), 128, h.box, p.G, d_mesh_acc, pot_ptr, stream);
            check(cudaStreamSynchronize(stream));
            pm_ms += (wall() - t_interp) * 1000.0;
            const double t_pm_dl = wall();
            stager.download(slabs[s].mesh_acc.data(), d_mesh_acc, size_t(ns) * sizeof(float3), stream);
            transfer_ms += (wall() - t_pm_dl) * 1000.0;
          }

          if (!is_pm_step) {
            mark_active_slab<<<blocks(ns), team, 0, stream>>>(
                slot_c.order.p, d_bins, d_flags, ns, ti);
            cub::CountingInputIterator<int> seq(0);
            check(cub::DeviceSelect::Flagged(
                d_select_storage.p, select_bytes, seq, d_flags,
                d_active_ids, d_active_count.p, ns, stream));
            check(cudaMemcpyAsync(&force_count, d_active_count.p, sizeof(int), cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
            active_ptr = d_active_ids;
          }
        }

        const double t_f0 = wall();
        if (force_count > 0) {
          const auto launch_w = [&](auto m2l_t, auto fsp_t) {
            tree_force_warp_slab<decltype(m2l_t)::value, decltype(fsp_t)::value>
                <<<blocks(force_count), team, 0, stream>>>(
                    d_x.p, d_old_acc, soft,
                    slot_m.view(), slot_c.view(), slot_p.view(),
                    d_force, ns, p, ti == 0, d_tree_invalid.p,
                    active_ptr, active_ptr ? d_active_count.p : nullptr,
                    use_compact);
          };
          if (warp_m2l) {
            if (fast_split) launch_w(std::true_type{}, std::true_type{});
            else launch_w(std::true_type{}, std::false_type{});
          } else {
            if (fast_split) launch_w(std::false_type{}, std::true_type{});
            else launch_w(std::false_type{}, std::false_type{});
          }
          check(cudaStreamSynchronize(stream));
        }
        force_ms += (wall() - t_f0) * 1000.0;

        const double t_k0 = wall();
        init_meta<<<1, 1, 0, stream>>>(d_stats.p, d_tree_invalid.p, slot_c.nodes);
        if (force_count > 0) {
          integrate_slab<<<blocks(force_count), team, 0, stream>>>(
              d_v, d_old_acc, d_bins, d_force, d_mesh_acc,
              use_compact ? nullptr : active_ptr,
              use_compact ? nullptr : slot_c.order.p,
              force_count, soft, p, ti, d_stats.p,
              d_kick_finish.p, d_kick_begin.p, is_pm_step ? pm_kick : 0.0,
              a, H_cur, dt_max_rms, short_span_cap, d_tree_invalid.p);
        }
        Meta slab_stats{};
        check(cudaMemcpyAsync(&slab_stats, d_stats.p, sizeof(Meta), cudaMemcpyDeviceToHost, stream));
        check(cudaStreamSynchronize(stream));
        kick_ms += (wall() - t_k0) * 1000.0;

        if (slab_stats.next < step_next) step_next = slab_stats.next;
        step_active += slab_stats.active;
        step_invalid |= slab_stats.invalid;

        const double t_dl0 = wall();
        if (use_compact) {
          if (force_count > 0) {
            stager.download(h_act_v.data(), d_v, size_t(force_count) * sizeof(float3), stream);
            stager.download(h_act_old.data(), d_old_acc, size_t(force_count) * sizeof(float), stream);
            stager.download(h_act_bins.data(), d_bins, size_t(force_count) * sizeof(unsigned char), stream);
            auto* sv = slabs[s].v.data();
            auto* so = slabs[s].old_acc.data();
            auto* sb = slabs[s].bins.data();
            const int* map_u = h_act_unsorted.data();
            const auto* av = h_act_v.data();
            const auto* ao = h_act_old.data();
            const auto* ab = h_act_bins.data();
#pragma omp parallel for schedule(static) if(force_count >= 16384)
            for (int idx = 0; idx < force_count; ++idx) {
              const int k = map_u[idx];
              sv[k] = av[idx];
              so[k] = ao[idx];
              sb[k] = ab[idx];
            }
          }
        } else {
          stager.download(slabs[s].v.data(), d_v, size_t(ns) * sizeof(float3), stream);
          stager.download(slabs[s].old_acc.data(), d_old_acc, size_t(ns) * sizeof(float), stream);
          stager.download(slabs[s].bins.data(), d_bins, size_t(ns) * sizeof(unsigned char), stream);
          if (is_final) {
            stager.download(h_raw_buf.data(), d_force, size_t(ns) * sizeof(float4), stream);
            if (!skip_output_write || force_comp_path) {
              stager.download(h_pot_buf.data(), d_pm_pot, size_t(ns) * sizeof(float), stream);
            } else {
              stager.download(sanity_pm_pot.data(), d_pm_pot, size_t(ns) * sizeof(float), stream);
            }
          }
        }
        transfer_ms += (wall() - t_dl0) * 1000.0;

        if (is_final) {
          const float inv_a = float(1.0 / a_end);
          const float* pm_p = (!skip_output_write || force_comp_path) ? h_pot_buf.data() : sanity_pm_pot.data();
          for (int k = 0; k < ns; ++k) {
            const float4 rf = h_raw_buf[k];
            const float pp = pm_p[k];
            const float pot_val = float(double(rf.w) + potential_norm) * p.G + pp;
            const double3 xk = slabs[s].x[k];
            const float3 vk = slabs[s].v[k];
            if (!std::isfinite(xk.x) || !std::isfinite(xk.y) || !std::isfinite(xk.z) ||
                xk.x < 0.0 || xk.x >= h.box || xk.y < 0.0 || xk.y >= h.box || xk.z < 0.0 || xk.z >= h.box ||
                !std::isfinite(vk.x) || !std::isfinite(vk.y) || !std::isfinite(vk.z) || !std::isfinite(pot_val)) {
              ++sanity_bad;
            } else {
              const float3 vf{vk.x * inv_a, vk.y * inv_a, vk.z * inv_a};
              sanity_v2_sum += double(vf.x) * vf.x + double(vf.y) * vf.y + double(vf.z) * vf.z;
              sanity_pot_sum += double(pot_val);
              if (pot_val < sanity_pot_min) sanity_pot_min = pot_val;
              if (pot_val > sanity_pot_max) sanity_pot_max = pot_val;
            }
            if (!skip_output_write) {
              const uint32_t gid = slabs[s].orig_idx[k];
              out_x[gid] = permute_out(xk, slab_axis);
              out_v[gid] = permute_out(float3{vk.x * inv_a, vk.y * inv_a, vk.z * inv_a}, slab_axis);
              out_pot[gid] = pot_val;
            }
            if (force_comp_path) {
              const uint32_t gid = slabs[s].orig_idx[k];
              const float3 sf3 = permute_out(float3{rf.x, rf.y, rf.z}, slab_axis);
              out_short_f[gid] = float4{sf3.x, sf3.y, sf3.z, rf.w};
              const float3 ma = permute_out(slabs[s].mesh_acc[k], slab_axis);
              out_long_f[gid] = float4{ma.x, ma.y, ma.z, pp};
            }
          }
          // Release non-coordinate per-slab host arrays immediately after final slab scatter
          slabs[s].v.clear(); slabs[s].v.shrink_to_fit();
          slabs[s].mesh_acc.clear(); slabs[s].mesh_acc.shrink_to_fit();
          slabs[s].old_acc.clear(); slabs[s].old_acc.shrink_to_fit();
          slabs[s].bins.clear(); slabs[s].bins.shrink_to_fit();
          slabs[s].orig_idx.clear(); slabs[s].orig_idx.shrink_to_fit();
          if (!mass_uniform) { slabs[s].mass.clear(); slabs[s].mass.shrink_to_fit(); }
        }
      }

      if (step_invalid || !step_active || (!is_final && (step_next <= ti || step_next > timeline)))
        throw std::runtime_error("Invalid slab state at step " + std::to_string(iteration) +
                                 " invalid=" + std::to_string(step_invalid) +
                                 " active=" + std::to_string(step_active));

      const double step_wall = wall() - step_start;
      total_wall_evo += step_wall;
      std::cout << "STEP {\"step\":" << iteration << ",\"a\":" << a
                << ",\"integer_time\":" << ti << ",\"active\":" << step_active
                << ",\"nodes\":" << step_nodes << ",\"wall_s\":" << step_wall
                << ",\"drift_ms\":" << drift_ms << ",\"key_sort_ms\":0"
                << ",\"tree_multipoles_ms\":" << tree_ms << ",\"force_ms\":" << force_ms
                << ",\"kick_bins_ms\":" << kick_ms << ",\"h2d_d2h_ms\":" << transfer_ms
                << ",\"slab_builds\":" << slab_builds << ",\"migrants\":" << total_migrants
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
      if (!fp) throw std::runtime_error("Could not open GRAV_FORCE_COMPONENTS output");
      write_bin(fp, out_short_f.data(), out_short_f.size());
      write_bin(fp, out_long_f.data(), out_long_f.size());
      if (fclose(fp)) throw std::runtime_error("Could not close GRAV_FORCE_COMPONENTS");
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
              << ",\"device_buffer_bytes\":" << device_bytes
              << ",\"memory_mode\":\"slab_device\""
              << ",\"final_particle_D2H_bytes\":" << output_bytes
              << ",\"export_s\":" << export_s << "}" << std::endl;

    cudaStreamDestroy(stream);
    return sanity_bad ? 6 : 0;
  } catch (const std::exception& e) {
    std::cerr << "Slab resident evolution failed: " << e.what() << std::endl;
    return 5;
  }
}
