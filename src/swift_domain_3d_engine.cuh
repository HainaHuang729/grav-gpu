#pragma once

#include "swift_cuda_fof.h"
#include "swift_domain_3d.h"

namespace swift3d {

template <class T>
inline size_t dev_len(const DevPtr<T>& d) {
  return d.bytes / sizeof(T);
}

__global__ void extract_home_slots_3d_kernel(const Key* keys, uint16_t* home_slot, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) home_slot[i] = uint16_t(keys[i] >> octree_shift);
}

__global__ void build_home_warp_slices_3d_kernel(
    const uint16_t* __restrict__ home_slot,
    int n_own,
    int n_interior_own,
    const int* __restrict__ active_sorted,
    const int* __restrict__ active_count,
    int n,
    const int* __restrict__ tree_invalid,
    SplitWarpSlice* __restrict__ out_slices,
    int* __restrict__ out_warp_counts) {
  const int tid = threadIdx.x;
  const int targets = active_count ? *active_count : n;
  if (*tree_invalid || targets <= 0 || n_own <= 0) {
    if (tid == 0) {
      out_warp_counts[0] = 0;
      out_warp_counts[1] = 0;
    }
    return;
  }
  __shared__ int s_begin[N_TOP_CELLS + 1];
  __shared__ int s_warp_off[N_TOP_CELLS + 1];

  for (int c = tid; c <= n_own; c += blockDim.x) {
    if (c == 0) {
      s_begin[0] = 0;
    } else if (c == n_own) {
      s_begin[n_own] = targets;
    } else {
      int lo = 0, hi = targets;
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        const int sid = active_sorted ? active_sorted[mid] : mid;
        if (int(home_slot[sid]) < c) lo = mid + 1;
        else hi = mid;
      }
      s_begin[c] = lo;
    }
  }
  __syncthreads();

  if (tid == 0) {
    int acc = 0;
    for (int c = 0; c < n_own; ++c) {
      s_warp_off[c] = acc;
      const int b = s_begin[c];
      const int e = s_begin[c + 1];
      if (b < e) {
        const int next32 = (b + 31) & ~31;
        if (b < next32) ++acc;
        if (next32 < e) acc += (e - next32 + 31) >> 5;
      }
    }
    s_warp_off[n_own] = acc;
    const int n_int_clamped = (n_interior_own < 0) ? 0 : ((n_interior_own > n_own) ? n_own : n_interior_own);
    out_warp_counts[0] = s_warp_off[n_int_clamped];
    out_warp_counts[1] = acc;
  }
  __syncthreads();

  for (int c = tid; c < n_own; c += blockDim.x) {
    const int b = s_begin[c];
    const int e = s_begin[c + 1];
    if (b < e) {
      int w = s_warp_off[c];
      const int next32 = (b + 31) & ~31;
      if (b < next32) {
        const int cnt = min(e, next32) - b;
        out_slices[w++] = {b, uint16_t(cnt), uint16_t(c)};
      }
      for (int pos = next32; pos < e; pos += 32, ++w) {
        const int cnt = min(32, e - pos);
        const uint16_t m2l_bit = (cnt == 32 || pos + cnt == targets) ? uint16_t(0x8000u) : uint16_t(0u);
        out_slices[w] = {pos, uint16_t(cnt), uint16_t(uint16_t(c) | m2l_bit)};
      }
    }
  }
}

template <bool use_m2l>
__global__ void __launch_bounds__(team, 5) traverse_m2p_split_3d_kernel(
    const double3* __restrict__ target_x,
    const float* __restrict__ target_old_acc,
    float uniform_soft,
    const float* __restrict__ target_soft_arr,
    const SplitTravTreeView loc_tv,
    const SplitTravTreeView gh_tv,
    const uint32_t* __restrict__ slot_nb_roots,
    const int* __restrict__ target_order,
    const SplitWarpSlice* __restrict__ slices,
    int warp_offset,
    int n_batch_warps,
    SplitWarpMeta* __restrict__ metas,
    int* __restrict__ head_p2p,
    int* __restrict__ head_m2l,
    SplitP2PChunk* __restrict__ p2p_chunks,
    int p2p_cap,
    SplitM2LChunk* __restrict__ m2l_chunks,
    int m2l_cap,
    int* __restrict__ p2p_warps,
    int* __restrict__ m2l_warps,
    int* __restrict__ split_counts,
    float4* __restrict__ force,
    int n,
    Physics p,
    bool initial,
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
  const int home_slot = int(sl.home_cell & 0x7FFFu);
  const bool slice_allow_m2l = (sl.home_cell & 0x8000u) != 0;
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
  const uint32_t* __restrict__ nb_ptr = slot_nb_roots + home_slot * 27;
  for (int nb_k = 0; nb_k < 27; ++nb_k) {
    const uint32_t entry = nb_ptr[nb_k];
    const uint32_t tree_id = entry >> 31;
    const int root_u = int(entry & 0x7FFFFFFFu);
    const Geometry* __restrict__ st_geom = tree_id ? gh_tv.geometry : loc_tv.geometry;
    const NodeMeta* __restrict__ st_meta = tree_id ? gh_tv.node_meta : loc_tv.node_meta;
    const CompactMultipole* __restrict__ st_cm = tree_id ? gh_tv.compact_m : loc_tv.compact_m;

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
                  s_m2l[warp_id][m2l_cnt] = (tree_id << 31) | uint32_t(u);
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
            s_p2p_meta[warp_id][p2p_cnt] = (tree_id << 31) | uint32_t(len);
          }
          ++p2p_cnt;
          if (p2p_cnt == split_p2p_chunk_sz) flush_p2p();
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

__global__ void __launch_bounds__(team, 4) eval_m2l_split_3d_kernel(
    const double3* __restrict__ target_x,
    const SplitTravTreeView loc_tv,
    const SplitTravTreeView gh_tv,
    const int* __restrict__ target_order,
    const SplitWarpMeta* __restrict__ metas,
    const int* __restrict__ head_m2l,
    const SplitM2LChunk* __restrict__ m2l_chunks,
    const int* __restrict__ m2l_warps,
    int n_m2l_warps,
    float4* __restrict__ force,
    int n,
    Physics p,
    const int* __restrict__ tree_invalid,
    const int* __restrict__ active_sorted,
    bool compact_targets) {
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
      const uint32_t tree_id = packed >> 31;
      const int u = int(packed & 0x7FFFFFFFu);
      const NodeMeta* __restrict__ st_meta = tree_id ? gh_tv.node_meta : loc_tv.node_meta;
      const CompactMultipole* __restrict__ st_cm = tree_id ? gh_tv.compact_m : loc_tv.compact_m;
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
__global__ void __launch_bounds__(team, uniform_soft ? 8 : 6) eval_p2p_split_3d_kernel(
    const double3* __restrict__ target_x,
    float uniform_soft_val,
    const float* __restrict__ target_soft_arr,
    const float4* __restrict__ loc_packed_sources,
    const float* __restrict__ loc_packed_soft,
    const int* __restrict__ loc_order,
    const float4* __restrict__ gh_packed_sources,
    const float* __restrict__ gh_packed_soft,
    const SplitWarpMeta* __restrict__ metas,
    const int* __restrict__ head_p2p,
    const SplitP2PChunk* __restrict__ p2p_chunks,
    const int* __restrict__ p2p_warps,
    int n_p2p_warps,
    float4* __restrict__ force,
    int n,
    Physics p,
    const int* __restrict__ tree_invalid,
    const int* __restrict__ active_sorted,
    bool compact_targets) {
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
    i = compact_targets ? tidx : loc_order[sorted_i];
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
      const int len = int(lmeta & 0x7FFFFFFFu);
      const uint32_t tree_id = lmeta >> 31;
      const float4* src_packed = tree_id ? gh_packed_sources : loc_packed_sources;
      const float* src_soft = tree_id ? gh_packed_soft : loc_packed_soft;
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
        const bool has_self = (tree_id == 0u) &&
            __any_sync(0xffffffff, lane_valid && sorted_i >= lstart + base && sorted_i < lstart + base + chunk);
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
              if (tree_id == 0u && sorted_i == lstart + base + k) continue;
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

__device__ __forceinline__ int cell_from_pos_12(double3 p, double inv_w) {
  const int tx = min(max(int(p.x * inv_w), 0), N_TOP - 1);
  const int ty = min(max(int(p.y * inv_w), 0), N_TOP - 1);
  const int tz = min(max(int(p.z * inv_w), 0), N_TOP - 1);
  return (tx * N_TOP + ty) * N_TOP + tz;
}

__global__ void drift_flag_migrants_3d_kernel(
    double3* d_x, const float3* d_v, const int* d_cell_owner, unsigned char* flags,
    int ns, int my_gpu, double drift_fac, double box, double inv_w) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= ns) return;
  double3 px = d_x[k];
  const float3 pv = d_v[k];
  px.x = wrapped_coord(px.x + drift_fac * double(pv.x), box);
  px.y = wrapped_coord(px.y + drift_fac * double(pv.y), box);
  px.z = wrapped_coord(px.z + drift_fac * double(pv.z), box);
  d_x[k] = px;
  const int cid = cell_from_pos_12(px, inv_w);
  flags[k] = (d_cell_owner[cid] != my_gpu) ? 1 : 0;
}

__global__ void flag_plan_migrants_3d_kernel(
    const double3* d_x, const int* d_new_owner, unsigned char* flags,
    int ns, int my_gpu, double inv_w) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= ns) return;
  const int cid = cell_from_pos_12(d_x[k], inv_w);
  flags[k] = (d_new_owner[cid] != my_gpu) ? 1 : 0;
}

__global__ void pack_migrants_3d_kernel(
    const double3* d_x, const float3* d_v, const float3* d_mesh_acc,
    const float* d_old_acc, const unsigned char* d_bins, const uint32_t* d_orig_idx,
    const float* d_mass, float first_mass, const float* d_soft, float first_soft,
    const int* mig_ids, int m_cnt, double inv_w, const int* d_cell_owner, DevMigrant* out) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= m_cnt) return;
  const int k = mig_ids[j];
  const double3 px = d_x[k];
  const int dest = d_cell_owner[cell_from_pos_12(px, inv_w)];
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

__global__ void mark_active_home_cells_3d_kernel(
    const uint16_t* __restrict__ home_slot,
    const int* __restrict__ order,
    const unsigned char* __restrict__ bins,
    const int* __restrict__ slot_to_cell,
    int ns,
    Tick ti,
    uint32_t* __restrict__ active_bits) {
  __shared__ uint32_t s_bits[54];
  for (int i = threadIdx.x; i < 54; i += blockDim.x) s_bits[i] = 0u;
  __syncthreads();
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < ns && is_active(ti, bins[order[k]])) {
    const int cid = slot_to_cell[int(home_slot[k])];
    atomicOr(&s_bits[cid >> 5], 1u << (cid & 31));
  }
  __syncthreads();
  for (int i = threadIdx.x; i < 54; i += blockDim.x) {
    if (s_bits[i]) atomicOr(&active_bits[i], s_bits[i]);
  }
}

__global__ void count_cells_3d_kernel(
    const double3* __restrict__ d_x, const unsigned char* __restrict__ d_bins,
    int ns, double inv_w, Tick ti, bool force_all,
    unsigned long long* __restrict__ cell_counts,
    unsigned long long* __restrict__ cell_act_counts) {
  __shared__ uint32_t s_cnt[N_TOP_CELLS];
  __shared__ uint32_t s_act[N_TOP_CELLS];
  for (int i = threadIdx.x; i < N_TOP_CELLS; i += blockDim.x) {
    s_cnt[i] = 0u;
    s_act[i] = 0u;
  }
  __syncthreads();
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < ns; k += blockDim.x * gridDim.x) {
    const int cid = cell_from_pos_12(d_x[k], inv_w);
    atomicAdd(&s_cnt[cid], 1u);
    if (force_all || is_active(ti, d_bins[k])) {
      atomicAdd(&s_act[cid], 1u);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < N_TOP_CELLS; i += blockDim.x) {
    if (s_cnt[i]) atomicAdd(&cell_counts[i], (unsigned long long)(s_cnt[i]));
    if (s_act[i]) atomicAdd(&cell_act_counts[i], (unsigned long long)(s_act[i]));
  }
}

__global__ void compute_node_vel_active_3d_kernel(
    const Geometry* __restrict__ nodes,
    const ForceNode* __restrict__ force_nodes,
    const float3* __restrict__ v_in,
    const float* __restrict__ mass_or_null,
    float uniform_mass,
    const int* __restrict__ sorted,
    int depth,
    const int* __restrict__ levels,
    const uint32_t* __restrict__ active_bits,
    const int* __restrict__ slot_to_cell,
    float4* __restrict__ node_vel_max) {
  const int i = levels[depth] + blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= levels[depth + 1]) return;
  if (!force_nodes[i].n) {
    node_vel_max[i] = {0.f, 0.f, 0.f, 0.f};
    return;
  }
  const Geometry geom = nodes[i];
  if (active_bits) {
    const int slot = int(geom.key >> octree_shift);
    const int cid = slot_to_cell[slot];
    if (((active_bits[cid >> 5] >> (cid & 31)) & 1u) == 0u) return;
  }
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

__global__ void count_own_slot_nodes_3d_kernel(
    const Geometry* __restrict__ geom,
    int n_own,
    int n_nodes,
    int* __restrict__ out_slot_desc_cnt,
    int* __restrict__ out_slot_pstart,
    int* __restrict__ out_slot_pcnt) {
  __shared__ uint32_t s_cnt[N_TOP_CELLS];
  for (int i = threadIdx.x; i < n_own; i += blockDim.x) s_cnt[i] = 0u;
  __syncthreads();

  for (int u = blockIdx.x * blockDim.x + threadIdx.x; u < n_nodes; u += blockDim.x * gridDim.x) {
    if (u < n_own) {
      const Geometry rg = geom[u];
      out_slot_pstart[u] = rg.start;
      out_slot_pcnt[u] = rg.end - rg.start;
    } else {
      const int s = int(geom[u].key >> octree_shift);
      if (s >= 0 && s < n_own) atomicAdd(&s_cnt[s], 1u);
    }
  }
  __syncthreads();

  for (int i = threadIdx.x; i < n_own; i += blockDim.x) {
    if (s_cnt[i]) atomicAdd(&out_slot_desc_cnt[i], int(s_cnt[i]));
  }
}

__global__ void assign_export_bidx_3d_kernel(
    const Geometry* __restrict__ src_geom,
    const int* __restrict__ slot_to_e,
    int n_own,
    int n_nodes,
    int E,
    int* __restrict__ exp_cursor,
    int* __restrict__ out_u_to_b,
    int* __restrict__ out_bnd_src_node_map) {
  const int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n_nodes) return;
  if (u < n_own) {
    const int e = slot_to_e[u];
    if (e >= 0) {
      out_u_to_b[u] = e;
      out_bnd_src_node_map[e] = u;
    }
    return;
  }
  const int s = int(src_geom[u].key >> octree_shift);
  if (s >= 0 && s < n_own) {
    const int e = slot_to_e[s];
    if (e >= 0) {
      const int pos = E + atomicAdd(&exp_cursor[e], 1);
      out_u_to_b[u] = pos;
      out_bnd_src_node_map[pos] = u;
    }
  }
}

__global__ void gather_export_bundle_3d_kernel(
    const Geometry* __restrict__ src_geom,
    const NodeMeta* __restrict__ src_meta,
    const CompactMultipole* __restrict__ src_cm,
    const int* __restrict__ slot_to_e,
    const int* __restrict__ exp_part_src_start,
    const int* __restrict__ exp_part_off,
    const int* __restrict__ u_to_b,
    const int* __restrict__ bnd_src_node_map,
    int E,
    int tot_nodes,
    Geometry* __restrict__ out_geom,
    NodeMeta* __restrict__ out_meta,
    CompactMultipole* __restrict__ out_cm) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= tot_nodes) return;
  const int u = bnd_src_node_map[b];
  Geometry g = src_geom[u];
  const int e = (b < E) ? b : slot_to_e[int(g.key >> octree_shift)];
  const int p_delta = exp_part_off[e] - exp_part_src_start[e];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int c = g.children[k];
    if (c >= 0) g.children[k] = u_to_b[c];
  }
  g.start += p_delta;
  g.end += p_delta;
  out_geom[b] = g;
  out_meta[b] = src_meta[u];
  out_cm[b] = src_cm[u];
}

__global__ void pack_export_particles_kernel(
    const float4* __restrict__ src_packed,
    const float* __restrict__ src_soft,
    const int* __restrict__ exp_part_src_start,
    const int* __restrict__ exp_part_off,
    const int* __restrict__ exp_part_cnt,
    int E,
    float4* __restrict__ out_packed,
    float* __restrict__ out_soft) {
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int e = gw >> 2;
  if (e >= E) return;
  const int sub = gw & 3;
  const int lane = threadIdx.x & 31;
  const int src_s = exp_part_src_start[e];
  const int dst_s = exp_part_off[e];
  const int cnt = exp_part_cnt[e];
  for (int k = (sub << 5) + lane; k < cnt; k += 128) {
    out_packed[dst_s + k] = src_packed[src_s + k];
    if (src_soft && out_soft) out_soft[dst_s + k] = src_soft[src_s + k];
  }
}

__global__ void install_ghost_bundle_kernel(
    const Geometry* __restrict__ in_geom,
    const NodeMeta* __restrict__ in_meta,
    const CompactMultipole* __restrict__ in_cm,
    const int* __restrict__ dst_ghost_slots,
    int E,
    int tot_nodes,
    int n_ghost,
    int dst_desc_base,
    int dst_part_base,
    Geometry* __restrict__ gh_geom,
    NodeMeta* __restrict__ gh_meta,
    CompactMultipole* __restrict__ gh_cm) {
  const int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= tot_nodes) return;
  const int dst_u = (b < E) ? dst_ghost_slots[b] : (n_ghost + dst_desc_base + (b - E));
  Geometry g = in_geom[b];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const int c = g.children[k];
    if (c >= 0) {
      g.children[k] = (c < E) ? dst_ghost_slots[c] : (n_ghost + dst_desc_base + (c - E));
    }
  }
  g.start += dst_part_base;
  g.end += dst_part_base;
  gh_geom[dst_u] = g;
  gh_meta[dst_u] = in_meta[b];
  gh_cm[dst_u] = in_cm[b];
}

struct Dom3DLazyRootDesc {
  int e;
  int src_part_start;
  int dst_gslot;
  int src_nonroot_off;
  int dst_desc_base;
  int n_cnt;
  int p_cnt;
  int lazy_node_off;
  int lazy_part_off;
  int dst_part_off;
};

__global__ void pack_lazy_edge_kernel(
    const NodeMeta* __restrict__ loc_meta,
    const int* __restrict__ bnd_src_node_map,
    const double3* __restrict__ loc_x,
    const float* __restrict__ loc_mass,
    float first_mass,
    const int* __restrict__ loc_order,
    const Dom3DLazyRootDesc* __restrict__ descs,
    int M,
    int E,
    int tot_lazy_nodes,
    float4* __restrict__ out_buf) {
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int m = gw >> 2;
  if (m >= M) return;
  const int sub = gw & 3;
  const int lane = threadIdx.x & 31;
  const Dom3DLazyRootDesc d = descs[m];
  for (int k = (sub << 5) + lane; k < d.n_cnt; k += 128) {
    const int b = (k == 0) ? d.e : (E + d.src_nonroot_off + (k - 1));
    const int u = bnd_src_node_map[b];
    out_buf[d.lazy_node_off + k] = loc_meta[u].center;
  }
  for (int k = (sub << 5) + lane; k < d.p_cnt; k += 128) {
    const int pid = loc_order[d.src_part_start + k];
    const double3 pos = loc_x[pid];
    const float mp = loc_mass ? loc_mass[pid] : first_mass;
    out_buf[tot_lazy_nodes + d.lazy_part_off + k] = {float(pos.x), float(pos.y), float(pos.z), mp};
  }
}

__global__ void unpack_lazy_edge_kernel(
    const float4* __restrict__ in_buf,
    const Dom3DLazyRootDesc* __restrict__ descs,
    int M,
    int n_ghost,
    int tot_lazy_nodes,
    NodeMeta* __restrict__ gh_meta,
    float4* __restrict__ gh_packed) {
  const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int m = gw >> 2;
  if (m >= M) return;
  const int sub = gw & 3;
  const int lane = threadIdx.x & 31;
  const Dom3DLazyRootDesc d = descs[m];
  for (int k = (sub << 5) + lane; k < d.n_cnt; k += 128) {
    const int dst_u = (k == 0) ? d.dst_gslot : (n_ghost + d.dst_desc_base + (k - 1));
    gh_meta[dst_u].center = in_buf[d.lazy_node_off + k];
  }
  for (int k = (sub << 5) + lane; k < d.p_cnt; k += 128) {
    gh_packed[d.dst_part_off + k] = in_buf[tot_lazy_nodes + d.lazy_part_off + k];
  }
}

inline void alloc_pinned_slack(
    PinnedHostBuf& buf, size_t need_bytes,
    unsigned int flags = cudaHostAllocPortable) {
  if (buf.bytes >= need_bytes) return;
  const size_t grown = std::max<size_t>(need_bytes + (need_bytes >> 1),
                                        need_bytes + (16ULL << 20));
  constexpr size_t align = 2ULL << 20;
  const size_t target = ((grown + align - 1ULL) / align) * align;
  buf.free();
  buf.bytes = target;
  check(cudaHostAlloc(&buf.p, buf.bytes, flags));
}

struct Dom3DEdgeState {
  static constexpr int MAX_CHUNKS = 8;
  static constexpr size_t CHUNK_BYTES = 8ULL << 20;

  int E = 0;
  int tot_nodes = 0;
  int tot_parts = 0;
  int n_chunks = 0;
  size_t chunk_sz = 0;

  cudaEvent_t ev_pack_done = nullptr;
  cudaEvent_t ev_d2h_chunk[MAX_CHUNKS]{};

  // Source-side metadata & bundle scratch (allocated on src GPU, persistent across rebalances)
  DevPtr<int> d_slot_to_e;
  DevPtr<int> d_exp_cursor, d_exp_part_src_start, d_exp_part_off, d_exp_part_cnt;
  DevPtr<int> d_bnd_src_node_map;
  DevPtr<char> d_send_wire, d_recv_wire;
  DevPtr<Dom3DLazyRootDesc> d_lazy_descs_src, d_lazy_descs_dst;
  DevPtr<float4> d_lazy_send, d_lazy_recv;

  // Destination-side slot map (allocated on dst GPU)
  DevPtr<int> d_dst_ghost_slots;

  PinnedHostBuf h_bounce;
  PinnedHostBuf h_meta_pinned;
  PinnedHostBuf h_lazy_descs_pinned;

  std::vector<int> h_node_cnt, h_part_cnt, h_part_src_start;
  std::vector<int> h_node_off, h_nonroot_off, h_part_off;
  std::vector<Dom3DLazyRootDesc> h_lazy_descs;

  size_t send_wire_bytes = 0;
  int lazy_M = 0;
  int lazy_tot_nodes = 0;
  int lazy_tot_parts = 0;
  int dst_desc_base = 0;
  int dst_part_base = 0;

  void init_events(int src_dev) {
    check(cudaSetDevice(src_dev));
    if (!ev_pack_done) check(cudaEventCreateWithFlags(&ev_pack_done, cudaEventDisableTiming));
    for (int c = 0; c < MAX_CHUNKS; ++c) {
      if (!ev_d2h_chunk[c]) check(cudaEventCreateWithFlags(&ev_d2h_chunk[c], cudaEventDisableTiming));
    }
  }

  void ensure_meta_src(int src_dev, int e_cnt, size_t& tracker) {
    const size_t need = size_t(e_cnt);
    const size_t need_p = need * 4ULL * sizeof(int);
    if (need > dev_len(d_exp_cursor)) {
      check(cudaSetDevice(src_dev));
      const size_t cap = std::max<size_t>(need * 2ULL, size_t(N_TOP_CELLS));
      d_exp_cursor.alloc(cap, tracker);
      d_exp_part_src_start.alloc(cap, tracker);
      d_exp_part_off.alloc(cap, tracker);
      d_exp_part_cnt.alloc(cap, tracker);
    }
    if (h_meta_pinned.bytes < need_p) {
      alloc_pinned_slack(h_meta_pinned, std::max<size_t>(need_p, size_t(N_TOP_CELLS) * 4ULL * sizeof(int)));
    }
  }

  void setup_plan(int src_dev, int dst_dev, int n_own_src,
                  const std::vector<int>& src_slots,
                  const std::vector<int>& dst_slots,
                  size_t& src_tracker, size_t& dst_tracker) {
    E = int(src_slots.size());
    tot_nodes = tot_parts = lazy_M = lazy_tot_nodes = lazy_tot_parts = dst_desc_base = dst_part_base = 0;
    send_wire_bytes = 0;
    n_chunks = 0;
    chunk_sz = 0;
    h_node_cnt.resize(E);
    h_part_cnt.resize(E);
    h_part_src_start.resize(E);
    h_node_off.resize(E);
    h_nonroot_off.resize(E);
    h_part_off.resize(E);

    check(cudaSetDevice(src_dev));
    if (dev_len(d_slot_to_e) < size_t(N_TOP_CELLS)) {
      d_slot_to_e.alloc(N_TOP_CELLS, src_tracker);
    }
    if (E == 0) return;

    std::array<int, N_TOP_CELLS> s2e;
    s2e.fill(-1);
    for (int e = 0; e < E; ++e) {
      if (src_slots[e] >= 0 && src_slots[e] < n_own_src) s2e[src_slots[e]] = e;
    }
    check(cudaMemcpy(d_slot_to_e.p, s2e.data(),
                     size_t(std::max(1, n_own_src)) * sizeof(int), cudaMemcpyHostToDevice));
    ensure_meta_src(src_dev, E, src_tracker);
    const size_t need_ld = size_t(E) * sizeof(Dom3DLazyRootDesc);
    if (h_lazy_descs_pinned.bytes < need_ld) {
      alloc_pinned_slack(h_lazy_descs_pinned, std::max<size_t>(need_ld, size_t(N_TOP_CELLS) * sizeof(Dom3DLazyRootDesc)));
    }
    if (size_t(E) > dev_len(d_lazy_descs_src)) {
      d_lazy_descs_src.alloc(std::max<size_t>(size_t(E), size_t(N_TOP_CELLS)), src_tracker);
    }

    check(cudaSetDevice(dst_dev));
    if (size_t(E) > dev_len(d_dst_ghost_slots)) {
      d_dst_ghost_slots.alloc(std::max<size_t>(size_t(E), size_t(N_TOP_CELLS)), dst_tracker);
    }
    check(cudaMemcpy(d_dst_ghost_slots.p, dst_slots.data(), size_t(E) * sizeof(int), cudaMemcpyHostToDevice));
    if (size_t(E) > dev_len(d_lazy_descs_dst)) {
      d_lazy_descs_dst.alloc(std::max<size_t>(size_t(E), size_t(N_TOP_CELLS)), dst_tracker);
    }
  }

  void ensure_bnd_nodes(int src_dev, int nn, size_t& tracker) {
    if (size_t(nn) <= dev_len(d_bnd_src_node_map)) return;
    check(cudaSetDevice(src_dev));
    const size_t cap = std::max(size_t(nn), size_t(nn) * 3 / 2 + 4096);
    d_bnd_src_node_map.alloc(cap, tracker);
  }

  void free_all(int src_dev, int dst_dev, size_t& src_tracker, size_t& dst_tracker) {
    check(cudaSetDevice(src_dev));
    d_slot_to_e.free(src_tracker);
    d_exp_cursor.free(src_tracker);
    d_exp_part_src_start.free(src_tracker);
    d_exp_part_off.free(src_tracker);
    d_exp_part_cnt.free(src_tracker);
    d_bnd_src_node_map.free(src_tracker);
    d_send_wire.free(src_tracker);
    d_lazy_descs_src.free(src_tracker);
    d_lazy_send.free(src_tracker);
    if (ev_pack_done) { cudaEventDestroy(ev_pack_done); ev_pack_done = nullptr; }
    for (int c = 0; c < MAX_CHUNKS; ++c) {
      if (ev_d2h_chunk[c]) { cudaEventDestroy(ev_d2h_chunk[c]); ev_d2h_chunk[c] = nullptr; }
    }

    check(cudaSetDevice(dst_dev));
    d_dst_ghost_slots.free(dst_tracker);
    d_recv_wire.free(dst_tracker);
    d_lazy_descs_dst.free(dst_tracker);
    d_lazy_recv.free(dst_tracker);
    h_bounce.free();
    h_meta_pinned.free();
    h_lazy_descs_pinned.free();
    E = tot_nodes = tot_parts = lazy_M = lazy_tot_nodes = lazy_tot_parts = dst_desc_base = dst_part_base = 0;
    send_wire_bytes = 0;
    n_chunks = 0;
    chunk_sz = 0;
  }
};

struct Dom3DGpuCtx {
  int dev = 0;
  size_t device_bytes = 0;
  cudaStream_t compute_stream = nullptr;
  cudaStream_t copy_stream = nullptr;
  cudaStream_t d2h_stream = nullptr;
  cudaEvent_t ev_force_start = nullptr, ev_force_end = nullptr;
  cudaEvent_t ev_prof0 = nullptr, ev_prof1 = nullptr;
  cudaEvent_t ev_ghost_start = nullptr, ev_ghost_ready = nullptr;
  cudaEvent_t ev_int_start = nullptr, ev_int_end = nullptr;
  cudaEvent_t ev_bnd_start = nullptr, ev_bnd_end = nullptr;

  // Resident particles
  size_t ns = 0, part_cap = 0;
  DevPtr<double3> d_x;
  DevPtr<float3> d_v, d_mesh_acc;
  DevPtr<float> d_old_acc, d_mass, d_soft;
  DevPtr<unsigned char> d_bins;
  DevPtr<uint32_t> d_orig_idx;

  // Domain plan mappings on GPU
  int n_own = 0, n_interior_own = 0, n_ghost = 0;
  DevPtr<int> d_cell_owner, d_slot_to_cell, d_cell_to_slot;
  DevPtr<uint32_t> d_slot_nb_roots;
  std::array<int, tree_levels + 2> h_levels{};

  // Reusable export staging per GPU
  DevPtr<int> d_own_slot_desc_cnt, d_own_slot_pstart, d_own_slot_pcnt;
  DevPtr<int> d_u_to_b;
  PinnedHostBuf h_slot_meta;

  // Local tree + split buffers
  size_t node_cap = 0;
  unsigned int nodes = 0;
  DevPtr<float4> packed_sources, node_vel_max, d_force;
  DevPtr<float> packed_soft, d_pm_pot;
  DevPtr<Geometry> geometry;
  DevPtr<ForceNode> force_nodes;
  DevPtr<NodeMeta> node_meta;
  DevPtr<CompactMultipole> compact_m;
  DevPtr<int> escape, first_child, order, node_count, levels;
  DevPtr<uint16_t> home_slot;

  // Build & scan scratch
  DevPtr<Key> d_key_tmp, d_keys_sorted;
  DevPtr<int> d_ord_tmp, d_active_ids, d_active_count, d_tree_invalid;
  DevPtr<int> d_tail_keep, d_tail_count;
  DevPtr<unsigned char> d_flags, d_tail_flags;
  DevPtr<double3> d_node_com;
  DevPtr<char> d_sort_storage, d_select_storage;
  DevPtr<uint32_t> d_active_bits;
  DevPtr<unsigned long long> d_cell_counts, d_cell_act_counts;
  DevPtr<DevMigrant> d_mig_out, d_mig_in;
  DevPtr<Meta> d_stats;
  DevPtr<double> d_kick_finish, d_kick_begin, d_pm_density;
  DevPtr<double4> d_pm_field;
  DevPtr<ScanPartial> d_scan_tmp;
  std::vector<ScanPartial> h_scan;
  PinnedHostBuf h_bounce;

  // Persistent resident ghost tree
  size_t gh_part_cap = 0, gh_node_cap = 0;
  unsigned int gh_nodes = 0;
  DevPtr<Geometry> gh_geometry;
  DevPtr<NodeMeta> gh_node_meta;
  DevPtr<CompactMultipole> gh_compact_m;
  DevPtr<float4> gh_packed_sources;
  DevPtr<float> gh_packed_soft;

  // Split traversal buffers
  DevPtr<SplitWarpSlice> d_split_slices;
  DevPtr<SplitWarpMeta> d_split_meta;
  DevPtr<int> d_split_head_p2p, d_split_head_m2l, d_split_p2p_warps, d_split_m2l_warps;
  DevPtr<int> d_split_counts, d_split_total_warps;
  DevPtr<SplitP2PChunk> d_split_p2p;
  DevPtr<SplitM2LChunk> d_split_m2l;
  size_t split_meta_cap = 0, split_p2p_cap = 0, split_m2l_cap = 0;
  int split_chunk_scale = 1;
  size_t sort_bytes = 0, select_bytes = 0;

  void alloc_particles(size_t new_cap, bool uniform_mass, bool uniform_soft) {
    part_cap = std::max<size_t>(new_cap, 4096);
    d_x.alloc(part_cap, device_bytes);
    d_v.alloc(part_cap, device_bytes);
    d_mesh_acc.alloc(part_cap, device_bytes);
    d_old_acc.alloc(part_cap, device_bytes);
    d_bins.alloc(part_cap, device_bytes);
    d_orig_idx.alloc(part_cap, device_bytes);
    if (!uniform_mass) d_mass.alloc(part_cap, device_bytes);
    if (!uniform_soft) d_soft.alloc(part_cap, device_bytes);

    packed_sources.alloc(part_cap, device_bytes);
    if (!uniform_soft) packed_soft.alloc(part_cap, device_bytes);
    order.alloc(part_cap, device_bytes);
    home_slot.alloc(part_cap, device_bytes);
    d_force.alloc(part_cap, device_bytes);
    d_pm_pot.alloc(part_cap, device_bytes);
    d_flags.alloc(part_cap, device_bytes);
    d_active_ids.alloc(part_cap, device_bytes);
    d_key_tmp.alloc(part_cap, device_bytes);
    d_keys_sorted.alloc(part_cap, device_bytes);
    d_ord_tmp.alloc(part_cap, device_bytes);

    size_t selb = 0;
    cub::CountingInputIterator<int> seq(0);
    check(cub::DeviceSelect::Flagged(
        nullptr, selb, seq, static_cast<unsigned char*>(nullptr),
        static_cast<int*>(nullptr), d_active_count.p, int(part_cap)));
    if (selb > select_bytes) {
      select_bytes = selb;
      d_select_storage.alloc(std::max<size_t>(select_bytes, 1), device_bytes);
    }
    size_t sb = 0;
    check(cub::DeviceRadixSort::SortPairs(
        nullptr, sb, static_cast<Key*>(nullptr), static_cast<Key*>(nullptr),
        static_cast<int*>(nullptr), static_cast<int*>(nullptr), int(part_cap), 0, 62));
    if (sb > sort_bytes) {
      sort_bytes = sb;
      d_sort_storage.alloc(std::max<size_t>(sort_bytes, 1), device_bytes);
    }
    const size_t req_sl = (part_cap + 31) / 32 + 2 * size_t(N_TOP_CELLS);
    if (req_sl > dev_len(d_split_slices)) d_split_slices.alloc(req_sl, device_bytes);
  }

  void alloc_local_tree_nodes(size_t ncap, bool need_nvel) {
    node_cap = std::max<size_t>(ncap, size_t(N_TOP_CELLS) * 2);
    geometry.alloc(node_cap, device_bytes);
    force_nodes.alloc(node_cap, device_bytes);
    node_meta.alloc(node_cap, device_bytes);
    compact_m.alloc(node_cap, device_bytes);
    if (need_nvel) node_vel_max.alloc(node_cap, device_bytes);
    escape.alloc(node_cap, device_bytes);
    first_child.alloc(node_cap, device_bytes);
    d_node_com.alloc(node_cap, device_bytes);
    d_u_to_b.alloc(node_cap, device_bytes);
    if (!d_node_count_allocated()) {
      node_count.alloc(1, device_bytes);
      levels.alloc(tree_levels + 2, device_bytes);
    }
  }
  bool d_node_count_allocated() const { return node_count.p != nullptr; }

  void ensure_ghost_cap(size_t need_parts, size_t need_nodes, bool uniform_soft) {
    const size_t np = std::max<size_t>(need_parts, 1);
    const size_t nn = std::max<size_t>(need_nodes, size_t(N_TOP_CELLS));
    if (np <= gh_part_cap && nn <= gh_node_cap && gh_geometry.p) return;
    check(cudaSetDevice(dev));
    if (np > gh_part_cap) {
      gh_part_cap = std::max(np, np * 5 / 4 + 4096);
      gh_packed_sources.alloc(gh_part_cap, device_bytes);
      if (!uniform_soft) gh_packed_soft.alloc(gh_part_cap, device_bytes);
    }
    if (nn > gh_node_cap) {
      gh_node_cap = std::max(nn, nn * 5 / 4 + 4096);
      gh_geometry.alloc(gh_node_cap, device_bytes);
      gh_node_meta.alloc(gh_node_cap, device_bytes);
      gh_compact_m.alloc(gh_node_cap, device_bytes);
    }
  }

  void ensure_split_bufs(size_t batch_warps, int scale = 1) {
    const size_t req_p2p = batch_warps * 32ULL * size_t(scale);
    const size_t req_m2l = batch_warps * 28ULL * size_t(scale);
    if (batch_warps <= split_meta_cap && req_p2p <= split_p2p_cap && req_m2l <= split_m2l_cap) return;
    check(cudaSetDevice(dev));
    check(cudaStreamSynchronize(compute_stream));
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

  void ensure_tail_cap(size_t m_cnt) {
    if (m_cnt <= dev_len(d_tail_flags)) return;
    d_tail_flags.alloc(m_cnt, device_bytes);
    d_tail_keep.alloc(m_cnt, device_bytes);
  }

  void ensure_mig_out(size_t n) { if (n > dev_len(d_mig_out)) d_mig_out.alloc(n, device_bytes); }
  void ensure_mig_in(size_t n) { if (n > dev_len(d_mig_in)) d_mig_in.alloc(n, device_bytes); }

  SplitTravTreeView local_split_view() const { return {geometry.p, node_meta.p, compact_m.p}; }
  SplitTravTreeView ghost_split_view() const { return {gh_geometry.p, gh_node_meta.p, gh_compact_m.p}; }

  void cleanup() {
    check(cudaSetDevice(dev));
    d_x.free(device_bytes); d_v.free(device_bytes); d_mesh_acc.free(device_bytes);
    d_old_acc.free(device_bytes); d_mass.free(device_bytes); d_soft.free(device_bytes);
    d_bins.free(device_bytes); d_orig_idx.free(device_bytes);
    d_cell_owner.free(device_bytes); d_slot_to_cell.free(device_bytes); d_cell_to_slot.free(device_bytes); d_slot_nb_roots.free(device_bytes);
    d_own_slot_desc_cnt.free(device_bytes); d_own_slot_pstart.free(device_bytes); d_own_slot_pcnt.free(device_bytes);
    d_u_to_b.free(device_bytes);
    h_slot_meta.free();
    packed_sources.free(device_bytes); node_vel_max.free(device_bytes); d_force.free(device_bytes);
    packed_soft.free(device_bytes); d_pm_pot.free(device_bytes);
    geometry.free(device_bytes); force_nodes.free(device_bytes);
    node_meta.free(device_bytes); compact_m.free(device_bytes);
    escape.free(device_bytes); first_child.free(device_bytes); order.free(device_bytes);
    node_count.free(device_bytes); levels.free(device_bytes); home_slot.free(device_bytes);
    d_key_tmp.free(device_bytes); d_keys_sorted.free(device_bytes); d_ord_tmp.free(device_bytes);
    d_active_ids.free(device_bytes); d_active_count.free(device_bytes); d_tree_invalid.free(device_bytes);
    d_tail_keep.free(device_bytes); d_tail_count.free(device_bytes);
    d_flags.free(device_bytes); d_tail_flags.free(device_bytes);
    d_node_com.free(device_bytes); d_sort_storage.free(device_bytes); d_select_storage.free(device_bytes);
    d_active_bits.free(device_bytes); d_cell_counts.free(device_bytes); d_cell_act_counts.free(device_bytes);
    d_mig_out.free(device_bytes); d_mig_in.free(device_bytes);
    d_stats.free(device_bytes); d_kick_finish.free(device_bytes); d_kick_begin.free(device_bytes);
    d_pm_density.free(device_bytes); d_pm_field.free(device_bytes); d_scan_tmp.free(device_bytes);
    gh_geometry.free(device_bytes); gh_node_meta.free(device_bytes); gh_compact_m.free(device_bytes);
    gh_packed_sources.free(device_bytes); gh_packed_soft.free(device_bytes);
    d_split_slices.free(device_bytes); d_split_meta.free(device_bytes);
    d_split_head_p2p.free(device_bytes); d_split_head_m2l.free(device_bytes);
    d_split_p2p_warps.free(device_bytes); d_split_m2l_warps.free(device_bytes);
    d_split_counts.free(device_bytes); d_split_total_warps.free(device_bytes);
    d_split_p2p.free(device_bytes); d_split_m2l.free(device_bytes);
    h_bounce.free();
    if (ev_force_start) cudaEventDestroy(ev_force_start);
    if (ev_force_end) cudaEventDestroy(ev_force_end);
    if (ev_prof0) cudaEventDestroy(ev_prof0);
    if (ev_prof1) cudaEventDestroy(ev_prof1);
    if (ev_ghost_start) cudaEventDestroy(ev_ghost_start);
    if (ev_ghost_ready) cudaEventDestroy(ev_ghost_ready);
    if (ev_int_start) cudaEventDestroy(ev_int_start);
    if (ev_int_end) cudaEventDestroy(ev_int_end);
    if (ev_bnd_start) cudaEventDestroy(ev_bnd_start);
    if (ev_bnd_end) cudaEventDestroy(ev_bnd_end);
    if (compute_stream) cudaStreamDestroy(compute_stream);
    if (copy_stream) cudaStreamDestroy(copy_stream);
    if (d2h_stream) cudaStreamDestroy(d2h_stream);
  }
};

struct Domain3DRunConfig {
  const char* input_path = nullptr;
  const char* output_path = nullptr;
  double started_wall = 0.0;
  double import_start = 0.0;
  StateInput h{};
  uint64_t N = 0;
  double a_end = 0.0;
  DomainMode domain_mode = DomainMode::Hilbert3D;
  int rebalance_interval = 8;
  double rebalance_min_gain = 0.02;
  std::array<uint64_t, N_TOP_CELLS> init_cell_counts{};
  double total_mass = 0.0;
  double first_mass_in = 0.0;
  double first_soft_in = 0.0;
  double min_mass = 0.0;
  bool mass_uniform = true;
  bool soft_uniform = true;
  bool has_soft_in = false;
  const char* soft_in_env = nullptr;

  int num_gpus = 1;
  std::vector<std::vector<int>> can_peer;
  std::string p2p_mode;

  bool periodic = true;
  bool comoving = true;
  bool dynamic_pm = false;
  int pm_intervals = 1;
  int short_intervals = 1;
  int pm_side = 128;
  float step_factor = 1.f;
  float mac = 0.f;
  float grav_G = 0.f;
  double dt_max_val = 0.0;
  double max_rms_fac = 0.0;
  double cosmo_h0 = 0.0, cosmo_om = 0.0, cosmo_ode = 0.0, cosmo_ocdm = 0.0;
  double cosmo_or = 0.0, cosmo_ok = 0.0, cosmo_w0 = -1.0, cosmo_wa = 0.0;
  float theta_val = 0.f, grav_eta_val = 0.f;
  int leaf_limit = 64;
  bool warp_m2l = true;
  bool fast_split = true;
  bool lazy_tree_enabled = true;
  bool split_force_enabled = true;
  bool pm_dist_enabled = true;
  bool split_profile = false;
  int split_batch_warps = 16384;
  int lazy_max_steps = 16;
  double lazy_disp_frac = 0.15;
  double lazy_active_frac = 0.50;
  bool skip_output_write = false;
  bool energy_log = false;

  const char* force0_out_path = nullptr;
  const char* force_comp_path = nullptr;
  const char* force3_out_path = nullptr;
  const char* midpoint_path = nullptr;
  std::vector<double> snap_scales;
  std::string out_prefix;
  bool run_fof = false;
  double fof_link_ratio = 0.2;
  int fof_min_size = 32;
  int requested_fof_slabs = 0;
  int force_check_n = 0;
  bool lc_enabled = false;
  bool lc_replicate = true;
  double3 lc_obs{};
  double lc_a_obs = 0.0;
  double lc_r_max_req = 0.0;
  const char* lc_out_env = nullptr;
};

inline int run_domain_3d_fof(const std::vector<Dom3DGpuCtx>& gpus,
                             const Domain3DRunConfig& cfg,
                             uint64_t N, double box, double fof_link,
                             const std::vector<double3>& out_x,
                             std::vector<int64_t>& fof_ids,
                             uint64_t& ngrp, uint64_t& ngrouped, uint64_t& largest) {
  double fof_el[9]{};
  fof_ids.resize(N);
  if (cfg.requested_fof_slabs > 1 && !out_x.empty()) {
    return whole_fof_slabbed_swift(
        reinterpret_cast<const double*>(out_x.data()), N,
        box, box, box, fof_link, cfg.fof_min_size, cfg.requested_fof_slabs,
        fof_ids.data(), &ngrp, &ngrouped, &largest, fof_el, nullptr);
  }
  const int ngpu = int(gpus.size());
  std::vector<int> dev_ids(ngpu);
  std::vector<const double*> dev_pos_ptrs(ngpu);
  std::vector<const uint32_t*> dev_orig_ptrs(ngpu);
  std::vector<uint64_t> dev_counts(ngpu);
  for (int g = 0; g < ngpu; ++g) {
    dev_ids[g] = gpus[g].dev;
    dev_pos_ptrs[g] = reinterpret_cast<const double*>(gpus[g].d_x.p);
    dev_orig_ptrs[g] = gpus[g].d_orig_idx.p;
    dev_counts[g] = uint64_t(gpus[g].ns);
  }
  return whole_fof_multigpu_swift(
      ngpu, dev_ids.data(), dev_pos_ptrs.data(), dev_orig_ptrs.data(), dev_counts.data(),
      N, box, box, box, fof_link, cfg.fof_min_size,
      fof_ids.data(), &ngrp, &ngrouped, &largest, fof_el);
}

template <typename ComputeCapFn, typename NodeCapFn>
inline int run_domain_3d_evolution(
    const Domain3DRunConfig& cfg,
    ComputeCapFn compute_part_cap,
    NodeCapFn compute_ncap) {
  auto blocks = [](int n) { return std::max(1, (n + team - 1) / team); };
  auto blocks_warps = [](int w) { return std::max(1, (w * 32 + team - 1) / team); };

  const int num_gpus = cfg.num_gpus;
  const uint64_t N = cfg.N;
  const StateInput& h = cfg.h;
  const float first_mass = float(cfg.first_mass_in);
  const float first_soft_plummer = float(cfg.first_soft_in);
  const bool mass_uniform = cfg.mass_uniform;
  const bool soft_uniform = cfg.soft_uniform;
  const int pm_side = cfg.pm_side;
  const int leaf_limit = cfg.leaf_limit;
  const double inv_cell_w = double(N_TOP) / h.box;

  std::array<double, N_TOP_CELLS> init_weights{};
  for (int c = 0; c < N_TOP_CELLS; ++c) init_weights[c] = double(cfg.init_cell_counts[c]);
  Domain3DPlan plan = build_domain_3d_plan(
      cfg.init_cell_counts, init_weights, num_gpus, cfg.domain_mode, pm_side, &cfg.can_peer);

  // Pass 2: stream input particles directly into per-GPU host staging
  std::vector<HostSlabStaging> h_gpu_stage(num_gpus);
  for (int g = 0; g < num_gpus; ++g) {
    size_t est = 0;
    for (int cid : plan.own_cells[g]) est += cfg.init_cell_counts[cid];
    h_gpu_stage[g].x.reserve(est);
    h_gpu_stage[g].v.reserve(est);
    h_gpu_stage[g].orig_idx.reserve(est);
    if (!mass_uniform) h_gpu_stage[g].mass.reserve(est);
    if (!soft_uniform) h_gpu_stage[g].soft.reserve(est);
  }

  {
    const uint64_t pos_offset = sizeof(StateInput);
    const uint64_t vel_offset = pos_offset + N * sizeof(double3);
    const uint64_t mass_offset = vel_offset + N * sizeof(float3);
    const uint64_t chunk = std::min<uint64_t>(N, 16ULL << 20);
    std::vector<double3> rx(chunk);
    std::vector<float3> rv(chunk);
    std::vector<float> rm, rs;
    if (!mass_uniform) rm.resize(chunk);
    if (!soft_uniform) rs.resize(chunk);

    FILE* fx = fopen(cfg.input_path, "rb");
    FILE* fv = fopen(cfg.input_path, "rb");
    FILE* fm = mass_uniform ? nullptr : fopen(cfg.input_path, "rb");
    FILE* fs = (!soft_uniform && cfg.has_soft_in) ? fopen(cfg.soft_in_env, "rb") : nullptr;
    if (!fx || !fv || (!mass_uniform && !fm) || (!soft_uniform && cfg.has_soft_in && !fs))
      throw std::runtime_error("Failed reopening input for 3D Pass 2 load");
    fseeko(fx, off_t(pos_offset), SEEK_SET);
    fseeko(fv, off_t(vel_offset), SEEK_SET);
    if (fm) fseeko(fm, off_t(mass_offset), SEEK_SET);

    const float a0_f = float(h.initial_scale);
    for (uint64_t base = 0; base < N; base += chunk) {
      const size_t cnt = size_t(std::min<uint64_t>(chunk, N - base));
      read_bin(fx, rx.data(), cnt);
      read_bin(fv, rv.data(), cnt);
      if (fm) read_bin(fm, rm.data(), cnt);
      if (fs) read_bin(fs, rs.data(), cnt);
      for (size_t k = 0; k < cnt; ++k) {
        double3 pos = rx[k];
        pos.x = wrapped_coord(pos.x, h.box);
        pos.y = wrapped_coord(pos.y, h.box);
        pos.z = wrapped_coord(pos.z, h.box);
        const int tx = std::min(std::max(int(pos.x * inv_cell_w), 0), N_TOP - 1);
        const int ty = std::min(std::max(int(pos.y * inv_cell_w), 0), N_TOP - 1);
        const int tz = std::min(std::max(int(pos.z * inv_cell_w), 0), N_TOP - 1);
        const int owner = plan.cell_owner[cell_id(tx, ty, tz)];
        auto& HS = h_gpu_stage[owner];
        HS.x.push_back(pos);
        const float3 vi = rv[k];
        HS.v.push_back(float3{vi.x * a0_f, vi.y * a0_f, vi.z * a0_f});
        HS.orig_idx.push_back(uint32_t(base + k));
        if (!mass_uniform) HS.mass.push_back(rm[k]);
        if (!soft_uniform) HS.soft.push_back(float(double(rs[k]) * kernel_gravity_softening_plummer_equivalent));
      }
    }
    fclose(fx); fclose(fv);
    if (fm) fclose(fm);
    if (fs) fclose(fs);
    if (const char* unlink_env = std::getenv("SWIFT_UNLINK_INPUT_AFTER_LOAD")) {
      if (std::string(unlink_env) == "1") std::remove(cfg.input_path);
    }
  }

  std::vector<Dom3DGpuCtx> gpus(num_gpus);
  for (int g = 0; g < num_gpus; ++g) {
    gpus[g].dev = g;
    check(cudaSetDevice(g));
    check(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
    check(cudaStreamCreateWithFlags(&gpus[g].compute_stream, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&gpus[g].copy_stream, cudaStreamNonBlocking));
    check(cudaStreamCreateWithFlags(&gpus[g].d2h_stream, cudaStreamNonBlocking));
    check(cudaEventCreateWithFlags(&gpus[g].ev_force_start, 0));
    check(cudaEventCreateWithFlags(&gpus[g].ev_force_end, 0));
    check(cudaEventCreateWithFlags(&gpus[g].ev_prof0, 0));
    check(cudaEventCreateWithFlags(&gpus[g].ev_prof1, 0));
    check(cudaEventCreateWithFlags(&gpus[g].ev_ghost_start, 0));
    check(cudaEventCreateWithFlags(&gpus[g].ev_ghost_ready, 0));
    check(cudaEventCreateWithFlags(&gpus[g].ev_int_start, 0));
    check(cudaEventCreateWithFlags(&gpus[g].ev_int_end, 0));
    check(cudaEventCreateWithFlags(&gpus[g].ev_bnd_start, 0));
    check(cudaEventCreateWithFlags(&gpus[g].ev_bnd_end, 0));
  }

  const size_t pm_points = size_t(pm_side) * size_t(pm_side) * size_t(pm_side);
  size_t peak_gpu_parts = 0;
  for (int g = 0; g < num_gpus; ++g) peak_gpu_parts = std::max(peak_gpu_parts, h_gpu_stage[g].x.size());

  auto upload_plan_to_gpu = [&](Dom3DGpuCtx& G, int g, const Domain3DPlan& pl) {
    check(cudaSetDevice(G.dev));
    G.n_own = int(pl.own_cells[g].size());
    G.n_interior_own = pl.n_interior_own[g];
    G.n_ghost = int(pl.ghost_cells[g].size());
    G.d_cell_owner.alloc(N_TOP_CELLS, G.device_bytes);
    G.d_cell_to_slot.alloc(N_TOP_CELLS, G.device_bytes);
    G.d_slot_to_cell.alloc(std::max(1, G.n_own), G.device_bytes);
    G.d_slot_nb_roots.alloc(std::max(1, G.n_own * 27), G.device_bytes);
    std::vector<int> h_c2s(N_TOP_CELLS, -1);
    for (int s = 0; s < G.n_own; ++s) h_c2s[pl.own_cells[g][s]] = s;
    check(cudaMemcpyAsync(G.d_cell_owner.p, pl.cell_owner.data(),
                          N_TOP_CELLS * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
    check(cudaMemcpyAsync(G.d_cell_to_slot.p, h_c2s.data(),
                          N_TOP_CELLS * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
    if (G.n_own > 0) {
      check(cudaMemcpyAsync(G.d_slot_to_cell.p, pl.own_cells[g].data(),
                            size_t(G.n_own) * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_slot_nb_roots.p, pl.slot_nb_roots[g].data(),
                            size_t(G.n_own) * 27ULL * sizeof(uint32_t), cudaMemcpyHostToDevice, G.compute_stream));
    }
    check(cudaStreamSynchronize(G.compute_stream));
  };

#pragma omp parallel num_threads(num_gpus)
  {
    const int g = omp_get_thread_num();
    auto& G = gpus[g];
    check(cudaSetDevice(G.dev));
    G.d_active_count.alloc(1, G.device_bytes);
    G.d_tail_count.alloc(1, G.device_bytes);
    G.d_tree_invalid.alloc(1, G.device_bytes);
    G.d_stats.alloc(1, G.device_bytes);
    G.d_kick_finish.alloc(57, G.device_bytes);
    G.d_kick_begin.alloc(57, G.device_bytes);
    G.d_active_bits.alloc(54, G.device_bytes);
    G.d_cell_counts.alloc(N_TOP_CELLS, G.device_bytes);
    G.d_cell_act_counts.alloc(N_TOP_CELLS, G.device_bytes);
    G.d_split_total_warps.alloc(2, G.device_bytes);
    G.d_own_slot_desc_cnt.alloc(N_TOP_CELLS, G.device_bytes);
    G.d_own_slot_pstart.alloc(N_TOP_CELLS, G.device_bytes);
    G.d_own_slot_pcnt.alloc(N_TOP_CELLS, G.device_bytes);
    G.h_slot_meta.alloc(3 * size_t(N_TOP_CELLS) * sizeof(int), cudaHostAllocPortable);
    if (!cfg.pm_dist_enabled) {
      G.d_pm_density.alloc(pm_points, G.device_bytes);
      G.d_pm_field.alloc(pm_points, G.device_bytes);
    }
    G.d_scan_tmp.alloc(scan_grid_max, G.device_bytes);
    G.h_scan.resize(scan_grid_max);
    G.ensure_mig_out(262144);
    G.ensure_mig_in(262144);

    const auto& HS = h_gpu_stage[g];
    G.ns = HS.x.size();
    const size_t init_cap = std::max(compute_part_cap(G.ns), size_t(double(G.ns) * 1.12) + 32768);
    const size_t init_ncap = compute_ncap(init_cap, leaf_limit);
    G.alloc_particles(init_cap, mass_uniform, soft_uniform);
    G.alloc_local_tree_nodes(init_ncap, cfg.lazy_tree_enabled);
    if (G.ns > 0) {
      check(cudaMemcpyAsync(G.d_x.p, HS.x.data(), G.ns * sizeof(double3), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_v.p, HS.v.data(), G.ns * sizeof(float3), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_orig_idx.p, HS.orig_idx.data(), G.ns * sizeof(uint32_t), cudaMemcpyHostToDevice, G.compute_stream));
      if (!mass_uniform)
        check(cudaMemcpyAsync(G.d_mass.p, HS.mass.data(), G.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
      if (!soft_uniform)
        check(cudaMemcpyAsync(G.d_soft.p, HS.soft.data(), G.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
    }
    check(cudaMemsetAsync(G.d_mesh_acc.p, 0, G.part_cap * sizeof(float3), G.compute_stream));
    check(cudaMemsetAsync(G.d_old_acc.p, 0, G.part_cap * sizeof(float), G.compute_stream));
    check(cudaMemsetAsync(G.d_bins.p, 1, G.part_cap * sizeof(unsigned char), G.compute_stream));
    upload_plan_to_gpu(G, g, plan);
    G.ensure_ghost_cap((num_gpus > 1) ? init_cap : 1, (num_gpus > 1) ? init_ncap : N_TOP_CELLS, soft_uniform);
    G.ensure_split_bufs(size_t(cfg.split_batch_warps), 1);
    G.h_bounce.alloc((num_gpus > 1) ? (64ULL << 20) : (80ULL << 20), cudaHostAllocPortable);
  }
  for (int g = 0; g < num_gpus; ++g) {
    std::vector<double3>().swap(h_gpu_stage[g].x);
    std::vector<float3>().swap(h_gpu_stage[g].v);
    std::vector<float>().swap(h_gpu_stage[g].mass);
    std::vector<float>().swap(h_gpu_stage[g].soft);
    std::vector<uint32_t>().swap(h_gpu_stage[g].orig_idx);
  }

  std::vector<std::vector<Dom3DEdgeState>> edges(num_gpus, std::vector<Dom3DEdgeState>(num_gpus));
  for (int src = 0; src < num_gpus; ++src) {
    for (int dst = 0; dst < num_gpus; ++dst) {
      if (src != dst) edges[src][dst].init_events(gpus[src].dev);
    }
  }
  auto setup_all_edges = [&](const Domain3DPlan& pl) {
    for (int src = 0; src < num_gpus; ++src) {
      for (int dst = 0; dst < num_gpus; ++dst) {
        if (src == dst) continue;
        edges[src][dst].setup_plan(
            gpus[src].dev, gpus[dst].dev, int(pl.own_cells[src].size()),
            pl.export_src_slots[src][dst], pl.export_dst_ghost_slots[src][dst],
            gpus[src].device_bytes, gpus[dst].device_bytes);
      }
    }
  };
  setup_all_edges(plan);
  for (int src = 0; src < num_gpus; ++src) {
    for (int dst = 0; dst < num_gpus; ++dst) {
      if (src == dst || cfg.can_peer[dst][src] || plan.export_cells[src][dst].empty()) continue;
      size_t est_p = 0;
      for (int cid : plan.export_cells[src][dst]) est_p += cfg.init_cell_counts[cid];
      const size_t est_w = std::max<size_t>(est_p * 48ULL, 16ULL << 20);
      alloc_pinned_slack(edges[src][dst].h_bounce, est_w);
    }
  }

  check(cudaSetDevice(0));
  const double load_s = wall() - cfg.import_start;

  SwiftDistPm dist_pm;
  if (cfg.pm_dist_enabled) {
    std::vector<int> devs(num_gpus);
    std::vector<cudaStream_t> streams(num_gpus);
    for (int g = 0; g < num_gpus; ++g) {
      devs[g] = gpus[g].dev;
      streams[g] = gpus[g].compute_stream;
    }
    dist_pm.init(devs, streams, cfg.can_peer, pm_side, plan.gpu_x_planes);
    for (int g = 0; g < num_gpus; ++g) gpus[g].device_bytes += dist_pm.device_bytes(g);
  }

  std::vector<double> knots;
  knots.push_back(h.initial_scale);
  for (double sa : cfg.snap_scales) knots.push_back(sa);
  knots.push_back(cfg.a_end);

  const double log_span = cfg.comoving ? log(knots[1] / knots[0]) : (knots[1] - knots[0]);
  const float soft = cfg.has_soft_in
      ? float(double(first_soft_plummer) * kernel_gravity_softening_plummer_equivalent)
      : float(h.plummer_softening * kernel_gravity_softening_plummer_equivalent);
  const double rho_crit0 = 3.0 * cfg.cosmo_h0 * cfg.cosmo_h0 / (8.0 * acos(-1.0) * double(cfg.grav_G));
  const double d_dm = cbrt(cfg.min_mass / (cfg.cosmo_ocdm * rho_crit0));
  const double r_s = 1.25 * h.box / double(pm_side);

  Physics p{knots[0], log_span,
      float(h.box), float(r_s), float(4.5 * r_s), cfg.grav_G,
      timeline / cfg.pm_intervals, timeline / cfg.short_intervals, cfg.step_factor, cfg.mac,
      cfg.dt_max_val, std::min(r_s, d_dm), cfg.max_rms_fac, cfg.dynamic_pm,
      cfg.cosmo_h0, cfg.cosmo_om, cfg.cosmo_ode, cfg.cosmo_or, cfg.cosmo_ok, cfg.cosmo_w0, cfg.cosmo_wa,
      cfg.theta_val, cfg.grav_eta_val,
      true, cfg.comoving,
      0.f, 1.f, 0.f, 0.f, 0.f};
  const double potential_norm = cfg.periodic
      ? (4.0 * acos(-1.0) * cfg.total_mass * double(p.split) * double(p.split) / (h.box * h.box * h.box))
      : 0.0;

  probe_header gh{};
  gh.top[0] = gh.top[1] = gh.top[2] = N_TOP;
  for (int axis = 0; axis < 3; ++axis) {
    gh.width[axis] = h.box / double(N_TOP);
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
    gpu_part_counts[g] = gpus[g].ns;
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
  std::cout << std::setprecision(12)
            << "MEMORY_PLAN {\"mode\":\"multi_gpu_domain_3d\",\"domain_mode\":\"" << domain_mode_name(cfg.domain_mode)
            << "\",\"num_gpus\":" << num_gpus
            << ",\"top_cells\":" << N_TOP_CELLS
            << ",\"particles\":" << N
            << ",\"w_imb\":" << plan.stats.w_imb << ",\"p_imb\":" << plan.stats.p_imb
            << ",\"ghost_cells_total\":" << plan.stats.ghost_total
            << ",\"max_x_slabs\":" << plan.stats.max_x_slabs
            << ",\"per_gpu_particles\":";
  print_vec(gpu_part_counts);
  std::cout << ",\"per_gpu_device_bytes\":";
  print_vec(gpu_dev_bytes);
  std::cout << ",\"per_gpu_free_vram_bytes\":";
  print_vec(gpu_free_vram);
  std::cout << ",\"can_peer\":[";
  for (int i = 0; i < num_gpus; ++i) {
    if (i) std::cout << ",";
    print_vec(cfg.can_peer[i]);
  }
  std::cout << "],\"p2p_mode\":\"" << cfg.p2p_mode
            << "\",\"uniform_mass\":" << (mass_uniform ? "true" : "false")
            << ",\"uniform_soft\":" << (soft_uniform ? "true" : "false")
            << ",\"device_buffer_bytes\":" << sum_dev_bytes
            << ",\"free_vram_bytes\":" << min_free_vram
            << ",\"total_vram_bytes\":" << total_vram_0 << "}" << std::endl;

  std::cout << std::setprecision(12)
            << "GPU_INPUT {\"particles\":" << N << ",\"num_gpus\":" << num_gpus
            << ",\"domain_mode\":\"" << domain_mode_name(cfg.domain_mode)
            << "\",\"rebalance_interval\":" << cfg.rebalance_interval
            << ",\"rebalance_min_gain\":" << cfg.rebalance_min_gain
            << ",\"periodic\":true,\"comoving\":" << (cfg.comoving ? "true" : "false")
            << ",\"pm_grid\":" << pm_side
            << ",\"dynamic_pm\":" << (cfg.dynamic_pm ? "true" : "false")
            << ",\"grav_G\":" << p.G << ",\"pm_intervals\":" << cfg.pm_intervals
            << ",\"short_cap_intervals\":" << cfg.short_intervals
            << ",\"short_step_factor\":" << cfg.step_factor
            << ",\"mac_tolerance\":" << cfg.mac
            << ",\"theta\":" << p.theta << ",\"grav_eta\":" << p.grav_eta
            << ",\"hubble0\":" << p.hubble0 << ",\"omega_m\":" << p.omega_m
            << ",\"omega_de\":" << p.omega_de << ",\"omega_r\":" << p.omega_r
            << ",\"omega_k\":" << p.omega_k << ",\"w0\":" << p.w0 << ",\"wa\":" << p.wa
            << ",\"force_check_n\":" << cfg.force_check_n
            << ",\"lightcone\":" << (cfg.lc_enabled ? "true" : "false")
            << ",\"ext_pot_M\":0,\"a_begin\":" << h.initial_scale << ",\"a_end\":" << cfg.a_end
            << ",\"warp_m2l\":" << (cfg.warp_m2l ? "true" : "false")
            << ",\"fast_split\":" << (cfg.fast_split ? "true" : "false")
            << ",\"split_force\":" << (cfg.split_force_enabled ? "true" : "false")
            << ",\"pm_dist\":" << (cfg.pm_dist_enabled ? "true" : "false")
            << ",\"split_batch_warps\":" << cfg.split_batch_warps
            << ",\"tree_leaf_limit\":" << gh.leaf_limit
            << ",\"softening_internal_mpc\":" << soft << ",\"box_mpc\":" << h.box
            << ",\"prepare_and_load_s\":" << load_s
            << ",\"device\":\"" << prop0.name << "\"}" << std::endl;

  // Auxiliary Ewald & Lightcone setup (1-GPU parity)
  size_t aux_dev_bytes = 0;
  const int fc_target_cap = int(std::min<uint64_t>(uint64_t(cfg.force_check_n), N));
  std::vector<EwaldMode> h_ewald_modes;
  DevPtr<EwaldMode> d_ewald_modes;
  DevPtr<double> d_ewald_C, d_ewald_S;
  DevPtr<double3> d_fc_tgt_x, d_fc_short_ref, d_fc_total_ref;
  DevPtr<float> d_fc_tgt_h;
  DevPtr<int> d_fc_lids, d_fc_tgt_slab;
  DevPtr<float4> d_fc_gpu_short;
  DevPtr<float3> d_fc_gpu_mesh;
  if (fc_target_cap > 0) {
    check(cudaSetDevice(0));
    d_fc_tgt_x.alloc(fc_target_cap, aux_dev_bytes);
    d_fc_tgt_h.alloc(fc_target_cap, aux_dev_bytes);
    d_fc_lids.alloc(fc_target_cap, aux_dev_bytes);
    d_fc_tgt_slab.alloc(fc_target_cap, aux_dev_bytes);
    check(cudaMemset(d_fc_tgt_slab.p, 0, size_t(fc_target_cap) * sizeof(int)));
    d_fc_gpu_short.alloc(fc_target_cap, aux_dev_bytes);
    d_fc_gpu_mesh.alloc(fc_target_cap, aux_dev_bytes);
    d_fc_short_ref.alloc(fc_target_cap, aux_dev_bytes);
    d_fc_total_ref.alloc(fc_target_cap, aux_dev_bytes);
    h_ewald_modes = build_ewald_modes(h.box);
    d_ewald_modes.alloc(h_ewald_modes.size(), aux_dev_bytes);
    d_ewald_C.alloc(h_ewald_modes.size(), aux_dev_bytes);
    d_ewald_S.alloc(h_ewald_modes.size(), aux_dev_bytes);
    check(cudaMemcpy(d_ewald_modes.p, h_ewald_modes.data(),
                     h_ewald_modes.size() * sizeof(EwaldMode), cudaMemcpyHostToDevice));
  }

  FILE* lc_fp = nullptr;
  uint64_t lc_total_hits = 0;
  double lc_r_max = 0.0;
  DevPtr<LightconeHit> d_lc_hits;
  DevPtr<unsigned int> d_lc_count;
  DevPtr<double3> d_lc_shifts;
  size_t lc_hit_cap = 0, lc_shift_cap = 0;
  AsyncLightconeWriter lc_writer;
  if (cfg.lc_enabled) {
    check(cudaSetDevice(0));
    lc_r_max = (cfg.lc_r_max_req > 0.0) ? cfg.lc_r_max_req
                                        : comoving_distance_mpc(h.initial_scale, cfg.lc_a_obs, p);
    lc_fp = fopen(cfg.lc_out_env, "wb");
    if (!lc_fp) throw std::runtime_error(std::string("Cannot open SWIFT_LIGHTCONE_OUT: ") + cfg.lc_out_env);
    const LightconeHeader blank_hdr{{'S','W','L','C','0','0','0','1'}, 0,
                                    {cfg.lc_obs.x, cfg.lc_obs.y, cfg.lc_obs.z},
                                    h.initial_scale, cfg.a_end, cfg.lc_a_obs, lc_r_max, h.box};
    write_bin(lc_fp, &blank_hdr, 1);
    lc_writer.start(lc_fp);
    lc_hit_cap = 262144;
    d_lc_hits.alloc(lc_hit_cap, aux_dev_bytes);
    d_lc_count.alloc(1, aux_dev_bytes);
  }

  // Host output containers
  std::vector<double3> out_x;
  std::vector<float3> out_v, out_short_f3, out_long_f3;
  std::vector<float> out_pot;
  std::vector<float4> out_short_f4, out_long_f4, out_force0_f4;
  if (cfg.force0_out_path) out_force0_f4.resize(N);

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
    ~WriterJoinGuard() { if (t.joinable()) t.join(); }
  } writer_guard{snap_writer};

  auto grow_resident_3d = [&](Dom3DGpuCtx& G, size_t need_n) {
    check(cudaSetDevice(G.dev));
    const size_t new_cap = std::max(compute_part_cap(need_n), size_t(double(need_n) * 1.12) + 32768);
    const size_t new_ncap = compute_ncap(new_cap, leaf_limit);
    std::vector<double3> sx(G.ns);
    std::vector<float3> sv(G.ns), sm(G.ns);
    std::vector<float> so(G.ns), smass, ssoft;
    std::vector<unsigned char> sb(G.ns);
    std::vector<uint32_t> sid(G.ns);
    if (G.ns > 0) {
      check(cudaMemcpyAsync(sx.data(), G.d_x.p, G.ns * sizeof(double3), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(sv.data(), G.d_v.p, G.ns * sizeof(float3), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(sm.data(), G.d_mesh_acc.p, G.ns * sizeof(float3), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(so.data(), G.d_old_acc.p, G.ns * sizeof(float), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(sb.data(), G.d_bins.p, G.ns * sizeof(unsigned char), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(sid.data(), G.d_orig_idx.p, G.ns * sizeof(uint32_t), cudaMemcpyDeviceToHost, G.compute_stream));
      if (!mass_uniform) {
        smass.resize(G.ns);
        check(cudaMemcpyAsync(smass.data(), G.d_mass.p, G.ns * sizeof(float), cudaMemcpyDeviceToHost, G.compute_stream));
      }
      if (!soft_uniform) {
        ssoft.resize(G.ns);
        check(cudaMemcpyAsync(ssoft.data(), G.d_soft.p, G.ns * sizeof(float), cudaMemcpyDeviceToHost, G.compute_stream));
      }
      check(cudaStreamSynchronize(G.compute_stream));
    }
    G.alloc_particles(new_cap, mass_uniform, soft_uniform);
    G.alloc_local_tree_nodes(new_ncap, cfg.lazy_tree_enabled);
    if (G.ns > 0) {
      check(cudaMemcpyAsync(G.d_x.p, sx.data(), G.ns * sizeof(double3), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_v.p, sv.data(), G.ns * sizeof(float3), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_mesh_acc.p, sm.data(), G.ns * sizeof(float3), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_old_acc.p, so.data(), G.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_bins.p, sb.data(), G.ns * sizeof(unsigned char), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_orig_idx.p, sid.data(), G.ns * sizeof(uint32_t), cudaMemcpyHostToDevice, G.compute_stream));
      if (!mass_uniform)
        check(cudaMemcpyAsync(G.d_mass.p, smass.data(), G.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
      if (!soft_uniform)
        check(cudaMemcpyAsync(G.d_soft.p, ssoft.data(), G.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaStreamSynchronize(G.compute_stream));
    }
  };

  auto build_local_tree_3d = [&](Dom3DGpuCtx& G, bool compute_nvel) {
    const int nk = int(G.ns);
    if (nk == 0) {
      zero_empty_roots<<<blocks(std::max(1, G.n_own)), team, 0, G.compute_stream>>>(
          G.force_nodes.p, G.escape.p, G.first_child.p, G.n_own);
      pack_split_nodes_kernel<<<blocks(std::max(1, G.n_own)), team, 0, G.compute_stream>>>(
          G.force_nodes.p, G.first_child.p, G.node_meta.p, G.compact_m.p, G.n_own);
      G.nodes = unsigned(G.n_own);
      G.h_levels.fill(G.n_own);
      G.h_levels[0] = 0;
      return;
    }
    for (int attempt = 0; attempt < 4; ++attempt) {
      check(cudaMemsetAsync(G.d_tree_invalid.p, 0, sizeof(int), G.compute_stream));
      const int p_grid = blocks(nk);
      key_particles_fast_3d<<<p_grid, team, 0, G.compute_stream>>>(
          G.d_x.p, nk, G.d_cell_to_slot.p, gh, G.d_key_tmp.p, G.d_ord_tmp.p);
      size_t sb = G.sort_bytes;
      check(cub::DeviceRadixSort::SortPairs(
          G.d_sort_storage.p, sb, G.d_key_tmp.p, G.d_keys_sorted.p,
          G.d_ord_tmp.p, G.order.p, nk, 0, 62, G.compute_stream));
      extract_home_slots_3d_kernel<<<p_grid, team, 0, G.compute_stream>>>(
          G.d_keys_sorted.p, G.home_slot.p, nk);
      roots<<<blocks(G.n_own), team, 0, G.compute_stream>>>(
          G.d_keys_sorted.p, nk, G.geometry.p, G.n_own);
      tree_offsets_begin<<<1, 1, 0, G.compute_stream>>>(G.levels.p, G.node_count.p, G.n_own);

      const int n_grid = blocks(int(G.node_cap));
      for (int depth = 0; depth < tree_levels; ++depth) {
        split<<<n_grid, team, 0, G.compute_stream>>>(
            G.d_keys_sorted.p, G.geometry.p, 0, 0, depth, gh.leaf_limit,
            G.node_count.p, int(G.node_cap), G.d_tree_invalid.p, G.levels.p);
        tree_offsets_end<<<1, 1, 0, G.compute_stream>>>(
            G.levels.p, G.node_count.p, depth, int(G.node_cap));
        tree_links<<<n_grid, team, 0, G.compute_stream>>>(
            G.geometry.p, G.escape.p, G.levels.p, depth, G.first_child.p);
      }
      unsigned int nc = 0;
      int inv_cap = 0;
      check(cudaMemcpyAsync(&nc, G.node_count.p, sizeof(unsigned int), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(&inv_cap, G.d_tree_invalid.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(G.h_levels.data(), G.levels.p, size_t(tree_levels + 2) * sizeof(int),
                            cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaStreamSynchronize(G.compute_stream));
      if (inv_cap != 0 || nc >= G.node_cap) {
        if (attempt < 3) {
          const size_t next_nc = std::max(G.node_cap * 3 / 2, size_t(nc) + 131072);
          G.alloc_local_tree_nodes(next_nc, cfg.lazy_tree_enabled);
          continue;
        }
        throw std::runtime_error("3D tree split overflow on GPU " + std::to_string(G.dev));
      }

      check_tree_limit<<<n_grid, team, 0, G.compute_stream>>>(
          G.geometry.p, G.levels.p, tree_levels, gh.leaf_limit, G.d_tree_invalid.p);
      tree_links<<<n_grid, team, 0, G.compute_stream>>>(
          G.geometry.p, G.escape.p, G.levels.p, tree_levels, G.first_child.p);
      for (int depth = tree_levels; depth >= 0; --depth) {
        multipoles_fast_3d<<<n_grid, team, 0, G.compute_stream>>>(
            G.geometry.p, G.force_nodes.p, G.d_node_com.p, G.d_x.p,
            mass_uniform ? nullptr : G.d_mass.p, first_mass, soft,
            G.order.p, depth, G.d_slot_to_cell.p, gh, G.levels.p,
            soft_uniform ? nullptr : G.d_soft.p);
      }
      gather_sources_slab<<<p_grid, team, 0, G.compute_stream>>>(
          G.d_x.p, mass_uniform ? nullptr : G.d_mass.p, first_mass,
          soft_uniform ? nullptr : G.d_soft.p,
          G.order.p, G.packed_sources.p, G.packed_soft.p, nk);
      if (compute_nvel) {
        for (int depth = tree_levels; depth >= 0; --depth) {
          const int lcnt = G.h_levels[depth + 1] - G.h_levels[depth];
          if (lcnt <= 0) continue;
          compute_node_vel_active_3d_kernel<<<blocks(lcnt), team, 0, G.compute_stream>>>(
              G.geometry.p, G.force_nodes.p, G.d_v.p,
              mass_uniform ? nullptr : G.d_mass.p, first_mass,
              G.order.p, depth, G.levels.p, nullptr, G.d_slot_to_cell.p, G.node_vel_max.p);
        }
      }
      pack_split_nodes_kernel<<<blocks(int(nc)), team, 0, G.compute_stream>>>(
          G.force_nodes.p, G.first_child.p, G.node_meta.p, G.compact_m.p, int(nc));
      check(cudaGetLastError());
      int inv_leaf = 0;
      check(cudaMemcpyAsync(&inv_leaf, G.d_tree_invalid.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaStreamSynchronize(G.compute_stream));
      if (inv_leaf != 0) {
        throw std::runtime_error("3D tree depth>14 leaf limit failed on GPU " + std::to_string(G.dev));
      }
      G.nodes = nc;
      return;
    }
  };

  // State across steps
  std::vector<double> gpu_force_ms(num_gpus, 0.0), gpu_kick_ms(num_gpus, 0.0);
  std::vector<double> gpu_ghost_ms(num_gpus, 0.0), gpu_xfer_ms(num_gpus, 0.0), gpu_tree_ms(num_gpus, 0.0);
  std::vector<uint64_t> gpu_active(num_gpus, 0), gpu_nodes(num_gpus, 0);
  std::vector<Tick> gpu_inactive_min(num_gpus, timeline), gpu_active_next(num_gpus, timeline);
  std::vector<double> gpu_v2_sum(num_gpus, 0.0);
  std::vector<float> gpu_max_v(num_gpus, 0.f);
  std::vector<unsigned int> gpu_invalid(num_gpus, 0);
  std::vector<int> gpu_builds(num_gpus, 0);
  std::vector<std::array<uint32_t, 54>> gpu_act_bits(num_gpus);
  std::array<uint8_t, N_TOP_CELLS> h_cell_active{};
  std::vector<std::array<unsigned long long, N_TOP_CELLS>> gpu_cell_cnt(num_gpus), gpu_cell_act(num_gpus);
  std::array<unsigned long long, N_TOP_CELLS> cell_act_accum{};
  int steps_since_rebal = 0;
  int rebalance_count = 0;

  std::vector<std::vector<DevMigrant>> h_mig_out(num_gpus), h_mig_in(num_gpus);
  unsigned int pm_updates = 0;
  double last_pm_span_scale = log_span / double(cfg.pm_intervals);
  float last_dt_max_rms = dt_displacement_default;
  size_t peak_gpu_seen = peak_gpu_parts;
  bool midpoint_written = false;
  uint64_t split_batches_run = 0, split_overflow_retries = 0;
  uint64_t total_lazy_steps = 0, total_full_rebuilds = 0;
  double cum_max_disp = 0.0;
  int lazy_steps_since_rebuild = 0;
  bool all_trees_built = false;
  bool ghost_installed = false;

  constexpr int max_steps_guard = 1000000;
  int global_iteration = 0;
  size_t sanity_bad = 0;
  double sanity_v2_sum = 0.0, sanity_pot_sum = 0.0;
  float sanity_pot_min = FLT_MAX, sanity_pot_max = -FLT_MAX;
  std::vector<double*> d_energy_accum(num_gpus, nullptr);
  std::vector<std::array<double, 10>> gpu_energy(num_gpus);
  if (cfg.energy_log) {
    for (int g = 0; g < num_gpus; ++g) {
      check(cudaSetDevice(g));
      check(cudaMalloc(&d_energy_accum[g], 10 * sizeof(double)));
    }
    check(cudaSetDevice(0));
  }
  const double evo_start = wall();

  for (size_t seg = 0; seg + 1 < knots.size(); ++seg) {
    const double seg_a_begin = knots[seg], seg_a_end = knots[seg + 1];
    const bool is_last_seg = (seg + 2 == knots.size());
    const double seg_span = cfg.comoving ? log(seg_a_end / seg_a_begin) : (seg_a_end - seg_a_begin);
    p.a_begin = seg_a_begin;
    p.log_span = seg_span;

    Tick ti = 0, previous = 0;
    Tick next_pm_boundary = timeline, current_pm_span = p.pm_span;
    Tick next_pm_span = p.pm_span, prev_pm_span = 0, short_span_cap = p.short_span_cap;
    double pm_kick = 0.0;
    float dt_max_rms = dt_displacement_default;
    all_trees_built = false;
    ghost_installed = false;
    lazy_steps_since_rebuild = 0;
    cum_max_disp = 0.0;

    if (seg > 0) {
      check(cudaSetDevice(0));
#pragma omp parallel num_threads(num_gpus)
      {
        const int g = omp_get_thread_num();
        auto& G = gpus[g];
        check(cudaSetDevice(G.dev));
        check(cudaMemsetAsync(G.d_mesh_acc.p, 0, G.part_cap * sizeof(float3), G.compute_stream));
        check(cudaMemsetAsync(G.d_old_acc.p, 0, G.part_cap * sizeof(float), G.compute_stream));
        check(cudaMemsetAsync(G.d_bins.p, 1, G.part_cap * sizeof(unsigned char), G.compute_stream));
        check(cudaStreamSynchronize(G.compute_stream));
      }
      check(cudaSetDevice(0));
    }

    for (int iteration = 0;; ++iteration, ++global_iteration) {
      if (iteration > max_steps_guard) throw std::runtime_error("Exceeded max_steps_guard");

      const double step_start = wall();
      const bool is_pm_step = (ti == 0 || ti == next_pm_boundary);
      const bool is_final = (ti == timeline);
      const bool is_run_final = is_final && is_last_seg;
      const bool midpoint = cfg.midpoint_path && ti == timeline / 2;
      const double closing_pm_kick = midpoint ? p.factors(ti - current_pm_span / 2, ti, false) : 0.0;
      const double drift_fac = (ti > 0) ? p.factors(previous, ti, true) : 0.0;
      const double a = p.scale(ti);
      const double H_cur = p.H(a);
      const bool need_vel_sum = (is_pm_step && !is_final && cfg.dynamic_pm);
      const bool do_fc_now = (fc_target_cap > 0) && ((ti == 0 && seg == 0) || is_run_final);
      int n_fc_tgt = 0;
      std::vector<int> h_fc_lids;

      const double wc = h.box / double(N_TOP);
      const double max_safe_disp = std::min(cfg.lazy_disp_frac * r_s, 0.35 * std::max(0.0, wc - double(p.cutoff)));
      const bool eligible_lazy = cfg.lazy_tree_enabled && (ti > 0) && !is_pm_step &&
          !is_final && !midpoint && !do_fc_now && all_trees_built &&
          (lazy_steps_since_rebuild < cfg.lazy_max_steps) && (max_safe_disp > 0.0);
      bool do_lazy_step = false;
      bool force_rebuild_all = is_pm_step || (ti == 0) || !all_trees_built || !ghost_installed;
      bool did_rebalance = false;
      double step_max_disp = 0.0;
      size_t total_migrants = 0;
      bool any_ns_changed = false;

      if (is_final || midpoint) {
        join_snap_writer();
        if (!cfg.skip_output_write) {
          out_x.resize(N); out_v.resize(N); out_pot.resize(N);
        }
        if (is_run_final && cfg.force_comp_path) {
          out_short_f4.resize(N); out_long_f4.resize(N);
        }
        if (is_run_final && cfg.force3_out_path) {
          out_short_f3.resize(N); out_long_f3.resize(N);
        }
      }

      double drift_ms = 0.0, pm_ms = 0.0, tree_ms = 0.0, force_ms = 0.0;
      double kick_ms = 0.0, ghost_ms = 0.0, export_ms = 0.0;
      const bool check_rebal_this_step = (num_gpus > 1 && cfg.rebalance_interval > 0 &&
          ti > 0 && !is_final && !midpoint && !do_fc_now &&
          (global_iteration % cfg.rebalance_interval == 0));

#pragma omp parallel num_threads(num_gpus)
      {
        const int g = omp_get_thread_num();
        auto& G = gpus[g];
        check(cudaSetDevice(G.dev));

        // ================= STEP A: Drift + Migrate + Optional Dynamic Rebalance =================
        const double t_a0 = wall();
        if (ti > 0 && cfg.lc_enabled && g == 0) {
          const double a_prev = p.scale(previous);
          if (a_prev < cfg.lc_a_obs) {
            const double a_step_end = std::min(a, cfg.lc_a_obs);
            const double tau_scale = (a > a_prev) ? std::min(1.0, (a_step_end - a_prev) / (a - a_prev)) : 1.0;
            const double step_drift_fac = drift_fac * tau_scale;
            const double chi_0 = comoving_distance_mpc(a_prev, cfg.lc_a_obs, p);
            const double chi_1 = comoving_distance_mpc(a_step_end, cfg.lc_a_obs, p);
            const double r_hi = std::min(chi_0, lc_r_max);
            if (chi_0 > chi_1 && chi_1 <= lc_r_max && G.ns > 0) {
              std::vector<double3> shifts;
              if (!cfg.lc_replicate) {
                shifts.push_back({0.0, 0.0, 0.0});
              } else {
                const double L = h.box, m_box = 0.5 * L;
                auto axis_rng = [&](double q) -> std::pair<int, int> {
                  return {int(std::floor((q - r_hi - m_box) / L)) - 1, int(std::ceil((q + r_hi + m_box) / L))};
                };
                const auto [nx0, nx1] = axis_rng(cfg.lc_obs.x);
                const auto [ny0, ny1] = axis_rng(cfg.lc_obs.y);
                const auto [nz0, nz1] = axis_rng(cfg.lc_obs.z);
                for (int nx = nx0; nx <= nx1; ++nx) {
                  const double bx0 = double(nx) * L - m_box, bx1 = double(nx + 1) * L + m_box;
                  const double dx_min = (cfg.lc_obs.x < bx0) ? (bx0 - cfg.lc_obs.x) : ((cfg.lc_obs.x > bx1) ? (cfg.lc_obs.x - bx1) : 0.0);
                  const double dx_max = std::max(std::abs(cfg.lc_obs.x - bx0), std::abs(cfg.lc_obs.x - bx1));
                  for (int ny = ny0; ny <= ny1; ++ny) {
                    const double by0 = double(ny) * L - m_box, by1 = double(ny + 1) * L + m_box;
                    const double dy_min = (cfg.lc_obs.y < by0) ? (by0 - cfg.lc_obs.y) : ((cfg.lc_obs.y > by1) ? (cfg.lc_obs.y - by1) : 0.0);
                    const double dy_max = std::max(std::abs(cfg.lc_obs.y - by0), std::abs(cfg.lc_obs.y - by1));
                    for (int nz = nz0; nz <= nz1; ++nz) {
                      const double bz0 = double(nz) * L - m_box, bz1 = double(nz + 1) * L + m_box;
                      const double dz_min = (cfg.lc_obs.z < bz0) ? (bz0 - cfg.lc_obs.z) : ((cfg.lc_obs.z > bz1) ? (cfg.lc_obs.z - bz1) : 0.0);
                      const double dz_max = std::max(std::abs(cfg.lc_obs.z - bz0), std::abs(cfg.lc_obs.z - bz1));
                      const double d_min = std::sqrt(dx_min*dx_min + dy_min*dy_min + dz_min*dz_min);
                      const double d_max = std::sqrt(dx_max*dx_max + dy_max*dy_max + dz_max*dz_max);
                      if (d_min <= r_hi && d_max >= chi_1) shifts.push_back({double(nx)*L, double(ny)*L, double(nz)*L});
                    }
                  }
                }
              }
              if (!shifts.empty()) {
                if (shifts.size() > lc_shift_cap) {
                  lc_shift_cap = shifts.size() + 64;
                  d_lc_shifts.alloc(lc_shift_cap, aux_dev_bytes);
                }
                check(cudaMemcpyAsync(d_lc_shifts.p, shifts.data(), shifts.size() * sizeof(double3),
                                      cudaMemcpyHostToDevice, G.compute_stream));
                unsigned int hit_cnt = 0;
                for (int retry = 0; retry < 2; ++retry) {
                  check(cudaMemsetAsync(d_lc_count.p, 0, sizeof(unsigned int), G.compute_stream));
                  lightcone_detect_kernel<<<blocks(int(G.ns)), team, 0, G.compute_stream>>>(
                      G.d_x.p, G.d_v.p, G.d_orig_idx.p,
                      mass_uniform ? nullptr : G.d_mass.p, first_mass, int(G.ns),
                      step_drift_fac, a_prev, a_step_end, chi_0, chi_1, lc_r_max,
                      cfg.lc_obs, d_lc_shifts.p, int(shifts.size()), 0,
                      d_lc_hits.p, d_lc_count.p, unsigned(lc_hit_cap));
                  check(cudaMemcpyAsync(&hit_cnt, d_lc_count.p, sizeof(unsigned int),
                                        cudaMemcpyDeviceToHost, G.compute_stream));
                  check(cudaStreamSynchronize(G.compute_stream));
                  if (size_t(hit_cnt) <= lc_hit_cap) break;
                  lc_hit_cap = std::max(size_t(hit_cnt) + 65536ULL, lc_hit_cap * 2ULL);
                  d_lc_hits.alloc(lc_hit_cap, aux_dev_bytes);
                }
                if (size_t(hit_cnt) > lc_hit_cap) throw std::runtime_error("Lightcone hit buffer overflow");
                if (hit_cnt > 0) {
                  auto [slot, lc_ptr] = lc_writer.acquire_slot(hit_cnt);
                  check(cudaMemcpyAsync(lc_ptr, d_lc_hits.p, size_t(hit_cnt) * sizeof(LightconeHit),
                                        cudaMemcpyDeviceToHost, G.compute_stream));
                  check(cudaStreamSynchronize(G.compute_stream));
                  lc_writer.submit_slot(slot, hit_cnt);
                  lc_total_hits += uint64_t(hit_cnt);
                }
              }
            }
          }
        }

        auto scan_local_gpu = [&](bool track_mv) {
          if (G.ns == 0) {
            gpu_active[g] = 0;
            gpu_inactive_min[g] = timeline;
            gpu_v2_sum[g] = 0.0;
            gpu_max_v[g] = 0.f;
            return;
          }
          const int g_blk = std::min(scan_grid_max, std::max(1, int((G.ns + scan_team - 1) / scan_team)));
          scan_slab_active_v2_kernel<<<g_blk, scan_team, 0, G.compute_stream>>>(
              G.d_bins.p, G.d_v.p, int(G.ns), ti, (ti == 0 || is_final || is_pm_step),
              need_vel_sum, track_mv, G.d_scan_tmp.p);
          check(cudaMemcpyAsync(G.h_scan.data(), G.d_scan_tmp.p,
                                size_t(g_blk) * sizeof(ScanPartial), cudaMemcpyDeviceToHost, G.compute_stream));
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
          gpu_active[g] = act;
          gpu_inactive_min[g] = nx;
          gpu_v2_sum[g] = v2;
          gpu_max_v[g] = mv;
        };

        auto relocate_by_device_owner = [&](const int* d_owner_map) {
          h_mig_out[g].clear();
          if (G.ns > 0) {
            flag_plan_migrants_3d_kernel<<<blocks(int(G.ns)), team, 0, G.compute_stream>>>(
                G.d_x.p, d_owner_map, G.d_flags.p, int(G.ns), g, inv_cell_w);
            cub::CountingInputIterator<int> seq(0);
            size_t selb = G.select_bytes;
            check(cub::DeviceSelect::Flagged(
                G.d_select_storage.p, selb, seq, G.d_flags.p,
                G.d_active_ids.p, G.d_active_count.p, int(G.ns), G.compute_stream));
            int m_cnt = 0;
            check(cudaMemcpyAsync(&m_cnt, G.d_active_count.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
            check(cudaStreamSynchronize(G.compute_stream));
            if (m_cnt > 0) {
              G.ensure_mig_out(size_t(m_cnt));
              pack_migrants_3d_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                  G.d_x.p, G.d_v.p, G.d_mesh_acc.p, G.d_old_acc.p, G.d_bins.p,
                  G.d_orig_idx.p, mass_uniform ? nullptr : G.d_mass.p, first_mass,
                  soft_uniform ? nullptr : G.d_soft.p, soft,
                  G.d_active_ids.p, m_cnt, inv_cell_w, d_owner_map, G.d_mig_out.p);
              h_mig_out[g].resize(size_t(m_cnt));
              check(cudaMemcpyAsync(h_mig_out[g].data(), G.d_mig_out.p,
                                    size_t(m_cnt) * sizeof(DevMigrant), cudaMemcpyDeviceToHost, G.compute_stream));
              const int keep_ns = int(G.ns) - m_cnt;
              G.ensure_tail_cap(size_t(m_cnt));
              mark_tail_survivors_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                  G.d_flags.p, keep_ns, m_cnt, G.d_tail_flags.p);
              cub::CountingInputIterator<int> tail_seq(keep_ns);
              size_t selb_t = G.select_bytes;
              check(cub::DeviceSelect::Flagged(
                  G.d_select_storage.p, selb_t, tail_seq, G.d_tail_flags.p,
                  G.d_tail_keep.p, G.d_tail_count.p, m_cnt, G.compute_stream));
              compact_migrants_parallel_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                  G.d_x.p, G.d_v.p, G.d_mesh_acc.p, G.d_old_acc.p, G.d_bins.p,
                  G.d_orig_idx.p, mass_uniform ? nullptr : G.d_mass.p,
                  soft_uniform ? nullptr : G.d_soft.p,
                  G.d_active_ids.p, G.d_tail_keep.p, G.d_tail_count.p);
              G.ns -= size_t(m_cnt);
              check(cudaStreamSynchronize(G.compute_stream));
            }
          }
#pragma omp barrier
          if (g == 0) {
            for (int d = 0; d < num_gpus; ++d) h_mig_in[d].clear();
            for (int s = 0; s < num_gpus; ++s) {
              for (const auto& m : h_mig_out[s]) h_mig_in[m.dst_slab].push_back(m);
            }
          }
#pragma omp barrier
          const auto& in_vec = h_mig_in[g];
          const size_t in_cnt = in_vec.size();
          if (in_cnt > 0) {
            if (G.ns + in_cnt > G.part_cap) grow_resident_3d(G, G.ns + in_cnt);
            G.ensure_mig_in(in_cnt);
            check(cudaMemcpyAsync(G.d_mig_in.p, in_vec.data(),
                                  in_cnt * sizeof(DevMigrant), cudaMemcpyHostToDevice, G.compute_stream));
            append_migrants_kernel<<<blocks(int(in_cnt)), team, 0, G.compute_stream>>>(
                G.d_x.p, G.d_v.p, G.d_mesh_acc.p, G.d_old_acc.p, G.d_bins.p,
                G.d_orig_idx.p, mass_uniform ? nullptr : G.d_mass.p,
                soft_uniform ? nullptr : G.d_soft.p,
                G.d_mig_in.p, int(G.ns), int(in_cnt));
            G.ns += in_cnt;
            check(cudaStreamSynchronize(G.compute_stream));
          }
        };

        if (ti > 0) {
          if (eligible_lazy) {
            scan_local_gpu(true);
#pragma omp barrier
            if (g == 0) {
              uint64_t tot_act = 0;
              float gmv = 0.f;
              for (int gg = 0; gg < num_gpus; ++gg) {
                tot_act += gpu_active[gg];
                gmv = std::max(gmv, gpu_max_v[gg]);
              }
              step_max_disp = drift_fac * double(gmv);
              do_lazy_step = (tot_act < uint64_t(double(N) * cfg.lazy_active_frac)) &&
                             ((cum_max_disp + step_max_disp) < max_safe_disp);
              if (!do_lazy_step && lazy_steps_since_rebuild > 0) force_rebuild_all = true;
            }
#pragma omp barrier
          }

          if (do_lazy_step) {
            h_mig_out[g].clear();
            if (G.ns > 0) {
              drift_only_kernel<<<blocks(int(G.ns)), team, 0, G.compute_stream>>>(
                  G.d_x.p, G.d_v.p, int(G.ns), drift_fac, h.box, true);
              check(cudaStreamSynchronize(G.compute_stream));
            }
          } else {
            h_mig_out[g].clear();
            if (G.ns > 0) {
              drift_flag_migrants_3d_kernel<<<blocks(int(G.ns)), team, 0, G.compute_stream>>>(
                  G.d_x.p, G.d_v.p, G.d_cell_owner.p, G.d_flags.p,
                  int(G.ns), g, drift_fac, h.box, inv_cell_w);
              cub::CountingInputIterator<int> seq(0);
              size_t selb = G.select_bytes;
              check(cub::DeviceSelect::Flagged(
                  G.d_select_storage.p, selb, seq, G.d_flags.p,
                  G.d_active_ids.p, G.d_active_count.p, int(G.ns), G.compute_stream));
              int m_cnt = 0;
              check(cudaMemcpyAsync(&m_cnt, G.d_active_count.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
              check(cudaStreamSynchronize(G.compute_stream));
              if (m_cnt > 0) {
                G.ensure_mig_out(size_t(m_cnt));
                pack_migrants_3d_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                    G.d_x.p, G.d_v.p, G.d_mesh_acc.p, G.d_old_acc.p, G.d_bins.p,
                    G.d_orig_idx.p, mass_uniform ? nullptr : G.d_mass.p, first_mass,
                    soft_uniform ? nullptr : G.d_soft.p, soft,
                    G.d_active_ids.p, m_cnt, inv_cell_w, G.d_cell_owner.p, G.d_mig_out.p);
                h_mig_out[g].resize(size_t(m_cnt));
                check(cudaMemcpyAsync(h_mig_out[g].data(), G.d_mig_out.p,
                                      size_t(m_cnt) * sizeof(DevMigrant), cudaMemcpyDeviceToHost, G.compute_stream));
                const int keep_ns = int(G.ns) - m_cnt;
                G.ensure_tail_cap(size_t(m_cnt));
                mark_tail_survivors_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                    G.d_flags.p, keep_ns, m_cnt, G.d_tail_flags.p);
                cub::CountingInputIterator<int> tail_seq(keep_ns);
                size_t selb_t = G.select_bytes;
                check(cub::DeviceSelect::Flagged(
                    G.d_select_storage.p, selb_t, tail_seq, G.d_tail_flags.p,
                    G.d_tail_keep.p, G.d_tail_count.p, m_cnt, G.compute_stream));
                compact_migrants_parallel_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                    G.d_x.p, G.d_v.p, G.d_mesh_acc.p, G.d_old_acc.p, G.d_bins.p,
                    G.d_orig_idx.p, mass_uniform ? nullptr : G.d_mass.p,
                    soft_uniform ? nullptr : G.d_soft.p,
                    G.d_active_ids.p, G.d_tail_keep.p, G.d_tail_count.p);
                G.ns -= size_t(m_cnt);
                check(cudaStreamSynchronize(G.compute_stream));
              }
            }
#pragma omp barrier
            if (g == 0) {
              for (int d = 0; d < num_gpus; ++d) h_mig_in[d].clear();
              for (int s = 0; s < num_gpus; ++s) {
                total_migrants += h_mig_out[s].size();
                for (const auto& m : h_mig_out[s]) h_mig_in[m.dst_slab].push_back(m);
              }
              if (total_migrants > 0) any_ns_changed = true;
              if (lazy_steps_since_rebuild > 0) force_rebuild_all = true;
            }
#pragma omp barrier
            const auto& in_vec = h_mig_in[g];
            const size_t in_cnt = in_vec.size();
            if (in_cnt > 0) {
              if (G.ns + in_cnt > G.part_cap) grow_resident_3d(G, G.ns + in_cnt);
              G.ensure_mig_in(in_cnt);
              check(cudaMemcpyAsync(G.d_mig_in.p, in_vec.data(),
                                    in_cnt * sizeof(DevMigrant), cudaMemcpyHostToDevice, G.compute_stream));
              append_migrants_kernel<<<blocks(int(in_cnt)), team, 0, G.compute_stream>>>(
                  G.d_x.p, G.d_v.p, G.d_mesh_acc.p, G.d_old_acc.p, G.d_bins.p,
                  G.d_orig_idx.p, mass_uniform ? nullptr : G.d_mass.p,
                  soft_uniform ? nullptr : G.d_soft.p,
                  G.d_mig_in.p, int(G.ns), int(in_cnt));
              G.ns += in_cnt;
              check(cudaStreamSynchronize(G.compute_stream));
            }
            if (!eligible_lazy || total_migrants > 0) scan_local_gpu(false);
          }
        } else {
          scan_local_gpu(false);
        }

#pragma omp barrier
        // Periodic Dynamic Rebalance check after particle drift
        if (check_rebal_this_step) {
          check(cudaMemsetAsync(G.d_cell_counts.p, 0, N_TOP_CELLS * sizeof(unsigned long long), G.compute_stream));
          check(cudaMemsetAsync(G.d_cell_act_counts.p, 0, N_TOP_CELLS * sizeof(unsigned long long), G.compute_stream));
          if (G.ns > 0) {
            const int c_grid = std::min(512, blocks(int(G.ns)));
            count_cells_3d_kernel<<<c_grid, team, 0, G.compute_stream>>>(
                G.d_x.p, G.d_bins.p, int(G.ns), inv_cell_w, ti, is_final,
                G.d_cell_counts.p, G.d_cell_act_counts.p);
          }
          check(cudaMemcpyAsync(gpu_cell_cnt[g].data(), G.d_cell_counts.p,
                                N_TOP_CELLS * sizeof(unsigned long long), cudaMemcpyDeviceToHost, G.compute_stream));
          check(cudaMemcpyAsync(gpu_cell_act[g].data(), G.d_cell_act_counts.p,
                                N_TOP_CELLS * sizeof(unsigned long long), cudaMemcpyDeviceToHost, G.compute_stream));
          check(cudaStreamSynchronize(G.compute_stream));
#pragma omp barrier
          if (g == 0) {
            std::array<uint64_t, N_TOP_CELLS> cur_cnt{};
            std::array<double, N_TOP_CELLS> cur_w{};
            const int win = std::max(1, steps_since_rebal + 1);
            for (int cid = 0; cid < N_TOP_CELLS; ++cid) {
              unsigned long long sc = 0, sa = 0;
              for (int gg = 0; gg < num_gpus; ++gg) {
                sc += gpu_cell_cnt[gg][cid];
                sa += gpu_cell_act[gg][cid];
              }
              cur_cnt[cid] = uint64_t(sc);
              cell_act_accum[cid] += sa;
              cur_w[cid] = double(sc) + 0.5 * double(cell_act_accum[cid]) / double(win);
            }
            cell_act_accum.fill(0ULL);
            steps_since_rebal = 0;
            const double cur_imb = detail::compute_load_imbalance(cur_w, plan.cell_owner, num_gpus);
            Domain3DPlan cand_plan = build_domain_3d_plan(
                cur_cnt, cur_w, num_gpus, cfg.domain_mode, pm_side, &cfg.can_peer);
            const double new_imb = cand_plan.stats.w_imb;
            const double gain = (cur_imb - new_imb) / std::max(cur_imb, 1e-12);
            if (gain >= cfg.rebalance_min_gain) {
              did_rebalance = true;
              plan = std::move(cand_plan);
              do_lazy_step = false;
              force_rebuild_all = true;
              if (cfg.pm_dist_enabled) dist_pm.reconfigure_x_planes(plan.gpu_x_planes);
              ++rebalance_count;
              std::cout << std::setprecision(6)
                        << "REBALANCE {\"step\":" << global_iteration
                        << ",\"a\":" << a
                        << ",\"mode\":\"" << domain_mode_name(cfg.domain_mode)
                        << "\",\"cur_imb\":" << cur_imb
                        << ",\"new_imb\":" << new_imb
                        << ",\"gain\":" << gain
                        << ",\"ghost_total\":" << plan.stats.ghost_total
                        << ",\"max_x_slabs\":" << plan.stats.max_x_slabs << "}" << std::endl;
            }
          }
#pragma omp barrier
          if (did_rebalance) {
            check(cudaSetDevice(G.dev));
            DevPtr<int> d_new_owner;
            d_new_owner.alloc(N_TOP_CELLS, G.device_bytes);
            check(cudaMemcpyAsync(d_new_owner.p, plan.cell_owner.data(),
                                  N_TOP_CELLS * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
            relocate_by_device_owner(d_new_owner.p);
            d_new_owner.free(G.device_bytes);
            upload_plan_to_gpu(G, g, plan);
            scan_local_gpu(false);
#pragma omp barrier
            if (g == 0) {
              setup_all_edges(plan);
              check(cudaSetDevice(0));
            }
#pragma omp barrier
            check(cudaSetDevice(G.dev));
          }
        } else if (g == 0 && ti > 0) {
          ++steps_since_rebal;
        }

        if (g == 0) {
          if (do_lazy_step) {
            cum_max_disp += step_max_disp;
            ++lazy_steps_since_rebuild;
            ++total_lazy_steps;
          } else {
            cum_max_disp = 0.0;
            lazy_steps_since_rebuild = 0;
            ++total_full_rebuilds;
          }
          for (int gg = 0; gg < num_gpus; ++gg) {
            peak_gpu_seen = std::max(peak_gpu_seen, gpus[gg].ns);
          }
          drift_ms = (wall() - t_a0) * 1000.0;

          if (is_pm_step) {
            ++pm_updates;
            if (is_final) {
              pm_kick = p.factors(timeline - current_pm_span / 2, timeline, false);
            } else {
              if (cfg.dynamic_pm) {
                double vel_norm_sum = 0.0;
                for (int gg = 0; gg < num_gpus; ++gg) vel_norm_sum += gpu_v2_sum[gg];
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
          }

          if (do_fc_now && gpus[0].ns > 0) {
            n_fc_tgt = std::min(fc_target_cap, int(gpus[0].ns));
            h_fc_lids.resize(n_fc_tgt);
            const int stride = std::max(1, int(gpus[0].ns) / n_fc_tgt);
            for (int k = 0; k < n_fc_tgt; ++k) h_fc_lids[k] = k * stride;
            check(cudaMemcpyAsync(d_fc_lids.p, h_fc_lids.data(), size_t(n_fc_tgt) * sizeof(int),
                                  cudaMemcpyHostToDevice, gpus[0].compute_stream));
          }
        }
#pragma omp barrier

        // ================= STEP B: Distributed PM =================
        if (is_pm_step) {
          const double t_pm0 = wall();
          if (cfg.pm_dist_enabled) {
            dist_pm.begin_deposit(g);
            if (G.ns > 0) {
              dist_pm.deposit(g, G.d_x.p, mass_uniform ? nullptr : G.d_mass.p, first_mass, G.ns, h.box);
            }
            check(cudaStreamSynchronize(G.compute_stream));
#pragma omp barrier
            if (g == 0) dist_pm.solve_and_gather(h.box, r_s);
#pragma omp barrier
            if (G.ns > 0) {
              dist_pm.interpolate(g, G.d_x.p, G.ns, h.box, p.G, G.d_mesh_acc.p, nullptr);
            }
            check(cudaStreamSynchronize(G.compute_stream));
#pragma omp barrier
          } else {
            check(cudaMemsetAsync(G.d_pm_density.p, 0, pm_points * sizeof(double), G.compute_stream));
            if (G.ns > 0) {
              swift_pm_cuda_deposit_buf(
                  G.d_x.p, mass_uniform ? nullptr : G.d_mass.p, first_mass,
                  G.ns, G.d_pm_density.p, pm_side, h.box, G.compute_stream);
            }
            swift_pm_cuda_solve_on_device(G.d_pm_density.p, G.d_pm_field.p, pm_side, h.box, r_s, G.compute_stream);
            if (G.ns > 0) {
              swift_pm_cuda_interpolate_buf(
                  G.d_x.p, G.ns, G.d_pm_field.p, pm_side, h.box, p.G, G.d_mesh_acc.p, nullptr, G.compute_stream);
            }
            check(cudaStreamSynchronize(G.compute_stream));
          }
          if (g == 0) pm_ms = (wall() - t_pm0) * 1000.0;
        }

        // ================= STEP C: Local Tree Build / Drift =================
        const double t_c0 = wall();
        if (do_lazy_step) {
          if (G.ns > 0) {
            drift_tree_nodes_kernel<<<blocks(int(G.nodes)), team, 0, G.compute_stream>>>(
                G.force_nodes.p, G.node_vel_max.p, G.node_meta.p,
                int(G.nodes), float(drift_fac), float(h.box));
            if (gpu_active[g] > 0) {
              gather_sources_slab<<<blocks(int(G.ns)), team, 0, G.compute_stream>>>(
                  G.d_x.p, mass_uniform ? nullptr : G.d_mass.p, first_mass,
                  nullptr, G.order.p, G.packed_sources.p, nullptr, int(G.ns));
              check(cudaMemsetAsync(G.d_active_bits.p, 0, 54 * sizeof(uint32_t), G.compute_stream));
              mark_active_home_cells_3d_kernel<<<blocks(int(G.ns)), team, 0, G.compute_stream>>>(
                  G.home_slot.p, G.order.p, G.d_bins.p, G.d_slot_to_cell.p,
                  int(G.ns), ti, G.d_active_bits.p);
              check(cudaMemcpyAsync(gpu_act_bits[g].data(), G.d_active_bits.p,
                                    54 * sizeof(uint32_t), cudaMemcpyDeviceToHost, G.compute_stream));
            } else {
              gpu_act_bits[g].fill(0u);
            }
          } else {
            gpu_act_bits[g].fill(0u);
          }
          gpu_builds[g] = 0;
          gpu_nodes[g] = G.nodes;
        } else {
          build_local_tree_3d(G, cfg.lazy_tree_enabled && !is_final && !(is_pm_step || gpu_active[g] > 0));
          gpu_builds[g] = 1;
          gpu_nodes[g] = G.nodes;
        }
        check(cudaStreamSynchronize(G.compute_stream));
        gpu_tree_ms[g] = (wall() - t_c0) * 1000.0;
#pragma omp barrier
        if (g == 0) {
          if (!do_lazy_step) { all_trees_built = true; }
          if (do_lazy_step) {
            h_cell_active.fill(0);
            for (int gg = 0; gg < num_gpus; ++gg) {
              for (int w = 0; w < 54; ++w) {
                uint32_t m = gpu_act_bits[gg][w];
                while (m) {
                  const int b = __builtin_ctz(m);
                  h_cell_active[(w << 5) + b] = 1;
                  m &= (m - 1u);
                }
              }
            }
          }
          double mx_t = 0.0;
          for (int gg = 0; gg < num_gpus; ++gg) mx_t = std::max(mx_t, gpu_tree_ms[gg]);
          tree_ms = mx_t;
        }
#pragma omp barrier

        // ================= STEP C2: Persistent 3D Ghost Rebuild or Selective Lazy Refresh (Full-Duplex Overlapped) =================
        if (num_gpus > 1) {
          if (do_lazy_step) {
            // Sub-phase 1: Sender packs active-adjacent exported subtrees & starts chunked D2H (NO stream sync before barrier)
            for (int hop = 1; hop < num_gpus; ++hop) {
              const int dst = (g + hop) % num_gpus;
              auto& ed = edges[g][dst];
              ed.h_lazy_descs.clear();
              ed.lazy_M = 0;
              ed.lazy_tot_nodes = 0;
              ed.lazy_tot_parts = 0;
              ed.n_chunks = 0;
              ed.chunk_sz = 0;
              if (ed.E == 0) continue;
              const auto& dst_gslots = plan.export_dst_ghost_slots[g][dst];
              int n_acc = 0, p_acc = 0;
              for (int e = 0; e < ed.E; ++e) {
                const int cid = plan.export_cells[g][dst][e];
                // Need exported cell cid iff ANY owned cell on dst is 26-adjacent AND active
                bool need_e = false;
                int tx, ty, tz;
                cell_coords(cid, tx, ty, tz);
                for (int dx = -1; dx <= 1 && !need_e; ++dx) {
                  const int nx = wrap12(tx + dx);
                  for (int dy = -1; dy <= 1 && !need_e; ++dy) {
                    const int ny = wrap12(ty + dy);
                    for (int dz = -1; dz <= 1; ++dz) {
                      const int nz = wrap12(tz + dz);
                      const int ncid = cell_id(nx, ny, nz);
                      if (plan.cell_owner[ncid] == dst && h_cell_active[ncid]) {
                        need_e = true; break;
                      }
                    }
                  }
                }
                if (!need_e) continue;
                const int n_cnt = ed.h_node_cnt[e];
                const int p_cnt = ed.h_part_cnt[e];
                Dom3DLazyRootDesc d{};
                d.e = e;
                d.src_part_start = ed.h_part_src_start[e];
                d.dst_gslot = dst_gslots[e];
                d.src_nonroot_off = ed.h_nonroot_off[e];
                d.dst_desc_base = ed.dst_desc_base + ed.h_nonroot_off[e];
                d.n_cnt = n_cnt;
                d.p_cnt = p_cnt;
                d.lazy_node_off = n_acc;
                d.lazy_part_off = p_acc;
                d.dst_part_off = ed.dst_part_base + ed.h_part_off[e];
                ed.h_lazy_descs.push_back(d);
                n_acc += n_cnt;
                p_acc += p_cnt;
              }
              ed.lazy_M = int(ed.h_lazy_descs.size());
              ed.lazy_tot_nodes = n_acc;
              ed.lazy_tot_parts = p_acc;
              if (ed.lazy_M == 0) continue;

              const size_t total_f4 = size_t(n_acc) + size_t(p_acc);
              const size_t ld_bytes = size_t(ed.lazy_M) * sizeof(Dom3DLazyRootDesc);
              if (size_t(ed.lazy_M) > dev_len(ed.d_lazy_descs_src)) {
                ed.d_lazy_descs_src.alloc(size_t(ed.lazy_M), G.device_bytes);
              }
              if (total_f4 > dev_len(ed.d_lazy_send)) {
                ed.d_lazy_send.alloc(std::max(total_f4, total_f4 * 3 / 2 + 4096), G.device_bytes);
              }
              if (ed.h_lazy_descs_pinned.bytes < ld_bytes) {
                alloc_pinned_slack(ed.h_lazy_descs_pinned, ld_bytes);
              }
              std::memcpy(ed.h_lazy_descs_pinned.p, ed.h_lazy_descs.data(), ld_bytes);
              check(cudaMemcpyAsync(ed.d_lazy_descs_src.p, ed.h_lazy_descs_pinned.p,
                                    ld_bytes, cudaMemcpyHostToDevice, G.compute_stream));
              pack_lazy_edge_kernel<<<blocks_warps(ed.lazy_M * 4), team, 0, G.compute_stream>>>(
                  G.node_meta.p, ed.d_bnd_src_node_map.p,
                  G.d_x.p, mass_uniform ? nullptr : G.d_mass.p, first_mass, G.order.p,
                  ed.d_lazy_descs_src.p, ed.lazy_M, ed.E, n_acc, ed.d_lazy_send.p);
              check(cudaEventRecord(ed.ev_pack_done, G.compute_stream));
              if (!cfg.can_peer[dst][g]) {
                const size_t bytes = total_f4 * sizeof(float4);
                alloc_pinned_slack(ed.h_bounce, bytes);
                check(cudaStreamWaitEvent(G.d2h_stream, ed.ev_pack_done, 0));
                int nc_c = std::min<int>(Dom3DEdgeState::MAX_CHUNKS,
                    std::max<int>(1, int((bytes + Dom3DEdgeState::CHUNK_BYTES - 1) / Dom3DEdgeState::CHUNK_BYTES)));
                size_t csz = ((bytes + size_t(nc_c) - 1) / size_t(nc_c) + 255ULL) & ~255ULL;
                csz = std::max<size_t>(csz, 256ULL);
                nc_c = std::max<int>(1, int((bytes + csz - 1) / csz));
                ed.n_chunks = nc_c;
                ed.chunk_sz = csz;
                const char* src_bytes = reinterpret_cast<const char*>(ed.d_lazy_send.p);
                char* hb = static_cast<char*>(ed.h_bounce.p);
                for (int c = 0; c < nc_c; ++c) {
                  const size_t off = size_t(c) * csz;
                  const size_t len = std::min(csz, bytes - off);
                  check(cudaMemcpyAsync(hb + off, src_bytes + off, len, cudaMemcpyDeviceToHost, G.d2h_stream));
                  check(cudaEventRecord(ed.ev_d2h_chunk[c], G.d2h_stream));
                }
              }
            }
#pragma omp barrier
            // Sub-phase 2: Receiver enqueues chunked H2D / Peer + unpack on copy_stream (NO stream sync)
            check(cudaEventRecord(G.ev_ghost_start, G.copy_stream));
            for (int hop = 1; hop < num_gpus; ++hop) {
              const int src = (g + num_gpus - hop) % num_gpus;
              auto& ed = edges[src][g];
              if (ed.lazy_M == 0) continue;
              const size_t total_f4 = size_t(ed.lazy_tot_nodes) + size_t(ed.lazy_tot_parts);
              const size_t bytes = total_f4 * sizeof(float4);
              const size_t ld_bytes = size_t(ed.lazy_M) * sizeof(Dom3DLazyRootDesc);
              if (size_t(ed.lazy_M) > dev_len(ed.d_lazy_descs_dst)) {
                ed.d_lazy_descs_dst.alloc(size_t(ed.lazy_M), G.device_bytes);
              }
              if (total_f4 > dev_len(ed.d_lazy_recv)) {
                ed.d_lazy_recv.alloc(std::max(total_f4, total_f4 * 3 / 2 + 4096), G.device_bytes);
              }
              check(cudaMemcpyAsync(ed.d_lazy_descs_dst.p, ed.h_lazy_descs_pinned.p,
                                    ld_bytes, cudaMemcpyHostToDevice, G.copy_stream));
              if (cfg.can_peer[g][src]) {
                check(cudaStreamWaitEvent(G.copy_stream, ed.ev_pack_done, 0));
                check(cudaMemcpyPeerAsync(ed.d_lazy_recv.p, G.dev, ed.d_lazy_send.p, gpus[src].dev,
                                          bytes, G.copy_stream));
              } else {
                char* dst_bytes = reinterpret_cast<char*>(ed.d_lazy_recv.p);
                const char* hb = static_cast<const char*>(ed.h_bounce.p);
                for (int c = 0; c < ed.n_chunks; ++c) {
                  const size_t off = size_t(c) * ed.chunk_sz;
                  const size_t len = std::min(ed.chunk_sz, bytes - off);
                  check(cudaStreamWaitEvent(G.copy_stream, ed.ev_d2h_chunk[c], 0));
                  check(cudaMemcpyAsync(dst_bytes + off, hb + off, len, cudaMemcpyHostToDevice, G.copy_stream));
                }
              }
              unpack_lazy_edge_kernel<<<blocks_warps(ed.lazy_M * 4), team, 0, G.copy_stream>>>(
                  ed.d_lazy_recv.p, ed.d_lazy_descs_dst.p, ed.lazy_M, G.n_ghost,
                  ed.lazy_tot_nodes, G.gh_node_meta.p, G.gh_packed_sources.p);
            }
            check(cudaEventRecord(G.ev_ghost_ready, G.copy_stream));
          } else {
            // Full Ghost Rebuild Sub-phase 1: Parallel per-slot count once per GPU, then ring export pack + chunked D2H
            int* h_slot_desc = static_cast<int*>(G.h_slot_meta.p);
            int* h_slot_pstart = h_slot_desc + N_TOP_CELLS;
            int* h_slot_pcnt = h_slot_pstart + N_TOP_CELLS;
            if (G.n_own > 0) {
              check(cudaMemsetAsync(G.d_own_slot_desc_cnt.p, 0, size_t(G.n_own) * sizeof(int), G.compute_stream));
              if (G.nodes > 0) {
                const int c_grid = std::min(512, blocks(int(G.nodes)));
                count_own_slot_nodes_3d_kernel<<<c_grid, team, 0, G.compute_stream>>>(
                    G.geometry.p, G.n_own, int(G.nodes),
                    G.d_own_slot_desc_cnt.p, G.d_own_slot_pstart.p, G.d_own_slot_pcnt.p);
              } else {
                check(cudaMemsetAsync(G.d_own_slot_pstart.p, 0, size_t(G.n_own) * sizeof(int), G.compute_stream));
                check(cudaMemsetAsync(G.d_own_slot_pcnt.p, 0, size_t(G.n_own) * sizeof(int), G.compute_stream));
              }
              check(cudaMemcpyAsync(h_slot_desc, G.d_own_slot_desc_cnt.p,
                                    size_t(G.n_own) * sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
              check(cudaMemcpyAsync(h_slot_pstart, G.d_own_slot_pstart.p,
                                    size_t(G.n_own) * sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
              check(cudaMemcpyAsync(h_slot_pcnt, G.d_own_slot_pcnt.p,
                                    size_t(G.n_own) * sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
              check(cudaStreamSynchronize(G.compute_stream));
            }

            for (int hop = 1; hop < num_gpus; ++hop) {
              const int dst = (g + hop) % num_gpus;
              auto& ed = edges[g][dst];
              ed.n_chunks = 0;
              ed.chunk_sz = 0;
              if (ed.E == 0) continue;
              const auto& src_slots = plan.export_src_slots[g][dst];
              int n_acc = 0, nr_acc = 0, p_acc = 0;
              for (int e = 0; e < ed.E; ++e) {
                const int s = src_slots[e];
                const int ncnt = 1 + h_slot_desc[s];
                const int pcnt = h_slot_pcnt[s];
                ed.h_node_cnt[e] = ncnt;
                ed.h_part_cnt[e] = pcnt;
                ed.h_part_src_start[e] = h_slot_pstart[s];
                ed.h_node_off[e] = n_acc;
                ed.h_nonroot_off[e] = nr_acc;
                ed.h_part_off[e] = p_acc;
                n_acc += ncnt;
                nr_acc += (ncnt - 1);
                p_acc += pcnt;
              }
              ed.tot_nodes = n_acc;
              ed.tot_parts = p_acc;

              ed.ensure_meta_src(G.dev, ed.E, G.device_bytes);
              ed.ensure_bnd_nodes(G.dev, std::max(1, ed.tot_nodes), G.device_bytes);
              int* hm = static_cast<int*>(ed.h_meta_pinned.p);
              std::memcpy(hm, ed.h_nonroot_off.data(), size_t(ed.E) * sizeof(int));
              std::memcpy(hm + ed.E, ed.h_part_src_start.data(), size_t(ed.E) * sizeof(int));
              std::memcpy(hm + 2 * ed.E, ed.h_part_off.data(), size_t(ed.E) * sizeof(int));
              std::memcpy(hm + 3 * ed.E, ed.h_part_cnt.data(), size_t(ed.E) * sizeof(int));
              check(cudaMemcpyAsync(ed.d_exp_cursor.p, hm, size_t(ed.E) * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
              check(cudaMemcpyAsync(ed.d_exp_part_src_start.p, hm + ed.E, size_t(ed.E) * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
              check(cudaMemcpyAsync(ed.d_exp_part_off.p, hm + 2 * ed.E, size_t(ed.E) * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
              check(cudaMemcpyAsync(ed.d_exp_part_cnt.p, hm + 3 * ed.E, size_t(ed.E) * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));

              auto align256 = [](size_t v) { return (v + 255ULL) & ~255ULL; };
              size_t w_off = 0;
              const size_t off_g = w_off; w_off += align256(size_t(ed.tot_nodes) * sizeof(Geometry));
              const size_t off_m = w_off; w_off += align256(size_t(ed.tot_nodes) * sizeof(NodeMeta));
              const size_t off_c = w_off; w_off += align256(size_t(ed.tot_nodes) * sizeof(CompactMultipole));
              const size_t off_p = w_off; w_off += align256(size_t(std::max(1, ed.tot_parts)) * sizeof(float4));
              const size_t off_s = w_off;
              if (!soft_uniform) w_off += align256(size_t(std::max(1, ed.tot_parts)) * sizeof(float));
              ed.send_wire_bytes = w_off;
              if (w_off > ed.d_send_wire.bytes) {
                ed.d_send_wire.alloc(std::max(w_off, w_off * 3 / 2 + 4096), G.device_bytes);
              }
              const size_t max_lazy_f4 = size_t(ed.tot_nodes) + size_t(ed.tot_parts);
              if (max_lazy_f4 > dev_len(ed.d_lazy_send)) {
                ed.d_lazy_send.alloc(std::max(max_lazy_f4, max_lazy_f4 * 3 / 2 + 4096), G.device_bytes);
              }

              Geometry* w_geom = reinterpret_cast<Geometry*>(ed.d_send_wire.p + off_g);
              NodeMeta* w_meta = reinterpret_cast<NodeMeta*>(ed.d_send_wire.p + off_m);
              CompactMultipole* w_cm = reinterpret_cast<CompactMultipole*>(ed.d_send_wire.p + off_c);
              float4* w_pack = reinterpret_cast<float4*>(ed.d_send_wire.p + off_p);
              float* w_soft = soft_uniform ? nullptr : reinterpret_cast<float*>(ed.d_send_wire.p + off_s);

              if (G.nodes > 0) {
                assign_export_bidx_3d_kernel<<<blocks(int(G.nodes)), team, 0, G.compute_stream>>>(
                    G.geometry.p, ed.d_slot_to_e.p, G.n_own, int(G.nodes),
                    ed.E, ed.d_exp_cursor.p, G.d_u_to_b.p, ed.d_bnd_src_node_map.p);
              }
              if (ed.tot_nodes > 0) {
                gather_export_bundle_3d_kernel<<<blocks(ed.tot_nodes), team, 0, G.compute_stream>>>(
                    G.geometry.p, G.node_meta.p, G.compact_m.p, ed.d_slot_to_e.p,
                    ed.d_exp_part_src_start.p, ed.d_exp_part_off.p,
                    G.d_u_to_b.p, ed.d_bnd_src_node_map.p, ed.E, ed.tot_nodes,
                    w_geom, w_meta, w_cm);
              }
              if (ed.tot_parts > 0) {
                pack_export_particles_kernel<<<blocks_warps(ed.E * 4), team, 0, G.compute_stream>>>(
                    G.packed_sources.p, soft_uniform ? nullptr : G.packed_soft.p,
                    ed.d_exp_part_src_start.p, ed.d_exp_part_off.p, ed.d_exp_part_cnt.p,
                    ed.E, w_pack, w_soft);
              }
              check(cudaEventRecord(ed.ev_pack_done, G.compute_stream));
              if (!cfg.can_peer[dst][g]) {
                alloc_pinned_slack(ed.h_bounce, ed.send_wire_bytes);
                check(cudaStreamWaitEvent(G.d2h_stream, ed.ev_pack_done, 0));
                int nc_c = std::min<int>(Dom3DEdgeState::MAX_CHUNKS,
                    std::max<int>(1, int((ed.send_wire_bytes + Dom3DEdgeState::CHUNK_BYTES - 1) / Dom3DEdgeState::CHUNK_BYTES)));
                size_t csz = ((ed.send_wire_bytes + size_t(nc_c) - 1) / size_t(nc_c) + 255ULL) & ~255ULL;
                csz = std::max<size_t>(csz, 256ULL);
                nc_c = std::max<int>(1, int((ed.send_wire_bytes + csz - 1) / csz));
                ed.n_chunks = nc_c;
                ed.chunk_sz = csz;
                char* hb = static_cast<char*>(ed.h_bounce.p);
                for (int c = 0; c < nc_c; ++c) {
                  const size_t off = size_t(c) * csz;
                  const size_t len = std::min(csz, ed.send_wire_bytes - off);
                  check(cudaMemcpyAsync(hb + off, ed.d_send_wire.p + off, len, cudaMemcpyDeviceToHost, G.d2h_stream));
                  check(cudaEventRecord(ed.ev_d2h_chunk[c], G.d2h_stream));
                }
              }
            }
#pragma omp barrier
            // Full Ghost Rebuild Sub-phase 2: Pull peer bundles via chunked H2D / Peer on copy_stream (NO stream sync)
            int desc_cursor = 0, part_cursor = 0;
            for (int src = 0; src < num_gpus; ++src) {
              if (src == g) continue;
              auto& ed = edges[src][g];
              ed.dst_desc_base = desc_cursor;
              ed.dst_part_base = part_cursor;
              if (ed.E > 0) {
                desc_cursor += (ed.tot_nodes - ed.E);
                part_cursor += ed.tot_parts;
              }
            }
            const size_t tot_gh_nodes = size_t(G.n_ghost) + size_t(desc_cursor);
            const size_t tot_gh_parts = size_t(part_cursor);
            G.ensure_ghost_cap(tot_gh_parts, tot_gh_nodes, soft_uniform);
            G.gh_nodes = unsigned(tot_gh_nodes);

            check(cudaEventRecord(G.ev_ghost_start, G.copy_stream));
            for (int hop = 1; hop < num_gpus; ++hop) {
              const int src = (g + num_gpus - hop) % num_gpus;
              auto& ed = edges[src][g];
              if (ed.E == 0) continue;
              const size_t wbytes = ed.send_wire_bytes;
              if (wbytes > ed.d_recv_wire.bytes) {
                ed.d_recv_wire.alloc(std::max(wbytes, wbytes * 3 / 2 + 4096), G.device_bytes);
              }
              const size_t max_lazy_f4 = size_t(ed.tot_nodes) + size_t(ed.tot_parts);
              if (max_lazy_f4 > dev_len(ed.d_lazy_recv)) {
                ed.d_lazy_recv.alloc(std::max(max_lazy_f4, max_lazy_f4 * 3 / 2 + 4096), G.device_bytes);
              }
              if (cfg.can_peer[g][src]) {
                check(cudaStreamWaitEvent(G.copy_stream, ed.ev_pack_done, 0));
                check(cudaMemcpyPeerAsync(ed.d_recv_wire.p, G.dev, ed.d_send_wire.p, gpus[src].dev, wbytes, G.copy_stream));
              } else {
                const char* hb = static_cast<const char*>(ed.h_bounce.p);
                for (int c = 0; c < ed.n_chunks; ++c) {
                  const size_t off = size_t(c) * ed.chunk_sz;
                  const size_t len = std::min(ed.chunk_sz, wbytes - off);
                  check(cudaStreamWaitEvent(G.copy_stream, ed.ev_d2h_chunk[c], 0));
                  check(cudaMemcpyAsync(ed.d_recv_wire.p + off, hb + off, len, cudaMemcpyHostToDevice, G.copy_stream));
                }
              }
              auto align256 = [](size_t v) { return (v + 255ULL) & ~255ULL; };
              size_t w_off = 0;
              const size_t off_g = w_off; w_off += align256(size_t(ed.tot_nodes) * sizeof(Geometry));
              const size_t off_m = w_off; w_off += align256(size_t(ed.tot_nodes) * sizeof(NodeMeta));
              const size_t off_c = w_off; w_off += align256(size_t(ed.tot_nodes) * sizeof(CompactMultipole));
              const size_t off_p = w_off; w_off += align256(size_t(std::max(1, ed.tot_parts)) * sizeof(float4));
              const size_t off_s = w_off;
              const Geometry* r_geom = reinterpret_cast<const Geometry*>(ed.d_recv_wire.p + off_g);
              const NodeMeta* r_meta = reinterpret_cast<const NodeMeta*>(ed.d_recv_wire.p + off_m);
              const CompactMultipole* r_cm = reinterpret_cast<const CompactMultipole*>(ed.d_recv_wire.p + off_c);
              const float4* r_pack = reinterpret_cast<const float4*>(ed.d_recv_wire.p + off_p);
              const float* r_soft = soft_uniform ? nullptr : reinterpret_cast<const float*>(ed.d_recv_wire.p + off_s);

              install_ghost_bundle_kernel<<<blocks(ed.tot_nodes), team, 0, G.copy_stream>>>(
                  r_geom, r_meta, r_cm, ed.d_dst_ghost_slots.p, ed.E, ed.tot_nodes,
                  G.n_ghost, ed.dst_desc_base, ed.dst_part_base,
                  G.gh_geometry.p, G.gh_node_meta.p, G.gh_compact_m.p);
              if (ed.tot_parts > 0) {
                check(cudaMemcpyAsync(G.gh_packed_sources.p + ed.dst_part_base, r_pack,
                                      size_t(ed.tot_parts) * sizeof(float4), cudaMemcpyDeviceToDevice, G.copy_stream));
                if (!soft_uniform && r_soft) {
                  check(cudaMemcpyAsync(G.gh_packed_soft.p + ed.dst_part_base, r_soft,
                                        size_t(ed.tot_parts) * sizeof(float), cudaMemcpyDeviceToDevice, G.copy_stream));
                }
              }
            }
            check(cudaEventRecord(G.ev_ghost_ready, G.copy_stream));
          }
        }

        // ================= STEP D: Two-Pass Overlapped 2-Tree Force + Integrate =================
        build_kick_table<<<1, 64, 0, G.compute_stream>>>(G.d_kick_finish.p, G.d_kick_begin.p, p, ti);
        {
          const int ns = int(G.ns);
          const int* active_ptr = nullptr;
          int force_count = ns;
          if (ns > 0 && !is_pm_step && !is_final) {
            mark_active_slab<<<blocks(ns), team, 0, G.compute_stream>>>(
                G.order.p, G.d_bins.p, G.d_flags.p, ns, ti);
            cub::CountingInputIterator<int> seq(0);
            size_t selb = G.select_bytes;
            check(cub::DeviceSelect::Flagged(
                G.d_select_storage.p, selb, seq, G.d_flags.p,
                G.d_active_ids.p, G.d_active_count.p, ns, G.compute_stream));
            int h_act = 0;
            check(cudaMemcpyAsync(&h_act, G.d_active_count.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
            check(cudaStreamSynchronize(G.compute_stream));
            force_count = h_act;
            active_ptr = G.d_active_ids.p;
          }

          if (force_count == 0) {
            gpu_force_ms[g] = 0.0;
            gpu_kick_ms[g] = 0.0;
            gpu_active_next[g] = timeline;
            gpu_invalid[g] = 0;
            gpu_xfer_ms[g] = 0.0;
          } else {
            check(cudaMemsetAsync(G.d_tree_invalid.p, 0, sizeof(int), G.compute_stream));
            const SplitTravTreeView loc_tv = G.local_split_view();
            const SplitTravTreeView gh_tv = G.ghost_split_view();
            build_home_warp_slices_3d_kernel<<<1, 1024, 0, G.compute_stream>>>(
                G.home_slot.p, G.n_own, G.n_interior_own,
                active_ptr, active_ptr ? G.d_active_count.p : nullptr,
                force_count, G.d_tree_invalid.p, G.d_split_slices.p, G.d_split_total_warps.p);
            int h_warp_cnt[2]{};
            check(cudaMemcpyAsync(h_warp_cnt, G.d_split_total_warps.p, 2 * sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
            check(cudaStreamSynchronize(G.compute_stream));
            const int n_int_warps = h_warp_cnt[0];
            const int total_warps = h_warp_cnt[1];

            uint64_t loc_batches = 0, loc_retries = 0;
            auto run_warp_range = [&](int w_begin, int w_end) {
              int warp_off = w_begin;
              int cur_batch = cfg.split_batch_warps;
              while (warp_off < w_end) {
                const int n_warps = std::min(cur_batch, w_end - warp_off);
                check(cudaMemsetAsync(G.d_split_counts.p, 0, 5 * sizeof(int), G.compute_stream));
                if (cfg.warp_m2l) {
                  traverse_m2p_split_3d_kernel<true><<<blocks(n_warps * 32), team, 0, G.compute_stream>>>(
                      G.d_x.p, G.d_old_acc.p, soft, soft_uniform ? nullptr : G.d_soft.p,
                      loc_tv, gh_tv, G.d_slot_nb_roots.p, G.order.p,
                      G.d_split_slices.p, warp_off, n_warps,
                      G.d_split_meta.p, G.d_split_head_p2p.p, G.d_split_head_m2l.p,
                      G.d_split_p2p.p, int(G.split_p2p_cap),
                      G.d_split_m2l.p, int(G.split_m2l_cap),
                      G.d_split_p2p_warps.p, G.d_split_m2l_warps.p, G.d_split_counts.p,
                      G.d_force.p, ns, p, ti == 0, G.d_tree_invalid.p, active_ptr, false);
                } else {
                  traverse_m2p_split_3d_kernel<false><<<blocks(n_warps * 32), team, 0, G.compute_stream>>>(
                      G.d_x.p, G.d_old_acc.p, soft, soft_uniform ? nullptr : G.d_soft.p,
                      loc_tv, gh_tv, G.d_slot_nb_roots.p, G.order.p,
                      G.d_split_slices.p, warp_off, n_warps,
                      G.d_split_meta.p, G.d_split_head_p2p.p, G.d_split_head_m2l.p,
                      G.d_split_p2p.p, int(G.split_p2p_cap),
                      G.d_split_m2l.p, int(G.split_m2l_cap),
                      G.d_split_p2p_warps.p, G.d_split_m2l_warps.p, G.d_split_counts.p,
                      G.d_force.p, ns, p, ti == 0, G.d_tree_invalid.p, active_ptr, false);
                }
                int h_counts[5]{};
                check(cudaMemcpyAsync(h_counts, G.d_split_counts.p, 5 * sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
                check(cudaStreamSynchronize(G.compute_stream));
                if (h_counts[4] != 0) {
                  ++loc_retries;
                  if (G.split_chunk_scale < 4 && cur_batch == cfg.split_batch_warps) {
                    G.ensure_split_bufs(size_t(cfg.split_batch_warps), G.split_chunk_scale * 2);
                    continue;
                  }
                  if (cur_batch > 256) { cur_batch = std::max(256, cur_batch / 2); continue; }
                  throw std::runtime_error("3D split traversal chunk overflow at minimum batch");
                }
                if (h_counts[3] > 0) {
                  eval_m2l_split_3d_kernel<<<blocks(h_counts[3] * 32), team, 0, G.compute_stream>>>(
                      G.d_x.p, loc_tv, gh_tv, G.order.p,
                      G.d_split_meta.p, G.d_split_head_m2l.p, G.d_split_m2l.p,
                      G.d_split_m2l_warps.p, h_counts[3],
                      G.d_force.p, ns, p, G.d_tree_invalid.p, active_ptr, false);
                }
                if (h_counts[2] > 0) {
                  const auto launch_p2p = [&](auto fsp_t, auto us_t) {
                    eval_p2p_split_3d_kernel<decltype(fsp_t)::value, decltype(us_t)::value>
                        <<<blocks(h_counts[2] * 32), team, 0, G.compute_stream>>>(
                            G.d_x.p, soft, soft_uniform ? nullptr : G.d_soft.p,
                            G.packed_sources.p, soft_uniform ? nullptr : G.packed_soft.p, G.order.p,
                            G.gh_packed_sources.p, soft_uniform ? nullptr : G.gh_packed_soft.p,
                            G.d_split_meta.p, G.d_split_head_p2p.p, G.d_split_p2p.p,
                            G.d_split_p2p_warps.p, h_counts[2],
                            G.d_force.p, ns, p, G.d_tree_invalid.p, active_ptr, false);
                  };
                  if (cfg.fast_split) {
                    if (soft_uniform) launch_p2p(std::true_type{}, std::true_type{});
                    else launch_p2p(std::true_type{}, std::false_type{});
                  } else {
                    if (soft_uniform) launch_p2p(std::false_type{}, std::true_type{});
                    else launch_p2p(std::false_type{}, std::false_type{});
                  }
                }
                ++loc_batches;
                warp_off += n_warps;
                if (cur_batch < cfg.split_batch_warps) cur_batch = cfg.split_batch_warps;
              }
            };

            // Pass 1: Interior warps (local_tree only, overlapped with GPU copy_stream / d2h_stream)
            const bool ran_int = (n_int_warps > 0);
            if (ran_int) {
              check(cudaEventRecord(G.ev_int_start, G.compute_stream));
              run_warp_range(0, n_int_warps);
              check(cudaEventRecord(G.ev_int_end, G.compute_stream));
            }
            // Pass 2: Wait on ghost tree ready, then execute boundary warps
            const bool ran_bnd = (n_int_warps < total_warps);
            if (ran_bnd) {
              if (num_gpus > 1) {
                check(cudaStreamWaitEvent(G.compute_stream, G.ev_ghost_ready, 0));
              }
              check(cudaEventRecord(G.ev_bnd_start, G.compute_stream));
              run_warp_range(n_int_warps, total_warps);
              check(cudaEventRecord(G.ev_bnd_end, G.compute_stream));
            }
            if (loc_batches || loc_retries) {
#pragma omp critical(split_stats_3d)
              { split_batches_run += loc_batches; split_overflow_retries += loc_retries; }
            }
            check(cudaGetLastError());
            check(cudaStreamSynchronize(G.compute_stream));
            double f_ms = 0.0;
            if (ran_int) {
              float ms_i = 0.f;
              check(cudaEventElapsedTime(&ms_i, G.ev_int_start, G.ev_int_end));
              f_ms += double(ms_i);
            }
            if (ran_bnd) {
              float ms_b = 0.f;
              check(cudaEventElapsedTime(&ms_b, G.ev_bnd_start, G.ev_bnd_end));
              f_ms += double(ms_b);
            }
            gpu_force_ms[g] = f_ms;

            if (cfg.force0_out_path && global_iteration == 0) {
              std::vector<float4> hf(ns);
              std::vector<uint32_t> hidx(ns);
              check(cudaMemcpy(hf.data(), G.d_force.p, size_t(ns) * sizeof(float4), cudaMemcpyDeviceToHost));
              check(cudaMemcpy(hidx.data(), G.d_orig_idx.p, size_t(ns) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
              for (int k = 0; k < ns; ++k) out_force0_f4[hidx[k]] = hf[k];
            }

            if (cfg.energy_log && (is_pm_step || is_final)) {
              check(cudaMemsetAsync(d_energy_accum[g], 0, 10 * sizeof(double), G.compute_stream));
              if (ns > 0) {
                if (cfg.pm_dist_enabled) {
                  dist_pm.interpolate(g, G.d_x.p, size_t(ns), h.box, p.G, G.d_mesh_acc.p, G.d_pm_pot.p);
                } else {
                  swift_pm_cuda_interpolate_buf(
                      G.d_x.p, size_t(ns), G.d_pm_field.p, pm_side, h.box, p.G,
                      G.d_mesh_acc.p, G.d_pm_pot.p, G.compute_stream);
                }
                const double closing_pm_e =
                    (ti == 0) ? 0.0
                    : (is_pm_step ? (is_final ? pm_kick : p.factors(ti - prev_pm_span / 2, ti, false))
                                  : p.factors(ti - current_pm_span / 2, ti, false));
                const int eblocks = std::min(256, (ns + 255) / 256);
                reduce_energy_mom_kernel<<<eblocks, 256, 0, G.compute_stream>>>(
                    G.d_x.p, G.d_v.p, G.d_force.p, G.d_mesh_acc.p,
                    G.d_pm_pot.p, G.d_bins.p,
                    mass_uniform ? nullptr : G.d_mass.p, first_mass,
                    ns, ti, G.d_kick_finish.p, closing_pm_e,
                    double(p.G), potential_norm, cfg.comoving ? (1.0 / a) : 1.0,
                    d_energy_accum[g]);
                check(cudaGetLastError());
              }
              check(cudaMemcpyAsync(gpu_energy[g].data(), d_energy_accum[g], 10 * sizeof(double),
                                    cudaMemcpyDeviceToHost, G.compute_stream));
            }

            if (do_fc_now && g == 0 && n_fc_tgt > 0) {
              gather_fc_samples_kernel<<<blocks(n_fc_tgt), team, 0, G.compute_stream>>>(
                  G.d_x.p, soft_uniform ? nullptr : G.d_soft.p, soft,
                  G.d_force.p, G.d_mesh_acc.p,
                  d_fc_lids.p, n_fc_tgt, 0,
                  d_fc_tgt_x.p, d_fc_tgt_h.p, d_fc_gpu_short.p,
                  is_pm_step ? d_fc_gpu_mesh.p : nullptr);
            }

            const double tk0 = wall();
            init_meta<<<1, 1, 0, G.compute_stream>>>(G.d_stats.p, G.d_tree_invalid.p, G.nodes);
            float3* synced_v = nullptr;
            if (midpoint) check(cudaMalloc(&synced_v, size_t(ns) * sizeof(float3)));
            integrate_slab<<<blocks(force_count), team, 0, G.compute_stream>>>(
                G.d_v.p, G.d_old_acc.p, G.d_bins.p, G.d_force.p, G.d_mesh_acc.p,
                active_ptr, G.order.p, force_count, soft,
                soft_uniform ? nullptr : G.d_soft.p, p, ti,
                G.d_stats.p, G.d_kick_finish.p, G.d_kick_begin.p,
                is_pm_step ? pm_kick : 0.0, a, H_cur, dt_max_rms, short_span_cap,
                G.d_tree_invalid.p, synced_v, closing_pm_kick);
            Meta g_stats{};
            check(cudaMemcpyAsync(&g_stats, G.d_stats.p, sizeof(Meta), cudaMemcpyDeviceToHost, G.compute_stream));
            check(cudaStreamSynchronize(G.compute_stream));
            gpu_kick_ms[g] = (wall() - tk0) * 1000.0;
            gpu_active_next[g] = g_stats.next;
            gpu_invalid[g] = g_stats.invalid;
            gpu_xfer_ms[g] = 0.0;

            if (cfg.lazy_tree_enabled && !is_final && ns > 0) {
              const uint32_t* act_bits_ptr = do_lazy_step ? G.d_active_bits.p : nullptr;
              for (int depth = tree_levels; depth >= 0; --depth) {
                const int lcnt = G.h_levels[depth + 1] - G.h_levels[depth];
                if (lcnt <= 0) continue;
                compute_node_vel_active_3d_kernel<<<blocks(lcnt), team, 0, G.compute_stream>>>(
                    G.geometry.p, G.force_nodes.p, G.d_v.p,
                    mass_uniform ? nullptr : G.d_mass.p, first_mass,
                    G.order.p, depth, G.levels.p, act_bits_ptr, G.d_slot_to_cell.p, G.node_vel_max.p);
              }
            }

            if (is_final || midpoint) {
              const double te0 = wall();
              if (cfg.pm_dist_enabled) {
                dist_pm.interpolate(g, G.d_x.p, size_t(ns), h.box, p.G, G.d_mesh_acc.p, G.d_pm_pot.p);
              } else {
                swift_pm_cuda_interpolate_buf(
                    G.d_x.p, size_t(ns), G.d_pm_field.p, pm_side, h.box, p.G,
                    G.d_mesh_acc.p, G.d_pm_pot.p, G.compute_stream);
              }
              constexpr int exp_chunk = 1 << 20;
              const size_t chunk_n = size_t(std::min(ns, exp_chunk));
              const bool need_mesh = is_run_final && (cfg.force_comp_path || cfg.force3_out_path);
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
              const float3* v_src = midpoint ? synced_v : G.d_v.p;
              size_t loc_bad = 0;
              double loc_v2 = 0.0, loc_psum = 0.0;
              float loc_pmin = FLT_MAX, loc_pmax = -FLT_MAX;
              double d2h_ms_acc = 0.0;

              for (int base = 0; base < ns; base += exp_chunk) {
                const int cnt = std::min(exp_chunk, ns - base);
                const double tc0 = (base == 0) ? te0 : wall();
                check(cudaMemcpyAsync(hx, G.d_x.p + base, size_t(cnt) * sizeof(double3), cudaMemcpyDeviceToHost, G.compute_stream));
                check(cudaMemcpyAsync(hv, v_src + base, size_t(cnt) * sizeof(float3), cudaMemcpyDeviceToHost, G.compute_stream));
                check(cudaMemcpyAsync(hraw, G.d_force.p + base, size_t(cnt) * sizeof(float4), cudaMemcpyDeviceToHost, G.compute_stream));
                check(cudaMemcpyAsync(hpot, G.d_pm_pot.p + base, size_t(cnt) * sizeof(float), cudaMemcpyDeviceToHost, G.compute_stream));
                check(cudaMemcpyAsync(hidx, G.d_orig_idx.p + base, size_t(cnt) * sizeof(uint32_t), cudaMemcpyDeviceToHost, G.compute_stream));
                if (need_mesh) {
                  check(cudaMemcpyAsync(hmesh, G.d_mesh_acc.p + base, size_t(cnt) * sizeof(float3), cudaMemcpyDeviceToHost, G.compute_stream));
                }
                check(cudaStreamSynchronize(G.compute_stream));
                d2h_ms_acc += (wall() - tc0) * 1000.0;

                for (int k = 0; k < cnt; ++k) {
                  const float4 rf = hraw[k];
                  const float pp = hpot[k];
                  const float pot = float(double(rf.w) + potential_norm) * p.G + pp;
                  if (!std::isfinite(hx[k].x) || !std::isfinite(hx[k].y) || !std::isfinite(hx[k].z) ||
                      (cfg.periodic && (hx[k].x < 0.0 || hx[k].x >= h.box ||
                                        hx[k].y < 0.0 || hx[k].y >= h.box ||
                                        hx[k].z < 0.0 || hx[k].z >= h.box)) ||
                      !std::isfinite(hv[k].x) || !std::isfinite(hv[k].y) || !std::isfinite(hv[k].z) ||
                      !std::isfinite(pot)) {
                    ++loc_bad;
                  } else {
                    const double vvx = double(hv[k].x) * inv_a;
                    const double vvy = double(hv[k].y) * inv_a;
                    const double vvz = double(hv[k].z) * inv_a;
                    loc_v2 += vvx * vvx + vvy * vvy + vvz * vvz;
                    loc_psum += double(pot);
                    if (pot < loc_pmin) loc_pmin = pot;
                    if (pot > loc_pmax) loc_pmax = pot;
                  }
                  if (!cfg.skip_output_write) {
                    const double vvx = double(hv[k].x) * inv_a;
                    const double vvy = double(hv[k].y) * inv_a;
                    const double vvz = double(hv[k].z) * inv_a;
                    const uint32_t oi = hidx[k];
                    out_x[oi] = hx[k];
                    out_v[oi] = float3{float(vvx), float(vvy), float(vvz)};
                    out_pot[oi] = pot;
                  }
                  if (need_mesh) {
                    const uint32_t oi = hidx[k];
                    const float3 ma = hmesh[k];
                    if (cfg.force_comp_path) {
                      out_short_f4[oi] = float4{rf.x, rf.y, rf.z, rf.w};
                      out_long_f4[oi] = float4{ma.x, ma.y, ma.z, pp};
                    }
                    if (cfg.force3_out_path) {
                      out_short_f3[oi] = float3{rf.x, rf.y, rf.z};
                      out_long_f3[oi] = ma;
                    }
                  }
                }
              }
              if (synced_v) check(cudaFree(synced_v));
              gpu_xfer_ms[g] = d2h_ms_acc;
#pragma omp critical(sanity_3d)
              {
                sanity_bad += loc_bad;
                sanity_v2_sum += loc_v2;
                sanity_pot_sum += loc_psum;
                sanity_pot_min = std::min(sanity_pot_min, loc_pmin);
                sanity_pot_max = std::max(sanity_pot_max, loc_pmax);
              }
            }
          }
          if (num_gpus > 1) {
            check(cudaStreamSynchronize(G.copy_stream));
            check(cudaStreamSynchronize(G.d2h_stream));
            float gms = 0.f;
            check(cudaEventElapsedTime(&gms, G.ev_ghost_start, G.ev_ghost_ready));
            gpu_ghost_ms[g] = double(gms);
          } else {
            gpu_ghost_ms[g] = 0.0;
          }
        }
#pragma omp barrier
      } // end omp parallel
      ghost_installed = true;

      Tick step_next = timeline;
      uint64_t step_active = 0, step_nodes = 0;
      unsigned int step_invalid = 0;
      int step_builds = 0;
      double step_ghost_ms = 0.0, step_xfer_ms = 0.0, step_force_ms = 0.0, step_kick_ms = 0.0;
      for (int gg = 0; gg < num_gpus; ++gg) {
        step_force_ms = (num_gpus == 1) ? gpu_force_ms[gg] : std::max(step_force_ms, gpu_force_ms[gg]);
        step_kick_ms = (num_gpus == 1) ? gpu_kick_ms[gg] : std::max(step_kick_ms, gpu_kick_ms[gg]);
        step_ghost_ms = (num_gpus == 1) ? gpu_ghost_ms[gg] : std::max(step_ghost_ms, gpu_ghost_ms[gg]);
        step_xfer_ms = (num_gpus == 1) ? gpu_xfer_ms[gg] : std::max(step_xfer_ms, gpu_xfer_ms[gg]);
        if (gpu_inactive_min[gg] < step_next) step_next = gpu_inactive_min[gg];
        if (gpu_active_next[gg] < step_next) step_next = gpu_active_next[gg];
        step_active += gpu_active[gg];
        step_nodes += gpu_nodes[gg];
        step_invalid |= gpu_invalid[gg];
        step_builds += gpu_builds[gg];
      }
      force_ms = step_force_ms;
      kick_ms = step_kick_ms;
      ghost_ms = step_ghost_ms;
      export_ms = step_xfer_ms;

      if (step_invalid || !step_active || (!is_final && (step_next <= ti || step_next > timeline)))
        throw std::runtime_error("Invalid 3D multi-GPU state at step " + std::to_string(global_iteration) +
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

      if (cfg.energy_log && (is_pm_step || is_final)) {
        double tot[10] = {0};
        for (int g = 0; g < num_gpus; ++g)
          for (int k = 0; k < 10; ++k) tot[k] += gpu_energy[g][k];
        const double inv_a_e = cfg.comoving ? (1.0 / a) : 1.0;
        const double U_phys = tot[1] * inv_a_e;
        std::cout << std::setprecision(12)
                  << "ENERGY {\"step\":" << global_iteration << ",\"a\":" << a
                  << ",\"T\":" << tot[0] << ",\"U\":" << U_phys
                  << ",\"W_com\":" << tot[1] << ",\"E\":" << (tot[0] + U_phys)
                  << ",\"Px\":" << tot[2] << ",\"Py\":" << tot[3] << ",\"Pz\":" << tot[4]
                  << ",\"Lx\":" << tot[5] << ",\"Ly\":" << tot[6] << ",\"Lz\":" << tot[7]
                  << ",\"M\":" << tot[8] << ",\"v_rms\":" << sqrt(tot[9] / double(N)) << "}" << std::endl;
      }

      if (do_fc_now && n_fc_tgt > 0) {
        check(cudaSetDevice(0));
        AllSlabsData all_slabs{};
        all_slabs.x[0] = gpus[0].d_x.p;
        all_slabs.mass[0] = mass_uniform ? nullptr : gpus[0].d_mass.p;
        all_slabs.soft[0] = soft_uniform ? nullptr : gpus[0].d_soft.p;
        all_slabs.ns[0] = int(gpus[0].ns);
        const int ne = int(h_ewald_modes.size());
        ewald_kspace_accum_kernel<<<ne, 256, 0, gpus[0].compute_stream>>>(
            all_slabs, first_mass, d_ewald_modes.p, ne, d_ewald_C.p, d_ewald_S.p);
        ewald_target_force_kernel<<<n_fc_tgt, 256, 0, gpus[0].compute_stream>>>(
            all_slabs, first_mass, soft, d_fc_tgt_x.p, d_fc_tgt_h.p,
            d_fc_tgt_slab.p, d_fc_lids.p, n_fc_tgt, h.box, r_s, double(p.G),
            d_ewald_modes.p, d_ewald_C.p, d_ewald_S.p, ne,
            d_fc_short_ref.p, d_fc_total_ref.p);
        check(cudaStreamSynchronize(gpus[0].compute_stream));

        std::vector<float4> h_gs(n_fc_tgt);
        std::vector<float3> h_gm(n_fc_tgt);
        std::vector<double3> h_rs(n_fc_tgt), h_rt(n_fc_tgt);
        check(cudaMemcpy(h_gs.data(), d_fc_gpu_short.p, size_t(n_fc_tgt) * sizeof(float4), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(h_gm.data(), d_fc_gpu_mesh.p, size_t(n_fc_tgt) * sizeof(float3), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(h_rs.data(), d_fc_short_ref.p, size_t(n_fc_tgt) * sizeof(double3), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(h_rt.data(), d_fc_total_ref.p, size_t(n_fc_tgt) * sizeof(double3), cudaMemcpyDeviceToHost));
        std::vector<double> rel_s(n_fc_tgt), rel_t(n_fc_tgt);
        double sq_g_s = 0.0, sq_r_s = 0.0, sq_g_t = 0.0, sq_r_t = 0.0;
        for (int k = 0; k < n_fc_tgt; ++k) {
          const double gx = double(h_gs[k].x) * double(p.G);
          const double gy = double(h_gs[k].y) * double(p.G);
          const double gz = double(h_gs[k].z) * double(p.G);
          const double rxv = h_rs[k].x, ryv = h_rs[k].y, rzv = h_rs[k].z;
          const double ds = sqrt((gx - rxv)*(gx - rxv) + (gy - ryv)*(gy - ryv) + (gz - rzv)*(gz - rzv));
          const double rs_norm = sqrt(rxv*rxv + ryv*ryv + rzv*rzv);
          rel_s[k] = (rs_norm > 0.0) ? (ds / rs_norm) : ds;
          sq_g_s += ds * ds; sq_r_s += rs_norm * rs_norm;
          const double txv = gx + double(h_gm[k].x);
          const double tyv = gy + double(h_gm[k].y);
          const double tzv = gz + double(h_gm[k].z);
          const double dt = sqrt((txv - h_rt[k].x)*(txv - h_rt[k].x) + (tyv - h_rt[k].y)*(tyv - h_rt[k].y) + (tzv - h_rt[k].z)*(tzv - h_rt[k].z));
          const double rt_norm = sqrt(h_rt[k].x*h_rt[k].x + h_rt[k].y*h_rt[k].y + h_rt[k].z*h_rt[k].z);
          rel_t[k] = (rt_norm > 0.0) ? (dt / rt_norm) : dt;
          sq_g_t += dt * dt; sq_r_t += rt_norm * rt_norm;
        }
        std::sort(rel_s.begin(), rel_s.end());
        std::sort(rel_t.begin(), rel_t.end());
        std::cout << std::setprecision(12)
                  << "FORCE_CHECK {\"step\":" << global_iteration << ",\"a\":" << a
                  << ",\"n\":" << n_fc_tgt << ",\"method\":\"ewald_periodic\""
                  << ",\"short_rms\":" << sqrt(sq_g_s / std::max(1e-300, sq_r_s))
                  << ",\"short_median\":" << rel_s[n_fc_tgt / 2]
                  << ",\"total_rms\":" << sqrt(sq_g_t / std::max(1e-300, sq_r_t))
                  << ",\"total_median\":" << rel_t[n_fc_tgt / 2] << "}" << std::endl;
      }

      if (midpoint && !midpoint_written) {
        FILE* out = fopen(cfg.midpoint_path, "wb");
        if (!out) throw std::runtime_error("Cannot open SWIFT_RESTART_MIDPOINT_OUT");
        const Footer footer{N, h.box, a};
        write_bin(out, &footer, 1);
        write_bin(out, out_x.data(), N);
        write_bin(out, out_v.data(), N);
        write_bin(out, out_pot.data(), N);
        if (fclose(out)) throw std::runtime_error("Failed closing SWIFT_RESTART_MIDPOINT_OUT");
        midpoint_written = true;
      }

      if (is_final) {
        if (!is_last_seg) {
          std::ostringstream ss;
          ss << cfg.out_prefix << "_" << std::setw(4) << std::setfill('0') << seg << ".bin";
          const std::string snap_path = ss.str();
          std::vector<int64_t> fof_ids;
          uint64_t fof_ngrp = 0, fof_ngrouped = 0, fof_largest = 0;
          std::string fof_path;
          double fof_link = 0.0;
          if (cfg.run_fof) {
            fof_link = cfg.fof_link_ratio * h.box / cbrt(double(N));
            const double fof_t0 = wall();
            const int fst = run_domain_3d_fof(
                gpus, cfg, N, h.box, fof_link, out_x, fof_ids, fof_ngrp, fof_ngrouped, fof_largest);
            check(cudaSetDevice(0));
            if (fst != 0) throw std::runtime_error("FoF failed at intermediate snapshot");
            fof_path = snap_path + ".fof.bin";
            std::cout << std::setprecision(12)
                      << "FOF_RESULT {\"index\":" << seg << ",\"a\":" << seg_a_end
                      << ",\"path\":\"" << fof_path << "\",\"groups\":" << fof_ngrp
                      << ",\"largest\":" << fof_largest << ",\"fof_s\":" << (wall() - fof_t0) << "}" << std::endl;
          }
          std::cout << std::setprecision(12)
                    << "SNAPSHOT_QUEUED {\"index\":" << seg << ",\"a\":" << seg_a_end
                    << ",\"path\":\"" << snap_path << "\",\"d2h_ms\":" << export_ms << "}" << std::endl;
          if (!cfg.skip_output_write) {
            pending_snap = {true, seg, seg_a_end, snap_path, 0.0};
            snap_writer = std::thread([
                wx = std::move(out_x), wv = std::move(out_v), wpot = std::move(out_pot),
                wfof = std::move(fof_ids), fof_path, fof_ngrp, fof_largest,
                fof_min_u64 = uint64_t(cfg.fof_min_size), fof_link,
                N, box = h.box, seg_a_end, snap_path,
                &pending_snap, &snap_writer_err]() {
              try {
                const double tw0 = wall();
                FILE* out = fopen(snap_path.c_str(), "wb");
                if (!out) throw std::runtime_error("Cannot open snapshot");
                const Footer hdr{N, box, seg_a_end};
                write_bin(out, &hdr, 1);
                write_bin(out, wx.data(), N);
                write_bin(out, wv.data(), N);
                write_bin(out, wpot.data(), N);
                if (fclose(out)) throw std::runtime_error("Cannot close snapshot");
                if (!fof_path.empty()) {
                  write_fof_bin(fof_path, N, fof_ngrp, fof_largest, fof_min_u64, box, seg_a_end, fof_link, wfof.data());
                }
                pending_snap.write_s = wall() - tw0;
              } catch (...) {
                snap_writer_err = std::current_exception();
              }
            });
          }
          sanity_bad = 0; sanity_v2_sum = sanity_pot_sum = 0.0;
          sanity_pot_min = FLT_MAX; sanity_pot_max = -FLT_MAX;
          break;
        }
        break;
      }
      previous = ti;
      ti = step_next;
    }
  }

  join_snap_writer();
  if (cfg.force0_out_path) {
    FILE* fp = fopen(cfg.force0_out_path, "wb");
    if (!fp) throw std::runtime_error("Could not open SWIFT_FORCE0_OUT output");
    write_bin(fp, out_force0_f4.data(), out_force0_f4.size());
    if (fclose(fp)) throw std::runtime_error("Could not close SWIFT_FORCE0_OUT");
  }
  if (cfg.force_comp_path) {
    FILE* fp = fopen(cfg.force_comp_path, "wb");
    if (!fp) throw std::runtime_error("Could not open SWIFT_FORCE_COMPONENTS output");
    write_bin(fp, out_short_f4.data(), out_short_f4.size());
    write_bin(fp, out_long_f4.data(), out_long_f4.size());
    if (fclose(fp)) throw std::runtime_error("Could not close SWIFT_FORCE_COMPONENTS");
  }
  if (cfg.force3_out_path) {
    FILE* fp = fopen(cfg.force3_out_path, "wb");
    if (!fp) throw std::runtime_error("Could not open SWIFT_FORCE3_OUT output");
    write_bin(fp, out_short_f3.data(), out_short_f3.size());
    write_bin(fp, out_long_f3.data(), out_long_f3.size());
    if (fclose(fp)) throw std::runtime_error("Could not close SWIFT_FORCE3_OUT");
  }

  const double start_output = wall();
  size_t output_bytes = 0;
  if (!cfg.skip_output_write) {
    FILE* output = fopen(cfg.output_path, "wb");
    if (!output) return 4;
    const Footer footer{N, h.box, cfg.a_end};
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
            << ",\"whole_program_s\":" << (wall() - cfg.started_wall)
            << ",\"pm_updates_including_initial\":" << pm_updates
            << ",\"fast_tree\":true,\"warp_force\":true"
            << ",\"warp_m2l\":" << (cfg.warp_m2l ? "true" : "false")
            << ",\"fast_split\":" << (cfg.fast_split ? "true" : "false")
            << ",\"split_force\":" << (cfg.split_force_enabled ? "true" : "false")
            << ",\"pm_dist\":" << (cfg.pm_dist_enabled ? "true" : "false")
            << ",\"domain_mode\":\"" << domain_mode_name(cfg.domain_mode)
            << "\",\"rebalances\":" << rebalance_count
            << ",\"split_batch_warps\":" << cfg.split_batch_warps
            << ",\"lazy_tree\":" << (cfg.lazy_tree_enabled ? "true" : "false")
            << ",\"split_batches\":" << split_batches_run
            << ",\"split_overflow_retries\":" << split_overflow_retries
            << ",\"lazy_steps\":" << total_lazy_steps
            << ",\"full_rebuilds\":" << total_full_rebuilds
            << ",\"dynamic_pm\":" << (cfg.dynamic_pm ? "true" : "false")
            << ",\"last_pm_span_scale\":" << last_pm_span_scale
            << ",\"last_dt_max_rms\":" << last_dt_max_rms
            << ",\"peak_slab_particles\":" << peak_gpu_seen
            << ",\"device_buffer_bytes\":" << sum_dev_bytes
            << ",\"memory_mode\":\"multi_gpu_domain_3d\",\"num_gpus\":" << num_gpus
            << ",\"final_particle_D2H_bytes\":" << output_bytes
            << ",\"export_s\":" << export_s << "}" << std::endl;

  if (cfg.run_fof && !cfg.skip_output_write) {
    const double fof_link = cfg.fof_link_ratio * h.box / cbrt(double(N));
    const double fof_t0 = wall();
    std::vector<int64_t> fof_ids;
    uint64_t ngrp = 0, ngrouped = 0, largest = 0;
    const int fst = run_domain_3d_fof(
        gpus, cfg, N, h.box, fof_link, out_x, fof_ids, ngrp, ngrouped, largest);
    check(cudaSetDevice(0));
    if (fst != 0) throw std::runtime_error("FoF failed");
    const std::string fof_path = std::string(cfg.output_path) + ".fof.bin";
    write_fof_bin(fof_path, N, ngrp, largest, uint64_t(cfg.fof_min_size),
                  h.box, cfg.a_end, fof_link, fof_ids.data());
    std::cout << std::setprecision(12)
              << "FOF_RESULT {\"index\":\"final\",\"a\":" << cfg.a_end
              << ",\"path\":\"" << fof_path << "\",\"groups\":" << ngrp
              << ",\"largest\":" << largest << ",\"fof_s\":" << (wall() - fof_t0) << "}" << std::endl;
  }

  for (int s = 0; s < num_gpus; ++s) {
    for (int d = 0; d < num_gpus; ++d) {
      if (s != d) edges[s][d].free_all(gpus[s].dev, gpus[d].dev, gpus[s].device_bytes, gpus[d].device_bytes);
    }
  }
  if (cfg.pm_dist_enabled) dist_pm.destroy();
  for (int g = 0; g < num_gpus; ++g) {
    if (d_energy_accum[g]) {
      check(cudaSetDevice(g));
      cudaFree(d_energy_accum[g]);
    }
  }
  check(cudaSetDevice(0));
  for (auto& G : gpus) G.cleanup();
  if (fc_target_cap > 0 || cfg.lc_enabled) {
    check(cudaSetDevice(0));
    d_ewald_modes.free(aux_dev_bytes); d_ewald_C.free(aux_dev_bytes); d_ewald_S.free(aux_dev_bytes);
    d_fc_tgt_x.free(aux_dev_bytes); d_fc_tgt_h.free(aux_dev_bytes); d_fc_lids.free(aux_dev_bytes); d_fc_tgt_slab.free(aux_dev_bytes);
    d_fc_gpu_short.free(aux_dev_bytes); d_fc_gpu_mesh.free(aux_dev_bytes);
    d_fc_short_ref.free(aux_dev_bytes); d_fc_total_ref.free(aux_dev_bytes);
    d_lc_hits.free(aux_dev_bytes); d_lc_count.free(aux_dev_bytes); d_lc_shifts.free(aux_dev_bytes);
  }
  if (cfg.lc_enabled && lc_fp) {
    lc_writer.finish();
    if (fseeko(lc_fp, 0, SEEK_SET) != 0) throw std::runtime_error("Cannot seek lightcone header");
    const LightconeHeader final_hdr{{'S','W','L','C','0','0','0','1'}, lc_total_hits,
                                    {cfg.lc_obs.x, cfg.lc_obs.y, cfg.lc_obs.z},
                                    h.initial_scale, cfg.a_end, cfg.lc_a_obs, lc_r_max, h.box};
    write_bin(lc_fp, &final_hdr, 1);
    if (fclose(lc_fp)) throw std::runtime_error("Cannot close lightcone");
    std::cout << std::setprecision(12)
              << "LIGHTCONE_RESULT {\"path\":\"" << cfg.lc_out_env
              << "\",\"hits\":" << lc_total_hits << "}" << std::endl;
  }
  return 0;
}

} // namespace swift3d
