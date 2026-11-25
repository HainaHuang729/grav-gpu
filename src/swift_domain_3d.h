#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace swift3d {

constexpr int N_TOP = 12;
constexpr int N_TOP_CELLS = N_TOP * N_TOP * N_TOP; // 1728

enum class DomainMode {
  Slab1D = 0,
  Hilbert3D = 1,
  Orb3D = 2,
};

inline DomainMode parse_domain_mode(const char* s, int num_gpus) {
  if (!s || !*s) {
    return (num_gpus > 1) ? DomainMode::Hilbert3D : DomainMode::Slab1D;
  }
  const std::string val(s);
  if (val == "auto") {
    return (num_gpus > 1) ? DomainMode::Hilbert3D : DomainMode::Slab1D;
  }
  if (val == "1d_slab") return DomainMode::Slab1D;
  if (val == "3d_hilbert") return DomainMode::Hilbert3D;
  if (val == "3d_orb") return DomainMode::Orb3D;
  throw std::runtime_error(
      "Invalid SWIFT_DOMAIN_MODE '" + val +
      "' (expected auto, 1d_slab, 3d_hilbert, or 3d_orb)");
}

inline const char* domain_mode_name(DomainMode m) {
  switch (m) {
    case DomainMode::Slab1D: return "1d_slab";
    case DomainMode::Hilbert3D: return "3d_hilbert";
    case DomainMode::Orb3D: return "3d_orb";
  }
  return "unknown";
}

inline int wrap12(int v) {
  int r = v % N_TOP;
  return (r < 0) ? (r + N_TOP) : r;
}

inline int cell_id(int tx, int ty, int tz) {
  return (tx * N_TOP + ty) * N_TOP + tz;
}

inline void cell_coords(int cid, int& tx, int& ty, int& tz) {
  tx = cid / (N_TOP * N_TOP);
  const int rem = cid % (N_TOP * N_TOP);
  ty = rem / N_TOP;
  tz = rem % N_TOP;
}

namespace detail {

inline int sgn_i(int v) { return (v > 0) - (v < 0); }

inline void gilbert_half_vec(int ax, int ay, int az, int len,
                             int& hx, int& hy, int& hz) {
  // Choose even split on internal sub-blocks > 2 to preserve face continuity.
  int h = len / 2;
  if ((h & 1) && len > 2) h -= 1;
  hx = sgn_i(ax) * h;
  hy = sgn_i(ay) * h;
  hz = sgn_i(az) * h;
}

inline void gilbert3d_rec(
    int x, int y, int z,
    int ax, int ay, int az,
    int bx, int by, int bz,
    int cx, int cy, int cz,
    int axis_rot,
    std::vector<int>& out) {
  const int w = std::abs(ax + ay + az);
  const int h = std::abs(bx + by + bz);
  const int d = std::abs(cx + cy + cz);

  const int dax = sgn_i(ax), day = sgn_i(ay), daz = sgn_i(az);
  const int dbx = sgn_i(bx), dby = sgn_i(by), dbz = sgn_i(bz);
  const int dcx = sgn_i(cx), dcy = sgn_i(cy), dcz = sgn_i(cz);

  auto emit_pt = [&](int px, int py, int pz) {
    int rx = px, ry = py, rz = pz;
    switch (axis_rot) {
      case 0: rx = px; ry = py; rz = pz; break;
      case 1: rx = py; ry = pz; rz = px; break;
      case 2: rx = pz; ry = px; rz = py; break;
      case 3: rx = px; ry = pz; rz = py; break;
      case 4: rx = py; ry = px; rz = pz; break;
      case 5: rx = pz; ry = py; rz = px; break;
    }
    out.push_back(cell_id(rx, ry, rz));
  };

  // Trivial 1D lines.
  if (h == 1 && d == 1) {
    for (int i = 0; i < w; ++i) {
      emit_pt(x, y, z);
      x += dax; y += day; z += daz;
    }
    return;
  }
  if (w == 1 && d == 1) {
    for (int i = 0; i < h; ++i) {
      emit_pt(x, y, z);
      x += dbx; y += dby; z += dbz;
    }
    return;
  }
  if (w == 1 && h == 1) {
    for (int i = 0; i < d; ++i) {
      emit_pt(x, y, z);
      x += dcx; y += dcy; z += dcz;
    }
    return;
  }

  int ax2 = 0, ay2 = 0, az2 = 0;
  int bx2 = 0, by2 = 0, bz2 = 0;
  int cx2 = 0, cy2 = 0, cz2 = 0;
  gilbert_half_vec(ax, ay, az, w, ax2, ay2, az2);
  gilbert_half_vec(bx, by, bz, h, bx2, by2, bz2);
  gilbert_half_vec(cx, cy, cz, d, cx2, cy2, cz2);

  // Prefer binary split along major axis when elongated.
  if (2 * w > 3 * h && 2 * w > 3 * d) {
    gilbert3d_rec(x, y, z, ax2, ay2, az2, bx, by, bz, cx, cy, cz, axis_rot, out);
    gilbert3d_rec(x + ax2, y + ay2, z + az2,
                  ax - ax2, ay - ay2, az - az2, bx, by, bz, cx, cy, cz, axis_rot, out);
  } else if (3 * h > 4 * d) {
    // Do not split c.
    gilbert3d_rec(x, y, z, bx2, by2, bz2, cx, cy, cz, ax2, ay2, az2, axis_rot, out);
    gilbert3d_rec(x + bx2, y + by2, z + bz2,
                  ax, ay, az, bx - bx2, by - by2, bz - bz2, cx, cy, cz, axis_rot, out);
    gilbert3d_rec(x + (ax - dax) + (bx2 - dbx),
                  y + (ay - day) + (by2 - dby),
                  z + (az - daz) + (bz2 - dbz),
                  -bx2, -by2, -bz2, cx, cy, cz, -(ax - ax2), -(ay - ay2), -(az - az2),
                  axis_rot, out);
  } else if (3 * d > 4 * h) {
    // Do not split b.
    gilbert3d_rec(x, y, z, cx2, cy2, cz2, ax2, ay2, az2, bx, by, bz, axis_rot, out);
    gilbert3d_rec(x + cx2, y + cy2, z + cz2,
                  ax, ay, az, bx, by, bz, cx - cx2, cy - cy2, cz - cz2, axis_rot, out);
    gilbert3d_rec(x + (ax - dax) + (cx2 - dcx),
                  y + (ay - day) + (cy2 - dcy),
                  z + (az - daz) + (cz2 - dcz),
                  -cx2, -cy2, -cz2, -(ax - ax2), -(ay - ay2), -(az - az2), bx, by, bz,
                  axis_rot, out);
  } else {
    // Regular 5-block 3D split.
    gilbert3d_rec(x, y, z, bx2, by2, bz2, cx2, cy2, cz2, ax2, ay2, az2, axis_rot, out);
    gilbert3d_rec(x + bx2, y + by2, z + bz2,
                  cx, cy, cz, ax2, ay2, az2, bx - bx2, by - by2, bz - bz2, axis_rot, out);
    gilbert3d_rec(x + (bx2 - dbx) + (cx - dcx),
                  y + (by2 - dby) + (cy - dcy),
                  z + (bz2 - dbz) + (cz - dcz),
                  ax, ay, az, -bx2, -by2, -bz2, -(cx - cx2), -(cy - cy2), -(cz - cz2),
                  axis_rot, out);
    gilbert3d_rec(x + (ax - dax) + bx2 + (cx - dcx),
                  y + (ay - day) + by2 + (cy - dcy),
                  z + (az - daz) + bz2 + (cz - dcz),
                  -cx, -cy, -cz, -(ax - ax2), -(ay - ay2), -(az - az2),
                  bx - bx2, by - by2, bz - bz2, axis_rot, out);
    gilbert3d_rec(x + (ax - dax) + (bx2 - dbx),
                  y + (ay - day) + (by2 - dby),
                  z + (az - daz) + (bz2 - dbz),
                  -bx2, -by2, -bz2, cx2, cy2, cz2, -(ax - ax2), -(ay - ay2), -(az - az2),
                  axis_rot, out);
  }
}

inline int count_ghost_cells_quick(const std::array<int, N_TOP_CELLS>& cell_owner,
                                   int num_gpus) {
  int total_ghost = 0;
  std::array<uint8_t, N_TOP_CELLS> mark{};
  for (int dst = 0; dst < num_gpus; ++dst) {
    mark.fill(0);
    for (int cid = 0; cid < N_TOP_CELLS; ++cid) {
      if (cell_owner[cid] != dst) continue;
      int tx, ty, tz;
      cell_coords(cid, tx, ty, tz);
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = wrap12(tx + dx);
        for (int dy = -1; dy <= 1; ++dy) {
          const int ny = wrap12(ty + dy);
          for (int dz = -1; dz <= 1; ++dz) {
            const int nz = wrap12(tz + dz);
            const int ncid = cell_id(nx, ny, nz);
            if (cell_owner[ncid] != dst) mark[ncid] = 1;
          }
        }
      }
    }
    for (int cid = 0; cid < N_TOP_CELLS; ++cid) total_ghost += mark[cid];
  }
  return total_ghost;
}

inline void count_x_slab_footprint(const std::array<int, N_TOP_CELLS>& cell_owner,
                                   int num_gpus, int& max_x_slabs, int& sum_x_slabs) {
  max_x_slabs = 0;
  sum_x_slabs = 0;
  for (int g = 0; g < num_gpus; ++g) {
    uint16_t mask = 0;
    for (int cid = 0; cid < N_TOP_CELLS; ++cid) {
      if (cell_owner[cid] == g) {
        mask |= uint16_t(1u << (cid / (N_TOP * N_TOP)));
      }
    }
    int cnt = 0;
    for (int tx = 0; tx < N_TOP; ++tx) cnt += (mask >> tx) & 1;
    max_x_slabs = std::max(max_x_slabs, cnt);
    sum_x_slabs += cnt;
  }
}

inline double compute_load_imbalance(const std::array<double, N_TOP_CELLS>& weights,
                                     const std::array<int, N_TOP_CELLS>& cell_owner,
                                     int num_gpus) {
  std::vector<double> sum_w(num_gpus, 0.0);
  double total_w = 0.0;
  for (int cid = 0; cid < N_TOP_CELLS; ++cid) {
    const double w = std::max(weights[cid], 0.0);
    sum_w[cell_owner[cid]] += w;
    total_w += w;
  }
  if (total_w <= 0.0 || num_gpus <= 1) return 1.0;
  const double avg = total_w / double(num_gpus);
  double mx = 0.0;
  for (double s : sum_w) mx = std::max(mx, s);
  return mx / avg;
}

inline void make_effective_weights(const std::array<double, N_TOP_CELLS>& weights,
                                   std::array<double, N_TOP_CELLS>& eff) {
  double sum = 0.0;
  for (int c = 0; c < N_TOP_CELLS; ++c) sum += std::max(weights[c], 0.0);
  const double eps = (sum > 0.0) ? (1e-9 * sum / double(N_TOP_CELLS)) : 1.0;
  for (int c = 0; c < N_TOP_CELLS; ++c) {
    eff[c] = std::max(weights[c], 0.0) + eps;
  }
}

// Compute balanced contiguous cuts along a 1D curve minimizing max GPU load first,
// then minimizing sum of squared deviations from the mean load.
inline std::vector<int> split_curve_optimal(
    const std::array<int, N_TOP_CELLS>& order,
    const std::array<double, N_TOP_CELLS>& eff_w,
    int num_gpus) {
  std::vector<int> cuts(num_gpus + 1, 0);
  cuts[num_gpus] = N_TOP_CELLS;
  if (num_gpus <= 1) return cuts;

  std::array<double, N_TOP_CELLS + 1> pref{};
  for (int i = 0; i < N_TOP_CELLS; ++i) {
    pref[i + 1] = pref[i] + eff_w[order[i]];
  }
  const double total = pref[N_TOP_CELLS];

  auto feasible = [&](double limit, std::vector<int>& out_cuts) -> bool {
    out_cuts.assign(num_gpus + 1, 0);
    out_cuts[num_gpus] = N_TOP_CELLS;
    int pos = 0;
    for (int g = 0; g < num_gpus - 1; ++g) {
      const int rem_gpus = num_gpus - g;
      const int max_end = N_TOP_CELLS - (rem_gpus - 1);
      const double rem_w = total - pref[pos];
      const double target_w = rem_w / double(rem_gpus);
      // Largest end with pref[end] - pref[pos] <= limit
      int hi = pos + 1;
      while (hi < max_end && (pref[hi + 1] - pref[pos]) <= limit) ++hi;
      // Ensure remaining suffix fits within (rem_gpus - 1) * limit via target choice,
      // and clamp [lo_suf, hi]. Also locate closest index to ideal target_w <= hi.
      int lo = pos + 1;
      while (lo < hi && (total - pref[lo]) > double(rem_gpus - 1) * limit) ++lo;
      if ((pref[lo] - pref[pos]) > limit) return false;
      int best_idx = lo;
      double best_err = std::abs((pref[lo] - pref[pos]) - target_w);
      for (int idx = lo + 1; idx <= hi; ++idx) {
        const double err = std::abs((pref[idx] - pref[pos]) - target_w);
        if (err < best_err) {
          best_err = err;
          best_idx = idx;
        }
      }
      out_cuts[g + 1] = best_idx;
      pos = best_idx;
    }
    return (total - pref[pos]) <= limit + 1e-12 * total;
  };

  double lo = total / double(num_gpus) * 0.999;
  double hi = total;
  for (int it = 0; it < 56; ++it) {
    const double mid = 0.5 * (lo + hi);
    std::vector<int> tmp;
    if (feasible(mid, tmp)) hi = mid;
    else lo = mid;
  }
  std::vector<int> best_cuts;
  if (!feasible(hi * (1.0 + 1e-7), best_cuts)) {
    feasible(total, best_cuts);
  }
  return best_cuts;
}

inline void orb_recurse(
    const std::array<double, N_TOP_CELLS>& eff_w,
    int x0, int dx, int y0, int dy, int z0, int dz,
    int g0, int num_g,
    std::array<int, N_TOP_CELLS>& cell_owner) {
  if (num_g <= 1) {
    for (int ix = 0; ix < dx; ++ix) {
      const int tx = wrap12(x0 + ix);
      for (int iy = 0; iy < dy; ++iy) {
        const int ty = wrap12(y0 + iy);
        for (int iz = 0; iz < dz; ++iz) {
          cell_owner[cell_id(tx, ty, wrap12(z0 + iz))] = g0;
        }
      }
    }
    return;
  }

  const int dims[3] = {dx, dy, dz};
  const int lo_b[3] = {x0, y0, z0};
  const int n_cells_box = dims[0] * dims[1] * dims[2];

  double total_w = 0.0;
  for (int ix = 0; ix < dx; ++ix)
    for (int iy = 0; iy < dy; ++iy)
      for (int iz = 0; iz < dz; ++iz)
        total_w += eff_w[cell_id(wrap12(x0 + ix), wrap12(y0 + iy), wrap12(z0 + iz))];

  int best_axis = -1;
  int best_s0 = 0;
  int best_dl = 1;
  int best_kl = num_g / 2;
  double best_score = std::numeric_limits<double>::infinity();

  int kl_candidates[2] = {num_g / 2, num_g - num_g / 2};
  int n_kl = (kl_candidates[0] == kl_candidates[1]) ? 1 : 2;

  for (int a = 0; a < 3; ++a) {
    if (dims[a] < 2) continue;
    const int slice_cells = n_cells_box / dims[a];
    std::array<double, N_TOP> slice_w{};
    for (int ix = 0; ix < dx; ++ix) {
      for (int iy = 0; iy < dy; ++iy) {
        for (int iz = 0; iz < dz; ++iz) {
          const int off = (a == 0) ? ix : ((a == 1) ? iy : iz);
          slice_w[off] += eff_w[cell_id(wrap12(x0 + ix), wrap12(y0 + iy), wrap12(z0 + iz))];
        }
      }
    }

    // When an axis spans the full periodic box (N_TOP), evaluate cyclic start offsets too.
    const int max_s0 = (dims[a] == N_TOP) ? N_TOP : 1;
    for (int s0 = 0; s0 < max_s0; ++s0) {
      for (int kli = 0; kli < n_kl; ++kli) {
        const int kl = kl_candidates[kli];
        const int kr = num_g - kl;
        double wl = 0.0;
        for (int dl = 1; dl < dims[a]; ++dl) {
          wl += slice_w[(s0 + dl - 1) % dims[a]];
          const int cl = dl * slice_cells;
          const int cr = (dims[a] - dl) * slice_cells;
          if (cl < kl || cr < kr) continue;
          const double wr = total_w - wl;
          const double avg = total_w / double(num_g);
          const double imb = std::max(wl / double(kl), wr / double(kr)) / avg;

          const int dr = dims[a] - dl;
          const int other0 = dims[(a + 1) % 3], other1 = dims[(a + 2) % 3];
          const double asp_l = double(std::max({dl, other0, other1})) / double(std::min({dl, other0, other1}));
          const double asp_r = double(std::max({dr, other0, other1})) / double(std::min({dr, other0, other1}));
          const double cut_area = double(other0 * other1) / double(N_TOP * N_TOP);
          const double x_pen = (a == 0) ? 0.0 : 0.00002;
          const double score = imb + 0.0001 * std::max(asp_l, asp_r) + 0.00005 * cut_area + x_pen;
          if (score < best_score) {
            best_score = score;
            best_axis = a;
            best_s0 = s0;
            best_dl = dl;
            best_kl = kl;
          }
        }
      }
    }
  }

  if (best_axis < 0) {
    throw std::runtime_error("ORB partitioning failed: box cannot accommodate num_gpus");
  }

  const int dr = dims[best_axis] - best_dl;
  const int base_l = wrap12(lo_b[best_axis] + best_s0);
  const int base_r = wrap12(lo_b[best_axis] + best_s0 + best_dl);
  if (best_axis == 0) {
    orb_recurse(eff_w, base_l, best_dl, y0, dy, z0, dz, g0, best_kl, cell_owner);
    orb_recurse(eff_w, base_r, dr,      y0, dy, z0, dz, g0 + best_kl, num_g - best_kl, cell_owner);
  } else if (best_axis == 1) {
    orb_recurse(eff_w, x0, dx, base_l, best_dl, z0, dz, g0, best_kl, cell_owner);
    orb_recurse(eff_w, x0, dx, base_r, dr,      z0, dz, g0 + best_kl, num_g - best_kl, cell_owner);
  } else {
    orb_recurse(eff_w, x0, dx, y0, dy, base_l, best_dl, g0, best_kl, cell_owner);
    orb_recurse(eff_w, x0, dx, y0, dy, base_r, dr,      g0 + best_kl, num_g - best_kl, cell_owner);
  }
}

// Reassign subdomain IDs to physical GPU ranks so the heaviest ghost exchanges use NVLink.
inline void remap_owners_for_nvlink(
    const std::array<uint64_t, N_TOP_CELLS>& counts,
    int num_gpus,
    const std::vector<std::vector<int>>& can_peer,
    std::array<int, N_TOP_CELLS>& cell_owner) {
  if (num_gpus < 2 || num_gpus > 6 || int(can_peer.size()) < num_gpus) return;
  bool has_peer = false;
  bool has_nop = false;
  for (int i = 0; i < num_gpus; ++i) {
    if (int(can_peer[i].size()) < num_gpus) return;
    for (int j = 0; j < num_gpus; ++j) {
      if (i == j) continue;
      if (can_peer[i][j]) has_peer = true;
      else has_nop = true;
    }
  }
  if (!has_peer || !has_nop) return;

  double edge_w[6][6]{};
  for (int dst = 0; dst < num_gpus; ++dst) {
    std::array<uint8_t, N_TOP_CELLS> mark{};
    for (int cid = 0; cid < N_TOP_CELLS; ++cid) {
      if (cell_owner[cid] != dst) continue;
      int tx, ty, tz;
      cell_coords(cid, tx, ty, tz);
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = wrap12(tx + dx);
        for (int dy = -1; dy <= 1; ++dy) {
          const int ny = wrap12(ty + dy);
          for (int dz = -1; dz <= 1; ++dz) {
            const int nz = wrap12(tz + dz);
            const int ncid = cell_id(nx, ny, nz);
            if (cell_owner[ncid] != dst) mark[ncid] = 1;
          }
        }
      }
    }
    for (int cid = 0; cid < N_TOP_CELLS; ++cid) {
      if (!mark[cid]) continue;
      const int src = cell_owner[cid];
      edge_w[src][dst] += double(std::max<uint64_t>(counts[cid], 1ULL));
    }
  }

  std::array<int, 6> perm{0, 1, 2, 3, 4, 5};
  std::array<int, 6> best_perm = perm;
  double best_max = std::numeric_limits<double>::infinity();
  double best_sum = std::numeric_limits<double>::infinity();
  do {
    double mx = 0.0, sm = 0.0;
    for (int gp = 0; gp < num_gpus; ++gp) {
      double send_p = 0.0, recv_p = 0.0;
      for (int gq = 0; gq < num_gpus; ++gq) {
        if (gp == gq) continue;
        if (!can_peer[gq][gp]) send_p += edge_w[perm[gp]][perm[gq]];
        if (!can_peer[gp][gq]) recv_p += edge_w[perm[gq]][perm[gp]];
      }
      mx = std::max(mx, std::max(send_p, recv_p));
      sm += send_p;
    }
    if (mx + 1e-9 < best_max ||
        (std::abs(mx - best_max) <= 1e-9 && sm + 1e-9 < best_sum)) {
      best_max = mx;
      best_sum = sm;
      best_perm = perm;
    }
  } while (std::next_permutation(perm.begin(), perm.begin() + num_gpus));

  int inv_perm[6]{};
  for (int gp = 0; gp < num_gpus; ++gp) inv_perm[best_perm[gp]] = gp;
  for (int cid = 0; cid < N_TOP_CELLS; ++cid) {
    cell_owner[cid] = inv_perm[cell_owner[cid]];
  }
}

} // namespace detail

inline std::array<int, N_TOP_CELLS> build_gilbert3d_12(int axis_rot) {
  std::vector<int> v;
  v.reserve(N_TOP_CELLS);
  detail::gilbert3d_rec(0, 0, 0, N_TOP, 0, 0, 0, N_TOP, 0, 0, 0, N_TOP,
                        axis_rot % 6, v);
  if (int(v.size()) != N_TOP_CELLS) {
    throw std::runtime_error("build_gilbert3d_12: unexpected size");
  }
  std::array<int, N_TOP_CELLS> arr{};
  std::copy(v.begin(), v.end(), arr.begin());
  return arr;
}

inline void partition_3d_hilbert(
    const std::array<double, N_TOP_CELLS>& weights,
    int num_gpus,
    std::array<int, N_TOP_CELLS>& cell_owner,
    int* chosen_rot_out = nullptr) {
  if (num_gpus <= 1) {
    cell_owner.fill(0);
    if (chosen_rot_out) *chosen_rot_out = 0;
    return;
  }
  std::array<double, N_TOP_CELLS> eff{};
  detail::make_effective_weights(weights, eff);

  double best_score = std::numeric_limits<double>::infinity();
  int best_rot = 0;
  std::array<int, N_TOP_CELLS> best_owner{};

  for (int rot = 0; rot < 6; ++rot) {
    const auto order = build_gilbert3d_12(rot);
    const auto cuts = detail::split_curve_optimal(order, eff, num_gpus);
    std::array<int, N_TOP_CELLS> cand{};
    for (int g = 0; g < num_gpus; ++g) {
      for (int i = cuts[g]; i < cuts[g + 1]; ++i) cand[order[i]] = g;
    }
    const double imb = detail::compute_load_imbalance(weights, cand, num_gpus);
    const int gh = detail::count_ghost_cells_quick(cand, num_gpus);
    int max_xs = 0, sum_xs = 0;
    detail::count_x_slab_footprint(cand, num_gpus, max_xs, sum_xs);
    const double score = imb + 0.04 * double(gh) / double(N_TOP_CELLS) +
                         0.006 * double(max_xs) / double(N_TOP);
    if (score < best_score) {
      best_score = score;
      best_rot = rot;
      best_owner = cand;
    }
  }
  cell_owner = best_owner;
  if (chosen_rot_out) *chosen_rot_out = best_rot;
}

inline void partition_3d_orb(
    const std::array<double, N_TOP_CELLS>& weights,
    int num_gpus,
    std::array<int, N_TOP_CELLS>& cell_owner) {
  if (num_gpus <= 1) {
    cell_owner.fill(0);
    return;
  }
  std::array<double, N_TOP_CELLS> eff{};
  detail::make_effective_weights(weights, eff);
  cell_owner.fill(0);
  detail::orb_recurse(eff, 0, N_TOP, 0, N_TOP, 0, N_TOP, 0, num_gpus, cell_owner);
}

struct DomainPartitionStats {
  double w_imb = 1.0;
  double p_imb = 1.0;
  int ghost_total = 0;
  int max_ghost_cells = 0;
  int min_own_cells = N_TOP_CELLS;
  int max_own_cells = N_TOP_CELLS;
  int max_x_slabs = N_TOP;
  int chosen_rot = 0;
};

struct Domain3DPlan {
  int num_gpus = 1;
  DomainMode mode = DomainMode::Hilbert3D;
  std::array<int, N_TOP_CELLS> cell_owner{};
  std::array<int, N_TOP_CELLS> cell_to_own_slot{};
  std::vector<std::array<int, N_TOP_CELLS>> cell_to_ghost_slot;
  std::vector<int> n_interior_own;
  std::vector<std::vector<int>> own_cells;
  std::vector<std::vector<int>> ghost_cells;
  std::vector<std::vector<int>> ghost_cell_owner;
  std::vector<std::vector<uint32_t>> slot_nb_roots;
  std::vector<std::vector<std::vector<int>>> export_cells;
  std::vector<std::vector<std::vector<int>>> export_src_slots;
  std::vector<std::vector<std::vector<int>>> export_dst_ghost_slots;
  std::vector<std::vector<int>> gpu_x_planes;
  DomainPartitionStats stats{};
};

inline Domain3DPlan build_domain_3d_plan_from_owners(
    const std::array<uint64_t, N_TOP_CELLS>& counts,
    const std::array<double, N_TOP_CELLS>& weights,
    const std::array<int, N_TOP_CELLS>& cell_owner,
    int num_gpus,
    DomainMode mode,
    int pm_side,
    int chosen_rot = 0,
    const std::vector<std::vector<int>>* can_peer = nullptr) {
  std::array<int, N_TOP_CELLS> eff_owner = cell_owner;
  for (int cid = 0; cid < N_TOP_CELLS; ++cid) {
    if (eff_owner[cid] < 0 || eff_owner[cid] >= num_gpus) {
      throw std::runtime_error("Invalid GPU owner in build_domain_3d_plan");
    }
  }
  if (can_peer != nullptr) {
    detail::remap_owners_for_nvlink(counts, num_gpus, *can_peer, eff_owner);
  }

  Domain3DPlan plan;
  plan.num_gpus = num_gpus;
  plan.mode = mode;
  plan.cell_owner = eff_owner;
  plan.cell_to_own_slot.fill(-1);
  plan.cell_to_ghost_slot.assign(num_gpus, {});
  for (auto& arr : plan.cell_to_ghost_slot) arr.fill(-1);
  plan.n_interior_own.assign(num_gpus, 0);
  plan.own_cells.assign(num_gpus, {});
  plan.ghost_cells.assign(num_gpus, {});
  plan.ghost_cell_owner.assign(num_gpus, {});
  plan.slot_nb_roots.assign(num_gpus, {});
  plan.export_cells.assign(num_gpus, std::vector<std::vector<int>>(num_gpus));
  plan.export_src_slots.assign(num_gpus, std::vector<std::vector<int>>(num_gpus));
  plan.export_dst_ghost_slots.assign(num_gpus, std::vector<std::vector<int>>(num_gpus));
  plan.gpu_x_planes.assign(num_gpus, {});

  // Partition owned cells into interior (all 26 neighbors local) then boundary slots.
  for (int g = 0; g < num_gpus; ++g) {
    std::vector<int> int_cells, bnd_cells;
    for (int cid = 0; cid < N_TOP_CELLS; ++cid) {
      if (eff_owner[cid] != g) continue;
      int tx, ty, tz;
      cell_coords(cid, tx, ty, tz);
      bool is_int = true;
      for (int dx = -1; dx <= 1 && is_int; ++dx) {
        const int nx = wrap12(tx + dx);
        for (int dy = -1; dy <= 1 && is_int; ++dy) {
          const int ny = wrap12(ty + dy);
          for (int dz = -1; dz <= 1; ++dz) {
            const int nz = wrap12(tz + dz);
            if (eff_owner[cell_id(nx, ny, nz)] != g) {
              is_int = false;
              break;
            }
          }
        }
      }
      if (is_int) int_cells.push_back(cid);
      else bnd_cells.push_back(cid);
    }
    plan.n_interior_own[g] = int(int_cells.size());
    plan.own_cells[g].reserve(int_cells.size() + bnd_cells.size());
    for (int cid : int_cells) {
      plan.cell_to_own_slot[cid] = int(plan.own_cells[g].size());
      plan.own_cells[g].push_back(cid);
    }
    for (int cid : bnd_cells) {
      plan.cell_to_own_slot[cid] = int(plan.own_cells[g].size());
      plan.own_cells[g].push_back(cid);
    }
  }

  for (int dst = 0; dst < num_gpus; ++dst) {
    std::array<uint8_t, N_TOP_CELLS> is_ghost{};
    for (int cid : plan.own_cells[dst]) {
      int tx, ty, tz;
      cell_coords(cid, tx, ty, tz);
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = wrap12(tx + dx);
        for (int dy = -1; dy <= 1; ++dy) {
          const int ny = wrap12(ty + dy);
          for (int dz = -1; dz <= 1; ++dz) {
            const int nz = wrap12(tz + dz);
            const int ncid = cell_id(nx, ny, nz);
            if (eff_owner[ncid] != dst) is_ghost[ncid] = 1;
          }
        }
      }
    }
    for (int cid = 0; cid < N_TOP_CELLS; ++cid) {
      if (!is_ghost[cid]) continue;
      const int gs = int(plan.ghost_cells[dst].size());
      const int src = eff_owner[cid];
      plan.cell_to_ghost_slot[dst][cid] = gs;
      plan.ghost_cells[dst].push_back(cid);
      plan.ghost_cell_owner[dst].push_back(src);
      plan.export_cells[src][dst].push_back(cid);
      plan.export_src_slots[src][dst].push_back(plan.cell_to_own_slot[cid]);
      plan.export_dst_ghost_slots[src][dst].push_back(gs);
    }

    // 27-neighbor lookup table for every owned home slot on dst.
    plan.slot_nb_roots[dst].resize(plan.own_cells[dst].size() * 27);
    for (size_t s = 0; s < plan.own_cells[dst].size(); ++s) {
      const int cid = plan.own_cells[dst][s];
      int tx, ty, tz;
      cell_coords(cid, tx, ty, tz);
      int k = 0;
      for (int dx = -1; dx <= 1; ++dx) {
        const int nx = wrap12(tx + dx);
        for (int dy = -1; dy <= 1; ++dy) {
          const int ny = wrap12(ty + dy);
          for (int dz = -1; dz <= 1; ++dz) {
            const int nz = wrap12(tz + dz);
            const int ncid = cell_id(nx, ny, nz);
            uint32_t encoded = 0;
            if (eff_owner[ncid] == dst) {
              encoded = uint32_t(plan.cell_to_own_slot[ncid]);
            } else {
              encoded = (1u << 31) | uint32_t(plan.cell_to_ghost_slot[dst][ncid]);
            }
            plan.slot_nb_roots[dst][s * 27 + k++] = encoded;
          }
        }
      }
    }

    // Distributed PM X planes covering all owned tx slices with CIC [-1, +1] margin.
    if (pm_side > 0) {
      std::array<bool, N_TOP> owns_tx{};
      for (int cid : plan.own_cells[dst]) owns_tx[cid / (N_TOP * N_TOP)] = true;
      std::vector<uint8_t> xmark(pm_side, 0);
      for (int tx = 0; tx < N_TOP; ++tx) {
        if (!owns_tx[tx]) continue;
        const int px_lo = int(std::floor(double(tx) * pm_side / double(N_TOP))) - 1;
        const int px_hi = int(std::ceil(double(tx + 1) * pm_side / double(N_TOP))) + 1;
        for (int px = px_lo; px <= px_hi; ++px) {
          int gx = px % pm_side;
          if (gx < 0) gx += pm_side;
          xmark[gx] = 1;
        }
      }
      for (int gx = 0; gx < pm_side; ++gx) {
        if (xmark[gx]) plan.gpu_x_planes[dst].push_back(gx);
      }
    }
  }

  // Stats
  plan.stats.chosen_rot = chosen_rot;
  plan.stats.w_imb = detail::compute_load_imbalance(weights, eff_owner, num_gpus);
  std::vector<double> g_cnt(num_gpus, 0.0);
  uint64_t total_cnt = 0;
  for (int cid = 0; cid < N_TOP_CELLS; ++cid) {
    g_cnt[eff_owner[cid]] += double(counts[cid]);
    total_cnt += counts[cid];
  }
  if (total_cnt > 0 && num_gpus > 1) {
    const double avg_c = double(total_cnt) / double(num_gpus);
    double mx = 0.0;
    for (double c : g_cnt) mx = std::max(mx, c);
    plan.stats.p_imb = mx / avg_c;
  } else {
    plan.stats.p_imb = 1.0;
  }
  plan.stats.ghost_total = 0;
  plan.stats.max_ghost_cells = 0;
  plan.stats.min_own_cells = N_TOP_CELLS;
  plan.stats.max_own_cells = 0;
  for (int g = 0; g < num_gpus; ++g) {
    const int no = int(plan.own_cells[g].size());
    const int ng = int(plan.ghost_cells[g].size());
    plan.stats.min_own_cells = std::min(plan.stats.min_own_cells, no);
    plan.stats.max_own_cells = std::max(plan.stats.max_own_cells, no);
    plan.stats.max_ghost_cells = std::max(plan.stats.max_ghost_cells, ng);
    plan.stats.ghost_total += ng;
  }
  int sum_xs = 0;
  detail::count_x_slab_footprint(eff_owner, num_gpus, plan.stats.max_x_slabs, sum_xs);
  return plan;
}

inline Domain3DPlan build_domain_3d_plan(
    const std::array<uint64_t, N_TOP_CELLS>& counts,
    const std::array<double, N_TOP_CELLS>& weights,
    int num_gpus,
    DomainMode mode,
    int pm_side,
    const std::vector<std::vector<int>>* can_peer = nullptr) {
  std::array<int, N_TOP_CELLS> cell_owner{};
  int chosen_rot = 0;
  if (mode == DomainMode::Orb3D) {
    partition_3d_orb(weights, num_gpus, cell_owner);
  } else {
    partition_3d_hilbert(weights, num_gpus, cell_owner, &chosen_rot);
  }
  if (can_peer != nullptr) {
    detail::remap_owners_for_nvlink(counts, num_gpus, *can_peer, cell_owner);
  }
  return build_domain_3d_plan_from_owners(
      counts, weights, cell_owner, num_gpus, mode, pm_side, chosen_rot, nullptr);
}

} // namespace swift3d
