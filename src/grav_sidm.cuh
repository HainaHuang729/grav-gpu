#ifndef GRAV_SIDM_CUH
#define GRAV_SIDM_CUH

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <algorithm>

// B2-a: GPU-Native Monte Carlo Self-Interacting Dark Matter (MC-SIDM).
// Operates directly on VRAM-resident Morton-sorted octree leaves (leaf_limit <= 64)
// using race-free warp round-robin tournament pairing for exact momentum & kinetic energy conservation.
//
// Controlled by:
//   GRAV_SIDM_SIGMA_M=1.0   (cross section per unit mass sigma_0/m in cm^2/g; 0 = disabled)
//   GRAV_SIDM_W0=0.0        (velocity scale w_0 in km/s for Yukawa sigma(v)=sigma_0/(1+(v/w_0)^2)^2; 0 = constant)
//   GRAV_SIDM_HS=0.0        (scattering search radius h_s in Mpc; 0 = auto = Plummer softening)
//   GRAV_SIDM_SEED=1337     (deterministic RNG seed)
//
// Unit conversion:
//   Internal units: Length = Mpc, Mass = 10^10 Msun, Velocity = km/s.
//   1 cm^2/g = (1.98841e43 g / (3.08567758e24 cm)^2) * [Mpc^2 / (10^10 Msun)]
//            = 2.088357e-6 Mpc^2 / (10^10 Msun).
//   In non-comoving dimensionless N-body units (e.g. G=1 Plummer/Hernquist test),
//   set GRAV_SIDM_SIGMA_INTERNAL > 0 directly in simulation [L^2 / M] units!

struct GravSidmConfig {
  double sigma_m_cm2_g = 0.0;      // cm^2 / g
  double sigma_m_internal = 0.0;   // internal [L^2 / M] units
  double w0_kms = 0.0;             // velocity-dependent scale w0 (0 = constant)
  double hs_override = 0.0;        // search radius h_s override (0 = Plummer softening)
  uint64_t seed = 1337ULL;
};

struct GravSidmStats {
  unsigned long long n_scatters;
  unsigned long long n_pairs_tested;
  double max_prob;
  double sum_dE_kin;
};

inline GravSidmConfig grav_sidm_init_from_env() {
  GravSidmConfig cfg{};
  constexpr double CM2_PER_G_TO_INTERNAL = 2.088357e-6;  // Mpc^2 / (10^10 Msun)
  if (const char* s = std::getenv("GRAV_SIDM_SIGMA_M")) {
    if (*s) {
      double v = std::atof(s);
      if (std::isfinite(v) && v > 0.0) {
        cfg.sigma_m_cm2_g = v;
        cfg.sigma_m_internal = v * CM2_PER_G_TO_INTERNAL;
      }
    }
  }
  if (const char* s = std::getenv("GRAV_SIDM_SIGMA_INTERNAL")) {
    if (*s) {
      double v = std::atof(s);
      if (std::isfinite(v) && v > 0.0) {
        cfg.sigma_m_internal = v;
      }
    }
  }
  if (const char* s = std::getenv("GRAV_SIDM_W0")) {
    if (*s) {
      double v = std::atof(s);
      if (std::isfinite(v) && v > 0.0) cfg.w0_kms = v;
    }
  }
  if (const char* s = std::getenv("GRAV_SIDM_HS")) {
    if (*s) {
      double v = std::atof(s);
      if (std::isfinite(v) && v > 0.0) cfg.hs_override = v;
    }
  }
  if (const char* s = std::getenv("GRAV_SIDM_SEED")) {
    if (*s) cfg.seed = std::strtoull(s, nullptr, 10);
  }
  return cfg;
}

// Host/device deterministic stateless hash RNG (SplitMix64 -> uniform [0, 1))
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
uint64_t grav_sidm_splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
double grav_sidm_u01(uint64_t h) {
  return double(h >> 11) * (1.0 / 9007199254740992.0);
}

// Velocity-dependent cross-section per unit mass: sigma(v)/m = (sigma_0/m) / (1 + (v/w0)^2)^2
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
double grav_sidm_sigma_of_v(double sigma0_m, double w0, double v_rel) {
  if (w0 <= 0.0) return sigma0_m;
  const double r = v_rel / w0;
  const double denom = 1.0 + r * r;
  return sigma0_m / (denom * denom);
}

// Top-hat spherical kernel W(r, h_s) = 3 / (4 * pi * h_s^3) for r < h_s (normalized: int W d^3r = 1)
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
double grav_sidm_pair_prob(double r2, double hs, double m_pair_avg,
                           double sigma_m, double v_rel, double dt_phys) {
  if (r2 >= hs * hs || hs <= 0.0 || v_rel <= 0.0 || dt_phys <= 0.0) return 0.0;
  const double inv_vol = 3.0 / (4.0 * 3.14159265358979323846 * hs * hs * hs);
  return m_pair_avg * sigma_m * v_rel * dt_phys * inv_vol;
}

#ifdef __CUDACC__
// Perform an elastic isotropic 2-body scatter in the center-of-mass frame.
// Preserves total momentum (m_i*v_i + m_j*v_j) and total kinetic energy exactly!
__host__ __device__ __forceinline__ void grav_sidm_elastic_kick(
    float3& vi, float3& vj, float mi, float mj,
    double u_cos, double u_phi, double& dE_kin_out) {
  const double m_sum = double(mi) + double(mj);
  const double inv_m = 1.0 / m_sum;
  const double vx_cm = (double(mi) * vi.x + double(mj) * vj.x) * inv_m;
  const double vy_cm = (double(mi) * vi.y + double(mj) * vj.y) * inv_m;
  const double vz_cm = (double(mi) * vi.z + double(mj) * vj.z) * inv_m;

  const double dvx = double(vi.x) - double(vj.x);
  const double dvy = double(vi.y) - double(vj.y);
  const double dvz = double(vi.z) - double(vj.z);
  const double v_rel = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);

  const double cos_th = 2.0 * u_cos - 1.0;
  const double sin_th = sqrt(fmax(0.0, 1.0 - cos_th * cos_th));
  const double phi = 6.28318530717958647692 * u_phi;
  const double nx = sin_th * cos(phi);
  const double ny = sin_th * sin(phi);
  const double nz = cos_th;

  const double e_before = 0.5 * double(mi) * (double(vi.x)*vi.x + double(vi.y)*vi.y + double(vi.z)*vi.z)
                        + 0.5 * double(mj) * (double(vj.x)*vj.x + double(vj.y)*vj.y + double(vj.z)*vj.z);

  const double fi = double(mj) * inv_m * v_rel;
  const double fj = double(mi) * inv_m * v_rel;
  vi.x = float(vx_cm + fi * nx);
  vi.y = float(vy_cm + fi * ny);
  vi.z = float(vz_cm + fi * nz);
  vj.x = float(vx_cm - fj * nx);
  vj.y = float(vy_cm - fj * ny);
  vj.z = float(vz_cm - fj * nz);

  const double e_after = 0.5 * double(mi) * (double(vi.x)*vi.x + double(vi.y)*vi.y + double(vi.z)*vi.z)
                       + 0.5 * double(mj) * (double(vj.x)*vj.x + double(vj.y)*vj.y + double(vj.z)*vj.z);
  dE_kin_out = e_after - e_before;
}

// Warp-level race-free round-robin MC-SIDM kernel over all octree leaf nodes.
// Each warp processes one leaf node (n_leaf <= 64 particles) in shared memory.
// In each round r in [0, m_even - 1), lane k pairs (p0, p1) disjointly so no particle
// is ever updated concurrently by two threads!
template <typename NodeGeomT>
__global__ void sidm_scatter_leaves_kernel(
    const NodeGeomT* __restrict__ geom,
    const int* __restrict__ order,
    const double3* __restrict__ pos,
    float3* __restrict__ vel,
    const float* __restrict__ mass_arr,
    float uniform_mass,
    int n_nodes,
    double box,
    bool periodic,
    double scale_a,
    bool comoving,
    double dt_phys,
    double sigma0_m,
    double w0_kms,
    double hs_phys,
    uint64_t step_seed,
    GravSidmStats* __restrict__ stats) {
  constexpr int WARPS_PER_BLOCK = 4;
  const int warp_in_block = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int warp_global = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;
  if (warp_global >= n_nodes) return;

  const NodeGeomT nd = geom[warp_global];
  // Check leaf (all 8 children < 0)
  bool is_leaf = true;
  for (int c = 0; c < 8; ++c) {
    if (nd.children[c] >= 0) { is_leaf = false; break; }
  }
  const int count = nd.end - nd.start;
  if (!is_leaf || count < 2) return;

  // Handle leaves with <= 64 particles directly in shared memory (default leaf_limit=64)
  const int m = (count <= 64) ? count : 64;
  __shared__ int s_idx[WARPS_PER_BLOCK][64];
  __shared__ double3 s_x[WARPS_PER_BLOCK][64];
  __shared__ float3 s_v[WARPS_PER_BLOCK][64];
  __shared__ float s_m[WARPS_PER_BLOCK][64];

  for (int k = lane; k < m; k += 32) {
    const int pid = order[nd.start + k];
    s_idx[warp_in_block][k] = pid;
    s_x[warp_in_block][k] = pos[pid];
    s_v[warp_in_block][k] = vel[pid];
    s_m[warp_in_block][k] = mass_arr ? mass_arr[pid] : uniform_mass;
  }
  __syncwarp();

  const int m_even = (m & 1) ? (m + 1) : m;
  const int n_rounds = m_even - 1;
  const int half = m_even >> 1;
  const double pos_to_phys = comoving ? scale_a : 1.0;
  const double vel_to_phys = comoving ? (1.0 / scale_a) : 1.0;

  unsigned long long loc_scatters = 0;
  unsigned long long loc_tested = 0;
  double loc_max_p = 0.0;
  double loc_dE = 0.0;

  // Circle-method round-robin tournament: every pair (i < j) visited exactly once,
  // and all pairs in a single round are 100% disjoint!
  for (int r = 0; r < n_rounds; ++r) {
    if (lane < half) {
      int a_slot, b_slot;
      if (lane == 0) {
        a_slot = 0;
        b_slot = 1 + r;
      } else {
        a_slot = 1 + ((r + lane) % (m_even - 1));
        b_slot = 1 + ((r + m_even - 1 - lane) % (m_even - 1));
      }
      if (a_slot < m && b_slot < m) {
        const double3 xa = s_x[warp_in_block][a_slot];
        const double3 xb = s_x[warp_in_block][b_slot];
        double dx = xa.x - xb.x;
        double dy = xa.y - xb.y;
        double dz = xa.z - xb.z;
        if (periodic) {
          const double hbox = 0.5 * box;
          if (dx > hbox) dx -= box; else if (dx < -hbox) dx += box;
          if (dy > hbox) dy -= box; else if (dy < -hbox) dy += box;
          if (dz > hbox) dz -= box; else if (dz < -hbox) dz += box;
        }
        const double r2_phys = (dx * dx + dy * dy + dz * dz) * (pos_to_phys * pos_to_phys);
        loc_tested += 1;
        if (r2_phys < hs_phys * hs_phys) {
          float3 va = s_v[warp_in_block][a_slot];
          float3 vb = s_v[warp_in_block][b_slot];
          const double dvx = (double(va.x) - double(vb.x)) * vel_to_phys;
          const double dvy = (double(va.y) - double(vb.y)) * vel_to_phys;
          const double dvz = (double(va.z) - double(vb.z)) * vel_to_phys;
          const double v_rel = sqrt(dvx * dvx + dvy * dvy + dvz * dvz);
          const double sig = grav_sidm_sigma_of_v(sigma0_m, w0_kms, v_rel);
          const float ma = s_m[warp_in_block][a_slot];
          const float mb = s_m[warp_in_block][b_slot];
          const double prob = grav_sidm_pair_prob(r2_phys, hs_phys, 0.5 * (double(ma) + double(mb)),
                                                  sig, v_rel, dt_phys);
          if (prob > loc_max_p) loc_max_p = prob;
          if (prob > 0.0) {
            const uint64_t key = step_seed
                ^ (uint64_t(s_idx[warp_in_block][a_slot]) * 0x9e3779b97f4a7c15ULL)
                ^ (uint64_t(s_idx[warp_in_block][b_slot]) * 0xbf58476d1ce4e5b9ULL);
            const uint64_t h0 = grav_sidm_splitmix64(key);
            if (grav_sidm_u01(h0) < prob) {
              const uint64_t h1 = grav_sidm_splitmix64(h0);
              const uint64_t h2 = grav_sidm_splitmix64(h1);
              double dE = 0.0;
              grav_sidm_elastic_kick(va, vb, ma, mb, grav_sidm_u01(h1), grav_sidm_u01(h2), dE);
              s_v[warp_in_block][a_slot] = va;
              s_v[warp_in_block][b_slot] = vb;
              loc_scatters += 1;
              loc_dE += dE * (vel_to_phys * vel_to_phys);
            }
          }
        }
      }
    }
    __syncwarp();
  }

  // Write updated velocities back to global memory
  for (int k = lane; k < m; k += 32) {
    vel[s_idx[warp_in_block][k]] = s_v[warp_in_block][k];
  }

  // Warp reduction for stats
  for (int off = 16; off > 0; off >>= 1) {
    loc_scatters += __shfl_down_sync(0xffffffff, loc_scatters, off);
    loc_tested += __shfl_down_sync(0xffffffff, loc_tested, off);
    double other_p = __shfl_down_sync(0xffffffff, loc_max_p, off);
    if (other_p > loc_max_p) loc_max_p = other_p;
    loc_dE += __shfl_down_sync(0xffffffff, loc_dE, off);
  }
  if (lane == 0 && stats) {
    if (loc_scatters) atomicAdd(&stats->n_scatters, loc_scatters);
    if (loc_tested) atomicAdd(&stats->n_pairs_tested, loc_tested);
    if (loc_dE != 0.0) atomicAdd(&stats->sum_dE_kin, loc_dE);
    // Track max_prob via atomicCAS on uint64 bitcast
    unsigned long long* addr = reinterpret_cast<unsigned long long*>(&stats->max_prob);
    unsigned long long old_bits = *addr;
    while (__longlong_as_double(old_bits) < loc_max_p) {
      unsigned long long prev = atomicCAS(addr, old_bits, __double_as_longlong(loc_max_p));
      if (prev == old_bits) break;
      old_bits = prev;
    }
  }
}
#endif  // __CUDACC__

#endif  // GRAV_SIDM_CUH
