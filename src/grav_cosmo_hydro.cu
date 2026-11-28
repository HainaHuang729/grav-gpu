// Full-GPU Comoving Cosmological Coupled PM + SPH Hydrodynamics Solver (gravityCU).
// Evolves PartType0 (Gas) + PartType1 (Dark Matter) simultaneously in a periodic expanding
// Friedmann box using:
//   - High-order GPU cuFFT PM (CIC/TSC/PCS + Interlacing + -ik via grav_cuda_pm.cu),
//     automatically inheriting B1 massive neutrinos (W_nu) and B2 f(R) modified gravity!
//   - Full-GPU 3D Wendland C2 SPH (spatial-hash neighbor list with periodic cell deduplication,
//     adaptive h_i + grad-h Omega_i, Monaghan viscosity + Balsara switch, Price conduction)
//   - Exact analytical Hubble adiabatic expansion u(a) ~ a^(-3*(gamma-1))
//   - Implicit subcycled Sutherland-Dopita radiative cooling + Schmidt-law star formation.

#include "grav_cuda_pm.h"
#include "grav_pm_config.cuh"
#include "grav_config_file.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double G_INTERNAL = 43.0092;  // Mpc, 10^10 Msun, km/s
constexpr double RHO_CRIT_0_H1 = 27.7536627;  // 10^10 Msun / Mpc^3 for h=1
// Conversion: u [(km/s)^2] -> T [K] for primordial ionized gas (mu = 0.6, gamma = 5/3):
//   T = (gamma - 1) * mu * m_p / k_B * u = (2/3) * 0.6 * (1.67262e-24 / 1.38065e-16) * 1e10 * u
//     = 48.4589 * u [(km/s)^2]
constexpr double U_KMS2_TO_KELVIN = 48.4589;

#define CH_CHECK(call)                                                         \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "COSMO_HYDRO_CUDA_ERR line=%d: %s\n", __LINE__,     \
                   cudaGetErrorString(err__));                                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct CosmoHydroHeader {
  int32_t n_gas;
  int32_t n_dm;
  double box;
  double a_init;
  double gamma;
  double omega_m;
  double omega_b;
  double omega_de;
  double hubble0;
};

struct CosmoGrid {
  int n;
  double dx;
  double box;
};

__device__ __forceinline__ double wendland_c2_W(double r, double h) {
  const double q = r / h;
  if (q >= 2.0) return 0.0;
  const double t = 1.0 - 0.5 * q;
  const double t2 = t * t;
  return (21.0 / (16.0 * PI * h * h * h)) * t2 * t2 * (1.0 + 2.0 * q);
}

__device__ __forceinline__ double wendland_c2_dW_over_r(double r, double h) {
  const double q = r / h;
  if (q >= 2.0) return 0.0;
  const double t = 1.0 - 0.5 * q;
  const double h2 = h * h;
  return -(105.0 / (16.0 * PI * h2 * h2 * h)) * t * t * t;
}

__device__ __forceinline__ double wrap_d(double d, double L) {
  const double hL = 0.5 * L;
  if (d > hL) d -= L;
  else if (d < -hL) d += L;
  return d;
}

__device__ __forceinline__ double wrap_p(double x, double L) {
  if (x < 0.0 || x >= L) {
    x -= floor(x / L) * L;
    if (x >= L) x = 0.0;
  }
  return x;
}

__global__ void cell_keys_kernel(const double3* __restrict__ pos,
                                 int* __restrict__ keys,
                                 int* __restrict__ idx,
                                 int n, CosmoGrid g) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double3 p = pos[i];
  const int cx = max(0, min(g.n - 1, int(floor(p.x / g.dx))));
  const int cy = max(0, min(g.n - 1, int(floor(p.y / g.dx))));
  const int cz = max(0, min(g.n - 1, int(floor(p.z / g.dx))));
  keys[i] = (cx * g.n + cy) * g.n + cz;
  idx[i] = i;
}

__global__ void cell_bounds_kernel(const int* __restrict__ keys,
                                   int* __restrict__ cstart,
                                   int* __restrict__ cend,
                                   int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int k = keys[i];
  if (i == 0 || k != keys[i - 1]) cstart[k] = i;
  if (i == n - 1 || k != keys[i + 1]) cend[k] = i + 1;
}

// Comoving SPH density, grad-h Omega_i, div(p), |curl(p)|, and adaptive h_com update
__global__ void cosmo_sph_density_kernel(
    const double3* __restrict__ pos,
    const double3* __restrict__ mom,  // p = a * v_pec
    const float* __restrict__ mass,
    double* __restrict__ h_arr,
    double* __restrict__ rho_arr,
    double* __restrict__ omega_arr,
    double* __restrict__ div_arr,
    double* __restrict__ curl_arr,
    const int* __restrict__ sorted_idx,
    const int* __restrict__ cstart,
    const int* __restrict__ cend,
    int n, CosmoGrid g,
    double eta_sph, double h_min, double h_max, int update_h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const double3 xi = pos[i];
  const double3 pi = mom[i];
  const double hi = h_arr[i];
  const double H = 2.0 * hi;
  const double H2 = H * H;

  const int cx0 = max(0, min(g.n - 1, int(floor(xi.x / g.dx))));
  const int cy0 = max(0, min(g.n - 1, int(floor(xi.y / g.dx))));
  const int cz0 = max(0, min(g.n - 1, int(floor(xi.z / g.dx))));
  const int r_cell = max(1, int(ceil(H / g.dx)));
  const int o_lo = -min(r_cell, g.n / 2);
  const int o_hi =  min(r_cell, (g.n - 1) / 2);

  double rho = 0.0, dh_sum = 0.0, div_p = 0.0;
  double cx_v = 0.0, cy_v = 0.0, cz_v = 0.0;

  for (int ox = o_lo; ox <= o_hi; ++ox) {
    const int cx = (cx0 + ox + g.n) % g.n;
    for (int oy = o_lo; oy <= o_hi; ++oy) {
      const int cy = (cy0 + oy + g.n) % g.n;
      for (int oz = o_lo; oz <= o_hi; ++oz) {
        const int cz = (cz0 + oz + g.n) % g.n;
        const int cid = (cx * g.n + cy) * g.n + cz;
        const int s = cstart[cid];
        if (s < 0) continue;
        const int e = cend[cid];
        for (int idx_p = s; idx_p < e; ++idx_p) {
          const int j = sorted_idx[idx_p];
          const double3 xj = pos[j];
          const double dx = wrap_d(xi.x - xj.x, g.box);
          const double dy = wrap_d(xi.y - xj.y, g.box);
          const double dz = wrap_d(xi.z - xj.z, g.box);
          const double r2 = dx * dx + dy * dy + dz * dz;
          if (r2 < H2) {
            const double r = sqrt(r2);
            const double mj = double(mass[j]);
            const double W = wendland_c2_W(r, hi);
            const double dW_r = wendland_c2_dW_over_r(r, hi);
            rho += mj * W;
            dh_sum += mj * (-(3.0 * W + r2 * dW_r) / hi);
            if (j != i && r2 > 1e-30) {
              const double3 pj = mom[j];
              const double dpx = pj.x - pi.x;
              const double dpy = pj.y - pi.y;
              const double dpz = pj.z - pi.z;
              const double gx = dx * dW_r;
              const double gy = dy * dW_r;
              const double gz = dz * dW_r;
              div_p += mj * (dpx * gx + dpy * gy + dpz * gz);
              cx_v += mj * (dpy * gz - dpz * gy);
              cy_v += mj * (dpz * gx - dpx * gz);
              cz_v += mj * (dpx * gy - dpy * gx);
            }
          }
        }
      }
    }
  }

  rho = fmax(rho, 1e-30);
  rho_arr[i] = rho;
  omega_arr[i] = fmax(0.5, fmin(2.0, 1.0 + (hi / (3.0 * rho)) * dh_sum));
  div_arr[i] = div_p / rho;
  curl_arr[i] = sqrt(cx_v * cx_v + cy_v * cy_v + cz_v * cz_v) / rho;

  if (update_h) {
    const double h_tgt = eta_sph * cbrt(double(mass[i]) / rho);
    h_arr[i] = 0.5 * hi + 0.5 * fmax(h_min, fmin(h_max, h_tgt));
  }
}

__global__ void cosmo_sph_eos_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ u,
    const double* __restrict__ div_p,
    const double* __restrict__ curl_p,
    const double* __restrict__ h_arr,
    double* __restrict__ P_com_arr,
    double* __restrict__ cs_arr,
    double* __restrict__ bal_arr,
    int n, double gamma_val, double a, int use_balsara) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double ui = fmax(u[i], 1e-12);
  P_com_arr[i] = (gamma_val - 1.0) * rho[i] * ui;
  const double cs = sqrt(gamma_val * (gamma_val - 1.0) * ui);
  cs_arr[i] = cs;
  if (use_balsara) {
    // div_p / a^2 has same units as cs / (a * h) -> compare |div_p| vs a * cs / h
    const double ad = fabs(div_p[i]);
    const double ac = fabs(curl_p[i]);
    bal_arr[i] = ad / (ad + ac + 1e-4 * a * cs / fmax(h_arr[i], 1e-12));
  } else {
    bal_arr[i] = 1.0;
  }
}

// Comoving SPH hydro dp/dt and du/dt (p = a * v_pec)
//   (dp_i/dt)_hydro = -sum_j m_j [ P_com,i/(om_i*rho_i^2) grad W_i + P_com,j/(om_j*rho_j^2) grad W_j + Pi_com grad W_avg ]
//   (du_i/dt)_hydro = (1/a^2) * sum_j m_j [ P_com,i/(om_i*rho_i^2) dWi + 0.5*Pi_com*dW_avg ] * (p_i - p_j).r_ij + cond
__global__ void cosmo_sph_force_kernel(
    const double3* __restrict__ pos,
    const double3* __restrict__ mom,
    const float* __restrict__ mass,
    const double* __restrict__ h_arr,
    const double* __restrict__ rho_arr,
    const double* __restrict__ omega_arr,
    const double* __restrict__ P_com_arr,
    const double* __restrict__ cs_arr,
    const double* __restrict__ u_arr,
    const double* __restrict__ bal_arr,
    double3* __restrict__ dp_dt_hydro,
    double* __restrict__ du_dt_hydro,
    double* __restrict__ dt_cfl_arr,
    const int* __restrict__ sorted_idx,
    const int* __restrict__ cstart,
    const int* __restrict__ cend,
    int n, CosmoGrid g,
    double a, double H_a,
    double alpha_visc, double alpha_cond, double cfl, double h_search_max) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const double3 xi = pos[i];
  const double3 pi = mom[i];
  const double hi = h_arr[i];
  const double rhoi = rho_arr[i];
  const double om_i = omega_arr[i];
  const double Pi = P_com_arr[i];
  const double csi = cs_arr[i];
  const double ui = u_arr[i];
  const double fi = bal_arr[i];
  const double P_rho2_i = Pi / (om_i * rhoi * rhoi);
  const double inv_a = 1.0 / a;
  const double inv_a2 = inv_a * inv_a;

  const double H_scan = 2.0 * fmax(hi, h_search_max);
  const int cx0 = max(0, min(g.n - 1, int(floor(xi.x / g.dx))));
  const int cy0 = max(0, min(g.n - 1, int(floor(xi.y / g.dx))));
  const int cz0 = max(0, min(g.n - 1, int(floor(xi.z / g.dx))));
  const int r_cell = max(1, int(ceil(H_scan / g.dx)));
  const int o_lo = -min(r_cell, g.n / 2);
  const int o_hi =  min(r_cell, (g.n - 1) / 2);

  double ax = 0.0, ay = 0.0, az = 0.0;
  double du = 0.0;
  double vsig_max = 2.0 * csi;

  for (int ox = o_lo; ox <= o_hi; ++ox) {
    const int cx = (cx0 + ox + g.n) % g.n;
    for (int oy = o_lo; oy <= o_hi; ++oy) {
      const int cy = (cy0 + oy + g.n) % g.n;
      for (int oz = o_lo; oz <= o_hi; ++oz) {
        const int cz = (cz0 + oz + g.n) % g.n;
        const int cid = (cx * g.n + cy) * g.n + cz;
        const int s = cstart[cid];
        if (s < 0) continue;
        const int e = cend[cid];
        for (int idx_p = s; idx_p < e; ++idx_p) {
          const int j = sorted_idx[idx_p];
          if (j == i) continue;
          const double hj = h_arr[j];
          const double H_pair = 2.0 * fmax(hi, hj);
          const double3 xj = pos[j];
          const double dx = wrap_d(xi.x - xj.x, g.box);
          const double dy = wrap_d(xi.y - xj.y, g.box);
          const double dz = wrap_d(xi.z - xj.z, g.box);
          const double r2 = dx * dx + dy * dy + dz * dz;
          if (r2 >= H_pair * H_pair || r2 <= 1e-30) continue;

          const double r = sqrt(r2);
          const double mj = double(mass[j]);
          const double rhoj = rho_arr[j];
          const double om_j = omega_arr[j];
          const double Pj = P_com_arr[j];
          const double csj = cs_arr[j];
          const double uj = u_arr[j];
          const double fj = bal_arr[j];

          const double dWi_r = wendland_c2_dW_over_r(r, hi);
          const double dWj_r = wendland_c2_dW_over_r(r, hj);
          const double dW_avg_r = 0.5 * (dWi_r + dWj_r);

          const double3 pj = mom[j];
          const double dpx = pi.x - pj.x;
          const double dpy = pi.y - pj.y;
          const double dpz = pi.z - pj.z;
          const double dp_dot_r = dpx * dx + dpy * dy + dpz * dz;
          // Physical radial approaching speed: v_phys.r_hat = (dp.r)/(a*r) + a*H*r
          const double w_phys = dp_dot_r * inv_a / r + a * H_a * r;

          const double rho_bar = 0.5 * (rhoi + rhoj);
          double Pi_visc = 0.0;
          double vsig = csi + csj;
          if (w_phys < 0.0) {
            vsig = csi + csj - 3.0 * w_phys;
            const double f_bar = 0.5 * (fi + fj);
            Pi_visc = -0.5 * alpha_visc * f_bar * vsig * w_phys / rho_bar;
          }
          if (vsig > vsig_max) vsig_max = vsig;

          const double P_rho2_j = Pj / (om_j * rhoj * rhoj);
          const double f_sc = -mj * (P_rho2_i * dWi_r + P_rho2_j * dWj_r + Pi_visc * dW_avg_r);
          ax += f_sc * dx;
          ay += f_sc * dy;
          az += f_sc * dz;

          // Physical PdV + viscous heating (dp_dot_r * inv_a2 = (v_phys_pec).dx / a)
          du += mj * (P_rho2_i * dWi_r + 0.5 * Pi_visc * dW_avg_r) * (dp_dot_r * inv_a2);

          if (alpha_cond > 0.0) {
            const double vsig_u = sqrt(fabs(Pi - Pj) / rho_bar);
            du += mj * alpha_cond * vsig_u * (ui - uj) * (r * dW_avg_r) * inv_a / rho_bar;
          }
        }
      }
    }
  }

  dp_dt_hydro[i] = {ax, ay, az};
  du_dt_hydro[i] = du;
  // Physical CFL dt = cfl * (a * h_com) / vsig_max
  dt_cfl_arr[i] = cfl * (a * hi) / fmax(vsig_max, 1e-6);
}

// Cosmological Sutherland-Dopita cooling + simple Schmidt-law star formation
__global__ void cosmo_cooling_sf_kernel(
    const double* __restrict__ rho_com,
    float* __restrict__ mass_gas,
    double* __restrict__ mass_star,
    double* __restrict__ u_arr,
    double* __restrict__ rad_lost,
    int n, double a, double dt_phys,
    double rho_b_mean_com,
    int enable_cooling, double cool_scale, double u_floor,
    int enable_sf, double sf_overdensity, double sf_u_max, double sf_eps) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double rho_phys = rho_com[i] / (a * a * a);
  double u = u_arr[i];

  if (enable_cooling && u > u_floor) {
    const double u0 = u;
    double t_rem = dt_phys;
    for (int sub = 0; sub < 32 && t_rem > 0.0 && u > u_floor; ++sub) {
      const double T = U_KMS2_TO_KELVIN * u;
      if (T <= 1.0e4) break;
      const double lT = log10(T);
      const double peak = exp(-0.5 * ((lT - 5.15) / 0.45) * ((lT - 5.15) / 0.45));
      const double brem = 0.05 * sqrt(T / 1.0e6);
      const double cut = 1.0 / (1.0 + exp(-(lT - 4.05) / 0.05));
      const double rate = cool_scale * rho_phys * cut * (peak + brem);
      if (rate <= 1e-30) break;
      const double dt_s = fmin(t_rem, 0.15 * (u - u_floor) / rate + 1e-12);
      u = fmax(u_floor, u - rate * dt_s);
      t_rem -= dt_s;
    }
    u_arr[i] = u;
    rad_lost[i] += double(mass_gas[i]) * (u0 - u);
  }

  // Simple Schmidt-law star formation: if delta_b > sf_overdensity and u < sf_u_max
  if (enable_sf && rho_com[i] > sf_overdensity * rho_b_mean_com && u < sf_u_max) {
    const double mg = double(mass_gas[i]);
    if (mg > 1e-12) {
      // t_dyn = 1 / sqrt(G * rho_phys) in Mpc / (km/s)
      const double t_dyn = 1.0 / sqrt(fmax(G_INTERNAL * rho_phys, 1e-30));
      const double frac = 1.0 - exp(-sf_eps * dt_phys / t_dyn);
      const double dm = mg * fmin(0.5, fmax(0.0, frac));
      mass_gas[i] = float(mg - dm);
      mass_star[i] += dm;
    }
  }
}

__global__ void kick_gas_kernel(
    double3* __restrict__ mom,
    double* __restrict__ u_arr,
    const float3* __restrict__ grav_acc,
    const double3* __restrict__ dp_hydro,
    const double* __restrict__ du_hydro,
    int n, double dt_grav_kick, double dt_phys_kick, double u_floor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // grav_acc from grav_pm_cuda_interpolate is -grad_x Phi_com -> dp/dt_grav = grav_acc / a
  // dt_grav_kick = dt_phys / a = int da / (a^2 H)
  mom[i].x += double(grav_acc[i].x) * dt_grav_kick + dp_hydro[i].x * dt_phys_kick;
  mom[i].y += double(grav_acc[i].y) * dt_grav_kick + dp_hydro[i].y * dt_phys_kick;
  mom[i].z += double(grav_acc[i].z) * dt_grav_kick + dp_hydro[i].z * dt_phys_kick;
  u_arr[i] = fmax(u_floor, u_arr[i] + du_hydro[i] * dt_phys_kick);
}

__global__ void kick_dm_kernel(
    double3* __restrict__ mom,
    const float3* __restrict__ grav_acc,
    int n, double dt_grav_kick) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  mom[i].x += double(grav_acc[i].x) * dt_grav_kick;
  mom[i].y += double(grav_acc[i].y) * dt_grav_kick;
  mom[i].z += double(grav_acc[i].z) * dt_grav_kick;
}

__global__ void drift_gas_kernel(
    double3* __restrict__ pos,
    const double3* __restrict__ mom,
    double* __restrict__ u_arr,
    int n, double dt_drift, double hubble_u_fac, double u_floor, double box) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  // dx/dt = v_pec / a = p / a^2 -> dt_drift = int da / (a^3 H)
  pos[i].x = wrap_p(pos[i].x + mom[i].x * dt_drift, box);
  pos[i].y = wrap_p(pos[i].y + mom[i].y * dt_drift, box);
  pos[i].z = wrap_p(pos[i].z + mom[i].z * dt_drift, box);
  // Exact adiabatic Hubble expansion: u *= (a_old / a_new)^(3*(gamma - 1))
  u_arr[i] = fmax(u_floor, u_arr[i] * hubble_u_fac);
}

__global__ void drift_dm_kernel(
    double3* __restrict__ pos,
    const double3* __restrict__ mom,
    int n, double dt_drift, double box) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  pos[i].x = wrap_p(pos[i].x + mom[i].x * dt_drift, box);
  pos[i].y = wrap_p(pos[i].y + mom[i].y * dt_drift, box);
  pos[i].z = wrap_p(pos[i].z + mom[i].z * dt_drift, box);
}

static double env_d(const char* k, double def) {
  const char* s = std::getenv(k);
  return (s && *s) ? std::atof(s) : def;
}
static int env_i(const char* k, int def) {
  const char* s = std::getenv(k);
  return (s && *s) ? std::atoi(s) : def;
}

static double hubble_H(double a, double H0, double om, double ode, double om_nu, double m_each) {
  const double a3 = 1.0 / (a * a * a);
  double mat = om * a3;
  if (om_nu > 0.0) {
    mat = (om - om_nu) * a3 + om_nu * a3 * grav_nu_rho_enhancement(a, m_each);
  }
  return H0 * std::sqrt(mat + ode);
}

// Simpson integration of int_{a0}^{a1} da / (a^p * H(a))
static double simpson_factor(double a0, double a1, int p_pow,
                             double H0, double om, double ode, double om_nu, double m_each) {
  if (a1 <= a0) return 0.0;
  const double am = 0.5 * (a0 + a1);
  auto f = [&](double a) {
    return 1.0 / (std::pow(a, p_pow) * hubble_H(a, H0, om, ode, om_nu, m_each));
  };
  return (a1 - a0) / 6.0 * (f(a0) + 4.0 * f(am) + f(a1));
}

}  // namespace

int main(int argc, char** argv) {
  grav_cfg::apply_cli_and_env_config(argc, argv);
  if (argc < 4) {
    std::fprintf(stderr, "Usage: %s <ic.bin> <out.bin> <a_end>\n", argv[0]);
    return 1;
  }
  const char* ic_path = argv[1];
  const char* out_path = argv[2];
  const double a_end = std::atof(argv[3]);
  const auto t0_wall = std::chrono::steady_clock::now();

  FILE* fin = std::fopen(ic_path, "rb");
  if (!fin) { std::fprintf(stderr, "Cannot open %s\n", ic_path); return 2; }
  CosmoHydroHeader hdr{};
  if (std::fread(&hdr, sizeof(hdr), 1, fin) != 1 || hdr.n_gas <= 0) {
    std::fprintf(stderr, "Invalid CosmoHydroHeader\n");
    std::fclose(fin);
    return 3;
  }
  const int n_gas = hdr.n_gas;
  const int n_dm = hdr.n_dm;
  std::vector<double3> h_gx(n_gas), h_gp(n_gas);
  std::vector<float> h_gm(n_gas);
  std::vector<double> h_gu(n_gas), h_gh(n_gas);
  auto must_read = [&](void* dst, size_t sz, size_t cnt) {
    if (std::fread(dst, sz, cnt, fin) != cnt) {
      std::fprintf(stderr, "Truncated cosmo hydro IC\n");
      std::exit(3);
    }
  };
  must_read(h_gx.data(), sizeof(double3), n_gas);
  must_read(h_gp.data(), sizeof(double3), n_gas);
  must_read(h_gm.data(), sizeof(float), n_gas);
  must_read(h_gu.data(), sizeof(double), n_gas);
  must_read(h_gh.data(), sizeof(double), n_gas);

  std::vector<double3> h_dx(n_dm), h_dp(n_dm);
  std::vector<float> h_dm(n_dm);
  if (n_dm > 0) {
    must_read(h_dx.data(), sizeof(double3), n_dm);
    must_read(h_dp.data(), sizeof(double3), n_dm);
    must_read(h_dm.data(), sizeof(float), n_dm);
  }
  std::fclose(fin);

  // Convert input peculiar velocity v_pec -> canonical momentum p = a_init * v_pec
  for (int i = 0; i < n_gas; ++i) {
    h_gp[i].x *= hdr.a_init; h_gp[i].y *= hdr.a_init; h_gp[i].z *= hdr.a_init;
  }
  for (int i = 0; i < n_dm; ++i) {
    h_dp[i].x *= hdr.a_init; h_dp[i].y *= hdr.a_init; h_dp[i].z *= hdr.a_init;
  }

  const int pm_side = env_i("GRAV_PM_GRID", 128);
  const double alpha_visc = env_d("GRAV_HYDRO_ALPHA_VISC", 1.0);
  const double alpha_cond = env_d("GRAV_HYDRO_ALPHA_COND", 0.15);
  const int use_balsara = env_i("GRAV_HYDRO_BALSARA", 1);
  const double cfl = env_d("GRAV_HYDRO_CFL", 0.25);
  const double max_dlna = env_d("GRAV_COSMO_MAX_DLNA", 0.015);
  const int enable_cool = env_i("GRAV_HYDRO_COOLING", 0);
  const double cool_scale = env_d("GRAV_HYDRO_COOLING_SCALE", 25.0);
  const double u_floor = env_d("GRAV_HYDRO_U_FLOOR", 0.0);  // 0 = pure adiabatic (no floor clamp)
  const int enable_sf = env_i("GRAV_COSMO_SF", 0);
  const double sf_overd = env_d("GRAV_COSMO_SF_OVERDENSITY", 50.0);
  const double sf_u_max = env_d("GRAV_COSMO_SF_U_MAX", 600.0);  // ~ 3e4 K
  const double sf_eps = env_d("GRAV_COSMO_SF_EPS", 0.05);
  const double eta_sph = env_d("GRAV_HYDRO_ETA", 1.2348);

  const GravPmConfig pm_cfg = grav_pm_get_config();
  const GravNuConfig nu_cfg = grav_nu_init_from_env(hdr.hubble0, hdr.omega_m);
  const GravFrConfig fr_cfg = grav_fr_init_from_env(hdr.hubble0, hdr.omega_m, hdr.omega_de);
  (void)fr_cfg;
  // For pure PM gravity, use tiny split r_s -> 0 so full Fourier gravity is included!
  const double pm_split = env_d("GRAV_COSMO_PM_SPLIT", 1e-6);

  std::printf("COSMO_HYDRO_CONFIG n_gas=%d n_dm=%d box=%.2f a=[%.4f->%.4f] pm=%d order=%d int=%d grad=%s cool=%d sf=%d\n",
              n_gas, n_dm, hdr.box, hdr.a_init, a_end, pm_side,
              pm_cfg.order, pm_cfg.interlace, pm_cfg.spectral_grad ? "ik" : "fd4",
              enable_cool, enable_sf);
  std::fflush(stdout);

  // Build spatial-hash grid for SPH
  double h_init_avg = 0.0;
  for (int i = 0; i < n_gas; ++i) h_init_avg += h_gh[i];
  h_init_avg /= double(n_gas);
  CosmoGrid sgrid{};
  sgrid.box = hdr.box;
  sgrid.n = std::max(4, std::min(64, int(std::floor(hdr.box / std::max(2.5 * h_init_avg, hdr.box / 64.0)))));
  sgrid.dx = hdr.box / double(sgrid.n);
  const int n_cells = sgrid.n * sgrid.n * sgrid.n;
  const double h_min = 0.05 * h_init_avg;
  const double h_max = std::min(0.25 * hdr.box, 4.0 * h_init_avg);

  const double rho_b_mean = hdr.omega_b * RHO_CRIT_0_H1 * std::pow(hdr.hubble0 / 100.0, 2);

  // Allocate GPU buffers
  double3 *d_gx, *d_gp, *d_gdp, *d_dx = nullptr, *d_dp = nullptr;
  float *d_gm, *d_dm = nullptr;
  float3 *d_gacc, *d_dacc = nullptr;
  double *d_gu, *d_gh, *d_grho, *d_gom, *d_gdiv, *d_gcurl, *d_gP, *d_gcs, *d_gbal, *d_gdu, *d_gdt;
  double *d_mstar, *d_rad;
  int *d_keys, *d_idx, *d_cstart, *d_cend;
  double *d_pm_rho;
  double4 *d_pm_fld, *d_pm_fld_s = nullptr;

  CH_CHECK(cudaMalloc(&d_gx, size_t(n_gas) * sizeof(double3)));
  CH_CHECK(cudaMalloc(&d_gp, size_t(n_gas) * sizeof(double3)));
  CH_CHECK(cudaMalloc(&d_gdp, size_t(n_gas) * sizeof(double3)));
  CH_CHECK(cudaMalloc(&d_gm, size_t(n_gas) * sizeof(float)));
  CH_CHECK(cudaMalloc(&d_gacc, size_t(n_gas) * sizeof(float3)));
  CH_CHECK(cudaMalloc(&d_gu, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_gh, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_grho, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_gom, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_gdiv, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_gcurl, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_gP, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_gcs, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_gbal, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_gdu, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_gdt, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_mstar, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_rad, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_keys, size_t(n_gas) * sizeof(int)));
  CH_CHECK(cudaMalloc(&d_idx, size_t(n_gas) * sizeof(int)));
  CH_CHECK(cudaMalloc(&d_cstart, size_t(n_cells) * sizeof(int)));
  CH_CHECK(cudaMalloc(&d_cend, size_t(n_cells) * sizeof(int)));

  const size_t pm_pts = size_t(pm_side) * pm_side * pm_side;
  CH_CHECK(cudaMalloc(&d_pm_rho, pm_pts * sizeof(double)));
  CH_CHECK(cudaMalloc(&d_pm_fld, pm_pts * sizeof(double4)));
  if (pm_cfg.interlace) CH_CHECK(cudaMalloc(&d_pm_fld_s, pm_pts * sizeof(double4)));

  CH_CHECK(cudaMemcpy(d_gx, h_gx.data(), size_t(n_gas) * sizeof(double3), cudaMemcpyHostToDevice));
  CH_CHECK(cudaMemcpy(d_gp, h_gp.data(), size_t(n_gas) * sizeof(double3), cudaMemcpyHostToDevice));
  CH_CHECK(cudaMemcpy(d_gm, h_gm.data(), size_t(n_gas) * sizeof(float), cudaMemcpyHostToDevice));
  CH_CHECK(cudaMemcpy(d_gu, h_gu.data(), size_t(n_gas) * sizeof(double), cudaMemcpyHostToDevice));
  CH_CHECK(cudaMemcpy(d_gh, h_gh.data(), size_t(n_gas) * sizeof(double), cudaMemcpyHostToDevice));
  CH_CHECK(cudaMemset(d_mstar, 0, size_t(n_gas) * sizeof(double)));
  CH_CHECK(cudaMemset(d_rad, 0, size_t(n_gas) * sizeof(double)));

  if (n_dm > 0) {
    CH_CHECK(cudaMalloc(&d_dx, size_t(n_dm) * sizeof(double3)));
    CH_CHECK(cudaMalloc(&d_dp, size_t(n_dm) * sizeof(double3)));
    CH_CHECK(cudaMalloc(&d_dm, size_t(n_dm) * sizeof(float)));
    CH_CHECK(cudaMalloc(&d_dacc, size_t(n_dm) * sizeof(float3)));
    CH_CHECK(cudaMemcpy(d_dx, h_dx.data(), size_t(n_dm) * sizeof(double3), cudaMemcpyHostToDevice));
    CH_CHECK(cudaMemcpy(d_dp, h_dp.data(), size_t(n_dm) * sizeof(double3), cudaMemcpyHostToDevice));
    CH_CHECK(cudaMemcpy(d_dm, h_dm.data(), size_t(n_dm) * sizeof(float), cudaMemcpyHostToDevice));
  }

  double m_baryon_init = 0.0;
  for (int i = 0; i < n_gas; ++i) m_baryon_init += double(h_gm[i]);

  const int B = 256;
  const int G_gas = (n_gas + B - 1) / B;
  const int G_dm = (n_dm + B - 1) / B;
  std::vector<double> h_dt_tmp(n_gas), h_h_tmp(n_gas);

  auto eval_forces = [&](double a_cur, double H_cur, int h_iters, double& dt_cfl_min) {
    // 1. Coupled PM Gravity (Gas + DM deposited onto same Fourier grid)
    grav_nu_update_step(a_cur, H_cur);
    grav_fr_update_step(a_cur);
    const int passes = pm_cfg.interlace ? 2 : 1;
    const float iscale = pm_cfg.interlace ? 0.5f : 1.0f;
    for (int pp = 0; pp < passes; ++pp) {
      const double sh = pp ? 0.5 : 0.0;
      double4* fld = pp ? d_pm_fld_s : d_pm_fld;
      CH_CHECK(cudaMemset(d_pm_rho, 0, pm_pts * sizeof(double)));
      grav_pm_cuda_deposit_buf_shift(d_gx, d_gm, 0.f, size_t(n_gas), d_pm_rho, pm_side, hdr.box, sh, nullptr);
      if (n_dm > 0) {
        grav_pm_cuda_deposit_buf_shift(d_dx, d_dm, 0.f, size_t(n_dm), d_pm_rho, pm_side, hdr.box, sh, nullptr);
      }
      grav_pm_cuda_solve_on_device(d_pm_rho, fld, pm_side, hdr.box, pm_split, nullptr);
      grav_pm_cuda_interpolate_buf_shift(d_gx, size_t(n_gas), fld, pm_side, hdr.box, float(G_INTERNAL),
                                         d_gacc, nullptr, sh, iscale, pp, nullptr);
      if (n_dm > 0) {
        grav_pm_cuda_interpolate_buf_shift(d_dx, size_t(n_dm), fld, pm_side, hdr.box, float(G_INTERNAL),
                                           d_dacc, nullptr, sh, iscale, pp, nullptr);
      }
    }

    // 2. Comoving GPU SPH Density + EOS + Hydro Force
    cell_keys_kernel<<<G_gas, B>>>(d_gx, d_keys, d_idx, n_gas, sgrid);
    thrust::sort_by_key(thrust::device_pointer_cast(d_keys),
                        thrust::device_pointer_cast(d_keys + n_gas),
                        thrust::device_pointer_cast(d_idx));
    CH_CHECK(cudaMemset(d_cstart, 0xff, size_t(n_cells) * sizeof(int)));
    CH_CHECK(cudaMemset(d_cend, 0, size_t(n_cells) * sizeof(int)));
    cell_bounds_kernel<<<G_gas, B>>>(d_keys, d_cstart, d_cend, n_gas);

    for (int it = 0; it < h_iters; ++it) {
      cosmo_sph_density_kernel<<<G_gas, B>>>(
          d_gx, d_gp, d_gm, d_gh, d_grho, d_gom, d_gdiv, d_gcurl,
          d_idx, d_cstart, d_cend, n_gas, sgrid,
          eta_sph, h_min, h_max, (it + 1 < h_iters) ? 1 : 0);
    }
    cosmo_sph_eos_kernel<<<G_gas, B>>>(
        d_grho, d_gu, d_gdiv, d_gcurl, d_gh, d_gP, d_gcs, d_gbal,
        n_gas, hdr.gamma, a_cur, use_balsara);

    CH_CHECK(cudaMemcpy(h_h_tmp.data(), d_gh, size_t(n_gas) * sizeof(double), cudaMemcpyDeviceToHost));
    double h_max_cur = h_min;
    for (int i = 0; i < n_gas; ++i) h_max_cur = std::max(h_max_cur, h_h_tmp[i]);

    cosmo_sph_force_kernel<<<G_gas, B>>>(
        d_gx, d_gp, d_gm, d_gh, d_grho, d_gom, d_gP, d_gcs, d_gu, d_gbal,
        d_gdp, d_gdu, d_gdt, d_idx, d_cstart, d_cend, n_gas, sgrid,
        a_cur, H_cur, alpha_visc, alpha_cond, cfl, h_max_cur);

    CH_CHECK(cudaMemcpy(h_dt_tmp.data(), d_gdt, size_t(n_gas) * sizeof(double), cudaMemcpyDeviceToHost));
    dt_cfl_min = 1e30;
    for (int i = 0; i < n_gas; ++i) dt_cfl_min = std::min(dt_cfl_min, h_dt_tmp[i]);
  };

  double a = hdr.a_init;
  double H_cur = hubble_H(a, hdr.hubble0, hdr.omega_m, hdr.omega_de, nu_cfg.omega_nu0, nu_cfg.m_each_ev);
  double dt_cfl = 0.0;
  eval_forces(a, H_cur, 4, dt_cfl);

  int step = 0;
  const int max_steps = env_i("GRAV_COSMO_MAX_STEPS", 4000);
  while (a < a_end - 1e-12 && step < max_steps) {
    // Choose next scale factor from min(max_dlna, H * dt_cfl)
    const double dlna_cfl = std::max(1e-5, H_cur * dt_cfl);
    const double dlna = std::min(max_dlna, dlna_cfl);
    const double a_next = std::min(a_end, a * std::exp(dlna));
    const double a_mid = std::sqrt(a * a_next);

    const double dt_grav_1 = simpson_factor(a, a_mid, 2, hdr.hubble0, hdr.omega_m, hdr.omega_de, nu_cfg.omega_nu0, nu_cfg.m_each_ev);
    const double dt_phys_1 = simpson_factor(a, a_mid, 1, hdr.hubble0, hdr.omega_m, hdr.omega_de, nu_cfg.omega_nu0, nu_cfg.m_each_ev);
    const double dt_drift  = simpson_factor(a, a_next, 3, hdr.hubble0, hdr.omega_m, hdr.omega_de, nu_cfg.omega_nu0, nu_cfg.m_each_ev);
    const double dt_grav_2 = simpson_factor(a_mid, a_next, 2, hdr.hubble0, hdr.omega_m, hdr.omega_de, nu_cfg.omega_nu0, nu_cfg.m_each_ev);
    const double dt_phys_2 = simpson_factor(a_mid, a_next, 1, hdr.hubble0, hdr.omega_m, hdr.omega_de, nu_cfg.omega_nu0, nu_cfg.m_each_ev);
    const double hubble_u_fac = std::pow(a / a_next, 3.0 * (hdr.gamma - 1.0));

    // KDK: Kick(a -> a_mid) -> Drift(a -> a_next) -> Force(a_next) -> Kick(a_mid -> a_next)
    kick_gas_kernel<<<G_gas, B>>>(d_gp, d_gu, d_gacc, d_gdp, d_gdu, n_gas, dt_grav_1, dt_phys_1, u_floor);
    if (n_dm > 0) kick_dm_kernel<<<G_dm, B>>>(d_dp, d_dacc, n_dm, dt_grav_1);

    drift_gas_kernel<<<G_gas, B>>>(d_gx, d_gp, d_gu, n_gas, dt_drift, hubble_u_fac, u_floor, hdr.box);
    if (n_dm > 0) drift_dm_kernel<<<G_dm, B>>>(d_dx, d_dp, n_dm, dt_drift, hdr.box);

    a = a_next;
    H_cur = hubble_H(a, hdr.hubble0, hdr.omega_m, hdr.omega_de, nu_cfg.omega_nu0, nu_cfg.m_each_ev);
    eval_forces(a, H_cur, 2, dt_cfl);

    kick_gas_kernel<<<G_gas, B>>>(d_gp, d_gu, d_gacc, d_gdp, d_gdu, n_gas, dt_grav_2, dt_phys_2, u_floor);
    if (n_dm > 0) kick_dm_kernel<<<G_dm, B>>>(d_dp, d_dacc, n_dm, dt_grav_2);

    if (enable_cool || enable_sf) {
      cosmo_cooling_sf_kernel<<<G_gas, B>>>(
          d_grho, d_gm, d_mstar, d_gu, d_rad, n_gas, a, dt_phys_1 + dt_phys_2,
          rho_b_mean, enable_cool, cool_scale, u_floor > 0.0 ? u_floor : (1.0e4 / U_KMS2_TO_KELVIN),
          enable_sf, sf_overd, sf_u_max, sf_eps);
    }
    ++step;
  }

  // Final density refresh & D2H export
  eval_forces(a, H_cur, 2, dt_cfl);
  std::vector<double> h_grho(n_gas), h_gP(n_gas), h_mstar(n_gas), h_rad(n_gas);
  CH_CHECK(cudaMemcpy(h_gx.data(), d_gx, size_t(n_gas) * sizeof(double3), cudaMemcpyDeviceToHost));
  CH_CHECK(cudaMemcpy(h_gp.data(), d_gp, size_t(n_gas) * sizeof(double3), cudaMemcpyDeviceToHost));
  CH_CHECK(cudaMemcpy(h_gm.data(), d_gm, size_t(n_gas) * sizeof(float), cudaMemcpyDeviceToHost));
  CH_CHECK(cudaMemcpy(h_gu.data(), d_gu, size_t(n_gas) * sizeof(double), cudaMemcpyDeviceToHost));
  CH_CHECK(cudaMemcpy(h_grho.data(), d_grho, size_t(n_gas) * sizeof(double), cudaMemcpyDeviceToHost));
  CH_CHECK(cudaMemcpy(h_gh.data(), d_gh, size_t(n_gas) * sizeof(double), cudaMemcpyDeviceToHost));
  CH_CHECK(cudaMemcpy(h_gP.data(), d_gP, size_t(n_gas) * sizeof(double), cudaMemcpyDeviceToHost));
  CH_CHECK(cudaMemcpy(h_mstar.data(), d_mstar, size_t(n_gas) * sizeof(double), cudaMemcpyDeviceToHost));
  CH_CHECK(cudaMemcpy(h_rad.data(), d_rad, size_t(n_gas) * sizeof(double), cudaMemcpyDeviceToHost));
  if (n_dm > 0) {
    CH_CHECK(cudaMemcpy(h_dx.data(), d_dx, size_t(n_dm) * sizeof(double3), cudaMemcpyDeviceToHost));
    CH_CHECK(cudaMemcpy(h_dp.data(), d_dp, size_t(n_dm) * sizeof(double3), cudaMemcpyDeviceToHost));
  }

  // Convert canonical momentum back to peculiar velocity v_pec = p / a
  const double inv_a_end = 1.0 / a;
  for (int i = 0; i < n_gas; ++i) {
    h_gp[i].x *= inv_a_end; h_gp[i].y *= inv_a_end; h_gp[i].z *= inv_a_end;
  }
  for (int i = 0; i < n_dm; ++i) {
    h_dp[i].x *= inv_a_end; h_dp[i].y *= inv_a_end; h_dp[i].z *= inv_a_end;
  }

  double m_gas_end = 0.0, m_star_end = 0.0, rad_tot = 0.0;
  for (int i = 0; i < n_gas; ++i) {
    m_gas_end += double(h_gm[i]);
    m_star_end += h_mstar[i];
    rad_tot += h_rad[i];
  }

  hdr.a_init = a;
  FILE* fout = std::fopen(out_path, "wb");
  if (!fout) { std::fprintf(stderr, "Cannot open %s\n", out_path); return 4; }
  std::fwrite(&hdr, sizeof(hdr), 1, fout);
  std::fwrite(h_gx.data(), sizeof(double3), n_gas, fout);
  std::fwrite(h_gp.data(), sizeof(double3), n_gas, fout);
  std::fwrite(h_gm.data(), sizeof(float), n_gas, fout);
  std::fwrite(h_gu.data(), sizeof(double), n_gas, fout);
  std::fwrite(h_grho.data(), sizeof(double), n_gas, fout);
  std::fwrite(h_mstar.data(), sizeof(double), n_gas, fout);
  if (n_dm > 0) {
    std::fwrite(h_dx.data(), sizeof(double3), n_dm, fout);
    std::fwrite(h_dp.data(), sizeof(double3), n_dm, fout);
  }
  std::fclose(fout);

  const double wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_wall).count();
  std::cout << std::setprecision(10)
            << "COSMO_HYDRO_RESULT {\"steps\":" << step
            << ",\"a_end\":" << a
            << ",\"wall_s\":" << wall_s
            << ",\"n_gas\":" << n_gas
            << ",\"n_dm\":" << n_dm
            << ",\"m_baryon_init\":" << m_baryon_init
            << ",\"m_gas_end\":" << m_gas_end
            << ",\"m_star_end\":" << m_star_end
            << ",\"baryon_mass_rel_err\":" << std::fabs((m_gas_end + m_star_end) - m_baryon_init) / m_baryon_init
            << ",\"E_rad_tot\":" << rad_tot
            << "}" << std::endl;
  return 0;
}
