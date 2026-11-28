#include "grav_cuda_hydro.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

#define HYDRO_CHECK(call)                                                      \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr, "HYDRO_CUDA_ERR line=%d: %s\n", __LINE__,           \
                   cudaGetErrorString(err__));                                 \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// 3D Wendland C2 kernel with support radius H = 2*h (q = r / h in [0, 2]):
//   W(q, h) = (21 / (16 * pi * h^3)) * (1 - q/2)^4 * (1 + 2*q)  -- wait: check normalization!
// Integral_0^2 4*pi*r^2 * (1 - q/2)^4 * (1 + 2*q) dr = 4*pi*h^3 * (16 / 21) -> norm = 21 / (16 * pi * h^3)!
// Wait: let's verify exact integral: let u = q/2 in [0,1], q = 2u, dq = 2du:
//   4*pi*h^3 * int_0^1 (2u)^2 * (1-u)^4 * (1+4u) * 2 du
//   = 32*pi*h^3 * int_0^1 u^2 (1-u)^4 (1+4u) du
//   int_0^1 u^2 (1-u)^4 du = 2!4!/7! = 1/105
//   int_0^1 u^3 (1-u)^4 du = 3!4!/8! = 1/280
//   sum = 1/105 + 4/280 = 1/105 + 1/70 = (2 + 3)/210 = 5/210 = 1/42.
//   32*pi*h^3 / 42 = (16/21)*pi*h^3 -> EXACT normalization = 21 / (16 * pi * h^3)!!
// Derivative dW/dr = (1/h) * dW/dq:
//   d/dq [(1 - q/2)^4 (1 + 2q)] = -2(1 - q/2)^3(1 + 2q) + 2(1 - q/2)^4
//     = 2(1 - q/2)^3 * [ (1 - q/2) - (1 + 2q) ] = -5 * q * (1 - q/2)^3.
//   So dW/dr = (21 / (16 * pi * h^4)) * (-5 * q * (1 - q/2)^3) = -(105 / (16 * pi * h^4)) * q * (1 - q/2)^3.
//   And (1/r) dW/dr = -(105 / (16 * pi * h^5)) * (1 - q/2)^3.
__device__ __forceinline__ double wendland_c2_W(double r, double h) {
  const double q = r / h;
  if (q >= 2.0) return 0.0;
  const double t = 1.0 - 0.5 * q;
  const double t2 = t * t;
  const double norm = 21.0 / (16.0 * PI * h * h * h);
  return norm * t2 * t2 * (1.0 + 2.0 * q);
}

// Returns (1/r) * dW/dr (so grad_i W = (r_i - r_j) * wendland_c2_dW_over_r)
__device__ __forceinline__ double wendland_c2_dW_over_r(double r, double h) {
  const double q = r / h;
  if (q >= 2.0) return 0.0;
  const double t = 1.0 - 0.5 * q;
  const double h2 = h * h;
  const double h5 = h2 * h2 * h;
  return -(105.0 / (16.0 * PI * h5)) * t * t * t;
}

__device__ __forceinline__ double wrap_delta(double d, double L, int per) {
  if (!per) return d;
  const double hL = 0.5 * L;
  if (d > hL) d -= L;
  else if (d < -hL) d += L;
  return d;
}

__device__ __forceinline__ double wrap_pos(double x, double L, int per) {
  if (!per) return x;
  if (x < 0.0 || x >= L) {
    x -= floor(x / L) * L;
    if (x >= L) x = 0.0;
  }
  return x;
}

struct CellGrid {
  int nx, ny, nz;
  double dx, dy, dz;
  double box[3];
  int periodic[3];
};

__global__ void compute_cell_keys_kernel(
    const double3* __restrict__ pos,
    int* __restrict__ keys,
    int* __restrict__ indices,
    int n,
    CellGrid g) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double3 p = pos[i];
  int cx = int(floor(p.x / g.dx));
  int cy = int(floor(p.y / g.dy));
  int cz = int(floor(p.z / g.dz));
  cx = max(0, min(g.nx - 1, cx));
  cy = max(0, min(g.ny - 1, cy));
  cz = max(0, min(g.nz - 1, cz));
  keys[i] = (cx * g.ny + cy) * g.nz + cz;
  indices[i] = i;
}

__global__ void find_cell_bounds_kernel(
    const int* __restrict__ sorted_keys,
    int* __restrict__ cell_start,
    int* __restrict__ cell_end,
    int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int k = sorted_keys[i];
  if (i == 0 || k != sorted_keys[i - 1]) cell_start[k] = i;
  if (i == n - 1 || k != sorted_keys[i + 1]) cell_end[k] = i + 1;
}

// Compute density, Omega_i (grad-h), velocity divergence, curl, and update h_i
__global__ void sph_density_kernel(
    const double3* __restrict__ pos,
    const double3* __restrict__ vel,
    const double* __restrict__ mass,
    double* __restrict__ h_arr,
    double* __restrict__ rho_arr,
    double* __restrict__ omega_arr,
    double* __restrict__ div_v_arr,
    double* __restrict__ curl_v_arr,
    const int* __restrict__ sorted_idx,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    int n,
    CellGrid g,
    double eta_sph,
    double h_min,
    double h_max,
    int update_h) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const double3 xi = pos[i];
  const double3 vi = vel[i];
  const double hi = h_arr[i];
  const double H = 2.0 * hi;
  const double H2 = H * H;

  const int cx0 = max(0, min(g.nx - 1, int(floor(xi.x / g.dx))));
  const int cy0 = max(0, min(g.ny - 1, int(floor(xi.y / g.dy))));
  const int cz0 = max(0, min(g.nz - 1, int(floor(xi.z / g.dz))));

  const int rx = max(1, int(ceil(H / g.dx)));
  const int ry = max(1, int(ceil(H / g.dy)));
  const int rz = max(1, int(ceil(H / g.dz)));
  const int ox_lo = g.periodic[0] ? -min(rx, g.nx / 2) : -rx;
  const int ox_hi = g.periodic[0] ?  min(rx, (g.nx - 1) / 2) : rx;
  const int oy_lo = g.periodic[1] ? -min(ry, g.ny / 2) : -ry;
  const int oy_hi = g.periodic[1] ?  min(ry, (g.ny - 1) / 2) : ry;
  const int oz_lo = g.periodic[2] ? -min(rz, g.nz / 2) : -rz;
  const int oz_hi = g.periodic[2] ?  min(rz, (g.nz - 1) / 2) : rz;

  double rho = 0.0;
  double dh_sum = 0.0;
  double div_v = 0.0;
  double curl_x = 0.0, curl_y = 0.0, curl_z = 0.0;

  for (int ox = ox_lo; ox <= ox_hi; ++ox) {
    int cx = cx0 + ox;
    if (g.periodic[0]) cx = (cx % g.nx + g.nx) % g.nx;
    else if (cx < 0 || cx >= g.nx) continue;
    for (int oy = oy_lo; oy <= oy_hi; ++oy) {
      int cy = cy0 + oy;
      if (g.periodic[1]) cy = (cy % g.ny + g.ny) % g.ny;
      else if (cy < 0 || cy >= g.ny) continue;
      for (int oz = oz_lo; oz <= oz_hi; ++oz) {
        int cz = cz0 + oz;
        if (g.periodic[2]) cz = (cz % g.nz + g.nz) % g.nz;
        else if (cz < 0 || cz >= g.nz) continue;

        const int cid = (cx * g.ny + cy) * g.nz + cz;
        const int s = cell_start[cid];
        if (s < 0) continue;
        const int e = cell_end[cid];

        for (int p = s; p < e; ++p) {
          const int j = sorted_idx[p];
          const double3 xj = pos[j];
          const double dx = wrap_delta(xi.x - xj.x, g.box[0], g.periodic[0]);
          const double dy = wrap_delta(xi.y - xj.y, g.box[1], g.periodic[1]);
          const double dz = wrap_delta(xi.z - xj.z, g.box[2], g.periodic[2]);
          const double r2 = dx * dx + dy * dy + dz * dz;
          if (r2 < H2) {
            const double r = sqrt(r2);
            const double mj = mass[j];
            const double W = wendland_c2_W(r, hi);
            const double dW_r = wendland_c2_dW_over_r(r, hi);
            rho += mj * W;
            // dW/dh = -(3/h)*W - (r/h)*dW/dr = -(3*W + r^2 * dW_r) / h
            dh_sum += mj * (-(3.0 * W + r2 * dW_r) / hi);
            if (j != i && r2 > 1e-30) {
              const double3 vj = vel[j];
              const double dvx = vj.x - vi.x;
              const double dvy = vj.y - vi.y;
              const double dvz = vj.z - vi.z;
              const double gx = dx * dW_r;
              const double gy = dy * dW_r;
              const double gz = dz * dW_r;
              div_v += mj * (dvx * gx + dvy * gy + dvz * gz);
              curl_x += mj * (dvy * gz - dvz * gy);
              curl_y += mj * (dvz * gx - dvx * gz);
              curl_z += mj * (dvx * gy - dvy * gx);
            }
          }
        }
      }
    }
  }

  rho = fmax(rho, 1e-30);
  rho_arr[i] = rho;
  const double omega = fmax(0.5, fmin(2.0, 1.0 + (hi / (3.0 * rho)) * dh_sum));
  omega_arr[i] = omega;
  div_v_arr[i] = div_v / rho;
  curl_v_arr[i] = sqrt(curl_x * curl_x + curl_y * curl_y + curl_z * curl_z) / rho;

  if (update_h) {
    const double h_target = eta_sph * cbrt(mass[i] / rho);
    // Damped fixed-point update for smooth convergence
    const double h_new = 0.5 * hi + 0.5 * fmax(h_min, fmin(h_max, h_target));
    h_arr[i] = h_new;
  }
}

__global__ void sph_eos_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ u,
    const double* __restrict__ div_v,
    const double* __restrict__ curl_v,
    const double* __restrict__ h_arr,
    double* __restrict__ P_arr,
    double* __restrict__ cs_arr,
    double* __restrict__ balsara_arr,
    int n,
    double gamma_val,
    int use_balsara) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const double ui = fmax(u[i], 1e-12);
  P_arr[i] = (gamma_val - 1.0) * rho[i] * ui;
  const double cs = sqrt(gamma_val * (gamma_val - 1.0) * ui);
  cs_arr[i] = cs;
  if (use_balsara) {
    const double ad = fabs(div_v[i]);
    const double ac = fabs(curl_v[i]);
    balsara_arr[i] = ad / (ad + ac + 1e-4 * cs / fmax(h_arr[i], 1e-12));
  } else {
    balsara_arr[i] = 1.0;
  }
}

// Conservative SPH hydro force + Monaghan viscosity + Price artificial conduction
__global__ void sph_force_kernel(
    const double3* __restrict__ pos,
    const double3* __restrict__ vel,
    const double* __restrict__ mass,
    const double* __restrict__ h_arr,
    const double* __restrict__ rho_arr,
    const double* __restrict__ omega_arr,
    const double* __restrict__ P_arr,
    const double* __restrict__ cs_arr,
    const double* __restrict__ u_arr,
    const double* __restrict__ balsara_arr,
    double3* __restrict__ acc_arr,
    double* __restrict__ du_dt_arr,
    double* __restrict__ dt_arr,
    const int* __restrict__ sorted_idx,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    int n,
    CellGrid g,
    double alpha_visc,
    double alpha_cond,
    double cfl,
    double h_search_max) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  const double3 xi = pos[i];
  const double3 vi = vel[i];
  const double hi = h_arr[i];
  const double rhoi = rho_arr[i];
  const double om_i = omega_arr[i];
  const double Pi = P_arr[i];
  const double csi = cs_arr[i];
  const double ui = u_arr[i];
  const double fi = balsara_arr[i];
  const double P_over_rho2_i = Pi / (om_i * rhoi * rhoi);

  const double H_scan = 2.0 * fmax(hi, h_search_max);
  const int cx0 = max(0, min(g.nx - 1, int(floor(xi.x / g.dx))));
  const int cy0 = max(0, min(g.ny - 1, int(floor(xi.y / g.dy))));
  const int cz0 = max(0, min(g.nz - 1, int(floor(xi.z / g.dz))));

  const int rx = max(1, int(ceil(H_scan / g.dx)));
  const int ry = max(1, int(ceil(H_scan / g.dy)));
  const int rz = max(1, int(ceil(H_scan / g.dz)));
  const int ox_lo = g.periodic[0] ? -min(rx, g.nx / 2) : -rx;
  const int ox_hi = g.periodic[0] ?  min(rx, (g.nx - 1) / 2) : rx;
  const int oy_lo = g.periodic[1] ? -min(ry, g.ny / 2) : -ry;
  const int oy_hi = g.periodic[1] ?  min(ry, (g.ny - 1) / 2) : ry;
  const int oz_lo = g.periodic[2] ? -min(rz, g.nz / 2) : -rz;
  const int oz_hi = g.periodic[2] ?  min(rz, (g.nz - 1) / 2) : rz;

  double ax = 0.0, ay = 0.0, az = 0.0;
  double du = 0.0;
  double vsig_max = 2.0 * csi;

  for (int ox = ox_lo; ox <= ox_hi; ++ox) {
    int cx = cx0 + ox;
    if (g.periodic[0]) cx = (cx % g.nx + g.nx) % g.nx;
    else if (cx < 0 || cx >= g.nx) continue;
    for (int oy = oy_lo; oy <= oy_hi; ++oy) {
      int cy = cy0 + oy;
      if (g.periodic[1]) cy = (cy % g.ny + g.ny) % g.ny;
      else if (cy < 0 || cy >= g.ny) continue;
      for (int oz = oz_lo; oz <= oz_hi; ++oz) {
        int cz = cz0 + oz;
        if (g.periodic[2]) cz = (cz % g.nz + g.nz) % g.nz;
        else if (cz < 0 || cz >= g.nz) continue;

        const int cid = (cx * g.ny + cy) * g.nz + cz;
        const int s = cell_start[cid];
        if (s < 0) continue;
        const int e = cell_end[cid];

        for (int p = s; p < e; ++p) {
          const int j = sorted_idx[p];
          if (j == i) continue;
          const double hj = h_arr[j];
          const double h_pair = fmax(hi, hj);
          const double H_pair = 2.0 * h_pair;
          const double3 xj = pos[j];
          const double dx = wrap_delta(xi.x - xj.x, g.box[0], g.periodic[0]);
          const double dy = wrap_delta(xi.y - xj.y, g.box[1], g.periodic[1]);
          const double dz = wrap_delta(xi.z - xj.z, g.box[2], g.periodic[2]);
          const double r2 = dx * dx + dy * dy + dz * dz;
          if (r2 >= H_pair * H_pair || r2 <= 1e-30) continue;

          const double r = sqrt(r2);
          const double mj = mass[j];
          const double rhoj = rho_arr[j];
          const double om_j = omega_arr[j];
          const double Pj = P_arr[j];
          const double csj = cs_arr[j];
          const double uj = u_arr[j];
          const double fj = balsara_arr[j];

          const double dWi_r = wendland_c2_dW_over_r(r, hi);
          const double dWj_r = wendland_c2_dW_over_r(r, hj);
          const double dW_avg_r = 0.5 * (dWi_r + dWj_r);

          const double3 vj = vel[j];
          const double dvx = vi.x - vj.x;
          const double dvy = vi.y - vj.y;
          const double dvz = vi.z - vj.z;
          const double v_dot_r = dvx * dx + dvy * dy + dvz * dz;
          const double w_ij = v_dot_r / r;

          const double rho_bar = 0.5 * (rhoi + rhoj);
          double Pi_visc = 0.0;
          double vsig = csi + csj;
          if (w_ij < 0.0) {
            vsig = csi + csj - 3.0 * w_ij;
            const double f_bar = 0.5 * (fi + fj);
            Pi_visc = -0.5 * alpha_visc * f_bar * vsig * w_ij / rho_bar;
          }
          if (vsig > vsig_max) vsig_max = vsig;

          // Momentum acceleration
          const double P_over_rho2_j = Pj / (om_j * rhoj * rhoj);
          const double f_scalar = -mj * (P_over_rho2_i * dWi_r + P_over_rho2_j * dWj_r + Pi_visc * dW_avg_r);
          ax += f_scalar * dx;
          ay += f_scalar * dy;
          az += f_scalar * dz;

          // PdV + viscous heating (strictly conservative with momentum equation)
          du += mj * (P_over_rho2_i * dWi_r + 0.5 * Pi_visc * dW_avg_r) * v_dot_r;

          // Price (2008) artificial thermal conduction (eliminates contact-discontinuity tension)
          if (alpha_cond > 0.0) {
            const double vsig_u = sqrt(fabs(Pi - Pj) / rho_bar);
            // Note: dW_avg_r <= 0, so r * dW_avg_r <= 0; conduction drives u_i toward u_j:
            // du_i/dt += m_j * alpha_u * vsig_u * (u_i - u_j) * (r * dW_avg_r) / rho_bar
            // When u_i > u_j, (u_i - u_j) > 0 and (r * dW_avg_r) < 0 -> du_i/dt < 0 (exact heat diffusion!)
            du += mj * alpha_cond * vsig_u * (ui - uj) * (r * dW_avg_r) / rho_bar;
          }
        }
      }
    }
  }

  acc_arr[i] = {ax, ay, az};
  du_dt_arr[i] = du;
  dt_arr[i] = cfl * hi / fmax(vsig_max, 1e-12);
}

// Direct O(N^2) tiled GPU self-gravity for isolated Santa Barbara cluster / coupled hydro benchmarks
// (for N <= 65536, tiled shared-memory GPU direct sum runs in ~2 ms/step on RTX 4090 with exact force!)
__global__ void direct_gravity_kernel(
    const double3* __restrict__ tgt_pos,
    int n_tgt,
    const double3* __restrict__ src_pos,
    const double* __restrict__ src_mass,
    int n_src,
    double3* __restrict__ tgt_acc,
    double* __restrict__ tgt_pot,
    double G,
    double eps2,
    int accumulate) {
  __shared__ double3 s_pos[256];
  __shared__ double s_m[256];

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  double3 xi = (i < n_tgt) ? tgt_pos[i] : double3{0.0, 0.0, 0.0};
  double ax = 0.0, ay = 0.0, az = 0.0, pot = 0.0;

  for (int base = 0; base < n_src; base += 256) {
    const int j = base + threadIdx.x;
    if (j < n_src) {
      s_pos[threadIdx.x] = src_pos[j];
      s_m[threadIdx.x] = src_mass[j];
    } else {
      s_pos[threadIdx.x] = {0.0, 0.0, 0.0};
      s_m[threadIdx.x] = 0.0;
    }
    __syncthreads();
    if (i < n_tgt) {
      const int cnt = min(256, n_src - base);
      for (int k = 0; k < cnt; ++k) {
        const double dx = s_pos[k].x - xi.x;
        const double dy = s_pos[k].y - xi.y;
        const double dz = s_pos[k].z - xi.z;
        const double r2 = dx * dx + dy * dy + dz * dz + eps2;
        const double inv_r = rsqrt(r2);
        const double inv_r3 = inv_r * inv_r * inv_r;
        const double mk = s_m[k];
        ax += mk * dx * inv_r3;
        ay += mk * dy * inv_r3;
        az += mk * dz * inv_r3;
        if (dx * dx + dy * dy + dz * dz > 1e-30) pot -= mk * inv_r;
      }
    }
    __syncthreads();
  }

  if (i < n_tgt) {
    if (accumulate) {
      tgt_acc[i].x += G * ax;
      tgt_acc[i].y += G * ay;
      tgt_acc[i].z += G * az;
      tgt_pot[i] += G * pot;
    } else {
      tgt_acc[i] = {G * ax, G * ay, G * az};
      tgt_pot[i] = G * pot;
    }
  }
}

// Tabulated / analytic Sutherland-Dopita (1993) primordial + solar-metallicity cooling function
// Lambda(T) in normalized dimensionless form peaking near T ~ 1.5e5 K with atomic H cutoff at T_floor.
__device__ __forceinline__ double cooling_lambda_norm(double u, double u_floor) {
  if (u <= u_floor) return 0.0;
  // Map u_floor -> T = 1e4 K; T = 1e4 * (u / u_floor)
  const double T = 1.0e4 * (u / u_floor);
  // Smooth two-peak Sutherland-Dopita fit: H/He collisional peak at 1.5e5 K + Bremsstrahlung T^0.5
  const double lT = log10(T);
  const double peak = exp(-0.5 * ((lT - 5.15) / 0.45) * ((lT - 5.15) / 0.45));
  const double brem = 0.05 * sqrt(T / 1.0e6);
  const double cutoff = 1.0 / (1.0 + exp(-(lT - 4.05) / 0.05));
  return cutoff * (peak + brem);
}

// Implicitly subcycled GPU radiative cooling kernel (strictly non-negative u >= u_floor)
__global__ void sph_cooling_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ mass,
    double* __restrict__ u_arr,
    double* __restrict__ rad_lost_arr,
    int n,
    double dt,
    double cooling_scale,
    double u_floor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  double u = u_arr[i];
  if (u <= u_floor) return;
  const double u0 = u;
  const double rhoi = rho[i];
  double t_rem = dt;
  // Subcycle with max 10% fractional cooling per sub-step for unconditional stability
  for (int sub = 0; sub < 64 && t_rem > 0.0 && u > u_floor; ++sub) {
    const double rate = cooling_scale * rhoi * cooling_lambda_norm(u, u_floor);
    if (rate <= 1e-30) break;
    const double dt_sub = fmin(t_rem, 0.1 * (u - u_floor) / rate + 1e-12);
    u = fmax(u_floor, u - rate * dt_sub);
    t_rem -= dt_sub;
  }
  u_arr[i] = u;
  rad_lost_arr[i] += mass[i] * (u0 - u);
}

__global__ void kick_drift_gas_kernel(
    double3* __restrict__ pos,
    double3* __restrict__ vel,
    double* __restrict__ u_arr,
    const double3* __restrict__ acc,
    const double* __restrict__ du_dt,
    int n,
    double dt_half,
    double dt_full,
    double u_floor,
    CellGrid g) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  vel[i].x += acc[i].x * dt_half;
  vel[i].y += acc[i].y * dt_half;
  vel[i].z += acc[i].z * dt_half;
  u_arr[i] = fmax(u_floor, u_arr[i] + du_dt[i] * dt_half);

  pos[i].x = wrap_pos(pos[i].x + vel[i].x * dt_full, g.box[0], g.periodic[0]);
  pos[i].y = wrap_pos(pos[i].y + vel[i].y * dt_full, g.box[1], g.periodic[1]);
  pos[i].z = wrap_pos(pos[i].z + vel[i].z * dt_full, g.box[2], g.periodic[2]);
}

__global__ void kick_finish_gas_kernel(
    double3* __restrict__ vel,
    double* __restrict__ u_arr,
    const double3* __restrict__ acc,
    const double* __restrict__ du_dt,
    int n,
    double dt_half,
    double u_floor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  vel[i].x += acc[i].x * dt_half;
  vel[i].y += acc[i].y * dt_half;
  vel[i].z += acc[i].z * dt_half;
  u_arr[i] = fmax(u_floor, u_arr[i] + du_dt[i] * dt_half);
}

__global__ void kick_drift_dm_kernel(
    double3* __restrict__ pos,
    double3* __restrict__ vel,
    const double3* __restrict__ acc,
    int n,
    double dt_half,
    double dt_full) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  vel[i].x += acc[i].x * dt_half;
  vel[i].y += acc[i].y * dt_half;
  vel[i].z += acc[i].z * dt_half;
  pos[i].x += vel[i].x * dt_full;
  pos[i].y += vel[i].y * dt_full;
  pos[i].z += vel[i].z * dt_full;
}

__global__ void kick_finish_dm_kernel(
    double3* __restrict__ vel,
    const double3* __restrict__ acc,
    int n,
    double dt_half) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  vel[i].x += acc[i].x * dt_half;
  vel[i].y += acc[i].y * dt_half;
  vel[i].z += acc[i].z * dt_half;
}

}  // namespace

int grav_hydro_evolve_gpu(GravHydroState* state, const GravHydroConfig* cfg,
                          double t_end, GravHydroStats* stats_out) {
  if (!state || !cfg || state->n_gas <= 0) return -1;
  const int n = state->n_gas;
  const int n_dm = state->n_dm;
  const auto t0_wall = std::chrono::steady_clock::now();

  // Choose spatial-hash grid dimensions (aim for cell size ~ 2 * h_init)
  double h_init_max = cfg->h_min;
  for (int i = 0; i < n; ++i) h_init_max = std::max(h_init_max, state->h[i]);
  CellGrid grid{};
  for (int d = 0; d < 3; ++d) {
    grid.box[d] = cfg->box[d];
    grid.periodic[d] = cfg->periodic[d];
  }
  const double target_cell = std::max(2.0 * h_init_max, 0.02 * std::min({cfg->box[0], cfg->box[1], cfg->box[2]}));
  grid.nx = std::max(1, std::min(64, int(std::floor(cfg->box[0] / target_cell))));
  grid.ny = std::max(1, std::min(64, int(std::floor(cfg->box[1] / target_cell))));
  grid.nz = std::max(1, std::min(64, int(std::floor(cfg->box[2] / target_cell))));
  grid.dx = cfg->box[0] / double(grid.nx);
  grid.dy = cfg->box[1] / double(grid.ny);
  grid.dz = cfg->box[2] / double(grid.nz);
  const int n_cells = grid.nx * grid.ny * grid.nz;

  // Allocate GPU buffers
  double3 *d_x = nullptr, *d_v = nullptr, *d_acc = nullptr;
  double *d_m = nullptr, *d_u = nullptr, *d_h = nullptr, *d_rho = nullptr;
  double *d_om = nullptr, *d_div = nullptr, *d_curl = nullptr;
  double *d_P = nullptr, *d_cs = nullptr, *d_bal = nullptr, *d_du = nullptr, *d_dt = nullptr;
  double *d_pot = nullptr, *d_rad = nullptr;
  int *d_keys = nullptr, *d_idx = nullptr, *d_cstart = nullptr, *d_cend = nullptr;

  HYDRO_CHECK(cudaMalloc(&d_x, size_t(n) * sizeof(double3)));
  HYDRO_CHECK(cudaMalloc(&d_v, size_t(n) * sizeof(double3)));
  HYDRO_CHECK(cudaMalloc(&d_acc, size_t(n) * sizeof(double3)));
  HYDRO_CHECK(cudaMalloc(&d_m, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_u, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_h, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_rho, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_om, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_div, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_curl, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_P, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_cs, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_bal, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_du, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_dt, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_pot, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_rad, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMalloc(&d_keys, size_t(n) * sizeof(int)));
  HYDRO_CHECK(cudaMalloc(&d_idx, size_t(n) * sizeof(int)));
  HYDRO_CHECK(cudaMalloc(&d_cstart, size_t(n_cells) * sizeof(int)));
  HYDRO_CHECK(cudaMalloc(&d_cend, size_t(n_cells) * sizeof(int)));
  HYDRO_CHECK(cudaMemset(d_pot, 0, size_t(n) * sizeof(double)));
  HYDRO_CHECK(cudaMemset(d_rad, 0, size_t(n) * sizeof(double)));

  double3 *d_dm_x = nullptr, *d_dm_v = nullptr, *d_dm_acc = nullptr;
  double *d_dm_m = nullptr, *d_dm_pot = nullptr;
  if (n_dm > 0) {
    HYDRO_CHECK(cudaMalloc(&d_dm_x, size_t(n_dm) * sizeof(double3)));
    HYDRO_CHECK(cudaMalloc(&d_dm_v, size_t(n_dm) * sizeof(double3)));
    HYDRO_CHECK(cudaMalloc(&d_dm_acc, size_t(n_dm) * sizeof(double3)));
    HYDRO_CHECK(cudaMalloc(&d_dm_m, size_t(n_dm) * sizeof(double)));
    HYDRO_CHECK(cudaMalloc(&d_dm_pot, size_t(n_dm) * sizeof(double)));
    std::vector<double3> h_dmx(n_dm), h_dmv(n_dm);
    for (int i = 0; i < n_dm; ++i) {
      h_dmx[i] = {state->dm_x[3 * i], state->dm_x[3 * i + 1], state->dm_x[3 * i + 2]};
      h_dmv[i] = {state->dm_v[3 * i], state->dm_v[3 * i + 1], state->dm_v[3 * i + 2]};
    }
    HYDRO_CHECK(cudaMemcpy(d_dm_x, h_dmx.data(), size_t(n_dm) * sizeof(double3), cudaMemcpyHostToDevice));
    HYDRO_CHECK(cudaMemcpy(d_dm_v, h_dmv.data(), size_t(n_dm) * sizeof(double3), cudaMemcpyHostToDevice));
    HYDRO_CHECK(cudaMemcpy(d_dm_m, state->dm_m, size_t(n_dm) * sizeof(double), cudaMemcpyHostToDevice));
  }

  // Upload gas state
  std::vector<double3> h_x(n), h_v(n);
  for (int i = 0; i < n; ++i) {
    h_x[i] = {state->x[3 * i], state->x[3 * i + 1], state->x[3 * i + 2]};
    h_v[i] = {state->v[3 * i], state->v[3 * i + 1], state->v[3 * i + 2]};
  }
  HYDRO_CHECK(cudaMemcpy(d_x, h_x.data(), size_t(n) * sizeof(double3), cudaMemcpyHostToDevice));
  HYDRO_CHECK(cudaMemcpy(d_v, h_v.data(), size_t(n) * sizeof(double3), cudaMemcpyHostToDevice));
  HYDRO_CHECK(cudaMemcpy(d_m, state->m, size_t(n) * sizeof(double), cudaMemcpyHostToDevice));
  HYDRO_CHECK(cudaMemcpy(d_u, state->u, size_t(n) * sizeof(double), cudaMemcpyHostToDevice));
  HYDRO_CHECK(cudaMemcpy(d_h, state->h, size_t(n) * sizeof(double), cudaMemcpyHostToDevice));

  const int B = 256;
  const int G_gas = (n + B - 1) / B;
  const int G_dm = (n_dm + B - 1) / B;
  std::vector<double> h_dt(n), h_h_tmp(n), h_u_tmp(n), h_pot_tmp(n), h_rad_tmp(n);
  std::vector<double3> h_v_tmp(n);

  auto compute_forces = [&](int n_h_iters, double& dt_min_out) -> int {
    compute_cell_keys_kernel<<<G_gas, B>>>(d_x, d_keys, d_idx, n, grid);
    thrust::sort_by_key(thrust::device_pointer_cast(d_keys),
                        thrust::device_pointer_cast(d_keys + n),
                        thrust::device_pointer_cast(d_idx));
    HYDRO_CHECK(cudaMemset(d_cstart, 0xff, size_t(n_cells) * sizeof(int)));
    HYDRO_CHECK(cudaMemset(d_cend, 0, size_t(n_cells) * sizeof(int)));
    find_cell_bounds_kernel<<<G_gas, B>>>(d_keys, d_cstart, d_cend, n);

    for (int it = 0; it < n_h_iters; ++it) {
      sph_density_kernel<<<G_gas, B>>>(
          d_x, d_v, d_m, d_h, d_rho, d_om, d_div, d_curl,
          d_idx, d_cstart, d_cend, n, grid,
          cfg->eta_sph, cfg->h_min, cfg->h_max, (it + 1 < n_h_iters) ? 1 : 0);
    }
    sph_eos_kernel<<<G_gas, B>>>(
        d_rho, d_u, d_div, d_curl, d_h, d_P, d_cs, d_bal,
        n, cfg->gamma, cfg->use_balsara);

    // Find max smoothing length for symmetric gather-scatter search radius
    HYDRO_CHECK(cudaMemcpy(h_h_tmp.data(), d_h, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost));
    double h_max_cur = cfg->h_min;
    for (int i = 0; i < n; ++i) h_max_cur = std::max(h_max_cur, h_h_tmp[i]);

    sph_force_kernel<<<G_gas, B>>>(
        d_x, d_v, d_m, d_h, d_rho, d_om, d_P, d_cs, d_u, d_bal,
        d_acc, d_du, d_dt, d_idx, d_cstart, d_cend, n, grid,
        cfg->alpha_visc, cfg->alpha_cond, cfg->cfl, h_max_cur);

    if (cfg->enable_gravity) {
      const double eps2 = cfg->grav_soft * cfg->grav_soft;
      HYDRO_CHECK(cudaMemset(d_pot, 0, size_t(n) * sizeof(double)));
      // Gas <- Gas gravity
      direct_gravity_kernel<<<G_gas, B>>>(d_x, n, d_x, d_m, n, d_acc, d_pot, cfg->grav_G, eps2, 1);
      if (n_dm > 0) {
        // Gas <- DM gravity
        direct_gravity_kernel<<<G_gas, B>>>(d_x, n, d_dm_x, d_dm_m, n_dm, d_acc, d_pot, cfg->grav_G, eps2, 1);
        // DM <- DM + Gas gravity
        direct_gravity_kernel<<<G_dm, B>>>(d_dm_x, n_dm, d_dm_x, d_dm_m, n_dm, d_dm_acc, d_dm_pot, cfg->grav_G, eps2, 0);
        direct_gravity_kernel<<<G_dm, B>>>(d_dm_x, n_dm, d_x, d_m, n, d_dm_acc, d_dm_pot, cfg->grav_G, eps2, 1);
      }
    }

    HYDRO_CHECK(cudaMemcpy(h_dt.data(), d_dt, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost));
    double dt_min = 1e30;
    for (int i = 0; i < n; ++i) dt_min = std::min(dt_min, h_dt[i]);
    if (cfg->enable_gravity) {
      // Gravity dynamical timestep guard
      dt_min = std::min(dt_min, 0.01);
    }
    dt_min_out = dt_min;
    return 0;
  };

  auto compute_energies = [&](double& ek, double& ei, double& ep, double& erad) -> int {
    HYDRO_CHECK(cudaMemcpy(h_v_tmp.data(), d_v, size_t(n) * sizeof(double3), cudaMemcpyDeviceToHost));
    HYDRO_CHECK(cudaMemcpy(h_u_tmp.data(), d_u, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost));
    HYDRO_CHECK(cudaMemcpy(h_pot_tmp.data(), d_pot, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost));
    HYDRO_CHECK(cudaMemcpy(h_rad_tmp.data(), d_rad, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost));
    ek = ei = ep = erad = 0.0;
    for (int i = 0; i < n; ++i) {
      const double mi = state->m[i];
      ek += 0.5 * mi * (h_v_tmp[i].x * h_v_tmp[i].x + h_v_tmp[i].y * h_v_tmp[i].y + h_v_tmp[i].z * h_v_tmp[i].z);
      ei += mi * h_u_tmp[i];
      ep += 0.5 * mi * h_pot_tmp[i];
      erad += h_rad_tmp[i];
    }
    if (n_dm > 0 && cfg->enable_gravity) {
      std::vector<double3> h_dmv_t(n_dm);
      std::vector<double> h_dmp_t(n_dm);
      HYDRO_CHECK(cudaMemcpy(h_dmv_t.data(), d_dm_v, size_t(n_dm) * sizeof(double3), cudaMemcpyDeviceToHost));
      HYDRO_CHECK(cudaMemcpy(h_dmp_t.data(), d_dm_pot, size_t(n_dm) * sizeof(double), cudaMemcpyDeviceToHost));
      for (int i = 0; i < n_dm; ++i) {
        const double mi = state->dm_m[i];
        ek += 0.5 * mi * (h_dmv_t[i].x * h_dmv_t[i].x + h_dmv_t[i].y * h_dmv_t[i].y + h_dmv_t[i].z * h_dmv_t[i].z);
        ep += 0.5 * mi * h_dmp_t[i];
      }
    }
    return 0;
  };

  double dt = 0.0;
  if (compute_forces(5, dt) != 0) return -1;

  GravHydroStats st{};
  if (compute_energies(st.E_kin_init, st.E_int_init, st.E_pot_init, st.E_rad_lost) != 0) return -1;
  st.E_tot_init = st.E_kin_init + st.E_int_init + st.E_pot_init;

  double t = 0.0;
  int step = 0;
  while (t < t_end - 1e-12 && step < cfg->max_steps) {
    const double step_dt = std::min(dt, t_end - t);
    const double dt_half = 0.5 * step_dt;

    // KDK Leapfrog: Kick(dt/2) + Drift(dt)
    kick_drift_gas_kernel<<<G_gas, B>>>(d_x, d_v, d_u, d_acc, d_du, n, dt_half, step_dt, cfg->u_floor, grid);
    if (n_dm > 0 && cfg->enable_gravity) {
      kick_drift_dm_kernel<<<G_dm, B>>>(d_dm_x, d_dm_v, d_dm_acc, n_dm, dt_half, step_dt);
    }

    // Recompute SPH density, h, forces, and self-gravity at new positions
    if (compute_forces(2, dt) != 0) return -1;

    // Kick finish (dt/2)
    kick_finish_gas_kernel<<<G_gas, B>>>(d_v, d_u, d_acc, d_du, n, dt_half, cfg->u_floor);
    if (n_dm > 0 && cfg->enable_gravity) {
      kick_finish_dm_kernel<<<G_dm, B>>>(d_dm_v, d_dm_acc, n_dm, dt_half);
    }

    // Optional GPU radiative cooling (implicit subcycled)
    if (cfg->enable_cooling) {
      sph_cooling_kernel<<<G_gas, B>>>(d_rho, d_m, d_u, d_rad, n, step_dt, cfg->cooling_unit_scale, cfg->u_floor);
    }

    t += step_dt;
    ++step;

    if (cfg->log_interval > 0 && (step % cfg->log_interval == 0 || t >= t_end - 1e-12)) {
      double ek, ei, ep, erad;
      if (compute_energies(ek, ei, ep, erad) != 0) return -1;
      const double etot = ek + ei + ep + erad;
      const double rel_dE = (std::fabs(st.E_tot_init) > 1e-12)
          ? std::fabs(etot - st.E_tot_init) / std::fabs(st.E_tot_init)
          : 0.0;
      if (rel_dE > st.max_rel_dE) st.max_rel_dE = rel_dE;
    }
  }

  // Final density & EOS refresh for clean output
  double dummy_dt = 0.0;
  if (compute_forces(2, dummy_dt) != 0) return -1;
  if (compute_energies(st.E_kin_final, st.E_int_final, st.E_pot_final, st.E_rad_lost) != 0) return -1;
  st.E_tot_final = st.E_kin_final + st.E_int_final + st.E_pot_final;
  st.steps = step;
  st.t_end = t;
  st.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_wall).count();

  // Copy final state back to host
  HYDRO_CHECK(cudaMemcpy(h_x.data(), d_x, size_t(n) * sizeof(double3), cudaMemcpyDeviceToHost));
  HYDRO_CHECK(cudaMemcpy(h_v.data(), d_v, size_t(n) * sizeof(double3), cudaMemcpyDeviceToHost));
  HYDRO_CHECK(cudaMemcpy(state->u, d_u, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost));
  HYDRO_CHECK(cudaMemcpy(state->rho, d_rho, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost));
  HYDRO_CHECK(cudaMemcpy(state->h, d_h, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost));
  HYDRO_CHECK(cudaMemcpy(state->P, d_P, size_t(n) * sizeof(double), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    state->x[3 * i] = h_x[i].x;
    state->x[3 * i + 1] = h_x[i].y;
    state->x[3 * i + 2] = h_x[i].z;
    state->v[3 * i] = h_v[i].x;
    state->v[3 * i + 1] = h_v[i].y;
    state->v[3 * i + 2] = h_v[i].z;
  }

  cudaFree(d_x); cudaFree(d_v); cudaFree(d_acc);
  cudaFree(d_m); cudaFree(d_u); cudaFree(d_h); cudaFree(d_rho);
  cudaFree(d_om); cudaFree(d_div); cudaFree(d_curl);
  cudaFree(d_P); cudaFree(d_cs); cudaFree(d_bal); cudaFree(d_du); cudaFree(d_dt);
  cudaFree(d_pot); cudaFree(d_rad);
  cudaFree(d_keys); cudaFree(d_idx); cudaFree(d_cstart); cudaFree(d_cend);
  if (n_dm > 0) {
    cudaFree(d_dm_x); cudaFree(d_dm_v); cudaFree(d_dm_acc);
    cudaFree(d_dm_m); cudaFree(d_dm_pot);
  }

  if (stats_out) *stats_out = st;
  return 0;
}
