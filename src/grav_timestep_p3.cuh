#ifndef GRAV_TIMESTEP_P3_CUH
#define GRAV_TIMESTEP_P3_CUH

#include <cmath>
#include <cstdlib>
#include <algorithm>

// P3: Adaptive dynamical/tidal & velocity-crossing timestepping + symplectic rung synchronization.
// Controlled by:
//   GRAV_TIDAL_TIMESTEP=1  (default: 1 in GRAV_PRECISION_MODE=high, else 0)
//   GRAV_TIDAL_ETA=0.05    (dynamical timescale factor: dt_dyn = eta_dyn * sqrt(|Phi|) / |a|)
//   GRAV_VEL_ETA=0.25      (softening-crossing factor:  dt_vel = eta_vel * eps_pl / |v|)
//   GRAV_SYNC_STEP=1       (force synchronous global-rung leapfrog for strict symplecticity)
//   GRAV_RUNG_HYSTERESIS=1 (10% margin on upward rung promotion to damp rung ping-pong)

struct GravP3Config {
  int tidal_timestep = 0;
  float tidal_eta = 0.05f;
  float vel_eta = 0.25f;
  int sync_step = 0;
  int rung_hysteresis = 0;
};

inline GravP3Config grav_p3_parse_env(int high_precision) {
  GravP3Config cfg{};
  const char* t_env = std::getenv("GRAV_TIDAL_TIMESTEP");
  cfg.tidal_timestep = (t_env && *t_env) ? (std::atoi(t_env) != 0 ? 1 : 0) : high_precision;

  const char* te_env = std::getenv("GRAV_TIDAL_ETA");
  if (te_env && *te_env) {
    float v = std::strtof(te_env, nullptr);
    if (std::isfinite(v) && v > 1e-4f && v < 10.f) cfg.tidal_eta = v;
  }

  const char* ve_env = std::getenv("GRAV_VEL_ETA");
  if (ve_env && *ve_env) {
    float v = std::strtof(ve_env, nullptr);
    if (std::isfinite(v) && v > 1e-4f && v < 10.f) cfg.vel_eta = v;
  }

  const char* s_env = std::getenv("GRAV_SYNC_STEP");
  cfg.sync_step = (s_env && *s_env) ? (std::atoi(s_env) != 0 ? 1 : 0) : 0;

  const char* h_env = std::getenv("GRAV_RUNG_HYSTERESIS");
  cfg.rung_hysteresis = (h_env && *h_env) ? (std::atoi(h_env) != 0 ? 1 : 0) : (cfg.tidal_timestep ? 1 : 0);

  return cfg;
}

// Compute physical P3 dynamical + velocity-crossing timestep cap (same physical time units as dt_acc).
//   pot_phys = G * raw.w / a (or G * raw.w in non-comoving)
//   acc_mag  = |a_total| (physical)
//   v_mag    = |v_phys|  (physical)
//   eps_pl   = Plummer-equivalent softening (physical)
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
float grav_p3_dt_limit(float dt_acc, float acc_mag, float abs_pot_phys,
                       float v_mag, float eps_pl,
                       int tidal_enabled, float tidal_eta, float vel_eta) {
  if (!tidal_enabled) return dt_acc;
  float dt = dt_acc;
  // 1. Local dynamical / tidal free-fall timescale: tau_dyn = sqrt(|Phi|) / |a| ~ sqrt(r^3 / GM)
  if (acc_mag > 1e-30f && abs_pot_phys > 1e-30f) {
    const float tau_dyn = sqrtf(abs_pot_phys) / acc_mag;
    const float dt_dyn = tidal_eta * tau_dyn;
    if (dt_dyn > 0.f && dt_dyn < dt) dt = dt_dyn;
  }
  // 2. Softening-crossing Courant criterion: prevents fast core-passing orbits from skipping eps_pl
  if (v_mag > 1e-30f && eps_pl > 0.f) {
    const float dt_vel = vel_eta * eps_pl / v_mag;
    if (dt_vel > 0.f && dt_vel < dt) dt = dt_vel;
  }
  return dt;
}

#endif  // GRAV_TIMESTEP_P3_CUH
