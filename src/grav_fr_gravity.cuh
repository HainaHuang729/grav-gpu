#ifndef GRAV_FR_GRAVITY_CUH
#define GRAV_FR_GRAVITY_CUH

#include <cmath>
#include <cstdlib>
#include <algorithm>

// B2-b: Hu-Sawicki (2007) n=1 f(R) Modified Gravity with Chameleon Screening.
// Controlled by:
//   GRAV_FR0=1e-4          (|f_R0|, background scalaron amplitude at z=0; 0 = GR)
//   GRAV_FR_SCREENING=1    (1 = apply Khoury-Weltman / Winther-Ferreira thin-shell chameleon screening)
//
// Physics:
//   Background scalaron:   |f_R_bar(a)| = |f_R0| * ((Omega_m + 4*Omega_DE) / (Omega_m*a^-3 + 4*Omega_DE))^2
//   Compton wavenumber:    m_fR(a) [1/Mpc] = (H0/c) * sqrt((Omega_m + 4*Omega_DE) / (2*|f_R0|))
//                                          * ((Omega_m*a^-3 + 4*Omega_DE) / (Omega_m + 4*Omega_DE))^(3/2)
//   Unscreened fifth-force: G_eff(k, a) / G = 1 + (1/3) * (k/a)^2 / ((k/a)^2 + m_fR(a)^2)
//   Thin-shell screening:   S(Phi_N) = min(1, 1.5 * |f_R_bar(a)| / |Phi_N / c^2|)

struct GravFrConfig {
  double f_R0 = 0.0;       // |f_R0| >= 0 (e.g., 1e-4, 1e-5, 1e-6)
  int screening = 1;       // 1 = enable chameleon screening
  double hubble0 = 67.36;  // km/s/Mpc
  double omega_m = 0.3153;
  double omega_de = 0.6847;
};

struct GravFrStepParams {
  int enabled = 0;
  int screening = 1;
  double inv_a2 = 1.0;        // 1 / a^2
  double m_fr2 = 0.0;         // m_fR(a)^2 in Mpc^-2
  double f_R_bar = 0.0;       // |f_R_bar(a)|
  double phi_screen = 0.0;    // 1.5 * |f_R_bar(a)| * c^2 in (km/s)^2
};

namespace grav_fr_detail {

inline GravFrConfig& config_storage() {
  static GravFrConfig cfg{};
  return cfg;
}

inline GravFrStepParams& step_params_storage() {
  static GravFrStepParams sp{};
  return sp;
}

}  // namespace grav_fr_detail

inline GravFrConfig grav_fr_init_from_env(double hubble0, double omega_m, double omega_de) {
  GravFrConfig cfg{};
  cfg.hubble0 = hubble0;
  cfg.omega_m = omega_m;
  cfg.omega_de = omega_de;
  const char* fr_env = std::getenv("GRAV_FR0");
  if (fr_env && *fr_env) {
    double v = std::fabs(std::atof(fr_env));
    if (std::isfinite(v) && v > 0.0) cfg.f_R0 = v;
  }
  const char* sc_env = std::getenv("GRAV_FR_SCREENING");
  if (sc_env && *sc_env) {
    cfg.screening = (std::atoi(sc_env) != 0) ? 1 : 0;
  }
  grav_fr_detail::config_storage() = cfg;
  return cfg;
}

// Compute background |f_R_bar(a)| for Hu-Sawicki n=1
inline double grav_fr_background(double a, double f_R0, double omega_m, double omega_de) {
  if (f_R0 <= 0.0 || a <= 0.0) return 0.0;
  const double R0 = omega_m + 4.0 * omega_de;
  const double Ra = omega_m / (a * a * a) + 4.0 * omega_de;
  const double ratio = R0 / Ra;
  return f_R0 * ratio * ratio;
}

// Compute physical scalaron mass m_fR(a) in Mpc^-1 for Hu-Sawicki n=1
inline double grav_fr_mass_inv_mpc(double a, double f_R0, double hubble0,
                                   double omega_m, double omega_de) {
  if (f_R0 <= 0.0 || a <= 0.0) return 0.0;
  constexpr double c_kms = 299792.458;
  const double R0 = omega_m + 4.0 * omega_de;
  const double Ra = omega_m / (a * a * a) + 4.0 * omega_de;
  const double m0 = (hubble0 / c_kms) * std::sqrt(R0 / (2.0 * f_R0));
  return m0 * std::pow(Ra / R0, 1.5);
}

inline void grav_fr_update_step(double a) {
  const auto& cfg = grav_fr_detail::config_storage();
  auto& sp = grav_fr_detail::step_params_storage();
  if (cfg.f_R0 <= 0.0 || a <= 0.0) {
    sp.enabled = 0;
    sp.m_fr2 = 0.0;
    sp.f_R_bar = 0.0;
    sp.phi_screen = 0.0;
    return;
  }
  constexpr double c_kms = 299792.458;
  const double m_fr = grav_fr_mass_inv_mpc(a, cfg.f_R0, cfg.hubble0, cfg.omega_m, cfg.omega_de);
  const double fr_bar = grav_fr_background(a, cfg.f_R0, cfg.omega_m, cfg.omega_de);
  sp.enabled = 1;
  sp.screening = cfg.screening;
  sp.inv_a2 = 1.0 / (a * a);
  sp.m_fr2 = m_fr * m_fr;
  sp.f_R_bar = fr_bar;
  sp.phi_screen = 1.5 * fr_bar * (c_kms * c_kms);
}

inline GravFrStepParams grav_fr_get_step_params() {
  return grav_fr_detail::step_params_storage();
}

// Linear k-space effective gravitational enhancement G_eff(k, a) / G
// k_comoving_inv_mpc = 2*pi*|n|/L_box [comoving Mpc^-1]
// Physical k_phys = k_comoving / a -> k_phys^2 = k_comoving^2 * inv_a2
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
double grav_fr_pm_boost(double k_comoving_inv_mpc, const GravFrStepParams& sp) {
  if (!sp.enabled || k_comoving_inv_mpc <= 0.0) return 1.0;
  const double kp2 = k_comoving_inv_mpc * k_comoving_inv_mpc * sp.inv_a2;
  return 1.0 + (1.0 / 3.0) * (kp2 / (kp2 + sp.m_fr2));
}

// Thin-shell chameleon screening factor S(Phi_N) in [0, 1]
// abs_phi_kms2 = |Phi_N| in (km/s)^2
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
double grav_fr_chameleon_screen(double abs_phi_kms2, const GravFrStepParams& sp) {
  if (!sp.enabled) return 0.0;
  if (!sp.screening || abs_phi_kms2 <= sp.phi_screen) return 1.0;
  return sp.phi_screen / abs_phi_kms2;
}

#endif  // GRAV_FR_GRAVITY_CUH
