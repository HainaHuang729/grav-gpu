#ifndef GRAV_NEUTRINO_CUH
#define GRAV_NEUTRINO_CUH

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// B1: Massive Neutrinos (background Fermi-Dirac H(a) + k-space Linear Response in PM FFT)
// Controlled by:
//   GRAV_NU_MASS_EV   : total sum(m_nu) in eV (default 0.0 = disabled)
//   GRAV_NU_NSPECIES  : number of degenerate massive species (default 3)
//   GRAV_NU_TABLE     : optional ASCII/CSV table with columns: a  k_inv_mpc  R_nu(k,a)

struct GravNuStepParams {
  int enabled = 0;
  double f_nu = 0.0;        // Omega_nu0 / Omega_m0
  double boost_coef = 0.0;  // f_nu / (1.0 - f_nu)
  double inv_k_fs2 = 0.0;   // 1.0 / k_fs(a)^2 [Mpc^2]
  // Optional 1D log-k table interpolated to current scale factor a (<= 256 bins)
  int n_k = 0;
  double log_k_min = 0.0;
  double inv_dlog_k = 0.0;
  double r_nu_tab[256];
};

struct GravNuConfig {
  double sum_m_nu_ev = 0.0;
  int n_species = 3;
  double m_each_ev = 0.0;
  double omega_nu0 = 0.0;
  double f_nu = 0.0;
  std::string table_path;

  // Parsed 2D table (if GRAV_NU_TABLE is provided): sorted unique a_grid and k_grid
  std::vector<double> a_grid;
  std::vector<double> log_k_grid;
  std::vector<double> r_nu_2d;  // [ia * nk + ik]
};

namespace grav_nu_detail {

inline GravNuStepParams& step_params_storage() {
  static GravNuStepParams p{};
  return p;
}

inline GravNuConfig& config_storage() {
  static GravNuConfig cfg{};
  return cfg;
}

inline void load_table_file(const std::string& path, GravNuConfig& cfg) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("GRAV_NU_TABLE: cannot open " + path);
  struct Row { double a, k, r; };
  std::vector<Row> rows;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    for (char& c : line) if (c == ',') c = ' ';
    std::istringstream iss(line);
    Row rw{};
    if (iss >> rw.a >> rw.k >> rw.r) {
      if (rw.a > 0.0 && rw.k > 0.0 && std::isfinite(rw.r)) rows.push_back(rw);
    }
  }
  if (rows.empty()) throw std::runtime_error("GRAV_NU_TABLE: no valid rows in " + path);
  std::vector<double> av, kv;
  for (const auto& r : rows) { av.push_back(r.a); kv.push_back(r.k); }
  std::sort(av.begin(), av.end());
  av.erase(std::unique(av.begin(), av.end(), [](double x, double y){ return std::fabs(x-y) < 1e-12; }), av.end());
  std::sort(kv.begin(), kv.end());
  kv.erase(std::unique(kv.begin(), kv.end(), [](double x, double y){ return std::fabs(x-y) < 1e-12 * std::max(1.0, x); }), kv.end());
  cfg.a_grid = av;
  cfg.log_k_grid.resize(kv.size());
  for (size_t i = 0; i < kv.size(); ++i) cfg.log_k_grid[i] = std::log(kv[i]);
  cfg.r_nu_2d.assign(av.size() * kv.size(), 0.0);
  for (const auto& r : rows) {
    size_t ia = size_t(std::lower_bound(av.begin(), av.end(), r.a - 1e-12) - av.begin());
    size_t ik = size_t(std::lower_bound(kv.begin(), kv.end(), r.k * (1.0 - 1e-9)) - kv.begin());
    if (ia < av.size() && ik < kv.size()) {
      cfg.r_nu_2d[ia * kv.size() + ik] = std::max(0.0, std::min(1.0, r.r));
    }
  }
}

}  // namespace grav_nu_detail

inline GravNuConfig grav_nu_init_from_env(double hubble0, double omega_m0) {
  GravNuConfig cfg{};
  const char* m_env = std::getenv("GRAV_NU_MASS_EV");
  if (m_env && *m_env) cfg.sum_m_nu_ev = std::atof(m_env);
  const char* n_env = std::getenv("GRAV_NU_NSPECIES");
  if (n_env && *n_env) cfg.n_species = std::max(1, std::atoi(n_env));
  const char* t_env = std::getenv("GRAV_NU_TABLE");
  if (t_env && *t_env) cfg.table_path = t_env;

  if (cfg.sum_m_nu_ev > 0.0) {
    const double h = hubble0 / 100.0;
    cfg.m_each_ev = cfg.sum_m_nu_ev / double(cfg.n_species);
    // Standard cosmological relation: Omega_nu,0 h^2 = sum(m_nu) / 93.14 eV
    cfg.omega_nu0 = cfg.sum_m_nu_ev / (93.14 * h * h);
    cfg.f_nu = (omega_m0 > 0.0) ? (cfg.omega_nu0 / omega_m0) : 0.0;
    if (cfg.f_nu >= 0.5) {
      throw std::runtime_error("GRAV_NU_MASS_EV implies unphysical f_nu >= 0.5");
    }
    if (!cfg.table_path.empty()) {
      grav_nu_detail::load_table_file(cfg.table_path, cfg);
    }
  }
  grav_nu_detail::config_storage() = cfg;
  return cfg;
}

// Komatsu+2011 (WMAP7 Eq. 25) Fermi-Dirac neutrino energy density enhancement factor:
// rho_nu(a) / (rho_nu0_NR * a^-3) = [1 + (3.151374 / y(a))^1.83]^(1/1.83)
// where y(a) = a * m_nu / (k_B T_nu0) with k_B T_nu0 = 1.6824e-4 eV -> 1/(k_B T_nu0) = 5943.9 eV^-1.
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
double grav_nu_rho_enhancement(double a, double m_each_ev) {
  if (m_each_ev <= 0.0) return 1.0;
  const double y = 5943.9 * a * m_each_ev;
  if (y > 1.0e5) return 1.0;
  const double ratio = 3.151374 / (y + 1.0e-30);
  return pow(1.0 + pow(ratio, 1.83), 1.0 / 1.83);
}

// Compute neutrino free-streaming wavenumber k_fs(a) in Mpc^-1:
//   c_nu(a) = 150 km/s * (1 eV / (a * m_each_ev))  [standard thermal velocity dispersion]
//   k_fs(a) = sqrt(3/2) * a * H(a) / c_nu(a)
// where H(a) is in km/s/Mpc and c_nu(a) is in km/s -> k_fs(a) in Mpc^-1.
inline double grav_nu_k_fs_inv_mpc(double a, double H_a_kms_mpc, double m_each_ev) {
  if (m_each_ev <= 0.0 || a <= 0.0 || H_a_kms_mpc <= 0.0) return 0.0;
  const double c_nu = 150.0 / (a * m_each_ev);
  return std::sqrt(1.5) * a * H_a_kms_mpc / c_nu;
}

inline void grav_nu_update_step(double a, double H_a_kms_mpc) {
  const auto& cfg = grav_nu_detail::config_storage();
  auto& sp = grav_nu_detail::step_params_storage();
  if (cfg.sum_m_nu_ev <= 0.0 || cfg.f_nu <= 0.0) {
    sp.enabled = 0;
    sp.f_nu = 0.0;
    sp.boost_coef = 0.0;
    sp.inv_k_fs2 = 0.0;
    sp.n_k = 0;
    return;
  }
  sp.enabled = 1;
  sp.f_nu = cfg.f_nu;
  sp.boost_coef = cfg.f_nu / (1.0 - cfg.f_nu);
  const double k_fs = grav_nu_k_fs_inv_mpc(a, H_a_kms_mpc, cfg.m_each_ev);
  sp.inv_k_fs2 = (k_fs > 0.0) ? (1.0 / (k_fs * k_fs)) : 0.0;

  if (!cfg.a_grid.empty() && cfg.log_k_grid.size() >= 2) {
    const size_t na = cfg.a_grid.size();
    const size_t nk_src = cfg.log_k_grid.size();
    // Interpolate in scale factor a
    size_t ia0 = 0, ia1 = 0;
    double ta = 0.0;
    if (a <= cfg.a_grid.front()) {
      ia0 = ia1 = 0;
    } else if (a >= cfg.a_grid.back()) {
      ia0 = ia1 = na - 1;
    } else {
      ia1 = size_t(std::lower_bound(cfg.a_grid.begin(), cfg.a_grid.end(), a) - cfg.a_grid.begin());
      ia0 = ia1 - 1;
      ta = (a - cfg.a_grid[ia0]) / (cfg.a_grid[ia1] - cfg.a_grid[ia0]);
    }
    // Resample onto uniform 256-bin log-k table for O(1) GPU lookup
    const int n_out = 256;
    const double lk_min = cfg.log_k_grid.front();
    const double lk_max = cfg.log_k_grid.back();
    const double d_lk = (lk_max - lk_min) / double(n_out - 1);
    sp.n_k = n_out;
    sp.log_k_min = lk_min;
    sp.inv_dlog_k = (d_lk > 0.0) ? (1.0 / d_lk) : 0.0;
    for (int j = 0; j < n_out; ++j) {
      const double lk = lk_min + double(j) * d_lk;
      auto it = std::lower_bound(cfg.log_k_grid.begin(), cfg.log_k_grid.end(), lk);
      size_t ik1 = size_t(it - cfg.log_k_grid.begin());
      size_t ik0 = (ik1 > 0) ? (ik1 - 1) : 0;
      if (ik1 >= nk_src) { ik0 = ik1 = nk_src - 1; }
      const double tk = (ik1 > ik0)
          ? (lk - cfg.log_k_grid[ik0]) / (cfg.log_k_grid[ik1] - cfg.log_k_grid[ik0])
          : 0.0;
      const double r0 = (1.0 - tk) * cfg.r_nu_2d[ia0 * nk_src + ik0] + tk * cfg.r_nu_2d[ia0 * nk_src + ik1];
      const double r1 = (1.0 - tk) * cfg.r_nu_2d[ia1 * nk_src + ik0] + tk * cfg.r_nu_2d[ia1 * nk_src + ik1];
      sp.r_nu_tab[j] = (1.0 - ta) * r0 + ta * r1;
    }
  } else {
    sp.n_k = 0;
  }
}

inline GravNuStepParams grav_nu_get_step_params() {
  return grav_nu_detail::step_params_storage();
}

// Device/host k-space boost factor W_nu(k, a) = 1 + (f_nu / (1 - f_nu)) * R_nu(k, a)
#ifdef __CUDACC__
__host__ __device__ __forceinline__
#else
inline
#endif
double grav_nu_pm_boost(double k_phys_inv_mpc, const GravNuStepParams& sp) {
  if (!sp.enabled) return 1.0;
  double r_nu = 0.0;
  if (sp.n_k >= 2 && k_phys_inv_mpc > 0.0) {
    const double lk = log(k_phys_inv_mpc);
    const double pos = (lk - sp.log_k_min) * sp.inv_dlog_k;
    if (pos <= 0.0) {
      r_nu = sp.r_nu_tab[0];
    } else if (pos >= double(sp.n_k - 1)) {
      r_nu = sp.r_nu_tab[sp.n_k - 1];
    } else {
      const int idx = int(pos);
      const double frac = pos - double(idx);
      r_nu = (1.0 - frac) * sp.r_nu_tab[idx] + frac * sp.r_nu_tab[idx + 1];
    }
  } else {
    // Quasi-static linear response: R_nu(k, a) = 1 / (1 + (k / k_fs(a))^2)
    r_nu = 1.0 / (1.0 + k_phys_inv_mpc * k_phys_inv_mpc * sp.inv_k_fs2);
  }
  return 1.0 + sp.boost_coef * r_nu;
}

#endif  // GRAV_NEUTRINO_CUH
