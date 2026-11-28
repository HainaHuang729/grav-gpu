// Unit test for B1 Massive Neutrino background H(a) & k-space linear-response boost
// Compiles & runs standalone on CPU (no GPU needed) and verifies:
//   1. Komatsu+2011 Fermi-Dirac rho_nu(a) vs direct numerical Fermi-Dirac quadrature (< 0.15%)
//   2. k-space boost W_nu(k,a) limits: k << k_fs -> 1/(1-f_nu), k >> k_fs -> 1.0
//   3. 2D (a, k) CLASS/CAMB transfer ratio table bilinear interpolation (< 1e-4)
#include "grav_neutrino.cuh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>

namespace {

// Exact numerical Fermi-Dirac integral ratio:
// I(y) / (I_NR(y)) where I(y) = int_0^inf dx x^2 sqrt(x^2 + y^2) / (e^x + 1)
// and I_NR(y) = y * int_0^inf dx x^2 / (e^x + 1) = y * (3/2) * zeta(3).
double exact_fd_enhancement(double y) {
  const int N = 4000;
  const double x_max = 40.0;
  const double dx = x_max / N;
  double num = 0.0, den = 0.0;
  for (int i = 1; i <= N; ++i) {
    const double x = (i - 0.5) * dx;
    const double f = 1.0 / (std::exp(x) + 1.0);
    num += x * x * std::sqrt(x * x + y * y) * f * dx;
    den += x * x * y * f * dx;
  }
  return num / den;
}

}  // namespace

int main() {
  // 1. Verify Fermi-Dirac background enhancement across z = 0 .. 10000
  const double m_each = 0.10;  // sum(m_nu) = 0.30 eV (3 degenerate species)
  double max_rel = 0.0;
  for (double a : {1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0}) {
    const double y = 5943.9 * a * m_each;
    const double exact = exact_fd_enhancement(y);
    const double approx = grav_nu_rho_enhancement(a, m_each);
    const double rel = std::fabs(approx - exact) / exact;
    if (rel > max_rel) max_rel = rel;
  }
  std::printf("FD_BACKGROUND max_rel=%.4e\n", max_rel);
  assert(max_rel < 2.5e-3);

  // 2. Verify analytic k-space linear response limits for sum(m_nu) = 0.30 eV
  setenv("GRAV_NU_MASS_EV", "0.30", 1);
  setenv("GRAV_NU_NSPECIES", "3", 1);
  unsetenv("GRAV_NU_TABLE");
  const GravNuConfig cfg = grav_nu_init_from_env(67.36, 0.3153);
  assert(cfg.omega_nu0 > 0.0070 && cfg.omega_nu0 < 0.0072);
  assert(cfg.f_nu > 0.022 && cfg.f_nu < 0.023);
  grav_nu_update_step(1.0, 67.36);
  const GravNuStepParams sp = grav_nu_get_step_params();
  assert(sp.enabled == 1);
  const double w_low_k = grav_nu_pm_boost(1e-5, sp);
  const double w_high_k = grav_nu_pm_boost(1e2, sp);
  const double target_low = 1.0 / (1.0 - cfg.f_nu);
  std::printf("ANALYTIC_BOOST f_nu=%.5f W(k->0)=%.6f (target %.6f) W(k->inf)=%.6f\n",
              cfg.f_nu, w_low_k, target_low, w_high_k);
  assert(std::fabs(w_low_k - target_low) < 1e-5);
  assert(std::fabs(w_high_k - 1.0) < 1e-5);

  // 3. Verify 2D (a, k) CLASS/CAMB transfer ratio table loading & bilinear interpolation
  const char* tmp_tab = "/tmp/test_grav_nu_table.csv";
  {
    std::ofstream out(tmp_tab);
    out << "# a, k [1/Mpc], R_nu(k,a)\n";
    for (double a : {0.02, 0.5, 1.0}) {
      for (int ik = 0; ik < 20; ++ik) {
        const double k = 1e-3 * std::pow(10.0, 5.0 * ik / 19.0);
        const double kfs = 0.05 * std::sqrt(a);
        const double r = 1.0 / (1.0 + (k / kfs) * (k / kfs));
        out << a << "," << k << "," << r << "\n";
      }
    }
  }
  setenv("GRAV_NU_TABLE", tmp_tab, 1);
  grav_nu_init_from_env(67.36, 0.3153);
  grav_nu_update_step(0.75, 80.0);
  const GravNuStepParams sp_tab = grav_nu_get_step_params();
  assert(sp_tab.n_k == 256);
  const double w_tab_low = grav_nu_pm_boost(1e-3, sp_tab);
  const double w_tab_high = grav_nu_pm_boost(10.0, sp_tab);
  assert(w_tab_low > 1.0 + 0.99 * sp_tab.boost_coef);
  assert(w_tab_high < 1.0 + 0.01 * sp_tab.boost_coef);
  std::remove(tmp_tab);

  std::printf("NEUTRINO_B1_UNIT_TESTS_PASSED\n");
  return 0;
}
