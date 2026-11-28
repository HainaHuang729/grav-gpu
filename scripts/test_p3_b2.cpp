// CPU unit tests for P3 (adaptive tidal/dynamical timestep) and B2 (f(R) gravity + MC-SIDM).
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "grav_timestep_p3.cuh"
#include "grav_fr_gravity.cuh"
#include "grav_sidm.cuh"

int main() {
  // 1. P3 dynamical / velocity-crossing timestep limits
  {
    const float dt_acc = 0.1f;
    // Deep halo core: |Phi| = 1.0, |a| = 10.0 -> tau_dyn = 0.1, dt_dyn = 0.05 * 0.1 = 0.005
    const float dt1 = grav_p3_dt_limit(dt_acc, 10.f, 1.f, 0.5f, 0.05f, 1, 0.05f, 0.25f);
    assert(std::fabs(dt1 - 0.005f) < 1e-6f);
    // Fast core-passing particle: v = 100, eps = 0.05 -> dt_vel = 0.25 * 0.05 / 100 = 1.25e-4
    const float dt2 = grav_p3_dt_limit(dt_acc, 1.f, 1.f, 100.f, 0.05f, 1, 0.05f, 0.25f);
    assert(std::fabs(dt2 - 1.25e-4f) < 1e-7f);
    // Disabled -> unchanged
    const float dt3 = grav_p3_dt_limit(dt_acc, 10.f, 1.f, 100.f, 0.05f, 0, 0.05f, 0.25f);
    assert(dt3 == dt_acc);
    std::printf("P3_TIMESTEP_OK dt_dyn=%.5f dt_vel=%.6f\n", dt1, dt2);
  }

  // 2. B2-b: Hu-Sawicki n=1 f(R) modified gravity + chameleon screening
  {
    setenv("GRAV_FR0", "1e-4", 1);
    setenv("GRAV_FR_SCREENING", "1", 1);
    const auto cfg = grav_fr_init_from_env(67.36, 0.3153, 0.6847);
    assert(std::fabs(cfg.f_R0 - 1e-4) < 1e-12);
    grav_fr_update_step(1.0);
    const auto sp = grav_fr_get_step_params();
    assert(sp.enabled == 1);
    // Check Compton wavelength at z=0 for |f_R0|=1e-4: lambda_C ~ 24.3 Mpc/h (~36.0 Mpc)
    const double lambda_c_mpc = 1.0 / std::sqrt(sp.m_fr2);
    assert(lambda_c_mpc > 30.0 && lambda_c_mpc < 42.0);
    // Check Yukawa boost limits: k -> 0 gives 1.0, k -> inf gives 4/3
    const double g_low = grav_fr_pm_boost(1e-6, sp);
    const double g_high = grav_fr_pm_boost(1e3, sp);
    assert(std::fabs(g_low - 1.0) < 1e-6);
    assert(std::fabs(g_high - (4.0 / 3.0)) < 1e-6);
    // Check chameleon screening: low potential -> S=1, deep potential -> S << 1
    const double s_unscreened = grav_fr_chameleon_screen(1e3, sp);
    const double s_screened = grav_fr_chameleon_screen(1e8, sp);
    assert(std::fabs(s_unscreened - 1.0) < 1e-12);
    assert(s_screened > 0.0 && s_screened < 0.2);
    std::printf("B2_FR_OK lambda_C=%.2fMpc G_eff(0)=%.6f G_eff(inf)=%.6f S_screen=%.4f\n",
                lambda_c_mpc, g_low, g_high, s_screened);
  }

  // 3. B2-a: MC-SIDM cross-section, probability, and RNG uniformity
  {
    setenv("GRAV_SIDM_SIGMA_M", "1.0", 1);
    setenv("GRAV_SIDM_W0", "100.0", 1);
    const auto scfg = grav_sidm_init_from_env();
    assert(std::fabs(scfg.sigma_m_internal - 2.088357e-6) < 1e-10);
    const double sig0 = grav_sidm_sigma_of_v(1.0, 100.0, 0.0);
    const double sig_w = grav_sidm_sigma_of_v(1.0, 100.0, 100.0);
    assert(std::fabs(sig0 - 1.0) < 1e-12);
    assert(std::fabs(sig_w - 0.25) < 1e-12);
    const double p_in = grav_sidm_pair_prob(0.01, 0.2, 1.0, 1.0, 1.0, 0.01);
    const double p_out = grav_sidm_pair_prob(0.05, 0.2, 1.0, 1.0, 1.0, 0.01);
    assert(p_in > 0.0 && p_out == 0.0);
    // Verify SplitMix64 uniform mean ~ 0.5
    double sum = 0.0;
    constexpr int M = 10000;
    for (int i = 0; i < M; ++i) sum += grav_sidm_u01(grav_sidm_splitmix64(uint64_t(i + 1)));
    assert(std::fabs(sum / M - 0.5) < 0.02);
    std::printf("B2_SIDM_OK sigma_int=%.6e sig(w0)=%.3f p_in=%.5f rng_mean=%.4f\n",
                scfg.sigma_m_internal, sig_w, p_in, sum / M);
  }

  std::printf("P3_B2_UNIT_TESTS_PASSED\n");
  return 0;
}
