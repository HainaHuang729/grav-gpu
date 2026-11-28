#ifndef GRAV_PHYSICS_STAGE_CUH
#define GRAV_PHYSICS_STAGE_CUH

#include <cmath>
#include "grav_neutrino.cuh"

// Shared Cosmological Background & Time-Integration Physics Model (v3.0 modular core).
// Supports flat/curved FLRW, w0/wa dynamic dark energy, Komatsu+2011 Fermi-Dirac massive
// neutrinos, and P3 adaptive dynamical/symplectic timestepping parameters.

struct GravPhysics {
  double a_begin, log_span;
  float box, split, cutoff, G;
  unsigned long long pm_span, short_span_cap;
  float short_step_factor, mac_tolerance;
  double dt_max, rms_length, max_dt_RMS_factor;
  bool dynamic_pm;
  double hubble0, omega_m, omega_de, omega_r, omega_k, w0, wa;
  float theta, grav_eta;
  bool periodic, comoving;
  float ext_M, ext_a, ext_cx, ext_cy, ext_cz;
  double omega_nu = 0.0, nu_m_each_ev = 0.0;
  int p3_tidal = 0, p3_sync = 0, p3_hyst = 0;
  float p3_tidal_eta = 0.05f, p3_vel_eta = 0.25f;

  static constexpr unsigned long long kTimeline = 1ULL << 57;

#ifdef __CUDACC__
  __host__ __device__
#endif
  double scale(unsigned long long ti) const {
    return comoving ? (a_begin * exp(log_span * (double(ti) / kTimeline))) : 1.0;
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
  double time_at(unsigned long long ti) const {
    return comoving ? scale(ti) : (a_begin + log_span * (double(ti) / kTimeline));
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
  double H(double a) const {
    if (!comoving) return 1.0;
    const double inv_a = 1.0 / a;
    const double a2 = inv_a * inv_a;
    const double a3 = a2 * inv_a;
    const double a4 = a2 * a2;
    double de_fac = 1.0;
    if (w0 != -1.0 || wa != 0.0) {
      de_fac = pow(a, -3.0 * (1.0 + w0 + wa)) * exp(-3.0 * wa * (1.0 - a));
    }
    double matter_term = omega_m * a3;
    if (omega_nu > 0.0) {
      const double om_cb = omega_m - omega_nu;
      matter_term = om_cb * a3 + omega_nu * a3 * grav_nu_rho_enhancement(a, nu_m_each_ev);
    }
    return hubble0 * sqrt(omega_r * a4 + matter_term + omega_k * a2 + omega_de * de_fac);
  }

#ifdef __CUDACC__
  __host__ __device__
#endif
  double factors(unsigned long long lo, unsigned long long hi, bool drift) const {
    if (lo == hi) return 0.0;
    if (!comoving) return log_span * (double(hi - lo) / kTimeline);
    const double a = scale(lo), b = scale(hi), m = (a + b) * 0.5;
    const double da = drift ? a * a * a : a * a;
    const double dm = drift ? m * m * m : m * m;
    const double db = drift ? b * b * b : b * b;
    return (b - a) / 6.0 * (1.0 / (da * H(a)) + 4.0 / (dm * H(m)) + 1.0 / (db * H(b)));
  }
};

#endif  // GRAV_PHYSICS_STAGE_CUH
