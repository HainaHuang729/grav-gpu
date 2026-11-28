#ifndef GRAV_PM_CONFIG_CUH
#define GRAV_PM_CONFIG_CUH

#include <cmath>
#include <cstdlib>
#include <cstring>

struct GravPmConfig {
  int order = 2;          // 2 = CIC (default), 3 = TSC, 4 = PCS
  int interlace = 0;      // 0 = single grid (default), 1 = symmetric half-cell interlacing
  int spectral_grad = 0;  // 0 = 4th-order FD (default), 1 = exact -i*k spectral gradient
  double asmth = 1.25;    // r_s / dx split ratio (default 1.25 = SWIFT)
};

namespace grav_pm_detail {

inline GravPmConfig& runtime_cfg_storage() {
  static GravPmConfig cfg{};
  return cfg;
}

inline bool& runtime_cfg_overridden() {
  static bool flag = false;
  return flag;
}

inline int parse_order_env() {
  const char* s = std::getenv("GRAV_PM_ORDER");
  if (!s || !*s) return 2;
  if (std::strcmp(s, "cic") == 0 || std::strcmp(s, "CIC") == 0) return 2;
  if (std::strcmp(s, "tsc") == 0 || std::strcmp(s, "TSC") == 0) return 3;
  if (std::strcmp(s, "pcs") == 0 || std::strcmp(s, "PCS") == 0) return 4;
  int v = std::atoi(s);
  return (v == 3 || v == 4) ? v : 2;
}

inline int parse_interlace_env() {
  const char* s = std::getenv("GRAV_PM_INTERLACE");
  return (s && *s && std::atoi(s) != 0) ? 1 : 0;
}

inline int parse_grad_env() {
  const char* s = std::getenv("GRAV_PM_GRAD");
  if (!s || !*s) return 0;
  if (std::strcmp(s, "ik") == 0 || std::strcmp(s, "IK") == 0 ||
      std::strcmp(s, "spectral") == 0 || std::strcmp(s, "1") == 0) {
    return 1;
  }
  return 0;
}

inline double parse_asmth_env() {
  const char* s = std::getenv("GRAV_PM_ASMTH");
  if (!s || !*s) return 1.25;
  double v = std::atof(s);
  return (std::isfinite(v) && v > 0.1 && v < 10.0) ? v : 1.25;
}

}  // namespace grav_pm_detail

inline GravPmConfig grav_pm_get_config() {
  if (grav_pm_detail::runtime_cfg_overridden()) {
    return grav_pm_detail::runtime_cfg_storage();
  }
  GravPmConfig cfg;
  cfg.order = grav_pm_detail::parse_order_env();
  cfg.interlace = grav_pm_detail::parse_interlace_env();
  cfg.spectral_grad = grav_pm_detail::parse_grad_env();
  cfg.asmth = grav_pm_detail::parse_asmth_env();
  return cfg;
}

inline void grav_pm_set_runtime_config(int order, int interlace, int spectral_grad, double asmth = 1.25) {
  auto& cfg = grav_pm_detail::runtime_cfg_storage();
  cfg.order = (order == 3 || order == 4) ? order : 2;
  cfg.interlace = interlace ? 1 : 0;
  cfg.spectral_grad = spectral_grad ? 1 : 0;
  cfg.asmth = (std::isfinite(asmth) && asmth > 0.1 && asmth < 10.0) ? asmth : 1.25;
  grav_pm_detail::runtime_cfg_overridden() = true;
}

inline int grav_pm_x_halo_lo(int order, int /*interlace*/) {
  return (order >= 3) ? 2 : 1;
}

inline int grav_pm_x_halo_hi(int order, int interlace) {
  if (order >= 4) return 3;
  if (order == 3 || interlace) return 2;
  return 1;
}

#ifdef __CUDACC__

template <int Order>
struct PmStencil {
  int base[3];
  double weight[3][Order];

  __device__ __forceinline__ PmStencil(double x0, double x1, double x2, int n, double box, double shift) {
    const double coords[3] = {x0, x1, x2};
    const double fac = double(n) / box;
    for (int axis = 0; axis < 3; ++axis) {
      double x = coords[axis];
      if (x < 0.0 || x >= box) x = x - floor(x / box) * box;
      double u = fac * x + shift;
      if (u >= double(n)) u -= double(n);
      if (u < 0.0) u += double(n);
      if constexpr (Order == 2) {
        int i0 = min(int(floor(u)), n - 1);
        const double d = u - double(i0);
        base[axis] = i0;
        weight[axis][0] = 1.0 - d;
        weight[axis][1] = d;
      } else if constexpr (Order == 3) {
        const int ic = int(floor(u + 0.5));
        const double d = u - double(ic);
        base[axis] = ic - 1;
        const double dm = 0.5 - d;
        const double dp = 0.5 + d;
        weight[axis][0] = 0.5 * dm * dm;
        weight[axis][1] = 0.75 - d * d;
        weight[axis][2] = 0.5 * dp * dp;
      } else {  // Order == 4 (PCS)
        const int i0 = int(floor(u));
        const double d = u - double(i0);
        const double omd = 1.0 - d;
        base[axis] = i0 - 1;
        weight[axis][0] = (omd * omd * omd) * (1.0 / 6.0);
        weight[axis][1] = (4.0 - 6.0 * d * d + 3.0 * d * d * d) * (1.0 / 6.0);
        weight[axis][2] = (4.0 - 6.0 * omd * omd + 3.0 * omd * omd * omd) * (1.0 / 6.0);
        weight[axis][3] = (d * d * d) * (1.0 / 6.0);
      }
    }
  }
};

__device__ __forceinline__ double grav_pm_deconv_weight(double inv_sinc_3d, int order) {
  const double s2 = inv_sinc_3d * inv_sinc_3d;
  const double s4 = s2 * s2;
  if (order == 3) return s4 * s2;
  if (order == 4) return s4 * s4;
  return s4;
}

#endif  // __CUDACC__
#endif  // GRAV_PM_CONFIG_CUH
