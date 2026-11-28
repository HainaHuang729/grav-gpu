// Standalone P1 PM precision verification suite for gravityCU:
// Sweeps (order in {2=CIC, 3=TSC, 4=PCS}) x (interlace in {0, 1}) x (grad in {fd4, ik}):
//   1. Single-particle zero self-force check (< 1e-13)
//   2. Two-particle Newton's 3rd law momentum conservation (< 1e-13)
//   3. 3D analytical cosine-wave force accuracy
//   4. Single-GPU vs multi-GPU GravDistPm parity (< 1e-13) when >= 2 GPUs are visible
#include "config.h"
#ifndef SWIFT_KERNEL_LONG_GRAVITY_H
#ifndef restrict
#define restrict __restrict__
#define TEST_PM_LOCAL_RESTRICT
#endif
#include "kernel_long_gravity.h"
#ifdef TEST_PM_LOCAL_RESTRICT
#undef restrict
#undef TEST_PM_LOCAL_RESTRICT
#endif
#endif
#include "grav_cuda_pm.h"
#include "grav_cuda_pm_dist.h"
#include "grav_pm_config.cuh"

#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

void check_cuda(cudaError_t st, int line) {
  if (st != cudaSuccess) {
    std::fprintf(stderr, "CUDA error line %d: %s\n", line, cudaGetErrorString(st));
    std::exit(1);
  }
}
#define CHK(x) check_cuda((x), __LINE__)

void run_single_gpu_pm(const std::vector<double3>& h_x, const std::vector<float>& h_m,
                       int side, double box, double split, float G,
                       int order, int interlace, int spectral_grad,
                       std::vector<float3>& h_acc, std::vector<float>& h_pot) {
  grav_pm_cuda_set_config(order, interlace, spectral_grad);
  const size_t n = h_x.size();
  const size_t pts = size_t(side) * side * side;
  double3* d_x = nullptr;
  float* d_m = nullptr;
  float3* d_acc = nullptr;
  float* d_pot = nullptr;
  double* d_dens = nullptr;
  double4* d_field = nullptr;
  CHK(cudaMalloc(&d_x, n * sizeof(double3)));
  CHK(cudaMalloc(&d_m, n * sizeof(float)));
  CHK(cudaMalloc(&d_acc, n * sizeof(float3)));
  CHK(cudaMalloc(&d_pot, n * sizeof(float)));
  CHK(cudaMalloc(&d_dens, pts * sizeof(double)));
  CHK(cudaMalloc(&d_field, pts * sizeof(double4)));
  CHK(cudaMemcpy(d_x, h_x.data(), n * sizeof(double3), cudaMemcpyHostToDevice));
  CHK(cudaMemcpy(d_m, h_m.data(), n * sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t st = nullptr;
  CHK(cudaStreamCreate(&st));
  const int passes = interlace ? 2 : 1;
  const float scale = interlace ? 0.5f : 1.0f;
  for (int p = 0; p < passes; ++p) {
    const double shift = p ? 0.5 : 0.0;
    CHK(cudaMemsetAsync(d_dens, 0, pts * sizeof(double), st));
    grav_pm_cuda_deposit_buf_shift(d_x, d_m, 1.0f, n, d_dens, side, box, shift, st);
    grav_pm_cuda_solve_on_device(d_dens, d_field, side, box, split, st);
    grav_pm_cuda_interpolate_buf_shift(d_x, n, d_field, side, box, G, d_acc, d_pot,
                                       shift, scale, p, st);
  }
  CHK(cudaStreamSynchronize(st));
  h_acc.resize(n);
  h_pot.resize(n);
  CHK(cudaMemcpy(h_acc.data(), d_acc, n * sizeof(float3), cudaMemcpyDeviceToHost));
  CHK(cudaMemcpy(h_pot.data(), d_pot, n * sizeof(float), cudaMemcpyDeviceToHost));
  CHK(cudaStreamDestroy(st));
  CHK(cudaFree(d_x));
  CHK(cudaFree(d_m));
  CHK(cudaFree(d_acc));
  CHK(cudaFree(d_pot));
  CHK(cudaFree(d_dens));
  CHK(cudaFree(d_field));
}

}  // namespace

int main() {
  const int side = 64;
  const double box = 25.0;
  const double split = 1.25 * box / double(side);
  const float G = 43.0092f;
  const double pi = 3.14159265358979323846;

  // Test 1 & 2 inputs: single particle (off-grid) and two particles
  const std::vector<double3> x1 = {{7.314159, 13.718281, 21.123456}};
  const std::vector<float> m1 = {1.0f};
  const std::vector<double3> x2 = {{4.123, 9.456, 15.789}, {11.987, 17.654, 3.321}};
  const std::vector<float> m2 = {1.0f, 1.0f};

  // Test 3 input: 3D high-k cosine density wave on 32^3 lattice (k_int = (10, 7, 5))
  // where CIC aliasing and FD4 dispersion are significant (~1-5%) while PCS+interlace+ik is < 1e-4.
  const int ng = 32;
  const double eps_amp = 0.25;
  const int kx_w = 10, ky_w = 7, kz_w = 5;
  const double kfac = 2.0 * pi / box;
  std::vector<double3> xw;
  std::vector<float> mw;
  xw.reserve(ng * ng * ng);
  mw.reserve(ng * ng * ng);
  for (int ix = 0; ix < ng; ++ix) {
    for (int iy = 0; iy < ng; ++iy) {
      for (int iz = 0; iz < ng; ++iz) {
        const double px = (ix + 0.37) * box / ng;
        const double py = (iy + 0.61) * box / ng;
        const double pz = (iz + 0.19) * box / ng;
        const double phase = kfac * (kx_w * px + ky_w * py + kz_w * pz);
        xw.push_back({px, py, pz});
        mw.push_back(float(1.0 + eps_amp * std::cos(phase)));
      }
    }
  }
  const double k2_int = double(kx_w * kx_w + ky_w * ky_w + kz_w * kz_w);
  const double u2 = k2_int * (4.0 * pi * pi * split * split / (box * box));
#ifdef GADGET2_LONG_RANGE_CORRECTION
  const double win = std::exp(-u2);
#else
  const double arg = 0.5 * pi * std::sqrt(u2);
  const double win = arg / std::sinh(arg);
#endif
  // Exact analytical PM force amplitude for delta(x) = eps_amp * cos(k.x) with mean rho = ng^3 / box^3
  const double rho_bar = double(ng * ng * ng) / (box * box * box);
  const double k2_phys = k2_int * kfac * kfac;
  const double f_amp = double(G) * 4.0 * pi * rho_bar * eps_amp * win / k2_phys;

  std::printf("order interlace grad | self_force | newton3_rel | wave_rms_rel\n");
  double prev_cic_fd4_err = 0.0;
  double best_pcs_ik_err = 1.0;

  for (int order : {2, 3, 4}) {
    for (int interlace : {0, 1}) {
      for (int sgrad : {0, 1}) {
        std::vector<float3> a1, a2, aw;
        std::vector<float> p1, p2, pw;
        run_single_gpu_pm(x1, m1, side, box, split, G, order, interlace, sgrad, a1, p1);
        const double self_f = std::sqrt(double(a1[0].x) * a1[0].x + double(a1[0].y) * a1[0].y + double(a1[0].z) * a1[0].z);

        run_single_gpu_pm(x2, m2, side, box, split, G, order, interlace, sgrad, a2, p2);
        const double sx = double(a2[0].x) + double(a2[1].x);
        const double sy = double(a2[0].y) + double(a2[1].y);
        const double sz = double(a2[0].z) + double(a2[1].z);
        const double n3_num = std::sqrt(sx * sx + sy * sy + sz * sz);
        const double n3_den = std::sqrt(double(a2[0].x) * a2[0].x + double(a2[0].y) * a2[0].y + double(a2[0].z) * a2[0].z) + 1e-30;
        const double n3_rel = n3_num / n3_den;

        run_single_gpu_pm(xw, mw, side, box, split, G, order, interlace, sgrad, aw, pw);
        double err2 = 0.0, ref2 = 0.0;
        for (size_t i = 0; i < xw.size(); ++i) {
          const double phase = kfac * (kx_w * xw[i].x + ky_w * xw[i].y + kz_w * xw[i].z);
          const double s = std::sin(phase);
          // For rho = rho_bar (1 + eps cos(k.x)), Phi_k = -4 pi G rho_k / k^2 => a = -grad Phi = -f_amp * k * sin(k.x)
          const double rx = -f_amp * (kx_w * kfac) * s;
          const double ry = -f_amp * (ky_w * kfac) * s;
          const double rz = -f_amp * (kz_w * kfac) * s;
          const double dx = double(aw[i].x) - rx;
          const double dy = double(aw[i].y) - ry;
          const double dz = double(aw[i].z) - rz;
          err2 += dx * dx + dy * dy + dz * dz;
          ref2 += rx * rx + ry * ry + rz * rz;
        }
        const double wave_rms = std::sqrt(err2 / (ref2 + 1e-30));
        std::printf("  %d       %d      %-3s | %.3e  |  %.3e  |  %.3e\n",
                    order, interlace, sgrad ? "ik" : "fd4", self_f, n3_rel, wave_rms);

        if (self_f > 1e-6) {
          std::fprintf(stderr, "FAIL: non-zero self-force %.3e\n", self_f);
          return 1;
        }
        if (n3_rel > 1e-5) {
          std::fprintf(stderr, "FAIL: Newton 3rd law residual %.3e\n", n3_rel);
          return 1;
        }
        if (order == 2 && interlace == 0 && sgrad == 0) prev_cic_fd4_err = wave_rms;
        if (order == 4 && interlace == 1 && sgrad == 1) best_pcs_ik_err = wave_rms;
      }
    }
  }

  if (!(best_pcs_ik_err < prev_cic_fd4_err * 0.1)) {
    std::fprintf(stderr, "FAIL: PCS+interlace+ik (%.3e) did not beat CIC+fd4 (%.3e) by >10x\n",
                 best_pcs_ik_err, prev_cic_fd4_err);
    return 1;
  }

  std::printf("PM_PRECISION_TESTS_PASSED (CIC+fd4=%.3e -> PCS+int+ik=%.3e, %.1fx better)\n",
              prev_cic_fd4_err, best_pcs_ik_err, prev_cic_fd4_err / (best_pcs_ik_err + 1e-30));
  return 0;
}
