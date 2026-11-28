#ifndef GRAV_CUDA_PM_H
#define GRAV_CUDA_PM_H

#include <stddef.h>

struct grav_pm_particle {
  double x[3];
  double mass;
};
struct grav_pm_force {
  float acceleration[3];
  float potential;
};
struct grav_pm_timing {
  double import_ms, deposit_ms, fft_ms, interpolate_ms, export_ms;
};

#ifdef __cplusplus
extern "C" {
#endif

/* The device entry point leaves all particle arrays and results on the GPU.
 * The host adapter supplies staging for the existing CPU-managed SWIFT tree. */
void grav_pm_cuda_device(const struct grav_pm_particle *device_particles,
                          size_t count, int side, double box, double split,
                          float grav_G, struct grav_pm_force *device_result,
                          struct grav_pm_timing *timing);
void grav_pm_cuda_host(const struct grav_pm_particle *particles, size_t count,
                        int side, double box, double split, float grav_G,
                        struct grav_pm_force *result, struct grav_pm_timing *timing);

/* Prepare plans before capture, then enqueue the PM stages without host waits.
 * Pass the same CUDA stream to both functions. */
void grav_pm_cuda_graph_prepare(int side, void *stream);
void grav_pm_cuda_graph_enqueue(const struct grav_pm_particle *device_particles,
                                 size_t count, int side, double box, double split,
                                 float grav_G, struct grav_pm_force *device_result,
                                 void *stream);
void grav_pm_cuda_graph_enqueue_strided(const void *device_particles,
                                         size_t stride_bytes,
                                         size_t count, int side, double box,
                                         double split, float grav_G,
                                         struct grav_pm_force *device_result,
                                         void *stream);

/* Streaming slab PM APIs: deposit multiple GPU slabs before solving, then
 * interpolate acceleration (and optional potential) slab-by-slab. */
void grav_pm_cuda_begin(int side, void *stream);
void grav_pm_cuda_deposit_double3(const void *d_x_double3,
                                   const float *d_mass_or_null,
                                   float uniform_mass, size_t count, int side,
                                   double box, void *stream);
void grav_pm_cuda_solve(int side, double box, double split, void *stream);
void grav_pm_cuda_interpolate_double3(const void *d_x_double3, size_t count,
                                       int side, double box, float grav_G,
                                       void *d_acc_float3, float *d_pot_or_null,
                                       void *stream);

/* Stateless / explicit-buffer multi-GPU PM APIs. */
void grav_pm_cuda_set_config(int order, int interlace, int spectral_grad);

void grav_pm_cuda_deposit_buf(const void *d_x_double3,
                               const float *d_mass_or_null,
                               float uniform_mass, size_t count,
                               double *d_density, int side, double box,
                               void *stream);
void grav_pm_cuda_deposit_buf_shift(const void *d_x_double3,
                                     const float *d_mass_or_null,
                                     float uniform_mass, size_t count,
                                     double *d_density, int side, double box,
                                     double shift, void *stream);
/* Note: overwrites d_density in-place with filtered real-space potential. */
void grav_pm_cuda_solve_on_device(double *d_density, void *d_field_double4,
                                   int side, double box, double split,
                                   void *stream);
void grav_pm_cuda_interpolate_buf(const void *d_x_double3, size_t count,
                                   const void *d_field_double4, int side,
                                   double box, float grav_G,
                                   void *d_acc_float3, float *d_pot_or_null,
                                   void *stream);
void grav_pm_cuda_interpolate_buf_shift(const void *d_x_double3, size_t count,
                                         const void *d_field_double4, int side,
                                         double box, float grav_G,
                                         void *d_acc_float3, float *d_pot_or_null,
                                         double shift, float acc_scale,
                                         int accumulate, void *stream);

#ifdef __cplusplus
}
#endif
#endif
