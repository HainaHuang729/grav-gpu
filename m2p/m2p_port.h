#ifndef GRAV_CUDA_M2P_PORT_H
#define GRAV_CUDA_M2P_PORT_H
#include "gpu_mapping.h"
#include "multipole_struct.h"

struct m2p_source_gpu {
  float4 center_radius;
  struct multipole expansion;
};

struct m2p_params_gpu {
  float tolerance, theta_sq;
  int adaptive, allow_soft;
};

struct m2p_batch_gpu {
  const struct m2p_source_gpu *sources;
  const float *old_accel;
  unsigned char *mask;
  struct m2p_params_gpu params;
};
#endif
