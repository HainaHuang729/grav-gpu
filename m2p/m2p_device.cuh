#include "m2p_port.h"
#include "m2p_native_math.cuh"

__global__ void native_m2p_pair(
    const int2* work, int swap, const int* pi, const int* pj, const int* counts,
    const int* offsets, const int* active_count, const int* active_offset,
    const int* active, const int* output_offset, const int* use_full,
    const float4* particles, const float* h, gravity_gpu_values_recv* out,
    m2p_batch_gpu m2p, int periodic, float r_s_inv, float Lx, float Ly, float Lz) {
  const int2 job=work[blockIdx.x];
  const int i=(swap?pj:pi)[job.x], j=(swap?pi:pj)[job.x];
  const int a=job.y+threadIdx.x;
  if (a>=active_count[i]) return;
  const int result=output_offset[2*job.x+swap]+a;
  m2p.mask[result]=0;
  if (counts[j]<=1) return;
  const int target=offsets[i]+active[active_offset[i]+a];
  const float4 x=particles[target];
  const auto* src=&m2p.sources[j];
  const float4 center=src->center_radius;
  // Match the direction used by SWIFT's native M2P force evaluator.
  float dx=center.x-x.x, dy=center.y-x.y, dz=center.z-x.z;
  if (periodic) {
    dx=nearestf1(dx,Lx);
    dy=nearestf1(dy,Ly);
    dz=nearestf1(dz,Lz);
  }
  const float r2=dx*dx+dy*dy+dz*dz;
  const float hmax=fmaxf(h[target],src->expansion.max_softening);
  if (!m2p.params.allow_soft && r2<=hmax*hmax) return;

  const float rho2=center.w*center.w;
  bool accept;
  if (m2p.params.adaptive) {
    const float old=m2p.old_accel[active_offset[i]+a];
    accept=rho2<r2 && 8.f*src->expansion.power[2]<m2p.params.tolerance*old*r2*r2;
  } else {
    accept=rho2<m2p.params.theta_sq*r2;
  }
  if (!accept) return;

  reduced_grav_tensor field{};
  swift_cuda_gravity_M2P(&src->expansion,dx,dy,dz,r2,hmax,
                        periodic && !use_full[job.x],r_s_inv,&field);
  out[result].values_i={field.F_100,field.F_010,field.F_001,field.F_000};
  m2p.mask[result]=1;
}
