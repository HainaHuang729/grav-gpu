#include "config.h"
#include "grav_cuda_pm.h"
#include "grav_pm_config.cuh"

// Read the split choice from SWIFT's source header in both standalone and
// integrated builds, even when that header is not in the include chain yet.
#ifndef SWIFT_KERNEL_LONG_GRAVITY_H
#ifndef restrict
#define restrict __restrict__
#define GRAV_PM_LOCAL_RESTRICT
#endif
#include "kernel_long_gravity.h"
#ifdef GRAV_PM_LOCAL_RESTRICT
#undef restrict
#undef GRAV_PM_LOCAL_RESTRICT
#endif
#endif

#include <cuda_runtime.h>
#include <cufft.h>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
constexpr int block_size = 256;
constexpr double pi = 3.1415926535897932384626433832795;

void check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "SWIFT CUDA PM: %s\n", cudaGetErrorString(status));
    std::abort();
  }
}
void check(cufftResult status) {
  if (status != CUFFT_SUCCESS) {
    std::fprintf(stderr, "SWIFT cuFFT error %d\n", int(status));
    std::abort();
  }
}

__device__ int wrapped_index(int x, int n) {
  const int r = x % n;
  return r < 0 ? r + n : r;
}
__device__ size_t index(int x, int y, int z, int n) {
  return (size_t(wrapped_index(x,n))*n+wrapped_index(y,n))*n+wrapped_index(z,n);
}

struct CIC {
  int cell[3];
  double weight[3][2];

  __device__ CIC(double x0, double x1, double x2, int n, double box) {
    const double coords[3] = {x0, x1, x2};
    double fac = n/box;
    for (int axis=0; axis<3; ++axis) {
      const double unwrapped = coords[axis];
      double x = unwrapped;
      // SWIFT positions can be just outside a periodic box at a rebuild.
      if (x<0 || x>=box) x = unwrapped-floor(unwrapped/box)*box;
      const double coordinate = fac*x;
      cell[axis] = min(int(coordinate),n-1);
      weight[axis][1] = coordinate-cell[axis];
      weight[axis][0] = 1.-weight[axis][1];
    }
  }
  __device__ explicit CIC(const grav_pm_particle& p, int n, double box)
      : CIC(p.x[0], p.x[1], p.x[2], n, box) {}
};

__device__ __forceinline__ const grav_pm_particle& pm_particle_at(
    const void* particles, size_t stride_bytes, size_t i) {
  return *reinterpret_cast<const grav_pm_particle*>(
      reinterpret_cast<const char*>(particles) + i * stride_bytes);
}

__global__ void deposit(const void* particles, size_t stride_bytes, size_t count,
                        double* density, int n, double box) {
  const size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i>=count) return;
  const auto& p = pm_particle_at(particles, stride_bytes, i);
  if (p.mass==0.) return;
  const CIC cic(p,n,box);
  for (int x=0;x<2;++x) for (int y=0;y<2;++y) for (int z=0;z<2;++z) {
    const size_t at = index(cic.cell[0]+x,cic.cell[1]+y,cic.cell[2]+z,n);
    atomicAdd(&density[at],p.mass*cic.weight[0][x]*cic.weight[1][y]*cic.weight[2][z]);
  }
}

__device__ double inverse_sinc(int mode, int n) {
  if (mode==0) return 1.;
  const double x = (pi/n)*mode;
  return x/sin(x);
}

__global__ void filter(cufftDoubleComplex* modes, int n, double box,
                       double split, int order, size_t count) {
  const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i>=count) return;
  if (i==0) {
    modes[0]={0.,0.};
    return;
  }
  const int half=n/2;
  int kx=(i/(half+1))/n, ky=(i/(half+1))%n, kz=i%(half+1);
  if(kx>half) kx-=n;
  if(ky>half) ky-=n;
  const double k2=double(kx)*kx+double(ky)*ky+double(kz)*kz;
  const double u2=k2*(4.*pi*pi*split*split/(box*box));
#ifdef GADGET2_LONG_RANGE_CORRECTION
  const double window=exp(-u2);
#else
  const double arg=0.5*pi*sqrt(u2);
  const double window=arg/(sinh(arg)+FLT_MIN);
#endif
  const double cic=inverse_sinc(kx,n)*inverse_sinc(ky,n)*inverse_sinc(kz,n);
  const double deconv=grav_pm_deconv_weight(cic, order);
  const double weight=(-1./(pi*box))*window/(k2+FLT_MIN)*deconv;
  modes[i].x*=weight;
  modes[i].y*=weight;
}

__global__ void spectral_grad_axis_kernel(const cufftDoubleComplex* __restrict__ phi_modes,
                                          cufftDoubleComplex* __restrict__ out_modes,
                                          int n, double two_pi_over_box, int axis, size_t count) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int half = n / 2;
  int kx = int((i / (half + 1)) / n);
  int ky = int((i / (half + 1)) % n);
  int kz = int(i % (half + 1));
  if (kx > half) kx -= n;
  if (ky > half) ky -= n;
  int k_int = (axis == 0) ? kx : ((axis == 1) ? ky : kz);
  if (k_int == half || k_int == -half || k_int == 0) {
    out_modes[i] = {0.0, 0.0};
    return;
  }
  const double k_phys = two_pi_over_box * double(k_int);
  const cufftDoubleComplex z = phi_modes[i];
  // -i * k * (re + i * im) = (+k * im, -k * re)
  out_modes[i] = {k_phys * z.y, -k_phys * z.x};
}

__global__ void pack_field_comp_kernel(const double* __restrict__ src,
                                       double4* __restrict__ field,
                                       size_t points, int axis) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= points) return;
  const double v = src[i];
  if (axis == 0) field[i].x = v;
  else if (axis == 1) field[i].y = v;
  else if (axis == 2) field[i].z = v;
  else field[i].w = v;
}

__device__ double stencil(double lo2,double lo1,double hi1,double hi2) {
  return (1./12.)*hi2-(2./3.)*hi1+(2./3.)*lo1-(1./12.)*lo2;
}
__global__ void differentiate(const double* potential, double4* field, int n, double fac) {
  const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i>=size_t(n)*n*n) return;
  const int x=(i/n)/n,y=(i/n)%n,z=i%n;
  field[i]={
    fac*stencil(potential[index(x-2,y,z,n)],potential[index(x-1,y,z,n)],
                potential[index(x+1,y,z,n)],potential[index(x+2,y,z,n)]),
    fac*stencil(potential[index(x,y-2,z,n)],potential[index(x,y-1,z,n)],
                potential[index(x,y+1,z,n)],potential[index(x,y+2,z,n)]),
    fac*stencil(potential[index(x,y,z-2,n)],potential[index(x,y,z-1,n)],
                potential[index(x,y,z+1,n)],potential[index(x,y,z+2,n)]),
    potential[i]};
}

__global__ void interpolate(const void* particles, size_t stride_bytes, grav_pm_force* result,
                             size_t count, const double4* field, int n, double box, float G) {
  const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i>=count) return;
  const CIC cic(pm_particle_at(particles, stride_bytes, i),n,box);
  double4 answer{};
  for (int x=0;x<2;++x) for (int y=0;y<2;++y) for (int z=0;z<2;++z) {
    const auto corner=field[index(cic.cell[0]+x,cic.cell[1]+y,cic.cell[2]+z,n)];
    const double w=cic.weight[0][x]*cic.weight[1][y]*cic.weight[2][z];
    answer.x+=w*corner.x;
    answer.y+=w*corner.y;
    answer.z+=w*corner.z;
    answer.w+=w*corner.w;
  }
  // The native solver stores each mesh result as float before applying G.
  result[i]={{G*float(answer.x), G*float(answer.y), G*float(answer.z)},G*float(answer.w)};
}

template <int Order>
__global__ void deposit_double3_t(const double3* x, const float* mass_or_null,
                                  float uniform_mass, size_t count,
                                  double* density, int n, double box, double shift) {
  const size_t i = size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i>=count) return;
  const double mp = double(mass_or_null ? mass_or_null[i] : uniform_mass);
  if (mp==0.) return;
  const double3 pos = x[i];
  const PmStencil<Order> st(pos.x, pos.y, pos.z, n, box, shift);
  #pragma unroll
  for (int ix=0; ix<Order; ++ix) {
    const double mx = mp * st.weight[0][ix];
    #pragma unroll
    for (int iy=0; iy<Order; ++iy) {
      const double mxy = mx * st.weight[1][iy];
      #pragma unroll
      for (int iz=0; iz<Order; ++iz) {
        const size_t at = index(st.base[0]+ix, st.base[1]+iy, st.base[2]+iz, n);
        atomicAdd(&density[at], mxy * st.weight[2][iz]);
      }
    }
  }
}

template <int Order>
__global__ void interpolate_double3_t(const double3* x, size_t count, const double4* field,
                                      int n, double box, float G,
                                      float3* acc_out, float* pot_out_or_null,
                                      double shift, float acc_scale, int accumulate) {
  const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if (i>=count) return;
  const double3 pos = x[i];
  const PmStencil<Order> st(pos.x, pos.y, pos.z, n, box, shift);
  double4 answer{};
  #pragma unroll
  for (int ix=0; ix<Order; ++ix) {
    #pragma unroll
    for (int iy=0; iy<Order; ++iy) {
      const double wxy = st.weight[0][ix] * st.weight[1][iy];
      #pragma unroll
      for (int iz=0; iz<Order; ++iz) {
        const auto corner=field[index(st.base[0]+ix, st.base[1]+iy, st.base[2]+iz, n)];
        const double w = wxy * st.weight[2][iz];
        answer.x+=w*corner.x;
        answer.y+=w*corner.y;
        answer.z+=w*corner.z;
        answer.w+=w*corner.w;
      }
    }
  }
  const float s = G * acc_scale;
  const float ax = s * float(answer.x);
  const float ay = s * float(answer.y);
  const float az = s * float(answer.z);
  if (accumulate) {
    acc_out[i].x += ax;
    acc_out[i].y += ay;
    acc_out[i].z += az;
    if (pot_out_or_null) pot_out_or_null[i] += s * float(answer.w);
  } else {
    acc_out[i] = {ax, ay, az};
    if (pot_out_or_null) pot_out_or_null[i] = s * float(answer.w);
  }
}

struct Workspace {
  double* potential=nullptr;
  double4* field=nullptr;
  cufftDoubleComplex* modes=nullptr;
  grav_pm_particle* inputs=nullptr;
  grav_pm_force* outputs=nullptr;
  cufftHandle forward, reverse;
  cudaStream_t stream;
  size_t capacity=0;
  int n=0;

  Workspace() { check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)); }
  ~Workspace() {
    clear_mesh();
    if (capacity) { cudaFree(inputs); cudaFree(outputs); }
    cudaStreamDestroy(stream);
  }
  void clear_mesh() {
    if (!n) return;
    cufftDestroy(forward);
    cufftDestroy(reverse);
    cudaFree(modes);
    cudaFree(field);
    cudaFree(potential);
  }
  void mesh_for(int side) {
    if (side==n) return;
    clear_mesh();
    n=side;
    const size_t points=size_t(n)*n*n;
    check(cudaMalloc(&potential,points*sizeof(*potential)));
    check(cudaMalloc(&field,points*sizeof(*field)));
    check(cudaMalloc(&modes,size_t(n)*n*(n/2+1)*sizeof(*modes)));
    check(cufftPlan3d(&forward,n,n,n,CUFFT_D2Z));
    check(cufftPlan3d(&reverse,n,n,n,CUFFT_Z2D));
    check(cufftSetStream(forward,stream));
    check(cufftSetStream(reverse,stream));
  }
  void particles_for(size_t count) {
    if (count<=capacity) return;
    if (capacity) { check(cudaFree(inputs)); check(cudaFree(outputs)); }
    check(cudaMalloc(&inputs,count*sizeof(*inputs)));
    check(cudaMalloc(&outputs,count*sizeof(*outputs)));
    capacity=count;
  }
};

Workspace& workspace() {
  static Workspace ws;
  return ws;
}

struct SolveWorkspace {
  cufftDoubleComplex* modes=nullptr;
  cufftDoubleComplex* grad_modes=nullptr;
  double* grad_real=nullptr;
  cufftHandle forward{}, reverse{};
  int n=0;
  int dev=-1;

  void clear() {
    if (!n) return;
    int prev=-1;
    if (dev>=0 && cudaGetDevice(&prev)==cudaSuccess && prev!=dev) cudaSetDevice(dev);
    cufftDestroy(forward);
    cufftDestroy(reverse);
    cudaFree(modes);
    if (grad_modes) cudaFree(grad_modes);
    if (grad_real) cudaFree(grad_real);
    modes=nullptr;
    grad_modes=nullptr;
    grad_real=nullptr;
    n=0;
    dev=-1;
    if (prev>=0 && cudaGetDevice(&prev)==cudaSuccess) cudaSetDevice(prev);
  }
  ~SolveWorkspace() { clear(); }
  void prepare(int side, cudaStream_t stream, bool need_spectral) {
    int cur=0;
    check(cudaGetDevice(&cur));
    if (n!=side || dev!=cur) {
      clear();
      n=side;
      dev=cur;
      check(cudaMalloc(&modes, size_t(n)*n*(n/2+1)*sizeof(*modes)));
      check(cufftPlan3d(&forward,n,n,n,CUFFT_D2Z));
      check(cufftPlan3d(&reverse,n,n,n,CUFFT_Z2D));
    }
    if (need_spectral && !grad_modes) {
      check(cudaMalloc(&grad_modes, size_t(n)*n*(n/2+1)*sizeof(*grad_modes)));
      check(cudaMalloc(&grad_real, size_t(n)*n*n*sizeof(*grad_real)));
    }
    check(cufftSetStream(forward,stream));
    check(cufftSetStream(reverse,stream));
  }
};

SolveWorkspace& solve_workspace() {
  static SolveWorkspace sw;
  return sw;
}

template<class Run> double measure(cudaStream_t stream, Run run) {
  cudaEvent_t start,stop;
  check(cudaEventCreate(&start));
  check(cudaEventCreate(&stop));
  check(cudaEventRecord(start,stream));
  run();
  check(cudaGetLastError());
  check(cudaEventRecord(stop,stream));
  check(cudaEventSynchronize(stop));
  float ms;
  check(cudaEventElapsedTime(&ms,start,stop));
  check(cudaEventDestroy(start));
  check(cudaEventDestroy(stop));
  return ms;
}

unsigned int blocks(size_t count) {
  return (count+block_size-1)/block_size;
}

void enqueue_deposit(Workspace& w,const void* particles,size_t stride_bytes,size_t count,
                     int side,double box,cudaStream_t stream) {
  const size_t points=size_t(side)*side*side;
  check(cudaMemsetAsync(w.potential,0,points*sizeof(*w.potential),stream));
  deposit<<<blocks(count),block_size,0,stream>>>(particles,stride_bytes,count,w.potential,side,box);
}

void enqueue_transform(Workspace& w,int side,double box,double split,cudaStream_t stream) {
  const auto cfg = grav_pm_get_config();
  const size_t modes=size_t(side)*side*(side/2+1);
  check(cufftExecD2Z(w.forward,w.potential,w.modes));
  filter<<<blocks(modes),block_size,0,stream>>>(w.modes,side,box,split,cfg.order,modes);
  check(cufftExecZ2D(w.reverse,w.modes,w.potential));
}

void enqueue_interpolate(Workspace& w,const void* particles,size_t stride_bytes,
                         grav_pm_force* result,size_t count,int side,double box,float G,
                         cudaStream_t stream) {
  const size_t points=size_t(side)*side*side;
  differentiate<<<blocks(points),block_size,0,stream>>>(w.potential,w.field,side,side/box);
  interpolate<<<blocks(count),block_size,0,stream>>>(
      particles,stride_bytes,result,count,w.field,side,box,G);
}

}

extern "C" void grav_pm_cuda_device(const grav_pm_particle* particles, size_t count,
                                      int side, double box, double split, float G,
                                      grav_pm_force* result, grav_pm_timing* timing) {
  *timing={};
  if (!count) return;
  auto& w=workspace();
  w.mesh_for(side);
  check(cufftSetStream(w.forward,w.stream));
  check(cufftSetStream(w.reverse,w.stream));
  timing->deposit_ms=measure(w.stream,[&]{
    enqueue_deposit(w,particles,sizeof(grav_pm_particle),count,side,box,w.stream);
  });
  timing->fft_ms=measure(w.stream,[&]{
    enqueue_transform(w,side,box,split,w.stream);
  });
  timing->interpolate_ms=measure(w.stream,[&]{
    enqueue_interpolate(w,particles,sizeof(grav_pm_particle),result,count,side,box,G,w.stream);
  });
}

extern "C" void grav_pm_cuda_graph_prepare(int side,void* stream) {
  auto& w=workspace();
  w.mesh_for(side);
  check(cufftSetStream(w.forward,static_cast<cudaStream_t>(stream)));
  check(cufftSetStream(w.reverse,static_cast<cudaStream_t>(stream)));
}

extern "C" void grav_pm_cuda_graph_enqueue_strided(const void* particles,
                                                    size_t stride_bytes,
                                                    size_t count,int side,double box,double split,float G,
                                                    grav_pm_force* result,void* handle) {
  auto& w=workspace();
  const auto stream=static_cast<cudaStream_t>(handle);
  enqueue_deposit(w,particles,stride_bytes,count,side,box,stream);
  enqueue_transform(w,side,box,split,stream);
  enqueue_interpolate(w,particles,stride_bytes,result,count,side,box,G,stream);
  check(cudaGetLastError());
}

extern "C" void grav_pm_cuda_graph_enqueue(const grav_pm_particle* particles,
                                            size_t count,int side,double box,double split,float G,
                                            grav_pm_force* result,void* handle) {
  grav_pm_cuda_graph_enqueue_strided(particles,sizeof(grav_pm_particle),count,side,box,split,G,result,handle);
}

extern "C" void grav_pm_cuda_begin(int side,void* handle) {
  auto& w=workspace();
  const auto stream=static_cast<cudaStream_t>(handle);
  w.mesh_for(side);
  check(cufftSetStream(w.forward,stream));
  check(cufftSetStream(w.reverse,stream));
  const size_t points=size_t(side)*side*side;
  check(cudaMemsetAsync(w.potential,0,points*sizeof(*w.potential),stream));
}

extern "C" void grav_pm_cuda_deposit_double3(const void* d_x_double3,
                                              const float* d_mass_or_null,
                                              float uniform_mass,size_t count,
                                              int side,double box,void* handle) {
  if (!count) return;
  auto& w=workspace();
  grav_pm_cuda_deposit_buf_shift(d_x_double3, d_mass_or_null, uniform_mass, count,
                                 w.potential, side, box, 0.0, handle);
}

extern "C" void grav_pm_cuda_solve(int side,double box,double split,void* handle) {
  auto& w=workspace();
  grav_pm_cuda_solve_on_device(w.potential, w.field, side, box, split, handle);
}

extern "C" void grav_pm_cuda_interpolate_double3(const void* d_x_double3,size_t count,
                                                  int side,double box,float grav_G,
                                                  void* d_acc_float3,float* d_pot_or_null,
                                                  void* handle) {
  if (!count) return;
  auto& w=workspace();
  grav_pm_cuda_interpolate_buf_shift(d_x_double3, count, w.field, side, box, grav_G,
                                     d_acc_float3, d_pot_or_null, 0.0, 1.0f, 0, handle);
}

extern "C" void grav_pm_cuda_host(const grav_pm_particle* particles, size_t count,
                                    int side, double box, double split, float G,
                                    grav_pm_force* result, grav_pm_timing* timing) {
  if (!count) { *timing={}; return; }
  auto& w=workspace();
  w.particles_for(count);
  const double upload=measure(w.stream,[&]{
    check(cudaMemcpyAsync(w.inputs,particles,count*sizeof(*particles),cudaMemcpyHostToDevice,w.stream));
  });
  grav_pm_cuda_device(w.inputs,count,side,box,split,G,w.outputs,timing);
  timing->import_ms=upload;
  timing->export_ms=measure(w.stream,[&]{
    check(cudaMemcpyAsync(result,w.outputs,count*sizeof(*result),cudaMemcpyDeviceToHost,w.stream));
  });
}

extern "C" void grav_pm_cuda_set_config(int order, int interlace, int spectral_grad) {
  grav_pm_set_runtime_config(order, interlace, spectral_grad);
}

extern "C" void grav_pm_cuda_deposit_buf_shift(const void* d_x_double3,
                                                const float* d_mass_or_null,
                                                float uniform_mass, size_t count,
                                                double* d_density, int side,
                                                double box, double shift, void* handle) {
  if (!count) return;
  const auto stream=static_cast<cudaStream_t>(handle);
  const auto cfg = grav_pm_get_config();
  if (cfg.order == 3) {
    deposit_double3_t<3><<<blocks(count),block_size,0,stream>>>(
        static_cast<const double3*>(d_x_double3),d_mass_or_null,uniform_mass,
        count,d_density,side,box,shift);
  } else if (cfg.order == 4) {
    deposit_double3_t<4><<<blocks(count),block_size,0,stream>>>(
        static_cast<const double3*>(d_x_double3),d_mass_or_null,uniform_mass,
        count,d_density,side,box,shift);
  } else {
    deposit_double3_t<2><<<blocks(count),block_size,0,stream>>>(
        static_cast<const double3*>(d_x_double3),d_mass_or_null,uniform_mass,
        count,d_density,side,box,shift);
  }
  check(cudaGetLastError());
}

extern "C" void grav_pm_cuda_deposit_buf(const void* d_x_double3,
                                          const float* d_mass_or_null,
                                          float uniform_mass, size_t count,
                                          double* d_density, int side,
                                          double box, void* handle) {
  grav_pm_cuda_deposit_buf_shift(d_x_double3, d_mass_or_null, uniform_mass, count,
                                 d_density, side, box, 0.0, handle);
}

extern "C" void grav_pm_cuda_solve_on_device(double* d_density, void* d_field_double4,
                                              int side, double box, double split,
                                              void* handle) {
  const auto stream=static_cast<cudaStream_t>(handle);
  const auto cfg = grav_pm_get_config();
  auto& sw=solve_workspace();
  sw.prepare(side,stream,cfg.spectral_grad != 0);
  const size_t modes_cnt=size_t(side)*side*(side/2+1);
  const size_t points=size_t(side)*side*side;
  check(cufftExecD2Z(sw.forward,d_density,sw.modes));
  filter<<<blocks(modes_cnt),block_size,0,stream>>>(sw.modes,side,box,split,cfg.order,modes_cnt);
  if (cfg.spectral_grad) {
    auto* field = static_cast<double4*>(d_field_double4);
    check(cufftExecZ2D(sw.reverse,sw.modes,d_density));
    pack_field_comp_kernel<<<blocks(points),block_size,0,stream>>>(d_density,field,points,3);
    const double two_pi_over_box = 2.0 * pi / box;
    for (int axis = 0; axis < 3; ++axis) {
      spectral_grad_axis_kernel<<<blocks(modes_cnt),block_size,0,stream>>>(
          sw.modes, sw.grad_modes, side, two_pi_over_box, axis, modes_cnt);
      check(cufftExecZ2D(sw.reverse,sw.grad_modes,sw.grad_real));
      pack_field_comp_kernel<<<blocks(points),block_size,0,stream>>>(sw.grad_real,field,points,axis);
    }
  } else {
    check(cufftExecZ2D(sw.reverse,sw.modes,d_density));
    differentiate<<<blocks(points),block_size,0,stream>>>(
        d_density,static_cast<double4*>(d_field_double4),side,side/box);
  }
  check(cudaGetLastError());
}

extern "C" void grav_pm_cuda_interpolate_buf_shift(const void* d_x_double3, size_t count,
                                                    const void* d_field_double4, int side,
                                                    double box, float grav_G,
                                                    void* d_acc_float3, float* d_pot_or_null,
                                                    double shift, float acc_scale,
                                                    int accumulate, void* handle) {
  if (!count) return;
  const auto stream=static_cast<cudaStream_t>(handle);
  const auto cfg = grav_pm_get_config();
  if (cfg.order == 3) {
    interpolate_double3_t<3><<<blocks(count),block_size,0,stream>>>(
        static_cast<const double3*>(d_x_double3),count,
        static_cast<const double4*>(d_field_double4),side,box,grav_G,
        static_cast<float3*>(d_acc_float3),d_pot_or_null,shift,acc_scale,accumulate);
  } else if (cfg.order == 4) {
    interpolate_double3_t<4><<<blocks(count),block_size,0,stream>>>(
        static_cast<const double3*>(d_x_double3),count,
        static_cast<const double4*>(d_field_double4),side,box,grav_G,
        static_cast<float3*>(d_acc_float3),d_pot_or_null,shift,acc_scale,accumulate);
  } else {
    interpolate_double3_t<2><<<blocks(count),block_size,0,stream>>>(
        static_cast<const double3*>(d_x_double3),count,
        static_cast<const double4*>(d_field_double4),side,box,grav_G,
        static_cast<float3*>(d_acc_float3),d_pot_or_null,shift,acc_scale,accumulate);
  }
  check(cudaGetLastError());
}

extern "C" void grav_pm_cuda_interpolate_buf(const void* d_x_double3, size_t count,
                                              const void* d_field_double4, int side,
                                              double box, float grav_G,
                                              void* d_acc_float3, float* d_pot_or_null,
                                              void* handle) {
  grav_pm_cuda_interpolate_buf_shift(d_x_double3, count, d_field_double4, side, box, grav_G,
                                     d_acc_float3, d_pot_or_null, 0.0, 1.0f, 0, handle);
}
