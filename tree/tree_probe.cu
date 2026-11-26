#include "tree_format.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <vector>

using Key=uint64_t;
constexpr int octree_shift=3*PROBE_LEVELS;

void check(cudaError_t status) {
  if (status!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
template<class T> struct Device {
  T* p;
  explicit Device(size_t n) { check(cudaMalloc(&p,n*sizeof(T))); }
  ~Device() { cudaFree(p); }
  Device(const Device&)=delete;
  Device& operator=(const Device&)=delete;
};
template<class F> float ms(F run) {
  cudaEvent_t a,b;
  check(cudaEventCreate(&a)); check(cudaEventCreate(&b)); check(cudaEventRecord(a));
  run(); check(cudaGetLastError()); check(cudaEventRecord(b)); check(cudaEventSynchronize(b));
  float time;
  check(cudaEventElapsedTime(&time,a,b)); check(cudaEventDestroy(a)); check(cudaEventDestroy(b));
  return time;
}

struct Geometry {
  Key key;
  int start, end;
  int children[8];
};

__global__ void key_particles(const probe_particle* in, int n, probe_header h,
                              Key* keys, int* order, int* invalid) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n) return;
  int top[3];
  double lower[3],width[3];
  for (int a=0;a<3;++a) {
    top[a]=int(in[i].pm.x[a]*h.inv_width[a]);
    if (top[a]<0 || top[a]>=h.top[a]) {
      atomicAdd(invalid,1);
      top[a]=0;
    }
    width[a]=h.width[a];
    lower[a]=top[a]*width[a];
  }
  Key key=Key((top[0]*h.top[1]+top[1])*h.top[2]+top[2]) << octree_shift;
  for (int level=1;level<=PROBE_LEVELS;++level) {
    unsigned octant=0;
    for (int a=0;a<3;++a) {
      width[a]*=0.5;
      if (in[i].pm.x[a]>lower[a]+width[a]) {
        lower[a]+=width[a];
        octant|=4>>a;
      }
    }
    key|=Key(octant)<<(3*(PROBE_LEVELS-level));
  }
  keys[i]=key; order[i]=i;
}

__device__ int lower_bound(const Key* keys,int a,int b,Key key) {
  while(a<b) {
    int c=a+(b-a)/2;
    if (keys[c]<key) a=c+1;
    else b=c;
  }
  return a;
}

__global__ void roots(const Key* keys,int n,Geometry* nodes,int nr) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=nr) return;
  auto& node=nodes[i];
  node.key=Key(i)<<octree_shift;
  node.start=lower_bound(keys,0,n,node.key);
  node.end=lower_bound(keys,node.start,n,Key(i+1)<<octree_shift);
  for(int& child:node.children) child=-1;
}

__global__ void split(const Key* keys,Geometry* nodes,int begin,int end,
                       int level,int limit,int* count,int capacity,int* invalid) {
  const int i=begin+blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=end) return;
  auto& parent=nodes[i];
  if (parent.end-parent.start<=limit) return;
  if (level>=PROBE_LEVELS) { atomicAdd(invalid,1); return; }
  const Key step=Key(1)<<(3*(PROBE_LEVELS-level-1));
  int start=parent.start;
  for(int k=0;k<8;++k) {
    const Key key=parent.key+step*k;
    const int stop=lower_bound(keys,start,parent.end,key+step);
    if(stop>start) {
      const int slot=atomicAdd(count,1);
      if (slot>=capacity) { atomicAdd(invalid,1); return; }
      parent.children[k]=slot;
      auto& node=nodes[slot];
      node.key=key; node.start=start; node.end=stop;
      for(int& child:node.children) child=-1;
    }
    start=stop;
  }
}

struct Exponent { int x,y,z; };
#define EXP(i,x,y,z) {x,y,z},
__device__ __constant__ Exponent exponent[PROBE_TERMS]={PROBE_TERM_LIST(EXP)};
constexpr Exponent host_exponent[PROBE_TERMS]={PROBE_TERM_LIST(EXP)};
#undef EXP

__device__ double powers(double x,int p) {
  const double x2=x*x;
  switch(p) {
    case 0:return 1.;
    case 1:return x;
    case 2:return x2/2.;
    case 3:return x2*x/6.;
    default:return x2*x2/24.;
  }
}
__device__ double product(const double r[3],int a,int b,int c) {
  return powers(r[0],a)*powers(r[1],b)*powers(r[2],c);
}
__device__ int factorial(int p) { return p==4?24:p==3?6:p==2?2:1; }

__device__ void power(probe_node& out) {
  float sum[5]={};
  for(int i=1;i<PROBE_TERMS;++i) {
    const auto a=exponent[i];
    const int d=a.x+a.y+a.z;
    const double weight=double(factorial(a.x)*factorial(a.y)*factorial(a.z))/factorial(d);
    sum[d]+=weight*out.moments[i]*out.moments[i];
  }
  out.power[0]=out.moments[0];
  for(int n=2;n<5;++n) out.power[n]=sqrtf(sum[n]);
}

__device__ double cell_radius(const probe_node& out,probe_header h) {
  int top=out.key>>octree_shift;
  double lower[3]={double(top/h.top[2]/h.top[1])*h.width[0],
                  double(top/h.top[2]%h.top[1])*h.width[1],double(top%h.top[2])*h.width[2]};
  double width[3]={h.width[0],h.width[1],h.width[2]};
  for(int level=1;level<=out.level;++level) {
    int child=(out.key>>(3*(PROBE_LEVELS-level)))&7;
    for(int a=0;a<3;++a) {
      width[a]*=0.5;
      if(child&(4>>a)) lower[a]+=width[a];
    }
  }
  double square=0;
  for(int a=0;a<3;++a) {
    double d=fmax(out.center[a]-lower[a],lower[a]+width[a]-out.center[a]);
    square+=d*d;
  }
  return sqrt(square);
}

__global__ void multipoles(const Geometry* geometry,probe_node* moments,
    const probe_particle* in,const int* sorted,int begin,int end,int level,probe_header h) {
  const int i=begin+blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=end) return;
  const auto& geom=geometry[i];
  probe_node out{};
  out.key=geom.key;out.level=level;out.count=geom.end-geom.start;
  if(!out.count) {moments[i]=out;return;}
  out.old_acceleration=FLT_MAX;
  bool leaf=true;
  for (int child:geom.children) leaf &= child<0;
  double mass=0;
  if(leaf) {
    for(int j=geom.start;j<geom.end;++j) {
      const auto& p=in[sorted[j]];
      mass+=p.pm.mass;
      for(int a=0;a<3;++a) out.center[a]+=p.pm.mass*p.pm.x[a];
      out.softening=fmaxf(out.softening,p.softening);
      out.old_acceleration=fminf(out.old_acceleration,p.old_acceleration);
    }
    for(double& center:out.center) center/=mass;
    double sums[PROBE_TERMS]={};
    double radius2=0;
    for(int j=geom.start;j<geom.end;++j) {
      const auto& p=in[sorted[j]];
      const double dx[3]={out.center[0]-p.pm.x[0],out.center[1]-p.pm.x[1],out.center[2]-p.pm.x[2]};
      radius2=fmax(radius2,dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
      for(int k=1;k<PROBE_TERMS;++k) {
        const auto term=exponent[k];
        sums[k]+=p.pm.mass*product(dx,term.x,term.y,term.z);
      }
    }
    out.radius=sqrt(radius2);
    for(int k=1;k<PROBE_TERMS;++k) out.moments[k]=sums[k];
  } else {
    for(int child:geom.children) {
      if(child<0) continue;
      const auto& m=moments[child];
      mass+=m.moments[0];
      for(int a=0;a<3;++a) out.center[a]+=m.moments[0]*m.center[a];
    }
    for(double& center:out.center) center/=mass;
    double radius=0;
    for(int child:geom.children) {
      if(child<0) continue;
      const auto& m=moments[child];
      const double dx[3]={out.center[0]-m.center[0],out.center[1]-m.center[1],out.center[2]-m.center[2]};
      radius=fmax(radius,m.radius+sqrt(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]));
      out.softening=fmaxf(out.softening,m.softening);
      out.old_acceleration=fminf(out.old_acceleration,m.old_acceleration);
      for(int k=1;k<PROBE_TERMS;++k) {
        const auto a=exponent[k];
        double translated=0;
        for(int j=0;j<=k;++j) {
          const auto b=exponent[j];
          if(b.x<=a.x && b.y<=a.y && b.z<=a.z)
            translated+=m.moments[j]*product(dx,a.x-b.x,a.y-b.y,a.z-b.z);
        }
        out.moments[k]+=float(translated);
      }
    }
    out.radius=fmin(radius,cell_radius(out,h));
    // Native M2M accumulates each child already rounded to float.
    float total=0;
    for (int child:geom.children) if (child>=0) total+=moments[child].moments[0];
    mass=total;
  }
  out.moments[0]=mass;
  power(out);
  moments[i]=out;
}

__global__ void pm_input(const probe_particle* in, grav_pm_particle* out,int n) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) out[i]=in[i].pm;
}

__global__ void pm_audit(const grav_pm_force* forces,int n,int* invalid) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n && (!isfinite(forces[i].potential) ||
      !isfinite(forces[i].acceleration[0]) || !isfinite(forces[i].acceleration[1]) ||
      !isfinite(forces[i].acceleration[2]))) atomicAdd(invalid,1);
}

constexpr int blocks(int n) {return (n+127)/128;}
int main(int argc,char** argv) {
  if(argc!=2) return 2;
  FILE* f=fopen(argv[1],"rb");
  probe_header h{};
  if (!f || fread(&h,sizeof(h),1,f)!=1 || h.particles!=1<<24 ||
      !h.nodes || h.nodes>2000000) return 3;
  const int n=h.particles,roots_count=h.top[0]*h.top[1]*h.top[2],capacity=2*h.nodes+roots_count;
  if (roots_count>=1<<(64-octree_shift)) return 4;
  std::vector<probe_particle> input(n);
  std::vector<probe_node> expected(h.nodes);
  if(fread(input.data(),sizeof(probe_particle),n,f)!=size_t(n) ||
      fread(expected.data(),sizeof(probe_node),h.nodes,f)!=h.nodes) return 5;
  fclose(f);

  Device<probe_particle> particles(n);
  Device<Key> keys(n);
  Device<int> order(n), cursor(1), invalid(1);
  Device<Geometry> geometry(capacity);
  Device<probe_node> moments(capacity);
  Device<grav_pm_particle> mesh_particles(n);
  Device<grav_pm_force> mesh_forces(n);
  check(cudaMemcpy(particles.p,input.data(),n*sizeof(probe_particle),cudaMemcpyHostToDevice));
  check(cudaMemset(invalid.p,0,sizeof(int)));
  const float sort_ms=ms([&]{
    key_particles<<<blocks(n),128>>>(particles.p,n,h,keys.p,order.p,invalid.p);
    thrust::sort_by_key(thrust::device_pointer_cast(keys.p),
        thrust::device_pointer_cast(keys.p+n),thrust::device_pointer_cast(order.p));
  });
  check(cudaMemcpy(cursor.p,&roots_count,sizeof(int),cudaMemcpyHostToDevice));
  roots<<<blocks(roots_count),128>>>(keys.p,n,geometry.p,roots_count);
  std::vector<int> levels{0,roots_count};
  int begin=0,end=roots_count;
  float topology_ms=0;
  for(int level=0;;++level) {
    topology_ms+=ms([&]{split<<<blocks(end-begin),128>>>(
        keys.p,geometry.p,begin,end,level,h.leaf_limit,cursor.p,capacity,invalid.p);});
    int next=0,bad=0;
    check(cudaMemcpy(&next,cursor.p,sizeof(int),cudaMemcpyDeviceToHost));
    check(cudaMemcpy(&bad,invalid.p,sizeof(int),cudaMemcpyDeviceToHost));
    if(bad || next>capacity) throw std::runtime_error("Invalid GPU tree topology");
    if(next==end) break;
    begin=end;end=next;levels.push_back(end);
  }
  const float multipole_ms=ms([&]{
    for(int l=int(levels.size())-2;l>=0;--l)
      multipoles<<<blocks(levels[l+1]-levels[l]),128>>>(
          geometry.p,moments.p,particles.p,order.p,levels[l],levels[l+1],l,h);
  });

  // Keep the particle state on the device between the tree and PM.
  const float mesh_view_ms=ms([&]{pm_input<<<blocks(n),128>>>(particles.p,mesh_particles.p,n);});
  grav_pm_timing pm{};
  grav_pm_cuda_device(mesh_particles.p,n,h.mesh,h.box,h.smoothing,h.grav_G,mesh_forces.p,&pm);
  pm_audit<<<blocks(n),128>>>(mesh_forces.p,n,invalid.p);
  int pm_invalid=0;
  check(cudaMemcpy(&pm_invalid,invalid.p,sizeof(int),cudaMemcpyDeviceToHost));

  std::vector<probe_node> actual(end);
  check(cudaMemcpy(actual.data(),moments.p,end*sizeof(probe_node),cudaMemcpyDeviceToHost));
  actual.erase(std::remove_if(actual.begin(),actual.end(),[](const auto& p){return p.count==0;}),actual.end());
  const auto ordered=[](const auto& x,const auto& y){return std::tie(x.key,x.level)<std::tie(y.key,y.level);};
  std::sort(actual.begin(),actual.end(),ordered);
  std::sort(expected.begin(),expected.end(),ordered);
  std::cerr<<"native_nodes "<<expected.size()<<" device_nodes "<<actual.size()<<" max_depth "<<levels.size()-2<<"\n";
  if(actual.size()!=expected.size()) return 6;
  double center_max=0,radius_absolute_max=0,radius_scaled_max=0,scaled_moment_max=0,relative_mass_max=0,scaled_power_max=0;
  size_t mismatched_cells=0,mismatched_soft=0,mismatched_acc=0;
  for(size_t i=0;i<actual.size();++i) {
    const auto& a=actual[i];
    const auto& b=expected[i];
    if(a.key!=b.key || a.level!=b.level || a.count!=b.count) {++mismatched_cells;continue;}
    double center=0;
    for(int k=0;k<3;++k) center=std::max(center,std::abs(a.center[k]-b.center[k]));
    center_max=std::max(center_max,center);
    relative_mass_max=std::max(relative_mass_max,std::abs(double(a.moments[0])-b.moments[0])/b.moments[0]);
    const double radius_delta=std::abs(double(a.radius)-b.radius);
    const double length=std::ldexp(h.width[0],-a.level);
    radius_absolute_max=std::max(radius_absolute_max,radius_delta);
    radius_scaled_max=std::max(radius_scaled_max,radius_delta/length);
    for(int d=2;d<5;++d) {
      const double scale=b.moments[0]*std::pow(length,d);
      scaled_power_max=std::max(scaled_power_max,std::abs(double(a.power[d])-b.power[d])/scale);
    }
    for(int m=1;m<PROBE_TERMS;++m) {
      const auto d=host_exponent[m];
      const double scale=b.moments[0]*std::pow(length,d.x+d.y+d.z);
      scaled_moment_max=std::max(scaled_moment_max,std::abs(double(a.moments[m])-b.moments[m])/scale);
    }
    mismatched_soft+=a.softening!=b.softening;
    mismatched_acc+=a.old_acceleration!=b.old_acceleration;
  }
  std::cout<<"{\"particles\":"<<n<<",\"nodes\":"<<actual.size()
      <<",\"max_tree_depth\":"<<levels.size()-2<<",\"topology_mismatches\":"<<mismatched_cells
      <<",\"softening_mismatches\":"<<mismatched_soft<<",\"mac_min_acceleration_mismatches\":"<<mismatched_acc
      <<",\"max_center_absolute_mpc\":"<<center_max<<",\"max_mass_relative\":"<<relative_mass_max
      <<",\"max_radius_absolute_mpc\":"<<radius_absolute_max
      <<",\"max_width_scaled_radius_error\":"<<radius_scaled_max
      <<",\"max_width_scaled_multipole_error\":"<<scaled_moment_max
      <<",\"max_width_scaled_multipole_power_error\":"<<scaled_power_max
      <<",\"key_and_sort_ms\":"<<sort_ms<<",\"split_kernel_ms\":"<<topology_ms<<",\"multipole_ms\":"<<multipole_ms
      <<",\"pm_device_input_view_ms\":"<<mesh_view_ms<<",\"pm_deposit_ms\":"<<pm.deposit_ms
      <<",\"pm_fft_ms\":"<<pm.fft_ms<<",\"pm_interpolate_ms\":"<<pm.interpolate_ms
      <<",\"pm_invalid_particles\":"<<pm_invalid<<"}\n";
  return mismatched_cells || mismatched_soft || mismatched_acc || pm_invalid ||
         relative_mass_max>2e-6 || scaled_moment_max>1e-6 || scaled_power_max>1e-6 || radius_scaled_max>1e-6 || center_max>1e-5 ? 7 : 0;
}
