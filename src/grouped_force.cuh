// Device work is organized in Morton-ordered tiles. Each tile shares a far-field
// expansion and a list of unresolved source leaves. A tile that cannot usefully
// share those data takes the ordinary per-particle path.
constexpr int force_tile=32;
constexpr int max_near_leaves=192;
constexpr float dense_force_fraction=1.f/16.f;

struct ForceTile {
  float3 center;
  int near_count;
  grav_tensor field;
};

__device__ bool dense_force_step(const int* active_count,int n) {
  return !active_count || *active_count>=int(n*dense_force_fraction);
}

__device__ __forceinline__ float4 evaluate_tile(const grav_tensor& f,float3 dx) {
  const float x=dx.x,y=dx.y,z=dx.z;
  const float x2=.5f*x*x,y2=.5f*y*y,z2=.5f*z*z;
  const float x3=x2*x/3.f,y3=y2*y/3.f,z3=z2*z/3.f;
  const float x4=x3*x*.25f,y4=y3*y*.25f,z4=z3*z*.25f;
  const float xy=x*y,xz=x*z,yz=y*z;
  const float a0=f.F_100;
  const float a1=f.F_010;
  const float a2=f.F_001;
  const float4 field{
      a0+f.F_200*x+f.F_110*y+f.F_101*z
        +f.F_300*x2+f.F_120*y2+f.F_102*z2+f.F_210*xy+f.F_201*xz+f.F_111*yz
        +f.F_400*x3+f.F_130*y3+f.F_103*z3+f.F_310*x2*y+f.F_301*x2*z
        +f.F_220*x*y2+f.F_202*x*z2+f.F_121*y2*z+f.F_112*y*z2+f.F_211*xy*z,
      a1+f.F_110*x+f.F_020*y+f.F_011*z
        +f.F_210*x2+f.F_030*y2+f.F_012*z2+f.F_120*xy+f.F_111*xz+f.F_021*yz
        +f.F_310*x3+f.F_040*y3+f.F_013*z3+f.F_220*x2*y+f.F_211*x2*z
        +f.F_130*x*y2+f.F_112*x*z2+f.F_031*y2*z+f.F_022*y*z2+f.F_121*xy*z,
      a2+f.F_101*x+f.F_011*y+f.F_002*z
        +f.F_201*x2+f.F_021*y2+f.F_003*z2+f.F_111*xy+f.F_102*xz+f.F_012*yz
        +f.F_301*x3+f.F_031*y3+f.F_004*z3+f.F_211*x2*y+f.F_202*x2*z
        +f.F_121*x*y2+f.F_103*x*z2+f.F_022*y2*z+f.F_013*y*z2+f.F_112*xy*z,
      -(f.F_000+a0*x+a1*y+a2*z
        +f.F_200*x2+f.F_020*y2+f.F_002*z2+f.F_110*xy+f.F_101*xz+f.F_011*yz
        +f.F_300*x3+f.F_030*y3+f.F_003*z3+f.F_210*x2*y+f.F_201*x2*z
        +f.F_120*x*y2+f.F_102*x*z2+f.F_021*y2*z+f.F_012*y*z2+f.F_111*xy*z
        +f.F_400*x4+f.F_040*y4+f.F_004*z4+f.F_310*x3*y+f.F_301*x3*z
        +f.F_130*x*y3+f.F_103*x*z3+f.F_031*y3*z+f.F_013*y*z3
        +f.F_220*x2*y2+f.F_202*x2*z2+f.F_022*y2*z2+f.F_211*x2*yz
        +f.F_121*y2*xz+f.F_112*z2*xy)
  };
  return field;
}

__global__ void build_force_tiles(const probe_particle* particles,const Key* keys,
    const int* order,const float4* packed,const ForceNode* source_nodes,const int* escape,
    const int* first_child,int n,Physics p,bool initial,const int* invalid,
    const int* active_sorted,const int* active_count,ForceTile* tiles,int* near_leaves) {
  const int group=blockIdx.x*blockDim.x+threadIdx.x;
  if(*invalid || !dense_force_step(active_count,n)) return;
  const int targets=active_count?*active_count:n;
  const int begin=group*force_tile;
  if(begin>=targets) return;
  const int end=min(begin+force_tile,targets);
  const int first=active_sorted?active_sorted[begin]:begin;
  const int last=active_sorted?active_sorted[end-1]:end-1;
  const int root=keys[first]>>octree_shift;
  auto& tile=tiles[group];
  tile.near_count=-1;
  if(end-begin<8 || root!=int(keys[last]>>octree_shift)) return;

  const auto f=packed[first];
  float3 lower{f.x,f.y,f.z},upper=lower;
  float old=FLT_MAX;
  for(int at=begin;at<end;++at) {
    const int i=active_sorted?active_sorted[at]:at;
    const auto a=packed[i];
    lower={fminf(lower.x,a.x),fminf(lower.y,a.y),fminf(lower.z,a.z)};
    upper={fmaxf(upper.x,a.x),fmaxf(upper.y,a.y),fmaxf(upper.z,a.z)};
    old=fminf(old,particles[order[i]].old_acceleration);
  }
  const float3 width{upper.x-lower.x,upper.y-lower.y,upper.z-lower.z};
  const float radius=.5f*sqrtf(norm2(width))+1e-5f;
  if(radius>1.5f*p.split) return;
  const float3 center{lower.x+.5f*width.x,lower.y+.5f*width.y,lower.z+.5f*width.z};
  const float inv_split=1/p.split;
  const int home[3]={root/144,root/12%12,root%12};
  int count=0,accepted=0;
  grav_tensor field{};
  int* leaves=near_leaves+size_t(group)*max_near_leaves;
  for(int dx=-1;dx<=1;++dx) for(int dy=-1;dy<=1;++dy) for(int dz=-1;dz<=1;++dz) {
    int node=(wrap_cell(home[0]+dx)*12+wrap_cell(home[1]+dy))*12+wrap_cell(home[2]+dz);
    while(node>=0) {
      const auto& source=source_nodes[node];
      const auto c=source.center;
      const float3 d=periodic_delta(center,{c.x,c.y,c.z},p.box);
      const float r2=norm2(d),reach=c.w+p.cutoff+radius;
      if(!source.n || r2>reach*reach) { node=escape[node]; continue; }
      const float extent=radius+c.w;
      // Constrain convergence and keep a guard band around softening and the PM split.
      if(source.n>1 && extent*extent<.4225f*r2) {
        const float distance=sqrtf(r2);
        const float closest=distance-radius;
        const float closest2=closest*closest;
        const float soft=source.m.max_softening;
        if(closest>soft && distance+radius<=c.w+p.cutoff &&
           (initial || 8.f*source.m.power[2]<p.mac_tolerance*old*closest2*closest2)) {
          potential_derivatives_M2L derivatives;
          grav_cuda_potential_derivatives_compute_M2L(d.x,d.y,d.z,r2,1/distance,soft,1,
                                                       inv_split,&derivatives);
          grav_cuda_gravity_M2L_apply(&field,&source.m,&derivatives);
          ++accepted;
          node=escape[node];
          continue;
        }
      }
      if(first_child[node]>=0) { node=first_child[node]; continue; }
      if(count==max_near_leaves) return;
      leaves[count++]=node;
      node=escape[node];
    }
  }
  if(!accepted) return;
  tile.center=center;
  tile.near_count=count;
  tile.field=field;
}
