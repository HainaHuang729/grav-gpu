from pathlib import Path
import shutil
import sys

here=Path(__file__).parent
root=Path(sys.argv[2]) if len(sys.argv)>2 else here.parent.parent.parent
build=Path(sys.argv[1])
tree=(root/"tree/tree_probe.cu").read_text()
# Retain the validated tree implementation, with GPU level boundaries.
tree=tree[:tree.index("int main(int argc,char** argv)")]
anchor="int level,int limit,int* count,int capacity,int* invalid) {\n  const int i=begin"
assert tree.count(anchor)==1
tree=tree.replace(anchor,"""int level,int limit,int* count,int capacity,int* invalid,
                       const int* levels=nullptr) {
  if (levels) { begin=levels[level]; end=levels[level+1]; }
  const int i=begin""")
anchor="int begin,int end,int level,probe_header h) {\n  const int i=begin"
assert tree.count(anchor)==1
tree=tree.replace(anchor,"""int begin,int end,int level,probe_header h,
    const int* levels=nullptr) {
  if (levels) { begin=levels[level]; end=levels[level+1]; }
  const int i=begin""")
# Refit nodes in their original root's periodic image. The measured child
# bounds replace the geometric cell cap because particles can leave old leaves.
anchor="__global__ void multipoles("
assert tree.count(anchor)==1
tree=tree.replace(anchor,"""__device__ double refit_coordinate(double x,double center,double box) {
  const float delta=float(x)-float(center);
  const float half=0.5f*float(box);
  if(delta>half) return x-box;
  if(delta<-half) return x+box;
  return x;
}

template<bool refit=false>
__global__ void multipoles(""",1)
anchor="probe_header h,\n    const int* levels=nullptr) {\n  if (levels) { begin=levels[level]; end=levels[level+1]; }\n  const int i=begin+"
assert tree.count(anchor)==1
tree=tree.replace(anchor,"probe_header h,\n    const int* levels=nullptr,int* invalid=nullptr,double max_root_offset=0) {\n  if (levels) { begin=levels[level]; end=levels[level+1]; }\n  const int i=begin+",1)
anchor="  if(leaf) {\n    for(int j=geom.start;"
assert tree.count(anchor)==1
tree=tree.replace(anchor,"""  if(leaf) {
    const int root=out.key>>octree_shift;
    const double root_center[3]={(root/h.top[2]/h.top[1]+0.5)*h.width[0],
        (root/h.top[2]%h.top[1]+0.5)*h.width[1],(root%h.top[2]+0.5)*h.width[2]};
    const double box=h.top[0]*h.width[0];
    bool left_root_buffer=false;
    for(int j=geom.start;""",1)
anchor="      for(int a=0;a<3;++a) out.center[a]+=p.pm.mass*p.pm.x[a];"
assert tree.count(anchor)==1
tree=tree.replace(anchor,"""      for(int a=0;a<3;++a) {
        double x=p.pm.x[a];
        if constexpr(refit) {
          x=refit_coordinate(x,root_center[a],box);
          left_root_buffer|=fabsf(float(x)-float(root_center[a]))>float(max_root_offset);
        }
        out.center[a]+=p.pm.mass*x;
      }""",1)
anchor="    for(double& center:out.center) center/=mass;"
assert tree.count(anchor)==2
tree=tree.replace(anchor,"""    if constexpr(refit) {
      if(left_root_buffer) atomicOr(invalid,16);
    }
"""+anchor,1)
anchor="      const double dx[3]={out.center[0]-p.pm.x[0],out.center[1]-p.pm.x[1],out.center[2]-p.pm.x[2]};"
assert tree.count(anchor)==1
tree=tree.replace(anchor,"""      double dx[3];
      for(int a=0;a<3;++a) {
        double x=p.pm.x[a];
        if constexpr(refit) x=refit_coordinate(x,root_center[a],box);
        dx[a]=out.center[a]-x;
      }""",1)
anchor="out.radius=fmin(radius,cell_radius(out,h));"
assert tree.count(anchor)==1
tree=tree.replace(anchor,"out.radius=refit?radius:fmin(radius,cell_radius(out,h));",1)
(build/"src/validated_device_tree.cuh").write_text(tree)
shutil.copyfile(root/"tree/tree_format.h",build/"src/tree_format.h")

# Specialize the short polynomial contractions for the configured quartic tree.
# The translation terms and their accumulation order are the same as above.
import re
terms=[(int(i),int(x),int(y),int(z)) for i,x,y,z in re.findall(
    r"X\((\d+),(\d+),(\d+),(\d+)\)", (root/"tree/tree_format.h").read_text())]
assert len(terms)==32 and terms[0]==(0,0,0,0)
kernel=tree[tree.index("template<bool refit=false>"):tree.index("__global__ void pm_input(")]
kernel=kernel.replace("void multipoles(", "void multipoles_specialized(",1)
monomial=lambda x,y,z:f"px[{x}]*py[{y}]*pz[{z}]"
axes="""      double px[5],py[5],pz[5];
#pragma unroll
      for(int k=0;k<5;++k) {
        px[k]=powers(dx[0],k);
        py[k]=powers(dx[1],k);
        pz[k]=powers(dx[2],k);
      }
"""
leaf_pattern="""      for(int k=1;k<PROBE_TERMS;++k) {
        const auto term=exponent[k];
        sums[k]+=p.pm.mass*product(dx,term.x,term.y,term.z);
      }"""
leaf=axes+"\n".join(f"      sums[{i}]+=p.pm.mass*({monomial(x,y,z)});" for i,x,y,z in terms[1:])
assert kernel.count(leaf_pattern)==1
kernel=kernel.replace(leaf_pattern,leaf,1)
parent_pattern="""      for(int k=1;k<PROBE_TERMS;++k) {
        const auto a=exponent[k];
        double translated=0;
        for(int j=0;j<=k;++j) {
          const auto b=exponent[j];
          if(b.x<=a.x && b.y<=a.y && b.z<=a.z)
            translated+=m.moments[j]*product(dx,a.x-b.x,a.y-b.y,a.z-b.z);
        }
        out.moments[k]+=float(translated);
      }"""
parent=[axes,"      double translated;"]
for i,x,y,z in terms[1:]:
    parent.append("      translated=0.;")
    for j,u,v,w in terms[:i+1]:
        if u<=x and v<=y and w<=z:
            parent.append(f"      translated+=m.moments[{j}]*({monomial(x-u,y-v,z-w)});")
    parent.append(f"      out.moments[{i}]+=float(translated);")
assert kernel.count(parent_pattern)==1
kernel=kernel.replace(parent_pattern,"\n".join(parent),1)
fast_keys="""
__device__ __forceinline__ Key expand_bits_21(uint32_t v) {
  Key x = Key(v & 0x1fffffU);
  x = (x | (x << 32)) & 0x001f00000000ffffULL;
  x = (x | (x << 16)) & 0x001f0000ff0000ffULL;
  x = (x | (x << 8))  & 0x100f00f00f00f00fULL;
  x = (x | (x << 4))  & 0x10c30c30c30c30c3ULL;
  x = (x | (x << 2))  & 0x1249249249249249ULL;
  return x;
}

__global__ void key_particles_fast(const probe_particle* in, int n, probe_header h,
                                   Key* keys, int* order, int* invalid) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n) return;
  int top[3];
  uint32_t ix[3];
  constexpr double scale=double(1u<<PROBE_LEVELS);
  for (int a=0;a<3;++a) {
    const double scaled=in[i].pm.x[a]*h.inv_width[a];
    int t=int(scaled);
    if (t<0 || t>=h.top[a]) {
      atomicAdd(invalid,1);
      t=0;
    }
    top[a]=t;
    const double q=(scaled-double(t))*scale;
    ix[a]=q<=0.0?0u:(q>=scale?((1u<<PROBE_LEVELS)-1u):uint32_t(q));
  }
  Key key=Key((top[0]*h.top[1]+top[1])*h.top[2]+top[2])<<octree_shift;
  key|=(expand_bits_21(ix[0])<<2)|(expand_bits_21(ix[1])<<1)|expand_bits_21(ix[2]);
  keys[i]=key; order[i]=i;
}

__global__ void key_particles_fast_slab(const double3* x, int n, int slab_x,
                                        probe_header h, Key* keys, int* order) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n) return;
  const double3 pos=x[i];
  constexpr double scale=double(1u<<PROBE_LEVELS);
  constexpr uint32_t max_q=(1u<<PROBE_LEVELS)-1u;
  auto quantize=[&](double q) -> uint32_t {
    return q<=0.0?0u:(q>=scale?max_q:uint32_t(q));
  };
  const uint32_t ix=quantize((pos.x*h.inv_width[0]-double(slab_x))*scale);
  const double sy=pos.y*h.inv_width[1];
  const int ty=min(max(int(sy),0),h.top[1]-1);
  const uint32_t iy=quantize((sy-double(ty))*scale);
  const double sz=pos.z*h.inv_width[2];
  const int tz=min(max(int(sz),0),h.top[2]-1);
  const uint32_t iz=quantize((sz-double(tz))*scale);
  const int local_top=ty*h.top[2]+tz;
  Key key=Key(local_top)<<octree_shift;
  key|=(expand_bits_21(ix)<<2)|(expand_bits_21(iy)<<1)|expand_bits_21(iz);
  keys[i]=key; order[i]=i;
}

__global__ void key_particles_fast_3d(const double3* x, int n,
                                      const int* cell_to_slot, probe_header h,
                                      Key* keys, int* order) {
  const int i=blockIdx.x*blockDim.x+threadIdx.x;
  if (i>=n) return;
  const double3 pos=x[i];
  constexpr double scale=double(1u<<PROBE_LEVELS);
  constexpr uint32_t max_q=(1u<<PROBE_LEVELS)-1u;
  auto quantize=[&](double q) -> uint32_t {
    return q<=0.0?0u:(q>=scale?max_q:uint32_t(q));
  };
  const double sx=pos.x*h.inv_width[0];
  const int tx=min(max(int(sx),0),h.top[0]-1);
  const uint32_t ix=quantize((sx-double(tx))*scale);
  const double sy=pos.y*h.inv_width[1];
  const int ty=min(max(int(sy),0),h.top[1]-1);
  const uint32_t iy=quantize((sy-double(ty))*scale);
  const double sz=pos.z*h.inv_width[2];
  const int tz=min(max(int(sz),0),h.top[2]-1);
  const uint32_t iz=quantize((sz-double(tz))*scale);
  const int cid=(tx*h.top[1]+ty)*h.top[2]+tz;
  const int local_top=max(cell_to_slot[cid],0);
  Key key=Key(local_top)<<octree_shift;
  key|=(expand_bits_21(ix)<<2)|(expand_bits_21(iy)<<1)|expand_bits_21(iz);
  keys[i]=key; order[i]=i;
}
"""
with (build/"src/validated_device_tree.cuh").open("a") as output:
    output.write("\n"+kernel+"\n"+fast_keys)

# Generate pure-FP32 cell-origin-shifted multipoles_fast.cuh writing directly to ForceNode.
import math
mname=lambda x,y,z: f"M_{x}{y}{z}"
fmono=lambda x,y,z: f"px[{x}]*py[{y}]*pz[{z}]"
leaf_acc="\n".join(f"      moments[{i}]+=mp*({fmono(x,y,z)});" for i,x,y,z in terms[1:])
m2m_acc=[]
for i,x,y,z in terms[1:]:
    m2m_acc.append("      translated=0.f;")
    for j,u,v,w in terms[:i+1]:
        if u<=x and v<=y and w<=z:
            m2m_acc.append(f"      translated+=cf.m.{mname(u,v,w)}*({fmono(x-u,y-v,z-w)});")
    m2m_acc.append(f"      moments[{i}]+=translated;")
m2m_str="\n".join(m2m_acc)
assign_str="\n".join(f"  fn.m.{mname(x,y,z)}=moments[{i}];" for i,x,y,z in terms)
p2_terms=[]
for i,x,y,z in terms[1:]:
    if x+y+z==2:
        w=float(math.factorial(x)*math.factorial(y)*math.factorial(z))/2.0
        p2_terms.append(f"({w}f)*(moments[{i}]*moments[{i}])")
p2_expr=" + ".join(p2_terms)

fast_mpole=f"""#pragma once
__device__ __forceinline__ float powers_f(float x,int p) {{
  const float x2=x*x;
  switch(p) {{
    case 0: return 1.f;
    case 1: return x;
    case 2: return 0.5f*x2;
    case 3: return (1.f/6.f)*x2*x;
    default: return (1.f/24.f)*x2*x2;
  }}
}}

template<bool track_vel=false>
__global__ void multipoles_fast(const Geometry* geometry,ForceNode* force_nodes,
    double3* node_com,const probe_particle* in,const float3* velocity,
    const int* sorted,int level,probe_header h,const int* levels) {{
  const int begin=levels[level];
  const int end=levels[level+1];
  const int i=begin+blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=end) return;
  const auto& geom=geometry[i];
  const int count=geom.end-geom.start;
  ForceNode& fn=force_nodes[i];
  fn.n=count;
  if(!count) {{
    fn.center={{}};
    fn.m={{}};
    node_com[i]={{}};
    return;
  }}
  const int top=geom.key>>octree_shift;
  double lower[3]={{
      double(top/h.top[2]/h.top[1])*h.width[0],
      double(top/h.top[2]%h.top[1])*h.width[1],
      double(top%h.top[2])*h.width[2]}};
  double width[3]={{h.width[0],h.width[1],h.width[2]}};
  for(int l=1;l<=level;++l) {{
    const int child=(geom.key>>(3*(PROBE_LEVELS-l)))&7;
    for(int a=0;a<3;++a) {{
      width[a]*=0.5;
      if(child&(4>>a)) lower[a]+=width[a];
    }}
  }}
  bool leaf=true;
  for(int child:geom.children) leaf&=child<0;

  float moments[PROBE_TERMS]={{}};
  double com[3]={{}};
  float radius=0.f,soft=0.f,mass=0.f;
  float vel_com[3]={{}},max_dv=0.f;

  if(leaf) {{
    float rel_cx=0.f,rel_cy=0.f,rel_cz=0.f;
    float vx_sum=0.f,vy_sum=0.f,vz_sum=0.f;
    for(int j=geom.start;j<geom.end;++j) {{
      const int pid=sorted[j];
      const auto& p=in[pid];
      const float mp=p.pm.mass;
      mass+=mp;
      rel_cx+=mp*float(p.pm.x[0]-lower[0]);
      rel_cy+=mp*float(p.pm.x[1]-lower[1]);
      rel_cz+=mp*float(p.pm.x[2]-lower[2]);
      soft=fmaxf(soft,p.softening);
      if constexpr(track_vel) {{
        const float3 vp=velocity[pid];
        vx_sum+=mp*vp.x; vy_sum+=mp*vp.y; vz_sum+=mp*vp.z;
      }}
    }}
    const float inv_m=1.f/mass;
    const float com_rel[3]={{rel_cx*inv_m,rel_cy*inv_m,rel_cz*inv_m}};
    com[0]=lower[0]+double(com_rel[0]);
    com[1]=lower[1]+double(com_rel[1]);
    com[2]=lower[2]+double(com_rel[2]);
    if constexpr(track_vel) {{
      vel_com[0]=vx_sum*inv_m; vel_com[1]=vy_sum*inv_m; vel_com[2]=vz_sum*inv_m;
    }}
    float radius2=0.f,max_dv2=0.f;
    for(int j=geom.start;j<geom.end;++j) {{
      const int pid=sorted[j];
      const auto& p=in[pid];
      const float mp=p.pm.mass;
      const float dx[3]={{
          com_rel[0]-float(p.pm.x[0]-lower[0]),
          com_rel[1]-float(p.pm.x[1]-lower[1]),
          com_rel[2]-float(p.pm.x[2]-lower[2])}};
      radius2=fmaxf(radius2,dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
      float px[5],py[5],pz[5];
#pragma unroll
      for(int k=0;k<5;++k) {{
        px[k]=powers_f(dx[0],k);
        py[k]=powers_f(dx[1],k);
        pz[k]=powers_f(dx[2],k);
      }}
{leaf_acc}
      if constexpr(track_vel) {{
        const float3 vp=velocity[pid];
        const float dvx=vp.x-vel_com[0],dvy=vp.y-vel_com[1],dvz=vp.z-vel_com[2];
        max_dv2=fmaxf(max_dv2,dvx*dvx+dvy*dvy+dvz*dvz);
      }}
    }}
    radius=sqrtf(radius2);
    if constexpr(track_vel) max_dv=sqrtf(max_dv2);
  }} else {{
    double dm=0.0;
    for(int child:geom.children) {{
      if(child<0) continue;
      const float cm=force_nodes[child].m.M_000;
      const double3 ccom=node_com[child];
      dm+=double(cm);
      com[0]+=double(cm)*ccom.x;
      com[1]+=double(cm)*ccom.y;
      com[2]+=double(cm)*ccom.z;
      if constexpr(track_vel) {{
        const auto& cm_m=force_nodes[child].m;
        vel_com[0]+=cm*cm_m.vel[0];
        vel_com[1]+=cm*cm_m.vel[1];
        vel_com[2]+=cm*cm_m.vel[2];
      }}
    }}
    const double inv_dm=1.0/dm;
    com[0]*=inv_dm; com[1]*=inv_dm; com[2]*=inv_dm;
    if constexpr(track_vel) {{
      const float inv_fm=float(inv_dm);
      vel_com[0]*=inv_fm; vel_com[1]*=inv_fm; vel_com[2]*=inv_fm;
    }}
    float total_m=0.f;
    for(int child:geom.children) {{
      if(child<0) continue;
      const ForceNode& cf=force_nodes[child];
      const double3 ccom=node_com[child];
      const float dx[3]={{float(com[0]-ccom.x),float(com[1]-ccom.y),float(com[2]-ccom.z)}};
      radius=fmaxf(radius,cf.center.w+sqrtf(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]));
      soft=fmaxf(soft,cf.m.max_softening);
      total_m+=cf.m.M_000;
      if constexpr(track_vel) {{
        const float dvx=cf.m.vel[0]-vel_com[0],dvy=cf.m.vel[1]-vel_com[1],dvz=cf.m.vel[2]-vel_com[2];
        max_dv=fmaxf(max_dv,cf.m.max_delta_vel[0]+sqrtf(dvx*dvx+dvy*dvy+dvz*dvz));
      }}
      float px[5],py[5],pz[5];
#pragma unroll
      for(int k=0;k<5;++k) {{
        px[k]=powers_f(dx[0],k);
        py[k]=powers_f(dx[1],k);
        pz[k]=powers_f(dx[2],k);
      }}
      float translated;
{m2m_str}
    }}
    mass=total_m;
    double sq=0.0;
    for(int a=0;a<3;++a) {{
      const double d=fmax(com[a]-lower[a],lower[a]+width[a]-com[a]);
      sq+=d*d;
    }}
    radius=fminf(radius,float(sqrt(sq)));
  }}
  moments[0]=mass;
  node_com[i]={{com[0],com[1],com[2]}};
  fn.center={{float(com[0]),float(com[1]),float(com[2]),radius}};
  fn.m.max_softening=soft;
  fn.m.power[0]=mass;
  fn.m.power[2]=sqrtf({p2_expr});
  if constexpr(track_vel) {{
    fn.m.vel[0]=vel_com[0]; fn.m.vel[1]=vel_com[1]; fn.m.vel[2]=vel_com[2];
    fn.m.max_delta_vel[0]=max_dv;
  }}
{assign_str}
}}

__global__ void multipoles_fast_slab(const Geometry* geometry,ForceNode* force_nodes,
    double3* node_com,const double3* x_in,const float* mass_or_null,float uniform_mass,
    float uniform_soft,const int* sorted,int level,int slab_x,probe_header h,
    const int* levels,const float* soft_or_null=nullptr) {{
  const int begin=levels[level];
  const int end=levels[level+1];
  const int i=begin+blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=end) return;
  const auto& geom=geometry[i];
  const int count=geom.end-geom.start;
  ForceNode& fn=force_nodes[i];
  fn.n=count;
  if(!count) {{
    fn.center={{}};
    fn.m={{}};
    node_com[i]={{}};
    return;
  }}
  const int local_top=geom.key>>octree_shift;
  double lower[3]={{
      double(slab_x)*h.width[0],
      double(local_top/h.top[2])*h.width[1],
      double(local_top%h.top[2])*h.width[2]}};
  double width[3]={{h.width[0],h.width[1],h.width[2]}};
  for(int l=1;l<=level;++l) {{
    const int child=(geom.key>>(3*(PROBE_LEVELS-l)))&7;
    for(int a=0;a<3;++a) {{
      width[a]*=0.5;
      if(child&(4>>a)) lower[a]+=width[a];
    }}
  }}
  bool leaf=true;
  for(int child:geom.children) leaf&=child<0;

  float moments[PROBE_TERMS]={{}};
  double com[3]={{}};
  float radius=0.f,soft=soft_or_null?0.f:uniform_soft,mass=0.f;

  if(leaf) {{
    float rel_cx=0.f,rel_cy=0.f,rel_cz=0.f;
    for(int j=geom.start;j<geom.end;++j) {{
      const int pid=sorted[j];
      const double3 pos=x_in[pid];
      const float mp=mass_or_null?mass_or_null[pid]:uniform_mass;
      if(soft_or_null) soft=fmaxf(soft,soft_or_null[pid]);
      mass+=mp;
      rel_cx+=mp*float(pos.x-lower[0]);
      rel_cy+=mp*float(pos.y-lower[1]);
      rel_cz+=mp*float(pos.z-lower[2]);
    }}
    const float inv_m=1.f/mass;
    const float com_rel[3]={{rel_cx*inv_m,rel_cy*inv_m,rel_cz*inv_m}};
    com[0]=lower[0]+double(com_rel[0]);
    com[1]=lower[1]+double(com_rel[1]);
    com[2]=lower[2]+double(com_rel[2]);
    float radius2=0.f;
    for(int j=geom.start;j<geom.end;++j) {{
      const int pid=sorted[j];
      const double3 pos=x_in[pid];
      const float mp=mass_or_null?mass_or_null[pid]:uniform_mass;
      const float dx[3]={{
          com_rel[0]-float(pos.x-lower[0]),
          com_rel[1]-float(pos.y-lower[1]),
          com_rel[2]-float(pos.z-lower[2])}};
      radius2=fmaxf(radius2,dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
      float px[5],py[5],pz[5];
#pragma unroll
      for(int k=0;k<5;++k) {{
        px[k]=powers_f(dx[0],k);
        py[k]=powers_f(dx[1],k);
        pz[k]=powers_f(dx[2],k);
      }}
{leaf_acc}
    }}
    radius=sqrtf(radius2);
  }} else {{
    double dm=0.0;
    for(int child:geom.children) {{
      if(child<0) continue;
      const float cm=force_nodes[child].m.M_000;
      const double3 ccom=node_com[child];
      dm+=double(cm);
      com[0]+=double(cm)*ccom.x;
      com[1]+=double(cm)*ccom.y;
      com[2]+=double(cm)*ccom.z;
    }}
    const double inv_dm=1.0/dm;
    com[0]*=inv_dm; com[1]*=inv_dm; com[2]*=inv_dm;
    float total_m=0.f;
    for(int child:geom.children) {{
      if(child<0) continue;
      const ForceNode& cf=force_nodes[child];
      const double3 ccom=node_com[child];
      const float dx[3]={{float(com[0]-ccom.x),float(com[1]-ccom.y),float(com[2]-ccom.z)}};
      radius=fmaxf(radius,cf.center.w+sqrtf(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]));
      soft=fmaxf(soft,cf.m.max_softening);
      total_m+=cf.m.M_000;
      float px[5],py[5],pz[5];
#pragma unroll
      for(int k=0;k<5;++k) {{
        px[k]=powers_f(dx[0],k);
        py[k]=powers_f(dx[1],k);
        pz[k]=powers_f(dx[2],k);
      }}
      float translated;
{m2m_str}
    }}
    mass=total_m;
    double sq=0.0;
    for(int a=0;a<3;++a) {{
      const double d=fmax(com[a]-lower[a],lower[a]+width[a]-com[a]);
      sq+=d*d;
    }}
    radius=fminf(radius,float(sqrt(sq)));
  }}
  moments[0]=mass;
  node_com[i]={{com[0],com[1],com[2]}};
  fn.center={{float(com[0]),float(com[1]),float(com[2]),radius}};
  fn.m.max_softening=soft;
  fn.m.power[0]=mass;
  fn.m.power[2]=sqrtf({p2_expr});
{assign_str}
}}

__global__ void multipoles_fast_3d(const Geometry* geometry,ForceNode* force_nodes,
    double3* node_com,const double3* x_in,const float* mass_or_null,float uniform_mass,
    float uniform_soft,const int* sorted,int level,const int* slot_to_cell,probe_header h,
    const int* levels,const float* soft_or_null=nullptr) {{
  const int begin=levels[level];
  const int end=levels[level+1];
  const int i=begin+blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=end) return;
  const auto& geom=geometry[i];
  const int count=geom.end-geom.start;
  ForceNode& fn=force_nodes[i];
  fn.n=count;
  if(!count) {{
    fn.center={{}};
    fn.m={{}};
    node_com[i]={{}};
    return;
  }}
  const int local_top=geom.key>>octree_shift;
  const int cid=slot_to_cell[local_top];
  const int yz_stride=h.top[1]*h.top[2];
  const int tx=cid/yz_stride;
  const int rem=cid%yz_stride;
  const int ty=rem/h.top[2];
  const int tz=rem%h.top[2];
  double lower[3]={{
      double(tx)*h.width[0],
      double(ty)*h.width[1],
      double(tz)*h.width[2]}};
  double width[3]={{h.width[0],h.width[1],h.width[2]}};
  for(int l=1;l<=level;++l) {{
    const int child=(geom.key>>(3*(PROBE_LEVELS-l)))&7;
    for(int a=0;a<3;++a) {{
      width[a]*=0.5;
      if(child&(4>>a)) lower[a]+=width[a];
    }}
  }}
  bool leaf=true;
  for(int child:geom.children) leaf&=child<0;

  float moments[PROBE_TERMS]={{}};
  double com[3]={{}};
  float radius=0.f,soft=soft_or_null?0.f:uniform_soft,mass=0.f;

  if(leaf) {{
    float rel_cx=0.f,rel_cy=0.f,rel_cz=0.f;
    for(int j=geom.start;j<geom.end;++j) {{
      const int pid=sorted[j];
      const double3 pos=x_in[pid];
      const float mp=mass_or_null?mass_or_null[pid]:uniform_mass;
      if(soft_or_null) soft=fmaxf(soft,soft_or_null[pid]);
      mass+=mp;
      rel_cx+=mp*float(pos.x-lower[0]);
      rel_cy+=mp*float(pos.y-lower[1]);
      rel_cz+=mp*float(pos.z-lower[2]);
    }}
    const float inv_m=1.f/mass;
    const float com_rel[3]={{rel_cx*inv_m,rel_cy*inv_m,rel_cz*inv_m}};
    com[0]=lower[0]+double(com_rel[0]);
    com[1]=lower[1]+double(com_rel[1]);
    com[2]=lower[2]+double(com_rel[2]);
    float radius2=0.f;
    for(int j=geom.start;j<geom.end;++j) {{
      const int pid=sorted[j];
      const double3 pos=x_in[pid];
      const float mp=mass_or_null?mass_or_null[pid]:uniform_mass;
      const float dx[3]={{
          com_rel[0]-float(pos.x-lower[0]),
          com_rel[1]-float(pos.y-lower[1]),
          com_rel[2]-float(pos.z-lower[2])}};
      radius2=fmaxf(radius2,dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]);
      float px[5],py[5],pz[5];
#pragma unroll
      for(int k=0;k<5;++k) {{
        px[k]=powers_f(dx[0],k);
        py[k]=powers_f(dx[1],k);
        pz[k]=powers_f(dx[2],k);
      }}
{leaf_acc}
    }}
    radius=sqrtf(radius2);
  }} else {{
    double dm=0.0;
    for(int child:geom.children) {{
      if(child<0) continue;
      const float cm=force_nodes[child].m.M_000;
      const double3 ccom=node_com[child];
      dm+=double(cm);
      com[0]+=double(cm)*ccom.x;
      com[1]+=double(cm)*ccom.y;
      com[2]+=double(cm)*ccom.z;
    }}
    const double inv_dm=1.0/dm;
    com[0]*=inv_dm; com[1]*=inv_dm; com[2]*=inv_dm;
    float total_m=0.f;
    for(int child:geom.children) {{
      if(child<0) continue;
      const ForceNode& cf=force_nodes[child];
      const double3 ccom=node_com[child];
      const float dx[3]={{float(com[0]-ccom.x),float(com[1]-ccom.y),float(com[2]-ccom.z)}};
      radius=fmaxf(radius,cf.center.w+sqrtf(dx[0]*dx[0]+dx[1]*dx[1]+dx[2]*dx[2]));
      soft=fmaxf(soft,cf.m.max_softening);
      total_m+=cf.m.M_000;
      float px[5],py[5],pz[5];
#pragma unroll
      for(int k=0;k<5;++k) {{
        px[k]=powers_f(dx[0],k);
        py[k]=powers_f(dx[1],k);
        pz[k]=powers_f(dx[2],k);
      }}
      float translated;
{m2m_str}
    }}
    mass=total_m;
    double sq=0.0;
    for(int a=0;a<3;++a) {{
      const double d=fmax(com[a]-lower[a],lower[a]+width[a]-com[a]);
      sq+=d*d;
    }}
    radius=fminf(radius,float(sqrt(sq)));
  }}
  moments[0]=mass;
  node_com[i]={{com[0],com[1],com[2]}};
  fn.center={{float(com[0]),float(com[1]),float(com[2]),radius}};
  fn.m.max_softening=soft;
  fn.m.power[0]=mass;
  fn.m.power[2]=sqrtf({p2_expr});
{assign_str}
}}
"""
(build/"src/multipoles_fast.cuh").write_text(fast_mpole)
