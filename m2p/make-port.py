from pathlib import Path
import os
import re
import shutil
import sys

base=Path(sys.argv[1])/"src"
here=Path(__file__).parent
for name in ("m2p_port.h","m2p_device.cuh"):
    shutil.copyfile(here/name,base/name)

# Use the configured SWIFT mathematical kernels rather than maintaining another
# copy of the derivatives or multipole contraction expressions.
functions=[
    ("kernel_gravity.h",[f"D_soft_{i}" for i in range(1,7)]),
    ("kernel_long_gravity.h",["kernel_long_grav_derivatives"]),
    ("gravity_derivatives.h",["potential_derivatives_compute_M2P"]),
    ("multipole.h",["gravity_M2P"]),
]
grouped=os.getenv("GRAV_GROUP_M2L")=="1"
if grouped:
    functions[-2][1].append("potential_derivatives_compute_M2L")
    functions[-1][1].append("gravity_M2L_apply")
names=[n for _,group in functions for n in group]
def extract(source,name):
    pattern=rf"__attribute__\(\([^\n]+\)\) INLINE static (?:float|void)\s+{name}\("
    m=re.search(pattern,source)
    assert m,name
    at=source.index("{",m.end())
    depth=1
    i=at+1
    while depth:
        depth+=(source[i]=="{")-(source[i]=="}")
        i+=1
    body=source[m.start():i]
    body=re.sub(r"__attribute__\(\([^\n]+\)\) INLINE static","static __device__ __forceinline__",body,count=1)
    return re.sub(r"\b("+"|".join(names)+r")\b",lambda m:"grav_cuda_"+m[0],body)

pieces=[]
for filename,group in functions:
    native=(base/filename).read_text()
    for name in group:
        pieces.append(extract(native,name))
(base/"m2p_native_math.cuh").write_text(
    "// Generated from the matching SWIFT checkout by make-port.py.\n"
    "#if defined(SWIFT_DEBUG_CHECKS) || defined(SWIFT_GRAVITY_FORCE_CHECKS)\n"
    "#error Disable native debug counters for this experimental device backend.\n#endif\n"
    +"\n\n".join(pieces)+"\n")

if os.getenv("GRAV_M2P_MATH_ONLY")=="1":
    raise SystemExit(0)

def replace(s,old,new):
    assert s.count(old)==1,old
    return s.replace(old,new,1)
def edit(file,f):
    p=base/file
    p.write_text(f(p.read_text()))

def header(s):
    s=replace(s,'#include "gpu_mapping.h"','#include "gpu_mapping.h"\n#include "m2p_port.h"')
    return replace(s,'  int pair_total_count;','''  int pair_total_count;
  int m2p_enabled;
  struct m2p_source_gpu *m2p_sources_h, *m2p_sources_d;
  float *m2p_old_h, *m2p_old_d;
  unsigned char *m2p_mask_d;''')
edit("runner_gpu.h",header)

def kernels(s):
    i=s.index("template <bool compact>\n__global__ void pair_grav_pp_kernel_tiled")
    pre=s[:i]
    s=s[i:]
    s=replace(s,"template <bool compact>","template <bool compact, bool with_m2p = false>")
    s=replace(s,"    int ncells) {","    int ncells, const unsigned char* m2p_mask = nullptr) {")
    s=replace(s,"  const int valid_pid = active_slot < active_count_i;",
               '''  bool valid_pid = active_slot < active_count_i;
  const int side = 2 * pair_id + swap;
  const int out_base = pair_side_active_offsets_d[side];
  if constexpr (with_m2p) {
    if (valid_pid && m2p_mask[out_base+active_slot]) valid_pid=false;
    if (!__syncthreads_or(valid_pid)) return;
  }''')
    s=replace(s,"""    const int side = 2 * pair_id + swap;
    const int out_base = pair_side_active_offsets_d[side];
    const int out = out_base + active_slot;""",
               "    const int out = out_base + active_slot;")
    return pre+s
edit("gpu_gravity_functions.cuh",kernels)

def dispatch(s):
    s=replace(s,'#include "gpu_gravity_functions.cuh"','#include "gpu_gravity_functions.cuh"\n#include "m2p_device.cuh"')
    i=s.index('extern "C" void pair_pp_offload_gpu(')
    pre=s[:i];s=s[i:]
    s=replace(s,'''    int work_j,
    GPUStream stream) {''','''    int work_j,
    struct m2p_batch_gpu m2p,
    GPUStream stream) {''')
    anchor='''  if (work != nullptr) {
    if (work_i > 0) launch(std::true_type{}, dim3(work_i), 0, work);'''
    s=replace(s,anchor,'''  if (m2p.mask != nullptr) {
    const int nwork[2]={work_i,work_j};
    for (int side=0;side<2;++side) {
      if (nwork[side]==0) continue;
      const int2* w=work+(side?work_i:0);
      native_m2p_pair<<<nwork[side],block,0,stream>>>(
          w,side,pair_pair_i_d,pair_pair_j_d,pair_counts_d,pair_offsets_d,
          pair_active_counts_d,pair_active_offsets_d,pair_active_index_d,
          pair_side_active_offsets_d,pair_use_full_d,send_pair_pos_mass_d,
          send_pair_h_d,gravity_gpu_values_recv_d,m2p,periodic,*r_s_inv,
          dim_0,dim_1,dim_2);
      pair_grav_pp_kernel_tiled<true,true><<<nwork[side],block,shmem,stream>>>(
          pair_cell_flags_d,pair_use_full_d,pair_side_active_offsets_d,
          send_pair_pos_mass_d,send_pair_h_d,gravity_gpu_values_recv_d,
          pair_counts_d,pair_offsets_d,pair_active_counts_d,pair_active_offsets_d,
          pair_active_index_d,pair_pair_i_d,pair_pair_j_d,npairs,nslots,periodic,
          *r_s_inv,side,w,dim_0,dim_1,dim_2,nslots,m2p.mask);
    }
  } else if (work != nullptr) {
    if (work_i > 0) launch(std::true_type{}, dim3(work_i), 0, work);''')
    return pre+s
edit("gpu_gravity_offload.cu",dispatch)

def runner(s):
    s=replace(s,'''    int work_i, int work_j,
    GPUStream stream);''','''    int work_i, int work_j,
    struct m2p_batch_gpu m2p,
    GPUStream stream);''')
    s=replace(s,'''  substream->pair_unique_cells[slot] = c;

  return slot;''','''  substream->pair_unique_cells[slot] = c;
  if (substream->m2p_enabled) {
    const struct gravity_tensors *g = c->grav.multipole;
    substream->m2p_sources_h[slot] = (struct m2p_source_gpu){
        .center_radius={(float)g->CoM[0],(float)g->CoM[1],(float)g->CoM[2],g->r_max},
        .expansion=g->m_pole};
    for (int a=0;a<active_count;++a) {
      const int i=substream->pair_active_index_h[active_base+a];
      substream->m2p_old_h[active_base+a]=c->grav.parts[i].old_a_grav_norm;
    }
  }

  return slot;''')
    s=replace(s,'''    pair_pp_offload_gpu(
''','''    struct m2p_batch_gpu m2p = {0};
    if (substream->m2p_enabled) {
      const struct gravity_props *props=e->gravity_properties;
      if (props->use_gadget_tolerance || props->consider_truncation_in_MAC)
        error("Device M2P currently supports the geometric or native Dehnen MAC");
      const float theta=props->theta_crit;
      m2p = (struct m2p_batch_gpu){
          .sources=substream->m2p_sources_d,.old_accel=substream->m2p_old_d,
          .mask=substream->m2p_mask_d,
          .params={props->adaptive_tolerance,theta*theta,props->use_advanced_MAC,
                   props->use_tree_below_softening}};
      const size_t sources_bytes=nslots*sizeof(*substream->m2p_sources_h);
      const size_t old_bytes=substream->pair_total_active_count*sizeof(float);
      pair_force_check(GPUMemcpyAsync(substream->m2p_sources_d,substream->m2p_sources_h,
                   sources_bytes,GPU_MEMCPY_HOST_TO_DEVICE,stream),"copy M2P sources");
      pair_force_check(GPUMemcpyAsync(substream->m2p_old_d,substream->m2p_old_h,
                   old_bytes,GPU_MEMCPY_HOST_TO_DEVICE,stream),"copy M2P target acceleration");
      substream->pair_uploaded_bytes += sources_bytes+old_bytes;
    }
    pair_pp_offload_gpu(
''')
    s=replace(s,'''        work_i, work_j,
        stream);''','''        work_i, work_j,
        m2p, stream);''')
    s=replace(s,'''    const size_t active_recv_bytes =
''','''    const char *m2p_mode=getenv("GRAV_BENCH_DEVICE_M2P");
    substream->m2p_enabled=m2p_mode && strcmp(m2p_mode,"1")==0;
    if (substream->m2p_enabled) {
      if (!substream->pair_direct_pack || !gpu->pair_compact_blocks || gpu->pair_pipeline)
        error("Device M2P requires compact direct-pack batches and synchronous completion");
      const size_t slots=gpu->grav_batch_ncells;
      const size_t particles=slots*gpu->grav_max_cell_size;
      pair_force_check(GPUHostMalloc((void **)&substream->m2p_sources_h,
                                      slots*sizeof(*substream->m2p_sources_h)),"allocate M2P host source");
      pair_force_check(GPUMalloc((void **)&substream->m2p_sources_d,
                                  slots*sizeof(*substream->m2p_sources_d)),"allocate M2P source");
      pair_force_check(GPUHostMalloc((void **)&substream->m2p_old_h,
                                      particles*sizeof(float)),"allocate M2P host target");
      pair_force_check(GPUMalloc((void **)&substream->m2p_old_d,
                                  particles*sizeof(float)),"allocate M2P target");
      pair_force_check(GPUMalloc((void **)&substream->m2p_mask_d,
                                  particles*sizeof(unsigned char)),"allocate M2P decisions");
    }

    const size_t active_recv_bytes =
''')
    s=replace(s,'    /* Pair buffers */','''    if (substream->m2p_enabled) {
      GPUFreeHost(substream->m2p_sources_h);
      GPUFree(substream->m2p_sources_d);
      GPUFreeHost(substream->m2p_old_h);
      GPUFree(substream->m2p_old_d);
      GPUFree(substream->m2p_mask_d);
    }
    /* Pair buffers */''')
    return s
edit("runner_gpu.c",runner)
