#include "grav_cuda_fof.h"

#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace grav_fof {

using Key = std::uint64_t;
using Atomic = cuda::atomic_ref<int, cuda::thread_scope_device>;

constexpr int threads = 256;
constexpr int max_grid_blocks = 65535;
constexpr int heavy_cell_threshold = 64;
constexpr std::int64_t grav_fof_none = 2147483647LL;

struct GpuError {
  cudaError_t status;
};

inline void check(cudaError_t status) {
  if (status != cudaSuccess) throw GpuError{status};
}

template <typename T>
struct Buffer {
  T* ptr = nullptr;
  Buffer() = default;
  explicit Buffer(std::size_t count) {
    if (count > 0) check(cudaMalloc(&ptr, sizeof(T) * count));
  }
  ~Buffer() {
    if (ptr) cudaFree(ptr);
  }
  void release() {
    if (ptr) {
      check(cudaFree(ptr));
      ptr = nullptr;
    }
  }
  void keep_only(T* used) {
    if (ptr != used) release();
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&& o) noexcept : ptr(o.ptr) { o.ptr = nullptr; }
  Buffer& operator=(Buffer&& o) noexcept {
    if (this != &o) {
      if (ptr) cudaFree(ptr);
      ptr = o.ptr;
      o.ptr = nullptr;
    }
    return *this;
  }
};

struct Timer {
  cudaEvent_t begin = nullptr, end = nullptr;
  Timer() {
    check(cudaEventCreate(&begin));
    cudaError_t st = cudaEventCreate(&end);
    if (st != cudaSuccess) {
      cudaEventDestroy(begin);
      throw GpuError{st};
    }
  }
  ~Timer() {
    if (begin) cudaEventDestroy(begin);
    if (end) cudaEventDestroy(end);
  }
  void start() { check(cudaEventRecord(begin)); }
  void stop(double* dst) {
    check(cudaGetLastError());
    check(cudaEventRecord(end));
    check(cudaEventSynchronize(end));
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, begin, end));
    *dst = double(ms) / 1000.0;
  }
};

struct Geometry {
  double lx, ly, lz;
  double hx, hy, hz;
  int nx, ny, nz;
  double limit;
  bool use_octants;
};

__device__ __forceinline__ double in_box(double x, double L) {
  if (x == 0.0) return 0.0;
  if (x > 0.0 && x < L) return x;
  double u = fmod(x, L);
  if (u < 0.0) u += L;
  if (u >= L) u = 0.0;
  return u;
}

__device__ __forceinline__ int axis_key(double x, double L, int n) {
  const double scale = double(n) / L;
  return min(n - 1, max(0, int(x * scale)));
}

__device__ __forceinline__ Key grid_key(int x, int y, int z, const Geometry& geo) {
  return Key(x) + Key(geo.nx) * (Key(y) + Key(geo.ny) * Key(z));
}

__device__ __forceinline__ Key hash_key(Key key) {
  key ^= key >> 30;
  key *= 0xbf58476d1ce4e5b9ULL;
  key ^= key >> 27;
  key *= 0x94d049bb133111ebULL;
  return key ^ (key >> 31);
}

__device__ __forceinline__ int find_cell(Key key, const Key* keys, const int* table, Key mask) {
  Key slot = hash_key(key) & mask;
  for (;;) {
    const int entry = table[slot];
    if (entry == 0) return -1;
    if (keys[entry - 1] == key) return entry - 1;
    slot = (slot + 1ULL) & mask;
  }
}

__device__ __forceinline__ int root(int* parent, int node) {
  for (;;) {
    const int p = Atomic(parent[node]).load(cuda::memory_order_relaxed);
    if (p == node) return node;
    const int grand = Atomic(parent[p]).load(cuda::memory_order_relaxed);
    if (grand != p) atomicMin(parent + node, grand);
    node = grand;
  }
}

__device__ __forceinline__ void link_edge(int* parent, int u, int v) {
  int a = root(parent, u), b = root(parent, v);
  while (a != b) {
    const int hi = max(a, b), lo = min(a, b);
    const int prior = atomicCAS(parent + hi, hi, lo);
    if (prior == hi) return;
    a = root(parent, prior);
    b = root(parent, lo);
  }
}

__device__ __forceinline__ double exact_image(double a, double b, double L) {
  const double d = a - b;
  if (d > 0.5 * L) return d - L;
  if (d < -0.5 * L) return d + L;
  return d;
}

__device__ __forceinline__ bool linked_d3(const double3 p, const double3 q, const Geometry& g) {
  const double x = exact_image(p.x, q.x, g.lx);
  const double y = exact_image(p.y, q.y, g.ly);
  const double z = exact_image(p.z, q.z, g.lz);
  return x * x + y * y + z * z <= g.limit;
}

__device__ __forceinline__ int wrap_cell(int i, int n) {
  return i < 0 ? i + n : (i >= n ? i - n : i);
}

__device__ __forceinline__ double axis_gap(double min_a, double max_a, double min_b, double max_b) {
  if (max_a < min_b) return min_b - max_a;
  if (min_a > max_b) return min_a - max_b;
  return 0.0;
}

__device__ __forceinline__ int particle_oct(const double3 p, const double3 mid) {
  return (p.x >= mid.x ? 1 : 0) | (p.y >= mid.y ? 2 : 0) | (p.z >= mid.z ? 4 : 0);
}

__global__ void grid_index_raw(const double3* raw_in, double3* pos_wrapped, Key* keys, int* rows,
                               int n, Geometry g, int* invalid) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    double3 p = raw_in[i];
    if (!(isfinite(p.x) && isfinite(p.y) && isfinite(p.z))) {
      atomicExch(invalid, 1);
      continue;
    }
    p.x = in_box(p.x, g.lx);
    p.y = in_box(p.y, g.ly);
    p.z = in_box(p.z, g.lz);
    pos_wrapped[i] = p;
    keys[i] = grid_key(axis_key(p.x, g.lx, g.nx),
                       axis_key(p.y, g.ly, g.ny),
                       axis_key(p.z, g.lz, g.nz), g);
    rows[i] = i;
  }
}

__global__ void gather_sorted_and_init_parent(const double3* pos_wrapped, const int* rows,
                                              double3* pos_sorted, int* parent, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
    pos_sorted[k] = pos_wrapped[rows[k]];
    parent[k] = k;
  }
}

__global__ void build_lookup(const Key* keys, int c_count, int* table, Key mask) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < c_count; i += stride) {
    Key slot = hash_key(keys[i]) & mask;
    while (atomicCAS(table + slot, 0, i + 1) != 0) slot = (slot + 1ULL) & mask;
  }
}

__global__ void intra_cell_light(const double3* pos_s, const Key* keys, const int* starts,
                                 int c_count, Geometry g, int* parent,
                                 double3* bmin_arr, double3* bmax_arr, int* cell_single_root,
                                 int* heavy_cells, int* heavy_cell_count) {
  const int stride = gridDim.x * blockDim.x;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < c_count; c += stride) {
    const int begin = starts[c];
    const int end = starts[c + 1];
    const int m = end - begin;
    if (m > heavy_cell_threshold) {
      const int idx = atomicAdd(heavy_cell_count, 1);
      heavy_cells[idx] = c;
      cell_single_root[c] = -1;
      continue;
    }

    double3 bmn = {DBL_MAX, DBL_MAX, DBL_MAX};
    double3 bmx = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
    if (m == 1) {
      const double3 p = pos_s[begin];
      bmin_arr[c] = p;
      bmax_arr[c] = p;
      cell_single_root[c] = begin;
      continue;
    }

    if (g.use_octants) {
      const Key key = keys[c];
      const int cx = int(key % Key(g.nx));
      const int cy = int((key / Key(g.nx)) % Key(g.ny));
      const int cz = int(key / (Key(g.nx) * Key(g.ny)));
      const double3 mid{(double(cx) + 0.5) * g.hx,
                        (double(cy) + 0.5) * g.hy,
                        (double(cz) + 0.5) * g.hz};

      int oct_rep[8];
      unsigned char p_oct[heavy_cell_threshold];
      for (int o = 0; o < 8; ++o) oct_rep[o] = -1;

      for (int off = 0; off < m; ++off) {
        const int k = begin + off;
        const double3 p = pos_s[k];
        bmn.x = fmin(bmn.x, p.x); bmn.y = fmin(bmn.y, p.y); bmn.z = fmin(bmn.z, p.z);
        bmx.x = fmax(bmx.x, p.x); bmx.y = fmax(bmx.y, p.y); bmx.z = fmax(bmx.z, p.z);
        const int oct = particle_oct(p, mid);
        p_oct[off] = static_cast<unsigned char>(oct);
        if (oct_rep[oct] < 0) oct_rep[oct] = k;
        else parent[k] = oct_rep[oct];
      }
      bmin_arr[c] = bmn;
      bmax_arr[c] = bmx;

      for (int i_off = 0; i_off < m; ++i_off) {
        const double3 pi = pos_s[begin + i_off];
        const int ri = oct_rep[p_oct[i_off]];
        for (int j_off = i_off + 1; j_off < m; ++j_off) {
          if (p_oct[i_off] == p_oct[j_off]) continue;
          const int rj = oct_rep[p_oct[j_off]];
          if (root(parent, ri) == root(parent, rj)) continue;
          if (linked_d3(pi, pos_s[begin + j_off], g)) {
            link_edge(parent, ri, rj);
          }
        }
      }

      int first_r = -1;
      bool all_same = true;
      for (int o = 0; o < 8; ++o) {
        if (oct_rep[o] < 0) continue;
        const int r = root(parent, oct_rep[o]);
        if (first_r < 0) first_r = r;
        else if (r != first_r) { all_same = false; break; }
      }
      cell_single_root[c] = all_same ? first_r : -1;
    } else {
      for (int k = begin; k < end; ++k) {
        const double3 p = pos_s[k];
        bmn.x = fmin(bmn.x, p.x); bmn.y = fmin(bmn.y, p.y); bmn.z = fmin(bmn.z, p.z);
        bmx.x = fmax(bmx.x, p.x); bmx.y = fmax(bmx.y, p.y); bmx.z = fmax(bmx.z, p.z);
      }
      bmin_arr[c] = bmn;
      bmax_arr[c] = bmx;
      for (int i = begin; i < end; ++i) {
        const double3 pi = pos_s[i];
        for (int j = i + 1; j < end; ++j) {
          if (root(parent, i) == root(parent, j)) continue;
          if (linked_d3(pi, pos_s[j], g)) {
            link_edge(parent, i, j);
          }
        }
      }
      const int r0 = root(parent, begin);
      bool all_same = true;
      for (int k = begin + 1; k < end; ++k) {
        if (root(parent, k) != r0) { all_same = false; break; }
      }
      cell_single_root[c] = all_same ? r0 : -1;
    }
  }
}

struct DeviceGuard {
  int saved = 0;
  bool valid = false;
  DeviceGuard() {
    valid = (cudaGetDevice(&saved) == cudaSuccess);
  }
  ~DeviceGuard() {
    if (valid) cudaSetDevice(saved);
  }
};

__device__ __forceinline__ double atomicAddDouble(double* addr, double val) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600
  return atomicAdd(addr, val);
#else
  unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(addr_ull, assumed, __double_as_longlong(__longlong_as_double(assumed) + val));
  } while (assumed != old);
  return __longlong_as_double(old);
#endif
}

// Fast single-instruction non-negative double max: valid ONLY for finite val >= +0.0.
__device__ __forceinline__ void atomicMaxNonNegDouble(double* addr, double val) {
  const double clamped = fmax(0.0, val);
  atomicMax(reinterpret_cast<unsigned long long*>(addr), __double_as_longlong(clamped));
}

using AtomicU32 = cuda::atomic_ref<std::uint32_t, cuda::thread_scope_device>;

__device__ __forceinline__ std::uint32_t root_u32(std::uint32_t* parent, std::uint32_t node) {
  for (;;) {
    const std::uint32_t p = AtomicU32(parent[node]).load(cuda::memory_order_relaxed);
    if (p == node) return node;
    const std::uint32_t grand = AtomicU32(parent[p]).load(cuda::memory_order_relaxed);
    if (grand != p) atomicMin(parent + node, grand);
    node = grand;
  }
}

__device__ __forceinline__ void link_edge_u32(std::uint32_t* parent, std::uint32_t u, std::uint32_t v) {
  std::uint32_t a = root_u32(parent, u);
  std::uint32_t b = root_u32(parent, v);
  while (a != b) {
    const std::uint32_t hi = (a > b) ? a : b;
    const std::uint32_t lo = (a < b) ? a : b;
    const std::uint32_t prior = atomicCAS(parent + hi, hi, lo);
    if (prior == hi) return;
    a = root_u32(parent, prior);
    b = root_u32(parent, lo);
  }
}

__device__ void atomicMinDouble(double* addr, double val) {
  unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_ull, assumed;
  do {
    assumed = old;
    if (__longlong_as_double(assumed) <= val) break;
    old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);
}

__device__ void atomicMaxDouble(double* addr, double val) {
  unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(addr);
  unsigned long long old = *addr_ull, assumed;
  do {
    assumed = old;
    if (__longlong_as_double(assumed) >= val) break;
    old = atomicCAS(addr_ull, assumed, __double_as_longlong(val));
  } while (assumed != old);
}

__global__ void intra_cell_heavy(const double3* pos_s, const Key* keys, const int* starts,
                                 const int* heavy_cells, int num_heavy, Geometry g,
                                 int* parent, double3* bmin_arr, double3* bmax_arr,
                                 int* cell_single_root) {
  if (blockIdx.x >= num_heavy) return;
  const int c = heavy_cells[blockIdx.x];
  const int begin = starts[c];
  const int end = starts[c + 1];
  const int tid = threadIdx.x;
  const double limit_eps = g.limit * (1.0 + 1e-12);

  __shared__ double3 s_bmin, s_bmax;
  __shared__ double3 s_oct_min[8], s_oct_max[8];
  __shared__ int s_oct_rep[8];
  __shared__ int s_do_pair, s_found;

  if (tid == 0) {
    s_bmin = {DBL_MAX, DBL_MAX, DBL_MAX};
    s_bmax = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
  }
  if (tid < 8) {
    s_oct_rep[tid] = end;
    s_oct_min[tid] = {DBL_MAX, DBL_MAX, DBL_MAX};
    s_oct_max[tid] = {-DBL_MAX, -DBL_MAX, -DBL_MAX};
  }
  __syncthreads();

  const Key key = keys[c];
  const int cx = int(key % Key(g.nx));
  const int cy = int((key / Key(g.nx)) % Key(g.ny));
  const int cz = int(key / (Key(g.nx) * Key(g.ny)));
  const double3 mid{(double(cx) + 0.5) * g.hx,
                    (double(cy) + 0.5) * g.hy,
                    (double(cz) + 0.5) * g.hz};

  double3 lmin{DBL_MAX, DBL_MAX, DBL_MAX}, lmax{-DBL_MAX, -DBL_MAX, -DBL_MAX};
  for (int k = begin + tid; k < end; k += blockDim.x) {
    const double3 p = pos_s[k];
    lmin.x = fmin(lmin.x, p.x); lmin.y = fmin(lmin.y, p.y); lmin.z = fmin(lmin.z, p.z);
    lmax.x = fmax(lmax.x, p.x); lmax.y = fmax(lmax.y, p.y); lmax.z = fmax(lmax.z, p.z);
    if (g.use_octants) {
      const int oct = particle_oct(p, mid);
      atomicMin(&s_oct_rep[oct], k);
    }
  }
  if (lmin.x <= lmax.x) {
    atomicMinDouble(&s_bmin.x, lmin.x);
    atomicMinDouble(&s_bmin.y, lmin.y);
    atomicMinDouble(&s_bmin.z, lmin.z);
    atomicMaxDouble(&s_bmax.x, lmax.x);
    atomicMaxDouble(&s_bmax.y, lmax.y);
    atomicMaxDouble(&s_bmax.z, lmax.z);
  }
  __syncthreads();

  if (tid == 0) {
    bmin_arr[c] = s_bmin;
    bmax_arr[c] = s_bmax;
  }

  if (g.use_octants) {
    for (int k = begin + tid; k < end; k += blockDim.x) {
      const double3 p = pos_s[k];
      const int oct = particle_oct(p, mid);
      if (k != s_oct_rep[oct]) parent[k] = s_oct_rep[oct];
      atomicMinDouble(&s_oct_min[oct].x, p.x);
      atomicMinDouble(&s_oct_min[oct].y, p.y);
      atomicMinDouble(&s_oct_min[oct].z, p.z);
      atomicMaxDouble(&s_oct_max[oct].x, p.x);
      atomicMaxDouble(&s_oct_max[oct].y, p.y);
      atomicMaxDouble(&s_oct_max[oct].z, p.z);
    }
    __syncthreads();

    if (tid == 0) {
      for (int a = 0; a < 8; ++a) {
        if (s_oct_rep[a] >= end) continue;
        const double3 pa = pos_s[s_oct_rep[a]];
        for (int b = a + 1; b < 8; ++b) {
          if (s_oct_rep[b] >= end) continue;
          if (root(parent, s_oct_rep[a]) == root(parent, s_oct_rep[b])) continue;
          if (linked_d3(pa, pos_s[s_oct_rep[b]], g)) {
            link_edge(parent, s_oct_rep[a], s_oct_rep[b]);
          }
        }
      }
    }
    __syncthreads();

    for (int a = 0; a < 8; ++a) {
      for (int b = a + 1; b < 8; ++b) {
        if (tid == 0) {
          int ok = 0;
          if (s_oct_rep[a] < end && s_oct_rep[b] < end &&
              root(parent, s_oct_rep[a]) != root(parent, s_oct_rep[b])) {
            const double gx = axis_gap(s_oct_min[a].x, s_oct_max[a].x, s_oct_min[b].x, s_oct_max[b].x);
            const double gy = axis_gap(s_oct_min[a].y, s_oct_max[a].y, s_oct_min[b].y, s_oct_max[b].y);
            const double gz = axis_gap(s_oct_min[a].z, s_oct_max[a].z, s_oct_min[b].z, s_oct_max[b].z);
            if (gx * gx + gy * gy + gz * gz <= limit_eps) ok = 1;
          }
          s_do_pair = ok;
          s_found = 0;
        }
        __syncthreads();
        if (!s_do_pair) continue;

        for (int i = begin; i < end; ++i) {
          if (Atomic(s_found).load(cuda::memory_order_relaxed)) break;
          const double3 pi = pos_s[i];
          if (particle_oct(pi, mid) != a) continue;
          const double gx = axis_gap(pi.x, pi.x, s_oct_min[b].x, s_oct_max[b].x);
          const double gy = axis_gap(pi.y, pi.y, s_oct_min[b].y, s_oct_max[b].y);
          const double gz = axis_gap(pi.z, pi.z, s_oct_min[b].z, s_oct_max[b].z);
          if (gx * gx + gy * gy + gz * gz > limit_eps) continue;
          for (int j = begin + tid; j < end; j += blockDim.x) {
            if (Atomic(s_found).load(cuda::memory_order_relaxed)) break;
            const double3 pj = pos_s[j];
            if (particle_oct(pj, mid) != b) continue;
            if (linked_d3(pi, pj, g)) {
              if (atomicExch(&s_found, 1) == 0) {
                link_edge(parent, s_oct_rep[a], s_oct_rep[b]);
              }
              break;
            }
          }
        }
        __syncthreads();
      }
    }

    if (tid == 0) {
      int first_r = -1;
      bool all_same = true;
      for (int o = 0; o < 8; ++o) {
        if (s_oct_rep[o] >= end) continue;
        const int r = root(parent, s_oct_rep[o]);
        if (first_r < 0) first_r = r;
        else if (r != first_r) { all_same = false; break; }
      }
      cell_single_root[c] = all_same ? first_r : -1;
    }
  } else {
    for (int i = begin; i < end; ++i) {
      const double3 pi = pos_s[i];
      for (int j = i + 1 + tid; j < end; j += blockDim.x) {
        if (root(parent, i) == root(parent, j)) continue;
        if (linked_d3(pi, pos_s[j], g)) link_edge(parent, i, j);
      }
    }
    __syncthreads();
    if (tid == 0) {
      const int r0 = root(parent, begin);
      bool all_same = true;
      for (int k = begin + 1; k < end; ++k) {
        if (root(parent, k) != r0) { all_same = false; break; }
      }
      cell_single_root[c] = all_same ? r0 : -1;
    }
  }
}

__device__ __forceinline__ bool cell_pair_bbox_close(
    int c, int nb, const Key* keys, const double3* bmin_arr, const double3* bmax_arr,
    const Geometry& g) {
  const Key key_a = keys[c];
  const Key key_b = keys[nb];
  const int ax = int(key_a % Key(g.nx));
  const int ay = int((key_a / Key(g.nx)) % Key(g.ny));
  const int az = int(key_a / (Key(g.nx) * Key(g.ny)));
  const int bx = int(key_b % Key(g.nx));
  const int by = int((key_b / Key(g.nx)) % Key(g.ny));
  const int bz = int(key_b / (Key(g.nx) * Key(g.ny)));

  double sx = 0.0, sy = 0.0, sz = 0.0;
  if (ax == g.nx - 1 && bx == 0) sx = g.lx;
  else if (ax == 0 && bx == g.nx - 1) sx = -g.lx;
  if (ay == g.ny - 1 && by == 0) sy = g.ly;
  else if (ay == 0 && by == g.ny - 1) sy = -g.ly;
  if (az == g.nz - 1 && bz == 0) sz = g.lz;
  else if (az == 0 && bz == g.nz - 1) sz = -g.lz;

  const double3 a_min = bmin_arr[c], a_max = bmax_arr[c];
  const double3 b_min = bmin_arr[nb], b_max = bmax_arr[nb];
  const double gx = axis_gap(a_min.x, a_max.x, b_min.x + sx, b_max.x + sx);
  const double gy = axis_gap(a_min.y, a_max.y, b_min.y + sy, b_max.y + sy);
  const double gz = axis_gap(a_min.z, a_max.z, b_min.z + sz, b_max.z + sz);
  return (gx * gx + gy * gy + gz * gz) <= g.limit * (1.0 + 1e-12);
}

__global__ void connect_light(const double3* pos_s, const Key* keys, const int* starts,
                              const int* lookup, Key mask, int c_count, Geometry g,
                              const double3* bmin_arr, const double3* bmax_arr,
                              const int* cell_single_root, int* parent,
                              int2* heavy_pairs, std::size_t max_heavy, int* heavy_pair_count) {
  const int stride = gridDim.x * blockDim.x;
  for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < c_count; c += stride) {
    const int begin = starts[c], end = starts[c + 1];
    const int mA = end - begin;
    const int rA = cell_single_root[c];
    const Key key = keys[c];
    const int x = int(key % Key(g.nx));
    const int y = int((key / Key(g.nx)) % Key(g.ny));
    const int z = int(key / (Key(g.nx) * Key(g.ny)));

    for (int dz = 0; dz <= 1; ++dz) {
      for (int dy = (dz == 0 ? 0 : -1); dy <= 1; ++dy) {
        for (int dx = (dz == 0 && dy == 0 ? 1 : -1); dx <= 1; ++dx) {
          const Key k = grid_key(wrap_cell(x + dx, g.nx),
                                 wrap_cell(y + dy, g.ny),
                                 wrap_cell(z + dz, g.nz), g);
          const int nb = find_cell(k, keys, lookup, mask);
          if (nb < 0) continue;

          if (!cell_pair_bbox_close(c, nb, keys, bmin_arr, bmax_arr, g)) continue;

          const int rB = cell_single_root[nb];
          if (rA >= 0 && rB >= 0 && root(parent, rA) == root(parent, rB)) continue;

          const int nb_begin = starts[nb], nb_end = starts[nb + 1];
          const int mB = nb_end - nb_begin;
          if (std::size_t(mA) * std::size_t(mB) > 1024ULL) {
            const int slot = atomicAdd(heavy_pair_count, 1);
            if (std::size_t(slot) < max_heavy) {
              heavy_pairs[slot] = int2{c, nb};
              continue;
            }
          }

          if (rA >= 0 && rB >= 0) {
            bool done = false;
            for (int i = begin; i < end && !done; ++i) {
              const double3 pi = pos_s[i];
              for (int j = nb_begin; j < nb_end; ++j) {
                if (linked_d3(pi, pos_s[j], g)) {
                  link_edge(parent, rA, rB);
                  done = true;
                  break;
                }
              }
            }
          } else {
            for (int i = begin; i < end; ++i) {
              const double3 pi = pos_s[i];
              for (int j = nb_begin; j < nb_end; ++j) {
                if (linked_d3(pi, pos_s[j], g)) {
                  link_edge(parent, rA >= 0 ? rA : i, rB >= 0 ? rB : j);
                  if (rB >= 0) break;
                }
              }
            }
          }
        }
      }
    }
  }
}

__global__ void connect_heavy(const double3* pos_s, const int* starts,
                              const int* cell_single_root, const int2* heavy_pairs,
                              int num_pairs, Geometry g, int* parent) {
  if (blockIdx.x >= num_pairs) return;
  const int2 pr = heavy_pairs[blockIdx.x];
  const int c = pr.x, nb = pr.y;
  const int beginA = starts[c], endA = starts[c + 1];
  const int beginB = starts[nb], endB = starts[nb + 1];
  const int rA = cell_single_root[c], rB = cell_single_root[nb];
  const int tid = threadIdx.x;

  __shared__ double3 s_tileB[threads];
  __shared__ int s_found;

  if (tid == 0) {
    s_found = (rA >= 0 && rB >= 0 && root(parent, rA) == root(parent, rB)) ? 1 : 0;
  }
  __syncthreads();
  if (s_found) return;

  if (rA >= 0 && rB >= 0) {
    for (int i = beginA; i < endA; ++i) {
      if (Atomic(s_found).load(cuda::memory_order_relaxed)) break;
      const double3 pi = pos_s[i];
      for (int j = beginB + tid; j < endB; j += blockDim.x) {
        if (Atomic(s_found).load(cuda::memory_order_relaxed)) break;
        if (linked_d3(pi, pos_s[j], g)) {
          if (atomicExch(&s_found, 1) == 0) {
            link_edge(parent, rA, rB);
          }
          break;
        }
      }
    }
    return;
  }

  for (int baseB = beginB; baseB < endB; baseB += threads) {
    const int tile_len = min(threads, endB - baseB);
    if (tid < tile_len) s_tileB[tid] = pos_s[baseB + tid];
    __syncthreads();

    for (int i = beginA + tid; i < endA; i += blockDim.x) {
      const double3 pi = pos_s[i];
      const int u = (rA >= 0) ? rA : i;
      for (int jj = 0; jj < tile_len; ++jj) {
        if (linked_d3(pi, s_tileB[jj], g)) {
          const int v = (rB >= 0) ? rB : (baseB + jj);
          link_edge(parent, u, v);
          if (rB >= 0) break;
        }
      }
    }
    __syncthreads();
  }
}

__global__ void finalize_sorted_roots(const int* parent, int* sorted_roots, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
    int p = k;
    while (p != parent[p]) p = parent[p];
    sorted_roots[k] = p;
  }
}

__global__ void scatter_raw_labels(const int* sorted_roots, const int* rows, int* labels, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
    labels[rows[k]] = rows[sorted_roots[k]];
  }
}

__global__ void count_root_sizes(const int* sorted_roots, int* root_counts, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
    atomicAdd(&root_counts[sorted_roots[k]], 1);
  }
}

__global__ void gather_root_sizes(const int* valid_roots, const int* root_counts,
                                  int* valid_sizes, int num_valid) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_valid; i += stride) {
    valid_sizes[i] = root_counts[valid_roots[i]];
  }
}

__global__ void fill_fof_sentinel(std::int64_t* root_to_ranked, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    root_to_ranked[i] = grav_fof_none;
  }
}

__global__ void assign_ranked_ids(const int* sorted_valid_roots, std::int64_t* root_to_ranked,
                                 int num_groups) {
  const int stride = gridDim.x * blockDim.x;
  for (int rank = blockIdx.x * blockDim.x + threadIdx.x; rank < num_groups; rank += stride) {
    root_to_ranked[sorted_valid_roots[rank]] = static_cast<std::int64_t>(rank) + 1LL;
  }
}

__global__ void scatter_ranked_labels(const int* sorted_roots, const int* rows,
                                     const std::uint32_t* dev_orig_idx,
                                     std::uint32_t gid_offset,
                                     const std::int64_t* root_to_ranked,
                                     std::int64_t* ranked_labels, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
    const std::uint32_t out_i = dev_orig_idx ? dev_orig_idx[rows[k]]
                                             : (gid_offset + static_cast<std::uint32_t>(rows[k]));
    ranked_labels[out_i] = root_to_ranked[sorted_roots[k]];
  }
}

__global__ void assign_vidx_kernel(const int* valid_roots, int* root_to_vidx, int num_valid) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_valid; i += stride) {
    root_to_vidx[valid_roots[i]] = i;
  }
}

__global__ void compute_valid_min_pid_kernel(const int* sorted_roots, const int* rows,
                                             const std::uint32_t* dev_orig_idx,
                                             std::uint32_t gid_offset,
                                             const int* root_to_vidx,
                                             std::uint32_t* valid_min_pid, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
    const int vi = root_to_vidx[sorted_roots[k]];
    if (vi >= 0) {
      const std::uint32_t pid = dev_orig_idx ? dev_orig_idx[rows[k]]
                                             : (gid_offset + static_cast<std::uint32_t>(rows[k]));
      atomicMin(&valid_min_pid[vi], pid);
    }
  }
}

__global__ void pack_valid_keys_kernel(const int* valid_sizes, const std::uint32_t* valid_min_pid,
                                       std::uint64_t* valid_keys, int num_valid) {
  const int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_valid; i += stride) {
    const std::uint64_t sz = static_cast<std::uint64_t>(static_cast<std::uint32_t>(valid_sizes[i]));
    const std::uint64_t inv_pid = static_cast<std::uint64_t>(0xFFFFFFFFU - valid_min_pid[i]);
    valid_keys[i] = (sz << 32) | inv_pid;
  }
}

__global__ void merge_slab_sorted_roots_u32_kernel(const int* sorted_roots, const int* rows,
                                                   const std::uint32_t* slab_gids,
                                                   std::uint32_t* global_parent, int n) {
  const int stride = gridDim.x * blockDim.x;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
    const int sr = sorted_roots[k];
    if (sr != k) {
      const std::uint32_t u = slab_gids[rows[k]];
      const std::uint32_t v = slab_gids[rows[sr]];
      if (u != v) link_edge_u32(global_parent, u, v);
    }
  }
}

__global__ void count_nonroot_edges_kernel(const int* sorted_roots, int n, int* edge_count) {
  int local = 0;
  const int stride = gridDim.x * blockDim.x;
  for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < n; k += stride) {
    if (sorted_roots[k] != k) ++local;
  }
  __shared__ int s_block_sum;
  if (threadIdx.x == 0) s_block_sum = 0;
  __syncthreads();
  if (local > 0) atomicAdd(&s_block_sum, local);
  __syncthreads();
  if (threadIdx.x == 0 && s_block_sum > 0) atomicAdd(edge_count, s_block_sum);
}

__global__ void collect_nonroot_edges_kernel(const int* sorted_roots, const int* rows,
                                             const std::uint32_t* dev_orig_idx,
                                             std::uint32_t gid_offset,
                                             int n, uint2* out_edges, int* write_cursor) {
  const std::uint64_t grid_stride = std::uint64_t(gridDim.x) * std::uint64_t(blockDim.x);
  const int lane = threadIdx.x & 31;
  for (std::uint64_t base = std::uint64_t(blockIdx.x) * std::uint64_t(blockDim.x);
       base < std::uint64_t(n); base += grid_stride) {
    const int k = static_cast<int>(base + std::uint64_t(threadIdx.x));
    int sr = k;
    bool keep = false;
    if (k < n) {
      sr = sorted_roots[k];
      keep = (sr != k);
    }
    const unsigned mask = __ballot_sync(0xFFFFFFFFU, keep);
    if (mask != 0U) {
      int w_base = 0;
      if (lane == 0) w_base = atomicAdd(write_cursor, __popc(mask));
      w_base = __shfl_sync(0xFFFFFFFFU, w_base, 0);
      if (keep) {
        const unsigned lower = (1U << lane) - 1U;
        const int pos = w_base + __popc(mask & lower);
        const std::uint32_t u = dev_orig_idx ? dev_orig_idx[rows[k]]
                                             : (gid_offset + static_cast<std::uint32_t>(rows[k]));
        const std::uint32_t v = dev_orig_idx ? dev_orig_idx[rows[sr]]
                                             : (gid_offset + static_cast<std::uint32_t>(rows[sr]));
        out_edges[pos] = uint2{u, v};
      }
    }
  }
}

struct GeqCut {
  int min_size;
  __host__ __device__ bool operator()(int cnt) const { return cnt >= min_size; }
};

struct KeyToSizeI64 {
  __host__ __device__ std::int64_t operator()(std::uint64_t k) const {
    return static_cast<std::int64_t>(k >> 32);
  }
};

inline int grid_for(int n) {
  return min(max_grid_blocks, max(1, (n + threads - 1) / threads));
}

inline int grid_for_u64(std::uint64_t n) {
  const std::uint64_t blocks = (n + std::uint64_t(threads) - 1ULL) / std::uint64_t(threads);
  return static_cast<int>(std::min<std::uint64_t>(std::uint64_t(max_grid_blocks), std::max<std::uint64_t>(1ULL, blocks)));
}

enum class OutMode {
  RawHost,
  Ranked,
  DeviceMerge,
};

int run_fof_core_impl(const double* pos_ptr, bool pos_is_device, bool allow_in_place_wrap,
                      int count, double lx, double ly, double lz, double link,
                      OutMode mode, int min_group_size,
                      const std::uint32_t* dev_orig_idx, std::uint32_t gid_offset,
                      int* raw_labels_host,
                      std::int64_t* ranked_labels_dev, std::int64_t* ranked_labels_host,
                      std::uint64_t* out_num_groups,
                      std::uint64_t* out_grouped_particles,
                      std::uint64_t* out_largest_group,
                      std::uint32_t* global_parent_dev, const std::uint32_t* slab_gids_dev,
                      std::vector<uint2>* out_edges_host,
                      double* elapsed, std::uint64_t* info) {
  const char* step = "initialization";
  try {
    if (!pos_ptr || count <= 0 || count > (1 << 30) || link <= 0.0 ||
        lx <= 0.0 || ly <= 0.0 || lz <= 0.0) {
      throw GpuError{cudaErrorInvalidValue};
    }
    if (mode == OutMode::Ranked && min_group_size < 2) {
      throw GpuError{cudaErrorInvalidValue};
    }
    if (mode == OutMode::Ranked && !ranked_labels_dev && !ranked_labels_host) {
      throw GpuError{cudaErrorInvalidValue};
    }
    if (mode == OutMode::RawHost && !raw_labels_host) {
      throw GpuError{cudaErrorInvalidValue};
    }
    if (mode == OutMode::DeviceMerge &&
        !((global_parent_dev && slab_gids_dev) || out_edges_host)) {
      throw GpuError{cudaErrorInvalidValue};
    }

    const int nx = static_cast<int>(std::floor(lx / link));
    const int ny = static_cast<int>(std::floor(ly / link));
    const int nz = static_cast<int>(std::floor(lz / link));
    if (nx < 3 || ny < 3 || nz < 3) {
      throw GpuError{cudaErrorInvalidValue};
    }
    const double hx = lx / double(nx);
    const double hy = ly / double(ny);
    const double hz = lz / double(nz);
    const double limit = link * link;
    const bool use_octants = (0.25 * (hx * hx + hy * hy + hz * hz) < 0.999999 * limit);
    const Geometry g{lx, ly, lz, hx, hy, hz, nx, ny, nz, limit, use_octants};

    if (elapsed) {
      for (int i = 0; i < 9; ++i) elapsed[i] = 0.0;
    }

    check(cudaFree(nullptr));
    std::size_t device_free = 0, device_total = 0;
    check(cudaMemGetInfo(&device_free, &device_total));
    if (info) {
      info[1] = device_free;
      info[2] = device_total;
    }

    Timer timer;
    double dummy_elapsed[9]{};
    double* el = elapsed ? elapsed : dummy_elapsed;

    step = "upload";
    Buffer<double3> pos_raw_buf;
    const double3* raw_in = nullptr;
    double3* pos_wrapped = nullptr;
    if (pos_is_device && allow_in_place_wrap) {
      pos_wrapped = const_cast<double3*>(reinterpret_cast<const double3*>(pos_ptr));
      raw_in = pos_wrapped;
      el[0] = 0.0;
    } else if (pos_is_device) {
      pos_raw_buf = Buffer<double3>(static_cast<std::size_t>(count));
      raw_in = reinterpret_cast<const double3*>(pos_ptr);
      pos_wrapped = pos_raw_buf.ptr;
      el[0] = 0.0;
    } else {
      pos_raw_buf = Buffer<double3>(static_cast<std::size_t>(count));
      timer.start();
      check(cudaMemcpy(pos_raw_buf.ptr, pos_ptr, static_cast<std::size_t>(count) * sizeof(double3),
                       cudaMemcpyHostToDevice));
      timer.stop(el + 0);
      raw_in = pos_raw_buf.ptr;
      pos_wrapped = pos_raw_buf.ptr;
    }

    step = "keying";
    Buffer<Key> key_a(count), key_b(count);
    Buffer<int> row_a(count), row_b(count), invalid(1);
    timer.start();
    check(cudaMemset(invalid.ptr, 0, sizeof(int)));
    grid_index_raw<<<grid_for(count), threads>>>(raw_in, pos_wrapped, key_a.ptr,
                                                 row_a.ptr, count, g, invalid.ptr);
    int bad = 0;
    check(cudaMemcpy(&bad, invalid.ptr, sizeof(int), cudaMemcpyDeviceToHost));
    timer.stop(el + 1);
    if (bad) throw GpuError{cudaErrorInvalidValue};
    invalid.release();

    step = "radix sort";
    cub::DoubleBuffer<Key> keys(key_a.ptr, key_b.ptr);
    cub::DoubleBuffer<int> rows(row_a.ptr, row_b.ptr);
    std::size_t sort_bytes = 0;
    Key maxkey = Key(g.nx) * Key(g.ny) * Key(g.nz) - 1ULL;
    int bits = 0;
    while (maxkey) {
      ++bits;
      maxkey >>= 1;
    }
    bits = std::max(bits, 1);
    check(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, keys, rows, count, 0, bits));
    if (info) info[3] = sort_bytes;
    {
      Buffer<unsigned char> work(sort_bytes);
      timer.start();
      check(cub::DeviceRadixSort::SortPairs(work.ptr, sort_bytes, keys, rows, count, 0, bits));
      timer.stop(el + 2);
    }
    key_a.keep_only(keys.Current());
    key_b.keep_only(keys.Current());
    row_a.keep_only(rows.Current());
    row_b.keep_only(rows.Current());

    step = "occupied grid";
    Buffer<Key> occupied(count);
    Buffer<int> starts(static_cast<std::size_t>(count) + 1), runs(1);
    std::size_t rle_bytes = 0;
    check(cub::DeviceRunLengthEncode::Encode(
        nullptr, rle_bytes, keys.Current(), occupied.ptr, starts.ptr, runs.ptr, count));
    if (info) info[4] = rle_bytes;
    int C = 0;
    {
      Buffer<unsigned char> work(rle_bytes);
      timer.start();
      check(cub::DeviceRunLengthEncode::Encode(
          work.ptr, rle_bytes, keys.Current(), occupied.ptr, starts.ptr, runs.ptr, count));
      check(cudaMemcpy(&C, runs.ptr, sizeof(int), cudaMemcpyDeviceToHost));
      timer.stop(el + 3);
    }
    key_a.release();
    key_b.release();
    runs.release();
    if (C <= 0 || C > count) throw GpuError{cudaErrorInvalidValue};
    if (info) info[0] = static_cast<std::uint64_t>(C);

    step = "cell offsets";
    std::size_t scan_bytes = 0;
    check(cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, starts.ptr, starts.ptr, C));
    {
      Buffer<unsigned char> work(scan_bytes);
      timer.start();
      check(cub::DeviceScan::ExclusiveSum(work.ptr, scan_bytes, starts.ptr, starts.ptr, C));
      check(cudaMemcpy(starts.ptr + C, &count, sizeof(int), cudaMemcpyHostToDevice));
      timer.stop(el + 4);
    }

    // Gather sorted positions and release unsorted pos_raw_buf immediately
    Buffer<double3> pos_s(static_cast<std::size_t>(count));
    Buffer<int> parent(static_cast<std::size_t>(count));
    gather_sorted_and_init_parent<<<grid_for(count), threads>>>(
        pos_wrapped, rows.Current(), pos_s.ptr, parent.ptr, count);
    check(cudaDeviceSynchronize());
    pos_raw_buf.release();

    Key slots = 1;
    while (slots < static_cast<Key>(C) * 2ULL) slots *= 2ULL;
    if (info) info[5] = slots;

    step = "hash and intra_cell";
    Buffer<int> lookup(slots);
    Buffer<double3> bmin_arr(static_cast<std::size_t>(C));
    Buffer<double3> bmax_arr(static_cast<std::size_t>(C));
    Buffer<int> cell_single_root(static_cast<std::size_t>(C));
    Buffer<int> heavy_cells(static_cast<std::size_t>(C));
    Buffer<int> heavy_cell_count(1);

    std::size_t latest_free = 0, latest_total = 0;
    check(cudaMemGetInfo(&latest_free, &latest_total));
    if (info) info[6] = latest_free;

    timer.start();
    check(cudaMemset(lookup.ptr, 0, slots * sizeof(int)));
    check(cudaMemset(heavy_cell_count.ptr, 0, sizeof(int)));
    build_lookup<<<grid_for(C), threads>>>(occupied.ptr, C, lookup.ptr, slots - 1ULL);
    intra_cell_light<<<grid_for(C), threads>>>(
        pos_s.ptr, occupied.ptr, starts.ptr, C, g, parent.ptr,
        bmin_arr.ptr, bmax_arr.ptr, cell_single_root.ptr,
        heavy_cells.ptr, heavy_cell_count.ptr);
    int n_heavy = 0;
    check(cudaMemcpy(&n_heavy, heavy_cell_count.ptr, sizeof(int), cudaMemcpyDeviceToHost));
    if (n_heavy > 0) {
      intra_cell_heavy<<<n_heavy, threads>>>(
          pos_s.ptr, occupied.ptr, starts.ptr, heavy_cells.ptr, n_heavy, g,
          parent.ptr, bmin_arr.ptr, bmax_arr.ptr, cell_single_root.ptr);
    }
    timer.stop(el + 5);
    heavy_cells.release();
    heavy_cell_count.release();

    step = "neighbors and union";
    const std::size_t max_heavy_pairs =
        std::max<std::size_t>(1ULL << 20, std::min<std::size_t>(static_cast<std::size_t>(C) * 2ULL, 32ULL << 20));
    Buffer<int2> heavy_pairs(max_heavy_pairs);
    Buffer<int> heavy_pair_count(1);

    timer.start();
    check(cudaMemset(heavy_pair_count.ptr, 0, sizeof(int)));
    connect_light<<<grid_for(C), threads>>>(
        pos_s.ptr, occupied.ptr, starts.ptr, lookup.ptr, slots - 1ULL, C, g,
        bmin_arr.ptr, bmax_arr.ptr, cell_single_root.ptr, parent.ptr,
        heavy_pairs.ptr, max_heavy_pairs, heavy_pair_count.ptr);
    int n_heavy_pairs = 0;
    check(cudaMemcpy(&n_heavy_pairs, heavy_pair_count.ptr, sizeof(int), cudaMemcpyDeviceToHost));
    const int n_dispatched = static_cast<int>(std::min<std::size_t>(std::size_t(n_heavy_pairs), max_heavy_pairs));
    if (n_dispatched > 0) {
      connect_heavy<<<n_dispatched, threads>>>(
          pos_s.ptr, starts.ptr, cell_single_root.ptr, heavy_pairs.ptr, n_dispatched, g, parent.ptr);
    }
    timer.stop(el + 6);

    heavy_pairs.release();
    heavy_pair_count.release();
    bmin_arr.release();
    bmax_arr.release();
    cell_single_root.release();
    lookup.release();
    occupied.release();
    starts.release();
    pos_s.release();

    step = "labels";
    Buffer<int> sorted_roots(static_cast<std::size_t>(count));
    timer.start();
    finalize_sorted_roots<<<grid_for(count), threads>>>(parent.ptr, sorted_roots.ptr, count);
    timer.stop(el + 7);
    parent.release();

    if (mode == OutMode::DeviceMerge) {
      if (global_parent_dev && slab_gids_dev) {
        step = "device merge";
        timer.start();
        merge_slab_sorted_roots_u32_kernel<<<grid_for(count), threads>>>(
            sorted_roots.ptr, rows.Current(), slab_gids_dev, global_parent_dev, count);
        timer.stop(el + 8);
      } else if (out_edges_host) {
        step = "collect edges";
        timer.start();
        Buffer<int> edge_cnt_dev(1);
        check(cudaMemset(edge_cnt_dev.ptr, 0, sizeof(int)));
        count_nonroot_edges_kernel<<<grid_for(count), threads>>>(
            sorted_roots.ptr, count, edge_cnt_dev.ptr);
        int n_edges = 0;
        check(cudaMemcpy(&n_edges, edge_cnt_dev.ptr, sizeof(int), cudaMemcpyDeviceToHost));
        out_edges_host->resize(static_cast<std::size_t>(n_edges));
        if (n_edges > 0) {
          Buffer<uint2> edges_dev(static_cast<std::size_t>(n_edges));
          check(cudaMemset(edge_cnt_dev.ptr, 0, sizeof(int)));
          collect_nonroot_edges_kernel<<<grid_for(count), threads>>>(
              sorted_roots.ptr, rows.Current(), dev_orig_idx, gid_offset,
              count, edges_dev.ptr, edge_cnt_dev.ptr);
          check(cudaMemcpy(out_edges_host->data(), edges_dev.ptr,
                           static_cast<std::size_t>(n_edges) * sizeof(uint2),
                           cudaMemcpyDeviceToHost));
        }
        timer.stop(el + 8);
      }
      return 0;
    }

    if (mode == OutMode::RawHost) {
      Buffer<int> labels(static_cast<std::size_t>(count));
      scatter_raw_labels<<<grid_for(count), threads>>>(sorted_roots.ptr, rows.Current(), labels.ptr, count);
      check(cudaDeviceSynchronize());
      row_a.release();
      row_b.release();
      sorted_roots.release();

      step = "download";
      timer.start();
      check(cudaMemcpy(raw_labels_host, labels.ptr, static_cast<std::size_t>(count) * sizeof(int),
                       cudaMemcpyDeviceToHost));
      timer.stop(el + 8);
      return 0;
    }

    step = "ranked relabel";
    Buffer<int> root_counts(static_cast<std::size_t>(count));
    check(cudaMemset(root_counts.ptr, 0, static_cast<std::size_t>(count) * sizeof(int)));
    count_root_sizes<<<grid_for(count), threads>>>(sorted_roots.ptr, root_counts.ptr, count);

    Buffer<int> valid_roots(static_cast<std::size_t>(count));
    Buffer<int> num_valid_dev(1);
    cub::CountingInputIterator<int> seq(0);
    cub::TransformInputIterator<bool, GeqCut, int*> pred(root_counts.ptr, GeqCut{min_group_size});
    std::size_t sel_bytes = 0;
    check(cub::DeviceSelect::Flagged(
        nullptr, sel_bytes, seq, pred, valid_roots.ptr, num_valid_dev.ptr, count));
    {
      Buffer<unsigned char> work(sel_bytes);
      check(cub::DeviceSelect::Flagged(
          work.ptr, sel_bytes, seq, pred, valid_roots.ptr, num_valid_dev.ptr, count));
    }
    int num_groups = 0;
    check(cudaMemcpy(&num_groups, num_valid_dev.ptr, sizeof(int), cudaMemcpyDeviceToHost));
    num_valid_dev.release();

    std::uint64_t grouped_total = 0;
    std::uint64_t largest_sz = 0;
    Buffer<std::int64_t> root_to_ranked(static_cast<std::size_t>(count));
    fill_fof_sentinel<<<grid_for(count), threads>>>(root_to_ranked.ptr, count);

    if (num_groups > 0) {
      Buffer<int> valid_sizes(static_cast<std::size_t>(num_groups));
      gather_root_sizes<<<grid_for(num_groups), threads>>>(
          valid_roots.ptr, root_counts.ptr, valid_sizes.ptr, num_groups);

      // Canonical deterministic tie-break: (size DESC, min global particle id ASC)
      check(cudaMemset(root_counts.ptr, 0xFF, static_cast<std::size_t>(count) * sizeof(int)));
      assign_vidx_kernel<<<grid_for(num_groups), threads>>>(
          valid_roots.ptr, root_counts.ptr, num_groups);
      Buffer<std::uint32_t> valid_min_pid(static_cast<std::size_t>(num_groups));
      check(cudaMemset(valid_min_pid.ptr, 0xFF, static_cast<std::size_t>(num_groups) * sizeof(std::uint32_t)));
      compute_valid_min_pid_kernel<<<grid_for(count), threads>>>(
          sorted_roots.ptr, rows.Current(), dev_orig_idx, gid_offset,
          root_counts.ptr, valid_min_pid.ptr, count);
      root_counts.release();

      Buffer<std::uint64_t> valid_keys(static_cast<std::size_t>(num_groups));
      pack_valid_keys_kernel<<<grid_for(num_groups), threads>>>(
          valid_sizes.ptr, valid_min_pid.ptr, valid_keys.ptr, num_groups);
      valid_sizes.release();
      valid_min_pid.release();

      Buffer<std::uint64_t> sorted_keys(static_cast<std::size_t>(num_groups));
      Buffer<int> sorted_valid_roots(static_cast<std::size_t>(num_groups));
      std::size_t gsort_bytes = 0;
      check(cub::DeviceRadixSort::SortPairsDescending(
          nullptr, gsort_bytes, valid_keys.ptr, sorted_keys.ptr,
          valid_roots.ptr, sorted_valid_roots.ptr, num_groups));
      {
        Buffer<unsigned char> work(gsort_bytes);
        check(cub::DeviceRadixSort::SortPairsDescending(
            work.ptr, gsort_bytes, valid_keys.ptr, sorted_keys.ptr,
            valid_roots.ptr, sorted_valid_roots.ptr, num_groups));
      }
      valid_keys.release();
      valid_roots.release();

      std::uint64_t top_key = 0;
      check(cudaMemcpy(&top_key, sorted_keys.ptr, sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
      largest_sz = top_key >> 32;

      Buffer<std::int64_t> sum_dev(1);
      cub::TransformInputIterator<std::int64_t, KeyToSizeI64, std::uint64_t*> cast_iter(
          sorted_keys.ptr, KeyToSizeI64{});
      std::size_t red_bytes = 0;
      check(cub::DeviceReduce::Sum(nullptr, red_bytes, cast_iter, sum_dev.ptr, num_groups));
      {
        Buffer<unsigned char> work(red_bytes);
        check(cub::DeviceReduce::Sum(work.ptr, red_bytes, cast_iter, sum_dev.ptr, num_groups));
      }
      std::int64_t grouped_i64 = 0;
      check(cudaMemcpy(&grouped_i64, sum_dev.ptr, sizeof(std::int64_t), cudaMemcpyDeviceToHost));
      grouped_total = static_cast<std::uint64_t>(grouped_i64);

      assign_ranked_ids<<<grid_for(num_groups), threads>>>(
          sorted_valid_roots.ptr, root_to_ranked.ptr, num_groups);
    } else {
      root_counts.release();
      valid_roots.release();
    }

    Buffer<std::int64_t> ranked_out_buf;
    std::int64_t* out_dev = ranked_labels_dev;
    if (!out_dev) {
      ranked_out_buf = Buffer<std::int64_t>(static_cast<std::size_t>(count));
      out_dev = ranked_out_buf.ptr;
    }
    scatter_ranked_labels<<<grid_for(count), threads>>>(
        sorted_roots.ptr, rows.Current(), dev_orig_idx, gid_offset,
        root_to_ranked.ptr, out_dev, count);
    check(cudaDeviceSynchronize());
    root_to_ranked.release();
    sorted_roots.release();
    row_a.release();
    row_b.release();

    step = "download";
    if (ranked_labels_host) {
      timer.start();
      check(cudaMemcpy(ranked_labels_host, out_dev,
                       static_cast<std::size_t>(count) * sizeof(std::int64_t),
                       cudaMemcpyDeviceToHost));
      timer.stop(el + 8);
    } else {
      el[8] = 0.0;
    }

    if (out_num_groups) *out_num_groups = static_cast<std::uint64_t>(num_groups);
    if (out_grouped_particles) *out_grouped_particles = grouped_total;
    if (out_largest_group) *out_largest_group = largest_sz;
    return 0;
  } catch (const GpuError& error) {
    std::fprintf(stderr, "GRAV_FOF_CUDA_ERROR %s %s\n", step, cudaGetErrorString(error.status));
    return int(error.status);
  }
}

int run_fof_core(const double* host, int count,
                 double lx, double ly, double lz, double link,
                 bool ranked_mode, int min_group_size,
                 int* raw_labels_host, std::int64_t* ranked_labels_host,
                 std::uint64_t* out_num_groups,
                 std::uint64_t* out_grouped_particles,
                 std::uint64_t* out_largest_group,
                 double* elapsed, std::uint64_t* info) {
  return run_fof_core_impl(
      host, false, false, count, lx, ly, lz, link,
      ranked_mode ? OutMode::Ranked : OutMode::RawHost, min_group_size,
      nullptr, 0U, raw_labels_host, nullptr, ranked_labels_host,
      out_num_groups, out_grouped_particles, out_largest_group,
      nullptr, nullptr, nullptr, elapsed, info);
}

// ============================================================================
// GPU Global Cross-Slab / Multi-GPU Parent Relabeling & Slab Kernels
// ============================================================================

__global__ void init_iota_u32_kernel(std::uint32_t* parent, std::uint64_t n) {
  const std::uint64_t stride = std::uint64_t(gridDim.x) * std::uint64_t(blockDim.x);
  for (std::uint64_t i = std::uint64_t(blockIdx.x) * std::uint64_t(blockDim.x) + threadIdx.x;
       i < n; i += stride) {
    parent[i] = static_cast<std::uint32_t>(i);
  }
}

__global__ void compress_parent_u32_kernel(std::uint32_t* parent, std::uint64_t n) {
  const std::uint64_t stride = std::uint64_t(gridDim.x) * std::uint64_t(blockDim.x);
  for (std::uint64_t i = std::uint64_t(blockIdx.x) * std::uint64_t(blockDim.x) + threadIdx.x;
       i < n; i += stride) {
    std::uint32_t r = static_cast<std::uint32_t>(i);
    for (;;) {
      const std::uint32_t p = AtomicU32(parent[r]).load(cuda::memory_order_relaxed);
      if (p == r) break;
      r = p;
    }
    if (parent[i] != r) AtomicU32(parent[i]).store(r, cuda::memory_order_relaxed);
  }
}

__global__ void count_root_sizes_u32_kernel(const std::uint32_t* parent,
                                            std::uint32_t* root_counts, std::uint64_t n) {
  const std::uint64_t stride = std::uint64_t(gridDim.x) * std::uint64_t(blockDim.x);
  for (std::uint64_t i = std::uint64_t(blockIdx.x) * std::uint64_t(blockDim.x) + threadIdx.x;
       i < n; i += stride) {
    atomicAdd(&root_counts[parent[i]], 1U);
  }
}

__global__ void count_valid_roots_u32_kernel(const std::uint32_t* parent,
                                             const std::uint32_t* root_counts,
                                             std::uint64_t n, int min_group_size,
                                             unsigned long long* out_num_valid) {
  unsigned long long local = 0ULL;
  const std::uint64_t stride = std::uint64_t(gridDim.x) * std::uint64_t(blockDim.x);
  const std::uint32_t min_sz = static_cast<std::uint32_t>(min_group_size);
  for (std::uint64_t i = std::uint64_t(blockIdx.x) * std::uint64_t(blockDim.x) + threadIdx.x;
       i < n; i += stride) {
    if (parent[i] == static_cast<std::uint32_t>(i) && root_counts[i] >= min_sz) {
      ++local;
    }
  }
  __shared__ unsigned long long s_block_sum;
  if (threadIdx.x == 0) s_block_sum = 0ULL;
  __syncthreads();
  if (local > 0ULL) atomicAdd(&s_block_sum, local);
  __syncthreads();
  if (threadIdx.x == 0 && s_block_sum > 0ULL) atomicAdd(out_num_valid, s_block_sum);
}

__global__ void collect_valid_roots_u32_kernel(const std::uint32_t* parent,
                                               const std::uint32_t* root_counts,
                                               std::uint64_t n, int min_group_size,
                                               std::uint64_t* valid_keys,
                                               std::uint32_t* valid_roots,
                                               unsigned long long* write_cursor) {
  const std::uint64_t grid_stride = std::uint64_t(gridDim.x) * std::uint64_t(blockDim.x);
  const std::uint32_t min_sz = static_cast<std::uint32_t>(min_group_size);
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  __shared__ unsigned s_warp_cnt[8];
  __shared__ unsigned long long s_block_base;

  for (std::uint64_t base = std::uint64_t(blockIdx.x) * std::uint64_t(blockDim.x);
       base < n; base += grid_stride) {
    const std::uint64_t i = base + std::uint64_t(threadIdx.x);
    std::uint32_t sz = 0U;
    bool keep = false;
    if (i < n && parent[i] == static_cast<std::uint32_t>(i)) {
      sz = root_counts[i];
      keep = (sz >= min_sz);
    }
    const unsigned mask = __ballot_sync(0xFFFFFFFFU, keep);
    if (lane == 0) s_warp_cnt[warp_id] = static_cast<unsigned>(__popc(mask));
    __syncthreads();

    unsigned warp_off = 0U;
    if (threadIdx.x == 0) {
      unsigned tot = 0U;
      for (int w = 0; w < 8; ++w) tot += s_warp_cnt[w];
      s_block_base = (tot > 0U) ? atomicAdd(write_cursor, static_cast<unsigned long long>(tot)) : 0ULL;
    }
    __syncthreads();

    if (keep) {
      for (int w = 0; w < warp_id; ++w) warp_off += s_warp_cnt[w];
      const unsigned lower = (1U << lane) - 1U;
      const std::uint64_t pos = s_block_base + std::uint64_t(warp_off) + std::uint64_t(__popc(mask & lower));
      valid_keys[pos] = (static_cast<std::uint64_t>(sz) << 32) |
                        static_cast<std::uint64_t>(0xFFFFFFFFU - static_cast<std::uint32_t>(i));
      valid_roots[pos] = static_cast<std::uint32_t>(i);
    }
    __syncthreads();
  }
}

__global__ void fill_fof_sentinel_i32_kernel(std::int32_t* root_to_ranked, std::uint64_t n) {
  const std::uint64_t stride = std::uint64_t(gridDim.x) * std::uint64_t(blockDim.x);
  const std::int32_t none_val = static_cast<std::int32_t>(grav_fof_none);
  for (std::uint64_t i = std::uint64_t(blockIdx.x) * std::uint64_t(blockDim.x) + threadIdx.x;
       i < n; i += stride) {
    root_to_ranked[i] = none_val;
  }
}

__global__ void assign_ranked_ids_i32_kernel(const std::uint32_t* sorted_valid_roots,
                                            std::int32_t* root_to_ranked, int num_groups) {
  const int stride = gridDim.x * blockDim.x;
  for (int rank = blockIdx.x * blockDim.x + threadIdx.x; rank < num_groups; rank += stride) {
    root_to_ranked[sorted_valid_roots[rank]] = rank + 1;
  }
}

__global__ void scatter_ranked_chunk_u32_kernel(const std::uint32_t* parent,
                                               const std::int32_t* root_to_ranked,
                                               std::int64_t* chunk_out,
                                               std::uint64_t base, std::uint64_t len) {
  const std::uint64_t stride = std::uint64_t(gridDim.x) * std::uint64_t(blockDim.x);
  for (std::uint64_t j = std::uint64_t(blockIdx.x) * std::uint64_t(blockDim.x) + threadIdx.x;
       j < len; j += stride) {
    chunk_out[j] = static_cast<std::int64_t>(root_to_ranked[parent[base + j]]);
  }
}

void finalize_and_relabel_ranked_u32_gpu(std::uint32_t* d_parent, std::uint64_t count,
                                        int min_group_size,
                                        std::int64_t* group_ids_host,
                                        std::uint64_t* out_num_groups,
                                        std::uint64_t* out_grouped_particles,
                                        std::uint64_t* out_largest_group,
                                        double* elapsed) {
  Timer timer;
  compress_parent_u32_kernel<<<grid_for_u64(count), threads>>>(d_parent, count);
  check(cudaGetLastError());

  Buffer<std::uint32_t> d_root_counts(static_cast<std::size_t>(count));
  check(cudaMemset(d_root_counts.ptr, 0, static_cast<std::size_t>(count) * sizeof(std::uint32_t)));
  count_root_sizes_u32_kernel<<<grid_for_u64(count), threads>>>(d_parent, d_root_counts.ptr, count);
  check(cudaGetLastError());

  Buffer<unsigned long long> d_num_valid(1);
  check(cudaMemset(d_num_valid.ptr, 0, sizeof(unsigned long long)));
  count_valid_roots_u32_kernel<<<grid_for_u64(count), threads>>>(
      d_parent, d_root_counts.ptr, count, min_group_size, d_num_valid.ptr);
  unsigned long long h_num_valid = 0ULL;
  check(cudaMemcpy(&h_num_valid, d_num_valid.ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  if (h_num_valid > std::uint64_t(std::numeric_limits<int>::max())) {
    throw GpuError{cudaErrorInvalidValue};
  }
  const int num_groups = static_cast<int>(h_num_valid);

  std::uint64_t grouped_total = 0ULL;
  std::uint64_t largest_sz = 0ULL;
  Buffer<std::uint32_t> d_sorted_roots;

  if (num_groups > 0) {
    Buffer<std::uint64_t> d_valid_keys(static_cast<std::size_t>(num_groups));
    Buffer<std::uint32_t> d_valid_roots(static_cast<std::size_t>(num_groups));
    check(cudaMemset(d_num_valid.ptr, 0, sizeof(unsigned long long)));
    collect_valid_roots_u32_kernel<<<grid_for_u64(count), threads>>>(
        d_parent, d_root_counts.ptr, count, min_group_size,
        d_valid_keys.ptr, d_valid_roots.ptr, d_num_valid.ptr);
    check(cudaGetLastError());
    d_num_valid.release();

    Buffer<std::uint64_t> d_sorted_keys(static_cast<std::size_t>(num_groups));
    d_sorted_roots = Buffer<std::uint32_t>(static_cast<std::size_t>(num_groups));
    std::size_t sort_bytes = 0;
    check(cub::DeviceRadixSort::SortPairsDescending(
        nullptr, sort_bytes, d_valid_keys.ptr, d_sorted_keys.ptr,
        d_valid_roots.ptr, d_sorted_roots.ptr, num_groups));
    {
      Buffer<unsigned char> work(sort_bytes);
      check(cub::DeviceRadixSort::SortPairsDescending(
          work.ptr, sort_bytes, d_valid_keys.ptr, d_sorted_keys.ptr,
          d_valid_roots.ptr, d_sorted_roots.ptr, num_groups));
    }
    d_valid_keys.release();
    d_valid_roots.release();

    std::uint64_t top_key = 0ULL;
    check(cudaMemcpy(&top_key, d_sorted_keys.ptr, sizeof(std::uint64_t), cudaMemcpyDeviceToHost));
    largest_sz = top_key >> 32;

    Buffer<std::int64_t> d_sum(1);
    cub::TransformInputIterator<std::int64_t, KeyToSizeI64, std::uint64_t*> cast_iter(
        d_sorted_keys.ptr, KeyToSizeI64{});
    std::size_t red_bytes = 0;
    check(cub::DeviceReduce::Sum(nullptr, red_bytes, cast_iter, d_sum.ptr, num_groups));
    {
      Buffer<unsigned char> work(red_bytes);
      check(cub::DeviceReduce::Sum(work.ptr, red_bytes, cast_iter, d_sum.ptr, num_groups));
    }
    std::int64_t grouped_i64 = 0;
    check(cudaMemcpy(&grouped_i64, d_sum.ptr, sizeof(std::int64_t), cudaMemcpyDeviceToHost));
    grouped_total = static_cast<std::uint64_t>(grouped_i64);
  } else {
    d_num_valid.release();
  }

  // Reuse d_root_counts storage (4B/entry) as int32_t root -> SWIFT group ID table
  std::int32_t* d_root_to_ranked = reinterpret_cast<std::int32_t*>(d_root_counts.ptr);
  fill_fof_sentinel_i32_kernel<<<grid_for_u64(count), threads>>>(d_root_to_ranked, count);
  if (num_groups > 0) {
    assign_ranked_ids_i32_kernel<<<grid_for(num_groups), threads>>>(
        d_sorted_roots.ptr, d_root_to_ranked, num_groups);
    d_sorted_roots.release();
  }
  check(cudaGetLastError());

  const std::uint64_t chunk_len = std::min<std::uint64_t>(count, 16ULL << 20);
  Buffer<std::int64_t> d_chunk_out(static_cast<std::size_t>(chunk_len));
  double dl_acc = 0.0;
  for (std::uint64_t base = 0; base < count; base += chunk_len) {
    const std::uint64_t cur = std::min<std::uint64_t>(chunk_len, count - base);
    scatter_ranked_chunk_u32_kernel<<<grid_for_u64(cur), threads>>>(
        d_parent, d_root_to_ranked, d_chunk_out.ptr, base, cur);
    double dt = 0.0;
    timer.start();
    check(cudaMemcpy(group_ids_host + base, d_chunk_out.ptr,
                     static_cast<std::size_t>(cur) * sizeof(std::int64_t),
                     cudaMemcpyDeviceToHost));
    timer.stop(&dt);
    dl_acc += dt;
  }
  if (elapsed) elapsed[8] += dl_acc;

  if (out_num_groups) *out_num_groups = static_cast<std::uint64_t>(num_groups);
  if (out_grouped_particles) *out_grouped_particles = grouped_total;
  if (out_largest_group) *out_largest_group = largest_sz;
}

__global__ void slab_bucket_count_kernel(const double3* pos, std::uint64_t count,
                                         double lx, int nx, int S,
                                         const int* cell_to_slab, const int* cell_first_col_slab,
                                         unsigned long long* prim_cnt, unsigned long long* halo_cnt,
                                         int* invalid) {
  __shared__ unsigned int s_prim[256];
  __shared__ unsigned int s_halo[256];
  const bool use_sh = (S <= 256);
  if (use_sh) {
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
      s_prim[s] = 0U;
      s_halo[s] = 0U;
    }
    __syncthreads();
  }
  const std::uint64_t stride = std::uint64_t(gridDim.x) * std::uint64_t(blockDim.x);
  for (std::uint64_t i = std::uint64_t(blockIdx.x) * std::uint64_t(blockDim.x) + threadIdx.x;
       i < count; i += stride) {
    const double3 p = pos[i];
    if (!(isfinite(p.x) && isfinite(p.y) && isfinite(p.z))) {
      atomicExch(invalid, 1);
      continue;
    }
    const double wx = in_box(p.x, lx);
    const int cx = axis_key(wx, lx, nx);
    const int ps = cell_to_slab[cx];
    const int hs = cell_first_col_slab[cx];
    if (use_sh) {
      atomicAdd(&s_prim[ps], 1U);
      if (hs >= 0) atomicAdd(&s_halo[hs], 1U);
    } else {
      atomicAdd(&prim_cnt[ps], 1ULL);
      if (hs >= 0) atomicAdd(&halo_cnt[hs], 1ULL);
    }
  }
  if (use_sh) {
    __syncthreads();
    for (int s = threadIdx.x; s < S; s += blockDim.x) {
      if (s_prim[s] > 0U) atomicAdd(&prim_cnt[s], static_cast<unsigned long long>(s_prim[s]));
      if (s_halo[s] > 0U) atomicAdd(&halo_cnt[s], static_cast<unsigned long long>(s_halo[s]));
    }
  }
}

__global__ void slab_bucket_scatter_kernel(const double3* pos, std::uint64_t count,
                                           double lx, int nx,
                                           const int* cell_to_slab, const int* cell_first_col_slab,
                                           unsigned long long* prim_cursor, unsigned long long* halo_cursor,
                                           std::uint32_t* prim_gids, std::uint32_t* halo_gids) {
  const std::uint64_t stride = std::uint64_t(gridDim.x) * std::uint64_t(blockDim.x);
  for (std::uint64_t i = std::uint64_t(blockIdx.x) * std::uint64_t(blockDim.x) + threadIdx.x;
       i < count; i += stride) {
    const double3 p = pos[i];
    const double wx = in_box(p.x, lx);
    const int cx = axis_key(wx, lx, nx);
    const int ps = cell_to_slab[cx];
    const unsigned long long p_slot = atomicAdd(&prim_cursor[ps], 1ULL);
    prim_gids[p_slot] = static_cast<std::uint32_t>(i);
    const int hs = cell_first_col_slab[cx];
    if (hs >= 0) {
      const unsigned long long h_slot = atomicAdd(&halo_cursor[hs], 1ULL);
      halo_gids[h_slot] = static_cast<std::uint32_t>(i);
    }
  }
}

__global__ void gather_slab_pos_gids_kernel(const double3* all_pos,
                                            const std::uint32_t* prim_gids, std::size_t n_prim,
                                            const std::uint32_t* halo_gids, std::size_t n_halo,
                                            double3* out_pos, std::uint32_t* out_gids) {
  const std::size_t total = n_prim + n_halo;
  const std::size_t stride = std::size_t(gridDim.x) * std::size_t(blockDim.x);
  for (std::size_t j = std::size_t(blockIdx.x) * std::size_t(blockDim.x) + threadIdx.x;
       j < total; j += stride) {
    const std::uint32_t gid = (j < n_prim) ? prim_gids[j] : halo_gids[j - n_prim];
    out_gids[j] = gid;
    out_pos[j] = all_pos[gid];
  }
}

__global__ void merge_edge_list_u32_kernel(const uint2* edges, std::size_t n_edges,
                                           std::uint32_t* global_parent) {
  const std::size_t stride = std::size_t(gridDim.x) * std::size_t(blockDim.x);
  for (std::size_t i = std::size_t(blockIdx.x) * std::size_t(blockDim.x) + threadIdx.x;
       i < n_edges; i += stride) {
    const uint2 e = edges[i];
    if (e.x != e.y) link_edge_u32(global_parent, e.x, e.y);
  }
}

__global__ void mark_coarse_occ_kernel(const double3* pos, std::uint64_t n,
                                       const std::uint32_t* dev_orig_idx,
                                       std::uint64_t total_count,
                                       double lx, double ly, double lz, int M,
                                       std::uint32_t* occ_bits, int* invalid) {
  const std::uint64_t stride = std::uint64_t(gridDim.x) * std::uint64_t(blockDim.x);
  for (std::uint64_t i = std::uint64_t(blockIdx.x) * std::uint64_t(blockDim.x) + threadIdx.x;
       i < n; i += stride) {
    const double3 raw = pos[i];
    if (!(isfinite(raw.x) && isfinite(raw.y) && isfinite(raw.z))) {
      atomicExch(invalid, 1);
      continue;
    }
    if (dev_orig_idx && std::uint64_t(dev_orig_idx[i]) >= total_count) {
      atomicExch(invalid, 1);
      continue;
    }
    const double wx = in_box(raw.x, lx);
    const double wy = in_box(raw.y, ly);
    const double wz = in_box(raw.z, lz);
    const int cx = axis_key(wx, lx, M);
    const int cy = axis_key(wy, ly, M);
    const int cz = axis_key(wz, lz, M);
    const std::uint32_t cid = static_cast<std::uint32_t>((cx * M + cy) * M + cz);
    atomicOr(&occ_bits[cid >> 5], 1U << (cid & 31U));
  }
}

__global__ void extract_boundary_particles_kernel(const double3* pos, std::uint64_t n,
                                                  const std::uint32_t* dev_orig_idx,
                                                  std::uint32_t gid_offset,
                                                  double lx, double ly, double lz,
                                                  double link, int M,
                                                  const std::uint32_t* other_occ,
                                                  double3* out_bpos, std::uint32_t* out_bgids,
                                                  unsigned long long* out_cnt, bool write_out) {
  const double Hx = lx / double(M);
  const double Hy = ly / double(M);
  const double Hz = lz / double(M);
  const double limit_eps = link * link * (1.0 + 1e-9);
  const std::uint64_t grid_stride = std::uint64_t(gridDim.x) * std::uint64_t(blockDim.x);
  const int lane = threadIdx.x & 31;

  for (std::uint64_t base = std::uint64_t(blockIdx.x) * std::uint64_t(blockDim.x);
       base < n; base += grid_stride) {
    const std::uint64_t i = base + std::uint64_t(threadIdx.x);
    double3 p{};
    bool is_bnd = false;
    if (i < n) {
      const double3 raw = pos[i];
      p.x = in_box(raw.x, lx);
      p.y = in_box(raw.y, ly);
      p.z = in_box(raw.z, lz);
      const int ix = axis_key(p.x, lx, M);
      const int iy = axis_key(p.y, ly, M);
      const int iz = axis_key(p.z, lz, M);

      for (int dz = -1; dz <= 1 && !is_bnd; ++dz) {
        const double gz = (dz == 0) ? 0.0
            : ((dz < 0) ? fmax(0.0, p.z - double(iz) * Hz) : fmax(0.0, double(iz + 1) * Hz - p.z));
        const double gz2 = gz * gz;
        if (gz2 > limit_eps) continue;
        const int nz_c = wrap_cell(iz + dz, M);
        for (int dy = -1; dy <= 1 && !is_bnd; ++dy) {
          const double gy = (dy == 0) ? 0.0
              : ((dy < 0) ? fmax(0.0, p.y - double(iy) * Hy) : fmax(0.0, double(iy + 1) * Hy - p.y));
          const double gyz2 = gz2 + gy * gy;
          if (gyz2 > limit_eps) continue;
          const int ny_c = wrap_cell(iy + dy, M);
          for (int dx = -1; dx <= 1; ++dx) {
            const double gx = (dx == 0) ? 0.0
                : ((dx < 0) ? fmax(0.0, p.x - double(ix) * Hx) : fmax(0.0, double(ix + 1) * Hx - p.x));
            if (gyz2 + gx * gx > limit_eps) continue;
            const int nx_c = wrap_cell(ix + dx, M);
            const std::uint32_t cid = static_cast<std::uint32_t>((nx_c * M + ny_c) * M + nz_c);
            if ((other_occ[cid >> 5] & (1U << (cid & 31U))) != 0U) {
              is_bnd = true;
              break;
            }
          }
        }
      }
    }
    const unsigned mask = __ballot_sync(0xFFFFFFFFU, is_bnd);
    if (mask != 0U) {
      unsigned long long w_base = 0ULL;
      if (lane == 0) w_base = atomicAdd(out_cnt, static_cast<unsigned long long>(__popc(mask)));
      if (write_out) {
        w_base = __shfl_sync(0xFFFFFFFFU, w_base, 0);
        if (is_bnd) {
          const unsigned lower = (1U << lane) - 1U;
          const std::uint64_t pos_idx = w_base + std::uint64_t(__popc(mask & lower));
          out_bpos[pos_idx] = p;
          out_bgids[pos_idx] = dev_orig_idx ? dev_orig_idx[i]
                                            : (gid_offset + static_cast<std::uint32_t>(i));
        }
      }
    }
  }
}

__global__ void collect_local_parent_edges_kernel(const std::uint32_t* local_parent,
                                                  const std::uint32_t* dev_orig_idx,
                                                  std::uint32_t gid_offset,
                                                  std::uint64_t n,
                                                  uint2* out_edges,
                                                  unsigned long long* write_cursor,
                                                  bool write_out) {
  const std::uint64_t grid_stride = std::uint64_t(gridDim.x) * std::uint64_t(blockDim.x);
  const int lane = threadIdx.x & 31;
  for (std::uint64_t base = std::uint64_t(blockIdx.x) * std::uint64_t(blockDim.x);
       base < n; base += grid_stride) {
    const std::uint64_t i = base + std::uint64_t(threadIdx.x);
    std::uint32_t r = 0U;
    bool keep = false;
    if (i < n) {
      r = local_parent[i];
      keep = (r != static_cast<std::uint32_t>(i));
    }
    const unsigned mask = __ballot_sync(0xFFFFFFFFU, keep);
    if (mask != 0U) {
      unsigned long long w_base = 0ULL;
      if (lane == 0) w_base = atomicAdd(write_cursor, static_cast<unsigned long long>(__popc(mask)));
      if (write_out) {
        w_base = __shfl_sync(0xFFFFFFFFU, w_base, 0);
        if (keep) {
          const unsigned lower = (1U << lane) - 1U;
          const std::uint64_t pos = w_base + std::uint64_t(__popc(mask & lower));
          const std::uint32_t u = dev_orig_idx ? dev_orig_idx[i]
                                               : (gid_offset + static_cast<std::uint32_t>(i));
          const std::uint32_t v = dev_orig_idx ? dev_orig_idx[r]
                                               : (gid_offset + r);
          out_edges[pos] = uint2{u, v};
        }
      }
    }
  }
}

// Helper for device-resident positions (large count) that internally sub-slabs along X into global_parent
int run_device_slabbed_merge_u32(const double3* d_pos, std::uint64_t count,
                                 double lx, double ly, double lz, double link,
                                 int num_slabs, std::uint32_t* d_global_parent,
                                 double* elapsed, std::uint64_t* info) {
  const int nx = static_cast<int>(std::floor(lx / link));
  const int S = std::max(2, std::min(num_slabs, std::max(1, nx / 4)));

  std::vector<int> cell_to_slab(nx, 0);
  std::vector<int> cell_first_col_slab(nx, -1);
  for (int s = 0; s < S; ++s) {
    const int c0 = int((std::uint64_t(s) * std::uint64_t(nx)) / std::uint64_t(S));
    const int c1 = int((std::uint64_t(s + 1) * std::uint64_t(nx)) / std::uint64_t(S));
    cell_first_col_slab[c0] = s;
    for (int cx = c0; cx < c1; ++cx) cell_to_slab[cx] = s;
  }

  Buffer<int> d_c2s(static_cast<std::size_t>(nx));
  Buffer<int> d_cfcs(static_cast<std::size_t>(nx));
  check(cudaMemcpy(d_c2s.ptr, cell_to_slab.data(), static_cast<std::size_t>(nx) * sizeof(int), cudaMemcpyHostToDevice));
  check(cudaMemcpy(d_cfcs.ptr, cell_first_col_slab.data(), static_cast<std::size_t>(nx) * sizeof(int), cudaMemcpyHostToDevice));

  Buffer<unsigned long long> d_prim_cnt(static_cast<std::size_t>(S));
  Buffer<unsigned long long> d_halo_cnt(static_cast<std::size_t>(S));
  Buffer<int> d_invalid(1);
  check(cudaMemset(d_prim_cnt.ptr, 0, static_cast<std::size_t>(S) * sizeof(unsigned long long)));
  check(cudaMemset(d_halo_cnt.ptr, 0, static_cast<std::size_t>(S) * sizeof(unsigned long long)));
  check(cudaMemset(d_invalid.ptr, 0, sizeof(int)));

  slab_bucket_count_kernel<<<grid_for_u64(count), threads>>>(
      d_pos, count, lx, nx, S, d_c2s.ptr, d_cfcs.ptr, d_prim_cnt.ptr, d_halo_cnt.ptr, d_invalid.ptr);
  int bad = 0;
  check(cudaMemcpy(&bad, d_invalid.ptr, sizeof(int), cudaMemcpyDeviceToHost));
  d_invalid.release();
  if (bad) return int(cudaErrorInvalidValue);

  std::vector<unsigned long long> h_prim_cnt(S), h_halo_cnt(S);
  check(cudaMemcpy(h_prim_cnt.data(), d_prim_cnt.ptr, static_cast<std::size_t>(S) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  check(cudaMemcpy(h_halo_cnt.data(), d_halo_cnt.ptr, static_cast<std::size_t>(S) * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

  std::vector<std::uint64_t> prim_off(S + 1, 0ULL), halo_off(S + 1, 0ULL);
  for (int s = 0; s < S; ++s) {
    prim_off[s + 1] = prim_off[s] + std::uint64_t(h_prim_cnt[s]);
    halo_off[s + 1] = halo_off[s] + std::uint64_t(h_halo_cnt[s]);
  }
  std::vector<unsigned long long> prim_cursor_h(S), halo_cursor_h(S);
  for (int s = 0; s < S; ++s) {
    prim_cursor_h[s] = static_cast<unsigned long long>(prim_off[s]);
    halo_cursor_h[s] = static_cast<unsigned long long>(halo_off[s]);
  }
  check(cudaMemcpy(d_prim_cnt.ptr, prim_cursor_h.data(), static_cast<std::size_t>(S) * sizeof(unsigned long long), cudaMemcpyHostToDevice));
  check(cudaMemcpy(d_halo_cnt.ptr, halo_cursor_h.data(), static_cast<std::size_t>(S) * sizeof(unsigned long long), cudaMemcpyHostToDevice));

  Buffer<std::uint32_t> d_prim_gids(static_cast<std::size_t>(count));
  Buffer<std::uint32_t> d_halo_gids(static_cast<std::size_t>(std::max<std::uint64_t>(1ULL, halo_off[S])));
  slab_bucket_scatter_kernel<<<grid_for_u64(count), threads>>>(
      d_pos, count, lx, nx, d_c2s.ptr, d_cfcs.ptr,
      d_prim_cnt.ptr, d_halo_cnt.ptr, d_prim_gids.ptr, d_halo_gids.ptr);
  check(cudaDeviceSynchronize());
  d_c2s.release();
  d_cfcs.release();
  d_prim_cnt.release();
  d_halo_cnt.release();

  for (int s = 0; s < S; ++s) {
    const std::size_t n_prim = static_cast<std::size_t>(prim_off[s + 1] - prim_off[s]);
    const int nxt = (s + 1) % S;
    const std::size_t n_halo = static_cast<std::size_t>(halo_off[nxt + 1] - halo_off[nxt]);
    const std::size_t m_s = n_prim + n_halo;
    if (m_s <= 1) continue;
    if (m_s > (1ULL << 30)) return int(cudaErrorInvalidValue);

    Buffer<double3> d_slab_pos(m_s);
    Buffer<std::uint32_t> d_slab_ids(m_s);
    gather_slab_pos_gids_kernel<<<grid_for(int(m_s)), threads>>>(
        d_pos, d_prim_gids.ptr + prim_off[s], n_prim,
        d_halo_gids.ptr + halo_off[nxt], n_halo,
        d_slab_pos.ptr, d_slab_ids.ptr);
    check(cudaGetLastError());

    double slab_el[9]{};
    const int st = run_fof_core_impl(
        reinterpret_cast<const double*>(d_slab_pos.ptr), true, true, int(m_s),
        lx, ly, lz, link, OutMode::DeviceMerge, 2,
        nullptr, 0U, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr,
        d_global_parent, d_slab_ids.ptr, nullptr,
        slab_el, info);
    if (st != 0) return st;
    if (elapsed) {
      for (int k = 0; k < 9; ++k) elapsed[k] += slab_el[k];
    }
  }
  return 0;
}

}  // namespace grav_fof

extern "C" int whole_fof(const double* host, int count,
                         double lx, double ly, double lz, double link,
                         int* labels_host, double* elapsed, std::uint64_t* info) {
  return grav_fof::run_fof_core_impl(
      host, false, false, count, lx, ly, lz, link,
      grav_fof::OutMode::RawHost, 2, nullptr, 0U,
      labels_host, nullptr, nullptr,
      nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, elapsed, info);
}

extern "C" int whole_fof_ranked(const double* host, int count,
                               double lx, double ly, double lz, double link,
                               int min_group_size,
                               std::int64_t* group_ids_host,
                               std::uint64_t* out_num_groups,
                               std::uint64_t* out_grouped_particles,
                               std::uint64_t* out_largest_group,
                               double* elapsed, std::uint64_t* info) {
  return grav_fof::run_fof_core_impl(
      host, false, false, count, lx, ly, lz, link,
      grav_fof::OutMode::Ranked, min_group_size, nullptr, 0U,
      nullptr, nullptr, group_ids_host,
      out_num_groups, out_grouped_particles, out_largest_group,
      nullptr, nullptr, nullptr, elapsed, info);
}

extern "C" int whole_fof_device(const double* dev_pos_xyz, int count,
                                      double lx, double ly, double lz, double link,
                                      int min_group_size,
                                      std::int64_t* group_ids_dev,
                                      std::int64_t* group_ids_host,
                                      std::uint64_t* out_num_groups,
                                      std::uint64_t* out_grouped_particles,
                                      std::uint64_t* out_largest_group,
                                      double* elapsed, std::uint64_t* info) {
  return grav_fof::run_fof_core_impl(
      dev_pos_xyz, true, false, count, lx, ly, lz, link,
      grav_fof::OutMode::Ranked, min_group_size, nullptr, 0U,
      nullptr, group_ids_dev, group_ids_host,
      out_num_groups, out_grouped_particles, out_largest_group,
      nullptr, nullptr, nullptr, elapsed, info);
}

namespace {

inline double host_in_box(double x, double L) {
  if (x == 0.0) return 0.0;
  if (x > 0.0 && x < L) return x;
  double u = std::fmod(x, L);
  if (u < 0.0) u += L;
  if (u >= L) u = 0.0;
  return u;
}

inline int host_axis_key(double x, double L, int n) {
  const double scale = double(n) / L;
  return std::min(n - 1, std::max(0, int(x * scale)));
}

inline double host_nearest_image(double d, double L) {
  if (d > 0.5 * L) return d - L;
  if (d < -0.5 * L) return d + L;
  return d;
}

inline std::uint32_t host_find(std::vector<std::uint32_t>& parent, std::uint32_t x) {
  std::uint32_t r = x;
  while (parent[r] != r) r = parent[r];
  while (x != r) {
    const std::uint32_t nxt = parent[x];
    parent[x] = r;
    x = nxt;
  }
  return r;
}

inline void host_union(std::vector<std::uint32_t>& parent,
                       std::vector<std::uint8_t>& rank,
                       std::uint32_t a, std::uint32_t b) {
  a = host_find(parent, a);
  b = host_find(parent, b);
  if (a == b) return;
  if (rank[a] < rank[b]) std::swap(a, b);
  parent[b] = a;
  if (rank[a] == rank[b]) ++rank[a];
}

// Host Union-Find fallback used ONLY if GPU d_parent cannot fit in VRAM
int whole_fof_slabbed_host_fallback(const double* host, std::uint64_t count,
                                    double lx, double ly, double lz, double link,
                                    int min_group_size, int S, int nx,
                                    std::int64_t* group_ids_host,
                                    std::uint64_t* out_num_groups,
                                    std::uint64_t* out_grouped_particles,
                                    std::uint64_t* out_largest_group,
                                    double* elapsed, std::uint64_t* info) {
  std::vector<int> cell_to_slab(nx, 0);
  std::vector<int> cell_first_col_slab(nx, -1);
  for (int s = 0; s < S; ++s) {
    const int c0 = int((std::uint64_t(s) * std::uint64_t(nx)) / std::uint64_t(S));
    const int c1 = int((std::uint64_t(s + 1) * std::uint64_t(nx)) / std::uint64_t(S));
    cell_first_col_slab[c0] = s;
    for (int cx = c0; cx < c1; ++cx) cell_to_slab[cx] = s;
  }

  std::vector<std::vector<std::uint32_t>> slab_primary(S);
  std::vector<std::vector<std::uint32_t>> slab_first_col(S);
  const std::size_t approx_slab = std::size_t(count / std::uint64_t(S) + count / (std::uint64_t(S) * 4ULL) + 64ULL);
  const std::size_t approx_col = std::size_t(count / std::uint64_t(nx) + count / (std::uint64_t(nx) * 2ULL) + 16ULL);
  for (int s = 0; s < S; ++s) {
    slab_primary[s].reserve(approx_slab);
    slab_first_col[s].reserve(approx_col);
  }

  for (std::uint64_t i = 0; i < count; ++i) {
    const double x = host[3 * i];
    const double y = host[3 * i + 1];
    const double z = host[3 * i + 2];
    if (!(std::isfinite(x) && std::isfinite(y) && std::isfinite(z))) {
      return int(cudaErrorInvalidValue);
    }
    const double wx = host_in_box(x, lx);
    const int cx = host_axis_key(wx, lx, nx);
    slab_primary[cell_to_slab[cx]].push_back(std::uint32_t(i));
    const int col_s = cell_first_col_slab[cx];
    if (col_s >= 0) slab_first_col[col_s].push_back(std::uint32_t(i));
  }

  std::vector<std::uint32_t> parent(count);
  for (std::uint64_t i = 0; i < count; ++i) parent[i] = std::uint32_t(i);
  std::vector<std::uint8_t> rank(count, 0);

  std::vector<double> slab_pos;
  std::vector<std::uint32_t> slab_gids;
  std::vector<int> local_roots;
  for (int s = 0; s < S; ++s) {
    const auto& prim = slab_primary[s];
    const auto& halo = slab_first_col[(s + 1) % S];
    const std::size_t m_s = prim.size() + halo.size();
    if (m_s <= 1) continue;
    if (m_s > (1ULL << 30)) return int(cudaErrorInvalidValue);

    slab_pos.resize(m_s * 3ULL);
    slab_gids.resize(m_s);
    local_roots.resize(m_s);

    std::size_t dst = 0;
    for (std::uint32_t gid : prim) {
      slab_gids[dst] = gid;
      const double* p = host + std::size_t(gid) * 3ULL;
      slab_pos[dst * 3ULL] = p[0];
      slab_pos[dst * 3ULL + 1ULL] = p[1];
      slab_pos[dst * 3ULL + 2ULL] = p[2];
      ++dst;
    }
    for (std::uint32_t gid : halo) {
      slab_gids[dst] = gid;
      const double* p = host + std::size_t(gid) * 3ULL;
      slab_pos[dst * 3ULL] = p[0];
      slab_pos[dst * 3ULL + 1ULL] = p[1];
      slab_pos[dst * 3ULL + 2ULL] = p[2];
      ++dst;
    }

    double slab_el[9]{};
    const int st = grav_fof::run_fof_core(
        slab_pos.data(), int(m_s), lx, ly, lz, link,
        false, 2, local_roots.data(), nullptr,
        nullptr, nullptr, nullptr, slab_el, info);
    if (st != 0) return st;
    if (elapsed) {
      for (int k = 0; k < 9; ++k) elapsed[k] += slab_el[k];
    }

    for (std::size_t j = 0; j < m_s; ++j) {
      const int r = local_roots[j];
      if (r != int(j)) host_union(parent, rank, slab_gids[j], slab_gids[r]);
    }
  }

  std::vector<std::vector<std::uint32_t>>().swap(slab_primary);
  std::vector<std::vector<std::uint32_t>>().swap(slab_first_col);
  std::vector<double>().swap(slab_pos);
  std::vector<std::uint32_t>().swap(slab_gids);
  std::vector<int>().swap(local_roots);
  std::vector<std::uint8_t>().swap(rank);

  for (std::uint64_t i = 0; i < count; ++i) {
    parent[i] = host_find(parent, std::uint32_t(i));
    group_ids_host[i] = 0;
  }
  for (std::uint64_t i = 0; i < count; ++i) {
    ++group_ids_host[parent[i]];
  }

  struct RootGroup {
    std::uint32_t root;
    std::uint64_t size;
  };
  std::vector<RootGroup> valid_groups;
  for (std::uint64_t i = 0; i < count; ++i) {
    if (parent[i] == std::uint32_t(i) && group_ids_host[i] >= min_group_size) {
      valid_groups.push_back({std::uint32_t(i), static_cast<std::uint64_t>(group_ids_host[i])});
    }
  }
  std::sort(valid_groups.begin(), valid_groups.end(), [](const RootGroup& a, const RootGroup& b) {
    if (a.size != b.size) return a.size > b.size;
    return a.root < b.root;
  });

  std::fill_n(group_ids_host, count, grav_fof::grav_fof_none);
  std::uint64_t grouped_total = 0;
  for (std::size_t g = 0; g < valid_groups.size(); ++g) {
    group_ids_host[valid_groups[g].root] = static_cast<std::int64_t>(g) + 1LL;
    grouped_total += valid_groups[g].size;
  }
  for (std::uint64_t i = 0; i < count; ++i) {
    group_ids_host[i] = group_ids_host[parent[i]];
  }

  if (out_num_groups) *out_num_groups = static_cast<std::uint64_t>(valid_groups.size());
  if (out_grouped_particles) *out_grouped_particles = grouped_total;
  if (out_largest_group) *out_largest_group = valid_groups.empty() ? 0ULL : valid_groups[0].size;
  return 0;
}

}  // namespace

extern "C" int whole_fof_slabbed(const double* host, std::uint64_t count,
                                       double lx, double ly, double lz, double link,
                                       int min_group_size, int num_slabs,
                                       std::int64_t* group_ids_host,
                                       std::uint64_t* out_num_groups,
                                       std::uint64_t* out_grouped_particles,
                                       std::uint64_t* out_largest_group,
                                       double* elapsed, std::uint64_t* info) {
  try {
    if (!host || !group_ids_host || count == 0 ||
        count > std::uint64_t(std::numeric_limits<std::uint32_t>::max()) - 1ULL ||
        lx <= 0.0 || ly <= 0.0 || lz <= 0.0 || link <= 0.0 ||
        min_group_size < 2 || num_slabs < 1) {
      return int(cudaErrorInvalidValue);
    }
    if (elapsed) {
      for (int i = 0; i < 9; ++i) elapsed[i] = 0.0;
    }
    const int nx = static_cast<int>(std::floor(lx / link));
    const int ny = static_cast<int>(std::floor(ly / link));
    const int nz = static_cast<int>(std::floor(lz / link));
    if (nx < 3 || ny < 3 || nz < 3) {
      return int(cudaErrorInvalidValue);
    }

    const int S = std::min(num_slabs, std::max(1, nx / 4));
    if (S <= 1) {
      if (count > (1ULL << 30)) return int(cudaErrorInvalidValue);
      return grav_fof::run_fof_core(
          host, int(count), lx, ly, lz, link,
          true, min_group_size, nullptr, group_ids_host,
          out_num_groups, out_grouped_particles, out_largest_group, elapsed, info);
    }

    grav_fof::check(cudaFree(nullptr));
    std::size_t free_mem = 0, total_mem = 0;
    grav_fof::check(cudaMemGetInfo(&free_mem, &total_mem));
    if (info) {
      info[1] = free_mem;
      info[2] = total_mem;
    }

    const bool force_stream = (std::getenv("GRAV_FOF_FORCE_STREAM") && std::atoi(std::getenv("GRAV_FOF_FORCE_STREAM")) != 0);
    const std::uint64_t max_slab_n = count / std::uint64_t(S) + count / (std::uint64_t(S) * 2ULL) +
                                     count / std::uint64_t(nx) + 1024ULL;
    const std::uint64_t slab_work_est = max_slab_n * 96ULL;

    // Tier 1: All-GPU slabbing + GPU union-find merge + GPU relabeling
    if (!force_stream && (count * 36ULL + slab_work_est) < std::uint64_t(0.70 * double(free_mem))) {
      grav_fof::Timer timer;
      grav_fof::Buffer<double3> d_host_pos(static_cast<std::size_t>(count));
      double up_s = 0.0;
      timer.start();
      grav_fof::check(cudaMemcpy(d_host_pos.ptr, host,
                                  static_cast<std::size_t>(count) * sizeof(double3),
                                  cudaMemcpyHostToDevice));
      timer.stop(&up_s);
      if (elapsed) elapsed[0] += up_s;

      grav_fof::Buffer<std::uint32_t> d_parent(static_cast<std::size_t>(count));
      grav_fof::init_iota_u32_kernel<<<grav_fof::grid_for_u64(count), grav_fof::threads>>>(
          d_parent.ptr, count);
      grav_fof::check(cudaGetLastError());

      const int st = grav_fof::run_device_slabbed_merge_u32(
          d_host_pos.ptr, count, lx, ly, lz, link, S, d_parent.ptr, elapsed, info);
      if (st != 0) return st;
      d_host_pos.release();

      grav_fof::finalize_and_relabel_ranked_u32_gpu(
          d_parent.ptr, count, min_group_size, group_ids_host,
          out_num_groups, out_grouped_particles, out_largest_group, elapsed);
      return 0;
    }

    // Tier 2: OpenMP host slab bucketing + streamed slab GPU union-find merge + GPU relabeling
    if ((count * 9ULL + slab_work_est) < std::uint64_t(0.85 * double(free_mem))) {
      std::vector<int> cell_to_slab(nx, 0);
      std::vector<int> cell_first_col_slab(nx, -1);
      for (int s = 0; s < S; ++s) {
        const int c0 = int((std::uint64_t(s) * std::uint64_t(nx)) / std::uint64_t(S));
        const int c1 = int((std::uint64_t(s + 1) * std::uint64_t(nx)) / std::uint64_t(S));
        cell_first_col_slab[c0] = s;
        for (int cx = c0; cx < c1; ++cx) cell_to_slab[cx] = s;
      }

      int nth = 1;
#ifdef _OPENMP
      nth = std::min(16, std::max(1, omp_get_max_threads()));
#endif
      std::vector<std::uint64_t> thr_prim_cnt(std::size_t(nth) * std::size_t(S), 0ULL);
      std::vector<std::uint64_t> thr_halo_cnt(std::size_t(nth) * std::size_t(S), 0ULL);
      int invalid_flag = 0;

#ifdef _OPENMP
#pragma omp parallel num_threads(nth) reduction(|:invalid_flag)
#endif
      {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
#else
        const int tid = 0, nt = 1;
#endif
        const std::uint64_t i0 = (count * std::uint64_t(tid)) / std::uint64_t(nt);
        const std::uint64_t i1 = (count * std::uint64_t(tid + 1)) / std::uint64_t(nt);
        std::uint64_t* lp = &thr_prim_cnt[std::size_t(tid) * std::size_t(S)];
        std::uint64_t* lh = &thr_halo_cnt[std::size_t(tid) * std::size_t(S)];
        for (std::uint64_t i = i0; i < i1; ++i) {
          const double x = host[3 * i];
          const double y = host[3 * i + 1];
          const double z = host[3 * i + 2];
          if (!(std::isfinite(x) && std::isfinite(y) && std::isfinite(z))) {
            invalid_flag = 1;
            continue;
          }
          const double wx = host_in_box(x, lx);
          const int cx = host_axis_key(wx, lx, nx);
          ++lp[cell_to_slab[cx]];
          const int hs = cell_first_col_slab[cx];
          if (hs >= 0) ++lh[hs];
        }
      }
      if (invalid_flag) return int(cudaErrorInvalidValue);

      std::vector<std::uint64_t> prim_off(S + 1, 0ULL), halo_off(S + 1, 0ULL);
      std::vector<std::uint64_t> thr_prim_cur(std::size_t(nth) * std::size_t(S), 0ULL);
      std::vector<std::uint64_t> thr_halo_cur(std::size_t(nth) * std::size_t(S), 0ULL);
      for (int s = 0; s < S; ++s) {
        std::uint64_t acc_p = 0ULL, acc_h = 0ULL;
        for (int t = 0; t < nth; ++t) {
          thr_prim_cur[std::size_t(t) * std::size_t(S) + std::size_t(s)] = prim_off[s] + acc_p;
          thr_halo_cur[std::size_t(t) * std::size_t(S) + std::size_t(s)] = halo_off[s] + acc_h;
          acc_p += thr_prim_cnt[std::size_t(t) * std::size_t(S) + std::size_t(s)];
          acc_h += thr_halo_cnt[std::size_t(t) * std::size_t(S) + std::size_t(s)];
        }
        prim_off[s + 1] = prim_off[s] + acc_p;
        halo_off[s + 1] = halo_off[s] + acc_h;
      }

      std::vector<std::uint32_t> prim_gids(count);
      std::vector<std::uint32_t> halo_gids(halo_off[S]);

#ifdef _OPENMP
#pragma omp parallel num_threads(nth)
#endif
      {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
#else
        const int tid = 0, nt = 1;
#endif
        const std::uint64_t i0 = (count * std::uint64_t(tid)) / std::uint64_t(nt);
        const std::uint64_t i1 = (count * std::uint64_t(tid + 1)) / std::uint64_t(nt);
        std::uint64_t* lp = &thr_prim_cur[std::size_t(tid) * std::size_t(S)];
        std::uint64_t* lh = &thr_halo_cur[std::size_t(tid) * std::size_t(S)];
        for (std::uint64_t i = i0; i < i1; ++i) {
          const double wx = host_in_box(host[3 * i], lx);
          const int cx = host_axis_key(wx, lx, nx);
          prim_gids[lp[cell_to_slab[cx]]++] = static_cast<std::uint32_t>(i);
          const int hs = cell_first_col_slab[cx];
          if (hs >= 0) halo_gids[lh[hs]++] = static_cast<std::uint32_t>(i);
        }
      }

      grav_fof::Buffer<std::uint32_t> d_parent(static_cast<std::size_t>(count));
      grav_fof::init_iota_u32_kernel<<<grav_fof::grid_for_u64(count), grav_fof::threads>>>(
          d_parent.ptr, count);
      grav_fof::check(cudaGetLastError());

      std::vector<double> slab_pos;
      std::vector<std::uint32_t> slab_gids;
      for (int s = 0; s < S; ++s) {
        const int nxt = (s + 1) % S;
        const std::size_t p_beg = static_cast<std::size_t>(prim_off[s]);
        const std::size_t p_cnt = static_cast<std::size_t>(prim_off[s + 1] - prim_off[s]);
        const std::size_t h_beg = static_cast<std::size_t>(halo_off[nxt]);
        const std::size_t h_cnt = static_cast<std::size_t>(halo_off[nxt + 1] - halo_off[nxt]);
        const std::size_t m_s = p_cnt + h_cnt;
        if (m_s <= 1) continue;
        if (m_s > (1ULL << 30)) return int(cudaErrorInvalidValue);

        slab_pos.resize(m_s * 3ULL);
        slab_gids.resize(m_s);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if(m_s > 4096)
#endif
        for (std::size_t j = 0; j < m_s; ++j) {
          const std::uint32_t gid = (j < p_cnt) ? prim_gids[p_beg + j] : halo_gids[h_beg + (j - p_cnt)];
          slab_gids[j] = gid;
          const double* src = host + std::size_t(gid) * 3ULL;
          slab_pos[j * 3ULL] = src[0];
          slab_pos[j * 3ULL + 1ULL] = src[1];
          slab_pos[j * 3ULL + 2ULL] = src[2];
        }

        grav_fof::Buffer<std::uint32_t> d_slab_gids(m_s);
        grav_fof::check(cudaMemcpy(d_slab_gids.ptr, slab_gids.data(),
                                    m_s * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
        double slab_el[9]{};
        const int st = grav_fof::run_fof_core_impl(
            slab_pos.data(), false, false, int(m_s), lx, ly, lz, link,
            grav_fof::OutMode::DeviceMerge, 2, nullptr, 0U,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            d_parent.ptr, d_slab_gids.ptr, nullptr, slab_el, info);
        if (st != 0) return st;
        if (elapsed) {
          for (int k = 0; k < 9; ++k) elapsed[k] += slab_el[k];
        }
      }

      std::vector<std::uint32_t>().swap(prim_gids);
      std::vector<std::uint32_t>().swap(halo_gids);
      std::vector<double>().swap(slab_pos);
      std::vector<std::uint32_t>().swap(slab_gids);

      grav_fof::finalize_and_relabel_ranked_u32_gpu(
          d_parent.ptr, count, min_group_size, group_ids_host,
          out_num_groups, out_grouped_particles, out_largest_group, elapsed);
      return 0;
    }

    return whole_fof_slabbed_host_fallback(
        host, count, lx, ly, lz, link, min_group_size, S, nx,
        group_ids_host, out_num_groups, out_grouped_particles, out_largest_group,
        elapsed, info);
  } catch (const grav_fof::GpuError& err) {
    return int(err.status);
  } catch (const std::bad_alloc&) {
    return int(cudaErrorMemoryAllocation);
  } catch (...) {
    return int(cudaErrorUnknown);
  }
}

extern "C" int whole_fof_multigpu(int num_gpus, const int* dev_ids,
                                        const double* const* dev_pos_xyz,
                                        const std::uint32_t* const* dev_orig_idx,
                                        const std::uint64_t* dev_counts,
                                        std::uint64_t total_count,
                                        double lx, double ly, double lz, double link,
                                        int min_group_size,
                                        std::int64_t* group_ids_host,
                                        std::uint64_t* out_num_groups,
                                        std::uint64_t* out_grouped_particles,
                                        std::uint64_t* out_largest_group,
                                        double* elapsed) {
  grav_fof::DeviceGuard dev_guard;
  try {
    if (num_gpus < 1 || !dev_ids || !dev_pos_xyz || !dev_counts || !group_ids_host ||
        total_count == 0 ||
        total_count > std::uint64_t(std::numeric_limits<std::uint32_t>::max()) - 1ULL ||
        lx <= 0.0 || ly <= 0.0 || lz <= 0.0 || link <= 0.0 || min_group_size < 2) {
      return int(cudaErrorInvalidValue);
    }
    const int nx = static_cast<int>(std::floor(lx / link));
    const int ny = static_cast<int>(std::floor(ly / link));
    const int nz = static_cast<int>(std::floor(lz / link));
    if (nx < 3 || ny < 3 || nz < 3) {
      return int(cudaErrorInvalidValue);
    }

    std::vector<std::uint64_t> prefix(std::size_t(num_gpus) + 1ULL, 0ULL);
    for (int g = 0; g < num_gpus; ++g) {
      if (dev_counts[g] > 0 && !dev_pos_xyz[g]) return int(cudaErrorInvalidValue);
      prefix[g + 1] = prefix[g] + dev_counts[g];
    }
    if (prefix[num_gpus] != total_count) return int(cudaErrorInvalidValue);

    if (elapsed) {
      for (int i = 0; i < 9; ++i) elapsed[i] = 0.0;
    }

    // Fast 1-GPU zero-copy path when single GPU fits directly
    if (num_gpus == 1) {
      grav_fof::check(cudaSetDevice(dev_ids[0]));
      grav_fof::check(cudaFree(nullptr));
      std::size_t free_mem = 0, total_mem = 0;
      grav_fof::check(cudaMemGetInfo(&free_mem, &total_mem));
      if (total_count <= (1ULL << 30) &&
          total_count * 96ULL < std::uint64_t(0.72 * double(free_mem))) {
        const std::uint32_t* orig0 = dev_orig_idx ? dev_orig_idx[0] : nullptr;
        return grav_fof::run_fof_core_impl(
            dev_pos_xyz[0], true, false, int(total_count),
            lx, ly, lz, link, grav_fof::OutMode::Ranked, min_group_size,
            orig0, 0U, nullptr, nullptr, group_ids_host,
            out_num_groups, out_grouped_particles, out_largest_group,
            nullptr, nullptr, nullptr, elapsed, nullptr);
      }
      // Large 1-GPU fallback via sub-slabbing on device
      grav_fof::Buffer<std::uint32_t> d_parent(static_cast<std::size_t>(total_count));
      grav_fof::init_iota_u32_kernel<<<grav_fof::grid_for_u64(total_count), grav_fof::threads>>>(
          d_parent.ptr, total_count);
      const std::uint32_t* orig0 = dev_orig_idx ? dev_orig_idx[0] : nullptr;
      const int S = std::max(2, std::min(16, std::max(2, nx / 4)));
      // Wait: if orig0 != nullptr we map edges through orig0 by passing local_parent -> collect edges!
      if (!orig0) {
        const int st = grav_fof::run_device_slabbed_merge_u32(
            reinterpret_cast<const double3*>(dev_pos_xyz[0]), total_count,
            lx, ly, lz, link, S, d_parent.ptr, elapsed, nullptr);
        if (st != 0) return st;
      } else {
        grav_fof::Buffer<std::uint32_t> d_local_parent(static_cast<std::size_t>(total_count));
        grav_fof::init_iota_u32_kernel<<<grav_fof::grid_for_u64(total_count), grav_fof::threads>>>(
            d_local_parent.ptr, total_count);
        const int st = grav_fof::run_device_slabbed_merge_u32(
            reinterpret_cast<const double3*>(dev_pos_xyz[0]), total_count,
            lx, ly, lz, link, S, d_local_parent.ptr, elapsed, nullptr);
        if (st != 0) return st;
        grav_fof::compress_parent_u32_kernel<<<grav_fof::grid_for_u64(total_count), grav_fof::threads>>>(
            d_local_parent.ptr, total_count);
        grav_fof::Buffer<unsigned long long> d_ecnt(1);
        grav_fof::check(cudaMemset(d_ecnt.ptr, 0, sizeof(unsigned long long)));
        grav_fof::collect_local_parent_edges_kernel<<<grav_fof::grid_for_u64(total_count), grav_fof::threads>>>(
            d_local_parent.ptr, orig0, 0U, total_count, nullptr, d_ecnt.ptr, false);
        unsigned long long n_ed = 0ULL;
        grav_fof::check(cudaMemcpy(&n_ed, d_ecnt.ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (n_ed > 0ULL) {
          grav_fof::Buffer<uint2> d_edges(static_cast<std::size_t>(n_ed));
          grav_fof::check(cudaMemset(d_ecnt.ptr, 0, sizeof(unsigned long long)));
          grav_fof::collect_local_parent_edges_kernel<<<grav_fof::grid_for_u64(total_count), grav_fof::threads>>>(
              d_local_parent.ptr, orig0, 0U, total_count, d_edges.ptr, d_ecnt.ptr, true);
          d_local_parent.release();
          grav_fof::merge_edge_list_u32_kernel<<<grav_fof::grid_for_u64(n_ed), grav_fof::threads>>>(
              d_edges.ptr, std::size_t(n_ed), d_parent.ptr);
        }
      }
      grav_fof::finalize_and_relabel_ranked_u32_gpu(
          d_parent.ptr, total_count, min_group_size, group_ids_host,
          out_num_groups, out_grouped_particles, out_largest_group, elapsed);
      return 0;
    }

    // Multi-GPU path (num_gpus >= 2): parallel per-GPU FoF + coarse periodic boundary exchange + merge on dev_ids[0]
    const int M = std::min(128, std::min(nx, std::min(ny, nz)));
    const std::size_t ncells_coarse = std::size_t(M) * std::size_t(M) * std::size_t(M);
    const std::size_t nwords = (ncells_coarse + 31ULL) / 32ULL;

    std::vector<std::vector<std::uint32_t>> h_occ(num_gpus, std::vector<std::uint32_t>(nwords, 0U));
    std::vector<std::vector<uint2>> h_edges(num_gpus);
    std::vector<int> s1_status(num_gpus, 0);
    std::vector<std::thread> workers;
    workers.reserve(num_gpus);

    for (int g = 0; g < num_gpus; ++g) {
      workers.emplace_back([&, g]() {
        try {
          const std::uint64_t n_g = dev_counts[g];
          if (n_g == 0ULL) return;
          grav_fof::check(cudaSetDevice(dev_ids[g]));
          grav_fof::check(cudaFree(nullptr));

          const std::uint32_t* orig_g = dev_orig_idx ? dev_orig_idx[g] : nullptr;
          const std::uint32_t gid_off_g = static_cast<std::uint32_t>(prefix[g]);
          const double3* pos_g = reinterpret_cast<const double3*>(dev_pos_xyz[g]);

          grav_fof::Buffer<std::uint32_t> d_occ(nwords);
          grav_fof::Buffer<int> d_invalid(1);
          grav_fof::check(cudaMemset(d_occ.ptr, 0, nwords * sizeof(std::uint32_t)));
          grav_fof::check(cudaMemset(d_invalid.ptr, 0, sizeof(int)));
          grav_fof::mark_coarse_occ_kernel<<<grav_fof::grid_for_u64(n_g), grav_fof::threads>>>(
              pos_g, n_g, orig_g, total_count, lx, ly, lz, M, d_occ.ptr, d_invalid.ptr);
          int bad = 0;
          grav_fof::check(cudaMemcpy(&bad, d_invalid.ptr, sizeof(int), cudaMemcpyDeviceToHost));
          if (bad) {
            s1_status[g] = int(cudaErrorInvalidValue);
            return;
          }
          grav_fof::check(cudaMemcpy(h_occ[g].data(), d_occ.ptr, nwords * sizeof(std::uint32_t),
                                      cudaMemcpyDeviceToHost));
          d_occ.release();
          d_invalid.release();

          std::size_t free_g = 0, total_g = 0;
          grav_fof::check(cudaMemGetInfo(&free_g, &total_g));
          if (n_g <= (1ULL << 30) && n_g * 96ULL < std::uint64_t(0.72 * double(free_g))) {
            s1_status[g] = grav_fof::run_fof_core_impl(
                dev_pos_xyz[g], true, false, int(n_g),
                lx, ly, lz, link, grav_fof::OutMode::DeviceMerge, 2,
                orig_g, gid_off_g, nullptr, nullptr, nullptr,
                nullptr, nullptr, nullptr,
                nullptr, nullptr, &h_edges[g], nullptr, nullptr);
          } else {
            grav_fof::Buffer<std::uint32_t> d_loc_parent(static_cast<std::size_t>(n_g));
            grav_fof::init_iota_u32_kernel<<<grav_fof::grid_for_u64(n_g), grav_fof::threads>>>(
                d_loc_parent.ptr, n_g);
            const int S_loc = std::max(2, std::min(16, std::max(2, nx / 4)));
            const int st = grav_fof::run_device_slabbed_merge_u32(
                pos_g, n_g, lx, ly, lz, link, S_loc, d_loc_parent.ptr, nullptr, nullptr);
            if (st != 0) {
              s1_status[g] = st;
              return;
            }
            grav_fof::compress_parent_u32_kernel<<<grav_fof::grid_for_u64(n_g), grav_fof::threads>>>(
                d_loc_parent.ptr, n_g);
            grav_fof::Buffer<unsigned long long> d_ecnt(1);
            grav_fof::check(cudaMemset(d_ecnt.ptr, 0, sizeof(unsigned long long)));
            grav_fof::collect_local_parent_edges_kernel<<<grav_fof::grid_for_u64(n_g), grav_fof::threads>>>(
                d_loc_parent.ptr, orig_g, gid_off_g, n_g, nullptr, d_ecnt.ptr, false);
            unsigned long long n_ed = 0ULL;
            grav_fof::check(cudaMemcpy(&n_ed, d_ecnt.ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            h_edges[g].resize(static_cast<std::size_t>(n_ed));
            if (n_ed > 0ULL) {
              grav_fof::Buffer<uint2> d_ed(static_cast<std::size_t>(n_ed));
              grav_fof::check(cudaMemset(d_ecnt.ptr, 0, sizeof(unsigned long long)));
              grav_fof::collect_local_parent_edges_kernel<<<grav_fof::grid_for_u64(n_g), grav_fof::threads>>>(
                  d_loc_parent.ptr, orig_g, gid_off_g, n_g, d_ed.ptr, d_ecnt.ptr, true);
              grav_fof::check(cudaMemcpy(h_edges[g].data(), d_ed.ptr,
                                          static_cast<std::size_t>(n_ed) * sizeof(uint2),
                                          cudaMemcpyDeviceToHost));
            }
          }
        } catch (const grav_fof::GpuError& err) {
          s1_status[g] = int(err.status);
        } catch (...) {
          s1_status[g] = int(cudaErrorUnknown);
        }
      });
    }
    for (auto& t : workers) t.join();
    for (int g = 0; g < num_gpus; ++g) {
      if (s1_status[g] != 0) return s1_status[g];
    }

    // Build complement bitmasks and extract periodic boundary particles per GPU in parallel
    std::vector<std::vector<double3>> h_bpos(num_gpus);
    std::vector<std::vector<std::uint32_t>> h_bgids(num_gpus);
    std::vector<int> s2_status(num_gpus, 0);
    workers.clear();

    for (int g = 0; g < num_gpus; ++g) {
      workers.emplace_back([&, g]() {
        try {
          const std::uint64_t n_g = dev_counts[g];
          if (n_g == 0ULL) return;
          std::vector<std::uint32_t> other_occ(nwords, 0U);
          bool any_other = false;
          for (int g2 = 0; g2 < num_gpus; ++g2) {
            if (g2 == g) continue;
            for (std::size_t w = 0; w < nwords; ++w) {
              const std::uint32_t bits = h_occ[g2][w];
              other_occ[w] |= bits;
              if (bits) any_other = true;
            }
          }
          if (!any_other) return;

          grav_fof::check(cudaSetDevice(dev_ids[g]));
          grav_fof::Buffer<std::uint32_t> d_other(nwords);
          grav_fof::check(cudaMemcpy(d_other.ptr, other_occ.data(),
                                      nwords * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
          grav_fof::Buffer<unsigned long long> d_bcnt(1);
          grav_fof::check(cudaMemset(d_bcnt.ptr, 0, sizeof(unsigned long long)));

          const double3* pos_g = reinterpret_cast<const double3*>(dev_pos_xyz[g]);
          const std::uint32_t* orig_g = dev_orig_idx ? dev_orig_idx[g] : nullptr;
          const std::uint32_t gid_off_g = static_cast<std::uint32_t>(prefix[g]);

          grav_fof::extract_boundary_particles_kernel<<<grav_fof::grid_for_u64(n_g), grav_fof::threads>>>(
              pos_g, n_g, orig_g, gid_off_g, lx, ly, lz, link, M, d_other.ptr,
              nullptr, nullptr, d_bcnt.ptr, false);
          unsigned long long b_cnt = 0ULL;
          grav_fof::check(cudaMemcpy(&b_cnt, d_bcnt.ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
          if (b_cnt == 0ULL) return;

          grav_fof::Buffer<double3> d_bp(static_cast<std::size_t>(b_cnt));
          grav_fof::Buffer<std::uint32_t> d_bg(static_cast<std::size_t>(b_cnt));
          grav_fof::check(cudaMemset(d_bcnt.ptr, 0, sizeof(unsigned long long)));
          grav_fof::extract_boundary_particles_kernel<<<grav_fof::grid_for_u64(n_g), grav_fof::threads>>>(
              pos_g, n_g, orig_g, gid_off_g, lx, ly, lz, link, M, d_other.ptr,
              d_bp.ptr, d_bg.ptr, d_bcnt.ptr, true);
          h_bpos[g].resize(static_cast<std::size_t>(b_cnt));
          h_bgids[g].resize(static_cast<std::size_t>(b_cnt));
          grav_fof::check(cudaMemcpy(h_bpos[g].data(), d_bp.ptr,
                                      static_cast<std::size_t>(b_cnt) * sizeof(double3),
                                      cudaMemcpyDeviceToHost));
          grav_fof::check(cudaMemcpy(h_bgids[g].data(), d_bg.ptr,
                                      static_cast<std::size_t>(b_cnt) * sizeof(std::uint32_t),
                                      cudaMemcpyDeviceToHost));
        } catch (const grav_fof::GpuError& err) {
          s2_status[g] = int(err.status);
        } catch (...) {
          s2_status[g] = int(cudaErrorUnknown);
        }
      });
    }
    for (auto& t : workers) t.join();
    for (int g = 0; g < num_gpus; ++g) {
      if (s2_status[g] != 0) return s2_status[g];
    }
    std::vector<std::vector<std::uint32_t>>().swap(h_occ);

    // Stage 2B: Global union-find merge + cross-domain boundary FoF + relabel on dev_ids[0]
    grav_fof::check(cudaSetDevice(dev_ids[0]));
    grav_fof::Buffer<std::uint32_t> d_parent(static_cast<std::size_t>(total_count));
    grav_fof::init_iota_u32_kernel<<<grav_fof::grid_for_u64(total_count), grav_fof::threads>>>(
        d_parent.ptr, total_count);
    grav_fof::check(cudaGetLastError());

    for (int g = 0; g < num_gpus; ++g) {
      const std::size_t ne = h_edges[g].size();
      if (ne > 0) {
        const std::size_t edge_chunk = 32ULL << 20;
        grav_fof::Buffer<uint2> d_ed(std::min(ne, edge_chunk));
        for (std::size_t off = 0; off < ne; off += edge_chunk) {
          const std::size_t cur = std::min(edge_chunk, ne - off);
          grav_fof::check(cudaMemcpy(d_ed.ptr, h_edges[g].data() + off,
                                      cur * sizeof(uint2), cudaMemcpyHostToDevice));
          grav_fof::merge_edge_list_u32_kernel<<<grav_fof::grid_for_u64(cur), grav_fof::threads>>>(
              d_ed.ptr, cur, d_parent.ptr);
        }
      }
      std::vector<uint2>().swap(h_edges[g]);
    }
    std::vector<std::vector<uint2>>().swap(h_edges);

    std::size_t total_B = 0;
    for (int g = 0; g < num_gpus; ++g) total_B += h_bpos[g].size();

    if (total_B > 1) {
      std::vector<double3> bpos_all;
      std::vector<std::uint32_t> bgids_all;
      bpos_all.reserve(total_B);
      bgids_all.reserve(total_B);
      for (int g = 0; g < num_gpus; ++g) {
        bpos_all.insert(bpos_all.end(), h_bpos[g].begin(), h_bpos[g].end());
        bgids_all.insert(bgids_all.end(), h_bgids[g].begin(), h_bgids[g].end());
        std::vector<double3>().swap(h_bpos[g]);
        std::vector<std::uint32_t>().swap(h_bgids[g]);
      }

      std::size_t free0 = 0, total0 = 0;
      grav_fof::check(cudaMemGetInfo(&free0, &total0));
      grav_fof::Buffer<std::uint32_t> d_bgids(total_B);
      grav_fof::check(cudaMemcpy(d_bgids.ptr, bgids_all.data(),
                                  total_B * sizeof(std::uint32_t), cudaMemcpyHostToDevice));

      if (total_B <= (1ULL << 30) &&
          std::uint64_t(total_B) * 96ULL < std::uint64_t(0.70 * double(free0))) {
        const int st = grav_fof::run_fof_core_impl(
            reinterpret_cast<const double*>(bpos_all.data()), false, false, int(total_B),
            lx, ly, lz, link, grav_fof::OutMode::DeviceMerge, 2, nullptr, 0U,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            d_parent.ptr, d_bgids.ptr, nullptr, elapsed, nullptr);
        if (st != 0) return st;
      } else {
        // Fallback if massive boundary (e.g. random partition across GPUs)
        grav_fof::Buffer<double3> d_bpos(total_B);
        grav_fof::check(cudaMemcpy(d_bpos.ptr, bpos_all.data(),
                                    total_B * sizeof(double3), cudaMemcpyHostToDevice));
        grav_fof::Buffer<std::uint32_t> d_bloc_parent(total_B);
        grav_fof::init_iota_u32_kernel<<<grav_fof::grid_for_u64(total_B), grav_fof::threads>>>(
            d_bloc_parent.ptr, total_B);
        const int S_bnd = std::max(2, std::min(16, std::max(2, nx / 4)));
        const int st = grav_fof::run_device_slabbed_merge_u32(
            d_bpos.ptr, total_B, lx, ly, lz, link, S_bnd, d_bloc_parent.ptr, elapsed, nullptr);
        if (st != 0) return st;
        d_bpos.release();
        grav_fof::compress_parent_u32_kernel<<<grav_fof::grid_for_u64(total_B), grav_fof::threads>>>(
            d_bloc_parent.ptr, total_B);
        grav_fof::Buffer<unsigned long long> d_ecnt(1);
        grav_fof::check(cudaMemset(d_ecnt.ptr, 0, sizeof(unsigned long long)));
        grav_fof::collect_local_parent_edges_kernel<<<grav_fof::grid_for_u64(total_B), grav_fof::threads>>>(
            d_bloc_parent.ptr, d_bgids.ptr, 0U, total_B, nullptr, d_ecnt.ptr, false);
        unsigned long long n_ed = 0ULL;
        grav_fof::check(cudaMemcpy(&n_ed, d_ecnt.ptr, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        if (n_ed > 0ULL) {
          grav_fof::Buffer<uint2> d_ed(static_cast<std::size_t>(n_ed));
          grav_fof::check(cudaMemset(d_ecnt.ptr, 0, sizeof(unsigned long long)));
          grav_fof::collect_local_parent_edges_kernel<<<grav_fof::grid_for_u64(total_B), grav_fof::threads>>>(
              d_bloc_parent.ptr, d_bgids.ptr, 0U, total_B, d_ed.ptr, d_ecnt.ptr, true);
          d_bloc_parent.release();
          grav_fof::merge_edge_list_u32_kernel<<<grav_fof::grid_for_u64(n_ed), grav_fof::threads>>>(
              d_ed.ptr, std::size_t(n_ed), d_parent.ptr);
        }
      }
    }

    grav_fof::finalize_and_relabel_ranked_u32_gpu(
        d_parent.ptr, total_count, min_group_size, group_ids_host,
        out_num_groups, out_grouped_particles, out_largest_group, elapsed);
    return 0;
  } catch (const grav_fof::GpuError& err) {
    return int(err.status);
  } catch (const std::bad_alloc&) {
    return int(cudaErrorMemoryAllocation);
  } catch (...) {
    return int(cudaErrorUnknown);
  }
}

// ============================================================================
// Full-GPU compute_fof_catalog & compute_fof_so Kernels + Launchers
// ============================================================================

namespace grav_fof {

__global__ void init_u32_val_kernel(std::uint32_t* arr, std::size_t n, std::uint32_t val) {
  const std::size_t stride = std::size_t(gridDim.x) * std::size_t(blockDim.x);
  for (std::size_t i = std::size_t(blockIdx.x) * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += stride) {
    arr[i] = val;
  }
}

__global__ void catalog_find_chunk_anchors_kernel(const std::int64_t* gids, std::size_t n,
                                                  std::uint64_t ng,
                                                  const unsigned char* has_anchor,
                                                  std::uint32_t* chunk_anchor_j) {
  const std::size_t stride = std::size_t(gridDim.x) * std::size_t(blockDim.x);
  for (std::size_t j = std::size_t(blockIdx.x) * std::size_t(blockDim.x) + threadIdx.x;
       j < n; j += stride) {
    const std::int64_t gid = gids[j];
    if (gid < 1 || gid > static_cast<std::int64_t>(ng)) continue;
    const std::size_t h = static_cast<std::size_t>(gid - 1);
    if (!has_anchor[h]) {
      atomicMin(&chunk_anchor_j[h], static_cast<std::uint32_t>(j));
    }
  }
}

__global__ void catalog_commit_anchors_kernel(const double3* pos, std::size_t cur,
                                              std::size_t ng, double lx, double ly, double lz,
                                              unsigned char* has_anchor,
                                              const std::uint32_t* chunk_anchor_j,
                                              double3* anchor) {
  const std::size_t stride = std::size_t(gridDim.x) * std::size_t(blockDim.x);
  for (std::size_t h = std::size_t(blockIdx.x) * std::size_t(blockDim.x) + threadIdx.x;
       h < ng; h += stride) {
    if (!has_anchor[h]) {
      const std::uint32_t jj = chunk_anchor_j[h];
      if (std::size_t(jj) < cur) {
        const double3 p = pos[jj];
        anchor[h] = double3{in_box(p.x, lx), in_box(p.y, ly), in_box(p.z, lz)};
        has_anchor[h] = 1;
      }
    }
  }
}

__global__ void catalog_pass1_kernel(const double3* pos, const float3* vel,
                                     const float* masses, double uniform_mass,
                                     const std::int64_t* gids, std::size_t n,
                                     std::uint64_t ng, double lx, double ly, double lz,
                                     const double3* anchor,
                                     unsigned long long* d_size, double* d_mass,
                                     double* d_com_x, double* d_com_y, double* d_com_z,
                                     double* d_bvel_x, double* d_bvel_y, double* d_bvel_z,
                                     int* d_invalid) {
  const std::size_t stride = std::size_t(gridDim.x) * std::size_t(blockDim.x);
  for (std::size_t j = std::size_t(blockIdx.x) * std::size_t(blockDim.x) + threadIdx.x;
       j < n; j += stride) {
    const std::int64_t gid = gids[j];
    if (gid < 1 || gid > static_cast<std::int64_t>(ng)) continue;
    const std::size_t h = static_cast<std::size_t>(gid - 1);
    const double mp = masses ? double(masses[j]) : uniform_mass;
    if (!isfinite(mp) || mp < 0.0) {
      atomicExch(d_invalid, 1);
      continue;
    }
    const double3 p_raw = pos[j];
    const double3 anc = anchor[h];
    const double dx = exact_image(in_box(p_raw.x, lx), anc.x, lx);
    const double dy = exact_image(in_box(p_raw.y, ly), anc.y, ly);
    const double dz = exact_image(in_box(p_raw.z, lz), anc.z, lz);
    double cx = mp * dx;
    double cy = mp * dy;
    double cz = mp * dz;
    double vx = 0.0, vy = 0.0, vz = 0.0;
    if (vel) {
      const float3 vv = vel[j];
      vx = mp * double(vv.x);
      vy = mp * double(vv.y);
      vz = mp * double(vv.z);
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    const unsigned same = __match_any_sync(__activemask(), static_cast<unsigned long long>(h));
    if (same == 0xFFFFFFFFU) {
      unsigned cnt = 1U;
      double msum = mp;
      for (int off = 16; off > 0; off >>= 1) {
        cnt += __shfl_down_sync(0xFFFFFFFFU, cnt, off);
        msum += __shfl_down_sync(0xFFFFFFFFU, msum, off);
        cx += __shfl_down_sync(0xFFFFFFFFU, cx, off);
        cy += __shfl_down_sync(0xFFFFFFFFU, cy, off);
        cz += __shfl_down_sync(0xFFFFFFFFU, cz, off);
        if (vel) {
          vx += __shfl_down_sync(0xFFFFFFFFU, vx, off);
          vy += __shfl_down_sync(0xFFFFFFFFU, vy, off);
          vz += __shfl_down_sync(0xFFFFFFFFU, vz, off);
        }
      }
      if ((threadIdx.x & 31) == 0) {
        atomicAdd(&d_size[h], static_cast<unsigned long long>(cnt));
        atomicAddDouble(&d_mass[h], msum);
        atomicAddDouble(&d_com_x[h], cx);
        atomicAddDouble(&d_com_y[h], cy);
        atomicAddDouble(&d_com_z[h], cz);
        if (vel) {
          atomicAddDouble(&d_bvel_x[h], vx);
          atomicAddDouble(&d_bvel_y[h], vy);
          atomicAddDouble(&d_bvel_z[h], vz);
        }
      }
      continue;
    }
#endif
    atomicAdd(&d_size[h], 1ULL);
    atomicAddDouble(&d_mass[h], mp);
    atomicAddDouble(&d_com_x[h], cx);
    atomicAddDouble(&d_com_y[h], cy);
    atomicAddDouble(&d_com_z[h], cz);
    if (vel) {
      atomicAddDouble(&d_bvel_x[h], vx);
      atomicAddDouble(&d_bvel_y[h], vy);
      atomicAddDouble(&d_bvel_z[h], vz);
    }
  }
}

__global__ void catalog_finalize_pass1_kernel(std::size_t ng, double lx, double ly, double lz,
                                              const double3* anchor,
                                              const unsigned long long* d_size,
                                              const double* d_mass,
                                              double* d_com_x, double* d_com_y, double* d_com_z,
                                              double* d_bvel_x, double* d_bvel_y, double* d_bvel_z,
                                              int* d_invalid) {
  const std::size_t stride = std::size_t(gridDim.x) * std::size_t(blockDim.x);
  for (std::size_t h = std::size_t(blockIdx.x) * std::size_t(blockDim.x) + threadIdx.x;
       h < ng; h += stride) {
    if (d_size[h] == 0ULL) {
      atomicExch(d_invalid, 1);
      continue;
    }
    const double m = d_mass[h];
    const double inv_m = (m > 0.0) ? (1.0 / m) : 0.0;
    const double3 anc = anchor[h];
    d_com_x[h] = in_box(anc.x + d_com_x[h] * inv_m, lx);
    d_com_y[h] = in_box(anc.y + d_com_y[h] * inv_m, ly);
    d_com_z[h] = in_box(anc.z + d_com_z[h] * inv_m, lz);
    d_bvel_x[h] *= inv_m;
    d_bvel_y[h] *= inv_m;
    d_bvel_z[h] *= inv_m;
  }
}

__global__ void catalog_pass2_kernel(const double3* pos, const float3* vel,
                                     const float* masses, double uniform_mass,
                                     const std::int64_t* gids, std::size_t n,
                                     std::uint64_t ng, double lx, double ly, double lz,
                                     const double* d_com_x, const double* d_com_y, const double* d_com_z,
                                     const double* d_bvel_x, const double* d_bvel_y, const double* d_bvel_z,
                                     double* d_max_r2, double* d_sum_mv2) {
  const std::size_t stride = std::size_t(gridDim.x) * std::size_t(blockDim.x);
  for (std::size_t j = std::size_t(blockIdx.x) * std::size_t(blockDim.x) + threadIdx.x;
       j < n; j += stride) {
    const std::int64_t gid = gids[j];
    if (gid < 1 || gid > static_cast<std::int64_t>(ng)) continue;
    const std::size_t h = static_cast<std::size_t>(gid - 1);
    const double3 p_raw = pos[j];
    const double dx = exact_image(in_box(p_raw.x, lx), d_com_x[h], lx);
    const double dy = exact_image(in_box(p_raw.y, ly), d_com_y[h], ly);
    const double dz = exact_image(in_box(p_raw.z, lz), d_com_z[h], lz);
    double r2 = fmax(0.0, dx * dx + dy * dy + dz * dz);
    double mv2 = 0.0;
    if (vel) {
      const double mp = masses ? double(masses[j]) : uniform_mass;
      const float3 vv = vel[j];
      const double dvx = double(vv.x) - d_bvel_x[h];
      const double dvy = double(vv.y) - d_bvel_y[h];
      const double dvz = double(vv.z) - d_bvel_z[h];
      mv2 = mp * (dvx * dvx + dvy * dvy + dvz * dvz);
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    const unsigned same = __match_any_sync(__activemask(), static_cast<unsigned long long>(h));
    if (same == 0xFFFFFFFFU) {
      for (int off = 16; off > 0; off >>= 1) {
        r2 = fmax(r2, __shfl_down_sync(0xFFFFFFFFU, r2, off));
        if (vel) mv2 += __shfl_down_sync(0xFFFFFFFFU, mv2, off);
      }
      if ((threadIdx.x & 31) == 0) {
        atomicMaxNonNegDouble(&d_max_r2[h], r2);
        if (vel) atomicAddDouble(&d_sum_mv2[h], mv2);
      }
      continue;
    }
#endif
    atomicMaxNonNegDouble(&d_max_r2[h], r2);
    if (vel) atomicAddDouble(&d_sum_mv2[h], mv2);
  }
}

__global__ void catalog_pack_rows_kernel(std::size_t ng,
                                         const unsigned long long* d_size,
                                         const double* d_mass,
                                         const double* d_com_x, const double* d_com_y, const double* d_com_z,
                                         const double* d_bvel_x, const double* d_bvel_y, const double* d_bvel_z,
                                         const double* d_max_r2, const double* d_sum_mv2,
                                         FofHaloRow* out_halos) {
  const std::size_t stride = std::size_t(gridDim.x) * std::size_t(blockDim.x);
  for (std::size_t h = std::size_t(blockIdx.x) * std::size_t(blockDim.x) + threadIdx.x;
       h < ng; h += stride) {
    FofHaloRow row{};
    row.group_id = static_cast<std::int64_t>(h) + 1LL;
    row.size = static_cast<std::uint64_t>(d_size[h]);
    row.mass = d_mass[h];
    row.com[0] = d_com_x[h];
    row.com[1] = d_com_y[h];
    row.com[2] = d_com_z[h];
    row.bulk_vel[0] = d_bvel_x[h];
    row.bulk_vel[1] = d_bvel_y[h];
    row.bulk_vel[2] = d_bvel_z[h];
    const double inv_m = (row.mass > 0.0) ? (1.0 / row.mass) : 0.0;
    row.r_max = sqrt(d_max_r2[h]);
    row.vel_disp = sqrt(d_sum_mv2[h] * inv_m);
    out_halos[h] = row;
  }
}

// ---------- SO Kernels ----------

struct SoGridParams {
  int gn;
  double lx, ly, lz;
  double cw_x, cw_y, cw_z;
  double thresh_c, thresh_m;
  double uniform_mass;
};

__host__ __device__ __forceinline__ void so_halo_span(
    int gn, double cw_x, double cw_y, double cw_z, double r_cap,
    int* dx_lo, int* dx_hi, int* dy_lo, int* dy_hi, int* dz_lo, int* dz_hi) {
  const int half_lo = (gn - 1) / 2;
  const int half_hi = gn / 2;
  const int rx = (half_hi < int(ceil(r_cap / cw_x))) ? half_hi : int(ceil(r_cap / cw_x));
  const int ry = (half_hi < int(ceil(r_cap / cw_y))) ? half_hi : int(ceil(r_cap / cw_y));
  const int rz = (half_hi < int(ceil(r_cap / cw_z))) ? half_hi : int(ceil(r_cap / cw_z));
  *dx_lo = -(rx < half_lo ? rx : half_lo); *dx_hi = (rx < half_hi ? rx : half_hi);
  *dy_lo = -(ry < half_lo ? ry : half_lo); *dy_hi = (ry < half_hi ? ry : half_hi);
  *dz_lo = -(rz < half_lo ? rz : half_lo); *dz_hi = (rz < half_hi ? rz : half_hi);
}

__device__ __forceinline__ double so_dist2(double3 q, double3 c, double lx, double ly, double lz) {
  const double dx = exact_image(q.x, c.x, lx);
  const double dy = exact_image(q.y, c.y, ly);
  const double dz = exact_image(q.z, c.z, lz);
  return fmax(0.0, dx * dx + dy * dy + dz * dz);
}

__global__ void so_grid_key_kernel(double3* pos_raw, const float* mass_raw,
                                   std::size_t n, int gn, double lx, double ly, double lz,
                                   std::uint32_t* cid_out, std::uint32_t* pid_out,
                                   std::uint32_t* cell_cnt, int* invalid) {
  const std::size_t stride = std::size_t(gridDim.x) * std::size_t(blockDim.x);
  for (std::size_t i = std::size_t(blockIdx.x) * std::size_t(blockDim.x) + threadIdx.x;
       i < n; i += stride) {
    double3 p = pos_raw[i];
    if (!(isfinite(p.x) && isfinite(p.y) && isfinite(p.z))) {
      atomicExch(invalid, 1);
      continue;
    }
    if (mass_raw) {
      const float m = mass_raw[i];
      if (!isfinite(m) || m < 0.0f) {
        atomicExch(invalid, 1);
        continue;
      }
    }
    p.x = in_box(p.x, lx);
    p.y = in_box(p.y, ly);
    p.z = in_box(p.z, lz);
    pos_raw[i] = p;
    const int cx = axis_key(p.x, lx, gn);
    const int cy = axis_key(p.y, ly, gn);
    const int cz = axis_key(p.z, lz, gn);
    const std::uint32_t cid = static_cast<std::uint32_t>((cx * gn + cy) * gn + cz);
    cid_out[i] = cid;
    pid_out[i] = static_cast<std::uint32_t>(i);
    atomicAdd(&cell_cnt[cid], 1U);
  }
}

__global__ void so_gather_sorted_kernel(const double3* pos_raw, const float* mass_raw,
                                        const std::uint32_t* pids, std::size_t n,
                                        double3* pos_s, float* mass_s) {
  const std::size_t stride = std::size_t(gridDim.x) * std::size_t(blockDim.x);
  for (std::size_t k = std::size_t(blockIdx.x) * std::size_t(blockDim.x) + threadIdx.x;
       k < n; k += stride) {
    const std::uint32_t pid = pids[k];
    pos_s[k] = pos_raw[pid];
    if (mass_s) mass_s[k] = mass_raw[pid];
  }
}

__global__ void so_grid_chunk_count_kernel(double3* chunk_pos, const float* chunk_mass,
                                           std::size_t cur, int gn, double lx, double ly, double lz,
                                           std::uint32_t* cell_cnt, int* invalid) {
  const std::size_t stride = std::size_t(gridDim.x) * std::size_t(blockDim.x);
  for (std::size_t j = std::size_t(blockIdx.x) * std::size_t(blockDim.x) + threadIdx.x;
       j < cur; j += stride) {
    double3 p = chunk_pos[j];
    if (!(isfinite(p.x) && isfinite(p.y) && isfinite(p.z))) {
      atomicExch(invalid, 1);
      continue;
    }
    if (chunk_mass) {
      const float m = chunk_mass[j];
      if (!isfinite(m) || m < 0.0f) {
        atomicExch(invalid, 1);
        continue;
      }
    }
    p.x = in_box(p.x, lx);
    p.y = in_box(p.y, ly);
    p.z = in_box(p.z, lz);
    chunk_pos[j] = p;
    const int cx = axis_key(p.x, lx, gn);
    const int cy = axis_key(p.y, ly, gn);
    const int cz = axis_key(p.z, lz, gn);
    const std::uint32_t cid = static_cast<std::uint32_t>((cx * gn + cy) * gn + cz);
    atomicAdd(&cell_cnt[cid], 1U);
  }
}

__global__ void so_grid_chunk_scatter_kernel(const double3* chunk_pos, const float* chunk_mass,
                                             std::size_t cur, int gn, double lx, double ly, double lz,
                                             std::uint32_t* write_pos,
                                             double3* pos_s, float* mass_s) {
  const std::size_t stride = std::size_t(gridDim.x) * std::size_t(blockDim.x);
  for (std::size_t j = std::size_t(blockIdx.x) * std::size_t(blockDim.x) + threadIdx.x;
       j < cur; j += stride) {
    double3 p = chunk_pos[j];
    p.x = in_box(p.x, lx);
    p.y = in_box(p.y, ly);
    p.z = in_box(p.z, lz);
    const int cx = axis_key(p.x, lx, gn);
    const int cy = axis_key(p.y, ly, gn);
    const int cz = axis_key(p.z, lz, gn);
    const std::uint32_t cid = static_cast<std::uint32_t>((cx * gn + cy) * gn + cz);
    const std::uint32_t dst = atomicAdd(&write_pos[cid], 1U);
    pos_s[dst] = p;
    if (mass_s) mass_s[dst] = chunk_mass[j];
  }
}

__global__ void so_count_light_kernel(const double3* pos_s, const std::uint32_t* cell_offsets,
                                      SoGridParams gp,
                                      const int* light_ids, int num_light,
                                      const double3* hpos, const double* hrcap,
                                      std::uint32_t* out_cnt) {
  const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp_global >= num_light) return;
  const int lane = threadIdx.x & 31;
  const int a = light_ids[warp_global];
  const double3 c = hpos[a];
  const double r_cap = hrcap[a];
  const double r_cap2 = r_cap * r_cap;
  const int icx = axis_key(c.x, gp.lx, gp.gn);
  const int icy = axis_key(c.y, gp.ly, gp.gn);
  const int icz = axis_key(c.z, gp.lz, gp.gn);
  int dx_lo, dx_hi, dy_lo, dy_hi, dz_lo, dz_hi;
  so_halo_span(gp.gn, gp.cw_x, gp.cw_y, gp.cw_z, r_cap,
               &dx_lo, &dx_hi, &dy_lo, &dy_hi, &dz_lo, &dz_hi);

  int local_cnt = 0;
  for (int ddx = dx_lo; ddx <= dx_hi; ++ddx) {
    const int gx = wrap_cell(icx + ddx, gp.gn);
    for (int ddy = dy_lo; ddy <= dy_hi; ++ddy) {
      const int gy = wrap_cell(icy + ddy, gp.gn);
      for (int ddz = dz_lo; ddz <= dz_hi; ++ddz) {
        const int gz = wrap_cell(icz + ddz, gp.gn);
        const std::size_t cid = (std::size_t(gx) * std::size_t(gp.gn) + std::size_t(gy)) * std::size_t(gp.gn) + std::size_t(gz);
        const std::uint32_t beg = cell_offsets[cid];
        const std::uint32_t end = cell_offsets[cid + 1];
        for (std::uint32_t p = beg + std::uint32_t(lane); p < end; p += 32U) {
          if (so_dist2(pos_s[p], c, gp.lx, gp.ly, gp.lz) <= r_cap2) ++local_cnt;
        }
      }
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    local_cnt += __shfl_down_sync(0xFFFFFFFFU, local_cnt, off);
  }
  if (lane == 0) out_cnt[a] = static_cast<std::uint32_t>(local_cnt);
}

__global__ void so_count_heavy_kernel(const double3* pos_s, const std::uint32_t* cell_offsets,
                                      SoGridParams gp,
                                      const int* heavy_ids, int num_heavy,
                                      const double3* hpos, const double* hrcap,
                                      std::uint32_t* out_cnt) {
  if (blockIdx.x >= num_heavy) return;
  const int a = heavy_ids[blockIdx.x];
  const double3 c = hpos[a];
  const double r_cap = hrcap[a];
  const double r_cap2 = r_cap * r_cap;
  const int icx = axis_key(c.x, gp.lx, gp.gn);
  const int icy = axis_key(c.y, gp.ly, gp.gn);
  const int icz = axis_key(c.z, gp.lz, gp.gn);
  int dx_lo, dx_hi, dy_lo, dy_hi, dz_lo, dz_hi;
  so_halo_span(gp.gn, gp.cw_x, gp.cw_y, gp.cw_z, r_cap,
               &dx_lo, &dx_hi, &dy_lo, &dy_hi, &dz_lo, &dz_hi);
  const int nx_w = dx_hi - dx_lo + 1;
  const int ny_w = dy_hi - dy_lo + 1;
  const int nz_w = dz_hi - dz_lo + 1;
  const int ncells_visit = nx_w * ny_w * nz_w;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  std::uint32_t local_cnt = 0U;
  for (int ci = warp_id; ci < ncells_visit; ci += 8) {
    const int ix = ci / (ny_w * nz_w);
    const int rem = ci - ix * (ny_w * nz_w);
    const int iy = rem / nz_w;
    const int iz = rem - iy * nz_w;
    const int gx = wrap_cell(icx + dx_lo + ix, gp.gn);
    const int gy = wrap_cell(icy + dy_lo + iy, gp.gn);
    const int gz = wrap_cell(icz + dz_lo + iz, gp.gn);
    const std::size_t cid = (std::size_t(gx) * std::size_t(gp.gn) + std::size_t(gy)) * std::size_t(gp.gn) + std::size_t(gz);
    const std::uint32_t beg = cell_offsets[cid];
    const std::uint32_t end = cell_offsets[cid + 1];
    for (std::uint32_t p = beg + std::uint32_t(lane); p < end; p += 32U) {
      if (so_dist2(pos_s[p], c, gp.lx, gp.ly, gp.lz) <= r_cap2) ++local_cnt;
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    local_cnt += __shfl_down_sync(0xFFFFFFFFU, local_cnt, off);
  }
  __shared__ std::uint32_t s_warp[8];
  if (lane == 0) s_warp[warp_id] = local_cnt;
  __syncthreads();
  if (threadIdx.x == 0) {
    std::uint32_t tot = 0U;
    for (int w = 0; w < 8; ++w) tot += s_warp[w];
    out_cnt[a] = tot;
  }
}

__global__ void so_fill_light_kernel(const double3* pos_s, const float* mass_s,
                                     const std::uint32_t* cell_offsets,
                                     SoGridParams gp,
                                     const int* light_bi, int num_light, int b_beg,
                                     const double3* hpos, const double* hrcap, const int* d_off,
                                     double* r2_in, float* m_in) {
  const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp_global >= num_light) return;
  const int lane = threadIdx.x & 31;
  const int bi = light_bi[warp_global];
  const int a = b_beg + bi;
  const double3 c = hpos[a];
  const double r_cap = hrcap[a];
  const double r_cap2 = r_cap * r_cap;
  const int icx = axis_key(c.x, gp.lx, gp.gn);
  const int icy = axis_key(c.y, gp.ly, gp.gn);
  const int icz = axis_key(c.z, gp.lz, gp.gn);
  int dx_lo, dx_hi, dy_lo, dy_hi, dz_lo, dz_hi;
  so_halo_span(gp.gn, gp.cw_x, gp.cw_y, gp.cw_z, r_cap,
               &dx_lo, &dx_hi, &dy_lo, &dy_hi, &dz_lo, &dz_hi);

  std::uint32_t cur_out = static_cast<std::uint32_t>(d_off[bi]);
  for (int ddx = dx_lo; ddx <= dx_hi; ++ddx) {
    const int gx = wrap_cell(icx + ddx, gp.gn);
    for (int ddy = dy_lo; ddy <= dy_hi; ++ddy) {
      const int gy = wrap_cell(icy + ddy, gp.gn);
      for (int ddz = dz_lo; ddz <= dz_hi; ++ddz) {
        const int gz = wrap_cell(icz + ddz, gp.gn);
        const std::size_t cid = (std::size_t(gx) * std::size_t(gp.gn) + std::size_t(gy)) * std::size_t(gp.gn) + std::size_t(gz);
        const std::uint32_t beg = cell_offsets[cid];
        const std::uint32_t end = cell_offsets[cid + 1];
        for (std::uint32_t p_base = beg; p_base < end; p_base += 32U) {
          const std::uint32_t p = p_base + std::uint32_t(lane);
          bool hit = false;
          double r2 = 0.0;
          if (p < end) {
            r2 = so_dist2(pos_s[p], c, gp.lx, gp.ly, gp.lz);
            hit = (r2 <= r_cap2);
          }
          const unsigned mask = __ballot_sync(0xFFFFFFFFU, hit);
          if (mask != 0U) {
            if (hit) {
              const unsigned lower = (1U << lane) - 1U;
              const std::uint32_t out_idx = cur_out + static_cast<std::uint32_t>(__popc(mask & lower));
              r2_in[out_idx] = r2;
              if (m_in) m_in[out_idx] = mass_s[p];
            }
            cur_out += static_cast<std::uint32_t>(__popc(mask));
          }
        }
      }
    }
  }
}

__global__ void so_fill_heavy_kernel(const double3* pos_s, const float* mass_s,
                                     const std::uint32_t* cell_offsets,
                                     SoGridParams gp,
                                     const int* heavy_bi, int num_heavy, int b_beg,
                                     const double3* hpos, const double* hrcap, const int* d_off,
                                     double* r2_in, float* m_in) {
  if (blockIdx.x >= num_heavy) return;
  const int bi = heavy_bi[blockIdx.x];
  const int a = b_beg + bi;
  __shared__ std::uint32_t s_write;
  if (threadIdx.x == 0) s_write = static_cast<std::uint32_t>(d_off[bi]);
  __syncthreads();

  const double3 c = hpos[a];
  const double r_cap = hrcap[a];
  const double r_cap2 = r_cap * r_cap;
  const int icx = axis_key(c.x, gp.lx, gp.gn);
  const int icy = axis_key(c.y, gp.ly, gp.gn);
  const int icz = axis_key(c.z, gp.lz, gp.gn);
  int dx_lo, dx_hi, dy_lo, dy_hi, dz_lo, dz_hi;
  so_halo_span(gp.gn, gp.cw_x, gp.cw_y, gp.cw_z, r_cap,
               &dx_lo, &dx_hi, &dy_lo, &dy_hi, &dz_lo, &dz_hi);
  const int nx_w = dx_hi - dx_lo + 1;
  const int ny_w = dy_hi - dy_lo + 1;
  const int nz_w = dz_hi - dz_lo + 1;
  const int ncells_visit = nx_w * ny_w * nz_w;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  for (int ci = warp_id; ci < ncells_visit; ci += 8) {
    const int ix = ci / (ny_w * nz_w);
    const int rem = ci - ix * (ny_w * nz_w);
    const int iy = rem / nz_w;
    const int iz = rem - iy * nz_w;
    const int gx = wrap_cell(icx + dx_lo + ix, gp.gn);
    const int gy = wrap_cell(icy + dy_lo + iy, gp.gn);
    const int gz = wrap_cell(icz + dz_lo + iz, gp.gn);
    const std::size_t cid = (std::size_t(gx) * std::size_t(gp.gn) + std::size_t(gy)) * std::size_t(gp.gn) + std::size_t(gz);
    const std::uint32_t beg = cell_offsets[cid];
    const std::uint32_t end = cell_offsets[cid + 1];
    for (std::uint32_t p_base = beg; p_base < end; p_base += 32U) {
      const std::uint32_t p = p_base + std::uint32_t(lane);
      bool hit = false;
      double r2 = 0.0;
      if (p < end) {
        r2 = so_dist2(pos_s[p], c, gp.lx, gp.ly, gp.lz);
        hit = (r2 <= r_cap2);
      }
      const unsigned mask = __ballot_sync(0xFFFFFFFFU, hit);
      if (mask != 0U) {
        std::uint32_t base_out = 0U;
        if (lane == 0) base_out = atomicAdd(&s_write, static_cast<std::uint32_t>(__popc(mask)));
        base_out = __shfl_sync(0xFFFFFFFFU, base_out, 0);
        if (hit) {
          const unsigned lower = (1U << lane) - 1U;
          const std::uint32_t out_idx = base_out + static_cast<std::uint32_t>(__popc(mask & lower));
          r2_in[out_idx] = r2;
          if (m_in) m_in[out_idx] = mass_s[p];
        }
      }
    }
  }
}

__device__ __forceinline__ void so_finish_threshold_uniform(
    const double* r2_arr, int K, long long k_last, double uniform_mass, double thresh,
    double* out_m, double* out_r, bool* saturated) {
  constexpr double four_thirds_pi = 4.18879020478639098462;
  *saturated = false;
  *out_m = 0.0;
  *out_r = 0.0;
  if (thresh <= 0.0 || K <= 0 || k_last < 0LL) return;
  const double cum_last = double(k_last + 1LL) * uniform_mass;
  if (k_last + 1LL == static_cast<long long>(K)) {
    *saturated = true;
    *out_r = sqrt(r2_arr[K - 1]);
    *out_m = double(K) * uniform_mass;
    return;
  }
  const double r0 = sqrt(r2_arr[k_last]);
  const double rho0 = cum_last / (four_thirds_pi * r0 * r0 * r0);
  const double cum1 = cum_last + uniform_mass;
  const double r1 = sqrt(r2_arr[k_last + 1LL]);
  const double rho1 = cum1 / (four_thirds_pi * r1 * r1 * r1);
  if (r1 > r0 && rho0 > thresh && rho1 > 0.0 && rho1 < rho0) {
    const double l0 = log(rho0), l1 = log(rho1), lt = log(thresh);
    const double t = fmin(1.0, fmax(0.0, (lt - l0) / (l1 - l0)));
    const double r_so = exp(log(r0) + t * (log(r1) - log(r0)));
    *out_r = r_so;
    *out_m = four_thirds_pi * r_so * r_so * r_so * thresh;
  } else {
    *out_r = r0;
    *out_m = cum_last;
  }
}

__device__ __forceinline__ void so_finish_threshold_nonuniform(
    const double* r2_arr, const float* m_arr, int K, long long k_last,
    double cum_last, double total_m, double thresh,
    double* out_m, double* out_r, bool* saturated) {
  constexpr double four_thirds_pi = 4.18879020478639098462;
  *saturated = false;
  *out_m = 0.0;
  *out_r = 0.0;
  if (thresh <= 0.0 || K <= 0 || k_last < 0LL) return;
  if (k_last + 1LL == static_cast<long long>(K)) {
    *saturated = true;
    *out_r = sqrt(r2_arr[K - 1]);
    *out_m = total_m;
    return;
  }
  const double r0 = sqrt(r2_arr[k_last]);
  const double rho0 = cum_last / (four_thirds_pi * r0 * r0 * r0);
  const double cum1 = cum_last + double(m_arr[k_last + 1LL]);
  const double r1 = sqrt(r2_arr[k_last + 1LL]);
  const double rho1 = cum1 / (four_thirds_pi * r1 * r1 * r1);
  if (r1 > r0 && rho0 > thresh && rho1 > 0.0 && rho1 < rho0) {
    const double l0 = log(rho0), l1 = log(rho1), lt = log(thresh);
    const double t = fmin(1.0, fmax(0.0, (lt - l0) / (l1 - l0)));
    const double r_so = exp(log(r0) + t * (log(r1) - log(r0)));
    *out_r = r_so;
    *out_m = four_thirds_pi * r_so * r_so * r_so * thresh;
  } else {
    *out_r = r0;
    *out_m = cum_last;
  }
}

__global__ void so_solve_uniform_light_kernel(const double* r2_out, const int* d_off,
                                              const int* light_bi, int num_light,
                                              SoGridParams gp,
                                              FofSoRow* batch_so, unsigned char* batch_sat) {
  constexpr double four_thirds_pi = 4.18879020478639098462;
  const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp_global >= num_light) return;
  const int lane = threadIdx.x & 31;
  const int bi = light_bi[warp_global];
  const int beg = d_off[bi];
  const int K = d_off[bi + 1] - beg;

  long long best_kc = -1LL, best_km = -1LL;
  for (int k = lane; k < K; k += 32) {
    const double r2 = r2_out[beg + k];
    if (r2 > 0.0) {
      const double r = sqrt(r2);
      const double rho = (double(k + 1) * gp.uniform_mass) / (four_thirds_pi * r2 * r);
      if (gp.thresh_c > 0.0 && rho >= gp.thresh_c) best_kc = static_cast<long long>(k);
      if (gp.thresh_m > 0.0 && rho >= gp.thresh_m) best_km = static_cast<long long>(k);
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    const long long okc = __shfl_down_sync(0xFFFFFFFFU, best_kc, off);
    if (okc > best_kc) best_kc = okc;
    const long long okm = __shfl_down_sync(0xFFFFFFFFU, best_km, off);
    if (okm > best_km) best_km = okm;
  }
  if (lane == 0) {
    FofSoRow row{};
    bool sc = false, sm = false;
    so_finish_threshold_uniform(r2_out + beg, K, best_kc, gp.uniform_mass, gp.thresh_c,
                                &row.m200c, &row.r200c, &sc);
    so_finish_threshold_uniform(r2_out + beg, K, best_km, gp.uniform_mass, gp.thresh_m,
                                &row.m200m, &row.r200m, &sm);
    batch_so[bi] = row;
    batch_sat[bi] = (sc || sm) ? 1 : 0;
  }
}

__global__ void so_solve_uniform_heavy_kernel(const double* r2_out, const int* d_off,
                                              const int* heavy_bi, int num_heavy,
                                              SoGridParams gp,
                                              FofSoRow* batch_so, unsigned char* batch_sat) {
  constexpr double four_thirds_pi = 4.18879020478639098462;
  if (blockIdx.x >= num_heavy) return;
  const int bi = heavy_bi[blockIdx.x];
  const int beg = d_off[bi];
  const int K = d_off[bi + 1] - beg;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  long long best_kc = -1LL, best_km = -1LL;
  for (int k = tid; k < K; k += blockDim.x) {
    const double r2 = r2_out[beg + k];
    if (r2 > 0.0) {
      const double r = sqrt(r2);
      const double rho = (double(k + 1) * gp.uniform_mass) / (four_thirds_pi * r2 * r);
      if (gp.thresh_c > 0.0 && rho >= gp.thresh_c) best_kc = static_cast<long long>(k);
      if (gp.thresh_m > 0.0 && rho >= gp.thresh_m) best_km = static_cast<long long>(k);
    }
  }
  for (int off = 16; off > 0; off >>= 1) {
    const long long okc = __shfl_down_sync(0xFFFFFFFFU, best_kc, off);
    if (okc > best_kc) best_kc = okc;
    const long long okm = __shfl_down_sync(0xFFFFFFFFU, best_km, off);
    if (okm > best_km) best_km = okm;
  }
  __shared__ long long s_kc[8], s_km[8];
  if (lane == 0) {
    s_kc[warp_id] = best_kc;
    s_km[warp_id] = best_km;
  }
  __syncthreads();
  if (tid == 0) {
    for (int w = 1; w < 8; ++w) {
      if (s_kc[w] > best_kc) best_kc = s_kc[w];
      if (s_km[w] > best_km) best_km = s_km[w];
    }
    FofSoRow row{};
    bool sc = false, sm = false;
    so_finish_threshold_uniform(r2_out + beg, K, best_kc, gp.uniform_mass, gp.thresh_c,
                                &row.m200c, &row.r200c, &sc);
    so_finish_threshold_uniform(r2_out + beg, K, best_km, gp.uniform_mass, gp.thresh_m,
                                &row.m200m, &row.r200m, &sm);
    batch_so[bi] = row;
    batch_sat[bi] = (sc || sm) ? 1 : 0;
  }
}

__global__ void so_solve_nonuniform_light_kernel(const double* r2_out, const float* m_out,
                                                 const int* d_off,
                                                 const int* light_bi, int num_light,
                                                 SoGridParams gp,
                                                 FofSoRow* batch_so, unsigned char* batch_sat) {
  constexpr double four_thirds_pi = 4.18879020478639098462;
  const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp_global >= num_light) return;
  const int lane = threadIdx.x & 31;
  const int bi = light_bi[warp_global];
  const int beg = d_off[bi];
  const int K = d_off[bi + 1] - beg;

  double running_m = 0.0;
  long long best_kc = -1LL, best_km = -1LL;
  double best_cc = 0.0, best_cm = 0.0;

  for (int base = 0; base < K; base += 32) {
    const int k = base + lane;
    const double mp = (k < K) ? double(m_out[beg + k]) : 0.0;
    double val = mp;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
      const double up = __shfl_up_sync(0xFFFFFFFFU, val, off);
      if (lane >= off) val += up;
    }
    const double tile_total = __shfl_sync(0xFFFFFFFFU, val, 31);
    const double cum_m = running_m + val;
    if (k < K) {
      const double r2 = r2_out[beg + k];
      if (r2 > 0.0) {
        const double r = sqrt(r2);
        const double rho = cum_m / (four_thirds_pi * r2 * r);
        if (gp.thresh_c > 0.0 && rho >= gp.thresh_c) {
          best_kc = static_cast<long long>(k);
          best_cc = cum_m;
        }
        if (gp.thresh_m > 0.0 && rho >= gp.thresh_m) {
          best_km = static_cast<long long>(k);
          best_cm = cum_m;
        }
      }
    }
    running_m += tile_total;
  }

  for (int off = 16; off > 0; off >>= 1) {
    const long long okc = __shfl_down_sync(0xFFFFFFFFU, best_kc, off);
    const double occ = __shfl_down_sync(0xFFFFFFFFU, best_cc, off);
    if (okc > best_kc) { best_kc = okc; best_cc = occ; }
    const long long okm = __shfl_down_sync(0xFFFFFFFFU, best_km, off);
    const double ocm = __shfl_down_sync(0xFFFFFFFFU, best_cm, off);
    if (okm > best_km) { best_km = okm; best_cm = ocm; }
  }

  if (lane == 0) {
    FofSoRow row{};
    bool sc = false, sm = false;
    so_finish_threshold_nonuniform(r2_out + beg, m_out + beg, K, best_kc, best_cc, running_m,
                                   gp.thresh_c, &row.m200c, &row.r200c, &sc);
    so_finish_threshold_nonuniform(r2_out + beg, m_out + beg, K, best_km, best_cm, running_m,
                                   gp.thresh_m, &row.m200m, &row.r200m, &sm);
    batch_so[bi] = row;
    batch_sat[bi] = (sc || sm) ? 1 : 0;
  }
}

__global__ void so_solve_nonuniform_heavy_kernel(const double* r2_out, const float* m_out,
                                                 const int* d_off,
                                                 const int* heavy_bi, int num_heavy,
                                                 SoGridParams gp,
                                                 FofSoRow* batch_so, unsigned char* batch_sat) {
  constexpr double four_thirds_pi = 4.18879020478639098462;
  if (blockIdx.x >= num_heavy) return;
  const int bi = heavy_bi[blockIdx.x];
  const int beg = d_off[bi];
  const int K = d_off[bi + 1] - beg;
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;

  using BlockScanD = cub::BlockScan<double, threads>;
  __shared__ typename BlockScanD::TempStorage s_scan;
  __shared__ long long s_kc[8], s_km[8];
  __shared__ double s_cc[8], s_cm[8];

  double running_m = 0.0;
  long long best_kc = -1LL, best_km = -1LL;
  double best_cc = 0.0, best_cm = 0.0;

  for (int base = 0; base < K; base += threads) {
    const int k = base + tid;
    const double mp = (k < K) ? double(m_out[beg + k]) : 0.0;
    double tile_cum = 0.0, tile_total = 0.0;
    BlockScanD(s_scan).InclusiveSum(mp, tile_cum, tile_total);
    __syncthreads();
    const double cum_m = running_m + tile_cum;
    if (k < K) {
      const double r2 = r2_out[beg + k];
      if (r2 > 0.0) {
        const double r = sqrt(r2);
        const double rho = cum_m / (four_thirds_pi * r2 * r);
        if (gp.thresh_c > 0.0 && rho >= gp.thresh_c) {
          best_kc = static_cast<long long>(k);
          best_cc = cum_m;
        }
        if (gp.thresh_m > 0.0 && rho >= gp.thresh_m) {
          best_km = static_cast<long long>(k);
          best_cm = cum_m;
        }
      }
    }
    running_m += tile_total;
  }

  for (int off = 16; off > 0; off >>= 1) {
    const long long okc = __shfl_down_sync(0xFFFFFFFFU, best_kc, off);
    const double occ = __shfl_down_sync(0xFFFFFFFFU, best_cc, off);
    if (okc > best_kc) { best_kc = okc; best_cc = occ; }
    const long long okm = __shfl_down_sync(0xFFFFFFFFU, best_km, off);
    const double ocm = __shfl_down_sync(0xFFFFFFFFU, best_cm, off);
    if (okm > best_km) { best_km = okm; best_cm = ocm; }
  }
  if (lane == 0) {
    s_kc[warp_id] = best_kc; s_cc[warp_id] = best_cc;
    s_km[warp_id] = best_km; s_cm[warp_id] = best_cm;
  }
  __syncthreads();

  if (tid == 0) {
    for (int w = 1; w < 8; ++w) {
      if (s_kc[w] > best_kc) { best_kc = s_kc[w]; best_cc = s_cc[w]; }
      if (s_km[w] > best_km) { best_km = s_km[w]; best_cm = s_cm[w]; }
    }
    FofSoRow row{};
    bool sc = false, sm = false;
    so_finish_threshold_nonuniform(r2_out + beg, m_out + beg, K, best_kc, best_cc, running_m,
                                   gp.thresh_c, &row.m200c, &row.r200c, &sc);
    so_finish_threshold_nonuniform(r2_out + beg, m_out + beg, K, best_km, best_cm, running_m,
                                   gp.thresh_m, &row.m200m, &row.r200m, &sm);
    batch_so[bi] = row;
    batch_sat[bi] = (sc || sm) ? 1 : 0;
  }
}

}  // namespace grav_fof

namespace {

// OpenMP CPU fallback for compute_fof_so, used ONLY if sorted positions exceed GPU VRAM
int compute_fof_so_cpu_fallback(const double* pos_xyz, const float* masses_or_null,
                                double uniform_mass, std::uint64_t count,
                                const double* centers_xyz, const std::uint64_t* sizes,
                                const double* fof_masses, const double* r_max_arr,
                                std::size_t ng, int gn,
                                double lx, double ly, double lz,
                                double thresh_c, double thresh_m, double min_thresh,
                                int min_so_size, FofSoRow* out_so) {
  const std::size_t ncells = std::size_t(gn) * std::size_t(gn) * std::size_t(gn);
  std::vector<std::uint32_t> cell_offsets(ncells + 1, 0);
  for (std::uint64_t i = 0; i < count; ++i) {
    const double px = host_in_box(pos_xyz[3 * i], lx);
    const double py = host_in_box(pos_xyz[3 * i + 1], ly);
    const double pz = host_in_box(pos_xyz[3 * i + 2], lz);
    if (!(std::isfinite(px) && std::isfinite(py) && std::isfinite(pz))) {
      return int(cudaErrorInvalidValue);
    }
    if (masses_or_null && (!std::isfinite(masses_or_null[i]) || masses_or_null[i] < 0.0f)) {
      return int(cudaErrorInvalidValue);
    }
    const int cx = host_axis_key(px, lx, gn);
    const int cy = host_axis_key(py, ly, gn);
    const int cz = host_axis_key(pz, lz, gn);
    const std::size_t cid = (std::size_t(cx) * std::size_t(gn) + std::size_t(cy)) * std::size_t(gn) + std::size_t(cz);
    ++cell_offsets[cid + 1];
  }
  for (std::size_t c = 0; c < ncells; ++c) cell_offsets[c + 1] += cell_offsets[c];

  std::vector<std::uint32_t> write_pos = cell_offsets;
  std::vector<std::uint32_t> cell_pids(count);
  for (std::uint64_t i = 0; i < count; ++i) {
    const double px = host_in_box(pos_xyz[3 * i], lx);
    const double py = host_in_box(pos_xyz[3 * i + 1], ly);
    const double pz = host_in_box(pos_xyz[3 * i + 2], lz);
    const int cx = host_axis_key(px, lx, gn);
    const int cy = host_axis_key(py, ly, gn);
    const int cz = host_axis_key(pz, lz, gn);
    const std::size_t cid = (std::size_t(cx) * std::size_t(gn) + std::size_t(cy)) * std::size_t(gn) + std::size_t(cz);
    cell_pids[write_pos[cid]++] = static_cast<std::uint32_t>(i);
  }
  std::vector<std::uint32_t>().swap(write_pos);

  constexpr double four_thirds_pi = 4.18879020478639098462;
  const double min_L = std::min({lx, ly, lz});
  const double cw_x = lx / double(gn);
  const double cw_y = ly / double(gn);
  const double cw_z = lz / double(gn);

#ifdef _OPENMP
#pragma omp parallel if(ng > 64)
#endif
  {
    std::vector<std::pair<double, double>> rm_list;
    rm_list.reserve(4096);
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 32)
#endif
    for (std::int64_t hi = 0; hi < std::int64_t(ng); ++hi) {
      const std::size_t h = static_cast<std::size_t>(hi);
      if ((min_so_size > 0 && sizes[h] < std::uint64_t(min_so_size)) || fof_masses[h] <= 0.0) {
        continue;
      }
      const double cx0 = host_in_box(centers_xyz[3 * h], lx);
      const double cy0 = host_in_box(centers_xyz[3 * h + 1], ly);
      const double cz0 = host_in_box(centers_xyz[3 * h + 2], lz);
      const int icx = host_axis_key(cx0, lx, gn);
      const int icy = host_axis_key(cy0, ly, gn);
      const int icz = host_axis_key(cz0, lz, gn);

      const double r_est = std::cbrt(1.5 * fof_masses[h] / (four_thirds_pi * min_thresh));
      double r_cap = std::min(0.25 * min_L, std::max(r_est, 1.25 * r_max_arr[h]));
      r_cap = std::max(r_cap, 1e-6 * min_L);

      FofSoRow so_res{};
      for (int attempt = 0; attempt < 3; ++attempt) {
        const double r_cap2 = r_cap * r_cap;
        const int rx = std::min(gn / 2, int(std::ceil(r_cap / cw_x)));
        const int ry = std::min(gn / 2, int(std::ceil(r_cap / cw_y)));
        const int rz = std::min(gn / 2, int(std::ceil(r_cap / cw_z)));
        const int dx_lo = -std::min(rx, (gn - 1) / 2), dx_hi = std::min(rx, gn / 2);
        const int dy_lo = -std::min(ry, (gn - 1) / 2), dy_hi = std::min(ry, gn / 2);
        const int dz_lo = -std::min(rz, (gn - 1) / 2), dz_hi = std::min(rz, gn / 2);

        rm_list.clear();
        for (int ddx = dx_lo; ddx <= dx_hi; ++ddx) {
          const int gx = ((icx + ddx) % gn + gn) % gn;
          for (int ddy = dy_lo; ddy <= dy_hi; ++ddy) {
            const int gy = ((icy + ddy) % gn + gn) % gn;
            for (int ddz = dz_lo; ddz <= dz_hi; ++ddz) {
              const int gz = ((icz + ddz) % gn + gn) % gn;
              const std::size_t cid = (std::size_t(gx) * std::size_t(gn) + std::size_t(gy)) * std::size_t(gn) + std::size_t(gz);
              const std::uint32_t beg = cell_offsets[cid];
              const std::uint32_t end = cell_offsets[cid + 1];
              for (std::uint32_t p = beg; p < end; ++p) {
                const std::uint32_t pid = cell_pids[p];
                const double px = host_in_box(pos_xyz[3 * std::uint64_t(pid)], lx);
                const double py = host_in_box(pos_xyz[3 * std::uint64_t(pid) + 1ULL], ly);
                const double pz = host_in_box(pos_xyz[3 * std::uint64_t(pid) + 2ULL], lz);
                const double dx = host_nearest_image(px - cx0, lx);
                const double dy = host_nearest_image(py - cy0, ly);
                const double dz = host_nearest_image(pz - cz0, lz);
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 <= r_cap2) {
                  const double mp = masses_or_null ? double(masses_or_null[pid]) : uniform_mass;
                  rm_list.emplace_back(r2, mp);
                }
              }
            }
          }
        }
        std::sort(rm_list.begin(), rm_list.end(), [](const auto& a, const auto& b) {
          return a.first < b.first;
        });

        const std::size_t K = rm_list.size();
        auto solve_so = [&](double thresh, double& out_m, double& out_r, bool& saturated) {
          saturated = false;
          out_m = 0.0;
          out_r = 0.0;
          if (thresh <= 0.0 || K == 0) return;
          std::int64_t k_last = -1;
          double cum_m = 0.0, cum_last = 0.0, total_m = 0.0;
          for (std::size_t k = 0; k < K; ++k) {
            cum_m += rm_list[k].second;
            const double r2 = rm_list[k].first;
            if (r2 > 0.0) {
              const double r = std::sqrt(r2);
              const double rho = cum_m / (four_thirds_pi * r2 * r);
              if (rho >= thresh) {
                k_last = std::int64_t(k);
                cum_last = cum_m;
              }
            }
          }
          total_m = cum_m;
          if (k_last < 0) return;
          if (k_last + 1 == std::int64_t(K)) {
            saturated = true;
            out_r = std::sqrt(rm_list.back().first);
            out_m = total_m;
            return;
          }
          const double r0 = std::sqrt(rm_list[k_last].first);
          const double rho0 = cum_last / (four_thirds_pi * r0 * r0 * r0);
          const double cum1 = cum_last + rm_list[k_last + 1].second;
          const double r1 = std::sqrt(rm_list[k_last + 1].first);
          const double rho1 = cum1 / (four_thirds_pi * r1 * r1 * r1);
          if (r1 > r0 && rho0 > thresh && rho1 > 0.0 && rho1 < rho0) {
            const double l0 = std::log(rho0), l1 = std::log(rho1), lt = std::log(thresh);
            const double t = std::clamp((lt - l0) / (l1 - l0), 0.0, 1.0);
            const double r_so = std::exp(std::log(r0) + t * (std::log(r1) - std::log(r0)));
            out_r = r_so;
            out_m = four_thirds_pi * r_so * r_so * r_so * thresh;
          } else {
            out_r = r0;
            out_m = cum_last;
          }
        };

        bool sat_c = false, sat_m = false;
        solve_so(thresh_c, so_res.m200c, so_res.r200c, sat_c);
        solve_so(thresh_m, so_res.m200m, so_res.r200m, sat_m);
        if ((sat_c || sat_m) && r_cap < 0.25 * min_L && attempt < 2) {
          r_cap = std::min(0.25 * min_L, r_cap * 2.0);
          continue;
        }
        break;
      }
      out_so[h] = so_res;
    }
  }
  return 0;
}

}  // namespace

extern "C" int compute_fof_catalog(const double* pos_xyz, const float* vel_xyz,
                                   const float* masses_or_null, double uniform_mass,
                                   const std::int64_t* group_ids,
                                   std::uint64_t count, std::uint64_t num_groups,
                                   double lx, double ly, double lz,
                                   FofHaloRow* out_halos) {
  static_assert(sizeof(FofHaloRow) == 88, "FofHaloRow layout mismatch");
  try {
    if (num_groups == 0) return 0;
    if (!pos_xyz || !group_ids || !out_halos || count == 0 ||
        lx <= 0.0 || ly <= 0.0 || lz <= 0.0) {
      return int(cudaErrorInvalidValue);
    }
    if (!masses_or_null && !(std::isfinite(uniform_mass) && uniform_mass > 0.0)) {
      return int(cudaErrorInvalidValue);
    }

    const std::size_t ng = static_cast<std::size_t>(num_groups);
    grav_fof::check(cudaFree(nullptr));
    std::size_t free_mem = 0, total_mem = 0;
    grav_fof::check(cudaMemGetInfo(&free_mem, &total_mem));

    grav_fof::Buffer<double3> d_anchor(ng);
    grav_fof::Buffer<unsigned char> d_has_anchor(ng);
    grav_fof::Buffer<std::uint32_t> d_chunk_anchor_j(ng);
    grav_fof::Buffer<unsigned long long> d_size(ng);
    grav_fof::Buffer<double> d_mass(ng), d_com_x(ng), d_com_y(ng), d_com_z(ng);
    grav_fof::Buffer<double> d_bvel_x(ng), d_bvel_y(ng), d_bvel_z(ng);
    grav_fof::Buffer<double> d_max_r2(ng), d_sum_mv2(ng);
    grav_fof::Buffer<int> d_invalid(1);

    grav_fof::check(cudaMemset(d_anchor.ptr, 0, ng * sizeof(double3)));
    grav_fof::check(cudaMemset(d_has_anchor.ptr, 0, ng * sizeof(unsigned char)));
    grav_fof::check(cudaMemset(d_size.ptr, 0, ng * sizeof(unsigned long long)));
    grav_fof::check(cudaMemset(d_mass.ptr, 0, ng * sizeof(double)));
    grav_fof::check(cudaMemset(d_com_x.ptr, 0, ng * sizeof(double)));
    grav_fof::check(cudaMemset(d_com_y.ptr, 0, ng * sizeof(double)));
    grav_fof::check(cudaMemset(d_com_z.ptr, 0, ng * sizeof(double)));
    grav_fof::check(cudaMemset(d_bvel_x.ptr, 0, ng * sizeof(double)));
    grav_fof::check(cudaMemset(d_bvel_y.ptr, 0, ng * sizeof(double)));
    grav_fof::check(cudaMemset(d_bvel_z.ptr, 0, ng * sizeof(double)));
    grav_fof::check(cudaMemset(d_max_r2.ptr, 0, ng * sizeof(double)));
    grav_fof::check(cudaMemset(d_sum_mv2.ptr, 0, ng * sizeof(double)));
    grav_fof::check(cudaMemset(d_invalid.ptr, 0, sizeof(int)));

    grav_fof::check(cudaMemGetInfo(&free_mem, &total_mem));
    std::uint64_t chunk_max = std::min<std::uint64_t>(
        count, std::min<std::uint64_t>(32ULL << 20, std::max<std::uint64_t>(1ULL << 20, free_mem / 128ULL)));
    if (const char* env_c = std::getenv("GRAV_FOF_CATALOG_CHUNK")) {
      const std::uint64_t v = std::strtoull(env_c, nullptr, 10);
      if (v > 0ULL) chunk_max = std::min(count, v);
    }
    const std::size_t buf_n = static_cast<std::size_t>(chunk_max);
    const bool single_chunk = (chunk_max >= count);

    grav_fof::Buffer<double3> d_pos(buf_n);
    grav_fof::Buffer<float3> d_vel(vel_xyz ? buf_n : 0ULL);
    grav_fof::Buffer<float> d_masses(masses_or_null ? buf_n : 0ULL);
    grav_fof::Buffer<std::int64_t> d_gids(buf_n);

    // Pass 1 (anchor selection + mass-weighted CoM & bulk velocity)
    for (std::uint64_t base = 0; base < count; base += chunk_max) {
      const std::size_t cur = static_cast<std::size_t>(std::min<std::uint64_t>(chunk_max, count - base));
      grav_fof::check(cudaMemcpy(d_pos.ptr, pos_xyz + base * 3ULL, cur * sizeof(double3), cudaMemcpyHostToDevice));
      if (vel_xyz) {
        grav_fof::check(cudaMemcpy(d_vel.ptr, vel_xyz + base * 3ULL, cur * sizeof(float3), cudaMemcpyHostToDevice));
      }
      if (masses_or_null) {
        grav_fof::check(cudaMemcpy(d_masses.ptr, masses_or_null + base, cur * sizeof(float), cudaMemcpyHostToDevice));
      }
      grav_fof::check(cudaMemcpy(d_gids.ptr, group_ids + base, cur * sizeof(std::int64_t), cudaMemcpyHostToDevice));

      grav_fof::init_u32_val_kernel<<<grav_fof::grid_for_u64(ng), grav_fof::threads>>>(
          d_chunk_anchor_j.ptr, ng, std::uint32_t(0xFFFFFFFFU));
      grav_fof::catalog_find_chunk_anchors_kernel<<<grav_fof::grid_for_u64(cur), grav_fof::threads>>>(
          d_gids.ptr, cur, num_groups, d_has_anchor.ptr, d_chunk_anchor_j.ptr);
      grav_fof::catalog_commit_anchors_kernel<<<grav_fof::grid_for_u64(ng), grav_fof::threads>>>(
          d_pos.ptr, cur, ng, lx, ly, lz, d_has_anchor.ptr, d_chunk_anchor_j.ptr, d_anchor.ptr);
      grav_fof::catalog_pass1_kernel<<<grav_fof::grid_for_u64(cur), grav_fof::threads>>>(
          d_pos.ptr, vel_xyz ? d_vel.ptr : nullptr, masses_or_null ? d_masses.ptr : nullptr, uniform_mass,
          d_gids.ptr, cur, num_groups, lx, ly, lz, d_anchor.ptr,
          d_size.ptr, d_mass.ptr, d_com_x.ptr, d_com_y.ptr, d_com_z.ptr,
          d_bvel_x.ptr, d_bvel_y.ptr, d_bvel_z.ptr, d_invalid.ptr);
      if (single_chunk) break;
    }

    int invalid_h = 0;
    grav_fof::check(cudaMemcpy(&invalid_h, d_invalid.ptr, sizeof(int), cudaMemcpyDeviceToHost));
    if (invalid_h) return int(cudaErrorInvalidValue);

    grav_fof::catalog_finalize_pass1_kernel<<<grav_fof::grid_for_u64(ng), grav_fof::threads>>>(
        ng, lx, ly, lz, d_anchor.ptr, d_size.ptr, d_mass.ptr,
        d_com_x.ptr, d_com_y.ptr, d_com_z.ptr, d_bvel_x.ptr, d_bvel_y.ptr, d_bvel_z.ptr,
        d_invalid.ptr);
    grav_fof::check(cudaMemcpy(&invalid_h, d_invalid.ptr, sizeof(int), cudaMemcpyDeviceToHost));
    if (invalid_h) return int(cudaErrorInvalidValue);

    d_anchor.release();
    d_has_anchor.release();
    d_chunk_anchor_j.release();
    d_invalid.release();

    // Pass 2 (Rmax & mass-weighted 3D velocity dispersion around finalized CoM)
    for (std::uint64_t base = 0; base < count; base += chunk_max) {
      const std::size_t cur = static_cast<std::size_t>(std::min<std::uint64_t>(chunk_max, count - base));
      if (!single_chunk) {
        grav_fof::check(cudaMemcpy(d_pos.ptr, pos_xyz + base * 3ULL, cur * sizeof(double3), cudaMemcpyHostToDevice));
        if (vel_xyz) {
          grav_fof::check(cudaMemcpy(d_vel.ptr, vel_xyz + base * 3ULL, cur * sizeof(float3), cudaMemcpyHostToDevice));
        }
        if (masses_or_null) {
          grav_fof::check(cudaMemcpy(d_masses.ptr, masses_or_null + base, cur * sizeof(float), cudaMemcpyHostToDevice));
        }
        grav_fof::check(cudaMemcpy(d_gids.ptr, group_ids + base, cur * sizeof(std::int64_t), cudaMemcpyHostToDevice));
      }
      grav_fof::catalog_pass2_kernel<<<grav_fof::grid_for_u64(cur), grav_fof::threads>>>(
          d_pos.ptr, vel_xyz ? d_vel.ptr : nullptr, masses_or_null ? d_masses.ptr : nullptr, uniform_mass,
          d_gids.ptr, cur, num_groups, lx, ly, lz,
          d_com_x.ptr, d_com_y.ptr, d_com_z.ptr,
          d_bvel_x.ptr, d_bvel_y.ptr, d_bvel_z.ptr,
          d_max_r2.ptr, d_sum_mv2.ptr);
      if (single_chunk) break;
    }

    d_pos.release();
    d_vel.release();
    d_masses.release();
    d_gids.release();

    grav_fof::Buffer<FofHaloRow> d_out(ng);
    grav_fof::catalog_pack_rows_kernel<<<grav_fof::grid_for_u64(ng), grav_fof::threads>>>(
        ng, d_size.ptr, d_mass.ptr, d_com_x.ptr, d_com_y.ptr, d_com_z.ptr,
        d_bvel_x.ptr, d_bvel_y.ptr, d_bvel_z.ptr, d_max_r2.ptr, d_sum_mv2.ptr, d_out.ptr);
    grav_fof::check(cudaMemcpy(out_halos, d_out.ptr, ng * sizeof(FofHaloRow), cudaMemcpyDeviceToHost));
    return 0;
  } catch (const grav_fof::GpuError& err) {
    return int(err.status);
  } catch (const std::bad_alloc&) {
    return int(cudaErrorMemoryAllocation);
  } catch (...) {
    return int(cudaErrorUnknown);
  }
}

extern "C" int compute_fof_so(const double* pos_xyz, const float* masses_or_null,
                              double uniform_mass, std::uint64_t count,
                              const double* centers_xyz, const std::uint64_t* sizes,
                              const double* fof_masses, const double* r_max_arr,
                              std::uint64_t num_groups,
                              double lx, double ly, double lz,
                              double rho_crit_com, double rho_mean_com,
                              int min_so_size, FofSoRow* out_so) {
  static_assert(sizeof(FofSoRow) == 32, "FofSoRow layout mismatch");
  try {
    if (num_groups == 0) return 0;
    if (!pos_xyz || !centers_xyz || !sizes || !fof_masses || !r_max_arr || !out_so ||
        count == 0 || lx <= 0.0 || ly <= 0.0 || lz <= 0.0 ||
        count > std::uint64_t(std::numeric_limits<std::uint32_t>::max())) {
      return int(cudaErrorInvalidValue);
    }
    if (!masses_or_null && !(std::isfinite(uniform_mass) && uniform_mass > 0.0)) {
      return int(cudaErrorInvalidValue);
    }
    if (!std::isfinite(rho_crit_com) || rho_crit_com < 0.0 ||
        !std::isfinite(rho_mean_com) || rho_mean_com < 0.0) {
      return int(cudaErrorInvalidValue);
    }

    const std::size_t ng = static_cast<std::size_t>(num_groups);
    std::memset(out_so, 0, ng * sizeof(FofSoRow));
    const double thresh_c = (rho_crit_com > 0.0) ? (200.0 * rho_crit_com) : 0.0;
    const double thresh_m = (rho_mean_com > 0.0) ? (200.0 * rho_mean_com) : 0.0;
    if (thresh_c <= 0.0 && thresh_m <= 0.0) return 0;

    double min_thresh = std::numeric_limits<double>::max();
    if (thresh_c > 0.0) min_thresh = std::min(min_thresh, thresh_c);
    if (thresh_m > 0.0) min_thresh = std::min(min_thresh, thresh_m);

    const int gn = std::clamp(int(std::cbrt(double(count) / 32.0)), 8, 128);
    const std::size_t ncells = std::size_t(gn) * std::size_t(gn) * std::size_t(gn);

    grav_fof::check(cudaFree(nullptr));
    std::size_t free_mem = 0, total_mem = 0;
    grav_fof::check(cudaMemGetInfo(&free_mem, &total_mem));

    const bool non_uniform = (masses_or_null != nullptr);
    const bool force_chunk_grid = (std::getenv("GRAV_FOF_SO_CHUNK_GRID") &&
                                   std::atoi(std::getenv("GRAV_FOF_SO_CHUNK_GRID")) != 0);
    const std::uint64_t resident_bytes = count * (non_uniform ? 28ULL : 24ULL) + std::uint64_t(ncells + 1) * 4ULL;
    const std::uint64_t sort_grid_bytes = count * (non_uniform ? 64ULL : 56ULL) + std::uint64_t(ncells + 1) * 4ULL;

    if (resident_bytes > std::uint64_t(0.75 * double(free_mem))) {
      return compute_fof_so_cpu_fallback(
          pos_xyz, masses_or_null, uniform_mass, count,
          centers_xyz, sizes, fof_masses, r_max_arr,
          ng, gn, lx, ly, lz, thresh_c, thresh_m, min_thresh, min_so_size, out_so);
    }

    // Step 1: Build spatially sorted particle buffers pos_s (+ mass_s) and cell_offsets on GPU
    grav_fof::Buffer<double3> d_pos_s;
    grav_fof::Buffer<float> d_mass_s;
    grav_fof::Buffer<std::uint32_t> d_cell_offsets(ncells + 1);
    grav_fof::Buffer<int> d_invalid(1);
    grav_fof::check(cudaMemset(d_invalid.ptr, 0, sizeof(int)));

    if (!force_chunk_grid && count <= (1ULL << 30) &&
        sort_grid_bytes < std::uint64_t(0.70 * double(free_mem))) {
      const int cnt_i = static_cast<int>(count);
      grav_fof::Buffer<double3> d_pos_raw(static_cast<std::size_t>(count));
      grav_fof::check(cudaMemcpy(d_pos_raw.ptr, pos_xyz, static_cast<std::size_t>(count) * sizeof(double3),
                                  cudaMemcpyHostToDevice));
      grav_fof::Buffer<float> d_mass_raw(non_uniform ? static_cast<std::size_t>(count) : 0ULL);
      if (non_uniform) {
        grav_fof::check(cudaMemcpy(d_mass_raw.ptr, masses_or_null, static_cast<std::size_t>(count) * sizeof(float),
                                    cudaMemcpyHostToDevice));
      }

      grav_fof::Buffer<std::uint32_t> d_cid_a(static_cast<std::size_t>(count));
      grav_fof::Buffer<std::uint32_t> d_cid_b(static_cast<std::size_t>(count));
      grav_fof::Buffer<std::uint32_t> d_pid_a(static_cast<std::size_t>(count));
      grav_fof::Buffer<std::uint32_t> d_pid_b(static_cast<std::size_t>(count));
      grav_fof::Buffer<std::uint32_t> d_cell_cnt(ncells);
      grav_fof::check(cudaMemset(d_cell_cnt.ptr, 0, ncells * sizeof(std::uint32_t)));

      grav_fof::so_grid_key_kernel<<<grav_fof::grid_for_u64(count), grav_fof::threads>>>(
          d_pos_raw.ptr, non_uniform ? d_mass_raw.ptr : nullptr,
          static_cast<std::size_t>(count), gn, lx, ly, lz,
          d_cid_a.ptr, d_pid_a.ptr, d_cell_cnt.ptr, d_invalid.ptr);
      int bad = 0;
      grav_fof::check(cudaMemcpy(&bad, d_invalid.ptr, sizeof(int), cudaMemcpyDeviceToHost));
      if (bad) return int(cudaErrorInvalidValue);

      std::size_t scan_bytes = 0;
      grav_fof::check(cub::DeviceScan::ExclusiveSum(
          nullptr, scan_bytes, d_cell_cnt.ptr, d_cell_offsets.ptr, int(ncells)));
      {
        grav_fof::Buffer<unsigned char> work(scan_bytes);
        grav_fof::check(cub::DeviceScan::ExclusiveSum(
            work.ptr, scan_bytes, d_cell_cnt.ptr, d_cell_offsets.ptr, int(ncells)));
      }
      const std::uint32_t total_u32 = static_cast<std::uint32_t>(count);
      grav_fof::check(cudaMemcpy(d_cell_offsets.ptr + ncells, &total_u32, sizeof(std::uint32_t),
                                  cudaMemcpyHostToDevice));
      d_cell_cnt.release();

      int bits = 0;
      std::size_t maxc = ncells - 1ULL;
      while (maxc) { ++bits; maxc >>= 1; }
      bits = std::max(bits, 1);

      cub::DoubleBuffer<std::uint32_t> cids(d_cid_a.ptr, d_cid_b.ptr);
      cub::DoubleBuffer<std::uint32_t> pids(d_pid_a.ptr, d_pid_b.ptr);
      std::size_t sort_bytes = 0;
      grav_fof::check(cub::DeviceRadixSort::SortPairs(nullptr, sort_bytes, cids, pids, cnt_i, 0, bits));
      {
        grav_fof::Buffer<unsigned char> work(sort_bytes);
        grav_fof::check(cub::DeviceRadixSort::SortPairs(work.ptr, sort_bytes, cids, pids, cnt_i, 0, bits));
      }
      d_cid_a.release();
      d_cid_b.release();
      d_pid_a.keep_only(pids.Current());
      d_pid_b.keep_only(pids.Current());

      d_pos_s = grav_fof::Buffer<double3>(static_cast<std::size_t>(count));
      if (non_uniform) d_mass_s = grav_fof::Buffer<float>(static_cast<std::size_t>(count));
      grav_fof::so_gather_sorted_kernel<<<grav_fof::grid_for_u64(count), grav_fof::threads>>>(
          d_pos_raw.ptr, non_uniform ? d_mass_raw.ptr : nullptr,
          pids.Current(), static_cast<std::size_t>(count),
          d_pos_s.ptr, non_uniform ? d_mass_s.ptr : nullptr);
      grav_fof::check(cudaDeviceSynchronize());
    } else {
      // Low-VRAM chunked 2-pass spatial histogram + scatter (handles up to 1024^3 in 48 GB VRAM)
      const std::uint64_t gchunk = std::min<std::uint64_t>(count, 16ULL << 20);
      grav_fof::Buffer<double3> d_cpos(static_cast<std::size_t>(gchunk));
      grav_fof::Buffer<float> d_cmass(non_uniform ? static_cast<std::size_t>(gchunk) : 0ULL);
      grav_fof::Buffer<std::uint32_t> d_cell_cnt(ncells);
      grav_fof::check(cudaMemset(d_cell_cnt.ptr, 0, ncells * sizeof(std::uint32_t)));

      for (std::uint64_t base = 0; base < count; base += gchunk) {
        const std::size_t cur = static_cast<std::size_t>(std::min<std::uint64_t>(gchunk, count - base));
        grav_fof::check(cudaMemcpy(d_cpos.ptr, pos_xyz + base * 3ULL, cur * sizeof(double3), cudaMemcpyHostToDevice));
        if (non_uniform) {
          grav_fof::check(cudaMemcpy(d_cmass.ptr, masses_or_null + base, cur * sizeof(float), cudaMemcpyHostToDevice));
        }
        grav_fof::so_grid_chunk_count_kernel<<<grav_fof::grid_for_u64(cur), grav_fof::threads>>>(
            d_cpos.ptr, non_uniform ? d_cmass.ptr : nullptr, cur, gn, lx, ly, lz,
            d_cell_cnt.ptr, d_invalid.ptr);
      }
      int bad = 0;
      grav_fof::check(cudaMemcpy(&bad, d_invalid.ptr, sizeof(int), cudaMemcpyDeviceToHost));
      if (bad) return int(cudaErrorInvalidValue);

      std::size_t scan_bytes = 0;
      grav_fof::check(cub::DeviceScan::ExclusiveSum(
          nullptr, scan_bytes, d_cell_cnt.ptr, d_cell_offsets.ptr, int(ncells)));
      {
        grav_fof::Buffer<unsigned char> work(scan_bytes);
        grav_fof::check(cub::DeviceScan::ExclusiveSum(
            work.ptr, scan_bytes, d_cell_cnt.ptr, d_cell_offsets.ptr, int(ncells)));
      }
      const std::uint32_t total_u32 = static_cast<std::uint32_t>(count);
      grav_fof::check(cudaMemcpy(d_cell_offsets.ptr + ncells, &total_u32, sizeof(std::uint32_t),
                                  cudaMemcpyHostToDevice));
      grav_fof::check(cudaMemcpy(d_cell_cnt.ptr, d_cell_offsets.ptr, ncells * sizeof(std::uint32_t),
                                  cudaMemcpyDeviceToDevice));

      d_pos_s = grav_fof::Buffer<double3>(static_cast<std::size_t>(count));
      if (non_uniform) d_mass_s = grav_fof::Buffer<float>(static_cast<std::size_t>(count));

      for (std::uint64_t base = 0; base < count; base += gchunk) {
        const std::size_t cur = static_cast<std::size_t>(std::min<std::uint64_t>(gchunk, count - base));
        grav_fof::check(cudaMemcpy(d_cpos.ptr, pos_xyz + base * 3ULL, cur * sizeof(double3), cudaMemcpyHostToDevice));
        if (non_uniform) {
          grav_fof::check(cudaMemcpy(d_cmass.ptr, masses_or_null + base, cur * sizeof(float), cudaMemcpyHostToDevice));
        }
        grav_fof::so_grid_chunk_scatter_kernel<<<grav_fof::grid_for_u64(cur), grav_fof::threads>>>(
            d_cpos.ptr, non_uniform ? d_cmass.ptr : nullptr, cur, gn, lx, ly, lz,
            d_cell_cnt.ptr, d_pos_s.ptr, non_uniform ? d_mass_s.ptr : nullptr);
      }
      grav_fof::check(cudaDeviceSynchronize());
    }
    d_invalid.release();

    constexpr double four_thirds_pi = 4.18879020478639098462;
    const double min_L = std::min({lx, ly, lz});
    const double cw_x = lx / double(gn);
    const double cw_y = ly / double(gn);
    const double cw_z = lz / double(gn);
    const grav_fof::SoGridParams gp{gn, lx, ly, lz, cw_x, cw_y, cw_z, thresh_c, thresh_m, uniform_mass};

    // Build initial active halo list
    std::vector<std::size_t> act_hidx;
    std::vector<double3> act_pos;
    std::vector<double> act_rcap;
    act_hidx.reserve(ng);
    act_pos.reserve(ng);
    act_rcap.reserve(ng);

    for (std::size_t h = 0; h < ng; ++h) {
      if ((min_so_size > 0 && sizes[h] < std::uint64_t(min_so_size)) || fof_masses[h] <= 0.0) {
        continue;
      }
      const double cx0 = host_in_box(centers_xyz[3 * h], lx);
      const double cy0 = host_in_box(centers_xyz[3 * h + 1], ly);
      const double cz0 = host_in_box(centers_xyz[3 * h + 2], lz);
      const double r_est = std::cbrt(1.5 * fof_masses[h] / (four_thirds_pi * min_thresh));
      double r_cap = std::min(0.25 * min_L, std::max(r_est, 1.25 * r_max_arr[h]));
      r_cap = std::max(r_cap, 1e-6 * min_L);
      act_hidx.push_back(h);
      act_pos.push_back(double3{cx0, cy0, cz0});
      act_rcap.push_back(r_cap);
    }

    for (int attempt = 0; attempt < 3 && !act_hidx.empty(); ++attempt) {
      const int n_act = static_cast<int>(act_hidx.size());
      std::vector<int> light_ids, heavy_ids;
      light_ids.reserve(n_act);
      heavy_ids.reserve(n_act / 8 + 1);
      std::vector<std::uint8_t> is_heavy(n_act, 0);

      for (int a = 0; a < n_act; ++a) {
        int dx0, dx1, dy0, dy1, dz0, dz1;
        grav_fof::so_halo_span(gn, cw_x, cw_y, cw_z, act_rcap[a],
                                &dx0, &dx1, &dy0, &dy1, &dz0, &dz1);
        const std::uint64_t nv = std::uint64_t(dx1 - dx0 + 1) *
                                 std::uint64_t(dy1 - dy0 + 1) *
                                 std::uint64_t(dz1 - dz0 + 1);
        if (nv <= 125ULL) {
          light_ids.push_back(a);
        } else {
          is_heavy[a] = 1;
          heavy_ids.push_back(a);
        }
      }

      grav_fof::Buffer<double3> d_hpos(n_act);
      grav_fof::Buffer<double> d_hrcap(n_act);
      grav_fof::Buffer<std::uint32_t> d_hcnt(n_act);
      grav_fof::check(cudaMemcpy(d_hpos.ptr, act_pos.data(), std::size_t(n_act) * sizeof(double3), cudaMemcpyHostToDevice));
      grav_fof::check(cudaMemcpy(d_hrcap.ptr, act_rcap.data(), std::size_t(n_act) * sizeof(double), cudaMemcpyHostToDevice));

      // Pass A: Count exact enclosed pairs per active halo
      if (!light_ids.empty()) {
        const int nl = static_cast<int>(light_ids.size());
        grav_fof::Buffer<int> d_lids(nl);
        grav_fof::check(cudaMemcpy(d_lids.ptr, light_ids.data(), std::size_t(nl) * sizeof(int), cudaMemcpyHostToDevice));
        const int warps_per_blk = grav_fof::threads / 32;
        const int nblks = (nl + warps_per_blk - 1) / warps_per_blk;
        grav_fof::so_count_light_kernel<<<nblks, grav_fof::threads>>>(
            d_pos_s.ptr, d_cell_offsets.ptr, gp, d_lids.ptr, nl,
            d_hpos.ptr, d_hrcap.ptr, d_hcnt.ptr);
      }
      if (!heavy_ids.empty()) {
        const int nh = static_cast<int>(heavy_ids.size());
        grav_fof::Buffer<int> d_hids(nh);
        grav_fof::check(cudaMemcpy(d_hids.ptr, heavy_ids.data(), std::size_t(nh) * sizeof(int), cudaMemcpyHostToDevice));
        grav_fof::so_count_heavy_kernel<<<nh, grav_fof::threads>>>(
            d_pos_s.ptr, d_cell_offsets.ptr, gp, d_hids.ptr, nh,
            d_hpos.ptr, d_hrcap.ptr, d_hcnt.ptr);
      }
      grav_fof::check(cudaGetLastError());

      std::vector<std::uint32_t> h_cnt(n_act);
      grav_fof::check(cudaMemcpy(h_cnt.data(), d_hcnt.ptr, std::size_t(n_act) * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
      d_hcnt.release();

      std::size_t cur_free = 0, cur_total = 0;
      grav_fof::check(cudaMemGetInfo(&cur_free, &cur_total));
      std::uint64_t max_batch_pairs = std::min<std::uint64_t>(
          64ULL << 20,
          std::max<std::uint64_t>(1ULL << 20, cur_free / (non_uniform ? 64ULL : 48ULL)));
      if (const char* env_b = std::getenv("GRAV_FOF_SO_BATCH_PAIRS")) {
        const std::uint64_t vb = std::strtoull(env_b, nullptr, 10);
        if (vb > 0ULL) max_batch_pairs = vb;
      }

      std::vector<std::size_t> next_hidx;
      std::vector<double3> next_pos;
      std::vector<double> next_rcap;

      for (int b_beg = 0; b_beg < n_act;) {
        int b_end = b_beg;
        std::uint64_t batch_K = 0ULL;
        while (b_end < n_act) {
          const std::uint64_t nextK = batch_K + std::uint64_t(h_cnt[b_end]);
          if (b_end > b_beg && nextK > max_batch_pairs) break;
          batch_K = nextK;
          ++b_end;
          if (batch_K >= max_batch_pairs) break;
        }
        const int batch_H = b_end - b_beg;
        if (batch_K > std::uint64_t(std::numeric_limits<int>::max())) {
          return int(cudaErrorMemoryAllocation);
        }

        std::vector<FofSoRow> batch_so(batch_H);
        std::vector<unsigned char> batch_sat(batch_H, 0);

        if (batch_K > 0ULL) {
          std::vector<int> h_off(batch_H + 1, 0);
          std::vector<int> b_light, b_heavy;
          b_light.reserve(batch_H);
          b_heavy.reserve(batch_H / 8 + 1);
          for (int bi = 0; bi < batch_H; ++bi) {
            h_off[bi + 1] = h_off[bi] + static_cast<int>(h_cnt[b_beg + bi]);
            if (is_heavy[b_beg + bi]) b_heavy.push_back(bi);
            else b_light.push_back(bi);
          }

          grav_fof::Buffer<int> d_off(batch_H + 1);
          grav_fof::check(cudaMemcpy(d_off.ptr, h_off.data(), std::size_t(batch_H + 1) * sizeof(int), cudaMemcpyHostToDevice));
          grav_fof::Buffer<int> d_blight, d_bheavy;
          if (!b_light.empty()) {
            d_blight = grav_fof::Buffer<int>(b_light.size());
            grav_fof::check(cudaMemcpy(d_blight.ptr, b_light.data(), b_light.size() * sizeof(int), cudaMemcpyHostToDevice));
          }
          if (!b_heavy.empty()) {
            d_bheavy = grav_fof::Buffer<int>(b_heavy.size());
            grav_fof::check(cudaMemcpy(d_bheavy.ptr, b_heavy.data(), b_heavy.size() * sizeof(int), cudaMemcpyHostToDevice));
          }

          const std::size_t bK_sz = static_cast<std::size_t>(batch_K);
          grav_fof::Buffer<double> d_r2_in(bK_sz), d_r2_out(bK_sz);
          grav_fof::Buffer<float> d_m_in(non_uniform ? bK_sz : 0ULL);
          grav_fof::Buffer<float> d_m_out(non_uniform ? bK_sz : 0ULL);

          // Pass B: Fill exact radial arrays
          if (!b_light.empty()) {
            const int nl = static_cast<int>(b_light.size());
            const int warps_per_blk = grav_fof::threads / 32;
            const int nblks = (nl + warps_per_blk - 1) / warps_per_blk;
            grav_fof::so_fill_light_kernel<<<nblks, grav_fof::threads>>>(
                d_pos_s.ptr, non_uniform ? d_mass_s.ptr : nullptr, d_cell_offsets.ptr, gp,
                d_blight.ptr, nl, b_beg, d_hpos.ptr, d_hrcap.ptr, d_off.ptr,
                d_r2_in.ptr, non_uniform ? d_m_in.ptr : nullptr);
          }
          if (!b_heavy.empty()) {
            const int nh = static_cast<int>(b_heavy.size());
            grav_fof::so_fill_heavy_kernel<<<nh, grav_fof::threads>>>(
                d_pos_s.ptr, non_uniform ? d_mass_s.ptr : nullptr, d_cell_offsets.ptr, gp,
                d_bheavy.ptr, nh, b_beg, d_hpos.ptr, d_hrcap.ptr, d_off.ptr,
                d_r2_in.ptr, non_uniform ? d_m_in.ptr : nullptr);
          }
          grav_fof::check(cudaGetLastError());

          // Pass C: Exact per-halo ascending sort of r2 (and paired mass if non-uniform)
          const int K_int = static_cast<int>(batch_K);
          std::size_t sort_bytes = 0;
          if (!non_uniform) {
            if (batch_H == 1) {
              grav_fof::check(cub::DeviceRadixSort::SortKeys(
                  nullptr, sort_bytes, d_r2_in.ptr, d_r2_out.ptr, K_int));
              grav_fof::Buffer<unsigned char> work(sort_bytes);
              grav_fof::check(cub::DeviceRadixSort::SortKeys(
                  work.ptr, sort_bytes, d_r2_in.ptr, d_r2_out.ptr, K_int));
            } else {
              grav_fof::check(cub::DeviceSegmentedRadixSort::SortKeys(
                  nullptr, sort_bytes, d_r2_in.ptr, d_r2_out.ptr, K_int, batch_H, d_off.ptr, d_off.ptr + 1));
              grav_fof::Buffer<unsigned char> work(sort_bytes);
              grav_fof::check(cub::DeviceSegmentedRadixSort::SortKeys(
                  work.ptr, sort_bytes, d_r2_in.ptr, d_r2_out.ptr, K_int, batch_H, d_off.ptr, d_off.ptr + 1));
            }
          } else {
            if (batch_H == 1) {
              grav_fof::check(cub::DeviceRadixSort::SortPairs(
                  nullptr, sort_bytes, d_r2_in.ptr, d_r2_out.ptr, d_m_in.ptr, d_m_out.ptr, K_int));
              grav_fof::Buffer<unsigned char> work(sort_bytes);
              grav_fof::check(cub::DeviceRadixSort::SortPairs(
                  work.ptr, sort_bytes, d_r2_in.ptr, d_r2_out.ptr, d_m_in.ptr, d_m_out.ptr, K_int));
            } else {
              grav_fof::check(cub::DeviceSegmentedRadixSort::SortPairs(
                  nullptr, sort_bytes, d_r2_in.ptr, d_r2_out.ptr, d_m_in.ptr, d_m_out.ptr, K_int, batch_H, d_off.ptr, d_off.ptr + 1));
              grav_fof::Buffer<unsigned char> work(sort_bytes);
              grav_fof::check(cub::DeviceSegmentedRadixSort::SortPairs(
                  work.ptr, sort_bytes, d_r2_in.ptr, d_r2_out.ptr, d_m_in.ptr, d_m_out.ptr, K_int, batch_H, d_off.ptr, d_off.ptr + 1));
            }
          }
          d_r2_in.release();
          d_m_in.release();

          // Pass D: Exact enclosed-density + log-log interpolation solver
          grav_fof::Buffer<FofSoRow> d_bso(batch_H);
          grav_fof::Buffer<unsigned char> d_bsat(batch_H);
          if (!non_uniform) {
            if (!b_light.empty()) {
              const int nl = static_cast<int>(b_light.size());
              const int warps_per_blk = grav_fof::threads / 32;
              const int nblks = (nl + warps_per_blk - 1) / warps_per_blk;
              grav_fof::so_solve_uniform_light_kernel<<<nblks, grav_fof::threads>>>(
                  d_r2_out.ptr, d_off.ptr, d_blight.ptr, nl, gp, d_bso.ptr, d_bsat.ptr);
            }
            if (!b_heavy.empty()) {
              const int nh = static_cast<int>(b_heavy.size());
              grav_fof::so_solve_uniform_heavy_kernel<<<nh, grav_fof::threads>>>(
                  d_r2_out.ptr, d_off.ptr, d_bheavy.ptr, nh, gp, d_bso.ptr, d_bsat.ptr);
            }
          } else {
            if (!b_light.empty()) {
              const int nl = static_cast<int>(b_light.size());
              const int warps_per_blk = grav_fof::threads / 32;
              const int nblks = (nl + warps_per_blk - 1) / warps_per_blk;
              grav_fof::so_solve_nonuniform_light_kernel<<<nblks, grav_fof::threads>>>(
                  d_r2_out.ptr, d_m_out.ptr, d_off.ptr, d_blight.ptr, nl, gp, d_bso.ptr, d_bsat.ptr);
            }
            if (!b_heavy.empty()) {
              const int nh = static_cast<int>(b_heavy.size());
              grav_fof::so_solve_nonuniform_heavy_kernel<<<nh, grav_fof::threads>>>(
                  d_r2_out.ptr, d_m_out.ptr, d_off.ptr, d_bheavy.ptr, nh, gp, d_bso.ptr, d_bsat.ptr);
            }
          }
          grav_fof::check(cudaMemcpy(batch_so.data(), d_bso.ptr, std::size_t(batch_H) * sizeof(FofSoRow), cudaMemcpyDeviceToHost));
          grav_fof::check(cudaMemcpy(batch_sat.data(), d_bsat.ptr, std::size_t(batch_H) * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        }

        for (int bi = 0; bi < batch_H; ++bi) {
          const int a = b_beg + bi;
          out_so[act_hidx[a]] = batch_so[bi];
          if (batch_sat[bi] && act_rcap[a] < 0.25 * min_L && attempt < 2) {
            next_hidx.push_back(act_hidx[a]);
            next_pos.push_back(act_pos[a]);
            next_rcap.push_back(std::min(0.25 * min_L, act_rcap[a] * 2.0));
          }
        }
        b_beg = b_end;
      }

      act_hidx.swap(next_hidx);
      act_pos.swap(next_pos);
      act_rcap.swap(next_rcap);
    }

    return 0;
  } catch (const grav_fof::GpuError& err) {
    return int(err.status);
  } catch (const std::bad_alloc&) {
    return int(cudaErrorMemoryAllocation);
  } catch (...) {
    return int(cudaErrorUnknown);
  }
}
