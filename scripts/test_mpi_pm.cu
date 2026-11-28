#include "grav_cuda_pm.h"
#include "grav_cuda_pm_mpi.h"
#include "grav_domain_3d.h"

#include <mpi.h>
#include <cuda_runtime.h>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int top_cell_for_pos(const double3& p, double box) {
  auto idx = [box](double v) {
    double w = v - std::floor(v / box) * box;
    int c = int(w * (double(grav3d::N_TOP) / box));
    if (c < 0) c = 0;
    if (c >= grav3d::N_TOP) c = grav3d::N_TOP - 1;
    return c;
  };
  return grav3d::cell_id(idx(p.x), idx(p.y), idx(p.z));
}

}  // namespace

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  {
    grav_mpi::GravMpiCtx mpi;
    mpi.init(MPI_COMM_WORLD);

    const int rank = mpi.world_rank;
    const int G = mpi.world_size;
    const int side = 128;
    const double box = 100.0;
    const double split = 1.25 * box / double(side);
    const float grav_G = 43.0092f;
    const size_t N = 64ULL * 64ULL * 64ULL;

    // Deterministic clustered particle distribution identical across all ranks
    std::vector<double3> all_x(N);
    std::vector<float> all_m(N);
    std::array<uint64_t, grav3d::N_TOP_CELLS> counts{};
    std::array<double, grav3d::N_TOP_CELLS> weights{};

    std::mt19937_64 rng(20261127ULL);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::normal_distribution<double> g01(0.0, 6.5);
    const double3 centers[4] = {
        {22.0, 27.0, 31.0}, {73.0, 21.0, 68.0}, {38.0, 77.0, 25.0}, {65.0, 64.0, 78.0}};
    auto wrap_box = [box](double v) {
      double r = v - std::floor(v / box) * box;
      return (r >= box) ? 0.0 : r;
    };

    for (size_t i = 0; i < N; ++i) {
      if ((i & 3U) == 0) {
        all_x[i] = {u01(rng) * box, u01(rng) * box, u01(rng) * box};
      } else {
        const auto& c = centers[i % 4];
        all_x[i] = {wrap_box(c.x + g01(rng)),
                    wrap_box(c.y + g01(rng)),
                    wrap_box(c.z + g01(rng))};
      }
      all_m[i] = float(0.8 + 0.4 * u01(rng));
      const int cid = top_cell_for_pos(all_x[i], box);
      counts[cid] += 1ULL;
    }
    for (int cid = 0; cid < grav3d::N_TOP_CELLS; ++cid) {
      weights[cid] = double(counts[cid]);
    }

    const grav3d::Domain3DPlan plan = grav3d::build_domain_3d_plan(
        counts, weights, G, grav3d::DomainMode::Hilbert3D, side);
    const grav3d::MultiNodeCommStats mstats =
        grav3d::compute_multinode_comm_stats(plan, mpi.rank_to_node, &counts);

    std::vector<double3> my_x;
    std::vector<float> my_m;
    my_x.reserve(N / size_t(G) + 4096);
    my_m.reserve(N / size_t(G) + 4096);
    for (size_t i = 0; i < N; ++i) {
      const int cid = top_cell_for_pos(all_x[i], box);
      if (plan.cell_owner[cid] == rank) {
        my_x.push_back(all_x[i]);
        my_m.push_back(all_m[i]);
      }
    }

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    double3* d_my_x = nullptr;
    float* d_my_m = nullptr;
    float3* d_my_acc = nullptr;
    float* d_my_pot = nullptr;
    const size_t my_n = my_x.size();
    cudaMalloc(&d_my_x, std::max<size_t>(1, my_n) * sizeof(double3));
    cudaMalloc(&d_my_m, std::max<size_t>(1, my_n) * sizeof(float));
    cudaMalloc(&d_my_acc, std::max<size_t>(1, my_n) * sizeof(float3));
    cudaMalloc(&d_my_pot, std::max<size_t>(1, my_n) * sizeof(float));
    if (my_n > 0) {
      cudaMemcpy(d_my_x, my_x.data(), my_n * sizeof(double3), cudaMemcpyHostToDevice);
      cudaMemcpy(d_my_m, my_m.data(), my_n * sizeof(float), cudaMemcpyHostToDevice);
    }

    GravMpiPm pm;
    pm.init(&mpi, stream, side, plan.gpu_x_planes[rank]);
    pm.begin_deposit();
    pm.deposit(d_my_x, d_my_m, 0.0f, my_n, box);
    pm.solve_and_gather(box, split);
    pm.interpolate(d_my_x, my_n, box, grav_G, d_my_acc, d_my_pot);
    cudaStreamSynchronize(stream);

    int my_x0 = 0, my_nx = 0;
    const size_t P = size_t(side) * size_t(side);
    std::vector<double4> my_slab(size_t(side) * P);
    pm.debug_copy_local_slab_field_to_host(my_slab.data(), my_x0, my_nx);

    // Gather canonical field slab onto rank 0 for exact double-precision comparison
    std::vector<int> slab_counts_bytes(G), slab_displs_bytes(G);
    int my_slab_bytes = int(size_t(my_nx) * P * sizeof(double4));
    MPI_Gather(&my_slab_bytes, 1, MPI_INT, slab_counts_bytes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<double4> gathered_field;
    if (rank == 0) {
      int tot = 0;
      for (int g = 0; g < G; ++g) {
        slab_displs_bytes[g] = tot;
        tot += slab_counts_bytes[g];
      }
      gathered_field.resize(size_t(side) * P);
    }
    MPI_Gatherv(my_slab.data(), my_slab_bytes, MPI_BYTE,
                rank == 0 ? gathered_field.data() : nullptr,
                slab_counts_bytes.data(), slab_displs_bytes.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
      // Single-GPU reference solve using grav_pm_cuda_*
      const size_t grid_elems = size_t(side) * P;
      double* d_ref_dens = nullptr;
      double4* d_ref_field = nullptr;
      double3* d_all_x = nullptr;
      float* d_all_m = nullptr;
      cudaMalloc(&d_ref_dens, grid_elems * sizeof(double));
      cudaMalloc(&d_ref_field, grid_elems * sizeof(double4));
      cudaMalloc(&d_all_x, N * sizeof(double3));
      cudaMalloc(&d_all_m, N * sizeof(float));
      cudaMemset(d_ref_dens, 0, grid_elems * sizeof(double));
      cudaMemcpy(d_all_x, all_x.data(), N * sizeof(double3), cudaMemcpyHostToDevice);
      cudaMemcpy(d_all_m, all_m.data(), N * sizeof(float), cudaMemcpyHostToDevice);

      grav_pm_cuda_deposit_buf(d_all_x, d_all_m, 0.0f, N, d_ref_dens, side, box, stream);
      grav_pm_cuda_solve_on_device(d_ref_dens, d_ref_field, side, box, split, stream);
      std::vector<double4> ref_field(grid_elems);
      cudaMemcpyAsync(ref_field.data(), d_ref_field, grid_elems * sizeof(double4),
                      cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);

      double max_field_err = 0.0, max_field_norm = 0.0;
      double max_pot_err = 0.0, max_pot_norm = 0.0;
      for (size_t i = 0; i < grid_elems; ++i) {
        const double4 a = gathered_field[i];
        const double4 b = ref_field[i];
        const double fn = std::sqrt(b.x * b.x + b.y * b.y + b.z * b.z);
        const double fe = std::sqrt((a.x - b.x) * (a.x - b.x) +
                                    (a.y - b.y) * (a.y - b.y) +
                                    (a.z - b.z) * (a.z - b.z));
        max_field_norm = std::max(max_field_norm, fn);
        max_field_err = std::max(max_field_err, fe);
        max_pot_norm = std::max(max_pot_norm, std::abs(b.w));
        max_pot_err = std::max(max_pot_err, std::abs(a.w - b.w));
      }
      const double max_rel_field = max_field_err / std::max(1e-30, max_field_norm);
      const double max_rel_pot = max_pot_err / std::max(1e-30, max_pot_norm);

      std::printf(
          "MPI_PM_TEST ranks=%d nodes=%d side=%d max_rel_field=%.3e max_rel_pot=%.3e "
          "pm_comm_ms=%.3f intra_ghost_cells=%llu inter_ghost_cells=%llu\n",
          G, mpi.num_nodes, side, max_rel_field, max_rel_pot, pm.last_comm_ms(),
          (unsigned long long)mstats.intra_node_ghost_cells,
          (unsigned long long)mstats.inter_node_ghost_cells);
      assert(max_rel_field < 1e-13);
      assert(max_rel_pot < 1e-13);
      std::printf("MPI_PM_PASSED\n");

      cudaFree(d_ref_dens);
      cudaFree(d_ref_field);
      cudaFree(d_all_x);
      cudaFree(d_all_m);
    }

    pm.destroy();
    cudaFree(d_my_x);
    cudaFree(d_my_m);
    cudaFree(d_my_acc);
    cudaFree(d_my_pot);
    cudaStreamDestroy(stream);
    mpi.destroy();
  }
  MPI_Finalize();
  return 0;
}
