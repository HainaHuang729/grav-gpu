#include "grav_domain_3d.h"
#include "grav_pm_config.cuh"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>

int main() {
  using namespace grav3d;
  std::array<uint64_t, N_TOP_CELLS> counts{};
  std::array<double, N_TOP_CELLS> weights{};
  for (int cid = 0; cid < N_TOP_CELLS; ++cid) {
    int tx, ty, tz;
    cell_coords(cid, tx, ty, tz);
    counts[cid] = 4096ULL + uint64_t((cid * 17) % 1024);
    weights[cid] = 1.0 + 1.5 * std::exp(-((tx - 4.0) * (tx - 4.0) + (ty - 6.0) * (ty - 6.0)) / 9.0);
  }

  for (int gpus : {1, 2, 4, 8}) {
    for (DomainMode mode : {DomainMode::Hilbert3D, DomainMode::Orb3D}) {
      const Domain3DPlan plan = build_domain_3d_plan(counts, weights, gpus, mode, 128);
      if (gpus >= 2 && gpus <= 4) {
        assert(plan.stats.w_imb < 1.05);
      }
      int total_ghost = 0;
      for (int dst = 0; dst < gpus; ++dst) {
        std::array<int, N_TOP_CELLS> seen{};
        for (int src = 0; src < gpus; ++src) {
          if (src == dst) assert(plan.export_cells[src][dst].empty());
          total_ghost += int(plan.export_cells[src][dst].size());
          for (size_t k = 0; k < plan.export_cells[src][dst].size(); ++k) {
            const int cid = plan.export_cells[src][dst][k];
            assert(plan.cell_owner[cid] == src);
            assert(plan.export_src_slots[src][dst][k] == plan.cell_to_own_slot[cid]);
            assert(plan.export_dst_ghost_slots[src][dst][k] == plan.cell_to_ghost_slot[dst][cid]);
            seen[cid]++;
          }
        }
        for (int cid : plan.ghost_cells[dst]) assert(seen[cid] == 1);
        assert(plan.slot_nb_roots[dst].size() == plan.own_cells[dst].size() * 27ULL);
      }
      assert(total_ghost == plan.stats.ghost_total);
    }
  }
  // Multi-node locality check: contiguous Hilbert curve rank assignment across nodes
  // must cut inter-node ghost volume relative to round-robin node assignment.
  for (int nodes : {2, 4}) {
    const int gpus = nodes * 4;
    const Domain3DPlan plan = build_domain_3d_plan(counts, weights, gpus, DomainMode::Hilbert3D, 128);
    std::vector<int> contig(gpus), rr(gpus);
    for (int r = 0; r < gpus; ++r) {
      contig[r] = r / 4;
      rr[r] = r % nodes;
    }
    const MultiNodeCommStats sc = compute_multinode_comm_stats(plan, contig, &counts);
    const MultiNodeCommStats sr = compute_multinode_comm_stats(plan, rr, &counts);
    assert(sc.intra_node_ghost_cells + sc.inter_node_ghost_cells == uint64_t(plan.stats.ghost_total));
    assert(sr.intra_node_ghost_cells + sr.inter_node_ghost_cells == uint64_t(plan.stats.ghost_total));
    assert(sc.inter_node_ghost_cells < sr.inter_node_ghost_cells);
    assert(sc.inter_node_ghost_parts < sr.inter_node_ghost_parts);
  }
  // P1 high-order PM halo-padding check (CIC [-1,+1] vs PCS+interlace [-2,+3])
  {
    assert(grav_pm_x_halo_lo(2, 0) == 1 && grav_pm_x_halo_hi(2, 0) == 1);
    assert(grav_pm_x_halo_lo(3, 0) == 2 && grav_pm_x_halo_hi(3, 0) == 2);
    assert(grav_pm_x_halo_lo(4, 1) == 2 && grav_pm_x_halo_hi(4, 1) == 3);
    const Domain3DPlan p_cic = build_domain_3d_plan(counts, weights, 4, DomainMode::Hilbert3D, 128, nullptr, 1, 1);
    const Domain3DPlan p_pcs = build_domain_3d_plan(counts, weights, 4, DomainMode::Hilbert3D, 128, nullptr, 2, 3);
    for (int g = 0; g < 4; ++g) {
      assert(p_pcs.gpu_x_planes[g].size() >= p_cic.gpu_x_planes[g].size());
    }
  }
  std::cout << "DOMAIN3D_SMOKE_PASSED\n";
  return 0;
}
