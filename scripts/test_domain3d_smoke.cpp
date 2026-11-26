#include "grav_domain_3d.h"

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
  std::cout << "DOMAIN3D_SMOKE_PASSED\n";
  return 0;
}
