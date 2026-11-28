#pragma once

#ifdef GRAV_ENABLE_MPI

#include "grav_cuda_pm_mpi.h"
#include "grav_domain_3d_engine.cuh"

namespace grav3d {

struct MpiGhostEdge {
  // Outgoing (my_rank -> peer)
  int E_out = 0;
  int out_tot_nodes = 0;
  int out_tot_parts = 0;
  size_t out_wire_bytes = 0;
  size_t out_off_g = 0, out_off_m = 0, out_off_c = 0, out_off_p = 0, out_off_s = 0;

  DevPtr<int> d_slot_to_e;
  DevPtr<int> d_exp_cursor, d_exp_part_src_start, d_exp_part_off, d_exp_part_cnt;
  DevPtr<int> d_bnd_src_node_map;
  DevPtr<char> d_send_wire;
  DevPtr<Dom3DLazyRootDesc> d_lazy_descs_src;
  DevPtr<float4> d_lazy_send;

  PinnedHostBuf h_meta_pinned;
  std::vector<int> h_node_cnt, h_part_cnt, h_part_src_start;
  std::vector<int> h_node_off, h_nonroot_off, h_part_off;
  std::vector<Dom3DLazyRootDesc> h_lazy_descs_send;
  int lazy_M_out = 0;
  int lazy_tot_nodes_out = 0;
  int lazy_tot_parts_out = 0;

  // Incoming (peer -> my_rank)
  int E_in = 0;
  int in_tot_nodes = 0;
  int in_tot_parts = 0;
  size_t in_wire_bytes = 0;
  size_t in_off_g = 0, in_off_m = 0, in_off_c = 0, in_off_p = 0, in_off_s = 0;
  int dst_desc_base = 0;
  int dst_part_base = 0;

  DevPtr<int> d_dst_ghost_slots;
  DevPtr<char> d_recv_wire;
  DevPtr<Dom3DLazyRootDesc> d_lazy_descs_dst;
  DevPtr<float4> d_lazy_recv;
  std::vector<Dom3DLazyRootDesc> h_lazy_descs_recv;
  int lazy_M_in = 0;
  int lazy_tot_nodes_in = 0;
  int lazy_tot_parts_in = 0;

  void ensure_meta_src(int dev, int e_cnt, size_t& tracker) {
    const size_t need = size_t(e_cnt);
    const size_t need_p = need * 4ULL * sizeof(int);
    if (need > dev_len(d_exp_cursor)) {
      check(cudaSetDevice(dev));
      const size_t cap = std::max<size_t>(need * 2ULL, size_t(N_TOP_CELLS));
      d_exp_cursor.alloc(cap, tracker);
      d_exp_part_src_start.alloc(cap, tracker);
      d_exp_part_off.alloc(cap, tracker);
      d_exp_part_cnt.alloc(cap, tracker);
    }
    if (h_meta_pinned.bytes < need_p) {
      alloc_pinned_slack(h_meta_pinned, std::max<size_t>(need_p, size_t(N_TOP_CELLS) * 4ULL * sizeof(int)));
    }
  }

  void ensure_bnd_nodes(int dev, int nn, size_t& tracker) {
    if (size_t(nn) <= dev_len(d_bnd_src_node_map)) return;
    check(cudaSetDevice(dev));
    const size_t cap = std::max(size_t(nn), size_t(nn) * 3 / 2 + 4096);
    d_bnd_src_node_map.alloc(cap, tracker);
  }

  void setup_plan(int dev, int n_own_src,
                  const std::vector<int>& out_src_slots,
                  const std::vector<int>& in_dst_slots,
                  size_t& tracker) {
    E_out = int(out_src_slots.size());
    E_in = int(in_dst_slots.size());
    out_tot_nodes = out_tot_parts = 0;
    in_tot_nodes = in_tot_parts = 0;
    out_wire_bytes = in_wire_bytes = 0;
    dst_desc_base = dst_part_base = 0;
    lazy_M_out = lazy_tot_nodes_out = lazy_tot_parts_out = 0;
    lazy_M_in = lazy_tot_nodes_in = lazy_tot_parts_in = 0;

    h_node_cnt.resize(E_out);
    h_part_cnt.resize(E_out);
    h_part_src_start.resize(E_out);
    h_node_off.resize(E_out);
    h_nonroot_off.resize(E_out);
    h_part_off.resize(E_out);

    check(cudaSetDevice(dev));
    if (dev_len(d_slot_to_e) < size_t(N_TOP_CELLS)) {
      d_slot_to_e.alloc(N_TOP_CELLS, tracker);
    }
    if (E_out > 0) {
      std::array<int, N_TOP_CELLS> s2e;
      s2e.fill(-1);
      for (int e = 0; e < E_out; ++e) {
        if (out_src_slots[e] >= 0 && out_src_slots[e] < n_own_src) s2e[out_src_slots[e]] = e;
      }
      check(cudaMemcpy(d_slot_to_e.p, s2e.data(),
                       size_t(std::max(1, n_own_src)) * sizeof(int), cudaMemcpyHostToDevice));
      ensure_meta_src(dev, E_out, tracker);
      if (size_t(E_out) > dev_len(d_lazy_descs_src)) {
        d_lazy_descs_src.alloc(std::max<size_t>(size_t(E_out), size_t(N_TOP_CELLS)), tracker);
      }
    }
    if (E_in > 0) {
      if (size_t(E_in) > dev_len(d_dst_ghost_slots)) {
        d_dst_ghost_slots.alloc(std::max<size_t>(size_t(E_in), size_t(N_TOP_CELLS)), tracker);
      }
      check(cudaMemcpy(d_dst_ghost_slots.p, in_dst_slots.data(),
                       size_t(E_in) * sizeof(int), cudaMemcpyHostToDevice));
      if (size_t(E_in) > dev_len(d_lazy_descs_dst)) {
        d_lazy_descs_dst.alloc(std::max<size_t>(size_t(E_in), size_t(N_TOP_CELLS)), tracker);
      }
    }
  }

  void free_all(int dev, size_t& tracker) {
    check(cudaSetDevice(dev));
    d_slot_to_e.free(tracker);
    d_exp_cursor.free(tracker);
    d_exp_part_src_start.free(tracker);
    d_exp_part_off.free(tracker);
    d_exp_part_cnt.free(tracker);
    d_bnd_src_node_map.free(tracker);
    d_send_wire.free(tracker);
    d_lazy_descs_src.free(tracker);
    d_lazy_send.free(tracker);
    d_dst_ghost_slots.free(tracker);
    d_recv_wire.free(tracker);
    d_lazy_descs_dst.free(tracker);
    d_lazy_recv.free(tracker);
    h_meta_pinned.free();
  }
};

template <typename ComputeCapFn, typename NodeCapFn>
inline int run_domain_3d_mpi_evolution(
    const Domain3DRunConfig& cfg,
    const grav_mpi::GravMpiCtx& mpi,
    ComputeCapFn compute_part_cap,
    NodeCapFn compute_ncap) {
  auto blocks = [](int n) { return std::max(1, (n + team - 1) / team); };
  auto blocks_warps = [](int w) { return std::max(1, (w * 32 + team - 1) / team); };

  const int num_ranks = mpi.world_size;
  const int my_rank = mpi.world_rank;
  const uint64_t N = cfg.N;
  const StateInput& h = cfg.h;
  const float first_mass = float(cfg.first_mass_in);
  const float first_soft_plummer = float(cfg.first_soft_in);
  const bool mass_uniform = cfg.mass_uniform;
  const bool soft_uniform = cfg.soft_uniform;
  const int pm_side = cfg.pm_side;
  const int leaf_limit = cfg.leaf_limit;
  const double inv_cell_w = double(N_TOP) / h.box;

  std::array<double, N_TOP_CELLS> init_weights{};
  for (int c = 0; c < N_TOP_CELLS; ++c) init_weights[c] = double(cfg.init_cell_counts[c]);
  Domain3DPlan plan = build_domain_3d_plan(
      cfg.init_cell_counts, init_weights, num_ranks, cfg.domain_mode, pm_side, nullptr);
  MultiNodeCommStats mstats = compute_multinode_comm_stats(plan, mpi.rank_to_node, &cfg.init_cell_counts);

  // Pass 2: each MPI rank streams input.bin and retains ONLY particles in its owned cells
  HostSlabStaging h_my_stage;
  {
    size_t est = 0;
    for (int cid : plan.own_cells[my_rank]) est += cfg.init_cell_counts[cid];
    h_my_stage.x.reserve(est);
    h_my_stage.v.reserve(est);
    h_my_stage.orig_idx.reserve(est);
    if (!mass_uniform) h_my_stage.mass.reserve(est);
    if (!soft_uniform) h_my_stage.soft.reserve(est);

    const uint64_t pos_offset = sizeof(StateInput);
    const uint64_t vel_offset = pos_offset + N * sizeof(double3);
    const uint64_t mass_offset = vel_offset + N * sizeof(float3);
    const uint64_t chunk = std::min<uint64_t>(N, 16ULL << 20);
    std::vector<double3> rx(chunk);
    std::vector<float3> rv(chunk);
    std::vector<float> rm, rs;
    if (!mass_uniform) rm.resize(chunk);
    if (!soft_uniform) rs.resize(chunk);

    FILE* fx = fopen(cfg.input_path, "rb");
    FILE* fv = fopen(cfg.input_path, "rb");
    FILE* fm = mass_uniform ? nullptr : fopen(cfg.input_path, "rb");
    FILE* fs = (!soft_uniform && cfg.has_soft_in) ? fopen(cfg.soft_in_env, "rb") : nullptr;
    if (!fx || !fv || (!mass_uniform && !fm) || (!soft_uniform && cfg.has_soft_in && !fs))
      throw std::runtime_error("Failed reopening input for MPI 3D Pass 2 load");
    fseeko(fx, off_t(pos_offset), SEEK_SET);
    fseeko(fv, off_t(vel_offset), SEEK_SET);
    if (fm) fseeko(fm, off_t(mass_offset), SEEK_SET);

    const float a0_f = float(h.initial_scale);
    for (uint64_t base = 0; base < N; base += chunk) {
      const size_t cnt = size_t(std::min<uint64_t>(chunk, N - base));
      read_bin(fx, rx.data(), cnt);
      read_bin(fv, rv.data(), cnt);
      if (fm) read_bin(fm, rm.data(), cnt);
      if (fs) read_bin(fs, rs.data(), cnt);
      for (size_t k = 0; k < cnt; ++k) {
        double3 pos = rx[k];
        pos.x = wrapped_coord(pos.x, h.box);
        pos.y = wrapped_coord(pos.y, h.box);
        pos.z = wrapped_coord(pos.z, h.box);
        const int tx = std::min(std::max(int(pos.x * inv_cell_w), 0), N_TOP - 1);
        const int ty = std::min(std::max(int(pos.y * inv_cell_w), 0), N_TOP - 1);
        const int tz = std::min(std::max(int(pos.z * inv_cell_w), 0), N_TOP - 1);
        const int owner = plan.cell_owner[cell_id(tx, ty, tz)];
        if (owner != my_rank) continue;
        h_my_stage.x.push_back(pos);
        const float3 vi = rv[k];
        h_my_stage.v.push_back(float3{vi.x * a0_f, vi.y * a0_f, vi.z * a0_f});
        h_my_stage.orig_idx.push_back(uint32_t(base + k));
        if (!mass_uniform) h_my_stage.mass.push_back(rm[k]);
        if (!soft_uniform) h_my_stage.soft.push_back(float(double(rs[k]) * kernel_gravity_softening_plummer_equivalent));
      }
    }
    fclose(fx); fclose(fv);
    if (fm) fclose(fm);
    if (fs) fclose(fs);
  }

  Dom3DGpuCtx G;
  G.dev = mpi.cuda_dev;
  check(cudaSetDevice(G.dev));
  check(cudaStreamCreateWithFlags(&G.compute_stream, cudaStreamNonBlocking));
  check(cudaStreamCreateWithFlags(&G.copy_stream, cudaStreamNonBlocking));
  check(cudaStreamCreateWithFlags(&G.d2h_stream, cudaStreamNonBlocking));
  check(cudaEventCreateWithFlags(&G.ev_force_start, 0));
  check(cudaEventCreateWithFlags(&G.ev_force_end, 0));
  check(cudaEventCreateWithFlags(&G.ev_prof0, 0));
  check(cudaEventCreateWithFlags(&G.ev_prof1, 0));
  check(cudaEventCreateWithFlags(&G.ev_ghost_start, 0));
  check(cudaEventCreateWithFlags(&G.ev_ghost_ready, 0));
  check(cudaEventCreateWithFlags(&G.ev_int_start, 0));
  check(cudaEventCreateWithFlags(&G.ev_int_end, 0));
  check(cudaEventCreateWithFlags(&G.ev_bnd_start, 0));
  check(cudaEventCreateWithFlags(&G.ev_bnd_end, 0));

  const size_t pm_points = size_t(pm_side) * size_t(pm_side) * size_t(pm_side);

  auto upload_plan_to_gpu = [&](const Domain3DPlan& pl) {
    check(cudaSetDevice(G.dev));
    G.n_own = int(pl.own_cells[my_rank].size());
    G.n_interior_own = pl.n_interior_own[my_rank];
    G.n_ghost = int(pl.ghost_cells[my_rank].size());
    G.d_cell_owner.alloc(N_TOP_CELLS, G.device_bytes);
    G.d_cell_to_slot.alloc(N_TOP_CELLS, G.device_bytes);
    G.d_slot_to_cell.alloc(std::max(1, G.n_own), G.device_bytes);
    G.d_slot_nb_roots.alloc(std::max(1, G.n_own * 27), G.device_bytes);
    std::vector<int> h_c2s(N_TOP_CELLS, -1);
    for (int s = 0; s < G.n_own; ++s) h_c2s[pl.own_cells[my_rank][s]] = s;
    check(cudaMemcpyAsync(G.d_cell_owner.p, pl.cell_owner.data(),
                          N_TOP_CELLS * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
    check(cudaMemcpyAsync(G.d_cell_to_slot.p, h_c2s.data(),
                          N_TOP_CELLS * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
    if (G.n_own > 0) {
      check(cudaMemcpyAsync(G.d_slot_to_cell.p, pl.own_cells[my_rank].data(),
                            size_t(G.n_own) * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_slot_nb_roots.p, pl.slot_nb_roots[my_rank].data(),
                            size_t(G.n_own) * 27ULL * sizeof(uint32_t), cudaMemcpyHostToDevice, G.compute_stream));
    }
    check(cudaStreamSynchronize(G.compute_stream));
  };

  G.d_active_count.alloc(1, G.device_bytes);
  G.d_tail_count.alloc(1, G.device_bytes);
  G.d_tree_invalid.alloc(1, G.device_bytes);
  G.d_stats.alloc(1, G.device_bytes);
  G.d_kick_finish.alloc(57, G.device_bytes);
  G.d_kick_begin.alloc(57, G.device_bytes);
  G.d_active_bits.alloc(54, G.device_bytes);
  G.d_cell_counts.alloc(N_TOP_CELLS, G.device_bytes);
  G.d_cell_act_counts.alloc(N_TOP_CELLS, G.device_bytes);
  G.d_split_total_warps.alloc(2, G.device_bytes);
  G.d_own_slot_desc_cnt.alloc(N_TOP_CELLS, G.device_bytes);
  G.d_own_slot_pstart.alloc(N_TOP_CELLS, G.device_bytes);
  G.d_own_slot_pcnt.alloc(N_TOP_CELLS, G.device_bytes);
  G.h_slot_meta.alloc(3 * size_t(N_TOP_CELLS) * sizeof(int), cudaHostAllocPortable);
  if (!cfg.pm_dist_enabled && num_ranks == 1) {
    G.d_pm_density.alloc(pm_points, G.device_bytes);
    G.d_pm_field.alloc(pm_points, G.device_bytes);
  }
  G.d_scan_tmp.alloc(scan_grid_max, G.device_bytes);
  G.h_scan.resize(scan_grid_max);
  G.ensure_mig_out(262144);
  G.ensure_mig_in(262144);

  G.ns = h_my_stage.x.size();
  const size_t init_cap = std::max(compute_part_cap(G.ns), size_t(double(G.ns) * 1.15) + 32768);
  const size_t init_ncap = compute_ncap(init_cap, leaf_limit);
  G.alloc_particles(init_cap, mass_uniform, soft_uniform);
  G.alloc_local_tree_nodes(init_ncap, cfg.lazy_tree_enabled);
  if (G.ns > 0) {
    check(cudaMemcpyAsync(G.d_x.p, h_my_stage.x.data(), G.ns * sizeof(double3), cudaMemcpyHostToDevice, G.compute_stream));
    check(cudaMemcpyAsync(G.d_v.p, h_my_stage.v.data(), G.ns * sizeof(float3), cudaMemcpyHostToDevice, G.compute_stream));
    check(cudaMemcpyAsync(G.d_orig_idx.p, h_my_stage.orig_idx.data(), G.ns * sizeof(uint32_t), cudaMemcpyHostToDevice, G.compute_stream));
    if (!mass_uniform)
      check(cudaMemcpyAsync(G.d_mass.p, h_my_stage.mass.data(), G.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
    if (!soft_uniform)
      check(cudaMemcpyAsync(G.d_soft.p, h_my_stage.soft.data(), G.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
  }
  check(cudaMemsetAsync(G.d_mesh_acc.p, 0, G.part_cap * sizeof(float3), G.compute_stream));
  check(cudaMemsetAsync(G.d_old_acc.p, 0, G.part_cap * sizeof(float), G.compute_stream));
  check(cudaMemsetAsync(G.d_bins.p, 1, G.part_cap * sizeof(unsigned char), G.compute_stream));
  upload_plan_to_gpu(plan);
  G.ensure_ghost_cap((num_ranks > 1) ? init_cap : 1, (num_ranks > 1) ? init_ncap : N_TOP_CELLS, soft_uniform);
  G.ensure_split_bufs(size_t(cfg.split_batch_warps), 1);
  G.h_bounce.alloc((num_ranks > 1) ? (64ULL << 20) : (80ULL << 20), cudaHostAllocPortable);

  {
    std::vector<double3>().swap(h_my_stage.x);
    std::vector<float3>().swap(h_my_stage.v);
    std::vector<float>().swap(h_my_stage.mass);
    std::vector<float>().swap(h_my_stage.soft);
    std::vector<uint32_t>().swap(h_my_stage.orig_idx);
  }

  std::vector<MpiGhostEdge> edges(num_ranks);
  auto setup_all_edges = [&](const Domain3DPlan& pl) {
    for (int p = 0; p < num_ranks; ++p) {
      if (p == my_rank) continue;
      edges[p].setup_plan(
          G.dev, int(pl.own_cells[my_rank].size()),
          pl.export_src_slots[my_rank][p],
          pl.export_dst_ghost_slots[p][my_rank],
          G.device_bytes);
    }
  };
  setup_all_edges(plan);

  GravMpiPm mpi_pm;
  const bool use_mpi_pm = cfg.pm_dist_enabled || (num_ranks > 1);
  if (use_mpi_pm) {
    mpi_pm.init(&mpi, G.compute_stream, pm_side, plan.gpu_x_planes[my_rank]);
    G.device_bytes += mpi_pm.device_bytes();
  }

  MPI_Barrier(mpi.world);
  const double load_s = wall() - cfg.import_start;

  std::vector<double> knots;
  knots.push_back(h.initial_scale);
  for (double sa : cfg.snap_scales) knots.push_back(sa);
  knots.push_back(cfg.a_end);

  const double log_span = cfg.comoving ? log(knots[1] / knots[0]) : (knots[1] - knots[0]);
  const float soft = cfg.has_soft_in
      ? float(double(first_soft_plummer) * kernel_gravity_softening_plummer_equivalent)
      : float(h.plummer_softening * kernel_gravity_softening_plummer_equivalent);
  const double rho_crit0 = 3.0 * cfg.cosmo_h0 * cfg.cosmo_h0 / (8.0 * acos(-1.0) * double(cfg.grav_G));
  const double d_dm = cbrt(cfg.min_mass / (cfg.cosmo_ocdm * rho_crit0));
  const double r_s = 1.25 * h.box / double(pm_side);

  Physics p{knots[0], log_span,
      float(h.box), float(r_s), float(4.5 * r_s), cfg.grav_G,
      timeline / cfg.pm_intervals, timeline / cfg.short_intervals, cfg.step_factor, cfg.mac,
      cfg.dt_max_val, std::min(r_s, d_dm), cfg.max_rms_fac, cfg.dynamic_pm,
      cfg.cosmo_h0, cfg.cosmo_om, cfg.cosmo_ode, cfg.cosmo_or, cfg.cosmo_ok, cfg.cosmo_w0, cfg.cosmo_wa,
      cfg.theta_val, cfg.grav_eta_val,
      true, cfg.comoving,
      0.f, 1.f, 0.f, 0.f, 0.f};
  const double potential_norm = cfg.periodic
      ? (4.0 * acos(-1.0) * cfg.total_mass * double(p.split) * double(p.split) / (h.box * h.box * h.box))
      : 0.0;

  probe_header gh{};
  gh.top[0] = gh.top[1] = gh.top[2] = N_TOP;
  for (int axis = 0; axis < 3; ++axis) {
    gh.width[axis] = h.box / double(N_TOP);
    gh.inv_width[axis] = 1.0 / gh.width[axis];
  }
  gh.particles = N;
  gh.leaf_limit = leaf_limit;

  // Report MEMORY_PLAN & GPU_INPUT from rank 0
  size_t fv_b = 0, tv_b = 0;
  check(cudaMemGetInfo(&fv_b, &tv_b));
  cudaDeviceProp prop0{};
  check(cudaGetDeviceProperties(&prop0, G.dev));

  uint64_t my_parts_u64 = uint64_t(G.ns);
  uint64_t my_dev_bytes_u64 = uint64_t(G.device_bytes);
  uint64_t my_free_vram_u64 = uint64_t(fv_b);
  std::vector<uint64_t> all_parts(num_ranks), all_dev_bytes(num_ranks), all_free_vram(num_ranks);
  MPI_Gather(&my_parts_u64, 1, MPI_UNSIGNED_LONG_LONG, all_parts.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, mpi.world);
  MPI_Gather(&my_dev_bytes_u64, 1, MPI_UNSIGNED_LONG_LONG, all_dev_bytes.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, mpi.world);
  MPI_Gather(&my_free_vram_u64, 1, MPI_UNSIGNED_LONG_LONG, all_free_vram.data(), 1, MPI_UNSIGNED_LONG_LONG, 0, mpi.world);

  uint64_t sum_dev_bytes = 0, min_free_vram = ~uint64_t(0), peak_gpu_seen = 0;
  if (my_rank == 0) {
    for (int r = 0; r < num_ranks; ++r) {
      sum_dev_bytes += all_dev_bytes[r];
      min_free_vram = std::min(min_free_vram, all_free_vram[r]);
      peak_gpu_seen = std::max(peak_gpu_seen, all_parts[r]);
    }
    auto print_vec = [&](const auto& vec) {
      std::cout << "[";
      for (size_t i = 0; i < vec.size(); ++i) std::cout << (i ? "," : "") << vec[i];
      std::cout << "]";
    };
    std::cout << std::setprecision(12)
              << "MEMORY_PLAN {\"mode\":\"mpi_domain_3d\",\"domain_mode\":\"" << domain_mode_name(cfg.domain_mode)
              << "\",\"num_gpus\":" << num_ranks << ",\"num_ranks\":" << num_ranks << ",\"num_nodes\":" << mpi.num_nodes
              << ",\"cuda_aware_mpi\":" << (mpi.cuda_aware ? "true" : "false")
              << ",\"top_cells\":" << N_TOP_CELLS
              << ",\"particles\":" << N
              << ",\"w_imb\":" << plan.stats.w_imb << ",\"p_imb\":" << plan.stats.p_imb
              << ",\"ghost_cells_total\":" << plan.stats.ghost_total
              << ",\"intra_node_ghost_cells\":" << mstats.intra_node_ghost_cells
              << ",\"inter_node_ghost_cells\":" << mstats.inter_node_ghost_cells
              << ",\"max_x_slabs\":" << plan.stats.max_x_slabs
              << ",\"per_gpu_particles\":";
    print_vec(all_parts);
    std::cout << ",\"per_gpu_device_bytes\":";
    print_vec(all_dev_bytes);
    std::cout << ",\"per_gpu_free_vram_bytes\":";
    print_vec(all_free_vram);
    std::cout << ",\"device_buffer_bytes\":" << sum_dev_bytes
              << ",\"free_vram_bytes\":" << min_free_vram
              << ",\"total_vram_bytes\":" << tv_b << "}" << std::endl;

    std::cout << std::setprecision(12)
              << "GPU_INPUT {\"particles\":" << N << ",\"num_gpus\":" << num_ranks
              << ",\"num_ranks\":" << num_ranks << ",\"num_nodes\":" << mpi.num_nodes
              << ",\"domain_mode\":\"" << domain_mode_name(cfg.domain_mode)
              << "\",\"rebalance_interval\":" << cfg.rebalance_interval
              << ",\"rebalance_min_gain\":" << cfg.rebalance_min_gain
              << ",\"periodic\":true,\"comoving\":" << (cfg.comoving ? "true" : "false")
              << ",\"pm_grid\":" << pm_side
              << ",\"dynamic_pm\":" << (cfg.dynamic_pm ? "true" : "false")
              << ",\"grav_G\":" << p.G << ",\"pm_intervals\":" << cfg.pm_intervals
              << ",\"short_cap_intervals\":" << cfg.short_intervals
              << ",\"short_step_factor\":" << cfg.step_factor
              << ",\"mac_tolerance\":" << cfg.mac
              << ",\"theta\":" << p.theta << ",\"grav_eta\":" << p.grav_eta
              << ",\"hubble0\":" << p.hubble0 << ",\"omega_m\":" << p.omega_m
              << ",\"omega_de\":" << p.omega_de << ",\"omega_r\":" << p.omega_r
              << ",\"omega_k\":" << p.omega_k << ",\"w0\":" << p.w0 << ",\"wa\":" << p.wa
              << ",\"force_check_n\":" << cfg.force_check_n
              << ",\"lightcone\":false"
              << ",\"ext_pot_M\":0,\"a_begin\":" << h.initial_scale << ",\"a_end\":" << cfg.a_end
              << ",\"warp_m2l\":" << (cfg.warp_m2l ? "true" : "false")
              << ",\"fast_split\":" << (cfg.fast_split ? "true" : "false")
              << ",\"split_force\":" << (cfg.split_force_enabled ? "true" : "false")
              << ",\"pm_dist\":true"
              << ",\"split_batch_warps\":" << cfg.split_batch_warps
              << ",\"tree_leaf_limit\":" << gh.leaf_limit
              << ",\"softening_internal_mpc\":" << soft << ",\"box_mpc\":" << h.box
              << ",\"prepare_and_load_s\":" << load_s
              << ",\"device\":\"" << prop0.name << "\"}" << std::endl;
  }
  MPI_Bcast(&peak_gpu_seen, 1, MPI_UNSIGNED_LONG_LONG, 0, mpi.world);

  // Auxiliary Ewald check setup (when GRAV_FORCE_CHECK_N > 0)
  size_t aux_dev_bytes = 0;
  const int fc_target_cap = int(std::min<uint64_t>(uint64_t(cfg.force_check_n), N));
  std::vector<EwaldMode> h_ewald_modes;
  DevPtr<EwaldMode> d_ewald_modes;
  DevPtr<double> d_ewald_C, d_ewald_S;
  DevPtr<double3> d_fc_tgt_x, d_fc_short_ref, d_fc_total_ref;
  DevPtr<float> d_fc_tgt_h;
  DevPtr<int> d_fc_lids, d_fc_tgt_slab;
  DevPtr<float4> d_fc_gpu_short;
  DevPtr<float3> d_fc_gpu_mesh;
  if (fc_target_cap > 0 && my_rank == 0) {
    d_fc_tgt_x.alloc(fc_target_cap, aux_dev_bytes);
    d_fc_tgt_h.alloc(fc_target_cap, aux_dev_bytes);
    d_fc_lids.alloc(fc_target_cap, aux_dev_bytes);
    d_fc_tgt_slab.alloc(fc_target_cap, aux_dev_bytes);
    check(cudaMemset(d_fc_tgt_slab.p, 0, size_t(fc_target_cap) * sizeof(int)));
    d_fc_gpu_short.alloc(fc_target_cap, aux_dev_bytes);
    d_fc_gpu_mesh.alloc(fc_target_cap, aux_dev_bytes);
    d_fc_short_ref.alloc(fc_target_cap, aux_dev_bytes);
    d_fc_total_ref.alloc(fc_target_cap, aux_dev_bytes);
    h_ewald_modes = build_ewald_modes(h.box);
    d_ewald_modes.alloc(h_ewald_modes.size(), aux_dev_bytes);
    d_ewald_C.alloc(h_ewald_modes.size(), aux_dev_bytes);
    d_ewald_S.alloc(h_ewald_modes.size(), aux_dev_bytes);
    check(cudaMemcpy(d_ewald_modes.p, h_ewald_modes.data(),
                     h_ewald_modes.size() * sizeof(EwaldMode), cudaMemcpyHostToDevice));
  }

  std::vector<double3> out_x;
  std::vector<float3> out_v;
  std::vector<float> out_pot;
  std::vector<float4> out_force0_f4;

  auto grow_resident_3d = [&](size_t need_n) {
    check(cudaSetDevice(G.dev));
    const size_t new_cap = std::max(compute_part_cap(need_n), size_t(double(need_n) * 1.15) + 32768);
    const size_t new_ncap = compute_ncap(new_cap, leaf_limit);
    std::vector<double3> sx(G.ns);
    std::vector<float3> sv(G.ns), sm(G.ns);
    std::vector<float> so(G.ns), smass, ssoft;
    std::vector<unsigned char> sb(G.ns);
    std::vector<uint32_t> sid(G.ns);
    if (G.ns > 0) {
      check(cudaMemcpyAsync(sx.data(), G.d_x.p, G.ns * sizeof(double3), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(sv.data(), G.d_v.p, G.ns * sizeof(float3), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(sm.data(), G.d_mesh_acc.p, G.ns * sizeof(float3), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(so.data(), G.d_old_acc.p, G.ns * sizeof(float), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(sb.data(), G.d_bins.p, G.ns * sizeof(unsigned char), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(sid.data(), G.d_orig_idx.p, G.ns * sizeof(uint32_t), cudaMemcpyDeviceToHost, G.compute_stream));
      if (!mass_uniform) {
        smass.resize(G.ns);
        check(cudaMemcpyAsync(smass.data(), G.d_mass.p, G.ns * sizeof(float), cudaMemcpyDeviceToHost, G.compute_stream));
      }
      if (!soft_uniform) {
        ssoft.resize(G.ns);
        check(cudaMemcpyAsync(ssoft.data(), G.d_soft.p, G.ns * sizeof(float), cudaMemcpyDeviceToHost, G.compute_stream));
      }
      check(cudaStreamSynchronize(G.compute_stream));
    }
    G.alloc_particles(new_cap, mass_uniform, soft_uniform);
    G.alloc_local_tree_nodes(new_ncap, cfg.lazy_tree_enabled);
    if (G.ns > 0) {
      check(cudaMemcpyAsync(G.d_x.p, sx.data(), G.ns * sizeof(double3), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_v.p, sv.data(), G.ns * sizeof(float3), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_mesh_acc.p, sm.data(), G.ns * sizeof(float3), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_old_acc.p, so.data(), G.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_bins.p, sb.data(), G.ns * sizeof(unsigned char), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaMemcpyAsync(G.d_orig_idx.p, sid.data(), G.ns * sizeof(uint32_t), cudaMemcpyHostToDevice, G.compute_stream));
      if (!mass_uniform)
        check(cudaMemcpyAsync(G.d_mass.p, smass.data(), G.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
      if (!soft_uniform)
        check(cudaMemcpyAsync(G.d_soft.p, ssoft.data(), G.ns * sizeof(float), cudaMemcpyHostToDevice, G.compute_stream));
      check(cudaStreamSynchronize(G.compute_stream));
    }
  };

  auto build_local_tree_3d = [&](bool compute_nvel) {
    const int nk = int(G.ns);
    if (nk == 0) {
      zero_empty_roots<<<blocks(std::max(1, G.n_own)), team, 0, G.compute_stream>>>(
          G.force_nodes.p, G.escape.p, G.first_child.p, G.n_own);
      pack_split_nodes_kernel<<<blocks(std::max(1, G.n_own)), team, 0, G.compute_stream>>>(
          G.force_nodes.p, G.first_child.p, G.node_meta.p, G.compact_m.p, G.n_own);
      G.nodes = unsigned(G.n_own);
      G.h_levels.fill(G.n_own);
      G.h_levels[0] = 0;
      return;
    }
    for (int attempt = 0; attempt < 4; ++attempt) {
      check(cudaMemsetAsync(G.d_tree_invalid.p, 0, sizeof(int), G.compute_stream));
      const int p_grid = blocks(nk);
      key_particles_fast_3d<<<p_grid, team, 0, G.compute_stream>>>(
          G.d_x.p, nk, G.d_cell_to_slot.p, gh, G.d_key_tmp.p, G.d_ord_tmp.p);
      size_t sb = G.sort_bytes;
      check(cub::DeviceRadixSort::SortPairs(
          G.d_sort_storage.p, sb, G.d_key_tmp.p, G.d_keys_sorted.p,
          G.d_ord_tmp.p, G.order.p, nk, 0, 62, G.compute_stream));
      extract_home_slots_3d_kernel<<<p_grid, team, 0, G.compute_stream>>>(
          G.d_keys_sorted.p, G.home_slot.p, nk);
      roots<<<blocks(G.n_own), team, 0, G.compute_stream>>>(
          G.d_keys_sorted.p, nk, G.geometry.p, G.n_own);
      tree_offsets_begin<<<1, 1, 0, G.compute_stream>>>(G.levels.p, G.node_count.p, G.n_own);

      const int n_grid = blocks(int(G.node_cap));
      for (int depth = 0; depth < tree_levels; ++depth) {
        split<<<n_grid, team, 0, G.compute_stream>>>(
            G.d_keys_sorted.p, G.geometry.p, 0, 0, depth, gh.leaf_limit,
            G.node_count.p, int(G.node_cap), G.d_tree_invalid.p, G.levels.p);
        tree_offsets_end<<<1, 1, 0, G.compute_stream>>>(
            G.levels.p, G.node_count.p, depth, int(G.node_cap));
        tree_links<<<n_grid, team, 0, G.compute_stream>>>(
            G.geometry.p, G.escape.p, G.levels.p, depth, G.first_child.p);
      }
      unsigned int nc = 0;
      int inv_cap = 0;
      check(cudaMemcpyAsync(&nc, G.node_count.p, sizeof(unsigned int), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(&inv_cap, G.d_tree_invalid.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaMemcpyAsync(G.h_levels.data(), G.levels.p, size_t(tree_levels + 2) * sizeof(int),
                            cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaStreamSynchronize(G.compute_stream));
      if (inv_cap != 0 || nc >= G.node_cap) {
        if (attempt < 3) {
          const size_t next_nc = std::max(G.node_cap * 3 / 2, size_t(nc) + 131072);
          G.alloc_local_tree_nodes(next_nc, cfg.lazy_tree_enabled);
          continue;
        }
        throw std::runtime_error("MPI 3D tree split overflow on rank " + std::to_string(my_rank));
      }

      check_tree_limit<<<n_grid, team, 0, G.compute_stream>>>(
          G.geometry.p, G.levels.p, tree_levels, gh.leaf_limit, G.d_tree_invalid.p);
      tree_links<<<n_grid, team, 0, G.compute_stream>>>(
          G.geometry.p, G.escape.p, G.levels.p, tree_levels, G.first_child.p);
      for (int depth = tree_levels; depth >= 0; --depth) {
        multipoles_fast_3d<<<n_grid, team, 0, G.compute_stream>>>(
            G.geometry.p, G.force_nodes.p, G.d_node_com.p, G.d_x.p,
            mass_uniform ? nullptr : G.d_mass.p, first_mass, soft,
            G.order.p, depth, G.d_slot_to_cell.p, gh, G.levels.p,
            soft_uniform ? nullptr : G.d_soft.p);
      }
      gather_sources_slab<<<p_grid, team, 0, G.compute_stream>>>(
          G.d_x.p, mass_uniform ? nullptr : G.d_mass.p, first_mass,
          soft_uniform ? nullptr : G.d_soft.p,
          G.order.p, G.packed_sources.p, G.packed_soft.p, nk);
      if (compute_nvel) {
        for (int depth = tree_levels; depth >= 0; --depth) {
          const int lcnt = G.h_levels[depth + 1] - G.h_levels[depth];
          if (lcnt <= 0) continue;
          compute_node_vel_active_3d_kernel<<<blocks(lcnt), team, 0, G.compute_stream>>>(
              G.geometry.p, G.force_nodes.p, G.d_v.p,
              mass_uniform ? nullptr : G.d_mass.p, first_mass,
              G.order.p, depth, G.levels.p, nullptr, G.d_slot_to_cell.p, G.node_vel_max.p);
        }
      }
      pack_split_nodes_kernel<<<blocks(int(nc)), team, 0, G.compute_stream>>>(
          G.force_nodes.p, G.first_child.p, G.node_meta.p, G.compact_m.p, int(nc));
      check(cudaGetLastError());
      int inv_leaf = 0;
      check(cudaMemcpyAsync(&inv_leaf, G.d_tree_invalid.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
      check(cudaStreamSynchronize(G.compute_stream));
      if (inv_leaf != 0) {
        throw std::runtime_error("MPI 3D tree depth>14 leaf limit failed on rank " + std::to_string(my_rank));
      }
      G.nodes = nc;
      return;
    }
  };

  std::array<uint8_t, N_TOP_CELLS> h_cell_active{};
  std::array<unsigned long long, N_TOP_CELLS> cell_act_accum{};
  int steps_since_rebal = 0;
  int rebalance_count = 0;
  unsigned int pm_updates = 0;
  double last_pm_span_scale = log_span / double(cfg.pm_intervals);
  float last_dt_max_rms = dt_displacement_default;
  uint64_t split_batches_run = 0, split_overflow_retries = 0;
  uint64_t total_lazy_steps = 0, total_full_rebuilds = 0;
  double cum_max_disp = 0.0;
  int lazy_steps_since_rebuild = 0;
  bool all_trees_built = false;
  bool ghost_installed = false;

  double pm_mpi_acc_ms = 0.0, ghost_mpi_acc_ms = 0.0, mig_mpi_acc_ms = 0.0;
  double total_ghost_exchange_ms = 0.0;

  constexpr int max_steps_guard = 1000000;
  int global_iteration = 0;
  size_t sanity_bad = 0;
  double sanity_v2_sum = 0.0, sanity_pot_sum = 0.0;
  float sanity_pot_min = FLT_MAX, sanity_pot_max = -FLT_MAX;

  double* d_energy_accum = nullptr;
  if (cfg.energy_log) {
    check(cudaMalloc(&d_energy_accum, 10 * sizeof(double)));
  }
  double* d_one_d3_dummy = nullptr;
  (void)d_one_d3_dummy;

  std::vector<std::vector<char>> h_ghost_send_scratch, h_ghost_recv_scratch;

  const double evo_start = wall();

  for (size_t seg = 0; seg + 1 < knots.size(); ++seg) {
    const double seg_a_begin = knots[seg], seg_a_end = knots[seg + 1];
    const bool is_last_seg = (seg + 2 == knots.size());
    const double seg_span = cfg.comoving ? log(seg_a_end / seg_a_begin) : (seg_a_end - seg_a_begin);
    p.a_begin = seg_a_begin;
    p.log_span = seg_span;

    Tick ti = 0, previous = 0;
    Tick next_pm_boundary = timeline, current_pm_span = p.pm_span;
    Tick next_pm_span = p.pm_span, prev_pm_span = 0, short_span_cap = p.short_span_cap;
    double pm_kick = 0.0;
    float dt_max_rms = dt_displacement_default;
    all_trees_built = false;
    ghost_installed = false;
    lazy_steps_since_rebuild = 0;
    cum_max_disp = 0.0;

    if (seg > 0) {
      check(cudaMemsetAsync(G.d_mesh_acc.p, 0, G.part_cap * sizeof(float3), G.compute_stream));
      check(cudaMemsetAsync(G.d_old_acc.p, 0, G.part_cap * sizeof(float), G.compute_stream));
      check(cudaMemsetAsync(G.d_bins.p, 1, G.part_cap * sizeof(unsigned char), G.compute_stream));
      check(cudaStreamSynchronize(G.compute_stream));
    }

    for (int iteration = 0;; ++iteration, ++global_iteration) {
      if (iteration > max_steps_guard) throw std::runtime_error("Exceeded max_steps_guard");

      const double step_start = wall();
      const bool is_pm_step = (ti == 0 || ti == next_pm_boundary);
      const bool is_final = (ti == timeline);
      const bool is_run_final = is_final && is_last_seg;
      const double drift_fac = (ti > 0) ? p.factors(previous, ti, true) : 0.0;
      const double a = p.scale(ti);
      const double H_cur = p.H(a);
      const bool need_vel_sum = (is_pm_step && !is_final && cfg.dynamic_pm);
      const bool do_fc_now = (fc_target_cap > 0) && ((ti == 0 && seg == 0) || is_run_final);

      const double wc = h.box / double(N_TOP);
      const double max_safe_disp = std::min(cfg.lazy_disp_frac * r_s, 0.35 * std::max(0.0, wc - double(p.cutoff)));
      const bool eligible_lazy = cfg.lazy_tree_enabled && (ti > 0) && !is_pm_step &&
          !is_final && !do_fc_now && all_trees_built &&
          (lazy_steps_since_rebuild < cfg.lazy_max_steps) && (max_safe_disp > 0.0);
      bool do_lazy_step = false;
      bool force_rebuild_all = is_pm_step || (ti == 0) || !all_trees_built || !ghost_installed;
      bool did_rebalance = false;
      double step_max_disp = 0.0;
      uint64_t total_migrants = 0;

      if (is_final && !cfg.skip_output_write && my_rank == 0) {
        out_x.resize(N); out_v.resize(N); out_pot.resize(N);
      }

      double drift_ms = 0.0, pm_ms = 0.0, tree_ms = 0.0, force_ms = 0.0;
      double kick_ms = 0.0, ghost_ms = 0.0, export_ms = 0.0;
      const bool check_rebal_this_step = (num_ranks > 1 && cfg.rebalance_interval > 0 &&
          ti > 0 && !is_final && !do_fc_now &&
          (global_iteration % cfg.rebalance_interval == 0));

      // ================= STEP A: Drift + MPI Migrate + Optional Dynamic Rebalance =================
      const double t_a0 = wall();
      uint64_t loc_active = 0;
      Tick loc_inactive_min = timeline, loc_active_next = timeline;
      double loc_v2_sum = 0.0;
      float loc_max_v = 0.f;

      auto scan_local_gpu = [&](bool track_mv) {
        if (G.ns == 0) {
          loc_active = 0;
          loc_inactive_min = timeline;
          loc_v2_sum = 0.0;
          loc_max_v = 0.f;
          return;
        }
        const int g_blk = std::min(scan_grid_max, std::max(1, int((G.ns + scan_team - 1) / scan_team)));
        scan_slab_active_v2_kernel<<<g_blk, scan_team, 0, G.compute_stream>>>(
            G.d_bins.p, G.d_v.p, int(G.ns), ti, (ti == 0 || is_final || is_pm_step),
            need_vel_sum, track_mv, G.d_scan_tmp.p);
        check(cudaMemcpyAsync(G.h_scan.data(), G.d_scan_tmp.p,
                              size_t(g_blk) * sizeof(ScanPartial), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaStreamSynchronize(G.compute_stream));
        uint64_t act = 0;
        Tick nx = timeline;
        double v2 = 0.0;
        float mv = 0.f;
        for (int b = 0; b < g_blk; ++b) {
          act += G.h_scan[b].active;
          if (G.h_scan[b].min_nx < nx) nx = G.h_scan[b].min_nx;
          v2 += G.h_scan[b].v2_sum;
          if (G.h_scan[b].max_v > mv) mv = G.h_scan[b].max_v;
        }
        loc_active = act;
        loc_inactive_min = nx;
        loc_v2_sum = v2;
        loc_max_v = mv;
      };

      auto exchange_migrants_mpi = [&](const std::vector<DevMigrant>& h_out) -> uint64_t {
        if (num_ranks == 1) return 0;
        const double tm0 = MPI_Wtime();
        std::vector<int> send_cnt(num_ranks, 0);
        for (const auto& m : h_out) {
          if (m.dst_slab >= 0 && m.dst_slab < num_ranks && m.dst_slab != my_rank) {
            ++send_cnt[m.dst_slab];
          }
        }
        std::vector<int> sdispls(num_ranks, 0);
        int tot_out = 0;
        for (int r = 0; r < num_ranks; ++r) {
          sdispls[r] = tot_out;
          tot_out += send_cnt[r];
        }
        std::vector<DevMigrant> send_packed(size_t(std::max(1, tot_out)));
        std::vector<int> cur = sdispls;
        for (const auto& m : h_out) {
          if (m.dst_slab >= 0 && m.dst_slab < num_ranks && m.dst_slab != my_rank) {
            send_packed[cur[m.dst_slab]++] = m;
          }
        }
        std::vector<int> recv_cnt(num_ranks, 0), recv_displs(num_ranks, 0);
        GRAV_MPI_CHECK(MPI_Alltoall(send_cnt.data(), 1, MPI_INT, recv_cnt.data(), 1, MPI_INT, mpi.world));
        int tot_in = 0;
        for (int r = 0; r < num_ranks; ++r) {
          recv_displs[r] = tot_in;
          tot_in += recv_cnt[r];
        }
        std::vector<int> send_b(num_ranks), sd_b(num_ranks), recv_b(num_ranks), rd_b(num_ranks);
        for (int r = 0; r < num_ranks; ++r) {
          send_b[r] = send_cnt[r] * int(sizeof(DevMigrant));
          sd_b[r] = sdispls[r] * int(sizeof(DevMigrant));
          recv_b[r] = recv_cnt[r] * int(sizeof(DevMigrant));
          rd_b[r] = recv_displs[r] * int(sizeof(DevMigrant));
        }
        std::vector<DevMigrant> in_vec(size_t(std::max(1, tot_in)));
        GRAV_MPI_CHECK(MPI_Alltoallv(
            send_packed.data(), send_b.data(), sd_b.data(), MPI_BYTE,
            in_vec.data(), recv_b.data(), rd_b.data(), MPI_BYTE, mpi.world));

        uint64_t loc_out_u64 = uint64_t(tot_out), glb_out_u64 = 0;
        GRAV_MPI_CHECK(MPI_Allreduce(&loc_out_u64, &glb_out_u64, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, mpi.world));
        mig_mpi_acc_ms += (MPI_Wtime() - tm0) * 1e3;

        if (tot_in > 0) {
          const size_t in_cnt = size_t(tot_in);
          if (G.ns + in_cnt > G.part_cap) grow_resident_3d(G.ns + in_cnt);
          G.ensure_mig_in(in_cnt);
          check(cudaMemcpyAsync(G.d_mig_in.p, in_vec.data(),
                                in_cnt * sizeof(DevMigrant), cudaMemcpyHostToDevice, G.compute_stream));
          append_migrants_kernel<<<blocks(int(in_cnt)), team, 0, G.compute_stream>>>(
              G.d_x.p, G.d_v.p, G.d_mesh_acc.p, G.d_old_acc.p, G.d_bins.p,
              G.d_orig_idx.p, mass_uniform ? nullptr : G.d_mass.p,
              soft_uniform ? nullptr : G.d_soft.p,
              G.d_mig_in.p, int(G.ns), int(in_cnt));
          G.ns += in_cnt;
          check(cudaStreamSynchronize(G.compute_stream));
        }
        return glb_out_u64;
      };

      auto relocate_by_device_owner = [&](const int* d_owner_map) {
        std::vector<DevMigrant> h_out;
        if (G.ns > 0) {
          flag_plan_migrants_3d_kernel<<<blocks(int(G.ns)), team, 0, G.compute_stream>>>(
              G.d_x.p, d_owner_map, G.d_flags.p, int(G.ns), my_rank, inv_cell_w);
          cub::CountingInputIterator<int> seq(0);
          size_t selb = G.select_bytes;
          check(cub::DeviceSelect::Flagged(
              G.d_select_storage.p, selb, seq, G.d_flags.p,
              G.d_active_ids.p, G.d_active_count.p, int(G.ns), G.compute_stream));
          int m_cnt = 0;
          check(cudaMemcpyAsync(&m_cnt, G.d_active_count.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
          check(cudaStreamSynchronize(G.compute_stream));
          if (m_cnt > 0) {
            G.ensure_mig_out(size_t(m_cnt));
            pack_migrants_3d_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                G.d_x.p, G.d_v.p, G.d_mesh_acc.p, G.d_old_acc.p, G.d_bins.p,
                G.d_orig_idx.p, mass_uniform ? nullptr : G.d_mass.p, first_mass,
                soft_uniform ? nullptr : G.d_soft.p, soft,
                G.d_active_ids.p, m_cnt, inv_cell_w, d_owner_map, G.d_mig_out.p);
            h_out.resize(size_t(m_cnt));
            check(cudaMemcpyAsync(h_out.data(), G.d_mig_out.p,
                                  size_t(m_cnt) * sizeof(DevMigrant), cudaMemcpyDeviceToHost, G.compute_stream));
            const int keep_ns = int(G.ns) - m_cnt;
            G.ensure_tail_cap(size_t(m_cnt));
            mark_tail_survivors_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                G.d_flags.p, keep_ns, m_cnt, G.d_tail_flags.p);
            cub::CountingInputIterator<int> tail_seq(keep_ns);
            size_t selb_t = G.select_bytes;
            check(cub::DeviceSelect::Flagged(
                G.d_select_storage.p, selb_t, tail_seq, G.d_tail_flags.p,
                G.d_tail_keep.p, G.d_tail_count.p, m_cnt, G.compute_stream));
            compact_migrants_parallel_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                G.d_x.p, G.d_v.p, G.d_mesh_acc.p, G.d_old_acc.p, G.d_bins.p,
                G.d_orig_idx.p, mass_uniform ? nullptr : G.d_mass.p,
                soft_uniform ? nullptr : G.d_soft.p,
                G.d_active_ids.p, G.d_tail_keep.p, G.d_tail_count.p);
            G.ns -= size_t(m_cnt);
            check(cudaStreamSynchronize(G.compute_stream));
          }
        }
        (void)exchange_migrants_mpi(h_out);
      };

      if (ti > 0) {
        if (eligible_lazy) {
          scan_local_gpu(true);
          uint64_t tot_act = loc_active;
          float gmv = loc_max_v;
          if (num_ranks > 1) {
            GRAV_MPI_CHECK(MPI_Allreduce(&loc_active, &tot_act, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, mpi.world));
            GRAV_MPI_CHECK(MPI_Allreduce(&loc_max_v, &gmv, 1, MPI_FLOAT, MPI_MAX, mpi.world));
          }
          step_max_disp = drift_fac * double(gmv);
          do_lazy_step = (tot_act < uint64_t(double(N) * cfg.lazy_active_frac)) &&
                         ((cum_max_disp + step_max_disp) < max_safe_disp);
          if (!do_lazy_step && lazy_steps_since_rebuild > 0) force_rebuild_all = true;
        }

        if (do_lazy_step) {
          if (G.ns > 0) {
            drift_only_kernel<<<blocks(int(G.ns)), team, 0, G.compute_stream>>>(
                G.d_x.p, G.d_v.p, int(G.ns), drift_fac, h.box, true);
            check(cudaStreamSynchronize(G.compute_stream));
          }
        } else {
          std::vector<DevMigrant> h_out;
          if (G.ns > 0) {
            drift_flag_migrants_3d_kernel<<<blocks(int(G.ns)), team, 0, G.compute_stream>>>(
                G.d_x.p, G.d_v.p, G.d_cell_owner.p, G.d_flags.p,
                int(G.ns), my_rank, drift_fac, h.box, inv_cell_w);
            cub::CountingInputIterator<int> seq(0);
            size_t selb = G.select_bytes;
            check(cub::DeviceSelect::Flagged(
                G.d_select_storage.p, selb, seq, G.d_flags.p,
                G.d_active_ids.p, G.d_active_count.p, int(G.ns), G.compute_stream));
            int m_cnt = 0;
            check(cudaMemcpyAsync(&m_cnt, G.d_active_count.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
            check(cudaStreamSynchronize(G.compute_stream));
            if (m_cnt > 0) {
              G.ensure_mig_out(size_t(m_cnt));
              pack_migrants_3d_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                  G.d_x.p, G.d_v.p, G.d_mesh_acc.p, G.d_old_acc.p, G.d_bins.p,
                  G.d_orig_idx.p, mass_uniform ? nullptr : G.d_mass.p, first_mass,
                  soft_uniform ? nullptr : G.d_soft.p, soft,
                  G.d_active_ids.p, m_cnt, inv_cell_w, G.d_cell_owner.p, G.d_mig_out.p);
              h_out.resize(size_t(m_cnt));
              check(cudaMemcpyAsync(h_out.data(), G.d_mig_out.p,
                                    size_t(m_cnt) * sizeof(DevMigrant), cudaMemcpyDeviceToHost, G.compute_stream));
              const int keep_ns = int(G.ns) - m_cnt;
              G.ensure_tail_cap(size_t(m_cnt));
              mark_tail_survivors_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                  G.d_flags.p, keep_ns, m_cnt, G.d_tail_flags.p);
              cub::CountingInputIterator<int> tail_seq(keep_ns);
              size_t selb_t = G.select_bytes;
              check(cub::DeviceSelect::Flagged(
                  G.d_select_storage.p, selb_t, tail_seq, G.d_tail_flags.p,
                  G.d_tail_keep.p, G.d_tail_count.p, m_cnt, G.compute_stream));
              compact_migrants_parallel_kernel<<<blocks(m_cnt), team, 0, G.compute_stream>>>(
                  G.d_x.p, G.d_v.p, G.d_mesh_acc.p, G.d_old_acc.p, G.d_bins.p,
                  G.d_orig_idx.p, mass_uniform ? nullptr : G.d_mass.p,
                  soft_uniform ? nullptr : G.d_soft.p,
                  G.d_active_ids.p, G.d_tail_keep.p, G.d_tail_count.p);
              G.ns -= size_t(m_cnt);
              check(cudaStreamSynchronize(G.compute_stream));
            }
          }
          total_migrants = exchange_migrants_mpi(h_out);
          if (lazy_steps_since_rebuild > 0) force_rebuild_all = true;
          if (!eligible_lazy || total_migrants > 0) scan_local_gpu(false);
        }
      } else {
        scan_local_gpu(false);
      }

      if (check_rebal_this_step) {
        check(cudaMemsetAsync(G.d_cell_counts.p, 0, N_TOP_CELLS * sizeof(unsigned long long), G.compute_stream));
        check(cudaMemsetAsync(G.d_cell_act_counts.p, 0, N_TOP_CELLS * sizeof(unsigned long long), G.compute_stream));
        if (G.ns > 0) {
          const int c_grid = std::min(512, blocks(int(G.ns)));
          count_cells_3d_kernel<<<c_grid, team, 0, G.compute_stream>>>(
              G.d_x.p, G.d_bins.p, int(G.ns), inv_cell_w, ti, is_final,
              G.d_cell_counts.p, G.d_cell_act_counts.p);
        }
        std::array<unsigned long long, N_TOP_CELLS> loc_cc{}, loc_ca{}, glb_cc{}, glb_ca{};
        check(cudaMemcpyAsync(loc_cc.data(), G.d_cell_counts.p,
                              N_TOP_CELLS * sizeof(unsigned long long), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaMemcpyAsync(loc_ca.data(), G.d_cell_act_counts.p,
                              N_TOP_CELLS * sizeof(unsigned long long), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaStreamSynchronize(G.compute_stream));

        GRAV_MPI_CHECK(MPI_Allreduce(loc_cc.data(), glb_cc.data(), N_TOP_CELLS, MPI_UNSIGNED_LONG_LONG, MPI_SUM, mpi.world));
        GRAV_MPI_CHECK(MPI_Allreduce(loc_ca.data(), glb_ca.data(), N_TOP_CELLS, MPI_UNSIGNED_LONG_LONG, MPI_SUM, mpi.world));

        std::array<uint64_t, N_TOP_CELLS> cur_cnt{};
        std::array<double, N_TOP_CELLS> cur_w{};
        const int win = std::max(1, steps_since_rebal + 1);
        for (int cid = 0; cid < N_TOP_CELLS; ++cid) {
          cur_cnt[cid] = uint64_t(glb_cc[cid]);
          cell_act_accum[cid] += glb_ca[cid];
          cur_w[cid] = double(glb_cc[cid]) + 0.5 * double(cell_act_accum[cid]) / double(win);
        }
        cell_act_accum.fill(0ULL);
        steps_since_rebal = 0;
        const double cur_imb = detail::compute_load_imbalance(cur_w, plan.cell_owner, num_ranks);
        Domain3DPlan cand_plan = build_domain_3d_plan(
            cur_cnt, cur_w, num_ranks, cfg.domain_mode, pm_side, nullptr);
        const double new_imb = cand_plan.stats.w_imb;
        const double gain = (cur_imb - new_imb) / std::max(cur_imb, 1e-12);
        if (gain >= cfg.rebalance_min_gain) {
          did_rebalance = true;
          plan = std::move(cand_plan);
          mstats = compute_multinode_comm_stats(plan, mpi.rank_to_node, &cur_cnt);
          force_rebuild_all = true;
          if (use_mpi_pm) mpi_pm.reconfigure_x_planes(plan.gpu_x_planes[my_rank]);
          ++rebalance_count;
          if (my_rank == 0) {
            std::cout << std::setprecision(6)
                      << "REBALANCE {\"step\":" << global_iteration
                      << ",\"a\":" << a
                      << ",\"mode\":\"" << domain_mode_name(cfg.domain_mode)
                      << "\",\"cur_imb\":" << cur_imb
                      << ",\"new_imb\":" << new_imb
                      << ",\"gain\":" << gain
                      << ",\"ghost_total\":" << plan.stats.ghost_total
                      << ",\"inter_node_ghost_cells\":" << mstats.inter_node_ghost_cells
                      << ",\"max_x_slabs\":" << plan.stats.max_x_slabs << "}" << std::endl;
          }
          DevPtr<int> d_new_owner;
          d_new_owner.alloc(N_TOP_CELLS, G.device_bytes);
          check(cudaMemcpyAsync(d_new_owner.p, plan.cell_owner.data(),
                                N_TOP_CELLS * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
          relocate_by_device_owner(d_new_owner.p);
          d_new_owner.free(G.device_bytes);
          upload_plan_to_gpu(plan);
          scan_local_gpu(false);
          setup_all_edges(plan);
        }
      } else if (ti > 0) {
        ++steps_since_rebal;
      }

      if (do_lazy_step) {
        cum_max_disp += step_max_disp;
        ++lazy_steps_since_rebuild;
        ++total_lazy_steps;
      } else {
        cum_max_disp = 0.0;
        lazy_steps_since_rebuild = 0;
        ++total_full_rebuilds;
      }
      {
        uint64_t loc_ns = uint64_t(G.ns), max_ns = 0;
        if (num_ranks > 1) {
          GRAV_MPI_CHECK(MPI_Allreduce(&loc_ns, &max_ns, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, mpi.world));
        } else {
          max_ns = loc_ns;
        }
        peak_gpu_seen = std::max(peak_gpu_seen, size_t(max_ns));
      }
      drift_ms = (wall() - t_a0) * 1000.0;

      // Collect global active & vel sums for PM timestep / cell active mask
      uint64_t glb_active_step = loc_active;
      double glb_v2_sum = loc_v2_sum;
      if (num_ranks > 1) {
        GRAV_MPI_CHECK(MPI_Allreduce(&loc_active, &glb_active_step, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, mpi.world));
        if (need_vel_sum) {
          GRAV_MPI_CHECK(MPI_Allreduce(&loc_v2_sum, &glb_v2_sum, 1, MPI_DOUBLE, MPI_SUM, mpi.world));
        }
      }

      if (is_pm_step) {
        ++pm_updates;
        if (is_final) {
          pm_kick = p.factors(timeline - current_pm_span / 2, timeline, false);
        } else {
          if (cfg.dynamic_pm) {
            const double rms_vel_dm = glb_v2_sum / double(N);
            const double dt_dm = rms_vel_dm > 0.0 ? (a * a * p.rms_length / sqrt(rms_vel_dm)) : 1e30;
            const float dt_rms = float(dt_dm * p.max_dt_RMS_factor);
            const double dt_mesh = fmin(double(dt_rms) * H_cur, p.dt_max);
            const double raw_ticks = dt_mesh * (double(timeline) / p.log_span);
            Tick trial = raw_ticks >= double(timeline) ? timeline : (raw_ticks < 4.0 ? 4ULL : Tick(raw_ticks));
            Tick new_dti = 1ULL << (63 - __builtin_clzll(trial));
            if (ti > 0) {
              const Tick old_dti = next_pm_span;
              if (new_dti > 2 * old_dti) new_dti = 2 * old_dti;
              if (new_dti > old_dti && (ti & (new_dti - 1ULL)) != 0) new_dti = old_dti;
              prev_pm_span = old_dti;
            } else {
              prev_pm_span = 0;
            }
            next_pm_span = new_dti;
            short_span_cap = new_dti;
            dt_max_rms = dt_rms;
          } else {
            prev_pm_span = (ti == 0) ? 0 : p.pm_span;
            next_pm_span = p.pm_span;
            short_span_cap = p.short_span_cap;
            dt_max_rms = dt_displacement_default;
          }
          current_pm_span = next_pm_span;
          last_pm_span_scale = double(current_pm_span) / double(timeline) * p.log_span;
          last_dt_max_rms = dt_max_rms;
          const Tick lo = (ti == 0) ? 0 : (ti - prev_pm_span / 2);
          const Tick hi = ti + next_pm_span / 2;
          pm_kick = p.factors(lo, hi, false);
          next_pm_boundary = ti + next_pm_span;
        }
      }

      // ================= STEP B: Distributed MPI PM =================
      if (is_pm_step) {
        const double t_pm0 = wall();
        if (use_mpi_pm) {
          mpi_pm.begin_deposit();
          if (G.ns > 0) {
            mpi_pm.deposit(G.d_x.p, mass_uniform ? nullptr : G.d_mass.p, first_mass, G.ns, h.box);
          }
          mpi_pm.solve_and_gather(h.box, r_s);
          pm_mpi_acc_ms += mpi_pm.last_comm_ms();
          if (G.ns > 0) {
            mpi_pm.interpolate(G.d_x.p, G.ns, h.box, p.G, G.d_mesh_acc.p, nullptr);
          }
          check(cudaStreamSynchronize(G.compute_stream));
        } else {
          check(cudaMemsetAsync(G.d_pm_density.p, 0, pm_points * sizeof(double), G.compute_stream));
          if (G.ns > 0) {
            grav_pm_cuda_deposit_buf(
                G.d_x.p, mass_uniform ? nullptr : G.d_mass.p, first_mass,
                G.ns, G.d_pm_density.p, pm_side, h.box, G.compute_stream);
          }
          grav_pm_cuda_solve_on_device(G.d_pm_density.p, G.d_pm_field.p, pm_side, h.box, r_s, G.compute_stream);
          if (G.ns > 0) {
            grav_pm_cuda_interpolate_buf(
                G.d_x.p, G.ns, G.d_pm_field.p, pm_side, h.box, p.G, G.d_mesh_acc.p, nullptr, G.compute_stream);
          }
          check(cudaStreamSynchronize(G.compute_stream));
        }
        pm_ms = (wall() - t_pm0) * 1000.0;
      }

      // ================= STEP C: Local Tree Build / Drift =================
      const double t_c0 = wall();
      int loc_build = 0;
      if (do_lazy_step) {
        std::array<uint32_t, 54> loc_act_bits{};
        if (G.ns > 0) {
          drift_tree_nodes_kernel<<<blocks(int(G.nodes)), team, 0, G.compute_stream>>>(
              G.force_nodes.p, G.node_vel_max.p, G.node_meta.p,
              int(G.nodes), float(drift_fac), float(h.box));
          if (loc_active > 0) {
            gather_sources_slab<<<blocks(int(G.ns)), team, 0, G.compute_stream>>>(
                G.d_x.p, mass_uniform ? nullptr : G.d_mass.p, first_mass,
                nullptr, G.order.p, G.packed_sources.p, nullptr, int(G.ns));
            check(cudaMemsetAsync(G.d_active_bits.p, 0, 54 * sizeof(uint32_t), G.compute_stream));
            mark_active_home_cells_3d_kernel<<<blocks(int(G.ns)), team, 0, G.compute_stream>>>(
                G.home_slot.p, G.order.p, G.d_bins.p, G.d_slot_to_cell.p,
                int(G.ns), ti, G.d_active_bits.p);
            check(cudaMemcpyAsync(loc_act_bits.data(), G.d_active_bits.p,
                                  54 * sizeof(uint32_t), cudaMemcpyDeviceToHost, G.compute_stream));
          }
        }
        check(cudaStreamSynchronize(G.compute_stream));
        std::array<uint32_t, 54> glb_act_bits = loc_act_bits;
        if (num_ranks > 1) {
          GRAV_MPI_CHECK(MPI_Allreduce(loc_act_bits.data(), glb_act_bits.data(), 54, MPI_UINT32_T, MPI_BOR, mpi.world));
        }
        h_cell_active.fill(0);
        for (int w = 0; w < 54; ++w) {
          uint32_t m = glb_act_bits[w];
          while (m) {
            const int b = __builtin_ctz(m);
            h_cell_active[(w << 5) + b] = 1;
            m &= (m - 1u);
          }
        }
      } else {
        build_local_tree_3d(cfg.lazy_tree_enabled && !is_final && !(is_pm_step || loc_active > 0));
        loc_build = 1;
        all_trees_built = true;
      }
      check(cudaStreamSynchronize(G.compute_stream));
      tree_ms = (wall() - t_c0) * 1000.0;

      // ================= STEP C2 (Part 1: Pack & Meta Exchange) =================
      const bool need_ghost = (num_ranks > 1);
      if (need_ghost) {
        if (do_lazy_step) {
          std::vector<int> send_meta(size_t(num_ranks) * 3, 0), recv_meta(size_t(num_ranks) * 3, 0);
          for (int dst = 0; dst < num_ranks; ++dst) {
            if (dst == my_rank) continue;
            auto& ed = edges[dst];
            ed.h_lazy_descs_send.clear();
            ed.lazy_M_out = ed.lazy_tot_nodes_out = ed.lazy_tot_parts_out = 0;
            if (ed.E_out == 0) continue;
            const auto& dst_gslots = plan.export_dst_ghost_slots[my_rank][dst];
            int n_acc = 0, p_acc = 0;
            for (int e = 0; e < ed.E_out; ++e) {
              const int cid = plan.export_cells[my_rank][dst][e];
              bool need_e = false;
              int tx, ty, tz;
              cell_coords(cid, tx, ty, tz);
              for (int dx = -1; dx <= 1 && !need_e; ++dx) {
                const int nx = wrap12(tx + dx);
                for (int dy = -1; dy <= 1 && !need_e; ++dy) {
                  const int ny = wrap12(ty + dy);
                  for (int dz = -1; dz <= 1; ++dz) {
                    const int nz = wrap12(tz + dz);
                    const int ncid = cell_id(nx, ny, nz);
                    if (plan.cell_owner[ncid] == dst && h_cell_active[ncid]) {
                      need_e = true; break;
                    }
                  }
                }
              }
              if (!need_e) continue;
              const int n_cnt = ed.h_node_cnt[e];
              const int p_cnt = ed.h_part_cnt[e];
              Dom3DLazyRootDesc d{};
              d.e = e;
              d.src_part_start = ed.h_part_src_start[e];
              d.dst_gslot = dst_gslots[e];
              d.src_nonroot_off = ed.h_nonroot_off[e];
              d.dst_desc_base = ed.h_nonroot_off[e];
              d.n_cnt = n_cnt;
              d.p_cnt = p_cnt;
              d.lazy_node_off = n_acc;
              d.lazy_part_off = p_acc;
              d.dst_part_off = ed.h_part_off[e];
              ed.h_lazy_descs_send.push_back(d);
              n_acc += n_cnt;
              p_acc += p_cnt;
            }
            ed.lazy_M_out = int(ed.h_lazy_descs_send.size());
            ed.lazy_tot_nodes_out = n_acc;
            ed.lazy_tot_parts_out = p_acc;
            send_meta[dst * 3 + 0] = ed.lazy_M_out;
            send_meta[dst * 3 + 1] = n_acc;
            send_meta[dst * 3 + 2] = p_acc;
            if (ed.lazy_M_out > 0) {
              const size_t total_f4 = size_t(n_acc) + size_t(p_acc);
              const size_t ld_bytes = size_t(ed.lazy_M_out) * sizeof(Dom3DLazyRootDesc);
              if (size_t(ed.lazy_M_out) > dev_len(ed.d_lazy_descs_src)) {
                ed.d_lazy_descs_src.alloc(size_t(ed.lazy_M_out), G.device_bytes);
              }
              if (total_f4 > dev_len(ed.d_lazy_send)) {
                ed.d_lazy_send.alloc(std::max(total_f4, total_f4 * 3 / 2 + 4096), G.device_bytes);
              }
              check(cudaMemcpyAsync(ed.d_lazy_descs_src.p, ed.h_lazy_descs_send.data(),
                                    ld_bytes, cudaMemcpyHostToDevice, G.compute_stream));
              pack_lazy_edge_kernel<<<blocks_warps(ed.lazy_M_out * 4), team, 0, G.compute_stream>>>(
                  G.node_meta.p, ed.d_bnd_src_node_map.p,
                  G.d_x.p, mass_uniform ? nullptr : G.d_mass.p, first_mass, G.order.p,
                  ed.d_lazy_descs_src.p, ed.lazy_M_out, ed.E_out, n_acc, ed.d_lazy_send.p);
            }
          }
          check(cudaStreamSynchronize(G.compute_stream));
          GRAV_MPI_CHECK(MPI_Alltoall(send_meta.data(), 3, MPI_INT, recv_meta.data(), 3, MPI_INT, mpi.world));
          for (int src = 0; src < num_ranks; ++src) {
            if (src == my_rank) continue;
            auto& ed = edges[src];
            ed.lazy_M_in = recv_meta[src * 3 + 0];
            ed.lazy_tot_nodes_in = recv_meta[src * 3 + 1];
            ed.lazy_tot_parts_in = recv_meta[src * 3 + 2];
            if (ed.lazy_M_in > 0) {
              ed.h_lazy_descs_recv.resize(size_t(ed.lazy_M_in));
              const size_t total_f4 = size_t(ed.lazy_tot_nodes_in) + size_t(ed.lazy_tot_parts_in);
              if (size_t(ed.lazy_M_in) > dev_len(ed.d_lazy_descs_dst)) {
                ed.d_lazy_descs_dst.alloc(size_t(ed.lazy_M_in), G.device_bytes);
              }
              if (total_f4 > dev_len(ed.d_lazy_recv)) {
                ed.d_lazy_recv.alloc(std::max(total_f4, total_f4 * 3 / 2 + 4096), G.device_bytes);
              }
            }
          }
        } else {
          // Full ghost rebuild pack on compute_stream
          int* h_slot_desc = static_cast<int*>(G.h_slot_meta.p);
          int* h_slot_pstart = h_slot_desc + N_TOP_CELLS;
          int* h_slot_pcnt = h_slot_pstart + N_TOP_CELLS;
          if (G.n_own > 0) {
            check(cudaMemsetAsync(G.d_own_slot_desc_cnt.p, 0, size_t(G.n_own) * sizeof(int), G.compute_stream));
            if (G.nodes > 0) {
              const int c_grid = std::min(512, blocks(int(G.nodes)));
              count_own_slot_nodes_3d_kernel<<<c_grid, team, 0, G.compute_stream>>>(
                  G.geometry.p, G.n_own, int(G.nodes),
                  G.d_own_slot_desc_cnt.p, G.d_own_slot_pstart.p, G.d_own_slot_pcnt.p);
            } else {
              check(cudaMemsetAsync(G.d_own_slot_pstart.p, 0, size_t(G.n_own) * sizeof(int), G.compute_stream));
              check(cudaMemsetAsync(G.d_own_slot_pcnt.p, 0, size_t(G.n_own) * sizeof(int), G.compute_stream));
            }
            check(cudaMemcpyAsync(h_slot_desc, G.d_own_slot_desc_cnt.p,
                                  size_t(G.n_own) * sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
            check(cudaMemcpyAsync(h_slot_pstart, G.d_own_slot_pstart.p,
                                  size_t(G.n_own) * sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
            check(cudaMemcpyAsync(h_slot_pcnt, G.d_own_slot_pcnt.p,
                                  size_t(G.n_own) * sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
            check(cudaStreamSynchronize(G.compute_stream));
          }

          std::vector<int> send_meta(size_t(num_ranks) * 8, 0), recv_meta(size_t(num_ranks) * 8, 0);
          for (int dst = 0; dst < num_ranks; ++dst) {
            if (dst == my_rank) continue;
            auto& ed = edges[dst];
            ed.out_tot_nodes = ed.out_tot_parts = 0;
            ed.out_wire_bytes = 0;
            if (ed.E_out == 0) continue;
            const auto& src_slots = plan.export_src_slots[my_rank][dst];
            int n_acc = 0, nr_acc = 0, p_acc = 0;
            for (int e = 0; e < ed.E_out; ++e) {
              const int s = src_slots[e];
              const int ncnt = 1 + h_slot_desc[s];
              const int pcnt = h_slot_pcnt[s];
              ed.h_node_cnt[e] = ncnt;
              ed.h_part_cnt[e] = pcnt;
              ed.h_part_src_start[e] = h_slot_pstart[s];
              ed.h_node_off[e] = n_acc;
              ed.h_nonroot_off[e] = nr_acc;
              ed.h_part_off[e] = p_acc;
              n_acc += ncnt;
              nr_acc += (ncnt - 1);
              p_acc += pcnt;
            }
            ed.out_tot_nodes = n_acc;
            ed.out_tot_parts = p_acc;
            ed.ensure_meta_src(G.dev, ed.E_out, G.device_bytes);
            ed.ensure_bnd_nodes(G.dev, std::max(1, ed.out_tot_nodes), G.device_bytes);
            int* hm = static_cast<int*>(ed.h_meta_pinned.p);
            std::memcpy(hm, ed.h_nonroot_off.data(), size_t(ed.E_out) * sizeof(int));
            std::memcpy(hm + ed.E_out, ed.h_part_src_start.data(), size_t(ed.E_out) * sizeof(int));
            std::memcpy(hm + 2 * ed.E_out, ed.h_part_off.data(), size_t(ed.E_out) * sizeof(int));
            std::memcpy(hm + 3 * ed.E_out, ed.h_part_cnt.data(), size_t(ed.E_out) * sizeof(int));
            check(cudaMemcpyAsync(ed.d_exp_cursor.p, hm, size_t(ed.E_out) * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
            check(cudaMemcpyAsync(ed.d_exp_part_src_start.p, hm + ed.E_out, size_t(ed.E_out) * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
            check(cudaMemcpyAsync(ed.d_exp_part_off.p, hm + 2 * ed.E_out, size_t(ed.E_out) * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));
            check(cudaMemcpyAsync(ed.d_exp_part_cnt.p, hm + 3 * ed.E_out, size_t(ed.E_out) * sizeof(int), cudaMemcpyHostToDevice, G.compute_stream));

            auto align256 = [](size_t v) { return (v + 255ULL) & ~255ULL; };
            size_t w_off = 0;
            ed.out_off_g = w_off; w_off += align256(size_t(ed.out_tot_nodes) * sizeof(Geometry));
            ed.out_off_m = w_off; w_off += align256(size_t(ed.out_tot_nodes) * sizeof(NodeMeta));
            ed.out_off_c = w_off; w_off += align256(size_t(ed.out_tot_nodes) * sizeof(CompactMultipole));
            ed.out_off_p = w_off; w_off += align256(size_t(std::max(1, ed.out_tot_parts)) * sizeof(float4));
            ed.out_off_s = w_off;
            if (!soft_uniform) w_off += align256(size_t(std::max(1, ed.out_tot_parts)) * sizeof(float));
            ed.out_wire_bytes = w_off;
            if (w_off > ed.d_send_wire.bytes) {
              ed.d_send_wire.alloc(std::max(w_off, w_off * 3 / 2 + 4096), G.device_bytes);
            }

            Geometry* w_geom = reinterpret_cast<Geometry*>(ed.d_send_wire.p + ed.out_off_g);
            NodeMeta* w_meta = reinterpret_cast<NodeMeta*>(ed.d_send_wire.p + ed.out_off_m);
            CompactMultipole* w_cm = reinterpret_cast<CompactMultipole*>(ed.d_send_wire.p + ed.out_off_c);
            float4* w_pack = reinterpret_cast<float4*>(ed.d_send_wire.p + ed.out_off_p);
            float* w_soft = soft_uniform ? nullptr : reinterpret_cast<float*>(ed.d_send_wire.p + ed.out_off_s);

            if (G.nodes > 0) {
              assign_export_bidx_3d_kernel<<<blocks(int(G.nodes)), team, 0, G.compute_stream>>>(
                  G.geometry.p, ed.d_slot_to_e.p, G.n_own, int(G.nodes),
                  ed.E_out, ed.d_exp_cursor.p, G.d_u_to_b.p, ed.d_bnd_src_node_map.p);
            }
            if (ed.out_tot_nodes > 0) {
              gather_export_bundle_3d_kernel<<<blocks(ed.out_tot_nodes), team, 0, G.compute_stream>>>(
                  G.geometry.p, G.node_meta.p, G.compact_m.p, ed.d_slot_to_e.p,
                  ed.d_exp_part_src_start.p, ed.d_exp_part_off.p,
                  G.d_u_to_b.p, ed.d_bnd_src_node_map.p, ed.E_out, ed.out_tot_nodes,
                  w_geom, w_meta, w_cm);
            }
            if (ed.out_tot_parts > 0) {
              pack_export_particles_kernel<<<blocks_warps(ed.E_out * 4), team, 0, G.compute_stream>>>(
                  G.packed_sources.p, soft_uniform ? nullptr : G.packed_soft.p,
                  ed.d_exp_part_src_start.p, ed.d_exp_part_off.p, ed.d_exp_part_cnt.p,
                  ed.E_out, w_pack, w_soft);
            }

            send_meta[dst * 8 + 0] = ed.out_tot_nodes;
            send_meta[dst * 8 + 1] = ed.out_tot_parts;
            send_meta[dst * 8 + 2] = int(ed.out_wire_bytes);
            send_meta[dst * 8 + 3] = int(ed.out_off_g);
            send_meta[dst * 8 + 4] = int(ed.out_off_m);
            send_meta[dst * 8 + 5] = int(ed.out_off_c);
            send_meta[dst * 8 + 6] = int(ed.out_off_p);
            send_meta[dst * 8 + 7] = int(ed.out_off_s);
          }
          check(cudaStreamSynchronize(G.compute_stream));
          GRAV_MPI_CHECK(MPI_Alltoall(send_meta.data(), 8, MPI_INT, recv_meta.data(), 8, MPI_INT, mpi.world));

          int desc_cursor = 0, part_cursor = 0;
          for (int src = 0; src < num_ranks; ++src) {
            if (src == my_rank) continue;
            auto& ed = edges[src];
            ed.in_tot_nodes = recv_meta[src * 8 + 0];
            ed.in_tot_parts = recv_meta[src * 8 + 1];
            ed.in_wire_bytes = size_t(recv_meta[src * 8 + 2]);
            ed.in_off_g = size_t(recv_meta[src * 8 + 3]);
            ed.in_off_m = size_t(recv_meta[src * 8 + 4]);
            ed.in_off_c = size_t(recv_meta[src * 8 + 5]);
            ed.in_off_p = size_t(recv_meta[src * 8 + 6]);
            ed.in_off_s = size_t(recv_meta[src * 8 + 7]);
            ed.dst_desc_base = desc_cursor;
            ed.dst_part_base = part_cursor;
            if (ed.E_in > 0) {
              desc_cursor += (ed.in_tot_nodes - ed.E_in);
              part_cursor += ed.in_tot_parts;
              if (ed.in_wire_bytes > ed.d_recv_wire.bytes) {
                ed.d_recv_wire.alloc(std::max(ed.in_wire_bytes, ed.in_wire_bytes * 3 / 2 + 4096), G.device_bytes);
              }
            }
          }
          const size_t tot_gh_nodes = size_t(G.n_ghost) + size_t(desc_cursor);
          const size_t tot_gh_parts = size_t(part_cursor);
          G.ensure_ghost_cap(tot_gh_parts, tot_gh_nodes, soft_uniform);
          G.gh_nodes = unsigned(tot_gh_nodes);
        }
      }

      // ================= STEP D: Overlapped Ghost Exchange + 2-Pass Force + Integrate =================
      build_kick_table<<<1, 64, 0, G.compute_stream>>>(G.d_kick_finish.p, G.d_kick_begin.p, p, ti);

      const int ns = int(G.ns);
      const int* active_ptr = nullptr;
      int force_count = ns;
      if (ns > 0 && !is_pm_step && !is_final) {
        mark_active_slab<<<blocks(ns), team, 0, G.compute_stream>>>(
            G.order.p, G.d_bins.p, G.d_flags.p, ns, ti);
        cub::CountingInputIterator<int> seq(0);
        size_t selb = G.select_bytes;
        check(cub::DeviceSelect::Flagged(
            G.d_select_storage.p, selb, seq, G.d_flags.p,
            G.d_active_ids.p, G.d_active_count.p, ns, G.compute_stream));
        int h_act = 0;
        check(cudaMemcpyAsync(&h_act, G.d_active_count.p, sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaStreamSynchronize(G.compute_stream));
        force_count = h_act;
        active_ptr = G.d_active_ids.p;
      }

      int n_int_warps = 0, total_warps = 0;
      uint64_t loc_batches = 0, loc_retries = 0;
      const SplitTravTreeView loc_tv = G.local_split_view();
      const SplitTravTreeView gh_tv = G.ghost_split_view();

      if (force_count > 0) {
        check(cudaMemsetAsync(G.d_tree_invalid.p, 0, sizeof(int), G.compute_stream));
        build_home_warp_slices_3d_kernel<<<1, 1024, 0, G.compute_stream>>>(
            G.home_slot.p, G.n_own, G.n_interior_own,
            active_ptr, active_ptr ? G.d_active_count.p : nullptr,
            force_count, G.d_tree_invalid.p, G.d_split_slices.p, G.d_split_total_warps.p);
        int h_warp_cnt[2]{};
        check(cudaMemcpyAsync(h_warp_cnt, G.d_split_total_warps.p, 2 * sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaStreamSynchronize(G.compute_stream));
        n_int_warps = h_warp_cnt[0];
        total_warps = h_warp_cnt[1];
      }

      auto run_warp_range = [&](int w_begin, int w_end) {
        check(cudaSetDevice(G.dev));
        int warp_off = w_begin;
        int cur_batch = cfg.split_batch_warps;
        while (warp_off < w_end) {
          const int n_warps = std::min(cur_batch, w_end - warp_off);
          check(cudaMemsetAsync(G.d_split_counts.p, 0, 5 * sizeof(int), G.compute_stream));
          if (cfg.warp_m2l) {
            traverse_m2p_split_3d_kernel<true><<<blocks(n_warps * 32), team, 0, G.compute_stream>>>(
                G.d_x.p, G.d_old_acc.p, soft, soft_uniform ? nullptr : G.d_soft.p,
                loc_tv, gh_tv, G.d_slot_nb_roots.p, G.order.p,
                G.d_split_slices.p, warp_off, n_warps,
                G.d_split_meta.p, G.d_split_head_p2p.p, G.d_split_head_m2l.p,
                G.d_split_p2p.p, int(G.split_p2p_cap),
                G.d_split_m2l.p, int(G.split_m2l_cap),
                G.d_split_p2p_warps.p, G.d_split_m2l_warps.p, G.d_split_counts.p,
                G.d_force.p, ns, p, ti == 0, G.d_tree_invalid.p, active_ptr, false);
          } else {
            traverse_m2p_split_3d_kernel<false><<<blocks(n_warps * 32), team, 0, G.compute_stream>>>(
                G.d_x.p, G.d_old_acc.p, soft, soft_uniform ? nullptr : G.d_soft.p,
                loc_tv, gh_tv, G.d_slot_nb_roots.p, G.order.p,
                G.d_split_slices.p, warp_off, n_warps,
                G.d_split_meta.p, G.d_split_head_p2p.p, G.d_split_head_m2l.p,
                G.d_split_p2p.p, int(G.split_p2p_cap),
                G.d_split_m2l.p, int(G.split_m2l_cap),
                G.d_split_p2p_warps.p, G.d_split_m2l_warps.p, G.d_split_counts.p,
                G.d_force.p, ns, p, ti == 0, G.d_tree_invalid.p, active_ptr, false);
          }
          int h_counts[5]{};
          check(cudaMemcpyAsync(h_counts, G.d_split_counts.p, 5 * sizeof(int), cudaMemcpyDeviceToHost, G.compute_stream));
          check(cudaStreamSynchronize(G.compute_stream));
          if (h_counts[4] != 0) {
            ++loc_retries;
            if (G.split_chunk_scale < 4 && cur_batch == cfg.split_batch_warps) {
              G.ensure_split_bufs(size_t(cfg.split_batch_warps), G.split_chunk_scale * 2);
              continue;
            }
            if (cur_batch > 256) { cur_batch = std::max(256, cur_batch / 2); continue; }
            throw std::runtime_error("MPI 3D split traversal chunk overflow at minimum batch");
          }
          if (h_counts[3] > 0) {
            eval_m2l_split_3d_kernel<<<blocks(h_counts[3] * 32), team, 0, G.compute_stream>>>(
                G.d_x.p, loc_tv, gh_tv, G.order.p,
                G.d_split_meta.p, G.d_split_head_m2l.p, G.d_split_m2l.p,
                G.d_split_m2l_warps.p, h_counts[3],
                G.d_force.p, ns, p, G.d_tree_invalid.p, active_ptr, false);
          }
          if (h_counts[2] > 0) {
            const auto launch_p2p = [&](auto fsp_t, auto us_t) {
              eval_p2p_split_3d_kernel<decltype(fsp_t)::value, decltype(us_t)::value>
                  <<<blocks(h_counts[2] * 32), team, 0, G.compute_stream>>>(
                      G.d_x.p, soft, soft_uniform ? nullptr : G.d_soft.p,
                      G.packed_sources.p, soft_uniform ? nullptr : G.packed_soft.p, G.order.p,
                      G.gh_packed_sources.p, soft_uniform ? nullptr : G.gh_packed_soft.p,
                      G.d_split_meta.p, G.d_split_head_p2p.p, G.d_split_p2p.p,
                      G.d_split_p2p_warps.p, h_counts[2],
                      G.d_force.p, ns, p, G.d_tree_invalid.p, active_ptr, false);
            };
            if (cfg.fast_split) {
              if (soft_uniform) launch_p2p(std::true_type{}, std::true_type{});
              else launch_p2p(std::true_type{}, std::false_type{});
            } else {
              if (soft_uniform) launch_p2p(std::false_type{}, std::true_type{});
              else launch_p2p(std::false_type{}, std::false_type{});
            }
          }
          ++loc_batches;
          warp_off += n_warps;
          if (cur_batch < cfg.split_batch_warps) cur_batch = cfg.split_batch_warps;
        }
      };

      // Launch Pass 1 interior warps on a CUDA-only helper thread so MPI ghost exchange
      // on the main thread overlaps 100% with GPU interior force evaluation!
      const bool ran_int = (force_count > 0 && n_int_warps > 0);
      std::thread int_thread;
      std::exception_ptr int_err = nullptr;
      double int_gpu_ms = 0.0;
      if (ran_int) {
        int_thread = std::thread([&]() {
          try {
            check(cudaSetDevice(G.dev));
            check(cudaEventRecord(G.ev_int_start, G.compute_stream));
            run_warp_range(0, n_int_warps);
            check(cudaEventRecord(G.ev_int_end, G.compute_stream));
            check(cudaStreamSynchronize(G.compute_stream));
            float ms_i = 0.f;
            check(cudaEventElapsedTime(&ms_i, G.ev_int_start, G.ev_int_end));
            int_gpu_ms = double(ms_i);
          } catch (...) {
            int_err = std::current_exception();
          }
        });
      }

      // Main thread performs MPI ghost exchange + GPU copy_stream unpack
      if (need_ghost) {
        const double tg0 = wall();
        if (do_lazy_step) {
          std::vector<MPI_Request> dreqs;
          for (int src = 0; src < num_ranks; ++src) {
            if (src == my_rank || edges[src].lazy_M_in == 0) continue;
            MPI_Request rq = MPI_REQUEST_NULL;
            GRAV_MPI_CHECK(MPI_Irecv(
                edges[src].h_lazy_descs_recv.data(),
                edges[src].lazy_M_in * int(sizeof(Dom3DLazyRootDesc)), MPI_BYTE,
                src, 61, mpi.world, &rq));
            dreqs.push_back(rq);
          }
          for (int dst = 0; dst < num_ranks; ++dst) {
            if (dst == my_rank || edges[dst].lazy_M_out == 0) continue;
            MPI_Request rq = MPI_REQUEST_NULL;
            GRAV_MPI_CHECK(MPI_Isend(
                edges[dst].h_lazy_descs_send.data(),
                edges[dst].lazy_M_out * int(sizeof(Dom3DLazyRootDesc)), MPI_BYTE,
                dst, 61, mpi.world, &rq));
            dreqs.push_back(rq);
          }
          if (!dreqs.empty()) {
            GRAV_MPI_CHECK(MPI_Waitall(int(dreqs.size()), dreqs.data(), MPI_STATUSES_IGNORE));
          }

          std::vector<int> s_peers, s_bytes, r_peers, r_bytes;
          std::vector<const void*> s_ptrs;
          std::vector<void*> r_ptrs;
          for (int dst = 0; dst < num_ranks; ++dst) {
            if (dst == my_rank || edges[dst].lazy_M_out == 0) continue;
            const size_t bytes = (size_t(edges[dst].lazy_tot_nodes_out) + size_t(edges[dst].lazy_tot_parts_out)) * sizeof(float4);
            if (bytes > 0) {
              s_peers.push_back(dst);
              s_ptrs.push_back(edges[dst].d_lazy_send.p);
              s_bytes.push_back(int(bytes));
            }
          }
          for (int src = 0; src < num_ranks; ++src) {
            if (src == my_rank || edges[src].lazy_M_in == 0) continue;
            const size_t bytes = (size_t(edges[src].lazy_tot_nodes_in) + size_t(edges[src].lazy_tot_parts_in)) * sizeof(float4);
            if (bytes > 0) {
              r_peers.push_back(src);
              r_ptrs.push_back(edges[src].d_lazy_recv.p);
              r_bytes.push_back(int(bytes));
            }
          }
          const double tm0 = MPI_Wtime();
          mpi.exchange_sparse_dev(
              s_peers, s_ptrs, s_bytes, h_ghost_send_scratch,
              r_peers, r_ptrs, r_bytes, h_ghost_recv_scratch,
              62, G.copy_stream);
          ghost_mpi_acc_ms += (MPI_Wtime() - tm0) * 1e3;

          for (int src = 0; src < num_ranks; ++src) {
            if (src == my_rank || edges[src].lazy_M_in == 0) continue;
            auto& ed = edges[src];
            for (auto& d : ed.h_lazy_descs_recv) {
              d.dst_desc_base += ed.dst_desc_base;
              d.dst_part_off += ed.dst_part_base;
            }
            const size_t ld_bytes = size_t(ed.lazy_M_in) * sizeof(Dom3DLazyRootDesc);
            check(cudaMemcpyAsync(ed.d_lazy_descs_dst.p, ed.h_lazy_descs_recv.data(),
                                  ld_bytes, cudaMemcpyHostToDevice, G.copy_stream));
            unpack_lazy_edge_kernel<<<blocks_warps(ed.lazy_M_in * 4), team, 0, G.copy_stream>>>(
                ed.d_lazy_recv.p, ed.d_lazy_descs_dst.p, ed.lazy_M_in, G.n_ghost,
                ed.lazy_tot_nodes_in, G.gh_node_meta.p, G.gh_packed_sources.p);
          }
          check(cudaStreamSynchronize(G.copy_stream));
        } else {
          std::vector<int> s_peers, s_bytes, r_peers, r_bytes;
          std::vector<const void*> s_ptrs;
          std::vector<void*> r_ptrs;
          for (int dst = 0; dst < num_ranks; ++dst) {
            if (dst == my_rank || edges[dst].E_out == 0 || edges[dst].out_wire_bytes == 0) continue;
            s_peers.push_back(dst);
            s_ptrs.push_back(edges[dst].d_send_wire.p);
            s_bytes.push_back(int(edges[dst].out_wire_bytes));
          }
          for (int src = 0; src < num_ranks; ++src) {
            if (src == my_rank || edges[src].E_in == 0 || edges[src].in_wire_bytes == 0) continue;
            r_peers.push_back(src);
            r_ptrs.push_back(edges[src].d_recv_wire.p);
            r_bytes.push_back(int(edges[src].in_wire_bytes));
          }
          const double tm0 = MPI_Wtime();
          mpi.exchange_sparse_dev(
              s_peers, s_ptrs, s_bytes, h_ghost_send_scratch,
              r_peers, r_ptrs, r_bytes, h_ghost_recv_scratch,
              63, G.copy_stream);
          ghost_mpi_acc_ms += (MPI_Wtime() - tm0) * 1e3;

          for (int src = 0; src < num_ranks; ++src) {
            if (src == my_rank || edges[src].E_in == 0) continue;
            auto& ed = edges[src];
            const Geometry* r_geom = reinterpret_cast<const Geometry*>(ed.d_recv_wire.p + ed.in_off_g);
            const NodeMeta* r_meta = reinterpret_cast<const NodeMeta*>(ed.d_recv_wire.p + ed.in_off_m);
            const CompactMultipole* r_cm = reinterpret_cast<const CompactMultipole*>(ed.d_recv_wire.p + ed.in_off_c);
            const float4* r_pack = reinterpret_cast<const float4*>(ed.d_recv_wire.p + ed.in_off_p);
            const float* r_soft = soft_uniform ? nullptr : reinterpret_cast<const float*>(ed.d_recv_wire.p + ed.in_off_s);

            install_ghost_bundle_kernel<<<blocks(ed.in_tot_nodes), team, 0, G.copy_stream>>>(
                r_geom, r_meta, r_cm, ed.d_dst_ghost_slots.p, ed.E_in, ed.in_tot_nodes,
                G.n_ghost, ed.dst_desc_base, ed.dst_part_base,
                G.gh_geometry.p, G.gh_node_meta.p, G.gh_compact_m.p);
            if (ed.in_tot_parts > 0) {
              check(cudaMemcpyAsync(G.gh_packed_sources.p + ed.dst_part_base, r_pack,
                                    size_t(ed.in_tot_parts) * sizeof(float4), cudaMemcpyDeviceToDevice, G.copy_stream));
              if (!soft_uniform && r_soft) {
                check(cudaMemcpyAsync(G.gh_packed_soft.p + ed.dst_part_base, r_soft,
                                      size_t(ed.in_tot_parts) * sizeof(float), cudaMemcpyDeviceToDevice, G.copy_stream));
              }
            }
          }
          check(cudaStreamSynchronize(G.copy_stream));
        }
        ghost_ms = (wall() - tg0) * 1000.0;
        total_ghost_exchange_ms += ghost_ms;
      }

      if (int_thread.joinable()) int_thread.join();
      if (int_err) std::rethrow_exception(int_err);

      // Pass 2: Boundary warps (only when force_count > 0)
      double bnd_gpu_ms = 0.0;
      unsigned int loc_invalid = 0;
      if (force_count == 0) {
        force_ms = 0.0;
      } else {
        const bool ran_bnd = (n_int_warps < total_warps);
        if (ran_bnd) {
          check(cudaEventRecord(G.ev_bnd_start, G.compute_stream));
          run_warp_range(n_int_warps, total_warps);
          check(cudaEventRecord(G.ev_bnd_end, G.compute_stream));
        }
        split_batches_run += loc_batches;
        split_overflow_retries += loc_retries;
        check(cudaGetLastError());
        check(cudaStreamSynchronize(G.compute_stream));
        force_ms = int_gpu_ms;
        if (ran_bnd) {
          float ms_b = 0.f;
          check(cudaEventElapsedTime(&ms_b, G.ev_bnd_start, G.ev_bnd_end));
          bnd_gpu_ms = double(ms_b);
          force_ms += bnd_gpu_ms;
        }
      }

      if (cfg.force0_out_path && global_iteration == 0) {
        std::vector<float4> hf(ns);
        std::vector<uint32_t> hidx(ns);
        if (ns > 0) {
          check(cudaMemcpy(hf.data(), G.d_force.p, size_t(ns) * sizeof(float4), cudaMemcpyDeviceToHost));
          check(cudaMemcpy(hidx.data(), G.d_orig_idx.p, size_t(ns) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        }
        if (num_ranks == 1) {
          out_force0_f4.resize(N);
          for (int k = 0; k < ns; ++k) out_force0_f4[hidx[k]] = hf[k];
        } else {
          std::vector<int> rc(num_ranks, 0);
          int my_ni = ns;
          MPI_Gather(&my_ni, 1, MPI_INT, rc.data(), 1, MPI_INT, 0, mpi.world);
          std::vector<int> bc4(num_ranks, 0), bd4(num_ranks, 0), bci(num_ranks, 0), bdi(num_ranks, 0);
          std::vector<float4> all_f;
          std::vector<uint32_t> all_id;
          if (my_rank == 0) {
            int acc = 0;
            for (int r = 0; r < num_ranks; ++r) {
              bd4[r] = acc * int(sizeof(float4));
              bc4[r] = rc[r] * int(sizeof(float4));
              bdi[r] = acc * int(sizeof(uint32_t));
              bci[r] = rc[r] * int(sizeof(uint32_t));
              acc += rc[r];
            }
            all_f.resize(acc);
            all_id.resize(acc);
            out_force0_f4.resize(N);
          }
          MPI_Gatherv(hf.data(), my_ni * int(sizeof(float4)), MPI_BYTE,
                      my_rank == 0 ? all_f.data() : nullptr, bc4.data(), bd4.data(), MPI_BYTE, 0, mpi.world);
          MPI_Gatherv(hidx.data(), my_ni * int(sizeof(uint32_t)), MPI_BYTE,
                      my_rank == 0 ? all_id.data() : nullptr, bci.data(), bdi.data(), MPI_BYTE, 0, mpi.world);
          if (my_rank == 0) {
            for (uint64_t i = 0; i < N; ++i) out_force0_f4[all_id[i]] = all_f[i];
          }
        }
      }

      if (cfg.energy_log && (is_pm_step || is_final)) {
        std::array<double, 10> loc_e{}, glb_e{};
        check(cudaMemsetAsync(d_energy_accum, 0, 10 * sizeof(double), G.compute_stream));
        if (ns > 0) {
          if (use_mpi_pm) {
            mpi_pm.interpolate(G.d_x.p, size_t(ns), h.box, p.G, G.d_mesh_acc.p, G.d_pm_pot.p);
          } else {
            grav_pm_cuda_interpolate_buf(
                G.d_x.p, size_t(ns), G.d_pm_field.p, pm_side, h.box, p.G,
                G.d_mesh_acc.p, G.d_pm_pot.p, G.compute_stream);
          }
          const double closing_pm_e =
              (ti == 0) ? 0.0
              : (is_pm_step ? (is_final ? pm_kick : p.factors(ti - prev_pm_span / 2, ti, false))
                            : p.factors(ti - current_pm_span / 2, ti, false));
          const int eblocks = std::min(256, (ns + 255) / 256);
          reduce_energy_mom_kernel<<<eblocks, 256, 0, G.compute_stream>>>(
              G.d_x.p, G.d_v.p, G.d_force.p, G.d_mesh_acc.p,
              G.d_pm_pot.p, G.d_bins.p,
              mass_uniform ? nullptr : G.d_mass.p, first_mass,
              ns, ti, G.d_kick_finish.p, closing_pm_e,
              double(p.G), potential_norm, cfg.comoving ? (1.0 / a) : 1.0,
              d_energy_accum);
          check(cudaGetLastError());
        }
        check(cudaMemcpyAsync(loc_e.data(), d_energy_accum, 10 * sizeof(double),
                              cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaStreamSynchronize(G.compute_stream));
        if (num_ranks > 1) {
          GRAV_MPI_CHECK(MPI_Reduce(loc_e.data(), glb_e.data(), 10, MPI_DOUBLE, MPI_SUM, 0, mpi.world));
        } else {
          glb_e = loc_e;
        }
        if (my_rank == 0) {
          const double inv_a_e = cfg.comoving ? (1.0 / a) : 1.0;
          const double U_phys = glb_e[1] * inv_a_e;
          std::cout << std::setprecision(12)
                    << "ENERGY {\"step\":" << global_iteration << ",\"a\":" << a
                    << ",\"T\":" << glb_e[0] << ",\"U\":" << U_phys
                    << ",\"W_com\":" << glb_e[1] << ",\"E\":" << (glb_e[0] + U_phys)
                    << ",\"Px\":" << glb_e[2] << ",\"Py\":" << glb_e[3] << ",\"Pz\":" << glb_e[4]
                    << ",\"Lx\":" << glb_e[5] << ",\"Ly\":" << glb_e[6] << ",\"Lz\":" << glb_e[7]
                    << ",\"M\":" << glb_e[8] << ",\"v_rms\":" << sqrt(glb_e[9] / double(N)) << "}" << std::endl;
        }
      }

      if (do_fc_now && fc_target_cap > 0) {
        std::vector<double3> my_hx(ns);
        std::vector<float> my_hm, my_hs;
        if (ns > 0) {
          check(cudaMemcpy(my_hx.data(), G.d_x.p, size_t(ns) * sizeof(double3), cudaMemcpyDeviceToHost));
          if (!mass_uniform) {
            my_hm.resize(ns);
            check(cudaMemcpy(my_hm.data(), G.d_mass.p, size_t(ns) * sizeof(float), cudaMemcpyDeviceToHost));
          }
          if (!soft_uniform) {
            my_hs.resize(ns);
            check(cudaMemcpy(my_hs.data(), G.d_soft.p, size_t(ns) * sizeof(float), cudaMemcpyDeviceToHost));
          }
        }
        std::vector<int> rcounts(num_ranks, 0), rdispls(num_ranks, 0);
        int my_n_i = ns;
        MPI_Gather(&my_n_i, 1, MPI_INT, rcounts.data(), 1, MPI_INT, 0, mpi.world);
        std::vector<double3> all_x;
        std::vector<float> all_m, all_s;
        std::vector<int> bcnt_x(num_ranks), bdsp_x(num_ranks), bcnt_f(num_ranks), bdsp_f(num_ranks);
        if (my_rank == 0) {
          int acc = 0;
          for (int r = 0; r < num_ranks; ++r) {
            rdispls[r] = acc;
            bcnt_x[r] = rcounts[r] * int(sizeof(double3));
            bdsp_x[r] = rdispls[r] * int(sizeof(double3));
            bcnt_f[r] = rcounts[r] * int(sizeof(float));
            bdsp_f[r] = rdispls[r] * int(sizeof(float));
            acc += rcounts[r];
          }
          all_x.resize(acc);
          if (!mass_uniform) all_m.resize(acc);
          if (!soft_uniform) all_s.resize(acc);
        }
        MPI_Gatherv(my_hx.data(), my_n_i * int(sizeof(double3)), MPI_BYTE,
                    my_rank == 0 ? all_x.data() : nullptr, bcnt_x.data(), bdsp_x.data(), MPI_BYTE, 0, mpi.world);
        if (!mass_uniform) {
          MPI_Gatherv(my_hm.data(), my_n_i * int(sizeof(float)), MPI_BYTE,
                      my_rank == 0 ? all_m.data() : nullptr, bcnt_f.data(), bdsp_f.data(), MPI_BYTE, 0, mpi.world);
        }
        if (!soft_uniform) {
          MPI_Gatherv(my_hs.data(), my_n_i * int(sizeof(float)), MPI_BYTE,
                      my_rank == 0 ? all_s.data() : nullptr, bcnt_f.data(), bdsp_f.data(), MPI_BYTE, 0, mpi.world);
        }

        if (my_rank == 0 && ns > 0) {
          const int n_fc_tgt = std::min(fc_target_cap, ns);
          std::vector<int> h_fc_lids(n_fc_tgt);
          const int stride = std::max(1, ns / n_fc_tgt);
          for (int k = 0; k < n_fc_tgt; ++k) h_fc_lids[k] = k * stride;
          check(cudaMemcpyAsync(d_fc_lids.p, h_fc_lids.data(), size_t(n_fc_tgt) * sizeof(int),
                                cudaMemcpyHostToDevice, G.compute_stream));
          gather_fc_samples_kernel<<<blocks(n_fc_tgt), team, 0, G.compute_stream>>>(
              G.d_x.p, soft_uniform ? nullptr : G.d_soft.p, soft,
              G.d_force.p, G.d_mesh_acc.p,
              d_fc_lids.p, n_fc_tgt, 0,
              d_fc_tgt_x.p, d_fc_tgt_h.p, d_fc_gpu_short.p,
              is_pm_step ? d_fc_gpu_mesh.p : nullptr);
          DevPtr<double3> d_all_x;
          DevPtr<float> d_all_m, d_all_s;
          size_t dummy_b = 0;
          d_all_x.alloc(all_x.size(), dummy_b);
          check(cudaMemcpyAsync(d_all_x.p, all_x.data(), all_x.size() * sizeof(double3),
                                cudaMemcpyHostToDevice, G.compute_stream));
          if (!mass_uniform) {
            d_all_m.alloc(all_m.size(), dummy_b);
            check(cudaMemcpyAsync(d_all_m.p, all_m.data(), all_m.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, G.compute_stream));
          }
          if (!soft_uniform) {
            d_all_s.alloc(all_s.size(), dummy_b);
            check(cudaMemcpyAsync(d_all_s.p, all_s.data(), all_s.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, G.compute_stream));
          }
          AllSlabsData all_slabs{};
          all_slabs.x[0] = d_all_x.p;
          all_slabs.mass[0] = mass_uniform ? nullptr : d_all_m.p;
          all_slabs.soft[0] = soft_uniform ? nullptr : d_all_s.p;
          all_slabs.ns[0] = int(all_x.size());
          const int ne = int(h_ewald_modes.size());
          ewald_kspace_accum_kernel<<<ne, 256, 0, G.compute_stream>>>(
              all_slabs, first_mass, d_ewald_modes.p, ne, d_ewald_C.p, d_ewald_S.p);
          ewald_target_force_kernel<<<n_fc_tgt, 256, 0, G.compute_stream>>>(
              all_slabs, first_mass, soft, d_fc_tgt_x.p, d_fc_tgt_h.p,
              d_fc_tgt_slab.p, d_fc_lids.p, n_fc_tgt, h.box, r_s, double(p.G),
              d_ewald_modes.p, d_ewald_C.p, d_ewald_S.p, ne,
              d_fc_short_ref.p, d_fc_total_ref.p);
          check(cudaStreamSynchronize(G.compute_stream));
          d_all_x.free(dummy_b);
          d_all_m.free(dummy_b);
          d_all_s.free(dummy_b);

          std::vector<float4> h_gs(n_fc_tgt);
          std::vector<float3> h_gm(n_fc_tgt);
          std::vector<double3> h_rs(n_fc_tgt), h_rt(n_fc_tgt);
          check(cudaMemcpy(h_gs.data(), d_fc_gpu_short.p, size_t(n_fc_tgt) * sizeof(float4), cudaMemcpyDeviceToHost));
          check(cudaMemcpy(h_gm.data(), d_fc_gpu_mesh.p, size_t(n_fc_tgt) * sizeof(float3), cudaMemcpyDeviceToHost));
          check(cudaMemcpy(h_rs.data(), d_fc_short_ref.p, size_t(n_fc_tgt) * sizeof(double3), cudaMemcpyDeviceToHost));
          check(cudaMemcpy(h_rt.data(), d_fc_total_ref.p, size_t(n_fc_tgt) * sizeof(double3), cudaMemcpyDeviceToHost));
          std::vector<double> rel_s(n_fc_tgt), rel_t(n_fc_tgt);
          double sq_g_s = 0.0, sq_r_s = 0.0, sq_g_t = 0.0, sq_r_t = 0.0;
          for (int k = 0; k < n_fc_tgt; ++k) {
            const double gx = double(h_gs[k].x) * double(p.G);
            const double gy = double(h_gs[k].y) * double(p.G);
            const double gz = double(h_gs[k].z) * double(p.G);
            const double rxv = h_rs[k].x, ryv = h_rs[k].y, rzv = h_rs[k].z;
            const double ds = sqrt((gx - rxv)*(gx - rxv) + (gy - ryv)*(gy - ryv) + (gz - rzv)*(gz - rzv));
            const double rs_norm = sqrt(rxv*rxv + ryv*ryv + rzv*rzv);
            rel_s[k] = (rs_norm > 0.0) ? (ds / rs_norm) : ds;
            sq_g_s += ds * ds; sq_r_s += rs_norm * rs_norm;
            const double txv = gx + double(h_gm[k].x);
            const double tyv = gy + double(h_gm[k].y);
            const double tzv = gz + double(h_gm[k].z);
            const double dt = sqrt((txv - h_rt[k].x)*(txv - h_rt[k].x) + (tyv - h_rt[k].y)*(tyv - h_rt[k].y) + (tzv - h_rt[k].z)*(tzv - h_rt[k].z));
            const double rt_norm = sqrt(h_rt[k].x*h_rt[k].x + h_rt[k].y*h_rt[k].y + h_rt[k].z*h_rt[k].z);
            rel_t[k] = (rt_norm > 0.0) ? (dt / rt_norm) : dt;
            sq_g_t += dt * dt; sq_r_t += rt_norm * rt_norm;
          }
          std::sort(rel_s.begin(), rel_s.end());
          std::sort(rel_t.begin(), rel_t.end());
          std::cout << std::setprecision(12)
                    << "FORCE_CHECK {\"step\":" << global_iteration << ",\"a\":" << a
                    << ",\"n\":" << n_fc_tgt << ",\"method\":\"ewald_periodic\""
                    << ",\"short_rms\":" << sqrt(sq_g_s / std::max(1e-300, sq_r_s))
                    << ",\"short_median\":" << rel_s[n_fc_tgt / 2]
                    << ",\"total_rms\":" << sqrt(sq_g_t / std::max(1e-300, sq_r_t))
                    << ",\"total_median\":" << rel_t[n_fc_tgt / 2] << "}" << std::endl;
        }
      }

      if (force_count == 0) {
        kick_ms = 0.0;
        loc_active_next = timeline;
        loc_invalid = 0;
      } else {
        const double tk0 = wall();
        init_meta<<<1, 1, 0, G.compute_stream>>>(G.d_stats.p, G.d_tree_invalid.p, G.nodes);
        integrate_slab<<<blocks(force_count), team, 0, G.compute_stream>>>(
            G.d_v.p, G.d_old_acc.p, G.d_bins.p, G.d_force.p, G.d_mesh_acc.p,
            active_ptr, G.order.p, force_count, soft,
            soft_uniform ? nullptr : G.d_soft.p, p, ti,
            G.d_stats.p, G.d_kick_finish.p, G.d_kick_begin.p,
            is_pm_step ? pm_kick : 0.0, a, H_cur, dt_max_rms, short_span_cap,
            G.d_tree_invalid.p, nullptr, 0.0);
        Meta g_stats{};
        check(cudaMemcpyAsync(&g_stats, G.d_stats.p, sizeof(Meta), cudaMemcpyDeviceToHost, G.compute_stream));
        check(cudaStreamSynchronize(G.compute_stream));
        kick_ms = (wall() - tk0) * 1000.0;
        loc_active_next = g_stats.next;
        loc_invalid = g_stats.invalid;

        if (cfg.lazy_tree_enabled && !is_final && ns > 0) {
          const uint32_t* act_bits_ptr = do_lazy_step ? G.d_active_bits.p : nullptr;
          for (int depth = tree_levels; depth >= 0; --depth) {
            const int lcnt = G.h_levels[depth + 1] - G.h_levels[depth];
            if (lcnt <= 0) continue;
            compute_node_vel_active_3d_kernel<<<blocks(lcnt), team, 0, G.compute_stream>>>(
                G.geometry.p, G.force_nodes.p, G.d_v.p,
                mass_uniform ? nullptr : G.d_mass.p, first_mass,
                G.order.p, depth, G.levels.p, act_bits_ptr, G.d_slot_to_cell.p, G.node_vel_max.p);
          }
          check(cudaStreamSynchronize(G.compute_stream));
        }
      }

      if (!is_final) {
        export_ms = 0.0;
      } else {
        const double te0 = wall();
        if (ns > 0) {
          if (use_mpi_pm) {
            mpi_pm.interpolate(G.d_x.p, size_t(ns), h.box, p.G, G.d_mesh_acc.p, G.d_pm_pot.p);
          } else {
            grav_pm_cuda_interpolate_buf(
                G.d_x.p, size_t(ns), G.d_pm_field.p, pm_side, h.box, p.G,
                G.d_mesh_acc.p, G.d_pm_pot.p, G.compute_stream);
          }
        }
        std::vector<double3> hx(ns);
        std::vector<float3> hv(ns);
        std::vector<float4> hraw(ns);
        std::vector<float> hpot(ns);
        std::vector<uint32_t> hidx(ns);
        if (ns > 0) {
          check(cudaMemcpyAsync(hx.data(), G.d_x.p, size_t(ns) * sizeof(double3), cudaMemcpyDeviceToHost, G.compute_stream));
          check(cudaMemcpyAsync(hv.data(), G.d_v.p, size_t(ns) * sizeof(float3), cudaMemcpyDeviceToHost, G.compute_stream));
          check(cudaMemcpyAsync(hraw.data(), G.d_force.p, size_t(ns) * sizeof(float4), cudaMemcpyDeviceToHost, G.compute_stream));
          check(cudaMemcpyAsync(hpot.data(), G.d_pm_pot.p, size_t(ns) * sizeof(float), cudaMemcpyDeviceToHost, G.compute_stream));
          check(cudaMemcpyAsync(hidx.data(), G.d_orig_idx.p, size_t(ns) * sizeof(uint32_t), cudaMemcpyDeviceToHost, G.compute_stream));
        }
        check(cudaStreamSynchronize(G.compute_stream));

        const float inv_a = float(1.0 / a);
        uint64_t l_bad = 0;
        double l_v2 = 0.0, l_psum = 0.0;
        float l_pmin = FLT_MAX, l_pmax = -FLT_MAX;
        std::vector<float3> hv_out(ns);
        std::vector<float> hpot_tot(ns);
        for (int k = 0; k < ns; ++k) {
          const float pot = float(double(hraw[k].w) + potential_norm) * p.G + hpot[k];
          hpot_tot[k] = pot;
          const double vvx = double(hv[k].x) * inv_a;
          const double vvy = double(hv[k].y) * inv_a;
          const double vvz = double(hv[k].z) * inv_a;
          hv_out[k] = {float(vvx), float(vvy), float(vvz)};
          if (!std::isfinite(hx[k].x) || !std::isfinite(hx[k].y) || !std::isfinite(hx[k].z) ||
              (cfg.periodic && (hx[k].x < 0.0 || hx[k].x >= h.box ||
                                hx[k].y < 0.0 || hx[k].y >= h.box ||
                                hx[k].z < 0.0 || hx[k].z >= h.box)) ||
              !std::isfinite(hv[k].x) || !std::isfinite(hv[k].y) || !std::isfinite(hv[k].z) ||
              !std::isfinite(pot)) {
            ++l_bad;
          } else {
            l_v2 += vvx * vvx + vvy * vvy + vvz * vvz;
            l_psum += double(pot);
            if (pot < l_pmin) l_pmin = pot;
            if (pot > l_pmax) l_pmax = pot;
          }
        }
        uint64_t g_bad = l_bad;
        double g_v2 = l_v2, g_psum = l_psum;
        float g_pmin = l_pmin, g_pmax = l_pmax;
        if (num_ranks > 1) {
          GRAV_MPI_CHECK(MPI_Reduce(&l_bad, &g_bad, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, mpi.world));
          GRAV_MPI_CHECK(MPI_Reduce(&l_v2, &g_v2, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.world));
          GRAV_MPI_CHECK(MPI_Reduce(&l_psum, &g_psum, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.world));
          GRAV_MPI_CHECK(MPI_Reduce(&l_pmin, &g_pmin, 1, MPI_FLOAT, MPI_MIN, 0, mpi.world));
          GRAV_MPI_CHECK(MPI_Reduce(&l_pmax, &g_pmax, 1, MPI_FLOAT, MPI_MAX, 0, mpi.world));
        }
        if (my_rank == 0) {
          sanity_bad += size_t(g_bad);
          sanity_v2_sum += g_v2;
          sanity_pot_sum += g_psum;
          sanity_pot_min = std::min(sanity_pot_min, g_pmin);
          sanity_pot_max = std::max(sanity_pot_max, g_pmax);
        }

        if (!cfg.skip_output_write) {
          std::vector<int> rcounts(num_ranks, 0);
          int my_ni = ns;
          MPI_Gather(&my_ni, 1, MPI_INT, rcounts.data(), 1, MPI_INT, 0, mpi.world);
          std::vector<double3> gx_all;
          std::vector<float3> gv_all;
          std::vector<float> gp_all;
          std::vector<uint32_t> gid_all;
          auto gather_typed = [&](const void* sbuf, size_t elem_sz, void* rbuf) {
            std::vector<int> rbc(num_ranks, 0), rbd(num_ranks, 0);
            if (my_rank == 0) {
              int acc = 0;
              for (int r = 0; r < num_ranks; ++r) {
                rbd[r] = acc;
                rbc[r] = rcounts[r] * int(elem_sz);
                acc += rbc[r];
              }
            }
            GRAV_MPI_CHECK(MPI_Gatherv(
                sbuf, my_ni * int(elem_sz), MPI_BYTE,
                rbuf, rbc.data(), rbd.data(), MPI_BYTE, 0, mpi.world));
          };
          if (my_rank == 0) {
            gx_all.resize(N); gv_all.resize(N); gp_all.resize(N); gid_all.resize(N);
          }
          gather_typed(hx.data(), sizeof(double3), my_rank == 0 ? gx_all.data() : nullptr);
          gather_typed(hv_out.data(), sizeof(float3), my_rank == 0 ? gv_all.data() : nullptr);
          gather_typed(hpot_tot.data(), sizeof(float), my_rank == 0 ? gp_all.data() : nullptr);
          gather_typed(hidx.data(), sizeof(uint32_t), my_rank == 0 ? gid_all.data() : nullptr);
          if (my_rank == 0) {
            for (uint64_t i = 0; i < N; ++i) {
              const uint32_t oi = gid_all[i];
              out_x[oi] = gx_all[i];
              out_v[oi] = gv_all[i];
              out_pot[oi] = gp_all[i];
            }
          }
        }
        export_ms = (wall() - te0) * 1000.0;
      }
      ghost_installed = true;

      // Global step synchronization / reductions
      Tick step_next = std::min(loc_inactive_min, loc_active_next);
      uint64_t step_active = glb_active_step;
      uint64_t step_nodes = uint64_t(G.nodes);
      unsigned int step_invalid = loc_invalid;
      int step_builds = loc_build;
      double step_ghost_ms = ghost_ms, step_xfer_ms = export_ms;
      double step_force_ms = force_ms, step_kick_ms = kick_ms;

      if (num_ranks > 1) {
        uint64_t loc_nxt_u64 = uint64_t(step_next), glb_nxt_u64 = 0;
        uint64_t loc_nd_u64 = step_nodes, glb_nd_u64 = 0;
        GRAV_MPI_CHECK(MPI_Allreduce(&loc_nxt_u64, &glb_nxt_u64, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, mpi.world));
        GRAV_MPI_CHECK(MPI_Allreduce(&loc_nd_u64, &glb_nd_u64, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, mpi.world));
        GRAV_MPI_CHECK(MPI_Allreduce(&loc_invalid, &step_invalid, 1, MPI_UNSIGNED, MPI_BOR, mpi.world));
        GRAV_MPI_CHECK(MPI_Allreduce(&loc_build, &step_builds, 1, MPI_INT, MPI_SUM, mpi.world));
        GRAV_MPI_CHECK(MPI_Reduce(&force_ms, &step_force_ms, 1, MPI_DOUBLE, MPI_MAX, 0, mpi.world));
        GRAV_MPI_CHECK(MPI_Reduce(&kick_ms, &step_kick_ms, 1, MPI_DOUBLE, MPI_MAX, 0, mpi.world));
        GRAV_MPI_CHECK(MPI_Reduce(&ghost_ms, &step_ghost_ms, 1, MPI_DOUBLE, MPI_MAX, 0, mpi.world));
        GRAV_MPI_CHECK(MPI_Reduce(&export_ms, &step_xfer_ms, 1, MPI_DOUBLE, MPI_MAX, 0, mpi.world));
        step_next = Tick(glb_nxt_u64);
        step_nodes = glb_nd_u64;
      }

      if (step_invalid || !step_active || (!is_final && (step_next <= ti || step_next > timeline)))
        throw std::runtime_error("Invalid MPI 3D state at step " + std::to_string(global_iteration) +
                                 " rank=" + std::to_string(my_rank) +
                                 " invalid=" + std::to_string(step_invalid) +
                                 " active=" + std::to_string(step_active));

      const double step_wall = wall() - step_start;
      if (my_rank == 0) {
        std::cout << "STEP {\"step\":" << global_iteration << ",\"a\":" << a
                  << ",\"integer_time\":" << ti << ",\"active\":" << step_active
                  << ",\"nodes\":" << step_nodes << ",\"wall_s\":" << step_wall
                  << ",\"drift_ms\":" << drift_ms << ",\"key_sort_ms\":0"
                  << ",\"tree_multipoles_ms\":" << tree_ms << ",\"force_ms\":" << step_force_ms
                  << ",\"kick_bins_ms\":" << step_kick_ms
                  << ",\"h2d_d2h_ms\":" << (step_ghost_ms + step_xfer_ms)
                  << ",\"ghost_ms\":" << step_ghost_ms
                  << ",\"slab_builds\":" << step_builds << ",\"migrants\":" << total_migrants
                  << ",\"pm\":" << (is_pm_step ? "true" : "false")
                  << ",\"lazy\":" << (do_lazy_step ? "true" : "false")
                  << ",\"pm_device_ms\":" << pm_ms << "}" << std::endl;
      }

      if (is_final) break;
      previous = ti;
      ti = step_next;
    }
  }

  if (cfg.force0_out_path && my_rank == 0) {
    FILE* fp = fopen(cfg.force0_out_path, "wb");
    if (!fp) throw std::runtime_error("Could not open GRAV_FORCE0_OUT output");
    write_bin(fp, out_force0_f4.data(), out_force0_f4.size());
    if (fclose(fp)) throw std::runtime_error("Could not close GRAV_FORCE0_OUT");
  }

  const double start_output = wall();
  size_t output_bytes = 0;
  if (!cfg.skip_output_write && my_rank == 0) {
    FILE* output = fopen(cfg.output_path, "wb");
    if (!output) return 4;
    const Footer footer{N, h.box, cfg.a_end};
    write_bin(output, &footer, 1);
    write_bin(output, out_x.data(), N);
    write_bin(output, out_v.data(), N);
    write_bin(output, out_pot.data(), N);
    if (fclose(output)) return 4;
    output_bytes = N * (sizeof(double3) + sizeof(float3) + sizeof(float));
  }
  const double export_s = wall() - start_output;
  MPI_Barrier(mpi.world);
  const double evo_wall = wall() - evo_start;

  uint64_t loc_db = uint64_t(G.device_bytes), sum_db = 0;
  double max_pm_mpi = pm_mpi_acc_ms, max_gh_mpi = ghost_mpi_acc_ms, max_mig_mpi = mig_mpi_acc_ms;
  if (num_ranks > 1) {
    GRAV_MPI_CHECK(MPI_Reduce(&loc_db, &sum_db, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, mpi.world));
    GRAV_MPI_CHECK(MPI_Reduce(&pm_mpi_acc_ms, &max_pm_mpi, 1, MPI_DOUBLE, MPI_MAX, 0, mpi.world));
    GRAV_MPI_CHECK(MPI_Reduce(&ghost_mpi_acc_ms, &max_gh_mpi, 1, MPI_DOUBLE, MPI_MAX, 0, mpi.world));
    GRAV_MPI_CHECK(MPI_Reduce(&mig_mpi_acc_ms, &max_mig_mpi, 1, MPI_DOUBLE, MPI_MAX, 0, mpi.world));
  } else {
    sum_db = loc_db;
  }

  if (my_rank == 0) {
    std::cout << std::setprecision(12)
              << "SANITY {\"particles\":" << N << ",\"bad_count\":" << sanity_bad
              << ",\"v_rms_kms_internal\":" << sqrt(sanity_v2_sum / double(N))
              << ",\"pot_mean\":" << (sanity_pot_sum / double(N))
              << ",\"pot_min\":" << sanity_pot_min << ",\"pot_max\":" << sanity_pot_max << "}" << std::endl;

    std::cout << std::setprecision(12)
              << "RESULT {\"evolution_wall_s\":" << evo_wall
              << ",\"whole_program_s\":" << (wall() - cfg.started_wall)
              << ",\"pm_updates_including_initial\":" << pm_updates
              << ",\"fast_tree\":true,\"warp_force\":true"
              << ",\"warp_m2l\":" << (cfg.warp_m2l ? "true" : "false")
              << ",\"fast_split\":" << (cfg.fast_split ? "true" : "false")
              << ",\"split_force\":" << (cfg.split_force_enabled ? "true" : "false")
              << ",\"pm_dist\":true"
              << ",\"domain_mode\":\"" << domain_mode_name(cfg.domain_mode)
              << "\",\"rebalances\":" << rebalance_count
              << ",\"split_batch_warps\":" << cfg.split_batch_warps
              << ",\"lazy_tree\":" << (cfg.lazy_tree_enabled ? "true" : "false")
              << ",\"split_batches\":" << split_batches_run
              << ",\"split_overflow_retries\":" << split_overflow_retries
              << ",\"lazy_steps\":" << total_lazy_steps
              << ",\"full_rebuilds\":" << total_full_rebuilds
              << ",\"dynamic_pm\":" << (cfg.dynamic_pm ? "true" : "false")
              << ",\"last_pm_span_scale\":" << last_pm_span_scale
              << ",\"last_dt_max_rms\":" << last_dt_max_rms
              << ",\"peak_slab_particles\":" << peak_gpu_seen
              << ",\"device_buffer_bytes\":" << sum_db
              << ",\"memory_mode\":\"mpi_domain_3d\",\"num_gpus\":" << num_ranks
              << ",\"num_ranks\":" << num_ranks << ",\"num_nodes\":" << mpi.num_nodes
              << ",\"pm_mpi_ms\":" << max_pm_mpi
              << ",\"ghost_mpi_ms\":" << max_gh_mpi
              << ",\"migrant_mpi_ms\":" << max_mig_mpi
              << ",\"intra_node_ghost_cells\":" << mstats.intra_node_ghost_cells
              << ",\"inter_node_ghost_cells\":" << mstats.inter_node_ghost_cells
              << ",\"final_particle_D2H_bytes\":" << output_bytes
              << ",\"export_s\":" << export_s << "}" << std::endl;
  }

  for (int p_i = 0; p_i < num_ranks; ++p_i) {
    if (p_i != my_rank) edges[p_i].free_all(G.dev, G.device_bytes);
  }
  if (use_mpi_pm) mpi_pm.destroy();
  if (d_energy_accum) {
    check(cudaSetDevice(G.dev));
    cudaFree(d_energy_accum);
  }
  check(cudaSetDevice(G.dev));
  G.cleanup();
  if (fc_target_cap > 0 && my_rank == 0) {
    check(cudaSetDevice(G.dev));
    d_ewald_modes.free(aux_dev_bytes); d_ewald_C.free(aux_dev_bytes); d_ewald_S.free(aux_dev_bytes);
    d_fc_tgt_x.free(aux_dev_bytes); d_fc_tgt_h.free(aux_dev_bytes); d_fc_lids.free(aux_dev_bytes); d_fc_tgt_slab.free(aux_dev_bytes);
    d_fc_gpu_short.free(aux_dev_bytes); d_fc_gpu_mesh.free(aux_dev_bytes);
    d_fc_short_ref.free(aux_dev_bytes); d_fc_total_ref.free(aux_dev_bytes);
  }
  int bad_exit = sanity_bad ? 6 : 0;
  MPI_Bcast(&bad_exit, 1, MPI_INT, 0, mpi.world);
  return bad_exit;
}

} // namespace grav3d

#endif
