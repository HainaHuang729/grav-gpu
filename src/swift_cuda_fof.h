#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ABI-compatible exact periodic GPU FoF (returns union-find root particle indices in [0, count)).
// elapsed[9]: upload, grid_key, radix_sort, rle, cell_offsets, hash_and_intra, neighbors_and_union, finalize_labels, download.
// info[7]: occupied_cells, init_free, total_mem, sort_bytes, rle_bytes, hash_slots, mid_free.
int whole_fof(const double* host, int count,
              double lx, double ly, double lz, double link,
              int* labels_host, double* elapsed, std::uint64_t* info);

// SWIFT/HBT+-compatible periodic GPU FoF:
// Outputs int64 group IDs (1..num_groups ordered by descending size; ungrouped/size < min_group_size = 2147483647).
int whole_fof_swift(const double* host, int count,
                    double lx, double ly, double lz, double link,
                    int min_group_size,
                    std::int64_t* swift_group_ids_host,
                    std::uint64_t* out_num_groups,
                    std::uint64_t* out_grouped_particles,
                    std::uint64_t* out_largest_group,
                    double* elapsed, std::uint64_t* info);

// Slab-streamed single-GPU exact periodic FoF (for large snapshots up to ~1024^3 in limited VRAM).
// Streams num_slabs x-cell slabs with +1 cell periodic halo through GPU FoF and merges via host Union-Find.
int whole_fof_slabbed_swift(const double* host, std::uint64_t count,
                            double lx, double ly, double lz, double link,
                            int min_group_size, int num_slabs,
                            std::int64_t* swift_group_ids_host,
                            std::uint64_t* out_num_groups,
                            std::uint64_t* out_grouped_particles,
                            std::uint64_t* out_largest_group,
                            double* elapsed, std::uint64_t* info);

struct FofHaloRow {
  std::int64_t group_id;
  std::uint64_t size;
  double mass;
  double com[3];
  double bulk_vel[3];
  double vel_disp;
  double r_max;
};

// Computes per-halo physical properties using two-pass periodic nearest-image CoM anchoring.
// masses_or_null may be nullptr when particles share uniform_mass; vel_xyz may be nullptr to skip velocity stats.
int compute_fof_catalog(const double* pos_xyz, const float* vel_xyz,
                        const float* masses_or_null, double uniform_mass,
                        const std::int64_t* swift_group_ids,
                        std::uint64_t count, std::uint64_t num_groups,
                        double lx, double ly, double lz,
                        FofHaloRow* out_halos);

struct FofSoRow {
  double m200c;
  double r200c;
  double m200m;
  double r200m;
};

// Computes spherical overdensity M200c/R200c and M200m/R200m (comoving radii, same mass units)
// around each halo's center using a periodic 3D spatial grid and log-log enclosed-density interpolation.
int compute_fof_so(const double* pos_xyz, const float* masses_or_null,
                   double uniform_mass, std::uint64_t count,
                   const double* centers_xyz, const std::uint64_t* sizes,
                   const double* fof_masses, const double* r_max_arr,
                   std::uint64_t num_groups,
                   double lx, double ly, double lz,
                   double rho_crit_com, double rho_mean_com,
                   int min_so_size, FofSoRow* out_so);

// Device-resident single-GPU SWIFT FoF: operates directly on read-only device positions dev_pos_xyz
// on the active CUDA device without H2D upload. Writes outputs to swift_group_ids_dev and/or
// swift_group_ids_host (at least one must be non-null).
int whole_fof_device_swift(const double* dev_pos_xyz, int count,
                           double lx, double ly, double lz, double link,
                           int min_group_size,
                           std::int64_t* swift_group_ids_dev,
                           std::int64_t* swift_group_ids_host,
                           std::uint64_t* out_num_groups,
                           std::uint64_t* out_grouped_particles,
                           std::uint64_t* out_largest_group,
                           double* elapsed, std::uint64_t* info);

// Multi-GPU device-resident SWIFT FoF across arbitrary 3D domain decompositions.
// dev_pos_xyz[g] points to dev_counts[g] particles (3*doubles) on CUDA device dev_ids[g] (read-only).
// Optional dev_orig_idx[g] maps local particles to global indices in [0, total_count); if nullptr or
// dev_orig_idx[g] == nullptr, contiguous prefix slices are assumed.
int whole_fof_multigpu_swift(int num_gpus, const int* dev_ids,
                             const double* const* dev_pos_xyz,
                             const std::uint32_t* const* dev_orig_idx,
                             const std::uint64_t* dev_counts,
                             std::uint64_t total_count,
                             double lx, double ly, double lz, double link,
                             int min_group_size,
                             std::int64_t* swift_group_ids_host,
                             std::uint64_t* out_num_groups,
                             std::uint64_t* out_grouped_particles,
                             std::uint64_t* out_largest_group,
                             double* elapsed);

#ifdef __cplusplus
}
#endif
