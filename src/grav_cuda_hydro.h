#ifndef GRAV_CUDA_HYDRO_H
#define GRAV_CUDA_HYDRO_H

#include <cstdint>

// B4: Full-GPU VRAM-Resident SPH Hydrodynamics + Tabulated Radiative Cooling + Coupled Self-Gravity.
// Designed for 100% GPU execution (spatial-hash neighbor search, Wendland C2 kernel,
// adaptive h_i iteration, Balsara viscosity switch, Price artificial conduction,
// KDK leapfrog, and implicit subcycled Sutherland-Dopita radiative cooling).

struct GravHydroConfig {
  double box[3] = {1.0, 1.0, 1.0};
  int periodic[3] = {1, 1, 1};
  double gamma = 5.0 / 3.0;          // Adiabatic index
  double eta_sph = 1.2348;           // h = eta * (m/rho)^(1/3) -> ~64 neighbors in 3D
  double alpha_visc = 1.0;           // Monaghan artificial viscosity strength
  double alpha_cond = 0.25;          // Price (2008) artificial thermal conduction
  double cfl = 0.25;                 // CFL timestep factor
  double h_min = 1e-4;               // Minimum smoothing length
  double h_max = 0.25;               // Maximum smoothing length
  int use_balsara = 1;               // 1 = enable Balsara curl/div viscosity switch
  int enable_cooling = 0;            // 1 = enable GPU tabulated radiative cooling
  double cooling_unit_scale = 1.0;   // Scaling factor for cooling rate in code units
  double u_floor = 1e-4;             // Minimum specific internal energy (T_floor)
  int enable_gravity = 0;            // 1 = enable coupled N-body self-gravity (gas + optional DM)
  double grav_G = 1.0;               // Gravitational constant G
  double grav_soft = 0.05;           // Plummer softening for self-gravity
  int max_steps = 20000;
  int log_interval = 50;
};

struct GravHydroState {
  int n_gas = 0;
  int n_dm = 0;                      // Optional collisionless DM particles for Santa Barbara cluster
  // Host arrays (size n_gas for gas, n_dm for DM)
  double* x = nullptr;               // [3 * n_gas]
  double* v = nullptr;               // [3 * n_gas]
  double* m = nullptr;               // [n_gas]
  double* u = nullptr;               // [n_gas] specific internal energy
  double* rho = nullptr;             // [n_gas]
  double* h = nullptr;               // [n_gas]
  double* P = nullptr;               // [n_gas]
  // Optional DM host arrays (size n_dm)
  double* dm_x = nullptr;            // [3 * n_dm]
  double* dm_v = nullptr;            // [3 * n_dm]
  double* dm_m = nullptr;            // [n_dm]
};

struct GravHydroStats {
  int steps = 0;
  double t_end = 0.0;
  double wall_s = 0.0;
  double E_kin_init = 0.0, E_int_init = 0.0, E_pot_init = 0.0, E_tot_init = 0.0;
  double E_kin_final = 0.0, E_int_final = 0.0, E_pot_final = 0.0, E_tot_final = 0.0;
  double E_rad_lost = 0.0;           // Cumulative energy radiated away by cooling
  double max_rel_dE = 0.0;
};

int grav_hydro_evolve_gpu(GravHydroState* state, const GravHydroConfig* cfg,
                          double t_end, GravHydroStats* stats_out);

#endif  // GRAV_CUDA_HYDRO_H
