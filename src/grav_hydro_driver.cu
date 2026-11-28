#include "grav_cuda_hydro.h"
#include "grav_config_file.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

struct HydroBinHeader {
  int32_t n_gas;
  int32_t n_dm;
  double box[3];
  int32_t periodic[3];
  int32_t pad;
  double gamma;
};

static double env_double(const char* name, double def) {
  const char* s = std::getenv(name);
  return (s && *s) ? std::atof(s) : def;
}

static int env_int(const char* name, int def) {
  const char* s = std::getenv(name);
  return (s && *s) ? std::atoi(s) : def;
}

int main(int argc, char** argv) {
  grav_cfg::apply_cli_and_env_config(argc, argv);
  if (argc < 4) {
    std::fprintf(stderr, "Usage: %s <ic.bin> <out.bin> <t_end>\n", argv[0]);
    return 1;
  }
  const char* ic_path = argv[1];
  const char* out_path = argv[2];
  const double t_end = std::atof(argv[3]);

  FILE* fin = std::fopen(ic_path, "rb");
  if (!fin) {
    std::fprintf(stderr, "Failed to open IC: %s\n", ic_path);
    return 2;
  }
  HydroBinHeader hdr{};
  if (std::fread(&hdr, sizeof(hdr), 1, fin) != 1 || hdr.n_gas <= 0) {
    std::fprintf(stderr, "Invalid hydro IC header\n");
    std::fclose(fin);
    return 3;
  }

  const int n = hdr.n_gas;
  const int n_dm = hdr.n_dm;
  std::vector<double> x(3 * n), v(3 * n), m(n), u(n), h(n), rho(n, 0.0), P(n, 0.0);
  if (std::fread(x.data(), sizeof(double), 3 * n, fin) != size_t(3 * n) ||
      std::fread(v.data(), sizeof(double), 3 * n, fin) != size_t(3 * n) ||
      std::fread(m.data(), sizeof(double), n, fin) != size_t(n) ||
      std::fread(u.data(), sizeof(double), n, fin) != size_t(n) ||
      std::fread(h.data(), sizeof(double), n, fin) != size_t(n)) {
    std::fprintf(stderr, "Truncated gas arrays in hydro IC\n");
    std::fclose(fin);
    return 4;
  }

  std::vector<double> dm_x, dm_v, dm_m;
  if (n_dm > 0) {
    dm_x.resize(3 * n_dm);
    dm_v.resize(3 * n_dm);
    dm_m.resize(n_dm);
    if (std::fread(dm_x.data(), sizeof(double), 3 * n_dm, fin) != size_t(3 * n_dm) ||
        std::fread(dm_v.data(), sizeof(double), 3 * n_dm, fin) != size_t(3 * n_dm) ||
        std::fread(dm_m.data(), sizeof(double), n_dm, fin) != size_t(n_dm)) {
      std::fprintf(stderr, "Truncated DM arrays in hydro IC\n");
      std::fclose(fin);
      return 5;
    }
  }
  std::fclose(fin);

  GravHydroConfig cfg{};
  for (int d = 0; d < 3; ++d) {
    cfg.box[d] = hdr.box[d];
    cfg.periodic[d] = hdr.periodic[d];
  }
  cfg.gamma = hdr.gamma;
  cfg.eta_sph = env_double("GRAV_HYDRO_ETA", 1.2348);
  cfg.alpha_visc = env_double("GRAV_HYDRO_ALPHA_VISC", 1.0);
  cfg.alpha_cond = env_double("GRAV_HYDRO_ALPHA_COND", 0.25);
  cfg.cfl = env_double("GRAV_HYDRO_CFL", 0.25);
  cfg.h_min = env_double("GRAV_HYDRO_H_MIN", 1e-4);
  cfg.h_max = env_double("GRAV_HYDRO_H_MAX", 0.25);
  cfg.use_balsara = env_int("GRAV_HYDRO_BALSARA", 1);
  cfg.enable_cooling = env_int("GRAV_HYDRO_COOLING", 0);
  cfg.cooling_unit_scale = env_double("GRAV_HYDRO_COOLING_SCALE", 1.0);
  cfg.u_floor = env_double("GRAV_HYDRO_U_FLOOR", 1e-4);
  cfg.enable_gravity = env_int("GRAV_HYDRO_GRAVITY", 0);
  cfg.grav_G = env_double("GRAV_HYDRO_G", 1.0);
  cfg.grav_soft = env_double("GRAV_HYDRO_SOFT", 0.05);
  cfg.max_steps = env_int("GRAV_HYDRO_MAX_STEPS", 20000);
  cfg.log_interval = env_int("GRAV_HYDRO_LOG_INTERVAL", 50);

  GravHydroState st{};
  st.n_gas = n;
  st.n_dm = n_dm;
  st.x = x.data();
  st.v = v.data();
  st.m = m.data();
  st.u = u.data();
  st.h = h.data();
  st.rho = rho.data();
  st.P = P.data();
  if (n_dm > 0) {
    st.dm_x = dm_x.data();
    st.dm_v = dm_v.data();
    st.dm_m = dm_m.data();
  }

  GravHydroStats stats{};
  const int rc = grav_hydro_evolve_gpu(&st, &cfg, t_end, &stats);
  if (rc != 0) {
    std::fprintf(stderr, "grav_hydro_evolve_gpu failed (rc=%d)\n", rc);
    return 6;
  }

  FILE* fout = std::fopen(out_path, "wb");
  if (!fout) {
    std::fprintf(stderr, "Failed to open output: %s\n", out_path);
    return 7;
  }
  std::fwrite(&hdr, sizeof(hdr), 1, fout);
  std::fwrite(x.data(), sizeof(double), 3 * n, fout);
  std::fwrite(v.data(), sizeof(double), 3 * n, fout);
  std::fwrite(m.data(), sizeof(double), n, fout);
  std::fwrite(u.data(), sizeof(double), n, fout);
  std::fwrite(rho.data(), sizeof(double), n, fout);
  std::fwrite(h.data(), sizeof(double), n, fout);
  std::fwrite(P.data(), sizeof(double), n, fout);
  std::fclose(fout);

  std::cout << std::setprecision(10)
            << "HYDRO_RESULT {\"n_gas\":" << n
            << ",\"n_dm\":" << n_dm
            << ",\"steps\":" << stats.steps
            << ",\"t_end\":" << stats.t_end
            << ",\"wall_s\":" << stats.wall_s
            << ",\"E_kin_init\":" << stats.E_kin_init
            << ",\"E_int_init\":" << stats.E_int_init
            << ",\"E_pot_init\":" << stats.E_pot_init
            << ",\"E_tot_init\":" << stats.E_tot_init
            << ",\"E_kin_final\":" << stats.E_kin_final
            << ",\"E_int_final\":" << stats.E_int_final
            << ",\"E_pot_final\":" << stats.E_pot_final
            << ",\"E_tot_final\":" << stats.E_tot_final
            << ",\"E_rad_lost\":" << stats.E_rad_lost
            << ",\"max_rel_dE\":" << stats.max_rel_dE
            << "}" << std::endl;
  return 0;
}
