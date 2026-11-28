#ifndef GRAV_CONFIG_FILE_H
#define GRAV_CONFIG_FILE_H

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// v3.0 Unified INI / TOML-lite Parameter File Loader for gravityCU.
// Supports:
//   --config <path.ini>  (CLI flag, stripped before positional args)
//   GRAV_CONFIG=<path>   (environment variable)
// Precedence: explicit GRAV_* environment variables ALWAYS override config file values
// (setenv(..., /*overwrite=*/0)), preserving full backward compatibility with Slurm sweeps!

namespace grav_cfg {

inline std::string trim(const std::string& s) {
  size_t a = 0;
  while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
  size_t b = s.size();
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
  return s.substr(a, b - a);
}

inline std::string lower_str(std::string s) {
  for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

inline std::string strip_quotes(const std::string& s) {
  if (s.size() >= 2 &&
      ((s.front() == '"' && s.back() == '"') || (s.front() == '\'' && s.back() == '\''))) {
    return s.substr(1, s.size() - 2);
  }
  return s;
}

inline std::string map_section_key_to_env(const std::string& sec, const std::string& key) {
  // Direct GRAV_* keys pass straight through
  if (key.size() >= 5 && key.compare(0, 5, "grav_") == 0) {
    std::string up = key;
    for (char& c : up) c = char(std::toupper(static_cast<unsigned char>(c)));
    return up;
  }
  static const std::unordered_map<std::string, std::string> kMap = {
      // [domain]
      {"domain.num_gpus", "GRAV_NUM_GPUS"},
      {"domain.mode", "GRAV_DOMAIN_MODE"},
      {"domain.p2p_mode", "GRAV_P2P_MODE"},
      {"domain.split_force", "GRAV_SPLIT_FORCE"},
      {"domain.lazy_tree", "GRAV_LAZY_TREE"},
      {"domain.rebalance_interval", "GRAV_REBALANCE_INTERVAL"},
      {"domain.rebalance_min_gain", "GRAV_REBALANCE_MIN_GAIN"},
      // [pm]
      {"pm.grid", "GRAV_PM_GRID"},
      {"pm.dist", "GRAV_PM_DIST"},
      {"pm.dynamic", "GRAV_DYNAMIC_PM"},
      {"pm.order", "GRAV_PM_ORDER"},
      {"pm.interlace", "GRAV_PM_INTERLACE"},
      {"pm.grad", "GRAV_PM_GRAD"},
      {"pm.asmth", "GRAV_PM_ASMTH"},
      {"pm.rcut", "GRAV_PM_RCUT"},
      // [precision]
      {"precision.mode", "GRAV_PRECISION_MODE"},
      {"precision.theta", "GRAV_THETA"},
      {"precision.mac", "GRAV_MAC"},
      {"precision.eta", "GRAV_ETA"},
      {"precision.dt_max", "GRAV_DT_MAX"},
      {"precision.tidal_timestep", "GRAV_TIDAL_TIMESTEP"},
      {"precision.tidal_eta", "GRAV_TIDAL_ETA"},
      {"precision.vel_eta", "GRAV_VEL_ETA"},
      {"precision.sync_step", "GRAV_SYNC_STEP"},
      // [cosmology]
      {"cosmology.periodic", "GRAV_PERIODIC"},
      {"cosmology.comoving", "GRAV_COMOVING"},
      {"cosmology.g", "GRAV_G"},
      {"cosmology.h0", "GRAV_COSMO_H0"},
      {"cosmology.h", "GRAV_COSMO_H"},
      {"cosmology.omega_m", "GRAV_COSMO_OMEGA_M"},
      {"cosmology.omega_de", "GRAV_COSMO_OMEGA_DE"},
      {"cosmology.omega_cdm", "GRAV_COSMO_OMEGA_CDM"},
      {"cosmology.w0", "GRAV_COSMO_W0"},
      {"cosmology.wa", "GRAV_COSMO_WA"},
      {"cosmology.nu_mass_ev", "GRAV_NU_MASS_EV"},
      {"cosmology.nu_nspecies", "GRAV_NU_NSPECIES"},
      {"cosmology.nu_table", "GRAV_NU_TABLE"},
      {"cosmology.f_r0", "GRAV_FR0"},
      {"cosmology.fr_screening", "GRAV_FR_SCREENING"},
      // [sidm]
      {"sidm.sigma_m", "GRAV_SIDM_SIGMA_M"},
      {"sidm.sigma_internal", "GRAV_SIDM_SIGMA_INTERNAL"},
      {"sidm.w0", "GRAV_SIDM_W0"},
      {"sidm.hs", "GRAV_SIDM_HS"},
      {"sidm.seed", "GRAV_SIDM_SEED"},
      // [zoomin]
      {"zoomin.soft_in", "GRAV_SOFT_IN"},
      // [fof]
      {"fof.enable", "GRAV_RUN_FOF"},
      {"fof.link_ratio", "GRAV_FOF_LINK_RATIO"},
      {"fof.link_mpc", "GRAV_FOF_LINK_MPC"},
      {"fof.min_size", "GRAV_FOF_MIN_SIZE"},
      // [hydro]
      {"hydro.alpha_visc", "GRAV_HYDRO_ALPHA_VISC"},
      {"hydro.alpha_cond", "GRAV_HYDRO_ALPHA_COND"},
      {"hydro.balsara", "GRAV_HYDRO_BALSARA"},
      {"hydro.cfl", "GRAV_HYDRO_CFL"},
      {"hydro.eta", "GRAV_HYDRO_ETA"},
      {"hydro.cooling", "GRAV_HYDRO_COOLING"},
      {"hydro.cooling_scale", "GRAV_HYDRO_COOLING_SCALE"},
      {"hydro.u_floor", "GRAV_HYDRO_U_FLOOR"},
      {"hydro.gravity", "GRAV_HYDRO_GRAVITY"},
      {"hydro.g", "GRAV_HYDRO_G"},
      {"hydro.soft", "GRAV_HYDRO_SOFT"},
      {"hydro.sf", "GRAV_COSMO_SF"},
      {"hydro.sf_overdensity", "GRAV_COSMO_SF_OVERDENSITY"},
      {"hydro.sf_u_max", "GRAV_COSMO_SF_U_MAX"},
      {"hydro.sf_eps", "GRAV_COSMO_SF_EPS"},
      {"hydro.max_dlna", "GRAV_COSMO_MAX_DLNA"},
      // [output]
      {"output.energy_log", "GRAV_ENERGY_LOG"},
      {"output.skip_write", "GRAV_SKIP_OUTPUT_WRITE"},
  };
  const std::string full = sec.empty() ? key : (sec + "." + key);
  auto it = kMap.find(full);
  if (it != kMap.end()) return it->second;
  return "";
}

// Parse INI/TOML-lite file and populate unset GRAV_* environment variables.
// Returns number of keys applied.
inline int load_config_file(const std::string& path) {
  std::ifstream in(path);
  if (!in.is_open()) {
    throw std::runtime_error("Cannot open gravityCU config file: " + path);
  }
  std::string section;
  std::string line;
  int applied = 0;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    // Strip inline comments (# or ;) outside quotes
    bool in_q = false;
    char q_ch = 0;
    for (size_t i = 0; i < line.size(); ++i) {
      const char c = line[i];
      if (!in_q && (c == '"' || c == '\'')) { in_q = true; q_ch = c; }
      else if (in_q && c == q_ch) { in_q = false; }
      else if (!in_q && (c == '#' || c == ';')) { line = line.substr(0, i); break; }
    }
    line = trim(line);
    if (line.empty()) continue;
    if (line.front() == '[' && line.back() == ']') {
      section = lower_str(trim(line.substr(1, line.size() - 2)));
      continue;
    }
    const size_t eq = line.find('=');
    if (eq == std::string::npos) {
      throw std::runtime_error("Malformed config line " + std::to_string(line_no) + ": " + line);
    }
    const std::string key = lower_str(trim(line.substr(0, eq)));
    const std::string val = strip_quotes(trim(line.substr(eq + 1)));
    const std::string env_name = map_section_key_to_env(section, key);
    if (!env_name.empty()) {
      // Do NOT overwrite explicit environment variables!
      if (setenv(env_name.c_str(), val.c_str(), 0) == 0) ++applied;
    }
  }
  return applied;
}

// Extract --config <path> (or --config=<path>) from argv in-place, or read GRAV_CONFIG env var.
inline std::string apply_cli_and_env_config(int& argc, char** argv) {
  std::string cfg_path;
  int write_idx = 1;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
      cfg_path = argv[++i];
    } else if (arg.rfind("--config=", 0) == 0) {
      cfg_path = arg.substr(9);
    } else {
      argv[write_idx++] = argv[i];
    }
  }
  argc = write_idx;
  if (cfg_path.empty()) {
    if (const char* env_c = std::getenv("GRAV_CONFIG")) {
      if (*env_c) cfg_path = env_c;
    }
  }
  if (!cfg_path.empty()) {
    const int n = load_config_file(cfg_path);
    std::printf("CONFIG_FILE path=%s applied_keys=%d\n", cfg_path.c_str(), n);
    std::fflush(stdout);
  }
  return cfg_path;
}

}  // namespace grav_cfg

#endif  // GRAV_CONFIG_FILE_H
