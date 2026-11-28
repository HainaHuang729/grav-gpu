// CPU unit test for v3.0 unified parameter file loader (grav_config_file.h) & GravPhysics.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

#include "grav_config_file.h"
#include "grav_physics_stage.cuh"

int main() {
  // 1. Write a temp INI/TOML-lite config and verify mapping + env precedence
  const std::string path = "/tmp/gcu_v3_test.ini";
  {
    std::ofstream f(path);
    f << "# Test config\n"
      << "[pm]\n"
      << "grid = 256\n"
      << "order = \"pcs\" # inline comment\n"
      << "interlace = 1\n"
      << "[cosmology]\n"
      << "nu_mass_ev = 0.15\n"
      << "f_r0 = 1e-5\n"
      << "[precision]\n"
      << "mode = high\n";
  }
  // Pre-set GRAV_PM_GRID=512 to verify env-var override precedence!
  setenv("GRAV_PM_GRID", "512", 1);
  unsetenv("GRAV_PM_ORDER");
  unsetenv("GRAV_NU_MASS_EV");
  unsetenv("GRAV_FR0");

  char arg0[] = "prog";
  char arg1[] = "--config";
  char arg2[] = "/tmp/gcu_v3_test.ini";
  char arg3[] = "in.bin";
  char* argv[] = {arg0, arg1, arg2, arg3};
  int argc = 4;
  const std::string loaded = grav_cfg::apply_cli_and_env_config(argc, argv);
  assert(loaded == path);
  assert(argc == 2);
  assert(std::strcmp(argv[1], "in.bin") == 0);
  // Env override preserved:
  assert(std::strcmp(std::getenv("GRAV_PM_GRID"), "512") == 0);
  // Config file values applied:
  assert(std::strcmp(std::getenv("GRAV_PM_ORDER"), "pcs") == 0);
  assert(std::strcmp(std::getenv("GRAV_NU_MASS_EV"), "0.15") == 0);
  assert(std::strcmp(std::getenv("GRAV_FR0"), "1e-5") == 0);
  std::remove(path.c_str());

  // 2. Verify GravPhysics H(a) & factors()
  GravPhysics p{};
  p.a_begin = 0.1;
  p.log_span = std::log(10.0);
  p.hubble0 = 67.36;
  p.omega_m = 0.3153;
  p.omega_de = 0.6847;
  p.w0 = -1.0;
  p.wa = 0.0;
  p.comoving = true;
  assert(std::fabs(p.H(1.0) - 67.36) < 1e-10);
  assert(std::fabs(p.scale(GravPhysics::kTimeline) - 1.0) < 1e-12);
  assert(p.factors(0, GravPhysics::kTimeline / 16, true) > 0.0);

  std::printf("CONFIG_V3_UNIT_TESTS_PASSED\n");
  return 0;
}
