#ifndef SWIFT_TREE_PROBE_FORMAT_H
#define SWIFT_TREE_PROBE_FORMAT_H
#include <stdint.h>
#include "swift_cuda_pm.h"

#define PROBE_TERMS 32
#define PROBE_LEVELS 17
#define PROBE_TERM_LIST(X) \
  X(0,0,0,0) \
  X(1,0,0,2) \
  X(2,0,1,1) \
  X(3,0,2,0) \
  X(4,1,0,1) \
  X(5,1,1,0) \
  X(6,2,0,0) \
  X(7,0,0,3) \
  X(8,0,1,2) \
  X(9,0,2,1) \
  X(10,0,3,0) \
  X(11,1,0,2) \
  X(12,1,1,1) \
  X(13,1,2,0) \
  X(14,2,0,1) \
  X(15,2,1,0) \
  X(16,3,0,0) \
  X(17,0,0,4) \
  X(18,0,1,3) \
  X(19,0,2,2) \
  X(20,0,3,1) \
  X(21,0,4,0) \
  X(22,1,0,3) \
  X(23,1,1,2) \
  X(24,1,2,1) \
  X(25,1,3,0) \
  X(26,2,0,2) \
  X(27,2,1,1) \
  X(28,2,2,0) \
  X(29,3,0,1) \
  X(30,3,1,0) \
  X(31,4,0,0)
struct probe_header {
  uint64_t particles, nodes;
  int top[3], leaf_limit, mesh;
  float grav_G;
  double width[3], inv_width[3], box, smoothing;
};
struct probe_particle {
  struct swift_pm_particle pm;
  float softening, old_acceleration;
};
struct probe_node {
  uint64_t key;
  int level, count;
  double center[3];
  float radius, softening, old_acceleration, moments[PROBE_TERMS], power[5];
};
#endif
