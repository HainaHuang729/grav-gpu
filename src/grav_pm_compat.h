#ifndef GRAV_PM_COMPAT_H
#define GRAV_PM_COMPAT_H

// Zero-dependency compatibility shim for PM FFT solvers (grav_cuda_pm*, grav_cosmo_hydro).
// Uses upstream SWIFT config.h / kernel_long_gravity.h automatically when KERNEL_BUILD_DIR
// is on the include path; otherwise defaults to GADGET2 Gaussian long-range split so
// `make pm`, `make pm-mpi`, `make test-pm`, and `make cosmo-hydro` build with zero external deps.

#if defined(__has_include)
#  if __has_include("config.h")
#    include "config.h"
#  endif
#  if __has_include("kernel_long_gravity.h")
#    ifndef __host__
#      define __host__
#    endif
#    include "kernel_long_gravity.h"
#  else
#    ifndef GADGET2_LONG_RANGE_CORRECTION
#      define GADGET2_LONG_RANGE_CORRECTION 1
#    endif
#  endif
#else
#  ifndef GADGET2_LONG_RANGE_CORRECTION
#    define GADGET2_LONG_RANGE_CORRECTION 1
#  endif
#endif

#endif  // GRAV_PM_COMPAT_H
