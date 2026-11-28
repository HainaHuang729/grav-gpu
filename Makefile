# Build & Smoke-Test Makefile for gravityCU

_DETECTED_NVCC := $(shell command -v nvcc 2>/dev/null)
CUDA_HOME ?= $(if $(_DETECTED_NVCC),$(patsubst %/bin/nvcc,%,$(_DETECTED_NVCC)),/usr/local/cuda)
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_86
CXX ?= g++
MPICXX ?= mpicxx
CC ?= gcc
PYTHON ?= python3

ifeq ($(wildcard $(CUDA_HOME)/include/cufft.h),)
  _CUDA_VER := $(notdir $(patsubst %/,%,$(CUDA_HOME)))
  _HPCSDK_MATH := $(firstword $(wildcard \
    $(CUDA_HOME)/../../math_libs/$(_CUDA_VER)/targets/x86_64-linux \
    $(CUDA_HOME)/../math_libs/$(_CUDA_VER)/targets/x86_64-linux \
    $(CUDA_HOME)/../../math_libs/*/targets/x86_64-linux \
    $(CUDA_HOME)/../math_libs/*/targets/x86_64-linux))
  ifneq ($(_HPCSDK_MATH),)
    MATHLIB_INC ?= $(_HPCSDK_MATH)/include
    MATHLIB_LIB ?= $(_HPCSDK_MATH)/lib
  endif
endif
MATHLIB_INC ?= $(CUDA_HOME)/include
MATHLIB_LIB ?= $(CUDA_HOME)/lib64

KERNEL_BUILD_DIR ?=
BUILD_DIR := build
GEN_DIR := $(BUILD_DIR)/gen

KERNEL_REQ_SRCS := \
	runner_gpu.h runner_gpu.c gpu_gravity_offload.cu gpu_gravity_functions.cuh \
	kernel_gravity.h kernel_long_gravity.h gravity_derivatives.h multipole.h

NVCC_COMMON := -O3 -std=c++17 -arch=$(CUDA_ARCH) --fmad=true -ccbin=$(CXX)

.PHONY: help all standalone fof pm pm-mpi cic engine mpi-engine test-domain3d test-fof test-mpi-pm test-pm-precision test-neutrino test-ic smoke clean check-kernels

help:
	@echo "Targets:"
	@echo "  make standalone                       Build standalone libs (fof, cic) — no SWIFT headers needed"
	@echo "  make fof                              Build $(BUILD_DIR)/grav_cuda_fof.so (CUDA FoF/Catalog/SO library)"
	@echo "  make cic                              Build $(BUILD_DIR)/libcic_deposit.so (C99 CIC helper)"
	@echo "  make test-domain3d                    Compile & run CPU unit test for 3D Hilbert/ORB domain planner"
	@echo "  make test-fof                         Build fof and run GPU ctypes FoF + Catalog + SO smoke test"
	@echo "  make pm KERNEL_BUILD_DIR=<dir>         Build $(BUILD_DIR)/grav_cuda_pm*.o (PM FFT objects)"
	@echo "  make engine KERNEL_BUILD_DIR=<dir>     Build multi_gpu_slab and slab_resident using upstream SWIFT headers"
	@echo "  make smoke KERNEL_BUILD_DIR=<dir>      Run full synthetic 64^3 multi-GPU evolution + FoF + SO smoke test"
	@echo "  make clean                            Remove $(BUILD_DIR)/"
	@echo ""
	@echo "Overrides: CUDA_HOME=$(CUDA_HOME) CUDA_ARCH=$(CUDA_ARCH) CXX=$(CXX) PYTHON=$(PYTHON)"

all: standalone

standalone: fof cic

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

fof: $(BUILD_DIR)/grav_cuda_fof.so

$(BUILD_DIR)/grav_hydro: src/grav_hydro_driver.cu src/grav_cuda_hydro.cu src/grav_cuda_hydro.h src/grav_config_file.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_COMMON) -Isrc src/grav_hydro_driver.cu src/grav_cuda_hydro.cu -o $@

hydro: $(BUILD_DIR)/grav_hydro

KERNEL_INC_OPT := $(if $(KERNEL_BUILD_DIR),-I$(KERNEL_BUILD_DIR) -I$(KERNEL_BUILD_DIR)/src,)

$(BUILD_DIR)/grav_cosmo_hydro: src/grav_cosmo_hydro.cu src/grav_cuda_pm.cu src/grav_cuda_pm.h src/grav_pm_compat.h src/grav_pm_config.cuh src/grav_config_file.h src/grav_neutrino.cuh src/grav_fr_gravity.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCC_COMMON) -Isrc $(KERNEL_INC_OPT) -I$(MATHLIB_INC) -L$(MATHLIB_LIB) \
		src/grav_cosmo_hydro.cu src/grav_cuda_pm.cu -lcufft -o $@

cosmo-hydro: $(BUILD_DIR)/grav_cosmo_hydro

$(BUILD_DIR)/grav_cuda_fof.so: src/grav_cuda_fof.cu src/grav_cuda_fof.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_COMMON) -Xcompiler "-fPIC -fopenmp" -shared -Isrc src/grav_cuda_fof.cu -o $@

pm: $(BUILD_DIR)/grav_cuda_pm.o $(BUILD_DIR)/grav_cuda_pm_dist.o

pm-mpi: $(BUILD_DIR)/grav_cuda_pm.o $(BUILD_DIR)/grav_cuda_pm_mpi.o

$(BUILD_DIR)/grav_cuda_pm.o: src/grav_cuda_pm.cu src/grav_cuda_pm.h src/grav_pm_compat.h src/grav_pm_config.cuh src/grav_neutrino.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCC_COMMON) -Xcompiler -fPIC -c -Isrc $(KERNEL_INC_OPT) -I$(MATHLIB_INC) src/grav_cuda_pm.cu -o $@

$(BUILD_DIR)/grav_cuda_pm_dist.o: src/grav_cuda_pm_dist.cu src/grav_cuda_pm_dist.h src/grav_pm_compat.h src/grav_pm_config.cuh src/grav_neutrino.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCC_COMMON) -Xcompiler -fPIC -c -Isrc $(KERNEL_INC_OPT) -I$(MATHLIB_INC) src/grav_cuda_pm_dist.cu -o $@

$(BUILD_DIR)/grav_cuda_pm_mpi.o: src/grav_cuda_pm_mpi.cu src/grav_cuda_pm_mpi.h src/grav_pm_compat.h src/grav_mpi_comm.h src/grav_pm_config.cuh src/grav_neutrino.cuh | $(BUILD_DIR)
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) --fmad=true -ccbin=$(MPICXX) -Xcompiler -fPIC -c -Isrc $(KERNEL_INC_OPT) -I$(MATHLIB_INC) src/grav_cuda_pm_mpi.cu -o $@

cic: $(BUILD_DIR)/libcic_deposit.so

$(BUILD_DIR)/libcic_deposit.so: src/cic_deposit.c | $(BUILD_DIR)
	$(CC) -O3 -shared -fPIC src/cic_deposit.c -lm -o $@

test-domain3d: $(BUILD_DIR)/test_domain3d
	./$(BUILD_DIR)/test_domain3d

$(BUILD_DIR)/test_domain3d: scripts/test_domain3d_smoke.cpp src/grav_domain_3d.h | $(BUILD_DIR)
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Werror -Isrc scripts/test_domain3d_smoke.cpp -o $@

$(BUILD_DIR)/test_mpi_pm: check-kernels scripts/test_mpi_pm.cu $(BUILD_DIR)/grav_cuda_pm.o $(BUILD_DIR)/grav_cuda_pm_mpi.o src/grav_domain_3d.h | $(BUILD_DIR)
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) --fmad=true -ccbin=$(MPICXX) \
		-Isrc -I$(KERNEL_BUILD_DIR) -I$(KERNEL_BUILD_DIR)/src -I$(MATHLIB_INC) \
		-L$(MATHLIB_LIB) scripts/test_mpi_pm.cu $(BUILD_DIR)/grav_cuda_pm.o $(BUILD_DIR)/grav_cuda_pm_mpi.o \
		-lcufft -o $@

test-mpi-pm: $(BUILD_DIR)/test_mpi_pm
	LD_LIBRARY_PATH="$(MATHLIB_LIB):$(CUDA_HOME)/lib64:$${LD_LIBRARY_PATH:-}" \
		mpirun -np 2 ./$(BUILD_DIR)/test_mpi_pm

$(BUILD_DIR)/test_pm_precision: check-kernels scripts/test_pm_precision.cu $(BUILD_DIR)/grav_cuda_pm.o $(BUILD_DIR)/grav_cuda_pm_dist.o src/grav_pm_config.cuh | $(BUILD_DIR)
	$(NVCC) $(NVCC_COMMON) -Isrc -I$(KERNEL_BUILD_DIR) -I$(KERNEL_BUILD_DIR)/src -I$(MATHLIB_INC) \
		-L$(MATHLIB_LIB) scripts/test_pm_precision.cu $(BUILD_DIR)/grav_cuda_pm.o $(BUILD_DIR)/grav_cuda_pm_dist.o \
		-lcufft -o $@

test-pm-precision: $(BUILD_DIR)/test_pm_precision
	LD_LIBRARY_PATH="$(MATHLIB_LIB):$(CUDA_HOME)/lib64:$${LD_LIBRARY_PATH:-}" \
		./$(BUILD_DIR)/test_pm_precision

test-neutrino: $(BUILD_DIR)/test_neutrino_b1
	./$(BUILD_DIR)/test_neutrino_b1

$(BUILD_DIR)/test_p3_b2: scripts/test_p3_b2.cpp src/grav_timestep_p3.cuh src/grav_fr_gravity.cuh src/grav_sidm.cuh | $(BUILD_DIR)
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Werror -Isrc $< -o $@

test-p3-b2: $(BUILD_DIR)/test_p3_b2
	./$(BUILD_DIR)/test_p3_b2

$(BUILD_DIR)/test_config_v3: scripts/test_config_v3.cpp src/grav_config_file.h src/grav_physics_stage.cuh src/grav_neutrino.cuh | $(BUILD_DIR)
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Werror -Isrc scripts/test_config_v3.cpp -o $@

test-config: $(BUILD_DIR)/test_config_v3
	./$(BUILD_DIR)/test_config_v3

test-zoomin:
	python3 scripts/zoomin_pipeline.py generate $(BUILD_DIR)/zoom_test.bin --soft-out $(BUILD_DIR)/zoom_test.soft.bin --n-eff 2048 --hr-half-width 0.35
	rm -f $(BUILD_DIR)/zoom_test.bin $(BUILD_DIR)/zoom_test.soft.bin

$(BUILD_DIR)/test_neutrino_b1: scripts/test_neutrino_b1.cpp src/grav_neutrino.cuh | $(BUILD_DIR)
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Werror -Isrc scripts/test_neutrino_b1.cpp -o $@

test-ic:
	$(PYTHON) scripts/test_ic_adapters.py

test-fof: fof
	LD_LIBRARY_PATH="$(CUDA_HOME)/lib64:$(MATHLIB_LIB):$${LD_LIBRARY_PATH:-}" \
		$(PYTHON) scripts/smoke_fof.py $(BUILD_DIR)/grav_cuda_fof.so

check-kernels:
	@if [ -z "$(KERNEL_BUILD_DIR)" ]; then \
		echo "ERROR: KERNEL_BUILD_DIR is not set."; \
		echo "Point it to an upstream GPU-configured SWIFT source/build tree containing config.h and src/multipole.h, e.g.:"; \
		echo "  make engine KERNEL_BUILD_DIR=/path/to/swiftsim"; \
		exit 1; \
	fi
	@test -f "$(KERNEL_BUILD_DIR)/config.h" || (echo "Missing $(KERNEL_BUILD_DIR)/config.h (run upstream SWIFT ./configure first)" && exit 1)
	@for f in $(KERNEL_REQ_SRCS); do \
		test -f "$(KERNEL_BUILD_DIR)/src/$$f" || (echo "Missing $(KERNEL_BUILD_DIR)/src/$$f" && exit 1); \
	done

$(GEN_DIR)/.prepared: tree/tree_probe.cu tree/tree_format.h m2p/make-port.py m2p/m2p_port.h m2p/m2p_device.cuh src/make-evolution.py
	@$(MAKE) --no-print-directory check-kernels
	@mkdir -p $(GEN_DIR)/src
	@cp "$(KERNEL_BUILD_DIR)/config.h" "$(GEN_DIR)/config.h"
	@for f in $(KERNEL_REQ_SRCS); do \
		cp "$(KERNEL_BUILD_DIR)/src/$$f" "$(GEN_DIR)/src/$$f"; \
	done
	$(PYTHON) src/make-evolution.py $(GEN_DIR) .
	GRAV_M2P_MATH_ONLY=1 GRAV_GROUP_M2L=1 $(PYTHON) m2p/make-port.py $(GEN_DIR)
	@touch $@

$(BUILD_DIR)/multi_gpu_slab: $(GEN_DIR)/.prepared src/multi_gpu_slab.cu src/grouped_force.cuh src/grav_domain_3d.h src/grav_domain_3d_engine.cuh src/grav_cuda_pm.cu src/grav_cuda_pm.h src/grav_cuda_pm_dist.cu src/grav_cuda_pm_dist.h src/grav_cuda_fof.cu src/grav_cuda_fof.h
	$(NVCC) $(NVCC_COMMON) -DGRAV_FAST_M2P_EXP -Xcompiler -fopenmp \
		-I$(GEN_DIR) -I$(GEN_DIR)/src -I$(KERNEL_BUILD_DIR)/src -Isrc -I$(MATHLIB_INC) \
		-L$(MATHLIB_LIB) src/multi_gpu_slab.cu src/grav_cuda_pm.cu src/grav_cuda_pm_dist.cu src/grav_cuda_fof.cu \
		-lcufft -o $@

$(BUILD_DIR)/slab_resident: $(GEN_DIR)/.prepared src/slab_resident.cu src/grouped_force.cuh src/grav_cuda_pm.cu src/grav_cuda_pm.h
	$(NVCC) $(NVCC_COMMON) -DGRAV_FAST_M2P_EXP -Xcompiler -fopenmp \
		-I$(GEN_DIR) -I$(GEN_DIR)/src -I$(KERNEL_BUILD_DIR)/src -Isrc -I$(MATHLIB_INC) \
		-L$(MATHLIB_LIB) src/slab_resident.cu src/grav_cuda_pm.cu \
		-lcufft -o $@

engine: $(BUILD_DIR)/multi_gpu_slab $(BUILD_DIR)/slab_resident

$(BUILD_DIR)/grav_mpi: $(GEN_DIR)/.prepared src/multi_gpu_slab.cu src/grouped_force.cuh src/grav_domain_3d.h src/grav_domain_3d_engine.cuh src/grav_domain_3d_mpi.cuh src/grav_mpi_comm.h src/grav_cuda_pm.cu src/grav_cuda_pm.h src/grav_cuda_pm_dist.cu src/grav_cuda_pm_dist.h src/grav_cuda_pm_mpi.cu src/grav_cuda_pm_mpi.h src/grav_cuda_fof.cu src/grav_cuda_fof.h
	$(NVCC) -O3 -std=c++17 -arch=$(CUDA_ARCH) --fmad=true -ccbin=$(MPICXX) \
		-DGRAV_FAST_M2P_EXP -DGRAV_ENABLE_MPI -Xcompiler -fopenmp \
		-I$(GEN_DIR) -I$(GEN_DIR)/src -I$(KERNEL_BUILD_DIR)/src -Isrc -I$(MATHLIB_INC) \
		-L$(MATHLIB_LIB) src/multi_gpu_slab.cu src/grav_cuda_pm.cu src/grav_cuda_pm_dist.cu src/grav_cuda_pm_mpi.cu src/grav_cuda_fof.cu \
		-lcufft -o $@

mpi-engine: $(BUILD_DIR)/grav_mpi

smoke: engine fof test-domain3d
	$(PYTHON) scripts/make_smoke_ic.py $(BUILD_DIR)/smoke_64.bin
	(LD_LIBRARY_PATH="$(MATHLIB_LIB):$(CUDA_HOME)/lib64:$${LD_LIBRARY_PATH:-}" \
		GRAV_PM_GRID=64 GRAV_DYNAMIC_PM=1 GRAV_LEAF_LIMIT=64 GRAV_G=43.0092 \
		GRAV_WARP_M2L=1 GRAV_FAST_SPLIT=1 GRAV_SPLIT_FORCE=1 GRAV_PM_DIST=1 \
		GRAV_DOMAIN_MODE=auto GRAV_LAZY_TREE=1 GRAV_RUN_FOF=1 GRAV_FOF_MIN_SIZE=32 \
		./$(BUILD_DIR)/multi_gpu_slab $(BUILD_DIR)/smoke_64.bin $(BUILD_DIR)/smoke_64_out.bin 0.92 \
		> $(BUILD_DIR)/smoke_64.log 2>&1) || (cat $(BUILD_DIR)/smoke_64.log; exit 1)
	LD_LIBRARY_PATH="$(CUDA_HOME)/lib64:$(MATHLIB_LIB):$${LD_LIBRARY_PATH:-}" \
		$(PYTHON) scripts/verify_smoke_e2e.py $(BUILD_DIR)/smoke_64_out.bin $(BUILD_DIR)/smoke_64_out.bin.fof.bin $(BUILD_DIR)/smoke_64.log $(BUILD_DIR)/grav_cuda_fof.so

clean:
	rm -rf $(BUILD_DIR)
