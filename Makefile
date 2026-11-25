# Build & Smoke-Test Makefile for swift-treepm-gpu

_DETECTED_NVCC := $(shell command -v nvcc 2>/dev/null)
CUDA_HOME ?= $(if $(_DETECTED_NVCC),$(patsubst %/bin/nvcc,%,$(_DETECTED_NVCC)),/usr/local/cuda)
NVCC ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= sm_86
CXX ?= g++
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

SWIFT_BUILD_DIR ?=
BUILD_DIR := build
GEN_DIR := $(BUILD_DIR)/gen

SWIFT_REQ_SRCS := \
	runner_gpu.h runner_gpu.c gpu_gravity_offload.cu gpu_gravity_functions.cuh \
	kernel_gravity.h kernel_long_gravity.h gravity_derivatives.h multipole.h

NVCC_COMMON := -O3 -std=c++17 -arch=$(CUDA_ARCH) --fmad=true -ccbin=$(CXX)

.PHONY: help all standalone fof pm cic engine test-domain3d test-fof smoke clean check-swift

help:
	@echo "Targets:"
	@echo "  make standalone                       Build standalone libs (fof, cic) — no SWIFT headers needed"
	@echo "  make fof                              Build $(BUILD_DIR)/swift_cuda_fof.so (CUDA FoF/Catalog/SO library)"
	@echo "  make cic                              Build $(BUILD_DIR)/libcic_deposit.so (C99 CIC helper)"
	@echo "  make test-domain3d                    Compile & run CPU unit test for 3D Hilbert/ORB domain planner"
	@echo "  make test-fof                         Build fof and run GPU ctypes FoF + Catalog + SO smoke test"
	@echo "  make pm SWIFT_BUILD_DIR=<dir>         Build $(BUILD_DIR)/swift_cuda_pm*.o (PM FFT objects)"
	@echo "  make engine SWIFT_BUILD_DIR=<dir>     Build multi_gpu_slab and slab_resident using upstream SWIFT headers"
	@echo "  make smoke SWIFT_BUILD_DIR=<dir>      Run full synthetic 64^3 multi-GPU evolution + FoF + SO smoke test"
	@echo "  make clean                            Remove $(BUILD_DIR)/"
	@echo ""
	@echo "Overrides: CUDA_HOME=$(CUDA_HOME) CUDA_ARCH=$(CUDA_ARCH) CXX=$(CXX) PYTHON=$(PYTHON)"

all: standalone

standalone: fof cic

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

fof: $(BUILD_DIR)/swift_cuda_fof.so

$(BUILD_DIR)/swift_cuda_fof.so: src/swift_cuda_fof.cu src/swift_cuda_fof.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_COMMON) -Xcompiler "-fPIC -fopenmp" -shared -Isrc src/swift_cuda_fof.cu -o $@

pm: check-swift $(BUILD_DIR)/swift_cuda_pm.o $(BUILD_DIR)/swift_cuda_pm_dist.o

$(BUILD_DIR)/swift_cuda_pm.o: src/swift_cuda_pm.cu src/swift_cuda_pm.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_COMMON) -Xcompiler -fPIC -c -Isrc -I$(SWIFT_BUILD_DIR) -I$(SWIFT_BUILD_DIR)/src -I$(MATHLIB_INC) src/swift_cuda_pm.cu -o $@

$(BUILD_DIR)/swift_cuda_pm_dist.o: src/swift_cuda_pm_dist.cu src/swift_cuda_pm_dist.h | $(BUILD_DIR)
	$(NVCC) $(NVCC_COMMON) -Xcompiler -fPIC -c -Isrc -I$(SWIFT_BUILD_DIR) -I$(SWIFT_BUILD_DIR)/src -I$(MATHLIB_INC) src/swift_cuda_pm_dist.cu -o $@

cic: $(BUILD_DIR)/libcic_deposit.so

$(BUILD_DIR)/libcic_deposit.so: src/cic_deposit.c | $(BUILD_DIR)
	$(CC) -O3 -shared -fPIC src/cic_deposit.c -lm -o $@

test-domain3d: $(BUILD_DIR)/test_domain3d
	./$(BUILD_DIR)/test_domain3d

$(BUILD_DIR)/test_domain3d: scripts/test_domain3d_smoke.cpp src/swift_domain_3d.h | $(BUILD_DIR)
	$(CXX) -O2 -std=c++17 -Wall -Wextra -Werror -Isrc scripts/test_domain3d_smoke.cpp -o $@

test-fof: fof
	LD_LIBRARY_PATH="$(CUDA_HOME)/lib64:$(MATHLIB_LIB):$${LD_LIBRARY_PATH:-}" \
		$(PYTHON) scripts/smoke_fof.py $(BUILD_DIR)/swift_cuda_fof.so

check-swift:
	@if [ -z "$(SWIFT_BUILD_DIR)" ]; then \
		echo "ERROR: SWIFT_BUILD_DIR is not set."; \
		echo "Point it to an upstream GPU-configured SWIFT source/build tree containing config.h and src/multipole.h, e.g.:"; \
		echo "  make engine SWIFT_BUILD_DIR=/path/to/swiftsim"; \
		exit 1; \
	fi
	@test -f "$(SWIFT_BUILD_DIR)/config.h" || (echo "Missing $(SWIFT_BUILD_DIR)/config.h (run upstream SWIFT ./configure first)" && exit 1)
	@for f in $(SWIFT_REQ_SRCS); do \
		test -f "$(SWIFT_BUILD_DIR)/src/$$f" || (echo "Missing $(SWIFT_BUILD_DIR)/src/$$f" && exit 1); \
	done

$(GEN_DIR)/.prepared: tree/tree_probe.cu tree/tree_format.h m2p/make-port.py m2p/m2p_port.h m2p/m2p_device.cuh src/make-evolution.py
	@$(MAKE) --no-print-directory check-swift
	@mkdir -p $(GEN_DIR)/src
	@cp "$(SWIFT_BUILD_DIR)/config.h" "$(GEN_DIR)/config.h"
	@for f in $(SWIFT_REQ_SRCS); do \
		cp "$(SWIFT_BUILD_DIR)/src/$$f" "$(GEN_DIR)/src/$$f"; \
	done
	$(PYTHON) src/make-evolution.py $(GEN_DIR) .
	SWIFT_M2P_MATH_ONLY=1 SWIFT_GROUP_M2L=1 $(PYTHON) m2p/make-port.py $(GEN_DIR)
	@touch $@

$(BUILD_DIR)/multi_gpu_slab: $(GEN_DIR)/.prepared src/multi_gpu_slab.cu src/grouped_force.cuh src/swift_domain_3d.h src/swift_domain_3d_engine.cuh src/swift_cuda_pm.cu src/swift_cuda_pm.h src/swift_cuda_pm_dist.cu src/swift_cuda_pm_dist.h src/swift_cuda_fof.cu src/swift_cuda_fof.h
	$(NVCC) $(NVCC_COMMON) -DSWIFT_FAST_M2P_EXP -Xcompiler -fopenmp \
		-I$(GEN_DIR) -I$(GEN_DIR)/src -I$(SWIFT_BUILD_DIR)/src -Isrc -I$(MATHLIB_INC) \
		-L$(MATHLIB_LIB) src/multi_gpu_slab.cu src/swift_cuda_pm.cu src/swift_cuda_pm_dist.cu src/swift_cuda_fof.cu \
		-lcufft -o $@

$(BUILD_DIR)/slab_resident: $(GEN_DIR)/.prepared src/slab_resident.cu src/grouped_force.cuh src/swift_cuda_pm.cu src/swift_cuda_pm.h
	$(NVCC) $(NVCC_COMMON) -DSWIFT_FAST_M2P_EXP -Xcompiler -fopenmp \
		-I$(GEN_DIR) -I$(GEN_DIR)/src -I$(SWIFT_BUILD_DIR)/src -Isrc -I$(MATHLIB_INC) \
		-L$(MATHLIB_LIB) src/slab_resident.cu src/swift_cuda_pm.cu \
		-lcufft -o $@

engine: $(BUILD_DIR)/multi_gpu_slab $(BUILD_DIR)/slab_resident

smoke: engine fof test-domain3d
	$(PYTHON) scripts/make_smoke_ic.py $(BUILD_DIR)/smoke_64.bin
	(LD_LIBRARY_PATH="$(MATHLIB_LIB):$(CUDA_HOME)/lib64:$${LD_LIBRARY_PATH:-}" \
		SWIFT_PM_GRID=64 SWIFT_DYNAMIC_PM=1 SWIFT_LEAF_LIMIT=64 SWIFT_GRAV_G=43.0092 \
		SWIFT_WARP_M2L=1 SWIFT_FAST_SPLIT=1 SWIFT_SPLIT_FORCE=1 SWIFT_PM_DIST=1 \
		SWIFT_DOMAIN_MODE=auto SWIFT_LAZY_TREE=1 SWIFT_RUN_FOF=1 SWIFT_FOF_MIN_SIZE=32 \
		./$(BUILD_DIR)/multi_gpu_slab $(BUILD_DIR)/smoke_64.bin $(BUILD_DIR)/smoke_64_out.bin 0.92 \
		> $(BUILD_DIR)/smoke_64.log 2>&1) || (cat $(BUILD_DIR)/smoke_64.log; exit 1)
	LD_LIBRARY_PATH="$(CUDA_HOME)/lib64:$(MATHLIB_LIB):$${LD_LIBRARY_PATH:-}" \
		$(PYTHON) scripts/verify_smoke_e2e.py $(BUILD_DIR)/smoke_64_out.bin $(BUILD_DIR)/smoke_64_out.bin.fof.bin $(BUILD_DIR)/smoke_64.log $(BUILD_DIR)/swift_cuda_fof.so

clean:
	rm -rf $(BUILD_DIR)
