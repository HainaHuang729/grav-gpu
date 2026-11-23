# `swift-treepm-gpu`: Multi-GPU VRAM-Resident 12-Slab CUDA TreePM Engine for Billion-Particle SWIFT Cosmological Simulations

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

`swift-treepm-gpu` is a multi-GPU CUDA TreePM gravity solver (`multi_gpu_slab.cu` / `slab_resident.cu`) engineered to evolve **$1024^3$ (`1,073,741,824`)** self-gravitating cosmological dark-matter particles **100% resident inside GPU VRAM** (`4× NVIDIA RTX A6000`, `4 × 48 GiB`) without host-device particle swapping, while preserving publication-grade accuracy against archival 256-core CPU SWIFT runs.

---

## Key Highlights

- **Zero-Swap Multi-GPU 12-Slab Resident Design (`src/multi_gpu_slab.cu`)**:
  Partitions the 12 Y-slabs across $G \in \{1, 2, 3, 4, 6\}$ GPUs (`139.7–142.4 GB` total footprint at $1024^3$, leaving `4.86 GB` minimum headroom at $z=0$ and `12.31 GB` at $z=2$).
- **AMD EPYC PCIe `NODE` P2P Deadlock Diagnosis (`SWIFT_P2P_MODE=nvlink`)**:
  Proved that cross-bridge `NODE` PCIe CUDA P2P (`SWIFT_P2P_MODE=all`) stalls at $1024^3$ Step 0 on EPYC multi-root nodes, whereas `SWIFT_P2P_MODE=nvlink` uses hardware NVLink4 atomics across bonded GPU pairs (`0↔1`, `2↔3`) and pinned host-bounce staging across PCIe bridges—delivering `2.40×` multi-GPU scaling over `1× RTX A6000` (`1.30×` over single-NVLink topology).
- **Verified $1024^3$ Speedups vs. 256-Core CHPC CPU SWIFT**:
  - **Late-time ($0055 \to 0056$, $z = 0.063 \to 0$)**: `4,388.62 s` (`73m 09s`) on dual-NVLink `4× RTX A6000` — **`2.40×` vs. `1× A6000`** (`10,540.44 s`) and **`7.26×` vs. 256-core CPU** (`31,878.77 s`).
  - **Mid-redshift ($0044 \to 0045$, $z = 2.033 \to 1.758$)**: `3,510.98 s` (`58m 31s`, idle node) / `4,362.17 s` (shared node) on dual-NVLink `4× A6000` — **`9.27×` vs. 256-core CPU pure evolution** (`9.64×` incl. dump step).
- **Publication-Grade Scientific Fidelity**:
  - `bad_count = 0` across all $256^3$ ($G = 1, 2, 4$) and $1024^3$ runs.
  - Matter power spectrum $P_\delta(k)$ ($k \in [0.3, 16]\,h\,\mathrm{Mpc}^{-1}$, $256^3$ FFT): **`0.00453%`** max deviation at $z=0$ and **`0.01019%`** at $z=1.758$ (`min_r >= 0.99999995`).
  - Velocity & momentum divergence $P_\theta(k)$: raw $z=1.758$ velocity median error **`0.0186 km/s`** ($P_\theta$ max dev **`0.01112%`**); end-of-run $z=0$ kick-aligned (`+2*s`) collapses $P_\theta$ error `17.3×` (`0.7018% → 0.0405%`).
  - Standalone native SWIFT Friends-of-Friends (`fof`): cumulative halo counts ($N \ge 32 \dots 65,536$) within `±0.32%` ($z=0$) and `±0.19%` ($z=1.758$); exact `198` vs. `198` (`0.0%`) match for $N \ge 262,144$ halos at $z=0$.
- **Full-History ($z = 200 \to 0$) Energy & Runtime Projection**:
  Calibrated against `20,194` archival CPU steps (`304.48 h` pure gravity, `322.66 h` total, `~82,600` core-h): `4× RTX A6000` projects **`34.6–35.8 h` (`1.44–1.49 days`)** pure evolution (`8.50–8.80×` speedup) and **`28.3–29.2 kWh`** whole-node energy (`~86%` savings vs. `213.1 kWh` on a `700 W` 256-core CPU node).

---

## Repository Structure

```text
swift-treepm-gpu/
├── src/                      # Multi-GPU & single-GPU CUDA slab TreePM kernels + PM solver + C CIC helper
│   ├── multi_gpu_slab.cu     # 12-slab VRAM-resident multi-GPU driver (NVLink + pinned bounce P2P)
│   ├── slab_resident.cu      # Single-GPU 12-slab VRAM-resident baseline driver
│   ├── swift_cuda_pm.cu/h    # Resident 128^3 cuFFT particle-mesh long-range gravity kernels
│   ├── make-evolution.py     # Instrumented GPU tree/FMM source builder
│   └── cic_deposit.c         # Deterministic single-pass C99 CIC grid deposition helper
├── scripts/                  # Snapshot packing, HDF5 export, power/FoF evaluation, full-history scaling
│   ├── prepare_1024_input.py
│   ├── to-hdf5-from-source.py
│   ├── eval_1024_snapshot.py
│   ├── model_1024_full_history.py
│   └── make_figures.py       # Reproducible publication figure generator
├── slurm/                    # Reference Slurm job templates for Gate 1 (256^3) and 1024^3 benchmarks
├── reports/                  # Verified JSON/CSV power spectra, FoF, timing & full-history model data
└── paper/                    # Physical Review D (revtex4-2) manuscript, BibTeX & vector PDF figures
    ├── main.tex
    ├── references.bib
    └── figures/
```

---

## Citation & Authorship

Author: **Haina Huang** (Department of Physics, The Chinese University of Hong Kong).  
See [paper/main.tex](paper/main.tex) and [reports/MULTI-GPU-AND-VALIDATION-REPORT.md](reports/MULTI-GPU-AND-VALIDATION-REPORT.md) for full derivations, kernel specifications, and benchmark provenance.
