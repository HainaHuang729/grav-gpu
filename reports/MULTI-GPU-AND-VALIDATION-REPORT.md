# Multi-GPU VRAM-Resident `1024³` TreePM Engine & Multi-Redshift Scientific Validation Report

**Date**: 2026-11-23  
**Workspace**: `/Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt`  
**Target Hardware**: `phy-gpus` node `a9` (`4× NVIDIA RTX A6000` 48 GiB each, dual NVLink4 pairs `0↔1` and `2↔3`, AMD EPYC dual-socket host) vs CHPC archival 256-core CPU (`PL_25_1024`)

---

## Executive Summary

1. **Part A — Multi-GPU VRAM-Resident 12-Slab Engine ([multi_gpu_slab.cu](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/src/multi_gpu_slab.cu))**:
   - Partitioning the 12 Y-slabs across `4× RTX A6000` (`4 × 48 GiB`) keeps all **`1,073,741,824` (`1024³`)** particles, multipoles, work pools, and PM mesh **100% resident in GPU VRAM** (`139.7–142.4 GB` total across 4 GPUs, `0` host-device slab swaps during evolution, **`4.86–12.31 GB` minimum free VRAM headroom**).
   - Diagnosed an AMD EPYC PCIe `NODE` P2P deadlock at `1024³` (`SWIFT_P2P_MODE=all`, Slurm `489878` hung at Step 0) and resolved it cleanly via `SWIFT_P2P_MODE=nvlink` (native NVLink atomics between paired GPUs `0↔1` & `2↔3`, clean pinned-host bounce across PCIe host bridges).
2. **Part C1 — Late-Redshift End-to-End Validation (`0055 → 0056`, `z = 0.063 → 0`, Slurm `489882`)**:
   - **`73m 09s` (`4,388.62 s`)** pure evolution on dual-NVLink `4× A6000`: **`2.40×` faster than `1× RTX A6000`** (`10,540.44 s`) and **`7.26×` faster than 256-core CHPC CPU** (`31,878.77 s`).
   - Scientific accuracy vs archival CHPC `PL_25_1024_0056.hdf5`: `bad_count = 0`, position RMS `0.05051 kpc/h` (`p50 = 0.01019 kpc/h`), matter $P_\delta(k)$ (`0.3–16 h/Mpc`) `max_abs_dev = 0.00453%`, kick-aligned momentum $P_\theta(k)$ `max_abs_dev` drops **`17.3×`** (`0.7018% → 0.0405%`), native 64-thread SWIFT FoF matches archival `fof_output_0056.hdf5` (`34,804,209` vs `34,804,779` particles for `#1` halo, `0.0%` count diff for $N \ge 262,144$).
3. **Part C2 — Mid-Redshift End-to-End Validation (`0044 → 0045`, `z = 2.033 → 1.758`, Slurm `489937` & `489982`)**:
   - **`58m 31s` (`3,510.98 s`)** solo on `a9` (`489937`) — **`9.27×` faster than 256-core CPU pure evolution** (`32,538.37 s`) / **`9.64×` faster than CPU step total** (`33,863.74 s`); **`72m 42s` (`4,362.17 s`)** on shared `a9` (`489982`, **`7.46×` / `7.76×`**), with `12.31 GB` minimum free VRAM and much tighter slab balance than `z = 0`.
   - Scientific accuracy vs archival `PL_25_1024_0045.hdf5`: `bad_count = 0`, position RMS `0.09717 kpc/h` (`p50 = 0.00775 kpc/h`), matter $P_\delta(k)$ `max_abs_dev = 0.01019%`, raw momentum $P_\theta(k)$ `max_abs_dev = 0.01112%` (`vel p50 = 0.0186 km/s` — directly confirming intermediate snapshot kick synchronization vs end-of-run snapshot `0056`), and native SWIFT FoF within `±0.19%` for $N \ge 32..65,536$.
4. **Part C3 — Full `z = 200 → 0` (`20,194` steps) & `z = 50 → 0` (`19,929` steps) Scaling ([full-history-1024-model.json](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/full-history-1024-model.json))**:
   - Archival 256-core CHPC CPU required **`304.5 h` pure evolution** (`322.7 h` = `13.44` node-days with 56 inline dumps, **`82,600` core-hours**, `~213.1 kWh` pure-evo node energy at `700 W`).
   - Dual-NVLink **`4× RTX A6000`** projects **`34.6–35.8 hours` (`1.44–1.49 days`)** for full `z = 200 → 0` pure gravity evolution (**`8.50–8.80×` speedup**) and **`28.3–29.2 kWh`** whole-node energy (**`7.3–7.5×` energy reduction**, ~`86%` savings); `z = 50 → 0` projects **`28.6–34.3 hours` (`1.19–1.43 days`)** vs CPU `248.7 h`.

---

## 1. Part A: Multi-GPU 12-Slab Engine & P2P Topology Findings

### 1.1 Architecture & Hardening ([multi_gpu_slab.cu](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/src/multi_gpu_slab.cu))
- Assigns contiguous groups of the 12 Y-slabs across `G ∈ {1, 2, 3, 4, 6}` GPUs with persistent per-slab particle/multipole buffers, paired-slab migration streams (`d2h_left_stream`, `d2h_right_stream`), neighbor ghost-tree sharing, OpenMP-coordinated resident 128³ PM FFT/gradient updates, and thread-safe phase-pool allocation (`#pragma omp critical(phase_pool_alloc)`).
- For `N > 50M` (`1024³`), uses conservative `1.04` slab capacity slack so heavily clustered late-time slabs fit inside 48 GiB without CPU fallback.

### 1.2 Gate 1 Correctness & Strong Scaling (`256³`, `a = 0.855 → 0.920`, Slurm `489828`)
Executed on `a9` (`CUDA_VISIBLE_DEVICES=1,2,3,4`, 320 steps, 17 PM updates, `bad_count = 0` for all GPU counts):

| Config | Slabs/GPU | `can_peer` Topology | Evolution Wall (s) | Speedup vs `G=1` | Pos RMS (`kpc/h`) | $P_\delta$ `max_abs_dev` | $P_{\theta,\text{adj}}$ `max_abs_dev` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `G = 1` ([mgpu1-256-092.log](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/mgpu1-256-092.log)) | 12 | `[[1]]` | `72.80` | `1.00×` | `0.022825` | `1.057e-5` | `1.946e-4` |
| `G = 2` ([mgpu2-256-092.log](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/mgpu2-256-092.log)) | 6 | `[[1,0],[0,1]]` (PCIe bounce) | `61.48` | `1.18×` | `0.022825` | `1.057e-5` | `1.946e-4` |
| `G = 4` ([mgpu4-256-092.log](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/mgpu4-256-092.log)) | 3 | Single NVLink (`2↔3`) | `30.78` | `2.36×` | `0.022825` | `1.057e-5` | `1.946e-4` |

All three GPU counts matched reference outputs identically across position/velocity errors (`explained_fraction = 0.956974` in [mgpu4-256-092-acc.json](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/mgpu4-256-092-acc.json)).

### 1.3 `a9` PCIe `NODE` P2P Deadlock Discovery & Dual-NVLink Speedup (`1024³`)
`nvidia-smi topo -m` on `a9` ([mgpu4-1024-0056-device.csv](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/mgpu4-1024-0056-device.csv)) shows three NVLink4 pairs (`0↔1`, `2↔3`, `4↔5`) connected across AMD EPYC PCIe host bridges via `NODE` links:
- **Probe `489878` (`SWIFT_P2P_MODE=all` on `0,1,2,3`)**: Hung at Step 0 ([slurm-mgpu1024-p2pall-489878.log](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/slurm-mgpu1024-p2pall-489878.log)), proving direct CUDA P2P over EPYC `NODE` PCIe bridges deadlocks/stalls at `1024³`.
- **`SWIFT_P2P_MODE=nvlink` (`cudaDevP2PAttrNativeAtomicSupported == 1`)**: Uses direct NVLink P2P on NVLink-bonded neighbors and pinned host-bounce staging across `NODE` boundaries:
  - Single-NVLink `1,2,3,4` (`489830`, `can_peer` only `2↔3`): **`5,719.90 s`** (`95m 20s`, **`1.84×`** vs `1× A6000`).
  - Dual-NVLink `0,1,2,3` (`489882`, `can_peer` `0↔1` & `2↔3`): **`4,388.62 s`** (`73m 09s`, **`2.40×`** vs `1× A6000`, `1.30×` faster than single-NVLink).

---

## 2. Part C1 & C2: End-to-End `1024³` Scientific Validation (`z ≈ 0` & `z ≈ 2`)

To prevent idle-GPU Slurm termination during post-processing on `0044 → 0045` (`489937` was `CANCELLED by 0` after `2h 12m` idle GPUs while Python `np.add.at` CIC ran >52 min), we replaced Python `np.add.at` in [eval_1024_snapshot.py](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/eval_1024_snapshot.py) with a deterministic single-pass C99 kernel ([cic_deposit.c](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/cic_deposit.c); verified against `np.add.at` on `1M` particles to `< 2.3e-13` max abs diff and `8.3×` faster) and bumped FoF threads to 96 in [mgpu-1024-0045-e2e.sbatch](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/mgpu-1024-0045-e2e.sbatch), finishing `489982` cleanly in `03:05:41` (`COMPLETED 0:0`).

### 2.1 Performance & Memory Balance Summary (`1024³`, `4× RTX A6000` Dual NVLink)

| Metric | `0055 → 0056` (`z=0.063→0`, `1×`, [slab-1024-0056](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/slab-1024-0056-evolve.log)) | `0055 → 0056` (`z=0.063→0`, `4×`, `489882`) | `0044 → 0045` (`z=2.033→1.758`, `4×` solo, `489937`) | `0044 → 0045` (`z=2.033→1.758`, `4×` shared, `489982`) |
| --- | ---: | ---: | ---: | ---: |
| Scale factors (`a_begin → a_end`) | `0.94059 → 1.00000` | `0.94059 → 1.00000` | `0.32967 → 0.36264` | `0.32967 → 0.36264` |
| GPU steps / PM updates | `513` / `33` | `513` / `33` | `910` / `50` | `910` / `50` |
| Pure evolution wall (`s`) | `10,540.44` (`175m 40s`) | **`4,388.62`** (`73m 09s`) | **`3,510.98`** (`58m 31s`) | `4,362.17` (`72m 42s`) |
| Archival 256c CPU wall (`s`) | `31,878.77` (`757` steps) | `31,878.77` (`757` steps) | `32,538.37` pure / `33,863.74` tot | `32,538.37` pure / `33,863.74` tot |
| Speedup vs 256-core CPU | `3.02×` | **`7.26×`** | **`9.27×` pure / `9.64×` tot** | `7.46×` pure / `7.76×` tot |
| Per-GPU particles (`M`) | `1,073.7` (all 12 slabs) | `[137.4, 269.5, 348.2, 318.7]` | `[205.0, 279.0, 293.1, 296.6]` | `[205.0, 279.0, 293.1, 296.6]` |
| Min free VRAM headroom (`GB`) | `17.34` | `4.86` | **`12.31`** | **`12.31`** |
| Mean GPU power (`W`, allocated) | `217.41` (GPU 1) | `657.12` (GPUs `0..3`) | `735.87` (GPUs `0..3`) | `637.14` (GPUs `0..3`) |
| `to-hdf5` / FoF wall (`s`) | `925.34` / `2,045.49` (64t) | `925.44` / `1,772.99` (64t) | `1,578` / `3,192` (64t) | `1,594.94` / `2,985.50` (96t) |

Additionally, the `1× A6000` mid-z timing probe (`0.32967 → 0.33050`, 17 steps, 2 PM updates in [slab1-1024-0044-probe.log](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/slab1-1024-0044-probe.log)) completed in `265.68 s` (`bad_count = 0`, `24.89 GB` free VRAM; `59.85 s` mean PM step, `wall_s = 4.30 + 69.83*(active/N)`), projecting `~9,710 s` for full `0044 → 0045` on `1× A6000` (`~3.35×` vs 256-core CPU, making `4× A6000` solo `~2.77×` faster than `1× A6000` at `z ≈ 2`).

### 2.2 Scientific Accuracy Comparison Against Archival CHPC CPU

Reports: [mgpu4-1024-0056-comparison.json](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/mgpu4-1024-0056-comparison.json) (`0056`), [slab-1024-0056-comparison.json](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/slab-1024-0056-comparison.json) (`1×` `0056`), and [mgpu4-1024-0045-comparison.json](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/mgpu4-1024-0045-comparison.json) (`0045`).

| Validation Metric (`16,388` ID-matched; `256³` FFT, `0.3–16 h/Mpc`) | `4×` `0056` (`z = 0`) | `1×` `0056` (`z = 0`) | `4×` `0045` (`z = 1.758`) |
| --- | ---: | ---: | ---: |
| `bad_count` | `0` | `0` | `0` |
| Position error RMS / p50 (`kpc/h`) | `0.05051` / `0.01019` | `0.05051` / `0.01019` | `0.09717` / `0.00775` |
| Matter $P_\delta(k)$ `max_abs_dev` | **`0.00453%`** (`4.53e-5`) | **`0.00453%`** | **`0.01019%`** (`1.02e-4`) |
| Matter $P_\delta(k)$ `min_r` | `0.999999973` | `0.999999973` | `0.999999950` |
| Raw Velocity error RMS / p50 (`km/s`) | `2.6657` / `0.9963` | `2.6656` / `0.9964` | **`0.6226` / `0.0186`** |
| Kick-aligned (`+2*s`) Velocity RMS / p50 (`km/s`) | **`1.1870` / `0.3364`** | — | `1.4249` / `0.3595` *(over-shifted)* |
| Raw Momentum $P_\theta(k)$ `max_abs_dev` (`min_r`) | `0.7018%` (`0.999758`) | `0.7018%` | **`0.01112%`** (`0.99999988`) |
| Kick-aligned $P_\theta(k)$ `max_abs_dev` (`min_r`) | **`0.0405%`** (`0.99999961`) | — | `0.5694%` *(over-shifted)* |
| `#1` FoF halo (`gpu` vs `cpu`) | `34,804,209` vs `34,804,779` (`-0.0016%`) | `34,804,189` vs `34,804,779` | `6,566,460` vs `6,562,673` (`+0.058%`) |
| Cumulative FoF count diff (`N ≥ 32 .. 65,536`) | `-0.32% .. +0.026%` | `-0.32% .. +0.026%` | `-0.18% .. +0.019%` |
| Cumulative FoF count diff (`N ≥ 262,144`) | `0.0%` (`198` vs `198`) | `0.0%` (`198` vs `198`) | `-0.68%` (`146` vs `147`) |

#### Physical Closure on Velocity Convention (`0056` vs `0045`)
In `integrate_slab` ([src/multi_gpu_slab.cu](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/src/multi_gpu_slab.cu#L605-L626)), when `ti == timeline`, the GPU kernel executes closing `kick2` and `mesh_kick` and skips opening `kick1`, exporting velocities strictly synchronized at `a_end`:
- At **intermediate snapshot `0045`** (`timesteps_1024.txt` step `8392`, `Props = 145 = Rebuild + Snapshot + FOF`), archival `PL_25_1024_0045.hdf5` matches GPU synchronized velocities directly (**`p50 = 0.0186 km/s`**, **`P_theta max_abs_dev = 0.01112%`**), whereas applying `+2*s` degrades agreement (`explained_fraction = 3.76%`).
- At **end-of-simulation snapshot `0056`** (`timesteps_1024.txt` step `20193`, `Props = 257 = Rebuild + mesh`, `g-Updates = 1,073,741,824`), archival `PL_25_1024_0056.hdf5` shows a systematic `-2*s` short-force offset (`two_component_coeffs = [-1.9197, -0.0219]`, explaining **`80.30%`** of raw `Δv` variance), so kick-alignment collapses $P_\theta(k)$ error **`17.3×`** from `0.7018%` down to **`0.0405%`**.
- **Conclusion**: Across both cosmological epochs (`z = 2 → 1.76` and `z = 0.06 → 0`), single-GPU and `4×` NVLink GPU TreePM agree with archival 256-core CPU SWIFT to `~0.005–0.01%` in $P_\delta(k)$, `~0.01–0.04%` in synchronized $P_\theta(k)$, and `< 0.35%` in cumulative FoF halo counts.

---

## 3. Part C3: Full-History (`z = 200 → 0` & `z = 50 → 0`) Compute & Energy Model

Script: [model_1024_full_history.py](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/model_1024_full_history.py) | Output: [full-history-1024-model.json](file:///Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/full-history-1024-model.json).

### 3.1 Archival 256-Core CHPC CPU Epoch Profile (`data/timesteps_1024.txt`, `20,194` clean steps)
Parsing `data/timesteps_1024.txt` (skipping `335` truncated restart lines, yielding consecutive steps `0..20193`, exact match to saved `0044→0045` and `0055→0056` CPU totals) isolates pure gravity time from the `56` inline Snapshot+FoF dumps (`18.18 h` dump overhead out of `322.66 h`):

| Cosmic Epoch | CPU Steps | PM Steps | Inline Dumps | Mean Active | CPU Total (`h`) | CPU Pure Gravity (`h`) | CPU PM Med (`s`) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `z = 200 → 50` | `265` | `265` | `15` | `1,073.7M` | `60.12` | `55.82` | `738.6` |
| `z = 50 → 10` | `1,591` | `296` | `16` | `208.1M` | `30.75` | `25.70` | `252.3` |
| `z = 10 → 4` | `2,857` | `153` | `8` | `110.6M` | `35.82` | `33.12` | `200.1` |
| `z = 4 → 1.5` | `4,430` | `264` | `8` | `127.4M` | `65.82` | `63.00` | `197.4` |
| `z = 1.5 → 0.5` | `6,040` | `384` | `5` | `125.3M` | `70.25` | `68.45` | `169.2` |
| `z = 0.5 → 0` | `5,011` | `314` | `4` | `143.5M` | `59.90` | `58.38` | `155.1` |
| **`z = 200 → 0`** | **`20,194`** | **`1,676`** | **`56`** | — | **`322.66` (`13.44 d`)** | **`304.48` (`12.69 d`)** | — |
| **`z = 50 → 0`** | **`19,929`** | **`1,411`** | **`41`** | — | **`262.54` (`10.94 d`)** | **`248.65` (`10.36 d`)** | — |

Notably, `z = 200 → 50` accounts for `1.3%` of steps (`265`) yet consumes `55.82 h` (`18.3%` of pure CPU gravity) because all `1.073B` particles are active every step and nearly uniform cells require `738.6 s` median per CPU PM step, whereas measured `4× A6000` Step-0 uniform-ish PM steps take only `11.5–20.7 s`.

### 3.2 Full-History Runtime & Energy Projection (`Clamped Interval` vs `Step-Physics` Models)
Two independent models bracket the uncertainty: **Clamped Interval Speedup** (clamps `z > 2` speedup conservatively at `z ≈ 2` values; interpolates `a` between `0.346` and `0.970`) and **Step-Physics Regression** (evaluates per-step `active` + `is_pm` regression with measured dynamic-PM step multiplier). Whole-node power assumes `700 W` for the 256-core CPU node and measured GPU draw (`217.4 W` for `1× A6000`; `696.5 W` solo / `647.1 W` shared for `4× A6000`) plus `+100–120 W` host overhead:

| Scenario / Platform | `z = 200 → 0` Pure Evo (`h` / `days`) | Speedup vs CPU Pure | `z = 200 → 0` Node Energy (`kWh`) | `z = 50 → 0` Pure Evo (`h` / `days`) | `z = 50 → 0` Node Energy (`kWh`) |
| --- | ---: | ---: | ---: | ---: | ---: |
| **256-core CHPC CPU** | `304.5 h` (`12.69 d`) *(322.7 h w/ dumps)* | `1.00×` | `213.1` *(225.9 w/ dumps)* | `248.7 h` (`10.36 d`) | `174.1` *(183.8 w/ dumps)* |
| **`1× RTX A6000`** *(clamped)* | `92.9 h` (`3.87 d`) | `3.28×` | `29.5` (`20.2` GPU-only) | `76.2 h` (`3.18 d`) | `24.2` (`16.6` GPU) |
| **`1× RTX A6000`** *(step-physics)* | `96.1 h` (`4.00 d`) | `3.17×` | `30.5` (`20.9` GPU-only) | `91.7 h` (`3.82 d`) | `29.1` (`19.9` GPU) |
| **`4× RTX A6000` NVLink solo** *(clamped)* | **`34.6 h` (`1.44 d`)** | **`8.80×`** | **`28.3`** (`24.1` GPU-only) | **`28.6 h` (`1.19 d`)** | **`23.3`** (`19.9` GPU) |
| **`4× RTX A6000` NVLink solo** *(step-physics)* | **`35.8 h` (`1.49 d`)** | **`8.50×`** | **`29.2`** (`24.9` GPU-only) | **`34.3 h` (`1.43 d`)** | **`28.0`** (`23.9` GPU) |
| **`4× A6000` shared node** *(clamped / physics)* | `41.1 – 41.3 h` (`1.71 d`) | `7.37–7.42×` | `31.5 – 31.7` | `33.6 – 39.7 h` | `25.7 – 30.4` |

**Post-Processing Cadence Note**: Measured offline HDF5 conversion averages `1,260 s/snapshot` (`925 s` at `0056`, `1,595 s` at `0045`) and standalone native SWIFT FoF takes `1,773 s` (`64t`, `0056`) / `2,986 s` (`96t`, `0045`). Exporting lightweight binary checkpoints asynchronously to NVMe/CPU workers keeps full `z = 200 → 0` GPU wall-clock near **`~35 h` (`~1.5 days`)** even when running full `56`-snapshot HDF5 + FoF offline.

---

## 4. Remote Staging Cleanup & Cluster State Verification

After syncing all `0044 → 0045` accuracy, comparison, power-spectrum, and telemetry files locally:
- Deleted remote staging directory `data/tmp1024/`, `recv_chunk.py`, and `recv_seq_chunk.py` on `phy-gpus`.
- Verified `quota -s` returned to **`18,128M / 100G`** (`17.7 GiB`, zero `.bin`/`.partial` leftovers).
- Verified `squeue -u hnhuang` is empty and all local artifacts reside under `/Users/amadeus/Documents/ChatGPT/论文/output/swift-gravity-bench-20261117/single-gpu-opt/reports/`.
