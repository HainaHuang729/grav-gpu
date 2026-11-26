"""Generate publication-quality vector PDF figures for the SWIFT-TreePM-GPU manuscript."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
OUT_DIR = ROOT_DIR / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8.4,
    "axes.titlesize": 9.2,
    "axes.labelsize": 8.4,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#222222",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
SLATE = "#4D4D4D"
RED = "#C53030"

CPU_FILL = "#EAF2F8"
CPU_EDGE = "#2C6B9E"
GPU_FILL = "#FDF2E9"
GPU_EDGE = "#C05621"
NVL_FILL = "#EDF7F1"
NVL_EDGE = "#1F7A5A"


def _box(ax, x, y, w, h, title, body, fc, ec, title_fs=7.4, body_fs=6.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        facecolor=fc, edgecolor=ec, linewidth=1.05, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h - 0.018, title,
        ha="center", va="top", fontsize=title_fs, fontweight="bold",
        color="#111111", zorder=3,
    )
    ax.text(
        x + w / 2, y + (h - 0.050) / 2, body,
        ha="center", va="center", fontsize=body_fs, color="#222222",
        linespacing=1.28, zorder=3,
    )


def _arrow(ax, x0, y0, x1, y1, color="#333333", lw=1.05, ls="-", rad=0.0, both=False):
    style = "<|-|>,head_width=2.6,head_length=3.8" if both else "-|>,head_width=2.6,head_length=3.8"
    arr = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style,
        color=color, linewidth=lw, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}", zorder=4,
    )
    ax.add_patch(arr)


def make_fig1_architecture():
    fig, ax = plt.subplots(figsize=(9.2, 4.15))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    # Top host lane
    host_band = FancyBboxPatch(
        (0.012, 0.72), 0.976, 0.25,
        boxstyle="round,pad=0.006,rounding_size=0.014",
        facecolor="#F7FAFC", edgecolor="#CBD5E1", linewidth=0.9, linestyle="--", zorder=0,
    )
    ax.add_patch(host_band)
    ax.text(0.022, 0.955, "AMD EPYC Host Orchestration (OpenMP 4-Thread Control + Pinned PCIe Staging)",
            fontsize=8.0, fontweight="bold", color=CPU_EDGE, va="top", zorder=2)

    _box(ax, 0.024, 0.74, 0.21, 0.16,
         "128^3 PM Grid Reduce",
         "Sum partial GPU densities\n3D Poisson solve + grads",
         CPU_FILL, CPU_EDGE)
    _box(ax, 0.268, 0.74, 0.46, 0.16,
         "Pinned-Host Cross-Bridge Staging (PCIe NODE Boundary)",
         "GRAV_P2P_MODE=nvlink: host-bounce ghost trees & migrants\nbetween GPU 1 (Slabs 3-5) <-> GPU 2 (Slabs 6-8) cleanly without deadlock",
         "#FFFBEB", "#B7791F")
    _box(ax, 0.762, 0.74, 0.214, 0.16,
         "Snapshot & FoF Pipeline",
         "D2H export at a_end -> HDF5\nNative SWIFT FoF (64-96T)",
         NVL_FILL, NVL_EDGE)

    # Middle multi-GPU resident islands (2 NVLink islands)
    isl_left = FancyBboxPatch(
        (0.012, 0.275), 0.435, 0.39,
        boxstyle="round,pad=0.006,rounding_size=0.014",
        facecolor="#FFF9F5", edgecolor="#FBD38D", linewidth=0.95, zorder=0,
    )
    isl_right = FancyBboxPatch(
        (0.553, 0.275), 0.435, 0.39,
        boxstyle="round,pad=0.006,rounding_size=0.014",
        facecolor="#FFF9F5", edgecolor="#FBD38D", linewidth=0.95, zorder=0,
    )
    ax.add_patch(isl_left)
    ax.add_patch(isl_right)
    ax.text(0.024, 0.652, "PCIe Island A (NVLink4 Pair: GPUs 0 <-> 1)",
            fontsize=7.4, fontweight="bold", color=GPU_EDGE, va="top", zorder=2)
    ax.text(0.565, 0.652, "PCIe Island B (NVLink4 Pair: GPUs 2 <-> 3)",
            fontsize=7.4, fontweight="bold", color=GPU_EDGE, va="top", zorder=2)

    g_w, g_h, g_y = 0.184, 0.30, 0.298
    _box(ax, 0.023, g_y, g_w, g_h,
         "GPU 0: RTX A6000",
         "Resident Slabs 0, 1, 2\n137-205M part (20-27 GB)\nTree, FMM, 128^3 PM",
         GPU_FILL, GPU_EDGE, title_fs=7.1, body_fs=6.0)
    _box(ax, 0.251, g_y, g_w, g_h,
         "GPU 1: RTX A6000",
         "Resident Slabs 3, 4, 5\n269-279M part (35.7-36 GB)\nPeer ghost: Slab 2 <-> 3",
         GPU_FILL, GPU_EDGE, title_fs=7.1, body_fs=6.0)
    _box(ax, 0.565, g_y, g_w, g_h,
         "GPU 2: RTX A6000",
         "Resident Slabs 6, 7, 8\n293-348M part (38.1-45.7 GB)\nPeer ghost: Slab 8 <-> 9",
         GPU_FILL, GPU_EDGE, title_fs=7.1, body_fs=6.0)
    _box(ax, 0.793, g_y, g_w, g_h,
         "GPU 3: RTX A6000",
         "Resident Slabs 9, 10, 11\n297-319M part (38.3-41.2 GB)\nPeriodic wrap: Slab 11 <-> 0",
         GPU_FILL, GPU_EDGE, title_fs=7.1, body_fs=6.0)

    # Direct NVLink arrows
    _arrow(ax, 0.211, 0.448, 0.247, 0.448, color=NVL_EDGE, lw=1.8, both=True)
    ax.text(0.229, 0.485, "NV4", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=NVL_EDGE)
    _arrow(ax, 0.753, 0.448, 0.789, 0.448, color=NVL_EDGE, lw=1.8, both=True)
    ax.text(0.771, 0.485, "NV4", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=NVL_EDGE)

    # Deadlocked direct NODE P2P marker cleanly centered in wide gap [0.439, 0.561]
    ax.plot([0.442, 0.558], [0.460, 0.460], color=RED, lw=1.2, ls=":", zorder=4)
    ax.text(0.500, 0.425, "X NODE P2P\n(stalls at\n1024^3)", ha="center", va="top",
            fontsize=5.6, fontweight="bold", color=RED, linespacing=1.15,
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec=RED, lw=0.6))

    # Arrows between Host and GPUs
    _arrow(ax, 0.115, 0.605, 0.115, 0.735, color=CPU_EDGE, lw=1.0, both=True)
    _arrow(ax, 0.343, 0.605, 0.343, 0.735, color="#B7791F", lw=1.25, both=True)
    _arrow(ax, 0.657, 0.605, 0.657, 0.735, color="#B7791F", lw=1.25, both=True)
    _arrow(ax, 0.885, 0.605, 0.885, 0.735, color=NVL_EDGE, lw=1.0)

    # Bottom resident per-step kernel bar
    step_band = FancyBboxPatch(
        (0.012, 0.048), 0.976, 0.195,
        boxstyle="round,pad=0.006,rounding_size=0.014",
        facecolor="#F8FAFC", edgecolor="#94A3B8", linewidth=0.9, zorder=0,
    )
    ax.add_patch(step_band)
    ax.text(0.024, 0.233, "Per-Step Zero-Swap Resident Execution Pipeline (0 Host Particle Swaps Across Evolution)",
            fontsize=7.3, fontweight="bold", color=SLATE, va="top", zorder=2)

    st_titles = [
        "1. Drift & Migrate",
        "2. Resident PM",
        "3. Slab Octrees",
        "4. Ghost Exchange",
        "5. Warp Tree-FMM",
        "6. Leapfrog Kick",
    ]
    st_bodies = [
        "Dual-stream D2D/host\nslab crossing",
        "cuFFT 128^3 grid\n(on PM steps)",
        "Sort, radix nodes &\nP2M multipoles",
        "Copy +/-2 neighbor\nghost trees",
        "Warp-level M2L/P2P\nfast_split kernels",
        "Close kick2 + dt bin\n+ open kick1",
    ]
    xs_st = np.linspace(0.024, 0.834, 6)
    w_st = 0.142
    for i, (sx, tt, bb) in enumerate(zip(xs_st, st_titles, st_bodies)):
        _box(ax, sx, 0.062, w_st, 0.120, tt, bb, "#FFFFFF", SLATE, title_fs=6.6, body_fs=5.6)
        if i < 5:
            _arrow(ax, sx + w_st + 0.002, 0.122, xs_st[i + 1] - 0.002, 0.122, color=SLATE, lw=0.9)

    ax.text(0.50, 0.006,
            "1,073,741,824 particles (1024^3) split across 12 Y-slabs (3 slabs/GPU) in 4 x 48 GiB VRAM; periodic boundary slabs 0 and 11 exchange via host staging.",
            ha="center", va="bottom", fontsize=6.1, color="#444444", style="italic")

    out_pdf = OUT_DIR / "fig1_architecture.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("Wrote", out_pdf)


def make_fig2_scaling_and_vram():
    c56_4 = json.loads((REPORTS_DIR / "mgpu4-1024-0056-comparison.json").read_text())
    c56_1 = json.loads((REPORTS_DIR / "slab-1024-0056-comparison.json").read_text())
    c45_4 = json.loads((REPORTS_DIR / "mgpu4-1024-0045-comparison.json").read_text())
    hist = json.loads((REPORTS_DIR / "full-history-1024-model.json").read_text())

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.6, 3.35), gridspec_kw={"width_ratios": [1.18, 1.0]}
    )

    # Panel (a): 1024^3 evolution time & speedup across z~0 and z~2
    # Groups: z = 0.063 -> 0 (0055 -> 0056) and z = 2.033 -> 1.758 (0044 -> 0045)
    cpu_0056 = hist["measured_anchors"]["cpu_0055_0056_pure_evo_s"] / 60.0
    gpu1_0056 = hist["measured_anchors"]["gpu1_z0"]["evolution_wall_s"] / 60.0
    gpu4_single_0056 = 5719.90190554 / 60.0  # Slurm 489830
    gpu4_dual_0056 = hist["measured_anchors"]["gpu4_z0"]["evolution_wall_s"] / 60.0

    cpu_0045 = hist["measured_anchors"]["cpu_0044_0045_pure_evo_s"] / 60.0
    gpu1_0045_est = hist["measured_anchors"]["gpu1_z2_probe_projected_0044_0045_s"] / 60.0
    gpu4_shared_0045 = hist["measured_anchors"]["gpu4_z2_shared"]["evolution_wall_s"] / 60.0
    gpu4_solo_0045 = hist["measured_anchors"]["gpu4_z2_solo"]["evolution_wall_s"] / 60.0

    x0 = np.array([0.0, 1.0])
    w = 0.175
    b_cpu = ax1.bar(x0 - 1.5 * w, [cpu_0056, cpu_0045], w, color="#7BAFD4", edgecolor="#222222",
                    linewidth=0.65, label="256-core CPU (CHPC)")
    b_g1_0 = ax1.bar([x0[0] - 0.5 * w], [gpu1_0056], w, color=BLUE, edgecolor="#222222",
                     linewidth=0.65, label="1x RTX A6000 (resident)")
    b_g1_2 = ax1.bar([x0[1] - 0.5 * w], [gpu1_0045_est], w, color=BLUE, edgecolor="#222222",
                     linewidth=0.65, hatch="///", alpha=0.85, label="1x A6000 (mid-z probe proj.)")
    b_mid = ax1.bar([x0[0] + 0.5 * w, x0[1] + 0.5 * w], [gpu4_single_0056, gpu4_shared_0045], w,
                    color=SKY, edgecolor="#222222", linewidth=0.65,
                    label="4x A6000 (1-NVLink / shared)")
    b_dual = ax1.bar(x0 + 1.5 * w, [gpu4_dual_0056, gpu4_solo_0045], w, color=ORANGE,
                     edgecolor="#222222", linewidth=0.65, label="4x A6000 Dual-NVLink (solo)")

    ax1.set_yscale("log")
    ax1.set_ylim(25, 2200)
    ax1.set_xticks(x0)
    ax1.set_xticklabels([r"$z = 0.063 \rightarrow 0$" + "\n(513 GPU steps)",
                         r"$z = 2.033 \rightarrow 1.758$" + "\n(910 GPU steps)"])
    ax1.set_ylabel("Pure evolution wall time (min)")
    ax1.set_title(r"(a) $1024^3$ Evolution Runtime & CPU Speedup")
    ax1.grid(axis="y", linestyle="--", alpha=0.28)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ann = [
        (x0[0] - 1.5 * w, cpu_0056, f"{cpu_0056:.0f}m", "#222222"),
        (x0[0] - 0.5 * w, gpu1_0056, f"{gpu1_0056:.0f}m\n(3.02x)", BLUE),
        (x0[0] + 0.5 * w, gpu4_single_0056, f"{gpu4_single_0056:.1f}m\n(5.57x)", "#1A608A"),
        (x0[0] + 1.5 * w, gpu4_dual_0056, f"{gpu4_dual_0056:.1f}m\n(7.26x)", GPU_EDGE),
        (x0[1] - 1.5 * w, cpu_0045, f"{cpu_0045:.0f}m", "#222222"),
        (x0[1] - 0.5 * w, gpu1_0045_est, f"{gpu1_0045_est:.0f}m\n(3.35x)", BLUE),
        (x0[1] + 0.5 * w, gpu4_shared_0045, f"{gpu4_shared_0045:.1f}m\n(7.46x)", "#1A608A"),
        (x0[1] + 1.5 * w, gpu4_solo_0045, f"{gpu4_solo_0045:.1f}m\n(9.27x)", GPU_EDGE),
    ]
    for cx, val, txt, col in ann:
        ax1.text(cx, val * 1.07, txt, ha="center", va="bottom", fontsize=5.9,
                 fontweight="bold" if col == GPU_EDGE else "normal", color=col)

    ax1.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.94, edgecolor="#CCCCCC", fontsize=6.0)

    # Panel (b): Per-GPU VRAM footprint vs 48 GiB ceiling
    mp_z0 = c56_4["timing_gpu"]["memory_plan"]
    mp_z2 = c45_4["timing_gpu"]["memory_plan"]
    vram_z0_gb = np.array(mp_z0["per_gpu_device_bytes"]) / 1e9
    vram_z2_gb = np.array(mp_z2["per_gpu_device_bytes"]) / 1e9
    part_z0_m = np.array(mp_z0["per_gpu_particles"]) / 1e6
    part_z2_m = np.array(mp_z2["per_gpu_particles"]) / 1e6
    ceil_gb = mp_z0["total_vram_bytes"] / 1e9  # 50.90 GB (47.4 GiB usable)

    xg = np.arange(4)
    wg = 0.34
    ax2.bar(xg - wg / 2, vram_z2_gb, wg, color=GREEN, edgecolor="#222222", linewidth=0.65,
            label=r"$z = 1.758$ (min free 12.31 GB)")
    ax2.bar(xg + wg / 2, vram_z0_gb, wg, color=BLUE, edgecolor="#222222", linewidth=0.65,
            label=r"$z = 0.0$ (min free 4.86 GB)")

    ax2.axhline(ceil_gb, color=RED, linestyle="--", linewidth=1.0)
    ax2.text(0.02, ceil_gb + 0.8, f"RTX A6000 Usable VRAM Ceiling ({ceil_gb:.1f} GB / 47.4 GiB)",
             fontsize=6.5, color=RED, ha="left", va="bottom", fontweight="bold")

    for i in range(4):
        ax2.text(xg[i] - wg / 2, vram_z2_gb[i] + 0.7, f"{vram_z2_gb[i]:.1f}\n({part_z2_m[i]:.0f}M)",
                 ha="center", va="bottom", fontsize=5.8, color=NVL_EDGE)
        ax2.text(xg[i] + wg / 2, vram_z0_gb[i] + 0.7, f"{vram_z0_gb[i]:.1f}\n({part_z0_m[i]:.0f}M)",
                 ha="center", va="bottom", fontsize=5.8, color=BLUE)

    ax2.set_ylim(0, 58)
    ax2.set_xticks(xg)
    ax2.set_xticklabels(["GPU 0\n(Slabs 0-2)", "GPU 1\n(Slabs 3-5)", "GPU 2\n(Slabs 6-8)", "GPU 3\n(Slabs 9-11)"])
    ax2.set_ylabel("Allocated Device Memory (GB)")
    ax2.set_title(r"(b) Per-GPU Resident VRAM & Particle Balance ($1024^3$)")
    ax2.grid(axis="y", linestyle="--", alpha=0.28)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.94, edgecolor="#CCCCCC", fontsize=6.5)

    fig.tight_layout(w_pad=1.6)
    out_pdf = OUT_DIR / "fig2_scaling_and_vram.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("Wrote", out_pdf)


def _read_pk_csv(path):
    ks, ratios, rs = [], [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            k = float(row["k_h_per_Mpc"])
            if 0.3 <= k <= 16.0:
                ks.append(k)
                ratios.append(float(row["P_gpu_over_cpu"]))
                rs.append(float(row["cross_correlation"]))
    return np.asarray(ks), np.asarray(ratios), np.asarray(rs)


def make_fig3_scientific_validation():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(9.4, 3.15))

    # Panel (a): Matter P_delta(k) fractional residual (%)
    k0, r0, _ = _read_pk_csv(REPORTS_DIR / "mgpu4-1024-0056-matter-power.csv")
    k2, r2, _ = _read_pk_csv(REPORTS_DIR / "mgpu4-1024-0045-matter-power.csv")

    ax1.axhspan(-0.015, 0.015, color="#F1F5F9", zorder=0)
    ax1.axhline(0.0, color="#64748B", lw=0.8, ls="--", zorder=1)
    ax1.semilogx(k0, (r0 - 1.0) * 100.0, color=BLUE, lw=1.6, marker="o", ms=2.4,
                 label=r"$z=0$ ($\max\ 0.00453\%$)")
    ax1.semilogx(k2, (r2 - 1.0) * 100.0, color=ORANGE, lw=1.5, marker="s", ms=2.3,
                 label=r"$z=1.758$ ($\max\ 0.01019\%$)")
    ax1.set_xlim(0.28, 16.5)
    ax1.set_ylim(-0.015, 0.006)
    ax1.set_xlabel(r"$k\ (h\,\mathrm{Mpc}^{-1})$")
    ax1.set_ylabel(r"$(P_\delta^{\mathrm{GPU}} / P_\delta^{\mathrm{CPU}} - 1)\ [\%]$")
    ax1.set_title(r"(a) Matter Power $P_\delta(k)$ Residual")
    ax1.grid(True, which="both", linestyle="--", alpha=0.25)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="lower left", frameon=True, framealpha=0.94, edgecolor="#CCCCCC", fontsize=6.6)
    ax1.text(0.97, 0.94, r"$\min\,r = 0.99999995$", transform=ax1.transAxes,
             ha="right", va="top", fontsize=6.4, color=SLATE,
             bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#CBD5E1", lw=0.7))

    # Panel (b): Momentum divergence P_theta(k) absolute % residual (raw vs kick-aligned)
    km0_raw, rm0_raw, _ = _read_pk_csv(REPORTS_DIR / "mgpu4-1024-0056-momentum-power.csv")
    km0_adj, rm0_adj, _ = _read_pk_csv(REPORTS_DIR / "mgpu4-1024-0056-momentum-adjusted-power.csv")
    km2_raw, rm2_raw, _ = _read_pk_csv(REPORTS_DIR / "mgpu4-1024-0045-momentum-power.csv")

    ax2.loglog(km0_raw, np.maximum(np.abs(rm0_raw - 1.0) * 100.0, 1e-5), color=BLUE, ls="--", lw=1.25,
               label=r"$z=0$ raw ($\max\ 0.702\%$)")
    ax2.loglog(km0_adj, np.maximum(np.abs(rm0_adj - 1.0) * 100.0, 1e-5), color=BLUE, lw=1.7, marker="o", ms=2.3,
               label=r"$z=0$ kick-aligned ($\max\ 0.0405\%$)")
    ax2.loglog(km2_raw, np.maximum(np.abs(rm2_raw - 1.0) * 100.0, 1e-5), color=ORANGE, lw=1.55, marker="s", ms=2.2,
               label=r"$z=1.758$ raw sync ($\max\ 0.0111\%$)")
    ax2.annotate(
        "17.3x drop\n(+2s align)",
        xy=(6.2, 0.036), xytext=(0.65, 0.045),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.0),
        fontsize=6.2, fontweight="bold", color=BLUE,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=BLUE, lw=0.7),
    )
    ax2.set_xlim(0.28, 16.5)
    ax2.set_ylim(2e-4, 2.0)
    ax2.set_xlabel(r"$k\ (h\,\mathrm{Mpc}^{-1})$")
    ax2.set_ylabel(r"$|P_\theta^{\mathrm{GPU}} / P_\theta^{\mathrm{CPU}} - 1|\ [\%]$")
    ax2.set_title(r"(b) Momentum Divergence $P_\theta(k)$")
    ax2.grid(True, which="both", linestyle="--", alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.94, edgecolor="#CCCCCC", fontsize=6.1)

    # Panel (c): Cumulative FoF HMF relative difference (%) + 1/sqrt(N) Poisson band
    c56 = json.loads((REPORTS_DIR / "mgpu4-1024-0056-comparison.json").read_text())
    c45 = json.loads((REPORTS_DIR / "mgpu4-1024-0045-comparison.json").read_text())
    thrs = np.array([32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144], dtype=np.int64)
    rel56 = np.array([c56["fof_cumulative_differences"][str(t)]["relative_to_cpu"] * 100.0 for t in thrs])
    rel45 = np.array([c45["fof_cumulative_differences"][str(t)]["relative_to_cpu"] * 100.0 for t in thrs])
    cnt45 = np.array([c45["fof_cumulative_differences"][str(t)]["cpu"] for t in thrs], dtype=float)
    shot = 100.0 / np.sqrt(cnt45)

    ax3.fill_between(thrs, -shot, shot, color="#CBD5E1", alpha=0.45, label=r"Poisson $\pm N_{\rm halo}^{-1/2}$")
    ax3.axhspan(-0.35, 0.35, color="#F1F5F9", alpha=0.65, zorder=0)
    ax3.axhline(0.0, color="#64748B", lw=0.8, ls="--")
    ax3.semilogx(thrs, rel56, color=BLUE, lw=1.6, marker="o", ms=3.5, label=r"$z=0$ ($\pm 0.32\%$, 198 vs 198)")
    ax3.semilogx(thrs, rel45, color=ORANGE, lw=1.5, marker="s", ms=3.3, label=r"$z=1.758$ ($\pm 0.18\%$, 146 vs 147)")
    ax3.set_xscale("log", base=2)
    ax3.set_xticks([32, 256, 4096, 65536, 262144])
    ax3.set_xticklabels(["32", "256", "4k", "64k", "256k"])
    ax3.set_ylim(-1.15, 0.95)
    ax3.set_xlabel(r"Minimum halo size $N_{\mathrm{min}}$ (particles)")
    ax3.set_ylabel(r"$(N_{\mathrm{FoF}}^{\mathrm{GPU}} / N_{\mathrm{FoF}}^{\mathrm{CPU}} - 1)\ [\%]$")
    ax3.set_title(r"(c) Native SWIFT FoF Mass Function")
    ax3.grid(True, which="both", linestyle="--", alpha=0.25)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.legend(loc="lower left", frameon=True, framealpha=0.94, edgecolor="#CCCCCC", fontsize=6.0)

    fig.tight_layout(w_pad=1.35)
    out_pdf = OUT_DIR / "fig3_scientific_validation.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("Wrote", out_pdf)


def make_fig4_full_history_scaling():
    hist = json.loads((REPORTS_DIR / "full-history-1024-model.json").read_text())
    epochs = hist["timesteps_summary"]["epoch_breakdown"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.6, 3.30), gridspec_kw={"width_ratios": [1.32, 1.0]}
    )

    # Panel (a): 6-epoch wall-clock comparison
    ep_labels = [e["epoch"].replace("->", r"$\to$") for e in epochs]
    cpu_pure_h = np.array([e["cpu_pure_evo_s"] / 3600.0 for e in epochs])
    cpu_dump_h = np.array([e["cpu_dump_overhead_s"] / 3600.0 for e in epochs])
    # Representative midpoints of each epoch in scale factor a for clamped interval GPU curves
    # Anchors from full-history-1024-model.json
    a_mids = np.array([0.012, 0.055, 0.145, 0.30, 0.53, 0.83])
    anchors = hist["measured_anchors"]
    sp_1_lo = anchors["cpu_0044_0045_pure_evo_s"] / anchors["gpu1_z2_probe_projected_0044_0045_s"]
    sp_1_hi = anchors["cpu_0055_0056_pure_evo_s"] / anchors["gpu1_z0"]["evolution_wall_s"]
    sp_4_lo = anchors["cpu_0044_0045_pure_evo_s"] / anchors["gpu4_z2_solo"]["evolution_wall_s"]
    sp_4_hi = anchors["cpu_0055_0056_pure_evo_s"] / anchors["gpu4_z0"]["evolution_wall_s"]
    a_lo, a_hi = anchors["gpu4_z2_solo"]["a_mid"], anchors["gpu4_z0"]["a_mid"]

    def lerp(a, lo, hi):
        t = np.clip((a - a_lo) / (a_hi - a_lo), 0.0, 1.0)
        return lo + (hi - lo) * t

    gpu1_ep_h = cpu_pure_h / lerp(a_mids, sp_1_lo, sp_1_hi)
    gpu4_ep_h = cpu_pure_h / lerp(a_mids, sp_4_lo, sp_4_hi)

    xe = np.arange(len(epochs))
    we = 0.25
    ax1.bar(xe - we, cpu_pure_h, we, color="#7BAFD4", edgecolor="#222222", linewidth=0.6,
            label="256c CPU pure gravity")
    ax1.bar(xe - we, cpu_dump_h, we, bottom=cpu_pure_h, color="#CBD5E1", edgecolor="#222222",
            linewidth=0.6, hatch="///", label="CPU 56 inline dumps")
    ax1.bar(xe, gpu1_ep_h, we, color=BLUE, edgecolor="#222222", linewidth=0.6,
            label="1x RTX A6000 (model)")
    ax1.bar(xe + we, gpu4_ep_h, we, color=ORANGE, edgecolor="#222222", linewidth=0.6,
            label="4x A6000 NVLink (model)")

    for i in range(len(epochs)):
        tot = cpu_pure_h[i] + cpu_dump_h[i]
        ax1.text(xe[i] - we, tot + 1.1, f"{tot:.1f}h", ha="center", va="bottom", fontsize=5.6, color="#222222")
        ax1.text(xe[i] + we + 0.04, gpu4_ep_h[i] + 1.1, f"{gpu4_ep_h[i]:.1f}h", ha="center", va="bottom",
                 fontsize=5.5, fontweight="bold", color=GPU_EDGE)

    ax1.set_ylim(0, 92)
    ax1.set_xticks(xe)
    ax1.set_xticklabels(ep_labels, fontsize=7.0)
    ax1.set_ylabel("Epoch wall-clock time (h)")
    ax1.set_title(r"(a) Six-Epoch Compute Breakdown ($20{,}194$ CPU Steps)")
    ax1.grid(axis="y", linestyle="--", alpha=0.28)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.94, edgecolor="#CCCCCC", fontsize=5.9)

    # Panel (b): Total runtime (h) & whole-node energy (kWh) for z=200->0 and z=50->0
    p200 = hist["projections"]["z200_to_z0"]
    p50 = hist["projections"]["z50_to_z0"]

    labels = [
        "256c CPU\n(z=200->0)",
        "1x A6000\n(z=200->0)",
        "4x A6000\n(z=200->0)",
        "4x A6000\n(z=50->0)",
    ]
    xb = np.arange(len(labels))
    wb = 0.34

    # Use midpoint of clamped interval and step-physics models with range error bars
    def mid_err(gm):
        a = gm["clamped_interval_model"]
        b = gm["step_physics_model"]
        return (
            0.5 * (a["pure_evo_hours"] + b["pure_evo_hours"]),
            0.5 * abs(a["pure_evo_hours"] - b["pure_evo_hours"]),
            0.5 * (a["whole_node_kwh"] + b["whole_node_kwh"]),
            0.5 * abs(a["whole_node_kwh"] - b["whole_node_kwh"]),
        )

    t1, te1, e1, ee1 = mid_err(p200["gpu_models"]["1x_RTX_A6000_slab_resident"])
    t4, te4, e4, ee4 = mid_err(p200["gpu_models"]["4x_RTX_A6000_NVLink_solo"])
    t50, te50, e50, ee50 = mid_err(p50["gpu_models"]["4x_RTX_A6000_NVLink_solo"])

    hrs = np.array([p200["cpu_256core"]["pure_evo_hours"], t1, t4, t50])
    hrs_err = np.array([0.0, te1, te4, te50])
    kwh = np.array([p200["cpu_256core"]["energy_pure_kwh"], e1, e4, e50])
    kwh_err = np.array([0.0, ee1, ee4, ee50])

    ax2.bar(xb - wb / 2, hrs, wb, yerr=hrs_err, capsize=2.5, color=BLUE, edgecolor="#222222",
            linewidth=0.65, label="Pure evolution wall time (h)")
    ax2.bar(xb + wb / 2, kwh, wb, yerr=kwh_err, capsize=2.5, color=GREEN, edgecolor="#222222",
            linewidth=0.65, label="Whole-node energy (kWh)")

    for i in range(len(xb)):
        ax2.text(xb[i] - wb / 2, hrs[i] + hrs_err[i] + 6, f"{hrs[i]:.1f}h",
                 ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=BLUE)
        ax2.text(xb[i] + wb / 2, kwh[i] + kwh_err[i] + 6, f"{kwh[i]:.1f}",
                 ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=NVL_EDGE)

    ax2.annotate(
        "8.5-8.8x faster\n86% less energy",
        xy=(2.0, 62), xytext=(2.05, 168),
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.05),
        ha="center", va="bottom", fontsize=6.7, fontweight="bold", color=GPU_EDGE,
        bbox=dict(boxstyle="round,pad=0.22", fc="#FFF6EF", ec=ORANGE, lw=0.7),
    )

    ax2.set_ylim(0, 355)
    ax2.set_xticks(xb)
    ax2.set_xticklabels(labels, fontsize=7.0)
    ax2.set_ylabel("Wall time (h) / Energy (kWh)")
    ax2.set_title(r"(b) Full-Run Runtime & Energy ($1024^3$)")
    ax2.grid(axis="y", linestyle="--", alpha=0.28)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.94, edgecolor="#CCCCCC", fontsize=6.3)

    fig.tight_layout(w_pad=1.5)
    out_pdf = OUT_DIR / "fig4_full_history_scaling.pdf"
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("Wrote", out_pdf)


if __name__ == "__main__":
    make_fig1_architecture()
    make_fig2_scaling_and_vram()
    make_fig3_scientific_validation()
    make_fig4_full_history_scaling()
