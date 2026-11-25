"""Generate CPU-vs-GPU speed and accuracy comparison figures (Opt1-5, 256^3 & 1024^3)."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT_DIR / "reports"
OUT_DIR = ROOT_DIR / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8.6,
    "axes.titlesize": 9.6,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.7,
    "legend.fontsize": 6.9,
    "axes.linewidth": 0.85,
    "axes.edgecolor": "#222222",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"
ORANGE = "#D55E00"
AMBER = "#E69F00"
GREEN = "#009E73"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
SLATE = "#475569"
CPU_COL = "#64748B"


def _load_json(name: str) -> dict:
    return json.loads((REPORTS_DIR / name).read_text())


def _read_pk_csv(name: str):
    ks, ratios, rs = [], [], []
    with (REPORTS_DIR / name).open() as f:
        for row in csv.DictReader(f):
            k = float(row["k_h_per_Mpc"])
            if 0.3 <= k <= 16.0:
                ks.append(k)
                ratios.append(float(row["P_gpu_over_cpu"]))
                rs.append(float(row["cross_correlation"]))
    return np.asarray(ks), np.asarray(ratios), np.asarray(rs)


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_both(fig, stem: str):
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.07)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.07)
    plt.close(fig)
    print(f"Wrote {pdf_path} and {png_path}")


def make_fig5_speedup_vs_cpu():
    o3 = _load_json("opt3-bench.json")["p32_256_abc"]
    o4a = _load_json("opt4a-bench.json")
    o4b = _load_json("opt4b-bench.json")
    o5 = _load_json("opt5-bench.json")
    base = _load_json("baseline-092-acc.json")
    hist = _load_json("full-history-1024-model.json")

    # CPU 256^3 pure evolution step-sum from native-092-timesteps.txt (320 steps; 815.98 s total in native-092.time)
    cpu_256_pure_s = 796.358
    t_base = base["execution"]["result"]["evolution_wall_s"]
    r3a = o3["runA_mono_eager"]
    r3b = o3["runB_split_eager"]
    r4a = o4a["runs"]
    r4b = o4b["runs"]

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.8), layout="constrained")
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    # ------------------------------------------------------------------
    # Panel (a): 256^3 (16.78M) Full Opt Progression vs 256-Thread CPU SWIFT
    # ------------------------------------------------------------------
    prog = [
        ("256-Thread CPU (native-092)", cpu_256_pure_s, CPU_COL),
        ("1x A6000 Initial Baseline", t_base, "#94A3B8"),
        ("1x A6000 12-Slab Pre-Opt", r3a["evo_s"], SKY),
        ("1x A6000 + Opt3 Split-Force", r3b["evo_s"], SKY),
        ("1x A6000 + Opt1-3 (1D Eager)", r4b["g1-1d-nolazy"]["evo_s"], BLUE),
        ("1x A6000 + Opt1-3 (1D Full)", r4b["g1-1d-full"]["evo_s"], BLUE),
        ("1x A6000 + Opt4B (3D Hilbert)", r4b["g1-3d-full"]["evo_s"], GREEN),
        ("4x A6000 Legacy 1D Slab", r4a["g4-legacy"]["evo_s"], AMBER),
        ("4x A6000 Opt4A (1D + Dist-PM)", r4a["g4-full"]["evo_s"], ORANGE),
        ("4x A6000 Final Opt4B (3D Hilb)", r4b["g4-3d-hilbert-full"]["evo_s"], ORANGE),
    ]
    labels_a = [p[0] for p in prog][::-1]
    vals_a = np.array([p[1] for p in prog][::-1])
    cols_a = [p[2] for p in prog][::-1]
    ya = np.arange(len(prog))

    ax_a.barh(ya, vals_a, height=0.66, color=cols_a, edgecolor="#222222", linewidth=0.65)
    ax_a.set_xscale("log")
    ax_a.set_xlim(9, 4800)
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels(labels_a, fontsize=7.6)
    ax_a.set_xlabel("Pure evolution wall time (s, log scale)")
    ax_a.set_title(r"(a) $256^3$ ($a=0.855\rightarrow 0.92$, 320 steps) vs. 256T CPU")
    ax_a.grid(axis="x", which="both", linestyle="--", alpha=0.25)
    _despine(ax_a)

    for yi, v, col in zip(ya, vals_a, cols_a):
        sp = cpu_256_pure_s / v
        tag = " (1.0x)" if abs(v - cpu_256_pure_s) < 1e-3 else f" ({sp:.1f}x)"
        is_best = (yi == 0)
        ax_a.text(
            v * 1.12, yi, f"{v:.1f} s{tag}",
            va="center", ha="left", fontsize=7.2,
            fontweight="bold" if is_best or col == CPU_COL else "normal",
            color=ORANGE if is_best else "#222222",
        )

    # ------------------------------------------------------------------
    # Panel (b): 256^3 Phase Breakdown Across Key GPU Optimizations
    # ------------------------------------------------------------------
    stages = [
        ("1G\nPre-Opt", r3a),
        ("1G 1D\nEager", r4b["g1-1d-nolazy"]),
        ("1G 1D\nFull", r4b["g1-1d-full"]),
        ("1G 3D\nHilbert", r4b["g1-3d-full"]),
        ("4G 1D\nFull", r4b["g4-1d-full"]),
        ("4G 3D\nHilbert", r4b["g4-3d-hilbert-full"]),
    ]
    xb = np.arange(len(stages))
    wb = 0.56
    c_force = np.array([s[1]["force_s"] for s in stages])
    c_tree = np.array([s[1]["tree_s"] for s in stages])
    c_ghost = np.array([s[1].get("ghost_s", 0.0) for s in stages])
    c_dyn = np.array([s[1]["drift_s"] + s[1]["kick_s"] + s[1]["pm_s"] for s in stages])
    c_tot = np.array([s[1]["evo_s"] for s in stages])
    c_other = np.maximum(0.0, c_tot - (c_force + c_tree + c_ghost + c_dyn))

    b0 = np.zeros_like(xb, dtype=float)
    segs = [
        ("Force (M2L+P2P)", c_force, BLUE),
        ("Tree Build / Lazy", c_tree, ORANGE),
        ("3D/1D Ghost Comm", c_ghost, GREEN),
        ("Drift + Kick + PM", c_dyn, SKY),
        ("Sort + Sync + D2H", c_other, "#CBD5E1"),
    ]
    for sname, svals, scol in segs:
        ax_b.bar(xb, svals, wb, bottom=b0, color=scol, edgecolor="#222222", linewidth=0.55, label=sname)
        b0 += svals

    for xi, tot in zip(xb, c_tot):
        ax_b.text(xi, tot + 1.8, f"{tot:.1f}s", ha="center", va="bottom",
                  fontsize=7.3, fontweight="bold" if xi == len(stages) - 1 else "normal",
                  color=ORANGE if xi == len(stages) - 1 else "#222222")

    ax_b.annotate(
        "Tree -96%\n(10.1 -> 0.38s)",
        xy=(4.85, c_force[5] + 0.5 * c_tree[5]), xytext=(3.15, 63),
        arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.0, shrinkB=2),
        ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=ORANGE,
        bbox=dict(boxstyle="round,pad=0.2", fc="#FFF7ED", ec=ORANGE, lw=0.7),
    )

    ax_b.set_ylim(0, 108)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels([s[0] for s in stages], fontsize=7.4)
    ax_b.set_ylabel("Wall time (s)")
    ax_b.set_title(r"(b) $256^3$ Kernel & Pipeline Time Breakdown")
    ax_b.grid(axis="y", linestyle="--", alpha=0.25)
    _despine(ax_b)
    ax_b.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CBD5E1", fontsize=6.7)

    # ------------------------------------------------------------------
    # Panel (c): Multi-Scale & Post-Processing Speedup Summary vs CPU
    # ------------------------------------------------------------------
    anc = hist["measured_anchors"]
    p200 = hist["projections"]["z200_to_z0"]
    cpu_0045_h = anc["cpu_0044_0045_pure_evo_s"] / 3600.0
    gpu1_0045_h = anc["gpu1_z2_probe_projected_0044_0045_s"] / 3600.0
    gpu4_0045_h = anc["gpu4_z2_solo"]["evolution_wall_s"] / 3600.0

    cpu_0056_h = anc["cpu_0055_0056_pure_evo_s"] / 3600.0
    gpu1_0056_h = anc["gpu1_z0"]["evolution_wall_s"] / 3600.0
    gpu4_0056_h = anc["gpu4_z0"]["evolution_wall_s"] / 3600.0

    cpu_full_h = p200["cpu_256core"]["pure_evo_hours"]
    gpu1_full_h = 0.5 * (
        p200["gpu_models"]["1x_RTX_A6000_slab_resident"]["clamped_interval_model"]["pure_evo_hours"]
        + p200["gpu_models"]["1x_RTX_A6000_slab_resident"]["step_physics_model"]["pure_evo_hours"]
    )
    gpu4_full_h = 0.5 * (
        p200["gpu_models"]["4x_RTX_A6000_NVLink_solo"]["clamped_interval_model"]["pure_evo_hours"]
        + p200["gpu_models"]["4x_RTX_A6000_NVLink_solo"]["step_physics_model"]["pure_evo_hours"]
    )

    t5 = o5["timings_256_054"]
    pm256_sp = o4a["speedups"]["pm256_dist_pm_speedup"]
    sp_256_full = cpu_256_pure_s / r4b["g4-3d-hilbert-full"]["evo_s"]
    sp_256_noio = cpu_256_pure_s / r4b["g4-3d-hilbert-full"]["evo_no_io_s"]
    sp_1024_z2 = anc["cpu_0044_0045_pure_evo_s"] / anc["gpu4_z2_solo"]["evolution_wall_s"]
    sp_1024_z0 = anc["cpu_0055_0056_pure_evo_s"] / anc["gpu4_z0"]["evolution_wall_s"]
    sp_1024_hist = cpu_full_h / gpu4_full_h

    summary_rows = [
        (f"256^3 4G 3D Evo vs 256T CPU\n({r4b['g4-3d-hilbert-full']['evo_s']:.1f}s vs {cpu_256_pure_s:.0f}s; {sp_256_noio:.1f}x no-IO)", sp_256_full, ORANGE),
        (f"256^3 GPU SO Uniform vs 8T CPU\n({t5['so_u_gpu_s']*1e3:.0f}ms vs {t5['so_u_cpu8t_s']:.2f}s)", t5["so_u_speedup"], GREEN),
        (f"256^3 GPU FoF Catalog vs CPU\n({t5['cat_gpu_s']*1e3:.0f}ms vs {t5['cat_cpu_s']:.2f}s)", t5["cat_speedup"], GREEN),
        (f"1024^3 z=2.03->1.76 (4G vs 256c)\n(58.5m vs 542.3m)", sp_1024_z2, BLUE),
        (f"1024^3 z=200->0 (4G vs 256c)\n({gpu4_full_h:.1f}h vs {cpu_full_h:.1f}h, -86% kWh)", sp_1024_hist, BLUE),
        (f"1024^3 z=0.06->0 (4G vs 256c)\n(73.1m vs 531.3m)", sp_1024_z0, BLUE),
        (f"256^3 GPU SO Non-Uni vs 8T CPU\n({t5['so_nu_gpu_s']*1e3:.0f}ms vs {t5['so_nu_cpu8t_s']:.2f}s)", t5["so_nu_speedup"], GREEN),
        (f"Dist-PM 256^3 Grid (4G vs 1G)\n(217ms vs 875ms, 17 steps)", pm256_sp, PURPLE),
    ]
    lab_c = [r[0] for r in summary_rows][::-1]
    val_c = [r[1] for r in summary_rows][::-1]
    col_c = [r[2] for r in summary_rows][::-1]
    yc = np.arange(len(summary_rows))

    ax_c.barh(yc, val_c, height=0.64, color=col_c, edgecolor="#222222", linewidth=0.65)
    ax_c.axvline(1.0, color=CPU_COL, linestyle="--", linewidth=0.9)
    ax_c.set_xlim(0, 56)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels(lab_c, fontsize=7.2)
    ax_c.set_xlabel("Speedup factor vs. reference (higher is faster)")
    ax_c.set_title("(c) Multi-Scale & FoF/SO Speedup Summary")
    ax_c.grid(axis="x", linestyle="--", alpha=0.25)
    _despine(ax_c)

    for yi, v, col in zip(yc, val_c, col_c):
        ax_c.text(v + 0.8, yi, f"{v:.2f}x", va="center", ha="left", fontsize=7.5,
                  fontweight="bold", color=col if col != PURPLE else "#8B3A62")

    legend_handles = [
        Patch(facecolor=ORANGE, edgecolor="#222222", label="Gravity 256^3 vs 256T CPU"),
        Patch(facecolor=BLUE, edgecolor="#222222", label="Gravity 1024^3 vs 256c CPU"),
        Patch(facecolor=GREEN, edgecolor="#222222", label="GPU Catalog/SO vs CPU"),
        Patch(facecolor=PURPLE, edgecolor="#222222", label="4-GPU vs 1-GPU Dist-PM"),
    ]
    ax_c.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.95,
                edgecolor="#CBD5E1", fontsize=6.5)

    # ------------------------------------------------------------------
    # Panel (d): 1024^3 (1.07B) GPU vs 256-Core CHPC CPU (Measured & Full Run)
    # ------------------------------------------------------------------
    xd = np.array([0.0, 1.05, 2.15])
    wd = 0.27
    vc = [cpu_0045_h, cpu_0056_h, cpu_full_h]
    vg1 = [gpu1_0045_h, gpu1_0056_h, gpu1_full_h]
    vg4 = [gpu4_0045_h, gpu4_0056_h, gpu4_full_h]

    ax_d.bar(xd - wd, vc, wd, color=CPU_COL, edgecolor="#222222", linewidth=0.6, label="256-Core CPU (CHPC)")
    ax_d.bar(xd, vg1, wd, color=BLUE, edgecolor="#222222", linewidth=0.6, label="1x RTX A6000 (Resident)")
    ax_d.bar(xd + wd, vg4, wd, color=ORANGE, edgecolor="#222222", linewidth=0.6, label="4x A6000 Dual-NVLink")

    ax_d.set_yscale("log")
    ax_d.set_ylim(0.5, 1100)
    ax_d.set_xlim(-0.5, 2.65)
    ax_d.set_xticks(xd)
    ax_d.set_xticklabels([
        r"$z=2.03\rightarrow 1.76$" + "\n(910 steps, meas.)",
        r"$z=0.06\rightarrow 0$" + "\n(513 steps, meas.)",
        r"$z=200\rightarrow 0$" + "\n(20,194 steps, model)",
    ], fontsize=7.5)
    ax_d.set_ylabel("Pure evolution wall time (h, log scale)")
    ax_d.set_title(r"(d) $1024^3$ ($1.07\times 10^9$ particles) vs. 256-Core CPU")
    ax_d.grid(axis="y", which="both", linestyle="--", alpha=0.25)
    _despine(ax_d)

    for i in range(3):
        ax_d.text(xd[i] - wd, vc[i] * 1.14, f"{vc[i]:.1f}h", ha="center", va="bottom", fontsize=6.9, color="#222222")
        sp1 = vc[i] / vg1[i]
        sp4 = vc[i] / vg4[i]
        ax_d.text(xd[i], vg1[i] * 1.14, f"{vg1[i]:.1f}h\n({sp1:.1f}x)", ha="center", va="bottom",
                  fontsize=6.4, color=BLUE)
        ax_d.text(xd[i] + wd + 0.03, vg4[i] * 1.14, f"{vg4[i]:.1f}h\n({sp4:.2f}x)", ha="center", va="bottom",
                  fontsize=6.6, fontweight="bold", color=ORANGE)

    ax_d.legend(loc="upper left", frameon=True, framealpha=0.95, edgecolor="#CBD5E1", fontsize=6.9)

    _save_both(fig, "fig5_opt_speedup_vs_cpu")


def make_fig6_accuracy_vs_cpu():
    base_256 = _load_json("baseline-092-acc.json")
    slab_256 = _load_json("mgpu4-256-092-acc.json")
    hilb_eager = _load_json("opt4b-g4-3d-hilbert-nolazy-acc.json")
    hilb_full = _load_json("opt4b-g4-3d-hilbert-full-acc.json")
    o4a = _load_json("opt4a-bench.json")
    o4b = _load_json("opt4b-bench.json")
    o5 = _load_json("opt5-bench.json")
    c56 = _load_json("mgpu4-1024-0056-accuracy.json")
    c45 = _load_json("mgpu4-1024-0045-accuracy.json")

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 7.6), layout="constrained")
    ax_a, ax_b = axes[0, 0], axes[0, 1]
    ax_c, ax_d = axes[1, 0], axes[1, 1]

    def _extract(acc, key):
        rows = acc[key]
        k = np.array([r["k"] for r in rows])
        ratio = np.array([r["ratio"] for r in rows])
        return k, ratio

    # ------------------------------------------------------------------
    # Panel (a): 256^3 Matter Power Spectrum Residual vs CPU native-092
    # ------------------------------------------------------------------
    ax_a.axhspan(-0.05, 0.05, color="#F1F5F9", zorder=0)
    ax_a.axhline(0.0, color=CPU_COL, lw=0.9, ls="--", zorder=1)

    series_a = [
        ("1x GPU Baseline", base_256, "#64748B", "--", "none", 1.3),
        ("4x GPU 1D Slab (Pre-Opt)", slab_256, SKY, "-", "s", 1.6),
        ("4x GPU 3D Hilbert (Opt4B Eager)", hilb_eager, BLUE, "-", "o", 1.8),
        ("4x GPU 3D Hilbert (Opt1-5 Full Lazy)", hilb_full, ORANGE, "-", "^", 1.8),
    ]
    for label, acc, col, ls, mk, lw in series_a:
        k, r = _extract(acc, "matter_vs_cpu")
        s = acc["matter_summary"]
        ax_a.semilogx(
            k, (r - 1.0) * 100.0,
            color=col, ls=ls, lw=lw, marker=mk if mk != "none" else None, ms=3.0,
            label=f"{label} (max {s['max_abs_dev']*100:.4f}%)",
        )

    ax_a.set_xlim(0.23, 8.6)
    ax_a.set_ylim(-0.048, 0.015)
    ax_a.set_xlabel(r"$k\ (h\,\mathrm{Mpc}^{-1})$")
    ax_a.set_ylabel(r"$(P_\delta^{\mathrm{GPU}} / P_\delta^{\mathrm{CPU}} - 1)\ [\%]$")
    ax_a.set_title(r"(a) $256^3$ Matter $P_\delta(k)$ vs. 256-Thread CPU SWIFT")
    ax_a.grid(True, which="both", linestyle="--", alpha=0.25)
    _despine(ax_a)
    ax_a.legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor="#CBD5E1", fontsize=6.6)
    ax_a.text(
        0.97, 0.94,
        f"3D Eager min r = {hilb_eager['matter_summary']['min_r']:.9f}\n"
        f"3D Lazy  min r = {hilb_full['matter_summary']['min_r']:.9f}",
        transform=ax_a.transAxes, ha="right", va="top", fontsize=6.8, color=SLATE,
        bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CBD5E1", lw=0.7),
    )

    # ------------------------------------------------------------------
    # Panel (b): 256^3 Momentum P_theta(k) Residual vs CPU native-092
    # ------------------------------------------------------------------
    series_b = [
        ("4x GPU 3D Hilbert Raw (-2s unaligned)", hilb_full, "momentum_vs_cpu", PURPLE, ":", "none", 1.4),
        ("1x GPU Baseline (kick-aligned)", base_256, "momentum_adjusted_vs_cpu", "#64748B", "--", "none", 1.4),
        ("4x GPU 1D Slab (kick-aligned)", slab_256, "momentum_adjusted_vs_cpu", SKY, "-", "s", 1.6),
        ("4x GPU 3D Hilbert Eager (kick-aligned)", hilb_eager, "momentum_adjusted_vs_cpu", BLUE, "-", "o", 1.8),
        ("4x GPU 3D Hilbert Full Lazy (kick-aligned)", hilb_full, "momentum_adjusted_vs_cpu", ORANGE, "-", "^", 1.8),
    ]
    for label, acc, key, col, ls, mk, lw in series_b:
        k, r = _extract(acc, key)
        sum_key = "momentum_summary" if key == "momentum_vs_cpu" else "momentum_adjusted_summary"
        mx = acc[sum_key]["max_abs_dev"] * 100.0
        ax_b.loglog(
            k, np.maximum(np.abs(r - 1.0) * 100.0, 1e-6),
            color=col, ls=ls, lw=lw, marker=mk if mk != "none" else None, ms=2.8,
            label=f"{label} ({mx:.3f}%)",
        )

    ax_b.axhline(0.1, color=GREEN, lw=0.95, ls="-.", alpha=0.85)
    ax_b.text(0.25, 0.115, "0.1% threshold", fontsize=6.8, color=GREEN, fontweight="bold")
    ax_b.set_xlim(0.23, 8.6)
    ax_b.set_ylim(5e-5, 22.0)
    ax_b.set_xlabel(r"$k\ (h\,\mathrm{Mpc}^{-1})$")
    ax_b.set_ylabel(r"$|P_\theta^{\mathrm{GPU}} / P_\theta^{\mathrm{CPU}} - 1|\ [\%]$")
    ax_b.set_title(r"(b) $256^3$ Momentum Divergence $P_\theta(k)$ vs. CPU")
    ax_b.grid(True, which="both", linestyle="--", alpha=0.25)
    _despine(ax_b)
    ax_b.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CBD5E1", fontsize=6.5)

    # ------------------------------------------------------------------
    # Panel (c): Cross-Component Verification Error Ladder (Bottom-Left)
    # ------------------------------------------------------------------
    dist_pm_128 = [c for c in o4a["t4a1_dist_pm_cases"] if c["pm_side"] == 128 and c["num_gpus"] == 4][0]
    cat_max = max(
        max(r["max_rel_com"], r["max_rel_bvel"], r["max_rel_vdisp"], r["max_rel_rmax"])
        for r in o5["parity_catalog"]
    )
    so_max = max(
        max(r["max_rel_m200c"], r["max_rel_r200c"], r["max_rel_m200m"], r["max_rel_r200m"])
        for r in o5["parity_so"]
    )
    vrms_3d = o4b["comparisons"]["g4_3d_hilbert_full_vs_g1_1d_full"]["rel_vrms_diff"]

    ladder = [
        ("Opt5 GPU SO vs 8T CPU (M200/R200)", so_max, GREEN),
        ("Opt4A Dist-PM Potential (4G vs 1G)", dist_pm_128["field_pot_max_rel_rms"], GREEN),
        ("Opt4A Dist-PM Force Field (4G vs 1G)", dist_pm_128["field_vec_max_rel_rms"], GREEN),
        ("Opt5 GPU FoF Catalog vs CPU", cat_max, GREEN),
        ("Opt4B 3D Hilbert vs 1D Slab (v_rms)", vrms_3d, BLUE),
        ("256^3 3D Hilb Eager P_delta max (vs CPU)", hilb_eager["matter_summary"]["max_abs_dev"], BLUE),
        ("1024^3 z=0 P_delta max (vs 256c CPU)", c56["matter_overall"]["max_abs_dev"], ORANGE),
        ("1024^3 z=1.76 P_delta max (vs 256c CPU)", c45["matter_overall"]["max_abs_dev"], ORANGE),
        ("256^3 3D Hilb Full P_delta max (vs CPU)", hilb_full["matter_summary"]["max_abs_dev"], ORANGE),
        ("1024^3 z=0 P_theta adj max (vs 256c CPU)", c56["momentum_adj_overall"]["max_abs_dev"], PURPLE),
    ]
    lab_c = [x[0] for x in ladder][::-1]
    val_c = np.array([x[1] for x in ladder][::-1])
    col_c = [x[2] for x in ladder][::-1]
    yc = np.arange(len(ladder))

    ax_c.hlines(yc, 1e-16, val_c, colors=col_c, lw=2.0, alpha=0.82)
    ax_c.scatter(val_c, yc, c=col_c, s=38, edgecolors="#111111", linewidths=0.7, zorder=4)
    ax_c.axvline(2.22e-16, color=SLATE, ls=":", lw=0.95)
    ax_c.axvline(1.19e-7, color=SLATE, ls=":", lw=0.95)
    ax_c.axvline(1e-3, color=GREEN, ls="-.", lw=1.0)

    n_l = len(ladder)
    ax_c.text(2.22e-16, n_l - 0.15, "fp64 eps", fontsize=6.7, color=SLATE, ha="left", va="bottom", fontweight="bold")
    ax_c.text(1.19e-7, n_l - 0.15, "fp32 eps", fontsize=6.7, color=SLATE, ha="center", va="bottom", fontweight="bold")
    ax_c.text(1e-3, n_l - 0.15, "0.1% limit", fontsize=6.7, color=GREEN, ha="center", va="bottom", fontweight="bold")

    ax_c.set_xscale("log")
    ax_c.set_xlim(1e-16, 1.5e-2)
    ax_c.set_ylim(-1.35, n_l + 0.55)
    ax_c.set_yticks(yc)
    ax_c.set_yticklabels(lab_c, fontsize=7.3)
    ax_c.set_xlabel("Maximum / RMS relative deviation vs. reference (log scale)")
    ax_c.set_title("(c) End-to-End Accuracy Ladder Across Subsystems")
    ax_c.grid(axis="x", which="both", linestyle="--", alpha=0.25)
    _despine(ax_c)

    for yi, v in zip(yc, val_c):
        if v > 6e-5:
            ax_c.text(v * 0.62, yi, f"{v:.2e}", va="center", ha="right", fontsize=6.9, color="#222222",
                      bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))
        else:
            ax_c.text(v * 1.55, yi, f"{v:.2e}", va="center", ha="left", fontsize=6.9, color="#222222")

    ax_c.text(
        0.50, -0.92,
        "Discrete parity: 7 GPU FoF modes & 4G 3D E2E -> 0 / 16.78M mismatches",
        ha="center", va="center", fontsize=6.3, fontweight="bold", color=GREEN,
        transform=ax_c.get_yaxis_transform(),
        bbox=dict(boxstyle="round,pad=0.20", fc="#F0FDF4", ec=GREEN, lw=0.75),
    )

    # ------------------------------------------------------------------
    # Panel (d): 1024^3 (1.07B) P_delta & P_theta vs 256-Core CHPC CPU
    # ------------------------------------------------------------------
    kd0, rd0, _ = _read_pk_csv("mgpu4-1024-0056-matter-power.csv")
    kd2, rd2, _ = _read_pk_csv("mgpu4-1024-0045-matter-power.csv")
    kt2, rt2, _ = _read_pk_csv("mgpu4-1024-0045-momentum-power.csv")
    kt0, rt0, _ = _read_pk_csv("mgpu4-1024-0056-momentum-adjusted-power.csv")

    ax_d.loglog(kd0, np.maximum(np.abs(rd0 - 1.0) * 100.0, 1e-5), color=BLUE, lw=1.7, marker="o", ms=2.4,
                label=f"z=0 Matter P_delta (max {c56['matter_overall']['max_abs_dev']*100:.5f}%)")
    ax_d.loglog(kd2, np.maximum(np.abs(rd2 - 1.0) * 100.0, 1e-5), color=ORANGE, lw=1.6, marker="s", ms=2.4,
                label=f"z=1.758 Matter P_delta (max {c45['matter_overall']['max_abs_dev']*100:.5f}%)")
    ax_d.loglog(kt2, np.maximum(np.abs(rt2 - 1.0) * 100.0, 1e-5), color=GREEN, lw=1.6, marker="^", ms=2.4,
                label=f"z=1.758 Raw P_theta (max {c45['momentum_overall']['max_abs_dev']*100:.4f}%)")
    ax_d.loglog(kt0, np.maximum(np.abs(rt0 - 1.0) * 100.0, 1e-5), color=PURPLE, lw=1.6, marker="d", ms=2.4,
                label=f"z=0 Aligned P_theta (max {c56['momentum_adj_overall']['max_abs_dev']*100:.4f}%)")

    ax_d.axhline(0.1, color=GREEN, lw=0.95, ls="-.", alpha=0.85)
    ax_d.text(0.32, 0.115, "0.1% precision target", fontsize=6.8, color=GREEN, fontweight="bold")
    ax_d.set_xlim(0.28, 16.5)
    ax_d.set_ylim(3e-4, 1.8)
    ax_d.set_xlabel(r"$k\ (h\,\mathrm{Mpc}^{-1})$")
    ax_d.set_ylabel(r"$|P^{\mathrm{GPU}} / P^{\mathrm{CPU}} - 1|\ [\%]$")
    ax_d.set_title(r"(d) $1024^3$ Power Spectra vs. 256-Core CHPC CPU")
    ax_d.grid(True, which="both", linestyle="--", alpha=0.25)
    _despine(ax_d)
    ax_d.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CBD5E1", fontsize=6.6)

    _save_both(fig, "fig6_accuracy_vs_cpu")


if __name__ == "__main__":
    make_fig5_speedup_vs_cpu()
    make_fig6_accuracy_vs_cpu()
