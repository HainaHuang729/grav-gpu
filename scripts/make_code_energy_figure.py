"""Generate multi-code runtime, hardware energy, and accuracy comparison figure (Fig 7)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT_DIR / "reports" / "code-energy-comparison.json"
OUT_DIR = ROOT_DIR / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavusans",
    "font.size": 8.4,
    "axes.titlesize": 9.4,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.3,
    "xtick.labelsize": 7.7,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 6.8,
    "axes.linewidth": 0.85,
    "axes.edgecolor": "#222222",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
PKD_LIGHT = "#E2A9C9"
CPU_DARK = "#475569"
CPU_SLATE = "#64748B"
HOST_GRAY = "#CBD5E1"
SKY = "#56B4E9"


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    data = json.loads(REPORT_PATH.read_text())
    by_id = {r["id"]: r for r in data["runs"]}

    selected = [
        ("gadget4_128c", "GADGET-4\n(128C CPU)", CPU_DARK),
        ("swift_cpu_256t", "CPU SWIFT\n(256T CPU)", CPU_SLATE),
        ("pkdgrav3_1g", "PKDGRAV3\n(1x A6000)", PURPLE),
        ("pkdgrav3_2g_mpi", "PKDGRAV3\n(2x A6000 MPI)", PKD_LIGHT),
        ("grav_gpu_1g_3d_lazy", "grav-gpu\n(1x A6000 Lazy)", BLUE),
        ("grav_gpu_2g_3d_lazy", "grav-gpu\n(2x A6000 Lazy)", GREEN),
        ("grav_gpu_4g_3d_lazy", "grav-gpu\n(4x A6000 Lazy)", ORANGE),
    ]

    labels = [s[1] for s in selected][::-1]
    runs = [by_id[s[0]] for s in selected][::-1]
    colors = [s[2] for s in selected][::-1]
    y = np.arange(len(selected))

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.9), layout="constrained")
    ax_t, ax_e, ax_a = axes

    # ------------------------------------------------------------------
    # Panel (a): Total Wall vs Pure Evolution Wall Time (s, log scale)
    # ------------------------------------------------------------------
    wall_vals = np.array([r["wall_s"] for r in runs])
    evo_vals = np.array([r["evo_s"] for r in runs])

    ax_t.barh(y, wall_vals, height=0.62, color=colors, alpha=0.35, edgecolor="#222222", linewidth=0.6)
    ax_t.barh(y, evo_vals, height=0.40, color=colors, edgecolor="#111111", linewidth=0.65)
    ax_t.set_xscale("log")
    ax_t.set_xlim(10, 26000)
    ax_t.set_yticks(y)
    ax_t.set_yticklabels(labels, fontsize=7.6)
    ax_t.set_xlabel("Time (s, log scale)")
    ax_t.set_title(r"(a) Wall & Evolution Time ($256^3,\ a=0.855\rightarrow 0.92$)")
    ax_t.grid(axis="x", which="both", linestyle="--", alpha=0.25)
    _despine(ax_t)

    t_swift_evo = by_id["swift_cpu_256t"]["evo_s"]
    for yi, r, col in zip(y, runs, colors):
        sp = t_swift_evo / r["evo_s"]
        is_best = r["id"] == "grav_gpu_4g_3d_lazy"
        txt = f"{r['evo_s']:.1f}s evo / {r['wall_s']:.1f}s ({sp:.1f}x)" if r["code"] == "grav-gpu" else f"{r['evo_s']:.0f}s / {r['wall_s']:.0f}s ({sp:.2f}x)"
        ax_t.text(
            r["wall_s"] * 1.13, yi, txt,
            va="center", ha="left", fontsize=6.9,
            fontweight="bold" if is_best or r["id"] == "swift_cpu_256t" else "normal",
            color=ORANGE if is_best else "#222222",
        )

    leg_t = [
        Patch(facecolor=CPU_SLATE, edgecolor="#111111", label="Evolution / compute time"),
        Patch(facecolor=CPU_SLATE, alpha=0.35, edgecolor="#222222", label="Total process wall time (incl. I/O)"),
    ]
    ax_t.legend(handles=leg_t, loc="lower right", frameon=True, framealpha=0.95, edgecolor="#CBD5E1", fontsize=6.5)

    # ------------------------------------------------------------------
    # Panel (b): Energy Cost per Run (Wh)
    # ------------------------------------------------------------------
    gpu_wh = np.array([r["gpu_energy_wh"] if r["gpu_energy_wh"] is not None else 0.0 for r in runs])
    host_wh = np.array([
        (r["node_est_energy_wh"] - r["gpu_energy_wh"]) if r["gpu_energy_wh"] is not None else r["node_scaled_energy_wh"]
        for r in runs
    ])
    tot_wh = gpu_wh + host_wh
    ref_700w = [r.get("node_700w_ref_wh") for r in runs]

    ax_e.barh(y, gpu_wh, height=0.58, color=colors, edgecolor="#111111", linewidth=0.65)
    ax_e.barh(y, host_wh, left=gpu_wh, height=0.58, color=HOST_GRAY, hatch="///", edgecolor="#334155", linewidth=0.6)

    for yi, r7 in zip(y, ref_700w):
        if r7 is not None:
            ax_e.scatter([r7], [yi], marker="D", s=34, color="white", edgecolors="#111111", linewidths=1.1, zorder=5)

    ax_e.set_xscale("log")
    ax_e.set_xlim(1.5, 3200)
    ax_e.set_yticks(y)
    ax_e.set_yticklabels([])
    ax_e.set_xlabel("Total energy per run (Wh, log scale)")
    ax_e.set_title("(b) Measured NVML GPU + Modeled Host Energy")
    ax_e.grid(axis="x", which="both", linestyle="--", alpha=0.25)
    _despine(ax_e)

    e_swift = by_id["swift_cpu_256t"]["node_scaled_energy_wh"]
    for yi, r, gw, tw in zip(y, runs, gpu_wh, tot_wh):
        sav = e_swift / tw
        if gw > 0:
            txt = f"{gw:.2f} GPU / {tw:.2f} Wh ({sav:.1f}x)" if tw < 20 else f"{gw:.1f} GPU / {tw:.1f} Wh ({sav:.1f}x)"
        else:
            txt = f"{tw:.1f} Wh (700W: {r['node_700w_ref_wh']:.0f} Wh)"
        x_anchor = max(tw, r.get("node_700w_ref_wh") or 0) * 1.13
        is_best = r["id"] == "grav_gpu_2g_3d_lazy"
        ax_e.text(
            x_anchor, yi, txt,
            va="center", ha="left", fontsize=6.9,
            fontweight="bold" if is_best or r["id"] == "grav_gpu_4g_3d_lazy" else "normal",
            color=GREEN if is_best else "#222222",
        )

    leg_e = [
        Patch(facecolor=BLUE, edgecolor="#111111", label="Measured NVML GPU (Wh)"),
        Patch(facecolor=HOST_GRAY, hatch="///", edgecolor="#334155", label="Modeled CPU/host power"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white", markeredgecolor="#111111", markersize=5.5, label="CPU flat 700 W node ref"),
    ]
    ax_e.legend(handles=leg_e, loc="lower right", frameon=True, framealpha=0.95, edgecolor="#CBD5E1", fontsize=6.4)

    # ------------------------------------------------------------------
    # Panel (c): Power Spectrum Accuracy vs CPU SWIFT 256T
    # ------------------------------------------------------------------
    acc_items = [
        ("GADGET-4\n(128C CPU)", by_id["gadget4_128c"]["accuracy"]["p_delta_max_abs_dev"] * 100.0, CPU_DARK),
        ("PKDGRAV3\n(1G / 2G MPI)", by_id["pkdgrav3_2g_mpi"]["accuracy"]["p_delta_max_abs_dev"] * 100.0, PURPLE),
        ("grav-gpu\n(1G / 2G / 4G Lazy)", by_id["grav_gpu_4g_3d_lazy"]["accuracy"]["p_delta_max_abs_dev"] * 100.0, ORANGE),
        ("grav-gpu\n(2G / 4G Eager)", by_id["grav_gpu_4g_3d_eager"]["accuracy"]["p_delta_max_abs_dev"] * 100.0, BLUE),
    ]
    acc_labels = [x[0] for x in acc_items][::-1]
    acc_vals = np.array([x[1] for x in acc_items][::-1])
    acc_cols = [x[2] for x in acc_items][::-1]
    ya = np.arange(len(acc_items))

    ax_a.hlines(ya, 1e-4, acc_vals, colors=acc_cols, lw=2.6, alpha=0.85)
    ax_a.scatter(acc_vals, ya, c=acc_cols, s=54, edgecolors="#111111", linewidths=0.8, zorder=4)
    ax_a.axvline(0.1, color=GREEN, ls="-.", lw=1.0)
    ax_a.axvline(1.0, color=CPU_SLATE, ls=":", lw=1.0)
    ax_a.text(0.108, len(acc_items) - 0.35, "0.1%", fontsize=6.9, color=GREEN, fontweight="bold")
    ax_a.text(1.08, len(acc_items) - 0.35, "1%", fontsize=6.9, color=CPU_SLATE, fontweight="bold")

    ax_a.set_xscale("log")
    ax_a.set_xlim(2e-4, 3.0)
    ax_a.set_ylim(-0.90, len(acc_items) - 0.15)
    ax_a.set_yticks(ya)
    ax_a.set_yticklabels(acc_labels, fontsize=7.5)
    ax_a.set_xlabel(r"$\max_k |P_\delta / P_\delta^{\mathrm{SWIFT}} - 1|\ [\%]$")
    ax_a.set_title("(c) Matter Spectrum Deviation vs. SWIFT 256T")
    ax_a.grid(axis="x", which="both", linestyle="--", alpha=0.25)
    _despine(ax_a)

    for yi, v in zip(ya, acc_vals):
        ax_a.text(
            v * 0.72 if v > 0.08 else v * 1.35,
            yi + (0.14 if v > 0.08 else 0.0),
            f"{v:.4f}%" if v < 0.01 else f"{v:.3f}%",
            va="center",
            ha="right" if v > 0.08 else "left",
            fontsize=7.0,
            color="#222222",
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.8),
        )

    ax_a.text(
        0.5, 0.05,
        r"$1024^3$ full history ($z=200\rightarrow 0$): 28.3-29.5 kWh (4G) vs 213.1 kWh (CPU)",
        transform=ax_a.transAxes, ha="center", va="bottom",
        fontsize=6.3, color=BLUE, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", fc="#EFF6FF", ec=BLUE, lw=0.7),
    )

    for ext in ("pdf", "png"):
        out_path = OUT_DIR / f"fig7_code_energy_comparison.{ext}"
        fig.savefig(out_path, dpi=300 if ext == "png" else None, bbox_inches="tight", pad_inches=0.07)
        print(f"Wrote {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
