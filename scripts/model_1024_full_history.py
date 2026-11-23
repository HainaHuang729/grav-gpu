"""Full-history (z=200->0 and z=50->0) 1024^3 compute, runtime, and energy scaling model.

Calibrates 1x RTX A6000 and 4x RTX A6000 (NVLink 12-slab VRAM-resident) against directly
measured 1024^3 runs at z~0 (0055->0056) and z~2 (0044->0045) and compares against the
complete 20,194-step archival 256-core CHPC CPU run in data/timesteps_1024.txt.
"""
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
TIMESTEPS_PATH = ROOT / "data" / "timesteps_1024.txt"
OUT_JSON = REPORTS / "full-history-1024-model.json"

N_PARTICLES = 1024**3

EPOCHS = [
    ("z=200->50", 200.5, 50.0),
    ("z=50->10", 50.0, 10.0),
    ("z=10->4", 10.0, 4.0),
    ("z=4->1.5", 4.0, 1.5),
    ("z=1.5->0.5", 1.5, 0.5),
    ("z=0.5->0", 0.5, -0.1),
]


def parse_clean_timesteps(path: Path):
    """Parse SWIFT timesteps_1024.txt, keeping the last valid 15-column row per step."""
    rows = {}
    raw_lines = 0
    skipped_lines = 0
    for line in path.read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        raw_lines += 1
        toks = line.split()
        if len(toks) != 15 or not toks[0].isdigit():
            skipped_lines += 1
            continue
        step = int(toks[0])
        props = int(toks[13])
        rows[step] = {
            "step": step,
            "time_internal": float(toks[1]),
            "a": float(toks[2]),
            "z": float(toks[3]),
            "dt_internal": float(toks[4]),
            "g_updates": int(toks[8]),
            "wall_s": float(toks[12]) / 1000.0,
            "props": props,
            "is_pm": bool(props & 256),
            "is_dump": bool(props & (16 | 128)),
            "dead_s": float(toks[14]) / 1000.0,
        }
    steps = [rows[k] for k in sorted(rows)]
    assert len(steps) == 20194, f"Expected 20194 steps, got {len(steps)}"
    assert [s["step"] for s in steps] == list(range(20194)), "Non-consecutive steps"
    return steps, raw_lines, skipped_lines


def verify_cpu_windows(steps):
    """Cross-check parsed CPU timesteps against saved 0055->0056 and 0044->0045 JSONs."""
    c56 = json.loads((REPORTS / "mgpu4-1024-0056-comparison.json").read_text())["timing_historical_cpu"]
    c45 = json.loads((REPORTS / "historical-cpu-0044-0045.json").read_text())

    w56 = [s for s in steps if c56["first_step"] <= s["step"] <= c56["last_step"]]
    assert len(w56) == c56["steps"]
    assert abs(sum(s["wall_s"] for s in w56) - c56["sum_step_s"]) < 1e-4

    w45 = [s for s in steps if c45["first_step"] <= s["step"] <= c45["last_step"]]
    assert len(w45) == c45["steps"]
    assert abs(sum(s["wall_s"] for s in w45) - c45["sum_step_s"]) < 1e-4


def assign_pure_gravity_split(steps):
    """Separate inline snapshot/FoF dump overhead from pure gravity step time per epoch."""
    epoch_stats = []
    for name, zhi, zlo in EPOCHS:
        seg = [s for s in steps if zlo <= s["z"] < zhi]
        pm_clean = [s["wall_s"] for s in seg if s["is_pm"] and not s["is_dump"]]
        non_clean = [s["wall_s"] for s in seg if not s["is_pm"] and not s["is_dump"]]
        pm_med = statistics.median(pm_clean) if pm_clean else 0.0
        non_med = statistics.median(non_clean) if non_clean else 0.0
        for s in seg:
            if s["is_dump"]:
                base = pm_med if s["is_pm"] else non_med
                base = min(s["wall_s"], base) if base > 0 else min(s["wall_s"], 100.0)
                s["cpu_pure_s"] = base
                s["cpu_dump_s"] = s["wall_s"] - base
            else:
                s["cpu_pure_s"] = s["wall_s"]
                s["cpu_dump_s"] = 0.0
        epoch_stats.append({
            "epoch": name,
            "steps": len(seg),
            "pm_steps": sum(1 for s in seg if s["is_pm"]),
            "dump_steps": sum(1 for s in seg if s["is_dump"]),
            "mean_active_particles": statistics.mean(s["g_updates"] for s in seg),
            "cpu_total_s": sum(s["wall_s"] for s in seg),
            "cpu_pure_evo_s": sum(s["cpu_pure_s"] for s in seg),
            "cpu_dump_overhead_s": sum(s["cpu_dump_s"] for s in seg),
            "cpu_pm_median_s": pm_med,
            "cpu_non_pm_median_s": non_med,
        })
    return epoch_stats


def parse_gpu_log(path: Path):
    """Extract STEP regressions, MEMORY_PLAN, SANITY, and RESULT from a GPU evolve log."""
    steps = []
    mem_plan = gpu_input = sanity = result = None
    for line in path.read_text().splitlines():
        if line.startswith("STEP "):
            steps.append(json.loads(line.split(" ", 1)[1]))
        elif line.startswith("MEMORY_PLAN "):
            mem_plan = json.loads(line.split(" ", 1)[1])
        elif line.startswith("GPU_INPUT "):
            gpu_input = json.loads(line.split(" ", 1)[1])
        elif line.startswith("SANITY "):
            sanity = json.loads(line.split(" ", 1)[1])
        elif line.startswith("RESULT "):
            result = json.loads(line.split(" ", 1)[1])
    assert result is not None and steps and sanity["bad_count"] == 0
    pm = [s for s in steps if s["pm"]]
    non = [s for s in steps if not s["pm"]]
    xs = [s["active"] / float(N_PARTICLES) for s in non]
    ys = [s["wall_s"] for s in non]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    varx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / varx if varx > 0 else 0.0
    intercept = my - slope * mx
    return {
        "log_file": str(path.relative_to(ROOT)),
        "a_begin": gpu_input["a_begin"],
        "a_end": gpu_input["a_end"],
        "a_mid": 0.5 * (gpu_input["a_begin"] + gpu_input["a_end"]),
        "steps": len(steps),
        "pm_count": len(pm),
        "non_pm_count": len(non),
        "evolution_wall_s": result["evolution_wall_s"],
        "whole_program_s": result["whole_program_s"],
        "pm_mean_s": statistics.mean(s["wall_s"] for s in pm),
        "pm_median_s": statistics.median(s["wall_s"] for s in pm),
        "step0_pm_s": pm[0]["wall_s"],
        "non_pm_intercept_s": intercept,
        "non_pm_slope_per_1024cubed_s": slope,
        "min_free_vram_gb": mem_plan["free_vram_bytes"] / 1e9,
    }


def parse_gpu_power(csv_path: Path, allocated_indices):
    """Compute allocated-GPU power draw (W) filtering strictly to allocated CUDA indices."""
    by_gpu = {idx: [] for idx in allocated_indices}
    with csv_path.open() as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            idx = int(row["index"])
            if idx not in by_gpu:
                continue
            pw = float(row["power.draw [W]"].split()[0])
            util = float(row["utilization.gpu [%]"].split()[0])
            mem = float(row["memory.used [MiB]"].split()[0])
            by_gpu[idx].append((pw, util, mem))
    per_gpu = {
        str(idx): {
            "samples": len(vals),
            "mean_w": statistics.mean(v[0] for v in vals),
            "max_w": max(v[0] for v in vals),
            "mean_util_pct": statistics.mean(v[1] for v in vals),
            "max_mem_mib": max(v[2] for v in vals),
        }
        for idx, vals in by_gpu.items()
    }
    return {
        "csv_file": str(csv_path.relative_to(ROOT)),
        "allocated_indices": allocated_indices,
        "total_mean_gpu_w": sum(v["mean_w"] for v in per_gpu.values()),
        "per_gpu": per_gpu,
    }


def lerp_clamped(a, a_lo, val_lo, a_hi, val_hi):
    if a <= a_lo:
        return val_lo
    if a >= a_hi:
        return val_hi
    return val_lo + (val_hi - val_lo) * ((a - a_lo) / (a_hi - a_lo))


def project_step_model(steps, anchors_lo, anchors_hi, step_ratio_lo, step_ratio_hi, cpu_to_gpu_pm_early_override=None):
    """Project GPU pure-evolution wall time via active-particle + PM regression across every step."""
    a_lo = anchors_lo["a_mid"]
    a_hi = anchors_hi["a_mid"]
    total_s = 0.0
    pm_s = 0.0
    non_pm_s = 0.0
    for s in steps:
        a = s["a"]
        act_frac = s["g_updates"] / float(N_PARTICLES)
        if s["is_pm"]:
            if cpu_to_gpu_pm_early_override is not None and s["z"] >= 50.0:
                base_step = anchors_lo["pm_mean_s"] * cpu_to_gpu_pm_early_override
            else:
                base_step = lerp_clamped(a, a_lo, anchors_lo["pm_mean_s"], a_hi, anchors_hi["pm_mean_s"])
            pm_s += base_step
        else:
            inter = lerp_clamped(a, a_lo, anchors_lo["non_pm_intercept_s"], a_hi, anchors_hi["non_pm_intercept_s"])
            slope = lerp_clamped(a, a_lo, anchors_lo["non_pm_slope_per_1024cubed_s"], a_hi, anchors_hi["non_pm_slope_per_1024cubed_s"])
            base_step = max(0.05, inter + slope * act_frac)
            non_pm_s += base_step
        # Step-count ratio (GPU dynamic-PM power-of-2 steps / CPU steps): 1.0 at z>=50 (all PM)
        if s["z"] >= 50.0:
            mult = 1.0
        else:
            mult = lerp_clamped(a, a_lo, step_ratio_lo, a_hi, step_ratio_hi)
        total_s += base_step * mult
    return {"total_s": total_s, "hours": total_s / 3600.0}


def summarize_range(steps, label, gpu_specs, cpu_node_w=700.0):
    cpu_total_s = sum(s["wall_s"] for s in steps)
    cpu_pure_s = sum(s["cpu_pure_s"] for s in steps)
    cpu_dump_s = sum(s["cpu_dump_s"] for s in steps)
    out = {
        "range": label,
        "cpu_steps": len(steps),
        "cpu_pm_steps": sum(1 for s in steps if s["is_pm"]),
        "cpu_dump_steps": sum(1 for s in steps if s["is_dump"]),
        "cpu_256core": {
            "pure_evo_s": cpu_pure_s,
            "pure_evo_hours": cpu_pure_s / 3600.0,
            "dump_overhead_s": cpu_dump_s,
            "total_wall_s": cpu_total_s,
            "total_wall_hours": cpu_total_s / 3600.0,
            "core_hours_total": 256.0 * cpu_total_s / 3600.0,
            "assumed_node_w": cpu_node_w,
            "energy_pure_kwh": (cpu_node_w * cpu_pure_s / 3600.0) / 1000.0,
            "energy_total_kwh": (cpu_node_w * cpu_total_s / 3600.0) / 1000.0,
        },
        "gpu_models": {},
    }
    for name, spec in gpu_specs.items():
        # Model 1: Conservative clamped interval speedup over CPU pure_evo
        m1_s = sum(
            s["cpu_pure_s"] / lerp_clamped(s["a"], spec["a_lo"], spec["speedup_lo"], spec["a_hi"], spec["speedup_hi"])
            for s in steps
        )
        # Model 2: Physics-calibrated step-level (active + PM) with dynamic-PM step multiplier
        m2 = project_step_model(
            steps,
            spec["anchor_lo"],
            spec["anchor_hi"],
            spec["step_ratio_lo"],
            spec["step_ratio_hi"],
        )
        m1_h = m1_s / 3600.0
        m2_h = m2["hours"]
        gpu_w = spec["mean_gpu_w"]
        node_w = spec["mean_node_w"]
        out["gpu_models"][name] = {
            "clamped_interval_model": {
                "pure_evo_s": m1_s,
                "pure_evo_hours": m1_h,
                "pure_evo_days": m1_h / 24.0,
                "effective_speedup_vs_cpu_pure": cpu_pure_s / m1_s,
                "gpu_only_kwh": gpu_w * m1_h / 1000.0,
                "whole_node_kwh": node_w * m1_h / 1000.0,
                "energy_ratio_node_vs_cpu_pure": (node_w * m1_h) / (cpu_node_w * cpu_pure_s / 3600.0),
            },
            "step_physics_model": {
                "pure_evo_s": m2["total_s"],
                "pure_evo_hours": m2_h,
                "pure_evo_days": m2_h / 24.0,
                "effective_speedup_vs_cpu_pure": cpu_pure_s / m2["total_s"],
                "gpu_only_kwh": gpu_w * m2_h / 1000.0,
                "whole_node_kwh": node_w * m2_h / 1000.0,
                "energy_ratio_node_vs_cpu_pure": (node_w * m2_h) / (cpu_node_w * cpu_pure_s / 3600.0),
            },
        }
    return out


def main():
    steps, raw_lines, skipped_lines = parse_clean_timesteps(TIMESTEPS_PATH)
    verify_cpu_windows(steps)
    epoch_stats = assign_pure_gravity_split(steps)

    # Parse all measured 1024^3 GPU evolution anchors & telemetry
    gpu1_z0 = parse_gpu_log(REPORTS / "slab-1024-0056-evolve.log")
    gpu1_z2_probe = parse_gpu_log(REPORTS / "slab1-1024-0044-probe.log")
    gpu4_z0 = parse_gpu_log(REPORTS / "mgpu4-1024-0056-evolve.log")
    gpu4_z2_solo = parse_gpu_log(REPORTS / "run-489937" / "mgpu4-1024-0045-evolve.log")
    gpu4_z2_shared = parse_gpu_log(REPORTS / "mgpu4-1024-0045-evolve.log")

    pwr_1_z0 = parse_gpu_power(REPORTS / "slab-1024-0056-device.csv", [1])
    pwr_4_z0 = parse_gpu_power(REPORTS / "mgpu4-1024-0056-gpu-telemetry.csv", [0, 1, 2, 3])
    pwr_4_z2_solo = parse_gpu_power(REPORTS / "run-489937" / "mgpu4-1024-0045-gpu-telemetry.csv", [0, 1, 2, 3])
    pwr_4_z2_shared = parse_gpu_power(REPORTS / "mgpu4-1024-0045-gpu-telemetry.csv", [0, 1, 2, 3])

    # Exact measured CPU pure evolution times on the two benchmark intervals
    c45 = json.loads((REPORTS / "historical-cpu-0044-0045.json").read_text())
    cpu_0044_0045_total_s = c45["sum_step_s"]
    cpu_0044_0045_pure_s = sum(s["cpu_pure_s"] for s in steps if c45["first_step"] <= s["step"] <= c45["last_step"])
    c56 = json.loads((REPORTS / "mgpu4-1024-0056-comparison.json").read_text())["timing_historical_cpu"]
    cpu_0055_0056_pure_s = c56["sum_step_s"]

    # Estimate 1x A6000 full 0044->0045 time using the 17-step probe + 910-step GPU workload distribution
    solo_steps_z2 = [json.loads(l.split(" ", 1)[1]) for l in (REPORTS / "run-489937" / "mgpu4-1024-0045-evolve.log").read_text().splitlines() if l.startswith("STEP ")]
    gpu1_z2_est_s = sum(
        gpu1_z2_probe["pm_mean_s"] if st["pm"] else (gpu1_z2_probe["non_pm_intercept_s"] + gpu1_z2_probe["non_pm_slope_per_1024cubed_s"] * (st["active"] / N_PARTICLES))
        for st in solo_steps_z2
    )

    gpu_specs = {
        "1x_RTX_A6000_slab_resident": {
            "a_lo": gpu4_z2_solo["a_mid"],
            "a_hi": gpu1_z0["a_mid"],
            "speedup_lo": cpu_0044_0045_pure_s / gpu1_z2_est_s,
            "speedup_hi": cpu_0055_0056_pure_s / gpu1_z0["evolution_wall_s"],
            "anchor_lo": gpu1_z2_probe,
            "anchor_hi": gpu1_z0,
            "step_ratio_lo": gpu4_z2_solo["steps"] / c45["steps"],
            "step_ratio_hi": gpu1_z0["steps"] / c56["steps"],
            "mean_gpu_w": pwr_1_z0["total_mean_gpu_w"],
            "mean_node_w": pwr_1_z0["total_mean_gpu_w"] + 100.0,
        },
        "4x_RTX_A6000_NVLink_solo": {
            "a_lo": gpu4_z2_solo["a_mid"],
            "a_hi": gpu4_z0["a_mid"],
            "speedup_lo": cpu_0044_0045_pure_s / gpu4_z2_solo["evolution_wall_s"],
            "speedup_hi": cpu_0055_0056_pure_s / gpu4_z0["evolution_wall_s"],
            "anchor_lo": gpu4_z2_solo,
            "anchor_hi": gpu4_z0,
            "step_ratio_lo": gpu4_z2_solo["steps"] / c45["steps"],
            "step_ratio_hi": gpu4_z0["steps"] / c56["steps"],
            "mean_gpu_w": 0.5 * (pwr_4_z2_solo["total_mean_gpu_w"] + pwr_4_z0["total_mean_gpu_w"]),
            "mean_node_w": 0.5 * (pwr_4_z2_solo["total_mean_gpu_w"] + pwr_4_z0["total_mean_gpu_w"]) + 120.0,
        },
        "4x_RTX_A6000_NVLink_shared_node": {
            "a_lo": gpu4_z2_shared["a_mid"],
            "a_hi": gpu4_z0["a_mid"],
            "speedup_lo": cpu_0044_0045_pure_s / gpu4_z2_shared["evolution_wall_s"],
            "speedup_hi": cpu_0055_0056_pure_s / gpu4_z0["evolution_wall_s"],
            "anchor_lo": gpu4_z2_shared,
            "anchor_hi": gpu4_z0,
            "step_ratio_lo": gpu4_z2_shared["steps"] / c45["steps"],
            "step_ratio_hi": gpu4_z0["steps"] / c56["steps"],
            "mean_gpu_w": 0.5 * (pwr_4_z2_shared["total_mean_gpu_w"] + pwr_4_z0["total_mean_gpu_w"]),
            "mean_node_w": 0.5 * (pwr_4_z2_shared["total_mean_gpu_w"] + pwr_4_z0["total_mean_gpu_w"]) + 120.0,
        },
    }

    z50_steps = [s for s in steps if s["a"] >= 1.0 / 51.0]
    post_hdf5_mean_s = 0.5 * (
        float((REPORTS / "mgpu4-1024-0056-tohdf5.seconds").read_text().strip())
        + float((REPORTS / "mgpu4-1024-0045-tohdf5.seconds").read_text().strip())
    )
    post_fof_96_s = float((REPORTS / "mgpu4-1024-0045-fof.seconds").read_text().strip())
    post_fof_64_s = float((REPORTS / "mgpu4-1024-0056-fof.seconds").read_text().strip())

    report = {
        "timesteps_summary": {
            "raw_noncomment_lines": raw_lines,
            "skipped_malformed_lines": skipped_lines,
            "clean_unique_steps": len(steps),
            "z50_start_step": z50_steps[0]["step"],
            "z50_start_a": z50_steps[0]["a"],
            "z50_start_z": z50_steps[0]["z"],
            "epoch_breakdown": epoch_stats,
        },
        "measured_anchors": {
            "cpu_0044_0045_total_s": cpu_0044_0045_total_s,
            "cpu_0044_0045_pure_evo_s": cpu_0044_0045_pure_s,
            "cpu_0055_0056_pure_evo_s": cpu_0055_0056_pure_s,
            "gpu1_z2_probe_projected_0044_0045_s": gpu1_z2_est_s,
            "gpu1_z0": gpu1_z0,
            "gpu1_z2_probe": gpu1_z2_probe,
            "gpu4_z0": gpu4_z0,
            "gpu4_z2_solo": gpu4_z2_solo,
            "gpu4_z2_shared": gpu4_z2_shared,
        },
        "measured_power": {
            "gpu1_z0": pwr_1_z0,
            "gpu4_z0": pwr_4_z0,
            "gpu4_z2_solo": pwr_4_z2_solo,
            "gpu4_z2_shared": pwr_4_z2_shared,
        },
        "postprocessing_per_snapshot_measured": {
            "to_hdf5_0056_s": float((REPORTS / "mgpu4-1024-0056-tohdf5.seconds").read_text().strip()),
            "to_hdf5_0045_s": float((REPORTS / "mgpu4-1024-0045-tohdf5.seconds").read_text().strip()),
            "to_hdf5_mean_s": post_hdf5_mean_s,
            "fof_0056_64threads_s": post_fof_64_s,
            "fof_0045_96threads_s": post_fof_96_s,
            "cpu_archival_56_inline_dumps_s": sum(e["cpu_dump_overhead_s"] for e in epoch_stats),
        },
        "projections": {
            "z200_to_z0": summarize_range(steps, "z=200->0", gpu_specs),
            "z50_to_z0": summarize_range(z50_steps, "z=50->0", gpu_specs),
        },
    }
    OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {OUT_JSON}")
    for rkey in ("z200_to_z0", "z50_to_z0"):
        r = report["projections"][rkey]
        c = r["cpu_256core"]
        print(f"\n=== {r['range']} ({r['cpu_steps']} CPU steps) ===")
        print(f"CPU 256-core: pure={c['pure_evo_hours']:.2f}h total_w_dumps={c['total_wall_hours']:.2f}h ({c['core_hours_total']:.0f} core-h) energy_pure={c['energy_pure_kwh']:.1f}kWh")
        for gname, gm in r["gpu_models"].items():
            m1, m2 = gm["clamped_interval_model"], gm["step_physics_model"]
            print(f"  {gname}: clamped={m1['pure_evo_hours']:.2f}h ({m1['effective_speedup_vs_cpu_pure']:.2f}x, {m1['whole_node_kwh']:.1f}kWh) | step_physics={m2['pure_evo_hours']:.2f}h ({m2['effective_speedup_vs_cpu_pure']:.2f}x, {m2['whole_node_kwh']:.1f}kWh)")


if __name__ == "__main__":
    main()
