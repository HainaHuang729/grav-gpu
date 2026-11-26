"""Generalized 1024^3 scientific accuracy comparison: GPU vs archival 256-core CPU.
Computes ID-matched particle position/velocity errors, 256^3 CIC matter P_delta(k) and
momentum divergence P_theta(k) spectra (raw + optional kick-aligned), FoF HMF & top-halo
comparison, and GPU vs CPU timing summary across any redshift snapshot."""
import argparse
import atexit
import csv
import ctypes
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

N = 1024**3
GRID = 256
CHUNK = 1 << 20
ID_STRIDE = 65521
H = 0.6736
HALO_EDGES = np.array([32, 64, 128, 256, 512, 1024, 4096, 16384, 65536, 262144])


def epoch(file):
    return float(np.asarray(file["Header"].attrs["Scale-factor"]).item())


def hubble_a(a):
    return 67.36 * np.sqrt(0.3153 / (a**3) + 0.6847)


def kick_factor_vec(left, right, a_end):
    def integrand(a):
        return 1.0 / (a * a * hubble_a(a))

    return (
        (right - left)
        / 6.0
        * (integrand(left) + 4.0 * integrand((left + right) * 0.5) + integrand(right))
        / a_end
    )


_CIC_FN = "UNSET"


def _get_cic_fn():
    """Load prebuilt or JIT-compiled single-pass C CIC kernel if available."""
    global _CIC_FN
    if _CIC_FN != "UNSET":
        return _CIC_FN
    lib_path = os.environ.get("GRAV_CIC_LIB")
    if lib_path:
        lib = ctypes.CDLL(lib_path)
    else:
        src = Path(__file__).with_name("cic_deposit.c")
        if not src.exists():
            src = Path(__file__).resolve().parent.parent / "src" / "cic_deposit.c"
        cc = shutil.which("cc") or shutil.which("gcc")
        if not src.exists() or not cc:
            _CIC_FN = None
            return None
        tmp_dir = tempfile.mkdtemp(prefix="grav_cic_")
        atexit.register(shutil.rmtree, tmp_dir, True)
        so_path = Path(tmp_dir) / "libcic_deposit.so"
        try:
            subprocess.run(
                [cc, "-O3", "-shared", "-fPIC", str(src), "-lm", "-o", str(so_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            lib = ctypes.CDLL(str(so_path))
        except Exception as exc:
            print(f"Warning: CIC C compilation failed ({exc}); using NumPy fallback", file=sys.stderr)
            _CIC_FN = None
            return None
    fn = lib.deposit_cic_c
    fn.argtypes = [ctypes.c_int64, ctypes.c_int] + [ctypes.c_void_p] * 11
    fn.restype = None
    _CIC_FN = fn
    return fn


def _deposit_cic_numpy(density, mom, coordinates, velocities, box, mom_adj=None, vel_adj=None):
    points = np.remainder(coordinates / box, 1.0) * GRID
    lower = points.astype(np.int32)
    frac = points - lower
    higher = (lower + 1) % GRID
    lower %= GRID
    vx = velocities[:, 0]
    vy = velocities[:, 1]
    vz = velocities[:, 2]
    has_adj = mom_adj is not None and vel_adj is not None
    if has_adj:
        vax = vel_adj[:, 0]
        vay = vel_adj[:, 1]
        vaz = vel_adj[:, 2]
        mom_adj_flats = [mom_adj[c].reshape(-1) for c in range(3)]
    n_grid = GRID * GRID * GRID
    g2 = GRID * GRID
    dens_flat = density.reshape(-1)
    mom_flats = [mom[c].reshape(-1) for c in range(3)]
    for x, wx in ((lower[:, 0], 1.0 - frac[:, 0]), (higher[:, 0], frac[:, 0])):
        x_off = x.astype(np.int64) * g2
        for y, wy in ((lower[:, 1], 1.0 - frac[:, 1]), (higher[:, 1], frac[:, 1])):
            wxy = wx * wy
            xy_off = x_off + y.astype(np.int64) * GRID
            for z, wz in ((lower[:, 2], 1.0 - frac[:, 2]), (higher[:, 2], frac[:, 2])):
                w = wxy * wz
                flat = xy_off + z
                dens_flat += np.bincount(flat, weights=w, minlength=n_grid)
                mom_flats[0] += np.bincount(flat, weights=w * vx, minlength=n_grid)
                mom_flats[1] += np.bincount(flat, weights=w * vy, minlength=n_grid)
                mom_flats[2] += np.bincount(flat, weights=w * vz, minlength=n_grid)
                if has_adj:
                    mom_adj_flats[0] += np.bincount(flat, weights=w * vax, minlength=n_grid)
                    mom_adj_flats[1] += np.bincount(flat, weights=w * vay, minlength=n_grid)
                    mom_adj_flats[2] += np.bincount(flat, weights=w * vaz, minlength=n_grid)


def deposit_cic_fields(density, mom, coordinates, velocities, box, mom_adj=None, vel_adj=None):
    fn = _get_cic_fn()
    if fn is None:
        _deposit_cic_numpy(density, mom, coordinates, velocities, box, mom_adj=mom_adj, vel_adj=vel_adj)
        return
    has_adj = mom_adj is not None and vel_adj is not None
    assert density.dtype == np.float64 and density.flags["C_CONTIGUOUS"]
    assert mom.dtype == np.float64 and mom.flags["C_CONTIGUOUS"]
    coords64 = np.ascontiguousarray(coordinates, dtype=np.float64)
    vels64 = np.ascontiguousarray(velocities, dtype=np.float64)
    box64 = np.ascontiguousarray(box, dtype=np.float64)
    if has_adj:
        assert mom_adj.dtype == np.float64 and mom_adj.flags["C_CONTIGUOUS"]
        va64 = np.ascontiguousarray(vel_adj, dtype=np.float64)
        va_ptr = va64.ctypes.data
        ma0_ptr, ma1_ptr, ma2_ptr = mom_adj[0].ctypes.data, mom_adj[1].ctypes.data, mom_adj[2].ctypes.data
    else:
        va_ptr = ma0_ptr = ma1_ptr = ma2_ptr = None
    fn(
        len(coords64),
        GRID,
        coords64.ctypes.data,
        vels64.ctypes.data,
        va_ptr,
        box64.ctypes.data,
        density.ctypes.data,
        mom[0].ctypes.data,
        mom[1].ctypes.data,
        mom[2].ctypes.data,
        ma0_ptr,
        ma1_ptr,
        ma2_ptr,
    )


def compute_short_long_kicks(short_chunk, long_chunk, a_begin, a_end, soft_mpc, grav_g, span_override, dt_rms):
    fs = short_chunk.astype(np.float64) * grav_g
    fl = long_chunk.astype(np.float64)
    span = np.log(a_end / a_begin)
    cap = span_override if span_override is not None else span / 16.0
    pm_prefactor = kick_factor_vec(a_end * np.exp(-cap * 0.5), a_end, a_end)
    current = np.linalg.norm(fs + fl, axis=1) / (a_end**2)
    dt_grav = np.sqrt(2.0 * 0.025 * soft_mpc * a_end / np.maximum(current, 1e-30))
    dln = hubble_a(a_end) * np.minimum(dt_grav, dt_rms)
    widths = np.minimum(cap, span * np.exp2(np.floor(np.log2(np.maximum(dln / span, 2.0**-54)))))
    s_vec = fs * kick_factor_vec(a_end * np.exp(-widths * 0.5), a_end, a_end)[:, None]
    l_vec = fl * pm_prefactor
    return s_vec, l_vec


def read_snapshot(path, force3_path=None, kick_cfg=None):
    density = np.zeros((GRID,) * 3, dtype=np.float64)
    mom = np.zeros((3, GRID, GRID, GRID), dtype=np.float64)
    mom_adj = np.zeros((3, GRID, GRID, GRID), dtype=np.float64) if force3_path is not None else None
    samples = []
    short_mm = long_mm = None
    if force3_path is not None:
        expected_bytes = 24 * N
        actual_bytes = force3_path.stat().st_size
        assert actual_bytes == expected_bytes, f"force3 size {actual_bytes} != {expected_bytes}"
        short_mm = np.memmap(force3_path, mode="r", dtype="<f4", offset=0, shape=(N, 3))
        long_mm = np.memmap(force3_path, mode="r", dtype="<f4", offset=12 * N, shape=(N, 3))

    with h5py.File(path, "r") as f:
        a = epoch(f)
        box = np.asarray(f["Header"].attrs["BoxSize"], dtype=float)
        assert 0.0 < a <= 1.0001 and box.shape == (3,) and np.all(box > 0)
        parts = f["PartType1"]
        assert parts["Coordinates"].shape == (N, 3)
        mass = float(parts["Masses"][0])
        assert mass > 0
        soft_mpc = float(np.median(parts["Softenings"][:8192])) if "Softenings" in parts else None
        if kick_cfg is not None and kick_cfg.get("soft_mpc") is None:
            assert soft_mpc is not None and np.isfinite(soft_mpc) and soft_mpc > 0
            kick_cfg["soft_mpc"] = soft_mpc

        for start in range(0, N, CHUNK):
            sl = slice(start, start + CHUNK)
            ids = parts["ParticleIDs"][sl]
            pos = parts["Coordinates"][sl]
            vel = parts["Velocities"][sl]
            assert np.isfinite(pos).all() and np.isfinite(vel).all()
            mask = (ids % ID_STRIDE) == 0
            if force3_path is not None:
                s_vec, l_vec = compute_short_long_kicks(
                    short_mm[sl], long_mm[sl],
                    kick_cfg["a_begin"], a, kick_cfg["soft_mpc"], kick_cfg["grav_g"],
                    kick_cfg["span_override"], kick_cfg["dt_rms"],
                )
                vel_adj = vel.astype(np.float64) + 2.0 * s_vec
                deposit_cic_fields(density, mom, pos, vel, box, mom_adj=mom_adj, vel_adj=vel_adj)
                samples.append((ids[mask], pos[mask], vel[mask], s_vec[mask], l_vec[mask]))
            else:
                deposit_cic_fields(density, mom, pos, vel, box)
                samples.append((ids[mask], pos[mask], vel[mask]))

    if force3_path is not None:
        ids, positions, velocities, s_samp, l_samp = (np.concatenate(fields) for fields in zip(*samples))
    else:
        ids, positions, velocities = (np.concatenate(fields) for fields in zip(*samples))
        s_samp = l_samp = None

    order = np.argsort(ids)
    assert len(ids) > 10000 and np.all(np.diff(ids[order]) > 0)
    assert abs(density.sum() - N) < 1e-3

    norm = float(GRID**3) / float(N)
    delta_k = np.fft.rfftn(density * norm - 1.0) / (GRID**3)

    box_h = box * H
    kx = 2.0 * np.pi * np.fft.fftfreq(GRID, d=box_h[0] / GRID)[:, None, None]
    ky = 2.0 * np.pi * np.fft.fftfreq(GRID, d=box_h[1] / GRID)[None, :, None]
    kz = 2.0 * np.pi * np.fft.rfftfreq(GRID, d=box_h[2] / GRID)[None, None, :]

    def to_theta_k(m_arr):
        return (
            kx * (np.fft.rfftn(m_arr[0] * norm) / (GRID**3))
            + ky * (np.fft.rfftn(m_arr[1] * norm) / (GRID**3))
            + kz * (np.fft.rfftn(m_arr[2] * norm) / (GRID**3))
        )

    theta_k = to_theta_k(mom)
    theta_adj_k = to_theta_k(mom_adj) if mom_adj is not None else None

    return {
        "file": path.name,
        "a": a,
        "box": box,
        "mass": mass,
        "ids": ids[order],
        "positions": positions[order],
        "velocities": velocities[order],
        "s_samp": s_samp[order] if s_samp is not None else None,
        "l_samp": l_samp[order] if l_samp is not None else None,
        "delta_k": delta_k,
        "theta_k": theta_k,
        "theta_adj_k": theta_adj_k,
    }


def summary(samples):
    return {
        "rms": float(np.sqrt(np.mean(samples**2))),
        "p50": float(np.percentile(samples, 50)),
        "p95": float(np.percentile(samples, 95)),
        "p99": float(np.percentile(samples, 99)),
        "maximum": float(np.max(samples)),
    }


def compare_field(cpu_k, gpu_k, box, csv_path=None):
    box_h = box * H
    volume = float(np.prod(box_h))
    freq = [
        np.fft.fftfreq(GRID) * GRID,
        np.fft.fftfreq(GRID) * GRID,
        np.fft.rfftfreq(GRID) * GRID,
    ]
    vectors = np.ix_(*(2.0 * np.pi * f / L for f, L in zip(freq, box_h)))
    k = np.sqrt(sum(v**2 for v in vectors))
    cic_power = np.ones_like(k)
    for f in np.ix_(*freq):
        cic_power *= np.sinc(f / GRID) ** 4
    fundamental = float(2.0 * np.pi / box_h[0])
    min_nyquist = float(np.min(np.pi * GRID / box_h))
    bins = np.arange(0.0, 0.5 * min_nyquist + fundamental, fundamental)
    weights = np.broadcast_to(np.r_[1, np.full(GRID // 2 - 1, 2), 1], k.shape)
    p_cpu = (np.abs(cpu_k) ** 2) / cic_power
    p_gpu = (np.abs(gpu_k) ** 2) / cic_power
    p_cross = np.real(cpu_k * np.conj(gpu_k)) / cic_power

    nmodes = np.histogram(k, bins=bins, weights=weights)[0]
    k_sum = np.histogram(k, bins=bins, weights=weights * k)[0]
    c_sum = np.histogram(k, bins=bins, weights=weights * p_cpu)[0]
    g_sum = np.histogram(k, bins=bins, weights=weights * p_gpu)[0]
    x_sum = np.histogram(k, bins=bins, weights=weights * p_cross)[0]

    rows = []
    for i in range(len(nmodes)):
        if nmodes[i] <= 0 or c_sum[i] <= 0 or g_sum[i] <= 0:
            continue
        k_mean = float(k_sum[i] / nmodes[i])
        pc = float(volume * c_sum[i] / nmodes[i])
        pg = float(volume * g_sum[i] / nmodes[i])
        px = float(volume * x_sum[i] / nmodes[i])
        ratio = pg / pc
        r_corr = px / np.sqrt(pc * pg)
        rows.append([k_mean, int(nmodes[i]), pc, pg, ratio, float(r_corr)])

    if csv_path is not None:
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k_h_per_Mpc", "modes", "P_cpu", "P_gpu", "P_gpu_over_cpu", "cross_correlation"])
            w.writerows(rows)

    data = np.asarray(rows)
    ranges = {}
    for lo, hi in [(0.3, 1.0), (1.0, 4.0), (4.0, 8.0), (8.0, 16.0)]:
        band = data[(data[:, 0] >= lo) & (data[:, 0] < hi)]
        if len(band) == 0:
            continue
        dev = np.abs(band[:, 4] - 1.0)
        ranges[f"{lo:g}-{hi:g}"] = {
            "bins": int(len(band)),
            "median_gpu_over_cpu_minus_1": float(np.median(band[:, 4] - 1.0)),
            "max_abs_dev": float(np.max(dev)),
            "rms_dev": float(np.sqrt(np.mean(dev**2))),
            "minimum_cross_correlation": float(np.min(band[:, 5])),
        }
    all_band = data[(data[:, 0] >= 0.3) & (data[:, 0] <= 16.0)]
    all_dev = np.abs(all_band[:, 4] - 1.0)
    return {
        "overall_0.3_16": {
            "bins": int(len(all_band)),
            "max_abs_dev": float(np.max(all_dev)),
            "rms_dev": float(np.sqrt(np.mean(all_dev**2))),
            "mean_ratio": float(np.mean(all_band[:, 4])),
            "min_r": float(np.min(all_band[:, 5])),
        },
        "bands": ranges,
    }


def halos(path, expected_a):
    with h5py.File(path, "r") as f:
        fa = epoch(f)
        assert abs(fa - expected_a) < 1e-5, f"FoF scale mismatch {fa} vs {expected_a}"
        sizes = f["Groups/Sizes"]
        n_groups = len(sizes)
        counts = np.zeros(len(HALO_EDGES), dtype=np.int64)
        top = np.array([], dtype=np.int64)
        for start in range(0, n_groups, CHUNK):
            vals = np.asarray(sizes[start:start + CHUNK], dtype=np.int64)
            if len(vals) == 0:
                continue
            for idx, minimum in enumerate(HALO_EDGES):
                counts[idx] += np.count_nonzero(vals >= minimum)
            cand = np.concatenate([top, vals[vals >= max(10000, int(top.min()) if len(top) >= 20 else 0)]])
            if len(cand) > 20:
                cand = np.partition(cand, -20)[-20:]
            top = np.sort(cand)[::-1]
        return {
            "number_of_groups": int(n_groups),
            "largest_group_particles": int(top[0]) if len(top) else 0,
            "top10_group_sizes": [int(x) for x in top[:10]],
            "cumulative_count": dict(zip(map(str, HALO_EDGES), counts.tolist())),
        }


def parse_gpu_log(log_path, fof_s_path, snap_s_path):
    steps = []
    mem_plan = result = sanity = gpu_input = None
    for line in log_path.read_text().splitlines():
        if line.startswith("MEMORY_PLAN "):
            mem_plan = json.loads(line.split(" ", 1)[1])
        elif line.startswith("GPU_INPUT "):
            gpu_input = json.loads(line.split(" ", 1)[1])
        elif line.startswith("STEP "):
            steps.append(json.loads(line.split(" ", 1)[1]))
        elif line.startswith("SANITY "):
            sanity = json.loads(line.split(" ", 1)[1])
        elif line.startswith("RESULT "):
            result = json.loads(line.split(" ", 1)[1])
    assert result is not None and steps
    pm_w = [s["wall_s"] for s in steps if s["pm"]]
    non_w = [s["wall_s"] for s in steps if not s["pm"]]
    return {
        "memory_plan": mem_plan,
        "gpu_input": gpu_input,
        "sanity": sanity,
        "result": result,
        "total_steps": len(steps),
        "pm_steps": {
            "count": len(pm_w),
            "sum_s": sum(pm_w),
            "median_s": statistics.median(pm_w) if pm_w else None,
            "mean_s": statistics.mean(pm_w) if pm_w else None,
        },
        "non_pm_steps": {
            "count": len(non_w),
            "sum_s": sum(non_w),
            "median_s": statistics.median(non_w) if non_w else None,
            "mean_s": statistics.mean(non_w) if non_w else None,
            "min_s": min(non_w) if non_w else None,
            "max_s": max(non_w) if non_w else None,
        },
        "snapshot_hdf5_wall_s": float(snap_s_path.read_text().strip()),
        "fof_wall_s": float(fof_s_path.read_text().strip()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cpu_snap", type=Path)
    ap.add_argument("gpu_snap", type=Path)
    ap.add_argument("cpu_fof", type=Path)
    ap.add_argument("gpu_fof", type=Path)
    ap.add_argument("gpu_log", type=Path)
    ap.add_argument("fof_s", type=Path)
    ap.add_argument("snap_s", type=Path)
    ap.add_argument("hist_cpu", type=Path)
    ap.add_argument("out_prefix", type=str)
    ap.add_argument("--force3-bin", type=Path, default=None)
    ap.add_argument("--a-begin", type=float, default=None)
    ap.add_argument("--soft-mpc", type=float, default=None)
    ap.add_argument("--grav-g", type=float, default=43.0092)
    args = ap.parse_args()

    timing_gpu = parse_gpu_log(args.gpu_log, args.fof_s, args.snap_s)
    kick_cfg = None
    if args.force3_bin is not None:
        gpu_in = timing_gpu.get("gpu_input") or {}
        res_obj = timing_gpu["result"]
        a_beg = args.a_begin if args.a_begin is not None else float(gpu_in["a_begin"])
        kick_cfg = {
            "a_begin": a_beg,
            "soft_mpc": args.soft_mpc,
            "grav_g": args.grav_g,
            "span_override": float(res_obj["last_pm_span_scale"]) if "last_pm_span_scale" in res_obj else None,
            "dt_rms": float(res_obj.get("last_dt_max_rms", 1.121728e-4)),
        }

    cpu = read_snapshot(args.cpu_snap)
    gpu = read_snapshot(args.gpu_snap, force3_path=args.force3_bin, kick_cfg=kick_cfg)
    assert abs(cpu["a"] - gpu["a"]) < 1e-5, f"Scale mismatch cpu={cpu['a']} gpu={gpu['a']}"
    assert np.allclose(cpu["box"], gpu["box"], rtol=1e-8, atol=0)
    assert np.isclose(cpu["mass"], gpu["mass"], rtol=1e-7, atol=0)
    assert np.array_equal(cpu["ids"], gpu["ids"])

    box = cpu["box"]
    dpos = (gpu["positions"] - cpu["positions"]).astype(np.float64)
    dpos -= box * np.rint(dpos / box)
    dvel = gpu["velocities"].astype(np.float64) - cpu["velocities"].astype(np.float64)

    cpu_fof = halos(args.cpu_fof, cpu["a"])
    gpu_fof = halos(args.gpu_fof, gpu["a"])
    halo_diffs = {}
    for thr, cnt in cpu_fof["cumulative_count"].items():
        diff = gpu_fof["cumulative_count"][thr] - cnt
        halo_diffs[thr] = {
            "cpu": cnt,
            "gpu": gpu_fof["cumulative_count"][thr],
            "gpu_minus_cpu": diff,
            "relative_to_cpu": diff / cnt if cnt else None,
        }

    matter = compare_field(cpu["delta_k"], gpu["delta_k"], box, Path(f"{args.out_prefix}-matter-power.csv"))
    momentum = compare_field(cpu["theta_k"], gpu["theta_k"], box, Path(f"{args.out_prefix}-momentum-power.csv"))
    momentum_adj = None
    vel_conv = None
    vel_adj_summary = None
    if args.force3_bin is not None:
        momentum_adj = compare_field(
            cpu["theta_k"], gpu["theta_adj_k"], box,
            Path(f"{args.out_prefix}-momentum-adjusted-power.csv"),
        )
        dv_flat = dvel.ravel()
        s_flat = gpu["s_samp"].ravel()
        l_flat = gpu["l_samp"].ravel()
        u = np.stack((s_flat, l_flat, dv_flat))
        qsum = u @ u.T
        lhs, rhs = qsum[:2, :2], qsum[:2, 2]
        sol = np.linalg.solve(lhs, rhs)
        y2 = float(qsum[2, 2])
        n_samp = len(cpu["ids"])
        dv_adj = dvel + 2.0 * gpu["s_samp"]
        vel_adj_summary = summary(np.linalg.norm(dv_adj, axis=1))
        vel_conv = {
            "sampled_particles": n_samp,
            "raw_vel_rms_km_s": float(np.sqrt(y2 / n_samp)),
            "two_component_coeffs": sol.tolist(),
            "unexplained_rms_km_s": float(np.sqrt(max(0.0, y2 - float(rhs @ sol)) / n_samp)),
            "explained_fraction": float(float(rhs @ sol) / y2),
            "fixed_short_kick_adj_vel_err_km_s": vel_adj_summary,
        }

    report = {
        "particles": N,
        "sampled_particle_ids": int(len(cpu["ids"])),
        "cpu_scale_factor": cpu["a"],
        "gpu_scale_factor": gpu["a"],
        "box_comoving_Mpc_h": (box * H).tolist(),
        "position_error_comoving_kpc_h": summary(np.linalg.norm(dpos, axis=1) * H * 1000.0),
        "velocity_error_km_s": summary(np.linalg.norm(dvel, axis=1)),
        "velocity_convention_check": vel_conv,
        "matter_power_k_h_Mpc": matter,
        "momentum_power_k_h_Mpc": momentum,
        "momentum_adjusted_power_k_h_Mpc": momentum_adj,
        "fof_cpu": cpu_fof,
        "fof_gpu": gpu_fof,
        "fof_cumulative_differences": halo_diffs,
        "timing_gpu": timing_gpu,
        "timing_historical_cpu": json.loads(args.hist_cpu.read_text()),
    }
    out_json = Path(f"{args.out_prefix}-comparison.json")
    out_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "a_end": gpu["a"],
        "pos_err_kpc_h": report["position_error_comoving_kpc_h"],
        "vel_err_km_s": report["velocity_error_km_s"],
        "vel_adj_err_km_s": vel_adj_summary,
        "velocity_convention_check": vel_conv,
        "matter_overall": matter["overall_0.3_16"],
        "momentum_overall": momentum["overall_0.3_16"],
        "momentum_adj_overall": momentum_adj["overall_0.3_16"] if momentum_adj else None,
        "fof_top5_cpu": cpu_fof["top10_group_sizes"][:5],
        "fof_top5_gpu": gpu_fof["top10_group_sizes"][:5],
        "fof_diffs": {k: v["relative_to_cpu"] for k, v in halo_diffs.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
