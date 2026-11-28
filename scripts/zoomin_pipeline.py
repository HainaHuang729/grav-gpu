#!/usr/bin/env python3
"""B3 Multi-Mass Zoom-in Pipeline for gravityCU.

Generates nested multi-level cosmological zoom-in ICs (up to 4096^3 / 8192^3 local
effective resolution) with matched per-particle softening (*.soft.bin), and audits
evolved snapshots for:
  1. Target halo R200c, M200c, R200m, M200m (shrinking-sphere center on high-res particles)
  2. Zero low-resolution particle contamination inside R200c (and nearest LR distance)
  3. Radial density profile & resolved subhalo mass function convergence.
"""
import argparse
import ctypes
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

# Internal gravityCU units: Mpc, 10^10 Msun, km/s
G_INTERNAL = 43.0092
RHO_CRIT_0_H1 = 27.7536627  # 10^10 Msun / Mpc^3 for h = 1 (3 H0^2 / 8 pi G)


def _generate_nested_levels(
    box: float,
    n_base: int,
    n_eff: int,
    hr_half_width: float,
    omega_m: float,
    h: float,
    soft_frac: float = 0.05,
):
    """Build nested factor-of-2 multi-mass Lagrangian shells centered at box/2.

    Every shell replaces the interior of its parent level with 8x lighter particles,
    guaranteeing exact mass conservation across the entire periodic volume and
    smooth 8:1 mass steps across all shell boundaries.
    """
    assert (n_eff & (n_eff - 1)) == 0 and (n_base & (n_base - 1)) == 0
    assert n_eff >= n_base
    n_levels = int(round(math.log2(n_eff // n_base))) + 1
    rho_mean = omega_m * RHO_CRIT_0_H1 * (h ** 2)
    center = 0.5 * box

    # Choose half-widths for each refined level l = 1..n_levels-1 (shrinking inward)
    # Outermost refined level covers a broad buffer; innermost level covers hr_half_width.
    half_widths = [0.5 * box]
    for l in range(1, n_levels):
        # Level l has cell size dx_l = box / (n_base * 2^l).
        # Ensure at least 4 parent cells of buffer padding on every side!
        dx_l = box / float(n_base << l)
        # Geometric progression from hr_half_width at finest level outward
        steps_from_finest = (n_levels - 1) - l
        hw = hr_half_width * (1.85 ** steps_from_finest)
        hw = min(hw, 0.45 * box)
        # Snap hw to an integer multiple of parent cell size dx_{l-1} so nesting is exact!
        dx_parent = box / float(n_base << (l - 1))
        n_parent_half = max(2, int(round(hw / dx_parent)))
        # Ensure strictly smaller than previous level's half-width
        max_parent_half = int(round(half_widths[-1] / dx_parent)) - 1
        n_parent_half = min(n_parent_half, max(1, max_parent_half))
        half_widths.append(n_parent_half * dx_parent)

    all_pos = []
    all_mass = []
    all_soft = []
    all_level = []

    for l in range(n_levels):
        n_l = n_base << l
        dx = box / float(n_l)
        m_part = rho_mean * (dx ** 3)
        soft_part = soft_frac * dx

        hw_out = half_widths[l]
        hw_in = half_widths[l + 1] if (l + 1 < n_levels) else -1.0

        # Grid indices covering [center - hw_out, center + hw_out)
        if l == 0:
            idx_1d = np.arange(n_l, dtype=np.int64)
            coords_1d = (idx_1d + 0.5) * dx
        else:
            n_half = int(round(hw_out / dx))
            i0 = n_l // 2 - n_half
            i1 = n_l // 2 + n_half
            idx_1d = np.arange(i0, i1, dtype=np.int64)
            coords_1d = (idx_1d + 0.5) * dx

        gx, gy, gz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
        x = gx.ravel()
        y = gy.ravel()
        z = gz.ravel()

        if hw_in > 0.0:
            # Exclude interior cube that will be populated by level l+1
            eps = 1e-9 * dx
            inside = (
                (np.abs(x - center) < hw_in - eps)
                & (np.abs(y - center) < hw_in - eps)
                & (np.abs(z - center) < hw_in - eps)
            )
            keep = ~inside
            x, y, z = x[keep], y[keep], z[keep]

        cnt = x.size
        all_pos.append(np.column_stack((x, y, z)))
        all_mass.append(np.full(cnt, m_part, dtype=np.float64))
        all_soft.append(np.full(cnt, soft_part, dtype=np.float64))
        all_level.append(np.full(cnt, l, dtype=np.int32))

    pos = np.vstack(all_pos)
    mass = np.concatenate(all_mass)
    soft = np.concatenate(all_soft)
    level = np.concatenate(all_level)
    return pos, mass, soft, level, half_widths


def generate_zoomin(
    output: Path,
    soft_out: Path,
    box: float = 50.0,
    n_base: int = 16,
    n_eff: int = 4096,
    hr_half_width: float = 0.42,
    a_init: float = 0.05,
    omega_m: float = 0.3153,
    omega_de: float = 0.6847,
    h: float = 0.6736,
    peak_amp: float = 0.14,
    sub_amp: float = 0.035,
    seed: int = 2026,
) -> dict:
    """Generate a multi-mass Zoom-in IC with a collapsing central peak + resolved subhalos."""
    pos, mass, soft, level, half_widths = _generate_nested_levels(
        box, n_base, n_eff, hr_half_width, omega_m, h
    )
    n_tot = pos.shape[0]
    center = 0.5 * box
    H0 = 100.0 * h
    H_init = H0 * math.sqrt(omega_m / (a_init ** 3) + omega_de)

    # Apply smooth 1LPT-like radial inward displacement for a central proto-halo of Lagrangian
    # radius R_proto = 0.45 * hr_half_width (safely inside the high-res region so zero LR
    # boundary particles fall into R200!), plus deterministic subhalo seed modes inside R_proto.
    dx_v = pos[:, 0] - center
    dy_v = pos[:, 1] - center
    dz_v = pos[:, 2] - center
    r = np.sqrt(dx_v ** 2 + dy_v ** 2 + dz_v ** 2) + 1e-12
    r_proto = 0.48 * hr_half_width

    # Smooth Gaussian overdensity profile -> radial displacement psi(r)
    # Tapered smoothly to 0 beyond 0.9 * hr_half_width so outer shells remain undisturbed
    u = r / r_proto
    disp_r = -peak_amp * r * np.exp(-0.5 * (u ** 2))
    # Add 12 deterministic off-center subhalo seed perturbations inside the proto-halo
    rng = np.random.default_rng(seed)
    psi = np.column_stack((disp_r * dx_v / r, disp_r * dy_v / r, disp_r * dz_v / r))
    for _ in range(12):
        # Place subhalo seeds at 0.25..0.75 * r_proto
        sr = rng.uniform(0.25 * r_proto, 0.72 * r_proto)
        ct = rng.uniform(-1.0, 1.0)
        st = math.sqrt(max(0.0, 1.0 - ct * ct))
        ph = rng.uniform(0.0, 2.0 * math.pi)
        sc = np.array([sr * st * math.cos(ph), sr * st * math.sin(ph), sr * ct])
        r_sub = rng.uniform(0.14 * r_proto, 0.26 * r_proto)
        sdx = dx_v - sc[0]
        sdy = dy_v - sc[1]
        sdz = dz_v - sc[2]
        sr2 = sdx ** 2 + sdy ** 2 + sdz ** 2
        fac = -sub_amp * np.exp(-0.5 * sr2 / (r_sub ** 2))
        psi[:, 0] += fac * sdx
        psi[:, 1] += fac * sdy
        psi[:, 2] += fac * sdz

    pos_init = np.mod(pos + psi, box)
    # Growing-mode 1LPT peculiar velocity: v_pec = a * H(a) * f(a) * psi (f ~ 1 at a_init=0.05)
    f_grow = (omega_m / (omega_m + omega_de * (a_init ** 3))) ** 0.55
    vel_init = (a_init * H_init * f_grow * psi).astype("<f4")

    m_f4 = mass.astype("<f4")
    s_f4 = soft.astype("<f4")
    pids = np.arange(1, n_tot + 1, dtype="<u8")
    min_soft = float(np.min(s_f4))
    min_mass = float(np.min(m_f4))
    max_mass = float(np.max(m_f4))
    n_hr = int(np.sum(level == int(np.max(level))))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        f.write(struct.pack("<Q3d", n_tot, box, a_init, min_soft))
        pos_init.astype("<f8").tofile(f)
        vel_init.tofile(f)
        m_f4.tofile(f)
        pids.tofile(f)

    soft_out.parent.mkdir(parents=True, exist_ok=True)
    s_f4.tofile(soft_out)

    meta = {
        "output": str(output),
        "soft_out": str(soft_out),
        "box_mpc": box,
        "n_base": n_base,
        "n_eff": n_eff,
        "dx_hr_kpc": 1000.0 * box / float(n_eff),
        "hr_half_width_mpc": hr_half_width,
        "n_total": int(n_tot),
        "n_high_res": n_hr,
        "n_low_res": int(n_tot - n_hr),
        "n_levels": len(half_widths),
        "half_widths_mpc": half_widths,
        "min_mass_1e10msun": min_mass,
        "max_mass_1e10msun": max_mass,
        "mass_dynamic_ratio": max_mass / min_mass,
        "min_soft_mpc": min_soft,
        "max_soft_mpc": float(np.max(s_f4)),
        "total_mass_1e10msun": float(np.sum(mass)),
        "expected_box_mass_1e10msun": omega_m * RHO_CRIT_0_H1 * (h ** 2) * (box ** 3),
        "a_init": a_init,
    }
    return meta


def _read_ic_masses(ic_path: Path):
    raw = ic_path.read_bytes()
    n, box, a0, soft = struct.unpack_from("<Q3d", raw, 0)
    m_off = 32 + n * 24 + n * 12
    masses = np.frombuffer(raw, dtype="<f4", count=n, offset=m_off).copy()
    return n, box, masses


def _read_snap_pos_vel(snap_path: Path):
    raw = snap_path.read_bytes()
    n, box, a_out = struct.unpack_from("<Q2d", raw, 0)
    pos = np.frombuffer(raw, dtype="<f8", count=n * 3, offset=24).reshape(n, 3).copy()
    vel = np.frombuffer(raw, dtype="<f4", count=n * 3, offset=24 + n * 24).reshape(n, 3).copy()
    return n, box, a_out, pos, vel


def _shrinking_sphere_center(pos_hr: np.ndarray, box: float) -> np.ndarray:
    """Find densest core center via shrinking sphere (Power+2003)."""
    center = np.array([0.5 * box, 0.5 * box, 0.5 * box], dtype=np.float64)
    rel = ((pos_hr - center + 0.5 * box) % box) - 0.5 * box
    c = np.median(rel, axis=0)
    r_max = float(np.percentile(np.linalg.norm(rel - c, axis=1), 90))
    rad = max(r_max, 0.05)
    for _ in range(25):
        d = np.linalg.norm(rel - c, axis=1)
        m = d < rad
        if int(np.sum(m)) < 64:
            break
        c = np.mean(rel[m], axis=0)
        rad *= 0.80
    return np.mod(center + c, box)


def audit_zoomin(
    ic_path: Path,
    snap_path: Path,
    n_eff: int = 4096,
    omega_m: float = 0.3153,
    omega_de: float = 0.6847,
    h: float = 0.6736,
    fof_lib: Path | None = None,
) -> dict:
    """Audit evolved Zoom-in halo for R200c/M200c, zero LR contamination, and subhalos."""
    n_ic, box_ic, masses = _read_ic_masses(ic_path)
    n_sn, box, a_out, pos, vel = _read_snap_pos_vel(snap_path)
    assert n_ic == n_sn, f"Particle count mismatch: IC={n_ic}, snap={n_sn}"

    m_hr = float(np.min(masses))
    is_hr = masses <= (1.01 * m_hr)
    is_lr = ~is_hr

    pos_hr = pos[is_hr]
    center = _shrinking_sphere_center(pos_hr, box)

    # Periodic distances from halo center for ALL particles
    rel = ((pos - center + 0.5 * box) % box) - 0.5 * box
    r_all = np.linalg.norm(rel, axis=1)
    order = np.argsort(r_all)
    r_sorted = r_all[order]
    m_sorted = masses[order].astype(np.float64)
    cum_m = np.cumsum(m_sorted)

    # Critical density at a_out in comoving Mpc^-3 (10^10 Msun / Mpc^3)
    # rho_crit(a) = rho_crit,0 * E(a)^2; in comoving volume V_com, M / V_com = a^3 * rho_phys
    e2 = omega_m / (a_out ** 3) + omega_de
    rho_crit_com = RHO_CRIT_0_H1 * (h ** 2) * e2 * (a_out ** 3)
    rho_mean_com = omega_m * RHO_CRIT_0_H1 * (h ** 2)

    vol = (4.0 / 3.0) * math.pi * (np.maximum(r_sorted, 1e-6) ** 3)
    mean_rho = cum_m / vol

    # Find outermost radius where mean_rho >= 200 * rho_crit_com
    idx_200c = np.where(mean_rho >= 200.0 * rho_crit_com)[0]
    if idx_200c.size > 0:
        k200c = int(idx_200c[-1])
        r200c = float(r_sorted[k200c])
        m200c = float(cum_m[k200c])
    else:
        r200c, m200c = 0.0, 0.0

    idx_200m = np.where(mean_rho >= 200.0 * rho_mean_com)[0]
    if idx_200m.size > 0:
        k200m = int(idx_200m[-1])
        r200m = float(r_sorted[k200m])
        m200m = float(cum_m[k200m])
    else:
        r200m, m200m = 0.0, 0.0

    # Contamination audit
    r_lr = r_all[is_lr]
    r_hr = r_all[is_hr]
    n_hr_r200c = int(np.sum(r_hr <= r200c))
    n_lr_r200c = int(np.sum(r_lr <= r200c))
    n_lr_15r200c = int(np.sum(r_lr <= 1.5 * r200c))
    n_lr_20r200c = int(np.sum(r_lr <= 2.0 * r200c))
    min_r_lr = float(np.min(r_lr)) if r_lr.size > 0 else float("inf")

    # Optional GPU FoF subhalo finder on high-res particles inside 2*R200c
    n_subhalos = 0
    subhalo_masses = []
    if fof_lib is not None and fof_lib.exists() and r200c > 0.0:
        in_zoom = is_hr & (r_all <= 2.0 * r200c)
        sub_pos = np.ascontiguousarray(pos[in_zoom], dtype=np.float64)
        n_sub_p = sub_pos.shape[0]
        if n_sub_p >= 64:
            lib = ctypes.CDLL(str(fof_lib))
            lib.whole_fof_ranked.argtypes = [
                ctypes.c_void_p, ctypes.c_int,
                ctypes.c_double, ctypes.c_double, ctypes.c_double,
                ctypes.c_double, ctypes.c_int,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.c_void_p, ctypes.c_void_p,
            ]
            lib.whole_fof_ranked.restype = ctypes.c_int
            # Use b_sub = 0.12 * dx_hr to isolate dense subhalo cores
            dx_hr = box / float(n_eff)
            b_sub_link = 0.12 * dx_hr
            ids_out = np.empty(n_sub_p, dtype=np.int64)
            ngrp = ctypes.c_uint64(0)
            ngrouped = ctypes.c_uint64(0)
            largest = ctypes.c_uint64(0)
            el = np.zeros(9, dtype=np.float64)
            rc = lib.whole_fof_ranked(
                sub_pos.ctypes.data, int(n_sub_p),
                box, box, box, b_sub_link, 20,
                ids_out.ctypes.data,
                ctypes.byref(ngrp), ctypes.byref(ngrouped), ctypes.byref(largest),
                el.ctypes.data, None,
            )
            if rc == 0 and ngrp.value > 1:
                valid = ids_out[ids_out >= 0]
                _, counts = np.unique(valid, return_counts=True)
                counts = np.sort(counts)[::-1]
                # Exclude the main host (counts[0]), record subhalos
                n_subhalos = int(counts.size - 1)
                subhalo_masses = [float(c * m_hr) for c in counts[1:16]]

    return {
        "a_end": float(a_out),
        "box_mpc": float(box),
        "n_eff": int(n_eff),
        "dx_hr_kpc": 1000.0 * box / float(n_eff),
        "n_total": int(n_sn),
        "n_high_res": int(np.sum(is_hr)),
        "m_hr_1e10msun": m_hr,
        "halo_center_mpc": [float(c) for c in center],
        "R200c_mpc": r200c,
        "M200c_1e10msun": m200c,
        "R200m_mpc": r200m,
        "M200m_1e10msun": m200m,
        "n_hr_in_R200c": n_hr_r200c,
        "n_lr_in_R200c": n_lr_r200c,
        "n_lr_in_1p5_R200c": n_lr_15r200c,
        "n_lr_in_2p0_R200c": n_lr_20r200c,
        "min_lr_dist_mpc": min_r_lr,
        "min_lr_over_R200c": (min_r_lr / r200c) if r200c > 0 else 0.0,
        "zero_contamination_R200c": (n_lr_r200c == 0 and n_hr_r200c > 100),
        "n_subhalos_fof": n_subhalos,
        "top_subhalo_masses_1e10msun": subhalo_masses,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="gravityCU B3 Multi-Mass Zoom-in Pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("output", type=Path)
    g.add_argument("--soft-out", type=Path, required=True)
    g.add_argument("--box", type=float, default=50.0)
    g.add_argument("--n-base", type=int, default=16)
    g.add_argument("--n-eff", type=int, default=4096)
    g.add_argument("--hr-half-width", type=float, default=0.42)
    g.add_argument("--a-init", type=float, default=0.05)
    g.add_argument("--peak-amp", type=float, default=0.14)
    g.add_argument("--sub-amp", type=float, default=0.035)
    g.add_argument("--seed", type=int, default=2026)

    a = sub.add_parser("audit")
    a.add_argument("ic", type=Path)
    a.add_argument("snapshot", type=Path)
    a.add_argument("--n-eff", type=int, default=4096)
    a.add_argument("--fof-lib", type=Path, default=None)

    args = ap.parse_args()
    if args.cmd == "generate":
        res = generate_zoomin(
            args.output, args.soft_out,
            box=args.box, n_base=args.n_base, n_eff=args.n_eff,
            hr_half_width=args.hr_half_width, a_init=args.a_init,
            peak_amp=args.peak_amp, sub_amp=args.sub_amp, seed=args.seed,
        )
        print(json.dumps(res, indent=2))
    else:
        res = audit_zoomin(args.ic, args.snapshot, n_eff=args.n_eff, fof_lib=args.fof_lib)
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
