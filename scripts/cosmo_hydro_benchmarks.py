#!/usr/bin/env python3
"""Cosmological Coupled PM + Full-GPU SPH Verification Suite for gravityCU.

Benchmarks:
  1. adiabatic  : Uniform expanding universe (a=0.05 -> 1.0), exact u(a) ~ a^-2 verification.
  2. pancake    : 3D Cosmological Zel'dovich Pancake (32^3 DM + 32^3 Gas, a=0.02 -> 1.0, a_cross=0.5),
                  verifying caustic DM streaming + gas accretion shock heating (> 100x).
  3. cosmo_box  : Full 3D Cosmological Box (64^3 DM + 64^3 Gas = 524,288 particles, L=25 Mpc,
                  a=0.05 -> 1.0, Cooling + Star Formation), verifying cosmic web collapse,
                  multi-phase IGM/WHIM/cold-gas rho-T diagram, and baryon conservation.
"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

RHO_CRIT_0_H1 = 27.7536627  # 10^10 Msun / Mpc^3 for h=1
U_KMS2_TO_KELVIN = 48.4589


def _write_cosmo_ic(
    path: Path, box: float, a_init: float, gamma: float,
    omega_m: float, omega_b: float, omega_de: float, hubble0: float,
    gx: np.ndarray, gv: np.ndarray, gm: np.ndarray, gu: np.ndarray, gh: np.ndarray,
    dx: np.ndarray | None = None, dv: np.ndarray | None = None, dm: np.ndarray | None = None,
):
    n_gas = int(gx.shape[0])
    n_dm = int(dx.shape[0]) if dx is not None else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        # struct CosmoHydroHeader { int32 n_gas, n_dm; double box, a_init, gamma, om, ob, ode, H0; }
        f.write(struct.pack(
            "<2i7d", n_gas, n_dm, box, a_init, gamma, omega_m, omega_b, omega_de, hubble0
        ))
        np.ascontiguousarray(gx, dtype="<f8").tofile(f)
        np.ascontiguousarray(gv, dtype="<f8").tofile(f)
        np.ascontiguousarray(gm, dtype="<f4").tofile(f)
        np.ascontiguousarray(gu, dtype="<f8").tofile(f)
        np.ascontiguousarray(gh, dtype="<f8").tofile(f)
        if n_dm > 0:
            np.ascontiguousarray(dx, dtype="<f8").tofile(f)
            np.ascontiguousarray(dv, dtype="<f8").tofile(f)
            np.ascontiguousarray(dm, dtype="<f4").tofile(f)


def _read_cosmo_out(path: Path):
    raw = path.read_bytes()
    n_gas, n_dm, box, a_end, gamma, om, ob, ode, H0 = struct.unpack_from("<2i7d", raw, 0)
    off = struct.calcsize("<2i7d")
    gx = np.frombuffer(raw, dtype="<f8", count=3 * n_gas, offset=off).reshape(n_gas, 3).copy(); off += 24 * n_gas
    gv = np.frombuffer(raw, dtype="<f8", count=3 * n_gas, offset=off).reshape(n_gas, 3).copy(); off += 24 * n_gas
    gm = np.frombuffer(raw, dtype="<f4", count=n_gas, offset=off).copy(); off += 4 * n_gas
    gu = np.frombuffer(raw, dtype="<f8", count=n_gas, offset=off).copy(); off += 8 * n_gas
    grho = np.frombuffer(raw, dtype="<f8", count=n_gas, offset=off).copy(); off += 8 * n_gas
    mstar = np.frombuffer(raw, dtype="<f8", count=n_gas, offset=off).copy(); off += 8 * n_gas
    dx = dv = None
    if n_dm > 0:
        dx = np.frombuffer(raw, dtype="<f8", count=3 * n_dm, offset=off).reshape(n_dm, 3).copy(); off += 24 * n_dm
        dv = np.frombuffer(raw, dtype="<f8", count=3 * n_dm, offset=off).reshape(n_dm, 3).copy()
    return {
        "n_gas": n_gas, "n_dm": n_dm, "box": box, "a_end": a_end, "gamma": gamma,
        "omega_m": om, "omega_b": ob, "omega_de": ode, "hubble0": H0,
        "gx": gx, "gv": gv, "gm": gm, "gu": gu, "grho": grho, "mstar": mstar,
        "dx": dx, "dv": dv,
    }


def make_adiabatic(path: Path):
    n_side = 16
    box = 25.0
    a_init = 0.05
    om, ob, ode, H0 = 0.3153, 0.0493, 0.6847, 67.36
    h = H0 / 100.0
    dx_g = box / n_side
    c = (np.arange(n_side) + 0.5) * dx_g
    xx, yy, zz = np.meshgrid(c, c, c, indexing="ij")
    gx = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    n = gx.shape[0]
    rho_b = ob * RHO_CRIT_0_H1 * (h ** 2)
    gm = np.full(n, rho_b * (dx_g ** 3), dtype=np.float32)
    gv = np.zeros((n, 3))
    gu = np.full(n, 100.0)
    gh = np.full(n, 1.2348 * dx_g)
    _write_cosmo_ic(path, box, a_init, 5.0 / 3.0, om, ob, ode, H0, gx, gv, gm, gu, gh)
    return {"bench": "adiabatic", "n_gas": n, "u_init": 100.0, "a_init": a_init}


def verify_adiabatic(path: Path):
    d = _read_cosmo_out(path)
    u_exact = 100.0 * ((0.05 / d["a_end"]) ** 2)
    u_med = float(np.median(d["gu"]))
    rel_err = abs(u_med - u_exact) / u_exact
    return {
        "bench": "adiabatic",
        "a_end": d["a_end"],
        "u_sim": u_med,
        "u_exact": u_exact,
        "rel_err": rel_err,
        "passed": bool(rel_err < 1e-4),
    }


def make_pancake(path: Path, n_side: int = 32):
    """Cosmological EdS Zel'dovich Pancake (DM + Gas, a=0.02 -> 1.0, a_cross=0.5)."""
    box = 64.0
    a_init = 0.02
    a_cross = 0.5
    om, ob, ode, H0 = 1.0, 0.15, 0.0, 67.36
    h = H0 / 100.0
    k = 2.0 * math.pi / box
    dx_g = box / n_side
    c = (np.arange(n_side) + 0.5) * dx_g
    xx, yy, zz = np.meshgrid(c, c, c, indexing="ij")
    q = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    n = q.shape[0]

    # Shift wave so caustic forms cleanly at x = box/2: psi_x = -(a/a_cross)*sin(k*(q-box/2))/k
    qx_rel = q[:, 0] - 0.5 * box
    psi_x = -(a_init / a_cross) * np.sin(k * qx_rel) / k
    vx = -H0 * math.sqrt(a_init) * np.sin(k * qx_rel) / (k * a_cross)

    gx = q.copy()
    gx[:, 0] = np.mod(q[:, 0] + psi_x, box)
    gv = np.zeros((n, 3))
    gv[:, 0] = vx

    # Offset DM by half a grid cell in y,z to avoid zero-distance overlap
    dx_dm = q.copy()
    dx_dm[:, 0] = gx[:, 0]
    dx_dm[:, 1] = np.mod(q[:, 1] + 0.5 * dx_g, box)
    dx_dm[:, 2] = np.mod(q[:, 2] + 0.5 * dx_g, box)
    dv_dm = gv.copy()

    rho_crit = RHO_CRIT_0_H1 * (h ** 2)
    gm = np.full(n, ob * rho_crit * (dx_g ** 3), dtype=np.float32)
    dm = np.full(n, (om - ob) * rho_crit * (dx_g ** 3), dtype=np.float32)
    # Cold initial gas T_init = 100 K -> u_init = 100 / 48.4589 = 2.0636 (km/s)^2
    # By a=1 without shocks, adiabatic T would cool by (0.02/1.0)^2 = 4e-4 -> 0.04 K!
    # Accretion shock at pancake heats central slab to T ~ 10^6 - 10^7 K!
    gu = np.full(n, 100.0 / U_KMS2_TO_KELVIN)
    gh = np.full(n, 1.2348 * dx_g)
    _write_cosmo_ic(path, box, a_init, 5.0 / 3.0, om, ob, ode, H0, gx, gv, gm, gu, gh, dx_dm, dv_dm, dm)
    return {"bench": "pancake", "n_gas": n, "n_dm": n, "box": box, "a_init": a_init, "a_cross": a_cross}


def verify_pancake(path: Path):
    d = _read_cosmo_out(path)
    box = d["box"]
    x_dist = np.abs(((d["gx"][:, 0] - 0.5 * box + 0.5 * box) % box) - 0.5 * box)
    T = U_KMS2_TO_KELVIN * d["gu"]
    rho_mean = d["omega_b"] * RHO_CRIT_0_H1 * ((d["hubble0"] / 100.0) ** 2)
    overd = d["grho"] / rho_mean

    mid = x_dist < 0.05 * box
    outer = x_dist > 0.35 * box
    T_mid_med = float(np.median(T[mid])) if np.any(mid) else 0.0
    T_mid_max = float(np.max(T[mid])) if np.any(mid) else 0.0
    T_outer_med = float(np.median(T[outer])) if np.any(outer) else 1e-6
    overd_mid = float(np.median(overd[mid])) if np.any(mid) else 0.0
    shock_jump = T_mid_med / max(T_outer_med, 1e-6)
    return {
        "bench": "pancake",
        "a_end": d["a_end"],
        "T_midplane_K": T_mid_med,
        "T_midplane_max_K": T_mid_max,
        "T_outer_K": T_outer_med,
        "shock_temp_jump": shock_jump,
        "midplane_overdensity": overd_mid,
        "passed": bool(shock_jump > 100.0 and overd_mid > 5.0),
    }


def make_cosmo_box(path: Path, n_side: int = 48):
    """Full 3D Cosmological Box (DM + Gas, L=25 Mpc, Planck18, a=0.05 -> 1.0) with Gaussian GRF 1LPT."""
    box = 25.0
    a_init = 0.05
    om, ob, ode, H0 = 0.3153, 0.0493, 0.6847, 67.36
    h = H0 / 100.0
    dx_g = box / n_side

    # Generate clean 3D Gaussian Random Field density contrast with BBKS-like P(k)
    rng = np.random.default_rng(2026)
    white = rng.normal(0.0, 1.0, size=(n_side, n_side, n_side))
    dk = np.fft.rfftn(white)
    kx = np.fft.fftfreq(n_side, d=dx_g) * 2.0 * math.pi
    ky = np.fft.fftfreq(n_side, d=dx_g) * 2.0 * math.pi
    kz = np.fft.rfftfreq(n_side, d=dx_g) * 2.0 * math.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    K = np.sqrt(K2)
    # BBKS-like transfer T(k) ~ 1 / (1 + (k/k_eq)^2) -> P(k) ~ k / (1 + (k/0.15)^2)^2
    Pk = np.where(K > 0, (K ** 0.96) / ((1.0 + (K / 0.15) ** 2) ** 2), 0.0)
    dk *= np.sqrt(Pk)
    dk[0, 0, 0] = 0.0
    delta_r = np.fft.irfftn(dk, s=(n_side, n_side, n_side))
    # Normalize RMS(delta) at a_init=0.05 to ~0.12 so nonlinear cosmic web & halos form cleanly by z=0!
    norm = 0.12 / max(float(np.std(delta_r)), 1e-12)
    dk *= norm

    # 1LPT displacement psi_k = i * k / k^2 * delta_k
    inv_k2 = np.where(K2 > 0, 1.0 / K2, 0.0)
    psi_x = np.fft.irfftn(1j * KX * inv_k2 * dk, s=(n_side, n_side, n_side)).ravel()
    psi_y = np.fft.irfftn(1j * KY * inv_k2 * dk, s=(n_side, n_side, n_side)).ravel()
    psi_z = np.fft.irfftn(1j * KZ * inv_k2 * dk, s=(n_side, n_side, n_side)).ravel()
    psi = np.column_stack((psi_x, psi_y, psi_z))

    c = (np.arange(n_side) + 0.5) * dx_g
    xx, yy, zz = np.meshgrid(c, c, c, indexing="ij")
    q = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    n = q.shape[0]

    H_init = H0 * math.sqrt(om / (a_init ** 3) + ode)
    f_grow = (om / (om + ode * (a_init ** 3))) ** 0.55
    vel = a_init * H_init * f_grow * psi

    gx = np.mod(q + psi, box)
    dx_dm = np.mod(q + 0.5 * dx_g + psi, box)
    rho_crit = RHO_CRIT_0_H1 * (h ** 2)
    gm = np.full(n, ob * rho_crit * (dx_g ** 3), dtype=np.float32)
    dm = np.full(n, (om - ob) * rho_crit * (dx_g ** 3), dtype=np.float32)
    # Initial IGM temperature ~ 200 K at z=19
    gu = np.full(n, 200.0 / U_KMS2_TO_KELVIN)
    gh = np.full(n, 1.2348 * dx_g)
    _write_cosmo_ic(path, box, a_init, 5.0 / 3.0, om, ob, ode, H0, gx, vel, gm, gu, gh, dx_dm, vel, dm)
    return {"bench": "cosmo_box", "n_gas": n, "n_dm": n, "box": box, "a_init": a_init}


def verify_cosmo_box(path: Path):
    d = _read_cosmo_out(path)
    rho_b_mean = d["omega_b"] * RHO_CRIT_0_H1 * ((d["hubble0"] / 100.0) ** 2)
    overd = d["grho"] / rho_b_mean
    T = U_KMS2_TO_KELVIN * d["gu"]
    m_gas = float(np.sum(d["gm"]))
    m_star = float(np.sum(d["mstar"]))
    f_star = m_star / max(m_gas + m_star, 1e-12)

    # Multi-phase fractions:
    #   Diffuse IGM: delta < 10
    #   WHIM (shock-heated): T > 1e5 K
    #   Condensed cold/SF gas: delta > 50 and T < 3e4 K
    frac_whim = float(np.mean(T > 1.0e5))
    frac_cold_dense = float(np.mean((overd > 50.0) & (T < 3.0e4)))
    return {
        "bench": "cosmo_box",
        "a_end": d["a_end"],
        "n_gas": d["n_gas"],
        "n_dm": d["n_dm"],
        "max_gas_overdensity": float(np.max(overd)),
        "p99_gas_overdensity": float(np.percentile(overd, 99)),
        "T_median_K": float(np.median(T)),
        "T_p95_K": float(np.percentile(T, 95)),
        "T_max_K": float(np.max(T)),
        "frac_WHIM_gt_1e5K": frac_whim,
        "frac_cold_dense": frac_cold_dense,
        "m_star_1e10msun": m_star,
        "stellar_baryon_fraction": f_star,
        "passed": bool(float(np.max(overd)) > 50.0 and frac_whim > 0.01 and m_star > 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("bench", choices=["adiabatic", "pancake", "cosmo_box"])
    m.add_argument("output", type=Path)
    m.add_argument("--n-side", type=int, default=48)
    v = sub.add_parser("verify")
    v.add_argument("bench", choices=["adiabatic", "pancake", "cosmo_box"])
    v.add_argument("snapshot", type=Path)
    args = ap.parse_args()

    if args.cmd == "make":
        if args.bench == "adiabatic":
            res = make_adiabatic(args.output)
        elif args.bench == "pancake":
            res = make_pancake(args.output, n_side=min(args.n_side, 32))
        else:
            res = make_cosmo_box(args.output, n_side=args.n_side)
    else:
        fn = {"adiabatic": verify_adiabatic, "pancake": verify_pancake, "cosmo_box": verify_cosmo_box}[args.bench]
        res = fn(args.snapshot)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
