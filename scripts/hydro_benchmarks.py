#!/usr/bin/env python3
"""B4 Full-GPU Hydrodynamics Verification Suite (Sod, Sedov-Taylor, K-H, Cooling, Santa Barbara).

Generates binary ICs for build/grav_hydro and quantitatively verifies evolved snapshots
against analytical Riemann, Sedov-Taylor, Kelvin-Helmholtz, cooling ODE, and Santa Barbara
cluster entropy-profile benchmarks.
"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np


def _write_hydro_ic(
    path: Path,
    box,
    periodic,
    gamma: float,
    x: np.ndarray,
    v: np.ndarray,
    m: np.ndarray,
    u: np.ndarray,
    h: np.ndarray,
    dm_x: np.ndarray | None = None,
    dm_v: np.ndarray | None = None,
    dm_m: np.ndarray | None = None,
):
    n_gas = int(x.shape[0])
    n_dm = int(dm_x.shape[0]) if dm_x is not None else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        # struct HydroBinHeader { int32 n_gas, n_dm; double box[3]; int32 periodic[3], pad; double gamma; }
        f.write(struct.pack(
            "<2i3d4id",
            n_gas, n_dm,
            float(box[0]), float(box[1]), float(box[2]),
            int(periodic[0]), int(periodic[1]), int(periodic[2]), 0,
            float(gamma),
        ))
        np.ascontiguousarray(x, dtype="<f8").tofile(f)
        np.ascontiguousarray(v, dtype="<f8").tofile(f)
        np.ascontiguousarray(m, dtype="<f8").tofile(f)
        np.ascontiguousarray(u, dtype="<f8").tofile(f)
        np.ascontiguousarray(h, dtype="<f8").tofile(f)
        if n_dm > 0:
            np.ascontiguousarray(dm_x, dtype="<f8").tofile(f)
            np.ascontiguousarray(dm_v, dtype="<f8").tofile(f)
            np.ascontiguousarray(dm_m, dtype="<f8").tofile(f)


def _read_hydro_out(path: Path):
    raw = path.read_bytes()
    n_gas, n_dm, bx, by, bz, px, py, pz, _, gamma = struct.unpack_from("<2i3d4id", raw, 0)
    off = struct.calcsize("<2i3d4id")
    x = np.frombuffer(raw, dtype="<f8", count=3 * n_gas, offset=off).reshape(n_gas, 3).copy(); off += 24 * n_gas
    v = np.frombuffer(raw, dtype="<f8", count=3 * n_gas, offset=off).reshape(n_gas, 3).copy(); off += 24 * n_gas
    m = np.frombuffer(raw, dtype="<f8", count=n_gas, offset=off).copy(); off += 8 * n_gas
    u = np.frombuffer(raw, dtype="<f8", count=n_gas, offset=off).copy(); off += 8 * n_gas
    rho = np.frombuffer(raw, dtype="<f8", count=n_gas, offset=off).copy(); off += 8 * n_gas
    h = np.frombuffer(raw, dtype="<f8", count=n_gas, offset=off).copy(); off += 8 * n_gas
    P = np.frombuffer(raw, dtype="<f8", count=n_gas, offset=off).copy()
    return {
        "n_gas": n_gas, "n_dm": n_dm, "box": (bx, by, bz), "gamma": gamma,
        "x": x, "v": v, "m": m, "u": u, "rho": rho, "h": h, "P": P,
    }


# ==================== 1. SOD SHOCK TUBE (3D PERIODIC MIRROR) ====================
def exact_sod_riemann(x_arr: np.ndarray, t: float, x0: float = 0.5, gamma: float = 5.0 / 3.0):
    """Exact analytical Riemann solution for Sod shock tube (rhoL=1, PL=1; rhoR=0.25, PR=0.1795)."""
    rhoL, PL, vL = 1.0, 1.0, 0.0
    rhoR, PR, vR = 0.25, 0.1795, 0.0
    cL = math.sqrt(gamma * PL / rhoL)
    cR = math.sqrt(gamma * PR / rhoR)

    def f_star(P):
        # Left rarefaction
        fL = (2.0 * cL / (gamma - 1.0)) * ((P / PL) ** ((gamma - 1.0) / (2.0 * gamma)) - 1.0)
        # Right shock
        AL = 2.0 / ((gamma + 1.0) * rhoR)
        BL = ((gamma - 1.0) / (gamma + 1.0)) * PR
        fR = (P - PR) * math.sqrt(AL / (P + BL))
        return fL + fR + (vR - vL)

    lo, hi = PR, PL
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f_star(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    P_star = 0.5 * (lo + hi)
    AL = 2.0 / ((gamma + 1.0) * rhoR)
    BL = ((gamma - 1.0) / (gamma + 1.0)) * PR
    v_star = vR + (P_star - PR) * math.sqrt(AL / (P_star + BL))
    rho_star_L = rhoL * ((P_star / PL) ** (1.0 / gamma))
    g1 = (gamma - 1.0) / (gamma + 1.0)
    rho_star_R = rhoR * ((P_star / PR + g1) / (g1 * (P_star / PR) + 1.0))
    c_star_L = cL * ((P_star / PL) ** ((gamma - 1.0) / (2.0 * gamma)))
    v_shock = vR + cR * math.sqrt(((gamma + 1.0) / (2.0 * gamma)) * (P_star / PR) + (gamma - 1.0) / (2.0 * gamma))

    rho_out = np.empty_like(x_arr)
    P_out = np.empty_like(x_arr)
    v_out = np.empty_like(x_arr)
    xi = (x_arr - x0) / max(t, 1e-12)
    for k, s in enumerate(xi):
        if s < -cL:
            rho_out[k], P_out[k], v_out[k] = rhoL, PL, vL
        elif s < v_star - c_star_L:
            v_f = (2.0 / (gamma + 1.0)) * (cL + s)
            c_f = cL - 0.5 * (gamma - 1.0) * v_f
            rho_out[k] = rhoL * ((c_f / cL) ** (2.0 / (gamma - 1.0)))
            P_out[k] = PL * ((c_f / cL) ** (2.0 * gamma / (gamma - 1.0)))
            v_out[k] = v_f
        elif s < v_star:
            rho_out[k], P_out[k], v_out[k] = rho_star_L, P_star, v_star
        elif s < v_shock:
            rho_out[k], P_out[k], v_out[k] = rho_star_R, P_star, v_star
        else:
            rho_out[k], P_out[k], v_out[k] = rhoR, PR, vR
    return rho_out, P_out, v_out, {"P_star": P_star, "v_star": v_star, "x_shock": x0 + v_shock * t, "x_contact": x0 + v_star * t}


def make_sod(path: Path):
    # 3D periodic mirror Sod tube with ISOTROPIC cubic lattices on both sides!
    # Left (rho=1.0, P=1.0) in x in [0, 1): nxL=160, nyL=8, nzL=8 (dx=dy=dz=0.00625, Ly=Lz=0.05)
    # Right (rho=0.25, P=0.1795) in x in [1, 2): nxR=100, nyR=5, nzR=5 -> 160*8*8 / (100*5*5) = 4.096 ~ 4.0 (equal mass!)
    gamma = 5.0 / 3.0
    Lx, Ly, Lz = 2.0, 0.05, 0.05
    nxL, nyL, nzL = 160, 8, 8
    nxR, nyR, nzR = 100, 5, 5
    pts, ms, us, hs = [], [], [], []
    for (x0, nx, ny, nz, rho0, P0) in [
        (0.0, nxL, nyL, nzL, 1.0, 1.0),
        (1.0, nxR, nyR, nzR, 0.25, 0.1795),
    ]:
        dx, dy, dz = 1.0 / nx, Ly / ny, Lz / nz
        xs = x0 + (np.arange(nx) + 0.5) * dx
        ys = (np.arange(ny) + 0.5) * dy
        zs = (np.arange(nz) + 0.5) * dz
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        p = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
        pts.append(p)
        vol_p = dx * dy * dz
        ms.append(np.full(p.shape[0], rho0 * vol_p))
        us.append(np.full(p.shape[0], P0 / ((gamma - 1.0) * rho0)))
        hs.append(np.full(p.shape[0], 1.2348 * (vol_p ** (1.0 / 3.0))))

    x = np.vstack(pts)
    n = x.shape[0]
    m = np.concatenate(ms)
    v = np.zeros((n, 3))
    u = np.concatenate(us)
    h = np.concatenate(hs)
    _write_hydro_ic(path, (Lx, Ly, Lz), (1, 1, 1), gamma, x, v, m, u, h)
    return {"test": "sod", "n_gas": n, "box": [Lx, Ly, Lz], "x_discont": 1.0}


def verify_sod(path: Path, t_end: float = 0.15):
    d = _read_hydro_out(path)
    x = d["x"][:, 0]
    # Evaluate inside [0.55, 1.45] around the x0=1.0 discontinuity (far from periodic wrap at x=0/2)
    mask = (x >= 0.55) & (x <= 1.45)
    xm = x[mask]
    rho_ex, P_ex, v_ex, info = exact_sod_riemann(xm, t_end, x0=1.0, gamma=d["gamma"])
    rho_l1 = float(np.mean(np.abs(d["rho"][mask] - rho_ex)) / np.mean(rho_ex))
    P_l1 = float(np.mean(np.abs(d["P"][mask] - P_ex)) / np.mean(P_ex))
    v_l1 = float(np.mean(np.abs(d["v"][mask, 0] - v_ex)) / max(info["v_star"], 1e-6))
    return {
        "test": "sod",
        "t_end": t_end,
        "rho_L1_rel": rho_l1,
        "P_L1_rel": P_l1,
        "vx_L1_rel": v_l1,
        "exact_info": info,
        "passed": bool(rho_l1 < 0.08 and P_l1 < 0.08),
    }


# ==================== 2. 3D SEDOV-TAYLOR BLAST WAVE ====================
def make_sedov(path: Path, n_side: int = 48):
    gamma = 5.0 / 3.0
    L = 1.0
    dx = L / n_side
    c1d = (np.arange(n_side) + 0.5) * dx
    gx, gy, gz = np.meshgrid(c1d, c1d, c1d, indexing="ij")
    x = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
    n = x.shape[0]
    m_part = 1.0 / n
    m = np.full(n, m_part)
    v = np.zeros((n, 3))
    h = np.full(n, 1.25 * dx)

    # Inject E0 = 1.0 smoothly into central Gaussian of width sigma = 2*dx
    r2 = np.sum((x - 0.5) ** 2, axis=1)
    sig = 2.0 * dx
    w = np.exp(-0.5 * r2 / (sig * sig))
    w /= np.sum(w)
    u = 1e-4 + (1.0 / m_part) * w
    _write_hydro_ic(path, (L, L, L), (1, 1, 1), gamma, x, v, m, u, h)
    return {"test": "sedov", "n_gas": n, "E0": 1.0, "rho0": 1.0}


def verify_sedov(path: Path, t_end: float = 0.06):
    d = _read_hydro_out(path)
    rel = ((d["x"] - 0.5 + 0.5) % 1.0) - 0.5
    r = np.linalg.norm(rel, axis=1)
    # Bin radial density in 40 shells up to r=0.48 to find shock peak radius
    edges = np.linspace(0.0, 0.48, 65)
    rmid = 0.5 * (edges[:-1] + edges[1:])
    rho_bin = np.zeros_like(rmid)
    for k in range(rmid.size):
        sel = (r >= edges[k]) & (r < edges[k + 1])
        if np.any(sel):
            rho_bin[k] = float(np.mean(d["rho"][sel]))
    k_peak = int(np.argmax(rho_bin))
    r_shock_sim = float(rmid[k_peak])
    rho_peak = float(rho_bin[k_peak])
    # Exact 3D Sedov-Taylor (gamma=5/3): R(t) = 1.15167 * (E0/rho0)^(1/5) * t^(2/5)
    r_shock_exact = 1.15167 * (t_end ** 0.4)
    rel_err = abs(r_shock_sim - r_shock_exact) / r_shock_exact
    return {
        "test": "sedov",
        "t_end": t_end,
        "R_shock_sim": r_shock_sim,
        "R_shock_exact": r_shock_exact,
        "R_shock_rel_err": rel_err,
        "rho_peak": rho_peak,
        "passed": bool(rel_err < 0.05 and rho_peak > 2.2),
    }


# ==================== 3. KELVIN-HELMHOLTZ INSTABILITY ====================
def make_kh(path: Path):
    gamma = 5.0 / 3.0
    Lx, Ly, Lz = 1.0, 1.0, 0.0625
    # 2D-in-3D periodic slab: high-density central band y in [0.25, 0.75] (rho=2, vx=+0.5)
    # low-density outer bands (rho=1, vx=-0.5), pressure P=2.5, equal-mass particles!
    nx1, ny1, nz1 = 64, 32, 4   # outer band (total height 0.5, rho=1)
    nx2, ny2, nz2 = 64, 64, 4   # inner band (height 0.5, rho=2 -> 2x particles)
    pts, vs, us, hs = [], [], [], []
    # Inner band y in [0.25, 0.75)
    x2 = (np.arange(nx2) + 0.5) * (Lx / nx2)
    y2 = 0.25 + (np.arange(ny2) + 0.5) * (0.5 / ny2)
    z2 = (np.arange(nz2) + 0.5) * (Lz / nz2)
    gx, gy, gz = np.meshgrid(x2, y2, z2, indexing="ij")
    p2 = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
    v2 = np.zeros_like(p2)
    v2[:, 0] = 0.5
    pts.append(p2); vs.append(v2)
    us.append(np.full(p2.shape[0], 2.5 / ((gamma - 1.0) * 2.0)))
    hs.append(np.full(p2.shape[0], 1.25 * ((Lx * 0.5 * Lz / p2.shape[0]) ** (1.0 / 3.0))))

    # Outer bands y in [0.0, 0.25) and [0.75, 1.0)
    x1 = (np.arange(nx1) + 0.5) * (Lx / nx1)
    y1a = (np.arange(ny1 // 2) + 0.5) * (0.25 / (ny1 // 2))
    y1b = 0.75 + (np.arange(ny1 // 2) + 0.5) * (0.25 / (ny1 // 2))
    y1 = np.concatenate([y1a, y1b])
    z1 = (np.arange(nz1) + 0.5) * (Lz / nz1)
    gx, gy, gz = np.meshgrid(x1, y1, z1, indexing="ij")
    p1 = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
    v1 = np.zeros_like(p1)
    v1[:, 0] = -0.5
    pts.append(p1); vs.append(v1)
    us.append(np.full(p1.shape[0], 2.5 / ((gamma - 1.0) * 1.0)))
    hs.append(np.full(p1.shape[0], 1.25 * ((Lx * 0.5 * Lz / p1.shape[0]) ** (1.0 / 3.0))))

    x = np.vstack(pts)
    v = np.vstack(vs)
    u = np.concatenate(us)
    h = np.concatenate(hs)
    n = x.shape[0]
    # Add classic McNally+2012 sinusoidal transverse seed at y=0.25 and y=0.75 interfaces
    sig_y = 0.025
    vy_seed = 0.025 * np.sin(4.0 * math.pi * x[:, 0]) * (
        np.exp(-0.5 * ((x[:, 1] - 0.25) / sig_y) ** 2)
        + np.exp(-0.5 * ((x[:, 1] - 0.75) / sig_y) ** 2)
    )
    v[:, 1] = vy_seed
    m_part = (1.0 * Lx * 0.5 * Lz) / p1.shape[0]
    m = np.full(n, m_part)
    eky_init = float(0.5 * np.sum(m * (v[:, 1] ** 2)))
    _write_hydro_ic(path, (Lx, Ly, Lz), (1, 1, 1), gamma, x, v, m, u, h)
    return {"test": "kh", "n_gas": n, "E_ky_init": eky_init}


def verify_kh(path: Path, eky_init: float):
    d = _read_hydro_out(path)
    eky_final = float(0.5 * np.sum(d["m"] * (d["v"][:, 1] ** 2)))
    growth = eky_final / max(eky_init, 1e-12)
    return {
        "test": "kh",
        "E_ky_init": eky_init,
        "E_ky_final": eky_final,
        "E_ky_growth_factor": growth,
        "passed": bool(growth > 5.0),
    }


# ==================== 4. RADIATIVE COOLING + SANTA BARBARA CLUSTER ====================
def make_cooling(path: Path):
    gamma = 5.0 / 3.0
    L = 1.0
    n_side = 12
    dx = L / n_side
    c1d = (np.arange(n_side) + 0.5) * dx
    gx, gy, gz = np.meshgrid(c1d, c1d, c1d, indexing="ij")
    x = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
    n = x.shape[0]
    m = np.full(n, 1.0 / n)
    v = np.zeros((n, 3))
    # u_floor = 0.01 -> T = 1e4 K; start at u0 = 1.0 -> T0 = 1e6 K
    u = np.full(n, 1.0)
    h = np.full(n, 1.25 * dx)
    _write_hydro_ic(path, (L, L, L), (1, 1, 1), gamma, x, v, m, u, h)
    return {"test": "cooling", "n_gas": n, "u_init": 1.0, "u_floor": 0.01}


def verify_cooling(path: Path, u_floor: float = 0.01):
    d = _read_hydro_out(path)
    u_med = float(np.median(d["u"]))
    u_max = float(np.max(d["u"]))
    return {
        "test": "cooling",
        "u_median": u_med,
        "u_max": u_max,
        "u_floor": u_floor,
        "passed": bool(abs(u_med - u_floor) / u_floor < 0.05),
    }


def make_santabarbara(path: Path, n_gas: int = 4096, n_dm: int = 4096):
    """3D self-gravitating adiabatic gas + DM cluster collapse (Santa Barbara-style)."""
    gamma = 5.0 / 3.0
    rng = np.random.default_rng(1999)
    box = 20.0
    f_b = 0.15
    M_tot = 1.0
    R_init = 2.0

    def sample_sphere(n_p):
        u_r = rng.uniform(0.0, 1.0, size=n_p)
        # Slightly centrally concentrated r ~ R_init * u^(1/2.5) for clean accretion shock
        r = R_init * (u_r ** (1.0 / 2.5))
        ct = rng.uniform(-1.0, 1.0, size=n_p)
        st = np.sqrt(np.maximum(0.0, 1.0 - ct * ct))
        ph = rng.uniform(0.0, 2.0 * math.pi, size=n_p)
        p = np.column_stack((r * st * np.cos(ph), r * st * np.sin(ph), r * ct))
        p -= p.mean(axis=0, keepdims=True)
        return p + 0.5 * box

    x_gas = sample_sphere(n_gas)
    x_dm = sample_sphere(n_dm)
    v_gas = np.zeros((n_gas, 3))
    v_dm = rng.normal(0.0, 0.05, size=(n_dm, 3))
    v_dm -= v_dm.mean(axis=0, keepdims=True)
    m_gas = np.full(n_gas, f_b * M_tot / n_gas)
    m_dm = np.full(n_dm, (1.0 - f_b) * M_tot / n_dm)
    # Cold initial gas (u0 = 0.01 << virial ~ 0.25) so virial accretion shock generates entropy K(r) ~ r^1.1
    u_gas = np.full(n_gas, 0.01)
    h_gas = np.full(n_gas, 0.22)
    _write_hydro_ic(path, (box, box, box), (0, 0, 0), gamma, x_gas, v_gas, m_gas, u_gas, h_gas, x_dm, v_dm, m_dm)
    return {"test": "santabarbara", "n_gas": n_gas, "n_dm": n_dm, "f_b": f_b}


def verify_santabarbara(path: Path):
    d = _read_hydro_out(path)
    # Center on densest gas particles (top 5% rho)
    top = np.argsort(d["rho"])[-max(32, d["n_gas"] // 20):]
    center = np.median(d["x"][top], axis=0)
    r = np.linalg.norm(d["x"] - center, axis=1)
    # Adiabatic entropy K = P / rho^(5/3) = (gamma - 1) * u / rho^(2/3)
    K = d["P"] / np.power(np.maximum(d["rho"], 1e-12), 5.0 / 3.0)
    edges = np.geomspace(0.08, 1.6, 13)
    r_prof, k_prof, rho_prof, u_prof = [], [], [], []
    for i in range(edges.size - 1):
        sel = (r >= edges[i]) & (r < edges[i + 1])
        if int(np.sum(sel)) >= 15:
            r_prof.append(float(math.sqrt(edges[i] * edges[i + 1])))
            k_prof.append(float(np.median(K[sel])))
            rho_prof.append(float(np.median(d["rho"][sel])))
            u_prof.append(float(np.median(d["u"][sel])))
    # Fit power-law slope d ln K / d ln r in outer virialized halo (r in [0.2, 1.4])
    r_arr, k_arr = np.array(r_prof), np.array(k_prof)
    fit_m = (r_arr >= 0.20) & (r_arr <= 1.40) & (k_arr > 0)
    if int(np.sum(fit_m)) >= 3:
        slope = float(np.polyfit(np.log(r_arr[fit_m]), np.log(k_arr[fit_m]), 1)[0])
    else:
        slope = 0.0
    k_rise = (k_arr[-1] / k_arr[0]) if (k_arr.size >= 2 and k_arr[0] > 0) else 0.0
    return {
        "test": "santabarbara",
        "entropy_slope_dlnK_dlnr": slope,
        "entropy_outer_over_inner": float(k_rise),
        "core_u_shock_heated": float(u_prof[0]) if u_prof else 0.0,
        "r_bins": r_prof,
        "K_bins": k_prof,
        "passed": bool(slope > 0.6 and slope < 1.8 and k_rise > 3.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("bench", choices=["sod", "sedov", "kh", "cooling", "santabarbara"])
    m.add_argument("output", type=Path)
    v = sub.add_parser("verify")
    v.add_argument("bench", choices=["sod", "sedov", "kh", "cooling", "santabarbara"])
    v.add_argument("snapshot", type=Path)
    v.add_argument("--t-end", type=float, default=0.15)
    v.add_argument("--eky-init", type=float, default=1e-4)
    args = ap.parse_args()

    if args.cmd == "make":
        fn = {"sod": make_sod, "sedov": make_sedov, "kh": make_kh,
              "cooling": make_cooling, "santabarbara": make_santabarbara}[args.bench]
        print(json.dumps(fn(args.output), indent=2))
    else:
        if args.bench == "sod":
            res = verify_sod(args.snapshot, args.t_end)
        elif args.bench == "sedov":
            res = verify_sedov(args.snapshot, args.t_end)
        elif args.bench == "kh":
            res = verify_kh(args.snapshot, args.eky_init)
        elif args.bench == "cooling":
            res = verify_cooling(args.snapshot)
        else:
            res = verify_santabarbara(args.snapshot)
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
