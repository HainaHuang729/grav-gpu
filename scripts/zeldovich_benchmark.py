#!/usr/bin/env python3
"""Generate and analytically verify a 1D Einstein-de Sitter Zel'dovich plane-wave collapse benchmark.

Physics (EdS: Omega_m = 1.0, Omega_Lambda = 0.0, H(a) = H0 * a^{-3/2}):
  x(q, a) = (q_x - (a / a_c) * sin(k * q_x) / k) mod L
  v_pec,x(q, a) = a * dx/dt = -H0 * sqrt(a) * sin(k * q_x) / (k * a_c)

Note: multi_gpu_slab reads/writes peculiar velocities v_pec = a * dx/dt in the .bin file
(scaling internally to canonical p = a * v_pec on load and back by 1/a on save).
"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np


G_GRAV = 43.0092
H0_DEFAULT = 67.36
BOX_DEFAULT = 37.114
SOFT_DEFAULT = 0.01


def _eds_mass_per_particle(n_total: int, box: float, h0: float) -> float:
    rho_crit0 = 3.0 * (h0 ** 2) / (8.0 * math.pi * G_GRAV)
    return rho_crit0 * (box ** 3) / float(n_total)


def generate_ic(
    output: Path,
    n_side: int = 64,
    box: float = BOX_DEFAULT,
    h0: float = H0_DEFAULT,
    a_init: float = 0.02,
    a_cross: float = 1.0,
    softening: float = SOFT_DEFAULT,
) -> dict:
    n_total = n_side ** 3
    k = 2.0 * math.pi / box
    dq = box / float(n_side)
    # 0.5-cell staggered grid removes zero-displacement aliasing along the wave
    coords_1d = (np.arange(n_side, dtype=np.float64) + 0.5) * dq
    qx, qy, qz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    qx_f = qx.ravel()
    qy_f = qy.ravel()
    qz_f = qz.ravel()

    disp = -(a_init / a_cross) * np.sin(k * qx_f) / k
    x = np.mod(qx_f + disp, box)
    y = qy_f.copy()
    z = qz_f.copy()
    pos = np.column_stack((x, y, z)).astype("<f8")

    # Peculiar velocity v_pec = a * dx/dt in EdS
    vx = (-h0 * math.sqrt(a_init) * np.sin(k * qx_f) / (k * a_cross)).astype("<f4")
    vel = np.column_stack((vx, np.zeros_like(vx), np.zeros_like(vx))).astype("<f4")

    m_part = _eds_mass_per_particle(n_total, box, h0)
    masses = np.full(n_total, np.float32(m_part), dtype="<f4")
    # Store 1-based ParticleIDs matching (ix, iy, iz) flattening
    pids = np.arange(1, n_total + 1, dtype="<u8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        f.write(struct.pack("<Q3d", n_total, box, a_init, softening))
        pos.tofile(f)
        vel.tofile(f)
        masses.tofile(f)
        pids.tofile(f)

    return {
        "output": str(output),
        "n_side": n_side,
        "n_total": n_total,
        "box": box,
        "h0": h0,
        "a_init": a_init,
        "a_cross": a_cross,
        "softening": softening,
        "m_particle": m_part,
        "delta_max_init": a_init / a_cross,
    }


def _read_snapshot_bin(path: Path):
    data = path.read_bytes()
    header_sz = 32
    n, box, scale = struct.unpack("<Q2d", data[:24])
    rem = len(data) - header_sz
    # Check whether file has footer-like 24B header or 32B StateInput header:
    # multi_gpu_slab writes Footer{N, box, a} (24B) followed by pos (24*N), vel (12*N), pot (4*N)
    # OR if ids included, check exact size!
    expected_24_nopot = 24 + n * (24 + 12)
    expected_24_pot = 24 + n * (24 + 12 + 4)
    expected_24_full = 24 + n * (24 + 12 + 4 + 8)
    expected_32_ic = 32 + n * (24 + 12 + 4 + 8)
    if len(data) == expected_24_pot:
        off = 24
        pos = np.frombuffer(data, dtype="<f8", count=n * 3, offset=off).reshape(n, 3)
        vel = np.frombuffer(data, dtype="<f4", count=n * 3, offset=off + n * 24).reshape(n, 3)
        return n, box, scale, pos, vel
    if len(data) == expected_24_nopot:
        off = 24
        pos = np.frombuffer(data, dtype="<f8", count=n * 3, offset=off).reshape(n, 3)
        vel = np.frombuffer(data, dtype="<f4", count=n * 3, offset=off + n * 24).reshape(n, 3)
        return n, box, scale, pos, vel
    if len(data) == expected_24_full:
        off = 24
        pos = np.frombuffer(data, dtype="<f8", count=n * 3, offset=off).reshape(n, 3)
        vel = np.frombuffer(data, dtype="<f4", count=n * 3, offset=off + n * 24).reshape(n, 3)
        return n, box, scale, pos, vel
    if len(data) == expected_32_ic:
        n, box, scale, _ = struct.unpack("<Q3d", data[:32])
        off = 32
        pos = np.frombuffer(data, dtype="<f8", count=n * 3, offset=off).reshape(n, 3)
        vel = np.frombuffer(data, dtype="<f4", count=n * 3, offset=off + n * 24).reshape(n, 3)
        return n, box, scale, pos, vel
    raise ValueError(f"Unexpected snapshot byte size {len(data)} for N={n}")


def verify_output(
    snapshot: Path,
    n_side: int = 64,
    h0: float = H0_DEFAULT,
    a_cross: float = 1.0,
) -> dict:
    n, box, a_out, pos, vel = _read_snapshot_bin(snapshot)
    if n != n_side ** 3:
        raise ValueError(f"Expected {n_side**3} particles, got {n}")

    k = 2.0 * math.pi / box
    dq = box / float(n_side)
    coords_1d = (np.arange(n_side, dtype=np.float64) + 0.5) * dq
    qx, qy, qz = np.meshgrid(coords_1d, coords_1d, coords_1d, indexing="ij")
    qx_f = qx.ravel()
    qy_f = qy.ravel()
    qz_f = qz.ravel()

    x_exact = np.mod(qx_f - (a_out / a_cross) * np.sin(k * qx_f) / k, box)
    vx_exact = -h0 * math.sqrt(a_out) * np.sin(k * qx_f) / (k * a_cross)

    dx = pos[:, 0] - x_exact
    dx = (dx + 0.5 * box) % box - 0.5 * box
    dy = (pos[:, 1] - qy_f + 0.5 * box) % box - 0.5 * box
    dz = (pos[:, 2] - qz_f + 0.5 * box) % box - 0.5 * box
    dr = np.sqrt(dx * dx + dy * dy + dz * dz)

    dvx = vel[:, 0].astype(np.float64) - vx_exact
    dvy = vel[:, 1].astype(np.float64)
    dvz = vel[:, 2].astype(np.float64)
    dv = np.sqrt(dvx * dvx + dvy * dvy + dvz * dvz)

    v_peak = h0 * math.sqrt(a_out) / (k * a_cross)
    x_amp = (a_out / a_cross) / k

    return {
        "n_particles": int(n),
        "box_mpc": float(box),
        "a_end": float(a_out),
        "delta_max": float(a_out / a_cross),
        "displacement_amp_mpc": float(x_amp),
        "velocity_peak_km_s": float(v_peak),
        "pos_rms_mpc": float(np.sqrt(np.mean(dr ** 2))),
        "pos_max_mpc": float(np.max(dr)),
        "pos_rms_over_box": float(np.sqrt(np.mean(dr ** 2)) / box),
        "pos_rms_over_amp": float(np.sqrt(np.mean(dr ** 2)) / x_amp),
        "vel_rms_km_s": float(np.sqrt(np.mean(dv ** 2))),
        "vel_max_km_s": float(np.max(dv)),
        "vel_rms_over_vpeak": float(np.sqrt(np.mean(dv ** 2)) / v_peak),
        "vel_transverse_rms_km_s": float(np.sqrt(np.mean(dvy ** 2 + dvz ** 2))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("output", type=Path)
    g.add_argument("--n-side", type=int, default=64)
    g.add_argument("--box", type=float, default=BOX_DEFAULT)
    g.add_argument("--h0", type=float, default=H0_DEFAULT)
    g.add_argument("--a-init", type=float, default=0.02)
    g.add_argument("--a-cross", type=float, default=1.0)
    g.add_argument("--softening", type=float, default=SOFT_DEFAULT)

    v = sub.add_parser("verify")
    v.add_argument("snapshot", type=Path)
    v.add_argument("--n-side", type=int, default=64)
    v.add_argument("--h0", type=float, default=H0_DEFAULT)
    v.add_argument("--a-cross", type=float, default=1.0)
    v.add_argument("--out-json", type=Path, default=None)

    args = ap.parse_args()
    if args.cmd == "generate":
        res = generate_ic(args.output, args.n_side, args.box, args.h0, args.a_init, args.a_cross, args.softening)
        print(json.dumps(res, indent=2))
    else:
        res = verify_output(args.snapshot, args.n_side, args.h0, args.a_cross)
        if args.out_json:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json.dumps(res, indent=2) + "\n")
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
