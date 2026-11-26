#!/usr/bin/env python3
"""Generate an equilibrium N-body Plummer sphere IC in dense .bin format for isolated conservation tests."""
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate an equilibrium isolated Plummer sphere IC .bin.")
    ap.add_argument("output", type=Path, help="Output .bin path")
    ap.add_argument("--n", type=int, default=32768, help="Number of particles (default: 32768)")
    ap.add_argument("--box", type=float, default=64.0, help="Box size (default: 64.0, center at box/2)")
    ap.add_argument("--scale-radius", type=float, default=1.0, help="Plummer scale radius a_pl (default: 1.0)")
    ap.add_argument("--total-mass", type=float, default=1.0, help="Total sphere mass M (default: 1.0)")
    ap.add_argument("--grav-g", type=float, default=1.0, help="Gravitational constant G (default: 1.0)")
    ap.add_argument("--softening", type=float, default=0.05, help="Plummer softening (default: 0.05)")
    ap.add_argument("--r-max", type=float, default=15.0, help="Outer cutoff radius (default: 15.0)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = ap.parse_args()

    n = int(args.n)
    a_pl = float(args.scale_radius)
    m_tot = float(args.total_mass)
    grav_g = float(args.grav_g)
    box = float(args.box)
    r_max = float(args.r_max)
    rng = np.random.default_rng(args.seed)

    # Sample radii from exact Plummer CDF with optional truncation at r_max
    u_max = 1.0 / math.pow(1.0 + (a_pl / r_max) ** 2, 1.5)
    u = rng.uniform(0.0, u_max, size=n)
    r = a_pl / np.sqrt(np.power(u, -2.0 / 3.0) - 1.0)
    cos_th = rng.uniform(-1.0, 1.0, size=n)
    sin_th = np.sqrt(np.maximum(0.0, 1.0 - cos_th ** 2))
    phi = rng.uniform(0.0, 2.0 * math.pi, size=n)
    pos = np.column_stack((
        r * sin_th * np.cos(phi),
        r * sin_th * np.sin(phi),
        r * cos_th,
    ))

    # Aarseth / Henon (1974) rejection sampling for isotropic f(E) ~ (-E)^(7/2)
    q = np.empty(n, dtype=np.float64)
    filled = 0
    while filled < n:
        batch = max(1024, (n - filled) * 4)
        x1 = rng.uniform(0.0, 1.0, size=batch)
        x2 = rng.uniform(0.0, 0.1, size=batch)
        acc = x1[x2 < (x1 ** 2) * np.power(1.0 - x1 ** 2, 3.5)]
        take = min(len(acc), n - filled)
        q[filled : filled + take] = acc[:take]
        filled += take

    v_esc = np.sqrt(2.0 * grav_g * m_tot / np.sqrt(r ** 2 + a_pl ** 2))
    v_mag = q * v_esc
    cv = rng.uniform(-1.0, 1.0, size=n)
    sv = np.sqrt(np.maximum(0.0, 1.0 - cv ** 2))
    pv = rng.uniform(0.0, 2.0 * math.pi, size=n)
    vel = np.column_stack((
        v_mag * sv * np.cos(pv),
        v_mag * sv * np.sin(pv),
        v_mag * cv,
    ))

    # Exact zero-momentum & zero-COM frame centered in box/2
    pos -= pos.mean(axis=0, keepdims=True)
    vel -= vel.mean(axis=0, keepdims=True)
    pos_box = (pos + 0.5 * box).astype("<f8")
    vel_f4 = vel.astype("<f4")
    m0 = np.float32(m_tot / float(n))
    masses = np.full(n, m0, dtype="<f4")
    pids = np.arange(1, n + 1, dtype="<u8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as out:
        out.write(struct.pack("<Q3d", n, box, 0.0, float(args.softening)))
        pos_box.tofile(out)
        vel_f4.tofile(out)
        masses.tofile(out)
        pids.tofile(out)

    ke = 0.5 * float(m0) * float(np.sum(vel ** 2))
    print(json.dumps({
        "output": str(args.output),
        "n": n,
        "box": box,
        "a_pl": a_pl,
        "total_mass": m_tot,
        "grav_g": grav_g,
        "softening": args.softening,
        "t_dyn_approx": math.sqrt((a_pl ** 3) / (grav_g * m_tot)),
        "initial_T_exact": ke,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
