#!/usr/bin/env python3
"""Generate a small synthetic clustered periodic dark-matter IC in dense .bin format for smoke tests."""
import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a synthetic periodic dark-matter IC .bin for smoke tests.")
    ap.add_argument("output", type=Path, help="Output .bin path")
    ap.add_argument("--n-side", type=int, default=64, help="N = n_side^3 particles (default: 64 -> 262,144)")
    ap.add_argument("--box", type=float, default=20.0, help="Periodic cubic box size [Mpc] (default: 20.0)")
    ap.add_argument("--a-init", type=float, default=0.85, help="Initial scale factor a_begin (default: 0.85)")
    ap.add_argument("--softening", type=float, default=0.01, help="Plummer softening [Mpc] (default: 0.01)")
    ap.add_argument("--n-clusters", type=int, default=16, help="Number of dense NFW-like Gaussian clusters")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = ap.parse_args()

    n = int(args.n_side) ** 3
    box = float(args.box)
    rng = np.random.default_rng(args.seed)

    # Mean comoving matter density in SWIFT internal units (L=Mpc, M=1e10 Msun, G=43.0092)
    h0 = 67.36
    omega_m = 0.3153
    grav_g = 43.0092
    rho_crit0 = 3.0 * h0 * h0 / (8.0 * math.pi * grav_g)
    total_mass = omega_m * rho_crit0 * (box ** 3)
    m0 = np.float32(total_mass / float(n))

    pos = np.empty((n, 3), dtype="<f8")
    vel = rng.normal(0.0, 25.0, size=(n, 3)).astype("<f4")

    # Place ~25% of particles in dense clusters (including one straddling the periodic boundary x~0/L)
    per_cluster = max(256, (n // 4) // max(1, args.n_clusters))
    n_clustered = min(n // 2, per_cluster * args.n_clusters)
    centers = rng.uniform(0.0, box, size=(args.n_clusters, 3))
    if args.n_clusters >= 1:
        centers[0] = np.array([0.02, box - 0.02, 0.5 * box], dtype=np.float64)

    sigma_pos = 0.15 * (box / float(args.n_side)) * 2.5
    cur = 0
    for c in range(args.n_clusters):
        cnt = min(per_cluster, n_clustered - cur)
        if cnt <= 0:
            break
        pos[cur : cur + cnt] = np.mod(centers[c] + rng.normal(0.0, sigma_pos, size=(cnt, 3)), box)
        # Local virialized internal velocities + bulk peculiar velocity
        vel[cur : cur + cnt] += rng.normal(rng.normal(0.0, 80.0, size=(3,)), 140.0, size=(cnt, 3)).astype("<f4")
        cur += cnt

    if cur < n:
        pos[cur:] = rng.uniform(0.0, box, size=(n - cur, 3))

    masses = np.full(n, m0, dtype="<f4")
    pids = np.arange(1, n + 1, dtype="<u8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as out:
        out.write(struct.pack("<Q3d", n, box, float(args.a_init), float(args.softening)))
        pos.tofile(out)
        vel.tofile(out)
        masses.tofile(out)
        pids.tofile(out)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "particles": n,
                "box_mpc": box,
                "a_init": args.a_init,
                "softening": args.softening,
                "particle_mass_1e10msun": float(m0),
                "clusters": args.n_clusters,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
