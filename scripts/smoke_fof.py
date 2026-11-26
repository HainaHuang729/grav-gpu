#!/usr/bin/env python3
"""Standalone ctypes smoke & parity test for build/grav_cuda_fof.so (no external data needed)."""
import ctypes
import math
import sys
from pathlib import Path

import numpy as np


class FofHaloRow(ctypes.Structure):
    _fields_ = [
        ("group_id", ctypes.c_int64),
        ("size", ctypes.c_uint64),
        ("mass", ctypes.c_double),
        ("com", ctypes.c_double * 3),
        ("bulk_vel", ctypes.c_double * 3),
        ("vel_disp", ctypes.c_double),
        ("r_max", ctypes.c_double),
    ]


class FofSoRow(ctypes.Structure):
    _fields_ = [
        ("m200c", ctypes.c_double),
        ("r200c", ctypes.c_double),
        ("m200m", ctypes.c_double),
        ("r200m", ctypes.c_double),
    ]


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <path/to/grav_cuda_fof.so>")
    lib_path = Path(sys.argv[1])
    lib = ctypes.CDLL(str(lib_path))

    assert ctypes.sizeof(FofHaloRow) == 88
    assert ctypes.sizeof(FofSoRow) == 32

    lib.whole_fof_ranked.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.whole_fof_ranked.restype = ctypes.c_int

    lib.whole_fof_slabbed.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.c_uint64,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint64),
    ]
    lib.whole_fof_slabbed.restype = ctypes.c_int

    lib.compute_fof_catalog.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_double, ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64, ctypes.c_uint64, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.POINTER(FofHaloRow),
    ]
    lib.compute_fof_catalog.restype = ctypes.c_int

    lib.compute_fof_so.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_float),
        ctypes.c_double, ctypes.c_uint64, ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint64, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.POINTER(FofSoRow),
    ]
    lib.compute_fof_so.restype = ctypes.c_int

    rng = np.random.default_rng(123)
    n = 200_000
    box = 25.0
    pos = rng.uniform(0.0, box, size=(n, 3)).astype(np.float64)
    vel = rng.normal(0.0, 50.0, size=(n, 3)).astype(np.float32)

    # Plant 12 well-separated dense halos, including one directly across the periodic corner (0, L, 0)
    n_halos = 12
    per_halo = 2500
    centers = [np.array([0.01, box - 0.01, 0.02], dtype=np.float64)]
    while len(centers) < n_halos:
        cand = rng.uniform(2.0, box - 2.0, size=(3,))
        if all(np.linalg.norm(((cand - c + 0.5 * box) % box) - 0.5 * box) >= 2.0 for c in centers):
            centers.append(cand)
    centers = np.asarray(centers)
    for h in range(n_halos):
        s = h * per_halo
        e = s + per_halo
        pos[s:e] = np.mod(centers[h] + rng.normal(0.0, 0.04, size=(per_halo, 3)), box)
        vel[s:e] = rng.normal(100.0 * (h + 1), 15.0, size=(per_halo, 3)).astype(np.float32)

    link = 0.2 * box / (n ** (1.0 / 3.0))
    ids_ref = np.empty(n, dtype=np.int64)
    ids_slab = np.empty(n, dtype=np.int64)
    ng1, gp1, lg1 = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
    ng2, gp2, lg2 = ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()

    rc = lib.whole_fof_ranked(
        pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_int(n),
        box, box, box, link, 32, ids_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(ng1), ctypes.byref(gp1), ctypes.byref(lg1), None, None)
    assert rc == 0, f"whole_fof_ranked failed: {rc}"

    rc = lib.whole_fof_slabbed(
        pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), ctypes.c_uint64(n),
        box, box, box, link, 32, 4, ids_slab.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(ng2), ctypes.byref(gp2), ctypes.byref(lg2), None, None)
    assert rc == 0, f"whole_fof_slabbed failed: {rc}"
    assert ng1.value == n_halos and ng2.value == ng1.value
    assert gp1.value == gp2.value and lg1.value == lg2.value
    assert np.array_equal(ids_ref, ids_slab), "Mismatch between whole_fof_ranked and whole_fof_slabbed"

    m0 = 1.0
    rows = (FofHaloRow * ng1.value)()
    rc = lib.compute_fof_catalog(
        pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        vel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_float)(), m0,
        ids_ref.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n, ng1.value, box, box, box, rows)
    assert rc == 0, f"compute_fof_catalog failed: {rc}"
    assert rows[0].size >= 2400 and rows[0].r_max > 0.0 and rows[0].vel_disp > 0.0

    # Verify SO calculation produces finite M200c/R200c and M200m/R200m
    rho_crit = 3.0 * (67.36 ** 2) / (8.0 * math.pi * 43.0092) * 1e-4
    rho_mean = 0.3153 * rho_crit
    centers_out = np.ascontiguousarray([tuple(r.com) for r in rows], dtype=np.float64)
    sizes_out = np.ascontiguousarray([r.size for r in rows], dtype=np.uint64)
    masses_out = np.ascontiguousarray([r.mass for r in rows], dtype=np.float64)
    rmax_out = np.ascontiguousarray([r.r_max for r in rows], dtype=np.float64)
    so = (FofSoRow * ng1.value)()
    rc = lib.compute_fof_so(
        pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_float)(), m0, n,
        centers_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sizes_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        masses_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rmax_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ng1.value, box, box, box, rho_crit, rho_mean, 32, so)
    assert rc == 0, f"compute_fof_so failed: {rc}"
    for r_so in so:
        assert r_so.m200c > 0.0 and r_so.r200c > 0.0 and r_so.m200m > 0.0 and r_so.r200m > 0.0

    print(
        f"FOF_SMOKE_PASSED groups={ng1.value} grouped={gp1.value} largest={lg1.value} "
        f"top_m200c={so[0].m200c:.2f} top_r200c={so[0].r200c:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
