#!/usr/bin/env python3
"""Verify end-to-end smoke run outputs (.bin, .fof.bin, log) and exercise GPU Catalog + SO via ctypes."""
import ctypes
import json
import math
import struct
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
    if len(sys.argv) != 5:
        raise SystemExit(
            f"Usage: {sys.argv[0]} <evolved.bin> <evolved.bin.fof.bin> <run.log> <swift_cuda_fof.so>"
        )
    out_bin, fof_bin, log_path, so_path = map(Path, sys.argv[1:5])

    log_txt = log_path.read_text()
    san = json.loads([l.split("SANITY ", 1)[1] for l in log_txt.splitlines() if l.startswith("SANITY ")][-1])
    res = json.loads([l.split("RESULT ", 1)[1] for l in log_txt.splitlines() if l.startswith("RESULT ")][-1])
    fof_res = json.loads([l.split("FOF_RESULT ", 1)[1] for l in log_txt.splitlines() if l.startswith("FOF_RESULT ")][-1])
    assert san["bad_count"] == 0, san

    with out_bin.open("rb") as f:
        n, box, a = struct.unpack("<Q2d", f.read(24))
    pos = np.ascontiguousarray(np.memmap(out_bin, dtype="<f8", mode="r", offset=24, shape=(n, 3)))
    vel = np.ascontiguousarray(np.memmap(out_bin, dtype="<f4", mode="r", offset=24 + 24 * n, shape=(n, 3)))

    with fof_bin.open("rb") as f:
        n_f, ngrp, largest, min_sz = struct.unpack("<4Q", f.read(32))
        fof_ids = np.ascontiguousarray(np.memmap(fof_bin, dtype="<i8", mode="r", offset=56, shape=(n_f,)))
    assert n_f == n and ngrp == fof_res["groups"] and ngrp > 0

    h0, omega_m, grav_g = 67.36, 0.3153, 43.0092
    omega_de = 1.0 - omega_m
    rho_crit0 = 3.0 * h0 * h0 / (8.0 * math.pi * grav_g)
    m0 = omega_m * rho_crit0 * (box ** 3) / float(n)
    h_a = h0 * math.sqrt(omega_m / (a ** 3) + omega_de)
    rho_crit_com = (a ** 3) * 3.0 * h_a * h_a / (8.0 * math.pi * grav_g)
    rho_mean_com = omega_m * rho_crit0

    lib = ctypes.CDLL(str(so_path))
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

    halos = (FofHaloRow * ngrp)()
    rc1 = lib.compute_fof_catalog(
        pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        vel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_float)(), m0,
        fof_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        n, ngrp, box, box, box, halos)
    assert rc1 == 0, f"compute_fof_catalog returned {rc1}"

    centers = np.ascontiguousarray([tuple(h.com) for h in halos], dtype=np.float64)
    sizes = np.ascontiguousarray([h.size for h in halos], dtype=np.uint64)
    masses = np.ascontiguousarray([h.mass for h in halos], dtype=np.float64)
    rmax = np.ascontiguousarray([h.r_max for h in halos], dtype=np.float64)
    so = (FofSoRow * ngrp)()
    rc2 = lib.compute_fof_so(
        pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ctypes.POINTER(ctypes.c_float)(), m0, n,
        centers.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        masses.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        rmax.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        ngrp, box, box, box, rho_crit_com, rho_mean_com, 32, so)
    assert rc2 == 0, f"compute_fof_so returned {rc2}"
    assert so[0].m200c > 0.0 and so[0].r200c > 0.0

    print(
        f"ALL_SMOKE_TESTS_PASSED mode={res['domain_mode']} gpus={res['num_gpus']} "
        f"evo_s={res['evolution_wall_s']:.3f} groups={ngrp} largest={largest} "
        f"top_m200c={so[0].m200c:.4g} top_r200c={so[0].r200c:.4g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
