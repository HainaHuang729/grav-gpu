#!/usr/bin/env python3
"""Unit tests for scripts/prepare_input.py (SWIFT, monofonIC Gadget shards, MUSIC Zoom-in)."""
import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PREP = ROOT / "scripts" / "prepare_input.py"
CM_PER_MPC = 3.0856775814913673e24
G_PER_1E10_MSUN = 1.988409870698051e43


def read_bin(path: Path):
    data = path.read_bytes()
    n, box, scale, soft = struct.unpack_from("<Q3d", data, 0)
    off = 32
    pos = np.frombuffer(data, dtype="<f8", count=n * 3, offset=off).reshape(n, 3)
    off += n * 3 * 8
    vel = np.frombuffer(data, dtype="<f4", count=n * 3, offset=off).reshape(n, 3)
    off += n * 3 * 4
    mass = np.frombuffer(data, dtype="<f4", count=n, offset=off)
    off += n * 4
    pids = np.frombuffer(data, dtype="<u8", count=n, offset=off)
    off += n * 8
    assert off == len(data), f"Unexpected trailing bytes: {len(data) - off}"
    return n, box, scale, soft, pos, vel, mass, pids


def run_prep(args):
    res = subprocess.run([sys.executable, str(PREP)] + [str(a) for a in args], capture_output=True, text=True)
    assert res.returncode == 0, f"prepare_input failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    return json.loads(res.stdout)


def test_swift_native(tmp: Path):
    src = tmp / "swift_ic.hdf5"
    dst = tmp / "swift_ic.bin"
    n = 64
    rng = np.random.default_rng(42)
    pos = rng.uniform(0.0, 37.114, size=(n, 3))
    vel = rng.normal(0.0, 120.0, size=(n, 3)).astype(np.float32)
    mass = np.full(n, 0.0125, dtype=np.float32)
    pids = np.arange(100, 100 + n, dtype=np.uint64)
    with h5py.File(src, "w") as f:
        hdr = f.create_group("Header")
        hdr.attrs["BoxSize"] = np.array([37.114, 37.114, 37.114])
        hdr.attrs["Scale-factor"] = np.array([0.02])
        u = f.create_group("Units")
        u.attrs["Unit length in cgs (U_L)"] = CM_PER_MPC
        u.attrs["Unit mass in cgs (U_M)"] = G_PER_1E10_MSUN
        pt = f.create_group("PartType1")
        pt.create_dataset("Coordinates", data=pos)
        pt.create_dataset("Velocities", data=vel)
        pt.create_dataset("Masses", data=mass)
        pt.create_dataset("ParticleIDs", data=pids)
        pt.create_dataset("Softenings", data=np.full(n, 0.00217, dtype=np.float32))

    meta = run_prep([src, dst])
    assert meta["format_detected"] == "swift"
    n_out, box, scale, soft, p_out, v_out, m_out, id_out = read_bin(dst)
    assert n_out == n
    assert math.isclose(box, 37.114, rel_tol=1e-12)
    assert math.isclose(scale, 0.02, rel_tol=1e-12)
    assert math.isclose(soft, 0.00217, rel_tol=1e-5)
    np.testing.assert_allclose(p_out, pos, rtol=1e-12)
    np.testing.assert_allclose(v_out, vel, rtol=1e-6)
    np.testing.assert_allclose(m_out, mass, rtol=1e-6)
    np.testing.assert_array_equal(id_out, pids)


def test_monofonic_gadget_shards(tmp: Path):
    # 2-shard Gadget-HDF5 in kpc/h, 1e10 Msun/h, MassTable only, Gadget u = v_pec / sqrt(a)
    h = 0.6736
    a = 0.02
    box_kpch = 25000.0  # 25 Mpc/h -> 25 / 0.6736 Mpc
    m_table_h = 0.05    # 1e10 Msun/h -> 0.05 / h
    rng = np.random.default_rng(7)
    pos_all = rng.uniform(0.0, box_kpch, size=(100, 3))
    v_pec_all = rng.normal(0.0, 50.0, size=(100, 3))
    u_gad_all = (v_pec_all / math.sqrt(a)).astype(np.float32)

    for idx, (s0, s1) in enumerate([(0, 40), (40, 100)]):
        p = tmp / f"mono_ic.{idx}.hdf5"
        with h5py.File(p, "w") as f:
            hdr = f.create_group("Header")
            hdr.attrs["BoxSize"] = box_kpch
            hdr.attrs["Time"] = a
            hdr.attrs["HubbleParam"] = h
            hdr.attrs["Omega0"] = 0.3153
            hdr.attrs["OmegaLambda"] = 0.6847
            hdr.attrs["OmegaBaryon"] = 0.0493
            hdr.attrs["NumFilesPerSnapshot"] = 2
            hdr.attrs["UnitLength_in_cm"] = CM_PER_MPC * 1e-3
            hdr.attrs["UnitMass_in_g"] = G_PER_1E10_MSUN
            hdr.attrs["MassTable"] = np.array([0.0, m_table_h, 0.0, 0.0, 0.0, 0.0])
            pt = f.create_group("PartType1")
            pt.create_dataset("Coordinates", data=pos_all[s0:s1])
            pt.create_dataset("Velocities", data=u_gad_all[s0:s1])
            pt.create_dataset("ParticleIDs", data=np.arange(s0, s1, dtype=np.uint64))

    dst = tmp / "mono_ic.bin"
    env_p = tmp / "mono_ic.cosmo.env"
    meta = run_prep([tmp / "mono_ic", dst, "--softening", "0.003", "--cosmo-env", env_p])
    assert meta["shards"] == 2
    assert meta["unit_scales"]["vel_mode"] == "gadget_sqrt_a"
    n_out, box, scale, soft, p_out, v_out, m_out, id_out = read_bin(dst)
    assert n_out == 100
    assert math.isclose(box, (box_kpch * 1e-3) / h, rel_tol=1e-12)
    np.testing.assert_allclose(p_out, (pos_all * 1e-3) / h, rtol=1e-12)
    np.testing.assert_allclose(v_out, v_pec_all.astype(np.float32), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(m_out, np.full(100, m_table_h / h, dtype=np.float32), rtol=1e-6)
    np.testing.assert_array_equal(id_out, np.arange(100, dtype=np.uint64))
    env_txt = env_p.read_text()
    assert "GRAV_COSMO_H0=67.36" in env_txt
    assert "GRAV_COSMO_OMEGA_M=0.3153" in env_txt


def test_music_zoomin_multimass(tmp: Path):
    h = 0.7
    a = 0.01
    src = tmp / "music_zoom.hdf5"
    dst = tmp / "music_zoom.bin"
    soft_p = tmp / "music_zoom.soft.bin"
    with h5py.File(src, "w") as f:
        hdr = f.create_group("Header")
        hdr.attrs["BoxSize"] = 50.0  # Mpc/h
        hdr.attrs["Time"] = a
        hdr.attrs["HubbleParam"] = h
        hdr.attrs["Omega0"] = 0.3
        hdr.attrs["OmegaLambda"] = 0.7
        hdr.attrs["MassTable"] = np.array([0.0, 0.001, 0.0, 0.0, 0.0, 0.0])
        pt1 = f.create_group("PartType1")
        pt1.create_dataset("Coordinates", data=np.full((30, 3), 25.0))
        pt1.create_dataset("Velocities", data=np.full((30, 3), 10.0, dtype=np.float32))
        pt2 = f.create_group("PartType2")
        pt2.create_dataset("Coordinates", data=np.full((20, 3), 5.0))
        pt2.create_dataset("Velocities", data=np.full((20, 3), -5.0, dtype=np.float32))
        pt2.create_dataset("Masses", data=np.full(20, 0.064, dtype=np.float32))

    meta = run_prep([
        src, dst,
        "--format", "music",
        "--part-types", "PartType1,PartType2",
        "--type-softenings", "PartType1:0.0005,PartType2:0.008",
        "--soft-out", soft_p,
    ])
    assert meta["particles"] == 50
    n_out, box, scale, soft, p_out, v_out, m_out, id_out = read_bin(dst)
    assert n_out == 50
    assert math.isclose(box, 50.0 / h, rel_tol=1e-12)
    np.testing.assert_allclose(m_out[:30], 0.001 / h, rtol=1e-6)
    np.testing.assert_allclose(m_out[30:], 0.064 / h, rtol=1e-6)
    np.testing.assert_allclose(v_out[:30], 10.0 * math.sqrt(a), rtol=1e-6)
    np.testing.assert_allclose(v_out[30:], -5.0 * math.sqrt(a), rtol=1e-6)
    soft_arr = np.fromfile(soft_p, dtype="<f4")
    assert len(soft_arr) == 50
    np.testing.assert_allclose(soft_arr[:30], 0.0005, rtol=1e-6)
    np.testing.assert_allclose(soft_arr[30:], 0.008, rtol=1e-6)
    # Fallback sequential IDs across merged types
    np.testing.assert_array_equal(id_out, np.arange(50, dtype=np.uint64))


def main():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_swift_native(tmp)
        test_monofonic_gadget_shards(tmp)
        test_music_zoomin_multimass(tmp)
    print("IC_ADAPTER_TESTS_PASSED")


if __name__ == "__main__":
    main()
