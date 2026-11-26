"""Convert a GPU TreePM snapshot (.bin, arbitrary particle count) into a SWIFT HDF5 snapshot,
reading Masses and ParticleIDs directly from the original input HDF5 snapshot, and optionally
attaching GPU-computed PartType1/FOFGroupIDs (.fof.bin) and FOFGroups/* catalog datasets."""
import argparse
import ctypes
import struct
from pathlib import Path

import h5py
import numpy as np

GRAV_FOF_NONE = 2147483647


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


HALO_ROW_DTYPE = np.dtype([
    ("group_id", "<i8"),
    ("size", "<u8"),
    ("mass", "<f8"),
    ("com", "<f8", (3,)),
    ("bulk_vel", "<f8", (3,)),
    ("vel_disp", "<f8"),
    ("r_max", "<f8"),
])


class FofSoRow(ctypes.Structure):
    _fields_ = [
        ("m200c", ctypes.c_double),
        ("r200c", ctypes.c_double),
        ("m200m", ctypes.c_double),
        ("r200m", ctypes.c_double),
    ]


SO_ROW_DTYPE = np.dtype([
    ("m200c", "<f8"),
    ("r200c", "<f8"),
    ("m200m", "<f8"),
    ("r200m", "<f8"),
])


def _cosmo_densities(a, h0, omega_m, omega_de, omega_r, omega_k, w0, wa, grav_g):
    inv_a = 1.0 / a
    a2 = inv_a * inv_a
    a3 = a2 * inv_a
    a4 = a2 * a2
    de_fac = 1.0
    if w0 != -1.0 or wa != 0.0:
        de_fac = (a ** (-3.0 * (1.0 + w0 + wa))) * np.exp(-3.0 * wa * (1.0 - a))
    h_a = h0 * np.sqrt(omega_r * a4 + omega_m * a3 + omega_k * a2 + omega_de * de_fac)
    rho_crit_0 = 3.0 * h0 * h0 / (8.0 * np.pi * grav_g)
    rho_crit_com = (a ** 3) * 3.0 * h_a * h_a / (8.0 * np.pi * grav_g)
    rho_mean_com = omega_m * rho_crit_0
    return float(rho_crit_com), float(rho_mean_com)


def _write_fof_groups(out, orig_parts, n, box, a, fof_bin, fof_lib_path, endpoint, masses_arr, first_mass, ngrp, args):
    if ctypes.sizeof(FofHaloRow) != 88 or HALO_ROW_DTYPE.itemsize != 88:
        raise RuntimeError("FofHaloRow ABI size mismatch")
    if ctypes.sizeof(FofSoRow) != 32 or SO_ROW_DTYPE.itemsize != 32:
        raise RuntimeError("FofSoRow ABI size mismatch")
    lib = ctypes.CDLL(str(fof_lib_path))
    lib.compute_fof_catalog.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(FofHaloRow),
    ]
    lib.compute_fof_catalog.restype = ctypes.c_int
    lib.compute_fof_so.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_uint64,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.POINTER(FofSoRow),
    ]
    lib.compute_fof_so.restype = ctypes.c_int

    so_arr = None
    if ngrp > 0:
        pos_mm = np.memmap(endpoint, dtype="<f8", mode="r", offset=24, shape=(n, 3))
        vel_mm = np.memmap(endpoint, dtype="<f4", mode="r", offset=24 + 24 * n, shape=(n, 3))
        fof_mm = np.memmap(fof_bin, dtype="<i8", mode="r", offset=56, shape=(n,))
        halos = (FofHaloRow * ngrp)()
        mass_ptr = (
            masses_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if masses_arr is not None
            else ctypes.POINTER(ctypes.c_float)()
        )
        rc = lib.compute_fof_catalog(
            pos_mm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            vel_mm.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mass_ptr,
            ctypes.c_double(float(first_mass)),
            fof_mm.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_uint64(n),
            ctypes.c_uint64(ngrp),
            ctypes.c_double(box),
            ctypes.c_double(box),
            ctypes.c_double(box),
            halos,
        )
        if rc != 0:
            raise RuntimeError(f"compute_fof_catalog failed with code {rc}")
        arr = np.frombuffer(halos, dtype=HALO_ROW_DTYPE, count=ngrp)

        if args.so_min_size >= 0:
            rho_crit_com, rho_mean_com = _cosmo_densities(
                a, args.h0, args.omega_m, args.omega_de,
                args.omega_r, args.omega_k, args.w0, args.wa, args.grav_g,
            )
            centers = np.ascontiguousarray(arr["com"], dtype="<f8")
            sizes_u64 = np.ascontiguousarray(arr["size"], dtype="<u8")
            fof_m = np.ascontiguousarray(arr["mass"], dtype="<f8")
            rmax_f64 = np.ascontiguousarray(arr["r_max"], dtype="<f8")
            so_buf = (FofSoRow * ngrp)()
            rc_so = lib.compute_fof_so(
                pos_mm.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                mass_ptr,
                ctypes.c_double(float(first_mass)),
                ctypes.c_uint64(n),
                centers.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                sizes_u64.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                fof_m.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                rmax_f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                ctypes.c_uint64(ngrp),
                ctypes.c_double(box),
                ctypes.c_double(box),
                ctypes.c_double(box),
                ctypes.c_double(rho_crit_com),
                ctypes.c_double(rho_mean_com),
                ctypes.c_int(args.so_min_size),
                so_buf,
            )
            if rc_so != 0:
                raise RuntimeError(f"compute_fof_so failed with code {rc_so}")
            so_arr = np.frombuffer(so_buf, dtype=SO_ROW_DTYPE, count=ngrp)
    else:
        arr = np.empty(0, dtype=HALO_ROW_DTYPE)
        if args.so_min_size >= 0:
            so_arr = np.empty(0, dtype=SO_ROW_DTYPE)

    fg = out.create_group("FOFGroups")
    specs = [
        ("GroupIDs", "<i8", arr["group_id"].astype("<i8")),
        ("Sizes", "<i8", arr["size"].astype("<i8")),
        ("Masses", "<f8", arr["mass"].astype("<f8")),
        ("CentresOfMass", "<f8", arr["com"].astype("<f8")),
        ("BulkVelocities", "<f4", arr["bulk_vel"].astype("<f4")),
        ("VelocityDispersions", "<f4", arr["vel_disp"].astype("<f4")),
        ("RadiiMax", "<f4", arr["r_max"].astype("<f4")),
    ]
    if so_arr is not None:
        specs.extend([
            ("M200c", "<f8", so_arr["m200c"].astype("<f8")),
            ("R200c", "<f4", so_arr["r200c"].astype("<f4")),
            ("M200m", "<f8", so_arr["m200m"].astype("<f8")),
            ("R200m", "<f4", so_arr["r200m"].astype("<f4")),
        ])
    chunk_len = max(1, min(8192, max(1, ngrp)))
    for dname, dtype, vals in specs:
        shape = vals.shape
        ds = fg.create_dataset(
            dname, shape=shape, dtype=dtype,
            chunks=(chunk_len,) + shape[1:] if ngrp > 0 else None,
            compression="gzip" if ngrp > 0 else None,
            compression_opts=1 if ngrp > 0 else None,
        )
        ds[...] = vals
        if dname in ("Masses", "M200c", "M200m", "CentresOfMass", "R200c", "R200m") and orig_parts is not None:
            ref_name = "Masses" if dname in ("Masses", "M200c", "M200m") else "Coordinates"
            if ref_name in orig_parts:
                for k, v in orig_parts[ref_name].attrs.items():
                    if k in ("Conversion factor to CGS (not including cosmological corrections)",
                             "Conversion factor to physical CGS (including cosmological corrections)",
                             "Expression for physical CGS units",
                             "U_I exponent", "U_L exponent", "U_M exponent",
                             "U_T exponent", "U_t exponent", "a-scale exponent", "h-scale exponent"):
                        ds.attrs[k] = v
        elif dname in ("BulkVelocities", "VelocityDispersions") and orig_parts is not None and "Velocities" in orig_parts:
            for k, v in orig_parts["Velocities"].attrs.items():
                if k in ("Conversion factor to CGS (not including cosmological corrections)",
                         "Conversion factor to physical CGS (including cosmological corrections)",
                         "Expression for physical CGS units",
                         "U_I exponent", "U_L exponent", "U_M exponent",
                         "U_T exponent", "U_t exponent", "a-scale exponent", "h-scale exponent"):
                    ds.attrs[k] = v


def _set_fof_attrs(ds, orig_parts):
    if "FOFGroupIDs" in orig_parts:
        ds.attrs.update(orig_parts["FOFGroupIDs"].attrs)
        return
    ds.attrs["Conversion factor to CGS (not including cosmological corrections)"] = np.array([1.0], dtype=np.float64)
    ds.attrs["Conversion factor to physical CGS (including cosmological corrections)"] = np.array([1.0], dtype=np.float64)
    ds.attrs["Description"] = np.bytes_(b"Friends-Of-Friends ID of the group the particles belong to")
    ds.attrs["Expression for physical CGS units"] = np.bytes_(b"[ - ] ")
    ds.attrs["Lossy compression filter"] = np.bytes_(b"None")
    for k in ("U_I exponent", "U_L exponent", "U_M exponent", "U_T exponent", "U_t exponent", "a-scale exponent", "h-scale exponent"):
        ds.attrs[k] = np.array([0.0], dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="Original SWIFT input HDF5 (provides Units, Masses, ParticleIDs)")
    ap.add_argument("endpoint", type=Path, help="GPU binary snapshot (.bin)")
    ap.add_argument("output", type=Path, help="Output SWIFT HDF5 path")
    ap.add_argument("--fof-bin", type=Path, default=None, help="Optional GPU FoF binary (.fof.bin)")
    ap.add_argument("--fof-lib", type=Path, default=None, help="Optional grav_cuda_fof.so to compute FOFGroups catalog")
    ap.add_argument("--so-min-size", type=int, default=20, help="Min FoF group size for SO (M200c/R200c/M200m/R200m); <0 disables")
    ap.add_argument("--h0", type=float, default=67.36)
    ap.add_argument("--omega-m", type=float, default=0.3153)
    ap.add_argument("--omega-de", type=float, default=0.6847)
    ap.add_argument("--omega-r", type=float, default=0.0)
    ap.add_argument("--omega-k", type=float, default=0.0)
    ap.add_argument("--w0", type=float, default=-1.0)
    ap.add_argument("--wa", type=float, default=0.0)
    ap.add_argument("--grav-g", type=float, default=43.0092)
    args = ap.parse_args()

    if args.fof_lib is not None and args.fof_bin is None:
        raise ValueError("--fof-lib requires --fof-bin")

    source, endpoint, output = args.source, args.endpoint, args.output
    with endpoint.open("rb") as f:
        n, box, a = struct.unpack("<Q2d", f.read(24))
    assert n > 0, f"Unexpected particle count {n}"
    assert 0.0 < a <= 1.0001, f"Unexpected scale factor {a}"
    expected_size = 24 + 40 * n
    actual_size = endpoint.stat().st_size
    assert actual_size == expected_size, f"Endpoint size {actual_size} != {expected_size}"

    fof_header = None
    if args.fof_bin is not None:
        with args.fof_bin.open("rb") as ff:
            n_f, ngrp_f, largest_f, min_sz_f, box_f, a_f, link_f = struct.unpack("<4Q3d", ff.read(56))
        assert n_f == n, f"FoF particle count {n_f} != {n}"
        assert abs(a_f - a) < 1e-8, f"FoF scale factor {a_f} != {a}"
        assert abs(box_f - box) < 1e-6, f"FoF box {box_f} != {box}"
        expected_fof_size = 56 + 8 * n
        actual_fof_size = args.fof_bin.stat().st_size
        assert actual_fof_size == expected_fof_size, f"FoF file size {actual_fof_size} != {expected_fof_size}"
        fof_header = (ngrp_f, largest_f, min_sz_f, link_f)

    bin_fields = {
        "Coordinates": ("<f8", 24, (n, 3)),
        "Velocities": ("<f4", 24 + 24 * n, (n, 3)),
        "Potentials": ("<f4", 24 + 36 * n, (n,)),
    }
    chunk = 1 << 20
    temporary = output.with_suffix(".hdf5.tmp")
    with h5py.File(source, "r") as original, h5py.File(temporary, "w") as out:
        orig_parts = original["PartType1"]
        assert orig_parts["ParticleIDs"].shape[0] == n, (
            f"Source ParticleIDs count {orig_parts['ParticleIDs'].shape[0]} != {n}"
        )
        for group in ("Units", "ICs_parameters"):
            if group in original:
                original.copy(group, out)
        header = out.create_group("Header")
        header.attrs.update(original["Header"].attrs)
        header.attrs["Scale-factor"] = a
        header.attrs["Redshift"] = 1.0 / a - 1.0
        header.attrs["OutputType"] = np.bytes_("GPU TreePM")
        hubble = 67.36
        header.attrs["Time"] = (
            2.0 * np.arcsinh(np.sqrt(0.6847 / 0.3153) * a**1.5)
            / (3.0 * hubble * np.sqrt(0.6847))
        )
        if fof_header is not None:
            header.attrs["GPU_FOF_NumGroups"] = np.uint64(fof_header[0])
            header.attrs["GPU_FOF_LargestGroup"] = np.uint64(fof_header[1])
            header.attrs["GPU_FOF_MinGroupSize"] = np.uint64(fof_header[2])
            header.attrs["GPU_FOF_LinkingLength"] = np.float64(fof_header[3])

        soft = float(np.median(orig_parts["Softenings"][:8192]))
        assert np.isfinite(soft) and soft > 0.0

        parts = out.create_group("PartType1")
        for name, (dtype, offset, shape) in bin_fields.items():
            data = np.memmap(endpoint, dtype=dtype, mode="r", offset=offset, shape=shape)
            ds = parts.create_dataset(
                name, shape=shape, dtype=dtype,
                chunks=(8192,) + shape[1:], shuffle=True,
                compression="gzip", compression_opts=1,
            )
            if name in orig_parts:
                ds.attrs.update(orig_parts[name].attrs)
            for start in range(0, n, chunk):
                block = np.asarray(data[start:start + chunk])
                if not np.isfinite(block).all():
                    raise ValueError(f"Non-finite {name} at {start}")
                if name == "Coordinates" and ((block < 0.0).any() or (block >= box).any()):
                    raise ValueError(f"Coordinate outside periodic box at {start}")
                ds[start:start + len(block)] = block

        first_mass = None
        masses_arr = None
        for name, dtype in (("Masses", "<f4"), ("ParticleIDs", "<u8")):
            src_ds = orig_parts[name]
            ds = parts.create_dataset(
                name, shape=(n,), dtype=dtype,
                chunks=(8192,), shuffle=True,
                compression="gzip", compression_opts=1,
            )
            ds.attrs.update(src_ds.attrs)
            for start in range(0, n, chunk):
                block = np.asarray(src_ds[start:start + chunk], dtype=dtype)
                if not np.isfinite(block).all():
                    raise ValueError(f"Non-finite {name} at {start}")
                ds[start:start + len(block)] = block
                if name == "Masses" and args.fof_lib is not None:
                    if first_mass is None:
                        first_mass = float(block[0])
                    if masses_arr is not None:
                        masses_arr[start:start + len(block)] = block
                    elif not np.all(block == first_mass):
                        masses_arr = np.empty(n, dtype="<f4")
                        if start > 0:
                            masses_arr[:start] = first_mass
                        masses_arr[start:start + len(block)] = block

        soft_ds = parts.create_dataset(
            "Softenings", shape=(n,), dtype="<f4", fillvalue=soft,
            chunks=(8192,), compression="gzip", compression_opts=1,
        )
        soft_ds.attrs.update(orig_parts["Softenings"].attrs)

        if args.fof_bin is not None:
            fof_data = np.memmap(args.fof_bin, dtype="<i8", mode="r", offset=56, shape=(n,))
            fof_ds = parts.create_dataset(
                "FOFGroupIDs", shape=(n,), dtype="<i8",
                chunks=(8192,), shuffle=True,
                compression="gzip", compression_opts=1,
            )
            _set_fof_attrs(fof_ds, orig_parts)
            ngrp = int(fof_header[0])
            for start in range(0, n, chunk):
                block = np.asarray(fof_data[start:start + chunk], dtype="<i8")
                if not np.all((block == GRAV_FOF_NONE) | ((block >= 1) & (block <= ngrp))):
                    raise ValueError(f"Invalid FOFGroupIDs entry at {start}")
                fof_ds[start:start + len(block)] = block
            if args.fof_lib is not None:
                _write_fof_groups(out, orig_parts, n, box, a, args.fof_bin, args.fof_lib, endpoint, masses_arr, first_mass, ngrp, args)

    temporary.replace(output)
    print(f"WROTE_HDF5 {output} bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
