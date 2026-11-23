"""Convert GPU 1024^3 endpoint.bin into a SWIFT HDF5 snapshot for standalone FoF,
reading Masses and ParticleIDs directly from the original input HDF5 snapshot."""
import struct
import sys
from pathlib import Path

import h5py
import numpy as np


def main():
    if len(sys.argv) != 4:
        raise SystemExit(f"Usage: {sys.argv[0]} <input_0055.hdf5> <endpoint.bin> <output_0056.hdf5>")
    source, endpoint, output = map(Path, sys.argv[1:4])
    with endpoint.open("rb") as f:
        n, box, a = struct.unpack("<Q2d", f.read(24))
    assert n == 1024**3, f"Unexpected particle count {n}"
    assert 0.0 < a <= 1.0001, f"Unexpected scale factor {a}"
    expected_size = 24 + 40 * n
    actual_size = endpoint.stat().st_size
    assert actual_size == expected_size, f"Endpoint size {actual_size} != {expected_size}"

    bin_fields = {
        "Coordinates": ("<f8", 24, (n, 3)),
        "Velocities": ("<f4", 24 + 24 * n, (n, 3)),
        "Potentials": ("<f4", 24 + 36 * n, (n,)),
    }
    chunk = 1 << 20
    temporary = output.with_suffix(".hdf5.tmp")
    with h5py.File(source, "r") as original, h5py.File(temporary, "w") as out:
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

        orig_parts = original["PartType1"]
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

        soft_ds = parts.create_dataset(
            "Softenings", shape=(n,), dtype="<f4", fillvalue=soft,
            chunks=(8192,), compression="gzip", compression_opts=1,
        )
        soft_ds.attrs.update(orig_parts["Softenings"].attrs)

    temporary.replace(output)
    print(f"WROTE_HDF5 {output} bytes={output.stat().st_size}")


if __name__ == "__main__":
    main()
