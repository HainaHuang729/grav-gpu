"""Convert any 1024^3 SWIFT HDF5 snapshot into dense internal binary input for GPU TreePM."""
import json
import struct
import sys
from pathlib import Path

import h5py
import numpy as np


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        raise SystemExit(f"Usage: {sys.argv[0]} <source.hdf5> <target.bin> [stride=1]")
    source, target = map(Path, sys.argv[1:3])
    stride = int(sys.argv[3]) if len(sys.argv) == 4 else 1
    block = 1 << 20
    with h5py.File(source, "r") as file, target.open("wb") as output:
        particles = file["PartType1"]
        header = file["Header"].attrs
        n = len(particles["ParticleIDs"])
        assert n == 1024**3 and stride > 0 and block % stride == 0
        box = np.asarray(header["BoxSize"], dtype=np.float64)
        a = float(np.asarray(header["Scale-factor"]).item())
        assert box.shape == (3,) and np.all(box == box[0]) and 0.0 < a <= 1.0001
        softening = float(np.median(particles["Softenings"][:8192]))
        assert np.isfinite(softening) and softening > 0.0
        output.write(struct.pack("<Q3d", n // stride, float(box[0]), a, softening))
        for field, dtype in (
            ("Coordinates", "<f8"),
            ("Velocities", "<f4"),
            ("Masses", "<f4"),
            ("ParticleIDs", "<u8"),
        ):
            for start in range(0, n, block):
                sample = np.asarray(particles[field][start:start + block:stride], dtype=dtype)
                if field == "Coordinates":
                    np.remainder(sample, float(box[0]), out=sample)
                if field == "Masses" and stride != 1:
                    sample *= stride
                if field != "ParticleIDs" and not np.isfinite(sample).all():
                    raise ValueError(f"Non-finite {field} at {start}")
                sample.tofile(output)
    print(json.dumps({
        "source": source.name,
        "particles": n // stride,
        "stride": stride,
        "box_mpc": float(box[0]),
        "scale": a,
        "softening": softening,
    }))


if __name__ == "__main__":
    main()
