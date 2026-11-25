#!/usr/bin/env python3
"""Convert a SWIFT HDF5 snapshot or IC file into the dense binary format read by multi_gpu_slab."""
import argparse
import json
import struct
import sys
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert a SWIFT HDF5 snapshot/IC into dense binary input for multi_gpu_slab."
    )
    ap.add_argument("source", type=Path, help="Input SWIFT HDF5 snapshot/IC file")
    ap.add_argument("target", type=Path, help="Output dense .bin file path")
    ap.add_argument("--part-type", default="PartType1", help="HDF5 particle group (default: PartType1)")
    ap.add_argument("--stride", type=int, default=1, help="Subsampling stride (default: 1 = all particles)")
    ap.add_argument(
        "--softening",
        type=float,
        default=None,
        help="Override Plummer-equivalent softening in internal length units (auto-read from Softenings if omitted)",
    )
    ap.add_argument("--block", type=int, default=1 << 20, help="HDF5 chunk size (default: 1048576)")
    args = ap.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.block < 1 or args.block % args.stride != 0:
        raise ValueError("--block must be divisible by --stride")

    with h5py.File(args.source, "r") as f:
        header = f["Header"].attrs
        if args.part_type not in f:
            raise KeyError(f"Particle group '{args.part_type}' not found in {args.source}")
        dark = f[args.part_type]
        n_total = len(dark["Coordinates"])
        if n_total == 0:
            raise ValueError(f"{args.part_type} is empty")

        box_arr = np.broadcast_to(np.asarray(header["BoxSize"], dtype=np.float64), (3,))
        if float(np.max(box_arr)) != float(np.min(box_arr)) or float(box_arr[0]) <= 0.0:
            raise ValueError(f"Expected positive cubic BoxSize, got {box_arr.tolist()}")
        box = float(box_arr[0])

        scale_key = "Scale-factor" if "Scale-factor" in header else "Time"
        scale = float(np.asarray(header[scale_key]).flat[0])
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid {scale_key}: {scale}")

        if args.softening is not None:
            softening = float(args.softening)
        elif "Softenings" in dark:
            sample = np.asarray(dark["Softenings"][: min(8192, n_total)], dtype=np.float64)
            if not np.isfinite(sample).all() or float(sample.min()) <= 0.0:
                raise ValueError("Invalid Softenings dataset; pass --softening explicitly")
            softening = float(np.median(sample))
        else:
            raise ValueError("No Softenings dataset found in HDF5; pass --softening explicitly")

        if not np.isfinite(softening) or softening <= 0.0:
            raise ValueError(f"Invalid softening: {softening}")

        out_indices = (n_total + args.stride - 1) // args.stride
        args.target.parent.mkdir(parents=True, exist_ok=True)
        with args.target.open("wb") as out:
            out.write(struct.pack("<Q3d", out_indices, box, scale, softening))
            for field, dtype in (
                ("Coordinates", "<f8"),
                ("Velocities", "<f4"),
                ("Masses", "<f4"),
                ("ParticleIDs", "<u8"),
            ):
                if field == "ParticleIDs" and "ParticleIDs" not in dark:
                    # Fallback sequential IDs when an IC file omits ParticleIDs
                    for start in range(0, n_total, args.block):
                        cnt = len(range(start, min(start + args.block, n_total), args.stride))
                        ids = np.arange(start, min(start + args.block, n_total), args.stride, dtype="<u8")
                        assert len(ids) == cnt
                        ids.tofile(out)
                    continue
                for start in range(0, n_total, args.block):
                    end = min(start + args.block, n_total)
                    chunk = np.asarray(dark[field][start:end:args.stride], dtype=dtype).copy()
                    if field == "Coordinates":
                        np.remainder(chunk, box, out=chunk)
                    elif field == "Masses" and args.stride != 1:
                        chunk *= float(args.stride)
                    if field != "ParticleIDs" and not np.isfinite(chunk).all():
                        raise ValueError(f"Non-finite values in {field} at offset {start}")
                    chunk.tofile(out)

    print(
        json.dumps(
            {
                "source": str(args.source),
                "target": str(args.target),
                "particles": out_indices,
                "stride": args.stride,
                "box_mpc": box,
                "scale": scale,
                "softening": softening,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
