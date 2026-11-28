#!/usr/bin/env python3
"""Convert SWIFT, monofonIC, MUSIC, or Gadget HDF5 IC/snapshots into gravityCU binary input.

Maintains 100% backward compatibility with the original SWIFT HDF5 CLI while adding:
  - Auto-detection of SWIFT, monofonIC (SWIFT/Gadget), MUSIC (single/zoom-in), and Gadget HDF5
  - Automatic unit conversion (kpc/h, Mpc/h -> Mpc; 1e10 Msun/h -> 1e10 Msun)
  - Automatic velocity convention alignment (Gadget u = v_pec / sqrt(a) -> v_pec)
  - Header MassTable fallback when PartTypeK/Masses is omitted
  - Multi-file HDF5 shard streaming (prefix.0.hdf5 .. prefix.{K-1}.hdf5)
  - Multi-type Zoom-in merging (--part-types PartType1,PartType2) + --soft-out (.soft.bin)
  - Optional --cosmo-env export of GRAV_COSMO_* environment variables
"""
import argparse
import json
import math
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

CM_PER_MPC = 3.0856775814913673e24
G_PER_1E10_MSUN = 1.988409870698051e43


def discover_shards(source: Path) -> List[Path]:
    """Return ordered list of HDF5 shard paths for single-file or multi-file snapshots."""
    if source.is_file():
        with h5py.File(source, "r") as f:
            n_files = int(np.asarray(f["Header"].attrs.get("NumFilesPerSnapshot", 1)).flat[0])
        if n_files <= 1:
            return [source]
        name = source.name
        if ".0.hdf5" in name:
            prefix = name[: -len(".0.hdf5")]
            shards = [source.parent / f"{prefix}.{i}.hdf5" for i in range(n_files)]
            if all(p.is_file() for p in shards):
                return shards
        return [source]

    # Prefix form: e.g. /path/to/ic -> /path/to/ic.0.hdf5 ...
    first = Path(f"{source}.0.hdf5")
    if first.is_file():
        return discover_shards(first)
    direct = Path(f"{source}.hdf5")
    if direct.is_file():
        return discover_shards(direct)
    raise FileNotFoundError(f"Input HDF5 file or shard prefix not found: {source}")


def detect_format(f: h5py.File, requested: str) -> str:
    if requested != "auto":
        return requested
    header = f["Header"].attrs
    if "Scale-factor" in header or "Units" in f or "Cosmology" in f:
        return "swift"
    if "ICs_parameters" in f or "MUSIC" in str(header.get("Code", "")):
        return "music"
    return "gadget-hdf5"


def parse_type_softenings(spec: Optional[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not spec:
        return out
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --type-softenings entry '{item}'; expected PartTypeK:val")
        k, v = item.split(":", 1)
        val = float(v)
        if not math.isfinite(val) or val <= 0.0:
            raise ValueError(f"Invalid softening for {k}: {v}")
        out[k.strip()] = val
    return out


def extract_cosmology(f: h5py.File) -> Dict[str, Optional[float]]:
    header = f["Header"].attrs
    cosmo_attrs = f["Cosmology"].attrs if "Cosmology" in f else {}

    def _get(keys: Tuple[str, ...]) -> Optional[float]:
        for src in (cosmo_attrs, header):
            for k in keys:
                if k in src:
                    val = float(np.asarray(src[k]).flat[0])
                    if math.isfinite(val):
                        return val
        return None

    h = _get(("h", "HubbleParam"))
    if h is None and "H0 [internal units]" in cosmo_attrs:
        h = float(np.asarray(cosmo_attrs["H0 [internal units]"]).flat[0]) / 100.0
    om = _get(("Omega_m", "Omega0"))
    ode = _get(("Omega_lambda", "OmegaLambda"))
    ob = _get(("Omega_b", "OmegaBaryon"))
    ocdm = _get(("Omega_cdm",))
    if ocdm is None and om is not None:
        ocdm = om - (ob if ob is not None else 0.0)
    w0 = _get(("w_0", "w0"))
    wa = _get(("w_a", "wa"))
    return {
        "h": h,
        "H0": (100.0 * h) if h is not None else None,
        "omega_m": om,
        "omega_de": ode,
        "omega_b": ob,
        "omega_cdm": ocdm,
        "w0": w0,
        "wa": wa,
    }


def resolve_unit_scales(
    f: h5py.File,
    fmt: str,
    h: Optional[float],
    length_unit: str,
    mass_unit: str,
    vel_convention: str,
    scale: float,
) -> Tuple[float, float, float, str]:
    """Return (length_to_mpc, mass_to_1e10msun, vel_to_kms, vel_mode_used)."""
    header = f["Header"].attrs
    units_attrs = f["Units"].attrs if "Units" in f else {}
    h_eff = h if (h is not None and h > 0.0) else 1.0

    # Length scale -> Mpc (no h)
    if length_unit == "mpc":
        l_scale = 1.0
    elif length_unit == "mpc_h":
        l_scale = 1.0 / h_eff
    elif length_unit == "kpc_h":
        l_scale = 1.0e-3 / h_eff
    else:  # auto
        if fmt == "swift":
            u_l_cm = float(np.asarray(units_attrs.get("Unit length in cgs (U_L)", CM_PER_MPC)).flat[0])
            l_scale = u_l_cm / CM_PER_MPC
        else:
            # Gadget / MUSIC / monofonIC gadget_hdf5: usually kpc/h (3.085678e21) or Mpc/h (3.085678e24)
            if "UnitLength_in_cm" in header:
                u_l_cm = float(np.asarray(header["UnitLength_in_cm"]).flat[0])
                l_scale = (u_l_cm / CM_PER_MPC) / h_eff
            else:
                # Heuristic: BoxSize > 5000 in typical cosmo boxes usually indicates kpc/h
                box_raw = float(np.asarray(header["BoxSize"]).flat[0])
                l_scale = (1.0e-3 / h_eff) if box_raw > 5000.0 else (1.0 / h_eff)

    # Mass scale -> 1e10 Msun (no h)
    if mass_unit == "1e10msun":
        m_scale = 1.0
    elif mass_unit == "1e10msun_h":
        m_scale = 1.0 / h_eff
    else:  # auto
        if fmt == "swift":
            u_m_g = float(np.asarray(units_attrs.get("Unit mass in cgs (U_M)", G_PER_1E10_MSUN)).flat[0])
            m_scale = u_m_g / G_PER_1E10_MSUN
        else:
            if "UnitMass_in_g" in header:
                u_m_g = float(np.asarray(header["UnitMass_in_g"]).flat[0])
                m_scale = (u_m_g / G_PER_1E10_MSUN) / h_eff
            else:
                m_scale = 1.0 / h_eff

    # Velocity scale -> SWIFT / gravityCU peculiar velocity v_pec (km/s)
    if vel_convention == "peculiar":
        v_scale = 1.0
        vel_mode = "peculiar"
    elif vel_convention == "gadget":
        v_scale = math.sqrt(scale)
        vel_mode = "gadget_sqrt_a"
    else:  # auto
        if fmt == "swift":
            v_scale = 1.0
            vel_mode = "peculiar"
        else:
            v_scale = math.sqrt(scale)
            vel_mode = "gadget_sqrt_a"

    return l_scale, m_scale, v_scale, vel_mode


def part_type_index(pt_name: str) -> int:
    if pt_name.startswith("PartType") and pt_name[8:].isdigit():
        return int(pt_name[8:])
    return -1


def write_cosmo_env(path: Path, cosmo: Dict[str, Optional[float]]) -> None:
    lines = ["# Auto-generated by gravityCU prepare_input.py"]
    mapping = [
        ("H0", "GRAV_COSMO_H0"),
        ("omega_m", "GRAV_COSMO_OMEGA_M"),
        ("omega_de", "GRAV_COSMO_OMEGA_DE"),
        ("omega_cdm", "GRAV_COSMO_OMEGA_CDM"),
        ("w0", "GRAV_COSMO_W0"),
        ("wa", "GRAV_COSMO_WA"),
    ]
    for key, env_name in mapping:
        val = cosmo.get(key)
        if val is not None:
            lines.append(f"export {env_name}={val:.10g}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert SWIFT, monofonIC, MUSIC, or Gadget HDF5 IC/snapshot into gravityCU dense .bin input."
    )
    ap.add_argument("source", type=Path, help="Input HDF5 file (or shard prefix / .0.hdf5)")
    ap.add_argument("target", type=Path, help="Output dense .bin file path")
    ap.add_argument(
        "--format",
        choices=("auto", "swift", "monofonic", "music", "gadget-hdf5"),
        default="auto",
        help="Input IC/snapshot convention (default: auto-detect)",
    )
    ap.add_argument("--part-type", default=None, help="Single HDF5 particle group (alias for --part-types)")
    ap.add_argument(
        "--part-types",
        default=None,
        help="Comma-separated HDF5 particle groups to merge (e.g. PartType1,PartType2 for MUSIC Zoom-in)",
    )
    ap.add_argument("--stride", type=int, default=1, help="Subsampling stride (default: 1 = all particles)")
    ap.add_argument(
        "--softening",
        type=float,
        default=None,
        help="Override global Plummer-equivalent softening in Mpc (auto-read from Softenings if omitted)",
    )
    ap.add_argument(
        "--type-softenings",
        default=None,
        help="Per-type Plummer softenings in Mpc for Zoom-in, e.g. 'PartType1:0.002,PartType2:0.01'",
    )
    ap.add_argument(
        "--soft-out",
        type=Path,
        default=None,
        help="Optional output path for per-particle float32 softening file (used with GRAV_SOFT_IN)",
    )
    ap.add_argument(
        "--cosmo-env",
        type=Path,
        default=None,
        help="Optional output path for shell-sourceable GRAV_COSMO_* environment variables",
    )
    ap.add_argument(
        "--length-unit",
        choices=("auto", "mpc", "mpc_h", "kpc_h"),
        default="auto",
        help="Override input length unit before converting to Mpc (no h)",
    )
    ap.add_argument(
        "--mass-unit",
        choices=("auto", "1e10msun", "1e10msun_h"),
        default="auto",
        help="Override input mass unit before converting to 1e10 Msun (no h)",
    )
    ap.add_argument(
        "--vel-convention",
        choices=("auto", "peculiar", "gadget"),
        default="auto",
        help="Override velocity convention: 'peculiar' (v_pec km/s) or 'gadget' (u = v_pec / sqrt(a))",
    )
    ap.add_argument("--block", type=int, default=1 << 20, help="HDF5 chunk size (default: 1048576)")
    args = ap.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")
    if args.block < 1 or args.block % args.stride != 0:
        raise ValueError("--block must be divisible by --stride")

    if args.part_types:
        part_types = [p.strip() for p in args.part_types.split(",") if p.strip()]
    elif args.part_type:
        part_types = [args.part_type.strip()]
    else:
        part_types = ["PartType1"]
    if not part_types:
        raise ValueError("No particle types specified")

    type_soft_map = parse_type_softenings(args.type_softenings)
    shards = discover_shards(args.source)

    with h5py.File(shards[0], "r") as f0:
        fmt = detect_format(f0, args.format)
        # monofonIC can output either SWIFT or Gadget HDF5; detect actual convention if requested
        if fmt == "monofonic":
            fmt = "swift" if ("Scale-factor" in f0["Header"].attrs or "Units" in f0) else "gadget-hdf5"
        header0 = f0["Header"].attrs
        cosmo = extract_cosmology(f0)

        box_arr = np.broadcast_to(np.asarray(header0["BoxSize"], dtype=np.float64), (3,))
        if float(np.max(box_arr)) != float(np.min(box_arr)) or float(box_arr[0]) <= 0.0:
            raise ValueError(f"Expected positive cubic BoxSize, got {box_arr.tolist()}")

        if "Scale-factor" in header0:
            scale = float(np.asarray(header0["Scale-factor"]).flat[0])
        elif "Time" in header0:
            scale = float(np.asarray(header0["Time"]).flat[0])
        elif "Redshift" in header0:
            z0 = float(np.asarray(header0["Redshift"]).flat[0])
            scale = 1.0 / (1.0 + z0)
        else:
            raise KeyError("Header missing Scale-factor, Time, and Redshift")
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid scale factor: {scale}")

        l_scale, m_scale, v_scale, vel_mode = resolve_unit_scales(
            f0, fmt, cosmo["h"], args.length_unit, args.mass_unit, args.vel_convention, scale
        )
        box = float(box_arr[0]) * l_scale

    # Pass 1: count particles across all shards and particle types, determine global softening
    plan: List[Tuple[Path, str, int, Optional[float]]] = []  # (shard, pt, n_pt, mass_table_val)
    total_out = 0
    inferred_soft: Optional[float] = args.softening

    for shard in shards:
        with h5py.File(shard, "r") as f:
            hdr = f["Header"].attrs
            mt = np.asarray(hdr.get("MassTable", np.zeros(6, dtype=np.float64)), dtype=np.float64)
            for pt in part_types:
                if pt not in f:
                    continue
                grp = f[pt]
                if "Coordinates" not in grp:
                    continue
                n_pt = len(grp["Coordinates"])
                if n_pt == 0:
                    continue
                pt_idx = part_type_index(pt)
                mt_val = float(mt[pt_idx]) if (0 <= pt_idx < len(mt) and mt[pt_idx] > 0.0) else None
                if "Masses" not in grp and mt_val is None:
                    raise KeyError(f"{shard}:{pt} has neither Masses dataset nor positive Header MassTable[{pt_idx}]")
                if inferred_soft is None:
                    if pt in type_soft_map:
                        inferred_soft = type_soft_map[pt]
                    elif "Softenings" in grp:
                        sample = np.asarray(grp["Softenings"][: min(8192, n_pt)], dtype=np.float64)
                        if not np.isfinite(sample).all() or float(sample.min()) <= 0.0:
                            raise ValueError(f"Invalid Softenings in {shard}:{pt}; pass --softening explicitly")
                        inferred_soft = float(np.median(sample)) * l_scale
                n_sel = (n_pt + args.stride - 1) // args.stride
                total_out += n_sel
                plan.append((shard, pt, n_pt, mt_val))

    if total_out == 0:
        raise ValueError(f"No particles found for {part_types} across {len(shards)} shard(s)")
    if inferred_soft is None:
        raise ValueError("No Softenings dataset found; pass --softening or --type-softenings explicitly")
    softening = float(inferred_soft)
    if not math.isfinite(softening) or softening <= 0.0:
        raise ValueError(f"Invalid softening: {softening}")

    # Pass 2: stream dense binary fields in exact multi_gpu_slab layout
    args.target.parent.mkdir(parents=True, exist_ok=True)
    with args.target.open("wb") as out:
        out.write(struct.pack("<Q3d", total_out, box, scale, softening))

        # 1) Coordinates (<f8, Mpc no-h, wrapped to [0, box))
        for shard, pt, n_pt, _ in plan:
            with h5py.File(shard, "r") as f:
                ds = f[pt]["Coordinates"]
                for start in range(0, n_pt, args.block):
                    end = min(start + args.block, n_pt)
                    chunk = np.asarray(ds[start:end:args.stride], dtype="<f8").copy()
                    if l_scale != 1.0:
                        chunk *= l_scale
                    np.remainder(chunk, box, out=chunk)
                    if not np.isfinite(chunk).all():
                        raise ValueError(f"Non-finite Coordinates in {shard}:{pt} at {start}")
                    chunk.tofile(out)

        # 2) Velocities (<f4, peculiar km/s)
        for shard, pt, n_pt, _ in plan:
            with h5py.File(shard, "r") as f:
                ds = f[pt]["Velocities"]
                for start in range(0, n_pt, args.block):
                    end = min(start + args.block, n_pt)
                    chunk = np.asarray(ds[start:end:args.stride], dtype="<f8")
                    if v_scale != 1.0:
                        chunk = chunk * v_scale
                    out_v = np.asarray(chunk, dtype="<f4")
                    if not np.isfinite(out_v).all():
                        raise ValueError(f"Non-finite Velocities in {shard}:{pt} at {start}")
                    out_v.tofile(out)

        # 3) Masses (<f4, 1e10 Msun no-h)
        for shard, pt, n_pt, mt_val in plan:
            with h5py.File(shard, "r") as f:
                grp = f[pt]
                has_masses = "Masses" in grp
                for start in range(0, n_pt, args.block):
                    end = min(start + args.block, n_pt)
                    cnt = len(range(start, end, args.stride))
                    if has_masses:
                        chunk = np.asarray(grp["Masses"][start:end:args.stride], dtype="<f8")
                    else:
                        chunk = np.full(cnt, mt_val, dtype="<f8")
                    if m_scale != 1.0:
                        chunk = chunk * m_scale
                    if args.stride != 1:
                        chunk = chunk * float(args.stride)
                    out_m = np.asarray(chunk, dtype="<f4")
                    if not np.isfinite(out_m).all():
                        raise ValueError(f"Non-finite Masses in {shard}:{pt} at {start}")
                    out_m.tofile(out)

        # 4) ParticleIDs (<u8)
        running_id = 0
        for shard, pt, n_pt, _ in plan:
            with h5py.File(shard, "r") as f:
                grp = f[pt]
                has_ids = "ParticleIDs" in grp
                for start in range(0, n_pt, args.block):
                    end = min(start + args.block, n_pt)
                    cnt = len(range(start, end, args.stride))
                    if has_ids:
                        ids = np.asarray(grp["ParticleIDs"][start:end:args.stride], dtype="<u8")
                    else:
                        ids = np.arange(running_id, running_id + cnt, dtype="<u8")
                    running_id += cnt
                    ids.tofile(out)

    # Optional per-particle softening file for Zoom-in (GRAV_SOFT_IN)
    if args.soft_out is not None:
        args.soft_out.parent.mkdir(parents=True, exist_ok=True)
        with args.soft_out.open("wb") as sout:
            for shard, pt, n_pt, _ in plan:
                with h5py.File(shard, "r") as f:
                    grp = f[pt]
                    override = type_soft_map.get(pt)
                    has_soft_ds = ("Softenings" in grp) and (override is None)
                    for start in range(0, n_pt, args.block):
                        end = min(start + args.block, n_pt)
                        cnt = len(range(start, end, args.stride))
                        if has_soft_ds:
                            s_chunk = np.asarray(grp["Softenings"][start:end:args.stride], dtype="<f8") * l_scale
                        else:
                            val = override if override is not None else softening
                            s_chunk = np.full(cnt, val, dtype="<f8")
                        np.asarray(s_chunk, dtype="<f4").tofile(sout)

    if args.cosmo_env is not None:
        write_cosmo_env(args.cosmo_env, cosmo)

    print(
        json.dumps(
            {
                "source": str(args.source),
                "shards": len(shards),
                "format_detected": fmt,
                "part_types": part_types,
                "target": str(args.target),
                "soft_out": str(args.soft_out) if args.soft_out else None,
                "cosmo_env": str(args.cosmo_env) if args.cosmo_env else None,
                "particles": total_out,
                "stride": args.stride,
                "box_mpc": box,
                "scale": scale,
                "softening": softening,
                "unit_scales": {
                    "length_to_mpc": l_scale,
                    "mass_to_1e10msun": m_scale,
                    "vel_to_kms": v_scale,
                    "vel_mode": vel_mode,
                },
                "cosmology": cosmo,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
