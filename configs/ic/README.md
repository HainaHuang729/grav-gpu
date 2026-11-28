# External IC Integration (`monofonIC` & `MUSIC`)

`gravityCU` keeps its CUDA solver core zero-dependency and connects cleanly to standard cosmological IC generators through `scripts/prepare_input.py`.

## 1. `monofonIC` (2LPT / 3LPT & Massive Neutrinos)

```bash
# 1. Generate SWIFT-format HDF5 IC (or Gadget-HDF5 shards)
mpirun -np 4 monofonIC configs/ic/monofonic_2lpt_swift.conf

# 2. Convert to gravityCU dense binary + auto-export cosmology environment
python3 scripts/prepare_input.py ic_monofonic_256.hdf5 ic_256.bin \
  --softening 0.0029 --cosmo-env ic_256.cosmo.env

# 3. Evolve on GPU (optionally with P1 high-order PM)
source ic_256.cosmo.env
GRAV_PM_ORDER=3 GRAV_PM_INTERLACE=1 GRAV_PM_GRAD=ik \
  ./build/multi_gpu_slab ic_256.bin out_256.bin 1.0
```

## 2. `MUSIC` (Multi-Mass Zoom-in)

```bash
# 1. Generate multi-level Zoom-in Gadget HDF5 IC
MUSIC configs/ic/music_zoomin_gadget.conf

# 2. Merge PartType1 (high-res) + PartType2 (coarse boundary) with per-particle softening
python3 scripts/prepare_input.py ic_music_zoom.hdf5 ic_zoom.bin \
  --format music --part-types PartType1,PartType2 \
  --type-softenings PartType1:0.0005,PartType2:0.008 \
  --soft-out ic_zoom.soft.bin --cosmo-env ic_zoom.cosmo.env

# 3. Run gravityCU with per-particle softening enabled
source ic_zoom.cosmo.env
GRAV_SOFT_IN=ic_zoom.soft.bin ./build/multi_gpu_slab ic_zoom.bin out_zoom.bin 1.0
```

## Automatic Convention Conversions in `prepare_input.py`
- **Units**: `kpc/h`, `Mpc/h` $\to$ `Mpc` (no $h$); `10^10 Msun/h` $\to$ `10^10 Msun` (no $h$).
- **Velocities**: Gadget internal $\mathbf{u} = \mathbf{v}_{\rm pec}/\sqrt{a}$ is automatically scaled by $\sqrt{a}$ to peculiar $\mathbf{v}_{\rm pec}$ (`km/s`).
- **Masses**: Falls back cleanly to `Header.attrs["MassTable"]` when `Masses` is omitted.
- **Shards**: Automatically streams multi-file `.0.hdf5 .. .{K-1}.hdf5` snapshots.
