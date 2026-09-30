# M9c clean elastic P/SV migration driver

M9c is the supported single-rank CPU `MODE=2`, `PHYSICS=1` path for raw
elastic P/SV Born-adjoint migration. It computes, in ascending physical-shot
order,

```text
image_lambda_raw = sum_s J_s(m0)^T d_s^mig |_lambda
image_mu_raw     = sum_s J_s(m0)^T d_s^mig |_mu
```

The driver calls the frozen M9b-1 operator through
`denise_elastic_psv_migrate()`. It does not call the historical `RTM_PSV()`
path and does not construct an FWI residual. In particular, it performs no
modeled-minus-observed subtraction, time reversal at the input boundary,
integration/differentiation, time window, trace kill, offset mute, filtering,
normalization, weighting, sign change, taper, smoothing, preconditioning,
illumination compensation, source normalization, or `DT`/`DH` scaling.

## Supported envelope

The initial executable path accepts exactly:

- elastic isotropic P/SV, `L=0`, `INVMAT1=3` (`lambda`, `mu`, `rho`);
- fixed density in the migrated parameter space;
- `FDORDER=4`, `NDT=1`, `DTINV=1`;
- `FREE_SURF=0`, `BOUNDARY=0`;
- one MPI rank with `NPROCX=NPROCY=1`;
- `SEISMO=1`, direct `vx` and `vy` data;
- `SRCREC=1`, `RUN_MULTIPLE_SHOTS=1`, one explosive source per shot;
- `QUELLART=3`, `INV_STF=0`, with already prepared source samples;
- common fixed receiver geometry (`READREC=1`);
- optional CPML (`FW=0` disables it; `FW>0` uses the normal DENISE
  `DAMPING`, `FPML`, `NPOWER`, and `K_MAX_PML` fields and the established
  reflection coefficient `0.001`).

Every unsupported setting fails closed. There is no fallback to legacy RTM.

## Public API

`include/denise_elastic_psv_migration.h` declares:

- `denise_elastic_psv_migrate()` for an explicit, global-free request;
- `denise_elastic_psv_migration_pack_components()` for strict component
  packing;
- `denise_elastic_psv_migration_result_destroy()` for result ownership;
- `denise_elastic_psv_migration_mode2()` for the executable adapter.

Model and image arrays at the API boundary are contiguous row-major `[y][x]`.
Source and receiver indices are zero-based. Prepared migration data is
chronological and time-major:

```text
((time * receiver_count + receiver) * 2 + component)
component 0 = vx
component 1 = vy
```

The executable performs the existing DENISE physical-coordinate rounding and
the one-based to zero-based conversion in one adapter function.

## Executable file contract

The positional parameter file adds records 120 through 122:

```text
MIGRATION_SOURCE_PREFIX =prepared/source
MIGRATION_DATA_PREFIX =prepared/migration
MIGRATION_IMAGE_PREFIX =image/migration
```

The normal model prefix supplies exact raw float32 files:

```text
MFILE.lam
MFILE.mu
MFILE.rho
```

These retain DENISE's existing x-major file traversal. The adapter converts
them once into the M9b-1 row-major view and never interprets them as `Vp` or
`Vs`.

`SOURCE_FILE` retains the normal count plus eight-column source geometry. For
the prepared-source contract, each row must use neutral metadata
`tshift=0`, `fc=0`, `amplitude=1`, `azimuth=0`, and source type `1`. Shot `s`
loads exactly `NT` raw float32 values from:

```text
MIGRATION_SOURCE_PREFIX.shot_s.bin
```

`REC_FILE.dat` contains the common receiver `(x,y)` rows in their canonical
order. Duplicate grid locations are rejected rather than silently collapsed.
`READREC=2` is outside M9c.

For every shot, the two prepared migration components are exact raw float32
files with `NT * NREC` values in `[time][receiver]` order:

```text
MIGRATION_DATA_PREFIX.vx.shot_s.bin
MIGRATION_DATA_PREFIX.vy.shot_s.bin
```

Missing files, extra or missing samples, nonfinite values, and receiver-count
mismatches are errors. Files are never truncated or padded.

## Output and ownership

The driver holds two global float64 images and processes only one shot context
at a time. Per shot it creates one M9b-1 context, selects the M9d1 segmented
checkpoint/replay backend when its complete logical retained-byte estimate is
strictly smaller than full trajectory storage,
applies `J^T`, accumulates into the global images in deterministic double
precision, and destroys the context. Production requests `min(32, NT)`
segments. Equal or larger estimates retain full storage. Allocator-private
bookkeeping and common context/working state are excluded from this metric.
The dominant declared allocation classes are:

```text
historical full equivalent = 4 * NT * NX * NY * sizeof(float)
retained replay            = checkpoint payloads + checkpoint objects
                           + start/end schedules + largest-segment operands
global images       = 2 * NX * NY * sizeof(double)
prepared shot data  = NT * NREC * 2 * sizeof(float)
```

The public `trajectory_bytes` diagnostic continues to report the historical
full-storage equivalent. Separate fields report checkpoint payload/total payload,
checkpoint objects, pointer table (zero for contiguous objects), schedules,
segment operands, complete retained replay bytes, segment geometry, working-state bytes
and initial/replayed forward-step counts. The scientific migration contract and
file formats are unchanged. See `m9d1_elastic_psv_checkpoint_replay.md` for the
checkpoint timing and exact storage layout.

The canonical output files are native-endian IEEE-754 float64, contiguous
row-major `[y][x]`, with no transposition or scaling:

```text
MIGRATION_IMAGE_PREFIX.image_lambda_raw.bin
MIGRATION_IMAGE_PREFIX.image_mu_raw.bin
```

Both complete images remain in memory until all shots succeed. The adapter
writes temporary files and publishes the final pair only after both writes
close successfully. A failed run removes temporary and final names so that it
cannot leave a successful-looking partial pair.

## M9d2 distributed production adapter

The existing direct serial API and valid one-rank MODE=2 path remain canonical.
M9d2 adds equal Cartesian tiles, owned model/data/images, root-only production
I/O and collective paired publication for NCOLORS=1. See
[the M9d2 contract](m9d2_elastic_psv_mpi.md) for topology, ownership, halo
transposes and failure gates. No independent shot groups are supported.
