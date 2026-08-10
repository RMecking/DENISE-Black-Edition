# Current DENISE build and execution workflow

This document records the pre-modernization workflow. It does not replace the
existing Makefiles or change any numerical implementation.

## Toolchain and build

The supported workflow is a Unix-like environment with GNU Make, a C compiler
wrapped by MPI, an MPI implementation, FFTW 3 development files, and the C++
runtime requested by the existing link flags. The M0 baseline was built on
Ubuntu 22.04 under WSL 2 with GCC 11.4.0 and Open MPI 4.1.2.

On Debian/Ubuntu the required system packages are:

```bash
sudo apt install build-essential openmpi-bin libopenmpi-dev libfftw3-dev
```

Build from the repository root:

```bash
make -C libcseife
make -C src denise
```

The first command produces `libcseife/libcseife.a`; the second produces
`bin/denise`. `make -C src snapmerge` additionally produces `bin/snapmerge`.

The active `src/Makefile` configuration is:

```text
CC=mpicc
CFLAGS=-O3 -w -fno-stack-protector -D_FORTIFY_SOURCE=0 -fcommon
IFLAGS=-I./../libcseife -I./../include
SFLAGS=-L./../libcseife
LFLAGS=-lm -lcseife -lfftw3 -lstdc++
```

The Makefile selects `model_ainos_visc.c`, `model_ainos.c`,
`model_acoustic.c`, `zinc_vti.c`, and `zinc_tti.c` as compiled-in model
generators. No solver kernel uses these generators when `READMOD=1`.

## Running DENISE

DENISE is launched from the directory relative to which paths in the main
parameter file should resolve:

```bash
mpirun -np <ranks> /path/to/bin/denise DENISE.inp FWI_workflow.inp
```

`denise.c` reads both `argv[1]` and `argv[2]` without checking `argc`, so two
arguments must be supplied even for `MODE=0`, where the workflow file is not
used. The MPI rank count must agree with `NPROCX * NPROCY`. The shipped
Marmousi example is normally run from `par/`:

```bash
cd par
mpirun -np 15 ../bin/denise DENISE_marm_OBC.inp FWI_workflow_marmousi.inp
```

`MODE` selects the operation: 0 forward modelling, 1 FWI, 2 RTM, and 3 the
FD-based gradient path. `PHYSICS` selects 1 isotropic P/SV, 2 acoustic, 3 VTI,
4 TTI, or 5 isotropic SH. Thus the M0 case uses `MODE=0` and `PHYSICS=5`.

## Inputs and outputs

The main input file is parsed positionally by `read_par.c`: the order of its
115 non-comment records is part of the current input contract. Comment lines
between parameter groups should be preserved. Important paths are:

- `MFILE`: model basename. With `READMOD=1`, elastic SH reads native 32-bit
  floats from `<MFILE>.vs` and `<MFILE>.rho`; elastic P/SV reads its velocity
  and density components in the corresponding physics reader. Grid samples
  are stored with x as the outer loop and y as the inner loop.
- `SOURCE_FILE`: text source geometry. The first line is the source count;
  subsequent records contain x, z, y, time shift, frequency, amplitude,
  azimuth, and source type. Source type 1 is the SH out-of-plane point force.
- `REC_FILE`: receiver basename when `READREC=1`; DENISE appends `.dat`. Each
  text line contains x and y in metres, adjusted by `REFREC` and rounded to the
  FD grid.
- `SEIS_FILE_*`: output basenames. `SEIS_FORMAT=1` writes SU traces (240-byte
  trace header followed by float samples), 2 writes receiver-major ASCII, and
  3 writes receiver-major native float binary. Multiple-shot forward outputs
  receive a `.shotN` suffix.
- `SNAP_FILE` and `SNAP_FORMAT`: snapshot basename and ASCII/binary format.
  Model and snapshot files are native float binaries when format 3 is chosen.

The full FWI configuration is split between the main DENISE input and the
workflow file (for example `par/FWI_workflow_marmousi.inp`). Logs are written
to `LOG_FILE.<rank>` unless rank zero is configured to print to stdout.
