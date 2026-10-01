# M9e-1 isolated elastic CUDA forward verification ABI

Candidate base: `bf20f6f15402c111468d55990be64b40d62ea653`.
Branch: `codex/m9e1-elastic-psv-cuda-forward`. Uncommitted implementation;
independent scientific verification and user acceptance remain separate gates.

## Scope and isolation

This is an opt-in internal verification operator for one rank and one GPU,
elastic L=0 P/SV, FD4, raw lambda/mu (INVMAT1=3), fixed density, BOUNDARY=0,
FREE_SURF=0, optional CPML, prepared explosive increments, direct vx/vy data,
and FULL corrected-strain storage. FREE_SURF=1 is rejected before propagation.
There is no MODE=2 routing, Born J/JT, migration, replay, density imaging, GSLS,
attenuation, or FWI residual interpretation in the CUDA operator.

`include/denise_cuda_m9_elastic.h` is an internal C ABI. Existing CPU structs
and functions remain unchanged. The host bridge includes the exact unchanged
CPU translation unit under private symbol names to reuse its validation,
material maps, profiles, ownership, and destruction. It never calls CPU J/JT.
Its symbols are hidden in the optional shared library. No M8e numerical source,
test, tolerance, build flag, checkpoint format, or dispatch is changed.

## State and arithmetic

Each device context owns one contiguous arena. Padded arrays have width NX+4,
height NY+4, stride NX+4, and physical (i,j) at (i+2,j+2); width-two halos
therefore represent negative indices directly. Five fields are ordered
vx, vy, sxx, syy, sxy. Eight memories are ordered sxx_x, sxy_y, sxy_x, syy_y,
vxx, vyx, vxy, vyy. Five material maps are lambda, mu, inverse x-face density,
inverse y-face density, harmonic corner mu. No rho/GSLS placeholders exist.

Profiles pack [x,x-half,y,y-half], each as [kappa,a,b] along its coordinate.
Source is [time]. Geometry is [receiver][i,j,ordinal]. Data is
[time][receiver][vx,vy]. Operands are [time][VXX,VYX,VXY,VYY][y][x]. All
numerical state, profiles, source, data, and operands are FP32.

The production graph is stress COPY X, stress COPY Y, momentum/CPML, sample,
velocity COPY X, velocity COPY Y, corrected strains, elastic stress, prepared
source, completed timestep. X copies only physical rows; Y subsequently copies
all columns, including the populated X halos and hence corners. Separate
kernel launches enforce these dependencies without cross-block barriers.

FD4 forms the raw difference first, multiplies by the already-rounded FP32
DT/DH, and only then applies memory_next=b*memory+a*q and q/kappa+memory_next.
This is M9 arithmetic, not M8e unscaled-CPML arithmetic. Strains are the four
dimensionless corrected increments. Stress uses raw lambda/mu and canonical
harmonic corner mu. Source adds the prepared sample to sxx and syy after
stress, without differentiation, another DT, normalization, or endpoint rules.
Receivers sample after momentum and before strains; duplicates retain ordinal
order. NT=1 is supported.

Canonical host material/profile bytes are copied exactly. Context inputs are
deep copied by that preparation, so caller arrays need not survive creation.
Nonlinear execution uploads supplied lambda/mu and their canonical harmonic
map; original density, profiles, source, geometry, and damping speed remain
fixed. Re-prepare restores the original background model and zeros the state.

## Memory and failure ownership

Let N=NX*NY, P=(NX+4)*(NY+4), T=NT, R=receiver_count. Device bytes are:

| Category | Requested bytes |
|---|---:|
| Five wavefields | 20P |
| Eight CPML arrays | 32P |
| Five material maps | 20P |
| Profiles | 24(NX+NY) |
| Source | 4T |
| Geometry and data | 12R + 8TR |
| FULL four operands | 16TN |
| Additional device workspace/metadata | 0 |

The kernel view is passed by value, not separately allocated on device. Checked
products/sums and cap/reserve/available-memory checks precede cudaMalloc and
any propagation. There are no hidden candidate device allocations. CUDA
driver internal allocations and allocator granularity are outside this ledger.
Host metadata diagnostics include the CUDA owner, private host owner, and CPU
preparation context; host requested allocations are tracked separately. Each
tracked host allocation also has a small alignment/accounting prefix, excluded
from the requested-byte ledger.

Every reachable setup, allocation, memset, transfer, launch, completion, and
event operation is checked. Results become valid only after a completed run.
Failure resets prepared/valid_steps, preserves the first meaningful diagnostic,
and makes output unavailable. Downloads use tracked temporary host storage and
publish to caller buffers only after all requested copies succeed. Destruction
is never synthetically faulted; real runtime cleanup errors retain the handle
and owned-byte ledger for explicit retry rather than reporting false success.

The fault API is a deterministic, process-local single-caller verification
facility; it is not a concurrent allocator service. It fails one indexed call
before execution. Owned device bytes, requested host bytes, and event counts
must all be zero after destruction. Driver-wide leak claims are not made.

## Reproduce the local gates

The ordinary `make -C libcseife` and `make -B -C src denise` remain
CUDA-independent. Optional targets use separate M9-local flags and objects:

```sh
make -C src cuda_m9_elastic_psv cuda_m9_elastic_psv_nofma \
  NVCC=/usr/local/cuda-12.8/bin/nvcc CUDA_ARCHS=86
python3 -m pytest -q tests/physics/test_m9e1_cuda_elastic_psv_forward.py
```

The default optional library is the ordinary FMA build. The explicit no-FMA
library uses --fmad=false; both use -O2, without fast math. No numerical backend
is selected automatically. CPU preparation is compiled with strict C99,
-Wall -Wextra -Werror -pedantic. The canonical CPU verification view uses O3
and the unchanged production source. New hardware tests skip only genuinely
missing compiler/device prerequisites, never compile or numerical failures.

The test suite uses canonical CPU plus independent FP64 M9b FD4, CPML,
face-density, and harmonic-shear primitives. Prepared FP32 source increments
are the input to all three layers. Final-state reference construction adds no
CUDA-derived expected values. Frozen per-array ceilings are relative L2 2e-6
and peak-normalized maximum 8e-6. Max absolute error, ULP against rounded FP32
reference, and both worst locations are reported. No-FMA requires byte identity
for each data/field/memory/operand array; no tolerance replaces that gate.

Twenty-five new tests cover background and nonlinear maps, heterogeneous and
non-square grids, CPML off/on, NT=1, arbitrary nonzero state/profiles, all 390
state basis vectors on a 5x6 grid, corners, pure uploads/zeros/trajectory-copy,
duplicate/scrambled receivers, source endpoints, repeated FULL execution,
invalid configuration/budget/device, every reachable fault position, and
physical timing/polarity/absorption controls. The complete new suite passes.
The unchanged CPU-M9 regression passes 440 tests; the required M8e regression
plus the existing FP32-contract suite passes 88. There are zero skips or
candidate-specific regressions in these runs.

For complete warm-runtime context-to-output timing, compile the CPU view as
a shared library and use the durable benchmark script:

```sh
cc -std=c99 -O3 -Wall -Wextra -Werror -pedantic -fPIC -shared -Iinclude \
  tests/utilities/m9e1_cuda_elastic_psv_forward_harness.c -lm \
  -o /tmp/m9e1_cpu.so
python3 tests/utilities/m9e1_forward_benchmark.py --cpu /tmp/m9e1_cpu.so \
  --nofma src/.cuda/m9/libdenise_m9_cuda_nofma.so \
  --fma src/.cuda/m9/libdenise_m9_cuda.so --output /tmp/m9e1_timing.json
```

The benchmark includes context creation, preparation, all outputs, and
destruction, separately from resident-forward CUDA-event timing. This milestone
does not assert a universal speedup or select a build on timing alone.

Local validation uses WSL Ubuntu 24.04.5, CUDA 12.8.93, RTX 3070 Laptop sm_86,
one logical device, GCC 13.3, and OpenMPI 4.1.6. Host ASan/UBSan runs the actual
production allocator and bridge with all 25 preparation/transfer allocation
sites. Native CUDA lifecycle passes UBSan and all 43 creation fault positions.
GPU ASan cannot initialize this WSL driver; the separate host-only view avoids
the driver and passes without suppressing candidate leaks. Compute-Sanitizer
cannot initialize the Windows WDDM debugger interface, so device memcheck,
initcheck, racecheck, and synccheck are not claimed as passed. No administrator
or machine configuration change was made. NESH is untouched.

Full per-fixture/per-array metrics, logs, JUnit results, exact command records,
retained tested binaries, raw/canonical file hashes, EOL/attributes, and candidate
snapshots are supplied with the implementation report. HEAD stays at the base;
the uncommitted candidate is identified by its complete file manifest.
