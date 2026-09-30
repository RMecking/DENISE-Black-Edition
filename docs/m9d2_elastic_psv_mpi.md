# M9d2: distributed elastic P/SV Born-adjoint and MODE=2

Candidate BASE: `bd3f7a9a9f0ba90ae681968cc3c82244b7981249`.
Implementation branch: `codex/m9d2-elastic-psv-mpi`. Publication is not authorized.
The serial M9b/M9c/M9d1 APIs and numerical source files remain canonical and
MPI-independent. MPI-specific headers, numerical context and migration driver
are additive. The MODE=2 adapter keeps its existing valid one-rank path.

## API and topology

`denise_elastic_psv_born_mpi_config` embeds the existing scientific config,
an explicit communicator, `nprocx` and `nprocy`. Models supplied in that config
are compact OWNED `[local_y][local_x]` tiles; dimensions and geometry are global.
All ranks supply identical scalar config, source signal and receiver geometry.
A metadata/source/geometry signature is checked collectively before setup.

Create duplicates the communicator; destroy collectively frees it before
`MPI_Finalize`. Tags belong to this duplicate. Numerical modules do not use
WORLD, SHOT_COMM, MYID, POS or INDEX. The production adapter explicitly passes
WORLD after requiring one shot group. A valid collective communicator and
identical collective call order are caller obligations; MPI cannot repair a
caller that substitutes a different/null communicator on only one participant.
All-null communicator calls are rejected before MPI access. Recoverable local
configuration/allocation/input failures are propagated collectively. The owned
communicator explicitly uses `MPI_ERRORS_ARE_FATAL` for unrecoverable transport
or protocol errors; no scientific result is silently returned after one.

Topology is equal rectangular tiles, rank `py*nprocx+px`, no reordering.
`nprocx*nprocy == comm_size`, NX and NY must divide exactly. Global dimensions
are at least five, as in the canonical operator. Local dimensions are at least
two: one neighboring tile must supply both FD4 stencil layers. A 6x6 global
grid decomposed 3x1 exercises exact local NX=2; 6x1 gives NX=1 and is rejected.
Processor coordinates are derived from the validated communicator rank and
never supplied as unchecked external indices. Uneven tiles are excluded.

For local `(i,j)`, global coordinates are `(offset_x+i,offset_y+j)` with
`offset_x=px*local_nx`, `offset_y=py*local_ny`. Field/model working arrays have
stride `local_nx+4` and extents `(local_ny+4,local_nx+4)`. Owned flat index is
`(j+2)*stride+i+2`; ghost coordinates range from -2 through local_extent+1.
Checkpoint, retained strain, input directions and returned images use compact
owned layout. Ghosts are never checkpointed or retained as trajectory operands.

## Global edges, maps and CPML

The serial derivatives wrap at global edges. Therefore left/right/top/bottom
neighbors are explicitly periodic, including self-neighbors in unsplit axes.
This reproduces the global operator; it does not imply periodic physical CPML.

Halo COPY exchanges x strips first, then y strips including x ghosts. This
provides diagonal material samples at four-rank corners and global wrap.
Inverse density maps use the canonical double intermediate expressions before
FP32 storage. Harmonic corner mu reads `(j,i)`, `(j,i+1)`, `(j+1,i)` and
`(j+1,i+1)` in exactly the serial expression order. Perturbed mu is exchanged
before its harmonic tangent is computed. Tests compare every owned material
map value bitwise to the canonical serial global map, covering all interfaces.

CPML profiles are generated with GLOBAL lengths and coordinates, then sliced
to owned coordinates. Temporary global one-dimensional profiles are released.
Profiles never depend on neighbor existence. Internal interfaces have no extra
damping. All profile coefficients are checked bitwise against global slices.
Checkpoint payloads contain only local memories whose canonical profile a is
nonzero. Interior memories must be exactly zero at capture. Strips can extend
across multiple ranks: the 32x24 FW=9, 4x1 gate crosses an internal boundary.
There is no restriction requiring a whole strip to fit on an edge rank.

## Forward, tangent and transpose graph

Each timestep copies three stresses, updates owned velocities, samples owned
receivers, copies the two updated velocities, computes corrected strains,
updates owned stresses, injects the unique source, and captures diagnostics or
checkpoint state. FD4 coefficients, expression order, FP32 state, receiver
time, source time and CPML recurrence are unchanged.

Source owner is `global_i/local_nx + nprocx*(global_j/local_ny)`. Only that rank
injects. Receivers have the same ownership rule, retaining ascending GLOBAL
ordinals locally. Trace assembly copies unique samples into chronological
`[time][global_receiver][vx,vy]`; it neither sums nor sorts by rank. Geometry
tests scramble receiver order and include boundary, far-side and diagonal
receivers. Source gates include interior, vertical/horizontal boundaries and
both sides of a four-rank corner.

J maintains its tangent fields across all timesteps and replay segments. It
uses the same stress and velocity copy edges as the background.

For COPY transpose, ghost cotangents are extracted and cleared, communicated
to their owner and ADDED once. The staged copy is transposed in reverse order:
y including x ghosts, then x. J-transpose reverses the strain/stress graph,
transposes velocity COPY, injects owned receiver cotangents, reverses velocity
update, then transposes stress COPY. No forward copy is used as an adjoint.

Harmonic-mu transpose produces four cell-centered contributions. Its separate
halo transpose returns remote edge/diagonal/wrap contributions before the next
timestep. Raw lambda/mu images accumulate in double precision. Cross-rank
addition grouping can differ from serial row-major scatter order; this is the
only accepted decomposition difference. Global relative L2 must be <=1e-12,
with maximum absolute/ULP/location and interface-strip errors recorded. This
does not relax the frozen M9b J <=1e-5, J-transpose <=6e-5 or dot bounds.

## Checkpoint/replay and memory selection

The M9d1 floor-partitioned schedule is identical on every rank. Checkpoints
contain five owned FP32 fields plus active local CPML strips and the original
48-byte checkpoint object metadata. Restore zeros ghosts and omitted memories;
stress/velocity copies regenerate ghosts before consumption. The continuation
roundtrip compares owned fields and all CPML memories after the following step.

FULL retains `NT*4*owned_cells*sizeof(float)` bytes. Complete replay retains
all checkpoint payloads, `(S-1)*sizeof(checkpoint)` object bytes, both S-entry
integer schedules, and `max_segment_length*4*owned_cells*sizeof(float)` operands.
There is no pointer table. Overflow arithmetic is checked. Forward/adjoint
scratch use padded local arrays; no rank retains a global model/trajectory/image.

Production chooses min(NT,32) replay segments only if EVERY rank has complete
replay bytes strictly smaller than local FULL bytes. Any equality or loss
selects FULL collectively. Explicit forced replay remains available for gates.
The mixed-boundary 32x24 NT=55 FW=3 4x1 case falls back to FULL even though
interior ranks benefit; NT=120 selects SEGMENTED everywhere. An exact equality
gate uses owned 2x2 cells, NT=69, S=32: FULL=replay=4416 bytes, so selects FULL.
Forced S=1, S=NT and S>NT (clamped) reproduce FULL bitwise.

Diagnostics report local FULL/complete replay/actual retained bytes; payload,
objects, pointer table (zero), schedules, operands and core transient working
bytes; communicator min/max/sum; local receiver data bytes; per-step wavefield
send bytes and actual stress/velocity COPY and transpose counters, material
transpose counters and replayed steps. Per-field COPY sends
`4*(local_nx+local_ny+4)` values, including diagonal/padding traffic. Receive
traffic is the same; reported bytes are SEND bytes, including self-neighbors.
Forward uses float and transpose uses double. Model setup sends mu and rho;
each tangent/nonlinear mu input adds one material COPY; J-transpose adds one
double material transpose per timestep. Core working diagnostics exclude the
compact-array adaptation buffers and caller-owned accumulators.

Context models/maps occupy `6*padded_cells*4` bytes. Halo send/receive buffers
occupy `2*max(2*local_ny,2*(local_nx+4))*8` bytes. Local one-dimensional profiles
occupy `6*(local_nx+local_ny)*4`; source samples occupy NT*4. Receiver metadata
currently allocates three global_receiver_count integer capacities, retaining
only the local valid entries; this is geometry metadata, not global trace data.
J/nonlinear temporary input adaptation adds two padded FP32 arrays. JT adds
two padded double image buffers plus its reported 15-array working state.
The migration caller holds four compact owned double images while stacking;
its result retains two. Prepared sources and local trace data are retained for
the shot list, in addition to the context's one source copy. Root-only temporary
global input/output assembly is released at phase completion.

## Migration and production MODE=2

The MPI migration request uses owned model arrays and local trace arrays, with
global geometry; returned result images are owned. Every communicator rank
visits every physical shot in strict ascending order. Each shot has a fresh
context, collective backend selection and double local image addition. Gather
copies rank tiles into root global row-major images without floating reductions
or averaging; a rank-coded synthetic image independently tests placement.

Root reads exact-size x-major .lam/.mu/.rho files, validates global CFL and
scatters compact tiles, releasing global models. Root strictly parses existing
source/receiver files; one-based file coordinates convert to zero-based once.
Geometry and prepared NT-float32 source signals are broadcast; only source
owners inject. Root reads chronological vx/vy files and scatters only each
owner's traces. Non-root ranks never retain global datasets.

The adapter requires NCOLORS=1 and rejects shot groups. Entry validation routes
invalid processor counts to collective cleanup before legacy division/splitting.
Only root writes native-endian float64 `.image_lambda_raw.bin` and
`.image_mu_raw.bin`. The original paired temporary-write/rename transaction is
retained. All ranks finish before publication; root status is then propagated.
Any setup/input/gather/publication failure removes both final paths and partial
temporary files. Non-root ranks never write, rename or remove image paths.

Status gates cover context arrays, material/profile setup, checkpoint/replay
allocation, shot geometry/data, prepare, directions/data, image buffers and I/O.
Fault injection is test-only linker `--wrap=calloc`; no production environment
switch can inject failures. A null context on a single participating caller is
a collective-call contract violation, since its communicator is unavailable.

Before legacy shot counting, MODE=2 uses a collective root geometry-count
preflight. Missing/invalid source declarations remove the configured temporary
and final image names on root before failure propagates. This closes the
legacy count_src early-abort gap on 1, 2 and 4 ranks; valid one-rank inputs
still return exactly the canonical source count and diagnostics.

## Verification and reproduction

`tests/physics/test_m9d2_elastic_psv_mpi.py` builds a strict C99 mpicc harness.
Machine-readable results are root-only files after collective assembly.
The immutable serial implementation is included only by that TEST harness to
expose maps/profiles for comparison; production MPI source has no serial-core
include or global-array fallback. Scientific oracle calculations come from the
unchanged M9b-0 Python oracle and the unchanged M9b-1 dot acceptance helper.

The matrix covers 1x1/2x1/1x2/2x2, interior/CPML, lambda/mu/joint and
vx/vy/both. It adds dense nonlinear FD, FULL/replay/strain equality, checkpoint
continuation, corner/wrap perturbations, source ownership, backend selection,
allocation/configuration failures, 20 repeated context reuse sweeps, real
multi-shot MODE=2 and paired output failures. Unchanged M9b/M9c/M9d1/H1 and
P/SV distributed/gradient tests, exact BASE CPU differential, strict builds and
local 2-/4-rank ASan/UBSan are required before the candidate is handed over.

The initial NEW FD fixture accidentally used epsilon=.01 rather than the
canonical M9b-1 epsilon=.05. An exact BASE control reproduced its .00757
interior/.00709 CPML errors, while MPI plus/minus bytes were identical to BASE.
At the pre-existing epsilon=.05 the BASE errors are .001048/.001071. The new
fixture was corrected to that existing step, retaining the .004 ceiling and
direction. The failed run and BASE control are preserved separately; no frozen
oracle, tolerance or production expression was changed to satisfy it.

No NESH access, CUDA, viscoelastic migration, free surface, uneven decomposition,
shot parallelism, compression, Revolve, disk checkpoints or image conditioning
is added. Local correctness is the gate; tiny-fixture MPI speedup is not required.
