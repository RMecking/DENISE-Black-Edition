# M9e-3: isolated FULL CUDA elastic J-transpose and raw migration

Candidate BASE/HEAD: `56800545b0128b5de62fe96aec06b1a01d082968`.
Branch: `codex/m9e3-cuda-jt-migration`. Implementation is uncommitted;
the exact candidate is identified by the separate raw/filtered blob manifest.
Independent scientific verification and user acceptance remain outstanding.

## ABI, ownership and precision

`denise_cuda_m9_create_migration` adds reverse ownership to the existing FULL
J constructor. The M9e-1 legacy and M9e-2 FULL constructors retain their exact
device budgets. Existing forward/J kernels and numerical expressions are
unchanged. Ordinary DENISE has no CUDA dependency.

The new sequence is create_migration -> prepare -> apply_j / apply_jt ->
image_download -> destroy. `apply_jt` accepts a finite FP32 chronological
`[time][receiver][vx,vy]` array with its exact value count. The caller supplies
receiver adjoint data; the backend performs no residual construction.
`image_download` requires two FP64 `[y][x]` row-major output arrays and the
exact physical cell count. `migrate` composes prepare, apply_jt and transactional
image_download for one shot.

Reverse physical fields are FP64 padded five-field arrays. Reverse CPML is
FP64 compact eight-channel state. Images are FP64 raw lambda/mu arrays, fixed
rho. Four compact FP64 scratch arrays hold horizontal strain bars, one current
channel and corner-mu bars. They are timestep workspace, with no surface
history. The prepared FP32 background operands remain immutable. No forward
recomputation is hidden in apply_jt.

All reverse state, images and scratch are zeroed on each apply_jt. Tangent and
reverse transients are separate. Successful J/JT calls preserve the background
and each other's independent result ownership. prepare/nonlinear/test mutation
invalidates all results. Failure invalidates prepared/J/JT state; destroy uses
the inherited tracked ownership ledger and uninjected cleanup.

## Reverse graph

Each reverse timestep runs from NT-1 to zero. There is no tangent source and
no source-gradient action.

1. FS1 consumes the output syy projection.
2. Accumulate direct raw normal-material images and corner shear bars.
3. FS1 applies the homogeneous rounded A closure to Qxx; delta-A contributes
   to the images in the preceding direct-material stage.
4. Harmonic corner transpose accumulates four centered mu contributions.
5. Reverse constitutive/strain channels and their CPML recurrences. FS0 uses
   the canonical CPU order Qxx, Qyx, Qxy, Qyy. FS1 first reverses vertical
   Qxy/Qyy, including velocity ghosts, then horizontal Qxx/Qyx. This dependency
   order is essential because ghost bars add into horizontal strain bars.
6. Inject receiver bars after reverse strains, before reverse momentum.
7. Reverse the four momentum CPML/derivative channels in canonical CPU order.
8. FS1 consumes the input syy projection.

Independent channels retain the CPU accumulation order. Each FD transpose
scatters into unwrapped width-two ghost locations. Surface transpose consumes
overwritten top ghosts before the periodic COPY transpose. Periodic COPY is
transposed as Y ADD/clear followed by X ADD/clear. No reverse COPY or averaging
exists. A single thread owns a row for x-scatter or a column for y-scatter;
halo-fold threads own disjoint rows/columns. No numerical atomicAdd is used.

Receiver injection assigns one thread per physical cell. It traverses the
original receiver list in order, adding both components. Duplicate receiver
positions therefore add deterministically, even with scrambled order.

For each CPML channel, with z the corrected cotangent and p the new-memory
cotangent: total=z+p; raw_bar=z/kappa+a*total; old_memory_bar=b*total.
Coefficients are the frozen FP32 profiles widened to double. Explicit double
multiply/add intrinsics preserve this CPU evaluation order in both builds.
All eight channels include arbitrary nonzero memory cotangents in their gates.

## Raw material and surface images

Bulk lambda adds (sxx_bar+syy_bar)*(background_Qxx+background_Qyy).
Bulk normal mu adds 2*(sxx_bar*background_Qxx+syy_bar*background_Qyy).
Corner shear bar is sxy_bar*(background_Qyx+background_Qxy). Harmonic mu uses
the rounded production corner coefficient H: common=0.25*H*H*corner_bar,
then common/mu_cell^2 for each of its four centered contributors. Each image
owner gathers its four corner contributions in ascending corner index, matching
the CPU scatter order, including periodic wrap and surface-adjacent corners.

The surface coefficients use double lambda/mu intermediates D=lambda+2*mu:
alpha and A are rounded to FP32 then widened, while canonical material
derivatives remain double. h is the reciprocal of the rounded FP32 DT/DH,
rounded to FP32, not a separately computed DH/DT.

For top sxx, homogeneous closure contributes A*sxx_bar to Qxx_bar.
Delta-A images add A_lambda*background_Qxx*sxx_bar and
A_mu*background_Qxx*sxx_bar. A_lambda=4*mu^2/D^2;
A_mu=4*(lambda^2+2*lambda*mu+2*mu^2)/D^2.

For vy[-m], factor=(2*m-1)*h*ghost_bar adds to vy[m-1], to Qxx through alpha,
and to lambda/mu images through background_Qxx*alpha_lambda/alpha_mu.
alpha_lambda=2*mu/D^2; alpha_mu=-2*lambda/D^2.
For vx[-m], ghost bars add to vx[m] and the four Qyx rows through
2*m*h*(35,-35,21,-5)/16. There is no material derivative of this extrapolation.

Stress mirrors subtract ghost cotangents into syy[m] and sxy[m-1] for m=1,2,
then clear the ghosts. syy[0]=0 has a projection transpose with no surviving
identity contribution. Projection, mirrors, A closure and velocity ghost
operations have separate isolated basis gates in addition to complete-step
matrix tests.

This is the canonical mixed-precision CPU M9 adjoint contract, including its
double FD4 transpose coefficients and material derivatives. It does not claim
a mathematical derivative of the floating-point rounding function itself.
FP32 forward/J, FP64 reverse and independent FP64 operator gates are distinct.

## Mandatory memory and transactions

Let N=NX*NY, P=(NX+4)*(NY+4), T=NT, R=receiver_count.
The inherited FULL-J byte count is

`B = 124*P + 24*(NX+NY) + 4*T + 12*R + 16*T*R + 16*T*N + 28*N`.

The new constructor adds

`A = ((8-B%8)%8) + 40*P + 64*N + 16*N + 32*N + 8*T*R`.

Thus total is B+A. The first term explicitly aligns the FP64 section.
40P is five padded reverse fields; 64N is eight CPML memories; 16N is two
images; 32N is four transient scratch arrays; 8TR is the FP32 receiver input.
Checked multiplication/addition and runtime device/budget discovery occur
before allocation. There is one cudaMalloc arena, with no hidden allocations.
Budget caps one byte below the mandatory count fail before ownership.

Image download uses one tracked host scratch allocation of 16N bytes and one
device-to-host copy. Only after all operations succeed are both caller arrays
published. Failed JT/download/migrate retain caller sentinels and invalidate
image validity. Repeated JT and J->JT->J are bitwise repeatable on the local GPU.

## Scoped evidence and reproducibility

The new durable module is `tests/physics/test_m9e3_cuda_jt_migration.py`.
Use the existing opt-in CUDA targets with CUDA_ARCHS=86 and CUDA Toolkit 12.8.
The tests discover CUDA via the inherited prerequisite fixture; compilation or
numerical failures never become skips. Both FMA and no-FMA builds participate.

The CPU production sources and independent M9b/M9d3 oracles are unchanged.
Frozen ceilings: J relative L2 1e-5; raw image relative L2 6e-5;
independent FP64 and local transpose 5e-13. Mixed-production dots use the
existing M9b triangle-inequality contract, not a fitted pair-relative tolerance.
The material-basis norm bound 7e-5 is the sum of the inherited J/JT ceilings.

The independent complete-step matrix uses FP64 arithmetic and the frozen
rounded coefficient nodes downloaded from the production static map. Its
375 columns cover 13 state and two raw material channels on a 5x5 grid.
All 337 output state/receiver basis cotangents, plus an arbitrary bar, are
compared against NumPy matrix transpose. This checks complete velocity/stress,
all eight CPML memories, receiver timing and surface/material dependency paths.
It supplements the unchanged full-trajectory FP64 oracle.

Regressions are limited to M9e-1 and M9e-2 because their shared CUDA translation
unit changed. Unrelated M8/CPU-M9/SH/oracles are checked by source identity.
The inherited constructor/destructor ownership code now includes the additive
JT owner, so the required M9e-2 suite also retains its existing fault coverage.
New fault sweeps arm JT/image operations after preparation, and separately cover
the new constructor and migration composition. The host-only ASan/UBSan harness
checks the actual new layout gate; the CUDA host UBSan harness checks actual
JT/image ownership. Compute-Sanitizer's WSL/WDDM debugger failure is classified
ENVIRONMENT / DRIVER LIMITATION, never a sanitizer pass.

Performance is diagnostic only: row/column scatters and receiver-list traversal
prioritize deterministic correctness. No optimization/acceptance speedup is
claimed. Reports include preparation, J including output download, JT, image
download, clean migration and owned bytes.

No replay, MODE=2 CUDA routing, MPI multi-GPU, viscoelastic migration,
density/Q/VpVs images, FWI residual construction, source inversion, normalization,
conditioning, smoothing, illumination compensation or MPI stacking is included.
All work and validation are local. No NESH connection or execution occurs.
