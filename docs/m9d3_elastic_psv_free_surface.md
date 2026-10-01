# M9d3 canonical elastic P/SV free surface

This extends the M9 Born/background/nonlinear/J/JT, segmented replay, equal-tile
MPI migration and MODE=2 paths with `FREE_SURF=1`. The accepted M9d3-0 reference
files are immutable scientific input. The historical `surface_elastic_PML_PSV`
kernel is neither called nor changed. `FREE_SURF=0` retains its canonical M9d2
equations and numerical outputs.

## Envelope and representation

Supported: isotropic elastic P/SV, raw cell lambda/mu, fixed rho, INVMAT1=3,
FDORDER=4, NDT=DTINV=1, BOUNDARY=0, a flat top, direct vx/vy receivers, the M9
explosive source, and left/right/bottom CPML. The global serial grid retains the
existing minimum of five cells per axis. MPI retains equal tiles and requires
local NX>=2 and local NY>=4 with a free surface (NY>=2 without it). Only top ranks
with global `oy==0` apply surface operations. Uneven tiles, topography, viscoelastic
migration, density images, CUDA, shot groups, conditioning and mode separation
are excluded.

An explosive source at one-based j=1 is rejected before operator preparation or
image production; j=2 is supported. Surface-valid receivers retain their declared
order and the M9 sampling time, including receivers on another rank. Lambda/mu
must give mu>0 and lambda+2mu>0. Bottom CPML must not overlap the four-row surface
closure: `FW < global_NY-3`.

The production representation remains mixed precision: materials, profiles,
forward/tangent states, eight CPML memories, data, four retained strains and
checkpoints are FP32; reverse state and raw images are FP64. Surface coefficients
are derived from stored FP32 materials, with A/alpha rounded to FP32 for forward
arithmetic; their analytic derivatives and image accumulation use double. There
is no alternate FP64 production ABI or file format.

## Geometry and boundary equations

The physical surface is y=DH/2, normal-stress/vx row r=0. Shear traction is
between sxy[-1] and sxy[0]. Let D=lambda+2mu, alpha=lambda/D and
A=4mu(lambda+mu)/D. For m=1,2:

```
syy[0] = 0
syy[-m] = -syy[m]
sxy[-m] = -sxy[m-1]

vx[-m] = vx[m] + 2m (DH/DT) sum(w[r] Qyx[r], r=0..3)
vy[-m] = vy[m-1] + (2m-1) (DH/DT) alpha Qxx[0]
w = (35,-35,21,-5)/16

sxx_new[0] = sxx_old[0] + A Qxx[0]
syy_new[0] = 0
```

Here Qxx and Qyx are the once-corrected horizontal FD4/CPML operands, including
DT/DH. Their CPML recurrence executes once per step. Ghosts are algebraic values
evaluated at stencil use, not an independent state. Surface shear uses the
existing harmonic-corner mu map; volume stress equations apply below row zero.

## Forward graph

1. Project owned top syy to zero.
2. Exchange stress halos X then Y (MPI); resolve top stress mirrors at stencil use.
3. Advance momentum and its four CPML recurrences.
4. Sample vx/vy in canonical receiver order.
5. Exchange velocity halos X then Y (MPI).
6. Compute corrected horizontal Qxx/Qyx once for all owned cells.
7. Resolve velocity ghosts using those operands and alpha, then compute corrected
   vertical Qxy/Qyy.
8. Retain/regenerate the four strain operands; update volume/shear stress and
   the surface plane-stress sxx/zero-syy map.
9. Inject the explosive source into sxx/syy at its valid physical cell.
10. End the complete timestep and capture a checkpoint where scheduled.

The background and nonlinear maps use the same step. The tangent follows the
same graph with volume lambda/mu terms, harmonic-mu tangent, delta-A Qxx at
surface sxx, and delta-alpha Qxx in the vy ghost. It does not add a volume sxx
material term again at the surface.

```
alpha_lambda = 2mu/D^2          alpha_mu = -2lambda/D^2
A_lambda = 4mu^2/D^2           A_mu = 4(lambda^2+2lambda*mu+2mu^2)/D^2
```

## Exact reverse and surface images

The reverse consumes the full discrete graph in reverse order: transpose stress
updates; consume the surface syy overwrite; accumulate A_lambda/A_mu times Qxx
and sxx-bar; transpose vertical strain CPML/FD4 into the velocity ghosts;
ADD ghost cotangents to their physical sources and horizontal operands;
accumulate alpha_lambda/alpha_mu times the ghost coefficient cotangent;
transpose horizontal strain CPML/FD4; return velocity-halo cotangents Y then X;
inject sampled receiver cotangents; transpose momentum CPML/FD4 and stress
mirrors; return stress-halo cotangents Y then X; consume the leading top syy
projection. Harmonic-corner image contributions return through the existing
material-halo transpose. Overwrites consume cotangents, and mirrors ADD.

For stress boundaries the composed map is `C=Bghost H Ztraction`; its reverse is
`C^T=Ztraction^T H^T Bghost^T`. Actual production halo/surface unit-basis tests
cover both stress components on 1x1, 2x1, 1x2 and 2x2. Top x derivatives consume
exchanged halos at split interfaces and global wrap edges; lower ranks use the
ordinary internal-interface graph.

The four isolated surface material-image terms are separately compared to the
independent reference before full image comparison. No surface normalization,
quadrature weight or image scale is introduced.

## CPML and checkpoint/replay

Profiles are constructed in global coordinates before MPI slicing. Top-y
profiles are identity, with a=0, b=1, kappa=1, above the bottom strip; horizontal
CPML remains active at the free surface. The FREE_SURF=1 horizontal inactive
profile samples follow the frozen reference's damping factor
`b=exp(-pi*FPML*DT), a=0, kappa=1`. This matters for arbitrary-state matrix tests;
zero-initialized inactive memories stay zero. FREE_SURF=0 profiles are unchanged.

Checkpoints retain the five physical FP32 fields plus only active entries of the
existing eight CPML fields. There is no ninth `psi_vxxs`, ghost payload, retained
surface coefficient, or new persistent field. Restore zeroes inactive memories;
MPI continuation poisons unowned field halos and verifies exact regeneration.
FULL and SEGMENTED data/J/JT/strain operands retain the M9d1/M9d2 equivalence
contract.

Complete retained replay bytes remain checkpoint payloads + the `(S-1)` actual
checkpoint objects + both S-entry integer schedule arrays + the four-strain
max-segment operand buffer. Pointer bytes are included in checkpoint objects;
there is no separate pointer table. Selection uses strict `< FULL`; equality
chooses FULL, and MPI selects SEGMENTED only if every rank benefits. The public
MPI summary estimates its usual min(NT,32) policy; tests of another requested
policy explicitly query that policy's estimator.

JT adds a transient two-cell-array double buffer for horizontal cotangents, on
all free-surface ranks. The serial working-state diagnostic counts 17 double
arrays rather than 15. MPI adds `2*(local_NX+4)*(local_NY+4)*sizeof(double)`
transient bytes. This buffer is absent for FREE_SURF=0, excluded from retained
replay bytes and released on every return path. Allocation-failure sweeps cover it.

## Migration, MODE=2 and verification layers

Serial and distributed migration call this exact JT. MODE=2 admits FREE_SURF=1
inside the envelope above; it preserves prepared source/FP32 receiver files,
shot order, root-only global publication, and the transactional lambda/mu pair
of native-endian row-major FP64 images. It does not route to historical RTM_PSV.

FP64 oracle algebra keeps its strict dot/matrix gates and epsilon ladder
1,.5,.25,.125. Production uses the inherited mixed-precision gates: background
2e-6, J 1e-5, raw JT 6e-5; the existing M9b independently bounded dot method;
centered nonlinear FD at epsilon=.05 with ceiling .004. Output quantization
alone can exceed the FP64 oracle dot ceiling, so these layers are reported
separately. No oracle, direction, epsilon or tolerance is adjusted for M9d3.

Durable checks are in `test_m9d3_*`: actual surface matrices and images, complete
timestep basis/sample/VJP, all-state histories, nine J/JT directions/components,
FULL/SEG replay, top depth/source restrictions, four MPI topologies, native
MODE=2, all reachable serial/MPI OOM indices, normal P packet reflection, and
the actual M9 oblique P/P and P/SV forward path with accepted timing,
polarization and independent analytic polarity coefficients.

This document describes an unpublished implementation candidate. Independent
scientific verification and user acceptance remain separate gates.
