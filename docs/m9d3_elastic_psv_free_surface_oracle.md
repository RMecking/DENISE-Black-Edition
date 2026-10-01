# M9d3-0: independent exact elastic P/SV flat free-surface oracle

Candidate for scientific review, BASE `2c05c71c14f020dd13322656f3a240b6edfac686`.
No production M9d3 implementation or production tolerance is accepted here.
Only new reference/tests/documentation are introduced. The formulas and gates
below are selected before examining production M9d3 output. No generated
field hash is a cross-platform mathematical acceptance criterion.

The production envelope to be reviewed is elastic isotropic P/SV (L=0),
lambda/mu with fixed rho, FDORDER=4, NDT=DTINV=1, FREE_SURF=1, BOUNDARY=0,
one flat top surface, direct vx/vy data, and existing M9 stress-source/raw
migration semantics. Left/right/bottom CPML is allowed. Viscoelasticity,
density imaging, CUDA, topography, non-flat surfaces, uneven tiles, shot
groups and higher production FD orders are outside this candidate.

## Geometry derived from derivative locations

For one-based DENISE `(i,j)`, normal stress/material are at
`((i-1/2)DH,(j-1/2)DH)`, vx at `(i DH,(j-1/2)DH)`, vy at
`((i-1/2)DH,j DH)`, and sxy at `(i DH,j DH)`. The backwards
vx difference is centered on normal stress and the forwards vy difference
on vy; the shear derivative pair is centered on the corner. These derivative
locations determine the relative offsets, independently of the historical
surface comment. DENISE source/receiver integer rounding establishes the
absolute origin (see existing `staggered_grid.py` and its physical tests).

The normal traction node `j=1` is therefore at **y=DH/2**. The shear traction
is between `sxy(j=0)` and `sxy(j=1)`, at that same y. There is no physical
sxy sample on the surface. In M9 row-major arrays `r=j-1, c=i-1`, the surface
normal/vx row is **r=0**. Normal coordinates are `(r+1/2)DH`; vy/sxy
coordinates are `(r+1)DH`. Relative to the surface, normal/vx rows have
y=r DH and vy/sxy rows have y=(r+1/2)DH. Ghost rows `r=-1,-2` correspond
to DENISE `j=0,-1`, not physical storage rows at the bottom of the array.

## Physical traction, stress extension, and constitutive law

The normal is `(0,-1)`, so zero traction requires syy=0 and sxy=0 at y=DH/2.
These are different discrete maps:

* Physical overwrite: `syy[0,c] := 0`.
* Normal-stress ghosts: `syy[-m,c] := -syy[m,c]`, m=1,2.
* Shear-stress ghosts: `sxy[-m,c] := -sxy[m-1,c]`, m=1,2.

Thus `9/16*(sxy[-1]+sxy[0])-1/16*(sxy[-2]+sxy[1])=0`, the cubic
midpoint traction interpolation. No physical shear value is zeroed. sxx is
not traction and is not zeroed or odd-mirrored. All ghost overwrites consume
their incoming cotangent; a reflected ghost adds `-ghost_bar` to its source.
The syy physical overwrite consumes its incoming cotangent completely.

Write `ex=DT*Dx(vx)` and `ey=DT*Dy(vy)` at a normal node. The elastic normal
increments are `(lambda+2mu)ex+lambda ey` and `lambda ex+(lambda+2mu)ey`.
Eliminating ey using zero normal traction gives

```
D = lambda + 2mu
alpha = lambda/D
ey_surface = -alpha ex_surface
A = D - lambda^2/D = 4mu(lambda+mu)/D
sxx_new[0] = sxx_old[0] + A ex_surface
syy_new[0] = 0
```

The surface sxx term **is required**. Keeping the bulk coefficient D gives a
different operator; zeroing sxx would impose an unphysical tangential stress
condition. The direct plane-stress increment is algebraically equivalent to
adding `-lambda^2/D*ex-lambda*ey` after a bulk normal update, provided the same
corrected ex/ey operands are used.

## Explicit velocity extension and FD4 closure

FD4 coefficients are `a1=9/8,a2=-1/24`; every derivative divides by DH and
uses `a1(f_+-f_-)+a2(f_outer+-f_outer-)` at its staggered location.
Let Qxx/Qyx denote the already horizontally CPML-corrected DT-scaled x
derivatives of vx/vy, computed **once**. The surface vy derivative is
extrapolated from four half-depth rows with weights `(35,-35,21,-5)/16`,
the cubic polynomial evaluation at y=0. Set `W=sum(w_r Qyx[r])`.
The chosen algebraic velocity extension is

```
vy[-m] = vy[m-1] + (2m-1)*(DH/DT)*alpha*Qxx[0]
vx[-m] = vx[m]   + 2m*(DH/DT)*W                     (m=1,2)
```

These positive mirrors plus affine terms impose the constitutive normal
derivative and the tangential shear derivative: `Dy vy(0)=-alpha Dx vx(0)`
and `Dy vx(0)=-Dx vy_surface`. The normal FD4 identity follows from
`a1+3a2=1`; the central derivative of the affine vx extension is exact.
This is an explicitly selected discrete closure, not a claim that continuum
traction uniquely specifies every ghost degree of freedom or that the whole
boundary has fourth-order truncation accuracy. The mirror closure and cubic
surface extrapolation are frozen; volume derivatives remain FD4.

No ghost is a persistent dynamic degree of freedom. Material directions act
on this velocity extension as well as on the direct plane-stress increment.
Bottom/external wrap follows M9d2's algebraic halo convention; bottom CPML
must attenuate physical waves before this artificial outer closure matters.

## Material tangent and exact reverse

At the surface:

```
alpha_lambda = 2mu/D^2; alpha_mu = -2lambda/D^2
A_lambda = 4mu^2/D^2
A_mu = 4(lambda^2+2lambda*mu+2mu^2)/D^2
delta_alpha = alpha_lambda*delta_lambda + alpha_mu*delta_mu
delta_A = A_lambda*delta_lambda + A_mu*delta_mu
delta_sxx_new[0] = delta_sxx_old[0] + A*delta_Qxx[0] + delta_A*Qxx[0]
delta_syy_new[0] = 0
delta_vy_ghost[-m] = delta_vy[m-1]
    + (2m-1)*(DH/DT)*(alpha*delta_Qxx[0]+delta_alpha*Qxx[0])
```

The vx extension has the analogous state tangent through W. Qxx/Qyx include
horizontal CPML's state tangent, but its base coefficient profiles are fixed
in the parameter operator (as in a frozen prepared M9 coefficient operator).
Nonlinear material FD uses those same profiles; no damping-speed derivative
is silently introduced. Density and face inverse masses are fixed.
The four-cell harmonic shear coefficient and its material tangent include
all four owner contributions, including halo dependencies.

Reverse differentiation first clears syy[0], then reverses the top sxx
increment: `Qxx_bar += A*sxx_bar`,
`g_lambda += A_lambda*Qxx*sxx_bar`,
`g_mu += A_mu*Qxx*sxx_bar`. Reverse the ghost extensions with ADD:
vy ghost cotangents give physical mirror ADD, Qxx ADD, and
`alpha_bar += (2m-1)*(DH/DT)*Qxx*ghost_bar`;
vx ghost cotangents give mirror ADD and `Qyx_bar[r] += 2m*(DH/DT)*w_r*ghost_bar`.
Convert alpha_bar to both physical material images via its derivatives.
Then reverse each CPML recurrence, FD4 stencil, sampling, velocity update,
and stress ghost construction in exact reverse order. Persistent paths have
identity cotangents except explicitly overwritten entries.

For each CPML memory `p'=b p+a q`, `q'=q/K+p'`, reverse uses
`t=q'_bar+p'_bar`, `q_bar=q'_bar/K+a t`, `p_bar=b t`.
No continuous-adjoint equation, fitted sign, DT/DH image weighting, or legacy
RTM normalization defines this VJP. Images are raw Euclidean lambda/mu
directional products.

## Timestep, MPI order and ownership

The frozen complete source-free graph is:

1. Project incoming physical syy[0] to zero; exchange stress x then y halos.
2. Regenerate top stress ghosts, overriding top wrap. Apply velocity update.
3. Sample direct vx/vy nodes (NDT=DTINV=1).
4. Exchange velocity x then y halos. Compute corrected Qxx/Qyx once.
5. Construct top velocity ghosts. Compute y strains and update stresses,
   using direct plane stress at row 0.
6. Inject the prescribed stress source, below row 0. This is the post-complete-
   timestep checkpoint boundary; surface ghosts are regenerated on next use.

For arbitrary stress input, the leading physical zero projection precedes
halo copying, so bottom wrap also receives syy[0]=0. In valid trajectories
that value already vanishes. Halo and surface maps are separate explicit
matrices: stress consumption uses `C=Bghost H Ztraction`. The transpose is
`Ztraction^T H^T Bghost^T`. A full H is H_y H_x, so reverse is H_x^T H_y^T.
Velocity surface construction follows exchange; reverse construction precedes
reverse exchange. There is no top wrap contribution after top ghost overwrite.

Only `pos_y==0` ranks construct surface ghosts or apply plane-stress law.
Interior y interfaces retain M9d2 halo copies. Top x derivatives consume
exchanged x halos, including interface-adjacent and global wrap edge cells.
The NumPy emulated rank graph uses actual local copied halos and owned FD4
updates, followed by gathering. It covers 1x1,2x1,1x2,2x2 without MPI C.
The local copy matrix's transpose independently checks rank ADD/clear.

For FD4 stress mirrors the deepest normal source is row 2, requiring at least
three owned rows. This chosen velocity surface cubic extrapolation uses rows
0..3, so the frozen **top local_ny minimum is 4**. An ordinary interior rank
needs at least 2 owned rows for M9d2 width-2 communication. With uniform tiles
the top requirement implies local_ny>=4 everywhere. local_nx>=2; global
NX,NY>=4 in tiny reference fixtures; production retains the existing M9d2
global NX,NY>=5 rule. Uneven domains and shot groups are excluded.

## CPML and checkpoint/replay

Top y profiles are K=1,a=0,b=1: no top absorber or advancing y memory. Left,
right and bottom remain active; fixture layers do not overlap the top closure.
Horizontal Qxx and Qyx and their memory updates participate in the boundary.
Qxx is corrected once and reused, preventing a second advance of the same
memory. The persistent checkpoint contains the five physical fields and the
active values of the eight existing CPML memories, after the complete timestep
and source injection. Inactive values must be exactly zero at capture and
restore as zero, preserving the M9d1/M9d2 packing invariant. The reference
checks packed restore, continuation data, final state and regenerated strains.
No new persistent surface state is introduced. A prepared model also restores
material, profiles and acquisition configuration; physical ghost/halo values
are reconstructed before consumption, not serialized.

## Historical audit (genealogy only)

At BASE, `surface_elastic_PML_PSV.c` zeroes physical syy(j=1), reflects
syy(j-m)=-syy(j+m) and sxy(j-m)=-sxy(j+m-1), then adds the bulk-to-plane-
stress correction. Its lambda/mu algebra agrees with the independent Schur
elimination for INVMAT1=3. `psv.c` places it after bulk update and source,
before stress exchange; only POS[2]==0 calls it. The elastic call passes
**psi_vxxs**, a separate memory from volume psi_vxx. Thus this is not two
updates of the same memory: with equal initial conditions and raw x operands
the two recurrences agree. Its additional surface memory would require replay
storage if preserved literally. The canonical oracle instead reuses the single
corrected volume Qxx and needs no additional surface checkpoint memory.

The historical kernel itself does not construct a P/SV velocity extension.
The canonical closure above is independently specified; ghost initialization
or stale MPI data are not a definition of the desired traction-free operator.
The historical source-at-surface behavior projects its normal load away while
retaining sxx, so this first M9d3 envelope explicitly rejects explosive j=1
sources. Sources at j=2 (within stencil reach) and safely below are supported;
receivers at j=1 are physical direct staggered velocity nodes and supported.
No historical code is modified and historical physics regression results
corroborate traction intent, not every new discrete closure choice.

Independent theoretical context: [Levander (1988), fourth-order P/SV
staggered-grid formulation](https://csim.kaust.edu.sa/files/ErSE210/Levander.PSV.pdf)
derives constitutive laws, stress images, and surface derivative elimination.
The geometry and exact matrices in this candidate are derived explicitly above.

## Frozen verification gates

All reference arrays/arithmetic are FP64. Dot residual is normalized by the
larger operand-norm product, not a cancellation-small signed dot. Local and
multistep dot maximum is **5e-13**; dense-forward-matrix/VJP relative comparison
maximum **5e-12**. Physical traction and ghost-parity scale-normalized residual
maximum **5e-14**. Tolerances are fixed before production comparison.
Nonlinear FD epsilons are **(1,.5,.25,.125)** with 1% deterministic physical
lambda/mu directions: require at least **3.9x** reduction per halving and final
relative difference **<2e-5**. The isolated timestep material dense-FD probe
uses independently chosen epsilon **1e-4** and relative difference **<1e-7**.
Repeatability, decomposition background/Born data and checkpoint replay require
exact array equality. Distributed adjoint images use the already frozen
matrix-relative bound 5e-12: the explicit rank ADD/clear transpose groups FP64
additions differently than global stencil scatter, as in M9d2. Initial global
reverse reuse could show artificial bitwise equality; it is replaced here by
the actual emulated-rank halo transpose. This correction tests the communication
graph rather than claiming an invalid cross-grouping bitwise invariant.
These gates cover separate lambda/mu/joint and vx/vy/both directions.

Explicit unit-basis matrices define each local forward map; `B.T` is the
authoritative local transpose oracle. The hand reverse is then checked against
that matrix. A dense full timestep state Jacobian includes every CPML state;
the dense parameter Jacobian independently checks boundary image dependence.
An independent upward normal P packet additionally checks the new reference
against geometric travel times (tolerance `2DT+0.005*t`), reflection polarity,
and zero transverse mode. The packet's initial velocity follows
`rho*v_t=stress_y`: upward positive stress has positive vy. This mode
initialization, rather than an explosive source calibration, fixes the sign.
Physical normal-incidence and oblique P/P-SV reflection checks are the existing
`tests/physics/test_free_surface.py` integrations with their original timing,
polarity/polarization and amplitude gates; their production kernel is evidence
of reflection physics, not the discrete Born/VJP authority for this candidate.

Run the new suite, existing M6.3c MPI/surface VJP suite, M9b oracle, and existing
elastic P/SV free-surface integrations. Numerical observations and exact final
file identities are reported at candidate handoff; no publication occurs here.

## Deterministic candidate observations

The coupled fixture is NX=NY=8, NT=32, DH=10, DT=0.0004, FW=2, rho=2000,
`lambda=6.44e9*(1+0.04*sin(0.7x+0.2y))`,
`mu=5.78e9*(1+0.03*cos(0.3x-0.5y))`, zero-based x/y in those formulas.
It has one source at one-based (4,2) and receiver ordinals
(1,1),(4,2),(5,1),(8,2),(2,7), spanning x ranks at the surface.
The existing independent M9b Ricker/time differentiation is reused, with
source_t0=0.003 and source amplitude 1e6. Material directions use NumPy RNG
seed 423 and 0.01 times each physical base field; data cotangents use seed 424.
No runtime output is used to retune these directions, epsilons or thresholds.

Windows CPython 3.12.10 / NumPy 2.5.3 and Linux/WSL2 CPython 3.13.15 /
NumPy 2.5.3 reproduce the numerical checks. The largest isolated local
boundary dot residual is 4.104713272577412e-17 (active x-CPML + sxx block);
largest composed halo/surface residual is 1.2897104413634255e-17.
Full timestep state+sampling matrix/VJP relative difference is
9.711379299721341e-18. Material matrix/image closure is
1.0001511958954003e-16; centered local material FD difference is
6.367200115166341e-11. Largest multistep J/JT dot residual across nine
lambda/mu/joint × vx/vy/both cases is 1.748550689804184e-17.

The nonlinear FD ladders in epsilon order (1,0.5,0.25,0.125) are:

```
lambda: [1.1906663309125964e-5, 2.976646091491787e-6,
         7.441603143060864e-7, 1.8604053294830266e-7]
mu:     [3.52600048746496e-4, 8.812705809239387e-5,
         2.2030330327888138e-5, 5.50749298998788e-6]
joint:  [3.211316064667286e-4, 8.026230027696242e-5,
         2.006428786368154e-5, 5.015991600462435e-6]
```

All traction, parity and algebraic stress-extension closure residuals are 0.0.
The packed checkpoint contains 544 FP64 values (five fields plus active
memories), and continuation/reconstructed-strain residuals are exactly 0.0.
For 2x1/1x2/2x2, background and Born equality is exact; worst image discrepancy
is 1.3589069900982896e-16. Active bottom and horizontal-memory norms are
35.446145800543846 and 167.40932716949976, while top y memories remain zero.

The independent 1-D normal P packet picks direct/reflected vy at 0.0520 and
0.1352 s versus 0.051666666666666666 and 0.135 s. Both have the incident upward
velocity polarity (positive for positive incident tensile stress); vx is
identically zero. The fixture sign was corrected from a downward-mode
initialization using the momentum equation, with the timing gates unchanged.

Existing untouched regressions: M6.3c MPI/surface VJP 5 passed; canonical Linux
M9b oracle 18 passed; Windows M9b 17 passed/1 intentional fingerprint skip;
DENISE normal P, oblique P/P-SV and MPI reflection integrations 3 passed,
using a binary compiled in this worktree from the exact BASE. The new final
suite contains 44 tests. Production sources and existing tests are unchanged.
