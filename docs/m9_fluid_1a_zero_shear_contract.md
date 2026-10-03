# M9-FLUID-1A frozen zero-shear oracle / future acceptance contract

BASE `13ed47dc862bffa44d4020529a5ac68c987a6964`; branch
`codex/m9-fluid-1a-oracles`. Independent reference arithmetic is FP64;
classification is derived from the **copied canonical FP32 background**.
No production fluid forward, J, JT, MPI or CUDA support is claimed here.
All newly introduced tests are reference/contract checks; future production
adapters must provide **actual** production products to the frozen helpers.
Reference self-consistency alone is not production acceptance.

## Material and parameter manifold

Z={mu==0 exactly}; solid mu>0. No epsilon, Vs floor or caller-supplied mask.
Negative mu is invalid. Fluid needs finite positive rho and lambda (mu=0).
Valid finite compressional stiffness and lambda+2mu>0 are required wherever
the active operator needs them. This is not permission for unrelated universal
validator hardening or changing accepted all-solid scientific configurations.
The reference explicitly retains a valid negative-lambda solid case.

Cell (j,i) contributes with (j,i+1),(j+1,i),(j+1,i+1), modulo NY/NX,
to its positive-x/positive-y shear corner. Inverse face density is
2/(rho00+rho10) and 2/(rho00+rho01). If all four mu>0, H is exactly
4/(1/m00+1/m10+1/m01+1/m11), with the old solid arithmetic ordering.
Any zero contributor gives **H=+0, dH=+0**. Never evaluate 1/0 or 0/0.
Tests enumerate all 16 occupancy patterns (0/1/2/3/4 zeros), isolated fluid,
strip, horizontal/vertical interface, pocket and each wrapped edge/corner.

Classification is immutable context state. Allowed perturbations satisfy
dMu[Z]=0; dLambda is allowed everywhere. A nonzero fluid dMu must be rejected
**before any output mutation**, even one cell. Nonlinear trials must preserve
the copied FP32 classification in both directions: fluid->solid and
solid->fluid reject. New classification requires a new prepared context.
No production J/JT fluid path may run until FLUID-2: **fail closed** even if
fluid forward works. Forward support is not migration support.

## Homogeneous acoustic authority and forward gate

NX=24, NY=20, DH=1, DT=0.1, NT=120, rho=1, lambda=4, mu=0, cp=2.
The independent authority is a separate pressure/velocity acoustic FD4
calculation, not a call to elastic production with mu=0. FD4 coefficients
9/8,-1/24; wrapped derivatives. Explosive source at zero-based (12,10),
equivalent one-based (13,11); receivers (15,10),(12,13),(10,8),(1,1),
all zero-based, direct vx/vy after velocity update.

Ricker fc=0.5, t0=3, amplitude 1, integer stress samples, one cumulative
integration then prescribed centered source difference, including both endpoint
rules; stress insertion after stress update is equal in both normal channels.
This is the existing prepared-source convention, independently written here.
No weighting, data processing, quadrature factors or normalization is introduced.

Finite nontrivial vx/vy, H=+0, zero shear increments and reachable sxy=0
from zero initial state are mandatory. Normal bulk increments are equal.
Independent elastic-constitutive study vs separate acoustic authority relative
data L2 <=5e-13; future production forward data relative L2 <=1e-5.
`require_homogeneous_products` accepts real data/corner/shear products; fake
zero data, nonzero H or shear fail. Validator relaxation alone cannot pass.
Plane (1,1) acoustic mode on 24x20 independently uses
K(k)=2/h*(9/8*sin(kh/2)-1/24*sin(3kh/2)),
sin(omega*DT/2)=cp*DT/2*sqrt(Kx^2+Ky^2).
Phase-speed relative departure from cp <=0.01; FP64 phase evolution relative
L2 <=5e-13. Neither frequency nor threshold is fitted to a computed wave.

## Physical interface gate and classification

See [the complete independent interface study](m9_fluid_1a_interface_investigation.md)
for explicit pressure/particle-speed conventions, normal impedance algebra,
oblique traction/slip equations, geometry, windows, normalization and refinement.
The authority is an independent continuum three-equation plane-wave solve;
the independently written zero-corner FD4 scheme is the subject of the study.
Both incident/reflected fluid P and transmitted solid P/SV are observed.
No welded tangential velocity condition is imposed. Normal flux conservation
and a nonzero tangential velocity jump independently check the reference.

The source is a coherent plane mode: no point-source spreading. A homogeneous
control removes the prepared source spectrum. Future production comparison
must supply equivalent coherent-source products (or an independently justified
Green-function/plane-mode projection of actual production shots), **not** compare
an unnormalized point-source peak to a plane coefficient. An adapter must
verify its aperture/outer-boundary/window contamination independently; Python
reconstruction cannot substitute for production numerical/halo behavior.

Fixed physical geometry/frequency; h=1,0.5,0.25,0.125. Finest complex relative
coefficient error <=0.02 and each halving <=max(0.8*previous error,0.002).
The 0.02 threshold, coefficients and directions are frozen before production
output. The extra window residual must be <0.0005. `require_interface_refinement`
is the unchanged future acceptance helper for actual production measurements.
Coarse-grid exact continuum equality and fourth-order discontinuity convergence
are not claimed. This is bounded horizontal propagating-wave evidence, not
general validation of curved interfaces, Scholte waves or arbitrary contrasts.

If independent evidence requires interface-specific derivatives, one-sided
operators, explicit acoustic-elastic coupling or another materially new spatial
discretization, **HOLD — INTERFACE REQUIRES CLASSIFICATION C**. Do not weaken
the physical gate. The documented Ricker-window measurement correction is
an oracle measurement repair before production comparison, not evidence of C.

## FREE_SURF=1 and CPML

Fluid surface: D=lambda, alpha=lambda/D=1, A=4mu(lambda+mu)/D=0.
For dMu=0, delta_alpha=delta_A=0 for all dLambda. Thus the M9d3 closure's
top-row raw gLambda in fluid is structurally **+0**: neither the affine vy
ghost nor direct surface sxx coefficient depends on fluid lambda. This does
not zero interior fluid lambda sensitivity or fluid strain operands.
Preserve source rejection at first physical surface row, incoming syy
projection, odd stress mirrors syy[-m]=-syy[m], sxy[-m]=-sxy[m-1],
cubic zero surface shear traction, and four-strain order VXX,VYX,VXY,VYY.
Surface vy[-m]=vy[m-1]+(2m-1)*(DH/DT)*alpha*Qxx[0]; the unchanged
vx mirror uses the (35,-35,21,-5)/16 extrapolation of Qyx. Direct A update
is used rather than numerical cancellation of a bulk update. No new state
or production J/JT is added. Flat top closure only, as existing M9d3.

Scalar CPML authority: p'=b*p+a*q, q'=q/K+p'. Transpose with
t=bar(q')+bar(p'): bar(q)=bar(q')/K+a*t, bar(p)=b*t.
Test both against an explicit independent 2x2 matrix and <=5e-13 operand-
scale absolute dot residual, not a cancellation-sensitive quotient.
Polynomial profiles: power=2, R=1e-3, Kmax=1, fpml=0.1,
d0=-3*vmax*ln(R)/(2*FW*DH); fixed copied background coefficient operator.

Predeclared absorbing studies: 64x64, DH=1, DT=0.1, NT=800, FW=10,
same prepared Ricker convention, fc=0.1,t0=15, source zero-based (32,20).
Homogeneous fluid and horizontal fluid-top/solid-bottom (same interface
material as the physical study), each with FREE_SURF=0/1. Require finite,
nontrivial data/state, advancing active CPML memories and **maximum diagnostic
norm in final 25% / maximum over whole run <=0.05**. Diagnostic norm is
sum[rho*(vx^2+vy^2)+(sxx^2+syy^2+2sxy^2)/(lambda+2mu)]. It is a declared
positive absorption/stability observable, not an exact solid elastic energy
or a claim of continuum-perfect reflectionlessness. Boundary source-free late
decay rather than source-normalized interface amplitude defines this gate.
All-fluid covers side/bottom/top overlap; interface crosses side CPML.
FREE_SURF=1 disables top y CPML (K=1,a=0,b=1 and zero y memories) while
retaining side/bottom. Future production must additionally compare actual
data to this independent reference, rather than use late decay alone.

## Frozen FLUID-2 J/JT matrix (not production support yet)

Directions: fluid-lambda-only; solid-lambda-only; solid-mu-only;
joint lambda/solid-mu; interface-localized lambda/solid-mu. Each combines
vx-only, vy-only and both data cotangents. Invalid single-fluid-cell dMu
rejection precedes output writes. Trial +/- epsilon*dMu keeps fluid mu zero.

JT returns raw Euclidean physical lambda/mu images, same sign contract
<Jv,d>=<v,JT d>, no DT/DH-squared factor, preprocessing or image weighting.
Raw gMu[Z] is exact **+0**. Mixed corners contribute no harmonic transpose
even to adjacent solids. Adjacent solid mu retains direct normal constitutive
sensitivity, applicable surface contributions and incident all-solid corners;
do NOT project those solid images to zero. Unit-basis material Jacobian,
explicit transpose and corner ADD/scatter include wrapped physical contributors.

Retain all existing gates: local FP64 absolute dot <=5e-13*operand scale;
production J relative L2 <=1e-5; raw JT relative L2 <=6e-5. Inherit existing
`_production_dot_metrics` / `_assert_production_dot_closes`, including absolute
residual <= (1e-5*||Jref v||*||d|| + oracle absolute bound +
6e-5*||v||*||JTref d||), operand-scale diagnostics and separate J/JT accuracy.
Do not substitute a dot quotient. No constants in existing gates change.

Full-operator dense centered FD: **epsilon=0.05**, relative ceiling **0.004**
on the corresponding established fixture class. Enumerate every lambda column
and only solid mu columns; fluid mu is not a parameter column. Classification,
rho, CPML profiles and acquisition remain fixed; nonlinear +/- evaluations
do not recalculate CPML damping speed. The new local material dense-FD check
uses this epsilon/ceiling but is not a substitute for the later wave-operator
dense FD. Never select epsilon after production results.

## Future execution matrix

- CPU MPI: x and y decomposition boundaries crossing interfaces, and mixed
  corner/decomposition combinations. Same physical four owners through actual
  production halos/ADD transpose; no global Python material reconstruction.
- FULL and replay: segments 1,3 and uneven schedule. Immutable classification
  is prepared context state, not a new checkpoint payload. Preserve existing
  five fields/eight memories/checkpoint boundary; regenerate actual four
  nonzero fluid strains rather than zeroing them because raw gMu is zero.
- Single GPU: zero-safe host maps, forward fluid, mixed corners, corner J,
  reverse harmonic, raw-fluid-gMu +0 projection, free surface, CPML, FULL J/JT,
  replay JT, CPU/CUDA data/raw-image comparison and all-solid regressions.
  No unsupported replay J or nonlinear replay API is required. M9e-5 is
  untouched and not an evidence dependency.

## Existing solid authority / provenance

Keep existing M9b/M9d scientific gates and their exact files unchanged. The
canonical Linux/x86_64/NumPy 2.5.3/FP64 six fingerprints are recorded, unchanged,
in `test_m9b_elastic_psv_born_oracle.py` and `m9b0_elastic_psv_born_oracle.md`:
shot1/shot2 lambda f104a285.../afde7d4e...; shot1/shot2 mu
33d6546c.../5138b9f0...; raw lambda c03d496e...; raw mu 9135afe8....
The complete existing values, not these abbreviated labels, remain authority.
Noncanonical runtime skips only exact publication fingerprints; all scientific
tests still execute. M9d3 local/multistep transpose, material derivative,
boundary, source/receiver, CPML, decomposition and replay checks stay frozen.
No existing oracle values, thresholds, production, include or parameter files
are edited, and no publication authority is exercised.

## Independent observations (not new thresholds)

All occupancy patterns give exact +0 for 1/2/3/4 zero contributors; the solid
formula identity is exact. Acoustic data norm is 4.0231159910, and independent
constitutive/acoustic relative data difference is 8.2268493194e-16. The local
dense material FD (54 allowed columns) relative error is 0.000177851235.
CPML late diagnostic norm ratios for homogeneous FS=0/1 and interface FS=0/1
are respectively 2.9020176478e-7, 0.002431973157, 6.1672283872e-7,
0.02303930978; all are below the unchanged 0.05 ceiling. Active memories
are nonzero, top y memories with FS=1 are exactly zero, and fluid strains
remain nonzero. See the companion interface study for refinement evidence.
