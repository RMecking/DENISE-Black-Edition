# M9b-0 — independent elastic P/SV Born / Born-adjoint oracle

Status: frozen independent reference for review; no elastic production-gradient
interface is asserted by this artifact. The executable production integration
gate remains the explicit M9b-1 RED frontier in
`tests/physics/test_m9b_elastic_psv_born_oracle.py`.

Reference provenance: base `modernization@8b7947ef0189a5e8289a843daa2701e0578f99d4`.
The NumPy implementation in `tests/utilities/elastic_psv_born_reference.py`
reimplements the discrete equations below; it does not call DENISE numerical
kernels. Its state, model, and CPML coefficient arrays and arithmetic are
FP64; this is the independent double-precision oracle, not a claim of
bitwise-equivalent rounding to the repository's float production kernels.
Production comparison tolerances must be separately justified. The frozen
tests are in `tests/physics/test_m9b_elastic_psv_born_oracle.py`.

## Publication-baseline provenance

The six output hashes are **publication baseline fingerprints established
after independent scientific review of the exact M9b-0 candidate**. They are
not described as pre-registered, pre-publication frozen, or historically
frozen: these three candidate files remain untracked, so the forthcoming
publication commit will establish their immutable repository genealogy.

The scientific oracle is defined by its independently derived equations,
nonlinear derivative checks, dot-product identities, falsification gates, and
unchanged thresholds. The hashes are deterministic reproducibility
fingerprints of that reviewed implementation, not the sole scientific
authority.

Exact fingerprint checking is a publication reproducibility contract, separate
from scientific acceptance. Its canonical runtime is Linux on `x86_64`/`AMD64`
with NumPy `2.5.3` and IEEE-754 FP64. CPython is intentionally not pinned to
one minor version: the fingerprints were generated twice in fresh processes
under each of these canonical Linux/WSL2 x86-64 FP64 runtimes at candidate base
`8b7947ef0189a5e8289a843daa2701e0578f99d4`:

- CPython 3.13.15, NumPy 2.5.3;
- CPython 3.14.7, NumPy 2.5.3.

All four runs produced the same complete values:

| Field | SHA-256 of canonical little-endian FP64 payload |
|---|---|
| shot 1 lambda | `f104a2850d1d62d9c28619e568bbdd7973164d95d832fcf9c0535358ce48a3ec` |
| shot 2 lambda | `afde7d4e5972dd8d89af25b10b45877c296d9ce619366e6fef279ccc85da3a4c` |
| shot 1 mu | `33d6546cc08511df16c92775618047add117d0f8a087f0b8ce8f59050b66954a` |
| shot 2 mu | `5138b9f06d3225701dbf71dd451a94492c47f496eb7f394458ec5672e78daef7` |
| `image_lambda_raw` | `c03d496e2dfc60f181387d7c8ef7ecfcfc31114cef6d9ce5466b69f7f45d1270` |
| `image_mu_raw` | `9135afe8873e71eb8b16dd3ddd148b845f87f46d82f5e4bf46292b64376829db` |

Windows CPython 3.12.10 / NumPy 2.5.3 produces the observed, scientifically
equivalent but byte-different family `4a63e971...`, `68b0cc7e...`,
`e3e6d75e...`, `a93fc02f...`, `611e9eef...`, and `0e59973b...` in the table's
field order. It is provenance, not a second universal baseline. Non-canonical
runtimes (including Windows, other NumPy versions, architectures, and macOS)
run every scientific numerical gate and within-runtime repeatability check,
but explicitly skip the canonical exact-hash assertion. A new fingerprint
family is never adopted automatically; it requires a separate reviewed
provenance update.

## Discrete state and material convention

The parameter channels are the physical cell-centered Lamé parameters
`lambda` and `mu`, in Pa. Density `rho` (kg/m³) is fixed. The base state uses
`rho=2000 kg/m³`, `Vp=3000 m/s`, `Vs=1700 m/s`, hence
`mu=rho Vs²` and `lambda=rho(Vp²-2 Vs²)`. DENISE's elastic P/SV reader loads
these as `ppi=lambda` and `pu=mu` for `INVMAT1=3`
(`src/PSV/readmod_elastic_PSV.c`). No velocity/Q parameter conversion,
preconditioning, model mask, or density perturbation is part of this operator.
Stress and Lamé parameters have units Pa; density has kg/m³; velocity and
sampled data have m/s. The corrected `VXX,VYX,VXY,VYY` operands are
dimensionless strain increments. Consequently the raw data-space product has
units `(m/s)²`, while the model image has the matching cotangent units
`(m/s)²/Pa`. Omitting `DH²` and `DT` intentionally makes these coordinate-array
transpose products, not volume/time quadrature approximations; changing mesh
or sample rates does not silently change the frozen inner product definition.

Arrays are y-by-x; test fixture coordinates are documented in the one-based
DENISE convention. The staggered-grid FD4 coefficients are `9/8` and `-1/24`.
Forward derivatives use the source/target staggering in
`src/PSV/update_v_PML_PSV.c` and `src/PSV/update_s_elastic_PML_PSV.c`; the
reference spells these as explicit periodic shifts and derives their scatter
transposes term by term.

The first frozen envelope is 2-D isotropic elastic P/SV with `L=0`,
`FDORDER=4`, `NDT=1`, `DTINV=1`, `FREE_SURF=0`, and `BOUNDARY=0`. At each time
step, the reference applies the same leapfrog order:

1. Update half-step `vx,vy` from stress derivatives, face reciprocal density,
   and CPML-corrected derivatives.
2. The receiver operator samples direct `vx,vy` at receiver nodes after this
   velocity update. Production calls `seismo_ssg` later in the loop, after the
   stress update/source insertion, but those operations do not modify velocity;
   this is exactly the direct-array `SEISMO=1` behavior in `src/seismo_ssg.c`.
3. Form the four velocity-gradient operands and apply their CPML recurrences.
   In this document `VXX,VYX,VXY,VYY` denote those corrected, dimensionless
   strain increments (raw derivative times `DT/DH`, then CPML correction).
4. Update normal and shear stresses, then add the explosive source equally to
`sxx` and `syy` as `src/PSV/psv.c` and `src/psource.c` do.

Thus `data[k]` is the direct velocity gather for the first/next half-step
`(k+1/2) DT`; it is not a stress-time sample. The source is a stress source in
Pa: active case `QUELLART=1` Ricker, one time integration (`N_ORDER=1`) followed
by the production-style `psource` difference. Its center is `t0=0.1 s`
(`1.5/fc`, `fc=15 Hz`) and amplitude is `1e6 Pa`. The 24-step microcase uses
`t0=0.002 s`. In zero-based reference notation, let `A_n` be the integrated
wavelet at production's one-based source sample `n`, with `A_0=0` and each
integration step adding `DT*w_n`. The source increment injected at stress
step `n` is `A_2/DT` for `n=1`, `(A_{n+1}-A_{n-1})/DT` for `1<n<NT`, and
`-A_{NT-1}/DT` at `n=NT`, matching `src/psource.c` endpoint handling.
Source injection is fixed under model perturbations.

For `INVMAT1=3`, the shear modulus at each staggered shear point is the
four-cell harmonic mean, as implemented by `src/av_mue.c`:

```text
mu_corner = 4 / (1/mu00 + 1/mu10 + 1/mu01 + 1/mu11)
```

The reference differentiates this map analytically and applies its exact
transpose to the cell-centered `mu` image. Face density is fixed and uses the
two-cell arithmetic average followed by reciprocal, as in `src/av_rho.c`.

## CPML and edge closure

The active fixture uses `NX=64, NY=56, DH=10 m, DT=0.0005 s, NT=700, FW=8`.
It has a single-rank P/SV domain with active CPML on all four sides, no free
surface, and no material contrast in the background. The microcase is
`41x37`, `FW=0`, `NT=24` and isolates the interior operator from CPML.

For every derivative channel, the reference implements the profile at the
correct center/half-grid staggering and the recurrence

```text
psi_new = b * psi_old + a * q
q_corrected = q / kappa + psi_new
```

The coefficients follow `src/PML_pro.c`: quadratic damping profile, reflection
coefficient `R=0.001`, `DAMPING=Vmax`, `kmax=1`, and `FPML=15 Hz`; the reference
uses FP64 coefficient arrays. The reverse recurrence differentiates both the
corrected derivative and persistent `psi` state. All eight velocity/stress
CPML memory channels participate in the reverse time step.

Single-rank halo closure is periodic, not zero padding or edge replication:
`src/initproc.c` assigns each edge rank's neighbor to the opposite rank. The
reference wraps edge values for FD stencils, face density, and the four-cell
harmonic shear map, and transposes that wrapping exactly. This closure is part
of this frozen fixture and is not a claim about physical periodic boundary
conditions outside the fixture.

The active time step is below the independently computed leapfrog bound. The
diagnostic uses the maximum staggered FD4 derivative symbol and
`omega_bound=Vmax*sqrt(2)*symbol_max/DH`, with `DT_limit=2/omega_bound`; the
oracle requires `DT/DT_limit < 1`.

## Born operator `J` and transpose `J^T`

Let the four stored base-state operands at stress step `k` be
`VXX[k], VYX[k], VXY[k], VYY[k]`. For cell-centered perturbations
`(delta_lambda, delta_mu)`, the direct material stress increments are

```text
delta_sxx += delta_lambda * (VXX + VYY) + 2 * delta_mu * VXX
delta_syy += delta_lambda * (VXX + VYY) + 2 * delta_mu * VYY
delta_sxy += delta_mu_corner * (VYX + VXY)
```

where `delta_mu_corner` is the analytic directional derivative of the same
four-cell harmonic average. The tangent state then follows the homogeneous
linearized velocity/stress/CPML recurrence, with fixed source, density, and
background coefficients. Receiver sampling produces
`J(delta_lambda,delta_mu)` in raw `vx,vy` data units (m/s).

`J^T` is a separate reverse-coded recurrence, not an autodiff trace or a
finite-difference quotient. It injects prepared data at the exact receiver,
component, and time sample; reverses the stress update and the four material
terms; reverses each CPML memory update; and scatters the FD4 derivatives with
their exact transposes. Its output arrays are named `image_lambda_raw` and
`image_mu_raw`.

The frozen adjoint contract is the raw Euclidean identity

```text
<J dm, d>_D = <dm_lambda, J^T d_lambda>_M
             + <dm_mu, J^T d_mu>_M
```

There is no `DT`, `DH²`, receiver/time normalization, component weighting,
shot normalization, sign flip, or FWI residual preprocessing in either inner
product. Accordingly these images are discrete raw transpose products, not a
claim that they are already continuum-volume gradients. Data `d` in the
adjoint test is explicitly prepared migration data, not `synthetic-observed`.

## Frozen experiments and acceptance

**Interior microcase.** Homogeneous stable elastic material, one source at
`(21,19)`, five direct-node receivers, all coordinates one-based. Dot checks
cover vx-only, vy-only, vx+vy and lambda-only, mu-only, and joint non-symmetric
model directions.

**Active-CPML case.** Two sources `(20,12)` and `(45,12)`; ten receivers at
`(12,12),(16,12),(20,12),(24,12),(28,12),(37,12),(41,12),(45,12),(49,12),
(53,12)`. The source pair and receiver set are symmetric around the center
between x=32 and x=33. A positive 2%-of-background lambda-only, one-cell-thick
reflector occupies one-based `y=34`, `x=16..49` with a symmetric cosine
envelope. The separate mu-only direction is a compact asymmetric 4-by-9 patch
at one-based `y=29..32`, `x=27..35`, with a deterministic nonuniform profile.
Additional deterministic, non-symmetric lambda/mu directions and vx/vy data
probe channel and transpose errors.

The acceptance assertions are deliberately distinct:

- **Born derivative:** central finite differences of the independently coded
  nonlinear forward map use epsilons `(1, 0.5, 0.25, 0.125)`. Each single and
  joint direction must decrease by at least `3.9x` per halving and finish
  below relative error `2e-5`. The quotient is only a check; it does not define
  `J` or `J^T`.
- **Adjoint identity:** interior and active-CPML tests cover each data component
  and both individual plus joint model channels. Dot products use compensated
  `math.fsum`. With unit roundoff `u=eps64/2`, the conservative envelope is
  `gamma_n = n*u/(1-n*u)`, `n=512*NT*NX*NY`, with absolute ceiling
  `8*gamma_n*max(||Jdm||||d||, ||dm||||J^T d||)`. Relative residual is
  normalized by this operand-norm scale, not by a potentially cancellation-
  small dot product. The measured pair condition is
  `kappa_pair=(||Jdm||||d||+||dm||||J^T d||)/max(|lhs|,|rhs|)`; the additional
  side-relative ceiling is `8*gamma_n*kappa_pair`. Both forms are asserted, as
  is a dimension-only relative ceiling below `2e-6`. The report includes each
  side's operand norms, both signed dot products, pair condition, residual,
  ceiling, and margin.
  The factor `512` is a conservative per-gridpoint/per-timestep operation
  envelope for the floating-point accumulation bound; it is not claimed to be
  an exactly counted FLOP total. Evaluation fails closed unless `n*u < 1`.
- **Active CPML:** both shot backgrounds must have nontrivial CPML-memory and
  layer wave-energy peaks and a peak layer-energy fraction above `0.5`; this
  prevents an inactive-PML test from masquerading as coverage.
- **Sampling/time/sign:** basis tests distinguish receiver, component, time,
  and sign. Separate active data checks require both one-sample time-shifted
  alternatives to differ materially from the correct alignment.
- **Channels / reflector:** the scientific normal-response test must peak
  within one cell of row 34,
  place more than 25% of lambda-image energy within two rows of the reflector,
  and have lateral-symmetry residual below `0.1`. A second check forms
  independent nonlinear central-difference data at epsilon `0.25`, then
  applies `J^T`; its complete two-channel image must agree with the analytic
  Born-data image within twice the measured data FD relative error (which must
  be below `5e-6`). The six canonical little-endian FP64/C-order SHA-256
  fingerprints are checked separately only in the canonical publication
  runtime.
- **Shot stack / reproducibility:** two-shot stacking must equal the ordered
  sum of single-shot raw images within a roundoff envelope; repeated gathers
  and both full images must be bitwise identical.

These numerical checks establish internal derivative/transpose consistency,
closure against a separately evaluated nonlinear FD response, and a frozen
normal-response fixture. They do not substitute for independent scientific
review of the equations or for production integration verification.

### Observed publication baseline after independent review

On the active-CPML fixture, the central-FD relative-error ladders for
epsilons `(1,0.5,0.25,0.125)` are:

```text
lambda: [4.1010105839607866e-5, 1.0252191809060108e-5,
         2.563026981360406e-6, 6.407552715516155e-7]
mu:     [8.455658764646848e-4, 2.1133832910802706e-4,
         5.283126191007636e-5, 1.3207607977318896e-5]
joint:  [8.409904021369265e-4, 2.10194030782529e-4,
         5.2545160532806956e-5, 1.3136080931864643e-5]
```

Across 18 dot cases (3 parameter directions by 3 data-component choices in
each of the two experiments), the maximum operand-scale relative residual is
`1.92719e-16`, and the maximum absolute residual is `6.93889e-17`. The
dimension-only relative ceilings range from `1.65564e-8` (interior) to
`1.14087e-6` (active CPML); the smallest observed absolute-ceiling/residual
margin is `8.59098e7`. The active-CPML-only maximum relative residual is
`9.108545822940204e-17`. Worst-relative-residual examples, including both dot
side norms and signed products, are:

| Case | lhs | rhs | abs residual | side norms `||Jdm||||d||`, `||dm||||J^T d||` | absolute ceiling |
|---|---:|---:|---:|---:|---:|
| Interior, lambda, vx+vy | -6.569903649078492e-2 | -6.569903649078498e-2 | 6.93889e-17 | 0.3600524, 0.2604076 | 5.96119e-9 |
| Active CPML, lambda, vy | -6.587772795086625e-5 | -6.587772795087007e-5 | 3.82181e-18 | 0.04195854, 0.02137756 | 4.78693e-8 |

The largest measured pair condition is `3263.449723640649`; the maximum
cancellation-aware pair-relative residual is `5.801372902328407e-14` and is
checked against the separately reported
`8*gamma_n*kappa_pair` ceiling. Repeating a complete dot-metric record gives
exact dictionary/value equality. A global image sign flip is rejected by more
than 100 absolute ceilings in every tested dot case.

Other active-case falsifications/diagnostics: the `-1/+1` strain shifts differ
from correct alignment by `0.05941781718355411/0.05934553167977237` of the
correct image norm, with shifted/correct norm ratios
`1.0026786520818187/0.997301134638389`. The lambda/mu swap relative data
separation is `3.4858525550535244`. The stacked normal lambda image peaks at
one-based `(i,j)=(33,34)`, puts `0.3611743807851241` of lambda-image energy
within two rows of the true strip, and has lateral-symmetry residual
`0.04157776763949436`. Raw image norms are `5.311744735976774e-15` (lambda)
and `6.5425155471439674e-15` (mu). The
two-shot stack additivity residuals are exactly `0.0` for both channels in the
reviewed deterministic accumulation. CPML memory-energy / layer-wave-energy /
maximum layer-energy-fraction peaks are
`(1171.8328872708441,1003.6354235694731,0.9300878001653744)` and
`(1171.8120844123166,957.0349383641894,0.9358038141859044)` for the two shots.
The discrete leapfrog ratio is `DT/DT_limit=0.2563262081801235`.

For the two-shot lambda reflector, central-FD Born data at `eps=0.25` have
relative error `2.5609504040270693e-6`. Applying the analytic transpose to those full
FD gathers gives a complete two-channel normal-image relative difference of
`2.53394728509814e-6` from the analytic-Born-gather image, below the unchanged
`5.1219008080541385e-6` acceptance ceiling. This scientific comparison is
independent of fingerprint identity; the complete FP64 gather/image publication
fingerprints are recorded above and checked separately only on the canonical
publication runtime.

## Production handoff and relation to legacy gradients

M9b-1 must connect the real elastic P/SV production forward background and
actual `delta_lambda` / `delta_mu` directional products to this contract, then
compare the resulting physical raw products to the frozen reference. It must
preserve the parameter directions, receiver sample, time alignment, sign,
unweighted raw Euclidean scales, CPML recurrence, and acceptance evidence
above. The four sufficient full-storage operands for the material VJP are the
post-CPML `VXX,VYX,VXY,VYY` terms for each stress update; no density direction
is present. A production implementation may recompute equivalent operands,
but must prove the same discrete values and ordering. An FWI objective gradient
is not directly interchangeable with this raw `J^T d` product.

The elastic stress material algebra in
`src/PSV/update_s_elastic_PML_PSV.c` has the same local Lamé structure as the
reference. Historical M7/M8 products are not accepted as this oracle: their
objective, residual sign/preprocessing, model parameterization, and any time or
grid scaling must be checked separately before comparison. M8's viscoelastic
replay illustrates the four corrected strain operands needed by an exact
material reverse, but its attenuation/rheology chain and objective convention
are outside this elastic fixed-density `J^T` contract. In the current M8 exact
viscoelastic implementation, the lambda-like native accumulation has the
same divergence-times-summed-normal-cotangent structure, and the shear native
term pairs the shear cotangent with `VYX+VXY`. Its normal-mu native expression
and signs are not the M9 physical-parameter expression as written; derive
the transformation and adjoint convention instead of transplanting it.
`FX,FY` feed a separate M8 density-gradient path and are not needed for this
M9 lambda/mu material VJP with fixed rho. M8's parameter/rheology map, receiver
residual injection (including its objective/sample exclusions), and physical
gradient scaling remain different and must not be assumed equivalent.

For M9d1 replay, the material VJP needs the four per-stress-step post-CPML
operands `VXX,VYX,VXY,VYY`, base cell-centered lambda/mu and staggered harmonic
mu coefficients, CPML coefficients/locations, and exact time/shot/receiver
metadata. With these operands stored, full background velocity/stress arrays
are not additionally needed to accumulate lambda/mu images; the forward
tangent or reverse dynamic states and CPML memories are still needed while
applying `J` or `J^T`. If the four operands are regenerated rather than stored,
checkpoint state must suffice to reproduce the nonlinear background exactly.
No production checkpoint scheme is implemented here.

No production files are changed by M9b-0. The RED frontier is intentional and
is not an xfail that can silently become green: it records the future adapter
and comparison required by M9b-1 while this oracle stays independently
executable and frozen.
