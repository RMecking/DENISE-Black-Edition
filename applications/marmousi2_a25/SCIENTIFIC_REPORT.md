# Frozen A2.5 scientific interpretation

The accepted Marmousi-II A2 RTM is a mathematically consistent raw adjoint
migration whose interpretation is limited by extreme interface sensitivity,
nonuniform/nonlocal operator response, aperture dependence, parameter coupling
and the historical discretization/bandwidth choice. Useful deep sensitivity
exists. Correct migration of the deep geological reflectors has not been
established. The conservative A2.5A verdict remains **INDETERMINATE / CONFOUNDED**.

## A2.5A — interpretation, not image repair

The raw water-bottom/first-solid sensitivity overwhelmingly dominates the JT
aggregate. Deeper sensitivity is nonzero and often odd/even coherent, while
disjoint aperture blocks can disagree strongly. Shot 68 is retained as the
documented norm/QC observation. The strong first-solid/OBC row remains preserved,
not removed or attributed to a unique cause.

At fixed original smooth2 density/background:
`g_Vp=2*rho*Vp*g_lambda`, `g_Vs=2*rho*Vs*(g_mu-2*g_lambda)`;
`g_lnVp=Vp*g_Vp`, `g_lnVs=Vs*g_Vs`. These are cotangent coordinate transforms,
not recovered information, reflectivity or final velocity updates. Physical
water has exactly `Vs=0`, `mu=0`: log-Vs is undefined there and its restricted
tangent is frozen; no epsilon, floor or repair mask is applied.

Derivatives/Laplacians use declared validity masks; diagnostic depth gain changes
visibility, not information. The first-solid-row-only spatial bandpass control
generates apparently deep coherent response. This severe interface-stripe
leakage is a first-class negative result: filtered structures cannot be shown
without their paired control/warning. Geological plausibility and subset
coherence alone do not establish reflector validity. Evaluation-only true-model
geometry never feeds conditioning.

`AUTHORITATIVE_RAW` evidence is immutable. `DERIVED_NUMERICAL` /
`DERIVED_DIAGNOSTIC` are deterministic reductions. Percentile normalization,
clipping, compression, raster colors and plotting are `DISPLAY_ONLY` and never
feed metrics. No final broadband RTM accuracy follows.

## A2.5B — discrete normal-operator columns

Accepted `H15=J15^T J15`: 900/900 applications, numerical content ID
`0ad22a66a152b9aad2fd89ee6fdb374228fec258bc23d9c797f7539b0e787721`.
Controlled `H5=J5^T J5`: 200/200 applications, numerical content ID
`5122f0654cd03dcb2921ccc84226849b07ae13ae30f0a5974b328579e8e7bb80`.
These are different source-dependent operators; H5 is not a conditioning,
re-display or replacement of H15. H15 anchors the A2 conclusions.

`H_log=A^T H_(lambda,mu) A`, with input
`dLambda=2*rho*Vp^2*dLnVp-4*rho*Vs^2*dLnVs`,
`dMu=2*rho*Vs^2*dLnVs`, and exact transpose
`gLnVp=2*rho*Vp^2*gLambda`,
`gLnVs=2*rho*Vs^2*(gMu-2*gLambda)`. Native tangent is FP32, output FP64;
mapping roundoff is recorded. No independent native-channel rescaling or
off-diagonal removal. Frozen Core Euclidean products add no DT, DH, area,
source or receiver weights.

Exact nine probes, coordinate normalization, CPML/boundary treatment and
predeclared windows are in `normal_operator_contract.json`. Columns are unit
discrete log-coordinate basis responses, not physical 100-percent anomalies.
Tier-2 central `vp_c2360` reuses Tier-1 exactly (no repeated 100-shot campaign).
The target is the fixed 200-m disk at injection; primary full-grid metrics
include CPML, with separately declared partitions. Secondary lobe uses the
4-connected half-height component inside the fixed 400-m disk. Windows were
frozen before results, never resized/recentered. R50/R90 are unweighted
grid-coordinate squared-norm radii about injection; full tied radius shell is
included. Labels are descriptive, not new numerical acceptance gates.

Verified J/JT/H behavior is consistent, but individual H columns can be strongly
nonlocal and coupled. Deep perturbations are not universally mapped to the
water bottom. Localization varies with position/depth; aperture dependence and
Vp/Vs cross-talk are substantial. Small relative interface tails can still be
large in absolute cotangent units; this does not prove that a single column
caused the RTM image. Sparse diagonal samples are not a full Hessian diagonal,
continuum Hessian, physical energy density, illumination or universal reflectivity
resolution map. A simple global scalar diagonal correction is not justified.
Randomized diag(H) is only a possible separately authorized future diagnostic.

## Mixed H15/H5 comparison

Full precision values and parent identities: `frequency_comparison.json`.

| Probe | H15 target fraction | H5 target fraction | H15 R90 (m) | H5 R90 (m) |
| --- | ---: | ---: | ---: | ---: |
| vp_c1560 | 0.309267506830992 | 0.443756791188564 | 2886.1739379323626 | 2674.023186137323 |
| vp_c2960 | 0.9341206263534154 | 0.7073661925874072 | 160 | 488.2622246293481 |

H5 improves target concentration for the moderate-depth probe yet remains
strongly nonlocal; it worsens deep-probe localization. Source construction was
reproducibly resolved: original QUELLART6 sample-0 spike, amplitude 1/tshift 0,
fifth-order causal frozen libcseife Butterworth; 15-Hz samples reproduced
bitwise before changing only the corner to 5 Hz. Historical FP32/FP64 casts,
zero recurrence state and prepared-source derivative convention are retained
in the externally hash-bound `source/gate_resolution.json`. No zero-phase or
invented source convention. Hashes/norms/amplitude/phase remain part of J, not
a unit-energy comparison.

Historical 15 Hz exceeds the conservative Taylor-FD4 advisory ~5.50625 Hz.
H5 is near/below that advisory, not automatically dispersion-free. Differences
include bandwidth and discretization sensitivity; a dispersion-only explanation
was not established. Accepted historical H15 is not invalidated.

## Runtime history and scope

Original ProcessPool/session interruptions, historical H5 exit195, failed
exact-single-probe identity gate, successful shot33 two-probe diagnostic and
final successful serial fresh-process continuation remain separate immutable
external records. **ROOT CAUSE UNRESOLVED**: no deterministic shot33/Core defect
was demonstrated; successful serial continuation does not identify the original
cause or prove pool stability. Numerical content identity is separate from
time-bearing bundle identity. The original HOLD/BLOCKED evidence is not erased.

No FWI iteration/acceptance, specific Hessian correction validation, FLUID-3/4,
CUDA physical-water acceptance, final broadband accuracy or scientific upgrade
is implied. See the short A3.0 handover in `README.md`.
