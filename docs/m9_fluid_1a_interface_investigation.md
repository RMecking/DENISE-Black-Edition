# M9-FLUID-1A independent physical interface investigation

BASE: `13ed47dc862bffa44d4020529a5ac68c987a6964`. No production
implementation/support is asserted. The classification-C stop gate is evaluated
before proceeding to freeze the full forward/Born acceptance package.

## Predeclared experiment and gate

The horizontal interface is y=0 (positive y downward). Fluid: rho=1, cp=2,
lambda=4, mu=0. Solid: rho=1.5, cp=3, cs=1.5, lambda=6.75, mu=3.375.
Frequency 0.25, fluid dominant wavelength 8. Angles 0 and 20 degrees from
the downward normal. Unit incident pressure; tensile-positive stress=-pressure.
P velocity polarization `(sin(alpha), cos(alpha))`; transmitted SV polarization
`(cos(beta), -sin(beta))`. Snell slowness is fixed. Solve normal traction and
normal velocity continuity and zero solid shear traction, **not** tangential
velocity continuity. At normal incidence Z=rho*cp: pressure R=(Zs-Zf)/(Zs+Zf),
pressure T=2Zs/(Zs+Zf), reflected vy/incident vy=-R, transmitted vy/incident
vy=2Zf/(Zs+Zf). The oblique three-equation solve also independently conserves
normal energy flux.

The investigated scheme is independently written FD4 velocity/stress algebra
with density-face arithmetic averaging, H=0 at mixed corners, unchanged
9/8,-1/24 derivatives. It is not the physical authority or a production call.
A complex horizontal Fourier mode is an exact reduction of the translation-
invariant 2-D horizontal-interface grid, not a pseudospectral x derivative:
its x symbol is the actual FD4 symbol at fixed physical kx. Vertical derivatives
remain explicit FD4. Thus it probes slip and P/SV mode conversion without
point-source geometric spreading.

Domain y=[-240,240], isotropic stress source nearest y=-64 after stress update,
Narrow-band pulse exp(-0.5*((t-40)/8)^2)*cos(2*pi*0.25*(t-40)),
sampling dt=0.04*h. Observe reflected pressure near
y=-32 and transmitted vx/vy near y=32. Fourier arrival window [0,140], covering
incident/reflected P, transmitted P and SV pulses; outer-wrap returns arrive
later. A homogeneous fluid control divides out source spectrum. Correct bulk
discrete propagation phase analytically using the FD4/leapfrog dispersion root;
do not fit phase or amplitude to production. Retain actual staggered locations
and half-step velocity times. Compare to independent continuum coefficients.

Refine h=(1,0.5,0.25,0.125), fixed geometry and frequency, hence 8/16/32/64
fluid points per wavelength. Predeclared before execution: **each nonzero
mode coefficient complex relative error <=0.02 on the finest grid**; SV zero
at normal incidence uses incident particle-speed scale 1/Zf. At each halving,
maximum mode error must be <=max(0.8*previous error,0.002). This modest
first-order-interface convergence gate does not claim fourth-order accuracy
at a material discontinuity or exact coefficients on a coarse 48-square grid.
No observed production output enters these thresholds.

### Measurement correction before production comparison

The initial Ricker (t0=16) probe had substantial spectrum near the vertical
propagation cutoff f=0.25*sin(20 degrees). Its slow-group-velocity coda was
not captured by the finite Fourier window: at h=0.25, expanding window/domain
from 140/240 to 240/480 changed maximum error from 0.04382 to 0.01157 without
any stencil change. This disproves an interface-classification conclusion
from that finite-window result. The narrow-band prepared pulse above suppresses
the cutoff spectrum analytically by its Gaussian frequency envelope. The
Ricker probe remains available as pulse="ricker" for reproducibility. Physical
reference, material, angles, refinement sequence and 2% threshold are unchanged;
no production output was inspected. Verify window invariance on the corrected
probe separately before interpreting refinement.

If these checks show a persistent interface discrepancy requiring changed
spatial interface treatment, STOP/HOLD classification C; do not supply a
weakened FLUID-1B oracle. This diagnostic is not a replacement implementation.

## Primary-source context

[Okamoto & Takenaka (2005)](https://doi.org/10.4294/zisin1948.57.3_355)
analyse the staggered fluid-solid interface and warn about high-order stencils
crossing discontinuities. [Martire et al., Appendix A (2022)](
https://doi.org/10.1093/gji/ggab308) provide an independent plane-wave
reflection/transmission context. The explicit three-equation slip solve here is
derived from velocity/stress plane waves, rather than copied coefficients.

## Independent observations and bounded classification

The narrow-pulse maximum mode-relative errors, in h order, are:

| h | normal incidence | 20-degree incidence |
|---|---:|---:|
| 1 | 0.0185635232 | 0.0624048876 |
| 0.5 | 0.00364412382 | 0.0340034360 |
| 0.25 | 0.000845938611 | 0.0174745319 |
| 0.125 | 0.000207337104 | 0.00879966685 |

The physical reference amplitudes (pressure R, P speed, SV speed) are
normal `(0.3846153846,0.3076923077,0)` and oblique
`(0.36029714,0.30407388,-0.15419740)`. The finest oblique measured
amplitudes are `(0.36034974-0.00038885i,0.30409128+0.00028737i,
-0.15410269+0.00135358i)`. All declared convergence/accuracy gates pass.
Extending the h=0.25 pulse window/domain to 180/360 changes normalized
coefficients by at most `1.221061962550866e-8`, below the separate 0.0005 gate.

This supports **classification B for this bounded horizontal propagating-wave
experiment**: zero-corner FD4 exhibits the required refinement without new
interface stencils. It is not classification A (the operator domain changes),
not a universal interface-accuracy claim, and not production fluid acceptance.
The classification-C stop condition remains binding for contradictory later
independent evidence. No gate or coefficient was weakened to reach this result.
