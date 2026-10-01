# M9e-2: isolated elastic CUDA free surface and FULL Born J

Base: `ae12ed5b35c9614189a4752f7cd1ee17cc57aaff`. This internal backend is
single rank, single NVIDIA GPU, L=0, FD4, fixed rho, raw lambda/mu, FULL corrected
background operands. It does not route MODE=2. There is no JT, migration image,
replay/checkpoint, MPI multi-GPU, viscoelasticity, density derivative, Vp/Vs/Q
mapping, observed-data residual, filtering, scaling, or conditioning here.

The existing `denise_cuda_m9_create` retains its M9e-1 FREE_SURF=0 rejection
contract and original device budget. Additive `denise_cuda_m9_create_full`
accepts FREE_SURF=0/1 and owns the separate tangent arena slices. Both use the
same isolated M9 backend, canonical CPU bridge, optional Make targets, and flat
width-two halo layout. Normal CPU DENISE and M8e numerics remain independent.

## Surface graph and arithmetic

Physical surface y=DH/2, zero-based top row 0:

1. Project top syy=0; stress COPY X, then COPY Y.
2. Mirror syy[-m]=-syy[m], sxy[-m]=-sxy[m-1], m=1,2.
3. Momentum/CPML; chronological receiver sample.
4. Velocity COPY X, then COPY Y.
5. Correct horizontal Qxx/Qyx once through their CPML recurrence.
6. Derive material-dependent top velocity ghosts from these corrected operands.
7. Correct vertical Qxy/Qyy; update volume normal stresses away from row 0 and
   shear stresses everywhere; apply separate top sxx closure; insert prepared
   explosive stress increments; complete timestep.

For D=lambda+2mu, alpha=lambda/D and A=4mu(lambda+mu)/D. CPU canonical double
intermediates are retained and alpha/A are rounded to FP32 before forward use.
Top sxx += A*Qxx, syy=0; there is no additional bulk-normal top increment.
The cubic Qyx extrapolation is the sequential FP32 sum of weights
(35,-35,21,-5)/16 over the four top rows. For m=1,2:

```
vx[-m] = vx[m] + (2*m*h)*sum(weights*Qyx)
vy[-m] = vy[m-1] + ((2*m-1)*h)*(alpha*Qxx[0])
h = 1.0f / coefficient; coefficient = rounded_FP32(DT/DH)
```

The standard build explicitly rounds the new surface FD, CPML, momentum,
ghost and constitutive FP32 nodes using round-to-nearest CUDA intrinsics.
This preserves canonical CPU association in long, cancellation-sensitive
reflections; it does not change precision or enable fast math. FREE_SURF=0
retains the existing M9e-1 arithmetic and FMA behavior. No-FMA compilation
still applies `-O2 --fmad=false` to the whole isolated backend. All state,
material maps, CPML memories, operands, source and output data remain FP32.
The h discriminator uses rounded DT=0.000318, DH=11.7: canonical h=36792.44921875,
while rounded DH/DT=36792.453125.

Profiles come from the unchanged canonical CPU bridge. Top-Y identity is
K=1,a=0,b=1; horizontal CPML remains active. Canonical inactive horizontal
b=exp(-pi*FPML*DT) is retained where specified by FREE_SURF=1. There is no
second Qxx/Qyx correction, top damping recurrence, or psi_vxxs history.

Source j=1 (physical one-based) rejects before propagation; j=2 is supported.
Receivers may occupy physical row 1 and duplicate/scrambled locations. Outputs
are `[time][receiver][vx,vy]`, sampled after momentum before velocity halos.
Canonical stronger grid constraints require NX,NY>=5; CPML layers must also
satisfy the unchanged CPU allocation/overlap constraints.

## Prepared background and J

```
create_full(config, options, &context)
prepare(context)                       // FULL Qxx,Qyx,Qxy,Qyy for every t
apply_j(context, delta_lambda, delta_mu, cells)
born_download(context, data, optional_fields, optional_psi, optional_q, optional_corner)
apply_j(context, another_lambda, another_mu, cells)
destroy(&context)
```

J requires an explicitly prepared valid FULL context, nonnull FP32 direction
arrays of exactly NX*NY cells, and finite directions. Pointers must identify
readable host arrays; the ABI cannot validate arbitrary foreign address ranges.
J never reruns background propagation. It resets its own five physical fields,
eight CPML memories, current four operands and chronological data. Background
source is not differentiated, and no tangent source is inserted. Fixed rho,
profiles and receiver geometry are shared read-only. Background FULL operands
are unchanged bitwise across repeated directions.

Bulk direct normal increments are delta_lambda*(Qxx+Qyy)+2*delta_mu*Qxx/Qyy.
Shear adds delta_corner*(Qyx+Qxy). Each corner tangent is evaluated with the
rounded FP32 background harmonic corner widened to double, in the exact
canonical CPU four-contributor evaluation order:

```
delta_corner = FP32(0.25*double(corner)^2 * sum(delta_mu/double(mu)^2))
```

Surface material derivatives retain double intermediates:
alpha_lambda=2mu/D^2, alpha_mu=-2lambda/D^2;
A_lambda=4mu^2/D^2,
A_mu=4(lambda^2+2lambda*mu+2mu^2)/D^2.
The tangent vy ghost adds rounded_FP32(delta_alpha)*background_Qxx to
alpha*tangent_Qxx before the canonical geometric factor. The top sxx direct
term is rounded_FP32(delta_A)*background_Qxx, alongside A*tangent_Qxx from
the homogeneous surface graph. Top syy remains zero; no bulk-normal direct
term is duplicated there. Qyx cubic extension propagates tangent operands
without an invented material derivative.

## Ownership, workspace and failure lifecycle

Let N=NX*NY, P=(NX+4)*(NY+4), T=NT, R=receiver_count. Checked mandatory device
bytes for FULL-J are:

```
124P + 24(NX+NY) + 4T + 12R + 16TR + 16TN + 28N
```

The inherited forward allocation is 72P+24(NX+NY)+4T+12R+8TR+16TN.
The additive J workspace is 52P+28N+8TR: five tangent fields 20P, eight tangent
memories 32P, two directions 8N, harmonic tangent 4N, four current tangent
operands 16N, and Born data 8TR. Only one cudaMalloc owns all slices. Budget,
overflow, configured cap and reserve are checked before device allocation.
Diagnostics account every category and actual ownership.

Surface-specific extra device bytes: **0**. Existing padded field halos hold
derived stress/velocity ghosts between dependency stages and are overwritten
before reuse each timestep. Current/background Qxx/Qyx are used directly;
there are no persistent alpha/A arrays, extrapolation copies or new physical
surface history. These halos are workspace, not future checkpoint content.

Optional transactional downloads allocate tracked host scratch only: at most
8TR+52N+16TN bytes for all background outputs and 8TR+72N bytes for all J
outputs. Test surface downloads use 20P host scratch. Scratch exists only for
the call and is freed before return; it adds zero VRAM. Host canonical bridge
preparation and metadata are tracked separately. No diagnostic cudaMalloc.

Every new CUDA operation/launch/completion and tracked allocation is injectable.
The first meaningful error is retained. A failed J or transfer invalidates
both prepared/background result state and J-valid state. Output publication
is transactional: caller arrays retain their sentinels on failed download.
Cleanup is not injected; real cleanup errors retain the still-owned context
for retry. Successful destruction yields zero device/host/event ownership.

## Local verification

Run the additive `tests/physics/test_m9e2_cuda_free_surface_born.py`; it compiles
both optional libraries and a separate strict-C CPU view. That view captures
final J arrays at destruction from the unchanged public canonical CPU J;
it does not copy its numerical loops. Frozen M9b/M9d3 Python oracles and all
existing test equations/thresholds remain unchanged. Test-only initial-state
evolution and pure projection/mirror/ghost/closure views use actual kernels.

No-FMA comparisons require byte identity for background data, five fields,
eight CPML memories, four FULL operands and comparable J data/state/operands/
corner. Standard-build background bounds remain L2<=2e-6, peak<=8e-6;
J receiver bounds remain L2<=1e-5, peak<=4e-5. GPU centered-FD epsilon=0.05
and relative error<=0.004 are frozen. Independent FD ladders, complete state
and material basis matrices, physical reflection controls, profiles, repeatability,
memory and fault sweeps complete the evidence. Timing is diagnostic only.

Host ASan/UBSan use the actual allocator and canonical surface bridge; CUDA
host lifecycle UBSan runs real FULL-J contexts and injected failures. If WSL's
WDDM debugger blocks Compute-Sanitizer, report **ENVIRONMENT / DRIVER LIMITATION**,
never PASS. No cluster, installation, configuration or publication is needed.
