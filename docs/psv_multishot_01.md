# PSV-MULTISHOT-01: discrete objective (2026-11-21)

Written before production changes. BASE b5ba9be91111103e421b784be22dfb5c13de3f66,
branch codex/psv-multishot-01. Capability: exact five-parameter L=1 FD4
visco-PSV FWI with sequential shots and workflow frequency stages in denise
and denise_cuda; demonstration target B3 small Mini-Marmousi.

For stage k let H_k be the existing finite-trace apply_tdfilt with workflow
ORDER, FC_END (lowpass), FC_START (optional highpass). With TIME_FILT=0 H=I.
For source q_s and observed velocity traces d_s, use

    J_k(m) = (1/2) sum_s sum_{component=x,y,receiver,t=2..NT}
                     [F_s(m; H_k q_s) - H_k d_s]^2.

The model-independent filter acts only on source samples and observations.
Neither modeled traces nor residuals receive a second filter. The adjoint
source is exactly the existing residual, with production sample one omitted;
no data-filter transpose belongs in this model derivative. We do not claim
H_k F_s(m;q_s) = F_s(m;H_k q_s) for finite traces. This is the smallest change
to the existing Köhn workflow and leaves all wave recurrences unchanged.

Shots run in increasing source-file order 1..NSRC within NCOLORS=1; CPU MPI
partitions space as before. NSRC>1 requires RUN_MULTIPLE_SHOTS=1. The five
per-shot FP64 adjoints are mapped and rounded to FP32, then accumulated in
FP32 in that order (CUDA retains direct assignment on the first shot to
preserve its signed-zero association). Each shot objective accumulates FP64 receiver contributions and uses the
existing spatial SHOT_COMM reduction; these global shot objectives are then
added in ascending shot order, in both the gradient and trial-forward paths.
All shots, including those outside TESTSHOT_*, enter objective and gradient.
Parallel shot colors remain fail-closed.

Acceptance criteria are unchanged: M7 EPSILONS=(.02,.01,.005), all five
parameter-class directions and GREEN_RELATIVE_ERROR_MAX=.02; CUDA relative
L2 <=2e-6; existing M8d spatial decomposition criteria. Prospective new
shot-sum check uses the same ascending FP32 addition order (bitwise equality
where both paths share that order; roundoff bound reported separately).
Existing single-shot, unfiltered oracles and tests are not edited.

## Revision 2 received during implementation

The Owner's R2 packet/addendum was read after the initial R1 implementation;
it independently prescribes the same source-plus-observation objective (E-7).
The legacy comparison against b07d81d78aabdda54ebc6023fe99265174291753
confirms identical apply_tdfilt, timedomain_filt and cseife filter sources.
The stage source call is conditional on TIME_FILT and INV_STF=0; observations
receive the same stage filter exactly once. QUELLART=6 spike shaping, when
selected, is part of source construction before the stage filter.

H is specifically the existing finite-trace implementation: causal single
pass Butterworth lowpass, then optional highpass, Cseife section coefficients
from ORDER, DT and the periods 1/FC and 1/FC_START. Each IIR section starts
with zero histories. timedomain_filt fills a Numerical Recipes vector at
indices 1..ns, while Cseife processes the passed indices 0..ns-1. dvector
initializes index 0 to zero. Consequently returned samples 1..ns-1 are
filtered and sample ns is left unchanged by each filter. This inherited
endpoint convention is part of H, is retained, and is not silently repaired.

The extra R2 B3 requirements are a held-out shot and CPU/GPU peak memory.
The supplied archive's three marine Vs models each contain 11,000 zero-Vs
cells. The requested B3 input is therefore incompatible with the P1 solid
interior/FD4/no-free-surface envelope; no floor, crop or contact replacement
is authorized or performed. Benchmark evidence remains pending Owner scope.

## Ratified R3 and R4 (2026-11-22)

R3 (SHA256 561f053f7d5d637a04e7e26614e7e0c06667de507655abe4f7d674c5ce1e32f4)
now authorizes the B3-S solid crop; it supersedes the pending marine choice
above. R4 (075d33ff6e84d53a13666fabe648569986ccad7e37f1e96d50daa46ae3b47fb9)
changes only the filtered objective/adjoint endpoint contract. The current
filtered objective is the E-7 expression above with **t=2..NT-1**. With
TIME_FILT=0 its original t=2..NT contract remains. Projection of the final
physical sample occurs after residual preprocessing and before the L2 sum.
The exact CPU transpose omits that injection; the CUDA chronological
production upload explicitly zeros physical NT. The generic compatibility
entry, legacy filter and unfiltered numerical operations are retained.

The 300x200 prerequisite measurement uses source fd7036, NT1500,53 receivers,
segments4, and isolated links sharing all objects except the PSV CUDA core.
One warmup pair precedes six alternating-order pairs. The prospective metric
is the median paired NoFMA/FMA ratio of synchronized forward+finish adapter
wall times: 1.0430379706 (+4.304%, below 15%; individual losses at most7.741%).
Evidence is retained under DENISE_PSV_EVIDENCE_ROOT at r3-timing/result.json
and logs/r3-timing.log.
After that measurement, the PSV core defaults to --fmad=false. Opt in with
make -C src denise_cuda CUDA_PSV_FMAD=true. Other CUDA/M9 flags are unchanged.
Precision-specific object names and an active link-config dependency prevent
stale object or binary reuse on a true/false/true toggle. Use separate output
paths for retained alternative configurations. Standard regression evidence
must name the actual selected configuration.

B3-S uses the package truth/start models cropped by all rows containing true
Vs=0, DH20m, FD4/L1, four-sided CPML/free-surface0, physical Qp=Qs80,
Vp/Vs inversion with fixed smoothed-start rho and fixed Q, training shots
1/26/51/76/100 and held-out38. Sources occupy solid row1 and receivers row2,
including the upper CPML; no silent shift into its interior is made. The
small workflow is2/3.5Hz, four iterations per stage. One candidate truth bank
is shared by all inversions. External stage evaluations generate synthetics
with the same once-filtered source, filter observed traces once, and apply
the same R4 endpoint exclusion to both candidate and Koehn. Filtering raw
synthetics afterwards would change E-7 and is not used. The full B3-S run on
the separately selected execution host remains separately authorized. Per-run ceilings:30min and8GiB.

## R7 review corrections (2026-11-23)

Every newly supported exact-FWI stage (TIME_FILT=1 or NSRC>1) requires
SEISMO=1 and N_ORDER=TIMEWIN=NORMALIZE=
OFFSET_MUTE=TRKILL=TIMELAPSE=0 and QUELLTYPB=1 (both velocity components).
LNORM=2 and GRAD_FORM=2 remain required. NORMALIZE is currently inactive
in this residual implementation, but its unqualified nonzero selector is
rejected as well. ENV only affects other, already rejected norms. Weighted
components, pressure, integration, windows, mute/kill and time-lapse need a
separate discrete-gradient contract. Unfiltered single-shot M8 residual-consumption
contracts are retained, including the N_ORDER and offset-mute adapter tests.
The former filtered N_ORDER endpoint-only test is replaced by rejection: an
endpoint check cannot qualify the missing integration transpose.

Filtered exact stages and multiple exact shots use all sources in ascending order, independently of
TESTSHOT_* (including TESTSHOT_INCR=0, without evaluating its modulo).
Only one unfiltered exact shot retains BASE test-shot selection, including
exclusion of shot1. The original unfiltered single-shot residual-consumption
contract stays intact.
The reload objective comparison now has zero absolute and relative tolerance,
matching the observed equality rather than admitting zero at tiny J.

Owner R7 explicitly chooses the whole shared CUDA PSV core to default to
NoFMA. CUDA_PSV_FMAD controls denise_cuda in MODE=0 general visco-PSV forward
and MODE=1 exact FWI, plus the denise_cuda_psv_forward and
denise_cuda_psv_fd4_l1 harness executables. The measured 300x200 adapter
slowdown was 4.304% median; this is not a universal forward timing guarantee.
M9 compilation rules are independent and unchanged. The existing complete
NoFMA forward fixture requires CPU/CUDA byte identity; it does not require
new-standard GPU bytes to equal the former FMA GPU output.

Benchmark scripts use DENISE_PSV_EVIDENCE_ROOT for retained inputs/results;
the default is benchmark-results/psv-multishot under the checkout. They derive
source paths from their checkout and use git -C there, with standard Git
environment overrides if needed. No enclosing private repository is assumed.
The legacy reload helper additionally requires --truth and --legacy-binary.
Historical R5 source and binary pins are prerequisites for reproducing the
old R5 data; the portable runner intentionally fails closed with other inputs.
Historical freezes retain their original hashes and numerical configuration;
these path changes do not authorize a benchmark replay.
