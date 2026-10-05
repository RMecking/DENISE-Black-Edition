# DENISE applications: frozen Marmousi-II A0/A1/A2

This package provides case preparation, input validation, forward-data QC and
raw migration-image integration around existing DENISE operators. It is an
unpublished publication candidate, not a new numerical implementation.

## Cases and accepted identities

| Directory | Meaning | Execution core |
|---|---|---|
| `marmousi2/` | Historical A0/A1-FD8 preparation/forward/QC baseline | `f315d157c2e817480990c97a3de45c4a23e465db` |
| `marmousi2_fd4/` | Historical matched Taylor-FD4 data and M9e-4 CUDA/water limitation | `f315d157c2e817480990c97a3de45c4a23e465db` |
| `marmousi2_fluid2/` | Accepted A2 physical-water CPU MODE=2 campaign, 100 independent shots | `f5ae2f3506f4ea8024b96d28bcbc3fc2470bb31d` |

The FD4 historical CUDA preflight rejected the unchanged 11000 zero-Vs/zero-mu
water cells. It is not the accepted CPU migration result. Keep these names and
all original evidence paths; do not consolidate or rewrite frozen provenance.
The copies of the two historical case manifests in this candidate describe
their original execution cores; they are not forward cases authorized at A2 HEAD.

External reference: `daniel-koehn/DENISE-Benchmark`, frozen revision
`931131e8dc53649320a5620befe751ad32d2f7bd`, `Marmousi-II/`.
No redistribution permission is assumed from the historical license audit.
Do not vendor copied models, geometry, SU data or benchmark documents into Git.
Import these into ignored directories from a separately acquired exact reference.

Python 3.10+ and NumPy are used by the backend. Pytest is needed for application
tests; matplotlib is additionally needed for raw RTM visualizations. POSIX MPI
and the exact authorized DENISE executable are needed only for new numerical
runs. Matplotlib/pytest are not supplied by `requirements-applications.txt`.

## Frozen-benchmark provenance policy

A completed benchmark is bound to its exact execution HEAD, executable/build
identity, model/source/data/input identities and retained result/metadata hashes.
The accepted A2 core remains `f5ae2f3506f4ea8024b96d28bcbc3fc2470bb31d`
after `modernization` advances. `origin/modernization` is an observed repository
state at final audit, not a completed-benchmark acceptance equality condition.
The explicit closure amendment preserves the earlier HOLD and execution-time
provenance; later Application Review accepted the campaign on that basis.

This does not authorize new runs. Before every new campaign, independently
verify its explicitly authorized execution SHA, exact binary/build, inputs,
numerical contract, backend and clean core/oracle state. Do not substitute the
moving branch tip, rewrite refs, bypass checks or edit old execution metadata.

## Implemented command families

Run commands from the corresponding execution checkout root, never inside the
accepted evidence directories to regenerate products. The following document
implemented interfaces, not permission to execute them in this packaging task.
Use a fresh authorized case/output area for any future numerical reproduction.

### Generic initialization and historical forward preparation/QC

```text
python -m tools.denise_case --help
python -m tools.denise_case init applications/new-case
python -m tools.denise_case --manifest applications/marmousi2/case.yaml prepare
python -m tools.denise_case --manifest applications/marmousi2/case.yaml validate
python -m tools.denise_case --manifest applications/marmousi2/case.yaml generate
python -m tools.denise_case --manifest applications/marmousi2/case.yaml forward --dry-run
python -m tools.denise_case --manifest applications/marmousi2/case.yaml forward
python -m tools.denise_case --manifest applications/marmousi2/case.yaml qc-models
python -m tools.denise_case --manifest applications/marmousi2/case.yaml qc-data
python -m tools.denise_case --manifest applications/marmousi2/case.yaml report
```

`init` makes a template with explicit REPLACE placeholders, not a ready benchmark.
The manifest resolves `provenance.benchmark_checkout` relative to the case root.
`forward --dry-run` still materializes files/metadata; it is not an audit of an
existing accepted run. Historical forward commands require the historical core,
not A2 HEAD. The FD4 preparation/forward/data-audit command family uses
`--manifest applications/marmousi2_fd4/case.yaml`; `audit-forward` validates the
100 vx +100 vy SU components. Historical `fd4-freeze`, `verify-baseline`,
`build-cuda`, `rtm`, `qc-rtm` and `report-rtm` remain separate legacy interfaces,
not commands for the accepted physical-water CPU campaign.

For a separately authorized historical-lane CUDA build, select the compiler and
architectures explicitly when needed:

```text
python -m tools.denise_case --manifest applications/marmousi2_fd4/case.yaml build-cuda --nvcc nvcc --cuda-archs "86 90"
```

API: `build_cuda(config, nvcc=..., cuda_archs=...)`; the pure
`cuda_build_spec(nvcc=..., cuda_archs=..., environ=...)` generates the specification
without compiling or probing. Each option follows explicit CLI/API argument,
then its `NVCC` / `CUDA_ARCHS` environment value, then repository/build-system
default. With both omitted and environment overrides absent, no NVCC or
CUDA_ARCHS assignment is injected: `make` resolves `nvcc` through PATH and uses
its existing architecture default. There is no application-specific toolkit
directory, architecture default, automatic GPU-based selection or WSL assumption. Choose only
architectures supported by the chosen toolkit. Actual future builds record the
selection/command/probe; original historical build receipts remain unchanged.
No compiler or build is run by command-specification tests.

### Physical-water CPU MODE=2 preparation and bounded/full migration

The accepted `marmousi2_fluid2/case.json` deliberately still references the named
historical sibling `app-marmousi2-a01/applications/marmousi2_fd4/case.yaml`.
Its accepted direct data, preparation/provenance and baseline receipts are
external audit dependencies. The manifest copy above does not relocate those
dependencies or constitute a substitute data set.

```text
python -m tools.denise_case.fluid2 --help
python -m tools.denise_case.fluid2 verify
python -m tools.denise_case.fluid2 plan
python -c "from tools.denise_case.fluid2 import prepare; prepare([1], 'shot_001')"
python -m tools.denise_case.fluid2 run --shots 1 --name shot_001
python -m tools.denise_case.fluid2 batch --shots 1,2,3,4,5
python -m tools.denise_case.fluid2 batch --shots 96,97,98,99,100
python -m tools.denise_case.fluid2_qc shot_001
```

There is no `fluid2 prepare` CLI subcommand: preparation is the imported API
shown above and occurs automatically through `run`/`batch`. `plan` records
20 deterministic batches, [1..5], [6..10], ... [96..100]. Execute every batch
in ascending physical-shot order only after authorization. Each shot is an
independent durable operation; never relaunch an attempted/incomplete shot or
skip solely because files exist. Reuse only exact hash/QC-validated metadata.
Raw lambda/mu pairs remain independent; the diagnostic [1,50,100] replay does
not count as three retained individual campaign results.

`fluid2.require_core(repo, authorized_sha=CORE_SHA)` requires a full explicitly
authorized SHA and exact local HEAD equality, clean protected production/core
tests and an empty index. The default is the frozen A2 core above. A wrong HEAD,
wrong explicit SHA or modified protected source still fails. The optional local
`origin/modernization` ref is returned as `observed_repository_state`; equal,
advanced and unavailable origin states do not determine PASS/FAIL. This check
never fetches, contacts a remote or rewrites refs, including for legacy callers
passing `remote=True`. Binary/build receipts, recorded core-source hashes and
inputs remain separate checks in the execution/resume paths. Authorization of
a future numerical campaign is still required; these commands are not run here.

### Aggregate QC and campaign report

The reusable post-processing API is
`tools.denise_case.campaign.aggregate_campaign` / `report_campaign`.
The corresponding implemented CLI consumes existing independent shot products:

```text
python -m tools.denise_case.campaign --help
python -m tools.denise_case.campaign aggregate --runs-root applications/marmousi2_fluid2/runs --shots 1-100 --shape 174,500 --campaign-receipt applications/marmousi2_fluid2/runs/campaign_100/summary.json --vs-model ../app-marmousi2-a01/applications/marmousi2_fd4/input/models/smooth2.vs --output applications/publication_candidate/runs/new-aggregate
python -m tools.denise_case.campaign report --runs-root applications/marmousi2_fluid2/runs --shots 1-100 --shape 174,500 --campaign-receipt applications/marmousi2_fluid2/runs/campaign_100/summary.json --aggregate-receipt applications/publication_candidate/runs/new-aggregate/receipt.json --vs-model ../app-marmousi2-a01/applications/marmousi2_fd4/input/models/smooth2.vs --output applications/publication_candidate/runs/new-report
```

Outputs must be fresh, caller-selected directories, outside individual shot
raw/derived/input directories; reports also stay outside the aggregate directory.
Existing output directories are rejected, not overwritten. Change the relative
input arguments when relocating the independently acquired evidence bundle;
historical absolute metadata paths are ignored for product resolution, never
rewritten. Products resolve as `shot_NNN/run_metadata.json`, `provenance.json`
and `raw/migration.image_{lambda,mu}_raw.bin` beneath `--runs-root`.

Aggregation validates singleton completed CPU receipts, exact requested physical
shot coverage, duplicate/missing IDs, frozen core and consistent recorded binary
identity, provenance/raw hashes, FP64 size/layout/shape and finite values. With
`--campaign-receipt`, retained metadata and raw identities must also agree.
Lambda and mu are summed separately in ascending physical-shot order using
FP64, without weights, averaging, normalization, smoothing, mute, illumination
weighting, preconditioning or parameter transform. `receipt.json` records order,
configuration, relative input/output identities and hashes. This is not execution.

Reporting verifies that aggregate order/configuration/input/output hashes match
the shots, then emits deterministic `summary.json` and self-contained `index.html`
with all retained scientific caveats and descriptive QC distributions. Pass
`--vs-model` to recompute raw water/solid QC using the original physical Vs mask;
without it reporting requires retained hash-bound execution/aggregate QC.
Optional repeatable `--qc-record PATH` consumes shot QC records only when their
raw identities match. The accepted historical aggregate receipt is also supported.
Shot 68 and first-solid-row observations are not new amplitude rejection gates.

Aggregation and QC/report generation never launch DENISE, rebuild binaries or
modify inputs. Execution, aggregation and reporting are distinct operations.
Original local campaign/plot/closure recipes remain excluded historical evidence;
do not rerun them over accepted products. `tools.denise_case.fluid2_report` remains
the excluded workstation-specific two-shot unblock helper, not this reporter.

## Scientific scope and retained observations

Taylor-FD4's conservative dispersion advisory is approximately 5.50625 Hz at
the true minimum nonzero velocity 881 m/s and DH=20 m. The preserved 15 Hz
filtered-spike low-pass exceeds it. A2 establishes workflow/operator integration,
physical-water restricted migration and deterministic raw image production/QC;
it does not establish final broadband imaging accuracy. No CUDA fluid support
or FLUID-3/4 acceptance is claimed.

Raw lambda/mu JT images are little-endian FP64, row-major [depth_y,x],
[174,500], 696000 bytes each; they are not velocities or display-processed
products. Model files instead use FP32 x-major/y-fastest. Raw water gMu remains
exact +0 in all 11000 cells, including sign bits, without floor, omission or
post-mask. P99 clipping is display-only; no smoothing/muting/normalization,
illumination weighting, parameter transform or stack averaging is implied.
Shot 68 remains a norm/QC observation, not a rejected shot; the strong first
solid-row contribution is preserved for later scientific interpretation.

## Publication boundary

Publish reviewed reusable Python sources, retained and new application tests, the
human-authored case manifests, case/root ignore policies and this guide.
Keep accepted reports/receipts/data/images/QC and local audit recipes outside
Git. Cache/build artifacts are disposable; accepted scientific evidence is NOT.
The exact publication and exclusion inventories and packaging test receipt are
under `publication_candidate/runs/`, retained locally for independent review.
The previous packaging assessment is [PUBLICATION_GATE.md](PUBLICATION_GATE.md);
the current blocker resolution is [HARDENING_REPORT.md](HARDENING_REPORT.md).
Nothing has been staged, committed, pushed or submitted as a PR.
