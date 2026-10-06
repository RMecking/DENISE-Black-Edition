# Marmousi-II A2.5A image interpretation

Solver-free Application post-processing of the accepted 100-shot CPU campaign.
This is an implementation candidate for independent scientific review, not a
new numerical baseline or a publication/merge authorization.

## Identities and use

Application BASE `54c69297a6c1d5fef94c396b49451ebaadcec2ab`, publication
`6c348d490862277d6c2c5d017bf2e4029ce54169`, historical execution Core
`f5ae2f3506f4ea8024b96d28bcbc3fc2470bb31d`, benchmark
`931131e8dc53649320a5620befe751ad32d2f7bd` have distinct meanings. A moving
remote branch is never a post-processing acceptance equality gate.

From this worktree, with Python and NumPy:

```text
python -m tools.denise_case image-condition --case applications/marmousi2_a25a/case.json --output <fresh-output-directory>
python -m pytest tests/applications
```

`case.json` points to sibling historical worktrees. Its frozen aggregate/model
hashes and physical IDs 1..100 are validated by the production entry point.
Other datasets may use the generic immutable loader/operators but must not be
silently branded as this frozen A2.5A campaign. No live `require_core`, solver,
compiler, or network connection is used by image conditioning.

Output must be fresh and outside accepted evidence, model/input directories and
their aliases. Existing outputs are refused, including on an interrupted run;
choose a new directory. No raw image is copied back or overwritten.

## Product contract

- `AUTHORITATIVE_RAW`: authenticated aggregate/individual-shot images and models;
  immutable bytes-backed arrays, hashes rechecked after processing.
- `DERIVED_NUMERICAL`: FP64 fixed-density cotangent transforms, nonperiodic
  centered derivatives with NaN/validity bitmaps, declared full-image Fourier
  filters, fixed diagnostic gains, ascending-ID shot subset reductions, and QC.
- `DISPLAY_ONLY`: percentile saturation, odd signed compression, physical-axis
  PNG/SVG sections and profiles; display pixels never feed scientific metrics.

The graph uses parallel branches, not an opaque filter/gain chain. Transforming
by spatial background coefficients precedes the transformed filter/derivative
branches; no commutation is assumed. Original smooth2 models are the sole
Jacobian background. Vs=0 gives transformed Vs zero naturally; no floor or
repair mask. Log-Vs sensitivity is a frozen-zero extension in water, where
physical log coordinates do not exist.

P99 is the primary display. P98/P99.5/full-linear, three signed compressions,
solid/deep/interface-excluded scale domains, both fixed bandpasses, padding and
taper controls, impulse/step/first-solid-stripe responses and all fixed gain
variants remain visible. Gain panels share their un-gained P99; other panel
scales are explicitly recorded. Undefined zero scale/denominators are not
epsilon repaired. Derivative edges are invalid, not fabricated values.

True Vp/Vs/rho, contrasts and edge maps are evaluation-only, never conditioning
inputs. Ten structural windows and true-Vp edge definition are frozen before
image processing. Ridge proximity, axial orientation and two displaced controls
are descriptive; strongest-per-column ridges can associate different features.
Layered geometry can survive lateral displacement; orientation |cos| has
isotropic baseline 2/pi. Mu/Vs correspondence with Vp geometry does not validate
Vs reflectivity. Surrogates shift odd/even aggregates cyclically, not 100
independent shots, and are not confirmatory geological p-values.

`receipt.json` binds all outputs, operations, parameters, input hashes, source
hashes, Python/NumPy/platform/FFT/renderer identities to a timestamp-free content
identity. Use the same environment for byte-identical reproduction; portability
of inputs/receipt paths does not imply cross-platform FFT bitwise identity.

## Scientific boundaries

Squared sensitivity norms are not physical seismic energy. Raw JT amplitudes
and Vp/Vs cotangents are not reflectivity or velocity updates. Shot 68 is retained.
First-solid row 22 (460 m center; interface midpoint450 m) is also the OBC row:
strong response is preserved and not attributed to one cause. Acquisition
coverage is not uniform illumination. Coherence alone does not establish geology.

The historical 15-Hz source low-pass exceeds the conservative Taylor-FD4
dispersion advisory of approximately5.50625 Hz. Post-processing cannot restore
missing bandwidth, identify illumination/Hessian information, or establish
final broadband accuracy. No FLUID-3/4 or CUDA physical-water acceptance follows.
Depth gain is not illumination compensation. Accepted A2.5B H15/H5 evidence
is consolidated separately in [A2.5 publication](../marmousi2_a25/README.md);
no J/JT/HVP or illumination runs occur in this offline image workflow.
