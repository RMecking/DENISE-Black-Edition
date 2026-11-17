# Focused legacy bugfix regression

Run from the repository root on the existing Linux CPU CI toolchain.

```sh
python3 -m pytest tests/test_legacy_bugfix_integration.py -q
```

The suite uses pytest, the standard library, the repository's GNU/OpenMPI build toolchain, and the sanitizer runtimes already used for native checks. Temporary executables, extracted driver blocks and generated SU records live in pytest's temporary directory. The existing non-physics CI selection includes this module. The historical writer, allocator and cleanup probes read the pinned base objects from Git. A private source export can supply those objects by setting `DENISE_BUGFIX_BASE_SOURCE`; every imported file is hash-checked.

| Provenance | SHA-256 |
|---|---|
| SU-INPUT-CORRECTION-01 | `b8cda883c32a04c76469b70fc0da95caf65d3329d1c3237354dd88af5740a461` |
| SU-WRITER-BOUNDARY-01 | `2677521b8cc4cd6c40c30d40c8ccbc84dcb0e3204390a64614624146051c147d` |
| DENISE-LEGACY-MEMORY-SAFETY-01 | `7be7da2773544045969b0978185a1128c03189d768cbad47e4fe11ca837ba0c1` |
| FD-SH-CLEANUP-01 | `91675bf43822e3c4fa50db1a7e31951c86f0a1a9e5fde9b052c4b6d6b00de2b9` |

The approved integration is based on `fe75412bce431a8597d26709ea13f0bdd67229a2`. The production diff and freeze identities, per-path hashes, review identity and fixture hashes are recorded in [reference-pins.json](../tests/utilities/legacy_bugfix/reference-pins.json). The fixture directory contains the original eight reader SU inputs and three accepted numerical outputs. The writer inputs and expected bytes are regenerated from the frozen independent recipe and compared to their original hashes. Tests compare the first-sample SH result, the 510-word and 530-word FD2 outputs, and the generated writer bytes exactly.

The syntax checks and focused local/one-rank harnesses cover the four correction packages. The SH–SU path connects source generation, the four-package build's SU writer and reader, B4 allocation, and per-shot cleanup. The separate exact SH MPI receiver fix and full-solver execution are outside this verification.

