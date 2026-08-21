# Test suite map

The verification harness is organized by responsibility:

- `tests/`: fast pure-Python tests for analytical helpers, gradient
  mathematics, provenance, parser/generator contracts, and test infrastructure.
- `tests/physics/`: black-box integration tests that execute DENISE through MPI.
- `tests/cases/`: deterministic compact model, source, receiver, and parameter
  generators shared by physics tests.
- `tests/utilities/`: subprocess/provenance support, SU readers, metrics,
  staggered-grid geometry, rheology oracles, and Taylor/gradient helpers.
- `tests/regression/`: reserved location for stored-reference regression cases.
- `tests/conftest.py`: DENISE/MPI discovery and mandatory skip-to-failure policy.

The principal verified families through M5.5 are homogeneous and boundary
elastic SH/PSV propagation, forward SH/PSV viscoelastic rheology, and exact
elastic SH and PSV gradients. Elastic gradients have finite-difference,
discrete-adjoint/math, MPI, and formal Taylor evidence within their documented
parameterizations. PSV `INVMAT1=2` is quarantined. Acoustic, VTI, TTI, RTM,
attenuation inversion, and optimizer convergence are not currently established
by this harness.

Use [docs/testing.md](../docs/testing.md) for build commands, QUICK/MANDATORY/
EXTENDED levels, the coverage matrix, subsystem-specific commands, failure
semantics, and instructions for adding tests. Use
[docs/verification.md](../docs/verification.md) for the detailed scientific
record, numerical results, limitations, and provenance history.
