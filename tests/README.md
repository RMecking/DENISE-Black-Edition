# Test layout

- `cases/`: deterministic input/model/geometry generators.
- `physics/`: analytical black-box physics checks.
- `regression/`: future stored-reference regression cases.
- `utilities/`: subprocess, metadata, seismogram, picking, and metric helpers.
- `conftest.py`: executable and MPI launcher discovery.

See `docs/verification.md` for installation, commands, tolerances, generated
artifacts, and failure inspection.

Use `--require-denise` for verification or CI. Development runs may omit the
flag to allow explicit skips when the executable or MPI launcher is unavailable.
