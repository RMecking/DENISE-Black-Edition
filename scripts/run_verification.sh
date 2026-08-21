#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "${script_dir}/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: ./scripts/run_verification.sh MODE

Modes:
  quick      Run pure-Python tests; DENISE and MPI are not required.
  mandatory  Build DENISE, run quick tests, then non-extended physics tests.
  extended   Build DENISE, then run extended physics tests.

MPIEXEC_FLAGS defaults to --oversubscribe when it is not already set.
EOF
}

run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    "$@"
}

run_quick() {
    run python3 -m pytest tests -m 'not integration' -q
}

build_denise() {
    run make -C libcseife
    run make -C src denise
}

if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

cd "${repository_root}"

case "$1" in
    quick)
        run_quick
        ;;
    mandatory)
        build_denise
        run_quick
        export MPIEXEC_FLAGS="${MPIEXEC_FLAGS:---oversubscribe}"
        run python3 -m pytest tests/physics -m 'not extended' --require-denise -v
        ;;
    extended)
        build_denise
        export MPIEXEC_FLAGS="${MPIEXEC_FLAGS:---oversubscribe}"
        run python3 -m pytest tests/physics -m extended --require-denise -v
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        printf 'Unknown mode: %s\n\n' "$1" >&2
        usage >&2
        exit 2
        ;;
esac
