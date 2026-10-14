"""Artifact-free public FD8 API integration; not a scientific accuracy gate.

Each case runs a fresh C process using the real public header and a normally
compiled shared library. No sealed fixture, private harness or reference cache.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import pytest


ROOT = Path(__file__).resolve().parents[2]
NORMAL_FLAGS = [
    "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
    "-fPIC", "-shared",
]
SCENARIOS = ("zero_state", "prepared_source", "capacity", "bad_abi")
CLIENT = r'''
#include "denise_elastic_psv_born.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define REQUIRE(condition) do { if (!(condition)) { \
    fprintf(stderr, "FD8 smoke failed at line %d: %s (%s)\n", __LINE__, \
            #condition, denise_elastic_psv_forward_fd8_last_error_v1()); \
    return 1; } } while (0)

int main(int argc, char **argv) {
    struct denise_elastic_psv_forward_fd8_config_v1 c = {0};
    struct denise_elastic_psv_forward_fd8 *h = NULL;
    float lambda[81], mu[81], rho[81], source[4] = {0};
    float data[9], repeated[9], sentinel[9];
    int receiver_i = 4, receiver_j = 4, p, nt, k;
    REQUIRE(argc == 2);
    REQUIRE(strcmp(argv[1], "zero_state") == 0 ||
            strcmp(argv[1], "prepared_source") == 0 ||
            strcmp(argv[1], "capacity") == 0 ||
            strcmp(argv[1], "bad_abi") == 0);
    nt = strcmp(argv[1], "prepared_source") == 0 ? 4 : 1;
    k = 2 * nt;
    for (p = 0; p < 81; ++p) {
        lambda[p] = 6.75f; mu[p] = 3.375f; rho[p] = 1.5f;
    }
    for (p = 0; p < 9; ++p) data[p] = repeated[p] = sentinel[p] = -123.5f;
    c.struct_size = sizeof(c); c.abi_version = 1;
    c.coefficient_family = DENISE_ELASTIC_PSV_FD8_TAYLOR8_V1;
    c.field_boundary_policy = DENISE_ELASTIC_PSV_FD8_ZERO_FIELD_V1;
    c.material_boundary_policy = DENISE_ELASTIC_PSV_FD8_REPLICATED_MATERIAL_V1;
    c.base.nx = c.base.ny = 9; c.base.nt = nt;
    c.base.dh = 1.0f; c.base.dt = 0.025f;
    c.base.invmat1 = 3; c.base.fdorder = 8;
    c.base.ndt = c.base.dtinv = c.base.mpi_size = 1;
    c.base.receiver_components = 2;
    c.base.lambda = lambda; c.base.mu = mu; c.base.rho = rho;
    c.base.source_i = c.base.source_j = 4;
    c.base.source_samples = source; c.base.receiver_count = 1;
    c.base.receiver_i = &receiver_i; c.base.receiver_j = &receiver_j;

    if (strcmp(argv[1], "bad_abi") == 0) {
        c.abi_version = 0;
        h = (struct denise_elastic_psv_forward_fd8 *)(uintptr_t)1;
        REQUIRE(denise_elastic_psv_forward_fd8_create_v1(&c, &h) == -1);
        REQUIRE(h == NULL);
    } else {
        if (nt == 4) source[0] = 0.001f;
        REQUIRE(denise_elastic_psv_forward_fd8_create_v1(&c, &h) == 0);
        REQUIRE(h != NULL);
        if (strcmp(argv[1], "capacity") == 0) {
            REQUIRE(denise_elastic_psv_forward_fd8_nonlinear_v1(
                h, lambda, mu, data, (size_t)(k - 1)) == -1);
            REQUIRE(memcmp(data, sentinel, sizeof(data)) == 0);
        }
        REQUIRE(denise_elastic_psv_forward_fd8_nonlinear_v1(
            h, lambda, mu, data, (size_t)k) == 0);
        REQUIRE(data[k] == sentinel[k]);
        for (p = 0; p < k; ++p) REQUIRE(isfinite(data[p]));
        if (nt == 4) {
            /* Predeclared receiver=(4,4) is source-coupled at next momentum.
             * Assert integration/chronology, not an amplitude/accuracy oracle. */
            REQUIRE(data[0] == 0.0f && data[1] == 0.0f);
            REQUIRE(data[2] != 0.0f || data[3] != 0.0f);
            REQUIRE(denise_elastic_psv_forward_fd8_nonlinear_v1(
                h, lambda, mu, repeated, (size_t)k) == 0);
            REQUIRE(memcmp(data, repeated, (size_t)k * sizeof(float)) == 0);
        } else {
            for (p = 0; p < k; ++p) REQUIRE(data[p] == 0.0f);
        }
        if (strcmp(argv[1], "capacity") == 0) {
            REQUIRE(denise_elastic_psv_forward_fd8_nonlinear_v1(
                h, lambda, mu, data, (size_t)(k + 1)) == 0);
            REQUIRE(data[k] == sentinel[k]);
        }
        denise_elastic_psv_forward_fd8_destroy_v1(&h);
        REQUIRE(h == NULL);
    }
    printf("{\"scenario\":\"%s\",\"status\":\"PASS\","
           "\"nx\":9,\"ny\":9,\"nt\":%d,\"receiver\":[4,4]}\n", argv[1], nt);
    return 0;
}
'''


def _raw(path):
    content = path.read_bytes()
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def _record(name, data):
    # Optional diagnostics only; ordinary CI requires no external artifact input.
    directory = os.environ.get("M9L1A_CI_SMOKE_EVIDENCE")
    if directory:
        with (Path(directory) / name).open("x", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)


@pytest.fixture(scope="module")
def public_client(tmp_path_factory):
    compiler = shutil.which("gcc")
    assert compiler, "gcc is required for the FD8 public API integration smoke"
    work = tmp_path_factory.mktemp("m9l1a-ci-smoke")
    library = work / "libdenise_m9l1a_ci.so"
    source = work / "public-client.c"
    executable = work / "public-client"
    source.write_text(CLIENT, encoding="utf-8")
    commands = [
        [compiler, *NORMAL_FLAGS, "-I", str(ROOT / "include"),
         str(ROOT / "src/PSV/elastic_psv_born.c"), "-lm", "-o", str(library)],
        [compiler, *[flag for flag in NORMAL_FLAGS if flag != "-shared"],
         "-I", str(ROOT / "include"), str(source), "-L", str(work),
         "-ldenise_m9l1a_ci", "-Wl,-rpath," + str(work), "-lm", "-o", str(executable)],
    ]
    started = time.perf_counter()
    for command in commands:
        completed = subprocess.run(command, capture_output=True, text=True)
        assert completed.returncode == 0, (
            "FD8 smoke compilation failed: " + repr(command) + "\n" +
            completed.stdout + completed.stderr
        )
    _record("public-api-build.json", {
        "commands": commands, "compiler": compiler,
        "compiler_version": subprocess.check_output([compiler, "--version"], text=True),
        "elapsed_seconds": time.perf_counter() - started,
        "library": {"path": str(library), **_raw(library)},
        "client": {"path": str(executable), **_raw(executable)},
        "client_source": {"path": str(source), **_raw(source)},
        "source": _raw(ROOT / "src/PSV/elastic_psv_born.c"),
        "header": _raw(ROOT / "include/denise_elastic_psv_born.h"),
    })
    return executable


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_public_fd8_ci_smoke(public_client, scenario):
    command = [str(public_client), scenario]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(completed.stdout)
    assert result["scenario"] == scenario and result["status"] == "PASS"
    _record(scenario + ".json", {
        "command": command, "exit_code": completed.returncode,
        "stdout": completed.stdout, "stderr": completed.stderr,
        "elapsed_seconds": time.perf_counter() - started,
    })
