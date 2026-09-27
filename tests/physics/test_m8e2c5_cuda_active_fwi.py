"""M8e-2C5 production integration gate for active CUDA exact-visco P/SV FWI."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from array import array
from pathlib import Path

import pytest

from tests.physics.test_visco_psv_active_fwi import (
    _point_model_at,
    _read_float_grid,
)
from tests.physics.test_visco_psv_physical_gradient_oracle import (
    PHYSICAL_FIELDS,
    ViscoPSVOracleConfig,
    _objective,
    _reference_model,
    _run_forward,
    _target_model,
    _write_case,
)
from tests.utilities.fwi_gradient import read_su_float_samples
from tests.test_m8e1b2a_cuda_psv_forward import _find_nvcc
from tests.utilities.runner import result_summary, run_denise


pytestmark = pytest.mark.integration


def _configure_cuda_envelope(directory: Path, q_mode: int) -> None:
    source = (directory / "source.dat").read_text(encoding="ascii").splitlines()
    fields = source[1].split()
    fields[-1] = "1"
    source[1] = " ".join(fields)
    (directory / "source.dat").write_text("\n".join(source) + "\n", encoding="ascii")
    parameters = directory / "denise.inp"
    parameters.write_text(
        parameters.read_text(encoding="ascii")
        + f"# positional parameter 116\nQ_PARAMETERIZATION_MODE ={q_mode}\n"
        + "# positional parameter 117\nQ_APPROX_FMIN =5.0\n"
        + "# positional parameter 118\nQ_APPROX_FMAX =40.0\n"
        + "# positional parameter 119\nQ_APPROX_DF =1.0\n",
        encoding="ascii",
    )


@pytest.fixture(scope="module")
def cuda_denise_binary(repository_root: Path) -> Path:
    make = shutil.which("make")
    nvcc = _find_nvcc()
    if not make or not nvcc:
        pytest.skip("CUDA build toolchain is unavailable")
    library = subprocess.run(
        [make, "-C", "libcseife"], cwd=repository_root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    assert library.returncode == 0, library.stdout
    build = subprocess.run(
        [make, "-C", "src", "denise_cuda", f"NVCC={nvcc}", "CUDA_ARCHS=86"],
        cwd=repository_root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    assert build.returncode == 0, build.stdout
    binary = repository_root / "bin" / "denise_cuda"
    assert binary.is_file()
    return binary


@pytest.fixture(scope="module")
def cuda_denise_nofma_binary(repository_root: Path) -> Path:
    make = shutil.which("make")
    nvcc = _find_nvcc()
    if not make or not nvcc:
        pytest.skip("CUDA build toolchain is unavailable")
    binary = repository_root / "bin" / "denise_cuda_h1_nofma"
    build = subprocess.run(
        [
            make, "-C", "src", "denise_cuda", f"NVCC={nvcc}",
            "CUDA_ARCHS=86", "CUDA_BUILD_DIR=.cuda-h1-nofma",
            f"CUDA_DENISE_BIN={binary}",
            "CUDA_CXXFLAGS=-O2 -std=c++14 --fmad=false",
        ],
        cwd=repository_root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    assert build.returncode == 0, build.stdout
    assert binary.is_file()
    return binary


def _prepare_observed(
    directory: Path, *, q_mode: int, config: ViscoPSVOracleConfig,
    repository_root: Path, denise_binary: Path, mpiexec: str,
):
    _write_case(
        directory, config=config, model=_target_model(_reference_model(config), config),
        mode=0,
    )
    _configure_cuda_envelope(directory, q_mode)
    return _run_forward(
        directory, repository_root=repository_root, denise_binary=denise_binary,
        mpiexec=mpiexec, config=config,
    )


def _set_workflow(directory: Path, **updates: int | float) -> None:
    path = directory / "workflow.inp"
    lines = path.read_text(encoding="ascii").splitlines()
    names, values = lines[0].split(), lines[1].split()
    assert len(names) == len(values)
    for name, value in updates.items():
        values[names.index(name)] = str(value)
    lines[1] = " ".join(values)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _read_raw_gradients(directory: Path, config: ViscoPSVOracleConfig):
    return {
        name: _read_float_grid(
            directory / "jacobian" / f"gradient.raw.{name}", config.cell_count
        )
        for name in PHYSICAL_FIELDS
    }


def _run_exact_gradient(
    directory: Path, *, repository_root: Path, binary: Path, mpiexec: str,
    backend: str, report_path: Path | None = None,
    residual_path: Path | None = None, raw_residual_test: bool = False,
    raw_residual_path: Path | None = None,
) -> None:
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("DENISE_PSV_BACKEND", backend)
        environment.setenv("DENISE_PSV_EXACT_SEGMENTS", "4")
        environment.setenv("DENISE_PSV_EXACT_VISCO_GRADIENT", "1")
        environment.setenv("DENISE_PSV_EXACT_DISTRIBUTED_GRADIENT_ONLY", "1")
        if report_path is not None:
            environment.setenv(
                "DENISE_CUDA_PSV_EXACT_FWI_REPORT_FILE", str(report_path)
            )
        if residual_path is not None:
            environment.setenv(
                "DENISE_CUDA_PSV_PRODUCTION_RESIDUAL_FILE", str(residual_path)
            )
        if raw_residual_path is not None:
            environment.setenv(
                "DENISE_CUDA_PSV_TEST_RAW_RESIDUAL_FILE", str(raw_residual_path)
            )
        if raw_residual_test:
            environment.setenv(
                "DENISE_CUDA_PSV_TEST_USE_RAW_MODELED_OBSERVED", "1"
            )
        result = run_denise(
            repository_root=repository_root, case_directory=directory,
            denise_binary=binary, mpiexec=mpiexec, ranks=1,
            configuration={"gate": "M8e-2C5-H1", "backend": backend},
            timeout_seconds=180.0,
        )
    assert result.returncode == 0, result_summary(result)


def _relative_l2(left, right) -> float:
    delta = math.fsum((a - b) ** 2 for a, b in zip(left, right, strict=True))
    scale = math.fsum(b * b for b in right)
    return math.sqrt(delta / max(scale, 1.0e-300))


@pytest.mark.parametrize("q_mode", [0, 1], ids=["legacy-q", "physical-q"])
def test_cuda_active_iteration_persists_and_cpu_reloads(
    tmp_path: Path, repository_root: Path, denise_binary: Path,
    cuda_denise_binary: Path, mpiexec: str, monkeypatch: pytest.MonkeyPatch,
    q_mode: int,
) -> None:
    config = ViscoPSVOracleConfig()
    base_model = _reference_model(config)
    truth_directory = tmp_path / "truth"
    observed = _prepare_observed(
        truth_directory, q_mode=q_mode, config=config,
        repository_root=repository_root, denise_binary=denise_binary,
        mpiexec=mpiexec,
    )

    fwi_directory = tmp_path / "fwi"
    _write_case(
        fwi_directory, config=config, model=base_model, mode=1,
        observed=truth_directory,
    )
    _configure_cuda_envelope(fwi_directory, q_mode)
    cuda_report_path = fwi_directory / "jacobian" / "gradient.cuda_active_fwi.json"
    monkeypatch.setenv("DENISE_PSV_BACKEND", "cuda")
    monkeypatch.setenv("DENISE_PSV_EXACT_SEGMENTS", "4")
    monkeypatch.setenv("DENISE_PSV_EXACT_VISCO_GRADIENT", "1")
    monkeypatch.setenv("DENISE_CUDA_PSV_EXACT_FWI_REPORT_FILE", str(cuda_report_path))
    result = run_denise(
        repository_root=repository_root, case_directory=fwi_directory,
        denise_binary=cuda_denise_binary, mpiexec=mpiexec, ranks=1,
        configuration={"gate": "M8e-2C5", "q_mode": q_mode, "segments": 4},
        timeout_seconds=180.0,
    )
    assert result.returncode == 0, result_summary(result)

    cuda_report = json.loads(cuda_report_path.read_text(encoding="utf-8"))
    assert cuda_report["complete"] is True
    assert cuda_report["segments"] == 4
    assert cuda_report["q_parameterization_mode"] == q_mode
    assert cuda_report["trial_forward_count"] >= 1
    assert cuda_report["trial_failure_count"] == 0
    assert cuda_report["trial_trace_d2h_calls"] == 2 * cuda_report["trial_forward_count"]
    assert cuda_report["context_destroy_calls"] == 1
    assert cuda_report["modeled_trace_d2h_calls"] == 2
    assert cuda_report["observed_trace_h2d_calls"] == 0
    assert cuda_report["observed_trace_h2d_bytes"] == 0
    assert cuda_report["production_residual_h2d_calls"] == 2
    assert cuda_report["production_residual_h2d_bytes"] == 2 * len(
        config.receivers_m
    ) * config.samples_per_trace * 4
    assert cuda_report["physical_gradient_d2h_calls"] == 5
    assert cuda_report["adjoint_initial_h2d_calls"] == 0
    assert cuda_report["adjoint_initial_zero_calls"] == 1
    assert cuda_report["reverse_timesteps"] == config.samples_per_trace
    assert cuda_report["reverse_segments"] == 4
    assert cuda_report["per_step_h2d_calls"] == 0
    assert cuda_report["per_step_d2h_calls"] == 0
    assert cuda_report["per_step_allocation_calls"] == 0
    assert cuda_report["per_step_free_calls"] == 0
    assert cuda_report["per_step_blocking_sync_calls"] == 0
    assert cuda_report["map_kernel_launches"] == 1
    assert cuda_report["physical_allocation_calls"] == 1
    fields = cuda_report["fields"]
    assert set(fields) == set(PHYSICAL_FIELDS)
    assert all(fields[name]["finite"] for name in PHYSICAL_FIELDS)
    assert all(fields[name]["nonzero"] > 0 for name in PHYSICAL_FIELDS)
    assert len({fields[name]["fnv1a64"] for name in PHYSICAL_FIELDS}) == 5
    assert all(
        fields[name]["download_fp32_fnv1a64"]
        == fields[name]["production_matrix_fnv1a64"]
        for name in PHYSICAL_FIELDS
    )

    active_report = json.loads(
        (fwi_directory / "jacobian" / "gradient.active_fwi.json").read_text(
            encoding="utf-8"
        )
    )
    assert active_report["accepted_alpha"] > 0.0
    assert active_report["accepted_objective"] < active_report["base_objective"]
    assert math.isclose(
        active_report["base_objective"], cuda_report["base_objective"],
        rel_tol=0.0, abs_tol=1.0e-15,
    )
    assert all(active_report["update_norms"][name] > 0.0 for name in PHYSICAL_FIELDS)

    persisted_prefix = fwi_directory / active_report["persisted_prefix"]
    accepted = {
        name: _read_float_grid(Path(f"{persisted_prefix}.{name}"), config.cell_count)
        for name in PHYSICAL_FIELDS
    }
    assert accepted["qp"] != _read_float_grid(
        fwi_directory / "model" / "current.qp", config.cell_count
    )
    assert accepted["qs"] != _read_float_grid(
        fwi_directory / "model" / "current.qs", config.cell_count
    )

    reload_directory = tmp_path / "reload"
    _write_case(reload_directory, config=config, model=accepted, mode=0)
    _configure_cuda_envelope(reload_directory, q_mode)
    reload_prefix = reload_directory / "model" / "accepted"
    for name in PHYSICAL_FIELDS:
        shutil.copyfile(Path(f"{persisted_prefix}.{name}"), Path(f"{reload_prefix}.{name}"))
    _point_model_at(reload_directory / "denise.inp", "model/accepted")
    monkeypatch.delenv("DENISE_PSV_BACKEND")
    reloaded = _run_forward(
        reload_directory, repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec, config=config,
    )
    reloaded_objective = _objective(reloaded, observed)
    assert math.isclose(
        reloaded_objective, active_report["accepted_objective"],
        rel_tol=2.0e-6, abs_tol=1.0e-10,
    )


def test_integrated_gradient_is_bit_exact_across_segmentation(
    tmp_path: Path, repository_root: Path, denise_binary: Path,
    cuda_denise_binary: Path, mpiexec: str,
) -> None:
    config = ViscoPSVOracleConfig()
    model = _reference_model(config)
    truth = tmp_path / "truth"
    _prepare_observed(
        truth, q_mode=1, config=config, repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec,
    )
    payloads = {}
    reports = {}
    for segments in (1, 4, config.samples_per_trace):
        case = tmp_path / f"segments-{segments}"
        _write_case(case, config=config, model=model, mode=1, observed=truth)
        _configure_cuda_envelope(case, 1)
        report_path = case / "jacobian" / "gradient.cuda_active_fwi.json"
        with pytest.MonkeyPatch.context() as environment:
            environment.setenv("DENISE_PSV_BACKEND", "cuda")
            environment.setenv("DENISE_PSV_EXACT_SEGMENTS", str(segments))
            environment.setenv("DENISE_PSV_EXACT_VISCO_GRADIENT", "1")
            environment.setenv("DENISE_PSV_EXACT_DISTRIBUTED_GRADIENT_ONLY", "1")
            environment.setenv("DENISE_CUDA_PSV_EXACT_FWI_REPORT_FILE", str(report_path))
            result = run_denise(
                repository_root=repository_root, case_directory=case,
                denise_binary=cuda_denise_binary, mpiexec=mpiexec, ranks=1,
                configuration={"gate": "M8e-2C5-segments", "segments": segments},
                timeout_seconds=180.0,
            )
        assert result.returncode == 0, result_summary(result)
        payloads[segments] = b"".join(
            (case / "jacobian" / f"gradient.raw.{name}").read_bytes()
            for name in PHYSICAL_FIELDS
        )
        reports[segments] = json.loads(report_path.read_text(encoding="utf-8"))
    assert payloads[1] == payloads[4] == payloads[config.samples_per_trace]
    hashes = {
        segments: tuple(report["fields"][name]["fnv1a64"] for name in PHYSICAL_FIELDS)
        for segments, report in reports.items()
    }
    assert hashes[1] == hashes[4] == hashes[config.samples_per_trace]


def test_n_order_one_consumes_authoritative_transformed_residual(
    tmp_path: Path, repository_root: Path, denise_binary: Path,
    cuda_denise_binary: Path, mpiexec: str,
) -> None:
    config = ViscoPSVOracleConfig()
    model = _reference_model(config)
    truth = tmp_path / "truth"
    _prepare_observed(
        truth, q_mode=1, config=config, repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec,
    )
    cases = {name: tmp_path / name for name in ("cpu", "cuda", "raw-old")}
    for case in cases.values():
        _write_case(case, config=config, model=model, mode=1, observed=truth)
        _configure_cuda_envelope(case, 1)
        _set_workflow(case, N_ORDER=1)

    _run_exact_gradient(
        cases["cpu"], repository_root=repository_root, binary=denise_binary,
        mpiexec=mpiexec, backend="cpu",
    )
    report_path = cases["cuda"] / "jacobian" / "gradient.cuda_active_fwi.json"
    residual_path = cases["cuda"] / "jacobian" / "production-residual.bin"
    raw_residual_path = cases["cuda"] / "jacobian" / "raw-residual.bin"
    _run_exact_gradient(
        cases["cuda"], repository_root=repository_root, binary=cuda_denise_binary,
        mpiexec=mpiexec, backend="cuda", report_path=report_path,
        residual_path=residual_path, raw_residual_path=raw_residual_path,
    )
    _run_exact_gradient(
        cases["raw-old"], repository_root=repository_root,
        binary=cuda_denise_binary, mpiexec=mpiexec, backend="cuda",
        raw_residual_test=True,
    )

    cpu = _read_raw_gradients(cases["cpu"], config)
    corrected = _read_raw_gradients(cases["cuda"], config)
    raw_old = _read_raw_gradients(cases["raw-old"], config)
    agreement = {
        name: _relative_l2(corrected[name], cpu[name]) for name in PHYSICAL_FIELDS
    }
    old_failure = {
        name: _relative_l2(raw_old[name], cpu[name]) for name in PHYSICAL_FIELDS
    }
    assert all(value <= 2.0e-6 for value in agreement.values()), agreement
    assert all(value >= 0.10 for value in old_failure.values()), old_failure

    residual = array("f")
    residual.frombytes(residual_path.read_bytes())
    component_samples = config.receiver_count * config.samples_per_trace
    assert len(residual) == 2 * component_samples
    raw = array("f")
    raw.frombytes(raw_residual_path.read_bytes())
    assert len(raw) == len(residual)
    for component in range(2):
        for receiver in range(config.receiver_count):
            sample = component * component_samples + receiver * config.samples_per_trace
            assert residual[sample] == 0.0
    transformed_relative = _relative_l2(residual, raw)
    assert transformed_relative >= 0.10
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["observed_trace_h2d_calls"] == 0
    assert report["production_residual_h2d_calls"] == 2
    print(
        "M8e-2C5-H1 N_ORDER=1",
        json.dumps({"cpu_cuda": agreement, "raw_old": old_failure,
                    "residual_transform": transformed_relative}, sort_keys=True),
    )


def test_canonical_production_residual_reproduces_pre_h1_path(
    tmp_path: Path, repository_root: Path, denise_binary: Path,
    cuda_denise_binary: Path, mpiexec: str,
) -> None:
    config = ViscoPSVOracleConfig()
    model = _reference_model(config)
    truth = tmp_path / "truth"
    _prepare_observed(
        truth, q_mode=1, config=config, repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec,
    )
    corrected, old = tmp_path / "corrected", tmp_path / "old"
    for case in (corrected, old):
        _write_case(case, config=config, model=model, mode=1, observed=truth)
        _configure_cuda_envelope(case, 1)
    report_path = corrected / "jacobian" / "gradient.cuda_active_fwi.json"
    _run_exact_gradient(
        corrected, repository_root=repository_root, binary=cuda_denise_binary,
        mpiexec=mpiexec, backend="cuda", report_path=report_path,
    )
    _run_exact_gradient(
        old, repository_root=repository_root, binary=cuda_denise_binary,
        mpiexec=mpiexec, backend="cuda", raw_residual_test=True,
    )
    canonical_relative = {}
    for name in PHYSICAL_FIELDS:
        corrected_values = _read_float_grid(
            corrected / "jacobian" / f"gradient.raw.{name}", config.cell_count
        )
        old_values = _read_float_grid(
            old / "jacobian" / f"gradient.raw.{name}", config.cell_count
        )
        canonical_relative[name] = _relative_l2(corrected_values, old_values)
        # calc_res_PSV materializes the authoritative subtraction as FP32,
        # while the historical kernel subtracted two FP32 operands in FP64.
        # Byte identity is therefore not mathematically expected. The direct
        # 2C1-2C4 gates retain byte/5e-13 checks for state, native gradients,
        # physical mapping, and replay operands; this integration delta is
        # constrained separately far below their forward FP32 input error.
        assert canonical_relative[name] <= 1.0e-8
    print("M8e-2C5-H1 canonical pre/post", json.dumps(canonical_relative))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["observed_trace_h2d_calls"] == 0
    assert report["production_residual_h2d_calls"] == 2


def test_offset_mute_consumes_generic_calc_res_output(
    tmp_path: Path, repository_root: Path, denise_binary: Path,
    cuda_denise_nofma_binary: Path, mpiexec: str,
) -> None:
    config = ViscoPSVOracleConfig()
    model = _reference_model(config)
    truth = tmp_path / "truth"
    _prepare_observed(
        truth, q_mode=1, config=config, repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec,
    )
    cpu_case, cuda_case = tmp_path / "cpu", tmp_path / "cuda"
    for case in (cpu_case, cuda_case):
        _write_case(case, config=config, model=model, mode=1, observed=truth)
        _configure_cuda_envelope(case, 1)
        _set_workflow(case, OFFSET_MUTE=1, OFFSETC=180.0)
    _run_exact_gradient(
        cpu_case, repository_root=repository_root, binary=denise_binary,
        mpiexec=mpiexec, backend="cpu",
    )
    residual_path = cuda_case / "jacobian" / "production-residual.bin"
    _run_exact_gradient(
        cuda_case, repository_root=repository_root,
        binary=cuda_denise_nofma_binary,
        mpiexec=mpiexec, backend="cuda", residual_path=residual_path,
    )
    cpu, cuda = _read_raw_gradients(cpu_case, config), _read_raw_gradients(
        cuda_case, config
    )
    agreement = {
        name: _relative_l2(cuda[name], cpu[name]) for name in PHYSICAL_FIELDS
    }
    assert all(value <= 2.0e-6 for value in agreement.values()), agreement
    residual = array("f")
    residual.frombytes(residual_path.read_bytes())
    samples = config.samples_per_trace
    component_samples = config.receiver_count * samples
    offsets = [
        math.hypot(x - config.source_x_m, y - config.source_y_m)
        for x, y in config.receivers_m
    ]
    for component in range(2):
        for receiver, offset in enumerate(offsets):
            trace = residual[
                component * component_samples + receiver * samples:
                component * component_samples + (receiver + 1) * samples
            ]
            if offset >= 180.0:
                assert not any(trace)
            else:
                assert any(trace[1:])
    print("M8e-2C5-H1 OFFSET_MUTE", json.dumps(agreement, sort_keys=True))


def test_adapter_reuses_closed_cuda_gradient_chain(repository_root: Path) -> None:
    source = (repository_root / "src/PSV/psv_cuda_exact_fwi.c").read_text(
        encoding="utf-8"
    )
    required = (
        "denise_cuda_psv_forward_segments_prepare",
        "denise_cuda_psv_forward_segment_capture",
        "denise_cuda_psv_adjoint_production_residual_prepare",
        "denise_cuda_psv_native_gradient_prepare",
        "denise_cuda_psv_physical_gradient_prepare",
        "denise_cuda_psv_adjoint_reverse_sweep",
        "denise_cuda_psv_physical_gradient_map",
        "denise_cuda_psv_physical_gradient_download",
    )
    assert all(name in source for name in required)
    assert "native[G" not in source
    assert "q_to_tau_derivative" not in source


def test_explicit_cpu_selector_preserves_active_fwi_path(
    tmp_path: Path, repository_root: Path, denise_binary: Path,
    cuda_denise_binary: Path, mpiexec: str,
) -> None:
    config = ViscoPSVOracleConfig()
    model = _reference_model(config)
    truth = tmp_path / "truth"
    _write_case(
        truth, config=config, model=_target_model(model, config), mode=0,
    )
    _run_forward(
        truth, repository_root=repository_root, denise_binary=denise_binary,
        mpiexec=mpiexec, config=config,
    )
    case = tmp_path / "fwi"
    _write_case(case, config=config, model=model, mode=1, observed=truth)
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("DENISE_PSV_BACKEND", "cpu")
        result = run_denise(
            repository_root=repository_root, case_directory=case,
            denise_binary=cuda_denise_binary, mpiexec=mpiexec, ranks=1,
            configuration={"gate": "M8e-2C5-explicit-cpu"},
            timeout_seconds=180.0,
        )
    assert result.returncode == 0, result_summary(result)
    report = json.loads(
        (case / "jacobian" / "gradient.active_fwi.json").read_text(encoding="utf-8")
    )
    assert report["accepted_objective"] < report["base_objective"]


@pytest.mark.parametrize(
    "selector,source_type,hide_gpu,expected",
    [
        ("bogus", 1, False, "unknown DENISE_PSV_BACKEND='bogus'"),
        ("cuda", 4, False, "violates frozen envelope"),
        ("cuda", 1, True, "no visible CUDA device"),
    ],
    ids=["unknown-selector", "unsupported-source", "hidden-gpu"],
)
def test_explicit_cuda_failures_precede_model_mutation(
    tmp_path: Path, repository_root: Path, denise_binary: Path,
    cuda_denise_binary: Path, mpiexec: str, selector: str, source_type: int,
    hide_gpu: bool, expected: str,
) -> None:
    config = ViscoPSVOracleConfig()
    truth = tmp_path / "truth"
    _prepare_observed(
        truth, q_mode=0, config=config, repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec,
    )
    case = tmp_path / "fwi"
    _write_case(
        case, config=config, model=_reference_model(config), mode=1,
        observed=truth,
    )
    _configure_cuda_envelope(case, 0)
    if source_type != 1:
        lines = (case / "source.dat").read_text(encoding="ascii").splitlines()
        fields = lines[1].split()
        fields[-1] = str(source_type)
        lines[1] = " ".join(fields)
        (case / "source.dat").write_text("\n".join(lines) + "\n", encoding="ascii")
    before = {
        name: (case / "model" / f"current.{name}").read_bytes()
        for name in PHYSICAL_FIELDS
    }
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("DENISE_PSV_BACKEND", selector)
        if hide_gpu:
            environment.setenv("CUDA_VISIBLE_DEVICES", "-1")
        result = run_denise(
            repository_root=repository_root, case_directory=case,
            denise_binary=cuda_denise_binary, mpiexec=mpiexec, ranks=1,
            configuration={"gate": "M8e-2C5-fail-closed", "selector": selector},
            timeout_seconds=90.0,
        )
    assert result.returncode != 0
    output = (
        result.stdout_path.read_text(encoding="utf-8", errors="replace")
        + result.stderr_path.read_text(encoding="utf-8", errors="replace")
    )
    assert expected in output
    assert all(
        (case / "model" / f"current.{name}").read_bytes() == before[name]
        for name in PHYSICAL_FIELDS
    )
    assert not list((case / "model").glob("accepted_stage_*"))


def test_trial_forward_failure_destroys_partial_context_without_persisting(
    tmp_path: Path, repository_root: Path, denise_binary: Path,
    cuda_denise_binary: Path, mpiexec: str,
) -> None:
    config = ViscoPSVOracleConfig()
    truth = tmp_path / "truth"
    _prepare_observed(
        truth, q_mode=0, config=config, repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec,
    )
    case = tmp_path / "fwi"
    _write_case(
        case, config=config, model=_reference_model(config), mode=1,
        observed=truth,
    )
    _configure_cuda_envelope(case, 0)
    before = {
        name: (case / "model" / f"current.{name}").read_bytes()
        for name in PHYSICAL_FIELDS
    }
    report_path = case / "jacobian" / "gradient.cuda_active_fwi.json"
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("DENISE_PSV_BACKEND", "cuda")
        environment.setenv("DENISE_PSV_EXACT_SEGMENTS", "4")
        environment.setenv("DENISE_CUDA_PSV_EXACT_FWI_REPORT_FILE", str(report_path))
        environment.setenv("DENISE_CUDA_PSV_TEST_FAIL_TRIAL_AFTER_CREATE", "1")
        result = run_denise(
            repository_root=repository_root, case_directory=case,
            denise_binary=cuda_denise_binary, mpiexec=mpiexec, ranks=1,
            configuration={"gate": "M8e-2C5-trial-failure"},
            timeout_seconds=90.0,
        )
    assert result.returncode != 0
    output = (
        result.stdout_path.read_text(encoding="utf-8", errors="replace")
        + result.stderr_path.read_text(encoding="utf-8", errors="replace")
    )
    assert "injected CUDA trial-forward failure after context creation" in output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["complete"] is True
    assert report["trial_forward_count"] == 0
    assert report["trial_failure_count"] == 1
    assert all(
        (case / "model" / f"current.{name}").read_bytes() == before[name]
        for name in PHYSICAL_FIELDS
    )
    assert not list((case / "model").glob("accepted_stage_*"))


def test_partial_production_residual_upload_fails_closed_and_destroys(
    tmp_path: Path, repository_root: Path, denise_binary: Path,
    cuda_denise_binary: Path, mpiexec: str,
) -> None:
    config = ViscoPSVOracleConfig()
    truth = tmp_path / "truth"
    _prepare_observed(
        truth, q_mode=0, config=config, repository_root=repository_root,
        denise_binary=denise_binary, mpiexec=mpiexec,
    )
    case = tmp_path / "fwi"
    _write_case(
        case, config=config, model=_reference_model(config), mode=1,
        observed=truth,
    )
    _configure_cuda_envelope(case, 0)
    before = {
        name: (case / "model" / f"current.{name}").read_bytes()
        for name in PHYSICAL_FIELDS
    }
    report_path = case / "jacobian" / "gradient.cuda_active_fwi.json"
    with pytest.MonkeyPatch.context() as environment:
        environment.setenv("DENISE_PSV_BACKEND", "cuda")
        environment.setenv("DENISE_PSV_EXACT_SEGMENTS", "4")
        environment.setenv("DENISE_CUDA_PSV_EXACT_FWI_REPORT_FILE", str(report_path))
        environment.setenv(
            "DENISE_CUDA_PSV_TEST_FAIL_PRODUCTION_RESIDUAL_AFTER_FIRST_COPY", "1"
        )
        result = run_denise(
            repository_root=repository_root, case_directory=case,
            denise_binary=cuda_denise_binary, mpiexec=mpiexec, ranks=1,
            configuration={"gate": "M8e-2C5-H1-partial-upload"},
            timeout_seconds=90.0,
        )
    assert result.returncode != 0
    output = (
        result.stdout_path.read_text(encoding="utf-8", errors="replace")
        + result.stderr_path.read_text(encoding="utf-8", errors="replace")
    )
    assert "injected production residual upload failure after first copy" in output
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["complete"] is False
    assert report["context_destroy_calls"] == 1
    assert all(
        (case / "model" / f"current.{name}").read_bytes() == before[name]
        for name in PHYSICAL_FIELDS
    )
    assert not list((case / "model").glob("accepted_stage_*"))
