"""Semantic B5B active-switch contract for the real viscoelastic SH driver."""

from __future__ import annotations

import ast
from pathlib import Path
import math
import re
import shutil
import subprocess
import struct

import pytest


def _write_model(path: Path, values: list[float], cells: int) -> None:
    """Production reads native floats in explicit x-major, then y-minor order."""
    assert len(values) == cells
    assert all(math.isfinite(value) and value > 0.0 for value in values)
    path.write_bytes(struct.pack(f"={len(values)}f", *values))
    assert path.stat().st_size == 4 * len(values)


def _write_sh_parameter(
    repository_root: Path, output: Path, *, mode: int = 0,
    model_prefix: str = "model/true", grad_method: int = 1,
    nprocx: int = 1, nprocy: int = 1, eps_scale: float = 1.0e6,
    stepmax: int = 10,
) -> None:
    """Invoke the repository's SH parameter serializer without its plotting imports."""
    serializer = repository_root / "par" / "pythonIO_SH" / "denise_sh_IO" / "denise_sh_out.py"
    tree = ast.parse(serializer.read_text(encoding="utf-8"), filename=str(serializer))
    writer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "write_denise_para")
    namespace: dict[str, object] = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[writer], type_ignores=[])), str(serializer), "exec"), namespace)
    # This is the SH serializer's complete field set.  Forward-relevant values
    # are deliberately small and physical; inactive FWI fields are safe parser
    # defaults required by read_par.c's positional legacy grammar.
    fields: dict[str, object] = {
        "filename": str(output), "descr": "pytest ephemeral viscoelastic SH forward",
        "MODE": mode, "PHYSICS": 5, "NPROCX": nprocx, "NPROCY": nprocy, "FD_ORDER": 4,
        "max_relative_error": 0, "NX": 32, "NY": 32, "DH": 10.0, "TIME": 0.08, "DT": 0.0005,
        "QUELLART": 1, "SIGNAL_FILE": "source/signal.dat", "TS": 0.0, "SOURCE_FILE": "source/one.dat",
        "RUN_MULTIPLE_SHOTS": 1, "FC_SPIKE_1": 0.0, "FC_SPIKE_2": 0.0, "ORDER_SPIKE": 2, "WRITE_STF": 0,
        "MFILE": model_prefix, "L": 1, "FL": 20.0, "FREE_SURF": 0, "FW": 6, "DAMPING": 2000.0,
        "FPML": 20.0, "npower": 2.0, "k_max_PML": 1.0, "SNAP": 0, "SNAP_SHOT": 1,
        "TSNAP1": 0.0, "TSNAP2": 0.0, "TSNAPINC": 0.0, "IDX": 1, "IDY": 1, "SNAP_FILE": "snap/field",
        "SEISMO": 1, "READREC": 1, "REC_FILE": "receiver/line", "NDT": 1,
        "SEIS_FILE_VX": "su/unused_x.su", "SEIS_FILE_VY": "su/observed_y.su",
        "SEIS_FILE_CURL": "su/unused_curl.su", "SEIS_FILE_DIV": "su/unused_div.su", "SEIS_FILE_P": "su/unused_p.su",
        "LOG_FILE": "log/denise", "ITERMAX": 1, "JACOBIAN": "jacobian/unused", "DATA_DIR": "su/observed",
        "TAPERLENGTH": 1, "GRADT1": 1, "GRADT2": 1, "GRADT3": 1, "GRADT4": 1,
        "INVMAT1": 1, "GRAD_FORM": 1, "QUELLTYPB": 2, "TESTSHOT_START": 1, "TESTSHOT_END": 1,
        "TESTSHOT_INCR": 1, "SWS_TAPER_GRAD_VERT": 0, "SWS_TAPER_GRAD_HOR": 0,
        "EXP_TAPER_GRAD_HOR": 1.0, "SWS_TAPER_GRAD_SOURCES": 0, "SWS_TAPER_CIRCULAR_PER_SHOT": 0,
        "SRTSHAPE": 1, "SRTRADIUS": 50.0, "SWS_TAPER_FILE": 0, "TFILE": "taper/unused",
        "INV_MOD_OUT": 0, "INV_MODELFILE": "model/inverted", "VPUPPERLIM": 5000.0, "VPLOWERLIM": 100.0,
        "VSUPPERLIM": 5000.0, "VSLOWERLIM": 100.0, "RHOUPPERLIM": 5000.0, "RHOLOWERLIM": 100.0,
        "QSUPPERLIM": 100.0, "QSLOWERLIM": 10.0, "GRAD_METHOD": grad_method, "PCG_BETA": 1, "NLBFGS": 1,
        # The real raw Q gradient is O(1e-7) here; 1e6 makes B5A's first
        # subtractive Q trial O(1e-1), safely above float resolution at Q=70.
        "MODEL_FILTER": 0, "FILT_SIZE": 1, "DTINV": 1, "EPS_SCALE": eps_scale, "STEPMAX": stepmax,
        # The active B5A bracket contracts require a strictly expanding factor.
        "SCALEFAC": 2.0, "TRKILL": 0, "TRKILL_FILE": "tracekill/unused", "PICKS_FILE": "picks/unused",
        "MISFIT_LOG_FILE": "log/misfit", "MIN_ITER": 1, "GRAD_FILTER": 0, "FILT_SIZE_GRAD": 1,
    }
    write = namespace["write_denise_para"]
    assert callable(write)
    write(fields)
    # Q controls are parser cases 116--119, newer than the SH serializer.
    output.write_text(output.read_text(encoding="utf-8") + "\nQ_PARAMETERIZATION_MODE = 1\nQ_APPROX_FMIN = 5.0\nQ_APPROX_FMAX = 40.0\nQ_APPROX_DF = 1.0\n", encoding="utf-8")


def _write_ephemeral_sh_forward_fixture(tmp_path: Path, repository_root: Path) -> dict[str, Path]:
    # 32 x 32 cells at DH=10 m with a six-cell CPML.  The source is at
    # (160, 160) m and the receiver line is x=100..220 m at y=160 m: each
    # lies at least four cells beyond the PML and permits measurable travel.
    nx = ny = 32
    dh, dt, time = 10.0, 0.0005, 0.08
    root = tmp_path / "sh_fixture"
    for name in ("model", "source", "receiver", "su", "log"):
        (root / name).mkdir(parents=True, exist_ok=True)
    true_prefix, start_prefix = root / "model" / "true", root / "model" / "start"
    cells = nx * ny
    # The nested order states readmod_visc_SH.c's x-then-y loop explicitly.
    # Vs=2000 m/s, rho=2000 kg/m^3, and physical Q=80 are homogeneous and
    # comfortably positive.  A 20-Hz source has a 100-m S wavelength (ten
    # cells); DT is deliberately well below DH/(Vs*sqrt(2)) ~= 3.5 ms.
    true_q, start_q = [], []
    for x in range(nx):
        for y in range(ny):
            true_q.append(80.0)
            start_q.append(70.0 if 14 <= x <= 18 and 14 <= y <= 18 else 80.0)
    for prefix, qs in ((true_prefix, true_q), (start_prefix, start_q)):
        _write_model(prefix.with_suffix(".vs"), [2000.0] * cells, cells)
        _write_model(prefix.with_suffix(".rho"), [2000.0] * cells, cells)
        _write_model(prefix.with_suffix(".qs"), qs, cells)
    source = root / "source" / "one.dat"
    source.write_text("1\n160 0 160 0 20 1 0 1\n", encoding="ascii")
    receiver = root / "receiver" / "line.dat"
    receiver.write_text("100 160\n120 160\n140 160\n180 160\n200 160\n220 160\n", encoding="ascii")
    par = root / "true_forward.inp"
    _write_sh_parameter(repository_root, par)
    workflow = root / "workflow.inp"
    workflow.write_text("PRO TIME_FILT FC_low FC_high ORDER TIME_WIN GAMMA TWIN- TWIN+ INV_VP_ITER INV_VS_ITER INV_RHO_ITER INV_QS_ITER SPATFILTER WD_DAMP WD_DAMP1 EPRECOND LNORM ROWI STF_INV OFFSETC_STF EPS_STF NORMALIZE OFFSET_MUTE OFFSETC SCALERHO SCALEQS ENV GAMMA_GRAV N_ORDER\n0.01 0 0 20 2 0 1 0 0 0 0 0 0 0 0.5 0.5 0 2 0 0 0 0.1 0 0 10 1 1 0 0 0\n", encoding="ascii")
    return {"root": root, "parameter": par, "workflow": workflow,
            "observed": root / "su" / "observed_y.su.shot1",
            "fwi_observed": root / "su" / "observed_y.su.shot1",
            "true": true_prefix, "start": start_prefix}


def _run_denise(
    repository_root: Path, fixture: dict[str, Path], parameter: Path, *, ranks: int = 1,
) -> subprocess.CompletedProcess[str]:
    executable = repository_root / "bin" / "denise"
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    assert executable.is_file(), "build the normal DENISE executable before this runtime test"
    assert launcher, "an MPI launcher is required"
    return subprocess.run(
        [launcher, "--oversubscribe", "-n", str(ranks), str(executable), str(parameter), str(fixture["workflow"])],
        cwd=fixture["root"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=120,
    )


def _compact(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return re.sub(r"\s+", "", text)


def test_production_fwi_object_is_linked_with_the_active_switch_harness(
    tmp_path: Path, repository_root: Path,
) -> None:
    compiler = shutil.which("mpicc")
    assert compiler, "mpicc is required for the B5B active-switch object link"
    fwi = tmp_path / "FWI_SH_visc.o"
    harness = tmp_path / "active_switch_harness.o"
    linked = tmp_path / "active_switch_link.o"
    for source, output in (
        (repository_root / "src/SH/FWI_SH_visc.c", fwi),
        (repository_root / "tests/utilities/m63c_c8c_active_fwi_switch_harness.c", harness),
    ):
        result = subprocess.run(
            [compiler, "-std=c99", "-fcommon", "-I", str(repository_root / "include"), "-c", str(source), "-o", str(output)],
            cwd=repository_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        assert result.returncode == 0, result.stdout
    linker = shutil.which("ld")
    assert linker, "ld is required for the relocatable B5B object link"
    result = subprocess.run(
        [linker, "-r", str(fwi), str(harness), "-o", str(linked)],
        cwd=repository_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout


def test_active_exact_steepest_driver_chain_and_legacy_exclusion(
    repository_root: Path,
) -> None:
    driver = _compact((repository_root / "src/SH/FWI_SH_visc.c").read_text(encoding="utf-8"))
    assert "if(GRAD_METHOD!=0){err(" in driver
    assert "Q_PARAMETERIZATION_MODE!=Q_PARAMETERIZATION_PHYSICAL" in driver
    chain = (
        "visco_sh_exact_prepare_visco_material(",
        "visco_sh_exact_objective_gradient(",
        "visco_sh_exact_build_steepest_subtractive_step(",
        "step_length_est_sh_visc_exact(",
        "visco_sh_exact_build_trial_parameter_state(",
        "exact_base_primary[exact_j][exact_i]=exact_trial_primary[exact_j][exact_i]",
        "visco_sh_exact_prepare_visco_material(",
    )
    offsets: list[int] = []
    start = 0
    for token in chain:
        offset = driver.index(token, start)
        offsets.append(offset)
        start = offset + len(token)
    assert offsets == sorted(offsets)
    active = driver[offsets[1]:offsets[-1] + len(chain[-1])]
    for legacy in ("grad_obj_sh(", "ass_gradSH_visc(", "step_length_est_sh(",
                   "calc_mat_change_test_SH_visc(", "obj_sh(", "descent("):
        assert legacy not in active


def test_active_commit_is_b2_subtractive_state_then_base_material_regeneration(
    repository_root: Path,
) -> None:
    driver = _compact((repository_root / "src/SH/FWI_SH_visc.c").read_text(encoding="utf-8"))
    accepted = driver[driver.index("exact_trial_state.alpha=exact_line_search_result.selected_alpha;"):]
    assert "exact_status=visco_sh_exact_build_trial_parameter_state(&exact_trial_state);" in accepted
    for field in ("primary", "rho", "q"):
        assert f"exact_base_{field}[exact_j][exact_i]=exact_trial_{field}[exact_j][exact_i];" in accepted
    assert "exact_status=visco_sh_exact_prepare_visco_material(&exact_material_request);" in accepted


def test_ephemeral_true_viscoelastic_sh_forward_produces_observed_data(
    tmp_path: Path, repository_root: Path,
) -> None:
    fixture = _write_ephemeral_sh_forward_fixture(tmp_path, repository_root)
    run = _run_denise(repository_root, fixture, fixture["parameter"])
    assert run.returncode == 0, run.stdout
    observed = fixture["observed"]
    assert observed.is_file() and observed.stat().st_size > 0
    samples = observed.read_bytes()
    # Production's default SU output is 240-byte headers followed by NT floats.
    nt, trace_size = 160, 240 + 4 * 160
    assert len(samples) == 6 * trace_size
    values = struct.unpack(f"={nt}f", samples[240:trace_size])
    assert all(value == value and abs(value) != float("inf") for value in values)
    assert any(value != 0.0 for value in values)
    assert len(set(values)) > 1
    assert fixture["observed"] == fixture["fwi_observed"]
    assert "(DATA_DIR) = su/observed" in fixture["parameter"].read_text(encoding="utf-8")
    # The starting Q differs only in an owned central patch for the next FWI task.
    assert fixture["start"].with_suffix(".qs").read_bytes() != fixture["true"].with_suffix(".qs").read_bytes()


def test_real_active_fwi_completes_one_exact_iteration(
    tmp_path: Path, repository_root: Path,
) -> None:
    """Run one real active FWI iteration without replacing any driver stage."""
    fixture = _write_ephemeral_sh_forward_fixture(tmp_path, repository_root)
    forward = _run_denise(repository_root, fixture, fixture["parameter"])
    assert forward.returncode == 0, forward.stdout
    assert fixture["observed"].is_file()
    fwi_parameter = fixture["root"] / "active_fwi.inp"
    start_before = {
        suffix: fixture["start"].with_suffix(suffix).read_bytes()
        for suffix in (".vs", ".rho", ".qs")
    }
    # This is exactly the active driver's documented supported configuration:
    # MODE=1, L=1, physical Q, a loaded start model, no gravity, and GRAD=0.
    _write_sh_parameter(repository_root, fwi_parameter, mode=1, model_prefix="model/start", grad_method=0)
    run = _run_denise(repository_root, fixture, fwi_parameter)
    assert run.returncode == 0, run.stdout
    assert "TDFWI ITERATION 1" in run.stdout
    assert "opteps_vp" in run.stdout
    assert start_before == {
        suffix: fixture["start"].with_suffix(suffix).read_bytes()
        for suffix in (".vs", ".rho", ".qs")
    }


@pytest.mark.parametrize("grad_method", [1, 2], ids=["pcg", "lbfgs"])
def test_real_active_fwi_rejects_unsupported_optimizers_before_model_mutation(
    tmp_path: Path, repository_root: Path, grad_method: int,
) -> None:
    fixture = _write_ephemeral_sh_forward_fixture(tmp_path, repository_root)
    before = {
        suffix: fixture["start"].with_suffix(suffix).read_bytes()
        for suffix in (".vs", ".rho", ".qs")
    }
    parameter = fixture["root"] / f"unsupported_{grad_method}.inp"
    _write_sh_parameter(repository_root, parameter, mode=1, model_prefix="model/start", grad_method=grad_method)
    run = _run_denise(repository_root, fixture, parameter)
    assert run.returncode != 0
    assert "Exact viscoelastic SH FWI currently supports GRAD_METHOD == 0 only." in run.stdout
    assert before == {
        suffix: fixture["start"].with_suffix(suffix).read_bytes()
        for suffix in (".vs", ".rho", ".qs")
    }


def test_real_active_fwi_completes_one_exact_iteration_with_two_ranks(
    tmp_path: Path, repository_root: Path,
) -> None:
    fixture = _write_ephemeral_sh_forward_fixture(tmp_path, repository_root)
    forward = _run_denise(repository_root, fixture, fixture["parameter"])
    assert forward.returncode == 0, forward.stdout
    parameter = fixture["root"] / "active_fwi_two_rank.inp"
    _write_sh_parameter(
        repository_root, parameter, mode=1, model_prefix="model/start", grad_method=0,
        nprocx=2, nprocy=1,
    )
    run = _run_denise(repository_root, fixture, parameter, ranks=2)
    assert run.returncode == 0, run.stdout
    assert "TDFWI ITERATION 1" in run.stdout
    assert "opteps_vp" in run.stdout


def test_real_active_fwi_two_rank_preacceptance_failure_keeps_start_model(
    tmp_path: Path, repository_root: Path,
) -> None:
    fixture = _write_ephemeral_sh_forward_fixture(tmp_path, repository_root)
    forward = _run_denise(repository_root, fixture, fixture["parameter"])
    assert forward.returncode == 0, forward.stdout
    before = {
        suffix: fixture["start"].with_suffix(suffix).read_bytes()
        for suffix in (".vs", ".rho", ".qs")
    }
    parameter = fixture["root"] / "active_fwi_two_rank_failure.inp"
    # With the measured raw-Q magnitude, this makes every B5A trial invisible
    # at Q=70 and exercises the collective failure before accepted B2.
    _write_sh_parameter(
        repository_root, parameter, mode=1, model_prefix="model/start", grad_method=0,
        nprocx=2, nprocy=1, eps_scale=0.1, stepmax=10,
    )
    run = _run_denise(repository_root, fixture, parameter, ranks=2)
    assert run.returncode != 0
    assert "Exact physical-Q SH line search failed." in run.stdout
    assert before == {
        suffix: fixture["start"].with_suffix(suffix).read_bytes()
        for suffix in (".vs", ".rho", ".qs")
    }
