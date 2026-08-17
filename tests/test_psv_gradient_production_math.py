from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from tests.cases.psv_fwi_gradient import (
    PSVFWIGradientConfig,
    heterogeneous_direction,
    heterogeneous_model,
)
from tests.utilities.psv_gradient import (
    density_average_jvp,
    density_average_vjp,
    harmonic_mu_jvp,
    harmonic_mu_vjp,
    matrix_inner,
)


def test_harmonic_four_cell_map_is_an_exact_transpose() -> None:
    mu = [[2.0, 3.0, 4.0], [5.0, 7.0, 11.0], [13.0, 17.0, 19.0]]
    delta = [[0.2, -0.1, 0.3], [-0.4, 0.6, -0.2], [0.5, 0.1, -0.3]]
    adjoint = [[-0.7, 0.4, 0.2], [0.1, -0.5, 0.8], [0.3, -0.2, 0.9]]
    assert math.isclose(
        matrix_inner(harmonic_mu_jvp(mu, delta), adjoint),
        matrix_inner(delta, harmonic_mu_vjp(mu, adjoint)),
        rel_tol=2.0e-15,
        abs_tol=2.0e-15,
    )


def test_rx_ry_density_maps_are_an_exact_joint_transpose() -> None:
    rho = [[2.0, 2.5, 3.0], [3.5, 4.0, 4.5], [5.0, 5.5, 6.0]]
    delta = [[0.2, -0.1, 0.3], [-0.4, 0.6, -0.2], [0.5, 0.1, -0.3]]
    adj_x = [[-0.7, 0.4, 0.2], [0.1, -0.5, 0.8], [0.3, -0.2, 0.9]]
    adj_y = [[0.6, -0.3, 0.5], [-0.2, 0.7, -0.4], [0.1, 0.8, -0.6]]
    delta_x, delta_y = density_average_jvp(rho, delta)
    assert math.isclose(
        matrix_inner(delta_x, adj_x) + matrix_inner(delta_y, adj_y),
        matrix_inner(delta, density_average_vjp(rho, adj_x, adj_y)),
        rel_tol=2.0e-15,
        abs_tol=2.0e-15,
    )


def test_production_selector_is_narrow_and_macro_free(repository_root: Path) -> None:
    psv = (repository_root / "src/PSV/psv.c").read_text(encoding="utf-8")
    assembler = (repository_root / "src/PSV/assemble_gradPSV_exact.c").read_text(
        encoding="utf-8"
    )
    combined = psv + assembler + (
        repository_root / "src/PSV/update_v_PML_PSV.c"
    ).read_text(encoding="utf-8")
    assert "(MODE==1)&&(mode==1)&&(L==0)" in psv
    assert "(INVMAT1==1)" in psv
    assert "((GRAD_FORM==1)||(GRAD_FORM==2))" in psv
    assert "M54_" not in combined
    assert "mu_corner*mu_corner/(4.0*mu_cell*mu_cell)" in assembler
    assert "-0.5*rx*rx*g_rx" in assembler
    assert "-0.5*ry*ry*g_ry" in assembler


def test_heterogeneous_holdout_is_positive_joint_and_crosses_both_seams() -> None:
    config = PSVFWIGradientConfig()
    model = heterogeneous_model(config)
    direction = heterogeneous_direction(config)
    for component in ("vp", "vs", "rho"):
        assert min(model[component]) > 0.0
        assert max(model[component]) > min(model[component])
        assert max(direction[component]) > 0.0 > min(direction[component])
        horizontal = (config.nx // 2) * config.ny
        vertical = config.ny // 2
        assert model[component][horizontal - 1] != model[component][horizontal]
        assert model[component][vertical - 1] != model[component][vertical]
    assert len({tuple(direction[name]) for name in direction}) == 3


def test_m54_production_patch_provenance(repository_root: Path) -> None:
    expected_files = {
        "include/fd.h",
        "src/Makefile",
        "src/PSV/FWI_PSV.c",
        "src/PSV/alloc_fwiPSV.c",
        "src/PSV/assemble_gradPSV_exact.c",
        "src/PSV/grad_obj_psv.c",
        "src/PSV/psv.c",
        "src/PSV/update_v_PML_PSV.c",
        "src/TTI/TTI.c",
        "src/VTI/VTI.c",
    }
    patch = repository_root / "tests/m5.4_psv_gradient_production_repair.patch"
    raw = patch.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "d5ff334ee244c0457944fb97746771ebbbb26b659c38e5aeaafdf6f15fad9edc"
    )
    text = raw.decode("utf-8")
    changed = set(re.findall(r"^diff --git a/(.+?) b/", text, flags=re.MULTILINE))
    assert changed == expected_files
    assert "src/PSV/update_s_elastic_PML_PSV.c" not in changed
