"""Freeze the narrow paired FD4/L=1 PSV factor-two precision contract."""

from __future__ import annotations

import re
from pathlib import Path


def test_fd4_gsls_factor_two_is_fp32_and_stress_remains_mixed(
    repository_root: Path,
) -> None:
    cpu_source = (
        repository_root / "src/PSV/update_s_visc_PML_PSV.c"
    ).read_text(encoding="utf-8")
    cuda_source = (
        repository_root / "src/CUDA/psv_fd4_l1.cu"
    ).read_text(encoding="utf-8")

    cpu_fd4 = cpu_source.split("case 4:", 1)[1].split("case 6:", 1)[0]
    cuda_fd4 = cuda_source.split(
        "__global__ void stress_fd4_l1", 1
    )[1].split("#undef A", 1)[0]

    expected = (
        (cpu_fd4, (
            ("2.0", "f[j][i]", "vyy"),
            ("2.0", "f[j][i]", "vxx"),
            ("2.0f", "d[j][i][l]", "vyy"),
            ("2.0f", "d[j][i][l]", "vxx"),
        )),
        (cuda_fd4, (
            ("2.0", "d.f[q]", "vyy"),
            ("2.0", "d.f[q]", "vxx"),
            ("2.0f", "d.d1[q]", "vyy"),
            ("2.0f", "d.d1[q]", "vxx"),
        )),
    )
    for region, expressions in expected:
        for factor, coefficient, derivative in expressions:
            pattern = (
                rf"(?<![\w.]){re.escape(factor)}(?![\w.])"
                rf"\s*\*\s*{re.escape(coefficient)}\s*\*\s*{derivative}\b"
            )
            assert len(re.findall(pattern, region)) == 1
