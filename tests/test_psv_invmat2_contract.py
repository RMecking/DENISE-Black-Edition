from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


DIAGNOSTIC = (
    "PSV INVMAT1=2 (Zp/Zs/rho impedance parameterization) is unsupported: "
    "the legacy PSV model-input/file contract is undefined. Use PSV "
    "INVMAT1=1 or INVMAT1=3 where applicable."
)


def test_psv_invmat2_gate_is_central_early_and_psv_scoped(
    repository_root: Path,
) -> None:
    source = (repository_root / "src/check_mode_phys.c").read_text(encoding="utf-8")
    compact = re.sub(r"\s+", "", source)

    assert "if((PHYSICS==1)&&(INVMAT1==2))" in compact
    assert source.count(DIAGNOSTIC) == 1
    assert source.index("INVMAT1==2") < source.index("switch (MODE)")

    # The validation is not a global numeric-INVMAT rejection.  Other solver
    # families retain their independent dispatch and implementation history.
    denise = (repository_root / "src/denise.c").read_text(encoding="utf-8")
    assert "INVMAT1==2" not in denise.replace(" ", "")
    for physics, dispatcher in ((2, "physics_AC"), (3, "physics_VTI"),
                                (4, "physics_TTI"), (5, "physics_SH")):
        assert f"if(PHYSICS=={physics})" in denise.replace(" ", "")
        assert f"{dispatcher}();" in denise


def test_historical_semantics_are_documented_without_inventing_files(
    repository_root: Path,
) -> None:
    write_par = (repository_root / "src/write_par.c").read_text(encoding="utf-8")
    documentation = (repository_root / "docs/verification.md").read_text(
        encoding="utf-8"
    )
    production_sources = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for root in (repository_root / "src", repository_root / "include")
        for path in root.rglob("*")
        if path.is_file() and path.suffix in {".c", ".h"}
    )

    assert "Historical parameters are Zp, Zs and rho" in write_par
    assert "unsupported for PSV" in write_par
    assert "model-input/file contract is undefined" in write_par
    assert "ppi=Zp=rho*Vp" in documentation
    assert "pu=Zs=rho*Vs" in documentation
    assert "prho=rho" in documentation
    assert "no supported `.zp` or `.zs` inputs" in documentation
    assert '".zp"' not in production_sources
    assert '".zs"' not in production_sources


def test_both_psv_readers_still_have_no_invmat2_file_contract(
    repository_root: Path,
) -> None:
    for relative in (
        "src/PSV/readmod_elastic_PSV.c",
        "src/PSV/readmod_visc_PSV.c",
    ):
        source = (repository_root / relative).read_text(encoding="utf-8")
        compact = re.sub(r"\s+", "", source)
        assert "INVMAT1==1" in compact
        assert "INVMAT1==3" in compact
        assert "INVMAT1==2" not in compact


def test_m541a_quarantine_provenance(repository_root: Path) -> None:
    artifact = json.loads(
        (repository_root / "tests/m5.4.1a_psv_invmat2_quarantine_validation.json")
        .read_text(encoding="utf-8")
    )
    patch = repository_root / "tests/m5.4.1a_psv_invmat2_quarantine.patch"
    raw = patch.read_bytes()
    changed_files = {
        line.removeprefix("diff --git a/").split(" b/", 1)[0]
        for line in raw.decode("utf-8").splitlines()
        if line.startswith("diff --git a/")
    }

    assert hashlib.sha256(raw).hexdigest() == artifact["production_patch_sha256"]
    assert changed_files == {"src/check_mode_phys.c", "src/write_par.c"}
    assert artifact["changed_production_files"] == sorted(changed_files)
    assert artifact["file_contract"] == "file contract undefined"
    assert artifact["post_fix_repeated_behavior"]["elastic"]["returncodes"] == [1, 1, 1]
    assert artifact["post_fix_repeated_behavior"]["viscoelastic"]["returncodes"] == [1, 1, 1]
    for medium in ("elastic", "viscoelastic"):
        mpi_rejection = artifact["post_fix_two_rank_rejection"][medium]
        assert mpi_rejection == {
            "diagnostic_present": True,
            "no_valid_seismograms": True,
            "ranks": 2,
            "returncode": 1,
            "timeout_or_hang": False,
        }
    assert artifact["verdict"] == "M5.4.1a PSV INVMAT1=2 SAFELY QUARANTINED"
