from __future__ import annotations

import json
import re
from pathlib import Path

from tests.utilities.sh_cross_code_inventory import (
    compare_sources,
    git_blob_oid,
)


INVENTORY = "m6.0_sh_cross_code_inventory.json"
OID = re.compile(r"[0-9a-f]{40}")


def test_source_comparison_ignores_comments_and_formatting(tmp_path: Path) -> None:
    left = tmp_path / "left.c"
    right = tmp_path / "right.c"
    left.write_text("int f(void) { /* old */ return 2; }\n", encoding="utf-8")
    right.write_text("// new\nint f(void)\n{\nreturn 2;\n}\n", encoding="utf-8")

    comparison = compare_sources(left, right)

    assert comparison["token_similarity"] == 1.0
    assert comparison["left_sha256"] != comparison["right_sha256"]


def test_inventory_contract_and_recorded_source_references(
    repository_root: Path,
) -> None:
    inventory = json.loads((repository_root / "tests" / INVENTORY).read_text())
    repositories = inventory["repositories"]

    assert inventory["scope"] == "source audit only"
    assert inventory["denise_sh_is_oracle"] is False
    assert inventory["production_files_changed"] == []
    assert inventory["final_verdict"] == "M6.0 SH CROSS-CODE AUDIT COMPLETE"
    assert repositories["black_edition"]["commit"] == (
        "ff8029e17b07d24644fb594f4102ed2ece9007b5"
    )
    assert repositories["denise_sh"]["commit"] == (
        "9a4efe1db13c2076580cb36b4fc3a9c63a664079"
    )

    manifests: dict[str, dict[str, str]] = {}
    for key, repository in repositories.items():
        manifest = repository["source_manifest"]
        assert manifest
        assert len(manifest) == len(set(manifest))
        assert all(OID.fullmatch(blob_oid) for blob_oid in manifest.values())
        manifests[key] = manifest

    for claim in inventory["source_genealogy"]:
        assert claim["evidence_type"] in {
            "source text",
            "build reachability",
            "call-site reachability",
            "git history",
            "runtime test",
        }
        for source in claim["sources"]:
            assert source["path"] in manifests[source["repository"]]

    for comparison in inventory["numeric_source_comparisons"]:
        assert 0.0 <= comparison["token_similarity"] <= 1.0
        assert comparison["interpretation"] == "genealogy evidence only"
        for side in ("left", "right"):
            source = comparison[side]
            assert source["path"] in manifests[source["repository"]]

    classes = {row["classification"] for row in inventory["capability_matrix"]}
    allowed_classes = {
        "IMPLEMENTATION EXISTS",
        "IMPLEMENTATION WIRED",
        "IMPLEMENTATION VERIFIED",
        "IMPLEMENTATION INCOMPLETE",
        "IMPLEMENTATION ABSENT",
    }
    assert classes <= allowed_classes
    assert {
        "IMPLEMENTATION EXISTS",
        "IMPLEMENTATION WIRED",
        "IMPLEMENTATION VERIFIED",
        "IMPLEMENTATION INCOMPLETE",
    } <= classes
    capabilities = {
        row["capability"]: row for row in inventory["capability_matrix"]
    }
    for capability in ("attenuation model update", "attenuation bounds"):
        assert capabilities[capability] == {
            "capability": capability,
            "black_edition": "IMPLEMENTATION INCOMPLETE",
            "denise_sh": "IMPLEMENTATION ABSENT",
            "classification": "IMPLEMENTATION INCOMPLETE",
        }
    assert len(inventory["audit_questions"]) == 9

    black = repositories["black_edition"]
    # A nested Windows worktree has a Windows-absolute .git pointer that Linux
    # Git cannot parse through /mnt.  Its parent is the same object database.
    git_repository = (
        repository_root.parent
        if (repository_root / ".git").is_file()
        and (repository_root.parent / ".git").is_dir()
        else repository_root
    )
    for path, expected in black["source_manifest"].items():
        assert git_blob_oid(git_repository, black["commit"], path) == expected
