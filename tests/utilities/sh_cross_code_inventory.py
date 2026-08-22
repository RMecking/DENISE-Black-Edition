"""Reproducible source-comparison helpers for the M6.0 SH audit.

Textual similarity is genealogy evidence only.  It is never interpreted as
proof of numerical or physical correctness.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
from pathlib import Path


_C_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_C_TOKEN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|"
    r"(?:0[xX][0-9A-Fa-f]+|\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)|"
    r"==|!=|<=|>=|&&|\|\||<<|>>|->|\+\+|--|"
    r"[^\s]"
)


def sha256_file(path: Path) -> str:
    """Return the byte-level SHA-256 for *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_c_tokens(path: Path) -> list[str]:
    """Return C tokens after removing comments and formatting differences."""

    source = path.read_text(encoding="utf-8", errors="replace")
    return _C_TOKEN.findall(_C_COMMENT.sub("", source))


def compare_sources(left: Path, right: Path) -> dict[str, object]:
    """Describe byte provenance and comment/whitespace-insensitive similarity."""

    left_tokens = normalized_c_tokens(left)
    right_tokens = normalized_c_tokens(right)
    ratio = difflib.SequenceMatcher(
        None, left_tokens, right_tokens, autojunk=False
    ).ratio()
    return {
        "left": str(left.resolve()),
        "right": str(right.resolve()),
        "left_sha256": sha256_file(left),
        "right_sha256": sha256_file(right),
        "left_tokens": len(left_tokens),
        "right_tokens": len(right_tokens),
        "token_similarity": ratio,
    }


def git_blob_oid(repository: Path, commit: str, path: str) -> str:
    """Return the blob OID for *path* in a recorded Git snapshot."""

    result = subprocess.run(
        ["git", "-C", str(repository), "ls-tree", commit, "--", path],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = result.stdout.strip().split()
    if len(fields) < 4 or fields[1] != "blob" or fields[3] != path:
        raise FileNotFoundError(f"{commit}:{path} is not a recorded blob")
    return fields[2]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two C sources for M6.0 genealogy evidence."
    )
    parser.add_argument(
        "paths",
        type=Path,
        nargs="+",
        help="One or more LEFT RIGHT path pairs.",
    )
    args = parser.parse_args()
    if len(args.paths) % 2:
        parser.error("paths must be supplied as LEFT RIGHT pairs")
    comparisons = [
        compare_sources(args.paths[index], args.paths[index + 1])
        for index in range(0, len(args.paths), 2)
    ]
    print(json.dumps(comparisons, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
