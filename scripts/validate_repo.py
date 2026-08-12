#!/usr/bin/env python3
"""Repository integrity validation for Project BioNeural (SDLC Phase 1).

Checks that all top-level and documentation artifacts exist and that the
internal documentation references are consistent. Exits non-zero on any failure,
so it can be used in CI and pre-commit.

Run:  python scripts/validate_repo.py
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# Top-level artifacts that must exist.
REQUIRED_TOP = [
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "SUPPORT.md",
    "CHANGELOG.md",
    "CITATION.cff",
    ".gitignore",
    ".gitattributes",
]

# Formal documents expected in docs/ (metadata fields apply to these only).
REQUIRED_DOCS = [
    "00_PROJECT_MANIFESTO_AND_MISSION.md",
    "01_RESEARCH_FOUNDATIONS_AND_STATE_OF_THE_ART.md",
    "02_SYSTEM_REQUIREMENTS_SPECIFICATION_SRS.md",
    "03_SYSTEM_ARCHITECTURE_DESIGN_DOCUMENT_SADD.md",
    "04_SDLC_ROADMAP_AND_MILESTONES.md",
]

# Metadata fields every formal document must declare (case-insensitive).
META_FIELDS = ("document id", "version", "status")

# Documents referenced from README.md must exist.
REF_LINK = re.compile(r"\]\(([^)#]+)(?:#[^)]+)?\)")
DOC_REF = re.compile(r"docs/([\w.]+\.md)")


def fail(problem: str) -> int:
    print(f"[FAIL] {problem}")
    return 1


def ok(message: str) -> None:
    print(f"[ OK ] {message}")


def main() -> int:
    errors = 0

    for name in REQUIRED_TOP:
        if not (ROOT / name).exists():
            errors += fail(f"missing required top-level file: {name}")

    for name in REQUIRED_DOCS:
        path = DOCS / name
        if not path.exists():
            errors += fail(f"missing required document: docs/{name}")
        else:
            text = path.read_text(encoding="utf-8").lower()
            for field in META_FIELDS:
                if field not in text:
                    errors += fail(f"{name}: missing '{field}' metadata field")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for group in DOC_REF.findall(readme):
        if not (DOCS / group).exists():
            errors += fail(f"README references missing document: docs/{group}")

    for doc_path in DOCS.glob("*.md"):
        text = doc_path.read_text(encoding="utf-8")
        for target in REF_LINK.findall(text):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            candidate = (DOCS / target).resolve()
            root_candidate = (ROOT / target).resolve()
            if not candidate.exists() and not root_candidate.exists():
                errors += fail(f"{doc_path.name}: broken internal link -> {target}")

    if errors:
        print(f"\n{errors} validation error(s) found.")
        return 1

    ok("all required artifacts present")
    ok("all document metadata fields present")
    ok("all internal documentation links resolve")
    print("\nRepository validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
