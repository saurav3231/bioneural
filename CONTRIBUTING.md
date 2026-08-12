# Contributing to Project BioNeural

Thank you for your interest in contributing! Project BioNeural is an open, sovereign research
initiative. This document describes how to contribute at every stage of the project — currently
**SDLC Phase 1 (Theoretical Research & Architecture Study)**.

## Code of Conduct

All contributors are expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md). Be respectful,
scientific, and constructive.

## How to Contribute

### 1. Report an Issue

Use the [issue templates](https://github.com/saurav3231/bioneural/issues/new/choose):

- **Bug report** — something is incorrect, inconsistent, or broken (including documentation defects).
- **Feature request** — a capability, requirement, or research topic you believe is missing.
- **Security report** — see [SECURITY.md](SECURITY.md). Do **not** open a public issue.

Good issues include:
- A clear title and one problem per issue.
- The affected file(s) and section(s).
- Expected vs. actual behavior, with evidence (references, mathematics, measurements).

### 2. Discuss First (for design/spec changes)

The repository is governed by formal standards documents (ISO/IEC/IEEE 29148, 42010, 12207).
Before proposing changes to requirements or architecture, open a discussion/issue describing the
change, its rationale, and its impact on the [traceability chain](docs/02_SYSTEM_REQUIREMENTS_SPECIFICATION_SRS.md#7-traceability).

### 3. Implement (Phase 2 onward)

| Phase | What can be contributed |
|---|---|
| **Phase 1 (current)** | Research review, mathematics, requirement analysis, documentation quality |
| Phase 2+ | Code: substrates, codecs, sensory front-ends, runtime |

#### Development environment

```console
git clone https://github.com/saurav3231/bioneural.git
cd bioneural
python -m venv .venv            # Python 3.9+
# On Windows: .venv\Scripts\activate
# On Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

#### Committing

This project uses [Conventional Commits](https://www.conventionalcommits.org/):

```text
feat: add LIF neuron core (Phase 2)
docs: clarify ITD/ILD fusion in SRS (FR-AUD-03)
fix: correct execution budget table in SADD
test: add gammatone filterbank unit tests
```

Branch naming: `feat/<topic>`, `docs/<topic>`, `fix/<topic>`.

### 4. Open a Pull Request

- Use the [pull request template](https://github.com/saurav3231/bioneural/blob/main/.github/PULL_REQUEST_TEMPLATE.md).
- Keep PRs small and focused; reference the related issue.
- Ensure CI passes (see below).
- For Phase 1 documentation changes, clearly describe the standard-compliance impact.

## Quality Gates

The repository uses continuous integration (`.github/workflows/ci.yml`) to enforce:

- **Repository integrity** — all top-level and documentation artifacts exist and are internally
  consistent (`python scripts/validate_repo.py`).
- **Formatting** — trailing whitespace, missing newlines, large binary files (`pre-commit`).
- **Markdown linting** — consistent, standards-compliant documentation (`markdownlint`).

Run locally:

```console
python scripts/validate_repo.py
pre-commit run --all-files
```

No contribution is merged with failing gates.

## Getting Help

- Documentation index: [`docs/README.md`](docs/README.md)
- Roadmap: [`docs/04_SDLC_ROADMAP_AND_MILESTONES.md`](docs/04_SDLC_ROADMAP_AND_MILESTONES.md)
- Questions: open a discussion in the repository's Issues/Discussions area.

---

The work you contribute helps make sovereign, offline, environment-friendly AI a reality for
everyone. Thank you.

— Project BioNeural
