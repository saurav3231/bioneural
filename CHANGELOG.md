# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned (Phase 2+)
- SNN substrate prototype (LIF core, event scheduler, STDP).
- Continuous cognitive substrate + rate⇄spike codecs.
- Sensory front-ends: binocular foveation, gammatone + ITD/ILD.
- Runtime governor, checkpointing, deterministic replay.

## [0.1.0-alpha] — 2026-08-12

### Added
- **P1-baseline**: Formal Phase 1 documentation baseline committed by the creator,
  Saurav Bhandari (Pokhara, Nepal).
- `README.md` — project landing page, manifesto, capability matrix, architecture, hardware profile.
- `docs/00_PROJECT_MANIFESTO_AND_MISSION.md` — philosophical whitepaper and Universal Access Pledge.
- `docs/01_RESEARCH_FOUNDATIONS_AND_STATE_OF_THE_ART.md` — academic state-of-the-art survey
  (dense AI vs. sparse SNNs, hybrid bridge, sensory-motor physics, benchmarks).
- `docs/02_SYSTEM_REQUIREMENTS_SPECIFICATION_SRS.md` — ISO/IEC/IEEE 29148 requirements baseline.
- `docs/03_SYSTEM_ARCHITECTURE_DESIGN_DOCUMENT_SADD.md` — IEEE 42010 architecture description
  with real-time budget tables.
- `docs/04_SDLC_ROADMAP_AND_MILESTONES.md` — ISO/IEC 12207 five-phase lifecycle.
- `pyproject.toml` — package metadata (author: Saurav Bhandari).
- `LICENSE` — MIT Open-Source License.
- `CITATION.cff`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`, `SUPPORT.md`.
- GitHub community files: issue/PR templates, CI workflow, repository validation script.

[Unreleased]: https://github.com/saurav3231/bioneural/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/saurav3231/bioneural/tree/v0.1.0-alpha
