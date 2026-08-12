<div align="center">

# 🧠 PROJECT BIONEURAL

### A fully simulated biomimetic head with an autonomous hybrid brain

Fusing **classical deep neural networks** with the **operational principles of the human brain** —
an event-driven, spiking, continuously-embodied artificial organism that runs sovereign and offline
on entry-level commodity hardware.

</div>

<p align="center">
  <a href="https://github.com/saurav3231/bioneural/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/saurav3231/bioneural/ci.yml?branch=main&label=CI&logo=github" alt="CI"></a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
  <a href="https://github.com/saurav3231/bioneural/releases">
    <img src="https://img.shields.io/github/v/release/saurav3231/bioneural?include_prereleases&label=release" alt="Release"></a>
  <a href="https://github.com/saurav3231/bioneural/issues">
    <img src="https://img.shields.io/github/issues/saurav3231/bioneural" alt="Issues"></a>
  <a href="https://github.com/saurav3231/bioneural/stargazers">
    <img src="https://img.shields.io/github/stars/saurav3231/bioneural?style=social" alt="Stars"></a>
  <a href="https://github.com/saurav3231/bioneural/blob/main/CITATION.cff">
    <img src="https://img.shields.io/badge/Citable-cff-blue.svg" alt="Citable"></a>
  <br>
  <img src="https://img.shields.io/badge/Creator-Saurav%20Bhandari%20%7C%20Pokhara%2C%20Nepal-blue.svg" alt="Creator">
  <img src="https://img.shields.io/badge/Hardware-RAM%20%3C4GB%20%7C%20CPU%202.0GHz%20%7C%20Zero%20GPU-orange.svg" alt="Target Hardware">
  <img src="https://img.shields.io/badge/Status-SDLC%20Phase%201%20%7C%20Theoretical%20Study-lightgrey.svg" alt="Status">
</p>

---

## Table of Contents

- [About](#about)
- [Cranial Capability Matrix](#cranial-capability-matrix)
- [High-Level Architecture](#high-level-architecture)
- [Hardware Resource Profile](#hardware-resource-profile)
- [The Manifesto](#the-manifesto)
- [Repository Structure](#repository-structure)
- [Documentation](#documentation)
- [Roadmap — SDLC Phases](#roadmap--sdlc-phases)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [Security](#security)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

---

## About

**PROJECT BIONEURAL** is a global-standard research initiative to build a complete computational
organism: an embodied head endowed with a **hybrid brain** that unites the differentiable,
high-level representational power of classical neural networks with the event-driven,
continuous-time dynamics of biologically inspired **spiking neural networks (SNNs)** and
limbic/subcortical loops. The organism exists entirely inside software, yet behaves — it sees, hears,
vocalizes, and moves — as a living creature.

Its mission, formalized in the [Project Manifesto](docs/00_PROJECT_MANIFESTO_AND_MISSION.md) and the
[Universal Access Pledge](docs/00_PROJECT_MANIFESTO_AND_MISSION.md#4-the-universal-access-pledge), is
founded on three motives:

1. **AI for Everyone** — break the multi-million-dollar GPU monopoly and deliver sovereign,
   high-power intelligence to anyone with an entry-level computer.
2. **Environmental Sustainability** — replace dense matrix multiplication with biological,
   event-driven sparse computation (< 2% active neurons) inside a < 15 W envelope.
3. **Beyond-Human Hybrid Architecture** — combine continuous-time spiking dynamics with compact deep
   semantic embeddings for sensory speed and multi-modal integration beyond biological limits.

> **Status:** `SDLC PHASE 1 — THEORETICAL RESEARCH & ARCHITECTURE STUDY` (in progress). This
> repository currently contains the formal research baseline: manifesto, state-of-the-art survey,
> requirements (ISO/IEC/IEEE 29148), architecture (IEEE 42010), and roadmap (ISO/IEC 12207). No
> implementation code exists yet — by design.

---

## Cranial Capability Matrix

| Modality | Capability | Biological Basis | SDLC |
|---|---|---|---|
| **Vision** | Stereoscopic foveated vision | Binocular retinas, foveal gaze, disparity depth | Phase 2 |
| **Vision** | Eye–head coordination | 18-DOF kinematic gaze + VOR reflex | Phase 3 |
| **Audition** | Binaural sound localization | Tonotopic cochlear analysis, ITD/ILD cues | Phase 2 |
| **Vocalization** | Physical articulatory speech | Source–filter glottal + vocal-tract synthesis | Phase 3 |
| **Kinematics** | 18-DOF head/facial kinematics | 3-DOF neck + binocular eyes + facial articulators | Phase 3 |
| **Brain** | Hybrid substrate | Deep substrate ⇄ continuous-time SNN | Phase 1 |
| **Brain** | Affect & drive | Limbic/subcortical loops (reward, salience) | Phase 4 |
| **Olfaction** | — | **Explicitly excluded** (FR-EXC) | — |

---

## High-Level Architecture

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                  THE HYBRID BRAIN                        │
                    │  ┌──────────────────────┐   ┌──────────────────────────┐ │
                    │  │ CONTINUOUS SUBSTRATE │   │   SNN SUBSTRATE         │ │
                    │  │ (deep embeddings,    │◄─►│ (LIF, STDP, events,     │ │
                    │  │  planning, language) │rate│  continuous time,       │ │
                    │  └──────────────────────┘⇄spike│  sparse <2% active)   │ │
                    │  ┌────────────────────────────────────────────────────┐ │
                    │  │   LIMBIC / SUBCORTICAL LOOPS (drive, salience,     │ │
                    │  │   VOR gating, homeostasis, efference copies)       │ │
                    │  └────────────────────────────────────────────────────┘ │
                    └──────────────────────────┬───────────────────────────────┘
                                               │
                    ┌──────────────────────────▼───────────────────────────────┐
                    │              THE EMBODIED HEAD PLANT                     │
                    │  ┌──────────┐  ┌───────────┐  ┌──────────────────────┐   │
                    │  │ FOVEA    │  │ COCHLEA   │  │ VOCAL TRACT          │   │
                    │  │ (stereo) │  │ (binaural)│  │ (source–filter voice)│   │
                    │  └──────────┘  └───────────┘  └──────────────────────┘   │
                    │  18-DOF PLANT: 3-DOF neck + 2×2-DOF eyes + 11-DOF face   │
                    └────────────────────────────────────────────────────────┘
```

Full formal treatment: [System Architecture Design Document](docs/03_SYSTEM_ARCHITECTURE_DESIGN_DOCUMENT_SADD.md).

---

## Hardware Resource Profile

BioNeural targets hardware people **already own** — the entire organism, brain and body, in less
memory than a single browser tab.

| Dimension | Requirement |
|---|---|
| CPU | Entry-level **single-core 2.0 GHz** (x86-64 / ARM64) |
| System RAM | **< 4 GB** host · **< 50 MB** active footprint |
| GPU / NPU | **Zero** — not required, not used |
| Power | **< 15 W** envelope (brain-inspired ~20 W metabolism) |
| Storage | < 250 MB (model + state) |
| Network | **None** — 100% offline sovereignty |
| Real-time tick | < 15 ms per cognitive cycle |

---

## The Manifesto

> *"No intelligence worth keeping is the monopoly of those who can afford the fire."*

Project BioNeural is a declaration of independence from centralized, cloud-locked, environmentally
destructive AI. It is engineered so that **sovereignty is a physical property of the artifact**:
zero paywalls, zero cloud dependencies, 100% offline operation, under the **MIT License** — for every
person on Earth. Read the full manifesto:
[`docs/00_PROJECT_MANIFESTO_AND_MISSION.md`](docs/00_PROJECT_MANIFESTO_AND_MISSION.md).

---

## Repository Structure

```
bioneural/
├── .github/                  # Issue/PR templates, CI workflow, community health files
├── docs/                     # Formal research & engineering baseline (Phase 1)
│   ├── 00_PROJECT_MANIFESTO_AND_MISSION.md
│   ├── 01_RESEARCH_FOUNDATIONS_AND_STATE_OF_THE_ART.md
│   ├── 02_SYSTEM_REQUIREMENTS_SPECIFICATION_SRS.md
│   ├── 03_SYSTEM_ARCHITECTURE_DESIGN_DOCUMENT_SADD.md
│   └── 04_SDLC_ROADMAP_AND_MILESTONES.md
├── scripts/                  # Repository validation & tooling
├── src/bioneural/            # Package scaffold (implementation begins Phase 2)
└── tests/                    # Test suite (Phase 1: package metadata smoke test)
├── CITATION.cff              # Academic citation metadata
├── LICENSE                   # MIT Open-Source License
└── pyproject.toml            # Package metadata (author: Saurav Bhandari)
```

---

## Documentation

| Document | Description |
|---|---|
| [00 — Project Manifesto & Mission](docs/00_PROJECT_MANIFESTO_AND_MISSION.md) | Philosophical whitepaper: the privilege gap, ecological crisis, ~20 W brain, Universal Access Pledge |
| [01 — Research Foundations & SoT-A](docs/01_RESEARCH_FOUNDATIONS_AND_STATE_OF_THE_ART.md) | Dense AI vs. sparse SNNs, hybrid bridge mathematics, sensory-motor physics, benchmarks |
| [02 — System Requirements (SRS)](docs/02_SYSTEM_REQUIREMENTS_SPECIFICATION_SRS.md) | ISO/IEC/IEEE 29148 formal requirements (FR-\*/NFR-\*) |
| [03 — Architecture (SADD)](docs/03_SYSTEM_ARCHITECTURE_DESIGN_DOCUMENT_SADD.md) | IEEE 42010 architecture description, 4+1 views, budget tables |
| [04 — SDLC Roadmap & Milestones](docs/04_SDLC_ROADMAP_AND_MILESTONES.md) | ISO/IEC 12207 five-phase R&D lifecycle |

See the [docs index](docs/README.md) for an overview.

---

## Roadmap — SDLC Phases

| Phase | Name | Status |
|---|---|---|
| **1** | **Theoretical Research & Architecture Study** | **In progress** |
| 2 | Substrate Prototypes (SNN + deep, sensory front-ends) | Planned |
| 3 | Embodied Head Plant (vision, audition, voice, 18-DOF) | Planned |
| 4 | Hybrid Brain Integration (limbic loops, autonomy) | Planned |
| 5 | Physical Neuromorphic Cranium | Planned |

---

## Getting Started

> **Note:** This is Phase 1 — a documentation/research baseline. There is no executable code yet.
> The full 5-phase lifecycle is defined in
> [`docs/04_SDLC_ROADMAP_AND_MILESTONES.md`](docs/04_SDLC_ROADMAP_AND_MILESTONES.md).

```console
git clone https://github.com/saurav3231/bioneural.git
cd bioneural
# Phase 1: repository validation
python scripts/validate_repo.py
```

Python requirement: **3.9+** (validated against the metadata in `pyproject.toml`).

---

## Contributing

Contributions are welcome — research feedback, requirement clarifications, architecture review, and
(Phase 2 onward) code. Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** and
**[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** before opening an issue or pull request.

- Found a defect or gap? [Open an issue](https://github.com/saurav3231/bioneural/issues/new/choose).
- Ready to help? Read the [roadmap](docs/04_SDLC_ROADMAP_AND_MILESTONES.md) and pick an open task.

---

## Security

Please report security vulnerabilities privately per our
**[SECURITY.md](SECURITY.md)** responsible-disclosure policy. Do **not** open a public issue for
security defects.

---

## License

This project is licensed under the **MIT License** —
© 2026 Saurav Bhandari (Pokhara, Nepal) & Project BioNeural Consortium.
See [LICENSE](LICENSE) for the full text.

---

## Citation

If you use BioNeural research in academic work, please cite it:

```bibtex
@software{bioneural,
  author  = {Bhandari, Saurav and {Project BioNeural Consortium}},
  title   = {Project BioNeural: A Biomimetic Head with an Autonomous Hybrid Brain},
  year    = {2026},
  url     = {https://github.com/saurav3231/bioneural},
  license = {MIT}
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff).

---

## Acknowledgments

- **Saurav Bhandari** — Creator & Principal Investigator (Pokhara, Nepal)
- The global open-source and neuromorphic computing research communities that make sovereign AI
  possible.

---

<p align="center">
  <sub>Project BioNeural is dedicated to the people of Pokhara, Nepal, and to every learner in the
  world who deserves intelligence they can hold in their own hands.</sub>
</p>
