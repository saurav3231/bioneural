# PROJECT BIONEURAL

[![Creator](https://img.shields.io/badge/Creator-Saurav%20Bhandari%20%7C%20Pokhara%2C%20Nepal-blue)](https://github.com/saurav-bhandari)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Target Hardware](https://img.shields.io/badge/Hardware-RAM%20%3C%204GB%20%7C%20CPU%202.0GHz%20%7C%20Zero%20GPU-orange.svg)](docs/02_SYSTEM_REQUIREMENTS_SPECIFICATION_SRS.md)

> A fully simulated biomimetic head with an autonomous hybrid brain — uniting classical neural networks
> with the operational principles of the human brain — that behaves like a living organism.

**Creator & Principal Investigator:** Saurav Bhandari — Student, Pokhara, Nepal
**Status:** `SDLC PHASE 1 — THEORETICAL RESEARCH & ARCHITECTURE STUDY` (in progress)
**Runtime Profile:** Single-Core CPU @ 2.0 GHz · RAM < 4 GB host (active footprint < 50 MB) · Zero GPU · fully offline · no cloud · no paywall
**License:** MIT Open-Source

---

## 1. Executive Summary

PROJECT BIONEURAL is a global-standard research initiative to construct a complete computational
organism: an embodied head endowed with a *hybrid brain* that fuses the differentiable, high-level
representational power of classical neural networks with the event-driven, continuous-time dynamics of
biologically inspired spiking neural networks (SNNs) and limbic/subcortical loops. The organism exists
entirely inside software, yet behaves — sees, hears, vocalizes, and moves — as a living creature.

The head possesses **full cranial embodiment**:

| Modality | Capability | Biological Basis |
|---|---|---|
| **Vision** | Stereoscopic foveated vision | Two retinas with foveal gaze, binocular disparity |
| **Audition** | Binaural sound localization | Tonotopic cochlear analysis, ITD/ILD cues |
| **Vocalization** | Physical articulatory speech | Source–filter glottal + tract synthesis |
| **Kinematics** | 18-DOF head/facial kinematics | 3-DOF neck + binocular eyes + facial articulators, vestibulo-ocular reflex (VOR) |
| **Olfaction** | **Explicitly excluded** | Deliberate scope boundary (see FR-EXC) |

No cloud. No GPU farm. No telemetry. The entire organism — brain and body — runs **on a single-core
2.0 GHz CPU inside a hard active memory budget of less than 50 MB** (on hosts with as little as 4 GB
of RAM), driven by spiking, event-driven, sparse computation in which **fewer than 2% of neurons are
active at any instant**, mirroring cortical energetics and keeping the total system inside a
**< 15 W power envelope**.

---

## 2. Project Manifesto

Artificial intelligence today is an expensive privilege reserved for technology giants — gated behind
multi-gigawatt data centers, unspeakable energy bills, proprietary clouds, and opaque access. BIONEURAL
rejects this design.

### The Five Motives

1. **Democratization of AI (AI for Everyone).** Advanced, autonomous, *sovereign* intelligence must be
   a birthright of every person on Earth. BIONEURAL breaks the multi-million-dollar GPU monopoly and
   runs entirely offline on the entry-level computers people already own — including single-core,
   4 GB machines.
2. **Ultra-Low Resource Footprint.** The complete organism operates within minimal CPU usage and a
   strict active memory ceiling of **< 50 MB**, on a host requiring **RAM < 4 GB, a single-core
   2.0 GHz CPU, and zero GPU** — less than a single web browser tab.
3. **Environmental Sustainability.** By replacing dense matrix multiplication with biological,
   event-driven sparse computation (< 2% active neurons), BIONEURAL eliminates the energy waste,
   carbon emissions, grid strain, and cooling-water depletion of dense multi-GPU data centers within
   the ~20 W metabolic envelope of the human brain.
4. **Rejection of the Classical AI Career.** BIONEURAL rejects dense all-to-all matrix multiplication,
   brute-force scaling laws, cloud lock-in, and disembodied statistical pattern matching in favor of
   a physically grounded, resilient, event-driven alternative.
5. **Beyond-Human Hybrid Architecture.** The organism bridges continuous-time spiking dynamics with
   compact deep semantic embeddings — biological temporal efficiency plus synthetic computational
   power — for sensory speed and multi-modal integration exceeding human biological limits.

> **The Universal Access Pledge** (full text in `docs/00_PROJECT_MANIFESTO_AND_MISSION.md`):
> Zero paywalls · Zero cloud dependencies · 100% offline sovereignty · MIT for everyone.

---

## 3. Multi-Sensory (Cranial) Capability Matrix

| Domain | Capability | Mechanism | Status |
|---|---|---|---|
| **Vision** | Stereoscopic foveated vision | Binocular retinas; central fovea at high acuity + periphery low-acuity, active saccades, disparity-based depth | Phase 2 |
| **Vision** | Eye–head coordination | 18-DOF kinematic gaze alignment + vestibulo-ocular reflex (VOR) | Phase 3 |
| **Audition** | Binaural localization | Interaural time difference (ITD) + interaural level difference (ILD) | Phase 2 |
| **Audition** | Tonotopic analysis | Gammatone filterbank cochlear decomposition | Phase 2 |
| **Vocalization** | Articulatory speech | Glottal source + vocal-tract filter (Klatt/formant synthesis) | Phase 3 |
| **Kinematics** | 18-DOF head motion | 3-DOF neck (yaw/pitch/roll) + 2×2-DOF binocular eyes + 11-DOF facial articulators | Phase 3 |
| **Brain** | Hybrid substrate | Continuous deep substrate ⇄ continuous-time spiking SNN | Phase 1 |
| **Brain** | Affect & drive | Limbic/subcortical loops (reward, salience, homeostasis) | Phase 4 |
| **Olfaction** | — | **Excluded by scope (FR-EXC)** | — |

---

## 4. High-Level Architecture

```
                          ┌──────────────────────────────────────────────────┐
                          │          THE HYBRID BRAIN (cognitive + limbic)  │
                          │                                                  │
                          │   ┌──────────────────────────────────────────┐   │
                          │   │      CONTINUOUS COGNITIVE SUBSTRATE     │   │
                          │   │  (compact deep semantic embeddings,     │   │
                          │   │   planning, language, episodic memory)  │   │
                          │   └──────────────────────┬────────────────────┘   │
                          │                          │  bidirectional gating  │
                          │                          │  (rate⇄spike codec)    │
                          │   ┌──────────────────────▼────────────────────┐   │
                          │   │   NEUROMORPHIC SNN SUBSTRATE (event-driven)│  │
                          │   │  continuous-time spiking dynamics,STDP,    │  │
                          │   │  sparse (<2% active), 3-D sensory cortex   │  │
                          │   └──────────────────────┬────────────────────┘   │
                          │                          │  drive / salience      │
                          │   ┌──────────────────────▼────────────────────┐   │
                          │   │   LIMBIC / SUBCORTICAL LOOPS              │   │
                          │   │  reward prediction, valence, salience,    │   │
                          │   │  homeostasis, VOR gating, arousal         │   │
                          │   └──────────────────────┬────────────────────┘   │
                          └──────────────────────────┼────────────────────────┘
                                                     │  efference copies + commands
                          ┌──────────────────────────▼────────────────────────┐
                          │              THE EMBODIED HEAD PLANT              │
                          │  ┌──────────┐ ┌───────────┐ ┌──────────────────┐  │
                          │  │ FOVEA    │ │ COCHLEA   │ │ VOCAL TRACT      │  │
                          │  │ (stereo) │ │ (binaura) │ │ source–filter    │  │
                          │  └──────────┘ └───────────┘ └──────────────────┘  │
                          │  ┌────────────────────────────────────────────┐  │
                          │  │ 18-DOF KINEMATIC PLANT: 3-DOF neck +       │  │
                          │  │ 2×2-DOF eyes + 11-DOF facial articulators  │  │
                          │  │ (incl. VOR-stabilized binocular gaze)      │  │
                          │  └────────────────────────────────────────────┘  │
                          └──────────────────────────────────────────────────┘
```

**Information flow.** Sensory streams enter the SNN substrate as event-driven spike trains; the
continuous substrate computes abstract cognition from them; limbic loops attach valence and drive;
efference copies coordinate predictive gaze and postural control; the body actuates and generates new
sensations — a closed, situated, animate loop.

---

## 5. SDLC Status Indicator

| Phase | Name | Objective | Status |
|---|---|---|---|
| **1** | **Theoretical Research & Architecture Study** | Manifesto, SoT-A, SRS, SADD, roadmap | **IN PROGRESS** |
| 2 | Substrate Prototypes | SNN + deep codec, sensory front-ends | Planned |
| 3 | Embodied Head Plant | Vision/audition/vocal/kinematics integration | Planned |
| 4 | Hybrid Brain Integration | Limbic loops, autonomy, closed-loop behavior | Planned |
| 5 | Physical Neuromorphic Cranium | Robotics / neuromorphic port | Planned |

The companion document `docs/04_SDLC_ROADMAP_AND_MILESTONES.md` specifies the full
ISO/IEC 12207:2017 lifecycle.

---

## 6. Hardware Resource Profile

| Dimension | Requirement | Rationale |
|---|---|---|
| CPU | Entry-level **single-core 2.0 GHz** (x86-64/ARM64); no accelerator required | Ubiquitous and disposable; sovereignty for all |
| System RAM | **< 4 GB host**; **active footprint < 50 MB** (process ceiling) | Runs on entry-level laptops/systems |
| Power budget | **< 15 W** (envelope, ~20 W brain-inspired) | Fits fanless embedded / laptop operation |
| Storage | < 250 MB (model + state) | Fits SD card / flash |
| Network | **None required** | 100% offline sovereignty (NFR-SOV) |
| GPU / NPU | **Zero — explicitly not required** | No accelerator dependency (NFR-PERF) |
| Compute discipline | **Sparse event-driven; < 2% active neurons** | Biological efficiency |
| Real-time tick | < 15 ms per cognitive cycle | Meets native sensorimotor latencies |

---

## 7. Document Map

| Document | Purpose |
|---|---|
| `docs/00_PROJECT_MANIFESTO_AND_MISSION.md` | Philosophical whitepaper: the privilege gap, ecological crisis, the ~20 W biological brain, Pokhara/global accessibility vision, Universal Access Pledge |
| `docs/01_RESEARCH_FOUNDATIONS_AND_STATE_OF_THE_ART.md` | Academic research survey: classical vs. sparse AI, hybrid bridge mathematics, sensory-motor physics, comparative benchmark tables |
| `docs/02_SYSTEM_REQUIREMENTS_SPECIFICATION_SRS.md` | ISO/IEC/IEEE 29148:2018 requirements baseline (FR-* and NFR-*) |
| `docs/03_SYSTEM_ARCHITECTURE_DESIGN_DOCUMENT_SADD.md` | IEEE 42010:2011 architecture description, 4+1 views, single-core CPU/memory budgets |
| `docs/04_SDLC_ROADMAP_AND_MILESTONES.md` | ISO/IEC 12207:2017 five-phase R&D lifecycle |

---

## 8. Quick Reference

- **Repository:** `A:\bioneural` (foundational documentation — Phase 1)
- **Creator & PI:** Saurav Bhandari (Student, Pokhara, Nepal)
- **Language of record:** English (ISO/IEC/IEEE compliant templates)
- **License:** MIT Open-Source — free to use, modify, and distribute (see `LICENSE`)
- **Contact of intent:** through the issue tracker of this repository

> *"No intelligence worth keeping is the monopoly of those who can afford the fire." — Project BioNeural*