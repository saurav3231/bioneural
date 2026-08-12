# PROJECT BIONEURAL — 02: SYSTEM REQUIREMENTS SPECIFICATION (SRS)

> *Formal requirements specification. Conforms to ISO/IEC/IEEE 29148:2018 — Systems and software
> engineering — Life cycle processes — Requirements engineering.*
> Document ID BN-DOC-002, version 0.1.0, status Draft (Phase 1), classification Public/Open.
> Creator & Principal Investigator: Saurav Bhandari (Student, Pokhara, Nepal) · License: MIT Open-Source

---

## 1. Introduction

### 1.1 Purpose

This System Requirements Specification (SRS) establishes the complete, verifiable baseline of
functional and non-functional requirements for Project BioNeural: a fully simulated biomimetic head
with an autonomous hybrid brain that behaves like a living organism, running entirely offline on
entry-level single-core commodity hardware (RAM < 4 GB, CPU 2.0 GHz, zero GPU) within a strict
resource envelope.

### 1.2 Scope

The SRS covers the delivered system:

- The **Hybrid Brain** (continuous cognitive substrate, event-driven neuromorphic SNN substrate,
  limbic/subcortical loops, bidirectional coupling codecs).
- The **Embodied Head Plant** (stereoscopic foveated vision, binaural gammatone audition, articulatory
  vocal synthesis, 18-DOF head/facial kinematics with vestibulo-ocular reflex).
- The **runtime**, **persistence**, and **self-hosting** layers.
- Explicitly **excluded**: olfaction/smell sensing (FR-EXC), cloud or network dependency
  (NFR-SOV), GPU/NPU acceleration (NFR-PERF), physical fabrication of the head (deferred to
  SDLC Phase 5 and covered in `docs/04_SDLC_ROADMAP_AND_MILESTONES.md`).

### 1.3 Document conventions

- Requirements are identified by stable prefixes: `FR-` (functional) and `NFR-` (non-functional).
- Each requirement carries a unique identifier, a background/purpose rationale, and a verifiable
  criterion.
- Requirement statement verbs follow ISO/IEC/IEEE 29148:2018 modality: **shall** (mandatory),
  **should** (recommended), **may** (permitted).
- Parent-child structure: `FR-BRN-01.01` is a child of `FR-BRN-01`.

### 1.4 Intended audience

Program management, architects, developers of all subsystems, verification & validation (V&V)
engineers, reviewers, and any contributor to the R&D lifecycle (ISO/IEC 12207:2017).

### 1.5 Applicable documents

| Ref | Document |
|---|---|
| [RD-00] | `00_PROJECT_MANIFESTO_AND_MISSION.md` — mission and pledge |
| [RD-01] | `01_RESEARCH_FOUNDATIONS_AND_STATE_OF_THE_ART.md` — science baseline |
| [RD-03] | `03_SYSTEM_ARCHITECTURE_DESIGN_DOCUMENT_SADD.md` — architecture description |
| [RD-04] | `04_SDLC_ROADMAP_AND_MILESTONES.md` — lifecycle plan |
| [RD-STD-1] | ISO/IEC/IEEE 29148:2018 — Requirements engineering |
| [RD-STD-2] | ISO/IEC/IEEE 42010:2011 — Architecture description |
| [RD-STD-3] | ISO/IEC/IEEE 12207:2017 — Software life cycle processes |

---

## 2. Overall Description

### 2.1 System perspective

BioNeural is a new self-contained system with no legacy components. It is deployed as a standalone
process on a commodity computer. The **playing field**: an offline, single-user, always-available
organism; no external services are involved.

### 2.2 Operating environment

| Element | Constraint | Source |
|---|---|---|
| CPU | Entry-level **single-core x86-64/ARM64 @ ≥ 2.0 GHz** (single active core for hot path) | FR-RES / NFR-PERF |
| System RAM | Host **< 4 GB**; process resident /**active footprint < 50 MB** | FR-RES |
| GPU / NPU | **Zero required** — no accelerator dependency | NFR-PERF |
| Power | < 15 W total envelope | NFR-GREEN |
| Storage | < 250 MB persistence (non-volatile state) | NFR-PERF |
| Network | None required; all operation offline | NFR-SOV |
| OS | Any POSIX-capable or Windows host with single-process sandbox | NFR-POR |

### 2.3 User classes

1. **End-user / operator** — runs the organism locally; may never configure beyond defaults.
2. **Researcher / developer** — configures, extends, trains, and debugs substrates.
3. **V&V / acceptance team** — executes acceptance criteria in §6.

### 2.4 Design constraints derived from the mission

- **Accessibility:** runs on **RAM < 4 GB, single-core 2.0 GHz CPU, zero GPU** (Motive 1 → FR-RES).
- **Sparsity:** < 2% of neurons active at any instant (Motive 3 → FR-BRN-03).
- **Sovereignty:** zero network calls (Motive 1 → NFR-SOV).
- **Latency:** cognitive cycle < 15 ms on a 2.0 GHz single core (biological real-time behavior → NFR-PERF).

### 2.5 Assumptions and dependencies

- A.1: Offline images/audio/speech corpora are bundled at build time; no runtime fetch.
- A.2: 16–22 kHz mono/stereo audio capture is available for audition.
- A.3: Real-time thread scheduling with ≤ 1 ms jitter on the host OS is achievable outside a
  browser sandbox.
- A.4: The system is single-user per process instance.

---

## 3. Functional Requirements

### 3.1 Resource envelope — `FR-RES`

| ID | Requirement |
|---|---|
| **FR-RES-01** | The complete runtime **shall** execute on an entry-level **single-core CPU @ ≥ 2.0 GHz** (x86-64/ARM64), on a host with **RAM < 4 GB**, with **zero GPU/NPU required**. |
| **FR-RES-01.01** | The active single-core hot path (brain tick + sensorimotor loop) **shall** operate within a CPU-tick budget of **< 15 ms** wall-clock per cycle. |
| **FR-RES-02** | The process **active/resident memory footprint shall** not exceed **50 MB** total over one hour of continuous autonomy, within a host system of **< 4 GB RAM**. |
| **FR-RES-03** | The cold-start time to first responsive state **shall** be **< 5 s** on reference hardware. |
| **FR-RES-04** | Model and learned memory persistence **shall** fit within **250 MB** on non-volatile storage. |
| **FR-RES-05** | The runtime **shall** run headless (no GUI) and with a minimal monitoring UI as an option. |

### 3.2 Stereoscopic foveated vision — `FR-VIS`

| ID | Requirement |
|---|---|
| **FR-VIS-01** | The system **shall** acquire visual frames (≥ 320×240 px per eye) at a selectable rate compatible with the 15 ms tick. |
| **FR-VIS-02** | The system **shall** render a **binocular** multi-resolution foveated representation: full-resolution central foveal crops (θ_fov ≤ 15° per eye) and coarser periphery whose resolution degrades with eccentricity. |
| **FR-VIS-02.01** | Peripheral resolution **shall** step down monotonically with eccentricity (≥ 4 resolution bands). |
| **FR-VIS-02.02** | The system **shall** compute **binocular disparity** in the foveal overlap region for metric depth/stereopsis, within the single-core budget. |
| **FR-VIS-03** | The system **shall** perform active gaze control: a saccade policy selecting fixation targets from peripheral salience and an internal task value map, with vergence control between the two eyes. |
| **FR-VIS-04** | The system **shall** maintain a short-term fixation buffer (min 250 ms at foveal resolution) to support microsaccadic integration. |
| **FR-VIS-05** | The system **shall** recognize and localize ≥ 16 object categories (closed test set) with ≥ 85% top-1 accuracy at foveated framing. |
| **FR-VIS-06** | The system **shall** fuse visual fixations with vestibular signals to stabilize gaze under head motion (see FR-KIN). |

### 3.3 Binaural tonotopic audition — `FR-AUD`

| ID | Requirement |
|---|---|
| **FR-AUD-01** | The system **shall** capture stereo audio at 16–22 kHz with synchronized left/right channels. |
| **FR-AUD-02** | The system **shall** decompose each channel with a **Gammatone filterbank** covering ≥ 16 channels spanning nominal speech range (≈ 100 Hz–8 kHz). |
| **FR-AUD-03** | The system **shall** estimate azimuth from binaural cues:** ITD** by cross-correlation of low-band envelopes (≤ 1.5 kHz) and **ILD** by level ratios (≥ 1.5 kHz), fused by maximum-likelihood. |
| **FR-AUD-03.01** | Azimuth localization accuracy **shall** be **< 5° RMS** under clean anechoic test conditions. |
| **FR-AUD-04** | The system **shall** detect and classify ≥ 8 auditory event classes (speech, alarm, footsteps, etc.) with ≥ 80% accuracy. |
| **FR-AUD-05** | The system **shall** emit picrosecond-resolution spike-latency representations of auditory events to the SNN substrate (≤ 1 sample-tick latency). |

### 3.4 Articulatory vocal synthesis — `FR-VOC`

| ID | Requirement |
|---|---|
| **FR-VOC-01** | The system **shall** synthesize speech via a **source–filter** articulatory model: glottal source (F0, harmonics, jitter/shimmer) + all-pole vocal-tract filter (F1–F3 from articulatory parameters). |
| **FR-VOC-02** | The system **shall** support ≥ 128 phonemes/phones, continuous F0 (80–500 Hz), and prosody control (duration, intensity, pitch contour) for ≥ 3 expressive intentions (neutral, question, emphasis). |
| **FR-VOC-03** | Articulatory accuracy **shall** achieve **≥ 90% phoneme intelligibility** on the bundled closed corpus. |
| **FR-VOC-04** | The vocal output **shall** run at 16–22 kHz output sample rate with < 2% CPU budget. |
| **FR-VOC-05** | Production latency from cognitive intent to first audio sample **shall** be **< 30 ms**. |

### 3.5 Head and facial kinematics — `FR-KIN` (18-DOF)

| ID | Requirement |
|---|---|
| **FR-KIN-01** | The system **shall** model an **18-DOF kinematic head** comprising: **3-DOF neck** (yaw ±45°, pitch ±30°, roll ±15° about a neck pivot, with mass/inertia and velocity limits); **2×2-DOF binocular eyes** (horizontal vergence + vertical gaze per eye); and **11-DOF facial articulators** (lips, jaw, brow, eyelids, cheeks). |
| **FR-KIN-01.01** | All 18 DOF **shall** expose actuator-space limits, velocities, and inertias consistent with a biological head. |
| **FR-KIN-02** | The system **shall** implement a **vestibulo-ocular reflex (VOR)**: eye angular velocity opposite to head ω, unity gain ±5% over 0.1–5 Hz maneuvers, applied per eye with **binocular conjugacy**. |
| **FR-KIN-02.01** | Combined gaze stability under head motion (0.5 Hz, ±10°) **shall** hold residual image slip **< 2°**. |
| **FR-KIN-03** | The 11-DOF facial articulators **shall** accompany vocalization and affect expression (lips track phonemes, brows/lids track valence). |
| **FR-KIN-04** | Kinematic state **shall** couple bidirectionally with efference copies in the brain (predictive loops). |

### 3.6 The Hybrid Brain — `FR-BRN`

| ID | Requirement |
|---|---|
| **FR-BRN-01** | The system **shall** operate a **Continuous Cognitive Substrate**: differentiable deep layers providing planning, language, episodic memory, and goal representations. |
| **FR-BRN-02** | The system **shall** operate a **Neuromorphic SNN Substrate**: continuous-time spiking neurons (LIF-compliant dynamics) orchestrated by an event-driven scheduler. |
| **FR-BRN-02.01** | The SNN substrate **shall** maintain a firing population duty cycle of **< 2%** active neurons averaged over 1 s. |
| **FR-BRN-02.02** | The SNN substrate **shall** run in continuous time with per-event scheduling (no fixed dense tick for spiking). |
| **FR-BRN-03** | The system **shall** couple substrates bidirectionally via **rate ⇄ spike codecs** (Poisson/real-valued rate encode + exponential-kernel spike decode). |
| **FR-BRN-04** | The system **shall** operate **limbic/subcortical loops**: reward prediction, salience weighting, valence, arousal, and homeostasis signals modulating both substrates via neuromodulatory gains. |
| **FR-BRN-05** | The system **shall** perform **online learning**: Hebbian/STDP and reward-modulated (three-factor) plasticity in the SNN substrate, plus incremental gradient updates in the continuous substrate. |
| **FR-BRN-06** | The system **shall** maintain a persistent **episodic memory** of sensory-motor events with capacity ≥ 10⁴ episodes within the 50 MB budget (compressed representation). |
| **FR-BRN-07** | The system **shall** exhibit autonomous closed-loop behavior: foveate, localize auditory events, orient gaze toward them, and vocalize responses without external supervision. |
| **FR-BRN-08** | The hybrid brain **shall** degrade gracefully (proportional degradation, no core dump) on under-specification inputs. |

### 3.7 Exclusion of olfaction — `FR-EXC`

| ID | Requirement |
|---|---|
| **FR-EXC-01** | The system **shall not** include any chemical/olfactory sensing modality. |
| **FR-EXC-02** | The sensory interface **may** define a reserved, future external interface for a chemical peripheral, provided it requires **no** changes to the hybrid brain core interfaces. |

---

## 4. Non-Functional Requirements

### 4.1 Performance — `NFR-PERF`

| ID | Requirement |
|---|---|
| **NFR-PERF-01** | Cognitive cycle latency **shall** be **< 15 ms** p95 on reference hardware (**single-core 2.0 GHz**) for the combined brain + sensory hot path. |
| **NFR-PERF-02** | Sensory sample-to-spike latency **shall** not exceed one tick (≤ 15 ms). |
| **NFR-PERF-03** | The system **shall** meet its latency budget while occupying **no more than a single active core @ 2.0 GHz** for real-time elements (zero GPU/NPU). |
| **NFR-PERF-04** | Persistence checkpoint (snapshot) **shall** complete in **< 100 ms** without blocking cognition. |
| **NFR-PERF-05** | Memory allocator **shall** keep peak heap fragmentation growth < 5% over a 24 h session. |

### 4.2 Energy / sustainability — `NFR-GREEN`

| ID | Requirement |
|---|---|
| **NFR-GREEN-01** | The full runtime **shall** operate within a **< 15 W** power envelope on the ~20 W brain-inspired metabolic budget, on a laptop-class device. |
| **NFR-GREEN-02** | The runtime **shall** enter an idle state with ≥ 95% reduced CPU duty when unattended for > 60 s. |
| **NFR-GREEN-03** | The system **shall not** require water or active cooled data-center infrastructure by design. |

### 4.3 Sovereignty / security / privacy — `NFR-SOV`

| ID | Requirement |
|---|---|
| **NFR-SOV-01** | The runtime **shall** perform **zero network transmissions**: no socket opens, no outbound DNS, no telemetry. |
| **NFR-SOV-02** | All model weights, corpora, and memory **shall** reside and operate **locally**. |
| **NFR-SOV-03** | The system **shall** run to parity of functionality **without any cloud account, subscription, or license server**. |
| **NFR-SOV-04** | Persistent user/sensory data **shall** be stored with at-rest encryption (AES-256 or better) when enabled. |

### 4.4 Reliability — `NFR-REL`

| ID | Requirement |
|---|---|
| **NFR-REL-01** | The system **shall** sustain ≥ 24 h continuous autonomy without fatal error (mean time to failure ≥ 24 h). |
| **NFR-REL-02** | A checkpoint-recovery mechanism **shall** restore a recent consistent state after abnormal termination. |
| **NFR-REL-03** | Numerical robustness **shall** be assured across the full range of sensor inputs (no NaN/overflow halts). |

### 4.5 Portability & deployability — `NFR-POR`

| ID | Requirement |
|---|---|
| **NFR-POR-01** | The runtime **shall** build and run on at least one 64-bit Linux, one 64-bit Windows, and one ARM64 target with identical behavior. |
| **NFR-POR-02** | The system **shall** be installed from a single self-contained artifact (< 250 MB) without external dependencies. |

### 4.6 Maintainability — `NFR-MAI`

| ID | Requirement |
|---|---|
| **NFR-MAI-01** | Each subsystem **shall** expose logging and a deterministic-replay mode for debugging within the resource envelope. |
| **NFR-MAI-02** | Architecture and interfaces **shall** be documented (see `03_SYSTEM_ARCHITECTURE_DESIGN_DOCUMENT_SADD.md`) and code must build from a single command. |

### 4.7 Accessibility & openness — `NFR-OPS` (operations/ethics)

| ID | Requirement |
|---|---|
| **NFR-OPS-01** | All authored artifacts **shall** be published under the **MIT Open-Source License** (no paywall) per the Universal Access Pledge. |
| **NFR-OPS-02** | The system **shall** expose an auditable, transparent decision record (attention, gaze, fixations, vocal intents) for research and safety review. |

---

## 5. Trade-off Analysis and Priorities

| Conflict | Resolution | Guiding requirement |
|---|---|---|
| Deep fidelity vs. 50 MB active | Capability tiering: sensor-tied core under hard cap; optional extensions flagged | FR-RES-02, FR-BRN-01 |
| Cognizability on single core 2.0 GHz | Event-driven scheduler; < 2% duty; spikes aggregated per tick when cost-critical | FR-BRN-02.01 |
| Foveal accuracy vs. latency | Binocular multi-resolution pyramid + microfixation integration; foveal-only disparity | FR-VIS-02, NFR-PERF-01 |
| Vocal quality vs. memory | Compact articulatory parameters in place of sampled audio | FR-VOC-01 |
| Offline learning vs. storage | Compressed episodic memory + incremental checkpointing | FR-BRN-06, NFR-PERF-04 |

Appendix A (traceability) and verification are explicitly mapped to the architecture in the SADD.

---

## 6. Verification & Validation Requirements

Verification methods align with ISO/IEC/IEEE 29148 clauses on V&V:

- **Inspection:** code review + architecture conformance to the SADD views.
- **Analysis:** benchmark harness measuring CPU tick, RAM RSS, power, latency (p95), duty cycle.
- **Demonstration:** end-to-end autonomy scenarios (sense → attend → vocal response) in a scripted
  challenge environment.
- **Test:** closed corpora for object recognition, azimuth localization, and phoneme intelligibility.

| Traceable acceptance matrix (excerpt) | | | |
|---|---|---|---|
| Requirement | Verification | Phase | Test fixture |
| FR-RES-01/02 | Analysis | 2 | RSS/CPU benchmark |
| FR-VIS-05 | Test | 3 | Object recognition corpus |
| FR-AUD-03.01 | Test | 3 | Anechoic ITD/ILD bench |
| FR-VOC-03 | Test | 3 | Phoneme intelligibility corpus |
| FR-KIN-02 | Demonstration | 3 | VOR wobble rig |
| FR-BRN-02.01 | Analysis | 2 | Spike-duty profiler |
| FR-BRN-07 | Demonstration | 4 | Autonomous closed-loop scenario |
| NFR-SOV-01 | Analysis | 2 | Socket-emission scan |
| NFR-GREEN-01 | Analysis | 3 | Power meter |

---

## 7. Traceability

- Functional requirements trace to: Manifesto motives (RD-00), research design decisions (RD-01),
  architectural elements (RD-03), and lifecycle milestones (RD-04).
- Every FR-* and NFR-* entry is realized by at least one architectural component in RD-03 and is
  scheduled in at least one SDLC phase in RD-04.
- No requirement is orphaned; no component lacks a requirement in the Phase 1 baseline.

*End of document.*