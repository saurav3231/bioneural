# PROJECT BIONEURAL — 04: SDLC ROADMAP AND MILESTONES

> *Lifecycle programme plan. Conforms to ISO/IEC/IEEE 12207:2017 — Systems and software
> engineering — Software life cycle processes.*
> Document ID BN-DOC-004, version 0.1.0, status Draft (Phase 1), classification Public/Open.
> Creator & Principal Investigator: Saurav Bhandari (Student, Pokhara, Nepal) · License: MIT Open-Source

---

## 1. Purpose and Scope

This document defines the five-phase research and development lifecycle for Project BioNeural,
mapping ISO/IEC 12207:2017 process groups onto concrete phases, milestones, gates, work products,
and exit criteria. It is the master schedule against which Phase 2–5 execution is planned and is
used to derive work breakdown structures (WBS), staffing, and acceptance reviews.

### 1.1 ISO/IEC 12207:2017 process map (what we deliberately apply)

| Process Group (12207:2017) | Program usage |
|---|---|
| Agreement | Universal Access Pledge licensing posture; contribution CLA |
| Organizational project-enabling | Governance, open community, metrics, risk, infrastructure |
| Technical management | Planning, decision analysis, risk, measurement, quality, change control, reviews |
| Technical (left side) | Business/mission analysis; stakeholder needs/requirements definition (→ SRS); system requirements definition; architecture definition (→ SADD); design; implementation; integration; verification; transition; validation; operation; maintenance; disposal |

Each phase of §3 maps the applicable technical processes to concrete deliverables.

---

## 2. Horizon & Guiding Principles

- **5 phases**, ending in a Physical Neuromorphic Cranium.
- **Phase gating:** no phase advances without exit criteria met (checklist per phase).
- **Budget discipline** (FR-RES, NFR-GREEN, NFR-PERF) is continuously instrumented from Phase 2
  onward, not retrofitted.
- **Continuous verification:** benchmarks, rejection suites, and determinism checks run in CI from
  Phase 2.
- **Open release cadence:** every phase produces a public, sovereign artifact.

---

## 3. The Five-Phase Lifecycle

### PHASE 1 — THEORETICAL RESEARCH & ARCHITECTURE STUDY (Current)

**Objective.** Lock the scientific foundation, requirements, architecture, and programme plan through
pure theoretical research — no implementation code. Binds the programme to the accessibility envelope
(**RAM < 4 GB, single-core 2.0 GHz CPU, zero GPU, < 50 MB active footprint**).

| Milestone | Deliverable(s) | Exit criteria |
|---|---|---|
| M1.1 Mission baseline | Manifesto (BN-DOC-000) ratified | Pledge + motives approved; creator identity recorded |
| M1.2 Research baseline | Foundations & SoT-A paper (BN-DOC-001) reviewed | Peer-review gate passed |
| M1.3 Requirements baseline | SRS (BN-DOC-002) approved | 100% FR/NFR triaged; no orphans |
| M1.4 Architecture baseline | SADD (BN-DOC-003) approved | Budget grid §4 validated by analysis |
| M1.5 Lifecycle plan | Roadmap (BN-DOC-004) approved | Phase gating avowed by all leads |
| M1.6 Accessibility proof | Analytical feasibility on single-core 2.0 GHz, < 4 GB RAM | Budget math confirms < 50 MB active + < 15 ms tick |

**Phase 1 exit gate:** all M1.x ✓; risk register opened; Phase 2 WBS released.

---

### PHASE 2 — SUBSTRATE PROTOTYPES

**Objective.** Stand up each brain substrate and the sensory codecs, measurable against the budget
grid.

| Milestone | Deliverable(s) | Exit criteria |
|---|---|---|
| M2.1 SNN core | LIF pools + event scheduler + STDP (benchmarked) | <2% duty; <3.5 ms per tick |
| M2.2 Continuous core | Small deep substrate + quantized planner | ≤16 MB whole-model RAM |
| M2.3 Codec | rate⇄spike interfaces; replay determinism | Proven on S1-style event flow |
| M2.4 Sensory front-ends | Binocular fovea pyramid goal design; gammatone+ITD/ILD | Vision 2 ms; audio 1.5 ms budget met |
| M2.5 Runtime governor | Tick accounting, checkpoint, replay, socket lock | 15 ms p95; zero-socket scan clean |
| M2.6 Phase gate code | Full instrumentation harness | Green budget + regression suite |

**Phase 2 exit gate:** integrated substrate demo (headless) holding all NFR budget cells at
≥ 90% of target; CI green.

---

### PHASE 3 — EMBODIED HEAD PLANT

**Objective.** Attach full embodiment: stereoscopic vision, audition, vocalization, 18-DOF kinematics
and VOR; validate closed sensory-motor coupling.

| Milestone | Deliverable(s) | Exit criteria |
|---|---|---|
| M3.1 Vision integration | Binocular foveated camera pipeline + saccade/vergence policy live | ≥85% object recognition; 2 ms budget |
| M3.2 Audition integration | Stereo gammatone + ITD/ILD fuse live | <5° RMS; event->salience latency ≤1 tick |
| M3.3 Vocal tract | Source–filter synthesizer live | ≥90% phoneme intelligibility; ≤30 ms intent→audio |
| M3.4 Kinematics + VOR | 18-DOF plant (neck/eyes/face) + binocular VOR filter live | <2° residual slip under 0.5 Hz ±10° |
| M3.5 Startle scenario | S1 (auditory startle → oriented gaze + vocal) | End-to-end closed loop, no supervision |
| M3.6 Validation | Scenario suite S1–S3 green | Power <15W; active RAM <50 MB on <4GB host |

**Phase 3 exit gate:** organism senses, localizes, and moves/vocalizes without external hinting.

---

### PHASE 4 — HYBRID BRAIN INTEGRATION & AUTONOMY

**Objective.** Unite substrates into full hybrid cognition: limbic loops, episodic memory, online
learning, autonomous closed-loop behavior.

| Milestone | Deliverable(s) | Exit criteria |
|---|---|---|
| M4.1 Limbic loops | Reward/valence/salience/arousal/homeostasis modules | Drive shapes behavior in S3/S4 |
| M4.2 Memory | Episodic compression; scenario replay | ≥10⁴ episodes; <100 ms checkpoint |
| M4.3 Online learning | STDP + three-factor + incremental gradient live | Stability within tick budget |
| M4.4 Autonomy suite | Seek&speak, homeostasis, memory CRUD scenarios | All S-scenarios (S1–S5) pass |
| M4.5 Long-run soak | 24 h continuous autonomy | MTTF ≥24 h; RSS drift <5% |

**Phase 4 exit gate:** a living, self-sustaining digital organism with affect and memory, running
sovereign on commodity hardware.

---

### PHASE 5 — PHYSICAL NEUROMORPHIC CRANIUM

**Objective.** Port the hybrid brain and head plant to physical neuromorphic hardware and a physical
/or animatronic cranium, while preserving behavioral identity with the simulation.

| Milestone | Deliverable(s) | Exit criteria |
|---|---|---|
| M5.1 Interface hardening | Neuromorphic target bound via ADR-7 interfaces | Bit-parity or equivalence harness green |
| M5.2 SNN on neuromorphic | SNN substrate mapped (Loihi-class or fpga SNN) | Duty & latency meet target on silicon |
| M5.3 Cranium integration | Physical cranium: face, neck, audio transducers, eyes | Mechanical + audio calibration pass |
| M5.4 Behavioral equivalence | Sim↔physical behavioral parity suite | ≥90% scenario equivalence |
| M5.5 Release & upkeep | Maintenance handover; disposal/upgrade pipeline | 12207 maintenance process documented |

**Phase 5 exit gate:** physical organism performs the embodied autonomy suite with behavioral parity
to the sovereign simulation; open-license release.

---

## 4. Master Schedule (indicative, calendar quarters)

| Phase | Focus | Target window (relative) | Gate |
|---|---|---|---|
| 1 Theoretical Research & Architecture Study | Manifesto, SoT-A, SRS, SADD, roadmap | Q1 | Study gate |
| 2 Substrate Prototypes | Brain + codecs | Q2–Q3 | Substrate gate |
| 3 Embodied Head Plant | Senses + voice + 18-DOF kinematics | Q3–Q4 | Embodiment gate |
| 4 Hybrid Integration | Limbic, memory, autonomy | Q4–Q6 | Autonomy gate |
| 5 Neuromorphic Cranium | Silicon + physical | Q6–Q8 | Release gate |

Windows are indicative; gates are decision criteria, not dates.

---

## 5. Verification & Validation Strategy (12207)

| Process | Phase | Action |
|---|---|---|
| Verification | 2–5 | Unit/benchmarks in CI; replay determinism on 2.0 GHz single-core reference rig |
| Validation | 3–5 | Scenario suite S1–S5; power/water/adherence audit |
| Acceptance | each gate | Formal gate review per §4 |
| Measurement/analysis | 2–5 | RSS, tick-p95, duty, power, WER-subscores logged |

### 6. Risk Register (top 8)

| # | Risk | Likelihood | Impact | Mitigation | Phase |
|---|---|---|---|---|---|
| R1 | Budget grid overshoots (SNN < 2%) | Med | High | Event-scheduler optimizations; duty governor | 2 |
| R2 | ITD accuracy at 16 kHz SNR | Med | Med | Upsampling; fusion MLE; adaptive threshold | 3 |
| R3 | Phoneme intelligibility under prosody | Med | Med | Databurst corpus; formant mapping ML | 3 |
| R4 | VOR phase lag at 5 Hz | Low | Med | Predictor + corner filter; higher vestibular rate | 3 |
| R5 | Memory leak / fragmentation | Med | Med | Arena allocator; 24 h soak (M4.5) | 2–4 |
| R6 | Online learning destabilizes behavior | Med | High | Safety limits; snapshot rollback; conservative STDP | 4 |
| R7 | SoT-A supersession | Low | Med | Quarterly literature watch; ADR re-review | all |
| R8 | Physical cranium supply/cost | Med | Med | Mixed-reality physical option; open-hardware BOM | 5 |

---

## 7. Governance & Compliance

- Phase gates chaired by program + V&V lead; decision log per ISO 12207 decision-management.
- Universal Access Pledge commitments (zero paywall **— MIT license —**, zero cloud) are contractually
  inherited by every release and milestone (NFR-OPS-01).
- Continual improvement + change control on requirements and architecture, traced through RD-00 →
  RD-04.

*End of document.*