# PROJECT BIONEURAL — 03: SYSTEM ARCHITECTURE DESIGN DOCUMENT (SADD)

> *Formal architecture description. Conforms to ISO/IEC/IEEE 42010:2011 — Systems and software
> engineering — Architecture description.*
> Document ID BN-DOC-003, version 0.1.0, status Draft (Phase 1), classification Public/Open.
> Creator & Principal Investigator: Saurav Bhandari (Student, Pokhara, Nepal) · License: MIT Open-Source

---

## 1. Introduction

### 1.1 Purpose

This System Architecture Design Document (SADD) defines the architecture for Project BioNeural: the
hybrid-brained biomimetic head. It adopts the **4+1 architectural views** model subsumed under
ISO/IEC/IEEE 42010:2011, presents block diagrams, and provides a **real-time single-core CPU
execution budget** that ties every subsystem to its millisecond and memory allocation — enforcing the
requirements of the SRS (`02_SYSTEM_REQUIREMENTS_SPECIFICATION_SRS.md`).

### 1.2 Architecture framework (42010:2011 mapping)

| 42010 element | This document |
|---|---|
| Architecture description | Entire SADD |
| Stakeholders and concerns | §2 (concerns), §8 (stakeholders) |
| Architecture viewpoints | §3 (4+1 views) |
| Architecture views | §3.1–3.5 |
| Architecture models | Diagrams, budgets, tables |
| Architecture rationales | Constraints, ADRs (decisions) |
| Architecture elements | Subsystem interfaces, budgets |

### 1.3 Conformance

The architecture **shall** realize every FR-*/NFR-* of the SRS; traceability table provided in §9.

---

## 2. Stakeholders, Concerns, and Constraints

| Stakeholder | Key concerns |
|---|---|
| Program leadership | Mission delivery, sovereignty pledge, sustainability, cost |
| Brain developers | Coupling fidelity, sparse execution, online learning |
| Sensory developers | Foveation latency, binaural precision, VOR stability |
| Embedded/firmware | < 4 GB host RAM, single-core hot path, power envelope, < 50 MB active |
| V&V engineers | Traceability, budget conformance, repeatability |
| End-users | Offline autonomy, instantaneous state, privacy |

**Constraining requirements:** FR-RES-01 (single-core 2.0 GHz, < 4 GB RAM, zero GPU), FR-RES-02
(< 50 MB active), NFR-PERF-01 (< 15 ms tick), FR-BRN-02.01 (< 2% sparsity), NFR-SOV-01 (zero
sockets), NFR-GREEN-01 (< 15 W).

---

## 3. The 4+1 Architecture Views

### 3.1 Logical View (component model)

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                      HYBRID BRAIN                        │
                    │  ┌──────────────────────┐   ┌──────────────────────────┐ │
                    │  │ CONTINUOUS SUBSTRATE │   │   SNN SUBSTRATE         │ │
                    │  │  - planner/cognition │◄─►│ - LIF neuron pools      │ │
                    │  │  - language          │rate│ - STDP synapses        │ │
                    │  │  - episodic memory   │⇄spike│ - event scheduler      │ │
                    │  │  - goals/values      │  │  - sensory cortex maps   │ │
                    │  └──────────┬───────────┘   └──────────┬───────────────┘ │
                    │             │  rate⇄spike codecs       │                 │
                    │  ┌──────────▼──────────────────────────▼───────────────┐ │
                    │  │          LIMBIC / SUBCORTICAL LOOPS                 │ │
                    │  │  reward, salience, valence, arousal, homeostasis,   │ │
                    │  │  VOR gating, efference-copy management              │ │
                    │  └──────────────────────────┬──────────────────────────┘ │
                    └─────────────────────────────┼────────────────────────────┘
                                                  │  commands + sensory data
                    ┌─────────────────────────────┼────────────────────────────┐
                    │             EMBODIED HEAD PLANT  (see §3.1.1)            │
                    └──────────────────────────────────────────────────────────┘
```

#### 3.1.1 Head Plant subsystem tree

```
HEAD PLANT (18-DOF)
├── SENSORS
│   ├── Vision         Binocular foveated camera seam → multi-resolution pyramids → disparity + salience
│   ├── Audition       L/R gammatone banks → envelope; ITD/ILD estimators → event azimuth
│   └── Vestibular     Angular/linear inertial estimates (for VOR)
├── EFFECTORS
│   ├── Vocal tract    Source (glottal) + filter (formants F1–F3) → 16–22 kHz output
│   ├── Neck (3-DOF)   Yaw/pitch/roll actuation (modeled plant, no physical motor)
│   ├── Eyes (2×2-DOF) Vergence + vertical gaze per eye (VOR-stabilized)
│   └── Face (11-DOF)  Lips, jaw, brow, eyelids, cheeks articulators (visual avatar / plot)
└── BRIDGE
    ├── Rate⇄Spike codecs above, plus sensor preprocessing and efference copies
```

#### 3.1.2 Key interfaces

| Interface | From → To | Data |
|---|---|---|
| IF-SPK | SNN substrate → Continuous substrate | Decoded rate vectors r̂(t) |
| IF-RAT | Continuous substrate → SNN substrate | Rate commands r(t) → spikes |
| IF-LIM | Limbic loops → both substrates | Neuromodulatory gain scalars (μ, dopamine-like δ) |
| IF-SEN | Sensory front-ends → SNN substrate | Spike-latency event streams |
| IF-EFF | Brain → head effectors | Motion/speech commands + efference copies |
| IF-PER | Any → storage | Compressed episodic + weights checkpoints |

---

### 3.2 Development View (module organization)

```
bioneural/
├── brain/
│   ├── continuous/   (deep substrate: layers, optimizer, memory tier)
│   ├── snn/          (LIF core, synapse/asynapse, event scheduler, STDP)
│   ├── codec/        (rate⇄spike bidirectional interfaces)
│   └── limbic/       (reward, valence, salience, arousal, homeostasis)
├── senses/
│   ├── vision/       (foveation, pyramid, saccade policy)
│   ├── audio/        (gammatone, ITD/ILD, event detect)
│   └── vestibular/   (VOR estimator)
├── body/
│   ├── vocal/        (source–filter synthesizer)
│   ├── kinematics18/ (3-DOF neck + 2×2-DOF eyes + 11-DOF face)
│   └── face/         (articulator mapping)
├── runtime/          (tick scheduler, budget governor, checkpoint, logging, replay)
├── storage/          (episodic store, weight snapshots, at-rest crypto)
└── test/             (benchmarks, closed corpora, scenario harness)
```

Build: single command; runtime single self-contained artifact (assignment NFR-POR-01/02).

---

### 3.3 Process View (runtime concurrency & tick)

The real-time core runs **one active thread** (hot path) to honor NFR-PERF-03; auxiliary, non-timely
service such as persistence, logging, and rendering run on background threads with strictly lower
priority and zero blocking of the hot path.

```
 [15 ms cognitive tick]  ───────────────────────────────────────────────────┐
                                                                             │
  t=0      SENSOR LATCH        vision frame      audio frame     vest est    │
  t=1→2    SENSOR PREPROCESS   fovea pyramid     gammatone+ITD/ILD          │
  t=3      SPIKE DELIVERY      sensory events → SNN event scheduler          │
  t=4→7    SNN STEP (event)    parallelizable pools, <2% duty, continuous-time│
  t=6→8    CODED UPLIFT        SNN spikes → rate r̂(t) → continuous substrate │
  t=8→11   COGNITION           plan, attention, language, goal update        │
  t=11→12  LIMBIC UPDATE       reward/salience/valence re-weight             │
  t=12→13  COMMAND DROP        efference copy, saccade, vocal intent, neck   │
  t=13→14  ACTUATION COMMIT    neck/face/vocal output committed              │
  t=14→15  BOOKKEEPING         budget accounting → governor                  │
```

Guarantees: no epoch, no fixed dense sweep; the SNN scheduler only touches events. Budget governor
drops lowest-priority model refresh when the tick budget is threatened.

**Concurrency guarantees.** Shared memory between hot path and background is lock-free (single
producer/consumer triple-buffer per stream); hot-path allocations limited to a preallocated arena.

---

### 3.4 Physical / Deployment View

```
┌────────────────────────────────────────────────────────────────┐
│  ENTRY-LEVEL HOST (< 4 GB RAM, single-core 2.0 GHz, zero GPU)  │
│                                                                │
│  CPU (single core 2.0 GHz) ─► [HOT PATH: brain tick <15 ms,    │
│                                sensorimotor loop]              │
│  CPU (spare cores, if any)  ─► [BACKGROUND: persistence, log]  │
│  RAM < 4 GB                ─► [ARENA ≤ 50 MB active: static +  │
│                                pool, no unbounded JIT alloc;   │
│                                model weights mmap-hinted]      │
│  GPU / NPU                 ─►  ❌ not required, not used       │
│  NETWORK                   ─►  ❌ sockets disabled (NFR-SOV-01) │
│  STORAGE                   ─►  < 250 MB snapshots + corpora    │
│  SENSORS                   ─►  USB cam x2 (optional), mic, IMU │
│  OUTPUT                    ─►  headless LPCM / VNC-style UI opt│
└────────────────────────────────────────────────────────────────┘
```

Platforms: single-core-capable Linux x86-64, Windows x86-64, Linux/FreeBSD ARM64 (NFR-POR-01).

---

### 3.5 Scenarios (+1)

| Scenario | Behavior exercised | Requirement |
|---|---|---|
| S1 *Startle* | Impulse sound from left → ITD/ILD event → SNN salience → orient 3-DOF neck + binocular gaze (VOR-stabilized); minimal vocal surprise. | FR-AUD; FR-KIN; FR-BRN-07; FR-BRN-04 |
| S2 *Seek & speak* | Off-screen target found by periphery → binocular foveated fixation + disparity depth; language command → articulatory speech with affect contour. | FR-VIS-03; FR-VOC-01/02; FR-BRN-07 |
| S3 *Homeostasis* | Battery/low-input energy profile → limbic arousal floor → reduced activity; checkpoints; returns to full goal pursuit after refill. | FR-BRN-04; NFR-GREEN-02 |
| S4 *CRUD memory* | Episode salience promotion/demotion; compression replay without breaking tick budget. | FR-BRN-06; NFR-PERF-04 |
| S5 *Offline audit* | Socket scan proves zero network; replay log reconstructs every decision. | NFR-SOV-01; NFR-OPS-02 |

---

## 4. Real-Time Single-Core CPU Execution Budget

Top-level allocation of a **15 ms** cognitive tick (target p95) on **one core @ 2.0 GHz**, plus
memory (active < 50 MB; host < 4 GB). All values are Phase 1 planning targets; actuals measured at
Phase 2 gate.

### 4.1 Per-subsystem budget (milliseconds and memory)

| Subsystem | Function | CPU budget (ms) | % tick | RAM (MB) | Notes |
|---|---|---|---|---|---|
| Sensor latch + preprocess (vision) | Frame capture, fovea pyramid, salience | 2.0 | 13.3% | 6.0 | Multi-res buffers, ring history |
| Audition front-end | Gammatone 2×16ch, ITD/ILD, event detect | 1.5 | 10.0% | 4.0 | Includes mic buffers |
| Vestibular estimator | VOR ω estimate, drift correction | 0.2 | 1.3% | 0.2 | |
| SNN substrate | Event scheduler, LIF pools, STDP | 3.5 | 23.3% | 12.0 | <2% duty; event-only |
| Rate⇄spike codec | Encode r→spikes; decode spikes→r̂ | 0.6 | 4.0% | 1.0 | Lookup tables |
| Continuous substrate | Deep fwd/update, planner, language | 3.0 | 20.0% | 16.0 | Small model, quantized |
| Limbic loops | Reward/valence/salience update | 0.5 | 3.3% | 1.5 | |
| Command & actuation | Neck/face/vocal intent commit, efference | 1.0 | 6.7% | 2.5 | Vocal synth off-tick |
| Vocal synthesis (streaming) | Source–filter on ac3d grid, waterfall to lpcm | 0.5 | 3.3% | 1.5 | Runs per audio block |
| Episodic memory | Compress/promote/replay op | 0.4 | 2.7% | 3.0 | Defer heavy ops to bg |
| Government/budgeting | Accounting, governor, logging | 0.3 | 2.0% | 0.3 | |
| Background (non-hot) | Persistence, render, audit | (off-core) | — | 2.0 | Async, non-blocking |
| **Total hot path** | — | **13.5 ms** | **90%** | **48.0 MB** | **Headroom 1.5 ms (10%)** |
| | | | | | ≤ 50 MB enforced |

### 4.2 Budget governance rules

1. No subsystem may exceed its allocation without governor approval in a change request.
2. If tick exceeds 15 ms p95 for 3 consecutive cycles → governor demotes lowest-priority refresh,
   then alerts (NFR-REL).
3. Memory arena: static segment + preallocated pools; no unbounded dynamic allocation on hot path
   (NFR-PERF-05).

---

## 5. Key Design Rationale & ADRs

| # | Decision | Rationale | Alternate rejected |
|---|---|---|---|
| ADR-1 | Hybrid (deep+S NN) not pure deep | temporal fidelity + energy + sparsity (RD-01 §2) | deep-only fails NFR-GREEN / FR-BRN-02 |
| ADR-2 | Event-driven SNN scheduler | < 2% duty (FR-BRN-02.01) unattainable by dense tick | fixed dense sweep rejected |
| ADR-3 | Binocular foveated multi-res vision + foveal-only disparity | FR-VIS-02/02.02 + budget: >20× pixel win, depth in-budget | full-frame dense rejected |
| ADR-4 | Gammatone + ITD/ILD | native temporal cues; low CPU (FR-AUD) | MFCC bags no ITD start |
| ADR-5 | Source–filter articulatory speech | FR-VOC-01; parameters ≪ sampled audio | sample-loops exceed 50MB |
| ADR-6 | 18-DOF modeled head (3-DOF neck + 2×2-DOF eyes + 11-DOF face) + VOR filter | FR-KIN-01/02; predictive gaze anchoring | physical robot deferred (Phase 5) |
| ADR-7 | Codec lookups + quantization | FR-RES-02; deterministic replay (NFR-MAI-01) | fp32 everywhere rejected |
| ADR-8 | Zero-socket discipline at OS layer | NFR-SOV-01 enforced structurally | rely-on-policy rejected |

---

## 6. Performance & Quality Characteristics

- **Latency:** tick p95 15 ms (NFR-PERF-01); speech→audio ≤30 ms (FR-VOC-05).
- **Memory:** 48 MB active reserved, 50 MB hard, host < 4 GB (FR-RES-01/02).
- **Compute:** single core @ 2.0 GHz, zero GPU (FR-RES-01).
- **Sparsity:** < 2% duty (FR-BRN-02.01).
- **Determinism:** fixed-seed replay equivalence (NFR-MAI-01).
- **Resilience:** graceful degradation, no crash on under-spec inputs (FR-BRN-08).

---

## 7. Evolution

- Phase 2: implement substrates + codecs on the budget grid; instrument governor.
- Phase 3: embody — attach vision/audio/vocal/neck modules, tune budgets.
- Phase 4: limbic loops, autonomy scenarios, memory compression.
- Phase 5: port hot path to neuromorphic target (Loihi/analog) via the same interfaces (ADR-7).

---

## 8. Stakeholder Traceability Table

| Concern | View (42010) | Document section |
|---|---|---|
| Sovereignty | Physical/deployment | §3.4, NFR-SOV |
| Energy | Budget/governance | §4.2, NFR-GREEN |
| Brain coupling | Logical/interfaces | §3.1.2, FR-BRN-03 |
| Real time | Process view | §3.3, §4 |
| Tracability to SRS | Traceability | §9 |

---

## 9. Architecture ↔ Requirements Traceability (excerpt)

| SRS Requirement | Architecture element | Telemetry/Model |
|---|---|---|
| FR-RES-02 | Arena + budget §4 | RSS monitor |
| FR-VIS-02/03 | Fovea pyramid, saccade policy | fixation trace |
| FR-AUD-03 | ITD cross-corr, ILD ratio | azimuth estimate |
| FR-VOC-01 | Source–filter departments | formant trace |
| FR-KIN-02 | VOR filter | gaze slip |
| FR-BRN-01/02 | Substrates + codecs | tick profile |
| FR-BRN-04 | Limbic loops | reward/salience |
| NFR-PERF-01 | Process view §3.3 / §4 | tick p95 |
| NFR-SOV-01 | Physical view §3.4 | socket scan |
| NFR-GREEN-01 | Process view + idle §3.3 | power meter |

*End of document.*
