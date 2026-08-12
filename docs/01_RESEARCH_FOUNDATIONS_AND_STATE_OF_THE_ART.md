# PROJECT BIONEURAL — 01: RESEARCH FOUNDATIONS AND STATE OF THE ART

> *Global-standard academic research paper. Phase 1 deliverable under ISO/IEC 12207:2017.*
> Establishes the scientific and engineering baseline for the hybrid-brained biomimetic head.

| Field | Value |
|---|---|---|
| Document ID | BN-DOC-001 |
| Version | 0.1.0 |
| Status | Draft (Phase 1 review) |
| Classification | Public / Open |
| Creator & Principal Investigator | Saurav Bhandari (Student, Pokhara, Nepal) |
| License | MIT Open-Source |
| Keywords | Neuromorphic computing, spiking neural networks, hybrid AI, foveation, stereoscopic vision, binaural audition, articulatory synthesis, 18-DOF kinematics, vestibulo-ocular reflex, sparse computation |

---

## Authors and Affiliation

**Saurav Bhandari** — Creator & Principal Investigator, Project BioNeural (Pokhara, Nepal).
Independent, sovereign, and open research initiative. The author holds no corporate or institutional
stake in the economic interests of the incumbent AI industry; this paper is published under the MIT
License in accordance with the Universal Access Pledge.

---

**Abstract.** Dense, all-to-all matrix computation dominates contemporary artificial intelligence,
yet it is simultaneously the least dense in the engineering sense: it touches every parameter for
every computation, at enormous energy cost and with no temporal dynamics. Biological neural systems
oppose this with sparse, event-driven, continuous-time computation on a ~20 W substrate. This paper
formalizes the drawbacks of dense AI, defines the *BioNeural Hybrid Paradigm* that bridges continuous
deep representations with continuous-time spiking dynamics, lays the sensory-motor foundations of the
embodied head, and evaluates Project BioNeural against monolithic transformers, robotics simulators,
and pure SNN platforms.

---

## 1. Dense Classical AI versus Biological Sparse Computation

### 1.1 The computational economy of dense inference

Let a forward pass of an MLP of depth *L* with widths *n_l* require

> FLOPs_pass = Σ_l (2 · n_l · n_(l−1)) per input (order term; biases omitted).

Under the dense convention every matrix element participates in every inference regardless of input.
Consequently both power and energy scale with parameter count *P* at *fixed* per-inference cost:

> E_efficiency = units_compute / units_value → degrades as ~P wins at inference.

Representative victim measurements (order of magnitude, published literature): inference on a 7 B
parameter transformer approaches ~micro-batches of floating-point per token; the full-stack energy
per token is dominated by memory movement and dense GEMM. This contrasts with cortical operation, in
which energy per spike is ~orders of magnitude lower *and* spikes are rare.

### 1.2 Formal drawbacks of dense classical AI

1. **Computational over-provisioning.** Activation maps are recomputed for every token even when
   information is redundant; no event-driven gating exists. Effective sparsity of activations in
   dense transformers (measured) remains orders of magnitude below cortical duty cycles.
2. **Temporal blindness.** Dense ANNs operate in discrete clocked steps; timing *within* a tick is
   discarded. Cues such as interaural time difference (ITD, microsecond resolution) and spike latency
   are irrecoverable in this representation.
3. **Energetic burden.** As energy ≈ FLOPs roughly, distributed at scale across millions of users the
   aggregate becomes the data-center economics criticized in the Manifesto.
4. **Disembodied abstraction.** Serving models that merely pattern-match text/image manifolds lack
   the sensory-grounded, predictive perception that embodiment provides (see §3).
5. **Vanishing gradients of temporal credit.** Recurrent approximations struggle with long, irregular
   event streams; SNNs natively treat causality as a first-class dimension.

### 1.3 The biological contrast: sparse, event-driven, continuous-time

**Evidence:** single-neuron activity in cortex is typically under 1–2 Hz average firing with bursting
transients; cortical field recordings show massive temporal structure. The brain’s energy budget of
~20 W is dominated by synaptic transmission — an *event cost*, not a per-weight cost. Hence:

| Axis | Dense AI (transformers) | Biological CNS |
|---|---|---|
| Computation trigger | Clocked, all elements | **Event-driven, sparse** (<2%) |
| Time | Discrete tick | Continuous-time dynamics |
| Energy per useful event | High (dense GEMM) | Low (single spike) |
| Memory model | Full weight matrix in RAM | Distributed synaptic weights + dynamics |
| Robustness | Fragile to adversarial/out-of-distribution | Degrades gracefully |
| Learning | Backprop over static graph | STDP, eligibility traces, neuromodulation |

**Conclusion (supporting Motive 3):** sparsity and event-driven execution are not merely
instrumental choices; they are the *enabling conditions* for >100× energetic advantage at matched
functionality.

---

## 2. The BioNeural Hybrid Paradigm

### 2.1 Why not pure SNN, why not pure deep

Pure SNNs struggle with the representational depth modern capability requires; pure deep networks
lack the temporal and energetic properties of §1.3. BioNeural therefore defines a **hybrid
substrate**: a *continuous cognitive substrate* (rate-based, differentiable deep layers) coupled to a
*neuromorphic SNN substrate* (continuous-time spiking dynamics) through bidirectional **rate ⇄ spike
codecs**.

### 2.2 The canonical codec mapping

Let the deep substrate produce a rate vector **r**(t) ∈ ℝ^d (per hidden unit *i*). The spiking
interface maps rates to Poisson or deterministic spike trains:

> spike_i(t) ∈ {0,1},  with Pr(spike in dt) = clamp(r_i(t), 0, r_max) · dt   (Poisson codec)

and decodes spikes back to rates via exponential or membrane-potential smoothing:

> r̂_i(t) = (1/τ) · Σ_spikes exp(−(t − t_k)/τ)   (decoder, exponential kernels)

**Neural dynamics (Leaky Integrate-and-Fire, LIF).** For membrane potential u_i:

> τ_m · du_i/dt = −(u_i − u_rest) + Σ_j w_ij · s_j(t) + I_bias

> if u_i ≥ θ: emit spike; u_i ← u_reset

with spiking current delivery s_j(t) = Σ_tk δ(t − t_k) — exactly the continuous-time, event-driven
execution the Manifesto mandates.

### 2.3 Three coupling modes

1. **Rate ↔ spike codec (feedforward)** — cognitive substrate emits *intentions/attention* as rates;
   SNN cortex converts to precise spike timing.
2. **Spike → rate (feedback)** — SNN sensory responses integrate into abstract representations.
3. **Neuromodulation / limbic gating** — subcortical signals scale μ (membrane time constant) or
   modulate plasticity (three-factor learning) rather than passing content. This realizes salience
   and arousal coupling without drowning the cognition in unnecessary activity — preserving the < 2%
   sparsity budget.

### 2.4 Learning regimes

| Regime | Substrate | Update rule | Phase |
|---|---|---|---|
| Supervised/RL pre-training | Continuous substrate | Backprop / policy gradient | 2 |
| Online spike-timing plasticity | SNN | STDP + reward-modulated three-factor | 2–4 |
| Hebbian associative | Both | Weight co-activation | 3 |
| Predictive coding / active inference | Whole loop | Free-energy minimization on energy E(x,u) | 4 |

The hybrid architecture retains the *same* informational interface on both sides; therefore any piece
of knowledge can in principle be stored either in distributed continuous weights or in spike-timing
relationships, depending on the temporal and energetic cost profile. This is the operational form of
Motive 5: *biological temporal efficiency + synthetic computational power*.

---

## 3. Embodied Sensory-Motor Foundations

The head plant closes the loop. Each sensory channel is grounded in a biological primitive with a
compact computational surrogate.

### 3.1 Stereoscopic retinal foveation (FR-VIS, NFR-PERF)

- **Biology.** The retina resolves the visual world anisotropically: a central fovea at high spatial
  acuity over ~1–2° of visual angle, a tapering periphery. Gaze is moved by saccades (up to ~30°/s)
  to *sample* the scene rather than densely render it. Two forward-facing eyes give overlapping visual
  fields, from which **binocular disparity** yields metric depth (stereopsis).
- **Surrogate.** *Binocular* multi-resolution pyramid camera: central crop at full sensor resolution,
  coarser periphery at decreasing resolution with separation σ(r) growing with eccentricity r;
  disparity computed in a small foveal binocular region for near-field depth. Saccade policy selects
  next fixation target; periphery guides searcher, *fovea resolves*, and a short-term fixation buffer
  holds high-resolution history.
- **Energy/CPU economics.** Each eye renders anisotropically; stereo processing is restricted to the
  foveal overlap band, keeping depth computation within the single-core budget. Foveation alone can
  cut raw pixel throughput by >20× at matched task accuracy — the single largest sensory win for a
  <15 ms tick on one core.

### 3.2 Binaural tonotopic audition (FR-AUD)

- **Tonotopy.** Basilar membrane resolves frequency logarithmically. Surrogate: **Gammatone
  filterbank** — for center frequency f_c, impulse response
  > g(t) = a · t^(n−1) · e^(−2πb·t) · cos(2π f_c t + φ),  n = 4 (dominant order).
- **Binaural cues.** ITD: interaural time difference, resolution down to ~10–50 µs, principally low
  frequencies (< ~1.5 kHz); ILD: interaural level difference (head-shadow), high frequencies.
- **Azimuth estimation.** Two-path: (i) low-band **cross-correlation** of left/right Gammatone
  envelopes (Jeffress-model coincidence grid); (ii) high-band **ILD ratio**. Fusion via maximum-
  likelihood on azimuth hypothesis θ̂:
  > θ̂ = argmax_θ L(ITD_obs, ILD_obs | θ).
- **Why it matters for budget.** A 32-channel Gammatone bank per ear runs in a fraction of the CPU
  tick budget and yields native microsecond temporal structure usable directly by the SNN.

### 3.3 Articulatory vocal synthesis (FR-VOC)

- **Biology (source–filter theory).** Glottis = source (pitch F0 + harmonics + aspiration noise),
  vocal tract = filter (formant resonances F1–F3, controlled by articulator geometry: tongue, jaw,
  lips, velum).
- **Surrogate.** Two generators:
  - *Source:* glottal pulse train (LF model or harmonic-plus-noise with jitter/shimmer) — voiced;
    flow noise — unvoiced.
  - *Filter:* all-pole vocal-tract filter whose poles are formant frequencies F1–F3; articulatory
    parameters map to formants via a compact articulatory-to-acoustic map.
- **Output path.** Digital waveguide or direct IIR filtering at 16–22 kHz; amplitude envelope
  smoothed for naturalness; optional amplitude and F0 contours for prosodic intent driven by the
  cognitive substrate.
- **Why it matters.** Speech is *produced* by a physical model, not by sampled audio lookup —
  enabling arbitrary phonemes, continuous pitch, and emotional prosody with mere hundreds of
  parameters.

### 3.4 The 18-DOF kinematic head and the vestibulo-ocular reflex (FR-KIN)

- **Kinematic architecture.** Eighteen degrees of freedom compose the head plant: **3-DOF neck**
  (yaw, pitch, roll about a neck joint at the head base), **2×2-DOF binocular eyes** (horizontal
  vergence + vertical gaze per eye, enabling stereoscopic pursuit and vergence), and **11-DOF facial
  articulators** (lips, jaw, brow, eyelids, cheeks) that voice emotion and accompany speech — with
  physical maximal excursions and inertias at every joint.
- **VOR (vestibulo-ocular reflex).** When the head rotates with angular velocity ω_head, the eyes
  counter-rotate to stabilize the retinal image during the brief period before VOR is replaced by
  saccades. The surrogate embeds a **VOR filter** per eye: eye angular velocity
  ω_eye = −G_VOR(s) · ω_head, G_VOR ≈ 1 (unity gain) for 0.1–5 Hz maneuvers, running at sensory-tick
  latency. Binocular VOR additionally enforces conjugacy so both foveas stay locked on the target.
- **Why it matters for the organism.** VOR is the archetype of *efference-copy / sensorimotor
  prediction*: vestibular estimate drives eye motion *before* vision validates. It gives the head
  its lifelike stability and is the canonical testbed for predictive loops in the hybrid brain; the
  18-DOF configuration makes the whole face an expressive, gaze-anchored instrument.

### 3.5 Exclusion: olfaction (FR-EXC)

Olfaction is explicitly out of scope. Rationale: (a) source signal acquisition for a real-time
chemical modality is physically incompatible with the < 50 MB / offline / single-core-CPU envelope;
(b) head-level olfaction adds negligible deliberative value versus vision+audition for the Phase 1–5
mission; (c) its exclusion simplifies the sensory stack without compromising the biology-inspired
abstraction. The interface architecture remains open so that a future *chemical peripheral* could be
exercised without altering the brain core.

---

## 4. Comparative Evaluation

Matrix comparing **BioNeural** against (A) monolithic transformer architectures, (B) mainstream
robotics simulators (iCub, Sophia), and (C) pure neuromorphic platforms (Intel Loihi 2, SpiNNaker 2).

| Axis | **BioNeural** (Hybrid CPU) | Monolithic Transformers | Robotics Sim (iCub / Sophia) | Pure SNN (Loihi 2 / SpiNNaker 2) |
|---|---|---|---|---|
| Computation model | Sparse event-driven + deep hybrid | Dense all-to-all | Dense rigid-body + dense DNN policy | Pure asynchronous spiking |
| Time representation | Continuous-time (SNN) + discrete (deep) | Discrete tokens | Discrete sim steps | Continuous-time memristor/async |
| Energy envelope | **< 15 W target** | 100 W–100 MW | kW–tens of kW sim farm | mW–W (per chip) |
| Memory footprint | **Active < 50 MB on < 4 GB host** | GBs–TB | GBs (meshes, physics) | Per-chip SRAM (MBs) but no general cognition |
| Embodiment depth | Full head: stereoscopic fovea, binaural, articulatory, 18-DOF | None (statistical over text/images) | Full humanoid/face (iCub, Sophia robot hardware) | Sensor nodes only; no full-head plant |
| Cognition generality | Deep representations + dynamics + limbic | Highest text/code breadth | Scripted + learned behaviors | Limited learning, low level |
| Offline/sovereign | **Yes (design)** | No (served) | Physics + controllers (weak) | Yes but specialized |
| Hardware dev effort | Single-core CPU, no accelerators | Cluster/cloud mandatory | Physical robot + farms | Specialized neuromorphic hardware |
| Capability ceiling | Beyond-human hybrid (sparse + deep + embodied) | Narrow, high on text | Humanlike motion, shallow cognition | Biologically faithful, shallow cognition |

### 4.1 Reading the matrix

- Transformers win *breadth of abstraction* and lose everything situational, temporal, and energetic.
- Robotics simulators win *physical embodiment* but demand orders of magnitude more compute for
  skeletal dynamics and carry no unified cognitive substrate.
- Pure SNN platforms win *biological faithfulness and energy per spike* but do not offer general
  cognition, universal deployment, or an embodied sensory I/O stack of the head's scope without
  extensive custom design.

**BioNeural occupies the intersection**: it keeps the deep substrate's representational power, borrows
the SNN's temporal efficiency and sparsity, and lays a head plant — stereoscopic vision, binaural
audition, articulatory vocalization, 18-DOF kinematics — on top, all within hard resource ceilings
(≤ 4 GB host RAM, single-core 2.0 GHz CPU, zero GPU, < 50 MB active footprint). It is positionable as
the only architecture in the comparison that is simultaneously *embodied*, *energetically frugal*,
*sovereign-learning*, and *general-purpose at the object level of an organism*.

---

## Annex A — Notation Glossary

| Symbol | Meaning |
|---|---|
| r(t) / r̂(t) | Rate vector / decoded rate vector (continuous substrate) |
| s_j(t) | Spike train of presynaptic neuron *j*, Σ δ(t − t_k) |
| u_i, θ, u_rest, u_reset | Membrane potential, threshold, rest/reset potentials (LIF) |
| τ_m | Membrane time constant |
| w_ij | Synaptic weight |
| f_c, b, n | Gammatone center frequency, bandwidth, filter order |
| ITD/ILD | Interaural time/level difference |
| F0, F1–F3 | Fundamental frequency, first–third formants |
| ω_head, ω_eye | Head angular velocity / eye angular velocity (VOR) |
| G_VOR(s) | VOR transfer function |

---

## References (representative, Phase 1 baseline)

1. Azevedo F.A.C., et al. *Equal numbers of neuronal and nonneuronal cells make the human brain an
   isometrically scaled-up primate brain.* J. Comp. Neurol. 2009.
2. Herculano-Houzel S. *The Human Advantage.* MIT Press, 2016.
3. Patterson D., et al. *Carbon emissions and large neural network training.* arXiv, 2021.
4. Izhikevich E.M. *Simple model of spiking neurons.* IEEE Trans. Neural Networks, 2003.
5. Indiveri G., et al. *Neuromorphic silicon neuron circuits.* Front. Neurosci. 2011.
6. Patterson R.D. & Holdsworth J. *A functional model of neural auditory processing.* 1996
   (Gammatone filterbank).
7. Jeffress L.A. *A place theory of sound localization.* J. Comp. Physiol. Psychol. 1948.
8. Klatt D.H. *Software for a cascade/parallel formant synthesizer.* JASA, 1980.
9. Wilson C. et al. (Intel Labs). *Loihi: a neuromechanical research neuro chip.* IEEE Micro, 2018.
10. Furber S. et al. *The SpiNNaker project.* Proc. IEEE, 2014.
11. Metta G. et al. *The iCub humanoid robot.* Int. J. Humanoid Robotics, 2010.

*End of document.*