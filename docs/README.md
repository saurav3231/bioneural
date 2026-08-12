# Project BioNeural — Documentation

This directory holds the formal research and engineering baseline for **SDLC Phase 1
(Theoretical Research & Architecture Study)**. All documents are written to international
standards and constitute the authoritative reference for the program.

## Document Index

| Doc | ID | Title | Standard | Status |
|---|---|---|---|---|
| [00](00_PROJECT_MANIFESTO_AND_MISSION.md) | BN-DOC-000 | Project Manifesto & Mission | — | Draft |
| [01](01_RESEARCH_FOUNDATIONS_AND_STATE_OF_THE_ART.md) | BN-DOC-001 | Research Foundations & State of the Art | academic survey | Draft |
| [02](02_SYSTEM_REQUIREMENTS_SPECIFICATION_SRS.md) | BN-DOC-002 | System Requirements Specification (SRS) | ISO/IEC/IEEE 29148:2018 | Draft |
| [03](03_SYSTEM_ARCHITECTURE_DESIGN_DOCUMENT_SADD.md) | BN-DOC-003 | System Architecture Design Document (SADD) | ISO/IEC/IEEE 42010:2011 | Draft |
| [04](04_SDLC_ROADMAP_AND_MILESTONES.md) | BN-DOC-004 | SDLC Roadmap & Milestones | ISO/IEC/IEEE 12207:2017 | Draft |

## Document Map

```
00 Manifesto ──► 01 Research Foundations ──► 02 SRS (requirements)
        │                 │                        │
        └─────────────────┴───────► 03 SADD (architecture) ◄──┐
                                             │                  │
                             04 SDLC Roadmap (lifecycle) ──────┘
```

- The **manifesto** fixes mission and pledge.
- **Research Foundations** supplies the scientific evidence base.
- The **SRS** is the verifiable requirements baseline (FR-\*/NFR-\*).
- The **SADD** realizes every requirement in architecture with time/memory budgets.
- The **roadmap** schedules all of the above across five SDLC phases.

## Reading Order

1. [00 — Manifesto](00_PROJECT_MANIFESTO_AND_MISSION.md)
2. [01 — Research Foundations](01_RESEARCH_FOUNDATIONS_AND_STATE_OF_THE_ART.md)
3. [02 — Requirements (SRS)](02_SYSTEM_REQUIREMENTS_SPECIFICATION_SRS.md)
4. [03 — Architecture (SADD)](03_SYSTEM_ARCHITECTURE_DESIGN_DOCUMENT_SADD.md)
5. [04 — Roadmap](04_SDLC_ROADMAP_AND_MILESTONES.md)

## Key Numbers

| Metric | Target |
|---|---|
| Host RAM | < 4 GB |
| Active footprint | < 50 MB |
| CPU | single-core 2.0 GHz |
| GPU | zero required |
| Power | < 15 W |
| Cognitive tick | < 15 ms |
| Active neurons | < 2% |

## Maintenance

Documents are version-controlled in this repository. Any change that alters requirements or
architecture must update the traceability chain in the SRS (§7) and the SADD (§9).
