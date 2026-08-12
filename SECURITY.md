# Security Policy

Project BioNeural takes the security and privacy of its users seriously. This project's core
promises — **100% offline sovereignty**, **zero cloud dependency**, and **no telemetry** — are
also its most important security properties (see `NFR-SOV` in the
[System Requirements Specification](docs/02_SYSTEM_REQUIREMENTS_SPECIFICATION_SRS.md)).

## Supported Versions

| Release | Supported |
|---|---|
| `v0.1.0-alpha` (SDLC Phase 1 baseline) | ✔ Current |
| Earlier development commits | ✖ |

Currently the repository contains **no executable code** (Phase 1 documentation baseline). Security
practices below apply to documentation accuracy and, from **Phase 2 onward**, to the runtime.

## Reporting a Vulnerability

Please report vulnerabilities privately. **Do not** open a public issue or pull request that
discloses a vulnerability.

- **Email:** `bhandarisaurav15@gmail.com` (Project Lead: Saurav Bhandari)
- **Include:** affected version/commit, subsystem (brain / senses / body / runtime / storage),
  description of the issue, steps to reproduce, and (if known) a suggested fix.

You will receive an acknowledgment within **72 hours**, and we will coordinate a disclosure timeline.
We ask that you allow us a reasonable period (typically 90 days) to publish a fix before public
disclosure. If you wish to follow coordinated disclosure, please include that in your report.

## Scope

In scope:

- Neural-network integrity — adversarial inputs that produce dangerous outputs.
- Memory/resource safety — crashes, unbounded allocation, or denial-of-service on entry-level
  hardware (single-core 2.0 GHz, < 4 GB RAM).
- Offline sovereignty — any path that leaks data or opens network connections at runtime
  (`NFR-SOV-01`).
- Supply chain — build/dependency integrity of the published artifact.

## Security Commitments

1. **Zero network transmissions** — the runtime performs no socket opens, DNS, or telemetry.
2. **At-rest encryption** — persistent sensory/user data is encrypted (AES-256 or better) when
   enabled.
3. **Responsible disclosure** — all vulnerabilities are handled privately and disclosed after fix.

## V&V

Security conformance is verified through the project's [V&V strategy](docs/04_SDLC_ROADMAP_AND_MILESTONES.md#5-verification--validation-strategy-12207),
including socket-emission scans and deterministic replay.

— Project BioNeural
