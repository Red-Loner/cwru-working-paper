# Integrity Report — Audited Evidence Stage

## Failure-mode audit

| Mode | Check | Status | Evidence |
|---|---|---|---|
| M1 — Hidden data dependence | Shared samples, recording overlap, normalization, and augmentation timing inspected | Cleared with scope caveat | Structural overlap audit; exact recording assignments; train-only normalization |
| M2 — Unsupported numerical claims | Main, H3, grouping, seed, and epoch claims traced to generated JSON | Cleared | `release_summary.json`; generated LaTeX tables; claim–evidence table |
| M3 — AI-invented evidence | Server outputs, hashes, logs, and repair trail retained | Cleared | 200 logs; server/local archive hash; prompt log; AI disclosure |
| M4 — Shortcut reliance | Negative control and two model families tested | Cleared within scope | Frequency reversal collapses accuracy; separate 1D/2D results |
| M5 — Bug-as-discovery | Protocol, metric, grouping, robustness, and convergence code re-audited | Cleared after repairs | Revision response items 1–13 |
| M6 — Methodological forgery | Implementation checked against prose and manifests | Cleared | Same 50% overlap; stratified random split; file-disjoint recording split |
| M7 — Framing lock-in | Causal wording, H3 null, seed limits, and grouping confound reviewed | Cleared | Protocol-gap language; sensitivity analysis; explicit limitations |

## Evidence completeness

- 40 unique main configurations × 5 configured seeds.
- Exactly 200 per-epoch CSVs, each covering epochs 1–50.
- Forty confusion-matrix records, each with five 4×4 matrices and an aggregate.
- Thirty grouping runs, all with exact assignments and 4×4 matrices.
- Recording grouping reproduces the main no-augmentation recording baseline exactly for every seed.
- Forty readable selected recordings and 200 seed/recording assignment rows.
- Random and recording split verification both report 50% overlap.
- Fifteen referenced result figures are present.
- H3 includes per-model primary tests and per-model sensitivity tests without the negative control.

## Interpretation boundaries

1. The random-to-recording difference is an operational protocol gap, not a causal estimate of leakage alone.
2. Fault-size grouping is a descriptive stress test because group and training coverage differ.
3. H3 is not supported; the null does not prove physical fidelity is irrelevant.
4. Best-validation epoch is a diagnostic, not proof of convergence.
5. Three seeds were inadequate in this configured ordering; five are not claimed universally sufficient.
6. The one-seed contamination curves are auxiliary only.
7. No cross-dataset generalization claim is made.

The final integrity report is issued only after the compiled PDF is rendered, visually inspected, and the complete release verifier passes.
