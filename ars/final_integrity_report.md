# Final Integrity Report

Date: 2026-07-23

## Release verdict

**PASS.** The audited A10 run, generated
tables, logs, split manifests, figures, manuscript source, and compiled PDF are
mutually consistent. The release verifier passes all machine-checkable
requirements. The 16-page PDF was rendered page by page and inspected; no
clipping, blank pages, unreadable tables, or missing figures were found.

The five supplied author names are included in standard pinyin form.
Affiliation and email are intentionally omitted because none were supplied.
No Git commit, push, tag, or public release was performed.

## Manual §16.1 checklist

| # | Question | Status | Evidence |
|---|---|---|---|
| 1 | Does every reported result exist in logs, tables, or figures? | PASS | `release_summary.json` reconciles 200 main runs, 30 grouping runs, 200 complete epoch logs, confusion matrices, mechanism audits, H3 sensitivity results, and generated paper tables. |
| 2 | Does every citation exist and support the adjacent claim? | PASS | `paper/refs.bib` contains eight DOI-backed literature records and one CWRU source URL; the manuscript makes no claim of a DOI for the dataset page. |
| 3 | Are dataset claims traceable? | PASS | The selected-file manifest fixes the exact 40 CWRU recordings and SHA-256 hashes. Server hashes match the readable D-drive copy. The damaged E-drive copy is explicitly excluded. |
| 4 | Are AI-assisted claims independently checked? | PASS | Numerical claims are generated from result JSON files, included through generated LaTeX tables where applicable, and checked by `src/verify_release.py`. |
| 5 | Are limitations explicit? | PASS | The paper lists eleven limitations, including artificial-fault/field-readiness scope, condition dependence, the leakage-prone comparator, fixed and offline augmentation, CNN-only models, the 50-epoch ceiling, limited physical metrics/sample size, unequal grouping coverage, reduced robustness scope, and AI-assisted-workflow risk. |
| 6 | Is the split design clear and reproducible? | PASS | Both random and recording-level protocols use 1,024-sample windows and 50% overlap. Exact per-seed recording assignments and class-stratification checks are stored in the release. |
| 7 | Are negative and null results reported honestly? | PASS | The FFT-magnitude-reversal negative control is reported as destructive; H3 is reported unsupported per model and after excluding that control; grouping comparisons are labeled descriptive rather than causal. |
| 8 | Is AI use disclosed? | PASS | Section 14 distinguishes initial DeepSeek/OpenCode assistance from the later Codex audit and A10 rerun, and states that numerical results came from executed code on selected files. |

## Seven failure modes

| Mode | Status | Release evidence |
|---|---|---|
| M1: Buggy self-review | CLEARED | Deterministic seeds, fail-fast data loading, protocol smoke tests, exact counts, finite-number checks, and a clean full rerun replace self-attestation. |
| M2: Hallucinated references | CLEARED | Bibliography entries are present and the CWRU web source is distinguished from DOI records. |
| M3: Hallucinated results | CLEARED | All headline values trace to generated JSON/CSV artifacts; the verifier checks counts, shapes, seeds, epochs, tables, figures, and PDF freshness. |
| M4: Shortcut learning | MITIGATED AND DISCLOSED | Recording-disjoint evaluation is the primary conservative protocol; random-window results are retained only as a leakage-prone comparator. |
| M5: Bug-as-discovery | CLEARED | Split, augmentation, M1, M2, M5, H3, robustness, and convergence defects were repaired before the clean A10 rerun. |
| M6: Methodology fabrication | CLEARED | Model shapes, batch size, augmentation behavior, seeds, runtime, and grouping assignments are reconciled to code and stored evidence. |
| M7: Framing lock-in | CLEARED | Alternative grouping stress tests, a destructive negative control, sensitivity analysis, seed diagnostics, and explicit null/limitation language are included. |

## Final evidence checks

- Main design: 2 models × 2 split protocols × 10 augmentation conditions × 5
  seeds = 200 runs.
- Grouping design: 2 models × 3 grouping strategies × 5 seeds = 30 runs.
- Main epoch logs: exactly 200 CSV files, each with 50 epochs.
- Confusion matrices: 40 configuration rows with valid per-seed and aggregate
  matrix shapes.
- Recording grouping baseline exactly reproduces the main no-augmentation
  recording-level result for every model and seed.
- A10 runtime logs preserve the one convergence-reporting failure and its
  successful fixed rerun rather than hiding the failure.
- `paper/main.pdf`: 16 pages, compiled after all source tables and figures.
- LaTeX log: no overfull/underfull boxes, undefined references, missing figures,
  or multiply defined labels.
- Visual inspection: all 16 rendered pages reviewed, including the title page,
  main table and protocol figures, grouping/H3 pages, limitations, reproducibility
  statement, AI-use disclosure, and references.

## Scope and owner actions

This is the course-permitted CWRU-only minimum scope; it does not support a
cross-dataset generalization claim. A second external dataset remains an
optional extension. The owner may decide whether to commit, push, tag, or
publish.

The E-drive CWRU data copy remains untouched because two selected recordings
there are unreadable. Use the verified D-drive copy or the hash-matched server
copy for reruns.
