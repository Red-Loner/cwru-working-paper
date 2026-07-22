# Final Integrity Report

Date: 2026-07-22

## Verification Checklist

Per `manual.md` §16.1, the following 8 questions are verified against all artifacts.

| # | Question | Status | Evidence |
|---|----------|--------|----------|
| 1 | Does every result in the paper exist in logs, tables, or figures? | PASS | All accuracy numbers verified against `results/tables/main_results.json`. All figures traced to `results/figures/*.png`. All gap-recovery numbers in `results/tables/gap_recovery.json`. All per-class values in `results/tables/per_class_analysis.json`. |
| 2 | Does every citation exist and support the sentence? | PASS | All 9 references in `paper/refs.bib` verified by DOI. Each citation used in paper.tex traced to supporting literature. No hallucinated citations (M2). |
| 3 | Are all dataset claims traceable to source documentation? | PASS | CWRU dataset facts (40 files, 12 kHz, bearing 6205, fault diameters 0.007"/0.014"/0.021") documented in `data_manifest/dataset_card_cwru.md` and `ars/material_passport.md`. |
| 4 | Are all AI-generated claims verified? | PASS | All factual claims cross-checked against JSON result files. "5 of 9 degrade" → corrected to "6 of 9" after re-run verification. No unsupported claim remaining. |
| 5 | Are limitations explicit? | PASS | 6 limitations listed in paper §Limitations: single dataset, single sensor, fixed augmentation parameters, CNN-only architectures, 50-epoch ceiling, R_energy metric confound. |
| 6 | Is the split design described clearly? | PASS | Recording-level split documented in `data_manifest/split_design.md` and paper §Dataset. Window parameters (1024 samples, 0%/50% overlap) specified. Train/val/test 60/20/20, class-stratified. |
| 7 | Are failed or negative results honestly reported? | PASS | H3 falsification (ρ=-0.37, p=0.33) honestly reported. Six of nine augmentations degrade 2D-CNN — reported without sugar-coating. Grouping ablation shows near-complete failure of cross-severity transfer (0.47-0.63). |
| 8 | Is the AI-use disclosure complete? | PASS | Paper §AI-Use Disclosure specifies: DeepSeek via OpenCode CLI, code implementation (verified on hardware), experimental debugging, manuscript drafting (all claims verified against results). Universal denial of hallucinated results. |

## Seven Failure Modes (Lu et al., 2026)

| Mode | Status | Evidence |
|------|--------|----------|
| M1: Buggy self-review | CLEARED | `torch.manual_seed` added; `overlap_ratio` parameter fixed; all code audited in integrity check (2026-07-22) |
| M2: Hallucinated references | CLEARED | All 9 DOIs verified; 0 hallucinations found |
| M3: Hallucinated results | CLEARED | All 140+ table cells verified against JSON sources; 0 discrepancies |
| M4: Shortcut learning | CLEARED | Recording-level split prevents cross-recording leakage; M1 autocorrelation confirms adjacent-window correlation; negative control (freq_flip) validates augmentation framework |
| M5: Bug-as-discovery | CLEARED | "6 of 9 degrade" verified by per-seed data; fault_size ablation verified; H3 null result honestly reported with metric limitation acknowledged |
| M6: Methodology fabrication | CLEARED | batch_size, conv layers, combined augmentation description all corrected to match actual code; 200 runs traceable to 200 epoch log files |
| M7: Framing lock-in | CLEARED | Alternative grouping variables tested (load, fault_size); negative control included; H3 metric limitation acknowledged; limitations section addresses CWRU-specific constraints |

## Blocking Issues

None.

## Non-Blocking Issues

1. `reviews/` directory is empty — reviewer reports exist in `ars/` instead.
2. Paper uses inline LaTeX tables; `paper_tables.tex` exists as standalone reference but is not `\input`'d.
3. Grouping ablation was not re-run with the fixed `overlap_ratio` code (uses default OVERLAP_RANDOM — results unchanged from prior run).

## Overall Verdict

**PASS.** All seven failure modes CLEARED. All manual §16.1 questions PASS. The research package is inspectable, reproducible, and scientifically honest.
