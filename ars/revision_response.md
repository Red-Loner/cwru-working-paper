# Revision Response

| # | Reviewer / Audit Concern | Response and Manuscript Change | Evidence |
|---|---|---|---|
| 1 | Random and recording protocols used different overlaps | Both now use 50% overlap, fixing segmentation and sample-count confounding | `src/config.py`; split verification JSON |
| 2 | Random split was described as stratified but was globally shuffled | Implemented deterministic per-class 60/20/20 allocation and generated seed-level checks | `src/preprocess.py`; `random_split_verification.json` |
| 3 | Five main logs and all confusion matrices were missing | Regenerated all evidence: 200 exact 50-epoch logs and 40 configuration-level matrix artifacts | `results/logs/epoch_logs/`; `confusion_matrices.json` |
| 4 | Grouping JSON was empty | Regenerated 30 grouping runs with per-seed matrices and exact split assignments | `grouping_ablation.json` |
| 5 | “SpecAugment” and frequency flipping operated on a fake reshaped image | Replaced with real-FFT bin masking plus waveform time masking and FFT-magnitude reversal; wording narrowed to waveform-compatible adaptation | `src/augmentation.py`; Method section |
| 6 | M1 zero-lag correlation did not measure shifted shared samples | Added exact shared fraction, aligned-overlap correlation, and aligned MAE | `adjacent_overlap_audit.json` |
| 7 | M5 treated energy amplification as preservation | Replaced with bounded symmetric envelope-spectrum band-energy fidelity | `energy_audit.json`; `fault_band_energy.py` |
| 8 | H3 silently overwrote one model | Added separate 1D and 2D correlations, a negative-control sensitivity analysis, and a non-inferential pooled description | `physical_fidelity_correlation.json` |
| 9 | M2 used an untrained network hidden layer | Replaced with paired standardized 32×32 STFT input-representation distances | `feature_diversity.json` |
| 10 | Grouping accuracy was presented as causal importance | Added exact group/recording counts and restricted interpretation to descriptive stress tests | generated grouping table; Limitations |
| 11 | Robustness curves used different seeds and noise draws | Paired all conditions on the same model seed and deterministic test-noise realization | `src/contamination_test.py` |
| 12 | Offline augmentation behavior was not disclosed | Paper now states that each seeded transform replaces training windows once per run and is not resampled per epoch | Dataset/Method and Limitations |
| 13 | “Convergence epoch” was overclaimed, and the script had an undefined variable | Renamed to best-validation epoch, fixed the variable, and regenerated 200-run summary and figure | `convergence_summary.json`; `convergence_repair.log` |
| 14 | Numerical claims were duplicated and stale | Main, recovery, per-class, and grouping tables are generated from JSON and included by `paper/main.tex` | `src/summarize_results.py`; `paper/generated_tables/` |
| 15 | E-drive selected data included damaged files | Excluded the E copy; all 40 server files were hash-matched to the readable D copy; loader now fails on selected-file read errors | local data-copy audit; dataset manifest |
| 16 | External-dataset search and CWRU-only scope were undocumented | Recorded official candidate sources and explicit deferral under the manual's minimum option | `dataset_discovery_report.md` |
| 17 | Software environment was not exactly locked | Added the exact Python/A10 package versions used by the audited run | `environment/requirements-a10-lock.txt` |
| 18 | Paper language causally equated the protocol gap with leakage | Reframed H2 and all results as an operational protocol gap that also includes unseen-recording shift | Abstract, Discussion, claim table |
| 19 | Seed recommendation relied on stale values | Recomputed first-three versus all-five diagnostics: shifts of 5.32pp (1D) and 7.75pp (2D); no universal sufficiency claim | `release_summary.json` |
| 20 | Release could regress silently | Added fail-fast checks for configurations, seeds, logs, matrices, assignments, generated tables, figures, claim-table schema, and PDF validity | `src/verify_release.py` |
| 21 | Course-required evidence was stored but not fully visible in the manuscript | Added explicit RQ/contributions, provenance, Macro-F1, selected per-class precision/recall/F1, plan-deviation disclosure, robustness boundaries, expanded limitations, and stronger reproduction/AI-verification statements | `paper/main.tex`; `paper/generated_tables/`; `ars/paper_requirement_traceability_20260723.md` |

All final numerical wording is derived from the audited A10 results. No Git push, tag, or public release was performed.
