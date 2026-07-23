# Final Reviewer Report

## Recommendation

**Accept with explicitly stated scope limitations.**

The repaired study now supports a reproducible, matched-preprocessing comparison between random-window and recording-level evaluation on the selected CWRU subset. Its strongest contribution is methodological: exact data hashes and split assignments, complete seed-level results, structural overlap evidence, a functioning negative control, and fail-fast release verification. The paper correctly treats the observed difference as an operational protocol gap rather than a causal estimate of leakage alone.

## Major findings verified

1. **Matched preprocessing**: both primary protocols use 1,024-sample windows and 50% overlap. The random split is class-stratified; the recording split is file-disjoint.
2. **Evidence completeness**: 40 configurations × 5 seeds are present, with exactly 200 complete 50-epoch logs and per-seed 4×4 confusion matrices.
3. **Primary result**: the no-augmentation protocol gaps are 0.0569 (1D) and 0.1143 (2D). Shift 5 gives the best tested recovery, 0.5255 and 0.3263.
4. **Negative control**: FFT-magnitude reversal sharply reduces both random and recording accuracy, so it behaves as a destructive control rather than an accidental augmentation.
5. **H3 reporting**: neither per-model correlation is significant; the sensitivity analysis excluding the high-leverage negative control is also null. The pooled correlation is correctly labeled non-inferential.
6. **Grouping interpretation**: fault-size grouping is the lowest-accuracy stress test, but the manuscript no longer converts this into causal variable importance because group counts and training coverage differ.
7. **Seed sensitivity**: the manuscript reports the large change from the first three to all five configured seeds without claiming that five seeds are universally sufficient.

## Residual limitations

- The study uses the course manual's CWRU-only minimum scope and cannot support cross-dataset claims.
- Only two CNN families and one sensor channel are evaluated.
- Augmentation is offline and sampled once per run; online per-epoch augmentation may differ.
- The H3 analysis has nine augmentation-level observations per model and one narrow physical metric.
- Grouping stress tests are not coverage-matched.
- Eight of 200 runs reach their best validation accuracy in epochs 45–50, so the 50-epoch ceiling remains relevant.
- The auxiliary contamination analysis uses one seed and should not be presented as a general robustness result.

## Required wording retained in the final manuscript

- Use “random-to-recording protocol gap,” not “measured leakage amount.”
- Use “H3 is not supported,” not “H3 is falsified.”
- Describe fault-size results as a stress test with unequal coverage.
- Describe best-validation epoch as a diagnostic, not proof of convergence.
- Limit augmentation claims to the tested offline transforms and parameters.

## Evidence inspected

- `results/tables/main_results.json`
- `results/tables/confusion_matrices.json`
- `results/tables/gap_recovery.json`
- `results/tables/grouping_ablation.json`
- `results/tables/physical_fidelity_correlation.json`
- `results/tables/release_summary.json`
- `data_manifest/dataset_file_manifest.json`
- `data_manifest/recording_split_assignments.json`
- `data_manifest/split_verification.json`
- `data_manifest/random_split_verification.json`
- `src/verify_release.py`
