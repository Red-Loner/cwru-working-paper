# Leakage Audit — CWRU Bearing Fault Diagnosis Experiment

## Audit Scope

This audit evaluates leakage risks for the factorial experiment comparing:
- Factor A: Random adjacent-window split (A1) vs. Recording-level split (A2)
- Factor B: 9 augmentation conditions (none, Gaussian noise ×3, time shift ×3, SpecAugment, combined, frequency-axis flip)
- Factor C: 2 model families (1D CNN, STFT + 2D CNN)

## Leakage Risk Assessment

### Leakage Path 1: Adjacent-Window Information Transfer

| Attribute | Assessment |
|-----------|------------|
| **Risk level** | **HIGH** |
| **Path** | Windows cut from the same continuous vibration recording with overlap (A1: 50% overlap). Adjacent windows share near-identical waveforms. When randomly assigned to train and test sets, the model sees test-time signal shapes nearly identical to training examples. |
| **Effect** | Model learns recording identity (motor speed, sensor transfer function, ambient noise floor) rather than fault signature. Estimated accuracy inflation: 10–30 percentage points. |
| **Detection** | Per-class accuracy gap between A1 and A2; M1 adjacent-window autocorrelation analysis; training curve divergence (M3). |
| **Correction** | Use recording-level split (A2) as primary evaluation protocol. A1 retained only as a leakage baseline. |
| **Paper reporting** | Report both A1 and A2 results for every experimental condition. Compute gap-recovery ratio Δ_recovery using A2 as denominator, A1 difference as numerator. Explicitly warn that A1 results are leakage-biased. |

### Leakage Path 2: Normalization Using Test Data

| Attribute | Assessment |
|-----------|------------|
| **Risk level** | **MEDIUM** |
| **Path** | Computing z-score mean/std on the full dataset (including test windows) before splitting. Test statistics leak into training input scale. |
| **Effect** | Modest but real accuracy inflation (1–5 percentage points, smaller than adjacent-window leakage but non-zero). |
| **Detection** | Compare "normalize-before-split" vs "normalize-after-split" accuracy difference. |
| **Correction** | Fit normalizer on training set only; apply same transform to val/test. |
| **Paper reporting** | Document normalization protocol explicitly: "Z-score normalization using mean and standard deviation computed from the training partition exclusively." |

### Leakage Path 3: Augmentation Before Splitting

| Attribute | Assessment |
|-----------|------------|
| **Risk level** | **MEDIUM** |
| **Path** | Applying augmentation to the full dataset before train/val/test split. Augmented copies of the same original window may end up in train AND test. |
| **Effect** | Augmentation-derived test windows closely resemble training windows, inflating accuracy. The augmentation-evaluation-confounding described in the RQ brief. |
| **Detection** | Compare "augment-before-split" vs "augment-after-split" accuracy. |
| **Correction** | Split first, then augment training partition only. Validation and test sets NEVER receive augmentation. |
| **Paper reporting** | Explicitly state: "Data augmentation is applied exclusively to the training partition after partitioning. No augmented samples appear in validation or test splits." |

### Leakage Path 4: Augmentation Leaking Test-Time Information

| Attribute | Assessment |
|-----------|------------|
| **Risk level** | **LOW** |
| **Path** | SpecAugment or frequency-axis manipulation using test-set statistics to set augmentation parameters. |
| **Effect** | Uncertain — low probability in this setup because augmentation parameters are fixed a priori (σ values, shift magnitudes, mask counts). |
| **Detection** | Parameter sweep design inherently prevents this — all augmentation strengths are predetermined. |
| **Correction** | N/A — already prevented by design. |
| **Paper reporting** | List all fixed augmentation hyperparameters in a table. |

### Leakage Path 5: Cross-Contamination in Contamination Test (Factor E)

| Attribute | Assessment |
|-----------|------------|
| **Risk level** | **LOW** |
| **Path** | Robustness curve tests apply noise contamination to test set only — but if noise parameters use information from the uncontaminated test set (e.g., setting σ=0.02 based on test-set std), this leaks information. |
| **Effect** | Negligible for this design — contamination levels (σ=0.02, σ=0.1) are set relative to signal statistics from training data only, not test data. |
| **Detection** | Verify contamination parameters are derived from training-set statistics. |
| **Correction** | Use training-set signal statistics (global std of normalized training windows) to set contamination σ values. |
| **Paper reporting** | Report: "Contamination noise magnitudes are set as fractions of the training-set signal standard deviation, not the test set." |

### Leakage Path 6: Cross-Load Test Leakage Through Fault Diameter

| Attribute | Assessment |
|-----------|------------|
| **Risk level** | **LOW-MEDIUM** |
| **Path** | If cross-load tests use the same fault-diameter recordings across train and test loads, the model may learn fault diameter rather than fault type, confounding cross-load generalization claims. |
| **Effect** | Cross-load accuracy slightly inflated if diameter is correlated with fault class. |
| **Detection** | Check: do all fault diameters appear in all splits? Report per-diameter cross-load accuracy. |
| **Correction** | If confounding found, report it as a limitation. Ensure all fault diameters are represented in both train and test loads. |
| **Paper reporting** | Disclose fault-diameter distribution in each split. Report per-diameter accuracy alongside per-class accuracy for cross-load tests. |

## Summary of Risk Levels

| Leakage Path | Risk Level | Blocking? | Mitigation |
|--------------|------------|-----------|------------|
| Adjacent-window transfer | HIGH | Yes — use A2 as primary | Recording-level split as primary protocol |
| Normalization leakage | MEDIUM | Yes — fix immediately | Fit normalization on training set only |
| Augmentation before split | MEDIUM | Yes — fix immediately | Augment only training partition post-split |
| Augmentation parameter leakage | LOW | No | Already prevented by fixed-parameter design |
| Contamination information leak | LOW | No | Use training-set statistics for contamination |
| Cross-load diameter confounding | LOW-MEDIUM | No | Disclose and report per-diameter analysis |

## Corrected Protocol (RECOMMENDED)

1. **Load** all .mat files; identify unique recording IDs.
2. **Split recordings** into train/val/test (60/20/20) by recording, stratified by fault class. Save assignment.
3. **Segment** windows within each split independently (A2: 0% overlap, A1: 50% overlap for baseline comparison).
4. **Compute normalization statistics** on training-set windows only. Apply to all three splits.
5. **Augment** training-set windows only (9 augmentation conditions, each applied to copies of training windows).
6. **Train** models on augmented training set; validate on unaugmented val set; test on unaugmented test set.
7. **Contamination tests**: apply noise to test set only, using parameters derived from training-set statistics.

## Paper Reporting Checklist

- [ ] Describe both split protocols (A1, A2) in the Experimental Setup section.
- [ ] State normalization protocol (training-set statistics only).
- [ ] State augmentation protocol (training partition only, post-split).
- [ ] Report all results for both A1 and A2 — never mix or report only one.
- [ ] Explicitly warn about leakage inflation in A1 results.
- [ ] Document recording-to-split assignment (supplementary material or repo file).
- [ ] Report per-class, per-diameter, and per-load breakdowns where informative.
