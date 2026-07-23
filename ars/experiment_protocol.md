# Experiment Protocol

## Experiment Name
Factorial evaluation of data augmentation × split protocol interaction on CWRU bearing fault diagnosis.

## Research Question
Under recording-level (leakage-safe) train-test splitting, to what extent can data augmentation recover the generalization performance lost when random adjacent-window leakage is removed? Does physical frequency-band energy preservation predict recovery efficacy?

## Hypothesis
- **H1**: Augmentation partially recovers the leakage gap, but recovery is incomplete.
- **H2**: The majority of inflated accuracy under random splitting is leakage-driven.
- **H3**: Physical frequency-band energy retention does not correlate with recovery (falsified: ρ=-0.37, p=0.33).

## Dataset(s)
- **Primary**: CWRU Bearing Dataset — Drive End 12 kHz, 40 .mat files, 4 fault classes (Normal, IR, OR, Ball), 3 fault diameters (0.007/0.014/0.021"), 4 motor loads (0/1/2/3 HP). 11,832 windows (1024 samples).
- **Additional**: None (minimum project scope per manual §19).

## Classes
4: Normal (0), Inner Race Fault (1), Outer Race Fault @6 o'clock (2), Ball Fault (3). Stratified 60/20/20 split.

## Segment Length
1024 samples (~85 ms at 12 kHz). Overlap: 50% (random split), 0% (recording split).

## Preprocessing
1. Load .mat files, extract DE channel.
2. Parse filename for fault_type, fault_diameter, motor_load.
3. Segment into sliding windows.
4. Z-score normalization: statistics from training set only.
5. Apply augmentation to training set only.

## Representation
- **Raw waveform**: 1×1024 (1D-CNN input).
- **STFT spectrogram**: n_fft=128, hop_length=64 → 1×32×32 (2D-CNN input).

## Model(s)
- **1D-CNN**: 4 conv layers (64→128→256→256 channels), AdaptiveAvgPool1d(16), FC(4096→128→4), Dropout 0.5.
- **2D-CNN**: 4 conv layers (32→64→128→256 channels), AdaptiveAvgPool2d(4), FC(4096→128→4), Dropout 0.5.

## Baselines
- **Simple**: 1D-CNN on raw waveform, no augmentation, random split (leakage ceiling).
- **Standard**: 2D-CNN on STFT, no augmentation, recording-level split (leakage-free floor).
- **Negative control**: Frequency-axis flip augmentation — physically destructive, should not improve performance.

## Proposed Method
Factorial experiment comparing 10 augmentation conditions × 2 split protocols × 2 models × 5 seeds.

## Ablations
1. **Single vs. combined augmentation**: Best individual (noise_01 for 1D, shift_5 for 2D) vs. combined (noise_005+shift_20).
2. **Grouping variable importance**: recording_id vs. motor_load vs. fault_size.

## Split Design
- **Random split**: All windows pooled, random permutation, 60/20/20, stratified by class.
- **Recording-level split**: Windows grouped by recording_id, each recording assigned entirely to one split, 60/20/20, stratified by class.

## Leakage Prevention
- Recording-level split (primary safe protocol).
- Z-score normalization from training set only.
- Augmentation applied after split, to training set only.
- No cross-recording contamination (each .mat file = one recording).
- No test-set tuning of any parameter.

## Seeds
[42, 123, 456, 789, 1024] — 5 seeds for stability assessment.

## Metrics
- **Primary**: Accuracy, Macro-F1 (mean ± std across 5 seeds).
- **Secondary**: Per-class precision/recall/F1, confusion matrix.
- **Gap-recovery**: Δ_rec = (Acc_aug - Acc_noaug) / (Acc_random - Acc_noaug).
- **Effect size**: Cohen's d.

## Non-Accuracy Dimension
**Physical consistency**: Spearman correlation between fault-band energy retention (R_energy) and gap recovery (Δ_rec). R_energy measured as energy ratio in BPFO/BPFI/BSF frequency bands (±5% tolerance, 2–3 harmonics) before vs. after augmentation.

## Mechanism Validation
1. **M1**: Adjacent-window Pearson correlation (quantify leakage severity).
2. **M5**: Fault-band energy retention audit (R_energy for each augmentation).
3. **M2**: Feature diversity analysis (cosine distance between original and augmented features).

## Expected Result
Augmentation provides partial but incomplete recovery. 1D-CNN benefits more than 2D-CNN. Physical fidelity does not predict recovery. Fault-size domain shift dominates as the hardest generalization challenge.

## Failure Interpretation
If all augmentations show Δ_rec ≈ 0: the performance gap is entirely protocol-driven and cannot be compensated.
If Δ_rec ≈ 1.0 for any augmentation: leakage is not the dominant factor; augmentation fully substitutes for recording diversity.

## Artifacts to Produce
- `results/tables/main_results.json` — 200 runs, 40 configurations × 5 seeds.
- `results/tables/gap_recovery.json` — Δ_rec, Cohen's d per augmentation.
- `results/tables/per_class_analysis.json` — per-class F1 breakdown.
- `results/tables/h4_correlation.json` — H3 Spearman test.
- `results/tables/energy_audit.json` — M5 R_energy values.
- `results/tables/convergence_summary.json` — training convergence statistics.
- `results/tables/falsification.json` — H1/H2 support verdict.
- `results/tables/contamination_robustness.json` — robustness curves.
- `results/figures/` — 14 publication-quality figures.
- `results/logs/epoch_logs/` — per-epoch CSV for all 200 runs.
