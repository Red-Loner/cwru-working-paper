# Experiment Protocol

## Experiment Name
Factorial evaluation of data augmentation × split protocol interaction on CWRU bearing fault diagnosis.

## Research Question
Under recording-level (leakage-safe) train-test splitting, to what extent can data augmentation recover the generalization performance lost when random adjacent-window leakage is removed? Does physical frequency-band energy preservation predict recovery efficacy?

## Hypothesis
- **H1**: Augmentation partially recovers the leakage gap, but recovery is incomplete.
- **H2**: The operational random-to-recording protocol gap exceeds every augmentation gain. The design does not causally partition exact shared-sample leakage from unseen-recording distribution shift.
- **H3**: Augmentations with higher physical fault-band energy fidelity produce greater gap recovery.

## Dataset(s)
- **Primary**: CWRU Bearing Dataset — Drive End 12 kHz, 40 .mat files, 4 fault classes (Normal, IR, OR, Ball), 3 fault diameters (0.007/0.014/0.021"), 4 motor loads (0/1/2/3 HP). 11,832 windows (1024 samples).
- **Additional**: None (minimum project scope per manual §19).

## Classes
4: Normal (0), Inner Race Fault (1), Outer Race Fault @6 o'clock (2), Ball Fault (3). Stratified 60/20/20 split.

## Segment Length
1024 samples (~85 ms at 12 kHz), 50% overlap for both protocols. The recording-level protocol controls leakage through file-disjoint partitions rather than a different stride.

## Preprocessing
1. Load .mat files, extract DE channel.
2. Parse filename for fault_type, fault_diameter, motor_load.
3. Segment into sliding windows.
4. Z-score normalization: statistics from training set only.
5. Sample the seeded transform once per run after splitting and use it to replace the normalized training windows. Do not concatenate originals or resample per epoch; validation and test remain unaugmented.
6. Keep the 50% overlap and all other preprocessing fixed across split protocols so the comparison isolates the split unit.

## Representation
- **Raw waveform**: 1×1024 (1D-CNN input).
- **STFT spectrogram**: n_fft=128, hop_length=64 → 1×32×32 (2D-CNN input).

## Model(s)
- **1D-CNN**: 4 conv layers (64→128→256→256 channels), AdaptiveAvgPool1d(16), FC(4096→128→4), Dropout 0.5.
- **2D-CNN**: 4 conv layers (32→64→128→128 channels), AdaptiveAvgPool2d(4×4), FC(2048→128→4), Dropout 0.5.

## Baselines
- **Simple**: 1D-CNN on raw waveform, no augmentation, random split (leakage-prone protocol ceiling).
- **Standard**: 2D-CNN on STFT, no augmentation, recording-level split (leakage-free floor).
- **Negative control**: Frequency-axis flip augmentation — physically destructive, should not improve performance.

## Proposed Method
Factorial experiment comparing 10 augmentation conditions × 2 split protocols × 2 models × 5 seeds. The `specaugment` condition is a waveform-compatible adaptation: two real-FFT frequency masks (up to 8 bins) followed by two contiguous time masks (up to 64 samples). It is not implemented by reshaping raw samples into a fake image.

## Ablations
1. **Single vs. combined augmentation**: Best individual (noise_01 for 1D, shift_5 for 2D) vs. combined (noise_005+shift_20).
2. **Grouping stress tests**: recording_id vs. motor_load vs. fault_size. Recording identity yields 23/7/10 train/validation/test recordings. Load grouping uses 2/1/1 of the four load groups and yields 20/10/10 recordings. Fault-size grouping uses 1/1/1 of the three diameter groups, with Normal recordings split separately as 2/1/1, and yields 14/13/13 recordings. Because group counts and training coverage differ, the resulting accuracy ranking is descriptive rather than causal.

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
- **Gap-recovery**: Δ_rec = (Acc_recording,aug − Acc_recording,none) / (Acc_random,none − Acc_recording,none).
- **Effect size**: Cohen's d.

## Non-Accuracy Dimension
**Physical consistency**: Spearman correlation between gap recovery (Δ_rec) and symmetric envelope-spectrum fault-band energy fidelity, `min(E_aug/E_orig, E_orig/E_aug)`. Fidelity is measured in BPFO/BPFI/BSF bands (±5% around the first three harmonics) before vs. after augmentation and reported separately for each model.

## Mechanism Validation
1. **M1**: Exact shared-sample fraction and aligned-overlap identity, alongside zero-lag Pearson similarity.
2. **M5**: Symmetric envelope-spectrum fault-band energy fidelity for each augmentation.
3. **M2**: Paired cosine distance between original and augmented standardized STFT input representations.

## Expected Result
Augmentation may provide partial but incomplete recovery, with model-dependent effects. The per-model physical-fidelity analysis may or may not support H3. Grouping stress tests are expected to differ, but unequal group counts preclude interpreting their ordering as causal variable importance.

## Failure Interpretation
If all augmentations show Δ_rec ≈ 0: the performance gap is entirely protocol-driven and cannot be compensated.
If Δ_rec ≈ 1.0 for any augmentation: the protocol gap does not dominate augmentation gain; augmentation nearly substitutes for the missing recording diversity under this operational comparison.

## Artifacts to Produce
- `results/tables/main_results.json` — 200 runs, 40 configurations × 5 seeds.
- `results/tables/gap_recovery.json` — Δ_rec, Cohen's d per augmentation.
- `results/tables/per_class_analysis.json` — per-class precision, recall, and F1 breakdown.
- `results/tables/confusion_matrices.json` — per-seed and aggregated 4×4 confusion matrices for all 40 configurations.
- `results/tables/physical_fidelity_correlation.json` — model-specific and pooled H3 Spearman tests.
- `results/tables/energy_audit.json` — M5 symmetric energy-fidelity values.
- `results/tables/adjacent_overlap_audit.json` — structural M1 leakage evidence.
- `results/tables/feature_diversity.json` — deterministic M2 representation comparison.
- `results/tables/convergence_summary.json` — training convergence statistics.
- `results/tables/falsification.json` — H1/H2 support verdict.
- `results/tables/contamination_robustness.json` — robustness curves.
- `results/tables/grouping_ablation.json` — 30 grouping-ablation runs.
- `data_manifest/dataset_file_manifest.json` — exact 40-file selection with SHA-256 hashes.
- `data_manifest/recording_split_assignments.json` — exact assignments for all five seeds.
- `data_manifest/random_split_verification.json` — class-stratified random-window counts for all five seeds.
- `results/figures/` — publication-quality figures, including recording-level confusion matrices.
- `results/logs/epoch_logs/` — per-epoch CSV for all 200 runs.
