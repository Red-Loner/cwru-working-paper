# Method Card

## Method Name

Comparative Evaluation Protocol Study: Leakage-Safe Splitting × Data Augmentation Interaction (Factorial Design)

## Problem Solved

Sliding-window preprocessing on continuous vibration recordings creates highly correlated adjacent windows. Random train/test splitting of these windows introduces data leakage: models learn recording identity rather than fault signatures. Recording-level splitting eliminates this leakage but reduces training diversity. This study measures whether data augmentation can compensate for the diversity loss under leakage-safe evaluation.

## Hypothesis

- **H1** (Partial recovery): Data augmentation partially recovers the accuracy gap between random-split and recording-level-split evaluation, but recovery is incomplete.
- **H2** (Leakage dominance): The majority of inflated accuracy under random splitting is attributable to recording-identity leakage.
- **H3** (Physical fidelity predicts recovery): Physical frequency-band energy retention under augmentation correlates with recovery efficacy.

## Input Format and Dimensions

| Component | Shape | Description |
|-----------|-------|-------------|
| Raw waveform window | `(1, 1024)` | 1024-point accelerometer segment, 12 kHz CWRU DE |
| STFT spectrogram | `(1, 32, 32)` | n_fft=128, hop_length=64, 50% overlap, mel-scale resizing |

## Output Format

| Component | Type | Description |
|-----------|------|-------------|
| Class label | `int ∈ {0,1,2,3}` | Normal, Inner Race Fault, Outer Race Fault, Ball Fault |
| Training log | CSV | Per-epoch train/val loss, accuracy, F1 |

## Key Parameters

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Window length | 1024 samples (≈85 ms) | Standard in CWRU literature; short enough to resolve transient fault impulses |
| Window stride | 512 samples | 50% overlap, common practice |
| n_fft | 128 | Fixed per ch40-41; yields 32×32 spectrogram after mel resizing |
| Optimizer | Adam | Standard choice for CNN training |
| Learning rate | 0.001 | Conservative, used across all conditions |
| Batch size | 32 | Fits GPU memory, stable gradient estimates |
| Epochs | 50 | Convergence verified (median epoch = 9–15 across configurations) |
| Random seeds | {42, 123, 456, 789, 1024} | 5 seeds for stability assessment |
| Train/Val/Test split | 60/20/20 | Standard ML practice, stratified by class |

## Why This Method Should Work

- **Statistical basis**: Recording-level splitting enforces independence between train and test samples. Any remaining performance reflects genuine fault-feature learning.
- **Physical basis**: CWRU bearing fault impulses produce characteristic frequency components (BPFO, BPFI, BSF). Augmentations that preserve these bands (time shifting, mild noise) should enable the model to learn fault-relevant features; augmentations that distort them (frequency-axis flipping, SpecAugment) should not.
- **Experimental design basis**: Full factorial crossing isolates interaction effects between split protocol and augmentation condition, avoiding the "one-variable-at-a-time" fallacy.

## Baselines

| Type | Baseline | Description |
|------|----------|-------------|
| Simple | 1D-CNN, no augmentation, random split | Leakage ceiling — what random-split evaluation reports |
| Standard | 1D-CNN, no augmentation, recording-level split | Leakage-free generalization floor |
| Standard | 2D-CNN, no augmentation, recording-level split | Second model family floor |
| Negative control | Frequency-axis flip augmentation, recording-level split | Physically destructive augmentation (violates ch45 constraint) — should provide zero or negative benefit |

## Ablation Plan

1. **Single vs. combined augmentation**: Compare best individual augmentation (noise_005 for 1D, shift_5 for 2D) against the combined augmentation condition. Hypothesis: stacking augmentations may amplify distribution shift beyond what training data can support.
2. **Grouping variable importance**: Compare three grouping protocols — by recording_id, by motor load, and by fault size. Hypothesis: fault-size domain shift dominates recording-level leakage as the harder generalization challenge.

## Mechanism Validation Plan

All mechanism validation is numerical (no subjective visual interpretation):

| Test | Method | Metric | Purpose |
|------|--------|--------|---------|
| M1 | Adjacent-window autocorrelation | Pearson r between windows 1–10 steps apart | Quantify leakage severity in raw signal |
| M5 | Fault-band energy audit | Energy ratio in BPFO/BPFI/BSF bands before/after augmentation | Verify physical safety of each augmentation |
| M2 | Feature-space diversity | Cosine distance between original and augmented features | Confirm augmentation increases feature diversity |

M5 includes a negative control: a frequency-shifted reference band (+100 Hz) to distinguish genuine fault-band retention from broadband noise retention.

## Expected Failure Cases

| Scenario | Expected Outcome | Interpretation |
|----------|-----------------|----------------|
| Δ_recovery ≈ 0 for all augmentations | Augmentation has no measurable effect under recording-level split | Falsifies H1; gap is purely protocol-driven |
| Δ_recovery ≈ 1.0 for any augmentation | Augmentation fully compensates; gap was not leakage-driven | Falsifies H2; challenges leakage orthodoxy |
| Spearman ρ ≈ 0 for H3 | No correlation between physical safety and recovery | Augmentation helps for reasons unrelated to physical features |
| Frequency-axis flip provides benefit | Negative control fails | Entire augmentation framework is suspect |

## Implementation Risks

| Risk | Mitigation |
|------|-----------|
| STFT device mismatch (hann_window on wrong device) | Fixed: explicitly place window on model's device |
| Resume logic seed_id mapping | Fixed: identify correct training run by matching all config fields |
| Fault_size split isolates Normal class | Fixed: Normal always in train; stratified sampling across fault classes |
| Augmentation hyperparameter sensitivity | Systematic sweep: 3 strengths per augmentation type |
| Fixed seed comparison insufficient | 5 seeds with mean ± std reported; convergence analysis confirms training stability |

## Physical Meaning Explanation

The STFT spectrogram's frequency axis carries physical units (Hz), determined by the sampling rate (12 kHz) and n_fft (128). The fault characteristic frequencies — BPFO ≈ 3.58×f_r, BPFI ≈ 5.42×f_r, BSF ≈ 2.32×f_r for the 6205 bearing — are derived from bearing geometry and shaft speed. Augmentations that modify these frequency components (e.g., frequency-axis flipping, broad-spectrum Gaussian noise at high σ) destroy the physical signature the model should learn. Per ch45, image-domain augmentations cannot be naively transferred to spectrograms because spectrogram axes have physical meaning, not arbitrary pixel coordinates.
