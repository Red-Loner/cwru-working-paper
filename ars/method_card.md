# Method Card

## Method Name

Comparative Evaluation Protocol Study: Leakage-Safe Splitting × Data Augmentation Interaction (Factorial Design)

## Problem Solved

Sliding-window preprocessing on continuous vibration recordings creates highly correlated adjacent windows. Random train/test splitting of these windows introduces data leakage: models learn recording identity rather than fault signatures. Recording-level splitting eliminates this leakage but reduces training diversity. This study measures whether data augmentation can compensate for the diversity loss under leakage-safe evaluation.

## Hypothesis

- **H1** (Partial recovery): Data augmentation partially recovers the accuracy gap between random-split and recording-level-split evaluation, but recovery is incomplete.
- **H2** (Protocol-gap dominance): The random-window to recording-level accuracy gap exceeds every augmentation gain. This does not causally apportion the gap between exact shared-sample leakage and broader unseen-recording shift.
- **H3** (Physical fidelity predicts recovery): Physical frequency-band energy retention under augmentation correlates with recovery efficacy.

## Input Format and Dimensions

| Component | Shape | Description |
|-----------|-------|-------------|
| Raw waveform window | `(1, 1024)` | 1024-point accelerometer segment, 12 kHz CWRU DE |
| STFT spectrogram | `(1, 32, 32)` | n_fft=128, hop_length=64; magnitude bins are cropped/padded to 32×32 and standardized |

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
| n_fft | 128 | Fixed per ch40-41; magnitude output is cropped/padded to 32×32 |
| Optimizer | Adam | Standard choice for CNN training |
| Learning rate | 0.001 | Conservative, used across all conditions |
| Batch size | 128 | Fits A10 memory and is fixed across all configurations |
| Epochs | 50 | Fixed training budget; best-validation epoch distribution is reported descriptively |
| Random seeds | {42, 123, 456, 789, 1024} | 5 seeds for stability assessment |
| Train/Val/Test split | 60/20/20 | Standard ML practice, stratified by class |

## Why This Method Should Work

- **Statistical basis**: Recording-level splitting enforces independence between train and test samples. Any remaining performance reflects genuine fault-feature learning.
- **Physical basis**: CWRU bearing fault impulses produce characteristic envelope-spectrum components (BPFO, BPFI, BSF). The waveform-compatible spectral/time mask removes real FFT bins and contiguous time samples; the negative control reverses FFT magnitudes across frequency bins. Their physical fidelity is measured rather than assumed.
- **Experimental design basis**: Full factorial crossing isolates interaction effects between split protocol and augmentation condition, avoiding the "one-variable-at-a-time" fallacy.

## Baselines

| Type | Baseline | Description |
|------|----------|-------------|
| Simple | 1D-CNN, no augmentation, random split | Leakage-prone protocol ceiling — what random-window evaluation reports |
| Standard | 1D-CNN, no augmentation, recording-level split | Leakage-free generalization floor |
| Standard | 2D-CNN, no augmentation, recording-level split | Second model family floor |
| Negative control | Frequency-axis flip augmentation, recording-level split | Physically destructive augmentation (violates ch45 constraint) — should provide zero or negative benefit |

## Ablation Plan

1. **Single vs. combined augmentation**: Compare best individual augmentation (noise_005 for 1D, shift_5 for 2D) against the combined augmentation condition. Hypothesis: stacking augmentations may amplify distribution shift beyond what training data can support.
2. **Grouping stress tests**: Compare three grouping protocols — by recording_id, by motor load, and by fault size. Their train/validation/test recording counts are respectively 23/7/10, 20/10/10, and 14/13/13. Load grouping assigns the four load groups as 2/1/1; fault-size grouping assigns the three diameter groups as 1/1/1, while Normal recordings are split separately as 2/1/1. Because group counts and training coverage are unequal, this ablation is descriptive and does not identify causal variable importance.

## Mechanism Validation Plan

All mechanism validation is numerical (no subjective visual interpretation):

| Test | Method | Metric | Purpose |
|------|--------|--------|---------|
| M1 | Structural adjacent-window audit | Exact shared-sample fraction, aligned-overlap correlation/MAE, and zero-lag Pearson r | Separate guaranteed sample duplication from data-dependent similarity |
| M5 | Envelope fault-band energy audit | Symmetric fidelity `min(E_aug/E_orig, E_orig/E_aug)` in BPFO/BPFI/BSF bands | Quantify preservation without treating energy amplification as automatically safe |
| M2 | Input-representation diversity | Paired cosine distance between standardized 32×32 STFT representations | Measure how strongly augmentation changes the model input |

M5 includes a negative control: a frequency-shifted reference band (+100 Hz) to distinguish genuine fault-band retention from broadband noise retention.

## Expected Failure Cases

| Scenario | Expected Outcome | Interpretation |
|----------|-----------------|----------------|
| Δ_recovery ≈ 0 for all augmentations | Augmentation has no measurable effect under recording-level split | Falsifies H1; gap is purely protocol-driven |
| Δ_recovery ≈ 1.0 for any augmentation | Augmentation fully compensates; gap was not leakage-driven | Falsifies H2; challenges leakage orthodoxy |
| Spearman ρ ≈ 0 for H3 | No measured association between energy fidelity and recovery | H3 is not supported under this protocol; no broader causal conclusion follows |
| Frequency-axis flip provides benefit | Negative control fails | Entire augmentation framework is suspect |

## Implementation Risks

| Risk | Mitigation |
|------|-----------|
| STFT device mismatch (hann_window on wrong device) | Fixed: explicitly place window on model's device |
| Resume logic seed_id mapping | Fixed: identify correct training run by matching all config fields |
| Fault-size split could isolate the Normal class | Fixed: Normal recordings are independently distributed 60/20/20 across train/validation/test |
| Grouping strategies have unequal group counts and training coverage | Report exact assignments per seed and treat the accuracy ordering as a descriptive stress test, not a causal ranking |
| Augmentation hyperparameter sensitivity | Systematic sweep: 3 strengths per augmentation type |
| Offline one-shot transforms may differ from online augmentation | Disclose that each seeded transform replaces the training windows once per run; do not generalize to per-epoch resampling |
| Fixed seed comparison insufficient | 5 seeds with mean ± std reported; the best-validation epoch distribution is retained as a descriptive diagnostic |

## Physical Meaning Explanation

The STFT spectrogram's frequency axis carries physical units (Hz), determined by the sampling rate (12 kHz) and n_fft (128). The fault characteristic frequencies — BPFO ≈ 3.58×f_r, BPFI ≈ 5.42×f_r, BSF ≈ 2.32×f_r for the 6205 bearing — are derived from bearing geometry and shaft speed. Operations such as FFT-magnitude reversal or broadband noise can distort these components; the M5 audit measures the distortion rather than assuming that any operation is safe or destructive. Per ch45, image-domain augmentations cannot be naively transferred to spectrograms because its axes have physical meaning.
