# Methodology Blueprint

Based on `ars/method_card.md`. This blueprint documents the complete experimental methodology.

## Research Design

**Type**: Factorial experiment (2 × 2 × 10 × 5 within-subjects)

| Factor | Levels | Description |
|--------|--------|-------------|
| A: Model | 2 | 1D-CNN (4 conv layers), 2D-CNN (4 conv layers, STFT input) |
| B: Split Protocol | 2 | Random adjacent-window split (leakage-prone) vs. Recording-level split (leakage-safe) |
| C: Augmentation | 10 | none, noise_001, noise_005, noise_01, shift_5, shift_20, shift_50, specaugment, combined (noise_005+shift_20), freq_flip (negative control) |
| D: Random Seed | 5 | [42, 123, 456, 789, 1024] |

**Total configurations**: 2 × 2 × 10 × 5 = 200 independent training runs

## Data Pipeline

1. Load 40 CWRU Drive-End .mat files (12 kHz sampling rate)
2. Parse filename to extract fault_type, fault_diameter, motor_load
3. Segment continuous recordings into sliding windows (1024 samples, fixed 50% overlap in both split protocols)
4. Assign labels: Normal=0, IR=1, OR=2, Ball=3
5. Split: 60% train / 20% validation / 20% test, stratified by class
6. Z-score normalization: statistics computed on training set only
7. Apply augmentation to training set only (never to val/test)

## Model Architectures

### 1D-CNN
- Input: (1, 1024) raw vibration waveform
- Conv1: 7→64 channels, kernel=7, stride=2
- Conv2: 64→128, kernel=5
- Conv3: 128→256, kernel=3
- Conv4: 256→256, kernel=3
- AdaptiveAvgPool1d(16)
- FC: 4096→128→4
- Dropout 0.5

### 2D-CNN
- Input: (1, 32, 32) STFT magnitude spectrogram (n_fft=128, hop=64)
- Conv1: 1→32, kernel=3
- Conv2: 32→64, kernel=3
- Conv3: 64→128, kernel=3
- Conv4: 128→128, kernel=3
- AdaptiveAvgPool2d(4)
- FC: 2048→128→4
- Dropout 0.5

## Training Protocol

- Optimizer: Adam, learning rate 0.001
- Loss: CrossEntropy
- Batch size: 128
- Epochs: 50 fixed; best-validation epoch distribution reported descriptively
- Device: NVIDIA A10 (24 GB)
- PyTorch 2.5.1+cu121

## Mechanism Validation (pre-classification)

| Test | Method | Metric |
|------|--------|--------|
| M1 | Structural adjacent-window audit | Exact shared fraction, aligned overlap correlation/MAE, and zero-lag r |
| M5 | Envelope fault-band energy audit | Symmetric fidelity `min(E_aug/E_orig, E_orig/E_aug)` in BPFO/BPFI/BSF bands |
| M2 | Input-representation diversity | Paired cosine distance between standardized STFT representations |

## Evaluation

- Primary: Accuracy, Macro-F1 (mean ± std over 5 seeds)
- Secondary: Per-class precision/recall/F1, confusion matrix
- Gap-recovery: Δ_rec = (Acc_recording,aug − Acc_recording,none) / (Acc_random,none − Acc_recording,none)
- Cohen's d for effect size

## Ablation

1. Single vs. combined augmentation: Compare best individual against combined
2. Grouping variable importance: recording_id vs. motor_load vs. fault_size

## Non-Accuracy Dimension

Physical consistency: model-specific Spearman correlation between symmetric fault-band fidelity and Δ_rec (H3 test)

## Software Environment

```
torch==2.5.1+cu121
numpy==2.2.6
scipy==1.15.3
tqdm==4.69.0
scikit-learn==1.7.2
matplotlib==3.10.9
```
