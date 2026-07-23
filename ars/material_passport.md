# Material Passport

## Primary Dataset: CWRU Bearing Data Center

| Field | Value |
|-------|-------|
| Dataset name | Case Western Reserve University Bearing Data Center |
| URL | https://engineering.case.edu/bearingdatacenter |
| Citation | CWRU Bearing Data Center, Case Western Reserve University |
| Access terms | Publicly downloadable benchmark; no explicit license statement was found in the local materials, so redistribution of raw files is avoided |
| Download date | 2026-07 |
| Signal type | Vibration acceleration (accelerometer) |
| Sensor location used | Drive-end (DE) |
| Sampling rate used | 12 kHz |
| Machine type | 2 HP Reliance Electric motor |
| Bearing type | 6205-2RS JEM SKF (DE), 6203-2RS JEM SKF (FE) |
| Fault types | Normal, Inner Race (IR), Outer Race (OR @ 6 o'clock), Ball |
| Fault diameters | 0.007", 0.014", 0.021" |
| Motor loads | 0, 1, 2, 3 HP |
| Approximate speeds | 1797, 1772, 1750, 1730 RPM |
| Total files | 40 (.mat format) |
| Preprocessing | Sliding windows: 1024 samples, 50% overlap in both split protocols |
| Total windows | 11,832 (50% overlap) |
| Split design | 60/20/20 train/val/test, stratified by class; recording-level split by recording_id |

## Leakage Risks (Audited)

| Risk | Level | Mitigation |
|------|-------|------------|
| Adjacent window leakage | HIGH when random split | Recording-level split eliminates |
| Cross-recording contamination | LOW | Each .mat file = one recording, never split |
| Normalization leakage | NONE | Statistics from training set only |
| Augmentation leakage | NONE | Applied after split, to training set only |
| Fault-size confounding | MEDIUM | Recording split does not stratify by fault size; tracked in the assignment manifest and acknowledged in limitations |
| Load confounding | LOW | Recording split stratifies by label only, but load varies within label |
| Grouping-ablation comparability | HIGH for causal ranking | Exact groups and recordings are saved per seed; unequal group counts/training coverage are disclosed, and results are interpreted descriptively |

## Augmentation Parameters

| Type | Parameters | Physical Safety |
|------|-----------|----------------|
| Gaussian noise | σ = 0.01, 0.05, 0.10 | Physical fidelity measured in `results/tables/energy_audit.json` |
| Time shift | ±5, ±20, ±50 samples | Physical fidelity measured in `results/tables/energy_audit.json` |
| Spectral/time mask | Two FFT-bin masks + two time-sample masks | Physical fidelity measured in `results/tables/energy_audit.json` |
| Combined | noise_005 + shift_20 | Physical fidelity measured in `results/tables/energy_audit.json` |
| FreqFlip | Full frequency-axis reversal | Negative control (deliberately destructive) |

## Physical Parameters (6205 Bearing)

| Parameter | Formula | Value @ 1797 RPM |
|-----------|---------|-----------------|
| BPFO | 3.5848 × f_r | 107.4 Hz |
| BPFI | 5.4152 × f_r | 162.2 Hz |
| BSF | 2.3178 × f_r | 69.4 Hz |
| FTF | 0.3983 × f_r | 11.9 Hz |

## Code Environment

- Python 3.10.12 with PyTorch 2.5.1+cu121
- Hardware: NVIDIA A10 (23,028 MiB), 16 vCPU, 64 GB RAM
- OS: Ubuntu 22.04 LTS
- GPU driver: 535.309.01, CUDA 12.2
