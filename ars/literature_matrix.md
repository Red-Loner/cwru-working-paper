# Literature Matrix

| # | Citation | Year | Venue | Relevance | Method | Dataset | Key Finding | Our Use |
|---|----------|------|-------|-----------|--------|---------|-------------|---------|
| 1 | Smith \& Randall, "Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study" | 2015 | Mechanical Systems and Signal Processing | High: CWRU benchmark baseline | Envelope analysis, statistical features | CWRU | Established CWRU as a benchmark; documented bearing geometry and fault frequencies | Baseline reference for fault frequency calculation |
| 2 | Wen et al., "A new convolutional neural network-based data-driven fault diagnosis method" | 2018 | IEEE Trans. Industrial Electronics | High: CNN for vibration signals | 1D-CNN (raw signal input) | CWRU, motor bearing, self-priming centrifugal pump | CNN on raw vibration achieves high accuracy | Model architecture inspiration for 1D-CNN baseline |
| 3 | Zhang et al., "A new deep learning model for fault diagnosis with good anti-noise and domain adaptation ability on raw vibration signals" | 2017 | Sensors | Medium: anti-noise CNN | CNN with wide kernels in first layer | CWRU | Wide-kernel CNN robust to noise | Noise augmentation design reference |
| 4 | Bergmeir \& Benítez, "On the use of cross-validation for time series predictor evaluation" | 2012 | Information Sciences | High: time-series leakage | Statistical analysis of cross-validation | Synthetic and real time series | Random CV inflates accuracy for dependent time series; blocked CV recommended | Core citation for leakage argument |
| 5 | Cerqueira et al., "Evaluating time series forecasting models: An empirical study on performance estimation methods" | 2020 | Machine Learning | High: evaluation methodology | Empirical comparison of evaluation methods | Multiple time series datasets | Cross-validation without blocking overestimates performance | Supports recording-level split design |
| 6 | Park et al., "SpecAugment: A simple data augmentation method for automatic speech recognition" | 2019 | Interspeech | Medium: augmentation method | SpecAugment (frequency + time masking) | Speech (LibriSpeech) | Time and frequency masking improves robustness | SpecAugment adaptation for vibration spectrograms |
| 7 | Shao et al., "Highly accurate machine fault diagnosis using deep transfer learning" | 2019 | IEEE Trans. Industrial Informatics | Medium: transfer learning | Deep transfer learning with CNN | CWRU, gearbox | Transfer learning improves accuracy under varying conditions | Reference for cross-condition generalization |
| 8 | Li et al., "Diagnosing rotating machines with weakly supervised data using deep transfer learning" | 2020 | IEEE Trans. Industrial Informatics | Low: transfer learning | Deep transfer learning | CWRU | Transfer learning under weak supervision | General reference for CWRU-based research |
| 9 | CWRU Bearing Data Center | — | — | High: primary data | — | CWRU | The dataset itself | Primary data source |

## Literature Gaps Identified

1. **Selected interaction gap**: the reviewed sources do not answer this project's controlled question about augmentation effectiveness under matched random-window and recording-level protocols. This limited matrix does not establish an exhaustive literature-first claim.
2. **Selected physical-fidelity gap**: this project directly audits the relationship between envelope-spectrum fault-band energy fidelity and diagnostic recovery; it does not claim that no prior physical audit exists.
3. **Recording-level evaluation need**: Bergmeir (2012) and Cerqueira (2020) motivate dependence-aware evaluation; this project measures the augmentation interaction for its fixed CWRU subset and protocol.

## Verified Sources

The eight scholarly references with DOI fields were checked by DOI resolution. The CWRU dataset citation was checked against its source URL. This verifies identity and accessibility, not every interpretive sentence; claim support is constrained in the paper and claim-evidence table.
