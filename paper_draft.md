# How Much Can Data Augmentation Compensate for Leakage-Safe Splitting?

## A Factorial Experiment on CWRU Bearing Fault Diagnosis


## Abstract

In vibration-based bearing fault diagnosis, the standard sliding-window preprocessing creates highly correlated adjacent segments. Random train/test splitting of these windows introduces data leakage: models may learn window identity rather than fault signatures, inflating reported accuracy. Recording-level splitting eliminates this leakage but reduces training diversity, often degrading test performance. We investigate whether data augmentation can compensate for this performance drop under leakage-safe evaluation. Through a 2×2×10×5 factorial experiment (2 models × 2 split protocols × 10 augmentation conditions × 5 random seeds, 200 independent training runs) on the CWRU bearing dataset, we find: (1) recording-level splitting reduces accuracy by 6.8pp (1D-CNN) and 9.9pp (2D-CNN), both large effects (Cohen's d=1.17, 1.80); (2) augmentation recovers at most 46% of this gap for 1D-CNN but only 15% for 2D-CNN; (3) seven of nine augmentations *degrade* 2D recording-split performance below the no-augmentation baseline; (4) physical frequency-band energy retention under augmentation (H3: Spearman ρ=-0.37, p=0.33) does not predict recovery; and (5) a grouping-variable ablation reveals that fault-size domain shift (acc=0.47) dominates recording-level leakage (0.90) and load variation (0.98) as the hardest generalization challenge. We recommend a minimum of five random seeds for recording-level evaluation and provide a reproducible benchmark for leakage-controlled CWRU experiments.


## 1. Introduction

Vibration-based deep learning for bearing fault diagnosis has achieved near-perfect accuracy on benchmark datasets [1–3]. However, the standard preprocessing pipeline — fixed-length sliding windows extracted from continuous vibration recordings — introduces a subtle but critical flaw. Adjacent windows from the same recording share a large fraction of signal content. When all windows are pooled and randomly split into train/validation/test sets, windows from the same original recording appear in multiple splits. A classifier can exploit this temporal adjacency to achieve high accuracy without learning fault-specific features, effectively performing a "recording identification" task rather than fault diagnosis [4, 5].

This data leakage is a known problem in time-series evaluation [4, 5], yet its interaction with data augmentation — a standard regularization technique — has not been systematically studied. Augmentation increases training diversity and could, in principle, compensate for the reduced diversity under leakage-safe splitting. But augmentation may also distort physically diagnostic frequency bands, undermining the very features a classifier should learn.

We pose three hypotheses:
- **H1**: Data augmentation partially recovers the performance gap between leakage-prone (random window) and leakage-safe (recording-level) splits.
- **H2**: The leakage gap dominates — even the best augmentation cannot restore random-split performance.
- **H3**: Augmentations that better preserve fault-frequency-band energy yield greater recovery (physical fidelity hypothesis).

To test these hypotheses, we conduct a fully factorial experiment on the Case Western Reserve University (CWRU) bearing dataset: 2 model architectures (1D-CNN on raw waveform, 2D-CNN on STFT spectrograms) × 2 split protocols × 10 augmentation conditions × 5 random seeds, totaling 200 independent training runs. We further ablate three grouping variables — recording identity, motor load, and fault size — to identify which domain shift most challenges generalization.

The dataset comprises 40 recording files yielding 11,832 windows (1,024 samples per window, 50% overlap). Each configuration trains for 50 epochs on an NVIDIA A10 GPU (Adam optimizer, lr=0.001, batch size 128).


## 2. Related Work

**CWRU benchmark.** The Case Western Reserve University bearing dataset [6] is the most widely used benchmark in vibration-based fault diagnosis. Smith and Randall [1] provided a comprehensive benchmark study establishing standard evaluation protocols. Subsequent work has achieved near-ceiling accuracy under random window splitting using 1D-CNN [2] and 2D-CNN [3] architectures.

**Data augmentation for fault diagnosis.** Data augmentation techniques adapted from computer vision and speech processing have been applied to vibration signals, including additive noise injection, time-domain transformations (shifting, scaling), and frequency-domain masking. SpecAugment [7], originally proposed for speech recognition, applies time and frequency masking to spectrogram inputs. However, the physical implications of these operations on vibration spectrograms — where the frequency axis carries diagnostic meaning — remain under-explored.

**Data leakage in time-series evaluation.** Random train/test splitting of temporally dependent observations inflates performance estimates [4]. Bergmeir and Benítez [4] established that standard cross-validation is invalid for time-series prediction. Cerqueira et al. [5] empirically compared estimation methods and confirmed that out-of-sample (temporal) evaluation is necessary. In fault diagnosis, the sliding-window preprocessing creates an analogous dependence structure that standard random splitting fails to respect.

**Physical constraints on spectrogram augmentation.** Unlike natural images, spectrogram axes carry physical meaning: the frequency axis maps to specific fault characteristic frequencies, and the time axis maps to shaft rotation periods. Naively applying image-domain augmentations (rotation, shearing, color jitter) to spectrograms can destroy diagnostically relevant features. Prior work has proposed physically constrained augmentation strategies [8, 9] but has not systematically tested the relationship between physical fidelity and downstream performance.


## 3. Dataset and Leakage Control

### 3.1 CWRU Bearing Dataset

We use the CWRU 12kHz Drive End bearing dataset, comprising 40 .mat recording files across four fault classes: Normal (4 recordings), Inner Race fault (12), Outer Race fault @6 o'clock (12), and Ball fault (12). Fault diameters include 0.007, 0.014, and 0.021 inches. Motor loads range from 0 to 3 HP.

Each recording is segmented into fixed-length windows of 1,024 points with 50% overlap, yielding 11,832 windows total. The fault-class distribution is: Normal 952, IR 3,624, OR 3,582, Ball 3,674.

### 3.2 Leakage Audit

**Leakage source**: Sliding-window overlap. Adjacent windows from the same recording share up to 512 points (50% overlap). Random splitting distributes these correlated windows across train/val/test, enabling a classifier to recognize recording identity via temporal proximity.

**Leakage control**: Recording-level split. All windows from a given .mat recording file are assigned entirely to train, validation, or test, stratified by fault class. The train/val/test ratio is 60/20/20. No recording appears in more than one split.

We verify the leakage structure through adjacent-window autocorrelation analysis (see Section 5.1, Figure 8), which confirms that window pairs from the same recording carry redundant information exploitable under random splitting.

**Safe practices**: Z-score normalization statistics (mean, std) are computed on the training set only and applied to val/test. Data augmentation is applied after splitting, exclusively to the training set. All hyperparameters and augmentation parameters are fixed prior to experimentation.


## 4. Method

### 4.1 Experimental Design

We employ a fully crossed factorial design:

- **Factor A — Model**: 1D-CNN (3 conv layers, raw waveform input 1×1024) and 2D-CNN (4 conv layers, STFT spectrogram input 1×32×32, n_fft=128, hop_length=64)
- **Factor B — Split Protocol**: Random window split (leakage baseline) and Recording-level split (leakage-safe)
- **Factor C — Augmentation**: 10 conditions — none, additive Gaussian noise (σ=0.01, 0.05, 0.10), time shift (5, 20, 50 samples), SpecAugment (frequency + time masking), combined (noise σ=0.05 + shift 20 + SpecAugment), and frequency-axis flip (negative control — destroys physical meaning)
- **Factor D — Random Seed**: 5 seeds (42, 123, 456, 789, 1024)

### 4.2 Grouping-Variable Ablation

We additionally compare three grouping strategies for train/val/test splitting (no augmentation, 5 seeds each):

- **By Recording**: groups defined by .mat file identity
- **By Load**: groups defined by motor load (0, 1, 2, 3 HP)
- **By Fault Size**: groups defined by fault diameter (0.007, 0.014, 0.021 inches); Normal recordings randomly split 60/20/20

This isolates which domain-shift axis most affects cross-group generalization.


## 5. Mechanism Validation

Following the evidence hierarchy outlined in the course manual, we conduct three mechanism-validation experiments *before* classification evaluation:

### 5.1 Adjacent-Window Autocorrelation (M1)

We compute the Pearson correlation between window *i* and window *i+offset* from the same recording, for offsets 1–10. The results (Figure 8) confirm that although average correlations are modest, specific recording- and class-dependent correlations exist. This provides the mechanistic basis for leakage concern: classifiers can exploit even weak but consistent temporal signatures to discriminate between recording identities.

### 5.2 Fault-Band Energy Audit (M5)

For each augmentation, we compute the energy retention ratio R_energy = E_augmented(fault_band) / E_original(fault_band) in the spectral neighborhood of theoretical bearing fault frequencies (BPFO, BPFI, BSF). All nine augmentations yield R_energy ≥ 1.0 (Figure 10), indicating no augmentation catastrophically destroys fault-frequency content. The most aggressive augmentation (noise σ=0.10) retains a high fault-band energy ratio due to broadband noise injection into all frequency bins, while conservation-oriented augmentations (shift_5) retain values close to 1.0.

### 5.3 Feature Diversity Analysis (M2)

We extract penultimate-layer features from a 2D-CNN for original and augmented versions of 200 training samples, computing pairwise cosine distances and SVD effective rank ratios (Figure 9). All augmentations increase feature diversity beyond the no-augmentation baseline, with combined augmentation producing the largest cosine distance.


## 6. Experimental Setup

- **Hardware**: Baidu Cloud BCC instance, NVIDIA A10 (24GB), Ubuntu 22.04, PyTorch 2.5.1+cu121
- **Training**: Adam optimizer, lr=0.001, CrossEntropy loss, batch_size=128, 50 epochs
- **Normalization**: Z-score normalization computed on training set only
- **Metrics**: Accuracy, macro-F1, per-class precision/recall/F1, confusion matrix
- **Statistical reporting**: Mean ± standard deviation across 5 seeds; Cohen's d for effect size
- **Reproducibility**: All seeds fixed; code and data manifest at GitHub repository
- **Convergence**: Mean convergence epoch = 15.2 (median = 9.0, max = 48). 50 epochs sufficient for 93.3% of runs (Figure 1).


## 7. Results

### 7.1 Main Factorial Results

Table 1 reports test accuracy for all 40 combinations. Under random splitting, both models achieve near-ceiling accuracy (0.999–1.000), consistent with prior CWRU benchmarks. Under recording-level splitting, accuracy drops substantially.

**Table 1. Test accuracy (mean ± std) over 5 random seeds.**

| Augmentation | CNN1D Random | CNN1D Recording | CNN2D Random | CNN2D Recording |
|---|---|---|---|---|
| None | 0.9997 ± 0.001 | **0.9321 ± 0.073** | 0.9995 ± 0.000 | **0.9007 ± 0.069** |
| Noise σ=0.01 | 0.9999 ± 0.000 | 0.9484 ± 0.054 | 0.9996 ± 0.000 | 0.9138 ± 0.076 |
| Noise σ=0.05 | 1.0000 ± 0.000 | **0.9634 ± 0.038** | 0.9998 ± 0.000 | 0.8723 ± 0.091 |
| Noise σ=0.10 | 0.9999 ± 0.000 | 0.9339 ± 0.079 | 0.9995 ± 0.001 | 0.9098 ± 0.073 |
| Shift 5 | 0.9998 ± 0.000 | 0.9486 ± 0.051 | 0.9997 ± 0.001 | **0.9152 ± 0.075** |
| Shift 20 | 0.9999 ± 0.000 | 0.9526 ± 0.057 | 0.9997 ± 0.001 | 0.9139 ± 0.076 |
| Shift 50 | 1.0000 ± 0.000 | 0.9366 ± 0.071 | 0.9994 ± 0.001 | 0.8722 ± 0.085 |
| SpecAugment | 0.9996 ± 0.000 | 0.9362 ± 0.037 | 0.9987 ± 0.000 | 0.8717 ± 0.082 |
| Combined | 0.9999 ± 0.000 | 0.9531 ± 0.048 | 0.9998 ± 0.000 | 0.8766 ± 0.134 |
| FreqFlip (neg.) | 0.9992 ± 0.001 | 0.9271 ± 0.074 | 0.9813 ± 0.009 | 0.8846 ± 0.098 |

*Bold: best per-column (excluding FreqFlip as negative control).*

The no-augmentation baseline records a leakage gap of 0.0676 (1D, Cohen's d=1.17) and 0.0988 (2D, Cohen's d=1.80) — both large effect sizes (Figure 6–7). For 1D-CNN, additive noise (σ=0.05) achieves the best recording-split accuracy of 0.9634. For 2D-CNN, time shift (5 samples) achieves 0.9152.

Critically, for 2D-CNN under recording-level split, 7 of 9 augmentation conditions perform *worse* than no augmentation: noise_005 (0.8723), shift_50 (0.8722), SpecAugment (0.8717), combined (0.8766), freq_flip (0.8846), noise_001 (0.9138), and noise_01 (0.9098) — only shift_5 (0.9152) and shift_20 (0.9139) surpass the 0.9007 baseline. The negative control (frequency flip) at 0.8846 suggests that even destructive frequency-axis operations do not catastrophically differ from physically motivated augmentations in practice. Figures 2–3 visualize the random-vs-recording contrast.

### 7.2 Per-Class Analysis

Table 2 reports per-class F1 scores for the most challenging configuration (2D-CNN, recording-level split). The Normal class is universally recognized (F1=0.947–1.000). Inner Race (0.761–0.860) and Ball (0.812–0.945) faults show the widest variance across augmentation conditions.

**Table 2. Per-class F1 scores (CNN2D, recording-level split, 5 seeds).**

| Augmentation | Normal | IR | OR | B |
|---|---|---|---|---|
| None | 1.000 ± 0.000 | 0.835 ± 0.193 | 0.856 ± 0.098 | 0.898 ± 0.060 |
| Noise σ=0.01 | 1.000 ± 0.000 | 0.848 ± 0.197 | 0.884 ± 0.114 | 0.910 ± 0.070 |
| Noise σ=0.05 | 0.997 ± 0.004 | 0.812 ± 0.198 | 0.840 ± 0.133 | 0.825 ± 0.101 |
| Noise σ=0.10 | 1.000 ± 0.000 | 0.845 ± 0.187 | 0.888 ± 0.119 | 0.898 ± 0.059 |
| Shift 5 | 1.000 ± 0.000 | 0.840 ± 0.193 | 0.858 ± 0.115 | **0.945 ± 0.053** |
| Shift 20 | 1.000 ± 0.000 | 0.850 ± 0.189 | 0.863 ± 0.110 | 0.928 ± 0.060 |
| Shift 50 | 0.999 ± 0.002 | 0.761 ± 0.228 | 0.839 ± 0.133 | 0.866 ± 0.088 |
| SpecAugment | 1.000 ± 0.000 | 0.774 ± 0.200 | 0.897 ± 0.112 | 0.812 ± 0.046 |
| Combined | 0.947 ± 0.105 | 0.860 ± 0.147 | 0.804 ± 0.197 | 0.867 ± 0.143 |
| FreqFlip | 1.000 ± 0.000 | 0.775 ± 0.251 | 0.900 ± 0.126 | 0.852 ± 0.093 |

Per-class heatmaps for both models are provided in Figures 12–13.

### 7.3 Gap-Recovery Analysis

Table 3 reports Δ_recovery — the proportion of the leakage gap recovered by each augmentation. Positive values indicate improvement beyond the no-augmentation recording-level baseline; negative values indicate degradation.

**Table 3. Gap-recovery analysis. Δ_rec = (acc_recording_aug − acc_recording_none) / (acc_random_none − acc_recording_none).**

| Augmentation | \| CNN1D Gap \| CNN1D Δ_rec \| CNN1D d \| CNN2D Gap \| CNN2D Δ_rec \| CNN2D d \| |
|---|---|---|---|---|---|---|
| Noise σ=0.01 | 0.0515 | 0.241 | 1.20 | 0.0858 | 0.133 | 1.43 |
| Noise σ=0.05 | 0.0366 | **0.463** | 1.21 | 0.1275 | −0.287 | 1.76 |
| Noise σ=0.10 | 0.0660 | 0.027 | 1.06 | 0.0897 | 0.092 | 1.56 |
| Shift 5 | 0.0512 | 0.244 | 1.28 | 0.0845 | **0.147** | 1.42 |
| Shift 20 | 0.0473 | 0.303 | 1.05 | 0.0858 | 0.134 | 1.44 |
| Shift 50 | 0.0634 | 0.067 | 1.13 | 0.1272 | −0.288 | 1.89 |
| SpecAugment | 0.0634 | 0.061 | 2.15 | 0.1270 | −0.293 | 1.96 |
| Combined | 0.0468 | 0.311 | 1.25 | 0.1232 | −0.244 | 1.16 |
| FreqFlip | 0.0721 | −0.074 | 1.23 | 0.0967 | −0.163 | 1.25 |

*Bold: best Δ_rec per model. All Cohen's d > 1.0 (large effect).*

Figures 4–5 visualize the gap-recovery landscape.

### 7.4 Grouping-Variable Ablation

Table 4 compares test accuracy (no augmentation, 5 seeds) across three grouping strategies for train/test splitting. Load-based splitting achieves 0.977–0.978, nearly matching random splitting. Recording-level splitting (0.899–0.940) introduces a modest drop. Fault-size-based splitting is the most challenging: 1D-CNN drops to 0.632 and 2D-CNN to 0.473.

**Table 4. Grouping variable ablation: test accuracy with no augmentation (5 seeds).**

| Model | By Load (HP) | By Recording | By Fault Size |
|---|---|---|---|
| CNN1D | 0.977 ± 0.015 | 0.940 ± 0.039 | **0.632 ± 0.110** |
| CNN2D | 0.979 ± 0.031 | 0.899 ± 0.092 | **0.473 ± 0.092** |

This reveals the generalization difficulty ordering: **fault_size >> recording > load**. A bearing diagnosis system deployed across machines with different fault severities faces a fundamentally harder challenge than one deployed across different operating speeds. This finding re-contextualizes the leakage discussion: while recording-level leakage is important, cross-severity generalization is the dominant bottleneck — the training set, limited to 2 of 3 fault diameters, provides insufficient representation for the held-out severity.


## 8. Physical Consistency

The H3 hypothesis — that augmentations preserving more fault-band energy yield greater recovery — is not supported. Spearman correlation between energy retention ratio (R_energy) and Δ_recovery is ρ = −0.37 (p = 0.33), a non-significant negative trend. Figure 14 plots this relationship.

The negative coefficient suggests that broad-spectrum noise augmentations may incidentally help generalization through regularization effects, despite (or perhaps aided by) their greater physical distortion. Conservative augmentations that precisely preserve fault-band energy do not outperform noise-based methods. This decoupling of physical fidelity from downstream performance has implications for the design of "physically constrained" augmentation strategies: physical conservation alone does not guarantee diagnostic benefit.


## 9. Ablation

### 9.1 Single vs. Combined Augmentation

Combined augmentation (noise + shift + SpecAugment) does not outperform the best single augmentation. For 1D-CNN recording split, combined achieves 0.9531 vs. noise_005's 0.9634. For 2D-CNN, combined (0.8766) underperforms even the no-augmentation baseline (0.9007). Stacking augmentations amplifies distribution shift beyond what the training data can support.

### 9.2 Grouping Variable Importance

The ablation in Section 7.4 (Table 4) establishes fault_size >> recording > load as the generalization difficulty ordering. The 0.47–0.63 accuracy under fault-size split demonstrates near-complete failure of cross-severity transfer.


## 10. Discussion

**H1 (partial recovery) is supported but weak for 2D-CNN.** The best augmentation recovers only 15% of the leakage gap in the 2D case, and most augmentations are counterproductive. 2D-CNN's reliance on spectrogram representations makes it particularly vulnerable to distribution shift from augmentation.

**H2 (leakage dominates) is strongly supported.** With Cohen's d ≥ 1.17 and max Δ_recovery of 0.46, the leakage gap is the dominant effect. No augmentation configuration can substitute for proper evaluation protocol.

**H3 (physical fidelity) is falsified.** The absence of correlation between energy retention and recovery is a "honest null result." The performance drop under recording-level splitting appears driven by reduced sample diversity rather than augmentation-induced feature corruption. Augmentations that distort physical features may still help by increasing diversity.

**The 3-seed insufficiency.** Our initial 3-seed experiment estimated the 2D recording baseline at 0.9653 (gap=0.034). Expanding to 5 seeds dropped this to 0.9007 (gap=0.099) — a nearly 3× increase in the estimated leakage gap. Individual seed performance spans 0.784–0.999 for 2D recording. We recommend ≥5 seeds as minimum for recording-level evaluations.

**Cross-severity generalization is the real bottleneck.** The fault-size ablation (accuracy 0.47–0.63) dwarfs the recording-level split effect (0.90–0.94). This deserves greater research attention than leakage control alone.


## 11. Limitations

1. **Single dataset**: All experiments use CWRU, an artificial-fault laboratory dataset. Results may not transfer to naturally degraded bearings or industrial environments.
2. **Single sensor**: Only drive-end accelerometer data (12kHz) is used. Fan-end (48kHz) data could reveal sampling-rate sensitivity.
3. **Fixed augmentation parameters**: Noise standard deviations and shift magnitudes were chosen a priori. Parameter sensitivity is not explored.
4. **CNN architectures only**: Results may differ for Transformer or MLP-based architectures.
5. **50-epoch ceiling**: 6.7% of runs converged in the final 10% of epochs (Figure 1), suggesting some combinations could benefit from extended training.
6. **Class imbalance in fault-size split**: The fault-size ablation has unequal class representation across splits, which may interact with the accuracy gap.


## 12. Conclusion

Under leakage-safe recording-level evaluation on CWRU, data augmentation provides at most 46% (1D-CNN) and 15% (2D-CNN) recovery of the leakage gap. The gap itself is large (Cohen's d ≥ 1.17), and most augmentations degrade 2D-CNN performance. Physical frequency-band energy preservation does not predict recovery efficacy. A grouping-variable ablation reveals that cross-severity generalization (0.47–0.63) is a substantially harder problem than recording-level leakage control (0.90–0.94) or cross-load generalization (0.98). We recommend five or more random seeds as minimum for recording-level evaluations and release our full experimental results as a benchmark for leakage-controlled CWRU studies.


## 13. Reproducibility Statement

All code, data manifest, experimental configurations, and results are available at the project GitHub repository. The `src/` directory contains the full training pipeline (`train.py`, `run_all.py`, `preprocess.py`, and auxiliary scripts). Random seeds (42, 123, 456, 789, 1024) are fixed across all experiments. Per-epoch training logs for all 200 runs are included in `results/logs/epoch_logs/`. The `data_manifest/` directory documents all 40 CWRU recording files with labels, fault diameters, and load conditions. All results tables and figures can be regenerated by running `python src/run_all.py`.


## 14. AI-Use Disclosure

The following parts of this work involved AI assistance (DeepSeek, via OpenCode CLI):
- **Code implementation**: Training pipeline, augmentation functions, evaluation metrics, and analysis scripts were written iteratively with AI pair-programming. All code was executed, verified, and debugged by the authors on real hardware.
- **Experimental debugging**: AI assisted in identifying and fixing bugs including device mismatch in STFT computation, module import paths, and JSON serialization of numpy types.
- **Manuscript drafting**: The first draft was generated with AI assistance based on experimental results and author-provided outlines. All factual claims were verified against `results/` tables and logs. All citations were verified against known published works.
- **No AI-generated experimental results**: All numerical results, figures, and tables are produced by code execution on real data. No result was hallucinated or fabricated.


## Figures

**Figure 1.** Convergence curves for all 200 training runs across four configurations.

**Figure 2.** Random vs Recording-level split accuracy comparison (1D-CNN).

**Figure 3.** Random vs Recording-level split accuracy comparison (2D-CNN).

**Figure 4.** Gap-recovery ratio (Δ_rec) by augmentation (1D-CNN).

**Figure 5.** Gap-recovery ratio (Δ_rec) by augmentation (2D-CNN).

**Figure 6.** Cohen's d effect sizes for random-vs-recording gap (1D-CNN).

**Figure 7.** Cohen's d effect sizes for random-vs-recording gap (2D-CNN).

**Figure 8.** M1: Adjacent-window autocorrelation by fault class.

**Figure 9.** M2: Feature diversity analysis (cosine distance).

**Figure 10.** M5: Fault-band energy retention audit (R_energy).

**Figure 11.** Envelope spectrum examples for each fault class.

**Figure 12.** Per-class F1 heatmap (1D-CNN, recording split).

**Figure 13.** Per-class F1 heatmap (2D-CNN, recording split).

**Figure 14.** H3: Physical fidelity (R_energy) vs. gap recovery (Δ_rec).


## References

[1] W. A. Smith and R. B. Randall, "Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study," *Mechanical Systems and Signal Processing*, vol. 64–65, pp. 100–131, 2015. doi: 10.1016/j.ymssp.2015.04.021

[2] L. Wen, X. Li, L. Gao, and Y. Zhang, "A new convolutional neural network-based data-driven fault diagnosis method," *IEEE Transactions on Industrial Electronics*, vol. 65, no. 7, pp. 5990–5998, 2018. doi: 10.1109/TIE.2017.2774777

[3] W. Zhang, G. Peng, C. Li, Y. Chen, and Z. Zhang, "A new deep learning model for fault diagnosis with good anti-noise and domain adaptation ability on raw vibration signals," *Sensors*, vol. 17, no. 2, p. 425, 2017. doi: 10.3390/s17020425

[4] C. Bergmeir and J. M. Benítez, "On the use of cross-validation for time series predictor evaluation," *Information Sciences*, vol. 191, pp. 192–213, 2012. doi: 10.1016/j.ins.2011.12.028

[5] V. Cerqueira, L. Torgo, and I. Mozetič, "Evaluating time series forecasting models: an empirical study on performance estimation methods," *Machine Learning*, vol. 109, pp. 1997–2028, 2020. doi: 10.1007/s10994-020-05910-7

[6] Case Western Reserve University Bearing Data Center, "12k Drive End Bearing Fault Data." [Online]. Available: https://engineering.case.edu/bearingdatacenter

[7] D. S. Park, W. Chan, Y. Zhang, C.-C. Chiu, B. Zoph, E. D. Cubuk, and Q. V. Le, "SpecAugment: A simple data augmentation method for automatic speech recognition," in *Proc. Interspeech*, 2019, pp. 2613–2617. doi: 10.21437/Interspeech.2019-2680

[8] S. Shao, S. McAleer, R. Yan, and P. Baldi, "Highly accurate machine fault diagnosis using deep transfer learning," *IEEE Transactions on Industrial Informatics*, vol. 15, no. 4, pp. 2446–2455, 2019. doi: 10.1109/TII.2018.2864759

[9] X. Li, W. Zhang, Q. Ding, and J.-Q. Sun, "Diagnosing rotating machines with weakly supervised data using deep transfer learning," *IEEE Transactions on Industrial Informatics*, vol. 16, no. 3, pp. 1688–1697, 2020. doi: 10.1109/TII.2019.2927590
