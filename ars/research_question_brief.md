# Research Question Brief

## Title

To What Extent Does Data Augmentation Compensate for Leakage-Safe Splitting? A Controlled Interaction Study with Physical Constraint Verification on CWRU Bearing Fault Diagnosis

数据增强在多大程度上能弥补泄漏安全切分的泛化损失？基于 CWRU 轴承故障诊断的受控交互研究（含物理约束验证）

## Chapter Dependencies

| 分段 | 依赖章节 | 用途 |
|------|----------|------|
| 第一段（基础） | ch30, ch37 | 信号数组操作、过拟合与泄漏控制概念 |
| 第二段（主力方法） | ch38, ch39, ch40-41, ch44, ch46 | 数据增强实现、1D-CNN、STFT+2D-CNN、数据集认知、频谱图预处理 |
| 第三段（验证） | **ch45**, **ch47**, ch49 | 频谱图物理约束（增强安全边界）、经典频域分析（故障频率计算）、评估五大挑战（研究动机） |

## Research Question

Under recording-level (leakage-safe) train-test splitting, to what extent can data augmentation — specifically Gaussian noise injection, time shifting, and waveform-compatible spectral/time masking — recover the generalization performance lost when random adjacent-window leakage is removed? Furthermore, does effectiveness correlate with symmetric preservation of envelope-spectrum fault-band energy?

在按原始记录切分（泄漏安全）的训练—测试协议下，数据增强（高斯噪声注入、时移、适配原始波形的频谱/时间掩蔽）能在多大程度上恢复因移除随机相邻窗口泄漏而损失的泛化性能？进一步地，增强的有效性是否与包络谱故障频带能量的对称保真度相关？

## Problem

Random adjacent-window train-test splitting is a known risk in CWRU bearing diagnosis. With the fixed 50% overlap used here, every pair of adjacent windows shares exactly 512 of 1,024 samples. When windows from one source recording are independently assigned to train and test sets, this creates a direct information-sharing path and can inflate the estimated performance; the magnitude is measured rather than assumed.

Recording-level splitting (all windows from the same original recording assigned exclusively to one split) prevents source-recording overlap and measures performance on the selected unseen recordings. Data augmentation — noise injection, time shifting, spectral masking — is frequently proposed to improve robustness and generalization. However, its recovery effect under recording-disjoint evaluation must be measured directly rather than inferred from random-window performance.

This gap is critical: if augmentation cannot meaningfully recover the generalization gap, then (a) reported "augmentation improves robustness" claims may be confounded by evaluation protocol, and (b) researchers should prioritize fixing their split protocol before investing in augmentation strategies.

A second, orthogonal concern is the **physical fidelity of augmentation**. ch45 establishes that frequency and time axes carry physical meaning; frequency-axis flipping or arbitrary masking can distort the signal. Even Gaussian noise and time shifting may alter envelope-spectrum fault-band energy. This study therefore evaluates augmentation through the dual lens of protocol integrity and measurable physical preservation without claiming an exhaustive literature first.

随机相邻窗口切分是 CWRU 轴承诊断中的已知风险。本实验固定使用 50% 重叠，因此每对相邻窗口恰好共享 512/1024 个采样点；若同一原始记录的窗口被独立分配到训练集和测试集，就形成直接的信息共享路径。虚高幅度由实验测量，而不预设为某个百分比。按原始记录切分可切断这一跨集合路径，数据增强能否弥补由此暴露的泛化差距则是本文要检验的问题。

第二个同等重要的关切是**增强的物理保真度**。ch45 确立了频率与时间轴带有物理量纲；频率轴翻转或任意掩蔽可能扭曲信号。即便高斯噪声、时移也可能改变包络谱故障频带能量。因此本文同时检查评估协议完整性与可测量的物理保真度，但不声称穷尽了全部既有研究。

## Bottleneck

The specific limitation this project studies is twofold:

**B1 (Protocol dimension):** Leakage-safe splitting provides a more independent estimate of CWRU generalization, but the amount that augmentation can compensate under this exact controlled protocol is unknown. The experiment decomposes the observed difference into the random-vs-recording gap and the fraction recovered by augmentation; it does not claim an exhaustive literature first.

**B2 (Physical dimension):** Data augmentation operations — even standard ones like Gaussian noise injection — modify the signal's spectral content. It is unknown which augmentation types preserve fault-frequency-band energy (BPFO/BPFI/BSF) and which distort it, and whether physical preservation correlates with generalization benefit. Without this check, "augmentation improves robustness" is an unvalidated claim — the improvement may come from physically irrelevant signal variation rather than from learning more fault-relevant features.

**Concrete compound gap**: The interaction effect (split type × augmentation condition) has not been isolated, and simultaneously, the physical fidelity of augmented signals has not been measured. Most CWRU studies evaluate augmentation under random splitting (confounded by leakage) or under recording-level splitting without a random-split baseline. Neither approach addresses physical safety.

本项目研究的具体瓶颈包含两个维度：

**B1（协议维度）：** 泄漏安全切分揭示了 CWRU 诊断模型的真实泛化能力，但数据增强能否、在多大程度上以及通过何种机制弥补由此产生的准确率损失——这一问题目前缺乏答案。尚无受控研究将准确率差距分解为"泄漏贡献的部分"与"增强可恢复的部分"。

**B2（物理维度）：** 数据增强操作——即使是常用的高斯噪声注入——也会改变信号的频谱内容。目前不清楚哪些增强类型保留了故障频带能量（BPFO/BPFI/BSF）、哪些扭曲了它，以及物理保留是否与泛化收益相关。缺少这一检查，"增强提升鲁棒性"的结论缺乏物理基础——性能提升可能来自物理无关的信号变化，而非学习到更多故障相关特征。

**复合空白**：切分类型 × 增强条件的交互效应未被分离，同时增强信号的物理保真度未被测量。大多数 CWRU 研究在随机切分下评估增强（被泄漏混淆），或在按记录切分下评估增强但缺乏随机切分基线。两者均未触及物理安全性。

## Hypothesis

**H1 (Partial recovery)**: Data augmentation will partially recover the accuracy gap between random-split and recording-level-split evaluation — but recovery will be incomplete.

**H2 (Protocol-gap dominance)**: The random-window to recording-level accuracy gap will exceed the gain from any augmentation, so recovery will saturate well below the random-split ceiling. This operational comparison does not identify what fraction of the gap comes from exact shared-sample leakage versus the broader challenge of generalizing to unseen recordings.

**H3 (Physical preservation → recovery correlation)**: Augmentations that preserve fault-frequency-band energy (BPFO/BPFI/BSF) — such as moderate time shifting — will provide stronger gap recovery than augmentations that distort fault-band energy — such as high-intensity Gaussian noise or frequency-axis manipulation. That is, the physical constraint of ch45 is not merely a theoretical concern: violating it measurably degrades augmentation's practical benefit.

**H1（部分恢复）**：数据增强将部分恢复随机切分与按记录切分之间的准确率差距——但恢复将是不完全的。

**H2（协议差距主导）**：随机窗口切分到按记录切分的准确率差距将大于任一增强带来的收益，增强恢复会在远低于随机切分天花板的水平上饱和。该操作性比较不能识别差距中有多少来自精确共享样本泄漏，又有多少来自对未见录制文件的更广义泛化难度。

**H3（物理保留 → 恢复相关性）**：保留故障频带能量（BPFO/BPFI/BSF）的增强——如适度时移——将比扭曲故障频带能量的增强——如高强度高斯噪声或频率轴操作——提供更强的差距恢复。即 ch45 的物理约束不仅是理论关切：违反它会在实践中可测量地削弱增强效果。

## Dataset(s)

- **Primary**: CWRU Bearing Dataset — Drive End (DE) accelerometer, 12 kHz sampling, all 4 motor loads (0/1/2/3 HP), all fault types (inner race / outer race @ 6 o'clock / ball) and diameters (0.007"/0.014"/0.021"), plus normal baseline
- **Additional**: None. The project uses the CWRU-only minimal-project option in manual §19 and therefore makes no cross-dataset generalization claim. Candidate CWRU-like datasets and the reason for deferral are documented in `data_manifest/dataset_discovery_report.md`.

## Method Family

Comparative evaluation protocol study with factorial design, incorporating physical constraint verification (per ch45 and ch47):

- **Factor A — Split protocol (2 levels)**: Random adjacent-window split vs. Recording-level split
- **Factor B — Augmentation condition (10 levels)**: No augmentation, Gaussian noise (σ = 0.01/0.05/0.1), time shifting (±5/±20/±50 samples), waveform-compatible spectral/time masking (two FFT-bin masks plus two time-sample masks), combined noise+shift, and **negative control: FFT-magnitude frequency-axis flipping**
- **Factor C — Model family (2 levels)**: Raw 1D CNN, STFT Spectrogram + 2D CNN
- **Additional grouping ablation**: recording identity vs. motor load (0/1/2/3 HP) vs. fault diameter
- **Factor E — Contamination level (3+ levels, for robustness curve)**: Clean test set, mild noise (σ = 0.02), moderate noise (σ = 0.1) — contamination applied only at test time to evaluate robustness degradation slope

The main three factors are fully crossed with five fixed random seeds (42, 123, 456, 789, 1024).

Fixed spectrogram parameters: STFT window = 128, overlap = 0.5, n_fft = 128 (per ch40-41), not varied across experiments.

## Baselines

| # | Baseline | What It Tests |
|---|----------|---------------|
| 1 | Raw 1D CNN, no augmentation, random split | Leakage-prone protocol ceiling; not a causal upper bound on leakage alone |
| 2 | Raw 1D CNN, no augmentation, recording-level split | True generalization floor: lower bound without augmentation |
| 3 | STFT + 2D CNN, no augmentation, random split | Second model family's leakage-prone protocol ceiling |
| 4 | STFT + 2D CNN, no augmentation, recording-level split | Second model family generalization floor |
| 5 | Gaussian noise augmentation (σ = 0.01, 0.05, 0.1) × both splits × both models | Augmentation effectiveness: noise sensitivity + physical preservation (H3) |
| 6 | Time shifting (±5, ±20, ±50 samples) × both splits × both models | Augmentation effectiveness: temporal diversity + physical preservation (H3) |
| 7 | Waveform-compatible spectral/time masking × both splits × both models | Augmentation effectiveness: spectral and temporal diversity without fake image reshaping |
| 8 | Combined augmentation (best moderate individual settings) × both splits × both models | Synergy or interference between augmentation types |
| 9 | **Negative control: FFT-magnitude frequency-axis flipping** × both splits × both models | Physically destructive augmentation (ch45: frequency carries physical meaning) — expected to degrade or provide zero benefit |

## Mechanism Validation

All mechanism validation is numerical — no subjective visual interpretation required. Tests are organized into two categories: **leakage evidence** (M1-M3) and **physical fidelity** (M4, M4-neg).

### Leakage Evidence Tests

| # | Test | Input | Metric | What It Shows |
|---|------|-------|--------|---------------|
| M1 | Structural adjacent-window audit | Raw windows from the same recording, offsets 1–10 | Exact shared-sample fraction, aligned-overlap correlation/MAE, plus zero-lag Pearson r | Separates guaranteed sample duplication from data-dependent waveform similarity |
| M2 | Input-representation diversity | Standardized 32×32 STFT representations of paired original/augmented samples | Cosine distance distribution and effective rank | Measures how strongly augmentation changes the actual 2D-CNN input representation |
| M3 | Training-curve diagnostic | Per-epoch train/validation curves | Best-validation epoch and late-epoch degradation counts | Describes training behavior without equating first-best epoch with formal convergence |
| M4 | Per-class gap decomposition | Accuracy by fault type (normal/IR/OR/ball) under each split × augmentation condition | Per-class Δ_recovery ratio | Reveals which fault types benefit most from augmentation and which are dominated by leakage — critical for evaluating whether "augmentation helps" is a class-uniform claim or varies by fault type |

### Physical Fidelity Tests (from 方向三 / ch45 constraint)

| # | Test | Input | Metric | What It Shows |
|---|------|-------|--------|---------------|
| M5 | **Envelope-spectrum fault-band fidelity** | Augmented vs. original standardized waveforms | `min(E_aug/E_orig, E_orig/E_aug)` in BPFO/BPFI/BSF bands (±5% around first three harmonics) | Penalizes both attenuation and amplification; values near 1 indicate energy preservation |
| M5-neg | **Frequency-shifted reference band** | Same as M5, shifted +100 Hz | Symmetric fidelity in shifted vs. true fault band | Tests whether apparent preservation is fault-band specific or spectrally broad |

The physical fidelity tests are performed on all augmentation types (B5-B9) to test H3: augmentations ranked by M5 energy retention should correlate with their Δ_recovery rankings.

## Metrics

**Primary**:
- Classification accuracy
- Macro-F1

**Secondary**:
- Per-class precision, recall
- **Gap-recovery ratio** (key contribution metric):
  ```
  Δ_recovery = (Acc_aug_recording − Acc_noaug_recording) / (Acc_noaug_random − Acc_noaug_recording)
  ```
  - Δ_recovery = 1.0 → augmentation fully compensates for leakage
  - Δ_recovery ≈ 0 → augmentation has no effect
  - Δ_recovery < 0 → augmentation harms performance under leakage-safe split
- **Symmetric fault-band energy fidelity** (physical metric, per M5):
  ```
  F_energy = min(E_aug / E_orig, E_orig / E_aug)
  ```
  - F_energy ≥ 0.95 → close energy preservation
  - F_energy < 0.80 → substantial energy distortion

**Stability**:
- Mean ± standard deviation across exactly 5 fixed random seeds (model initialization + augmentation randomness)
- Reported in all tables as μ ± σ

**Non-accuracy dimensions** (§1.8):

| Dimension | Test | Metric |
|-----------|------|--------|
| **Cross-load generalization** | Group train/validation/test partitions by motor load across five seeds | Accuracy and Macro-F1 under held-out-load evaluation |
| **Robustness curve** (from 方向三) | Test models trained with each augmentation on progressively contaminated test sets (clean → mild noise σ=0.02 → moderate noise σ=0.1) | Degradation slope: ΔAcc / Δnoise_level. A flatter slope indicates augmentation-induced robustness. Compare slopes across augmentation types and split protocols. |
| **Physical consistency** | Correlation between M5 fidelity ranking and Δ_recovery ranking across augmentation types | Model-specific Spearman rank correlation between F_energy and Δ_recovery; positive significant association supports H3, while a null result means H3 is not supported. |

## Expected Contribution

1. **Controlled characterization** of the interaction between evaluation protocol (random vs. recording-level split) and data augmentation in the selected CWRU subset
2. **A quantitative metric** (gap-recovery ratio Δ_recovery) for measuring augmentation effectiveness under leakage-safe conditions — applicable to future CWRU studies
3. **Physical-fidelity audit of augmentation**: quantifying how strongly each augmentation preserves symmetric fault-frequency-band energy (BPFO/BPFI/BSF) without equating one metric with complete physical safety
4. **Correlation between physical preservation and practical benefit**: testing whether augmentations that maintain fault-band energy provide stronger generalization recovery — if confirmed, this provides a physics-based criterion for selecting and designing augmentation strategies for bearing diagnosis
5. **Practical guidance**: evidence on whether researchers should prioritize fixing evaluation protocols, selecting physically safe augmentations, or both
6. **A reproducible experimental protocol** (2 splits × 10 augmentation conditions × 2 models × 5 seeds), plus grouping and contamination checks

## Main Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Augmentation recovery effect is too small to be meaningful | Medium | Framed as "establishing the boundary of augmentation effectiveness" — a careful negative result with good controls is valid per §17.2 |
| Sensitivity to augmentation hyperparameters may dominate the signal | Medium | Systematic sensitivity sweep (3 strengths per augmentation type) rather than picking one arbitrary setting |
| Only 4 CWRU motor loads limits cross-load generalizability | High | Acknowledged explicitly; external candidates are documented in `data_manifest/dataset_discovery_report.md` but deliberately deferred under the minimal-project option |
| Results limited to 2 model families | Medium | Acknowledged as scope limitation; 1D CNN and 2D CNN cover the two dominant CWRU model paradigms |
| STFT parameters add a hidden degree of freedom | Low | Fixed STFT parameters (window=128, overlap=0.5, n_fft=128) — documented and not varied |
| Fault-frequency band energy retention is measured using theoretical fault frequencies, but CWRU's actual shaft speed may deviate from nominal values (1797/1772/1750/1730 RPM) | Medium | (1) Compute fault frequencies as a ±2% band around theoretical values to account for speed variation; (2) Cross-validate with envelope spectrum peak detection to confirm actual fault frequencies; (3) If discrepancy is large, report both theoretical-band and detected-peak energy retention |
| Robustness curve requires generating contaminated test sets — parameters (noise σ, magnitude) are arbitrary | Low | Fixed contamination parameters determined before seeing results; contamination applied only at test time, never during training (leakage prevention per §6.3) |

## What Would Falsify the Hypothesis

| Hypothesis | Falsification Condition |
|-----------|------------------------|
| **H1** (partial recovery) | Δ_recovery ≈ 0 for all augmentation conditions → augmentation has no measurable effect under either split protocol — the performance gap is entirely determined by split protocol alone |
| **H2** (protocol-gap dominance) | Δ_recovery ≈ 1.0 for any augmentation condition → augmentation nearly compensates for the operational protocol gap, so the gap does not dominate augmentation gain |
| **H3** (physical preservation → recovery correlation) | Spearman ρ ≈ 0 between F_energy (M5) and Δ_recovery → no measured association between energy fidelity and generalization benefit under this protocol. This does not invalidate the physical constraint; it limits the downstream claim. |

The most scientifically valuable outcome is not confirmation of all three hypotheses, but a well-controlled experiment that clearly establishes **where the boundaries lie** — whether Δ_recovery saturates at 0.1, 0.3, or 0.5, and whether physical preservation (H3) meaningfully constrains the effectiveness ranking of augmentation strategies.
