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

Under recording-level (leakage-safe) train-test splitting, to what extent can data augmentation — specifically Gaussian noise injection, time shifting, and SpecAugment — recover the generalization performance lost when random adjacent-window leakage is removed? Furthermore, does the effectiveness of augmentation correlate with its preservation of fault-frequency-band energy — i.e., do augmentations that maintain physical signal integrity provide stronger recovery than those that disrupt it?

在按原始记录切分（泄漏安全）的训练—测试协议下，数据增强（高斯噪声注入、时移、SpecAugment）能在多大程度上恢复因移除随机相邻窗口泄漏而损失的泛化性能？进一步地，增强的有效性是否与其保留故障频带物理特征的程度相关——即保持信号物理完整性的增强是否比破坏物理特征的增强提供更强的恢复？

## Problem

In CWRU bearing fault diagnosis, a substantial portion of the literature employs random adjacent-window train-test splitting. Windows cut from the same continuous vibration recording share near-identical waveforms. When these adjacent windows are randomly assigned to train and test sets, the model learns to recognize recording identity rather than fault signature, inflating reported accuracy by 10–30 percentage points.

Recording-level splitting (all windows from the same original recording assigned exclusively to one split) corrects this leakage but exposes the model's true generalization capability. Data augmentation — noise injection, time shifting, spectral masking — is frequently proposed as a remedy to improve robustness and generalization. However, the actual recovery effect of augmentation under leakage-safe protocols has not been systematically characterized. It is unknown whether augmentation genuinely improves fault-feature learning, or whether its reported benefits are themselves partially artifacts of leakage-prone evaluation.

This gap is critical: if augmentation cannot meaningfully recover the generalization gap, then (a) reported "augmentation improves robustness" claims may be confounded by evaluation protocol, and (b) researchers should prioritize fixing their split protocol before investing in augmentation strategies.

A second, orthogonal concern is the **physical safety of augmentation**. ch45 establishes that spectrogram axes carry physical units (frequency and time); augmentations borrowed from ImageNet — such as frequency-axis flipping, extreme scaling, or arbitrary masking — can distort the physical meaning of the signal. Even commonly used augmentations (Gaussian noise, time shifting) may alter the fault-frequency-band energy that is the physical basis of bearing diagnosis. **No existing CWRU study simultaneously evaluates augmentation through the dual lens of (1) evaluation protocol integrity and (2) physical feature preservation.** This is the compound gap RQ-4 addresses.

在 CWRU 轴承故障诊断中，大量文献采用随机相邻窗口训练—测试切分。同一段连续振动记录切出的相邻窗口波形几乎相同，当相邻窗口被随机分配到训练集和测试集时，模型学到的是记录身份而非故障特征，使报告的准确率虚高 10-30 个百分点。按原始记录切分修正了这一泄漏，但暴露了模型真实的泛化能力。数据增强（噪声注入、时移、频谱掩蔽）常被提议为提升鲁棒性和泛化能力的补救手段，但其在泄漏安全协议下的实际恢复效果尚未被系统刻画。如果增强无法有效弥补泛化差距，那么(a) 文献中"增强提升鲁棒性"的结论本身可能被评估协议的泄漏所混淆，(b) 研究者在投入增强策略之前，应优先修正其切分协议。

第二个同等重要的关切是**增强的物理安全性**。ch45 确立了频谱图坐标轴带有物理量纲（频率和时间）；借用自 ImageNet 的增强——如频率轴翻转、极端缩放、任意掩蔽——可能扭曲信号的物理含义。即便是常用的增强（高斯噪声、时移）也可能改变故障频带能量——即轴承诊断的物理基础。**目前尚无 CWRU 研究同时从(1)评估协议完整性和(2)物理特征保留这两个维度来评估数据增强。** 这是 RQ-4 解决的复合空白。

## Bottleneck

The specific limitation this project studies is twofold:

**B1 (Protocol dimension):** Leakage-safe splitting reveals the true generalization capability of CWRU diagnostic models, but it is unknown whether, how much, and through what mechanism data augmentation can compensate for the resulting accuracy loss. No existing controlled study decomposes the accuracy gap into "what leakage gave us" versus "what augmentation can recover" — the interaction between split protocol and augmentation strategy is an unexplored dimension.

**B2 (Physical dimension):** Data augmentation operations — even standard ones like Gaussian noise injection — modify the signal's spectral content. It is unknown which augmentation types preserve fault-frequency-band energy (BPFO/BPFI/BSF) and which distort it, and whether physical preservation correlates with generalization benefit. Without this check, "augmentation improves robustness" is an unvalidated claim — the improvement may come from physically irrelevant signal variation rather than from learning more fault-relevant features.

**Concrete compound gap**: The interaction effect (split type × augmentation condition) has not been isolated, and simultaneously, the physical fidelity of augmented signals has not been measured. Most CWRU studies evaluate augmentation under random splitting (confounded by leakage) or under recording-level splitting without a random-split baseline. Neither approach addresses physical safety.

本项目研究的具体瓶颈包含两个维度：

**B1（协议维度）：** 泄漏安全切分揭示了 CWRU 诊断模型的真实泛化能力，但数据增强能否、在多大程度上以及通过何种机制弥补由此产生的准确率损失——这一问题目前缺乏答案。尚无受控研究将准确率差距分解为"泄漏贡献的部分"与"增强可恢复的部分"。

**B2（物理维度）：** 数据增强操作——即使是常用的高斯噪声注入——也会改变信号的频谱内容。目前不清楚哪些增强类型保留了故障频带能量（BPFO/BPFI/BSF）、哪些扭曲了它，以及物理保留是否与泛化收益相关。缺少这一检查，"增强提升鲁棒性"的结论缺乏物理基础——性能提升可能来自物理无关的信号变化，而非学习到更多故障相关特征。

**复合空白**：切分类型 × 增强条件的交互效应未被分离，同时增强信号的物理保真度未被测量。大多数 CWRU 研究在随机切分下评估增强（被泄漏混淆），或在按记录切分下评估增强但缺乏随机切分基线。两者均未触及物理安全性。

## Hypothesis

**H1 (Partial recovery)**: Data augmentation will partially recover the accuracy gap between random-split and recording-level-split evaluation — but recovery will be incomplete.

**H2 (Leakage dominance)**: The majority of the inflated accuracy under random splitting is attributable to recording-identity leakage rather than generalizable fault-feature learning. Augmentation recovery will saturate well below the random-split ceiling.

**H3 (Diminishing returns)**: Stronger augmentation (larger noise magnitude, larger time shifts, combining multiple augmentation types) will recover more of the gap, but with diminishing returns and potential accuracy degradation at extreme strengths.

**H4 (Physical preservation → recovery correlation)**: Augmentations that preserve fault-frequency-band energy (BPFO/BPFI/BSF) — such as moderate time shifting — will provide stronger gap recovery than augmentations that distort fault-band energy — such as high-intensity Gaussian noise or frequency-axis manipulation. That is, the physical constraint of ch45 is not merely a theoretical concern: violating it measurably degrades augmentation's practical benefit.

**H1（部分恢复）**：数据增强将部分恢复随机切分与按记录切分之间的准确率差距——但恢复将是不完全的。

**H2（泄漏主导）**：随机切分下虚高的准确率主要由记录身份泄漏驱动，而非泛化的故障特征学习。增强恢复将在远低于随机切分天花板的水平上饱和。

**H3（边际递减）**：更强的增强（更大噪声幅度、更大时移、多种增强叠加）将恢复更多差距，但呈边际递减趋势，且在极端强度下可能出现准确率下降。

**H4（物理保留 → 恢复相关性）**：保留故障频带能量（BPFO/BPFI/BSF）的增强——如适度时移——将比扭曲故障频带能量的增强——如高强度高斯噪声或频率轴操作——提供更强的差距恢复。即 ch45 的物理约束不仅是理论关切：违反它会在实践中可测量地削弱增强效果。

## Dataset(s)

- **Primary**: CWRU Bearing Dataset — Drive End (DE) accelerometer, 12 kHz sampling, all 4 motor loads (0/1/2/3 HP), all fault types (inner race / outer race @ 6 o'clock / ball) and diameters (0.007"/0.014"/0.021"), plus normal baseline
- **Additional**: [TBD — to be identified in Section 5: Dataset Discovery]

## Method Family

Comparative evaluation protocol study with factorial design, incorporating physical constraint verification (per ch45 and ch47):

- **Factor A — Split protocol (2 levels)**: Random adjacent-window split vs. Recording-level split
- **Factor B — Augmentation condition (7+ levels)**: No augmentation, Gaussian noise (3 strengths: σ = 0.01/0.05/0.1), time shifting (3 magnitudes: ±5/±20/±50 samples), SpecAugment (2 frequency-time masks), combined augmentation (best moderate settings), **negative control: frequency-axis flipping** (physically destructive per ch45 — expected to harm performance)
- **Factor C — Model family (2 levels)**: Raw 1D CNN, STFT Spectrogram + 2D CNN
- **Factor D — Load condition (4 levels, for cross-load test)**: 0/1/2/3 HP
- **Factor E — Contamination level (3+ levels, for robustness curve)**: Clean test set, mild noise (σ = 0.02), moderate noise (σ = 0.1) — contamination applied only at test time to evaluate robustness degradation slope

All factors crossed where applicable. Fixed random seeds (≥3) for stability assessment.

Fixed spectrogram parameters: STFT window = 128, overlap = 0.5, n_fft = 128 (per ch40-41), not varied across experiments.

## Baselines

| # | Baseline | What It Tests |
|---|----------|---------------|
| 1 | Raw 1D CNN, no augmentation, random split | Leakage ceiling: upper bound of what leakage can produce |
| 2 | Raw 1D CNN, no augmentation, recording-level split | True generalization floor: lower bound without augmentation |
| 3 | STFT + 2D CNN, no augmentation, random split | Second model family leakage ceiling |
| 4 | STFT + 2D CNN, no augmentation, recording-level split | Second model family generalization floor |
| 5 | Gaussian noise augmentation (σ = 0.01, 0.05, 0.1) × both splits × both models | Augmentation effectiveness: noise sensitivity + physical preservation (H4) |
| 6 | Time shifting (±5, ±20, ±50 samples) × both splits × both models | Augmentation effectiveness: temporal diversity + physical preservation (H4) |
| 7 | SpecAugment (2 frequency-time masks) × both splits × both models | Augmentation effectiveness: spectral diversity |
| 8 | Combined augmentation (best moderate individual settings) × both splits × both models | Synergy or interference between augmentation types |
| 9 | **Negative control: frequency-axis flipping** × recording-level split × both models | Physically destructive augmentation (ch45: frequency axis carries physical meaning) — should degrade or provide zero benefit; validates that the augmentation benefit framework is physically grounded, not merely "any perturbation helps" |

## Mechanism Validation

All mechanism validation is numerical — no subjective visual interpretation required. Tests are organized into two categories: **leakage evidence** (M1-M3) and **physical fidelity** (M4, M4-neg).

### Leakage Evidence Tests

| # | Test | Input | Metric | What It Shows |
|---|------|-------|--------|---------------|
| M1 | Adjacent-window autocorrelation | Raw signal windows from same recording, varying offset (1–10 windows apart) | Pearson r between window pairs | Quantifies leakage severity: high r between adjacent windows confirms recording-identity signal dominates over fault-feature variation |
| M2 | Feature-space diversity from augmentation | Last hidden layer features of original and augmented samples | Cosine distance distribution, effective rank of feature matrix | Verifies that augmentation actually increases feature diversity rather than adding trivial perturbation — augmentation that does not change feature-space geometry cannot improve generalization |
| M3 | Training curve divergence | Train/val accuracy curves over epochs | Gap between train and val curves (Δ_train-val) under recording-level split vs random split | Shows whether augmentation stabilizes training (narrows the generalization gap) or introduces training instability (widens it) under leakage-safe conditions |
| M4 | Per-class gap decomposition | Accuracy by fault type (normal/IR/OR/ball) under each split × augmentation condition | Per-class Δ_recovery ratio | Reveals which fault types benefit most from augmentation and which are dominated by leakage — critical for evaluating whether "augmentation helps" is a class-uniform claim or varies by fault type |

### Physical Fidelity Tests (from 方向三 / ch45 constraint)

| # | Test | Input | Metric | What It Shows |
|---|------|-------|--------|---------------|
| M5 | **Fault-frequency band energy retention** | Augmented signal spectra vs original signal spectra | Energy ratio in BPFO/BPFI/BSF bands (±5% tolerance around theoretical frequency and 2nd-3rd harmonics) before vs after augmentation | Quantifies physical safety: an augmentation that preserves >95% of fault-band energy is "physically safe"; one that preserves <80% is "physically destructive." Per ch47, fault frequencies are computed from CWRU bearing geometry (6205: BPFO ≈ 3.58×f_r, BPFI ≈ 5.42×f_r, BSF ≈ 2.32×f_r at 1797 RPM) |
| M5-neg | **Negative control: frequency-shifted reference band** | Same as M5, but using a frequency band shifted +100 Hz away from the true fault frequency | Energy ratio in shifted band vs true fault band | Distinguishes genuine fault-band retention from broadband noise retention. If the true band and shifted band show equal energy retention, then "preservation" is trivial (the augmentation is spectrally flat) rather than fault-specific |

The physical fidelity tests are performed on all augmentation types (B5-B9) to test H4: augmentations ranked by M5 energy retention should correlate with their Δ_recovery rankings.

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
- **Fault-band energy retention ratio** (physical safety metric, per M5):
  ```
  R_energy = Σ(energy in BPFO/BPFI/BSF bands of augmented signal) / Σ(energy in same bands of original signal)
  ```
  - R_energy ≥ 0.95 → physically safe augmentation
  - R_energy < 0.80 → physically destructive augmentation

**Stability**:
- Mean ± standard deviation across ≥3 random seeds (model initialization + augmentation randomness)
- Reported in all tables as μ ± σ

**Non-accuracy dimensions** (§1.8):

| Dimension | Test | Metric |
|-----------|------|--------|
| **Cross-load generalization** | Train on one load, test on another (all 12 load-pair combinations) | Accuracy retention ratio: Acc_cross-load / Acc_same-load |
| **Robustness curve** (from 方向三) | Test models trained with each augmentation on progressively contaminated test sets (clean → mild noise σ=0.02 → moderate noise σ=0.1) | Degradation slope: ΔAcc / Δnoise_level. A flatter slope indicates augmentation-induced robustness. Compare slopes across augmentation types and split protocols. |
| **Physical consistency** | Correlation between M5 energy retention ranking and Δ_recovery ranking across augmentation types | Spearman rank correlation ρ between R_energy and Δ_recovery. ρ > 0.7 supports H4; ρ ≈ 0 falsifies H4. |

## Expected Contribution

1. **First systematic characterization** of the interaction between evaluation protocol (random vs. recording-level split) and data augmentation in CWRU bearing fault diagnosis
2. **A quantitative metric** (gap-recovery ratio Δ_recovery) for measuring augmentation effectiveness under leakage-safe conditions — applicable to future CWRU studies
3. **Physical safety audit of augmentation**: quantifying which augmentation types preserve fault-frequency-band energy (BPFO/BPFI/BSF) and which distort it, establishing a direct test of ch45's physical constraint in the context of diagnostic performance
4. **Correlation between physical preservation and practical benefit**: testing whether augmentations that maintain fault-band energy provide stronger generalization recovery — if confirmed, this provides a physics-based criterion for selecting and designing augmentation strategies for bearing diagnosis
5. **Practical guidance**: evidence on whether researchers should prioritize fixing evaluation protocols, selecting physically safe augmentations, or both
6. **A reproducible experimental protocol** (2 splits × 7 augmentations × 2 models × 3 seeds × 3 contamination levels) that can serve as a diagnostic toolkit for evaluating any CWRU study's robustness claims

## Main Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Augmentation recovery effect is too small to be meaningful | Medium | Framed as "establishing the boundary of augmentation effectiveness" — a careful negative result with good controls is valid per §17.2 |
| Sensitivity to augmentation hyperparameters may dominate the signal | Medium | Systematic sensitivity sweep (3 strengths per augmentation type) rather than picking one arbitrary setting |
| Only 4 CWRU motor loads limits cross-load generalizability | High | Acknowledged as explicit limitation (§1.10); one additional CWRU-like dataset planned (Paderborn PU preferred) to mitigate |
| Results limited to 2 model families | Medium | Acknowledged as scope limitation; 1D CNN and 2D CNN cover the two dominant CWRU model paradigms |
| STFT parameters add a hidden degree of freedom | Low | Fixed STFT parameters (window=128, overlap=0.5, n_fft=128) — documented and not varied |
| Fault-frequency band energy retention is measured using theoretical fault frequencies, but CWRU's actual shaft speed may deviate from nominal values (1797/1772/1750/1730 RPM) | Medium | (1) Compute fault frequencies as a ±2% band around theoretical values to account for speed variation; (2) Cross-validate with envelope spectrum peak detection to confirm actual fault frequencies; (3) If discrepancy is large, report both theoretical-band and detected-peak energy retention |
| Robustness curve requires generating contaminated test sets — parameters (noise σ, magnitude) are arbitrary | Low | Fixed contamination parameters determined before seeing results; contamination applied only at test time, never during training (leakage prevention per §6.3) |

## What Would Falsify the Hypothesis

| Hypothesis | Falsification Condition |
|-----------|------------------------|
| **H1** (partial recovery) | Δ_recovery ≈ 0 for all augmentation conditions → augmentation has no measurable effect under either split protocol — the performance gap is entirely determined by split protocol alone |
| **H2** (leakage dominance) | Δ_recovery ≈ 1.0 for any augmentation condition → augmentation fully compensates, meaning the gap was NOT primarily leakage-driven — a surprising but valid result that would challenge the leakage orthodoxy |
| **H3** (diminishing returns) | Accuracy increases monotonically with augmentation strength without saturation or degradation → augmentation benefits are unbounded, contradicting the expectation that stronger perturbation introduces task-irrelevant distortion |
| **H4** (physical preservation → recovery correlation) | Spearman ρ ≈ 0 between R_energy (M5) and Δ_recovery → no correlation between physical safety and generalization benefit. Augmentation helps (or doesn't help) for reasons unrelated to fault-frequency-band energy preservation. This would suggest that ch45's physical constraint, while theoretically valid, does not manifest in measurable diagnostic performance differences under these experimental conditions |

The most scientifically valuable outcome is not confirmation of all four hypotheses, but a well-controlled experiment that clearly establishes **where the boundaries lie** — whether Δ_recovery saturates at 0.1, 0.3, or 0.5, and whether physical preservation (H4) meaningfully constrains the effectiveness ranking of augmentation strategies.
