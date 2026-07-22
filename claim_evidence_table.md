# Claim–Evidence Table

| # | Claim | Evidence Source | Strength | Counter-claim / Limitation |
|---|-------|----------------|----------|----------------------------|
| C1 | Random window splitting inflates accuracy due to adjacent-window leakage | M1: auto-correlation across offset windows (Fig. `m1_adjacent_autocorrelation.png`); | Strong | Magnitude of inflation varies by recording and fault type |
| C2 | Recording-level split eliminates leakage but reduces test accuracy | Table 1: Protocol R (random) vs Protocol S (recording): 1D baseline 0.9997→0.9321, 2D baseline 0.9995→0.9007 | Strong | Effect size confirmed: Cohen's d=1.17 (1D), d=1.80 (2D) |
| C3 | Augmentation partially recovers the leakage gap, with limited efficacy | Table 2, Fig. `gap_recovery_1d.png`/`gap_recovery_2d.png`: max Δ_recovery=0.46 (1D, noise_005), 0.15 (2D, shift_5) | Strong | H1 supported but effect small for 2D |
| C4 | Leakage dominates over augmentation benefits | H2: max Δ_recovery << 1.0 for both models. No augmentation restores recording-level performance to random-level. | Strong | Confirmed by falsification analysis |
| C5 | Most augmentations degrade 2D recording performance | Table 2: 7/9 augmentations show negative Δ_recovery for 2D | Moderate | 1D shows mostly positive Δ_recovery; model-dependent |
| C6 | Three random seeds insufficient to characterize recording-level variance | 3-seed vs 5-seed comparison: 2D recording baseline moved from 0.9653 to 0.9007 (6.5pp drop). individual seeds for 2D range from 0.784 to 0.999. | Strong | 5 seeds still limited for 13% std; methodological recommendation |
| C7 | Fault-size grouping dominates as the hardest domain shift | Ablation Table: by Fault Size acc=0.63/0.47 vs by Load=0.98/0.98 | Strong | Grouping variable ranking: fault_size >> recording > load |
| C8 | Physical frequency energy preservation does not correlate with recovery | H4: Spearman ρ=-0.37, p=0.33. Fig. `h4_correlation.png` | Strong | Negative ρ implies weak inverse trend — augmentation that preserves MORE energy may hurt performance slightly |
| C9 | All 9 augmentations pass physical safety audit | M5: energy retention ratio ≥1.0 for all augmentations (no order-of-magnitude drops in fault-band energy) | Moderate | Safety ≠ efficacy; supports H4 null result |
| C10 | 50 training epochs sufficient for convergence | Convergence analysis: mean convergence epoch=15.2, median=9, max=48; 6.7% of runs converge in last 10% epochs | Strong | Minor early-stopping optimization possible |
| C11 | The recording-level gap is a structural property, not a model artifact | Consistent across 1D-CNN and 2D-CNN; effect size >1.0 for both | Strong | |
| C12 | Cross-load generalization is surprisingly easy | Load-based split yields acc=0.977–0.978, nearly matching random split | Strong | Frequency-domain features mostly load-invariant for CWRU |
