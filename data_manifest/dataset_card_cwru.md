# Dataset Card — CWRU Bearing Dataset

- **Dataset name**: Case Western Reserve University (CWRU) Bearing Data Center Dataset
- **Source URL**: https://engineering.case.edu/bearingdatacenter
- **Citation**: K.A. Loparo, Case Western Reserve University, "Bearing Data Center," https://engineering.case.edu/bearingdatacenter
- **License or terms**: Publicly available for research; attribution requested. No formal license specified — academic use is standard.
- **Download date**: [TBD — fill in actual download date]
- **Signal type**: Vibration acceleration (accelerometer)
- **Sampling rate**: 12 kHz (Drive End, DE) / 48 kHz (Fan End, FE, available but less commonly used)
- **Machine or system**: 2 HP Reliance Electric motor driving a bearing test rig. Test bearings: SKF 6205-2RS JEM (Drive End), SKF 6203-2RS JEM (Fan End).
- **Fault classes**:
  - Normal (unfaulted)
  - Inner race fault (IR, IR007/IR014/IR021 for 0.007"/0.014"/0.021" diameter)
  - Outer race fault (OR, OR007@6/OR014@6/OR021@6 for 0.007"/0.014"/0.021" diameter at 6 o'clock position)
  - Ball fault (B, B007/B014/B021 for 0.007"/0.014"/0.021" diameter)
- **Operating conditions**: Motor loads: 0 HP / 1 HP / 2 HP / 3 HP, corresponding to approximate shaft speeds 1797/1772/1750/1730 RPM
- **File organization**: `.mat` files, one per load × fault condition. Each file contains DE and FE time series. Filename encodes fault location, diameter, and load.
- **Preprocessing**: Raw accelerometer time series. Requires: loading from `.mat`, selecting DE channel, normalization (e.g., z-score per file or per channel), segmentation into fixed-length windows.
- **Segmentation**: Fixed-length windows (e.g., 1024 or 2048 samples from 12 kHz signal → ~85 ms or ~170 ms per window). Overlap TBD (e.g., 50% for random split, 0% for recording-level — to be decided during leakage audit).
- **Train/validation/test split**: Two variants per experimental design:
  1. **Random adjacent-window split** (leakage baseline): All windows from all recordings pooled, randomly split 60/20/20.
  2. **Recording-level split** (leakage-safe): Each original `.mat` recording assigned entirely to one split. Train/val/test = 60/20/20 by number of recordings, stratified by fault class.
- **Leakage risks**:
  - **Adjacent-window leakage (HIGH)**: Windows cut from the same continuous recording with overlap share near-identical waveforms. Model learns recording identity, not fault signature. Estimated inflation: 10–30% accuracy.
  - **Normalization leakage (MEDIUM)**: Normalizing before splitting leaks test-set statistics into training.
  - **Augmentation leakage (LOW-MEDIUM)**: Augmenting before splitting, or augmenting with label-dependent transformations.
- **Mitigation**:
  - Recording-level split as the primary (safe) protocol.
  - Normalize per-training-set statistics only.
  - Apply augmentation only to training split after splitting.
- **Reason for inclusion**: Primary dataset. CWRU is the most widely used benchmark in bearing fault diagnosis literature, enabling comparison with published baselines. The multiple loads and fault diameters support the factorial design (split × augmentation × load × contamination).
- **Known limitations**:
  - Artificial seeded faults (EDM machining) differ from naturally evolving fatigue faults.
  - Only 4 motor loads (0–3 HP) — limited operating condition diversity.
  - Drive End bearing (SKF 6205) only; Fan End data available but less commonly used.
  - Single sensor location per bearing (one accelerometer per end).
  - Single test rig; results may not transfer to other machine geometries.
  - Shaft speed may deviate from nominal values (1797/1772/1750/1730 RPM).
  - High CWRU accuracy does NOT prove industrial field readiness.
