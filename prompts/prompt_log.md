# Prompt Log

Per `Final_Experiment/PROMPT_PROTOCOL.md`, this log records all AI interactions during the research process.

Entries from 2026-07-21/22 describe the repository state at that time. The 2026-07-23 independent audit found that several earlier “verified” statements did not hold in the local release; the corrective entries and error list below supersede them.

| Date | Tool | Objective | Input Files/Sources | Prompt Summary | Output Used | Verification |
|------|------|-----------|---------------------|----------------|-------------|-------------|
| 2026-07-21 | DeepSeek (OpenCode) | Set up server environment | `setup_server.sh` | Install PyTorch CUDA 12.1, verify GPU, configure conda environment | Server setup completed; GPU confirmed via nvidia-smi | Manual verification on hardware |
| 2026-07-21 | DeepSeek (OpenCode) | Implement training pipeline | `manual.md` §7-10 | Implement train.py with factorial design (2 models × 2 splits × 10 augmentations × 5 seeds), recording-level split, batch_size=128, 50 epochs | `src/train.py`, `src/preprocess.py`, `src/config.py` | Verified on server; 200 runs completed |
| 2026-07-21 | DeepSeek (OpenCode) | Fix STFT device mismatch bug | `src/train.py` | hann_window placed on wrong device causing crash on GPU | Bug fixed: hann_window explicitly moved to model device | Re-ran; no crash |
| 2026-07-21 | DeepSeek (OpenCode) | Implement mechanism validation | `manual.md` §9 | M1 (autocorrelation), M2 (feature diversity), M5 (fault-band energy) | `src/mechanism_validation/*.py` | Results verified in `results/figures/` |
| 2026-07-21 | DeepSeek (OpenCode) | Implement analysis and figures | `manual.md` §10-13 | Gap-recovery analysis, Cohen's d, convergence analysis, per-class analysis, 14 publication figures | `src/analyze_results.py`, `src/plot_figures.py` | All figures verified; numbers cross-checked against JSON |
| 2026-07-21 | DeepSeek (OpenCode) | Implement contamination robustness test | `manual.md` §11.2 | Test all trained models on progressively noisier test sets (σ=0.00, 0.02, 0.10) | `src/contamination_test.py` | Results verified in `contamination_robustness.json` |
| 2026-07-21 | DeepSeek (OpenCode) | Implement grouping variable ablation | `innovation_hints_cn.md` §3 (Direction 1) | Compare 3 grouping strategies: load, recording_id, fault_size | `src/grouping_ablation.py` | Results verified in `grouping_ablation.json` |
| 2026-07-21 | DeepSeek (OpenCode) | Draft paper from evidence | `manual.md` §14, `claim_evidence_table.md` | Generate paper.tex with all 16 sections, LaTeX format, English body with Chinese margin notes | `paper.tex` (15 sections, 450 lines) | All numbers verified against results JSON files |
| 2026-07-21 | DeepSeek (OpenCode) | Verify all 9 references | `references.bib` | DOI verification via web fetch for all 9 citations | All 9 DOIs confirmed valid | Verified via actual DOI resolution |
| 2026-07-21 | DeepSeek (OpenCode) | Simulate peer review | `manual.md` §15 | Critical reviewer audit: 10 issues identified (minor revision verdict) | `ars/reviewer_report.md` | 7 issues fixed in revision |
| 2026-07-21 | DeepSeek (OpenCode) | Integrity check against Lu et al. (2026) 7 failure modes | `paper.tex`, `src/`, `results/` | Audit all code and paper for M1-M7 failures | `ars/integrity_report.md` | All 7 modes CLEARED |
| 2026-07-22 | DeepSeek (OpenCode) | Fix paper errors found in self-check | `paper.tex` | batch_size 128→64→128 (revert), 1D-CNN 3→4 conv layers, combined description, manual_seed, overlap_ratio, R_energy limitation | Updated `paper.tex`, `train.py`, `preprocess.py` | All changes verified against code and JSON |
| 2026-07-22 | DeepSeek (OpenCode) | Re-run full pipeline with fixed code | Server: `src/run_all.py` | Upload fixed code, clear old results, run 200 experiments + all 9 pipeline steps | All 8 tables + 14 figures regenerated | 9/9 steps [OK], 46 min runtime |
| 2026-07-22 | DeepSeek (OpenCode) | Update all numbers in paper | `main_results.json`, `gap_recovery.json`, `per_class_analysis.json` | Update abstract, all 3 tables, inline numbers, conclusion in paper.tex and paper_draft.md | Updated paper.tex, paper_draft.md, paper_tables.tex | All numbers cross-checked against new JSON |
| 2026-07-22 | DeepSeek (OpenCode) | §17 compliance audit and fixes | `manual.md` §17, §2.3 | Create paper/ directory, refs.bib, literature_matrix, methodology_blueprint, material_passport, final_integrity_report | All missing files created | Structure matches manual §17 checklist |
| 2026-07-22 | DeepSeek (OpenCode) | Clean up artifacts and duplicates | `.gitignore` | Remove LaTeX build artifacts, __pycache__, duplicate files (paper.pdf, paper.tex, references.bib at root) | Git history cleaned; .gitignore updated | Root directory has only 6 files |
| 2026-07-22 | DeepSeek (OpenCode) | Distribute figures to proper sections | `paper/main.tex` | Move 14 figures from clustered `\section*{Figures}` block to their respective sections | Figures now appear near referencing text | Paths fixed to `../results/figures/` |
| 2026-07-23 | OpenAI Codex | Re-audit course compliance after local file changes | Course `CAPSTONE_GUIDE.md`, `Final_Experiment/manual.md`, local repository | Compare every §17 requirement with files and inspect whether existing artifacts actually support manuscript claims; identify leakage, validity, and reproducibility risks | `ars/course_requirement_audit_20260723.md`; repair plan | Read-only inventory, JSON/log counts, code-to-document comparison; no missing result was fabricated |
| 2026-07-23 | OpenAI Codex | Audit all available CWRU data copies | D-drive, E-drive, A10 raw tree, project selection code | Validate the exact 40 selected recordings, readability, sizes, and SHA-256 identity; refuse silent skips | `data_manifest/local_data_copy_audit.md`; fail-fast loader and dataset manifest | E-drive corruption isolated; all 40 selected D-drive and A10 hashes match |
| 2026-07-23 | OpenAI Codex | Repair experiment protocol and evidence pipeline | `src/`, method/protocol cards, prior results | Hold preprocessing fixed across split protocols; make random split stratified; replace fake reshape-based frequency transforms; add confusion matrices and exact split manifests; correct M1/M2/M5/H3 analyses | Updated source, protocol smoke tests, release verifier, environment lock | Synthetic protocol tests and 40-file audit passed before the clean A10 run |
| 2026-07-23 | OpenAI Codex | Execute an isolated audited rerun on NVIDIA A10 | Repaired source, hash-matched server data, fixed seeds | Run 200 main trainings, 30 grouping stress tests, mechanism/robustness analyses, and figures in a new directory; do not mutate the old server project | Complete audited result package: 200 logs, 40 main rows, 30 grouping rows, 15 figures | Data/result verification passed; one convergence-script variable error was repaired and that stage rerun successfully |
| 2026-07-23 | OpenAI Codex | Reconcile final evidence and manuscript | Final JSON, figures, generated tables, paper and audit documents | Add H3 negative-control sensitivity, mark pooled H3 non-inferential, replace stale numerical duplicates, add automatic LaTeX table generation | Final paper source, Chinese summary, claim table, reviewer response, integrity materials | Values traced to `release_summary.json`; final acceptance requires PDF render and release verifier |

## Tool Summary

| Tool/Model | Purpose | Sessions |
|------------|---------|----------|
| DeepSeek (via OpenCode CLI) | Code implementation, debugging, paper drafting, review, integrity checking | ~17 sessions |
| paramiko (Python SSH library) | Server deployment, remote execution, file transfer | ~8 sessions |
| OpenAI Codex | Independent compliance audit, data-copy validation, protocol repair, rerun orchestration, evidence/document reconciliation | 1 extended session |
| OpenSSH client | Key-based A10 connection, isolated execution, progress inspection, artifact transfer | 1 audited rerun |

## Errors Found and Corrected

1. **STFT device mismatch**: `torch.hann_window` on CPU while model on GPU → fixed by moving window to correct device.
2. **Resume logic seed_id mapping**: Incorrectly mapped cached results → fixed by matching all config fields.
3. **fault_size_split Normal class isolation**: Normal recordings all assigned to train → fixed by stratified distribution.
4. **overlap_ratio not used**: Recording-level windows always created with 50% overlap → fixed with parameterized `build_datasets`.
5. **torch.manual_seed missing**: Model weights not seeded → fixed by adding manual_seed.
6. **batch_size claim mismatch**: Paper said 128, local config had 64 → verified server used 128, corrected local and paper.
7. **1D-CNN layer count wrong**: Paper said 3 conv layers, code has 4 → corrected in paper.
8. **Combined augmentation description**: Paper said noise+shift+SpecAugment, code only noise+shift → corrected in paper.
9. **R_energy metric confound**: All augmentations increase broadband energy, inflating metric → added to Limitations.
10. **Figure paths broken**: `results/figures/` → `../results/figures/` when paper moved to `paper/` directory.
11. **Release evidence incomplete**: 195/200 epoch logs, empty grouping JSON, and no confusion matrices → full clean regeneration and exact verifier added.
12. **Split-factor confound**: random used 50% overlap while recording used 0% → fixed both at 50% so only the split unit changes.
13. **Random split not stratified**: prose claimed class stratification but code globally shuffled → deterministic per-class allocation and verification added.
14. **Fake frequency transforms**: raw samples were reshaped to 32×32 and treated as a spectrum → replaced with real-FFT spectral/time masking and FFT-magnitude reversal.
15. **M1 statistic misaligned**: zero-lag Pearson r did not measure shifted shared samples → exact overlap fraction and aligned identity audit added.
16. **M5 amplification error**: any energy increase was labeled physically safe → symmetric envelope-spectrum fidelity now penalizes attenuation and amplification.
17. **H3 model overwrite**: augmentation-only keys retained one model's recovery values → 1D, 2D, and pooled correlations reported separately.
18. **M2 untrained feature claim**: an untrained CNN layer was used as “feature space” → deterministic standardized STFT input-representation analysis used.
19. **Data-copy corruption**: 13 E-drive `.mat` files were unreadable, including 2 selected recordings → E copy excluded; D/server selected hashes verified.
20. **Citation verification wording**: dataset URL was counted as a DOI and literature gaps were overclaimed → eight DOI records and one source URL are now distinguished; claims narrowed.
21. **Grouping-ablation overclaim**: load grouping trained on 2/4 load groups, while fault-size grouping trained on 1/3 diameter groups → exact assignments are retained and the accuracy ordering is interpreted only as a descriptive stress test.
22. **Robustness comparison was unpaired**: augmentation conditions used different model seeds and test-noise draws → fixed to a common initialization seed and deterministic noise realization at each contamination level.
23. **Offline augmentation scope was implicit**: transforms replace the training windows once per run rather than being resampled per epoch or concatenated → disclosed in the manuscript, method card, protocol, and README.
24. **Convergence stage crashed after computing statistics**: output formatting referenced undefined `conv_epochs` → fixed to `best_epochs` and regenerated the 200-run JSON and figure.
25. **H3 negative-control leverage and pooled dependence**: the destructive control had extreme recovery values and the pooled analysis duplicated fidelity values → added per-model sensitivity without the control and labeled pooling non-inferential.
26. **Compiled PDF became stale after final source and figure repairs**: regenerated the PDF after the last plot/table/source change, scanned a clean LaTeX log, rendered the then-current document with Poppler, and visually checked every page.
27. **Course requirements were present but not fully visible in the manuscript**: re-audited `CAPSTONE_GUIDE.md`, the complete `Final_Experiment` specification set, and the project research brief → added explicit bottleneck/RQ/contributions, dataset provenance, generated Macro-F1 and per-class precision/recall/F1 tables, plan-deviation disclosure, robustness boundaries, expanded limitations, and stronger reproduction/AI-verification statements.
28. **Requirement-driven revision changed the final pagination**: recompiled the revised manuscript to 16 pages, scanned the final LaTeX log, rendered every page with Poppler, and visually inspected the complete document, including both new generated metric tables.

## Verification Methods

- All numerical claims cross-checked against `results/tables/*.json` files.
- All 9 references verified by DOI resolution.
- The final A10 evidence passed exact 200-run/30-run, log, matrix, manifest, and finite-number checks.
- All 40 selected server recordings were hash-matched to the readable D-drive copy before the audited rerun.
- Final figures and the compiled 16-page PDF were rendered and visually inspected after the requirement-driven revision.
- Integrity checklist (7 failure modes) manually audited against code and paper.
