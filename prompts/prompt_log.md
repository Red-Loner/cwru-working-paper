# Prompt Log

Per `Final_Experiment/PROMPT_PROTOCOL.md`, this log records all AI interactions during the research process.

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

## Tool Summary

| Tool/Model | Purpose | Sessions |
|------------|---------|----------|
| DeepSeek (via OpenCode CLI) | Code implementation, debugging, paper drafting, review, integrity checking | ~17 sessions |
| paramiko (Python SSH library) | Server deployment, remote execution, file transfer | ~8 sessions |

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

## Verification Methods

- All numerical claims cross-checked against `results/tables/*.json` files.
- All 9 references verified by DOI resolution.
- All code tested on NVIDIA A10 GPU (200 runs, 46 min).
- Figures visually inspected for correctness.
- Integrity checklist (7 failure modes) manually audited against code and paper.
