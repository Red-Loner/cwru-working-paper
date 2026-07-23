# Course-Requirement Audit — 2026-07-23

This audit compares the capstone instructions in `Final_Experiment/manual.md`
and `CAPSTONE_GUIDE.md` with the working-paper repository and the available
CWRU data copies. It records both missing artifacts and cases where a file
existed but did not support its claim.

## §17 release structure

| Requirement | Canonical artifact | Release check |
|---|---|---|
| Paper PDF | `paper/main.pdf` | PASS: final 16-page PDF compiled after all audited tables and figures and visually inspected page by page |
| Paper source | `paper/main.tex` | Present; reconciled to generated final tables and release summary |
| References | `paper/refs.bib` | Present; 8 DOI records and CWRU URL distinguished |
| Source code | `src/` | Present; protocol smoke and release-verification scripts added |
| Environment | `environment/requirements.txt`, `requirements-a10-lock.txt` | Present |
| Data manifest | `data_manifest/` | Exact file hashes and both split-protocol checks generated |
| Tables | `results/tables/` | Regenerated; 40 main rows, 30 grouping rows, H3 sensitivity, and canonical summary verified |
| Figures | `results/figures/` | Fifteen regenerated figures present, including confusion matrices |
| Logs | `results/logs/epoch_logs/` | Exactly 200 complete 50-epoch CSV files verified |
| Prompt log | `prompts/prompt_log.md` | Present; audit, repair, rerun, and reconciliation recorded |
| Material passport | `ars/material_passport.md` | Present and corrected |
| Integrity report | `ars/integrity_report.md` | Rewritten against audited evidence |
| Reviewer report | `ars/reviewer_reports.md` | Final evidence-aware review present |
| Revision response | `ars/revision_response.md` | Twenty-one audit/reviewer concerns mapped to repairs |
| Final integrity report | `ars/final_integrity_report.md` | PASS: regenerated after final PDF compilation and visual QA |
| Claim-evidence table | `claim_evidence_table.md` | Rewritten with required six-column schema and 17 traceable claims |
| README | `README.md` | Required sections and audited runtime/scope present |

## Material omissions and mismatches found

| Finding | Why it mattered | Repair |
|---|---|---|
| `grouping_ablation.json` was zero bytes locally | Paper cited evidence that did not exist in the release | Recovered old evidence for diagnosis, then scheduled a clean 30-run regeneration |
| Only 195/200 epoch logs were local | Reproducibility statement was false | Clean run plus exact expected-filename and 50-epoch verification |
| No confusion-matrix artifact | A required secondary metric was not reproducible | Per-seed and aggregated matrices added for all 40 configurations, plus paper figure |
| No recording-to-split manifest | File-disjointness could not be independently checked | Five-seed assignment and verification JSON files added |
| Random split used 50% overlap but recording split used 0% | Split effect was confounded by segmentation and sample count | Both protocols fixed at 1,024 samples and 50% overlap |
| Random split was described as stratified but code globally shuffled | Paper and implementation disagreed | Deterministic class-stratified allocation plus generated verification |
| `SpecAugment` and `freq_flip` reshaped raw samples into a fake 32×32 image | Named transforms were not operating on a real frequency axis | Replaced by real-FFT spectral/time masking and FFT-magnitude reversal; description narrowed |
| M1 used zero-lag Pearson correlation to infer overlap leakage | Shared samples are shifted, so that statistic missed the guaranteed duplication | Added exact shared fraction and aligned-overlap correlation/MAE |
| M5 treated any energy increase as “safe” | Broadband noise amplification was mislabeled preservation | Replaced with symmetric envelope-spectrum energy fidelity in [0,1] |
| H3 mapping silently retained only one model | Correlation evidence was overwritten by augmentation key | Separate 1D, 2D, and pooled correlations added |
| M2 used an untrained CNN hidden layer | “Feature diversity” lacked a learned or deterministic basis | Replaced with paired standardized STFT input-representation distances |
| Grouping accuracies were presented as a causal importance ranking | Load grouping trains on 2 of 4 groups, whereas fault-size grouping trains on only 1 of 3; training coverage is not matched | Exact per-seed assignments added and interpretation restricted to descriptive stress tests |
| Robustness curves used different model seeds and test-noise draws across augmentation conditions | Curves were not paired and could attribute random variation to augmentation | Fixed common initialization and deterministic per-level test noise |
| Offline augmentation behavior was not explicit | One-shot replacement can behave differently from online resampling or dataset expansion | Exact behavior disclosed in paper, protocol, method card, and README |
| Convergence analysis referenced undefined `conv_epochs` | The pipeline recorded a failed stage and omitted its JSON/figure | Replaced with `best_epochs`, reran successfully over all 200 logs, and retained the repair log |
| “Convergence epoch” was the first best validation epoch | A descriptive index was overclaimed as formal convergence | Field and prose renamed to best-validation epoch diagnostic |
| Method/model details conflicted across files | Batch size, 2D channels/FC size, STFT handling, and counts were stale | Method card, blueprint, protocol, README, and manuscript reconciled to code |
| External-dataset requirement was undocumented | Minimal CWRU-only scope had no search trail | Candidate sources/access terms and explicit deferral recorded |
| No exact software lock | Lower-bound requirements do not recreate the A10 run | Exact A10 package lock added |
| E-drive selected data contained corruption | Two required selected recordings could be silently lost in weak loaders | Fail-fast loader added; D-drive and server selected hashes verified identical |

## Data-copy decision

Do not run this project from the E-drive data tree. Two selected files
(`Normal_3HP_1730rpm.mat` and `IR007_1HP_1772rpm.mat`) are unreadable there.
All 40 selected SHA-256 hashes match between the readable D-drive copy and the
A10 copy used for the audited run. Details are in
`data_manifest/local_data_copy_audit.md`.

## Scope boundary and remaining owner actions

- The course guide prefers a second CWRU-like dataset, while `manual.md` §19
  permits a CWRU-only minimum project. This release uses that minimum option
  and makes no cross-dataset claim. Candidate discovery is documented, but no
  external dataset was downloaded or tested.
- Git commit, remote push, tag, and public release are repository-owner
  actions. This repair does not push or publish without explicit authorization.
- The E-drive damaged data copy is not overwritten or deleted.

The A10 pipeline has finished, the generated evidence passes
`python src/verify_release.py`, and the revised 16-page PDF has been compiled
and visually checked. The author list is complete; affiliation and email are
intentionally omitted because none were supplied. Only owner-authorized Git
publication actions remain outside this audit.
