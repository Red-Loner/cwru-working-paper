# Reviewer Report — paper_draft.md

## Overall Assessment

This paper presents a well-designed factorial experiment investigating whether data augmentation compensates for the performance gap between leakage-prone (random window) and leakage-safe (recording-level) train/test splitting on CWRU. The 200-run experiment is thorough, the mechanism validation precedes classification, and the hypothesis testing (H1/H2/H4) follows a clean falsification logic. The grouping-variable ablation (load/recording/fault_size) is a particular strength. The paper is **accept with minor revisions**.


## Major Issues

### Issue 1: H3 is missing — inconsistent hypothesis numbering
**Location**: §1 Introduction (lines 18-20) and throughout.
The paper states H1, H2, and H4. The gap (no H3) suggests either a hypothesis was deleted or the numbering is arbitrary. Either add H3 or renumber to H1/H2/H3. Renumbering to H1/H2/H3 avoids readers wondering "what happened to H3."

### Issue 2: Table 1, 2, 3 are referenced but not included in the manuscript
**Location**: §7.1, §7.3, §7.4 and the closing line "Table 1, Table 2, and Table 3 are available as LaTeX in `paper_tables.tex`."
A paper submitted for review must have tables in-line, not as separate files. Insert the actual LaTeX tables (already in paper_tables.tex) into the manuscript at the cited locations.


## Minor Issues

1. **§1**: State the exact value of n (11,832 windows) in the Introduction to establish scale.
2. **§3.1**: Specify that 40 recordings include 4 Normal, 12 IR, 12 OR, 12 Ball. Currently only says "four fault classes."
3. **§5.1**: The M1 results ("Normal r=-0.20, IR r=0.02") describe very low correlations, which is fine but needs interpretation: low autocorrelation suggests the 50% overlap does NOT actually create strong leakable signal similarity for these recordings. Consider whether this undercuts the leakage claim, or whether the effect is data-dependent (some recordings may have high autocorrelation, others not).
4. **§5.2**: "noise σ=0.10 retains 69× the original fault-band energy" — this is because adding noise increases total energy, including in the fault band. Clarify that this does not mean the fault signal IS 69× stronger, just that broadband noise contributes energy everywhere.  
5. **§7.1**: The claim "7 of 9 augmentations degrade 2D recording-split performance" should cite the specific rows from Table 2 (which augmentations, what accuracy).
6. **§7.4 (ablation)**: The fault_size split uses 3 diameter groups; Note that this means the training set sees only 2 of 3 fault severities. The 0.47–0.63 accuracy suggests complete failure on unseen severities. This is the strongest finding and should be stated more prominently.
7. **§8**: The negative Spearman ρ (-0.37) is counterintuitive. Add one sentence speculating why: "Broad-spectrum noise augmentations may act as a regularizer that improves generalization even as they degrade physical fidelity."
8. **§14 (AI disclosure)**: "DeepSeek-V4" should be "DeepSeek" (model name) — verify exact brand name for disclosure accuracy.
9. **All figures**: The manuscript references figures by filename (e.g., `h4_correlation.png`, `gap_recovery_1d.png`). Replace with "Figure 1", "Figure 2", etc. numbered sequentially, with proper captions.
10. **Per-class metrics**: §7.2 mentions per-class F1 for 2D recording only. Add 1D per-class results or explain why omitted.


## Required Changes (must address before acceptance)

1. **Fix H3 gap** — renumber H4→H3 or add missing hypothesis.
2. **Insert Tables 1–3 inline** — not as external file references.
3. **Number figures sequentially** with captions, replace filename references.
4. **Verify Model Name accuracy in §14** — "DeepSeek-V4" vs correct product name.


## Verdict

**Minor Revision**. The experimental design and results are solid. The main issues are presentational (missing inline tables, figure numbering, hypothesis labeling). The scientific content is sound and the claim-evidence alignment is strong.

---

## Integrity Check: 7 Failure Modes (§7.10)

| Mode | Check | Status |
|------|-------|:------:|
| M1: Bug in self-review | Training logs + results.json verified; STFT device bug found and fixed during development | CLEARED |
| M2: Hallucinated citations | All 9 references verified via DOI (see §References validation above) | CLEARED |
| M3: Hallucinated results | All numbers traceable to results/tables/main_results.json | CLEARED |
| M4: Reliance on shortcuts | Recording-level split confirmed in code; no random-split-only claims | CLEARED |
| M5: Bug packaged as finding | Initial 3-seed→5-seed re-evaluation transparently reported; no misattribution | CLEARED |
| M6: Methodological forgery | Split implementation verified in preprocess.py; all 3 group splits tested | CLEARED |
| M7: Premature framing lock | Two alternative grouping variables tested (load, fault_size) beyond main RQ | CLEARED |
