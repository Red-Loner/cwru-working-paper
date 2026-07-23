# Paper-Requirement Traceability — 2026-07-23

This audit identifies which files in the course-material directory impose
requirements on the working paper and distinguishes binding specifications
from instructional background.

## Requirement sources

| Source file | Role | Paper requirements used |
|---|---|---|
| `CAPSTONE_GUIDE.md` | Course-level summary | Produce a reproducible working paper and evidence package; use bounded claims, leakage control, mechanism evidence, ARS review, AI disclosure, and final integrity checks. It explicitly defers operational detail to `Final_Experiment/manual.md`. |
| `Final_Experiment/manual.md` | Primary operational specification | Required 16-part paper structure; problem/bottleneck/hypothesis/method; fair baselines and a negative control; mechanism validation before classification; accuracy, Macro-F1, per-class precision/recall, confusion matrices, and seed variation; one non-accuracy dimension; ablation; limitations; reproducibility statement; AI-use disclosure. |
| `Final_Experiment/RESEARCH_STRUCTURE.md` | Minimum research-evidence standard | Problem, bottleneck, hypothesis, mechanism validation, baselines, downstream evaluation, stability, non-accuracy evidence, ablation, discussion, limitations, and reproducibility. |
| `Final_Experiment/CLAIM_EVIDENCE_RULES.md` | Claim-writing constraint | Every important claim must trace to an experiment, figure, table, verified citation, dataset card, limitation, or clearly marked hypothesis; avoid field-readiness, causal, and physical-reasoning overclaims. |
| `Final_Experiment/STUDENT_PRIVACY_AND_IP_RULES.md` | Integrity and disclosure constraint | State data provenance and access terms; verify citations; disclose AI tools, assisted work, and verification; do not claim field readiness, general solution, unfair superiority, causality from correlation, or unsupported physical interpretation. |
| `Final_Experiment/PROMPT_PROTOCOL.md` | AI-assisted writing constraint | Draft from supplied evidence, separate facts from suggestions, record verification and errors, and do not write around missing or failed experiments. |
| `Final_Experiment/ARS_INTEGRATION.md` | Review/revision process requirement | Draft from the claim-evidence table, run integrity and reviewer stages, revise point by point, and retain required ARS artifacts. |
| `Final_Experiment/DATASET_DISCOVERY_GUIDE.md` | Dataset provenance/scope guidance | Record source URL, citation, access terms, download date, signal/classes/conditions, preprocessing, split, leakage risks, and reason for inclusion or exclusion. It permits CWRU plus a discovery report for a short course. |
| `research_question_brief.md` | Project-specific initial plan | Defines the split × augmentation question, physical-fidelity hypothesis, planned factors, metrics, contamination/cross-load extensions, mechanism tests, expected contributions, risks, and falsification conditions. It is a planning document; deviations must be disclosed rather than silently presented as completed tests. |
| `Final_Experiment/README.md` | Routing index | Identifies the preceding files as the intended specification set. |

Templates under `Final_Experiment/templates/` prescribe the format of supporting
artifacts, not additional manuscript sections. Chapter PDFs, worksheets,
answers, figures, and chapter experiment READMEs provide technical instruction
and examples; they do not override the final-experiment specification or add
separate submission requirements.

## Manuscript improvements made from this audit

| Requirement | Previous state | Revision |
|---|---|---|
| Explicit research question and bottleneck | Implied across the Introduction | Added a direct bottleneck statement and research question. |
| Bounded contribution statement | Contributions were implicit | Added three benchmark-scoped contributions and an explicit no-industrial-generalization boundary. |
| Dataset provenance and access terms | Present only in the dataset card/material passport | Added official source citation, download date, access-term uncertainty, non-redistribution policy, and exact manifest path to the Dataset section. |
| Macro-F1 | Named in Experimental Setup but absent from Results tables | Added an auto-generated all-configuration Macro-F1 table with mean ± standard deviation. |
| Per-class precision and recall | Stored in JSON but the paper displayed only F1 | Added an auto-generated precision/recall/F1 table for both models under baseline, best augmentation, and destructive control. |
| Plan-versus-execution transparency | Initial cross-load/contamination plan was not discussed | Added a Method subsection that separates primary inference from descriptive grouping and one-seed contamination evidence and records the uncompleted all-pairs cross-load plan. |
| Robustness boundary | Auxiliary contamination results were absent from the manuscript | Added a clearly labeled exploratory paragraph that makes no robustness-superiority claim. |
| Required limitations | Several were distributed across the paper rather than explicit in Limitations | Added artificial-fault/field-readiness, condition dependence, random-split comparator, reduced robustness scope, and AI-assisted-workflow limitations. |
| Reproduction instructions | Repository command was present but data/device setup was implicit | Added `CWRU_DATA_ROOT`, optional CUDA device configuration, run command, and verifier scope. |
| AI-use verification | Tools and activities were disclosed | Added artifact, dataset, and citation verification procedures plus author responsibility. |
| Claim traceability | Existing table covered 13 claims | Added Macro-F1, per-class error, dataset provenance, and plan-deviation claims. |

All numerical additions are generated from `results/tables/main_results.json`;
no result was manually invented or transcribed from memory.
