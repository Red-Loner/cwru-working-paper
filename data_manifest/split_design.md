# Split Design — CWRU Bearing Fault Diagnosis

## Overview

This document defines the train/validation/test split protocols for the factorial experiment evaluating augmentation × split interaction.

## Factor A: Split Protocol (2 levels)

### A1. Random Adjacent-Window Split (LEAKAGE BASELINE)

Purpose: Upper bound — measures what leakage produces, not true generalization.

Procedure:
1. Load all `.mat` files, extract DE 12 kHz signal.
2. Segment each signal into fixed-length windows (length = 1024 samples, overlap = 50%).
3. Pool all windows from all recordings into a single dataset.
4. Randomly split windows: train 60% / val 20% / test 20%, stratified by fault class.
5. Report: "Random split (leakage-prone)".

Risk: Adjacent windows from the same recording appear in train AND test. Model learns recording identity → inflated accuracy by 10–30 percentage points.

Justification: This is the de facto standard in much CWRU literature. Including it as a baseline (A1) quantifies the leakage premium and provides reference for the gap-recovery ratio.

### A2. Recording-Level Split (LEAKAGE-SAFE)

Purpose: True generalization — evaluates whether the model learns fault features, not recording identity.

Procedure:
1. Load all `.mat` files, identify unique recordings.
2. Group recordings by fault class and load condition.
3. Assign each recording entirely to ONE split: train (60% of recordings), val (20%), test (20%), stratified by fault class.
4. Segment windows within each split independently. Overlap = 0% (non-overlapping windows to maximize independence within a recording).
5. Report: "Recording-level split (leakage-safe)".

Rule: NO two windows from the same original `.mat` recording appear in different splits.

Validation:
- Check: every recording's windows belong to exactly one split.
- Report: number of recordings per split, number of windows per split, class distribution per split.

## Fixed Parameters (Not Varied)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Window length | 1024 samples (~85 ms @ 12 kHz) | Standard in CWRU literature; captures multiple shaft rotations |
| Overlap (recording-level) | 0% | Maximizes window independence when windows share a recording |
| Overlap (random split) | 50% | Typical practice in literature baselines |
| Normalization | Z-score, per-channel, using training-set statistics only | Prevents test-set information leakage |
| Random seeds | ≥ 3 (e.g., 42, 123, 456) | Stability assessment |
| Train/val/test ratio | 60/20/20 | Standard ML practice |

## Stratification Strategy

- Primary: fault class (normal, IR, OR, ball)
- Secondary (recording-level split only): load condition (0/1/2/3 HP) — ensure each split contains all loads to avoid confounding load effects with split effects
- Fault diameter (0.007"/0.014"/0.021") is NOT used for stratification but is tracked for per-diameter sub-analysis

## Verification Checklist

- [ ] All recordings identified and documented (count, class, load, duration)
- [ ] Recording-to-split assignment recorded and saved (for reproducibility)
- [ ] No recording appears in more than one split (recording-level)
- [ ] Class distribution balanced across splits (±5%)
- [ ] Normalization fitted on training set only
- [ ] Augmentation applied to training set only (post-split)
- [ ] Window count per class per split documented for class-imbalance disclosure
