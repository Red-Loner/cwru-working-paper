# Split Design — CWRU Bearing Fault Diagnosis

## Overview

This document defines the train/validation/test split protocols for the factorial experiment evaluating augmentation × split interaction.

## Factor A: Split Protocol (2 levels)

### A1. Random Adjacent-Window Split (LEAKAGE BASELINE)

Purpose: Leakage-prone reference estimate under window-level randomization.

Procedure:
1. Load all `.mat` files, extract DE 12 kHz signal.
2. Segment each signal into fixed-length windows (length = 1024 samples, overlap = 50%).
3. Pool all windows from all recordings into a single dataset.
4. Randomly split windows: train 60% / val 20% / test 20%, stratified by fault class.
5. Report: "Random split (leakage-prone)".

Risk: Adjacent windows from the same recording can appear in train and test. The model can exploit shared samples and recording-specific signatures; the observed inflation is measured rather than prespecified.

Justification: Including this commonly encountered baseline quantifies the protocol difference and provides the denominator for the gap-recovery ratio.

### A2. Recording-Level Split (LEAKAGE-SAFE)

Purpose: Recording-disjoint estimate that prevents source-recording identity from crossing partitions.

Procedure:
1. Load all `.mat` files, identify unique recordings.
2. Group recordings by fault class.
3. Assign each recording entirely to ONE split: train (60% of recordings), val (20%), test (20%), stratified by fault class.
4. Segment each recording with the same 50% overlap used by Protocol R, then assign all of that recording's windows to its preselected partition. This keeps preprocessing fixed while preventing cross-partition recording overlap.
5. Report: "Recording-level split (leakage-safe)".

Rule: NO two windows from the same original `.mat` recording appear in different splits.

Validation:
- Check: every recording's windows belong to exactly one split.
- Report: number of recordings per split, number of windows per split, class distribution per split.

## Fixed Parameters (Not Varied)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Window length | 1024 samples (~85 ms @ 12 kHz) | Standard in CWRU literature; captures multiple shaft rotations |
| Overlap (recording-level) | 50% | Fixed to match the random protocol; recordings remain disjoint across partitions |
| Overlap (random split) | 50% | Creates the intended leakage-prone baseline when adjacent windows are split independently |
| Normalization | Z-score, per-channel, using training-set statistics only | Prevents test-set information leakage |
| Random seeds | 5 (42, 123, 456, 789, 1024) | Stability assessment |
| Train/val/test ratio | 60/20/20 | Standard ML practice |

## Stratification Strategy

- Primary: fault class (normal, IR, OR, ball)
- Load condition and fault diameter are tracked but are not additional stratification keys. Their distribution can vary across seeds and is examined separately through grouping ablations.

## Verification Checklist

- [x] All 40 selected recordings identified with class, load, size, SHA-256, and signal length in `dataset_file_manifest.json`
- [x] Recording-to-split assignment saved for all five seeds in `recording_split_assignments.json`
- [x] No recording appears in more than one split; checked programmatically in `split_verification.json`
- [x] Every selected recording appears in exactly one split for every seed
- [x] Normalization is fitted on training windows only
- [x] Augmentation is applied to the training partition only, after splitting
- [x] Recording, window, and per-class counts are documented per seed and split
