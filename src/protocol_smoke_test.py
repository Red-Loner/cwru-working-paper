"""Small deterministic checks for the experiment's high-risk protocol logic."""

import numpy as np

import config
from augmentation import AUGMENTATION_CONDITIONS, apply_augmentation
from mechanism_validation.fault_band_energy import fault_band_energy
from preprocess import (
    fault_size_based_assignment,
    load_based_assignment,
    random_split,
)


def check_fixed_segmentation():
    assert config.OVERLAP_RANDOM == config.OVERLAP_RECORDING == 0.5


def check_stratified_random_split():
    windows = np.arange(400 * 8, dtype=np.float32).reshape(400, 8)
    labels = np.repeat(np.arange(4), 100)
    partitions = random_split(windows, labels, seed=42)
    counts = [
        np.bincount(partition_labels, minlength=4).tolist()
        for _, partition_labels in partitions
    ]
    assert counts == [[60] * 4, [20] * 4, [20] * 4], counts


def check_augmentations():
    rng = np.random.RandomState(42)
    waveforms = rng.randn(8, config.WINDOW_LENGTH).astype(np.float32)
    for augmentation in AUGMENTATION_CONDITIONS:
        np.random.seed(42)
        augmented = apply_augmentation(waveforms, augmentation)
        assert augmented.shape == waveforms.shape, augmentation
        assert augmented.dtype == np.float32, augmentation
        assert np.isfinite(augmented).all(), augmentation
    assert not np.array_equal(
        apply_augmentation(waveforms, "specaugment"), waveforms
    )
    assert not np.array_equal(
        apply_augmentation(waveforms, "freq_flip"), waveforms
    )


def check_physical_metric():
    time = np.arange(config.WINDOW_LENGTH) / config.SAMPLE_RATE
    signal = (
        np.sin(2 * np.pi * 162.0 * time)
        + 0.2 * np.sin(2 * np.pi * 324.0 * time)
    ).astype(np.float32)
    identical = fault_band_energy(signal, signal, "IR", 1797)
    assert abs(identical["raw_energy_ratio"] - 1.0) < 1e-6
    assert abs(identical["symmetric_fidelity"] - 1.0) < 1e-6
    scaled = fault_band_energy(signal, signal * 2, "IR", 1797)
    assert 0.0 <= scaled["symmetric_fidelity"] <= 1.0


def check_group_assignments():
    records = []
    for load in range(4):
        records.append({
            "recording_id": f"normal_load_{load}",
            "load": load,
            "diameter": "None",
        })
        for diameter in ("007", "014", "021"):
            for fault_type in ("IR", "OR", "B"):
                records.append({
                    "recording_id": f"{fault_type}_{diameter}_load_{load}",
                    "load": load,
                    "diameter": diameter,
                })

    expected_ids = {record["recording_id"] for record in records}
    checks = (
        (load_based_assignment(records, seed=42), [2, 1, 1]),
        (fault_size_based_assignment(records, seed=42), [1, 1, 1]),
    )
    for assignment, expected_group_counts in checks:
        assert [
            len(assignment["groups"][split])
            for split in ("train", "validation", "test")
        ] == expected_group_counts
        recording_sets = [
            set(assignment["recordings"][split])
            for split in ("train", "validation", "test")
        ]
        assert not (recording_sets[0] & recording_sets[1])
        assert not (recording_sets[0] & recording_sets[2])
        assert not (recording_sets[1] & recording_sets[2])
        assert set().union(*recording_sets) == expected_ids


def main():
    check_fixed_segmentation()
    check_stratified_random_split()
    check_augmentations()
    check_physical_metric()
    check_group_assignments()
    print("Protocol smoke tests PASS")


if __name__ == "__main__":
    main()
