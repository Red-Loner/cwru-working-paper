import csv
import hashlib
import json
import os
from collections import Counter

import config
import numpy as np
from preprocess import (
    LABEL_MAP,
    build_datasets,
    load_and_segment,
    load_one_file,
    random_split,
    recording_level_assignment,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MANIFEST_DIR = os.path.join(PROJECT_ROOT, "data_manifest")


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def windows_for_signal(signal_length, overlap_ratio):
    stride = max(1, int(config.WINDOW_LENGTH * (1 - overlap_ratio)))
    if signal_length < config.WINDOW_LENGTH:
        return 0
    return 1 + (signal_length - config.WINDOW_LENGTH) // stride


def build_file_manifest(records):
    rows = []
    failures = []
    for rec in records:
        try:
            signal = load_one_file(rec["path"])
            readable = True
            error = None
        except Exception as exc:
            signal = []
            readable = False
            error = f"{type(exc).__name__}: {exc}"
            failures.append(f"{rec['recording_id']}: {error}")

        rows.append(
            {
                "recording_id": rec["recording_id"],
                "relative_path": os.path.relpath(rec["path"], config.DATA_ROOT),
                "fault_type": rec["fault_type"],
                "fault_diameter": rec["diameter"],
                "load_hp": rec["load"],
                "size_bytes": os.path.getsize(rec["path"]),
                "sha256": sha256_file(rec["path"]),
                "readable": readable,
                "read_error": error,
                "de_samples": len(signal),
                "windows_50pct_overlap": windows_for_signal(
                    len(signal), config.OVERLAP_RECORDING
                ),
            }
        )

    if failures:
        raise RuntimeError(
            "Selected dataset contains unreadable files:\n  - " + "\n  - ".join(failures)
        )
    return rows


def build_split_manifest(records, file_rows):
    metadata = {row["recording_id"]: row for row in file_rows}
    manifests = []
    verification = []

    for seed in config.RANDOM_SEEDS:
        assignment = recording_level_assignment(
            records,
            train_r=config.TRAIN_RATIO,
            val_r=config.VAL_RATIO,
            seed=seed,
        )
        all_ids = assignment["train"] + assignment["validation"] + assignment["test"]
        unique_ids = set(all_ids)
        if len(all_ids) != len(unique_ids) or len(unique_ids) != len(records):
            raise RuntimeError(f"Seed {seed}: recording split is not disjoint and exhaustive.")

        seed_counts = {}
        for split_name, recording_ids in assignment.items():
            class_counts = Counter(metadata[rid]["fault_type"] for rid in recording_ids)
            window_count = sum(
                metadata[rid]["windows_50pct_overlap"]
                for rid in recording_ids
            )
            seed_counts[split_name] = {
                "recordings": len(recording_ids),
                "windows": window_count,
                "class_recordings": dict(sorted(class_counts.items())),
            }
            for rid in recording_ids:
                row = metadata[rid]
                manifests.append(
                    {
                        "seed": seed,
                        "split": split_name,
                        "recording_id": rid,
                        "fault_type": row["fault_type"],
                        "fault_diameter": row["fault_diameter"],
                        "load_hp": row["load_hp"],
                        "windows": row["windows_50pct_overlap"],
                    }
                )

        verification.append(
            {
                "seed": seed,
                "disjoint": True,
                "all_selected_recordings_assigned": True,
                "selected_recordings": len(records),
                "splits": seed_counts,
            }
        )
    return manifests, verification


def verify_random_window_split():
    windows, labels, _, _ = build_datasets(
        overlap_ratio=config.OVERLAP_RANDOM
    )
    inverse_labels = {value: key for key, value in LABEL_MAP.items()}
    verification = []
    for seed in config.RANDOM_SEEDS:
        partitions = random_split(
            windows,
            labels,
            train_r=config.TRAIN_RATIO,
            val_r=config.VAL_RATIO,
            seed=seed,
        )
        split_rows = {}
        for split_name, (_, split_labels) in zip(
            ("train", "validation", "test"), partitions
        ):
            split_rows[split_name] = {
                "windows": int(len(split_labels)),
                "class_windows": {
                    inverse_labels[label]: int(np.sum(split_labels == label))
                    for label in sorted(inverse_labels)
                },
            }

        for label, class_name in inverse_labels.items():
            total = int(np.sum(labels == label))
            expected_train = int(total * config.TRAIN_RATIO)
            expected_val_end = int(
                total * (config.TRAIN_RATIO + config.VAL_RATIO)
            )
            expected = {
                "train": expected_train,
                "validation": expected_val_end - expected_train,
                "test": total - expected_val_end,
            }
            actual = {
                split_name: split_rows[split_name]["class_windows"][class_name]
                for split_name in split_rows
            }
            if actual != expected:
                raise RuntimeError(
                    f"Seed {seed}: random split is not class-stratified for "
                    f"{class_name}: expected {expected}, found {actual}."
                )

        verification.append(
            {
                "seed": seed,
                "class_stratified": True,
                "all_windows_assigned_once": (
                    sum(row["windows"] for row in split_rows.values())
                    == len(windows)
                ),
                "splits": split_rows,
            }
        )
    return verification


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def main():
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    if config.OVERLAP_RANDOM != config.OVERLAP_RECORDING:
        raise RuntimeError(
            "Split protocols must use the same overlap so their comparison "
            "isolates the split unit."
        )
    records = load_and_segment(config.DATA_ROOT)
    if len(records) != 40:
        raise RuntimeError(
            f"Expected the documented 40-recording subset, found {len(records)}."
        )

    file_rows = build_file_manifest(records)
    split_rows, verification = build_split_manifest(records, file_rows)
    random_verification = verify_random_window_split()

    write_json(
        os.path.join(MANIFEST_DIR, "dataset_file_manifest.json"),
        {
            "data_root_at_audit": os.path.abspath(config.DATA_ROOT),
            "selection_rule": {
                "normal": "all 4 normal baseline recordings",
                "fault_dataset": "12 kHz drive-end",
                "fault_types": ["B", "IR", "OR@6"],
                "fault_diameters": ["007", "014", "021"],
                "loads_hp": [0, 1, 2, 3],
            },
            "recording_count": len(file_rows),
            "class_counts": dict(
                sorted(Counter(row["fault_type"] for row in file_rows).items())
            ),
            "files": file_rows,
        },
    )

    csv_path = os.path.join(MANIFEST_DIR, "dataset_file_manifest.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=file_rows[0].keys())
        writer.writeheader()
        writer.writerows(file_rows)

    write_json(
        os.path.join(MANIFEST_DIR, "recording_split_assignments.json"),
        split_rows,
    )
    write_json(
        os.path.join(MANIFEST_DIR, "split_verification.json"),
        {
            "protocol": "recording-level, class-stratified, 60/20/20",
            "window_length": config.WINDOW_LENGTH,
            "overlap": config.OVERLAP_RECORDING,
            "normalization": "fit on training windows only",
            "augmentation": "training partition only, after splitting",
            "seeds": verification,
        },
    )
    write_json(
        os.path.join(MANIFEST_DIR, "random_split_verification.json"),
        {
            "protocol": "random-window, class-stratified, 60/20/20",
            "window_length": config.WINDOW_LENGTH,
            "overlap": config.OVERLAP_RANDOM,
            "seeds": random_verification,
        },
    )
    print(
        f"Dataset audit PASS: {len(file_rows)} readable recordings; "
        f"{len(split_rows)} seed/recording assignments written."
    )


if __name__ == "__main__":
    main()
