import os
import glob
import scipy.io
import numpy as np
from config import DATA_ROOT, WINDOW_LENGTH, OVERLAP_RANDOM, OVERLAP_RECORDING

LABEL_MAP = {"Normal": 0, "IR": 1, "OR": 2, "B": 3}

def load_one_file(filepath):
    mat = scipy.io.loadmat(filepath)
    for key in mat:
        if "DE_time" in key:
            return mat[key].flatten()
    raise KeyError(f"No DE_time variable found in {filepath}")

def parse_filename(filename):
    base = os.path.basename(filename).replace(".mat", "")
    parts = base.split("_")
    if "Normal" in base:
        fault_type = "Normal"
        diameter = "None"
    else:
        fault_type = parts[0][:2] if (parts[0].startswith("IR") or parts[0].startswith("OR")) else parts[0][0]
        diameter_part = parts[0]
        if "@" in diameter_part:
            diameter_part = diameter_part.split("@")[0]
        digits_start = None
        for i, ch in enumerate(diameter_part):
            if ch.isdigit():
                digits_start = i
                break
        if digits_start is not None:
            diameter = diameter_part[digits_start:]
        else:
            diameter = "None"
    load_str = [p for p in parts if "HP" in p][0]
    load = int(load_str.replace("HP", ""))
    return fault_type, diameter, load

def sliding_windows(signal, window_length, overlap_ratio):
    stride = int(window_length * (1 - overlap_ratio))
    if stride <= 0:
        stride = 1
    windows = []
    for start in range(0, len(signal) - window_length + 1, stride):
        windows.append(signal[start:start + window_length])
    return np.array(windows, dtype=np.float32)

def load_and_segment(data_root, fault_diameters=None):
    if fault_diameters is None:
        fault_diameters = ["007", "014", "021"]

    normal_dir = os.path.join(data_root, "01_Normal_Baseline")
    fault_dir = os.path.join(data_root, "02_12k_Drive_End_Fault")
    for required_dir in (normal_dir, fault_dir):
        if not os.path.isdir(required_dir):
            raise FileNotFoundError(f"Required dataset directory not found: {required_dir}")

    records = []
    for fpath in sorted(glob.glob(os.path.join(normal_dir, "*.mat"))):
        records.append({"path": fpath, "recording_id": os.path.basename(fpath),
                        "fault_type": "Normal", "diameter": "None", "load": None})

    for fpath in sorted(glob.glob(os.path.join(fault_dir, "*.mat"))):
        fname = os.path.basename(fpath)
        fault_type, diameter, load = parse_filename(fname)
        if fault_type == "OR" and "@6" not in fname:
            continue
        if diameter not in fault_diameters:
            continue
        records.append({"path": fpath, "recording_id": fname,
                        "fault_type": fault_type, "diameter": diameter, "load": load})

    for rec in records:
        if rec["load"] is None:
            rec["load"] = parse_filename(rec["path"])[2]

    return records

def build_datasets(random_seed=42, fault_diameters=None, overlap_ratio=None):
    if overlap_ratio is None:
        overlap_ratio = OVERLAP_RANDOM
    records = load_and_segment(DATA_ROOT, fault_diameters)

    all_windows = []
    window_labels = []
    window_recording_ids = []
    loaded_records = []
    failures = []

    for rec in records:
        try:
            signal = load_one_file(rec["path"])
        except Exception as e:
            failures.append(f"{rec['recording_id']}: {e}")
            continue
        windows = sliding_windows(signal, WINDOW_LENGTH, overlap_ratio)
        if len(windows) == 0:
            failures.append(
                f"{rec['recording_id']}: signal is shorter than WINDOW_LENGTH={WINDOW_LENGTH}"
            )
            continue
        all_windows.append(windows)
        window_labels.extend([LABEL_MAP[rec["fault_type"]]] * len(windows))
        window_recording_ids.extend([rec["recording_id"]] * len(windows))
        loaded_records.append(rec)

    if failures:
        details = "\n  - ".join(failures)
        raise RuntimeError(
            "Dataset validation failed; selected recordings must never be silently skipped:\n"
            f"  - {details}"
        )
    if not all_windows:
        raise RuntimeError("No readable windows were produced from the selected recordings.")

    all_windows = np.concatenate(all_windows, axis=0)
    window_labels = np.array(window_labels, dtype=np.int64)
    window_recording_ids = np.array(window_recording_ids)

    return all_windows, window_labels, window_recording_ids, loaded_records

def random_split(all_windows, window_labels, train_r=0.6, val_r=0.2, seed=42):
    rng = np.random.RandomState(seed)
    split_indices = {"train": [], "validation": [], "test": []}
    for label in sorted(np.unique(window_labels)):
        class_indices = np.flatnonzero(window_labels == label)
        class_indices = rng.permutation(class_indices)
        train_end = int(len(class_indices) * train_r)
        val_end = int(len(class_indices) * (train_r + val_r))
        split_indices["train"].extend(class_indices[:train_end])
        split_indices["validation"].extend(class_indices[train_end:val_end])
        split_indices["test"].extend(class_indices[val_end:])

    # Shuffle within each partition after stratified allocation so batches do
    # not retain class blocks. The seed makes the allocation reproducible.
    train_indices = rng.permutation(split_indices["train"])
    val_indices = rng.permutation(split_indices["validation"])
    test_indices = rng.permutation(split_indices["test"])
    return (
        (all_windows[train_indices], window_labels[train_indices]),
        (all_windows[val_indices], window_labels[val_indices]),
        (all_windows[test_indices], window_labels[test_indices]),
    )

def recording_level_assignment(records, train_r=0.6, val_r=0.2, seed=42):
    rng = np.random.RandomState(seed)
    unique_recordings = list(set(rec["recording_id"] for rec in records))
    unique_recordings.sort()

    label_of_recording = {}
    for rec in records:
        label_of_recording[rec["recording_id"]] = LABEL_MAP[rec["fault_type"]]

    class_recordings = {}
    for rid in unique_recordings:
        lbl = label_of_recording[rid]
        class_recordings.setdefault(lbl, []).append(rid)

    train_ids, val_ids, test_ids = [], [], []
    for lbl, rids in class_recordings.items():
        rids_sorted = sorted(rids)
        rng.shuffle(rids_sorted)
        n = len(rids_sorted)
        train_end = max(1, int(n * train_r))
        val_end = max(1, int(n * (train_r + val_r)))
        train_ids.extend(rids_sorted[:train_end])
        val_ids.extend(rids_sorted[train_end:val_end])
        test_ids.extend(rids_sorted[val_end:])

    return {
        "train": sorted(train_ids),
        "validation": sorted(val_ids),
        "test": sorted(test_ids),
    }


def recording_level_split(all_windows, window_labels, window_recording_ids, records,
                           train_r=0.6, val_r=0.2, seed=42):
    assignment = recording_level_assignment(
        records, train_r=train_r, val_r=val_r, seed=seed
    )
    train_set = set(assignment["train"])
    val_set = set(assignment["validation"])
    test_set = set(assignment["test"])

    train_mask = np.isin(window_recording_ids, list(train_set))
    val_mask = np.isin(window_recording_ids, list(val_set))
    test_mask = np.isin(window_recording_ids, list(test_set))

    return (
        (all_windows[train_mask], window_labels[train_mask]),
        (all_windows[val_mask], window_labels[val_mask]),
        (all_windows[test_mask], window_labels[test_mask]),
    )

def normalize_dataset(train_x, val_x, test_x):
    mean = train_x.mean()
    std = train_x.std()
    if std == 0:
        std = 1.0
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std


def load_based_assignment(records, train_r=0.6, val_r=0.2, seed=42):
    rng = np.random.RandomState(seed)
    load_of_recording = {
        rec["recording_id"]: rec["load"] for rec in records
    }
    unique_loads = sorted(set(load_of_recording.values()))
    rng.shuffle(unique_loads)
    n = len(unique_loads)
    train_end = max(1, int(n * train_r))
    val_end = max(1, int(n * (train_r + val_r)))
    groups = {
        "train": sorted(unique_loads[:train_end]),
        "validation": sorted(unique_loads[train_end:val_end]),
        "test": sorted(unique_loads[val_end:]),
    }
    recordings = {
        split_name: sorted(
            rid
            for rid, load in load_of_recording.items()
            if load in set(split_groups)
        )
        for split_name, split_groups in groups.items()
    }
    return {"groups": groups, "recordings": recordings}


def load_based_split(all_windows, window_labels, window_recording_ids, records,
                     train_r=0.6, val_r=0.2, seed=42):
    assignment = load_based_assignment(
        records, train_r=train_r, val_r=val_r, seed=seed
    )

    train_mask = np.isin(
        window_recording_ids, assignment["recordings"]["train"]
    )
    val_mask = np.isin(
        window_recording_ids, assignment["recordings"]["validation"]
    )
    test_mask = np.isin(
        window_recording_ids, assignment["recordings"]["test"]
    )

    return (
        (all_windows[train_mask], window_labels[train_mask]),
        (all_windows[val_mask], window_labels[val_mask]),
        (all_windows[test_mask], window_labels[test_mask]),
    )


def fault_size_based_assignment(records, train_r=0.6, val_r=0.2, seed=42):
    rng = np.random.RandomState(seed)
    diam_of_recording = {
        rec["recording_id"]: rec["diameter"] for rec in records
    }

    normal_ids = [rid for rid, d in diam_of_recording.items() if d == "None"]
    fault_ids = [rid for rid, d in diam_of_recording.items() if d != "None"]

    rng.shuffle(normal_ids)
    n_norm = len(normal_ids)
    norm_train_end = max(1, int(n_norm * train_r))
    norm_val_end = max(1, int(n_norm * (train_r + val_r)))

    unique_diams = sorted(set(d for d in diam_of_recording.values() if d != "None"))
    rng.shuffle(unique_diams)
    n = len(unique_diams)
    train_end = max(1, int(n * train_r))
    val_end = max(1, int(n * (train_r + val_r)))
    groups = {
        "train": sorted(unique_diams[:train_end]),
        "validation": sorted(unique_diams[train_end:val_end]),
        "test": sorted(unique_diams[val_end:]),
    }
    train_diams = set(groups["train"])
    val_diams = set(groups["validation"])

    train_ids = normal_ids[:norm_train_end]
    val_ids = normal_ids[norm_train_end:norm_val_end]
    test_ids = normal_ids[norm_val_end:]

    for rid in fault_ids:
        d = diam_of_recording[rid]
        if d in train_diams:
            train_ids.append(rid)
        elif d in val_diams:
            val_ids.append(rid)
        else:
            test_ids.append(rid)

    recordings = {
        "train": sorted(train_ids),
        "validation": sorted(val_ids),
        "test": sorted(test_ids),
    }
    return {"groups": groups, "recordings": recordings}


def fault_size_based_split(all_windows, window_labels, window_recording_ids, records,
                           train_r=0.6, val_r=0.2, seed=42):
    assignment = fault_size_based_assignment(
        records, train_r=train_r, val_r=val_r, seed=seed
    )
    train_mask = np.isin(
        window_recording_ids, assignment["recordings"]["train"]
    )
    val_mask = np.isin(
        window_recording_ids, assignment["recordings"]["validation"]
    )
    test_mask = np.isin(
        window_recording_ids, assignment["recordings"]["test"]
    )

    return (
        (all_windows[train_mask], window_labels[train_mask]),
        (all_windows[val_mask], window_labels[val_mask]),
        (all_windows[test_mask], window_labels[test_mask]),
    )
