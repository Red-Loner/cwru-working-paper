import os
import glob
import scipy.io
import numpy as np
from config import DATA_ROOT, WINDOW_LENGTH, OVERLAP_RANDOM, OVERLAP_RECORDING

LABEL_MAP = {"Normal": 0, "IR": 1, "OR": 2, "B": 3}

def load_one_file(filepath):
    mat = scipy.io.loadmat(filepath)
    for key in mat:
        if key.startswith("DE"):
            return mat[key].flatten()
    raise KeyError(f"No DE variable found in {filepath}")

def parse_filename(filename):
    base = os.path.basename(filename).replace(".mat", "")
    parts = base.split("_")
    if "Normal" in base:
        fault_type = "Normal"
        diameter = "None"
    else:
        fault_type = parts[0][:2] if parts[0].startswith("IR") or parts[0].startswith("OR") else parts[0][0]
        fault_type = "Normal" if fault_type == "Norm" else fault_type
        for p in parts[0]:
            if p.isdigit():
                diameter = parts[0][parts[0].index(p):]
                break
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

def build_datasets(random_seed=42, fault_diameters=None):
    records = load_and_segment(DATA_ROOT, fault_diameters)

    all_windows = []
    window_labels = []
    window_recording_ids = []

    for rec in records:
        signal = load_one_file(rec["path"])
        windows = sliding_windows(signal, WINDOW_LENGTH, OVERLAP_RANDOM)
        all_windows.append(windows)
        window_labels.extend([LABEL_MAP[rec["fault_type"]]] * len(windows))
        window_recording_ids.extend([rec["recording_id"]] * len(windows))

    all_windows = np.concatenate(all_windows, axis=0)
    window_labels = np.array(window_labels, dtype=np.int64)
    window_recording_ids = np.array(window_recording_ids)

    return all_windows, window_labels, window_recording_ids, records

def random_split(all_windows, window_labels, train_r=0.6, val_r=0.2, seed=42):
    rng = np.random.RandomState(seed)
    n = len(all_windows)
    indices = rng.permutation(n)
    train_end = int(n * train_r)
    val_end = int(n * (train_r + val_r))
    return (
        (all_windows[indices[:train_end]], window_labels[indices[:train_end]]),
        (all_windows[indices[train_end:val_end]], window_labels[indices[train_end:val_end]]),
        (all_windows[indices[val_end:]], window_labels[indices[val_end:]]),
    )

def recording_level_split(all_windows, window_labels, window_recording_ids, records,
                           train_r=0.6, val_r=0.2, seed=42):
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

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

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
