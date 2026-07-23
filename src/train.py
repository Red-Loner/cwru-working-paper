import os
import json
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np

import config
from preprocess import (
    build_datasets, random_split, recording_level_split, normalize_dataset,
    LABEL_MAP,
)
from augmentation import AUGMENTATION_CONDITIONS, apply_augmentation
from models.cnn1d import CNN1D
from models.cnn2d import CNN2D

IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}


def set_reproducible_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_stft(x_batch, n_fft=128):
    b, l = x_batch.shape
    hop_length = n_fft // 2
    freqs = n_fft // 2 + 1
    frames = (l - n_fft) // hop_length + 1
    if isinstance(x_batch, torch.Tensor):
        x_t = x_batch.float()
    else:
        x_t = torch.from_numpy(x_batch).float()
    window = torch.hann_window(n_fft, device=x_t.device)
    spec = torch.stft(
        x_t, n_fft=n_fft, hop_length=hop_length, win_length=n_fft,
        window=window, return_complex=True, onesided=True
    )
    mag = spec.abs()
    mag = mag[:, :32, :32]
    if mag.size(-1) > 32:
        mag = mag[:, :, :32]
    if mag.size(-1) < 32:
        pad_w = 32 - mag.size(-1)
        mag = torch.nn.functional.pad(mag, (0, pad_w))
    mag = (mag - mag.mean(dim=(1, 2), keepdim=True)) / (mag.std(dim=(1, 2), keepdim=True) + 1e-8)
    return mag.unsqueeze(1)


def train_epoch(model, loader, optimizer, criterion, device, model_type="1d"):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if model_type == "2d":
            x = to_stft(x.squeeze(1))
            x = x.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += x.size(0)
    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(model, loader, device, model_type="1d"):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if model_type == "2d":
            x = to_stft(x.squeeze(1))
            x = x.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    acc = accuracy_score(labels, preds)
    mf1 = f1_score(labels, preds, average="macro")
    label_ids = list(range(len(LABEL_MAP)))
    cm = confusion_matrix(labels, preds, labels=label_ids).tolist()
    report = classification_report(
        labels,
        preds,
        labels=label_ids,
        target_names=list(LABEL_MAP.keys()),
        output_dict=True,
        zero_division=0,
    )
    per_class = {}
    for k in LABEL_MAP:
        if k in report:
            per_class[k] = {
                "precision": round(report[k]["precision"], 4),
                "recall": round(report[k]["recall"], 4),
                "f1": round(report[k]["f1-score"], 4),
            }
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(mf1, 4),
        "loss": round(total_loss / len(labels), 4),
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def run_experiment(model_type, split_type, aug_type, seed, log_dir=None):
    set_reproducible_seed(seed)
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    overlap = config.OVERLAP_RECORDING if split_type == "recording" else config.OVERLAP_RANDOM
    windows, labels, rec_ids, records = build_datasets(random_seed=seed, overlap_ratio=overlap)

    if split_type == "random":
        (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = random_split(
            windows, labels, train_r=config.TRAIN_RATIO,
            val_r=config.VAL_RATIO, seed=seed)
    else:
        (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = recording_level_split(
            windows, labels, rec_ids, records,
            train_r=config.TRAIN_RATIO, val_r=config.VAL_RATIO, seed=seed)

    tr_x, va_x, te_x = normalize_dataset(tr_x, va_x, te_x)

    if aug_type != "none":
        tr_x = apply_augmentation(tr_x, aug_type)

    tr_ds = TensorDataset(torch.tensor(tr_x, dtype=torch.float32).unsqueeze(1),
                          torch.tensor(tr_y, dtype=torch.long))
    va_ds = TensorDataset(torch.tensor(va_x, dtype=torch.float32).unsqueeze(1),
                          torch.tensor(va_y, dtype=torch.long))
    te_ds = TensorDataset(torch.tensor(te_x, dtype=torch.float32).unsqueeze(1),
                          torch.tensor(te_y, dtype=torch.long))

    tr_loader = DataLoader(tr_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=config.BATCH_SIZE)
    te_loader = DataLoader(te_ds, batch_size=config.BATCH_SIZE)

    model = CNN1D() if model_type == "1d" else CNN2D()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    epoch_log = []
    best_va_acc = 0
    best_state = None
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, tr_loader, optimizer, criterion, device, model_type)
        va_result = evaluate(model, va_loader, device, model_type)
        epoch_log.append([epoch, train_loss, train_acc, va_result["accuracy"], va_result["loss"]])
        if va_result["accuracy"] > best_va_acc:
            best_va_acc = va_result["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if log_dir:
        log_path = os.path.join(log_dir, f"{model_type}_{split_type}_{aug_type}_s{seed}.csv")
        with open(log_path, "w") as f:
            f.write("epoch,train_loss,train_acc,val_acc,val_loss\n")
            for row in epoch_log:
                f.write(",".join(str(x) for x in row) + "\n")

    if best_state is not None:
        model.load_state_dict(best_state)
    te_result = evaluate(model, te_loader, device, model_type)
    te_result["seed"] = seed
    te_result["best_val_acc"] = round(best_va_acc, 4)
    te_result["final_epoch"] = config.NUM_EPOCHS
    return te_result


def load_existing_results(results_dir):
    path = os.path.join(results_dir, "tables", "main_results.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        rows = json.load(f)
    existing = {}
    for row in rows:
        key = (row["model"], row["split"], row["augmentation"])
        existing[key] = row
    return existing


def write_result_artifacts(results):
    tables_dir = os.path.join(config.RESULTS_DIR, "tables")
    main_path = os.path.join(tables_dir, "main_results.json")
    with open(main_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

    confusion_rows = []
    for row in results:
        if not row.get("per_seed_results"):
            continue
        confusion_rows.append(
            {
                "model": row["model"],
                "split": row["split"],
                "augmentation": row["augmentation"],
                "class_order": list(LABEL_MAP.keys()),
                "per_seed": [
                    {
                        "seed": seed_row["seed"],
                        "confusion_matrix": seed_row["confusion_matrix"],
                    }
                    for seed_row in row["per_seed_results"]
                ],
                "confusion_matrix_sum": row["confusion_matrix_sum"],
            }
        )
    confusion_path = os.path.join(tables_dir, "confusion_matrices.json")
    with open(confusion_path, "w", encoding="utf-8") as handle:
        json.dump(confusion_rows, handle, indent=2, ensure_ascii=False)


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "tables"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "logs"), exist_ok=True)
    epoch_log_dir = os.path.join(config.RESULTS_DIR, "logs", "epoch_logs")
    os.makedirs(epoch_log_dir, exist_ok=True)

    existing = load_existing_results(config.RESULTS_DIR)

    models = ["1d", "2d"]
    splits = ["random", "recording"]
    augs = AUGMENTATION_CONDITIONS

    all_results = []
    total_new = 0
    total_skip = 0

    for model_type in models:
        for split_type in splits:
            for aug_type in augs:
                key = (model_type, split_type, aug_type)
                old_entry = existing.get(key, None)
                old_seed_results = {}
                if old_entry and old_entry.get("per_seed_results"):
                    old_seed_results = {
                        seed_row["seed"]: seed_row
                        for seed_row in old_entry["per_seed_results"]
                        if "confusion_matrix" in seed_row
                        and "per_class" in seed_row
                    }

                new_runs = []
                for seed in config.RANDOM_SEEDS:
                    if seed in old_seed_results:
                        print(f"[{model_type} | {split_type} | {aug_type} | seed={seed}] SKIPPED (cached)")
                        total_skip += 1
                        new_runs.append(old_seed_results[seed])
                        continue

                    print(f"[{model_type} | {split_type} | {aug_type} | seed={seed}] RUNNING")
                    total_new += 1
                    try:
                        result = run_experiment(model_type, split_type, aug_type, seed, log_dir=epoch_log_dir)
                        new_runs.append(result)
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        new_runs.append(None)

                valid = [r for r in new_runs if r is not None]
                if not valid:
                    row = old_entry if old_entry else None
                    if row:
                        all_results.append(row)
                    continue

                avg_acc = np.mean([r["accuracy"] for r in valid])
                std_acc = np.std([r["accuracy"] for r in valid])
                avg_f1 = np.mean([r["macro_f1"] for r in valid])
                std_f1 = np.std([r["macro_f1"] for r in valid])

                per_class_agg = {}
                for cls_name in LABEL_MAP:
                    c_precisions = []
                    c_recalls = []
                    c_f1s = []
                    for r in valid:
                        if cls_name in r.get("per_class", {}):
                            c_precisions.append(r["per_class"][cls_name]["precision"])
                            c_recalls.append(r["per_class"][cls_name]["recall"])
                            c_f1s.append(r["per_class"][cls_name]["f1"])
                    if c_precisions:
                        per_class_agg[cls_name] = {
                            "precision_mean": round(float(np.mean(c_precisions)), 4),
                            "precision_std": round(float(np.std(c_precisions)), 4),
                            "recall_mean": round(float(np.mean(c_recalls)), 4),
                            "recall_std": round(float(np.std(c_recalls)), 4),
                            "f1_mean": round(float(np.mean(c_f1s)), 4),
                            "f1_std": round(float(np.std(c_f1s)), 4),
                        }

                confusion_matrices = [
                    np.asarray(r["confusion_matrix"], dtype=np.int64) for r in valid
                ]
                confusion_sum = np.sum(confusion_matrices, axis=0).tolist()

                row = {
                    "model": model_type,
                    "split": split_type,
                    "augmentation": aug_type,
                    "accuracy_mean": round(float(avg_acc), 4),
                    "accuracy_std": round(float(std_acc), 4),
                    "macro_f1_mean": round(float(avg_f1), 4),
                    "macro_f1_std": round(float(std_f1), 4),
                    "per_seed_accuracy": [r["accuracy"] for r in valid],
                    "per_seed_macro_f1": [r["macro_f1"] for r in valid],
                    "per_seed_ids": [s for s in config.RANDOM_SEEDS
                                     if new_runs[config.RANDOM_SEEDS.index(s)] is not None],
                    "per_seed_results": valid,
                    "per_seed_confusion_matrices": [
                        r["confusion_matrix"] for r in valid
                    ],
                    "confusion_matrix_sum": confusion_sum,
                    "per_class": per_class_agg,
                }
                all_results.append(row)
                print(f"  → Acc: {avg_acc:.4f} ± {std_acc:.4f}  F1: {avg_f1:.4f} ± {std_f1:.4f}")

                write_result_artifacts(all_results)

    write_result_artifacts(all_results)

    print(f"\nResults saved to {config.RESULTS_DIR}/tables/main_results.json")
    print(f"Total combinations: {len(all_results)}")
    print(f"New runs: {total_new}, Cached: {total_skip}")
    print(f"Seeds per combo: {len(config.RANDOM_SEEDS)}")


if __name__ == "__main__":
    main()
