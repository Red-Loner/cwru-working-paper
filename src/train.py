import os
import json
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

def to_stft(x_batch, n_fft=128):
    b, l = x_batch.shape
    hop_length = n_fft // 2
    freqs = n_fft // 2 + 1
    frames = (l - n_fft) // hop_length + 1
    x_t = torch.tensor(x_batch, dtype=torch.float32)
    window = torch.hann_window(n_fft)
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
    cm = confusion_matrix(labels, preds).tolist()
    report = classification_report(labels, preds, target_names=list(LABEL_MAP.keys()),
                                   output_dict=True, zero_division=0)
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(mf1, 4),
        "loss": round(total_loss / len(labels), 4),
        "confusion_matrix": cm,
        "per_class": {k: {"precision": round(v["precision"], 4),
                          "recall": round(v["recall"], 4)}
                      for k, v in report.items() if k in LABEL_MAP},
    }


def run_experiment(model_type, split_type, aug_type, seed):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    windows, labels, rec_ids, records = build_datasets(random_seed=seed)

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

    best_va_acc = 0
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, tr_loader, optimizer, criterion, device, model_type)
        va_result = evaluate(model, va_loader, device, model_type)
        if va_result["accuracy"] > best_va_acc:
            best_va_acc = va_result["accuracy"]
            torch.save(model.state_dict(), f"best_{model_type}_{split_type}_{aug_type}_seed{seed}.pt")

    model.load_state_dict(torch.load(f"best_{model_type}_{split_type}_{aug_type}_seed{seed}.pt", map_location=device))
    te_result = evaluate(model, te_loader, device, model_type)
    return te_result


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "tables"), exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "logs"), exist_ok=True)

    models = ["1d", "2d"]
    splits = ["random", "recording"]
    augs = AUGMENTATION_CONDITIONS

    all_results = []

    for model_type in models:
        for split_type in splits:
            for aug_type in augs:
                seed_results = []
                for seed in config.RANDOM_SEEDS:
                    print(f"[{model_type} | {split_type} | {aug_type} | seed={seed}]")
                    try:
                        result = run_experiment(model_type, split_type, aug_type, seed)
                        seed_results.append(result)
                    except Exception as e:
                        print(f"  ERROR: {e}")
                        seed_results.append(None)

                valid = [r for r in seed_results if r is not None]
                if not valid:
                    continue
                avg_acc = np.mean([r["accuracy"] for r in valid])
                std_acc = np.std([r["accuracy"] for r in valid])
                avg_f1 = np.mean([r["macro_f1"] for r in valid])
                std_f1 = np.std([r["macro_f1"] for r in valid])

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
                }
                all_results.append(row)
                print(f"  → Acc: {avg_acc:.4f} ± {std_acc:.4f}  F1: {avg_f1:.4f} ± {std_f1:.4f}")

    with open(os.path.join(config.RESULTS_DIR, "tables", "main_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {config.RESULTS_DIR}/tables/main_results.json")
    print(f"Total combinations: {len(all_results)}")


if __name__ == "__main__":
    main()
