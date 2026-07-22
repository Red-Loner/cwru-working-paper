import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

import config
from preprocess import (
    build_datasets, random_split, recording_level_split,
    load_based_split, fault_size_based_split, normalize_dataset, LABEL_MAP,
)
from models.cnn1d import CNN1D
from models.cnn2d import CNN2D
from train import train_epoch, evaluate


SPLIT_FUNCTIONS = {
    "recording": recording_level_split,
    "load": load_based_split,
    "fault_size": fault_size_based_split,
}

SPLIT_LABELS = {
    "recording": "By Recording",
    "load": "By Load (HP)",
    "fault_size": "By Fault Size",
}


def run_single(model_type, split_name, split_fn, seed):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    windows, labels, rec_ids, records = build_datasets(random_seed=seed)

    (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = split_fn(
        windows, labels, rec_ids, records,
        train_r=config.TRAIN_RATIO, val_r=config.VAL_RATIO, seed=seed)

    tr_x, va_x, te_x = normalize_dataset(tr_x, va_x, te_x)

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
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    best_va_acc = 0
    best_state = None
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, tr_loader, optimizer, criterion, device, model_type)
        va_result = evaluate(model, va_loader, device, model_type)
        if va_result["accuracy"] > best_va_acc:
            best_va_acc = va_result["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    te_result = evaluate(model, te_loader, device, model_type)
    return te_result


def load_existing_ablation(results_dir):
    path = os.path.join(results_dir, "tables", "grouping_ablation.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    existing = {}
    for row in rows:
        key = (row["model"], row["split"])
        if key not in existing:
            existing[key] = {}
        existing[key][row["seed"]] = row
    return existing


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.RESULTS_DIR, "tables"), exist_ok=True)

    existing = load_existing_ablation(config.RESULTS_DIR)

    models = ["1d", "2d"]
    splits = ["recording", "load", "fault_size"]
    seeds = config.RANDOM_SEEDS

    all_results = []

    for model_type in models:
        for split_name in splits:
            split_fn = SPLIT_FUNCTIONS[split_name]
            for seed in seeds:
                key = (model_type, split_name)
                if key in existing and seed in existing[key]:
                    print(f"[{model_type} | {split_name} | seed={seed}] SKIPPED (cached)")
                    all_results.append(existing[key][seed])
                    continue

                print(f"[{model_type} | {split_name} | seed={seed}] RUNNING")
                try:
                    result = run_single(model_type, split_name, split_fn, seed)
                    row = {
                        "model": model_type,
                        "split": split_name,
                        "seed": seed,
                        "accuracy": result["accuracy"],
                        "macro_f1": result["macro_f1"],
                        "per_class": result.get("per_class", {}),
                    }
                    all_results.append(row)
                    print(f"  → Acc: {result['accuracy']:.4f}  F1: {result['macro_f1']:.4f}")
                except Exception as e:
                    print(f"  ERROR: {e}")
                    row = {
                        "model": model_type,
                        "split": split_name,
                        "seed": seed,
                        "accuracy": None,
                        "macro_f1": None,
                        "error": str(e),
                    }
                    all_results.append(row)

                path = os.path.join(config.RESULTS_DIR, "tables", "grouping_ablation.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2, ensure_ascii=False)

    path = os.path.join(config.RESULTS_DIR, "tables", "grouping_ablation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n=== ABLATION SUMMARY ===")
    for model_type in models:
        for split_name in splits:
            rows = [r for r in all_results if r["model"] == model_type and r["split"] == split_name and r["accuracy"] is not None]
            if rows:
                accs = [r["accuracy"] for r in rows]
                f1s = [r["macro_f1"] for r in rows]
                print(f"  [{model_type}] {SPLIT_LABELS[split_name]:15s} acc={np.mean(accs):.4f} ± {np.std(accs):.4f}  f1={np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            else:
                print(f"  [{model_type}] {SPLIT_LABELS[split_name]:15s} NO RESULTS")

    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
