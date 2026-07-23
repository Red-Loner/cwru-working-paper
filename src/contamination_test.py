import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from preprocess import build_datasets, recording_level_split, random_split, normalize_dataset
from augmentation import apply_augmentation, AUGMENTATION_CONDITIONS
from train import set_reproducible_seed, to_stft
from models.cnn1d import CNN1D
from models.cnn2d import CNN2D
from config import (
    DATA_ROOT, RANDOM_SEEDS, TRAIN_RATIO, VAL_RATIO, BATCH_SIZE,
    LEARNING_RATE, NUM_EPOCHS, DEVICE, RESULTS_DIR,
)


def add_noise(x, sigma, rng):
    noise = rng.randn(*x.shape).astype(np.float32) * sigma
    return x + noise


def quick_train(model, train_loader, device, epochs=20):
    import torch.nn as nn
    import torch.optim as optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()


@torch.no_grad()
def test_acc(model, test_x, test_y, device, model_type="1d"):
    model.eval()
    if model_type == "2d":
        x_t = torch.tensor(test_x, dtype=torch.float32)
        x_t = to_stft(x_t)
        ds = TensorDataset(x_t, torch.tensor(test_y, dtype=torch.long))
    else:
        ds = TensorDataset(torch.tensor(test_x, dtype=torch.float32).unsqueeze(1),
                           torch.tensor(test_y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=BATCH_SIZE)
    preds, labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        preds.append(out.argmax(1).cpu().numpy())
        labels.append(y.cpu().numpy())
    return accuracy_score(np.concatenate(labels), np.concatenate(preds))


def run_contamination_test(model_type="2d", split_type="recording", seed=42):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    windows, labels, rec_ids, records = build_datasets(random_seed=seed)
    if split_type == "random":
        (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = random_split(
            windows, labels, train_r=TRAIN_RATIO, val_r=VAL_RATIO, seed=seed)
    else:
        (tr_x, tr_y), (va_x, va_y), (te_x, te_y) = recording_level_split(
            windows, labels, rec_ids, records, train_r=TRAIN_RATIO, val_r=VAL_RATIO, seed=seed)
    noise_levels = [0, 0.01, 0.02, 0.05, 0.08, 0.1]
    results = {}
    for aug_type in ["none", "noise_005", "shift_20", "combined"]:
        # Use paired model initialization and data-loader randomness so the
        # robustness comparison isolates the augmentation condition.
        set_reproducible_seed(seed)
        tr_x_norm, va_x_norm, te_x_norm = normalize_dataset(tr_x.copy(), va_x.copy(), te_x.copy())

        if aug_type != "none":
            tr_aug = apply_augmentation(tr_x_norm, aug_type)
        else:
            tr_aug = tr_x_norm

        if model_type == "2d":
            tr_tensor = to_stft(torch.tensor(tr_aug, dtype=torch.float32))
        else:
            tr_tensor = torch.tensor(tr_aug, dtype=torch.float32).unsqueeze(1)
        tr_ds = TensorDataset(tr_tensor, torch.tensor(tr_y, dtype=torch.long))
        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = CNN1D() if model_type == "1d" else CNN2D()
        model = model.to(device)
        quick_train(model, tr_loader, device, epochs=NUM_EPOCHS)

        noise_accs = []
        for sigma in noise_levels:
            te_noisy = te_x_norm.copy()
            if sigma > 0:
                # Recreate the same test perturbation for every augmentation.
                noise_rng = np.random.RandomState(
                    seed + int(round(sigma * 10_000))
                )
                te_noisy = add_noise(te_noisy, sigma, noise_rng)
            acc = test_acc(model, te_noisy, te_y, device, model_type)
            noise_accs.append({"noise_sigma": sigma, "accuracy": round(float(acc), 4)})
        results[aug_type] = noise_accs
        print(f"  {aug_type}: clean_acc={noise_accs[0]['accuracy']:.4f}, "
              f"noise_0.1={noise_accs[-1]['accuracy']:.4f}")
    return results


def main():
    output_dir = os.path.join(RESULTS_DIR, "tables")
    os.makedirs(output_dir, exist_ok=True)
    all_results = {}
    for split_type in ["recording", "random"]:
        print(f"Contamination test: {split_type} split, 2D model")
        key = f"2d_{split_type}"
        all_results[key] = run_contamination_test("2d", split_type, seed=RANDOM_SEEDS[0])
    output_path = os.path.join(output_dir, "contamination_robustness.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
