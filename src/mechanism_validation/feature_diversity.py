import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader, TensorDataset
from preprocess import build_datasets, recording_level_split, normalize_dataset
from augmentation import apply_augmentation, AUGMENTATION_CONDITIONS
from models.cnn1d import CNN1D
from models.cnn2d import CNN2D
from config import DATA_ROOT, RANDOM_SEEDS, TRAIN_RATIO, VAL_RATIO, RESULTS_DIR, DEVICE


def extract_features(model, loader):
    model.eval()
    features = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE)
            feat = model.features(x).cpu().numpy()
            features.append(feat)
    return np.concatenate(features, axis=0)


def feature_diversity(x_orig, x_aug, model_type="1d", seed=42):
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    windows, labels, rec_ids, records = build_datasets(random_seed=seed)
    (tr_x, tr_y), _, _ = recording_level_split(
        windows, labels, rec_ids, records,
        train_r=TRAIN_RATIO, val_r=VAL_RATIO, seed=seed)

    tr_norm, _, _ = normalize_dataset(tr_x, np.zeros_like(tr_x), np.zeros_like(tr_x))
    tr_aug = apply_augmentation(tr_norm, x_aug)

    model = CNN1D() if model_type == "1d" else CNN2D()
    model.features = lambda x: model.fc1(model.pool4(
        model.bn4(model.conv4(
            model.pool3(model.bn3(model.conv3(
                model.pool2(model.bn2(model.conv2(
                    model.pool1(model.bn1(model.conv1(x.unsqueeze(1) if x.dim() == 2 else x)))))))))))).view(x.size(0), -1))
    model = model.to(device)

    n_samples = min(200, len(tr_norm))
    indices = np.random.choice(len(tr_norm), n_samples, replace=False)
    tr_sub = tr_norm[indices]
    tr_aug_sub = tr_aug[indices]

    ds_orig = TensorDataset(torch.tensor(tr_sub, dtype=torch.float32), torch.zeros(n_samples, dtype=torch.long))
    ds_aug = TensorDataset(torch.tensor(tr_aug_sub, dtype=torch.float32), torch.zeros(n_samples, dtype=torch.long))
    loader_orig = DataLoader(ds_orig, batch_size=32)
    loader_aug = DataLoader(ds_aug, batch_size=32)

    feat_orig = extract_features(model, loader_orig)
    feat_aug = extract_features(model, loader_aug)

    cos_dists = np.array([cosine(feat_orig[i], feat_aug[i]) for i in range(n_samples)])
    cos_mean = float(np.mean(cos_dists))
    cos_std = float(np.std(cos_dists))

    combined = np.concatenate([feat_orig, feat_aug], axis=0)
    U, S, Vt = np.linalg.svd(combined, full_matrices=False)
    effective_rank = float(np.sum(S > 1e-4 * S[0]) / len(S))

    return {"cosine_distance_mean": cos_mean, "cosine_distance_std": cos_std,
            "effective_rank_ratio": effective_rank}


def plot_feature_diversity(aug_diversity, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)

    augs = list(aug_diversity.keys())
    cos_means = [aug_diversity[a]["cosine_distance_mean"] for a in augs]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    bars = plt.bar(augs, cos_means, color="steelblue")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.ylabel("Mean Cosine Distance")
    plt.title("M2: Feature Diversity (Cosine Distance)")
    plt.tight_layout()

    path = os.path.join(output_dir, "m2_feature_diversity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return aug_diversity


def main():
    augs = ["none", "noise_005", "shift_20", "specaugment", "combined", "freq_flip"]
    results = {}
    for aug in augs:
        print(f"Computing feature diversity: {aug}")
        try:
            results[aug] = feature_diversity(aug, aug, model_type="2d", seed=RANDOM_SEEDS[0])
            print(f"  cosine_dist={results[aug]['cosine_distance_mean']:.4f}, "
                  f"rank_ratio={results[aug]['effective_rank_ratio']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")
    plot_feature_diversity(results)


if __name__ == "__main__":
    main()
