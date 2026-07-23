import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from augmentation import apply_augmentation
from config import RANDOM_SEEDS, RESULTS_DIR, TRAIN_RATIO, VAL_RATIO
from preprocess import (
    build_datasets,
    normalize_dataset,
    recording_level_split,
)
from train import to_stft


def representation_diversity(aug_type, seed=42, n_samples=200):
    """Compare deterministic STFT representations before and after augmentation."""
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    windows, labels, rec_ids, records = build_datasets(
        random_seed=seed
    )
    (train_x, _), _, _ = recording_level_split(
        windows,
        labels,
        rec_ids,
        records,
        train_r=TRAIN_RATIO,
        val_r=VAL_RATIO,
        seed=seed,
    )
    train_norm, _, _ = normalize_dataset(
        train_x,
        np.zeros_like(train_x),
        np.zeros_like(train_x),
    )
    sample_count = min(n_samples, len(train_norm))
    indices = rng.choice(len(train_norm), sample_count, replace=False)
    original = train_norm[indices]
    augmented = apply_augmentation(original, aug_type)

    original_rep = (
        to_stft(original).detach().cpu().numpy().reshape(sample_count, -1)
    )
    augmented_rep = (
        to_stft(augmented).detach().cpu().numpy().reshape(sample_count, -1)
    )

    numerator = np.sum(original_rep * augmented_rep, axis=1)
    denominator = (
        np.linalg.norm(original_rep, axis=1)
        * np.linalg.norm(augmented_rep, axis=1)
    )
    cosine_distance = 1.0 - np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 0,
    )

    combined = np.concatenate([original_rep, augmented_rep], axis=0)
    singular_values = np.linalg.svd(combined, compute_uv=False)
    threshold = 1e-4 * singular_values[0]
    effective_rank = int(np.sum(singular_values > threshold))

    return {
        "representation": "standardized 32x32 STFT magnitude",
        "sample_count": sample_count,
        "cosine_distance_mean": float(np.mean(cosine_distance)),
        "cosine_distance_std": float(np.std(cosine_distance)),
        "effective_rank": effective_rank,
        "effective_rank_ratio": float(effective_rank / len(singular_values)),
    }


def plot_feature_diversity(aug_diversity, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)

    augmentations = list(aug_diversity)
    cosine_means = [
        aug_diversity[augmentation]["cosine_distance_mean"]
        for augmentation in augmentations
    ]
    cosine_stds = [
        aug_diversity[augmentation]["cosine_distance_std"]
        for augmentation in augmentations
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        augmentations,
        cosine_means,
        yerr=cosine_stds,
        capsize=3,
        color="steelblue",
    )
    ax.set_xticks(np.arange(len(augmentations)))
    ax.set_xticklabels(augmentations, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Paired STFT cosine distance")
    ax.set_title("M2: Input-Representation Diversity")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()

    path = os.path.join(output_dir, "m2_feature_diversity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    augmentations = [
        "none",
        "noise_005",
        "shift_20",
        "specaugment",
        "combined",
        "freq_flip",
    ]
    results = {}
    for augmentation in augmentations:
        print(f"Computing representation diversity: {augmentation}")
        results[augmentation] = representation_diversity(
            augmentation,
            seed=RANDOM_SEEDS[0],
        )
        print(
            f"  cosine_dist={results[augmentation]['cosine_distance_mean']:.4f}, "
            f"rank_ratio={results[augmentation]['effective_rank_ratio']:.4f}"
        )

    plot_feature_diversity(results)
    output_path = os.path.join(
        RESULTS_DIR, "tables", "feature_diversity.json"
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
