import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS_DIR

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def plot_gap_recovery(recovery_rows, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)
    for model in ["1d", "2d"]:
        rows = [r for r in recovery_rows if r["model"] == model and r["augmentation"] != "none"]
        if not rows:
            continue
        rows.sort(key=lambda r: r["delta_recovery"], reverse=True)
        augs = [r["augmentation"] for r in rows]
        deltas = [r["delta_recovery"] for r in rows]
        bars = plt.figure(figsize=(10, 5))
        colors = ["green" if d > 0 else "red" for d in deltas]
        plt.barh(augs, deltas, color=colors)
        plt.axvline(x=0, color="black", linewidth=0.8)
        plt.axvline(x=1.0, color="blue", linestyle="--", linewidth=1, alpha=0.5, label="Δ_recovery = 1.0 (full)")
        plt.xlabel("Δ_recovery (Gap-Recovery Ratio)")
        plt.title(f"Gap-Recovery Ratio by Augmentation ({model.upper()} Model)")
        plt.legend()
        plt.tight_layout()
        path = os.path.join(output_dir, f"gap_recovery_{model}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


def plot_random_vs_recording(results, model_type="2d", output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)
    random_accs = {}
    recording_accs = {}
    random_stds = {}
    recording_stds = {}
    for row in results:
        if row["model"] != model_type:
            continue
        if row["split"] == "random":
            random_accs[row["augmentation"]] = row["accuracy_mean"]
            random_stds[row["augmentation"]] = row.get("accuracy_std", 0)
        else:
            recording_accs[row["augmentation"]] = row["accuracy_mean"]
            recording_stds[row["augmentation"]] = row.get("accuracy_std", 0)
    augs = sorted(set(list(random_accs.keys()) + list(recording_accs.keys())))
    x = np.arange(len(augs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    r_vals = [random_accs.get(a, 0) for a in augs]
    rc_vals = [recording_accs.get(a, 0) for a in augs]
    r_errs = [random_stds.get(a, 0) for a in augs]
    rc_errs = [recording_stds.get(a, 0) for a in augs]
    ax.bar(x - width / 2, r_vals, width, yerr=r_errs, label="Random Split (leakage-prone)", color="coral", alpha=0.8,
           capsize=2, error_kw={"linewidth": 0.8})
    ax.bar(x + width / 2, rc_vals, width, yerr=rc_errs, label="Recording-Level Split (recording-disjoint)", color="steelblue", alpha=0.8,
           capsize=2, error_kw={"linewidth": 0.8})
    ax.set_xlabel("Augmentation Condition")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Random vs Recording-Level Split ({model_type.upper()} Model, {len(results[0].get('per_seed_accuracy',[]))} seeds)")
    ax.set_xticks(x)
    ax.set_xticklabels(augs, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    lower_candidates = [
        value - error
        for value, error in zip(r_vals + rc_vals, r_errs + rc_errs)
    ]
    ax.set_ylim(max(0.0, min(lower_candidates) - 0.03), 1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f"random_vs_recording_{model_type}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_physical_fidelity_correlation(correlation_data, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)
    by_model = correlation_data.get("by_model", {})
    if not by_model:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True)
    for ax, model in zip(axes, ("1d", "2d")):
        model_data = by_model.get(model, {})
        fidelities = model_data.get("energy_fidelity", [])
        deltas = model_data.get("delta_recovery", [])
        augmentations = model_data.get("augmentations", [])
        rho = model_data.get("spearman_rho")
        p_value = model_data.get("p_value")
        for index, augmentation in enumerate(augmentations):
            color = "green" if deltas[index] > 0 else "red"
            ax.scatter(
                fidelities[index],
                deltas[index],
                s=90,
                alpha=0.75,
                color=color,
                edgecolors="black",
                linewidth=0.5,
            )
            ax.annotate(
                augmentation,
                (fidelities[index], deltas[index]),
                textcoords="offset points",
                xytext=(5, 5 if index % 2 == 0 else -10),
                fontsize=7,
            )
        ax.axhline(
            y=0,
            color="black",
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
        )
        ax.set_xlabel("Symmetric fault-band energy fidelity")
        ax.set_ylabel("Gap-recovery ratio")
        if rho is None:
            subtitle = "correlation unavailable"
        else:
            subtitle = f"Spearman ρ={rho:.3f}, p={p_value:.3f}"
        ax.set_title(f"{model.upper()} model\n{subtitle}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("H3: Physical Fidelity vs Gap Recovery")
    fig.tight_layout()
    path = os.path.join(
        output_dir, "h3_physical_fidelity_correlation.png"
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_per_class_heatmap(results, model_type="2d", split_type="recording", output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)
    rows = [r for r in results if r["model"] == model_type and r["split"] == split_type
            and "per_class" in r and r["per_class"]]
    if not rows:
        return
    classes = list(rows[0]["per_class"].keys())
    augs = [r["augmentation"] for r in rows]
    f1_matrix = np.zeros((len(augs), len(classes)))
    for i, row in enumerate(rows):
        for j, cls in enumerate(classes):
            if cls in row["per_class"]:
                f1_matrix[i, j] = row["per_class"][cls]["f1_mean"]
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.5), max(6, len(augs) * 0.5)))
    color_min = max(0.0, float(np.min(f1_matrix)) - 0.05)
    im = ax.imshow(
        f1_matrix,
        cmap="RdYlGn",
        vmin=color_min,
        vmax=1.0,
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(np.arange(len(augs)))
    ax.set_yticklabels(augs)
    for i in range(len(augs)):
        for j in range(len(classes)):
            text = ax.text(j, i, f"{f1_matrix[i, j]:.3f}", ha="center", va="center",
                          color="black" if 0.75 < f1_matrix[i, j] < 0.98 else "white",
                          fontsize=8)
    ax.set_title(f"Per-Class F1 Score ({model_type.upper()} / {split_type} split)")
    plt.colorbar(im, ax=ax, label="F1 Score")
    plt.tight_layout()
    path = os.path.join(output_dir, f"per_class_{model_type}_{split_type}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_cohens_d(recovery_rows, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)
    for model in ["1d", "2d"]:
        rows = [r for r in recovery_rows if r["model"] == model and r["cohens_d"] is not None]
        if not rows:
            continue
        rows.sort(key=lambda r: r["cohens_d"], reverse=True)
        augs = [r["augmentation"] for r in rows]
        d_vals = [r["cohens_d"] for r in rows]
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["green" if d > 0 else "red" for d in d_vals]
        ax.barh(augs, d_vals, color=colors)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.axvline(x=0.2, color="gray", linestyle=":", linewidth=0.8, alpha=0.5, label="d=0.2 (small)")
        ax.axvline(x=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="d=0.5 (medium)")
        ax.axvline(x=0.8, color="gray", linestyle="-", linewidth=0.8, alpha=0.5, label="d=0.8 (large)")
        ax.set_xlabel("Cohen's d (random vs recording gap)")
        ax.set_title(f"Effect Size by Augmentation ({model.upper()} Model)")
        ax.legend(fontsize=8)
        plt.tight_layout()
        path = os.path.join(output_dir, f"cohens_d_{model}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {path}")


def plot_confusion_matrices(results, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)
    selections = []
    for model in ("1d", "2d"):
        candidates = [
            row
            for row in results
            if row["model"] == model
            and row["split"] == "recording"
            and row["augmentation"] != "none"
        ]
        best = max(candidates, key=lambda row: row["accuracy_mean"])
        selections.extend(
            [
                (model, "none", f"{model.upper()} / no augmentation"),
                (
                    model,
                    best["augmentation"],
                    f"{model.upper()} / best: {best['augmentation']}",
                ),
            ]
        )
    class_names = ["Normal", "IR", "OR", "B"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    plotted = 0
    for ax, (model, augmentation, title) in zip(axes.flat, selections):
        row = next(
            (
                item
                for item in results
                if item["model"] == model
                and item["split"] == "recording"
                and item["augmentation"] == augmentation
            ),
            None,
        )
        if row is None or "confusion_matrix_sum" not in row:
            ax.axis("off")
            continue
        matrix = np.asarray(row["confusion_matrix_sum"], dtype=float)
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix,
            row_totals,
            out=np.zeros_like(matrix),
            where=row_totals != 0,
        )
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        for row_idx in range(normalized.shape[0]):
            for col_idx in range(normalized.shape[1]):
                value = normalized[row_idx, col_idx]
                ax.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}\n(n={int(matrix[row_idx, col_idx])})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value > 0.55 else "black",
                )
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    fig.suptitle("Recording-Level Test Confusion Matrices (5 Seeds Aggregated)")
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.subplots_adjust(top=0.92, right=0.9, wspace=0.3, hspace=0.3)
    path = os.path.join(output_dir, "confusion_matrices_recording.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def main():
    output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, "tables", "main_results.json")
    if not os.path.exists(results_path):
        print("No main_results.json found. Run src/train.py first.")
        return
    with open(results_path, "r") as f:
        results = json.load(f)
    recovery_path = os.path.join(RESULTS_DIR, "tables", "gap_recovery.json")
    if os.path.exists(recovery_path):
        with open(recovery_path, "r") as f:
            recovery = json.load(f)
        plot_gap_recovery(recovery, output_dir)
        plot_cohens_d(recovery, output_dir)
    plot_random_vs_recording(results, "1d", output_dir)
    plot_random_vs_recording(results, "2d", output_dir)
    correlation_path = os.path.join(
        RESULTS_DIR,
        "tables",
        "physical_fidelity_correlation.json",
    )
    if os.path.exists(correlation_path):
        with open(correlation_path, "r") as f:
            correlation_data = json.load(f)
        plot_physical_fidelity_correlation(
            correlation_data, output_dir
        )
    plot_per_class_heatmap(results, "1d", "recording", output_dir)
    plot_per_class_heatmap(results, "2d", "recording", output_dir)
    plot_confusion_matrices(results, output_dir)
    print("All figures generated.")


if __name__ == "__main__":
    main()
