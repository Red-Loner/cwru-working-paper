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
    for row in results:
        if row["model"] != model_type:
            continue
        if row["split"] == "random":
            random_accs[row["augmentation"]] = row["accuracy_mean"]
        else:
            recording_accs[row["augmentation"]] = row["accuracy_mean"]
    augs = sorted(set(list(random_accs.keys()) + list(recording_accs.keys())))
    x = np.arange(len(augs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    r_vals = [random_accs.get(a, 0) for a in augs]
    rc_vals = [recording_accs.get(a, 0) for a in augs]
    ax.bar(x - width / 2, r_vals, width, label="Random Split (leakage)", color="coral", alpha=0.8)
    ax.bar(x + width / 2, rc_vals, width, label="Recording-Level Split (safe)", color="steelblue", alpha=0.8)
    ax.set_xlabel("Augmentation Condition")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Random vs Recording-Level Split ({model_type.upper()} Model)")
    ax.set_xticks(x)
    ax.set_xticklabels(augs, rotation=45, ha="right", fontsize=8)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, f"random_vs_recording_{model_type}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_h4_correlation(h4_data, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)
    if not h4_data.get("augmentations"):
        return
    energies = h4_data["energy_retention"]
    deltas = h4_data["delta_recovery"]
    augs = h4_data["augmentations"]
    rho = h4_data.get("spearman_rho")
    p = h4_data.get("p_value")
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, a in enumerate(augs):
        ax.scatter(energies[i], deltas[i], s=120, alpha=0.7)
        ax.annotate(a, (energies[i], deltas[i]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    if rho is not None:
        z = np.polyfit(energies, deltas, 1)
        x_line = np.linspace(min(energies) - 0.02, max(energies) + 0.02, 50)
        y_line = np.polyval(z, x_line)
        ax.plot(x_line, y_line, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Fault-Band Energy Retention (R_energy)")
    ax.set_ylabel("Gap-Recovery Ratio (Δ_recovery)")
    title = "H4: Physical Preservation vs Gap Recovery"
    if rho is not None:
        title += f"\nSpearman ρ = {rho:.3f}, p = {p:.3f}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "h4_correlation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_confusion_summary(results, model_type="2d", split_type="recording", output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(RESULTS_DIR, "tables", "main_results.json")
    if not os.path.exists(cm_path):
        print("No results file found for confusion matrix")
        return


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
    plot_random_vs_recording(results, "1d", output_dir)
    plot_random_vs_recording(results, "2d", output_dir)
    h4_path = os.path.join(RESULTS_DIR, "tables", "h4_correlation.json")
    if os.path.exists(h4_path):
        with open(h4_path, "r") as f:
            h4_data = json.load(f)
        plot_h4_correlation(h4_data, output_dir)
    print("All figures generated.")


if __name__ == "__main__":
    main()
