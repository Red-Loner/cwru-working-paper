import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import RESULTS_DIR


def analyze_convergence():
    log_dir = os.path.join(RESULTS_DIR, "logs", "epoch_logs")
    if not os.path.exists(log_dir):
        print("No epoch logs found")
        return

    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
    if not csv_files:
        print("No epoch logs found")
        return

    summary = {}
    for fp in csv_files:
        basename = os.path.basename(fp).replace(".csv", "")
        try:
            data = np.loadtxt(fp, delimiter=",", skiprows=1)
        except Exception:
            continue
        if data.ndim == 1:
            continue
        epochs = data[:, 0]
        val_acc = data[:, 3]
        last10 = val_acc[-10:].mean()
        best = val_acc.max()
        converged_epoch = np.argmax(val_acc) + 1
        summary[basename] = {
            "best_val_acc": round(float(best), 4),
            "last10_val_acc": round(float(last10), 4),
            "converged_at_epoch": int(converged_epoch),
            "final_epoch": int(epochs[-1]),
        }

    conv_epochs = [v["converged_at_epoch"] for v in summary.values()]
    if conv_epochs:
        print(f"Total runs: {len(conv_epochs)}")
        print(f"Convergence epoch: mean={np.mean(conv_epochs):.1f}, "
              f"median={np.median(conv_epochs):.1f}, max={max(conv_epochs)}")
        late_conv = sum(1 for e in conv_epochs if e >= 45)
        print(f"Runs converging in last 10% of epochs (>=45): {late_conv}/{len(conv_epochs)}")
        not_conv = sum(1 for v in summary.values()
                       if v["last10_val_acc"] < v["best_val_acc"] - 0.01)
        print(f"Runs with degrading last-10-epoch val_acc: {not_conv}/{len(conv_epochs)}")

    parts = ["1d_random", "1d_recording", "2d_random", "2d_recording"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, prefix in enumerate(parts):
        ax = axes[idx // 2][idx % 2]
        matching = [f for f in csv_files if f.replace("\\", "/").split("/")[-1].startswith(prefix)]
        for fp in matching[:15]:
            try:
                data = np.loadtxt(fp, delimiter=",", skiprows=1)
                label = os.path.basename(fp).replace(".csv", "").replace(prefix + "_", "")
                ax.plot(data[:, 0], data[:, 3], alpha=0.5, linewidth=0.8, label=label[:20])
            except Exception:
                continue
        ax.set_title(prefix.replace("_", " ").upper())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Accuracy")
        ax.legend(fontsize=6, loc="lower right")
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "figures", "convergence_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    with open(os.path.join(RESULTS_DIR, "tables", "convergence_summary.json"), "w") as f:
        import json
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Convergence summary saved")


if __name__ == "__main__":
    analyze_convergence()
