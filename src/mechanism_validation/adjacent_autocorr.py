import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from preprocess import load_and_segment, load_one_file, sliding_windows
from config import DATA_ROOT, WINDOW_LENGTH, OVERLAP_RECORDING, RESULTS_DIR


def adjacent_overlap_audit(records, offsets=None):
    if offsets is None:
        offsets = list(range(1, 11))
    stride = int(WINDOW_LENGTH * (1 - OVERLAP_RECORDING))
    results = {}
    for rec in records:
        signal = load_one_file(rec["path"])
        windows = sliding_windows(signal, WINDOW_LENGTH, OVERLAP_RECORDING)
        if len(windows) < max(offsets) + 1:
            continue
        zero_lag_corrs = []
        shared_fractions = []
        aligned_overlap_corrs = []
        aligned_overlap_mae = []
        for offset in offsets:
            # Zero-lag similarity is data dependent and need not be high even
            # when adjacent windows contain exactly duplicated samples.
            w1 = windows[:-offset].reshape(-1)
            w2 = windows[offset:].reshape(-1)
            r, _ = pearsonr(w1, w2)
            zero_lag_corrs.append(float(r))

            sample_shift = offset * stride
            shared = max(0, WINDOW_LENGTH - sample_shift)
            shared_fractions.append(shared / WINDOW_LENGTH)
            if shared:
                aligned_left = windows[:-offset, sample_shift:].reshape(-1)
                aligned_right = windows[offset:, :shared].reshape(-1)
                if np.std(aligned_left) == 0 or np.std(aligned_right) == 0:
                    aligned_r = 1.0 if np.array_equal(aligned_left, aligned_right) else 0.0
                else:
                    aligned_r, _ = pearsonr(aligned_left, aligned_right)
                aligned_overlap_corrs.append(float(aligned_r))
                aligned_overlap_mae.append(
                    float(np.mean(np.abs(aligned_left - aligned_right)))
                )
            else:
                aligned_overlap_corrs.append(None)
                aligned_overlap_mae.append(None)
        results[rec["recording_id"]] = {
            "fault_type": rec["fault_type"],
            "offsets": offsets,
            "stride_samples": stride,
            "shared_fraction": shared_fractions,
            "aligned_overlap_correlation": aligned_overlap_corrs,
            "aligned_overlap_mae": aligned_overlap_mae,
            "zero_lag_correlation": zero_lag_corrs,
        }
    return results


def plot_overlap_audit(results, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)

    colors = {"Normal": "green", "IR": "red", "OR": "blue", "B": "orange"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    example = next(iter(results.values()))
    axes[0].plot(
        example["offsets"],
        example["shared_fraction"],
        marker="o",
        color="black",
    )
    axes[0].set_xlabel("Window offset (stride count)")
    axes[0].set_ylabel("Exactly shared sample fraction")
    axes[0].set_title("Structural overlap from 50% windowing")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(True, alpha=0.3)

    for rid, data in results.items():
        lbl = data["fault_type"]
        axes[1].plot(
            data["offsets"],
            data["zero_lag_correlation"],
            color=colors.get(lbl, "gray"),
            alpha=0.35,
            linewidth=0.8,
        )
    for lbl, c in colors.items():
        axes[1].plot([], [], color=c, label=lbl)
    axes[1].set_xlabel("Window offset (stride count)")
    axes[1].set_ylabel("Zero-lag Pearson r")
    axes[1].set_title("Data-dependent same-recording similarity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle("M1: Adjacent-Window Leakage Audit")
    fig.tight_layout()
    path = os.path.join(output_dir, "m1_adjacent_autocorrelation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")

    summary = {
        "stride_samples": example["stride_samples"],
        "adjacent_shared_fraction": example["shared_fraction"][0],
        "adjacent_aligned_overlap_correlation_mean": float(
            np.mean(
                [
                    data["aligned_overlap_correlation"][0]
                    for data in results.values()
                ]
            )
        ),
        "adjacent_aligned_overlap_mae_max": float(
            np.max(
                [
                    data["aligned_overlap_mae"][0]
                    for data in results.values()
                ]
            )
        ),
        "zero_lag_correlation_by_class": {},
    }
    for lbl in ["Normal", "IR", "OR", "B"]:
        all_corrs = []
        for data in results.values():
            if data["fault_type"] == lbl:
                all_corrs.append(data["zero_lag_correlation"])
        if all_corrs:
            mean_corr = np.mean(all_corrs, axis=0)
            summary["zero_lag_correlation_by_class"][lbl] = dict(
                zip(map(str, example["offsets"]), mean_corr.tolist())
            )
    return summary


def main():
    import json

    records = load_and_segment(DATA_ROOT)
    results = adjacent_overlap_audit(records)
    summary = plot_overlap_audit(results)
    output_path = os.path.join(
        RESULTS_DIR, "tables", "adjacent_overlap_audit.json"
    )
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"summary": summary, "recordings": results},
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(
        "Adjacent windows: "
        f"{summary['adjacent_shared_fraction']:.0%} exact sample overlap, "
        f"aligned r={summary['adjacent_aligned_overlap_correlation_mean']:.4f}, "
        f"max MAE={summary['adjacent_aligned_overlap_mae_max']:.3g}"
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
