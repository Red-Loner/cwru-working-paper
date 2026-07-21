import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from preprocess import load_and_segment, load_one_file, sliding_windows
from config import DATA_ROOT, WINDOW_LENGTH, OVERLAP_RECORDING, RESULTS_DIR


def adjacent_autocorrelation(records, offsets=None):
    if offsets is None:
        offsets = list(range(1, 11))
    results = {}
    for rec in records:
        signal = load_one_file(rec["path"])
        windows = sliding_windows(signal, WINDOW_LENGTH, OVERLAP_RECORDING)
        if len(windows) < max(offsets) + 1:
            continue
        corrs = []
        for offset in offsets:
            w1 = windows[:-offset].reshape(-1)
            w2 = windows[offset:].reshape(-1)
            r, _ = pearsonr(w1, w2)
            corrs.append(r)
        results[rec["recording_id"]] = {"fault_type": rec["fault_type"],
                                          "offsets": offsets, "correlations": corrs}
    return results


def plot_autocorr(results, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)

    colors = {"Normal": "green", "IR": "red", "OR": "blue", "B": "orange"}
    plt.figure(figsize=(10, 6))
    for rid, data in results.items():
        lbl = data["fault_type"]
        plt.plot(data["offsets"], data["correlations"], color=colors.get(lbl, "gray"),
                 alpha=0.4, linewidth=0.8)
    for lbl, c in colors.items():
        plt.plot([], [], color=c, label=lbl)
    plt.axhline(y=0.7, color="red", linestyle="--", linewidth=1, alpha=0.5)
    plt.xlabel("Window Offset (stride count)")
    plt.ylabel("Pearson r")
    plt.title("M1: Adjacent-Window Autocorrelation by Fault Type")
    plt.legend()
    plt.grid(True, alpha=0.3)
    path = os.path.join(output_dir, "m1_adjacent_autocorrelation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")

    offset_means = {}
    for lbl in ["Normal", "IR", "OR", "B"]:
        all_corrs = []
        for rid, data in results.items():
            if data["fault_type"] == lbl:
                all_corrs.append(data["correlations"])
        if all_corrs:
            mean_corr = np.mean(all_corrs, axis=0)
            offset_means[lbl] = dict(zip(results[list(results.keys())[0]]["offsets"], mean_corr.tolist()))
    return offset_means


def main():
    records = load_and_segment(DATA_ROOT)
    results = adjacent_autocorrelation(records)
    means = plot_autocorr(results)
    for lbl, corrs in means.items():
        print(f"{lbl}: offset=1 r={corrs.get(1, 0):.4f}, offset=5 r={corrs.get(5, 0):.4f}, offset=10 r={corrs.get(10, 0):.4f}")


if __name__ == "__main__":
    main()
