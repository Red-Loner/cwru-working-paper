import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from preprocess import load_and_segment, load_one_file
from config import DATA_ROOT, SAMPLE_RATE, RESULTS_DIR

BEARING_PARAMS_6205 = {"bpfo": 3.5848, "bpfi": 5.4152, "bsf": 2.3178, "ftf": 0.3983}


def compute_fault_frequencies(shaft_speed_rpm, params=None):
    if params is None:
        params = BEARING_PARAMS_6205
    shaft_hz = shaft_speed_rpm / 60.0
    return {"BPFO": params["bpfo"] * shaft_hz, "BPFI": params["bpfi"] * shaft_hz,
            "BSF": params["bsf"] * shaft_hz}


def compute_envelope_spectrum(signal, fs=SAMPLE_RATE):
    analytic = hilbert(signal.astype(np.float64))
    envelope = np.abs(analytic)
    envelope -= np.mean(envelope)
    n = len(envelope)
    spec = np.abs(np.fft.rfft(envelope)) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spec


def detect_fault_peaks(freqs, env_spec, fault_type, shaft_speed_rpm, max_freq=500):
    fault_freqs = compute_fault_frequencies(shaft_speed_rpm)
    target_map = {"IR": "BPFI", "OR": "BPFO", "B": "BSF"}
    target_name = target_map.get(fault_type, "BPFO")
    target = fault_freqs[target_name]
    peaks = []
    for h in range(1, 6):
        hz = target * h
        if hz > max_freq:
            break
        idx = np.argmin(np.abs(freqs - hz))
        lo = max(0, idx - 5)
        hi = min(len(freqs), idx + 5)
        peak_val = np.max(env_spec[lo:hi])
        peak_idx = lo + np.argmax(env_spec[lo:hi])
        peaks.append((freqs[peak_idx], peak_val))
    return target_name, target, peaks


def plot_envelope_example(recordings, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)
    shaft_speed_map = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
    examples = {}
    for rec in recordings:
        ft = rec["fault_type"]
        if ft not in examples:
            examples[ft] = rec
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    plot_order = ["Normal", "IR", "OR", "B"]
    for i, ft in enumerate(plot_order):
        ax = axes[i]
        rec = examples.get(ft)
        if rec is None:
            ax.set_title(f"{ft} (no data)")
            continue
        signal = load_one_file(rec["path"])
        speed = shaft_speed_map.get(rec["load"], 1797)
        freqs, env_spec = compute_envelope_spectrum(signal[:200000])
        ax.semilogy(freqs, env_spec, color="black", linewidth=0.8)
        ax.set_xlim(0, 500)
        if ft != "Normal":
            target_name, target, peaks = detect_fault_peaks(freqs, env_spec, ft, speed)
            for pf, pv in peaks:
                ax.axvline(x=pf, color="red", linestyle="--", linewidth=0.6, alpha=0.7)
            ax.set_title(f"{ft} ({rec['load']} HP, {rec['diameter']}\")")
        else:
            ax.set_title(f"Normal ({rec['load']} HP)")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (log)")
        ax.grid(True, alpha=0.3)
    plt.suptitle("Envelope Spectra by Fault Type", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "envelope_spectra_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def main():
    records = load_and_segment(DATA_ROOT)
    print(f"Loaded {len(records)} recordings")
    plot_envelope_example(records)


if __name__ == "__main__":
    main()
