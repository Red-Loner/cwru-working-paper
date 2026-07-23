import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from preprocess import load_and_segment, load_one_file, sliding_windows
from augmentation import apply_augmentation, AUGMENTATION_CONDITIONS
from config import DATA_ROOT, WINDOW_LENGTH, OVERLAP_RECORDING, RESULTS_DIR, SAMPLE_RATE

BEARING_PARAMS_6205 = {
    "bpfo": 3.5848,
    "bpfi": 5.4152,
    "bsf": 2.3178,
    "ftf": 0.3983,
}


def compute_fault_frequencies(shaft_speed_rpm, params=None):
    if params is None:
        params = BEARING_PARAMS_6205
    shaft_hz = shaft_speed_rpm / 60.0
    return {
        "BPFO": params["bpfo"] * shaft_hz,
        "BPFI": params["bpfi"] * shaft_hz,
        "BSF": params["bsf"] * shaft_hz,
        "FTF": params["ftf"] * shaft_hz,
    }


def compute_spectrum(signal, fs=SAMPLE_RATE):
    n = len(signal)
    spec = np.abs(np.fft.rfft(signal)) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, spec


def compute_envelope_spectrum(signal, fs=SAMPLE_RATE):
    analytic = hilbert(signal)
    envelope = np.abs(analytic)
    envelope -= np.mean(envelope)
    return compute_spectrum(envelope, fs)


def symmetric_energy_fidelity(ratio):
    if ratio <= 0 or not np.isfinite(ratio):
        return 0.0
    return float(min(ratio, 1.0 / ratio))


def fault_band_energy(signal_original, signal_augmented, fault_type, shaft_speed_rpm,
                      fs=SAMPLE_RATE, tolerance_pct=5):
    # Bearing characteristic frequencies are modulation frequencies, so the
    # envelope spectrum is the physically appropriate representation.
    freqs, spec_orig = compute_envelope_spectrum(signal_original, fs)
    _, spec_aug = compute_envelope_spectrum(signal_augmented, fs)
    fault_freqs = compute_fault_frequencies(shaft_speed_rpm)
    fault_key = {"IR": "BPFI", "OR": "BPFO", "B": "BSF"}.get(fault_type, "BPFO")
    target = fault_freqs[fault_key]
    mask = np.zeros_like(freqs, dtype=bool)
    for harmonic in (1, 2, 3):
        center = target * harmonic
        tolerance = center * tolerance_pct / 100.0
        mask |= (freqs >= center - tolerance) & (freqs <= center + tolerance)
    original_energy = float(np.sum(spec_orig[mask] ** 2))
    augmented_energy = float(np.sum(spec_aug[mask] ** 2))
    ratio = augmented_energy / original_energy if original_energy > 0 else 1.0
    return {
        "fault_frequency": fault_key,
        "raw_energy_ratio": float(ratio),
        "symmetric_fidelity": symmetric_energy_fidelity(ratio),
    }


def shifted_band_energy(signal_original, signal_augmented, fault_type, shaft_speed_rpm,
                         shift_hz=100, fs=SAMPLE_RATE, tolerance_pct=5):
    freqs, spec_orig = compute_envelope_spectrum(signal_original, fs)
    _, spec_aug = compute_envelope_spectrum(signal_augmented, fs)
    fault_freqs = compute_fault_frequencies(shaft_speed_rpm)
    fault_key = {"IR": "BPFI", "OR": "BPFO", "B": "BSF"}.get(fault_type, "BPFO")
    target = fault_freqs.get(fault_key, fault_freqs["BPFO"]) + shift_hz
    mask = np.zeros_like(freqs, dtype=bool)
    for harmonic in (1, 2, 3):
        center = target * harmonic
        tolerance = center * tolerance_pct / 100.0
        mask |= (freqs >= center - tolerance) & (freqs <= center + tolerance)
    e_orig = np.sum(spec_orig[mask] ** 2)
    e_aug = np.sum(spec_aug[mask] ** 2)
    if e_orig > 0:
        return symmetric_energy_fidelity(float(e_aug / e_orig))
    return 1.0


def audit_all_augmentations(records, shaft_speed_map=None, n_samples=50):
    if shaft_speed_map is None:
        shaft_speed_map = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
    results = {}
    for aug_type in AUGMENTATION_CONDITIONS:
        results[aug_type] = {
            "raw_ratios": [],
            "fidelity": [],
            "shifted_fidelity": [],
        }
    for rec in records:
        if rec["fault_type"] == "Normal":
            continue
        signal = load_one_file(rec["path"])
        windows = sliding_windows(signal, WINDOW_LENGTH, OVERLAP_RECORDING)
        speed = shaft_speed_map.get(rec["load"], 1797)
        n = min(n_samples, len(windows))
        indices = np.random.choice(len(windows), n, replace=False)
        for idx in indices:
            orig = windows[idx].astype(np.float32)
            orig_std = float(np.std(orig))
            if orig_std > 0:
                orig = (orig - float(np.mean(orig))) / orig_std
            for aug_type in AUGMENTATION_CONDITIONS:
                if aug_type == "none":
                    aug_sig = orig
                else:
                    aug_sig = apply_augmentation(orig[np.newaxis, :], aug_type)[0]
                energy = fault_band_energy(
                    orig, aug_sig, rec["fault_type"], speed
                )
                results[aug_type]["raw_ratios"].append(
                    energy["raw_energy_ratio"]
                )
                results[aug_type]["fidelity"].append(
                    energy["symmetric_fidelity"]
                )
                shifted_fidelity = shifted_band_energy(
                    orig, aug_sig, rec["fault_type"], speed
                )
                results[aug_type]["shifted_fidelity"].append(
                    shifted_fidelity
                )
    summary = {}
    for aug_type, data in results.items():
        if data["fidelity"]:
            raw_ratios = np.asarray(data["raw_ratios"])
            summary[aug_type] = {
                "metric": "min(E_aug/E_orig, E_orig/E_aug) on envelope-spectrum fault bands",
                "energy_retention_mean": float(np.mean(data["fidelity"])),
                "energy_retention_std": float(np.std(data["fidelity"])),
                "raw_energy_ratio_mean": float(np.mean(raw_ratios)),
                "raw_energy_ratio_median": float(np.median(raw_ratios)),
                "shifted_band_mean": float(
                    np.mean(data["shifted_fidelity"])
                ),
            }
    return summary


def plot_fault_band_energy(audit_summary, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(output_dir, exist_ok=True)
    augs = list(audit_summary.keys())
    ratios = [audit_summary[a]["energy_retention_mean"] for a in augs]
    shifted = [audit_summary[a]["shifted_band_mean"] for a in augs]
    x = np.arange(len(augs))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 5))
    bars1 = ax.bar(x - width / 2, ratios, width, label="Fault-band fidelity", color="steelblue")
    bars2 = ax.bar(x + width / 2, shifted, width, label="Shifted-band fidelity (+100Hz)", color="coral")
    ax.axhline(y=0.95, color="green", linestyle="--", linewidth=1, alpha=0.7, label="95% threshold")
    ax.axhline(y=0.80, color="red", linestyle="--", linewidth=1, alpha=0.7, label="80% threshold")
    ax.set_xlabel("Augmentation Type")
    ax.set_ylabel("Symmetric energy fidelity (0–1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("M5: Envelope-Spectrum Fault-Band Fidelity")
    ax.set_xticks(x)
    ax.set_xticklabels(augs, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "m5_fault_band_energy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")
    return audit_summary


def main():
    import json
    records = load_and_segment(DATA_ROOT)
    print(f"Loaded {len(records)} recordings for energy audit")
    fault_records = [r for r in records if r["fault_type"] != "Normal"]
    print(f"  Fault recordings: {len(fault_records)}")
    np.random.seed(42)
    summary = audit_all_augmentations(fault_records, n_samples=30)
    plot_fault_band_energy(summary)
    output_dir = os.path.join(RESULTS_DIR, "tables")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "energy_audit.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    for aug in sorted(summary.keys(), key=lambda k: summary[k]["energy_retention_mean"], reverse=True):
        r = summary[aug]["energy_retention_mean"]
        status = "PHYSICALLY SAFE" if r >= 0.95 else ("DESTRUCTIVE" if r < 0.80 else "MARGINAL")
        print(f"  {aug:15s}  R_energy={r:.4f}  [{status}]")


if __name__ == "__main__":
    main()
