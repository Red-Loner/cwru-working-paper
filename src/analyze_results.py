import os
import json
import numpy as np
from scipy.stats import spearmanr
from config import RESULTS_DIR


def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    pooled_std = max(pooled_std, 1e-8)
    return (np.mean(x) - np.mean(y)) / pooled_std


def compute_gap_recovery(results):
    splits = {}
    for row in results:
        key = (row["model"], row["augmentation"])
        if key not in splits:
            splits[key] = {}
        splits[key][row["split"]] = row["accuracy_mean"]
    rows = []
    for (model, aug), acc in splits.items():
        if "random" not in acc or "recording" not in acc:
            continue
        acc_random = acc["random"]
        acc_recording = acc["recording"]
        gap = acc_random - acc_recording
        noaug_key = (model, "none")
        noaug_gap = None
        noaug_recording = 0
        if noaug_key in splits:
            noaug_random = splits[noaug_key].get("random", 0)
            noaug_recording = splits[noaug_key].get("recording", 0)
            noaug_gap = noaug_random - noaug_recording
        if noaug_gap and noaug_gap > 0:
            delta_recovery = (acc_recording - noaug_recording) / noaug_gap
        else:
            delta_recovery = 0.0

        random_seeds = []
        recording_seeds = []
        for r in results:
            if r["model"] == model and r["augmentation"] == aug:
                if r["split"] == "random":
                    random_seeds = r.get("per_seed_accuracy", [])
                elif r["split"] == "recording":
                    recording_seeds = r.get("per_seed_accuracy", [])

        effect_size = None
        if len(random_seeds) >= 5 and len(recording_seeds) >= 5:
            effect_size = round(cohens_d(random_seeds, recording_seeds), 4)

        rows.append({
            "model": model,
            "augmentation": aug,
            "acc_random": round(acc_random, 4),
            "acc_recording": round(acc_recording, 4),
            "leakage_gap": round(gap, 4),
            "delta_recovery": round(delta_recovery, 4),
            "cohens_d": effect_size,
        })
    return rows


def compute_h4_correlation(gap_recovery_rows, energy_audit):
    delta_recovery_map = {}
    for row in gap_recovery_rows:
        if row["augmentation"] != "none":
            delta_recovery_map[row["augmentation"]] = row["delta_recovery"]
    augs = []
    deltas = []
    energies = []
    for aug in delta_recovery_map:
        if aug in energy_audit:
            augs.append(aug)
            deltas.append(delta_recovery_map[aug])
            energies.append(energy_audit[aug]["energy_retention_mean"])
    if len(augs) >= 4:
        rho, p_value = spearmanr(energies, deltas)
    else:
        rho, p_value = float("nan"), float("nan")
    return {
        "augmentations": augs,
        "energy_retention": energies,
        "delta_recovery": deltas,
        "spearman_rho": round(float(rho), 4) if not np.isnan(rho) else None,
        "p_value": round(float(p_value), 4) if not np.isnan(p_value) else None,
        "h4_supported": bool(rho > 0.5 and p_value < 0.05) if not bool(np.isnan(rho)) else None,
    }


def compute_falsification(results):
    noaug_random_acc = None
    noaug_recording_acc = None
    for row in results:
        if row["model"] == "2d" and row["augmentation"] == "none":
            if row["split"] == "random":
                noaug_random_acc = row["accuracy_mean"]
            elif row["split"] == "recording":
                noaug_recording_acc = row["accuracy_mean"]
    if noaug_random_acc is None or noaug_recording_acc is None:
        return {"error": "Missing baseline results"}
    gap = noaug_random_acc - noaug_recording_acc
    falsification = {}
    if gap < 0.02:
        falsification["H2_null"] = "gap too small — leakage effect negligible"
    else:
        falsification["H2_null"] = f"leakage gap = {gap:.4f} — test H1/H2"
    recording_by_aug = {}
    for row in results:
        if row["model"] == "2d" and row["split"] == "recording":
            recording_by_aug[row["augmentation"]] = row["accuracy_mean"]
    recovery_values = []
    for aug, acc in recording_by_aug.items():
        if aug == "none":
            continue
        delta = (acc - noaug_recording_acc) / gap if gap > 0 else 0
        recovery_values.append(delta)
    if recovery_values:
        max_recovery = max(recovery_values)
        falsification["H1_result"] = f"max_delta_recovery = {max_recovery:.4f}"
        if max_recovery < 0.02:
            falsification["H1"] = "FALSIFIED: no meaningful recovery"
        elif max_recovery > 0.9:
            falsification["H2"] = "FALSIFIED: augmentation fully compensates"
        else:
            falsification["H1"] = "SUPPORTED: partial recovery"
            falsification["H2"] = "SUPPORTED: leakage dominates"
    else:
        falsification["H1"] = "UNKNOWN"
        falsification["H2"] = "UNKNOWN"

    falsification["effect_sizes"] = {}
    for model in ["1d", "2d"]:
        random_seeds = []
        recording_seeds = []
        for row in results:
            if row["model"] == model and row["augmentation"] == "none":
                if row["split"] == "random":
                    random_seeds = row.get("per_seed_accuracy", [])
                elif row["split"] == "recording":
                    recording_seeds = row.get("per_seed_accuracy", [])
        if len(random_seeds) >= 3 and len(recording_seeds) >= 3:
            falsification["effect_sizes"][model] = {
                "cohens_d_random_vs_recording": round(cohens_d(random_seeds, recording_seeds), 4),
                "random_seeds": random_seeds,
                "recording_seeds": recording_seeds,
            }
    return falsification


def compute_per_class_analysis(results):
    per_class_rows = []
    for row in results:
        if "per_class" in row and row["per_class"]:
            for cls_name, metrics in row["per_class"].items():
                per_class_rows.append({
                    "model": row["model"],
                    "split": row["split"],
                    "augmentation": row["augmentation"],
                    "class": cls_name,
                    **metrics,
                })
    return per_class_rows


def main():
    results_path = os.path.join(RESULTS_DIR, "tables", "main_results.json")
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
    with open(results_path, "r") as f:
        results = json.load(f)

    recovery = compute_gap_recovery(results)
    output_path = os.path.join(RESULTS_DIR, "tables", "gap_recovery.json")
    with open(output_path, "w") as f:
        json.dump(recovery, f, indent=2, ensure_ascii=False)
    print(f"Gap-recovery table saved to {output_path}")
    for r in recovery:
        es = f" d={r['cohens_d']}" if r['cohens_d'] is not None else ""
        print(f"  [{r['model']}] {r['augmentation']:15s}  "
              f"random={r['acc_random']:.4f}  recording={r['acc_recording']:.4f}  "
              f"gap={r['leakage_gap']:.4f}  Δ={r['delta_recovery']:.4f}{es}")

    falsification = compute_falsification(results)
    fp_path = os.path.join(RESULTS_DIR, "tables", "falsification.json")
    with open(fp_path, "w") as f:
        json.dump(falsification, f, indent=2, ensure_ascii=False)
    print(f"Falsification analysis saved to {fp_path}")
    for k, v in falsification.items():
        print(f"  {k}: {v}")

    per_class = compute_per_class_analysis(results)
    pc_path = os.path.join(RESULTS_DIR, "tables", "per_class_analysis.json")
    with open(pc_path, "w") as f:
        json.dump(per_class, f, indent=2, ensure_ascii=False)
    print(f"Per-class analysis saved to {pc_path} ({len(per_class)} rows)")

    energy_path = os.path.join(RESULTS_DIR, "tables", "energy_audit.json")
    if os.path.exists(energy_path):
        with open(energy_path, "r") as f:
            energy_audit = json.load(f)
        h4 = compute_h4_correlation(recovery, energy_audit)
        h4_path = os.path.join(RESULTS_DIR, "tables", "h4_correlation.json")
        with open(h4_path, "w") as f:
            json.dump(h4, f, indent=2, ensure_ascii=False)
        print(f"H4 correlation: rho={h4['spearman_rho']}, p={h4['p_value']}, "
              f"supported={h4['h4_supported']}")


if __name__ == "__main__":
    main()
