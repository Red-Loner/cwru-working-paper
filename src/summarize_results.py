"""Generate one canonical numerical summary and standalone LaTeX tables."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"

AUGMENTATIONS = (
    "none",
    "noise_001",
    "noise_005",
    "noise_01",
    "shift_5",
    "shift_20",
    "shift_50",
    "specaugment",
    "combined",
    "freq_flip",
)
DISPLAY = {
    "none": "None",
    "noise_001": "Noise 0.01",
    "noise_005": "Noise 0.05",
    "noise_01": "Noise 0.10",
    "shift_5": "Shift 5",
    "shift_20": "Shift 20",
    "shift_50": "Shift 50",
    "specaugment": "Spectral/time mask",
    "combined": "Combined",
    "freq_flip": "FFT magnitude reversal",
}


def load(name):
    with (TABLES / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main_lookup(rows):
    return {
        (row["model"], row["split"], row["augmentation"]): row
        for row in rows
    }


def grouping_summary(rows):
    summary = {}
    for model in ("1d", "2d"):
        summary[model] = {}
        for split in ("load", "recording", "fault_size"):
            values = [
                row["accuracy"]
                for row in rows
                if row["model"] == model and row["split"] == split
            ]
            summary[model][split] = {
                "mean": round(statistics.fmean(values), 4),
                "std": round(statistics.pstdev(values), 4),
                "per_seed": values,
            }
    return summary


def build_summary(main_rows, recovery_rows, grouping_rows, convergence, physical):
    lookup = main_lookup(main_rows)
    recovery_lookup = {
        (row["model"], row["augmentation"]): row for row in recovery_rows
    }
    models = {}
    for model in ("1d", "2d"):
        random_baseline = lookup[(model, "random", "none")]
        recording_baseline = lookup[(model, "recording", "none")]
        augmentation_rows = [
            lookup[(model, "recording", augmentation)]
            for augmentation in AUGMENTATIONS
            if augmentation != "none"
        ]
        best = max(augmentation_rows, key=lambda row: row["accuracy_mean"])
        improved = [
            row["augmentation"]
            for row in augmentation_rows
            if row["accuracy_mean"] > recording_baseline["accuracy_mean"]
        ]
        degraded = [
            row["augmentation"]
            for row in augmentation_rows
            if row["accuracy_mean"] < recording_baseline["accuracy_mean"]
        ]
        best_recovery = recovery_lookup[(model, best["augmentation"])]
        recording_seed_values = recording_baseline["per_seed_accuracy"]
        first_three = recording_seed_values[:3]
        models[model] = {
            "random_baseline": {
                "accuracy_mean": random_baseline["accuracy_mean"],
                "accuracy_std": random_baseline["accuracy_std"],
            },
            "recording_baseline": {
                "accuracy_mean": recording_baseline["accuracy_mean"],
                "accuracy_std": recording_baseline["accuracy_std"],
                "per_seed_accuracy": recording_baseline["per_seed_accuracy"],
            },
            "baseline_protocol_gap": round(
                random_baseline["accuracy_mean"]
                - recording_baseline["accuracy_mean"],
                4,
            ),
            "baseline_cohens_d": recovery_lookup[(model, "none")]["cohens_d"],
            "best_recording_augmentation": best["augmentation"],
            "best_recording_accuracy": best["accuracy_mean"],
            "best_delta_recovery": best_recovery["delta_recovery"],
            "augmentations_improved": improved,
            "augmentations_degraded": degraded,
            "recording_baseline_seed_diagnostic": {
                "first_three_mean": round(statistics.fmean(first_three), 4),
                "first_three_std": round(
                    statistics.pstdev(first_three), 4
                ),
                "all_five_mean": recording_baseline["accuracy_mean"],
                "all_five_std": recording_baseline["accuracy_std"],
                "absolute_mean_change_after_seeds_4_and_5": round(
                    abs(
                        statistics.fmean(first_three)
                        - recording_baseline["accuracy_mean"]
                    ),
                    4,
                ),
                "all_five_range": [
                    min(recording_seed_values),
                    max(recording_seed_values),
                ],
            },
        }

    best_epochs = [
        row["best_validation_epoch"] for row in convergence.values()
    ]
    convergence_summary = {
        "run_count": len(best_epochs),
        "best_validation_epoch_mean": round(statistics.fmean(best_epochs), 2),
        "best_validation_epoch_median": statistics.median(best_epochs),
        "best_validation_epoch_max": max(best_epochs),
        "best_validation_epoch_45_to_50": sum(
            epoch >= 45 for epoch in best_epochs
        ),
    }
    return {
        "main_run_count": sum(
            len(row["per_seed_results"]) for row in main_rows
        ),
        "configuration_count": len(main_rows),
        "models": models,
        "grouping": {
            "results": grouping_summary(grouping_rows),
            "assignment_design": {
                "load_group_counts_train_validation_test": [2, 1, 1],
                "fault_size_group_counts_train_validation_test": [1, 1, 1],
                "normal_recording_counts_in_fault_size_split": [2, 1, 1],
                "recording_strategy_recording_counts": [23, 7, 10],
                "load_strategy_recording_counts": [20, 10, 10],
                "fault_size_strategy_recording_counts": [14, 13, 13],
                "interpretation": (
                    "Descriptive stress tests only: group counts and training "
                    "coverage differ, so the accuracy ordering is not causal."
                ),
            },
        },
        "best_validation_epoch_diagnostic": convergence_summary,
        "physical_fidelity_correlation": physical,
    }


def metric_cell(row):
    return f"{row['accuracy_mean']:.4f} $\\pm$ {row['accuracy_std']:.4f}"


def macro_f1_cell(row):
    return (
        f"{row['macro_f1_mean']:.4f} $\\pm$ "
        f"{row['macro_f1_std']:.4f}"
    )


def write_latex_tables(main_rows, recovery_rows, grouping_rows):
    lookup = main_lookup(main_rows)
    recovery = {
        (row["model"], row["augmentation"]): row for row in recovery_rows
    }
    lines = [
        "% AUTO-GENERATED by src/summarize_results.py; do not hand-edit.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Main results: accuracy (mean $\pm$ std) across 5 seeds.}",
        r"\label{tab:main}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Augmentation & 1D Random & 1D Recording & 2D Random & 2D Recording \\",
        r"\midrule",
    ]
    for augmentation in AUGMENTATIONS:
        cells = [
            DISPLAY[augmentation],
            metric_cell(lookup[("1d", "random", augmentation)]),
            metric_cell(lookup[("1d", "recording", augmentation)]),
            metric_cell(lookup[("2d", "random", augmentation)]),
            metric_cell(lookup[("2d", "recording", augmentation)]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Macro-F1 (mean $\pm$ std) across 5 seeds for all factorial configurations.}",
            r"\label{tab:macro-f1}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Augmentation & 1D Random & 1D Recording & 2D Random & 2D Recording \\",
            r"\midrule",
        ]
    )
    for augmentation in AUGMENTATIONS:
        cells = [
            DISPLAY[augmentation],
            macro_f1_cell(lookup[("1d", "random", augmentation)]),
            macro_f1_cell(lookup[("1d", "recording", augmentation)]),
            macro_f1_cell(lookup[("2d", "random", augmentation)]),
            macro_f1_cell(lookup[("2d", "recording", augmentation)]),
        ]
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Gap recovery relative to the no-augmentation random-to-recording protocol gap.}",
            r"\label{tab:gap}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            r"Augmentation & Gap (1D) & $\Delta_{\rm rec}$ (1D) & $d$ (1D) & Gap (2D) & $\Delta_{\rm rec}$ (2D) & $d$ (2D) \\",
            r"\midrule",
        ]
    )
    for augmentation in AUGMENTATIONS:
        one_d = recovery[("1d", augmentation)]
        two_d = recovery[("2d", augmentation)]
        lines.append(
            f"{DISPLAY[augmentation]} & "
            f"{one_d['leakage_gap']:.4f} & {one_d['delta_recovery']:+.4f} & {one_d['cohens_d']:.4f} & "
            f"{two_d['leakage_gap']:.4f} & {two_d['delta_recovery']:+.4f} & {two_d['cohens_d']:.4f} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Per-class F1 under recording-level splitting for the 2D-CNN (mean $\pm$ std).}",
            r"\label{tab:perclass}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Augmentation & Normal & IR & OR & Ball \\",
            r"\midrule",
        ]
    )
    for augmentation in AUGMENTATIONS:
        row = lookup[("2d", "recording", augmentation)]
        cells = [DISPLAY[augmentation]]
        for class_name in ("Normal", "IR", "OR", "B"):
            metrics = row["per_class"][class_name]
            cells.append(
                f"{metrics['f1_mean']:.3f} $\\pm$ {metrics['f1_std']:.3f}"
            )
        lines.append(" & ".join(cells) + r" \\")
    group = grouping_summary(grouping_rows)
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Recording-level per-class precision, recall, and F1 (mean $\pm$ std across 5 seeds) for the baseline, best augmentation, and destructive control.}",
            r"\label{tab:perclass-prf}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lllcccc}",
            r"\toprule",
            r"Model & Augmentation & Metric & Normal & IR & OR & Ball \\",
            r"\midrule",
        ]
    )
    selected_augmentations = ("none", "shift_5", "freq_flip")
    metric_specs = (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
    )
    for model, model_label in (("1d", "CNN--1D"), ("2d", "CNN--2D")):
        for augmentation in selected_augmentations:
            row = lookup[(model, "recording", augmentation)]
            for metric_index, (metric_key, metric_label) in enumerate(
                metric_specs
            ):
                cells = [
                    model_label if augmentation == "none" and metric_index == 0
                    else "",
                    DISPLAY[augmentation] if metric_index == 0 else "",
                    metric_label,
                ]
                for class_name in ("Normal", "IR", "OR", "B"):
                    metrics = row["per_class"][class_name]
                    cells.append(
                        f"{metrics[f'{metric_key}_mean']:.3f} $\\pm$ "
                        f"{metrics[f'{metric_key}_std']:.3f}"
                    )
                lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\addlinespace")
        if model == "1d":
            lines.append(r"\midrule")
    if lines[-1] == r"\addlinespace":
        lines.pop()
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
            "",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Descriptive grouping stress-test accuracy (mean $\pm$ std across 5 seeds). Load and fault-size strategies use unequal group counts and are not causally comparable.}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Model & By Load (2/1/1) & By Recording & By Fault Size (1/1/1) \\",
            r"\midrule",
        ]
    )
    for model, label in (("1d", "CNN--1D"), ("2d", "CNN--2D")):
        cells = [label]
        for split in ("load", "recording", "fault_size"):
            cells.append(
                f"{group[model][split]['mean']:.4f} $\\pm$ "
                f"{group[model][split]['std']:.4f}"
            )
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    content = "\n".join(lines)
    (ROOT / "paper_tables.tex").write_text(content, encoding="utf-8")

    fragments = content.strip().split("\n\n")
    fragment_dir = ROOT / "paper" / "generated_tables"
    fragment_dir.mkdir(parents=True, exist_ok=True)
    fragment_names = (
        "main_results.tex",
        "macro_f1.tex",
        "gap_recovery.tex",
        "per_class.tex",
        "per_class_prf.tex",
        "grouping.tex",
    )
    if len(fragments) != len(fragment_names):
        raise RuntimeError(
            f"Expected {len(fragment_names)} LaTeX table fragments, "
            f"found {len(fragments)}"
        )
    for name, fragment in zip(fragment_names, fragments):
        if not fragment.startswith("% AUTO-GENERATED"):
            fragment = (
                "% AUTO-GENERATED by src/summarize_results.py; "
                "do not hand-edit.\n" + fragment
            )
        (fragment_dir / name).write_text(
            fragment + "\n", encoding="utf-8"
        )


def main():
    main_rows = load("main_results.json")
    recovery_rows = load("gap_recovery.json")
    grouping_rows = load("grouping_ablation.json")
    convergence = load("convergence_summary.json")
    physical = load("physical_fidelity_correlation.json")

    summary = build_summary(
        main_rows,
        recovery_rows,
        grouping_rows,
        convergence,
        physical,
    )
    with (TABLES / "release_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_latex_tables(main_rows, recovery_rows, grouping_rows)
    print(f"Saved {TABLES / 'release_summary.json'}")
    print(f"Saved {ROOT / 'paper_tables.tex'}")


if __name__ == "__main__":
    main()
