"""Fail-fast verification for the submitted research package.

This script does not train models.  It checks that the generated evidence is
complete and internally consistent before the paper is compiled or released.
Run from any directory with:

    python src/verify_release.py
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"
EPOCH_LOGS = RESULTS / "logs" / "epoch_logs"
MANIFESTS = ROOT / "data_manifest"

MODELS = ("1d", "2d")
SPLITS = ("random", "recording")
AUGMENTATIONS = (
    "none",
    "noise_001",
    "noise_005",
    "noise_01",
    "shift_5",
    "shift_20",
    "shift_50",
    "freq_flip",
    "specaugment",
    "combined",
)
SEEDS = (42, 123, 456, 789, 1024)
CLASSES = ("Normal", "IR", "OR", "B")


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path):
    require(path.is_file(), f"missing JSON artifact: {path.relative_to(ROOT)}")
    require(path.stat().st_size > 0, f"empty JSON artifact: {path.relative_to(ROOT)}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_finite(value, location: str) -> None:
    if isinstance(value, dict):
        require("error" not in value, f"error field found at {location}: {value.get('error')}")
        for key, item in value.items():
            require_finite(item, f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, f"{location}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"non-finite number at {location}")


def verify_matrix(matrix, location: str) -> None:
    require(isinstance(matrix, list) and len(matrix) == len(CLASSES),
            f"{location} must have four rows")
    for row in matrix:
        require(isinstance(row, list) and len(row) == len(CLASSES),
                f"{location} must be 4 x 4")
        require(all(isinstance(value, int) and value >= 0 for value in row),
                f"{location} must contain non-negative integers")


def verify_main_results() -> None:
    rows = load_json(TABLES / "main_results.json")
    expected = {
        (model, split, augmentation)
        for model in MODELS
        for split in SPLITS
        for augmentation in AUGMENTATIONS
    }
    actual = {(row["model"], row["split"], row["augmentation"]) for row in rows}
    require(len(rows) == 40 and actual == expected,
            "main_results.json must contain exactly the 40 factorial combinations")

    for row in rows:
        key = f"{row['model']}/{row['split']}/{row['augmentation']}"
        require(row.get("per_seed_ids") == list(SEEDS),
                f"{key} has missing or misordered seeds")
        seed_rows = row.get("per_seed_results", [])
        require([item.get("seed") for item in seed_rows] == list(SEEDS),
                f"{key} must contain five complete per-seed results")
        require(len(row.get("per_seed_accuracy", [])) == 5,
                f"{key} must contain five seed accuracies")
        for index, seed_row in enumerate(seed_rows):
            verify_matrix(seed_row.get("confusion_matrix"),
                          f"{key}/seed={SEEDS[index]}/confusion_matrix")
        verify_matrix(row.get("confusion_matrix_sum"), f"{key}/confusion_matrix_sum")
    require_finite(rows, "main_results")


def verify_confusion_artifact() -> None:
    rows = load_json(TABLES / "confusion_matrices.json")
    require(len(rows) == 40, "confusion_matrices.json must contain 40 combinations")
    for row in rows:
        key = f"{row['model']}/{row['split']}/{row['augmentation']}"
        require(tuple(row.get("class_order", [])) == CLASSES,
                f"{key} has the wrong confusion-matrix class order")
        per_seed = row.get("per_seed", [])
        require([item.get("seed") for item in per_seed] == list(SEEDS),
                f"{key} must contain five per-seed confusion matrices")
        for item in per_seed:
            verify_matrix(item.get("confusion_matrix"),
                          f"{key}/seed={item.get('seed')}/confusion_matrix")
        verify_matrix(row.get("confusion_matrix_sum"), f"{key}/confusion_matrix_sum")


def verify_epoch_logs() -> None:
    expected_names = {
        f"{model}_{split}_{augmentation}_s{seed}.csv"
        for model in MODELS
        for split in SPLITS
        for augmentation in AUGMENTATIONS
        for seed in SEEDS
    }
    actual_paths = list(EPOCH_LOGS.glob("*.csv"))
    actual_names = {path.name for path in actual_paths}
    require(actual_names == expected_names,
            f"epoch logs differ from the expected 200 files "
            f"(missing={len(expected_names - actual_names)}, "
            f"unexpected={len(actual_names - expected_names)})")
    for path in actual_paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == 50, f"{path.name} must contain exactly 50 epochs")
        require(rows[0]["epoch"] == "1" and rows[-1]["epoch"] == "50",
                f"{path.name} has an invalid epoch range")


def verify_grouping_ablation() -> None:
    rows = load_json(TABLES / "grouping_ablation.json")
    expected = {
        (model, split, seed)
        for model in MODELS
        for split in ("recording", "load", "fault_size")
        for seed in SEEDS
    }
    actual = {(row["model"], row["split"], row["seed"]) for row in rows}
    require(len(rows) == 30 and actual == expected,
            "grouping_ablation.json must contain exactly 30 unique runs")
    for row in rows:
        verify_matrix(
            row.get("confusion_matrix"),
            f"grouping/{row['model']}/{row['split']}/seed={row['seed']}",
        )
        assignment = row.get("split_assignment")
        require(
            isinstance(assignment, dict)
            and set(assignment.get("recordings", {}))
            == {"train", "validation", "test"},
            f"grouping/{row['model']}/{row['split']}/seed={row['seed']} "
            "must include a complete split assignment",
        )
        recording_sets = [
            set(assignment["recordings"][split])
            for split in ("train", "validation", "test")
        ]
        expected_recording_counts = {
            "recording": [23, 7, 10],
            "load": [20, 10, 10],
            "fault_size": [14, 13, 13],
        }[row["split"]]
        require(
            all(recording_sets)
            and not (recording_sets[0] & recording_sets[1])
            and not (recording_sets[0] & recording_sets[2])
            and not (recording_sets[1] & recording_sets[2])
            and len(set().union(*recording_sets)) == 40,
            f"grouping/{row['model']}/{row['split']}/seed={row['seed']} "
            "must assign 40 recordings to non-empty, disjoint partitions",
        )
        require(
            [len(item) for item in recording_sets]
            == expected_recording_counts,
            f"grouping/{row['model']}/{row['split']}/seed={row['seed']} "
            f"must use recording counts {expected_recording_counts}",
        )
        expected_group_counts = {
            "recording": None,
            "load": [2, 1, 1],
            "fault_size": [1, 1, 1],
        }[row["split"]]
        groups = assignment.get("groups")
        if expected_group_counts is None:
            require(
                groups is None,
                f"grouping/{row['model']}/recording/seed={row['seed']} "
                "must not report a higher-level grouping variable",
            )
        else:
            require(
                isinstance(groups, dict)
                and [
                    len(groups.get(split, []))
                    for split in ("train", "validation", "test")
                ] == expected_group_counts,
                f"grouping/{row['model']}/{row['split']}/seed={row['seed']} "
                f"must use group counts {expected_group_counts}",
            )
    require_finite(rows, "grouping_ablation")

    main_rows = load_json(TABLES / "main_results.json")
    for model in MODELS:
        main_baseline = next(
            row
            for row in main_rows
            if row["model"] == model
            and row["split"] == "recording"
            and row["augmentation"] == "none"
        )
        grouping_accuracy = {
            row["seed"]: row["accuracy"]
            for row in rows
            if row["model"] == model and row["split"] == "recording"
        }
        expected_accuracy = dict(
            zip(main_baseline["per_seed_ids"], main_baseline["per_seed_accuracy"])
        )
        require(
            grouping_accuracy == expected_accuracy,
            f"{model} recording-group ablation does not reproduce the main baseline",
        )


def verify_manifests() -> None:
    dataset = load_json(MANIFESTS / "dataset_file_manifest.json")
    files = dataset.get("files", [])
    require(dataset.get("recording_count") == 40 and len(files) == 40,
            "dataset manifest must identify exactly 40 selected recordings")
    require(all(item.get("readable") is True for item in files),
            "all selected recordings must be marked readable")
    require(len({item.get("recording_id") for item in files}) == 40,
            "dataset manifest contains duplicate recording IDs")

    assignments = load_json(MANIFESTS / "recording_split_assignments.json")
    require(len(assignments) == 200,
            "recording split manifest must contain 40 recordings x 5 seeds")
    for seed in SEEDS:
        seed_rows = [row for row in assignments if row.get("seed") == seed]
        require(len(seed_rows) == 40, f"seed {seed} must assign all 40 recordings")
        require(len({row.get("recording_id") for row in seed_rows}) == 40,
                f"seed {seed} contains duplicate recording assignments")
        require({row.get("split") for row in seed_rows} == {"train", "validation", "test"},
                f"seed {seed} does not contain all three partitions")

    split_check = load_json(MANIFESTS / "split_verification.json")
    require([item.get("seed") for item in split_check.get("seeds", [])] == list(SEEDS),
            "split verification must cover the five configured seeds")
    require(all(item.get("disjoint") and item.get("all_selected_recordings_assigned")
                for item in split_check["seeds"]),
            "recording partitions must be disjoint and exhaustive")

    random_check = load_json(MANIFESTS / "random_split_verification.json")
    require(random_check.get("overlap") == split_check.get("overlap") == 0.5,
            "both split protocols must use the fixed 50% overlap")
    require([item.get("seed") for item in random_check.get("seeds", [])] == list(SEEDS),
            "random split verification must cover the five configured seeds")
    require(all(item.get("class_stratified") and item.get("all_windows_assigned_once")
                for item in random_check["seeds"]),
            "random-window partitions must be stratified and exhaustive")


def verify_release_files() -> None:
    required = (
        ROOT / "README.md",
        ROOT / "claim_evidence_table.md",
        ROOT / "paper" / "main.tex",
        ROOT / "paper" / "main.pdf",
        ROOT / "paper" / "refs.bib",
        ROOT / "paper_tables.tex",
        ROOT / "paper" / "generated_tables" / "main_results.tex",
        ROOT / "paper" / "generated_tables" / "macro_f1.tex",
        ROOT / "paper" / "generated_tables" / "gap_recovery.tex",
        ROOT / "paper" / "generated_tables" / "per_class.tex",
        ROOT / "paper" / "generated_tables" / "per_class_prf.tex",
        ROOT / "paper" / "generated_tables" / "grouping.tex",
        ROOT / "environment" / "requirements-a10-lock.txt",
        ROOT / "ars" / "research_question_brief.md",
        ROOT / "ars" / "method_card.md",
        ROOT / "ars" / "experiment_protocol.md",
        ROOT / "ars" / "reviewer_reports.md",
        ROOT / "ars" / "revision_response.md",
        ROOT / "ars" / "final_integrity_report.md",
        ROOT / "ars" / "paper_requirement_traceability_20260723.md",
        ROOT / "data_manifest" / "dataset_card_cwru.md",
        ROOT / "data_manifest" / "dataset_discovery_report.md",
        ROOT / "data_manifest" / "split_design.md",
        ROOT / "prompts" / "prompt_log.md",
        FIGURES / "m1_adjacent_autocorrelation.png",
        FIGURES / "m5_fault_band_energy.png",
        FIGURES / "m2_feature_diversity.png",
        FIGURES / "envelope_spectra_examples.png",
        FIGURES / "convergence_curves.png",
        FIGURES / "cohens_d_1d.png",
        FIGURES / "cohens_d_2d.png",
        FIGURES / "random_vs_recording_1d.png",
        FIGURES / "random_vs_recording_2d.png",
        FIGURES / "per_class_1d_recording.png",
        FIGURES / "per_class_2d_recording.png",
        FIGURES / "gap_recovery_1d.png",
        FIGURES / "gap_recovery_2d.png",
        FIGURES / "confusion_matrices_recording.png",
        FIGURES / "h3_physical_fidelity_correlation.png",
    )
    for path in required:
        require(path.is_file() and path.stat().st_size > 0,
                f"missing or empty release artifact: {path.relative_to(ROOT)}")

    for name in (
        "gap_recovery.json",
        "per_class_analysis.json",
        "falsification.json",
        "energy_audit.json",
        "adjacent_overlap_audit.json",
        "feature_diversity.json",
        "convergence_summary.json",
        "contamination_robustness.json",
        "physical_fidelity_correlation.json",
        "release_summary.json",
    ):
        require_finite(load_json(TABLES / name), name)

    summary = load_json(TABLES / "release_summary.json")
    require(
        summary.get("main_run_count") == 200
        and summary.get("configuration_count") == 40,
        "release summary must report 200 runs across 40 configurations",
    )
    grouping_design = summary.get("grouping", {}).get(
        "assignment_design", {}
    )
    require(
        grouping_design.get(
            "load_group_counts_train_validation_test"
        ) == [2, 1, 1]
        and grouping_design.get(
            "fault_size_group_counts_train_validation_test"
        ) == [1, 1, 1]
        and grouping_design.get(
            "recording_strategy_recording_counts"
        ) == [23, 7, 10]
        and grouping_design.get(
            "load_strategy_recording_counts"
        ) == [20, 10, 10]
        and grouping_design.get(
            "fault_size_strategy_recording_counts"
        ) == [14, 13, 13],
        "release summary must disclose unequal grouping-ablation group counts",
    )

    physical = load_json(TABLES / "physical_fidelity_correlation.json")
    require(
        set(physical.get("by_model", {})) == set(MODELS)
        and all(
            len(physical["by_model"][model].get("augmentations", [])) == 9
            for model in MODELS
        ),
        "H3 correlation must be reported separately for both models "
        "across all nine augmentations",
    )
    sensitivity = physical.get(
        "sensitivity_without_negative_control", {}
    )
    require(
        set(sensitivity) == set(MODELS)
        and all(
            len(sensitivity[model].get("augmentations", [])) == 8
            and sensitivity[model].get("excluded_augmentations")
            == ["freq_flip"]
            for model in MODELS
        ),
        "H3 must include an eight-augmentation per-model sensitivity "
        "analysis excluding the negative control",
    )
    require(
        physical.get("pooled_exploratory", {}).get("inferential") is False,
        "pooled H3 correlation must be labeled non-inferential",
    )


def verify_document_contracts() -> None:
    claim_path = ROOT / "claim_evidence_table.md"
    claim_text = claim_path.read_text(encoding="utf-8")
    expected_header = (
        "| Claim | Evidence | Artifact | Strength | Risk | Revision Needed |"
    )
    require(
        expected_header in claim_text,
        "claim-evidence table must use the six required columns",
    )
    require(
        claim_text.count("\n| C") >= 10,
        "claim-evidence table must contain at least ten traceable claims",
    )
    require(
        "TODO" not in claim_text and "TBD" not in claim_text,
        "claim-evidence table contains an unfinished placeholder",
    )

    main_tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    for fragment in (
        "main_results",
        "macro_f1",
        "gap_recovery",
        "per_class",
        "per_class_prf",
        "grouping",
    ):
        require(
            f"\\input{{generated_tables/{fragment}.tex}}" in main_tex,
            f"paper must source the generated {fragment} table",
        )

    pdf_path = ROOT / "paper" / "main.pdf"
    with pdf_path.open("rb") as handle:
        require(
            handle.read(5) == b"%PDF-",
            "paper/main.pdf is not a valid PDF file",
        )
    require(
        pdf_path.stat().st_size >= 100_000,
        "paper/main.pdf is unexpectedly small",
    )
    pdf_dependencies = [
        ROOT / "paper" / "main.tex",
        ROOT / "paper" / "refs.bib",
        *(
            ROOT / "paper" / "generated_tables" / f"{name}.tex"
            for name in (
                "main_results",
                "macro_f1",
                "gap_recovery",
                "per_class",
                "per_class_prf",
                "grouping",
            )
        ),
        *FIGURES.glob("*.png"),
    ]
    require(
        pdf_path.stat().st_mtime
        >= max(path.stat().st_mtime for path in pdf_dependencies),
        "paper/main.pdf is older than a source table or figure",
    )


def main() -> None:
    checks = (
        ("main results", verify_main_results),
        ("confusion matrices", verify_confusion_artifact),
        ("epoch logs", verify_epoch_logs),
        ("grouping ablation", verify_grouping_ablation),
        ("dataset and split manifests", verify_manifests),
        ("release files", verify_release_files),
        ("document contracts", verify_document_contracts),
    )
    for label, check in checks:
        check()
        print(f"[PASS] {label}")
    print("[PASS] release evidence is complete and internally consistent")


if __name__ == "__main__":
    try:
        main()
    except (VerificationError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
