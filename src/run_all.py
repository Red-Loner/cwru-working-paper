import os
import sys
import subprocess
import json
import time
from config import RESULTS_DIR


def run_step(script_path, step_name):
    print(f"\n{'=' * 60}")
    print(f"STEP: {step_name}")
    print(f"{'=' * 60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, script_path], capture_output=False,
                            cwd=os.path.dirname(script_path))
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"[OK] {step_name} completed in {elapsed:.1f}s")
    else:
        print(f"[FAIL] {step_name} returned code {result.returncode}")
    return result.returncode


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)

    pipeline = [
        ("src/train.py", "Full Factorial Training (2M x 2S x 10A x 3 seeds = 120 runs)"),
        ("src/mechanism_validation/adjacent_autocorr.py", "M1: Adjacent-Window Autocorrelation"),
        ("src/mechanism_validation/fault_band_energy.py", "M5: Fault-Frequency Energy Audit"),
        ("src/mechanism_validation/feature_diversity.py", "M2: Feature Diversity Analysis"),
        ("src/mechanism_validation/envelope_spectrum.py", "Envelope Spectrum Examples"),
        ("src/contamination_test.py", "Robustness Contamination Test"),
        ("src/analyze_results.py", "Gap-Recovery Analysis + Hypothesis Testing"),
        ("src/plot_figures.py", "Generate All Paper Figures"),
    ]

    failed = []
    for script, name in pipeline:
        spath = os.path.join(base, script)
        rc = run_step(spath, name)
        if rc != 0:
            failed.append(name)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"COMPLETED WITH {len(failed)} FAILURES:")
        for f in failed:
            print(f"  - {f}")
    else:
        print("ALL STEPS COMPLETED SUCCESSFULLY.")

    output_artifacts = []
    for root, dirs, files in os.walk(os.path.join(base, "..", "results")):
        for f in files:
            output_artifacts.append(os.path.relpath(os.path.join(root, f), os.path.join(base, "..")))
    print(f"\nOutput artifacts ({len(output_artifacts)}):")
    for a in sorted(output_artifacts):
        print(f"  {a}")


if __name__ == "__main__":
    main()
