import os
import sys
import subprocess
import json
import time
from config import RESULTS_DIR


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(script_path, step_name):
    print(f"\n{'=' * 60}")
    print(f"STEP: {step_name}")
    print(f"{'=' * 60}")
    t0 = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = BASE_DIR + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, script_path], capture_output=False,
                            cwd=BASE_DIR, env=env)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"[OK] {step_name} completed in {elapsed:.1f}s")
    else:
        print(f"[FAIL] {step_name} returned code {result.returncode}")
    return result.returncode


def main():
    os.makedirs(os.path.join(RESULTS_DIR, "tables"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)

    pipeline = [
        ("train.py", "Full Factorial Training (2M x 2S x 10A x 5 seeds = 200 runs)"),
        ("mechanism_validation/adjacent_autocorr.py", "M1: Adjacent-Window Autocorrelation"),
        ("mechanism_validation/fault_band_energy.py", "M5: Fault-Frequency Energy Audit"),
        ("mechanism_validation/feature_diversity.py", "M2: Feature Diversity Analysis"),
        ("mechanism_validation/envelope_spectrum.py", "Envelope Spectrum Examples"),
        ("mechanism_validation/convergence_analysis.py", "Convergence Analysis"),
        ("contamination_test.py", "Robustness Contamination Test"),
        ("analyze_results.py", "Gap-Recovery Analysis + Hypothesis Testing"),
        ("plot_figures.py", "Generate All Paper Figures"),
    ]

    failed = []
    for script, name in pipeline:
        spath = os.path.join(BASE_DIR, script)
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
    for root, dirs, files in os.walk(os.path.join(BASE_DIR, "..", "results")):
        for f in files:
            output_artifacts.append(os.path.relpath(os.path.join(root, f), os.path.join(BASE_DIR, "..")))
    print(f"\nOutput artifacts ({len(output_artifacts)}):")
    for a in sorted(output_artifacts):
        print(f"  {a}")


if __name__ == "__main__":
    main()
