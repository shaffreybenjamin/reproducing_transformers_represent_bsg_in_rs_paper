"""Orchestrator: Run Hankel two-factor spectral-OOM on all 7 processes.

Generates individual results files and a comprehensive summary report comparing
elbow-detected dimensionality vs true dimensionality across all processes.
"""
from pathlib import Path
import subprocess
import sys

# Process metadata: (name, script, true_d)
PROCESSES = [
    ("arch", "fig1_arch_hankel.py", 4),
    ("mess3", "fig1_mess3_hankel.py", 3),
    ("fern", "fig1_fern_hankel.py", 3),
    ("strata", "fig1_strata_hankel.py", 3),
    ("wing", "fig1_wing_hankel.py", 3),
    ("zero_one_random", "fig1_zero_one_random_hankel.py", 3),
    ("rrxor", "fig1_rrxor_hankel.py", 5),
]

SCRIPT_DIR = Path(__file__).parent


def run_process(name, script, true_d):
    """Run a single process script and capture output."""
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"{'='*70}")

    try:
        # Use the venv's python interpreter for subprocess
        venv_python = str(Path(__file__).parent.parent / ".venv" / "bin" / "python3")
        result = subprocess.run(
            [venv_python, "-B", script],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=600
        )

        success = result.returncode == 0
        output = result.stdout + ("\n" + result.stderr if result.stderr else "")

        # Parse the elbow-detected d from output
        d_detected = None
        for line in output.split('\n'):
            if "Elbow-detected d:" in line:
                parts = line.split("Elbow-detected d:")
                if len(parts) > 1:
                    try:
                        d_detected = int(parts[1].split()[0])
                    except:
                        pass
                break

        return {
            "name": name,
            "true_d": true_d,
            "d_detected": d_detected,
            "success": success,
            "output": output,
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "true_d": true_d,
            "d_detected": None,
            "success": False,
            "output": f"TIMEOUT: {name} took too long (>10 min)",
        }
    except Exception as e:
        return {
            "name": name,
            "true_d": true_d,
            "d_detected": None,
            "success": False,
            "output": f"ERROR: {str(e)}",
        }


def main():
    print("\n" + "="*70)
    print("HANKEL TWO-FACTOR SPECTRAL-OOM: ALL PROCESSES")
    print("="*70)
    print(f"Testing dimensionality selection on {len(PROCESSES)} processes...")

    results = []
    for name, script, true_d in PROCESSES:
        res = run_process(name, script, true_d)
        results.append(res)

    # Generate summary report
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)

    summary_lines = []
    summary_lines.append("")
    summary_lines.append("Process              | True d | Detected d | Match | Status")
    summary_lines.append("-" * 65)

    correct_count = 0
    for res in results:
        name = res["name"]
        true_d = res["true_d"]
        d_det = res["d_detected"]
        match = "✓" if d_det == true_d else ("?" if d_det is None else "✗")
        status = "OK" if res["success"] else "FAIL"

        if d_det == true_d:
            correct_count += 1

        line = f"{name:20} | {true_d:6} | {d_det if d_det is not None else '?':10} | {match:5} | {status}"
        summary_lines.append(line)

    summary_lines.append("-" * 65)
    summary_lines.append(f"Correct: {correct_count}/{len(PROCESSES)} ({100*correct_count/len(PROCESSES):.1f}%)")
    summary_lines.append("")

    summary = "\n".join(summary_lines)
    print(summary)

    # Write summary to file
    out_path = SCRIPT_DIR / "HANKEL_ELBOW_RESULTS.txt"
    with open(out_path, "w") as f:
        f.write("HANKEL TWO-FACTOR SPECTRAL-OOM: D-SELECTION RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(summary)
        f.write("\n" + "=" * 70 + "\n\n")

        # Detailed results
        f.write("DETAILED RESULTS\n")
        f.write("=" * 70 + "\n")
        for res in results:
            f.write(f"\n{res['name'].upper()}\n")
            f.write("-" * 70 + "\n")
            f.write(f"True d: {res['true_d']}\n")
            f.write(f"Detected d: {res['d_detected']}\n")
            f.write(f"Status: {'SUCCESS' if res['success'] else 'FAILED'}\n")
            f.write(f"\nOutput:\n{res['output']}\n")

    print(f"\nFull report saved to: {out_path}")

    # Exit with success code only if all matched
    if correct_count == len(PROCESSES):
        print("\n✓ All processes: elbow method correctly selected d!")
        return 0
    else:
        print(f"\n✗ Hankel elbow: {len(PROCESSES) - correct_count} process(es) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
