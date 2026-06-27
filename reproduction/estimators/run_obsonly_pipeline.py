"""Run observability-only spectral-OOM on all 7 processes."""
import subprocess
import sys
from pathlib import Path

venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python3"

processes = [
    ("arch", "fig1_arch_hankel.py"),
    ("mess3", "fig1_mess3_hankel.py"),
    ("fern", "fig1_fern_hankel.py"),
    ("strata", "fig1_strata_hankel.py"),
    ("wing", "fig1_wing_hankel.py"),
    ("zero_one_random", "fig1_zero_one_random_hankel.py"),
    ("rrxor", "fig1_rrxor_hankel.py"),
]

results = []

print("=" * 70)
print("OBSERVABILITY-ONLY SPECTRAL-OOM: ALL PROCESSES")
print("=" * 70)
print("Testing observability matrix (no two-factor Hankel) on 7 processes...\n")

for name, script in processes:
    print("=" * 70)
    print(f"Running: {name}")
    print("=" * 70)

    try:
        result = subprocess.run(
            [str(venv_python), "-B", script],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=120
        )
        output = result.stdout + result.stderr
        print(output)

        # Parse the result line
        for line in output.split('\n'):
            if "Elbow-detected d:" in line and "True d:" in line:
                parts = line.split()
                d_det = int(parts[2])
                d_true = int(parts[5])
                match = "Match: True" in line
                results.append((name, d_true, d_det, match))
                break
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {name} took too long")
        results.append((name, "?", "?", False))
    except Exception as e:
        print(f"ERROR: {name} - {e}")
        results.append((name, "?", "?", False))

    print()

# Print summary
print("\n" + "=" * 70)
print("SUMMARY REPORT")
print("=" * 70)
print()
print(f"{'Process':<20} | {'True d':<8} | {'Detected d':<12} | {'Match':<8}")
print("-" * 70)
for name, d_true, d_det, match in results:
    status = "✓" if match else "✗"
    print(f"{name:<20} | {str(d_true):<8} | {str(d_det):<12} | {status:<8}")

correct = sum(1 for _, _, _, m in results if m)
total = len(results)
print("-" * 70)
print(f"Correct: {correct}/{total} ({100*correct//total}%)\n")

if correct == total:
    print("✓ All processes correct!")
elif correct > 0:
    print(f"⚠ Partial success: {correct} process(es) correct")
else:
    print("✗ Observability-only: all processes failed")
