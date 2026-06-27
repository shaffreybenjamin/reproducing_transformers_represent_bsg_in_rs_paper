"""Run observability-only spectral-OOM on all 7 processes to compare with two-factor Hankel."""
import subprocess
import sys
from pathlib import Path

venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python3"

processes = [
    ("arch", "fig1_arch_hankel.py"),
    ("mess3", "fig1_mess3_hankel_test_observability.py"),
    ("fern", "fig1_fern_hankel.py"),
    ("strata", "fig1_strata_hankel.py"),
    ("wing", "fig1_wing_hankel.py"),
    ("zero_one_random", "fig1_zero_one_random_hankel.py"),
    ("rrxor", "fig1_rrxor_hankel.py"),
]

print("=" * 70)
print("OBSERVABILITY-ONLY SPECTRAL-OOM: ALL PROCESSES")
print("=" * 70)
print("Testing dimensionality selection on 7 processes...\n")

for name, script in processes:
    print("=" * 70)
    print(f"Running: {name}")
    print("=" * 70)
    print()

    result = subprocess.run(
        [str(venv_python), "-B", script],
        cwd=Path(__file__).parent,
        capture_output=False
    )
    print()

print("All processes completed!")
