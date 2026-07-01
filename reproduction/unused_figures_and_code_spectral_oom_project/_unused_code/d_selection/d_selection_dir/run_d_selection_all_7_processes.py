"""Orchestrator: Run d-selection on all 7 processes and generate comprehensive report.

This script runs all 7 processes in sequence and produces:
1. Individual result files for each process (d_selection_PROCESS_results.txt)
2. A master summary report (D_SELECTION_COMPREHENSIVE_REPORT.txt)
3. A CSV table for easy comparison

Usage:
    python run_d_selection_all_7_processes.py
"""

from pathlib import Path
import subprocess
import sys

PROCESSES = [
    ("zero_one_random", 3),
    ("fern", 3),
    ("strata", 3),
    ("wing", 3),
    ("arch", 4),
    ("mess3", 3),
    ("rrxor", 5),
]

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR


def run_process(process_name, true_d):
    """Run d-selection for a single process."""
    script = SCRIPT_DIR / f"fig1_{process_name}_d_selection.py"
    if not script.exists():
        print(f"  ERROR: {script} not found")
        return None

    print(f"\n{'='*70}")
    print(f"Running {process_name} (true d={true_d})...")
    print(f"{'='*70}")

    result = subprocess.run([sys.executable, str(script)], cwd=SCRIPT_DIR)
    return result.returncode == 0


def read_results_file(process_name):
    """Read individual process results file."""
    result_file = OUT_DIR / f"d_selection_{process_name}_results.txt"
    if result_file.exists():
        with open(result_file, "r") as f:
            return f.read()
    return None


def main():
    print("\n" + "="*70)
    print("D-SELECTION COMPREHENSIVE PIPELINE")
    print("Testing effective rank + MI saturation on 7 known processes")
    print("="*70)

    results_summary = {}

    # Run all processes
    for process_name, true_d in PROCESSES:
        success = run_process(process_name, true_d)
        if success:
            print(f"  ✓ {process_name} completed successfully")
            results_summary[process_name] = {
                "true_d": true_d,
                "success": True,
            }
        else:
            print(f"  ✗ {process_name} failed")
            results_summary[process_name] = {
                "true_d": true_d,
                "success": False,
            }

    # Generate comprehensive report
    print(f"\n{'='*70}")
    print("GENERATING COMPREHENSIVE REPORT")
    print(f"{'='*70}")

    report_lines = [
        "D-SELECTION COMPREHENSIVE REPORT",
        "="*70,
        f"Analysis of effective rank + MI saturation methods",
        f"Tested on 7 known processes\n",
    ]

    # Add per-process results
    for process_name, true_d in PROCESSES:
        report_lines.append(f"\n{'-'*70}")
        report_lines.append(f"{process_name.upper()} (true d={true_d})")
        report_lines.append(f"{'-'*70}")

        result_file = OUT_DIR / f"d_selection_{process_name}_results.txt"
        if result_file.exists():
            with open(result_file, "r") as f:
                content = f.read()
                report_lines.append(content)
        else:
            report_lines.append("(Results file not found)")

    # Add summary table
    report_lines.append(f"\n\n{'='*70}")
    report_lines.append("SUMMARY TABLE")
    report_lines.append(f"{'='*70}")
    report_lines.append(
        f"{'Process':<20} {'True d':<10} {'ER95 (OOM)':<12} {'ER99 (OOM)':<12} "
        f"{'MI (OOM)':<12} {'ER95 (CCA)':<12} {'ER99 (CCA)':<12} {'MI (CCA)':<12}"
    )
    report_lines.append("-" * 100)

    # Parse individual files to build table
    for process_name, true_d in PROCESSES:
        result_file = OUT_DIR / f"d_selection_{process_name}_results.txt"
        if result_file.exists():
            with open(result_file, "r") as f:
                content = f.read()
                # Extract values (simple parsing)
                import re

                er95_oom = re.search(r"Effective Rank \(95%\): (\d+)", content)
                er99_oom = re.search(
                    r"Effective Rank \(99%\): (\d+)", content.split("CCA/RRR:")[0]
                )
                mi_oom = re.search(r"MI Saturation: (\d+)", content.split("CCA/RRR:")[0])

                er95_cca = re.search(
                    r"Effective Rank \(95%\): (\d+)", content.split("CCA/RRR:")[1]
                )
                er99_cca = re.search(r"Effective Rank \(99%\): (\d+)", content.split("CCA/RRR:")[1])
                mi_cca = re.search(r"MI Saturation: (\d+)", content.split("CCA/RRR:")[1])

                er95_oom_val = er95_oom.group(1) if er95_oom else "?"
                er99_oom_val = er99_oom.group(1) if er99_oom else "?"
                mi_oom_val = mi_oom.group(1) if mi_oom else "?"
                er95_cca_val = er95_cca.group(1) if er95_cca else "?"
                er99_cca_val = er99_cca.group(1) if er99_cca else "?"
                mi_cca_val = mi_cca.group(1) if mi_cca else "?"

                report_lines.append(
                    f"{process_name:<20} {true_d:<10} {er95_oom_val:<12} {er99_oom_val:<12} "
                    f"{mi_oom_val:<12} {er95_cca_val:<12} {er99_cca_val:<12} {mi_cca_val:<12}"
                )

    report_text = "\n".join(report_lines)

    # Save report
    report_file = OUT_DIR / "D_SELECTION_COMPREHENSIVE_REPORT.txt"
    with open(report_file, "w") as f:
        f.write(report_text)

    print(f"\n✓ Comprehensive report saved to: {report_file}")

    # Print summary to stdout
    print(f"\n{report_text}")


if __name__ == "__main__":
    main()
