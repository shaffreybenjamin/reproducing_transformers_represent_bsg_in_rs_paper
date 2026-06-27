"""
Analyze Hankel singular values for manual elbow inspection.

Runs spectral-OOM on all 7 processes and outputs singular value spectra
without applying any automatic elbow detection method. This allows manual
visual inspection to determine where the true elbow should be.
"""
from pathlib import Path
import subprocess
import sys
import numpy as np

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


def extract_sv_from_output(output_text):
    """Extract singular values from script output."""
    lines = output_text.split('\n')
    sv_list = None

    for i, line in enumerate(lines):
        if "Singular value spectrum (top 20):" in line:
            # Next line contains the array
            if i + 1 < len(lines):
                sv_str = lines[i + 1].strip()
                # Parse numpy array format
                sv_str = sv_str.replace('[', '').replace(']', '')
                try:
                    sv_list = np.array([float(x) for x in sv_str.split()])
                except:
                    pass
            break

    return sv_list


def run_process_get_sv(name, script, true_d):
    """Run a process and extract singular values."""
    venv_python = str(Path(__file__).parent.parent / ".venv" / "bin" / "python3")

    try:
        result = subprocess.run(
            [venv_python, "-B", script],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=600
        )

        sv = extract_sv_from_output(result.stdout)

        return {
            "name": name,
            "true_d": true_d,
            "sv": sv,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "true_d": true_d,
            "sv": None,
            "success": False,
        }
    except Exception as e:
        return {
            "name": name,
            "true_d": true_d,
            "sv": None,
            "success": False,
        }


def main():
    print("\n" + "="*80)
    print("HANKEL SINGULAR VALUES ANALYSIS: MANUAL ELBOW INSPECTION")
    print("="*80)
    print("\nRunning spectral-OOM on all 7 processes to extract singular values...")
    print("Inspect the spectra below and determine where YOU think the elbow should be.")
    print("Compare your judgment with the true d value shown.\n")

    results = []
    for name, script, true_d in PROCESSES:
        print(f"Processing {name:20s}...", end=" ", flush=True)
        res = run_process_get_sv(name, script, true_d)
        results.append(res)
        print("✓" if res["success"] else "✗")

    # Generate detailed report
    print("\n" + "="*80)
    print("SINGULAR VALUE SPECTRA")
    print("="*80)

    out_path = SCRIPT_DIR / "HANKEL_SINGULAR_VALUES_REPORT.txt"
    with open(out_path, "w") as f:
        f.write("HANKEL SINGULAR VALUES ANALYSIS: MANUAL ELBOW INSPECTION\n")
        f.write("="*80 + "\n\n")
        f.write("Below are the singular value spectra for each process.\n")
        f.write("TRUE_D indicates the known number of hidden states.\n")
        f.write("Visually inspect where you think the elbow (knee) occurs.\n\n")

        for res in results:
            name = res["name"]
            true_d = res["true_d"]
            sv = res["sv"]

            output = f"\n{name.upper()}\n"
            output += "-" * 80 + "\n"
            output += f"True d: {true_d}\n"

            if sv is None:
                output += "ERROR: Could not extract singular values\n"
            else:
                output += f"Number of singular values: {len(sv)}\n\n"

                # Show all singular values
                output += "Raw singular values:\n"
                output += "  " + str(np.round(sv, 4)) + "\n\n"

                # Show normalized spectrum
                sv_norm = sv / sv[0]
                output += "Normalized spectrum (sv / sv[0]):\n"
                output += "  " + str(np.round(sv_norm, 4)) + "\n\n"

                # Show log spectrum and gaps
                log_sv = np.log(np.clip(sv_norm, 1e-12, None))
                output += "Log spectrum:\n"
                output += "  " + str(np.round(log_sv, 4)) + "\n\n"

                gaps = np.diff(log_sv)
                output += "Log gaps (negative values show drop magnitude):\n"
                output += "  " + str(np.round(gaps, 4)) + "\n\n"

                # Formatted table for easy inspection
                output += "Index | SV (raw)  | SV (norm) | Log(SV)  | Gap (magnitude)\n"
                output += "------|-----------|-----------|----------|----------------\n"
                for i in range(min(20, len(sv))):
                    gap_str = f"{abs(gaps[i]):.4f}" if i < len(gaps) else "-"
                    output += f"{i:5d} | {sv[i]:9.4f} | {sv_norm[i]:9.4f} | {log_sv[i]:8.4f} | {gap_str}\n"

                if len(sv) > 20:
                    output += f"... ({len(sv) - 20} more values)\n"

            output += "\n"
            print(output)
            f.write(output)

        f.write("\n" + "="*80 + "\n")
        f.write("INSTRUCTIONS FOR MANUAL ELBOW SELECTION:\n")
        f.write("="*80 + "\n\n")
        f.write("1. Look for where the singular values transition from 'large' to 'small'\n")
        f.write("2. Check the 'Gap (magnitude)' column - large gaps indicate structural breaks\n")
        f.write("3. The elbow is typically where the largest gap occurs\n")
        f.write("4. Compare the normalized spectrum - smoother decay = harder to detect\n")
        f.write("5. Remember: the true dimension equals the number of kept singular values\n")
        f.write("   (not the gap index)\n\n")

    print(f"\nFull analysis saved to: {out_path}")
    print(f"\nYou can now inspect the spectra and determine the true elbows manually.")
    print(f"The report shows raw, normalized, and log spectra plus gap magnitudes.")


if __name__ == "__main__":
    main()
