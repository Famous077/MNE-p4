"""
Final verification — runs everything and gives a pass/fail summary.

    python verify.py
"""

import sys
import subprocess
from pathlib import Path


def run_command(description, cmd):
    """Run a command and return success/failure."""
    print(f"\n{'─' * 50}")
    print(f"  {description}")
    print(f"  $ {cmd}")
    print(f"{'─' * 50}")

    result = subprocess.run(
        cmd, shell=True,
        capture_output=False
    )
    return result.returncode == 0


def main():
    print("=" * 60)
    print("  FINAL VERIFICATION")
    print("  Checking that everything works end-to-end")
    print("=" * 60)

    results = []

    # check imports
    results.append((
        "Import check",
        run_command(
            "Verify all modules import correctly",
            f"{sys.executable} -c \"from src import RawMffNew, "
            f"MFFFileInfo, create_demo_mff; print('All imports OK')\""
        )
    ))

    # run demo
    results.append((
        "Demo script",
        run_command(
            "Run the full demonstration",
            f"{sys.executable} demo.py"
        )
    ))
    results.append((
        "Test suite",
        run_command(
            "Run all tests",
            f"{sys.executable} -m pytest tests/ -v --tb=short"
        )
    ))

    print("\n" + "=" * 60)
    print("  VERIFICATION SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  ✓ ALL VERIFICATIONS PASSED")
        print("  This PoC is ready for review!")
    else:
        print("  ✗ SOME VERIFICATIONS FAILED")
        print("  Check the output above for details.")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())