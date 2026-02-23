import subprocess
import sys
import os

def run_command(command, description):
    print(f"--- {description} ---")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"FAILED: {description}")
        return False
    print(f"PASSED: {description}")
    return True

def main():
    # 1. Static Analysis
    # We ignore some errors that are expected due to missing stubs or environment setup
    static_passed = run_command("uv run mypy src main.py --ignore-missing-imports", "Static Analysis (mypy)")

    # 2. Unit Tests
    # We use unittest to discover and run tests in the tests folder
    test_passed = run_command("uv run python -m unittest discover tests", "Unit Tests")

    if static_passed and test_passed:
        print("\nALL TESTS PASSED!")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
