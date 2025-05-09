"""
Test Runner Script for Computer Vision Project

This script demonstrates how to run the test suite for the project.
It includes commands for running different types of tests.
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add the src directory to the path
project_root = Path(__file__).resolve().parent.parent
src_dir = project_root / "src"
sys.path.insert(0, str(project_root))

# Simple error handler
def print_error(e, exit_code=None):
    """Simple error handler"""
    print(f"\nError: {str(e)}")
    if exit_code is not None:
        sys.exit(exit_code)


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Run tests for the Computer Vision Project"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["unit", "integration", "all", "basic"],
        default="basic",
        help="Type of tests to run (unit, integration, all, or basic)"
    )
    
    parser.add_argument(
        "--skip-gpu", "-s",
        action="store_true",
        help="Skip tests that require a GPU"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )
    
    return parser.parse_args()


def run_tests(test_type="basic", skip_gpu=False, verbose=False):
    """
    Run tests of the specified type
    
    Args:
        test_type: Type of tests to run (unit, integration, all, or basic)
        skip_gpu: Whether to skip GPU tests
        verbose: Whether to show verbose output
    """
    # Compose the pytest command
    cmd = ["pytest"]
      # Add test type
    if test_type == "unit":
        cmd.append("-m")
        cmd.append("unit")
    elif test_type == "integration":
        cmd.append("-m")
        cmd.append("integration")
    elif test_type == "basic":
        # Run specific basic test files
        cmd.extend([
            "tests/test_error_handling_basic.py",
            "tests/test_models_basic.py",
            "tests/test_validation_basic.py",
            "tests/test_video_utils_basic.py"
        ])
    
    # Skip GPU tests if requested
    if skip_gpu:
        if "-m" in cmd:
            # If we already have a marker, we need to combine them
            idx = cmd.index("-m") + 1
            cmd[idx] = f"{cmd[idx]} and not gpu"
        else:
            cmd.append("-m")
            cmd.append("not gpu")
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Run the tests
    try:
        print(f"\nRunning {test_type} tests...")
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("\nTests completed successfully!")
    
    except subprocess.CalledProcessError as e:
        print_error(f"Tests failed with exit code {e.returncode}", exit_code=e.returncode)
    
    except Exception as e:
        print_error(e, exit_code=1)


if __name__ == "__main__":
    """Run the test suite"""
    args = parse_args()
    run_tests(
        test_type=args.type,
        skip_gpu=args.skip_gpu,
        verbose=args.verbose
    )
