"""
Test Runner Script for Computer Vision Project

This script demonstrates how to run the test suite for the project.
It includes commands for running different types of tests and generating reports.
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

# Import error handling if available
try:
    from src.error_handling import print_friendly_error
except ImportError:
    def print_friendly_error(e, exit_code=None):
        """Simple error handler if the module is not available"""
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
    
    parser.add_argument(
        "--coverage", "-c",
        action="store_true",
        help="Generate a coverage report (requires pytest-cov)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save the coverage report",
        default=None
    )
    
    return parser.parse_args()


def run_tests(test_type="all", skip_gpu=False, verbose=False, report=False, output=None):
    """
    Run tests of the specified type
    
    Args:
        test_type: Type of tests to run (unit, integration, or all)
        skip_gpu: Whether to skip GPU tests
        verbose: Whether to show verbose output
        report: Whether to generate a coverage report
        output: Path to save the report
    """
    # Compose the pytest command
    cmd = ["pytest"]
    
    # Add test type
    if test_type == "unit":
        cmd.append("-m")
        cmd.append("unit")
    elif test_type == "integration":
        cmd.append("-m")
        cmd.append("integration")    elif test_type == "basic":
        # Run specific basic test files that are known to exist
        basic_test_files = [
            "tests/test_error_handling_basic.py",
            "tests/test_validation_basic.py",
            "tests/test_video_utils_basic.py"
        ]
        
        # Check if test_models_basic.py exists and include it if it does
        models_basic_path = Path("tests/test_models_basic.py")
        print(f"Looking for file: {models_basic_path}, exists: {models_basic_path.exists()}")
        if models_basic_path.exists():
            basic_test_files.append(str(models_basic_path))
            print(f"Added {models_basic_path} to test files.")
            
        cmd.extend(basic_test_files)
    
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
      
    # Add coverage reporting only if explicitly requested
    if report:
        try:
            # Check if pytest-cov is installed
            import pytest_cov
            
            # Add coverage arguments
            cmd.append("--cov=src")
            cmd.append("--cov-report=term")
            
            if output:
                output_dir = Path(output)
                output_dir.mkdir(exist_ok=True, parents=True)
                
                # HTML report
                cmd.append(f"--cov-report=html:{output}/html")
                
                # XML report for CI systems
                cmd.append(f"--cov-report=xml:{output}/coverage.xml")
        except ImportError:
            print("Warning: pytest-cov not found. Skipping coverage report.")
            report = False
    
    # Run the tests
    try:
        print(f"\nRunning {test_type} tests...")
        print(f"Command: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("\nTests completed successfully!")
        
        if report and output:
            print(f"\nCoverage report saved to: {output}")
    
    except subprocess.CalledProcessError as e:
        print_friendly_error(f"Tests failed with exit code {e.returncode}", exit_code=e.returncode)
    
    except Exception as e:
        print_friendly_error(e, exit_code=1)


if __name__ == "__main__":
    """Run the test suite"""
    args = parse_args()
    run_tests(
        test_type=args.type,
        skip_gpu=args.skip_gpu,
        verbose=args.verbose,
        report=args.coverage,
        output=args.output
    )
