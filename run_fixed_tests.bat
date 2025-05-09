@echo off
REM Basic test runner script for Computer Vision Project using the fixed test runner
REM This script runs only basic tests without coverage reporting

echo Running Computer Vision Project Basic Tests with Fixed Runner
echo ==========================================================

echo.
echo Running basic tests...
echo --------------------
python tests/run_tests_fixed.py --type basic --verbose

echo.
echo Basic tests completed
echo ==========================================================

pause
