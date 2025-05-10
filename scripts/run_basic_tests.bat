@echo off
REM Basic test runner for Computer Vision Project

echo Running basic tests...
echo --------------------------
python ../tests/run_tests.py --type basic --verbose
IF %ERRORLEVEL% NEQ 0 (
    echo Basic tests failed!
    exit /b 1
)
echo Basic tests completed successfully.
