@echo off
REM Test runner script for Computer Vision Project
REM This script runs all tests and generates a coverage report

echo Running Computer Vision Project Test Suite
echo ==========================================

REM Create reports directory if it doesn't exist
if not exist reports\coverage mkdir reports\coverage

echo.
echo Running basic tests first...
echo --------------------------
python tests/run_tests_fixed.py --type basic --verbose
if %ERRORLEVEL% neq 0 (
    echo Basic tests failed! Fix these issues before continuing.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Running unit tests...
echo --------------------
python tests/run_tests_fixed.py --type unit --verbose
if %ERRORLEVEL% neq 0 (
    echo Unit tests failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Running integration tests...
echo --------------------------
python tests/run_tests_fixed.py --type integration --skip-gpu --verbose
if %ERRORLEVEL% neq 0 (
    echo Integration tests failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Generating coverage report...
echo ----------------------------
python tests/run_tests_fixed.py --type all --coverage --output reports/coverage
if %ERRORLEVEL% neq 0 (
    echo Coverage report generation failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Test suite completed successfully
echo ==========================================
echo Coverage report available at: reports/coverage/html/index.html
echo.

pause
