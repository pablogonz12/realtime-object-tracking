@echo off
REM Test runner for Computer Vision Project
REM This script runs all tests and generates a coverage report

echo Running Computer Vision Project Test Suite
echo ==========================================

REM Create reports directory if it doesn't exist
mkdir .\reports\coverage 2>NUL

echo.
echo Running basic tests first...
echo --------------------------
python ..\tests\run_tests.py --type basic --verbose
IF %ERRORLEVEL% NEQ 0 (
    echo Basic tests failed! Fix these issues before continuing.
    exit /b 1
)

echo.
echo Running unit tests...
echo --------------------
python ..\tests\run_tests.py --type unit --verbose
IF %ERRORLEVEL% NEQ 0 (
    echo Unit tests failed!
    exit /b 1
)

echo.
echo Running integration tests...
echo --------------------------
python ..\tests\run_tests.py --type integration --skip-gpu --verbose
IF %ERRORLEVEL% NEQ 0 (
    echo Integration tests failed!
    exit /b 1
)

echo.
echo Generating coverage report...
echo ----------------------------
python ..\tests\run_tests.py --type all --coverage --output ..\reports\coverage
IF %ERRORLEVEL% NEQ 0 (
    echo Coverage report generation failed!
    exit /b 1
)

echo.
echo Test suite completed successfully
echo ==========================================
echo Coverage report available at: reports\coverage\html\index.html
echo.
