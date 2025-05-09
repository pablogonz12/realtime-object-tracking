#!/bin/bash
# Test runner script for Computer Vision Project
# This script runs all tests and generates a coverage report

echo "Running Computer Vision Project Test Suite"
echo "=========================================="

# Create reports directory if it doesn't exist
mkdir -p reports/coverage

echo ""
echo "Running basic tests first..."
echo "--------------------------"
python tests/run_tests_fixed.py --type basic --verbose
if [ $? -ne 0 ]; then
    echo "Basic tests failed! Fix these issues before continuing."
    exit 1
fi

echo ""
echo "Running unit tests..."
echo "--------------------"
python tests/run_tests_fixed.py --type unit --verbose
if [ $? -ne 0 ]; then
    echo "Unit tests failed!"
    exit 1
fi

echo ""
echo "Running integration tests..."
echo "--------------------------"
python tests/run_tests_fixed.py --type integration --skip-gpu --verbose
if [ $? -ne 0 ]; then
    echo "Integration tests failed!"
    exit 1
fi

echo ""
echo "Generating coverage report..."
echo "----------------------------"
python tests/run_tests_fixed.py --type all --coverage --output reports/coverage
if [ $? -ne 0 ]; then
    echo "Coverage report generation failed!"
    exit 1
fi

echo ""
echo "Test suite completed successfully"
echo "=========================================="
echo "Coverage report available at: reports/coverage/html/index.html"
echo ""
