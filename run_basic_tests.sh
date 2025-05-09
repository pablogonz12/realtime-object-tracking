#!/bin/bash
# Basic test runner script for Computer Vision Project
# This script runs only basic tests without coverage reporting

echo "Running Computer Vision Project Basic Tests"
echo "=========================================="

echo ""
echo "Running basic tests..."
echo "--------------------"
python tests/run_tests.py --type basic --verbose

echo ""
echo "Basic tests completed"
echo "=========================================="
