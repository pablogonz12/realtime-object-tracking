#!/bin/bash
# Basic test runner script for Computer Vision Project

echo "Running basic tests..."
echo "--------------------------"
python ../tests/run_tests.py --type basic --verbose
if [ $? -ne 0 ]; then
    echo "Basic tests failed! Fix these issues before continuing."
    exit 1
fi
echo "Basic tests completed successfully."