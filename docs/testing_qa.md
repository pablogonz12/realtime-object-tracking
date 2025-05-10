# Testing & Quality Assurance

The project includes a comprehensive testing framework to ensure reliability and robustness.

## Running Tests

We provide several ways to run the tests:

1. **Basic Tests Only** (Recommended for quick validation):
   ```
   # Windows
   run_basic_tests.bat
   
   # Unix/Mac
   ./run_basic_tests.sh
   ```

2. **All Tests with Coverage Report**:
   ```
   # Windows
   run_tests.bat
   
   # Unix/Mac
   ./run_tests.sh
   ```

3. **Custom Test Configuration**:
   ```
   python tests/run_tests_fixed.py --type [basic|unit|integration|all] [options]
   ```
   Options:
   - `--verbose` or `-v`: Show detailed test output
   - `--skip-gpu` or `-s`: Skip tests requiring GPU
   - `--coverage` or `-c`: Generate coverage report
   - `--output` or `-o`: Path to save coverage report

## Test Structure

- `tests/test_*_basic.py` - Basic test files that don't require complex data
- `tests/test_*.py` - Full test suite for each module
- `tests/conftest.py` - Pytest fixtures and configuration
- `pytest.ini` - Test discovery and marker configuration

## Test Markers

Tests are categorized using markers:
- `@pytest.mark.unit` - Unit tests for individual functions
- `@pytest.mark.integration` - Tests for module interactions
- `@pytest.mark.gpu` - Tests requiring GPU (can be skipped)
- `@pytest.mark.slow` - Time-consuming tests (can be skipped)

- **GPU Tests**: Tests that require a CUDA-capable GPU (skipped if not available)
  ```
  python tests/run_tests.py --skip-gpu  # Skip GPU tests
  ```

## Coverage Reports

Generate coverage reports to identify untested code:

```
python tests/run_tests.py --report --output reports/coverage
```

This will generate:
- An HTML report at `reports/coverage/html/index.html`
- An XML report at `reports/coverage/coverage.xml`

## Error Handling

The project includes robust error handling with:

1. **Input Validation**: All user inputs are validated
2. **Helpful Error Messages**: Clear and informative error messages
3. **Logging**: Comprehensive logging of operations and errors
4. **Graceful Recovery**: The system attempts to recover from errors when possible

Validation checks include:
- File path existence and format
- Model name validity
- Parameter ranges (confidence thresholds, IoU settings)
- Image and video format compatibility
- Device availability (CUDA, CPU, MPS)

## Quality Assurance Measures

- **Automated Testing**: CI/CD pipeline runs tests on each commit
- **Code Coverage**: Aim for >80% test coverage
- **Input Validation**: All user inputs are validated
- **Error Handling**: Graceful handling of errors with informative messages
- **Logging**: Comprehensive logging for debugging and auditing

# Test Organization

The project includes two types of tests:

1. **Basic Tests**: These are lightweight tests designed to quickly verify core functionality. They are located in files with `_basic` in their names (e.g., `test_models_basic.py`).

2. **Full Tests**: These are comprehensive tests that cover all edge cases and scenarios. They are located in files without `_basic` in their names (e.g., `test_models.py`).

## When to Use
- Use **basic tests** during development for quick feedback.
- Use **full tests** before deployment or during CI/CD pipelines to ensure complete coverage.
