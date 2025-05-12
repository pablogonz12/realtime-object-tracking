'''Nox sessions for linting, testing, and coverage.

Inspired by noxfile.py in Google Cloud Platform Python client libraries.
'''

import nox


@nox.session
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.run("python", "-m", "pip", "install", "-v", "-r", "requirements.txt", "-r", "dev-requirements.txt") 
    session.run("pytest", *session.posargs) # Pass any arguments after -- to pytest


@nox.session
def basic_tests(session: nox.Session) -> None:
    """Run basic tests only."""
    session.run("python", "-m", "pip", "install", "-v", "-r", "requirements.txt", "-r", "dev-requirements.txt")
    # Run tests matching the pattern *_basic.py in the tests directory
    session.run("pytest", "tests/test_error_handling_basic.py", "tests/test_models_basic.py", "tests/test_validation_basic.py", "tests/test_video_utils_basic.py", *session.posargs)


# Example of a linting session (can be expanded)
@nox.session
def lint(session: nox.Session) -> None:
    """Lint using flake8."""
    # No need to install from requirements.txt if only linting tools are needed
    session.install("-r", "dev-requirements.txt") 
    session.run("flake8", "src", "tests", "noxfile.py")
