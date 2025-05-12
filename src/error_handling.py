"""
Error handling and logging utilities for the Computer Vision Project.
Provides functions to log errors and handle them gracefully.
"""
import logging # Added import
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, Callable
import functools # Added for decorators
import sys # Added for sys.exit

# Configure logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Create a logger
logger = logging.getLogger("cvproject")
logger.setLevel(logging.INFO)

# Create a file handler
log_file = LOG_DIR / f"cvproject_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)  # Less verbose on console

# Create a formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def log_info(message: str) -> None:
    """Logs an informational message."""
    logger.info(message)


def log_warning(message: str) -> None:
    """Logs a warning message."""
    logger.warning(message)


def log_error(message: str, exc_info: bool = False) -> None:
    """Logs an error message.

    Args:
        message (str): The error message.
        exc_info (bool): If True, exception information is added to the log.
    """
    logger.error(message, exc_info=exc_info)


def log_exception(e: Exception, context: Optional[str] = None) -> None:
    """Logs an exception with optional context.

    Args:
        e (Exception): The exception to log.
        context (Optional[str]): Additional context about where the exception occurred.
    """
    if context:
        logger.exception(f"Exception occurred in {context}: {e}")
    else:
        logger.exception(f"An exception occurred: {e}")


def handle_exceptions(func: Callable) -> Callable:
    """
    A decorator that wraps a function to catch and log any exceptions.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            log_exception(e, context=func.__name__)
            # Optionally, re-raise the exception or return a default value
            # For now, we'll just log and let it propagate or be handled by outer layers
            raise
    return wrapper


def with_error_handling(message: str = "An error occurred") -> Callable:
    """
    A decorator factory that wraps a function to catch exceptions and log a custom message.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_error(f"{message} in {func.__name__}: {e}", exc_info=True)
                # Optionally, re-raise or return a default
                raise
        return wrapper
    return decorator


def robust_function(fallback_value: Any = None) -> Callable:
    """
    A decorator factory that makes a function robust by catching exceptions
    and returning a fallback value.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_exception(e, context=f"Robust call to {func.__name__}")
                return fallback_value
        return wrapper
    return decorator


def print_friendly_error(e: Exception, exit_code: Optional[int] = None) -> None:
    """
    Prints a user-friendly error message to the console and logs the full exception.
    Optionally exits the program.

    Args:
        e (Exception): The exception that occurred.
        exit_code (Optional[int]): If provided, the program will exit with this code.
    """
    error_type = type(e).__name__
    error_message = str(e)
    
    friendly_message = f"Oops! Something went wrong ({error_type})."
    if error_message:
        friendly_message += f"\nDetails: {error_message}"
    
    print(f"\n❌ {friendly_message}")
    print("Please check the logs for more detailed information.")
    
    log_exception(e, context="User-facing error")
    
    if exit_code is not None:
        print(f"Exiting with code {exit_code}.")
        sys.exit(exit_code)
