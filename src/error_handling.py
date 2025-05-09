"""
Error handling and logging utilities for the Computer Vision Project.
Provides functions to log errors and handle them gracefully.
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict, List, Union, Callable

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
    """Log an informational message"""
    logger.info(message)


def log_warning(message: str) -> None:
    """Log a warning message"""
    logger.warning(message)


def log_error(message: str, exc_info: bool = False) -> None:
    """Log an error message"""
    logger.error(message, exc_info=exc_info)


def log_exception(e: Exception, context: Optional[str] = None) -> None:
    """
    Log an exception with context.
    
    Args:
        e: The exception to log
        context: Optional context string to include in the log
    """
    if context:
        logger.error(f"{context}: {str(e)}", exc_info=True)
    else:
        logger.error(str(e), exc_info=True)


def handle_exceptions(func: Callable) -> Callable:
    """
    Decorator to handle exceptions in functions.
    Logs the exception and raises a friendly error message.
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function that handles exceptions
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the exception
            log_exception(e, f"Error in {func.__name__}")
            
            # Re-raise with a more user-friendly message
            raise type(e)(f"Error in {func.__name__}: {str(e)}") from e
    
    return wrapper


def with_error_handling(message: str = "An error occurred") -> Callable:
    """
    More configurable decorator for exception handling.
    
    Args:
        message: Custom error message prefix
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the exception
                log_exception(e, f"{message} in {func.__name__}")
                
                # Re-raise with a more user-friendly message
                raise type(e)(f"{message} in {func.__name__}: {str(e)}") from e
        
        return wrapper
    
    return decorator


def robust_function(fallback_value: Any = None) -> Callable:
    """
    Decorator for functions that should never crash.
    Returns a fallback value if an exception occurs.
    
    Args:
        fallback_value: Value to return if an exception occurs
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the exception
                log_exception(e, f"Error in {func.__name__} (fallback used)")
                
                # Return the fallback value
                return fallback_value
        
        return wrapper
    
    return decorator


def print_friendly_error(e: Exception, exit_code: Optional[int] = None) -> None:
    """
    Print a user-friendly error message and optionally exit.
    
    Args:
        e: The exception to print
        exit_code: If provided, the program will exit with this code
    """
    from src.validation import CVProjectError
    
    # Different handling for different error types
    if isinstance(e, CVProjectError):
        # These are our custom errors with friendly messages
        print(f"\nError: {str(e)}")
    else:
        # Generic errors get a more technical message
        print(f"\nAn unexpected error occurred: {str(e)}")
        print("See the log file for more details.")
    
    # Exit if requested
    if exit_code is not None:
        sys.exit(exit_code)
