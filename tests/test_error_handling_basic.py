"""
Basic unit tests for error handling utilities.
"""

import pytest
import os
import logging
from pathlib import Path

from src.error_handling import (
    log_info, log_warning, log_error, log_exception
)

@pytest.mark.unit
class TestErrorHandling:
    """Tests for error handling functions"""
    
    def test_logger_exists(self):
        """Test that the logger exists"""
        # Import the logger directly
        from src.error_handling import logger
        
        assert logger is not None, "Logger should exist"
        assert isinstance(logger, logging.Logger), "Logger should be a Logger instance"
        assert logger.name == "cvproject", "Logger should have the correct name"
    
    def test_logging_functions(self, caplog):
        """Test that logging functions work"""
        # Set log level to capture all messages
        caplog.set_level(logging.INFO)
        
        # Test each logging function
        test_message = "This is a test message"
        
        log_info(test_message)
        assert test_message in caplog.text, "Info message should be logged"
        
        log_warning(test_message)
        assert test_message in caplog.text, "Warning message should be logged"
        
        log_error(test_message)
        assert test_message in caplog.text, "Error message should be logged"
        
        # Test exception logging
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            log_exception(e)
        
        assert "Test exception" in caplog.text, "Exception should be logged"
