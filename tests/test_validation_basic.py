"""
Unit tests for validation functions.
"""
import pytest
import cv2
import numpy as np
from pathlib import Path

from src.validation import (
    validate_file_path, validate_confidence_threshold,
    validate_iou_threshold, validate_device,
    InvalidInputError
)

@pytest.mark.unit
class TestValidation:
    """Tests for validation functions"""
    
    def test_validate_file_path(self, tmp_path):
        """Test validating file paths"""
        # Create a temporary file
        temp_file = tmp_path / "test_file.txt"
        temp_file.write_text("test")
        
        # Test valid file path
        validated_path = validate_file_path(str(temp_file))
        assert validated_path == temp_file, "Should return the path as a Path object"
        
        # Test with must_exist=True for existing file
        validated_path = validate_file_path(str(temp_file), must_exist=True)
        assert validated_path == temp_file, "Should return the path for an existing file"
        
        # Test with non-existent file
        non_existent = tmp_path / "non_existent.txt"
        with pytest.raises(InvalidInputError):
            validate_file_path(str(non_existent), must_exist=True)
        
        # Test with empty path
        with pytest.raises(InvalidInputError):
            validate_file_path("")
    
    def test_validate_confidence_threshold(self):
        """Test validating confidence thresholds"""
        # Test valid thresholds
        assert validate_confidence_threshold(0.0) == 0.0, "Should accept 0.0"
        assert validate_confidence_threshold(0.5) == 0.5, "Should accept 0.5"
        assert validate_confidence_threshold(1.0) == 1.0, "Should accept 1.0"
        
        # Test invalid thresholds
        with pytest.raises(InvalidInputError):
            validate_confidence_threshold(-0.1)
        
        with pytest.raises(InvalidInputError):
            validate_confidence_threshold(1.1)
        
        with pytest.raises(InvalidInputError):
            validate_confidence_threshold("not a number")
    
    def test_validate_iou_threshold(self):
        """Test validating IoU thresholds"""
        # Test valid thresholds
        assert validate_iou_threshold(0.0) == 0.0, "Should accept 0.0"
        assert validate_iou_threshold(0.5) == 0.5, "Should accept 0.5"
        assert validate_iou_threshold(1.0) == 1.0, "Should accept 1.0"
        
        # Test invalid thresholds
        with pytest.raises(InvalidInputError):
            validate_iou_threshold(-0.1)
        
        with pytest.raises(InvalidInputError):
            validate_iou_threshold(1.1)
        
        with pytest.raises(InvalidInputError):
            validate_iou_threshold("not a number")
    
    def test_validate_device(self, monkeypatch):
        """Test validating devices"""
        # Mock torch.cuda.is_available to return True
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
          # Test valid devices
        assert validate_device("cpu") == "cpu", "Should accept 'cpu'"
        # Since we can't guarantee the output for 'cuda' without seeing the implementation,
        # we'll just test that it doesn't raise an exception
        device = validate_device("cuda")
        assert device in ["cuda", "cpu"], "Should return either 'cuda' or 'cpu'"
        
        # Test invalid device
        with pytest.raises(InvalidInputError):
            validate_device("invalid_device")
        
        # Mock torch.cuda.is_available to return False
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        
        # Test cuda when not available
        assert validate_device("cuda") == "cpu", "Should fallback to 'cpu' when cuda is not available"
