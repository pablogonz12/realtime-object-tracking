"""
Unit tests for validation and error handling utilities.
"""
import pytest
import numpy as np
import torch
from pathlib import Path

from src.validation import (
    validate_file_path, validate_model_name, validate_image, validate_video,
    validate_confidence_threshold, validate_iou_threshold, validate_device,
    InvalidInputError
)

@pytest.mark.unit
class TestValidation:
    """Tests for input validation functions"""
    
    def test_validate_file_path(self, sample_image_path):
        """Test file path validation"""
        # Test with an existing file
        path = validate_file_path(sample_image_path)
        assert isinstance(path, Path), "Should return a Path object"
        
        # Test with a non-existent file
        with pytest.raises(InvalidInputError):
            validate_file_path("non_existent_file.txt")
        
        # Test with a non-existent file, but must_exist=False
        path = validate_file_path("non_existent_file.txt", must_exist=False)
        assert isinstance(path, Path), "Should return a Path object"
        
        # Test with an empty path
        with pytest.raises(InvalidInputError):
            validate_file_path("")
    
    def test_validate_model_name(self):
        """Test model name validation"""
        available_models = ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg"]
        
        # Test with a valid model name
        model_name = validate_model_name("yolov8n-seg", available_models)
        assert model_name == "yolov8n-seg", "Should return the same model name"
        
        # Test with an invalid model name
        with pytest.raises(InvalidInputError):
            validate_model_name("invalid_model", available_models)
        
        # Test with an empty model name
        with pytest.raises(InvalidInputError):
            validate_model_name("", available_models)
    
    def test_validate_image(self, sample_image_path):
        """Test image validation"""
        # Test with a valid image path
        image = validate_image(sample_image_path)
        assert isinstance(image, str), "Should return a string path"
        
        # Test with a valid numpy array
        img = np.ones((640, 640, 3), dtype=np.uint8)
        image = validate_image(img)
        assert isinstance(image, np.ndarray), "Should return the numpy array"
        
        # Test with an invalid image path
        with pytest.raises(InvalidInputError):
            validate_image("non_existent_image.jpg")
        
        # Test with an invalid numpy array (wrong dimensions)
        img = np.ones((640, 640), dtype=np.uint8)  # 2D instead of 3D
        with pytest.raises(InvalidInputError):
            validate_image(img)
        
        # Test with an invalid numpy array (wrong channels)
        img = np.ones((640, 640, 4), dtype=np.uint8)  # 4 channels instead of 3
        with pytest.raises(InvalidInputError):
            validate_image(img)
        
        # Test with an invalid type
        with pytest.raises(InvalidInputError):
            validate_image(123)
    
    def test_validate_confidence_threshold(self):
        """Test confidence threshold validation"""
        # Test with valid values
        assert validate_confidence_threshold(0.0) == 0.0
        assert validate_confidence_threshold(0.5) == 0.5
        assert validate_confidence_threshold(1.0) == 1.0
        
        # Test with invalid values
        with pytest.raises(InvalidInputError):
            validate_confidence_threshold(-0.1)
        
        with pytest.raises(InvalidInputError):
            validate_confidence_threshold(1.1)
        
        with pytest.raises(InvalidInputError):
            validate_confidence_threshold("0.5")
    
    def test_validate_iou_threshold(self):
        """Test IoU threshold validation"""
        # Test with valid values
        assert validate_iou_threshold(0.0) == 0.0
        assert validate_iou_threshold(0.5) == 0.5
        assert validate_iou_threshold(1.0) == 1.0
        
        # Test with invalid values
        with pytest.raises(InvalidInputError):
            validate_iou_threshold(-0.1)
        
        with pytest.raises(InvalidInputError):
            validate_iou_threshold(1.1)
        
        with pytest.raises(InvalidInputError):
            validate_iou_threshold("0.5")
    
    def test_validate_device(self):
        """Test device validation"""
        # Test with valid values
        assert validate_device("cpu") == "cpu"
        
        # Test CUDA with mock availability
        original_cuda_is_available = torch.cuda.is_available
        try:
            # Mock CUDA availability
            torch.cuda.is_available = lambda: True
            assert validate_device("cuda") == "cuda"
            
            # Mock CUDA unavailability
            torch.cuda.is_available = lambda: False
            assert validate_device("cuda") == "cpu"  # Should fall back to CPU
        finally:
            # Restore original function
            torch.cuda.is_available = original_cuda_is_available
        
        # Test with invalid value
        with pytest.raises(InvalidInputError):
            validate_device("invalid_device")
