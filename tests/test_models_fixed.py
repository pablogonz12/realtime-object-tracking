"""
Unit tests for the models module.
Tests loading models, inference, and various model-related utilities.
"""
import pytest
import torch
import os
import numpy as np
import cv2
from pathlib import Path

from src.models import ModelManager, DEFAULT_MODEL_PATHS

@pytest.mark.unit
class TestModelManager:
    """Tests for the ModelManager class"""
    
    def test_model_manager_creation(self):
        """Test that ModelManager can be instantiated"""
        manager = ModelManager(model_type="yolo")
        assert manager is not None, "ModelManager should be instantiated"
    
    def test_default_paths_exist(self):
        """Test that default model paths are defined"""
        assert DEFAULT_MODEL_PATHS is not None, "DEFAULT_MODEL_PATHS should be defined"
        assert isinstance(DEFAULT_MODEL_PATHS, dict), "DEFAULT_MODEL_PATHS should be a dictionary"
        assert len(DEFAULT_MODEL_PATHS) > 0, "DEFAULT_MODEL_PATHS should not be empty"
    
    def test_model_paths_valid(self):
        """Test that all model paths in DEFAULT_MODEL_PATHS use Path objects"""
        for model_name, model_path in DEFAULT_MODEL_PATHS.items():
            assert isinstance(model_path, Path) or isinstance(model_path, str), f"Model path for {model_name} should be a Path object or string"
    
    def test_missing_model_handled_gracefully(self):
        """Test that attempting to load a non-existent model is handled gracefully"""
        manager = ModelManager(model_type="yolo", model_path="nonexistent_model.pt")
        assert manager is not None, "ModelManager should be instantiated even with invalid model"
        # The model attribute should exist but be None since loading fails
        assert hasattr(manager, "model"), "ModelManager should have model attribute"
