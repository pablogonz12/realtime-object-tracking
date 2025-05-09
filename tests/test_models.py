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
        manager = ModelManager()
        assert manager is not None, "ModelManager should be instantiated"
    
    def test_default_paths_exist(self):
        """Test that default model paths are defined"""
        assert DEFAULT_MODEL_PATHS is not None, "DEFAULT_MODEL_PATHS should be defined"
        assert isinstance(DEFAULT_MODEL_PATHS, dict), "DEFAULT_MODEL_PATHS should be a dictionary"
        assert len(DEFAULT_MODEL_PATHS) > 0, "DEFAULT_MODEL_PATHS should not be empty"
        model = manager.load_model(model_name)
        assert model is not None, f"Failed to load model {model_name}"
    
    @pytest.mark.parametrize("model_name", ["yolov8n-seg"])
    def test_model_info(self, model_name):
        """Test getting model info"""
        # Skip if model file doesn't exist
        model_path = DEFAULT_MODEL_PATHS.get(model_name)
        if not os.path.exists(model_path):
            pytest.skip(f"Model file {model_path} not found. Skipping to avoid download.")
            
        manager = ModelManager()
        model_info = manager.get_model_info(model_name)
        assert isinstance(model_info, dict), "Model info should be a dictionary"
        assert "name" in model_info, "Model info should contain 'name'"
        assert "type" in model_info, "Model info should contain 'type'"
    
    def test_available_models(self):
        """Test getting the list of available models"""
        manager = ModelManager()
        available_models = manager.get_available_models()
        assert isinstance(available_models, list), "Available models should be a list"
        assert len(available_models) > 0, "There should be at least one available model"
    
    @pytest.mark.parametrize("model_name", ["yolov8n-seg"])
    def test_inference_with_image_path(self, sample_image_path, model_name):
        """Test model inference with an image path"""
        # Skip if model file doesn't exist
        model_path = DEFAULT_MODEL_PATHS.get(model_name)
        if not os.path.exists(model_path):
            pytest.skip(f"Model file {model_path} not found. Skipping to avoid download.")
        
        manager = ModelManager()
        model = manager.load_model(model_name)
        results = manager.inference(model, sample_image_path)
        
        assert results is not None, "Inference should return results"
        assert "boxes" in results, "Results should contain 'boxes'"
        assert "scores" in results, "Results should contain 'scores'"
        assert "labels" in results, "Results should contain 'labels'"
        assert "masks" in results, "Results should contain 'masks'"
        assert "time" in results, "Results should contain 'time'"
    
    @pytest.mark.parametrize("model_name", ["yolov8n-seg"])
    def test_inference_with_numpy_image(self, sample_image_path, model_name):
        """Test model inference with a numpy image"""
        # Skip if model file doesn't exist
        model_path = DEFAULT_MODEL_PATHS.get(model_name)
        if not os.path.exists(model_path):
            pytest.skip(f"Model file {model_path} not found. Skipping to avoid download.")
        
        # Load the sample image as numpy array
        img = cv2.imread(sample_image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
        manager = ModelManager()
        model = manager.load_model(model_name)
        results = manager.inference(model, img)
        
        assert results is not None, "Inference should return results"
        assert "boxes" in results, "Results should contain 'boxes'"
    
    @pytest.mark.gpu
    @pytest.mark.parametrize("model_name", ["yolov8n-seg"])
    def test_gpu_inference(self, sample_image_path, model_name):
        """Test model inference on GPU if available"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
            
        # Skip if model file doesn't exist
        model_path = DEFAULT_MODEL_PATHS.get(model_name)
        if not os.path.exists(model_path):
            pytest.skip(f"Model file {model_path} not found. Skipping to avoid download.")
        
        manager = ModelManager()
        model = manager.load_model(model_name, device="cuda")
        results = manager.inference(model, sample_image_path)
        
        assert results is not None, "GPU inference should return results"
        assert "boxes" in results, "Results should contain 'boxes'"
