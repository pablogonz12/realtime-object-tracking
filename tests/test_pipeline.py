"""
End-to-end test for the entire Computer Vision Project pipeline.
Tests the complete workflow from model loading to evaluation and demo video creation.
"""

import pytest
import os
import sys
import tempfile
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.models import ModelManager
from src.validation import InvalidInputError


@pytest.mark.integration
class TestPipeline:
    """Test the complete pipeline from end to end"""
    
    @pytest.mark.slow
    def test_pipeline_minimal(self, sample_image_path, sample_video_path):
        """
        Test a minimal version of the entire pipeline.
        This test:
        1. Loads a model
        2. Performs inference on a sample image
        3. Creates a demo video with the model
        """
        # Skip if model files don't exist
        model_name = "yolov8n-seg"
        model_path = Path("models/pts") / f"{model_name}.pt"
        if not model_path.exists():
            pytest.skip(f"Model file {model_path} not found. Skipping to avoid download.")
        
        try:
            # 1. Load model
            manager = ModelManager()
            model = manager.load_model(model_name)
            assert model is not None, "Failed to load model"
            
            # 2. Perform inference on a sample image
            results = manager.inference(model, sample_image_path)
            assert results is not None, "Inference failed"
            assert "boxes" in results, "Results should contain 'boxes'"
            assert "scores" in results, "Results should contain 'scores'"
            assert "labels" in results, "Results should contain 'labels'"
            assert "masks" in results, "Results should contain 'masks'"
            
            # 3. Process a video (minimal version to keep the test fast)
            with tempfile.TemporaryDirectory() as temp_dir:
                # Use a temporary directory for output
                output_path = Path(temp_dir) / "output.mp4"
                
                # Import video processing utilities
                from src.video_utils import load_video, extract_frames, save_video
                
                # Load the sample video
                cap, width, height, fps, frame_count = load_video(sample_video_path)
                
                # Extract a small number of frames to keep the test fast
                frames = extract_frames(cap, num_frames=5)
                
                # Process each frame with the model
                processed_frames = []
                for frame in frames:
                    # Run inference
                    frame_results = manager.inference(model, frame)
                    
                    # Visualize detections
                    from src.video_utils import visualize_detection
                    vis_frame = visualize_detection(frame, frame_results)
                    processed_frames.append(vis_frame)
                
                # Save the processed frames to a video
                save_video(processed_frames, str(output_path), fps, width, height)
                
                # Verify the output video was created
                assert output_path.exists(), "Output video should exist"
                assert output_path.stat().st_size > 0, "Output video should not be empty"
                
                # Clean up
                cap.release()
        
        except Exception as e:
            pytest.fail(f"Pipeline test failed with error: {str(e)}")
    
    @pytest.mark.slow
    def test_pipeline_error_handling(self, sample_image_path):
        """Test that the pipeline handles errors gracefully"""
        # Test with an invalid model name
        with pytest.raises(InvalidInputError):
            manager = ModelManager()
            manager.load_model("invalid_model_name")
        
        # Test with an invalid confidence threshold
        model_name = "yolov8n-seg"
        model_path = Path("models/pts") / f"{model_name}.pt"
        if not model_path.exists():
            pytest.skip(f"Model file {model_path} not found. Skipping to avoid download.")
        
        with pytest.raises(InvalidInputError):
            from src.validation import validate_confidence_threshold
            validate_confidence_threshold(-0.1)
        
        # Test with an invalid IoU threshold
        with pytest.raises(InvalidInputError):
            from src.validation import validate_iou_threshold
            validate_iou_threshold(1.1)
        
        # Test with an invalid device
        with pytest.raises(InvalidInputError):
            from src.validation import validate_device
            validate_device("invalid_device")
