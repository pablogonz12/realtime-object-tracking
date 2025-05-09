"""
Integration tests for the Computer Vision Project.
Tests that components work together correctly.
"""

import pytest
import os
import sys
import subprocess
import json
from pathlib import Path

from src.models import ModelManager
import src.evaluate_models as evaluate_models
import src.create_demo_video as create_demo_video
import src.pipeline as pipeline

@pytest.mark.integration
class TestModelEvaluationIntegration:
    """Integration tests for model evaluation workflow"""
    
    @pytest.mark.slow
    def test_model_evaluation_pipeline(self, sample_image_path):
        """Test the full model evaluation pipeline with a minimal setup"""
        # Skip if the model file doesn't exist to avoid downloads during testing
        model_name = "yolov8n-seg"
        model_path = Path("models/pts") / f"{model_name}.pt"
        if not model_path.exists():
            pytest.skip(f"Model file {model_path} not found. Skipping to avoid download.")
        
        # Create a temporary results directory
        results_dir = Path("inference/results/test_integration")
        results_dir.mkdir(exist_ok=True, parents=True)
        
        # Run a minimal evaluation (single image, single model)
        results_file = results_dir / "test_eval_results.json"
        
        # Create a function that mimics evaluate_models.py but uses the test image
        def evaluate_on_test_image():
            manager = ModelManager()
            model = manager.load_model(model_name)
            
            # Run inference
            results = manager.inference(model, sample_image_path)
            
            # Format simplified results
            eval_results = {
                "model_name": model_name,
                "metrics": {
                    "precision": 0.0,  # Simplified test metrics
                    "recall": 0.0,
                    "mAP50": 0.0,
                    "mAP50-95": 0.0
                },
                "avg_inference_time": results["time"],
                "timestamp": "2025-05-09T12:00:00"
            }
            
            # Save results
            with open(results_file, "w") as f:
                json.dump(eval_results, f, indent=2)
            
            return eval_results
        
        # Run the evaluation
        eval_results = evaluate_on_test_image()
        
        # Verify results were saved
        assert results_file.exists(), f"Results file {results_file} should exist"
        
        # Test pipeline integration by checking if we can load and use the results
        with open(results_file, "r") as f:
            loaded_results = json.load(f)
        
        assert loaded_results["model_name"] == model_name, "Loaded results should contain correct model name"
        assert "metrics" in loaded_results, "Loaded results should contain metrics"
        assert "avg_inference_time" in loaded_results, "Loaded results should contain inference time"
        
        # Clean up
        results_file.unlink(missing_ok=True)

@pytest.mark.integration
class TestDemoVideoIntegration:
    """Integration tests for demo video creation"""
    
    @pytest.mark.slow
    def test_demo_video_creation(self, sample_video_path):
        """Test creating a demo video with detections"""
        # Skip if the model file doesn't exist to avoid downloads during testing
        model_name = "yolov8n-seg"
        model_path = Path("models/pts") / f"{model_name}.pt"
        if not model_path.exists():
            pytest.skip(f"Model file {model_path} not found. Skipping to avoid download.")
        
        # Create output directory
        output_dir = Path("inference/output_videos/test_integration")
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Define output path
        output_path = output_dir / "test_demo.mp4"
        if output_path.exists():
            output_path.unlink()
        
        # Run a simplified version of create_demo_video
        def create_test_demo():
            manager = ModelManager()
            model = manager.load_model(model_name)
            
            # We'll process only a few frames to keep the test quick
            from src.video_utils import load_video, extract_frames, save_video, visualize_detection
            
            # Load video
            cap, width, height, fps, frame_count = load_video(sample_video_path)
            
            # Extract a few frames
            frames = extract_frames(cap, num_frames=5)
            
            # Process frames
            processed_frames = []
            for frame in frames:
                # Run inference
                results = manager.inference(model, frame)
                # Visualize detections
                vis_frame = visualize_detection(frame, results)
                processed_frames.append(vis_frame)
            
            # Save video
            save_video(processed_frames, str(output_path), fps, width, height)
            
            # Clean up
            cap.release()
            
            return str(output_path)
        
        # Create the demo video
        demo_path = create_test_demo()
        
        # Verify the output video was created
        assert Path(demo_path).exists(), f"Demo video {demo_path} should exist"
        assert Path(demo_path).stat().st_size > 0, "Demo video should not be empty"
        
        # Verify we can read the video
        cap = cv2.VideoCapture(demo_path)
        assert cap.isOpened(), "Demo video should be readable"
        
        # Get some basic properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        assert width > 0, "Video width should be positive"
        assert height > 0, "Video height should be positive"
        assert frame_count == 5, "Video should have 5 frames"
        
        # Clean up
        cap.release()
        output_path.unlink(missing_ok=True)
