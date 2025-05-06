"""
Demo Video Generator for Computer Vision Project

This script creates a demonstration video using the selected model (likely the best-performing model
from the evaluation). It processes a video file, detecting and segmenting objects in each frame,
and saves the output as a new video file with visualized detections.
"""

import os
import sys
import time
import json
from pathlib import Path
import argparse

# Add the parent directory to sys.path for proper imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import project components - handle both relative and absolute imports
try:
    # Try relative imports first (when running from within src directory)
    from models import ModelManager, DEFAULT_MODEL_PATHS
    from video_utils import process_video_with_model
    # For pipeline functions, handle both import styles
    try:
        from pipeline import determine_best_model, find_latest_results
    except (ImportError, AttributeError):
        # If pipeline hasn't been fully imported yet, define the functions here
        def find_latest_results():
            """Find the latest evaluation results file"""
            result_files = list(RESULTS_DIR.glob("evaluation_results_*.json"))
            if not result_files:
                return None
            
            # Sort by modification time (newest first)
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            return latest_file
            
        def determine_best_model(results):
            """Simplified version for this file only"""
            if not results:
                return None, None
                
            # Use mAP and FPS as main criteria
            best_model = None
            best_score = -1
            best_details = {}
            
            for model_name, metrics in results.items():
                if "coco_metrics" not in metrics:
                    continue
                    
                coco_metrics = metrics["coco_metrics"]
                fps = metrics.get("fps", 0)
                mAP = coco_metrics.get("AP_IoU=0.50:0.95", 0)
                
                # Simple weighted score
                weighted_score = (mAP * 0.7) + ((min(fps, 30) / 30) * 0.3)
                
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_model = model_name
                    best_details = {
                        "mAP": mAP,
                        "FPS": fps,
                        "weighted_score": weighted_score
                    }
            
            return best_model, best_details
except ImportError:
    # Fall back to absolute imports (when running from project root)
    from src.models import ModelManager, DEFAULT_MODEL_PATHS
    from src.video_utils import process_video_with_model
    try:
        from src.pipeline import determine_best_model, find_latest_results
    except (ImportError, AttributeError):
        # Define inline versions if needed
        def find_latest_results():
            """Find the latest evaluation results file"""
            result_files = list(RESULTS_DIR.glob("evaluation_results_*.json"))
            if not result_files:
                return None
            
            # Sort by modification time (newest first)
            latest_file = max(result_files, key=lambda f: f.stat().st_mtime)
            return latest_file
            
        def determine_best_model(results):
            """Simplified version for this file only"""
            if not results:
                return None, None
                
            # Use mAP and FPS as main criteria
            best_model = None
            best_score = -1
            best_details = {}
            
            for model_name, metrics in results.items():
                if "coco_metrics" not in metrics:
                    continue
                    
                coco_metrics = metrics["coco_metrics"]
                fps = metrics.get("fps", 0)
                mAP = coco_metrics.get("AP_IoU=0.50:0.95", 0)
                
                # Simple weighted score
                weighted_score = (mAP * 0.7) + ((min(fps, 30) / 30) * 0.3)
                
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_model = model_name
                    best_details = {
                        "mAP": mAP,
                        "FPS": fps,
                        "weighted_score": weighted_score
                    }
            
            return best_model, best_details

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data_sets" / "video_data" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "inference" / "output_videos"
RESULTS_DIR = PROJECT_ROOT / "inference" / "results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def create_demo_video(model_type, video_path, output_path=None, model_path=None, config_path=None, progress_callback=None, 
                     conf_threshold=0.25, iou_threshold=0.45):
    """
    Create a demonstration video with object detection and segmentation overlay
    
    Args:
        model_type (str): Type of model to use ('mask-rcnn', 'yolo-seg')
        video_path (str or Path): Path to input video file
        output_path (str or Path, optional): Path to save output video file
        model_path (str or Path, optional): Path to model weights file
        config_path (str or Path, optional): Path to model configuration file
        progress_callback (callable, optional): Callback function to report progress (frame_idx, total_frames)
        conf_threshold (float): Confidence threshold for predictions (0-1)
        iou_threshold (float): IoU threshold for NMS (0-1)
    
    Returns:
        str: Path to the output video file
    """
    # Input validation
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return None
    
    # Determine output path if not specified
    if output_path is None:
        output_filename = f"{video_path.stem}_{model_type}_demo.mp4"
        output_path = OUTPUT_DIR / output_filename
    else:
        output_path = Path(output_path)
    
    # Determine model path if not specified
    if model_path is None and model_type in DEFAULT_MODEL_PATHS:
        model_path = DEFAULT_MODEL_PATHS[model_type]
    
    print(f"Processing video: {video_path}")
    print(f"Using model: {model_type}")
    print(f"Output will be saved to: {output_path}")
    print(f"Detection thresholds: confidence={conf_threshold}, IoU={iou_threshold}")
    
    # Initialize model
    try:
        model_manager = ModelManager(
            model_type, 
            model_path, 
            config_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        if model_manager.model_wrapper is None:
            print(f"Error: Failed to initialize {model_type} model")
            return None
    except Exception as e:
        print(f"Error initializing {model_type} model: {e}")
        return None
    
    # Define callback for progress reporting
    def processing_callback(frame, frame_count, total_frames, model_type, progress=None):
        if progress_callback and total_frames > 0 and frame_count % 30 == 0:
            progress_callback(frame_count, total_frames)
        # Show console progress every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            print(f"Progress: {progress:.1f}% (Frame {frame_count}/{total_frames})")
    
    # Process the video using the shared video utility
    stats = process_video_with_model(
        video_path=video_path,
        model_manager=model_manager,
        output_path=output_path,
        callback=processing_callback,
        stats_callback=None,
        add_fps=True
    )
    
    # Print summary statistics
    print("\nVideo Processing Summary:")
    print(f"- Total frames: {stats['frames_processed']} / {stats['total_frames']}")
    print(f"- Total processing time: {stats['processing_time']:.2f}s")
    print(f"- Average processing speed: {stats['actual_fps']:.2f} FPS")
    
    # Calculate and report detection statistics
    total_detections = sum(stats['detections'].values())
    print(f"- Total detections: {total_detections}")
    print(f"- Average detections per frame: {total_detections / max(stats['frames_processed'], 1):.2f}")
    
    # Show detection classes
    if stats['detections']:
        classes = sorted(stats['detections'].keys())
        print(f"- Detected classes: {', '.join(classes)}")
    else:
        print("- No detections found")
        
    print(f"- Output video saved to: {output_path}")
    
    return str(output_path)

def find_best_model_from_evaluation():
    """Find the best model from the latest evaluation results"""
    latest_results_file = find_latest_results()
    
    if not latest_results_file:
        print("No evaluation results found. Please run evaluate_models.py first.")
        return None
    
    print(f"Loading evaluation results from: {latest_results_file}")
    
    try:
        with open(latest_results_file, 'r') as f:
            results = json.load(f)
        
        best_model, details = determine_best_model(results)
        
        if best_model:
            print(f"\nBest model determined from evaluation: {best_model}")
            print(f"Selection criteria:")
            for criterion, value in details.items():
                print(f"- {criterion}: {value}")
            
            return best_model
        else:
            print("Could not determine best model from results.")
            return None
    
    except Exception as e:
        print(f"Error loading evaluation results: {e}")
        return None

def list_available_videos():
    """List videos available in the sample video directory"""
    if not VIDEO_DIR.is_dir():
        print(f"Error: Sample video directory not found: {VIDEO_DIR}")
        return []
    
    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    if not videos:
        print("No sample videos found.")
        return []
    
    print("\nAvailable sample videos:")
    for i, video in enumerate(videos, 1):
        print(f"{i}. {video.name}")
        
    return videos

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create demo video with object detection/segmentation")
    
    # Model selection options
    model_group = parser.add_argument_group("Model Selection")
    model_selection = model_group.add_mutually_exclusive_group()
    model_selection.add_argument("--model", "-m", type=str, 
                       help="Model type to use (e.g., yolov8n-seg)")
    model_selection.add_argument("--best-model", action="store_true",
                       help="Use best model from latest evaluation results")
    model_group.add_argument("--model-path", type=str,
                       help="Path to model weights file (if not using default)")
    
    # Video options
    video_group = parser.add_argument_group("Video Options")
    video_group.add_argument("--video", "-v", type=str,
                       help="Video file path (if not specified, will list available samples)")
    video_group.add_argument("--output", "-o", type=str, 
                       help="Output video file path (if not specified, will use default naming)")
    
    # Detection parameters
    detection_group = parser.add_argument_group("Detection Parameters")
    detection_group.add_argument("--conf-threshold", type=float, default=0.25,
                       help="Confidence threshold for detections (default: 0.25)")
    detection_group.add_argument("--iou-threshold", type=float, default=0.45,
                       help="IoU threshold for NMS (default: 0.45)")
    
    # Other options
    other_group = parser.add_argument_group("Other Options")
    other_group.add_argument("--list-models", action="store_true",
                       help="List available models and exit")
    
    args = parser.parse_args()
    
    # List available models if requested
    if args.list_models:
        print("\nAvailable models:")
        for i, (model_key, path) in enumerate(DEFAULT_MODEL_PATHS.items(), 1):
            print(f"{i}. {model_key}")
        return
    
    # Determine model type from args
    model_type = None
    
    if args.best_model:
        model_type = find_best_model_from_evaluation()
        if not model_type:
            print("Could not determine best model. Please specify a model with --model.")
            return
    elif args.model:
        model_type = args.model
    else:
        # No model specified - show available models and ask user to choose
        print("\nAvailable models:")
        model_keys = list(DEFAULT_MODEL_PATHS.keys())
        for i, model_key in enumerate(model_keys, 1):
            print(f"{i}. {model_key}")
            
        try:
            selection = int(input("\nEnter model number to use (or 0 to exit): "))
            if selection == 0:
                print("Exiting...")
                return
            if selection < 1 or selection > len(model_keys):
                print(f"Invalid selection. Please choose a number between 1 and {len(model_keys)}")
                return
            
            model_type = model_keys[selection - 1]
            print(f"Selected model: {model_type}")
        except ValueError:
            print("Invalid input. Please enter a number.")
            return
    
    # If video not specified, show list of available videos
    if args.video is None:
        videos = list_available_videos()
        if not videos:
            return
        
        # Ask user to select a video
        try:
            selection = int(input("\nEnter video number to process (or 0 to exit): "))
            if selection == 0:
                print("Exiting...")
                return
            if selection < 1 or selection > len(videos):
                print(f"Invalid selection. Please choose a number between 1 and {len(videos)}")
                return
            
            video_path = videos[selection - 1]
        except ValueError:
            print("Invalid input. Please enter a number.")
            return
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            return
    
    # Create the demo video
    output_path = create_demo_video(
        model_type, 
        video_path, 
        args.output, 
        args.model_path,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )
    
    if output_path:
        print(f"Demo video creation complete! Video saved to: {output_path}")
    else:
        print("Failed to create demo video.")

if __name__ == "__main__":
    main()