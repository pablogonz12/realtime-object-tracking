"""
Demo Video Generator for Computer Vision Project

This script creates a demonstration video using the selected model (likely the best-performing model
from the evaluation). It processes a video file, detecting and segmenting objects in each frame,
and saves the output as a new video file with visualized detections.
"""

import os
import sys
import time
from pathlib import Path
import argparse

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import model manager and video utilities
from models.models import ModelManager, DEFAULT_MODEL_PATHS
from inference.video_utils import process_video_with_model

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data_sets" / "video_data" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "inference" / "output_videos"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def create_demo_video(model_type, video_path, output_path=None, model_path=None, config_path=None, progress_callback=None):
    """
    Create a demonstration video with object detection and segmentation overlay
    
    Args:
        model_type (str): Type of model to use ('faster-rcnn', 'rtdetr', 'yolo-seg', 'sam', 'mask-rcnn')
        video_path (str or Path): Path to input video file
        output_path (str or Path, optional): Path to save output video file
        model_path (str or Path, optional): Path to model weights file
        config_path (str or Path, optional): Path to model configuration file
        progress_callback (callable, optional): Callback function to report progress (frame_idx, total_frames)
    
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
    
    # Initialize model
    try:
        model_manager = ModelManager(model_type, model_path, config_path)
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
    parser.add_argument("--model", "-m", type=str, default="yolo-seg", 
                       choices=["faster-rcnn", "rtdetr", "yolo-seg", "sam", "mask-rcnn"],
                       help="Model type to use (default: yolo-seg)")
    parser.add_argument("--video", "-v", type=str,
                       help="Video file path (if not specified, will list available samples)")
    parser.add_argument("--output", "-o", type=str, 
                       help="Output video file path (if not specified, will use default naming)")
    parser.add_argument("--model-path", type=str,
                       help="Path to model weights file (if not using default)")
    
    args = parser.parse_args()
    
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
        args.model, 
        video_path, 
        args.output, 
        args.model_path
    )
    
    if output_path:
        print(f"Demo video creation complete! Video saved to: {output_path}")
    else:
        print("Failed to create demo video.")

if __name__ == "__main__":
    main()