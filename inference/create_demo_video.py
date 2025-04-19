"""
Demo Video Generator for Computer Vision Project

This script creates a demonstration video using the selected model (likely the best-performing model
from the evaluation). It processes a video file, detecting and segmenting objects in each frame,
and saves the output as a new video file with visualized detections.
"""

import os
import sys
import time
import cv2
import numpy as np
from pathlib import Path
import argparse

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import model manager
from models.models import ModelManager, DEFAULT_MODEL_PATHS

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = PROJECT_ROOT / "data_sets" / "video_data" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "inference" / "output_videos"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

def create_demo_video(model_type, video_path, output_path=None, model_path=None, config_path=None):
    """
    Create a demonstration video with object detection and segmentation overlay
    
    Args:
        model_type (str): Type of model to use ('faster-rcnn', 'rtdetr', 'yolo-seg')
        video_path (str or Path): Path to input video file
        output_path (str or Path, optional): Path to save output video file
        model_path (str or Path, optional): Path to model weights file
        config_path (str or Path, optional): Path to model configuration file
    
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
    
    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return None
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4 format
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Initialize counters and timing variables
    frame_count = 0
    processed_frame_count = 0
    total_detection_count = 0
    detection_classes = set()
    start_time = time.time()
    
    # Process each frame
    print(f"Starting video processing with {model_type} model...")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Show progress
            if frame_count % 30 == 0:  # Update every 30 frames
                elapsed_time = time.time() - start_time
                fps_processing = frame_count / elapsed_time if elapsed_time > 0 else 0
                progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                print(f"Progress: {progress:.1f}% (Frame {frame_count}/{total_frames}, "
                      f"Processing Speed: {fps_processing:.1f} FPS)")
            
            # Process frame with model
            try:
                detections, segmentations, annotated_frame = model_manager.predict(frame)
                processed_frame_count += 1
                
                # Count detections
                total_detection_count += len(detections)
                
                # Track unique detection classes
                for det in detections:
                    detection_classes.add(det.get("class_name", "unknown"))
                
                # Add info overlay to the top of the frame
                overlay = annotated_frame.copy()
                
                # Create semi-transparent rectangle for info panel
                cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
                
                # Add model info text
                info_text = f"Model: {model_type.upper()} | "
                if model_type == "yolo-seg":
                    info_text += f"Detections: {len(detections)} | Segmentations: {len(segmentations) if segmentations else 0} | "
                else:
                    info_text += f"Detections: {len(detections)} | "
                info_text += f"Frame: {frame_count}/{total_frames}"
                
                # Blend the overlay with the frame
                alpha = 0.7
                cv2.addWeighted(overlay, alpha, annotated_frame, 1 - alpha, 0, annotated_frame)
                
                # Add text to the blended image
                cv2.putText(annotated_frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame to output video
                out.write(annotated_frame)
                
                # Save sample frames for documentation (every 100 frames)
                if frame_count % 100 == 0:
                    sample_dir = OUTPUT_DIR / "samples"
                    sample_dir.mkdir(exist_ok=True)
                    cv2.imwrite(str(sample_dir / f"{model_type}_frame_{frame_count}.jpg"), annotated_frame)
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                # Write original frame on error
                cv2.putText(frame, f"Processing Error - Frame {frame_count}", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                out.write(frame)
    
    except KeyboardInterrupt:
        print("Processing interrupted by user.")
    except Exception as e:
        print(f"Error during video processing: {e}")
    finally:
        # Release resources
        cap.release()
        out.release()
        
        # Print summary statistics
        elapsed_time = time.time() - start_time
        print("\nVideo Processing Summary:")
        print(f"- Total frames: {frame_count}")
        print(f"- Processed frames: {processed_frame_count}")
        print(f"- Total processing time: {elapsed_time:.2f}s")
        print(f"- Average processing speed: {frame_count / elapsed_time:.2f} FPS")
        print(f"- Total detections: {total_detection_count}")
        print(f"- Average detections per frame: {total_detection_count / max(processed_frame_count, 1):.2f}")
        print(f"- Detected classes: {', '.join(sorted(detection_classes))}")
        print(f"- Output video saved to: {output_path}")
        
        return str(output_path)

def list_available_videos():
    """List videos available in the sample video directory"""
    if not VIDEO_DIR.exists():
        print(f"Error: Video directory not found: {VIDEO_DIR}")
        return []
    
    videos = list(VIDEO_DIR.glob("*.mp4")) + list(VIDEO_DIR.glob("*.avi")) + list(VIDEO_DIR.glob("*.mov"))
    
    if not videos:
        print(f"No video files found in {VIDEO_DIR}")
        return []
    
    print(f"Found {len(videos)} sample videos:")
    for i, video in enumerate(videos):
        print(f"{i+1}. {video.name}")
    
    return videos

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create demo video with object detection/segmentation")
    parser.add_argument("--model", "-m", type=str, default="yolo-seg", 
                       choices=["faster-rcnn", "rtdetr", "yolo-seg"],
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