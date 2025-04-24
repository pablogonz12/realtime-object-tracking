"""
Video Processing Utilities for Computer Vision Project

This module provides shared utilities for processing video files with computer vision models.
It standardizes the visualization and processing logic across all parts of the application.
"""

import cv2
import numpy as np
import time
from pathlib import Path
import threading

def process_video_with_model(
    video_path, 
    model_manager, 
    output_path=None, 
    callback=None, 
    stats_callback=None, 
    stop_event=None,
    pause_event=None,
    add_fps=True
):
    """
    Process a video with the specified model and generate outputs.
    
    Args:
        video_path (str or Path): Path to the input video file
        model_manager: Model manager instance with predict() method
        output_path (str or Path, optional): Path to save the processed video (if None, won't save)
        callback (callable, optional): Callback function for frame updates 
                                       callback(annotated_frame, frame_count, total_frames, model_type)
        stats_callback (callable, optional): Callback function for final statistics
        stop_event (threading.Event, optional): Event to signal process termination
        pause_event (threading.Event, optional): Event to signal process pausing
        add_fps (bool): Whether to add FPS counter to video
        
    Returns:
        dict: Processing statistics
    """
    # Initialize statistics
    stats = {
        "frames_processed": 0,
        "total_frames": 0,
        "processing_time": 0.0,
        "actual_fps": 0.0,
        "detections": {},
        "model_type": model_manager.model_type,
        "model_device": model_manager.device
    }
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video: {video_path}")
        return stats
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stats["total_frames"] = total_frames
    
    # Initialize video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Initialize counters
    frame_count = 0
    processing_errors = 0
    start_time = time.time()
    
    try:
        # Process each frame
        while cap.isOpened():
            # Check if we should stop
            if stop_event and stop_event.is_set():
                print("Stopping video processing due to stop event")
                break
                
            # Check if we should pause
            if pause_event and pause_event.is_set():
                time.sleep(0.1)  # Sleep briefly to avoid CPU spinning
                continue
            
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # Update progress if callback provided
            if callback and frame_count % 5 == 0:  # Update every 5 frames
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0
                callback(None, frame_count, total_frames, model_manager.model_type, progress=progress)
            
            try:
                # Process frame based on model type
                if model_manager.model_type == "sam":
                    # For SAM, create a grid of prompt points
                    h, w = frame.shape[:2]
                    grid_size = 3
                    y_points = np.linspace(h // 4, 3 * h // 4, grid_size, dtype=int)
                    x_points = np.linspace(w // 4, 3 * w // 4, grid_size, dtype=int)
                    input_points = np.array([[x, y] for y in y_points for x in x_points])
                    input_labels = np.ones(len(input_points), dtype=np.int32)
                    
                    # Run prediction with points and labels
                    results = model_manager.predict(frame, input_points=input_points, input_labels=input_labels)
                else:
                    # For other models (YOLO-Seg, Mask R-CNN)
                    results = model_manager.predict(frame)
                    
                # Unpack results
                if results and len(results) == 3:
                    detections, segmentations, annotated_frame = results
                    
                    # Count detections by class for statistics
                    for det in detections:
                        class_name = det.get('class_name', 'unknown')
                        stats["detections"][class_name] = stats["detections"].get(class_name, 0) + 1
                    
                    # Add overlay with more information if it's not already there
                    if add_fps:
                        # Add semi-transparent overlay at the top
                        overlay = annotated_frame.copy()
                        cv2.rectangle(overlay, (0, 0), (width, 40), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
                        
                        # Add model info and frame count
                        elapsed = time.time() - start_time
                        if elapsed > 0:
                            current_fps = frame_count / elapsed
                        else:
                            current_fps = 0
                        
                        info_text = f"Model: {model_manager.model_type.upper()} | "
                        info_text += f"FPS: {current_fps:.1f} | "
                        info_text += f"Frame: {frame_count}/{total_frames}"
                        
                        cv2.putText(annotated_frame, info_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                    # Write to output video if enabled
                    if out:
                        out.write(annotated_frame)
                    
                    # Update via callback
                    if callback:
                        callback(annotated_frame, frame_count, total_frames, model_manager.model_type)
                else:
                    # Handle invalid results
                    msg = "Model returned unexpected results format"
                    print(f"Error: {msg}")
                    annotated_frame = frame.copy()
                    cv2.putText(annotated_frame, msg, (30, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    if out:
                        out.write(annotated_frame)
                    
                    if callback:
                        callback(annotated_frame, frame_count, total_frames, model_manager.model_type)
                        
            except Exception as e:
                processing_errors += 1
                print(f"Error processing frame {frame_count}: {e}")
                import traceback
                traceback.print_exc()
                
                # Show the error frame
                error_frame = frame.copy()
                cv2.putText(error_frame, f"Error: {str(e)[:50]}", (30, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                if out:
                    out.write(error_frame)
                    
                if callback:
                    callback(error_frame, frame_count, total_frames, model_manager.model_type)
            
            # Small delay for UI updates if needed
            time.sleep(0.01)
    
    except Exception as e:
        print(f"Fatal error in video processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Update statistics
        stats["frames_processed"] = frame_count
        stats["processing_time"] = time.time() - start_time
        stats["actual_fps"] = frame_count / stats["processing_time"] if stats["processing_time"] > 0 else 0
        
        # Clean up resources
        cap.release()
        if out:
            out.write
            out.release()
            
        # Provide final statistics
        if stats_callback:
            stats_callback(stats)
            
        return stats


def process_webcam_with_model(
    model_manager, 
    callback=None, 
    stats_callback=None, 
    stop_event=None,
    pause_event=None,
    camera_id=0,
    process_width=640,
    process_height=480,
    frame_skip=2
):
    """
    Process webcam input with the specified model in real-time.
    
    Args:
        model_manager: Model manager instance with predict() method
        callback (callable, optional): Callback function for frame updates
        stats_callback (callable, optional): Callback function for final statistics
        stop_event (threading.Event, optional): Event to signal process termination
        pause_event (threading.Event, optional): Event to signal process pausing
        camera_id (int): Camera device ID
        process_width (int): Width for processing (resize for performance)
        process_height (int): Height for processing (resize for performance)
        frame_skip (int): Process every Nth frame
        
    Returns:
        dict: Processing statistics
    """
    # Initialize statistics
    stats = {
        "frames_processed": 0,
        "total_frames": 0,
        "processing_time": 0.0,
        "actual_fps": 0.0,
        "detections": {},
        "model_type": model_manager.model_type,
        "model_device": model_manager.device
    }
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Cannot open webcam with ID {camera_id}")
        return stats
    
    # Get webcam properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Performance optimizations
    webcam_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    target_display_fps = min(15, webcam_fps / frame_skip)
    frame_interval_ms = int(1000 / target_display_fps) if target_display_fps > 0 else 100
    
    # Initialize counters
    frame_count = 0
    total_frames = 0
    processing_errors = 0
    start_time = time.time()
    last_update_time = 0
    
    try:
        # Process frames
        while cap.isOpened():
            # Check if we should stop
            if stop_event and stop_event.is_set():
                print("Stopping webcam processing due to stop event")
                break
                
            # Check if we should pause
            if pause_event and pause_event.is_set():
                time.sleep(0.1)  # Sleep briefly to avoid CPU spinning
                continue
            
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                print("Error reading from webcam")
                break
                
            total_frames += 1
            
            # Apply frame skipping
            if total_frames % frame_skip != 0:
                continue
            
            frame_count += 1
            current_time = int(time.time() * 1000)
            
            try:
                # Resize frame for processing
                frame_resized = cv2.resize(frame, (process_width, process_height))
                
                # Process frame based on model type
                if model_manager.model_type == "sam":
                    # For SAM, create a grid of prompt points
                    h, w = frame_resized.shape[:2]
                    grid_size = 3
                    y_points = np.linspace(h // 4, 3 * h // 4, grid_size, dtype=int)
                    x_points = np.linspace(w // 4, 3 * w // 4, grid_size, dtype=int)
                    input_points = np.array([[x, y] for y in y_points for x in x_points])
                    input_labels = np.ones(len(input_points), dtype=np.int32)
                    
                    # Run prediction with points and labels
                    results = model_manager.predict(frame_resized, input_points=input_points, input_labels=input_labels)
                else:
                    # For other models (YOLO-Seg, Mask R-CNN)
                    results = model_manager.predict(frame_resized)
                
                # Unpack results
                if results and len(results) == 3:
                    detections, segmentations, annotated_frame = results
                    
                    # Count detections by class for statistics
                    for det in detections:
                        class_name = det.get('class_name', 'Unknown')
                        if class_name != 'Unknown':
                            stats["detections"][class_name] = stats["detections"].get(class_name, 0) + 1
                    
                    # Limit display update rate
                    if (current_time - last_update_time) >= frame_interval_ms:
                        # Add overlay with more information
                        overlay = annotated_frame.copy()
                        cv2.rectangle(overlay, (0, 0), (process_width, 40), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
                        
                        # Add model info and stats
                        elapsed = time.time() - start_time
                        current_fps = frame_count / elapsed if elapsed > 0 else 0
                        
                        info_text = f"Model: {model_manager.model_type.upper()} | "
                        info_text += f"FPS: {current_fps:.1f} | "
                        info_text += f"Frame: {frame_count}"
                        
                        cv2.putText(annotated_frame, info_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Update via callback
                        if callback:
                            callback(annotated_frame, frame_count, total_frames, model_manager.model_type)
                        
                        last_update_time = current_time
                else:
                    # Handle invalid results if update interval passed
                    if (current_time - last_update_time) >= frame_interval_ms:
                        msg = "Model returned unexpected results format"
                        print(f"Error: {msg}")
                        annotated_frame = frame_resized.copy()
                        cv2.putText(annotated_frame, msg, (30, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        if callback:
                            callback(annotated_frame, frame_count, total_frames, model_manager.model_type)
                        
                        last_update_time = current_time
            
            except Exception as e:
                processing_errors += 1
                print(f"Error processing webcam frame {frame_count}: {e}")
                
                # Only show errors occasionally to avoid spamming
                if processing_errors <= 3 or processing_errors % 50 == 0:
                    if (current_time - last_update_time) >= frame_interval_ms:
                        error_frame = frame.copy()
                        cv2.putText(error_frame, f"Error: {str(e)[:50]}", (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if callback:
                            callback(error_frame, frame_count, total_frames, model_manager.model_type)
                        
                        last_update_time = current_time
    
    except Exception as e:
        print(f"Fatal error in webcam processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Update statistics
        stats["frames_processed"] = frame_count
        stats["total_frames"] = total_frames
        stats["processing_time"] = time.time() - start_time
        stats["actual_fps"] = frame_count / stats["processing_time"] if stats["processing_time"] > 0 else 0
        
        # Clean up resources
        cap.release()
        
        # Provide final statistics
        if stats_callback:
            stats_callback(stats)
            
        return stats