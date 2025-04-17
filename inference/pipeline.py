"""
Unified inference pipeline for processing video frames using selected models.
This file is now less critical as ModelManager handles model loading and prediction,
but can be kept for potential future complex pipeline logic or testing.
"""
import sys
import os
import cv2
import time
import numpy as np

# Add the project root directory to the Python module search path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # Go up two levels
sys.path.insert(0, project_root)

# Import the central ModelManager
from models.models import ModelManager

class InferencePipeline:
    def __init__(self, model_type='yolo', model_path=None, config_path=None):
        """
        Initializes the inference pipeline using ModelManager.

        Args:
            model_type (str): The model to use ('yolo', 'rtdetr', 'sam').
            model_path (str, optional): Path to model weights. Defaults handled by ModelManager.
            config_path (str, optional): Path to config file (if needed).
        """
        self.model_type = model_type.lower()
        print(f"Initializing Inference Pipeline for model type: {self.model_type}")
        try:
            # Use ModelManager to handle model loading and device selection
            self.model_manager = ModelManager(
                model_type=self.model_type,
                model_path=model_path,
                config_path=config_path
            )
            if self.model_manager.model_wrapper is None:
                 print("Warning: ModelManager failed to initialize a model wrapper in the pipeline.")
        except ValueError as e:
             print(f"Error initializing ModelManager in pipeline: {e}")
             self.model_manager = None
        except Exception as e:
             print(f"Unexpected error initializing ModelManager in pipeline: {e}")
             import traceback
             traceback.print_exc()
             self.model_manager = None

        print(f"Pipeline initialized. ModelManager ready: {self.model_manager is not None and self.model_manager.model_wrapper is not None}")


    def process_frame(self, frame, input_points=None, input_labels=None):
        """
        Processes a single video frame using the managed model.

        Args:
            frame (np.ndarray): Input video frame (BGR format).
            input_points (Optional[np.ndarray]): Point prompts for SAM.
            input_labels (Optional[np.ndarray]): Labels for input points for SAM.


        Returns:
            tuple: Contains results depending on the model:
                   - YOLO/RT-DETR: (detections, segmentations, annotated_frame)
                   - SAM: (masks, scores, annotated_frame)
                   Returns ([], [], frame) if model processing fails.
        """
        if self.model_manager is None or self.model_manager.model_wrapper is None:
            print("Error: ModelManager or its wrapper not available in pipeline.")
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Pipeline Error: No Model", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return [], [], annotated_frame # Return empty results and annotated frame

        start_time = time.time()

        try:
            # Delegate prediction to the ModelManager
            results = self.model_manager.predict(frame, input_points=input_points, input_labels=input_labels)

            end_time = time.time()
            processing_time = end_time - start_time
            fps = 1.0 / processing_time if processing_time > 0 else float('inf')
            # print(f"Frame processed by {self.model_type} via Manager in {processing_time:.4f} seconds ({fps:.2f} FPS)") # Reduce verbosity

            # Ensure results tuple has the expected structure (list, list, frame)
            if not (isinstance(results, tuple) and len(results) == 3 and isinstance(results[2], np.ndarray)):
                 print(f"Warning: Unexpected result format from ModelManager.predict: {type(results)}")
                 annotated_frame = frame.copy()
                 cv2.putText(annotated_frame, "Pipeline Error: Bad Result Format", (10, 60),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                 return [], [], annotated_frame

            # Add FPS counter to the annotated frame returned by the manager
            annotated_frame = results[2] # Annotated frame is the last element
            cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Return the results (unpacking the tuple)
            return results[0], results[1], annotated_frame

        except Exception as e:
            print(f"Error during pipeline processing frame with {self.model_type}: {e}")
            import traceback
            traceback.print_exc()
            # Return original frame with error text in case of prediction error
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, f"Pipeline Error: {self.model_type}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return [], [], annotated_frame


# Example usage (for testing) - Updated for ModelManager
if __name__ == '__main__':
    # Create a dummy frame
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "Input Frame", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Test with Faster R-CNN
    print("\n--- Testing Faster R-CNN Pipeline ---")
    try:
        pipeline_fasterrcnn = InferencePipeline(model_type='faster-rcnn')
        if pipeline_fasterrcnn.model_manager and pipeline_fasterrcnn.model_manager.model_wrapper:
            dets, segs, annotated_fasterrcnn = pipeline_fasterrcnn.process_frame(dummy_frame.copy())
            print(f"Faster R-CNN Detections: {len(dets)}")
            print(f"Faster R-CNN Segmentations: {len(segs)}")
            # cv2.imshow("Faster R-CNN Pipeline Test", annotated_fasterrcnn)
        else:
            print("Faster R-CNN Pipeline failed to initialize ModelManager.")
    except Exception as e:
        print(f"Failed to test Faster R-CNN pipeline: {e}")
        import traceback
        traceback.print_exc()


    # Test with RT-DETR
    print("\n--- Testing RT-DETR Pipeline ---")
    try:
        pipeline_rtdetr = InferencePipeline(model_type='rtdetr')
        if pipeline_rtdetr.model_manager and pipeline_rtdetr.model_manager.model_wrapper:
            dets, segs, annotated_rtdetr = pipeline_rtdetr.process_frame(dummy_frame.copy())
            print(f"RT-DETR Detections: {len(dets)}")
            print(f"RT-DETR Segmentations: {len(segs)}")
            # cv2.imshow("RT-DETR Pipeline Test", annotated_rtdetr)
        else:
            print("RT-DETR Pipeline failed to initialize ModelManager.")
    except Exception as e:
        print(f"Failed to test RT-DETR pipeline: {e}")
        import traceback
        traceback.print_exc()

    # Test with SAM
    print("\n--- Testing SAM Pipeline ---")
    try:
        pipeline_sam = InferencePipeline(model_type='sam')
        if pipeline_sam.model_manager and pipeline_sam.model_manager.model_wrapper:
            # Create dummy points for SAM test
            h, w = dummy_frame.shape[:2]
            points = np.array([[w // 2, h // 2], [w // 4, h // 4]])
            labels = np.array([1, 1]) # Foreground points
            masks, scores, annotated_sam = pipeline_sam.process_frame(dummy_frame.copy(), input_points=points, input_labels=labels)
            print(f"SAM Masks: {len(masks)}")
            print(f"SAM Scores: {scores}")
            # cv2.imshow("SAM Pipeline Test", annotated_sam)
        else:
            print("SAM Pipeline failed to initialize ModelManager.")
    except Exception as e:
        print(f"Failed to test SAM pipeline: {e}")
        import traceback
        traceback.print_exc()

    # print("\nDisplaying results (press 'q' in window or any key here)...")
    # key = cv2.waitKey(0)
    # if key == ord('q'):
    #      print("Exiting.")
    # cv2.destroyAllWindows()
    print("\nPipeline tests complete.")
