"""
Consolidated Computer Vision Models
This file combines model wrappers (Faster R-CNN, RT-DETR, YOLO-Seg) into a single file
for easier maintenance and use.
"""

import cv2
import numpy as np
import torch
import os
import requests
import time
import torchvision
from torchvision.transforms import functional as F
from pathlib import Path

# COCO Class Names (80 classes + background)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# --- Model URLs and Default Paths ---
# Using Path for better path handling
MODELS_DIR = Path("models/pts")
MODELS_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

DEFAULT_MODEL_PATHS = {
    'faster-rcnn': MODELS_DIR / 'fasterrcnn_resnet50_fpn.pt',
    'rtdetr': MODELS_DIR / 'rtdetr-l.pt',
    'yolo-seg': MODELS_DIR / 'yolov8n-seg.pt' # Added YOLO Segmentation model
}

MODEL_URLS = {
    "rtdetr-l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/rtdetr-l.pt",
    "yolov8n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt" # Added YOLO-Seg URL
}

# --- Helper Function for Downloading ---
def _download_model_if_needed(model_path: Path):
    """Downloads the model file if it does not exist."""
    if model_path.exists():
        print(f"Model found: {model_path}")
        return True

    model_name = model_path.name
    # Skip download for Faster R-CNN as we'll use torchvision's pretrained model
    if model_name == 'fasterrcnn_resnet50_fpn.pt':
        print(f"Note: {model_name} will be loaded from torchvision, not downloaded.")
        # Create an empty file as a placeholder
        open(model_path, 'w').close()
        return True
        
    if model_name not in MODEL_URLS:
        print(f"Error: No download URL defined for model: {model_name}")
        return False

    url = MODEL_URLS[model_name]
    print(f"Downloading {model_name} from {url} to {model_path}...")

    try:
        response = requests.get(url, stream=True, timeout=60) # Added timeout
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte

        with open(model_path, "wb") as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Basic progress indication
                    progress = int(50 * downloaded_size / total_size) if total_size else 0
                    print(f"\rDownloading: [{'=' * progress}{' ' * (50 - progress)}] {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB", end='')
        print(f"\nModel {model_name} downloaded successfully.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"\nError downloading model {model_name}: {e}")
        # Clean up potentially incomplete file
        if model_path.exists():
            try:
                os.remove(model_path)
                print(f"Removed incomplete download: {model_path}")
            except OSError as rm_err:
                print(f"Error removing incomplete file {model_path}: {rm_err}")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred during download: {e}")
        return False


# --- Faster R-CNN Wrapper ---
class FasterRCNNWrapper:
    """Wrapper for Faster R-CNN model inference using torchvision."""
    
    def __init__(self, model_path=None):
        """
        Initializes the Faster R-CNN model wrapper.
        
        Args:
            model_path (str or Path, optional): Path is ignored as we use pretrained torchvision model.
        """
        print("Initializing FasterRCNNWrapper")
        
        # Determine device (cuda or cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            print("Loading Faster R-CNN model from torchvision...")
            # Load the pretrained model - use weights='DEFAULT' for newer torchvision, use pretrained=True for older versions
            try:
                self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
            except TypeError:
                # Fall back to older torchvision API
                self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            
            # Set model to evaluation mode
            self.model.eval()
            # Move model to the correct device
            self.model.to(self.device)
            # Store COCO class names
            self.class_names = COCO_CLASSES
            print("Faster R-CNN model loaded successfully.")
        except Exception as e:
            print(f"Error loading Faster R-CNN model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def predict(self, frame: np.ndarray):
        """
        Performs object detection on a single frame.
        
        Args:
            frame (np.ndarray): Input video frame (BGR format).
            
        Returns:
            tuple: A tuple containing:
                - detections (list): List of detected objects (dict with 'box', 'class_id', 'class_name', 'score').
                - segmentations (list): Empty list (Faster R-CNN doesn't provide segmentation).
                - annotated_frame (np.ndarray): Frame with visualizations drawn.
        """
        if self.model is None:
            print("Warning: Faster R-CNN model not loaded. Returning empty results.")
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Faster R-CNN Model Not Loaded", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame
            
        try:
            # Start timing
            start_time = time.time()
            
            # Preprocess the input frame
            # Convert BGR (OpenCV) to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to a PyTorch tensor
            img_tensor = F.to_tensor(rgb_frame)
            
            # Move to the correct device and add batch dimension
            img_tensor = img_tensor.to(self.device).unsqueeze(0)
            
            preprocess_time = time.time() - start_time
            inference_start = time.time()
            
            # Perform inference
            with torch.no_grad():
                predictions = self.model(img_tensor)
                
            inference_time = time.time() - inference_start
            postprocess_start = time.time()
            
            # Process predictions
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            
            # Filter predictions based on confidence threshold
            confidence_threshold = 0.5
            confident_detections = scores >= confidence_threshold
            
            # Format detections
            detections = []
            annotated_frame = frame.copy()
            
            for idx in range(len(boxes)):
                if scores[idx] >= confidence_threshold:
                    # Extract coordinates
                    x1, y1, x2, y2 = boxes[idx].astype(int)
                    class_id = labels[idx].item()
                    score = scores[idx].item()
                    
                    # Get class name (subtract 1 if your model uses 0 as background)
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Unknown {class_id}"
                    
                    # Add to detections list
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "class_id": class_id,
                        "class_name": class_name,
                        "score": score
                    })
                    
                    # Annotate frame
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{class_name}: {score:.2f}"
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            postprocess_time = time.time() - postprocess_start
            total_time = time.time() - start_time
            
            # Display timing information on frame and in console
            cv2.putText(annotated_frame, f"FPS: {1/total_time:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add detailed timing report to console
            print(f"Faster R-CNN timing: Total={total_time:.3f}s (FPS: {1/total_time:.1f}) | " 
                  f"Preprocess={preprocess_time:.3f}s | Inference={inference_time:.3f}s | "
                  f"Postprocess={postprocess_time:.3f}s | Detections={len(detections)}")
            
            return detections, [], annotated_frame
            
        except Exception as e:
            print(f"Error during Faster R-CNN prediction: {e}")
            import traceback
            traceback.print_exc()
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Faster R-CNN Prediction Error", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame


# --- YOLO Wrapper ---
class YOLOWrapper:
    """Wrapper for YOLO model inference using the Ultralytics library."""

    def __init__(self, model_path=DEFAULT_MODEL_PATHS['yolo-seg']):
        """
        Initializes the YOLO model wrapper. Handles both detection and segmentation models.

        Args:
            model_path (str or Path): Path to the YOLO model weights file (.pt).
        """
        self.model_path = Path(model_path)
        print(f"Initializing YOLOWrapper with model_path: {self.model_path}")

        # Determine if it's a default model for download check
        is_default_seg = self.model_path == DEFAULT_MODEL_PATHS.get('yolo-seg')

        # Download model if it doesn't exist and is a default model
        if is_default_seg:
            if not _download_model_if_needed(self.model_path):
                print(f"Error: YOLO model file {self.model_path} could not be found or downloaded.")
                self.model = None
                return
        elif not self.model_path.exists():
             print(f"Error: Custom YOLO model file not found: {self.model_path}")
             self.model = None
             return

        # Determine device (cuda or cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        try:
            from ultralytics import YOLO # Import here to avoid dependency if not used
            print("Loading YOLO model...")
            self.model = YOLO(self.model_path)

            # Ensure the model is moved to the correct device
            self.model.to(self.device)

            # Perform a dummy inference to check if model loaded correctly (optional)
            # _ = self.model(np.zeros((64, 64, 3), dtype=np.uint8))
            print("YOLO model loaded successfully.")
        except ImportError:
             print("Error: 'ultralytics' library not found. Please install it (`pip install ultralytics`)")
             self.model = None
        except Exception as e:
            print(f"Error loading YOLO model from {self.model_path}: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def predict(self, frame: np.ndarray):
        """
        Performs object detection and segmentation on a single frame.

        Args:
            frame (np.ndarray): Input video frame (BGR format).

        Returns:
            tuple: A tuple containing:
                - detections (list): List of detected objects (dict with 'box', 'class_id', 'class_name', 'score').
                - segmentations (list): List of segmentation masks (if available, else empty).
                - annotated_frame (np.ndarray): Frame with visualizations drawn.
        """
        if self.model is None:
            print("Warning: YOLO model not loaded. Returning empty results.")
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "YOLO Model Not Loaded", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame

        try:
            # Start timing
            start_time = time.time()
            
            # Perform prediction
            results = self.model(frame, verbose=False) # verbose=False reduces console spam
            
            inference_time = time.time() - start_time
            postprocess_start = time.time()

            detections = []
            segmentations = [] # Placeholder for potential segmentation masks
            
            # Create a copy of the frame for annotation
            annotated_frame = results[0].plot()

            # Extract detection details (optional, as plot() already visualizes)
            for result in results:
                boxes = result.boxes
                names = result.names # Class names mapping
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    score = float(box.conf[0])
                    class_name = names.get(class_id, f"ID:{class_id}") # Get name or use ID
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "class_id": class_id,
                        "class_name": class_name,
                        "score": score
                    })

                # Extract segmentation masks if available (e.g., for yolov8n-seg.pt)
                if result.masks is not None:
                    # Convert masks to numpy arrays (ensure they are boolean or uint8)
                    for mask_tensor in result.masks.data:
                        # Mask tensor is usually float, convert to boolean or uint8
                        mask_np = mask_tensor.cpu().numpy().astype(np.uint8) * 255 # Or bool
                        segmentations.append(mask_np)
            
            postprocess_time = time.time() - postprocess_start
            total_time = time.time() - start_time
            
            # Display timing information on frame
            cv2.putText(annotated_frame, f"FPS: {1/total_time:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add detailed timing report to console
            print(f"YOLO timing: Total={total_time:.3f}s (FPS: {1/total_time:.1f}) | " 
                  f"Inference={inference_time:.3f}s | Postprocess={postprocess_time:.3f}s | "
                  f"Detections={len(detections)} | Segmentations={len(segmentations)}")

            return detections, segmentations, annotated_frame
            
        except Exception as e:
            print(f"Error during YOLO prediction: {e}")
            import traceback
            traceback.print_exc()
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "YOLO Prediction Error", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame


# --- RT-DETR Wrapper ---
class RTDETRWrapper:
    """Wrapper for RT-DETR model inference using the Ultralytics library."""

    def __init__(self, model_path=DEFAULT_MODEL_PATHS['rtdetr'], config_path=None):
        """
        Initializes the RT-DETR model wrapper.

        Args:
            model_path (str or Path): Path to the RT-DETR model weights file.
            config_path (str, optional): Path to the configuration file (usually not needed with Ultralytics).
        """
        self.model_path = Path(model_path)
        print(f"Initializing RTDETRWrapper with model_path: {self.model_path}")

        # Download model if it doesn't exist and is a default model
        if self.model_path == DEFAULT_MODEL_PATHS['rtdetr']:
            if not _download_model_if_needed(self.model_path):
                print(f"Error: RT-DETR model file {self.model_path} could not be found or downloaded.")
                self.model = None
                return
        elif not self.model_path.exists():
             print(f"Error: Custom RT-DETR model file not found: {self.model_path}")
             self.model = None
             return

        try:
            # RT-DETR is now integrated into Ultralytics
            from ultralytics import RTDETR # Import here
            print("Loading RT-DETR model...")
            self.model = RTDETR(self.model_path)
            # Perform a dummy inference to check if model loaded correctly (optional)
            # _ = self.model(np.zeros((64, 64, 3), dtype=np.uint8))
            print("RT-DETR model loaded successfully.")
        except ImportError:
             print("Error: 'ultralytics' library not found. Please install it (`pip install ultralytics`)")
             self.model = None
        except Exception as e:
            print(f"Error loading RT-DETR model from {self.model_path}: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def predict(self, frame: np.ndarray):
        """
        Performs object detection on a single frame.

        Args:
            frame (np.ndarray): Input video frame (BGR format).

        Returns:
            tuple: A tuple containing:
                - detections (list): List of detected objects (dict with 'box', 'class_id', 'class_name', 'score').
                - segmentations (list): Empty list (RT-DETR is primarily detection).
                - annotated_frame (np.ndarray): Frame with visualizations drawn.
        """
        if self.model is None:
            print("Warning: RT-DETR model not loaded. Returning empty results.")
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "RT-DETR Model Not Loaded", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame

        try:
            # Start timing
            start_time = time.time()
            
            # Perform prediction
            results = self.model(frame, verbose=False)
            
            inference_time = time.time() - start_time
            postprocess_start = time.time()
            
            detections = []
            segmentations = [] # RT-DETR typically doesn't output segmentations
            
            # Create a copy of the frame for annotation
            annotated_frame = frame.copy()
            
            # Extract detection details
            for result in results:
                boxes = result.boxes
                names = result.names  # Class names mapping
                
                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls[0])
                    score = float(box.conf[0])
                    
                    # Get class name - fallback to COCO classes if not available in result.names
                    if class_id in names:
                        class_name = names[class_id]
                    elif class_id < len(COCO_CLASSES):
                        class_name = COCO_CLASSES[class_id + 1]  # +1 because COCO_CLASSES starts with background
                    else:
                        class_name = f"ID:{class_id}"
                    
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "class_id": class_id,
                        "class_name": class_name,
                        "score": score
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label with class name instead of ID
                    label_text = f"{class_name}: {score:.2f}"
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            postprocess_time = time.time() - postprocess_start
            total_time = time.time() - start_time
            
            # Display timing information on frame and in console
            cv2.putText(annotated_frame, f"FPS: {1/total_time:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add detailed timing report to console
            print(f"RT-DETR timing: Total={total_time:.3f}s (FPS: {1/total_time:.1f}) | " 
                  f"Inference={inference_time:.3f}s | Postprocess={postprocess_time:.3f}s | "
                  f"Detections={len(detections)}")

            return detections, segmentations, annotated_frame
            
        except Exception as e:
            print(f"Error during RT-DETR prediction: {e}")
            import traceback
            traceback.print_exc()
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "RT-DETR Prediction Error", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame


# --- Model Manager ---
class ModelManager:
    """Manager class for handling different model types"""

    def __init__(self, model_type, model_path=None, config_path=None):
        """
        Initializes the appropriate model wrapper based on the model type.

        Args:
            model_type (str): Type of the model ('faster-rcnn', 'rtdetr', 'yolo-seg').
            model_path (str or Path, optional): Path to the model weights file.
                                                If None, uses default path for the type.
            config_path (str, optional): Path to the configuration file (currently unused).
        """
        self.model_type = model_type.lower()
        self.config_path = config_path # Store config path even if unused by current wrappers

        # Determine model path: Use provided path or default for the type
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATHS.get(self.model_type)
        if not self.model_path:
             raise ValueError(f"Model path not specified and no default path found for type: {self.model_type}")

        self.device = self._get_device()
        print(f"Using device: {self.device}")
        self.model_wrapper = self._initialize_model()

        if self.model_wrapper is None or not hasattr(self.model_wrapper, 'model') or self.model_wrapper.model is None:
             print(f"Warning: ModelManager failed to initialize a valid model wrapper for type '{self.model_type}' with path '{self.model_path}'.")
             # Optionally raise an error here if initialization must succeed
             # raise RuntimeError("Model wrapper initialization failed.")


    def _get_device(self):
        """Determine the device to use with enhanced detection for problematic CUDA setups."""
        # Check if we have an environment variable to force a specific device
        force_device = os.environ.get("CV_FORCE_DEVICE", "").lower()
        
        # First attempt: Check standard PyTorch CUDA availability
        cuda_available = torch.cuda.is_available()
        
        # Second attempt: Check if NVIDIA drivers are installed even if PyTorch doesn't see them
        nvidia_driver_available = False
        try:
            import subprocess
            nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if nvidia_smi.returncode == 0:
                nvidia_driver_available = True
                print("NVIDIA drivers detected via nvidia-smi")
                # Parse nvidia-smi output to get GPU name
                for line in nvidia_smi.stdout.split('\n'):
                    if "NVIDIA" in line and "RTX" in line:
                        gpu_name = line.strip()
                        print(f"Found GPU via nvidia-smi: {gpu_name}")
        except:
            pass
            
        # Determine if we should force CUDA based on environment and hardware detection
        force_cuda = (force_device == "cuda" or 
                     (force_device == "auto" and (cuda_available or nvidia_driver_available)))
                     
        # Force CUDA if user explicitly asked for it or if we detected NVIDIA GPU but PyTorch didn't
        if force_cuda:
            if cuda_available:
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device() 
                device_name = torch.cuda.get_device_name(current_device)
                print(f"Using CUDA with GPU: {device_name}")
                return 'cuda'
            elif nvidia_driver_available and force_device == "cuda":
                # If user forced CUDA and we see NVIDIA drivers but PyTorch doesn't recognize them,
                # still attempt to use CUDA and warn the user
                print("WARNING: NVIDIA GPU detected but PyTorch can't access it through CUDA.")
                print("This may be due to PyTorch being installed without CUDA support.")
                print("Attempting to use CUDA anyway, but this may cause errors.")
                print("Try reinstalling PyTorch with CUDA support using:")
                print("pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118")
                
                # We'll still return 'cuda' to make an attempt, even if it might fail
                # This will allow error messages to be more informative
                return 'cuda'
        
        # Fallback to CPU
        print("Using CPU for model inference (no CUDA GPU detected or CPU forced)")
        return 'cpu'

    def _initialize_model(self):
        """Initialize the appropriate model based on type"""
        print(f"Initializing model manager for type: {self.model_type}")
        try:
            if self.model_type == 'faster-rcnn':
                # Faster R-CNN wrapper from torchvision
                return FasterRCNNWrapper(self.model_path)
            elif self.model_type == 'yolo-seg': # Added case for YOLO segmentation
                # Reuse YOLOWrapper, it handles segmentation models
                return YOLOWrapper(self.model_path)
            elif self.model_type == 'rtdetr':
                # RTDETR wrapper handles device internally via Ultralytics
                return RTDETRWrapper(self.model_path, self.config_path)
            elif self.model_type == 'sam':
                # SAM model isn't implemented for this task
                print(f"Note: SAM model is not implemented for the project evaluation task. Please use one of the three main models: faster-rcnn, rtdetr, or yolo-seg.")
                return None
            else:
                print(f"Error: Unsupported model type: {self.model_type}")
                return None # Return None for unsupported types
        except Exception as e:
             print(f"Error during model wrapper initialization for {self.model_type}: {e}")
             import traceback
             traceback.print_exc()
             return None


    def predict(self, frame, input_points=None, input_labels=None):
        """
        Performs prediction using the selected model.

        Args:
            frame (np.ndarray): Input video frame.
            input_points (Optional[np.ndarray]): Point prompts for SAM (if applicable).
            input_labels (Optional[np.ndarray]): Labels for input points for SAM (if applicable).

        Returns:
            tuple: Prediction results. Format depends on the model type:
                   - YOLO/RT-DETR/YOLO-Seg: (detections, segmentations, annotated_frame)
                   Returns ([], [], frame) if the model wrapper is not valid or prediction fails.
        """
        if self.model_wrapper is None or not hasattr(self.model_wrapper, 'predict'):
            print(f"Error: Model wrapper for {self.model_type} is not initialized or lacks predict method.")
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, f"{self.model_type.upper()} Model Error", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame # Return empty results and original frame

        try:
            # Removed SAM-specific logic
            # All current models (YOLO, YOLO-Seg, RT-DETR) use the same predict signature
            # Ignore input_points and input_labels if passed
            if input_points is not None or input_labels is not None:
                 print(f"Warning: input_points/input_labels provided but model type is {self.model_type}. Ignoring.")
            return self.model_wrapper.predict(frame)
        except Exception as e:
            print(f"Error during prediction with {self.model_type}: {e}")
            import traceback
            traceback.print_exc()
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, f"{self.model_type.upper()} Predict Error", (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Return empty lists and the annotated error frame
            return [], [], annotated_frame