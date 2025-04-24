"""
Consolidated Computer Vision Models
This file combines model wrappers (Mask R-CNN and YOLO-Seg) into a single file
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
CONFIGS_DIR = Path("models/configs") # Define config directory
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATHS = {
    'yolo-seg': MODELS_DIR / 'yolov8n-seg.pt',
    'mask-rcnn': MODELS_DIR / 'maskrcnn_resnet50_fpn.pt'
}

MODEL_URLS = {
    "yolov8n-seg.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"
    # Mask R-CNN uses torchvision built-in
}

# --- Helper Function for Downloading ---
def _download_model_if_needed(model_path: Path):
    """Downloads the model file if it does not exist."""
    if model_path.exists():
        # Check if file size is greater than zero for placeholder files
        if model_path.stat().st_size > 0:
             print(f"Model found: {model_path}")
             return True
        else:
             print(f"Placeholder found, proceeding with download check: {model_path}")

    model_name = model_path.name
    # Skip download for Mask R-CNN as we'll use torchvision's pretrained model
    if model_name == 'maskrcnn_resnet50_fpn.pt':
        print(f"Note: {model_name} will be loaded from torchvision, not downloaded.")
        # Create an empty file as a placeholder if it doesn't exist
        if not model_path.exists():
            model_path.touch()
        return True # Indicate that it's handled

    if model_name not in MODEL_URLS:
        print(f"Error: No download URL defined for model: {model_name}")
        return False

    url = MODEL_URLS[model_name]
    print(f"Downloading {model_name} from {url} to {model_path}...")

    try:
        # Standard download method
        response = requests.get(url, stream=True, timeout=120) # Increased timeout
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192 # Increased block size

        with open(model_path, "wb") as f:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    # Basic progress indication
                    progress = int(50 * downloaded_size / total_size) if total_size else 0
                    print(f"\\rDownloading: [{'=' * progress}{' ' * (50 - progress)}] {downloaded_size / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB", end='')
        print(f"\\nModel {model_name} downloaded successfully.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"\\nError downloading model {model_name}: {e}")
        # Clean up potentially incomplete file
        if model_path.exists():
            try:
                os.remove(model_path)
                print(f"Removed incomplete download: {model_path}")
            except OSError as rm_err:
                print(f"Error removing incomplete file {model_path}: {rm_err}")
        return False
    except Exception as e:
        print(f"\\nAn unexpected error occurred during download: {e}")
        # Clean up potentially incomplete file if it exists
        if model_path.exists():
             try:
                 os.remove(model_path)
                 print(f"Removed potentially corrupt file: {model_path}")
             except OSError as rm_err:
                 print(f"Error removing file {model_path}: {rm_err}")
        return False


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
                    # CRITICAL FIX: Convert YOLO's 0-indexed class IDs to COCO's 1-indexed class IDs
                    yolo_class_id = int(box.cls[0])
                    class_id = yolo_class_id + 1  # Add 1 to convert from 0-indexed to 1-indexed
                    score = float(box.conf[0])
                    class_name = names.get(yolo_class_id, f"ID:{yolo_class_id}") # Get name or use ID
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "class_id": class_id,  # Use the adjusted class_id
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


# --- Mask R-CNN Wrapper ---
class MaskRCNNWrapper:
    """Wrapper for Mask R-CNN model inference using torchvision."""
    
    def __init__(self, model_path=None):
        """
        Initializes the Mask R-CNN model wrapper.
        
        Args:
            model_path (str or Path, optional): Path is ignored as we use pretrained torchvision model.
        """
        print("Initializing MaskRCNNWrapper")
        
        # Determine device (cuda or cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            print("Loading Mask R-CNN model from torchvision...")
            # Load the pretrained model - use weights='DEFAULT' for newer torchvision, use pretrained=True for older versions
            try:
                self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
            except TypeError:
                # Fall back to older torchvision API
                self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
            
            # Set model to evaluation mode
            self.model.eval()
            # Move model to the correct device
            self.model.to(self.device)
            # Store COCO class names
            self.class_names = COCO_CLASSES
            print("Mask R-CNN model loaded successfully.")
        except Exception as e:
            print(f"Error loading Mask R-CNN model: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def predict(self, frame: np.ndarray):
        """
        Performs instance segmentation on a single frame.
        
        Args:
            frame (np.ndarray): Input video frame (BGR format).
            
        Returns:
            tuple: A tuple containing:
                - detections (list): List of detected objects (dict with 'box', 'class_id', 'class_name', 'score').
                - segmentations (list): List of segmentation masks (binary masks for each detection).
                - annotated_frame (np.ndarray): Frame with visualizations drawn.
        """
        if self.model is None:
            print("Warning: Mask R-CNN model not loaded. Returning empty results.")
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Mask R-CNN Model Not Loaded", (50, 50),
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
            masks = predictions[0]['masks'].cpu().numpy()
            
            # Filter predictions based on confidence threshold
            confidence_threshold = 0.5
            confident_indices = scores >= confidence_threshold
            
            # Format detections and segmentations
            detections = []
            segmentations = []
            annotated_frame = frame.copy()
            
            # Create a semi-transparent overlay for masks
            mask_overlay = np.zeros_like(frame)
            
            # Define colors for different classes (random but consistent)
            np.random.seed(42)  # For consistent colors across runs
            colors = [(int(np.random.randint(0, 256)), 
                       int(np.random.randint(0, 256)), 
                       int(np.random.randint(0, 256))) for _ in range(len(self.class_names))]
            
            for i, box in enumerate(boxes):
                if scores[i] >= confidence_threshold:
                    # Extract coordinates
                    x1, y1, x2, y2 = box.astype(int)
                    class_id = int(labels[i].item())
                    score = float(scores[i].item())
                    
                    # Get the mask for this detection (threshold at 0.5)
                    mask = masks[i, 0] > 0.5  # First channel of mask, threshold at 0.5
                    segmentations.append(mask.astype(np.uint8) * 255)  # Convert to uint8 for consistency
                    
                    # Get class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"Unknown {class_id}"
                    
                    # Add to detections list
                    detections.append({
                        "box": [x1, y1, x2, y2],
                        "class_id": class_id,
                        "class_name": class_name,
                        "score": score
                    })
                    
                    # Use consistent color based on class ID
                    color = colors[class_id % len(colors)]
                    
                    # Visualize instance mask - add color where mask is True
                    colored_mask = np.zeros_like(frame)
                    mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))
                    colored_mask[mask_resized > 0] = color
                    
                    # Add mask overlay to frame
                    cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.5, 0, annotated_frame)
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with class name and score
                    label_text = f"{class_name}: {score:.2f}"
                    cv2.rectangle(annotated_frame, (x1, y1-25), (x1 + len(label_text)*9, y1), color, -1)
                    cv2.putText(annotated_frame, label_text, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            postprocess_time = time.time() - postprocess_start
            total_time = time.time() - start_time
            
            # Display timing information on frame
            cv2.putText(annotated_frame, f"FPS: {1/total_time:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Add detailed timing report to console
            print(f"Mask R-CNN timing: Total={total_time:.3f}s (FPS: {1/total_time:.1f}) | " 
                  f"Preprocess={preprocess_time:.3f}s | Inference={inference_time:.3f}s | "
                  f"Postprocess={postprocess_time:.3f}s | Detections={len(detections)} | Masks={len(segmentations)}")
            
            return detections, segmentations, annotated_frame
            
        except Exception as e:
            print(f"Error during Mask R-CNN prediction: {e}")
            import traceback
            traceback.print_exc()
            annotated_frame = frame.copy()
            cv2.putText(annotated_frame, "Mask R-CNN Prediction Error", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame


# --- Model Manager ---
class ModelManager:
    """Manager class for handling different model types"""

    def __init__(self, model_type, model_path=None, config_path=None):
        """
        Initializes the appropriate model wrapper based on the model type.

        Args:
            model_type (str): Type of the model ('mask-rcnn', 'yolo-seg').
            model_path (str or Path, optional): Path to the model weights file.
                                                If None, uses default path for the type.
            config_path (str, optional): Path to the configuration file (not used currently).
        """
        self.model_type = model_type.lower()
        self.config_path = config_path # Store config path

        # Determine model path: Use provided path or default for the type
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATHS.get(self.model_type)
        if not self.model_path:
             raise ValueError(f"Model path not specified and no default path found for type: {self.model_type}")

        # Device selection logic remains the same
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
            nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False) # Use check=False
            if nvidia_smi.returncode == 0:
                nvidia_driver_available = True
                print("NVIDIA drivers detected via nvidia-smi")
                # Parse nvidia-smi output to get GPU name
                for line in nvidia_smi.stdout.split('\\n'):
                    if "NVIDIA" in line and ("RTX" in line or "GeForce" in line or "Tesla" in line): # Broader check
                        gpu_name = line.strip()
                        print(f"Found GPU via nvidia-smi: {gpu_name}")
                        break # Found one, no need to check further
        except FileNotFoundError:
             print("nvidia-smi command not found. Cannot check for NVIDIA drivers this way.")
        except Exception as e:
            print(f"Error running nvidia-smi: {e}")

        # Determine if we should force CUDA based on environment and hardware detection
        force_cuda = (force_device == "cuda" or
                     (force_device != "cpu" and (cuda_available or nvidia_driver_available))) # Default to GPU if available unless CPU forced

        # Use CUDA if forced or available and not overridden by CPU force
        if force_cuda and cuda_available:
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"Using CUDA with GPU: {device_name} (Available: {device_count})")
            return 'cuda'
        elif force_cuda and nvidia_driver_available and not cuda_available:
            # If user wants CUDA (or default auto) and we see NVIDIA drivers but PyTorch doesn't recognize them,
            # warn the user but still try CPU as CUDA won't work.
            print("WARNING: NVIDIA GPU detected but PyTorch cannot access it via CUDA.")
            print("This is likely due to PyTorch being installed without CUDA support or a driver/CUDA toolkit mismatch.")
            print("Please check your PyTorch installation and CUDA setup.")
            print("Falling back to CPU.")
            return 'cpu'
        else:
            # Fallback to CPU
            if force_device == "cpu":
                 print("Forcing CPU usage based on environment variable.")
            else:
                 print("Using CPU for model inference (No CUDA GPU detected/available or CPU forced).")
            return 'cpu'

    def _initialize_model(self):
        """Initialize the appropriate model based on type"""
        print(f"Initializing model manager for type: {self.model_type}")
        try:
            if self.model_type == 'mask-rcnn':
                return MaskRCNNWrapper(self.model_path)
            elif self.model_type == 'yolo-seg':
                return YOLOWrapper(self.model_path)
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
            input_points (Optional[np.ndarray]): Not used by current models.
            input_labels (Optional[np.ndarray]): Not used by current models.

        Returns:
            tuple: Prediction results: (detections, segmentations, annotated_frame)
                   Returns ([], [], frame) if the model wrapper is not valid or prediction fails.
        """
        if self.model_wrapper is None or not hasattr(self.model_wrapper, 'predict'):
            print(f"Error: Model wrapper for {self.model_type} is not initialized or lacks predict method.")
            annotated_frame = frame.copy()
            # Add error text to frame
            cv2.putText(annotated_frame, f"{self.model_type.upper()} Model Error", (10, 60), # Adjusted position
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return [], [], annotated_frame # Return empty results and annotated frame

        try:
            # All current models (Mask R-CNN, YOLO-Seg) use the same predict signature
            return self.model_wrapper.predict(frame)
        except Exception as e:
            print(f"Error during prediction with {self.model_type}: {e}")
            import traceback
            traceback.print_exc()
            annotated_frame = frame.copy()
            # Add error text to frame
            cv2.putText(annotated_frame, f"{self.model_type.upper()} Predict Error", (10, 60), # Adjusted position
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Return empty lists and the annotated error frame
            return [], [], annotated_frame

# Define COLORS for visualization if not already defined globally
# Generate random colors for visualization
np.random.seed(42) # for reproducibility
COLORS = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(COCO_CLASSES))] # Use COCO classes length for consistency