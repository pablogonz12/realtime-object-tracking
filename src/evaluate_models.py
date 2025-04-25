"""
Model Evaluation Script for Computer Vision Project

This script evaluates two state-of-the-art object detection and segmentation models:
1. Mask R-CNN (TorchVision)
2. YOLO-Seg (YOLOv8 with segmentation by Ultralytics)

The evaluation measures:
- Validity: Precision, Recall, mAP
- Reliability: Consistency across different runs
- Objectivity: Performance on standard COCO validation dataset
- Efficiency: FPS, latency, and resource usage
"""

import os
import sys
import time
import json
import torch
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import tempfile
import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Import model manager
from src.models import ModelManager, DEFAULT_MODEL_PATHS

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
COCO_VAL_DIR = PROJECT_ROOT / "data_sets" / "image_data" / "coco" / "val2017"
COCO_ANNOT_FILE = PROJECT_ROOT / "data_sets" / "image_data" / "coco" / "annotations" / "instances_val2017.json"
RESULTS_DIR = PROJECT_ROOT / "inference" / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# Configure models for evaluation
MODELS_TO_EVALUATE = [
    {"type": "mask-rcnn", "path": DEFAULT_MODEL_PATHS['mask-rcnn'], "config": None},
    {"type": "yolo-seg", "path": DEFAULT_MODEL_PATHS['yolo-seg'], "config": None}
]

# COCO val2017 has 5000 images total
COCO_VAL_TOTAL_IMAGES = 5000

class ModelEvaluator:
    """Class to evaluate model performance on COCO validation set"""
    
    def __init__(self, models_list=None):
        """
        Initialize the evaluator with models to test
        
        Args:
            models_list (list): List of model configurations to evaluate
        """
        self.models_list = models_list or MODELS_TO_EVALUATE
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        self.coco_gt = None
        self.coco_gt_api = None
        
        # Load COCO annotations if file exists
        if COCO_ANNOT_FILE.exists():
            with open(COCO_ANNOT_FILE, 'r') as f:
                self.coco_gt = json.load(f)
            print(f"Loaded COCO annotations from {COCO_ANNOT_FILE}")
            
            # Initialize COCO API for annotations
            try:
                self.coco_gt_api = COCO(COCO_ANNOT_FILE)
                print(f"Successfully initialized COCO API with {len(self.coco_gt_api.getImgIds())} images")
                print(f"Found {len(self.coco_gt_api.getCatIds())} categories in annotations")
            except Exception as e:
                print(f"Warning: Failed to initialize COCO API: {e}")
        else:
            print(f"Warning: COCO annotations file not found at {COCO_ANNOT_FILE}")
    
    def load_coco_val_images(self, max_images=100):
        """
        Load COCO validation images for evaluation
        
        Args:
            max_images (int): Maximum number of images to load
            
        Returns:
            list: List of (image, image_id) tuples
        """
        # Check if we have the COCO dataset
        if not COCO_VAL_DIR.exists() or not COCO_ANNOT_FILE.exists():
            print(f"COCO dataset not found. Downloading {max_images} images...")
            try:
                # Import and use DatasetManager to download the dataset
                from data_sets.dataset_manager import DatasetManager
                dataset_manager = DatasetManager()
                dataset_manager.setup_coco(subset_size=max_images)
                print("Dataset download complete. Now loading images...")
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                import traceback
                traceback.print_exc()
                return []
                
        if not COCO_VAL_DIR.exists():
            print(f"Error: COCO validation directory not found at {COCO_VAL_DIR}")
            return []
        
        # If COCO API is initialized, use it to get image IDs
        if self.coco_gt_api:
            print("Using COCO API to get image IDs")
            image_ids = sorted(self.coco_gt_api.getImgIds())[:max_images]
            print(f"Selected {len(image_ids)} images from annotation file")
            
            images = []
            for img_id in tqdm(image_ids, desc="Loading images"):
                img_info = self.coco_gt_api.loadImgs(img_id)[0]
                filename = img_info['file_name']
                img_path = COCO_VAL_DIR / filename
                
                if not img_path.exists():
                    print(f"Warning: Image file not found: {img_path}")
                    continue
                
                # Load image
                img = cv2.imread(str(img_path))
                if img is None:
                    print(f"Warning: Could not load image {img_path}")
                    continue
                    
                images.append((img, img_id))
        else:
            # Fallback to directory search if COCO API not available
            print("Falling back to directory search for images")
            image_files = list(COCO_VAL_DIR.glob("*.jpg"))
            if not image_files:
                print(f"Error: No images found in {COCO_VAL_DIR}")
                return []
            
            # Limit number of images (useful for quick testing)
            image_files = image_files[:max_images]
            print(f"Loading {len(image_files)} images from directory")
            
            images = []
            for img_file in tqdm(image_files, desc="Loading images"):
                # Extract image ID from filename (e.g., 000000000139.jpg -> 139)
                img_id = int(img_file.stem)
                
                # Load image
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"Warning: Could not load image {img_file}")
                    continue
                    
                images.append((img, img_id))
        
        print(f"Loaded {len(images)} images")
        return images
    
    def evaluate_model(self, model_config, images, save_visualizations=True, progress_callback=None):
        """
        Evaluate a single model on the validation set
        
        Args:
            model_config (dict): Model configuration with type, path, and config
            images (list): List of (image, image_id) tuples
            save_visualizations (bool): Whether to save visualization images
            progress_callback (callable, optional): Function to report progress. 
                                                   Called with (model_type, current_image_index, total_images).
            
        Returns:
            dict: Performance metrics
        """
        model_type = model_config["type"]
        model_path = model_config["path"]
        config_path = model_config["config"]
        
        print(f"\nEvaluating {model_type.upper()} model...")
        
        # Initialize the model
        try:
            model_manager = ModelManager(model_type, model_path, config_path)
            if model_manager.model_wrapper is None:
                print(f"Error: Failed to initialize {model_type} model")
                return None
        except Exception as e:
            print(f"Error initializing {model_type} model: {e}")
            return None
        
        # Initialize metrics
        metrics = {
            "model_type": model_type,
            "total_images": len(images),
            "successful_inferences": 0,
            "total_detections": 0,
            "inference_times": [],
            "detection_counts": {},
            "classes_detected": set(),
            "failures": 0
        }
        
        # Create directory for model visualizations
        vis_dir = RESULTS_DIR / f"{model_type}_visualizations"
        if save_visualizations:
            vis_dir.mkdir(exist_ok=True)
        
        # For COCO evaluation, we need to store predictions in COCO format
        coco_bbox_predictions = []  # For bounding box predictions
        coco_segm_predictions = []  # For segmentation mask predictions
        
        # Process each image
        for idx, (image, image_id) in enumerate(tqdm(images, desc=f"Evaluating {model_type}")):
            # Report progress via callback
            if progress_callback:
                try:
                    progress_callback(model_type, idx + 1, len(images))
                except Exception as cb_err:
                    print(f"Warning: Progress callback failed: {cb_err}") # Avoid crashing evaluation due to callback error

            try:
                # Run inference with timing
                start_time = time.time()
                
                # Standard prediction for all our models
                detections, segmentations, annotated_image = model_manager.predict(image)
                    
                inference_time = time.time() - start_time
                
                # Record inference time
                metrics["inference_times"].append(inference_time)
                
                # Count successful inferences
                metrics["successful_inferences"] += 1
                
                # Count total detections
                metrics["total_detections"] += len(detections)
                
                # Update class detection counts
                for det in detections:
                    class_name = det.get("class_name", "unknown")
                    metrics["detection_counts"][class_name] = metrics["detection_counts"].get(class_name, 0) + 1
                    metrics["classes_detected"].add(class_name)
                
                # Save visualization if requested
                if save_visualizations and idx < 20:  # Limit to first 20 images
                    cv2.imwrite(str(vis_dir / f"img_{image_id}_{model_type}.jpg"), annotated_image)
                
                # Format detections for COCO evaluation
                for det_idx, det in enumerate(detections):
                    # Get class ID (COCO dataset uses categories 1-90)
                    class_id = det.get("class_id")

                    # --- DEBUGGING START ---
                    # if model_type == "yolo-seg" and idx < 3 and det_idx < 5:
                    #     pass # Or remove the block entirely
                    # --- DEBUGGING END ---

                    # Skip if class_id is None or not recognized
                    if class_id is None:
                        continue
                    
                    # Get bounding box coordinates
                    box = det.get("box", [0, 0, 0, 0])
                    if len(box) == 4:
                        x1, y1, x2, y2 = box
                        # COCO format is [x, y, width, height]
                        coco_box = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
                        
                        # Add prediction to the list in COCO format
                        pred = {
                            'image_id': int(image_id),
                            'category_id': int(class_id), # Ensure category_id is integer
                            'bbox': coco_box,
                            'score': float(det.get("score", 0.0))
                        }
                        
                        # --- DEBUGGING START ---
                        if model_type == "yolo-seg" and idx < 3 and det_idx < 5:
                            print(f"DEBUG ({model_type}, img_id={image_id}, det_idx={det_idx}): COCO_pred={pred}")
                        # --- DEBUGGING END ---
                        
                        coco_bbox_predictions.append(pred)
                
                # If segmentation masks are available (for YOLO-Seg), add them to COCO predictions
                if segmentations and model_type == "yolo-seg":
                    for i, mask in enumerate(segmentations):
                        if i < len(detections) and mask is not None:  # Make sure we have a detection to pair with
                            det = detections[i]
                            class_id = det.get("class_id") # Get class_id again for segmentation
                            
                            # Skip if no class_id or if mask is invalid
                            if class_id is None or mask.shape[0] == 0:
                                continue
                            
                            # Convert binary mask to RLE format that COCO expects
                            try:
                                # Ensure mask is binary (0/1)
                                binary_mask = (mask > 0).astype(np.uint8)
                                
                                # Get mask dimensions
                                mask_h, mask_w = binary_mask.shape[:2]
                                
                                # Calculate area 
                                area = float(np.sum(binary_mask))
                                if area < 1:  # Skip tiny masks
                                    continue
                                    
                                # Convert to RLE format
                                rle = maskUtils.encode(np.asfortranarray(binary_mask))
                                rle['counts'] = rle['counts'].decode('utf-8')  # Convert bytes to str for JSON
                                
                                # Create COCO format segmentation prediction
                                segm_pred = {
                                    'image_id': int(image_id),
                                    'category_id': int(class_id), # Ensure category_id is integer
                                    'segmentation': rle,
                                    'score': float(det.get("score", 0.0)),
                                    'area': area
                                }
                                
                                coco_segm_predictions.append(segm_pred)
                                
                            except Exception as mask_err:
                                print(f"Error converting segmentation mask: {mask_err}")
                                continue
            
            except Exception as e:
                print(f"Error processing image {image_id} with {model_type}: {e}")
                metrics["failures"] += 1
        
        # Calculate standard statistics
        if metrics["inference_times"]:
            metrics["mean_inference_time"] = np.mean(metrics["inference_times"])
            metrics["median_inference_time"] = np.median(metrics["inference_times"])
            metrics["min_inference_time"] = np.min(metrics["inference_times"])
            metrics["max_inference_time"] = np.max(metrics["inference_times"])
            metrics["std_inference_time"] = np.std(metrics["inference_times"])
            metrics["fps"] = 1.0 / metrics["mean_inference_time"]
        
        metrics["classes_detected"] = list(metrics["classes_detected"])
        metrics["unique_classes_detected"] = len(metrics["classes_detected"])
        metrics["avg_detections_per_image"] = metrics["total_detections"] / max(metrics["successful_inferences"], 1)
        
        # Run COCO evaluation if we have ground truth annotations and predictions
        if self.coco_gt_api is not None and (coco_bbox_predictions or coco_segm_predictions):
            try:
                # Evaluate bounding box predictions
                if coco_bbox_predictions:
                    print(f"Running COCO bbox evaluation for {model_type} with {len(coco_bbox_predictions)} predictions")
                    
                    # --- DEBUGGING START ---
                    if model_type == "yolo-seg" and len(coco_bbox_predictions) > 0:
                        print(f"DEBUG ({model_type}): First few formatted bbox preds: {coco_bbox_predictions[:5]}")
                    # --- DEBUGGING END ---

                    # Save predictions to a temporary file
                    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
                        json.dump(coco_bbox_predictions, f)
                        dt_bbox_file = f.name
                    
                    # Initialize the COCO detection object with predictions
                    coco_dt = self.coco_gt_api.loadRes(dt_bbox_file)
                    
                    # Create the COCO evaluator for bounding box
                    coco_bbox_eval = COCOeval(self.coco_gt_api, coco_dt, 'bbox')
                    
                    # Run evaluation
                    coco_bbox_eval.evaluate()
                    coco_bbox_eval.accumulate()
                    coco_bbox_eval.summarize()
                    
                    # Add COCO bbox metrics to our results
                    metrics["coco_metrics"] = {
                        "AP_IoU=0.50:0.95": float(coco_bbox_eval.stats[0]),  # AP at IoU=0.50:0.95
                        "AP_IoU=0.50": float(coco_bbox_eval.stats[1]),       # AP at IoU=0.50
                        "AP_IoU=0.75": float(coco_bbox_eval.stats[2]),       # AP at IoU=0.75
                        "AP_small": float(coco_bbox_eval.stats[3]),          # AP for small objects
                        "AP_medium": float(coco_bbox_eval.stats[4]),         # AP for medium objects
                        "AP_large": float(coco_bbox_eval.stats[5]),          # AP for large objects
                        "AR_max=1": float(coco_bbox_eval.stats[6]),          # AR given 1 detection per image
                        "AR_max=10": float(coco_bbox_eval.stats[7]),         # AR given 10 detections per image
                        "AR_max=100": float(coco_bbox_eval.stats[8]),        # AR given 100 detections per image
                        "AR_small": float(coco_bbox_eval.stats[9]),          # AR for small objects
                        "AR_medium": float(coco_bbox_eval.stats[10]),        # AR for medium objects
                        "AR_large": float(coco_bbox_eval.stats[11])          # AR for large objects
                    }
                    
                    # Clean up the temporary file
                    os.unlink(dt_bbox_file)
                
                # Evaluate segmentation predictions (only for yolo-seg)
                if model_type == "yolo-seg" and coco_segm_predictions:
                    print(f"Running COCO segmentation evaluation for {model_type} with {len(coco_segm_predictions)} predictions")
                    
                    # --- DEBUGGING START ---
                    if len(coco_segm_predictions) > 0:
                        print(f"DEBUG ({model_type}): First few formatted segm preds: {coco_segm_predictions[:5]}")
                    # --- DEBUGGING END ---

                    # Save predictions to a temporary file
                    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
                        json.dump(coco_segm_predictions, f)
                        dt_segm_file = f.name
                    
                    # Initialize the COCO detection object with predictions
                    coco_dt = self.coco_gt_api.loadRes(dt_segm_file)
                    
                    # Create the COCO evaluator for segmentation
                    coco_segm_eval = COCOeval(self.coco_gt_api, coco_dt, 'segm')
                    
                    # Run evaluation
                    coco_segm_eval.evaluate()
                    coco_segm_eval.accumulate()
                    coco_segm_eval.summarize()
                    
                    # Add COCO segmentation metrics to our results
                    metrics["coco_segm_metrics"] = {
                        "Segm_AP_IoU=0.50:0.95": float(coco_segm_eval.stats[0]),  # AP at IoU=0.50:0.95
                        "Segm_AP_IoU=0.50": float(coco_segm_eval.stats[1]),       # AP at IoU=0.50
                        "Segm_AP_IoU=0.75": float(coco_segm_eval.stats[2]),       # AP at IoU=0.75
                        "Segm_AP_small": float(coco_segm_eval.stats[3]),          # AP for small objects
                        "Segm_AP_medium": float(coco_segm_eval.stats[4]),         # AP for medium objects
                        "Segm_AP_large": float(coco_segm_eval.stats[5]),          # AP for large objects
                        "Segm_AR_max=1": float(coco_segm_eval.stats[6]),          # AR given 1 detection per image
                        "Segm_AR_max=10": float(coco_segm_eval.stats[7]),         # Segm AR given 10 detections per image
                        "Segm_AR_max=100": float(coco_segm_eval.stats[8]),        # AR given 100 detections per image
                        "Segm_AR_small": float(coco_segm_eval.stats[9]),          # AR for small objects
                        "Segm_AR_medium": float(coco_segm_eval.stats[10]),        # AR for medium objects
                        "Segm_AR_large": float(coco_segm_eval.stats[11])          # AR for large objects
                    }
                    
                    # Clean up the temporary file
                    os.unlink(dt_segm_file)
                    
            except Exception as e:
                print(f"Error during COCO evaluation for {model_type}: {e}")
                import traceback
                traceback.print_exc()
                
                metrics["coco_metrics"] = {"error": str(e)}
        
        # Print summary
        print(f"\n{model_type.upper()} Evaluation Summary:")
        print(f"- Successfully processed: {metrics['successful_inferences']}/{metrics['total_images']} images")
        print(f"- Average FPS: {metrics['fps']:.2f}")
        print(f"- Total detections: {metrics['total_detections']}")
        print(f"- Unique classes detected: {metrics['unique_classes_detected']}")
        print(f"- Average detections per image: {metrics['avg_detections_per_image']:.2f}")
        
        # Print COCO metrics if available
        if "coco_metrics" in metrics and metrics["coco_metrics"] and "error" not in metrics["coco_metrics"]:
            print("- COCO Detection Metrics:")
            print(f"  • mAP (IoU=0.50:0.95): {metrics['coco_metrics']['AP_IoU=0.50:0.95']:.4f}")
            print(f"  • AP (IoU=0.50): {metrics['coco_metrics']['AP_IoU=0.50']:.4f}")
            print(f"  • AP (IoU=0.75): {metrics['coco_metrics']['AP_IoU=0.75']:.4f}")
        
        # Print segmentation metrics if available
        if "coco_segm_metrics" in metrics and metrics.get("coco_segm_metrics") and "error" not in metrics.get("coco_segm_metrics", {}):
            print("- COCO Segmentation Metrics:")
            print(f"  • Segmentation mAP (IoU=0.50:0.95): {metrics['coco_segm_metrics']['Segm_AP_IoU=0.50:0.95']:.4f}")
            print(f"  • Segmentation AP (IoU=0.50): {metrics['coco_segm_metrics']['Segm_AP_IoU=0.50']:.4f}")
            print(f"  • Segmentation AP (IoU=0.75): {metrics['coco_segm_metrics']['Segm_AP_IoU=0.75']:.4f}")
        
        return metrics
    
    def run_reliability_test(self, max_images=20, num_runs=3):
        """
        Run multiple evaluations to test model reliability
        
        Args:
            max_images (int): Maximum number of images to evaluate
            num_runs (int): Number of evaluation runs
        
        Returns:
            dict: Reliability metrics for each model
        """
        print(f"\nRunning reliability test with {num_runs} evaluation runs")
        
        # Load validation images
        images = self.load_coco_val_images(max_images)
        if not images:
            print("Error: No images to evaluate")
            return {}
        
        # Run multiple evaluations for each model
        reliability_results = {}
        
        for model_config in self.models_list:
            model_type = model_config["type"]
            print(f"\nTesting reliability of {model_type} model...")
            
            run_metrics = []
            for run in range(num_runs):
                print(f"Run {run+1}/{num_runs}:")
                metrics = self.evaluate_model(model_config, images, save_visualizations=(run==0))
                if metrics:
                    run_metrics.append(metrics)
            
            if not run_metrics:
                print(f"No successful evaluation runs for {model_type}")
                continue
            
            # Calculate reliability metrics
            fps_values = [m.get("fps", 0) for m in run_metrics]
            detection_counts = [m.get("total_detections", 0) for m in run_metrics]
            
            reliability_results[model_type] = {
                "fps_mean": np.mean(fps_values),
                "fps_std": np.std(fps_values),
                "fps_var": np.var(fps_values),
                "fps_coef_var": np.std(fps_values) / np.mean(fps_values) if np.mean(fps_values) > 0 else 0,
                "detection_mean": np.mean(detection_counts),
                "detection_std": np.std(detection_counts),
                "run_metrics": run_metrics
            }
            
            print(f"{model_type} Reliability Results:")
            print(f"- Mean FPS: {reliability_results[model_type]['fps_mean']:.2f} ± {reliability_results[model_type]['fps_std']:.2f}")
            print(f"- FPS Coefficient of Variation: {reliability_results[model_type]['fps_coef_var']:.3f}")
            print(f"- Mean Detections: {reliability_results[model_type]['detection_mean']:.1f} ± {reliability_results[model_type]['detection_std']:.1f}")
        
        # Save reliability results
        if reliability_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            reliability_file = RESULTS_DIR / f"reliability_results_{timestamp}.json"
            
            # Convert to serializable format 
            serializable_results = {}
            for model_type, metrics in reliability_results.items():
                serializable_metrics = {}
                for key, value in metrics.items():
                    if key != "run_metrics":  # Skip detailed run metrics - too large
                        serializable_metrics[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
                serializable_results[model_type] = serializable_metrics
            
            with open(reliability_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            print(f"Reliability results saved to {reliability_file}")
        
        return reliability_results
    
    def run_evaluation(self, max_images=50, save_visualizations=True, progress_callback=None):
        """
        Run evaluation on all configured models
        
        Args:
            max_images (int): Maximum number of images to evaluate
            save_visualizations (bool): Whether to save visualized detections
            progress_callback (callable, optional): Function to report progress during model evaluation.
        """
        print(f"\nEvaluating models using {max_images} images from COCO val2017 dataset")
        print(f"This represents {(max_images/COCO_VAL_TOTAL_IMAGES)*100:.1f}% of the full validation set")
        
        # Load validation images
        images = self.load_coco_val_images(max_images)
        if not images:
            print("Error: No images to evaluate")
            return
        
        # Evaluate each model
        results = {}
        for model_config in self.models_list:
            model_type = model_config["type"]
            # Pass the progress callback down to evaluate_model
            metrics = self.evaluate_model(model_config, images, save_visualizations, progress_callback)
            if metrics:
                results[model_type] = metrics
        
        # Save results
        self.results = results
        self.save_results()

    def save_results(self):
        """Save evaluation results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = RESULTS_DIR / f"evaluation_results_{timestamp}.json"
        
        # Convert sets to lists for JSON serialization
        serializable_results = {}
        for model_type, metrics in self.results.items():
            serializable_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, set):
                    serializable_metrics[key] = list(value)
                else:
                    serializable_metrics[key] = value
            serializable_results[model_type] = serializable_metrics
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate computer vision models on COCO validation set")
    parser.add_argument("--images", type=int, default=50,
                        help=f"Number of COCO val2017 images to use (max {COCO_VAL_TOTAL_IMAGES}, default: 50)")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip saving individual detection visualizations (dashboard will still be generated)")
    parser.add_argument("--models", type=str, nargs="+", choices=["mask-rcnn", "yolo-seg"],
                        help="Specific models to evaluate (default: all)")
    args = parser.parse_args()

    # Validate args
    max_images = min(max(1, args.images), COCO_VAL_TOTAL_IMAGES)
    if max_images != args.images:
        print(f"Adjusted number of images to {max_images} (valid range: 1-{COCO_VAL_TOTAL_IMAGES})")

    save_individual_visualizations = not args.no_vis # Renamed for clarity

    # Filter models if specified
    models_to_evaluate = MODELS_TO_EVALUATE
    if args.models:
        models_to_evaluate = [m for m in MODELS_TO_EVALUATE if m["type"] in args.models]
        print(f"Evaluating selected models: {', '.join([m['type'] for m in models_to_evaluate])}")

    print("Starting model evaluation...")

    # Create evaluator with possibly filtered models list
    evaluator = ModelEvaluator(models_to_evaluate)

    # Run evaluation
    evaluator.run_evaluation(max_images, save_individual_visualizations)

    print("Evaluation complete!")

    # Always generate the metrics dashboard after evaluation
    print("\\nGenerating metrics dashboard...")
    try:
        # Ensure MetricsVisualizer is imported
        from src.metrics_visualizer import MetricsVisualizer
        visualizer = MetricsVisualizer()  # Will automatically use latest results file

        # Create the dashboard
        dashboard_path = visualizer.create_metrics_dashboard(show_plot=False) # Set show_plot=False to only save

        if dashboard_path:
            print(f"Metrics dashboard generated successfully at: {dashboard_path}")
        else:
            print("Failed to generate metrics dashboard.")

    except ImportError as e:
        print(f"Warning: Could not import metrics visualizer: {e}")
        print("Skipping dashboard generation...")
    except Exception as e:
        print(f"Error generating dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()