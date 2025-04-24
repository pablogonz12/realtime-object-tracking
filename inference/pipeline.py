"""
Computer Vision Project Pipeline Script

This script serves as the main pipeline for the computer vision project evaluation task.
It coordinates the evaluation of two state-of-the-art computer vision models and the
creation of a demonstration video showcasing the best-performing model.

The pipeline:
1. Evaluates both models on the COCO validation dataset
2. Compares their performance metrics
3. Identifies the best-performing model based on selected criteria
4. Creates a demonstration video with the best model

Task requirements:
- Evaluate two SoA approaches on validity, reliability, and objectivity
- Present a qualitative analysis of the different approaches
- Create a demo video with the best-performing model
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import evaluation and demo scripts
from inference.evaluate_models import ModelEvaluator
from inference.create_demo_video import create_demo_video, list_available_videos

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "inference" / "results"
VIDEO_DIR = PROJECT_ROOT / "data_sets" / "video_data" / "samples"
OUTPUT_DIR = PROJECT_ROOT / "inference" / "output_videos"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Model types to evaluate
MODEL_TYPES = ["mask-rcnn", "yolo-seg"]

def run_evaluation(max_images=100, save_visualizations=True):
    """
    Run evaluation on all models
    
    Args:
        max_images (int): Maximum number of images to evaluate
        save_visualizations (bool): Whether to save visualized detections
        
    Returns:
        dict: Evaluation results
    """
    print("\n=== STARTING MODEL EVALUATION ===\n")
    
    # Create evaluator and run evaluation
    evaluator = ModelEvaluator()
    evaluator.run_evaluation(max_images, save_visualizations)
    
    # Return results
    return evaluator.results

def determine_best_model(results):
    """
    Determine the best model based on multiple criteria
    
    Args:
        results (dict): Evaluation results from ModelEvaluator
        
    Returns:
        str: Name of the best model
        dict: Scoring details
    """
    if not results or not all(model in results for model in MODEL_TYPES):
        print("Error: Not all models have evaluation results")
        # Default to YOLO-Seg if results are incomplete
        return "yolo-seg", {"note": "Default selection due to incomplete evaluation"}
    
    # Define scoring weights
    weights = {
        "fps": 0.3,                      # Performance (30%)
        "successful_inferences": 0.1,     # Reliability (10%)
        "total_detections": 0.15,         # Detection quantity (15%)
        "unique_classes_detected": 0.15,   # Detection variety (15%)
        "AP_IoU=0.50": 0.3,              # Detection accuracy (30%)
    }
    
    # Normalize metrics and calculate scores
    scores = {}
    metrics = {}
    
    # Extract metrics
    for metric in weights.keys():
        if metric == "AP_IoU=0.50":
            # Get mAP from COCO metrics if available
            metrics[metric] = {
                model: (results[model].get("coco_metrics", {}) or {}).get(metric, 0) 
                for model in MODEL_TYPES
            }
        else:
            metrics[metric] = {
                model: results[model].get(metric, 0) 
                for model in MODEL_TYPES
            }
    
    # Add bonus for segmentation capabilities (only for YOLO-Seg)
    if "yolo-seg" in results and results["yolo-seg"].get("coco_segm_metrics"):
        segm_ap = results["yolo-seg"]["coco_segm_metrics"].get("Segm_AP_IoU=0.50", 0)
        # Add segmentation bonus
        metrics["segmentation_bonus"] = {"mask-rcnn": 0, "yolo-seg": segm_ap}
        weights["segmentation_bonus"] = 0.1  # 10% weight for segmentation capability
    
    # Find max values for normalization
    max_values = {metric: max(values.values()) for metric, values in metrics.items()}
    
    # Calculate normalized scores
    for model in MODEL_TYPES:
        scores[model] = sum(
            weights[metric] * (metrics[metric][model] / max_values[metric]) 
            for metric in weights
            if max_values[metric] > 0  # Avoid division by zero
        )
    
    # Find the best model
    best_model = max(scores, key=scores.get)
    
    # Prepare scoring details
    scoring_details = {
        "scores": scores,
        "weights": weights,
        "raw_metrics": {model: {metric: metrics[metric][model] for metric in weights} 
                        for model in MODEL_TYPES},
        "best_model": best_model
    }
    
    return best_model, scoring_details

def create_comparison_report(results, scoring_details):
    """
    Create a comprehensive comparison report of the models
    
    Args:
        results (dict): Evaluation results from ModelEvaluator
        scoring_details (dict): Details from the best model determination
        
    Returns:
        str: Path to the saved report
    """
    print("\n=== CREATING COMPARISON REPORT ===\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"model_comparison_report_{timestamp}.html"
    
    # Create HTML report
    with open(report_path, 'w') as f:
        f.write('<html>\n<head>\n')
        f.write('<title>Computer Vision Model Comparison Report</title>\n')
        f.write('<style>\n')
        f.write('body { font-family: Arial, sans-serif; margin: 40px; }\n')
        f.write('h1 { color: #2c3e50; }\n')
        f.write('h2 { color: #3498db; margin-top: 30px; }\n')
        f.write('table { border-collapse: collapse; width: 100%; margin: 20px 0; }\n')
        f.write('th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }\n')
        f.write('th { background-color: #f2f2f2; }\n')
        f.write('tr:nth-child(even) { background-color: #f9f9f9; }\n')
        f.write('.highlight { background-color: #e8f4f8; font-weight: bold; }\n')
        f.write('.chart { width: 100%; max-width: 800px; margin: 30px 0; }\n')
        f.write('.winner { background-color: #d5f5e3; }\n')
        f.write('</style>\n')
        f.write('</head>\n<body>\n')
        
        # Header
        f.write(f'<h1>Computer Vision Model Comparison Report</h1>\n')
        f.write(f'<p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>\n')
        
        # Best Model Section
        best_model = scoring_details.get('best_model', 'Unknown')
        f.write(f'<h2>Best Performing Model: {best_model.upper()}</h2>\n')
        f.write('<p>Based on a weighted evaluation of performance metrics including speed, detection accuracy, and versatility.</p>\n')
        
        # Summary Scores Table
        f.write('<h2>Model Scores Summary</h2>\n')
        f.write('<table>\n')
        f.write('<tr><th>Model</th><th>Overall Score</th><th>FPS</th><th>Detections</th><th>Classes Detected</th></tr>\n')
        
        scores = scoring_details.get('scores', {})
        raw_metrics = scoring_details.get('raw_metrics', {})
        
        for model in MODEL_TYPES:
            # Determine if this is the best model
            is_best = model == best_model
            row_class = ' class="winner"' if is_best else ''
            
            score = scores.get(model, 0)
            fps = raw_metrics.get(model, {}).get('fps', 0)
            detections = raw_metrics.get(model, {}).get('total_detections', 0)
            classes = raw_metrics.get(model, {}).get('unique_classes_detected', 0)
            
            f.write(f'<tr{row_class}><td>{model.upper()}</td><td>{score:.3f}</td><td>{fps:.2f}</td>'
                   f'<td>{detections}</td><td>{classes}</td></tr>\n')
            
        f.write('</table>\n')
        
        # Detailed Metrics
        f.write('<h2>Detailed Performance Metrics</h2>\n')
        f.write('<table>\n')
        f.write('<tr><th>Metric</th>')
        for model in MODEL_TYPES:
            f.write(f'<th>{model.upper()}</th>')
        f.write('</tr>\n')
        
        # Common metrics to include
        detailed_metrics = [
            'fps', 'mean_inference_time', 'total_detections', 'unique_classes_detected',
            'avg_detections_per_image', 'successful_inferences', 'AP_IoU=0.50'
        ]
        
        metric_display_names = {
            'fps': 'Frames Per Second',
            'mean_inference_time': 'Avg. Inference Time (s)',
            'total_detections': 'Total Detections',
            'unique_classes_detected': 'Unique Classes Detected',
            'avg_detections_per_image': 'Avg. Detections per Image',
            'successful_inferences': 'Successfully Processed Images',
            'AP_IoU=0.50': 'Average Precision (IoU=0.50)'
        }
        
        # Add rows for each metric
        for metric in detailed_metrics:
            display_name = metric_display_names.get(metric, metric)
            f.write(f'<tr><td>{display_name}</td>')
            
            for model in MODEL_TYPES:
                # Highlight the best value in each row
                metric_values = [results[m].get(metric, 0) for m in MODEL_TYPES if m in results]
                if metric in ['mean_inference_time']:  # Lower is better
                    is_best = (results[model].get(metric, float('inf')) == min(metric_values)) if metric_values else False
                else:  # Higher is better
                    is_best = (results[model].get(metric, 0) == max(metric_values)) if metric_values else False
                
                cell_class = ' class="highlight"' if is_best else ''
                
                value = results[model].get(metric, 'N/A')
                if isinstance(value, (int, float)):
                    if metric in ['fps', 'mean_inference_time', 'avg_detections_per_image']:
                        formatted_value = f'{value:.2f}'
                    else:
                        formatted_value = f'{value}'
                else:
                    formatted_value = str(value)
                
                f.write(f'<td{cell_class}>{formatted_value}</td>')
            
            f.write('</tr>\n')
            
        f.write('</table>\n')
        
        # Add segmentation analysis section if available
        if "yolo-seg" in results and results["yolo-seg"].get("coco_segm_metrics"):
            f.write('<h2>Segmentation Analysis</h2>\n')
            f.write('<p>In addition to bounding box detection, YOLOv8-Seg provides pixel-level segmentation masks.</p>\n')
            
            segm_metrics = results["yolo-seg"]["coco_segm_metrics"]
            
            f.write('<table>\n')
            f.write('<tr><th>Metric</th><th>Value</th><th>Description</th></tr>\n')
            
            metrics_to_show = [
                ("Segm_AP_IoU=0.50:0.95", "Mean Average Precision across IoU thresholds"),
                ("Segm_AP_IoU=0.50", "Average Precision at IoU threshold 0.50"),
                ("Segm_AP_IoU=0.75", "Average Precision at IoU threshold 0.75"),
                ("Segm_AP_small", "AP for small objects"),
                ("Segm_AP_medium", "AP for medium objects"),
                ("Segm_AP_large", "AP for large objects"),
            ]
            
            for key, description in metrics_to_show:
                value = segm_metrics.get(key, "N/A")
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                    
                f.write(f'<tr><td>{key}</td><td>{formatted_value}</td><td>{description}</td></tr>\n')
                
            f.write('</table>\n')
            
            f.write('<p>Segmentation provides more precise object delineation, which is valuable for tasks requiring '
                    'fine-grained understanding of object boundaries, such as instance counting, area measurement, and '
                    'precise localization in robotics applications.</p>\n')
        
        # Add reliability analysis section
        f.write('<h2>Reliability Analysis</h2>\n')
        f.write('<p>Reliability in computer vision models refers to their consistency in performance across multiple runs '
                'and different conditions. The following metrics evaluate the stability of each model:</p>\n')
        
        f.write('<table>\n')
        f.write('<tr><th>Model</th><th>Success Rate (%)</th><th>Processing Stability</th><th>Detection Consistency</th></tr>\n')
        
        for model in MODEL_TYPES:
            metrics = results[model]
            success_rate = (metrics.get("successful_inferences", 0) / metrics.get("total_images", 1)) * 100
            
            # Check for variance data (if we ran reliability tests)
            if "fps_var" in metrics:
                processing_stability = f"CoV: {metrics['fps_coef_var']:.3f}"
                detection_consistency = f"±{metrics['detection_std']:.1f} objects"
            else:
                # If we don't have explicit reliability data, just show success rate
                processing_stability = "Not measured"
                detection_consistency = "Not measured"
            
            f.write(f'<tr><td>{model.upper()}</td><td>{success_rate:.1f}%</td>'
                   f'<td>{processing_stability}</td><td>{detection_consistency}</td></tr>\n')
        
        f.write('</table>\n')
        
        f.write('<p>A model with high reliability will have a high success rate (indicating few runtime failures) '
                'and low coefficient of variation (CoV) in processing speed (indicating consistent performance).</p>\n')
        
        # Add qualitative analysis section
        f.write('<h2>Qualitative Analysis</h2>\n')
        
        # Compare models features
        f.write('<h3>Feature Comparison</h3>\n')
        
        f.write('<table>\n')
        f.write('<tr><th>Feature</th><th>Mask R-CNN</th><th>YOLOv8-Seg</th></tr>\n')
        
        features = [
            ("Architecture", "Two-Stage CNN", "Single-Stage CNN"),
            ("Segmentation Support", "Yes", "Yes"),
            ("Real-time Processing", "No", "Yes"),
            ("Precision Focus", "High", "Balanced"),
            ("Small Object Detection", "Strong", "Medium"),
            ("Implementation Complexity", "High", "Low")
        ]
        
        for feature, mask_rcnn, yolo_seg in features:
            f.write(f'<tr><td>{feature}</td><td>{mask_rcnn}</td><td>{yolo_seg}</td></tr>\n')
        
        f.write('</table>\n')
        
        f.write('<h2>Conclusion</h2>\n')
        
        # Add conclusion text based on the best model
        if best_model == 'yolo-seg':
            f.write('<p>The <strong>YOLO-Seg</strong> model performed best overall, offering an excellent balance of speed '
                    'and detection quality. Its segmentation capabilities provide more detailed object boundaries, which is '
                    'valuable for applications requiring precise object localization. The model demonstrated strong detection '
                    'performance across diverse object categories while maintaining real-time processing speeds.</p>\n')
        elif best_model == 'mask-rcnn':
            f.write('<p>The <strong>Mask R-CNN</strong> model achieved the highest evaluation score, showcasing '
                    'its mature and refined detection capabilities. While not as fast as some newer models, it demonstrated '
                    'superior detection accuracy and consistency across diverse objects. Its two-stage detection approach '
                    'provides reliable results, making it well-suited for applications where detection quality is prioritized '
                    'over processing speed.</p>\n')
            
        # Add methodology notes
        f.write('<h2>Methodology Notes</h2>\n')
        f.write('<p>This evaluation was performed on a subset of the COCO validation dataset. '
                'The scoring system weighted multiple factors including:</p>\n')
        
        f.write('<ul>\n')
        weights = scoring_details.get('weights', {})
        for metric, weight in weights.items():
            f.write(f'<li><strong>{metric_display_names.get(metric, metric)}</strong>: {weight*100:.0f}%</li>\n')
        f.write('</ul>\n')
        
        # Footer
        f.write('<p><em>Generated by Computer Vision Project Evaluation Pipeline</em></p>\n')
        f.write('</body>\n</html>\n')
    
    print(f"Comparison report saved to: {report_path}")
    return str(report_path)

def run_demo_video(best_model, video_path=None):
    """
    Create a demonstration video with the best model
    
    Args:
        best_model (str): The model type to use
        video_path (str or Path, optional): Specific video to use. If None, will prompt for selection.
        
    Returns:
        str: Path to the output video file
    """
    print(f"\n=== CREATING DEMO VIDEO WITH {best_model.upper()} ===\n")
    
    # If no video specified, list available videos and select the best one for the model
    if video_path is None:
        videos = list_available_videos()
        if not videos:
            return None
        
        # Intelligently select a video based on the model
        # For segmentation models, prefer videos with people/objects for segmentation
        suggested_keywords = {
            "yolo-seg": ["person", "people", "fruit", "detection"],
            "mask-rcnn": ["person", "detection", "bicycle", "car"]
        }
        
        # Try to find a video matching the suggested keywords for this model
        keywords = suggested_keywords.get(best_model, ["detection"])
        suggested_video = None
        
        for video in videos:
            for keyword in keywords:
                if keyword.lower() in video.name.lower():
                    suggested_video = video
                    break
            if suggested_video:
                break
                
        # If no matching video found, use the first video
        if not suggested_video and videos:
            suggested_video = videos[0]
            
        if suggested_video:
            print(f"Selected video: {suggested_video}")
            video_path = suggested_video
        else:
            print("No suitable video found.")
            return None
    
    # Create the demo video
    output_path = create_demo_video(best_model, video_path)
    return output_path

def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description="Run the Computer Vision Project evaluation pipeline")
    parser.add_argument("--skip-eval", action="store_true", help="Skip the evaluation step")
    parser.add_argument("--images", type=int, default=50, help="Number of images to use in evaluation")
    parser.add_argument("--model", type=str, choices=MODEL_TYPES, help="Force using a specific model for the demo video")
    parser.add_argument("--video", type=str, help="Specific video file to use for the demo")
    
    args = parser.parse_args()
    
    print("\n===== COMPUTER VISION PROJECT PIPELINE =====\n")
    
    # Step 1: Evaluate models
    results = None
    if not args.skip_eval:
        results = run_evaluation(max_images=args.images)
    else:
        print("Skipping evaluation step...")
        # Try to load most recent results from the results directory
        result_files = list(RESULTS_DIR.glob("evaluation_results_*.json"))
        if result_files:
            # Get the most recent file
            latest_result = max(result_files, key=lambda p: p.stat().st_mtime)
            print(f"Loading previous evaluation results from {latest_result}")
            with open(latest_result, 'r') as f:
                results = json.load(f)
    
    # Step 2: Determine best model
    if args.model:
        best_model = args.model
        print(f"Using specified model: {best_model.upper()}")
        scoring_details = {"note": "User-specified model"}
    elif results:
        best_model, scoring_details = determine_best_model(results)
        print(f"Best model determined to be: {best_model.upper()}")
    else:
        print("No evaluation results available. Using default model (yolo-seg).")
        best_model = "yolo-seg"
        scoring_details = {"note": "Default model due to missing evaluation"}
    
    # Step 3: Create comparison report
    if results:
        report_path = create_comparison_report(results, scoring_details)
        print(f"Comparison report created: {report_path}")
    else:
        print("Skipping comparison report due to missing evaluation results.")
    
    # Step 4: Create demo video
    video_path = args.video if args.video else None
    if video_path and not os.path.exists(video_path):
        print(f"Warning: Specified video not found: {video_path}")
        video_path = None
        
    output_path = run_demo_video(best_model, video_path)
    
    if output_path:
        print(f"\nDemo video created: {output_path}")
        print("\nPipeline execution complete!")
        print("\nFor your report, consider including:")
        print("1. The comparison report with model evaluation metrics")
        print("2. A link to the demo video (upload it to your preferred hosting platform)")
        print("3. Screenshots from the visualizations for qualitative analysis")
    else:
        print("\nFailed to create demo video. Pipeline execution incomplete.")

if __name__ == "__main__":
    main()
