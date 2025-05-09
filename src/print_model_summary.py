"""
Print Model Summary Table

A simple utility module to print a formatted summary table of model evaluation results
to the terminal. This provides a cleaner view than the debug statements, displaying
the raw (not normalized) values in a nicely formatted table.
"""

import json
from pathlib import Path
import glob
import os

def print_summary_table(results_file=None, results_dir="inference/results"):
    """
    Prints a nicely formatted summary table of key model metrics to the terminal.
    
    Args:
        results_file (str): Path to specific results file. If None, uses the latest file.
        results_dir (str): Directory containing results files.
    """
    # Resolve results directory path
    results_dir = Path(results_dir)
    
    # Load results file
    if results_file:
        results_path = Path(results_file)
    else:
        # Find the latest results file
        list_of_files = glob.glob(str(results_dir / 'evaluation_results_*.json'))
        if not list_of_files:
            print(f"Error: No 'evaluation_results_*.json' files found in {results_dir}")
            return
        results_path = Path(max(list_of_files, key=os.path.getctime))
    
    # Check if file exists
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return
    
    # Load JSON data
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        print(f"Using results from: {results_path.name}")
    except Exception as e:
        print(f"Error loading results file: {e}")
        return
    
    # Extract models (skip metadata)
    models = [model for model in results.keys() if model != "evaluation_metadata"]
    if not models:
        print("No model data found in results file.")
        return
    
    # Extract metrics for each model
    metrics_data = {}
    metrics_to_extract = [
        "mAP (IoU=0.50:0.95)",
        "F1-Score",
        "Large Objects AP",
        "Medium Objects AP", 
        "Small Objects AP",
        "Speed (FPS)"
    ]
    
    for model in models:
        metrics_data[model] = {}
        
        # Extract mAP
        if "coco_metrics" in results[model]:
            coco_metrics = results[model]["coco_metrics"]
            metrics_data[model]["mAP (IoU=0.50:0.95)"] = coco_metrics.get("AP_IoU=0.50:0.95", 0.0)
            metrics_data[model]["Large Objects AP"] = coco_metrics.get("AP_large", 0.0)
            metrics_data[model]["Medium Objects AP"] = coco_metrics.get("AP_medium", 0.0)
            metrics_data[model]["Small Objects AP"] = coco_metrics.get("AP_small", 0.0)
            
            # Calculate F1-Score from precision and recall
            ap50 = coco_metrics.get("AP_IoU=0.50", 0.0)  # Use as precision proxy
            recall = coco_metrics.get("AR_max=100", 0.0)  # Use as recall proxy
            
            if ap50 + recall > 0:
                f1_score = 2 * (ap50 * recall) / (ap50 + recall)
            else:
                f1_score = 0.0
                
            metrics_data[model]["F1-Score"] = f1_score
        
        # Extract speed
        metrics_data[model]["Speed (FPS)"] = results[model].get("fps", 0.0)
    
    # Print summary table header
    print("\n" + "="*90)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*90)
    
    # Format headers
    headers = ["Model".ljust(15)]
    for metric in metrics_to_extract:
        display_name = metric
        if metric == "mAP (IoU=0.50:0.95)":
            display_name = "mAP"
        elif metric == "Large Objects AP":
            display_name = "Large Objects"
        elif metric == "Medium Objects AP":
            display_name = "Medium Objects"
        elif metric == "Small Objects AP":
            display_name = "Small Objects"
        headers.append(display_name.ljust(15))
    
    print("  ".join(headers))
    print("-"*90)
    
    # Print each model's metrics
    for model in models:
        row = [model.ljust(15)]
        for metric in metrics_to_extract:
            value = metrics_data[model].get(metric, 0.0)
            
            # Format based on metric type
            if metric == "Speed (FPS)":
                formatted = f"{value:.1f}"
            else:
                formatted = f"{value:.3f}"
                
            row.append(formatted.ljust(15))
        
        print("  ".join(row))
    
    print("="*90)
    print("Higher values are better for all metrics.")
    print("="*90 + "\n")

if __name__ == "__main__":
    # When run directly, print the summary table
    print_summary_table()
