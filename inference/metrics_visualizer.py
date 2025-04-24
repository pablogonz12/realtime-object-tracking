"""
Performance Metrics Visualizer for Computer Vision Project

This module provides comprehensive visualization capabilities for model evaluation results,
enabling clear comparison of metrics like precision, recall, F1-score, mAP, and efficiency
across different object detection and segmentation models.
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
from datetime import datetime

# Configure paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "inference" / "results"
VISUALIZATIONS_DIR = PROJECT_ROOT / "inference" / "results" / "visualizations"
VISUALIZATIONS_DIR.mkdir(exist_ok=True, parents=True)

# Define a consistent color scheme for models
MODEL_COLORS = {
    "yolo-seg": "#2ca02c",     # Green
    "mask-rcnn": "#d62728",    # Red
    "yolact": "#1f77b4"        # Blue
}

# Define full names for models for better labeling
MODEL_NAMES = {
    "yolo-seg": "YOLOv8-Seg",
    "mask-rcnn": "Mask R-CNN",
    "yolact": "YOLACT"
}

class MetricsVisualizer:
    """Class to visualize and compare model performance metrics"""
    
    def __init__(self, results_path=None):
        """
        Initialize the visualizer with results data
        
        Args:
            results_path (str, optional): Path to a specific results JSON file
                                         If None, will use the latest available
        """
        self.results = {}
        self.results_file = None
        
        # If no path provided, find the latest results file
        if results_path is None:
            result_files = sorted(RESULTS_DIR.glob("evaluation_results_*.json"), 
                                 key=lambda x: x.stat().st_mtime, reverse=True)
            if result_files:
                self.results_file = result_files[0]
                print(f"Using latest evaluation results: {self.results_file.name}")
            else:
                print("Warning: No evaluation results found")
                return
        else:
            self.results_file = Path(results_path)
            
        # Load results data
        if self.results_file and self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    self.results = json.load(f)
                print(f"Loaded results for {len(self.results)} models from {self.results_file}")
                
                # Convert possible strings to numbers for metrics
                for model_type, metrics in self.results.items():
                    if "coco_metrics" in metrics:
                        for key, value in metrics["coco_metrics"].items():
                            if isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                metrics["coco_metrics"][key] = float(value)
            except Exception as e:
                print(f"Error loading results from {self.results_file}: {e}")
    
    def load_multiple_results(self, pattern="evaluation_results_*.json"):
        """
        Load and aggregate multiple evaluation result files
        
        Args:
            pattern (str): Glob pattern to match result files
            
        Returns:
            dict: Aggregated results by model type
        """
        result_files = sorted(RESULTS_DIR.glob(pattern), 
                             key=lambda x: x.stat().st_mtime)
        if not result_files:
            print(f"No result files found matching pattern: {pattern}")
            return {}
        
        print(f"Found {len(result_files)} result files")
        
        # Aggregate results across all files
        aggregated_results = {}
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    results = json.load(f)
                
                # Add to aggregated results
                for model_type, metrics in results.items():
                    if model_type not in aggregated_results:
                        aggregated_results[model_type] = []
                    
                    # Ensure metrics are numeric
                    if "coco_metrics" in metrics:
                        for key, value in metrics["coco_metrics"].items():
                            if isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                metrics["coco_metrics"][key] = float(value)
                    
                    # Add file timestamp as metadata
                    timestamp_str = file_path.stem.replace("evaluation_results_", "")
                    try:
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        metrics["timestamp"] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    except:
                        metrics["timestamp"] = timestamp_str
                    
                    metrics["file"] = file_path.name
                    aggregated_results[model_type].append(metrics)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return aggregated_results
    
    def create_metrics_dashboard(self, output_path=None, show_plot=True):
        """
        Generate a comprehensive dashboard with all key metrics
        
        Args:
            output_path (str, optional): Path to save the dashboard image
            show_plot (bool): Whether to display the plot
            
        Returns:
            str: Path to the saved dashboard image
        """
        if not self.results:
            print("No results available to visualize")
            return None
        
        # Create figure with grid layout for dashboard
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig)
        
        # Extract model types and ensure consistent order
        model_types = list(self.results.keys())
        
        # Extract metrics for comparison
        metrics_data = {
            'Model': [],
            'FPS': [],
            'mAP (IoU=0.50:0.95)': [],
            'AP (IoU=0.50)': [],
            'AP (IoU=0.75)': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': [],
            'Small Objects AP': [],
            'Medium Objects AP': [],
            'Large Objects AP': []
        }
        
        for model_type in model_types:
            metrics = self.results[model_type]
            
            # Get friendly name or use original
            model_name = MODEL_NAMES.get(model_type, model_type)
            metrics_data['Model'].append(model_name)
            
            # Performance metrics
            metrics_data['FPS'].append(metrics.get('fps', 0))
            
            # COCO metrics if available - use consistent key access for all model types
            coco_metrics = metrics.get('coco_metrics', {})
            
            # Ensure we have standard metrics with proper fallbacks if any are missing
            metrics_data['mAP (IoU=0.50:0.95)'].append(coco_metrics.get('AP_IoU=0.50:0.95', 0))
            metrics_data['AP (IoU=0.50)'].append(coco_metrics.get('AP_IoU=0.50', 0))
            metrics_data['AP (IoU=0.75)'].append(coco_metrics.get('AP_IoU=0.75', 0))
            metrics_data['Small Objects AP'].append(coco_metrics.get('AP_small', 0))
            metrics_data['Medium Objects AP'].append(coco_metrics.get('AP_medium', 0))
            metrics_data['Large Objects AP'].append(coco_metrics.get('AP_large', 0))
            
            # Calculate precision, recall, F1 if possible
            tp = metrics.get('true_positives', 0)
            fp = metrics.get('false_positives', 0)
            fn = metrics.get('false_negatives', 0)
            
            # If we don't have these metrics directly, estimate from COCO metrics
            if tp == 0 and fp == 0 and fn == 0 and coco_metrics:
                # Rough approximation based on AP@0.5 and AR metrics
                precision = coco_metrics.get('AP_IoU=0.50', 0) 
                recall = coco_metrics.get('AR_max=100', 0) if 'AR_max=100' in coco_metrics else precision
            else:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics_data['Precision'].append(precision)
            metrics_data['Recall'].append(recall)
            metrics_data['F1-Score'].append(f1)
        
        # Create a DataFrame for easier plotting
        df = pd.DataFrame(metrics_data)
        
        # 1. Precision-Recall-F1 Bar Chart
        ax1 = fig.add_subplot(gs[0, 0])
        metrics_subset = df.melt(id_vars=['Model'], 
                               value_vars=['Precision', 'Recall', 'F1-Score'],
                               var_name='Metric', value_name='Value')
        
        sns.barplot(x='Model', y='Value', hue='Metric', data=metrics_subset, ax=ax1)
        ax1.set_title('Precision, Recall, F1-Score Comparison', fontsize=14)
        ax1.set_ylim(0, 1.0)
        ax1.set_xlabel('')
        ax1.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 2. AP Comparison across IoU thresholds
        ax2 = fig.add_subplot(gs[0, 1])
        ap_metrics = df.melt(id_vars=['Model'], 
                           value_vars=['mAP (IoU=0.50:0.95)', 'AP (IoU=0.50)', 'AP (IoU=0.75)'],
                           var_name='AP Metric', value_name='Value')
        
        sns.barplot(x='Model', y='Value', hue='AP Metric', data=ap_metrics, ax=ax2)
        ax2.set_title('Average Precision (AP) at Different IoU Thresholds', fontsize=14)
        ax2.set_ylim(0, 1.0)
        ax2.set_xlabel('')
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. Object Size Performance
        ax3 = fig.add_subplot(gs[0, 2])
        size_metrics = df.melt(id_vars=['Model'], 
                             value_vars=['Small Objects AP', 'Medium Objects AP', 'Large Objects AP'],
                             var_name='Object Size', value_name='AP Value')
        
        sns.barplot(x='Model', y='AP Value', hue='Object Size', data=size_metrics, ax=ax3)
        ax3.set_title('Performance by Object Size (AP)', fontsize=14)
        ax3.set_ylim(0, 1.0)
        ax3.set_xlabel('')
        ax3.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 4. Speed (FPS) Comparison
        ax4 = fig.add_subplot(gs[1, 0])
        model_colors = [MODEL_COLORS.get(model, 'gray') for model in model_types]
        bars = ax4.bar([MODEL_NAMES.get(m, m) for m in model_types], 
                     [self.results[m].get('fps', 0) for m in model_types],
                     color=model_colors)
        
        ax4.set_title('Processing Speed (FPS)', fontsize=14)
        ax4.set_xlabel('')
        ax4.set_ylabel('Frames Per Second')
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add FPS values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom')
        
        # 5. Detection Counts
        ax5 = fig.add_subplot(gs[1, 1])
        
        # Extract detection data
        detection_data = []
        labels = []
        model_names_with_data = []
        positions = []
        
        for i, model_type in enumerate(model_types):
            metrics = self.results[model_type]
            detection_counts = metrics.get('detection_counts', {})
            
            # Sort by count (descending)
            sorted_counts = sorted(detection_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Get top 10 classes
            top_classes = sorted_counts[:10]
            
            # Only add to positions and data if we have data
            if top_classes:
                detection_data.append([count for _, count in top_classes])
                labels.append([cls for cls, _ in top_classes])
                model_names_with_data.append(MODEL_NAMES.get(model_type, model_type))
                positions.append(i+1)
        
        # Create box plot only if we have data
        if detection_data:
            # Create box plot
            for i, (model, data, pos) in enumerate(zip(model_names_with_data, detection_data, positions)):
                color = MODEL_COLORS.get(model_types[model_types.index(next(m for m in model_types if MODEL_NAMES.get(m, m) == model))], 'gray')
                ax5.boxplot(data, positions=[pos], widths=0.6, patch_artist=True,
                          boxprops=dict(facecolor=color, alpha=0.7))
            
            ax5.set_title('Distribution of Detections by Class (Top 10)', fontsize=14)
            ax5.set_xlabel('')
            ax5.set_ylabel('Detections Count')
            
            # Set ticks only for positions that have data - ensure positions and labels match
            if positions:
                # This is the fix: make sure we're setting exactly the same number of positions as labels
                ax5.set_xticks(positions)
                ax5.set_xticklabels(model_names_with_data)
            ax5.grid(axis='y', linestyle='--', alpha=0.7)
        else:
            ax5.set_title('No Detection Data Available', fontsize=14)
            ax5.set_xlabel('')
            ax5.set_ylabel('Detections Count')
            ax5.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 6. F1-Score vs FPS scatter plot
        ax6 = fig.add_subplot(gs[1, 2])
        for i, model_type in enumerate(model_types):
            metrics = self.results[model_type]
            fps = metrics.get('fps', 0)
            
            # Get friendly name
            model_name = MODEL_NAMES.get(model_type, model_type)
            
            # Get F1 score
            f1 = metrics_data['F1-Score'][i]
            
            # Plot point
            ax6.scatter(fps, f1, s=100, color=MODEL_COLORS.get(model_type, 'gray'), 
                      label=model_name, alpha=0.8)
            
            # Add model name as annotation
            ax6.annotate(model_name, (fps, f1), xytext=(5, 5), 
                       textcoords='offset points')
        
        ax6.set_title('F1-Score vs. Speed Performance', fontsize=14)
        ax6.set_xlabel('Speed (FPS)')
        ax6.set_ylabel('F1-Score')
        ax6.grid(True, linestyle='--', alpha=0.7)
        
        # 7. Radar Chart for Overall Performance
        ax7 = fig.add_subplot(gs[2, 0], polar=True)
        
        # Categories for radar chart
        categories = ['mAP', 'F1-Score', 'Speed', 'Small\nObjects', 
                    'Medium\nObjects', 'Large\nObjects']
        
        N = len(categories)
        theta = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        theta += theta[:1]  # Close the polygon
        
        # Plot each model
        for i, model_type in enumerate(model_types):
            metrics = self.results[model_type]
            coco_metrics = metrics.get('coco_metrics', {})
            
            # Normalize FPS score on a 0-1 scale 
            fps = metrics.get('fps', 0)
            max_fps = max([self.results[m].get('fps', 0) for m in model_types])
            fps_score = fps / max_fps if max_fps > 0 else 0
            
            # Create data array for radar chart
            values = [
                coco_metrics.get('AP_IoU=0.50:0.95', 0),  # mAP
                metrics_data['F1-Score'][i],              # F1-Score
                fps_score,                                # Speed (normalized)
                coco_metrics.get('AP_small', 0),          # Small objects
                coco_metrics.get('AP_medium', 0),         # Medium objects
                coco_metrics.get('AP_large', 0)           # Large objects
            ]
            
            # Close the polygon
            values += values[:1]
            
            # Plot the polygon
            color = MODEL_COLORS.get(model_type, 'gray')
            ax7.plot(theta, values, color=color, linewidth=2, 
                   label=MODEL_NAMES.get(model_type, model_type))
            ax7.fill(theta, values, color=color, alpha=0.2)
        
        # Add category labels
        ax7.set_xticks(theta[:-1])
        ax7.set_xticklabels(categories, fontsize=10)
        ax7.set_ylim(0, 1)
        ax7.set_title('Multi-dimensional Performance Comparison', fontsize=14)
        ax7.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # 8. Summary table
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis('tight')
        ax8.axis('off')
        
        # Create summary table data
        table_data = []
        table_columns = ['Model', 'mAP', 'AP@50', 'AP@75', 'F1-Score', 'FPS', 
                       'Small AP', 'Med AP', 'Large AP']
        
        for model_type in model_types:
            metrics = self.results[model_type]
            coco_metrics = metrics.get('coco_metrics', {})
            
            model_name = MODEL_NAMES.get(model_type, model_type)
            
            # Find index in metrics_data
            idx = metrics_data['Model'].index(model_name) if model_name in metrics_data['Model'] else 0
            
            row = [
                model_name,
                f"{coco_metrics.get('AP_IoU=0.50:0.95', 0):.3f}",
                f"{coco_metrics.get('AP_IoU=0.50', 0):.3f}",
                f"{coco_metrics.get('AP_IoU=0.75', 0):.3f}",
                f"{metrics_data['F1-Score'][idx]:.3f}",
                f"{metrics.get('fps', 0):.1f}",
                f"{coco_metrics.get('AP_small', 0):.3f}",
                f"{coco_metrics.get('AP_medium', 0):.3f}",
                f"{coco_metrics.get('AP_large', 0):.3f}"
            ]
            table_data.append(row)
        
        # Create table
        table = ax8.table(cellText=table_data, colLabels=table_columns, 
                        loc='center', cellLoc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        
        # Apply alternating row colors
        for i, row in enumerate(table_data):
            color = MODEL_COLORS.get(model_types[i], 'gray')
            
            # Make header row colors lighter for better readability
            for j in range(len(row)):
                cell = table[(i + 1, j)]  # +1 for header row
                cell.set_facecolor(mcolors.to_rgba(color, alpha=0.1))
                
                # Highlight best values in each column
                if j > 0 and j < len(row) - 1:  # Skip model name column
                    try:
                        value = float(row[j].replace(',', '.'))
                        column_values = [float(table_data[k][j].replace(',', '.')) 
                                      for k in range(len(table_data))]
                        if value >= max(column_values) * 0.99:  # Allow for slight numerical differences
                            cell.set_facecolor(mcolors.to_rgba('green', alpha=0.3))
                    except ValueError:
                        pass
        
        # Add main title and description
        plt.suptitle('Comprehensive Model Performance Comparison', fontsize=22, y=0.98)
        plt.figtext(0.5, 0.93, 
                  f"Evaluation results from {self.results_file.name if self.results_file else 'unknown source'}", 
                  ha='center', fontsize=14, style='italic')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        
        # Save the figure if output path provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = VISUALIZATIONS_DIR / f"metrics_dashboard_{timestamp}.png"
        else:
            output_path = Path(output_path)
            # Ensure parent directory exists
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
        return str(output_path)
    
    def create_precision_recall_curve(self, output_path=None, show_plot=True):
        """
        Generate precision-recall curves for models
        
        Args:
            output_path (str, optional): Path to save the curve image
            show_plot (bool): Whether to display the plot
            
        Returns:
            str: Path to the saved curve image
        """
        if not self.results:
            print("No results available to visualize")
            return None
            
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Legend items
        legend_items = []
        
        # Plot each model's PR curve if data is available
        for model_type, metrics in self.results.items():
            # Check if we have PR curve data
            if 'precision_values' in metrics and 'recall_values' in metrics:
                precision = metrics['precision_values']
                recall = metrics['recall_values']
                
                # Plot the curve
                color = MODEL_COLORS.get(model_type, None)
                line, = plt.plot(recall, precision, label=MODEL_NAMES.get(model_type, model_type),
                               linewidth=2, color=color)
                legend_items.append(line)
            else:
                # If we don't have curve data but have single points from COCO metrics
                coco_metrics = metrics.get('coco_metrics', {})
                if 'AP_IoU=0.50' in coco_metrics:
                    # Get precision and recall estimates 
                    ap50 = coco_metrics.get('AP_IoU=0.50', 0)
                    recall_max = coco_metrics.get('AR_max=100', ap50) # Use AP50 if AR not available
                    
                    # Plot a single point
                    color = MODEL_COLORS.get(model_type, None)
                    plt.scatter(recall_max, ap50, s=100, label=f"{MODEL_NAMES.get(model_type, model_type)} (est.)",
                              marker='o', color=color)
        
        # Add labels and title
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title('Precision-Recall Curves', fontsize=18)
        plt.grid(linestyle='--', alpha=0.7)
        plt.xlim(0, 1.05)
        plt.ylim(0, 1.05)
        
        # Add legend
        plt.legend(loc='lower left', fontsize=12)
        
        # Save the figure if output path provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = VISUALIZATIONS_DIR / f"precision_recall_curve_{timestamp}.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Precision-recall curve saved to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
            
        return str(output_path)

    def create_reliability_visualization(self, reliability_data=None, output_path=None, show_plot=True):
        """
        Visualize model reliability based on repeated evaluation runs
        
        Args:
            reliability_data (dict): Data from reliability test, if None will try to load from files
            output_path (str, optional): Path to save the visualizations
            show_plot (bool): Whether to display the plots
            
        Returns:
            str: Path to the saved visualization
        """
        # If no reliability data provided, try to load from files
        if reliability_data is None:
            # Find reliability results files
            reliability_files = sorted(RESULTS_DIR.glob("reliability_results_*.json"),
                                     key=lambda x: x.stat().st_mtime, reverse=True)
            
            if not reliability_files:
                print("No reliability test results found")
                return None
                
            # Load the latest file
            latest_file = reliability_files[0]
            try:
                with open(latest_file, 'r') as f:
                    reliability_data = json.load(f)
                print(f"Loaded reliability data from {latest_file.name}")
            except Exception as e:
                print(f"Error loading reliability data: {e}")
                return None
        
        if not reliability_data:
            print("No reliability data available")
            return None
            
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract model types
        model_types = list(reliability_data.keys())
        
        # 1. FPS Consistency (Coefficient of Variation)
        coef_vars = []
        model_names = []
        
        for model_type in model_types:
            data = reliability_data[model_type]
            if isinstance(data, dict) and 'fps_coef_var' in data:
                coef_vars.append(data['fps_coef_var'])
                model_names.append(MODEL_NAMES.get(model_type, model_type))
        
        if coef_vars:
            # Lower coefficient of variation is better (more consistent)
            ax = axes[0, 0]
            bars = ax.bar(model_names, coef_vars, 
                        color=[MODEL_COLORS.get(mt, 'gray') for mt in model_types])
            
            ax.set_title('FPS Consistency (Lower is Better)', fontsize=14)
            ax.set_ylabel('Coefficient of Variation')
            ax.set_ylim(0, max(coef_vars) * 1.2)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                      f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 2. FPS Box Plot
        fps_data = []
        
        for model_type in model_types:
            data = reliability_data[model_type]
            if isinstance(data, dict) and 'run_metrics' in data:
                fps_values = [run.get('fps', 0) for run in data['run_metrics']]
                fps_data.append(fps_values)
            else:
                fps_data.append([data.get('fps_mean', 0)])
                
        if fps_data:
            ax = axes[0, 1]
            box = ax.boxplot(fps_data, patch_artist=True)
            
            # Color the boxes
            for i, patch in enumerate(box['boxes']):
                color = MODEL_COLORS.get(model_types[i % len(model_types)], 'gray')
                patch.set_facecolor(mcolors.to_rgba(color, alpha=0.6))
            
            ax.set_title('FPS Distribution Across Runs', fontsize=14)
            ax.set_ylabel('Frames Per Second')
            ax.set_xticklabels([MODEL_NAMES.get(mt, mt) for mt in model_types])
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 3. Detection Consistency
        det_vars = []
        
        for model_type in model_types:
            data = reliability_data[model_type]
            if isinstance(data, dict) and 'detection_mean' in data and 'detection_std' in data:
                # Calculate coefficient of variation for detections
                # (standard deviation / mean)
                mean = data['detection_mean']
                std = data['detection_std']
                coef_var = std / mean if mean > 0 else 0
                det_vars.append(coef_var)
            else:
                det_vars.append(0)
                
        if det_vars:
            ax = axes[1, 0]
            bars = ax.bar([MODEL_NAMES.get(mt, mt) for mt in model_types], det_vars, 
                        color=[MODEL_COLORS.get(mt, 'gray') for mt in model_types])
            
            ax.set_title('Detection Count Consistency (Lower is Better)', fontsize=14)
            ax.set_ylabel('Coefficient of Variation')
            ax.set_ylim(0, max(det_vars) * 1.2 if max(det_vars) > 0 else 0.1)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                      f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # 4. Detection Count Box Plot
        det_data = []
        
        for model_type in model_types:
            data = reliability_data[model_type]
            if isinstance(data, dict) and 'run_metrics' in data:
                det_values = [run.get('total_detections', 0) for run in data['run_metrics']]
                det_data.append(det_values)
            else:
                det_data.append([data.get('detection_mean', 0)])
                
        if det_data:
            ax = axes[1, 1]
            box = ax.boxplot(det_data, patch_artist=True)
            
            # Color the boxes
            for i, patch in enumerate(box['boxes']):
                color = MODEL_COLORS.get(model_types[i % len(model_types)], 'gray')
                patch.set_facecolor(mcolors.to_rgba(color, alpha=0.6))
            
            ax.set_title('Detection Count Distribution Across Runs', fontsize=14)
            ax.set_ylabel('Number of Detections')
            ax.set_xticklabels([MODEL_NAMES.get(mt, mt) for mt in model_types])
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add main title
        plt.suptitle('Model Reliability Analysis', fontsize=18)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure if output path provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = VISUALIZATIONS_DIR / f"reliability_analysis_{timestamp}.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Reliability visualization saved to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
        return str(output_path)
    
    def create_performance_over_time_chart(self, output_path=None, show_plot=True):
        """
        Create chart showing model performance trends over time across multiple evaluations
        
        Args:
            output_path (str, optional): Path to save the chart
            show_plot (bool): Whether to display the plot
            
        Returns:
            str: Path to the saved chart
        """
        # Load multiple results
        aggregated_results = self.load_multiple_results()
        
        if not aggregated_results:
            print("Not enough results data to create trend chart")
            return None
            
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Track all file timestamps for X axis
        timestamps = set()
        
        # Extract data for each model type
        for model_type, runs in aggregated_results.items():
            # Extract model name
            model_name = MODEL_NAMES.get(model_type, model_type)
            color = MODEL_COLORS.get(model_type, 'gray')
            
            # Get timestamps, AP and FPS values
            run_timestamps = []
            ap_values = []
            fps_values = []
            
            for run in runs:
                if 'timestamp' in run:
                    run_timestamps.append(run['timestamp'])
                    timestamps.add(run['timestamp'])
                else:
                    continue
                
                # Get AP@50 value
                coco_metrics = run.get('coco_metrics', {})
                ap50 = coco_metrics.get('AP_IoU=0.50', 0) if coco_metrics else 0
                ap_values.append(ap50)
                
                # Get FPS value
                fps = run.get('fps', 0)
                fps_values.append(fps)
            
            # Sort by timestamp
            if run_timestamps:
                # Create datapoints
                datapoints = [(t, ap, fps) for t, ap, fps in zip(run_timestamps, ap_values, fps_values)]
                datapoints.sort(key=lambda x: x[0])  # Sort by timestamp
                
                # Extract sorted data
                sorted_timestamps = [dp[0] for dp in datapoints]
                sorted_ap = [dp[1] for dp in datapoints]
                sorted_fps = [dp[2] for dp in datapoints]
                
                # Plot AP over time
                axes[0].plot(sorted_timestamps, sorted_ap, 'o-', label=model_name, color=color, linewidth=2)
                
                # Plot FPS over time
                axes[1].plot(sorted_timestamps, sorted_fps, 'o-', label=model_name, color=color, linewidth=2)
        
        # Configure AP plot
        axes[0].set_title('Average Precision (AP@IoU=0.50) Over Time', fontsize=14)
        axes[0].set_ylabel('AP@IoU=0.50')
        axes[0].set_ylim(0, 1.0)
        axes[0].grid(True, linestyle='--', alpha=0.7)
        axes[0].legend()
        
        # Rotate x-axis labels
        axes[0].tick_params(axis='x', rotation=45)
        
        # Configure FPS plot
        axes[1].set_title('Processing Speed (FPS) Over Time', fontsize=14)
        axes[1].set_ylabel('Frames Per Second')
        axes[1].grid(True, linestyle='--', alpha=0.7)
        axes[1].legend()
        
        # Rotate x-axis labels
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add main title
        plt.suptitle('Model Performance Trends Over Time', fontsize=18)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save the figure if output path provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = VISUALIZATIONS_DIR / f"performance_trends_{timestamp}.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Performance trends chart saved to {output_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
        return str(output_path)
    
    def generate_comprehensive_report(self, output_path=None):
        """
        Generate a comprehensive HTML report with all visualizations and metrics
        
        Args:
            output_path (str, optional): Path to save the report
            
        Returns:
            str: Path to the saved HTML report
        """
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        # Create timestamp for report
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate all visualizations and save them
        vis_dir = VISUALIZATIONS_DIR / f"report_{report_timestamp}"
        vis_dir.mkdir(exist_ok=True, parents=True)
        
        # Generate with show_plot=False to avoid displaying them
        dashboard_path = self.create_metrics_dashboard(
            output_path=vis_dir / "dashboard.png", 
            show_plot=False
        )
        
        pr_curve_path = self.create_precision_recall_curve(
            output_path=vis_dir / "precision_recall.png", 
            show_plot=False
        )
        
        reliability_path = self.create_reliability_visualization(
            output_path=vis_dir / "reliability.png", 
            show_plot=False
        )
        
        trends_path = self.create_performance_over_time_chart(
            output_path=vis_dir / "performance_trends.png", 
            show_plot=False
        )
        
        # Default output path if none provided
        if output_path is None:
            output_path = VISUALIZATIONS_DIR / f"comprehensive_report_{report_timestamp}.html"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Computer Vision Model Evaluation Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f8f9fa;
                }}
                h1, h2, h3 {{
                    color: #2C5AA0;
                }}
                h1 {{
                    border-bottom: 2px solid #ddd;
                    padding-bottom: 10px;
                }}
                .report-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 30px;
                }}
                .timestamp {{
                    font-style: italic;
                    color: #666;
                }}
                .visualization-section {{
                    margin: 30px 0;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 20px;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                }}
                .visualization-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .visualization-container img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #eee;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #2C5AA0;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                .footer {{
                    margin-top: 40px;
                    text-align: center;
                    color: #666;
                    font-size: 0.9em;
                }}
                .highlight {{
                    background-color: #e8f4f8;
                    padding: 15px;
                    border-left: 4px solid #2C5AA0;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="report-header">
                <h1>Computer Vision Model Evaluation Report</h1>
                <div class="timestamp">Generated on: {timestamp}</div>
            </div>
            
            <div class="highlight">
                <h3>Executive Summary</h3>
                <p>This report provides a comprehensive analysis of multiple object detection and segmentation models 
                evaluated on the COCO dataset. Performance metrics include precision, recall, F1-score, 
                mean Average Precision (mAP) at various IoU thresholds, and processing speed (FPS).</p>
            </div>
            
            <div class="visualization-section">
                <h2>Performance Dashboard</h2>
                <p>A comprehensive view of key performance metrics across all evaluated models.</p>
                <div class="visualization-container">
                    <img src="{os.path.relpath(dashboard_path, output_path.parent)}" 
                    alt="Performance Dashboard" title="Click to enlarge">
                </div>
            </div>
            
            <div class="visualization-section">
                <h2>Precision-Recall Curves</h2>
                <p>Precision-Recall curves show the trade-off between precision and recall at different detection thresholds.</p>
                <div class="visualization-container">
                    <img src="{os.path.relpath(pr_curve_path, output_path.parent)}" 
                    alt="Precision-Recall Curves" title="Click to enlarge">
                </div>
            </div>
            
            <div class="visualization-section">
                <h2>Reliability Analysis</h2>
                <p>This analysis shows the consistency of model performance across multiple evaluation runs.</p>
                <div class="visualization-container">
                    <img src="{os.path.relpath(reliability_path, output_path.parent)}" 
                    alt="Reliability Analysis" title="Click to enlarge">
                </div>
            </div>
            
            <div class="visualization-section">
                <h2>Performance Trends</h2>
                <p>How model performance has evolved over time across multiple evaluation runs.</p>
                <div class="visualization-container">
                    <img src="{os.path.relpath(trends_path, output_path.parent)}" 
                    alt="Performance Trends" title="Click to enlarge">
                </div>
            </div>
            
            <div class="visualization-section">
                <h2>Detailed Metrics</h2>
                <p>The table below presents detailed metrics for each evaluated model.</p>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>mAP (IoU=0.50:0.95)</th>
                            <th>AP@50</th>
                            <th>AP@75</th>
                            <th>Small Objects AP</th>
                            <th>Medium Objects AP</th>
                            <th>Large Objects AP</th>
                            <th>FPS</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        # Add table rows for each model
        for model_type, metrics in self.results.items():
            coco_metrics = metrics.get('coco_metrics', {})
            fps = metrics.get('fps', 0)
            
            model_name = MODEL_NAMES.get(model_type, model_type)
            
            html_content += f"""
                        <tr>
                            <td>{model_name}</td>
                            <td>{coco_metrics.get('AP_IoU=0.50:0.95', 0):.3f}</td>
                            <td>{coco_metrics.get('AP_IoU=0.50', 0):.3f}</td>
                            <td>{coco_metrics.get('AP_IoU=0.75', 0):.3f}</td>
                            <td>{coco_metrics.get('AP_small', 0):.3f}</td>
                            <td>{coco_metrics.get('AP_medium', 0):.3f}</td>
                            <td>{coco_metrics.get('AP_large', 0):.3f}</td>
                            <td>{fps:.1f}</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
            </div>
            
            <div class="visualization-section">
                <h2>Conclusions and Recommendations</h2>
                <p>Based on the evaluation results, the following conclusions can be drawn:</p>
                <ul>
        """
        
        # Add model-specific conclusions
        best_map = 0
        best_map_model = ""
        best_ap50 = 0
        best_ap50_model = ""
        best_fps = 0
        best_fps_model = ""
        
        for model_type, metrics in self.results.items():
            coco_metrics = metrics.get('coco_metrics', {})
            fps = metrics.get('fps', 0)
            model_name = MODEL_NAMES.get(model_type, model_type)
            
            map_score = coco_metrics.get('AP_IoU=0.50:0.95', 0)
            ap50_score = coco_metrics.get('AP_IoU=0.50', 0)
            
            if map_score > best_map:
                best_map = map_score
                best_map_model = model_name
                
            if ap50_score > best_ap50:
                best_ap50 = ap50_score
                best_ap50_model = model_name
                
            if fps > best_fps:
                best_fps = fps
                best_fps_model = model_name
        
        if best_map_model:
            html_content += f"""
                    <li><strong>{best_map_model}</strong> achieves the highest overall accuracy (mAP = {best_map:.3f}) 
                    across all IoU thresholds.</li>
            """
            
        if best_ap50_model and best_ap50_model != best_map_model:
            html_content += f"""
                    <li><strong>{best_ap50_model}</strong> performs best at IoU=0.50 (AP@50 = {best_ap50:.3f}).</li>
            """
            
        if best_fps_model:
            html_content += f"""
                    <li><strong>{best_fps_model}</strong> is the fastest model, processing at {best_fps:.1f} FPS.</li>
            """
            
        # Add recommendations based on use case
        html_content += """
                </ul>
                <p>Recommendations for different use cases:</p>
                <ul>
        """
        
        if best_map_model:
            html_content += f"""
                    <li>For applications requiring the highest detection accuracy: <strong>{best_map_model}</strong></li>
            """
            
        if best_fps_model:
            html_content += f"""
                    <li>For real-time applications: <strong>{best_fps_model}</strong></li>
            """
            
        # Add general recommendations
        small_gap = abs(best_map - best_ap50) < 0.1
        if small_gap and best_fps > 5:
            best_model = best_map_model if best_map > best_ap50 else best_ap50_model
            html_content += f"""
                    <li>Best overall balance of accuracy and speed: <strong>{best_model}</strong></li>
            """
            
        html_content += """
                </ul>
            </div>
            
            <div class="footer">
                <p>Generated by Computer Vision Model Evaluation Framework</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html_content)
            
        print(f"Comprehensive report generated at {output_path}")
        return str(output_path)
    
    def _determine_fastest_model(self):
        """Helper method to determine the fastest model"""
        if not self.results:
            return "N/A"
            
        fastest_model = None
        max_fps = 0
        
        for model_type, metrics in self.results.items():
            fps = metrics.get('fps', 0)
            if fps > max_fps:
                max_fps = fps
                fastest_model = model_type
                
        return MODEL_NAMES.get(fastest_model, fastest_model) if fastest_model else "N/A"
    
    def _determine_most_accurate_model(self):
        """Helper method to determine the most accurate model based on AP@50"""
        if not self.results:
            return "N/A"
            
        most_accurate_model = None
        max_ap = 0
        
        for model_type, metrics in self.results.items():
            coco_metrics = metrics.get('coco_metrics', {})
            ap = coco_metrics.get('AP_IoU=0.50', 0) if coco_metrics else 0
            
            if ap > max_ap:
                max_ap = ap
                most_accurate_model = model_type
                
        return MODEL_NAMES.get(most_accurate_model, most_accurate_model) if most_accurate_model else "N/A"
    
    def _determine_best_small_object_model(self):
        """Helper method to determine the best model for small objects"""
        if not self.results:
            return "N/A"
            
        best_model = None
        max_ap_small = 0
        
        for model_type, metrics in self.results.items():
            coco_metrics = metrics.get('coco_metrics', {})
            ap_small = coco_metrics.get('AP_small', 0) if coco_metrics else 0
            
            if ap_small > max_ap_small:
                max_ap_small = ap_small
                best_model = model_type
                
        return MODEL_NAMES.get(best_model, best_model) if best_model else "N/A"

def main():
    """Main function to run the metrics visualizer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize model performance metrics")
    parser.add_argument("--results", type=str, help="Path to specific results file (optional)")
    parser.add_argument("--output-dir", type=str, help="Directory to save visualizations")
    parser.add_argument("--no-display", action="store_true", help="Don't display plots")
    parser.add_argument("--report", action="store_true", help="Generate comprehensive HTML report")
    
    args = parser.parse_args()
    
    # Set output directory if specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = VISUALIZATIONS_DIR
    
    # Initialize visualizer
    visualizer = MetricsVisualizer(args.results)
    
    if args.report:
        # Generate full report
        visualizer.generate_comprehensive_report()
    else:
        # Generate individual visualizations
        visualizer.create_metrics_dashboard(
            output_path=output_dir / "metrics_dashboard.png",
            show_plot=not args.no_display
        )
        
        visualizer.create_precision_recall_curve(
            output_path=output_dir / "precision_recall_curve.png",
            show_plot=not args.no_display
        )
        
        visualizer.create_reliability_visualization(
            output_path=output_dir / "reliability_analysis.png",
            show_plot=not args.no_display
        )
        
        visualizer.create_performance_over_time_chart(
            output_path=output_dir / "performance_trends.png", 
            show_plot=not args.no_display
        )

if __name__ == "__main__":
    main()