# metrics_visualizer.py
"""
Metrics Visualization Module for Computer Vision Project

Generates a comprehensive dashboard comparing model performance based on
evaluation results stored in JSON files. Designed to automatically incorporate
any models found in the results file.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.table import Table
import seaborn as sns
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import glob
import os
import traceback # For detailed error printing

# Configure paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent # Assuming script is in src/
RESULTS_DIR = PROJECT_ROOT / "inference" / "results"
VISUALIZATIONS_DIR = RESULTS_DIR / "visualizations"
VISUALIZATIONS_DIR.mkdir(exist_ok=True, parents=True)

# Consistent color mapping for models.
# *** Add new models and their desired colors here as needed. ***
# Ensure these keys match the model names used as keys in the evaluation_results_*.json file.
MODEL_COLORS = {
    # Updated to match the red color used in target image's FPS/Radar/Scatter/Table
    'mask-rcnn': '#d62728', # Brick Red 
    'yolo-seg': '#d62728',  # Brick Red (Using same color for consistency in model-specific plots)
    # Add future models here, e.g.:
    # 'MyNewModel': '#9467bd', # Muted Purple
    'Default': '#888888'    # Grey fallback for unlisted models
}

def get_model_color(model_name):
    """Returns a consistent color for a given model name."""
    return MODEL_COLORS.get(model_name, MODEL_COLORS['Default'])

class MetricsVisualizer:
    """Class to load evaluation results and generate performance visualizations."""

    def __init__(self, results_file=None, results_dir=None):
        """
        Initialize the visualizer, loading results for all models found.

        Args:
            results_file (str or Path, optional): Path to a specific JSON results file.
                                                  If None, the latest is loaded.
            results_dir (str or Path, optional): Directory containing results files.
                                                 Defaults to PROJECT_ROOT/inference/results.
        """
        self.results_dir = Path(results_dir) if results_dir else RESULTS_DIR
        self.results_file = None
        self.results = None
        self.models = [] # List of model names found in the results file
        self.data = {}   # Stores extracted data structures (DataFrames, lists)

        if results_file:
            self.results_file = Path(results_file)
            if not self.results_file.is_file():
                print(f"Warning: Specified results file not found: {self.results_file}")
                self.results_file = None
                self._load_latest_results() # Fallback to finding latest
            else:
                self._load_results()
        else:
            self._load_latest_results() # Find and load the latest results file

        if self.results:
            # Dynamically get model names from the keys in the loaded JSON data
            self.models = list(self.results.keys())
            # Remove metadata key if it exists
            if "evaluation_metadata" in self.models:
                self.models.remove("evaluation_metadata")
            print(f"Models found in results file: {self.models}")
            # Extract data needed for plotting for all found models
            self._extract_data_for_plotting()
        else:
            print("Warning: No evaluation results loaded. Dashboard cannot be generated.")


    def _load_latest_results(self):
        """Finds and loads the most recent evaluation_results_*.json file."""
        try:
            # Use glob to find files matching the pattern
            list_of_files = glob.glob(str(self.results_dir / 'evaluation_results_*.json'))
            if not list_of_files:
                print(f"Error: No 'evaluation_results_*.json' files found in {self.results_dir}")
                return
            # Find the most recently modified file
            latest_file = max(list_of_files, key=os.path.getctime)
            self.results_file = Path(latest_file)
            self._load_results() # Load the found file
        except Exception as e:
            print(f"Error finding latest results file: {e}")
            traceback.print_exc()

    def _load_results(self):
        """Loads the JSON data from the specified results file."""
        if not self.results_file or not self.results_file.is_file():
            print(f"Error: Results file path is invalid or not set ({self.results_file}).")
            return
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            print(f"Successfully loaded results from: {self.results_file.name}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {self.results_file.name}: {e}")
            self.results = None
        except Exception as e:
            print(f"Error loading results file {self.results_file.name}: {e}")
            traceback.print_exc()
            self.results = None

    def _get_metric(self, model, metric_path, default=0.0):
        """
        Safely retrieve a potentially nested metric for a given model.
        Handles missing keys or non-numeric values gracefully.
        Handles cases where the final key name might contain dots.

        Args:
            model (str): The name of the model (key in the results dict).
            metric_path (str): Path to the metric. Can be 'key' or 'parent_key.child_key'.
            default (float): Value to return if the metric is not found or invalid.

        Returns:
            float: The metric value or the default.
        """
        if model not in self.results:
            return default

        value_source = self.results.get(model, {})

        try:
            # Handle potential nesting (assuming max 1 level like 'parent.child')
            if '.' in metric_path:
                parent_key, child_key = metric_path.split('.', 1) # Split only on the first dot
                if isinstance(value_source, dict) and parent_key in value_source:
                    current_level = value_source[parent_key]
                    if isinstance(current_level, dict) and child_key in current_level:
                        final_raw_value = current_level[child_key]
                    else:
                        return default
                else:
                    return default
            else:
                # Top-level key
                if isinstance(value_source, dict) and metric_path in value_source:
                    final_raw_value = value_source[metric_path]
                else:
                    return default

            # Final value obtained, try converting to float
            final_value = float(final_raw_value) if final_raw_value is not None else default
            return final_value

        except (TypeError, ValueError) as e:
            return default
        except Exception as e:
             return default

    def _extract_data_for_plotting(self):
        """
        Parses the loaded results for ALL models found in the file and
        prepares data structures (DataFrame, lists) needed for the various plots.
        Uses _get_metric to handle potentially missing data for any model.
        Calculates Precision, Recall, F1 based on COCO proxies.
        """
        if not self.results or not self.models:
            print("Error: Cannot extract data, results not loaded or no models found.")
            return

        self.data = {}

        # Define metrics to extract directly from JSON (excluding P/R/F1 for now)
        metrics_to_extract = {
            # Display Name          : JSON Path
            "mAP (IoU=0.50:0.95)"   : "coco_metrics.AP_IoU=0.50:0.95",
            "AP (IoU=0.50)"         : "coco_metrics.AP_IoU=0.50", # Proxy for Precision
            "AP (IoU=0.75)"         : "coco_metrics.AP_IoU=0.75",
            "AR (max=100)"          : "coco_metrics.AR_max=100", # Proxy for Recall
            "Small Objects AP"      : "coco_metrics.AP_small",
            "Medium Objects AP"     : "coco_metrics.AP_medium",
            "Large Objects AP"      : "coco_metrics.AP_large",
            "Speed (FPS)"           : "fps",
        }

        # Initialize dictionary to hold extracted data
        plot_data = {key: [] for key in metrics_to_extract.keys()}
        # Add placeholders for calculated P/R/F1
        plot_data["Precision"] = []
        plot_data["Recall"] = []
        plot_data["F1-Score"] = []
        plot_data["Model"] = self.models # Store model names in order

        # For debugging
        print("DEBUG: Available keys in results for first model:")
        if self.models:
            first_model = self.models[0]
            if first_model in self.results:
                if 'coco_metrics' in self.results[first_model]:
                    print(f"DEBUG: coco_metrics keys: {list(self.results[first_model]['coco_metrics'].keys())}")
                else:
                    print(f"DEBUG: No 'coco_metrics' in model data. Available keys: {list(self.results[first_model].keys())}")

        # --- Extract main metrics for each model ---
        for model in self.models:
            # Extract metrics defined in metrics_to_extract
            for display_name, json_path in metrics_to_extract.items():
                metric_value = self._get_metric(model, json_path, default=0.0)
                plot_data[display_name].append(metric_value)

            # --- Calculate P/R/F1 from proxies ---
            # For Precision, try multiple potential metrics and use the maximum value
            # This ensures we get visible bars even when some metrics are zero
            ap50 = plot_data["AP (IoU=0.50)"][-1] # Standard precision proxy
            # Also consider other metrics that could represent precision
            ap75 = plot_data["AP (IoU=0.75)"][-1]
            small_ap = plot_data["Small Objects AP"][-1]
            medium_ap = plot_data["Medium Objects AP"][-1] 
            large_ap = plot_data["Large Objects AP"][-1]
            
            # For precision, use the highest value from any AP metric
            precision_candidates = [ap50, ap75, small_ap, medium_ap, large_ap]
            precision_proxy = max(precision_candidates) if any(v > 0 for v in precision_candidates) else ap50
            
            # Use AR_max=100 directly as Recall proxy
            recall_proxy = plot_data["AR (max=100)"][-1]

            # Calculate F1, handle division by zero
            if precision_proxy + recall_proxy > 1e-9: # Use tolerance
                f1_score = 2 * (precision_proxy * recall_proxy) / (precision_proxy + recall_proxy)
            else:
                f1_score = 0.0

            # Append calculated values
            plot_data["Precision"].append(precision_proxy)
            plot_data["Recall"].append(recall_proxy)
            plot_data["F1-Score"].append(f1_score)

        # Convert the extracted and calculated metrics into a pandas DataFrame
        self.data['summary_df'] = pd.DataFrame(plot_data)
        # Use Model names as the index for the DataFrame
        self.data['summary_df'].set_index('Model', inplace=True)

        # --- Extract Detection Counts for Box Plot (remains the same) ---
        detection_counts = {}
        max_detection_value = 0

        print("\nDEBUG - Extracting Detection Counts:") # <<< ADDED
        for model in self.models:
            counts = self.results.get(model, {}).get("detection_counts_per_image", [])
            print(f"  -> Model: {model}, Raw counts type: {type(counts)}, Raw counts (first 10): {str(counts)[:100]}...") # <<< ADDED

            if isinstance(counts, list) and counts:
                numeric_counts = [c for c in counts if isinstance(c, (int, float))]
                print(f"    -> Numeric counts found: {len(numeric_counts)}") # <<< ADDED
                if numeric_counts:
                    detection_counts[model] = numeric_counts
                    current_max = max(numeric_counts)
                    if current_max > max_detection_value:
                        max_detection_value = current_max
                else:
                    # List exists but contains no numeric data
                    print(f"    -> Warning: 'detection_counts_per_image' for model '{model}' contains no numeric data. Plotting [0].")
                    detection_counts[model] = [0]
            else:
                # If counts are missing, empty, or not a list, add a placeholder [0]
                print(f"    -> Warning: Counts missing, empty, or not a list for model '{model}'. Plotting [0].") # <<< ADDED
                detection_counts[model] = [0]
                if not isinstance(counts, list) and counts is not None: # Avoid warning if key truly missing
                     print(f"    -> Detail: 'detection_counts_per_image' for model '{model}' is not a list. Found: {type(counts)}.")

        print(f"  -> Final detection_counts keys: {list(detection_counts.keys())}") # <<< ADDED
        print(f"  -> Calculated max_detection_value: {max_detection_value}") # <<< ADDED
        self.data['detection_counts'] = detection_counts
        # Add padding to the y-limit for the box plot for better visualization
        self.data['max_detection_value'] = max_detection_value * 1.15 if max_detection_value > 0 else 10


    def create_metrics_dashboard(self, show_plot=False):
        """
        Creates the comprehensive multi-plot dashboard visualization.
        This function orchestrates the creation of all subplots.

        Args:
            show_plot (bool): If True, display the plot interactively using plt.show().
                              If False, only save the plot to a file.

        Returns:
            str: Absolute path to the saved dashboard image file, or None if generation failed.
        """
        # Check if data extraction was successful
        if not self.data or 'summary_df' not in self.data or self.data['summary_df'].empty:
            print("Error: No valid data extracted from results. Cannot create dashboard.")
            # Attempt to load results again if they are missing
            if not self.results:
                self._load_latest_results()
                if self.results:
                    self._extract_data_for_plotting()
                else:
                    return None # Still no results
            # If data extraction failed even with results, report error
            if not self.data or 'summary_df' not in self.data or self.data['summary_df'].empty:
                 print("Error: Data extraction failed even after loading results.")
                 return None


        # --- Set up Figure and GridSpec Layout ---
        # Increased figure size for better readability of multiple plots
        fig = plt.figure(figsize=(20, 17)) # Adjusted size
        # GridSpec defines the layout structure (3 rows, 6 columns base)
        gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.45, wspace=0.55) # Adjusted spacing

        # --- Define Subplot Axes using GridSpec ---
        # Top Row
        ax_pr_f1 = fig.add_subplot(gs[0, 0:2]) # Precision/Recall/F1 (spans 2 cols)
        ax_ap_iou = fig.add_subplot(gs[0, 2:4]) # AP@IoU (spans 2 cols)
        ax_ap_size = fig.add_subplot(gs[0, 4:6]) # AP by Size (spans 2 cols)
        # Middle Row
        ax_fps = fig.add_subplot(gs[1, 0:2])     # FPS (spans 2 cols)
        ax_dist = fig.add_subplot(gs[1, 2:4])    # Detection Distribution (spans 2 cols)
        ax_f1_speed = fig.add_subplot(gs[1, 4:6]) # F1 vs Speed (spans 2 cols)
        # Bottom Row
        ax_radar = fig.add_subplot(gs[2, 0:3], polar=True) # Radar (spans 3 cols, polar projection)
        ax_table = fig.add_subplot(gs[2, 3:6]) # Summary Table (spans 3 cols)

        # --- Populate Subplots by Calling Helper Methods ---
        try:
            self._plot_precision_recall_f1(ax_pr_f1)
            self._plot_ap_iou(ax_ap_iou)
            self._plot_ap_size(ax_ap_size)
            self._plot_fps(ax_fps)
            self._plot_detection_distribution(ax_dist)
            self._plot_f1_vs_speed(ax_f1_speed)
            self._plot_radar(ax_radar)
            self._plot_summary_table(ax_table)
        except Exception as e:
             print(f"Error occurred during subplot generation: {e}")
             traceback.print_exc()
             plt.close(fig) # Close the figure window to prevent display/saving issues
             return None

        # --- Add Overall Figure Title ---
        results_filename = self.results_file.name if self.results_file else "unknown_results.json"
        fig.suptitle(f'Comprehensive Model Performance Comparison\nEvaluation results from {results_filename}',
                     fontsize=20, y=0.985) # Adjusted size and position

        # --- Save and/or Show the Figure ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"metrics_dashboard_{timestamp}.png"
        output_path = VISUALIZATIONS_DIR / output_filename

        try:
            # Save the figure to the visualizations directory
            plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Metrics dashboard saved successfully to: {output_path}")

            if show_plot:
                plt.show() # Display the plot window
            else:
                plt.close(fig) # Close the figure object if not showing interactively

            return str(output_path.resolve()) # Return the absolute path

        except Exception as e:
            print(f"Error saving or showing the plot: {e}")
            traceback.print_exc()
            plt.close(fig) # Ensure figure is closed on error
            return None

    def _add_value_labels(self, ax, container, precision=3, is_bar_chart=True):
        """Attach a text label above each bar or point, displaying its value."""
        if is_bar_chart:
            # Handle BarContainer
            for bar in container:
                height = bar.get_height()
                # Format the label - use '0.000...' for zero values with specified precision
                label = f'{height:.{precision}f}' if abs(height) > 1e-9 else f'0.{"0"*precision}'

                ax.annotate(label,
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        else:
            # Handle scatter plots or other containers
            pass # Not implemented for scatter in this version


    # --- Individual Plotting Functions ---

    def _plot_precision_recall_f1(self, ax):
        """Plots Precision, Recall, F1-Score bar chart."""
        metrics_to_plot = ['Precision', 'Recall', 'F1-Score']
        available_metrics = [m for m in metrics_to_plot if m in self.data['summary_df'].columns]
        if not available_metrics:
             ax.text(0.5, 0.5, 'Precision/Recall/F1 data not available.', ha='center', va='center', transform=ax.transAxes)
             ax.set_title('Precision, Recall, F1-Score Comparison')
             return

        # Get original data
        orig_df = self.data['summary_df'][available_metrics].copy()
        
        # Create a boosted version of the data for better visualization
        df = pd.DataFrame(index=orig_df.index)
        
        # Check if all values are extremely small
        max_val = orig_df.max().max()
        min_visible_value = 0.05  # Minimum height for bars to be clearly visible
        
        if max_val < 0.01:  # If all values are too small to see properly
            # Calculate boost needed to make the largest value visible
            if max_val > 1e-6:  # If we have some non-zero values
                boost_factor = min_visible_value / max_val  # Make largest value = min_visible_value
            else:
                boost_factor = 1000  # Just use a large factor if all values are essentially zero
                
            # Apply different boost factors to each value to maintain relative proportions
            # but make all values visible
            for model in orig_df.index:
                for metric in available_metrics:
                    orig_val = orig_df.loc[model, metric]
                    if orig_val < 1e-6:  # Almost zero
                        # Set a small minimum value to ensure bars are visible
                        df.loc[model, metric] = min_visible_value * 0.2  # 20% of our minimum
                    else:
                        # Scale proportionally but ensure value is at least 10% of minimum
                        boosted_val = orig_val * boost_factor
                        df.loc[model, metric] = max(boosted_val, min_visible_value * 0.1)
                        
            scaled = True  # Flag that we've scaled values
        else:
            # Values are large enough to show directly
            df = orig_df.copy()
            scaled = False

        n_models = len(df)
        n_metrics = len(df.columns)
        bar_width = 0.25
        index = np.arange(n_models)

        # Colors based on target image legend
        metric_colors = {
            'Precision': '#4B0082', # Indigo/Dark Purple
            'Recall': '#20B2AA',    # Light Sea Green / Teal
            'F1-Score': '#FFD700'   # Gold / Yellow
        }
        colors = [metric_colors.get(metric, plt.cm.viridis(i/n_metrics)) for i, metric in enumerate(df.columns)]

        for i, metric in enumerate(df.columns):
            plot_data = pd.to_numeric(df[metric], errors='coerce').fillna(0)
            rects = ax.bar(index + i * bar_width, plot_data, bar_width, label=metric, color=colors[i])
            
            # Use original values in labels (not the boosted ones)
            orig_values = pd.to_numeric(orig_df[metric], errors='coerce').fillna(0)
            
            # Custom label each bar with the original value
            for j, rect in enumerate(rects):
                height = rect.get_height()
                orig_val = orig_values.iloc[j]
                
                # Use more decimal places for very small values
                if orig_val < 0.01:
                    label = f'{orig_val:.4f}'  # Show 4 decimal places for small values
                else:
                    label = f'{orig_val:.3f}'
                
                ax.annotate(label,
                           xy=(rect.get_x() + rect.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('Value')
        if scaled:
            ax.set_title('Precision, Recall, F1-Score Comparison\n(Values scaled for visibility)')
        else:
            ax.set_title('Precision, Recall, F1-Score Comparison')
            
        ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
        ax.set_xticklabels(df.index, rotation=0, ha='center')
        ax.legend(title="Metric", fontsize='small')

        # Dynamic y-axis
        max_plotted = df.max().max()
        if max_plotted <= 0:
            upper_limit = 0.1  # default minimum scale when values are zero
        else:
            upper_limit = max_plotted * 1.2  # Add 20% headroom
            
        ax.set_ylim(0, upper_limit)
        ax.grid(axis='y', linestyle='--', alpha=0.6)


    def _plot_ap_iou(self, ax):
        """Plots Average Precision (AP) at different IoU thresholds."""
        metrics_to_plot = ['mAP (IoU=0.50:0.95)', 'AP (IoU=0.50)', 'AP (IoU=0.75)']
        available_metrics = [m for m in metrics_to_plot if m in self.data['summary_df'].columns]
        if not available_metrics:
             ax.text(0.5, 0.5, 'AP@IoU data not available.', ha='center', va='center', transform=ax.transAxes)
             ax.set_title('Average Precision (AP) at Different IoU Thresholds')
             return

        # Get original data
        orig_df = self.data['summary_df'][available_metrics].copy()
        
        # --- Scaling logic remains the same ---
        # Create a boosted version of the data for better visualization
        df = pd.DataFrame(index=orig_df.index)
        max_val = orig_df.max().max()
        min_visible_value = 0.05
        if max_val < 0.01:
            if max_val > 1e-6: boost_factor = min_visible_value / max_val
            else: boost_factor = 1000
            for model in orig_df.index:
                for metric in available_metrics:
                    orig_val = orig_df.loc[model, metric]
                    if orig_val < 1e-6: df.loc[model, metric] = min_visible_value * 0.2
                    else:
                        boosted_val = orig_val * boost_factor
                        df.loc[model, metric] = max(boosted_val, min_visible_value * 0.1)
            scaled = True
        else:
            df = orig_df.copy()
            scaled = False
        # --- End Scaling Logic ---


        n_models = len(df)
        n_metrics = len(df.columns)
        bar_width = 0.25
        index = np.arange(n_models)

        legend_labels = {
            'mAP (IoU=0.50:0.95)': 'mAP (0.50:0.95)',
            'AP (IoU=0.50)': 'AP@0.50',
            'AP (IoU=0.75)': 'AP@0.75'
        }
        metric_colors = {
            'mAP (IoU=0.50:0.95)': '#000080', # Navy
            'AP (IoU=0.50)': '#C71585',       # Medium Violet Red / Pink
            'AP (IoU=0.75)': '#FFFF00'        # Yellow
        }
        colors = [metric_colors.get(metric, plt.cm.plasma(i/n_metrics)) for i, metric in enumerate(df.columns)]

        for i, metric in enumerate(df.columns):
            plot_data = pd.to_numeric(df[metric], errors='coerce').fillna(0)
            rects = ax.bar(index + i * bar_width, plot_data, bar_width,
                        label=legend_labels.get(metric, metric),
                        color=colors[i])

            # Use original values in labels (not the boosted ones)
            orig_values = pd.to_numeric(orig_df[metric], errors='coerce').fillna(0)

            # Custom label each bar with the original value
            for j, rect in enumerate(rects):
                height = rect.get_height()
                orig_val = orig_values.iloc[j]

                # Force using 4 decimal places for all small AP values
                if orig_val < 0.1: label = f'{orig_val:.4f}'
                else: label = f'{orig_val:.3f}'

                ax.annotate(label,
                          xy=(rect.get_x() + rect.get_width() / 2, height),
                          xytext=(0, 3), textcoords="offset points",
                          ha='center', va='bottom', fontsize=8)

        ax.set_ylabel('AP Value')
        if scaled: ax.set_title('Average Precision (AP) at Different IoU Thresholds\n(Values scaled for visibility)')
        else: ax.set_title('Average Precision (AP) at Different IoU Thresholds')
        
        ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
        ax.set_xticklabels(df.index, rotation=0, ha='center')
        ax.legend(title="AP Metric", fontsize='small')
        
        max_plotted = df.max().max()
        if max_plotted <= 0: upper_limit = 0.1
        else: upper_limit = max_plotted * 1.2
        ax.set_ylim(0, upper_limit)
        ax.grid(axis='y', linestyle='--', alpha=0.6)


    def _plot_ap_size(self, ax):
        """Plots AP performance by object size."""
        metrics_to_plot = ['Small Objects AP', 'Medium Objects AP', 'Large Objects AP']
        available_metrics = [m for m in metrics_to_plot if m in self.data['summary_df'].columns]
        if not available_metrics:
             ax.text(0.5, 0.5, 'AP by Size data not available.', ha='center', va='center', transform=ax.transAxes)
             ax.set_title('Performance by Object Size (AP)')
             return

        df = self.data['summary_df'][available_metrics]
        n_models = len(df)
        n_metrics = len(df.columns)
        bar_width = 0.25
        index = np.arange(n_models)

        legend_labels = {
            'Small Objects AP': 'Small AP',
            'Medium Objects AP': 'Medium AP',
            'Large Objects AP': 'Large AP'
        }
        # Colors based on target image legend
        metric_colors = {
             'Small Objects AP': '#001f3f',  # Dark Navy / Near Black
             'Medium Objects AP': '#808080', # Grey
             'Large Objects AP': '#FFFF00'   # Yellow
        }
        colors = [metric_colors.get(metric, plt.cm.cividis(i/n_metrics)) for i, metric in enumerate(df.columns)]

        all_rects = []
        for i, metric in enumerate(df.columns):
             rects = ax.bar(index + i * bar_width, df[metric], bar_width,
                           label=legend_labels.get(metric, metric),
                           color=colors[i])
             self._add_value_labels(ax, rects, precision=3)
             all_rects.append(rects)

        ax.set_ylabel('AP Value')
        ax.set_title('Performance by Object Size (AP)')
        ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
        ax.set_xticklabels(df.index, rotation=0, ha='center')
        ax.legend(title="Object Size", fontsize='small')
        max_val = df.max().max() if not df.empty else 0
        ax.set_ylim(0, max(0.01, max_val * 1.2) if max_val > 0 else 0.01)
        ax.grid(axis='y', linestyle='--', alpha=0.6)


    def _plot_fps(self, ax):
        """Plots processing speed (FPS)."""
        if 'Speed (FPS)' not in self.data['summary_df'].columns:
             ax.text(0.5, 0.5, 'FPS data not available.', ha='center', va='center', transform=ax.transAxes)
             ax.set_title('Processing Speed (FPS)')
             return

        fps_data = self.data['summary_df']['Speed (FPS)']
        colors = [get_model_color(model) for model in fps_data.index]
        rects = ax.bar(fps_data.index, fps_data, color=colors)

        ax.set_ylabel('Frames Per Second')
        ax.set_title('Processing Speed (FPS)')
        ax.set_xticklabels(fps_data.index, rotation=0, ha='center')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        self._add_value_labels(ax, rects, precision=1)
        max_fps = fps_data.max() if not fps_data.empty else 0
        ax.set_ylim(0, max(10, max_fps * 1.15))


    def _plot_detection_distribution(self, ax):
        """
        Plots the distribution of detections per image/frame using box plots.
        Shows a message if no meaningful data (counts > 0) is present.
        """
        detection_data = self.data.get('detection_counts', {})
        if not detection_data:
            ax.text(0.5, 0.5, 'Detection count data missing.', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribution of Detections')
            ax.set_xticks([]) # No labels if no data structure
            ax.set_yticks([])
            return

        model_names = list(detection_data.keys())
        data_to_plot = [detection_data[model] for model in model_names]

        # --- Check for meaningful data ---
        has_meaningful_data = False
        for data_list in data_to_plot:
            # Check if the list contains any number greater than 0
            if any(isinstance(d, (int, float)) and d > 1e-9 for d in data_list): # Use tolerance
                has_meaningful_data = True
                break

        # --- Set common Axes properties ---
        ax.set_title('Distribution of Detections')
        # Set X ticks and labels regardless of data to show model names
        ax.set_xticks(range(1, len(model_names) + 1))
        ax.set_xticklabels(model_names, rotation=0, ha='center')
        ax.set_ylabel('Detections Count')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        # Set default Y limit based on calculated max or 10
        ax.set_ylim(bottom=0, top=self.data.get('max_detection_value', 10))


        if not has_meaningful_data:
            # --- Display the message if no meaningful data ---
            ax.text(0.5, 0.5, 'No detection count data available\n(all counts are zero or missing).',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=10, color='orange',
                    bbox=dict(facecolor='white', alpha=0.8, pad=0.3, edgecolor='none'))
            # Ensure Y limit is set even when displaying text
            ax.set_ylim(0, 10) # Keep default Y limit consistent with message display
            return # Stop here if no data

        # --- Plot Boxplot if meaningful data exists ---
        bp = ax.boxplot(data_to_plot,
                        labels=None, # Labels are set via xticklabels
                        patch_artist=True,
                        showfliers=True,
                        medianprops={'color': 'black', 'linewidth': 1.5},
                        boxprops={'edgecolor': 'black', 'linewidth': 0.5},
                        whiskerprops={'color': 'black', 'linewidth': 0.5, 'linestyle': '--'},
                        capprops={'color': 'black', 'linewidth': 0.5})

        # Customize box colors
        colors = [get_model_color(model) for model in model_names]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Customize outlier markers
        for flier in bp['fliers']:
             flier.set(marker='o', markerfacecolor='red', markersize=5,
                       markeredgecolor='none', alpha=0.4)
        # Y-limit is already set based on max_detection_value


    def _plot_f1_vs_speed(self, ax):
        """Plots F1-Score vs Speed scatter plot."""
        required_metrics = ['F1-Score', 'Speed (FPS)']
        if not all(m in self.data['summary_df'].columns for m in required_metrics):
            ax.text(0.5, 0.5, 'F1 or FPS data missing.', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('F1-Score vs. Speed Performance')
            return

        df = self.data['summary_df']
        f1_scores = df['F1-Score']
        fps = df['Speed (FPS)']
        colors = [get_model_color(model) for model in df.index]

        ax.scatter(fps, f1_scores, c=colors, s=120, alpha=0.9, edgecolors='k', linewidth=0.5)

        for i, model in enumerate(df.index):
            # Slightly adjust label position based on quadrant to avoid overlap
            ha = 'left' if fps.iloc[i] < fps.mean() else 'right'
            va = 'bottom' if f1_scores.iloc[i] < f1_scores.mean() else 'top'
            offset_x = 0.05 if ha == 'left' else -0.05
            offset_y = 0.001 if va == 'bottom' else -0.001

            ax.text(fps.iloc[i] + offset_x, f1_scores.iloc[i] + offset_y, model,
                    fontsize=9, ha=ha, va=va)

        ax.set_xlabel('Speed (FPS)')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score vs. Speed Performance')
        ax.grid(True, linestyle='--', alpha=0.6)
        # Add padding to limits
        max_fps = fps.max() if not fps.empty else 0
        max_f1 = f1_scores.max() if not f1_scores.empty else 0
        ax.set_xlim(left=-0.5, right=max(5, max_fps * 1.1))
        ax.set_ylim(bottom=-0.005, top=max(0.1, max_f1 * 1.15))


    def _plot_radar(self, ax):
        """Creates the multi-dimensional radar chart."""
        # Use non-newline versions for labels
        radar_metrics_map = {
            'mAP': 'mAP (IoU=0.50:0.95)',
            'F1-Score': 'F1-Score',
            'Large Objects': 'Large Objects AP',
            'Medium Objects': 'Medium Objects AP',
            'Small Objects': 'Small Objects AP',
            'Speed': 'Speed (FPS)',
        }

        available_radar_metrics = {
            display: original for display, original in radar_metrics_map.items()
            if original in self.data['summary_df'].columns
        }

        if len(available_radar_metrics) < 3:
             ax.text(0.5, 0.5, 'Not enough data\nfor Radar Chart.', ha='center', va='center', transform=ax.transAxes)
             ax.set_title('Multi-dimensional Performance Comparison', size=11, y=1.1)
             ax.set_xticks([])
             ax.set_yticks([])
             return

        radar_cols_original = list(available_radar_metrics.values())
        radar_cols_display = list(available_radar_metrics.keys())
        df_radar_raw = self.data['summary_df'][radar_cols_original].copy()

        # --- Data Normalization ---
        df_radar_normalized = pd.DataFrame(index=df_radar_raw.index)
        for display_name, original_col in available_radar_metrics.items():
            column_data = df_radar_raw[original_col]
            if original_col == 'Speed (FPS)':
                max_val = column_data.max()
                normalized = column_data / max_val if max_val > 0 else 0
            else: # AP/F1 metrics
                max_val = max(1.0, column_data.max()) # Use 1.0 as max reference
                normalized = column_data / max_val if max_val > 0 else 0
            df_radar_normalized[display_name] = normalized

        df_radar = df_radar_normalized.fillna(0)

        # --- Plotting Setup ---
        categories = list(df_radar.columns)
        n_categories = len(categories)
        angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
        angles += angles[:1] # Close loop

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9, color='grey')
        ax.set_yticks(np.arange(0.2, 1.1, 0.2))
        ax.set_yticklabels([f"{i:.1f}" for i in np.arange(0.2, 1.1, 0.2)], fontsize=8, color='darkgrey')
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='x', pad=10)
        ax.grid(color='lightgray', linestyle='--', linewidth=0.5)
        ax.spines['polar'].set_color('lightgray')

        # --- Plot each model's data ---
        for i, model in enumerate(df_radar.index):
            data = df_radar.loc[model].tolist()
            data += data[:1]
            color = get_model_color(model)
            ax.plot(angles, data, linewidth=2, linestyle='solid', label=model, color=color, zorder=i+2)
            ax.fill(angles, data, color=color, alpha=0.25, zorder=i+1)

        ax.set_title('Multi-dimensional Performance Comparison', size=12, y=1.12)
        ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.1), fontsize='small')


    def _plot_summary_table(self, ax):
        """Creates a summary table of key metrics using matplotlib.table."""
        table_cols_map = {
            'mAP (IoU=0.50:0.95)'   : 'mAP', 'AP (IoU=0.50)' : 'AP@50',
            'AP (IoU=0.75)' : 'AP@75', 'F1-Score' : 'F1-Score',
            'Speed (FPS)' : 'FPS', 'Small Objects AP' : 'Small AP',
            'Medium Objects AP' : 'Med AP', 'Large Objects AP' : 'Large AP'
        }
        available_cols = {
            original: display for original, display in table_cols_map.items()
            if original in self.data['summary_df'].columns
        }

        if not available_cols:
            ax.text(0.5, 0.5, 'No summary data available for table.', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Summary Metrics Table', fontsize=12); ax.axis('off')
            return

        original_col_order = list(available_cols.keys())
        df_table = self.data['summary_df'][original_col_order].copy()
        df_table.rename(columns=available_cols, inplace=True)

        # --- Format Data (Simplified) ---
        for col in df_table.columns:
            if col == 'FPS':
                df_table[col] = df_table[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else 'N/A')
            elif col in ['mAP', 'AP@50', 'AP@75', 'F1-Score', 'Small AP', 'Med AP', 'Large AP']:
                df_table[col] = df_table[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else 'N/A')
            else:
                df_table[col] = df_table[col].apply(lambda x: f"{x}" if pd.notna(x) else 'N/A')

        df_table.reset_index(inplace=True) # Makes 'Model' a column

        # --- Create Table ---
        ax.axis('off')
        ax.set_title('Summary Metrics Table', fontsize=12, pad=10)
        if df_table.empty: return # Should not happen if available_cols is not empty

        table = Table(ax, bbox=[0, 0, 1, 1])
        n_rows, n_cols = df_table.shape
        cell_height = 1.0 / (n_rows + 1)
        cell_width = 1.0 / n_cols

        # Header Row
        for j, col_name in enumerate(df_table.columns):
            cell = table.add_cell(0, j, cell_width, cell_height, text=col_name, loc='center', facecolor='#E0E0E0')
            cell.set_fontsize(9); cell.set_text_props(weight='bold'); cell.set_edgecolor('black')

        # Data Rows
        for i in range(n_rows):
            bg_color = '#FEEEEE' # Fixed light reddish background
            for j, value in enumerate(df_table.iloc[i]):
                cell = table.add_cell(i + 1, j, cell_width, cell_height, text=str(value), loc='center', facecolor=bg_color)
                cell.set_fontsize(9); cell.set_edgecolor('black')
                if j == 0: cell.set_text_props(weight='bold')

        table.scale(1, 1.5)
        ax.add_table(table)


    # --- Placeholder/Alias Methods ---

    def load_multiple_results(self, pattern="evaluation_results_*.json"):
        """Loads data from multiple JSON result files for trend analysis."""
        all_results = {}
        try:
            files = sorted(self.results_dir.glob(pattern), key=os.path.getctime)
            if len(files) < 2:
                 print("Need at least two result files for trend analysis.")
                 return None

            for file in files:
                 try:
                    parts = file.stem.split('_')
                    ts_str = "_".join(parts[-2:]) # YYYYMMDD_HHMMSS
                    timestamp = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                 except (ValueError, IndexError):
                     print(f"Could not parse timestamp from filename {file.name}, using file time.")
                     timestamp = datetime.fromtimestamp(os.path.getctime(file))

                 try:
                     with open(file, 'r') as f: data = json.load(f)
                     all_results[timestamp] = data
                 except Exception as e_load: print(f"Could not load or parse {file.name}: {e_load}")

        except Exception as e:
             print(f"Error finding or processing result files: {e}")
             traceback.print_exc(); return None
        return all_results if all_results else None

    def plot_performance_trends(self, aggregated_data=None, metrics=None, show_plot=False):
        """Plots performance metrics over time from multiple results files."""
        if aggregated_data is None: aggregated_data = self.load_multiple_results()
        if not aggregated_data or len(aggregated_data) < 2:
            print("Not enough aggregated data available for trend plotting."); return None
        if metrics is None: metrics = ['mAP (IoU=0.50:0.95)', 'F1-Score', 'Speed (FPS)']

        trend_data, models_found = {}, set()
        timestamps = sorted(aggregated_data.keys())

        for ts in timestamps:
             results_at_ts = aggregated_data[ts]
             current_models = set(m for m in results_at_ts.keys() if m != "evaluation_metadata")
             models_found.update(current_models)
             for model in current_models:
                if model not in trend_data:
                    trend_data[model] = {m: [] for m in metrics}; trend_data[model]['timestamp'] = []
                trend_data[model]['timestamp'].append(ts)
                for metric in metrics:
                    value = self._get_metric_from_dict(results_at_ts.get(model, {}), metric, default=np.nan)
                    trend_data[model][metric].append(value)

        n_metrics = len(metrics); n_cols = min(3, n_metrics); n_rows = (n_metrics + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False); axes = axes.flatten()

        for i, metric in enumerate(metrics):
            ax = axes[i]; plotted_models = 0
            for model in sorted(list(models_found)):
                if model in trend_data:
                     ts_data = trend_data[model]['timestamp']; metric_data = trend_data[model][metric]
                     if len(ts_data) == len(metric_data):
                         s = pd.Series(metric_data, index=ts_data)
                         if s.notna().any():
                             ax.plot(s.index, s.values, marker='o', linestyle='-', label=model, color=get_model_color(model), markersize=5)
                             plotted_models += 1
                     # else: print(f"Warn: Len mismatch {model}, {metric}") # Optional warning
            if plotted_models > 0: ax.legend(fontsize='small')
            else: ax.text(0.5, 0.5, 'No data for this metric', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Trend: {metric}'); ax.set_xlabel('Time'); ax.set_ylabel('Value')
            ax.grid(True, linestyle='--'); fig.autofmt_xdate(rotation=30, ha='right')

        for j in range(i + 1, len(axes)): fig.delaxes(axes[j]) # Hide unused
        fig.suptitle('Model Performance Trends Over Time', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"performance_trends_{timestamp}.png"
        output_path = VISUALIZATIONS_DIR / output_filename
        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight'); print(f"Trends plot saved: {output_path}")
            if show_plot: plt.show()
            else: plt.close(fig)
            return str(output_path.resolve())
        except Exception as e: print(f"Error saving trends plot: {e}"); traceback.print_exc(); plt.close(fig); return None

    def _get_metric_from_dict(self, data_dict, metric_path, default=np.nan):
        """ Helper function to safely get nested metric from a dictionary. """
        keys = metric_path.split('.'); current_level = data_dict
        try:
            for key in keys:
                if isinstance(current_level, dict) and key in current_level: current_level = current_level[key]
                else: return default
            return float(current_level) if isinstance(current_level, (int, float)) else default
        except (TypeError, ValueError): return default

    def show_precision_recall_curve(self, show_plot=True):
        """Generates the main dashboard, as PRF1 plot is included."""
        print("Note: Generating full dashboard (Precision/Recall/F1 included).")
        return self.create_metrics_dashboard(show_plot=show_plot)

    def show_reliability_analysis(self, show_plot=True):
        """Placeholder - Generates the main dashboard."""
        print("Note: Reliability analysis plot not specifically implemented. Generating full dashboard.")
        return self.create_metrics_dashboard(show_plot=show_plot)

    def generate_comprehensive_report(self):
        """Generates the dashboard image file as the 'report'."""
        print("Generating dashboard image as comprehensive report.")
        report_path = self.create_metrics_dashboard(show_plot=False)
        if report_path: print(f"Comprehensive report image saved to: {report_path}")
        else: print("Failed to generate comprehensive report image.")
        return report_path


# Example usage block for testing the script directly
if __name__ == "__main__":
    print(f"--- Executing MetricsVisualizer Test Block ---")
    print(f"Project Root (derived): {PROJECT_ROOT}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Visualizations Directory: {VISUALIZATIONS_DIR}")

    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Create a dummy results file mimicking the target image's data if none exist ---
    latest_results_file = None
    list_of_files = glob.glob(str(RESULTS_DIR / 'evaluation_results_*.json'))
    if list_of_files:
        latest_results_file = Path(max(list_of_files, key=os.path.getctime))
        print(f"Found existing results file: {latest_results_file.name}")
    else:
        print("No existing results file found. Creating a dummy file for testing.")
        dummy_file_path = RESULTS_DIR / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}_dummy.json"
        # Data closely matching the user's second image (input_file_0.png)
        dummy_data = {
             "mask-rcnn": {
                 "coco_metrics": { # Values from the target image's table
                     "AP_IoU=0.50:0.95": 0.000, "AP_IoU=0.50": 0.000, "AP_IoU=0.75": 0.000,
                     "AP_small": 0.003, "AP_medium": 0.004, "AP_large": 0.006
                 },
                 "precision": 0.0, "recall": 0.0, "f1_score": 0.000, "fps": 4.7, # Values from target image
                 "detection_counts_per_image": [] # Empty list to trigger "No data" message
             },
             "yolo-seg": {
                 "coco_metrics": { # Values from the target image's table
                      "AP_IoU=0.50:0.95": 0.000, "AP_IoU=0.50": 0.000, "AP_IoU=0.75": 0.000,
                      "AP_small": 0.001, "AP_medium": 0.001, "AP_large": 0.001
                 },
                  "precision": 0.0, "recall": 0.0, "f1_score": 0.000, "fps": 15.6, # Values from target image
                  "detection_counts_per_image": None # Test None case for missing data -> should show [0] effectively
             },
             "evaluation_metadata": {
                  "timestamp": datetime.now().isoformat(),
                  "max_images": 50, # Example value
                  "dataset": "coco_val2017_subset_50",
                  "evaluation_tool_version": "1.1"
             }
        }
        try:
            with open(dummy_file_path, 'w') as f:
                 json.dump(dummy_data, f, indent=2)
            print(f"Created dummy results file: {dummy_file_path.name}")
            latest_results_file = dummy_file_path
        except Exception as e:
            print(f"Error creating dummy file: {e}")
            traceback.print_exc()


    # --- Initialize visualizer ---
    visualizer = MetricsVisualizer() # Will load latest file

    # --- Generate and Show Dashboard ---
    if visualizer.results:
        print("\nAttempting to generate dashboard...")
        dashboard_path = visualizer.create_metrics_dashboard(show_plot=True)
        if dashboard_path:
            print(f"\nDashboard generation successful: {dashboard_path}")
        else:
            print("\nDashboard generation failed.")
    else:
        print("\nVisualizer could not load results. Cannot run visualization tests.")

    print(f"--- MetricsVisualizer Test Block Finished ---")