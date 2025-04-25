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
    'Faster R-CNN': '#1f77b4', # Muted Blue (Matches your image)
    'RT-DETR': '#ff7f0e',    # Safety Orange (Matches your image)
    'YOLOv8-Seg': '#2ca02c', # Cooked Asparagus Green (Matches your image)
    # Add future models here, e.g.:
    # 'MyNewModel': '#9467bd', # Muted Purple
    'Default': '#d62728'    # Brick Red (Fallback for unlisted models)
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
            # This makes the class automatically adapt to the models present in the file.
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
            self.results = None

    def _get_metric(self, model, metric_path, default=0.0):
        """
        Safely retrieve a potentially nested metric for a given model.
        Handles missing keys or non-numeric values gracefully.

        Args:
            model (str): The name of the model (key in the results dict).
            metric_path (str): Dot-separated path to the metric (e.g., "coco_metrics.AP_small").
            default (float): Value to return if the metric is not found or invalid.

        Returns:
            float: The metric value or the default.
        """
        value = self.results.get(model, {}) # Get the dictionary for the specific model
        keys = metric_path.split('.')
        try:
            current_level = value
            for key in keys:
                # Check if the current level is a dictionary and contains the key
                if isinstance(current_level, dict) and key in current_level:
                    current_level = current_level[key]
                else:
                    # Key not found at this level
                    # print(f"Debug: Key '{key}' not found for model '{model}' in path '{metric_path}'")
                    return default
            # Final value obtained, try converting to float
            # Handle None explicitly before float conversion
            return float(current_level) if current_level is not None else default
        except (TypeError, ValueError) as e:
            # Handles cases where the final value isn't a number (e.g., a string, list)
            # print(f"Debug: Error converting value to float for {metric_path} in {model}. Value: {current_level}. Error: {e}")
            return default
        except Exception as e:
             # Catch any other unexpected errors during access
            # print(f"Debug: Unexpected error accessing {metric_path} for {model}: {e}")
             return default


    def _extract_data_for_plotting(self):
        """
        Parses the loaded results for ALL models found in the file and
        prepares data structures (DataFrame, lists) needed for the various plots.
        Uses _get_metric to handle potentially missing data for any model.
        """
        if not self.results or not self.models:
            print("Error: Cannot extract data, results not loaded or no models found.")
            return

        self.data = {}

        # Define all metrics needed across all plots and their paths in the JSON
        metrics_to_extract = {
            # Display Name          : JSON Path
            "Precision"             : "precision",
            "Recall"                : "recall",
            "F1-Score"              : "f1_score",
            "mAP (IoU=0.50:0.95)"   : "coco_metrics.AP_IoU=0.50:0.95",
            "AP (IoU=0.50)"         : "coco_metrics.AP_IoU=0.50",
            "AP (IoU=0.75)"         : "coco_metrics.AP_IoU=0.75",
            "Small Objects AP"      : "coco_metrics.AP_small",
            "Medium Objects AP"     : "coco_metrics.AP_medium",
            "Large Objects AP"      : "coco_metrics.AP_large",
            "Speed (FPS)"           : "fps",
        }

        # Initialize dictionary to hold extracted data
        plot_data = {key: [] for key in metrics_to_extract.keys()}
        plot_data["Model"] = self.models # Store model names in order

        # --- Extract main metrics for each model ---
        # This loop inherently handles any number of models found in self.models
        for model in self.models:
            for display_name, json_path in metrics_to_extract.items():
                # Use the safe getter function; handles missing metrics per model
                metric_value = self._get_metric(model, json_path, default=0.0)
                plot_data[display_name].append(metric_value)

        # Convert the extracted metrics into a pandas DataFrame for convenience
        self.data['summary_df'] = pd.DataFrame(plot_data)
        # Use Model names as the index for the DataFrame
        self.data['summary_df'].set_index('Model', inplace=True)

        # --- Extract Detection Counts for Box Plot ---
        # Assumes 'detection_counts_per_image' is a list stored for each model in the JSON
        detection_counts = {}
        max_detection_value = 0 # To set an appropriate y-axis limit later

        for model in self.models:
            # Safely get the list of counts, default to empty list if missing
            counts = self.results.get(model, {}).get("detection_counts_per_image", [])

            if isinstance(counts, list) and counts: # Check if it's a non-empty list
                detection_counts[model] = counts
                current_max = max(counts)
                if current_max > max_detection_value:
                    max_detection_value = current_max
            else:
                # If counts are missing, empty, or not a list, add a placeholder [0]
                # This allows the model to appear on the box plot axis but shows no distribution.
                detection_counts[model] = [0]
                if not isinstance(counts, list):
                     print(f"Warning: 'detection_counts_per_image' for model '{model}' is not a list. Found: {type(counts)}. Plotting [0].")
                elif not counts:
                      print(f"Warning: 'detection_counts_per_image' for model '{model}' is empty. Plotting [0].")
                # Implicitly handles case where the key was missing entirely


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
             import traceback
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
            plt.close(fig) # Ensure figure is closed on error
            return None

    def _add_value_labels(self, ax, container, precision=3, is_bar_chart=True):
        """Attach a text label above each bar or point, displaying its value."""
        if is_bar_chart:
            # Handle BarContainer
            for bar in container:
                height = bar.get_height()
                # Only add label if height is significantly different from zero (optional)
                # if abs(height) > 1e-6:
                ax.annotate(f'{height:.{precision}f}' if abs(height) > 1e-9 else '0', # Format or show '0'
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8)
        else:
            # Handle scatter plots or other containers (assuming container is iterable points)
             # This part might need adjustment based on the specific plot type if not bar chart
             pass # Placeholder - specific logic needed for scatter labels if required


    # --- Individual Plotting Functions ---
    # These functions take an Axes object and plot the specific data onto it.
    # They rely on self.data being populated by _extract_data_for_plotting.

    def _plot_precision_recall_f1(self, ax):
        """Plots Precision, Recall, F1-Score bar chart."""
        metrics_to_plot = ['Precision', 'Recall', 'F1-Score']
        # Check if required columns exist
        available_metrics = [m for m in metrics_to_plot if m in self.data['summary_df'].columns]
        if not available_metrics:
             ax.text(0.5, 0.5, 'Precision/Recall/F1 data not available.', ha='center', va='center', transform=ax.transAxes)
             ax.set_title('Precision, Recall, F1-Score Comparison')
             return

        df = self.data['summary_df'][available_metrics]
        n_models = len(df)
        n_metrics = len(df.columns)
        bar_width = 0.25
        index = np.arange(n_models)

        # Use a perceptually uniform colormap like 'viridis'
        colors = plt.cm.viridis(np.linspace(0, 1, n_metrics))

        all_rects = [] # To store all bar containers for labeling
        for i, metric in enumerate(df.columns):
            rects = ax.bar(index + i * bar_width, df[metric], bar_width,
                           label=metric, color=colors[i])
            self._add_value_labels(ax, rects, precision=3) # Add labels per metric group
            all_rects.append(rects)

        ax.set_ylabel('Value')
        ax.set_title('Precision, Recall, F1-Score Comparison')
        ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
        ax.set_xticklabels(df.index, rotation=0, ha='center')
        ax.legend(title="Metric", fontsize='small')
        ax.set_ylim(0, 1.05) # Standard 0-1 scale for these metrics
        ax.grid(axis='y', linestyle='--', alpha=0.6)


    def _plot_ap_iou(self, ax):
        """Plots Average Precision (AP) at different IoU thresholds."""
        metrics_to_plot = ['mAP (IoU=0.50:0.95)', 'AP (IoU=0.50)', 'AP (IoU=0.75)']
        available_metrics = [m for m in metrics_to_plot if m in self.data['summary_df'].columns]
        if not available_metrics:
             ax.text(0.5, 0.5, 'AP@IoU data not available.', ha='center', va='center', transform=ax.transAxes)
             ax.set_title('Average Precision (AP) at Different IoU Thresholds')
             return

        df = self.data['summary_df'][available_metrics]
        n_models = len(df)
        n_metrics = len(df.columns)
        bar_width = 0.25
        index = np.arange(n_models)

        # Shorten labels slightly for the legend if needed
        legend_labels = {
            'mAP (IoU=0.50:0.95)': 'mAP (0.50:0.95)',
            'AP (IoU=0.50)': 'AP@0.50',
            'AP (IoU=0.75)': 'AP@0.75'
        }
        # Use a different colormap like 'plasma'
        colors = plt.cm.plasma(np.linspace(0, 1, n_metrics))

        all_rects = []
        for i, metric in enumerate(df.columns):
             rects = ax.bar(index + i * bar_width, df[metric], bar_width,
                           label=legend_labels.get(metric, metric), # Use shortened label
                           color=colors[i])
             self._add_value_labels(ax, rects, precision=3)
             all_rects.append(rects)

        ax.set_ylabel('AP Value')
        ax.set_title('Average Precision (AP) at Different IoU Thresholds')
        ax.set_xticks(index + bar_width * (n_metrics - 1) / 2)
        ax.set_xticklabels(df.index, rotation=0, ha='center')
        ax.legend(title="AP Metric", fontsize='small')
        # Dynamic y-limit based on data, but ensure it's at least 0-1
        max_val = df.max().max() if not df.empty else 0
        ax.set_ylim(0, max(1.0, max_val * 1.1) if max_val > 0 else 1.0)
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
        # Use 'cividis' colormap
        colors = plt.cm.cividis(np.linspace(0, 1, n_metrics))

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
        ax.set_ylim(0, max(1.0, max_val * 1.1) if max_val > 0 else 1.0)
        ax.grid(axis='y', linestyle='--', alpha=0.6)


    def _plot_fps(self, ax):
        """Plots processing speed (FPS)."""
        if 'Speed (FPS)' not in self.data['summary_df'].columns:
             ax.text(0.5, 0.5, 'FPS data not available.', ha='center', va='center', transform=ax.transAxes)
             ax.set_title('Processing Speed (FPS)')
             return

        fps_data = self.data['summary_df']['Speed (FPS)']
        # Get colors based on the model names (index of the series)
        colors = [get_model_color(model) for model in fps_data.index]
        rects = ax.bar(fps_data.index, fps_data, color=colors)

        ax.set_ylabel('Frames Per Second')
        ax.set_title('Processing Speed (FPS)')
        ax.set_xticklabels(fps_data.index, rotation=0, ha='center')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        # Add value labels with 1 decimal place for FPS
        self._add_value_labels(ax, rects, precision=1)
        # Dynamic Y limit based on max FPS, with padding
        max_fps = fps_data.max() if not fps_data.empty else 0
        ax.set_ylim(0, max(10, max_fps * 1.15)) # Ensure a minimum limit


    def _plot_detection_distribution(self, ax):
        """Plots the distribution of detections per image/frame using box plots."""
        # Use the pre-processed detection counts dictionary
        detection_data = self.data.get('detection_counts', {})
        if not detection_data:
            ax.text(0.5, 0.5, 'Detection count data missing.', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Distribution of Detections')
            return

        model_names = list(detection_data.keys())
        # Data needs to be a list of lists for boxplot
        data_to_plot = [detection_data[model] for model in model_names]

        # Check if there's actually any data points across all models (beyond the default [0])
        if not any(any(d > 0 for d in data_list) for data_list in data_to_plot if data_list != [0]):
             ax.text(0.5, 0.5, 'No detection count data available\n(all counts are zero or missing).',
                     horizontalalignment='center', verticalalignment='center',
                     transform=ax.transAxes, fontsize=9, color='orange')
             ax.set_title('Distribution of Detections')
             ax.set_xticks(range(1, len(model_names) + 1)) # Set ticks even if no data
             ax.set_xticklabels(model_names, rotation=0, ha='center')
             ax.set_ylabel('Detections Count')
             ax.set_ylim(0, self.data.get('max_detection_value', 10)) # Use calculated max or default
             ax.grid(axis='y', linestyle='--', alpha=0.6)
             return

        # Create the boxplot
        bp = ax.boxplot(data_to_plot, labels=model_names, patch_artist=True,
                        showfliers=True, # Show outliers as in the example image
                        medianprops={'color': 'black', 'linewidth': 1.5}, # Thicker median line
                        boxprops={'edgecolor': 'black', 'linewidth': 0.5},
                        whiskerprops={'color': 'black', 'linewidth': 0.5, 'linestyle': '--'},
                        capprops={'color': 'black', 'linewidth': 0.5})

        # Customize box colors using the defined model colors
        colors = [get_model_color(model) for model in model_names]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7) # Semi-transparent fill

        # Customize outlier markers (fliers)
        for flier in bp['fliers']:
             flier.set(marker='o', markerfacecolor='red', markersize=5,
                       markeredgecolor='none', alpha=0.4)

        ax.set_ylabel('Detections Count')
        ax.set_title('Distribution of Detections per Image')
        ax.set_xticklabels(model_names, rotation=0, ha='center')
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        # Set y-limit based on max observed value (calculated in _extract_data)
        ax.set_ylim(bottom=0, top=self.data.get('max_detection_value', 10))


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

        # Plot points with model-specific colors
        ax.scatter(fps, f1_scores, c=colors, s=120, alpha=0.9, edgecolors='k', linewidth=0.5)

        # Add model name labels near each point
        for i, model in enumerate(df.index):
            # Add a small offset to prevent labels overlapping points exactly
            ax.text(fps.iloc[i] * 1.02, f1_scores.iloc[i] * 1.02, model, fontsize=9)

        ax.set_xlabel('Speed (FPS)')
        ax.set_ylabel('F1-Score')
        ax.set_title('F1-Score vs. Speed Performance')
        ax.grid(True, linestyle='--', alpha=0.6)
        # Set limits dynamically, ensuring non-negative range
        ax.set_xlim(left=0, right=max(5, fps.max() * 1.1) if not fps.empty else 5)
        ax.set_ylim(bottom=0, top=max(0.1, f1_scores.max() * 1.1) if not f1_scores.empty else 0.1)


    def _plot_radar(self, ax):
        """Creates the multi-dimensional radar chart."""
        # Select metrics for the radar chart axes
        # Ensure these keys match the columns in summary_df after extraction
        radar_metrics_map = {
            'mAP': 'mAP (IoU=0.50:0.95)',
            'F1-Score': 'F1-Score',
            'Large\nObjects': 'Large Objects AP', # Use newlines for better label spacing
            'Medium\nObjects': 'Medium Objects AP',
            'Small\nObjects': 'Small Objects AP',
            'Speed': 'Speed (FPS)',
        }
        
        # Filter available metrics from the map based on df columns
        available_radar_metrics = {
            display: original for display, original in radar_metrics_map.items()
            if original in self.data['summary_df'].columns
        }

        if len(available_radar_metrics) < 3: # Need at least 3 axes for a radar chart
             ax.text(0.5, 0.5, 'Not enough data for Radar Chart.', ha='center', va='center', transform=ax.transAxes)
             ax.set_title('Multi-dimensional Performance Comparison', size=11, y=1.1)
             ax.set_xticks([])
             ax.set_yticks([])
             return

        radar_cols_original = list(available_radar_metrics.values())
        radar_cols_display = list(available_radar_metrics.keys())
        df_radar_raw = self.data['summary_df'][radar_cols_original].copy()


        # --- Data Normalization (Crucial for Radar Charts) ---
        df_radar_normalized = pd.DataFrame(index=df_radar_raw.index)

        for display_name, original_col in available_radar_metrics.items():
            column_data = df_radar_raw[original_col]

            if original_col == 'Speed (FPS)':
                # Normalize FPS relative to max observed FPS
                max_val = column_data.max()
                normalized = column_data / max_val if max_val > 0 else 0
            else:
                # Normalize AP/F1 scores (typically 0-1 range)
                # Using 1.0 as the theoretical max, or observed max if higher (unlikely for AP/F1)
                max_val = max(1.0, column_data.max())
                normalized = column_data / max_val if max_val > 0 else 0

            df_radar_normalized[display_name] = normalized # Use display name for normalized df

        # Handle potential NaN values resulting from division by zero
        df_radar = df_radar_normalized.fillna(0)

        # --- Plotting Setup ---
        categories = list(df_radar.columns) # Use display names now
        n_categories = len(categories)

        # Create angles for the radar chart axes (evenly spaced)
        angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
        angles += angles[:1] # Close the loop by repeating the first angle

        # Configure the radar axes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9, color='grey') # Display category names
        # Configure the radial tick marks (0.2, 0.4, ..., 1.0)
        ax.set_yticks(np.arange(0.2, 1.1, 0.2))
        ax.set_yticklabels([f"{i:.1f}" for i in np.arange(0.2, 1.1, 0.2)], fontsize=8, color='darkgrey')
        ax.set_ylim(0, 1.0) # Normalized scale
        ax.tick_params(axis='x', pad=10) # Pad labels outward
        # Set background grid color
        ax.grid(color='lightgray', linestyle='--', linewidth=0.5)
        # Change spine color
        ax.spines['polar'].set_color('lightgray')


        # --- Plot each model's data ---
        # This loop handles any number of models present in the data
        for i, model in enumerate(df_radar.index):
            data = df_radar.loc[model].tolist()
            data += data[:1] # Close the loop for plotting
            color = get_model_color(model) # Get consistent color

            # Plot the line
            ax.plot(angles, data, linewidth=2, linestyle='solid', label=model, color=color, zorder=i+2)
            # Fill the area under the line
            ax.fill(angles, data, color=color, alpha=0.25, zorder=i+1)

        ax.set_title('Multi-dimensional Performance Comparison', size=12, y=1.12) # Adjust title position
        # Place legend outside the plot area for clarity
        # Adjust bbox_to_anchor: (x, y, width, height) relative to axes
        ax.legend(loc='lower right', bbox_to_anchor=(1.15, -0.1), fontsize='small')


    def _plot_summary_table(self, ax):
        """Creates a summary table of key metrics using matplotlib.table."""
        # Define the desired columns and their display names
        table_cols_map = {
            'mAP (IoU=0.50:0.95)'   : 'mAP',
            'AP (IoU=0.50)'         : 'AP@50',
            'AP (IoU=0.75)'         : 'AP@75',
            'F1-Score'              : 'F1-Score',
            'Speed (FPS)'           : 'FPS',
            'Small Objects AP'      : 'Small AP',
            'Medium Objects AP'     : 'Med AP',
            'Large Objects AP'      : 'Large AP'
        }
        # Filter map based on columns actually available in the summary dataframe
        available_cols = {
            original: display for original, display in table_cols_map.items()
            if original in self.data['summary_df'].columns
        }
        
        if not available_cols:
            ax.text(0.5, 0.5, 'No summary data available for table.', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Summary Metrics Table', fontsize=12)
            ax.axis('off')
            return

        original_col_order = list(available_cols.keys())
        display_col_order = list(available_cols.values())

        df_table = self.data['summary_df'][original_col_order].copy()
        df_table.rename(columns=available_cols, inplace=True) # Rename columns for display

        # --- Format Data for Display ---
        for col in df_table.columns:
            if col == 'FPS':
                df_table[col] = df_table[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else 'N/A')
            elif col in ['mAP', 'AP@50', 'AP@75', 'F1-Score', 'Small AP', 'Med AP', 'Large AP']:
                 df_table[col] = df_table[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else '0.000')
            else: # Handle any other unexpected column types
                 df_table[col] = df_table[col].apply(lambda x: f"{x}" if pd.notna(x) else 'N/A')


        # Reset index to make 'Model' a column for the table content
        df_table.reset_index(inplace=True)
        # df_table.rename(columns={'index': 'Model'}, inplace=True) # Already renamed via reset_index

        # --- Create Matplotlib Table ---
        ax.axis('off') # Hide the axes background
        ax.set_title('Summary Metrics Table', fontsize=12, pad=10)

        if df_table.empty:
             ax.text(0.5, 0.5, 'No summary data available.', ha='center', va='center', transform=ax.transAxes)
             return

        # Create the table object - bbox specifies table area relative to axes [left, bottom, width, height]
        table = Table(ax, bbox=[0, 0, 1, 1])

        n_rows, n_cols = df_table.shape
        cell_height = 1.0 / (n_rows + 1) # +1 for header row
        cell_width = 1.0 / n_cols

        # Add Header Row
        for j, col_name in enumerate(df_table.columns):
            cell = table.add_cell(0, j, cell_width, cell_height, text=col_name,
                                  loc='center', facecolor='#E0E0E0') # Light grey header
            cell.set_fontsize(9)
            cell.set_text_props(weight='bold')

        # Add Data Rows
        # This loop iterates through models found in the dataframe
        for i in range(n_rows):
            model_name = df_table.iloc[i, 0] # Get model name (first column)
            row_color = get_model_color(model_name) # Get model's specific color
            # Create a light background color based on model color
            bg_color = plt.cm.colors.to_rgba(row_color, alpha=0.15)

            for j, value in enumerate(df_table.iloc[i]):
                cell = table.add_cell(i + 1, j, cell_width, cell_height, text=value,
                                      loc='center', facecolor=bg_color)
                cell.set_fontsize(9)
                # Make model name bold in the first column
                if j == 0:
                     cell.set_text_props(weight='bold')

        # Adjust column widths automatically based on content (optional but good)
        # table.auto_set_column_width(col=list(range(n_cols)))
        table.scale(1, 1.5) # Scale table height slightly if needed

        ax.add_table(table)


    # --- Placeholder/Alias Methods for Compatibility with app.py ---
    # These can be expanded later if specific plots are desired instead of the full dashboard.

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
                    # Try to extract timestamp from filename, fallback to modification time
                    parts = file.stem.split('_')
                    if len(parts) >= 3: # Expecting evaluation_results_YYYYMMDD_HHMMSS format
                         timestamp_str = parts[-2] + "_" + parts[-1]
                         timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    else:
                         timestamp = datetime.fromtimestamp(os.path.getctime(file))

                    with open(file, 'r') as f:
                         data = json.load(f)
                         all_results[timestamp] = data
                 except (ValueError, IndexError) as e:
                      print(f"Could not parse timestamp from filename {file.name}, using file time. Error: {e}")
                      timestamp = datetime.fromtimestamp(os.path.getctime(file))
                      with open(file, 'r') as f:
                         data = json.load(f)
                         all_results[timestamp] = data
                 except Exception as e:
                     print(f"Could not load or parse {file.name}: {e}")
        except Exception as e:
             print(f"Error finding or processing result files: {e}")
             return None

        return all_results if all_results else None

    def plot_performance_trends(self, aggregated_data=None, metrics=None, show_plot=False):
        """Plots performance metrics over time from multiple results files."""
        if aggregated_data is None:
            aggregated_data = self.load_multiple_results()

        if not aggregated_data or len(aggregated_data) < 2:
            print("Not enough aggregated data available for trend plotting.")
            return None

        # Default metrics to plot if none are specified
        if metrics is None:
            metrics = ['mAP (IoU=0.50:0.95)', 'F1-Score', 'Speed (FPS)'] # Default trend metrics

        # Prepare data structure for plotting trends
        trend_data = {}
        timestamps = sorted(aggregated_data.keys()) # Chronological order

        # Extract metrics for each model at each timestamp
        models_found = set()
        for ts in timestamps:
             results_at_ts = aggregated_data[ts]
             current_models = set(results_at_ts.keys()) - {"evaluation_metadata"}
             models_found.update(current_models)

             for model in current_models:
                if model not in trend_data:
                    # Initialize structure for this model
                    trend_data[model] = {m: [] for m in metrics}
                    trend_data[model]['timestamp'] = []

                # Add timestamp for this model
                trend_data[model]['timestamp'].append(ts)
                # Extract each requested metric safely
                for metric in metrics:
                    value = self._get_metric({"temp_model":results_at_ts.get(model,{})}, f"temp_model.{metric}", default=np.nan) # Use safe getter
                    trend_data[model][metric].append(value)

        # --- Plotting ---
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics) # Max 3 plots per row
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
        axes = axes.flatten() # Makes iterating through axes easier

        for i, metric in enumerate(metrics):
            ax = axes[i]
            plotted_models = 0
            for model in models_found: # Iterate through all models encountered
                if model in trend_data: # Check if model has data
                     model_ts = trend_data[model]['timestamp']
                     model_metric_data = trend_data[model][metric]

                     # Ensure lengths match before plotting
                     if len(model_ts) == len(model_metric_data):
                          # Plot only if there's non-NaN data
                          if pd.Series(model_metric_data).notna().any():
                               ax.plot(model_ts, model_metric_data, marker='o', linestyle='-',
                                       label=model, color=get_model_color(model), markersize=5)
                               plotted_models += 1
                     else:
                          print(f"Warning: Timestamp/data length mismatch for {model}, metric {metric}. Skipping.")

            if plotted_models > 0:
                ax.legend(fontsize='small')
            else:
                ax.text(0.5, 0.5, 'No data for this metric', ha='center', va='center', transform=ax.transAxes)

            ax.set_title(f'Trend: {metric}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.6)
            # Improve date formatting and rotation
            fig.autofmt_xdate(rotation=30, ha='right')

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle('Model Performance Trends Over Time', fontsize=16, y=1.02)
        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout to prevent title overlap

        # --- Save and Show ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"performance_trends_{timestamp}.png"
        output_path = VISUALIZATIONS_DIR / output_filename

        try:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Performance trends plot saved to: {output_path}")
            if show_plot: plt.show()
            else: plt.close(fig)
            return str(output_path.resolve())
        except Exception as e:
            print(f"Error saving or showing trends plot: {e}")
            plt.close(fig)
            return None


    # --- Aliases providing the dashboard for specific requests from app.py ---
    def show_precision_recall_curve(self, show_plot=True):
        """Generates the main dashboard, as PR curve is included."""
        print("Note: Generating full dashboard (Precision/Recall/F1 included).")
        return self.create_metrics_dashboard(show_plot=show_plot)

    def show_reliability_analysis(self, show_plot=True):
        """Placeholder - Generates the main dashboard."""
        print("Note: Reliability analysis not specifically implemented. Generating full dashboard.")
        return self.create_metrics_dashboard(show_plot=show_plot)

    def generate_comprehensive_report(self):
        """Generates the dashboard image file as the 'report'."""
        print("Generating dashboard image as comprehensive report.")
        report_path = self.create_metrics_dashboard(show_plot=False) # Don't show interactively
        if report_path:
             print(f"Comprehensive report image saved to: {report_path}")
        else:
             print("Failed to generate comprehensive report image.")
        return report_path


# Example usage block for testing the script directly
if __name__ == "__main__":
    print(f"Executing MetricsVisualizer Test Block")
    print(f"Project Root (derived): {PROJECT_ROOT}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Visualizations Directory: {VISUALIZATIONS_DIR}")

    # Ensure results directory exists
    RESULTS_DIR.mkdir(exist_ok=True)

    # --- Create a dummy results file mimicking the user's image data if none exist ---
    latest_results_file = None
    list_of_files = glob.glob(str(RESULTS_DIR / 'evaluation_results_*.json'))
    if list_of_files:
        latest_results_file = Path(max(list_of_files, key=os.path.getctime))
        print(f"Found existing results file: {latest_results_file.name}")
    else:
        print("No existing results file found. Creating a dummy file for testing.")
        dummy_file_path = RESULTS_DIR / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        # Data closely matching the user's image
        dummy_data = {
            "Faster R-CNN": {
                "coco_metrics": {
                    "AP_IoU=0.50:0.95": 0.174, "AP_IoU=0.50": 0.271, "AP_IoU=0.75": 0.189,
                    "AP_small": 0.085, "AP_medium": 0.189, "AP_large": 0.243,
                    "AR_IoU=0.50:0.95_maxDets=1": 0.1,"AR_IoU=0.50:0.95_maxDets=10": 0.2, "AR_IoU=0.50:0.95_maxDets=100": 0.21, # Example AR values
                    "AR_small_maxDets=100": 0.1, "AR_medium_maxDets=100": 0.22, "AR_large_maxDets=100": 0.3
                },
                "precision": 0.26, "recall": 0.21, "f1_score": 0.235, "fps": 4.8,
                "detection_counts_per_image": list(np.random.randint(300, 1500, size=50)) + list(np.random.randint(0,50, size=50)) # Example counts
            },
            "RT-DETR": { # Simulating failed/zero evaluation from image
                "coco_metrics": {
                    "AP_IoU=0.50:0.95": 0.000, "AP_IoU=0.50": 0.000, "AP_IoU=0.75": 0.000,
                    "AP_small": 0.000, "AP_medium": 0.000, "AP_large": 0.000,
                    "AR_IoU=0.50:0.95_maxDets=1": 0.0,"AR_IoU=0.50:0.95_maxDets=10": 0.0, "AR_IoU=0.50:0.95_maxDets=100": 0.0,
                    "AR_small_maxDets=100": 0.0, "AR_medium_maxDets=100": 0.0, "AR_large_maxDets=100": 0.0
                },
                 "precision": 0.0, "recall": 0.0, "f1_score": 0.000, "fps": 10.4,
                 "detection_counts_per_image": list(np.random.randint(800, 2000, size=70)) + list(np.random.randint(0, 100, size=30)) + [7500] # Add outlier
            },
            "YOLOv8-Seg": { # Simulating near-zero eval from image
                "coco_metrics": {
                     "AP_IoU=0.50:0.95": 0.000, "AP_IoU=0.50": 0.000, "AP_IoU=0.75": 0.000, # Adjusted to match image table exactly
                     "AP_small": 0.001, "AP_medium": 0.000, "AP_large": 0.000,
                     "AR_IoU=0.50:0.95_maxDets=1": 0.0,"AR_IoU=0.50:0.95_maxDets=10": 0.0, "AR_IoU=0.50:0.95_maxDets=100": 0.005,
                     "AR_small_maxDets=100": 0.001, "AR_medium_maxDets=100": 0.002, "AR_large_maxDets=100": 0.001
                },
                 "precision": 0.0, "recall": 0.0, "f1_score": 0.000, "fps": 26.4, # Adjusted to match image table exactly
                 "detection_counts_per_image": list(np.random.randint(100, 800, size=80)) + list(np.random.randint(0,50, size=20)) + [5000] # Add outlier
            },
             "evaluation_metadata": {
                 "timestamp": datetime.now().isoformat(),
                 "max_images": 100,
                 "dataset": "coco_val2017_subset_100",
                 "evaluation_tool_version": "1.0"
            }
        }
        try:
            with open(dummy_file_path, 'w') as f:
                 json.dump(dummy_data, f, indent=2)
            print(f"Created dummy results file: {dummy_file_path.name}")
            latest_results_file = dummy_file_path
        except Exception as e:
            print(f"Error creating dummy file: {e}")


    # --- Initialize visualizer (will load latest file, potentially the dummy one) ---
    visualizer = MetricsVisualizer()

    # --- Generate and Show Dashboard ---
    if visualizer.results:
        print("\nAttempting to generate dashboard...")
        dashboard_path = visualizer.create_metrics_dashboard(show_plot=True)
        if dashboard_path:
            print(f"\nDashboard generation successful: {dashboard_path}")
        else:
            print("\nDashboard generation failed.")

        # Optional: Test trend plotting if multiple files were created/exist
        # print("\nAttempting to generate trends plot...")
        # trends_path = visualizer.plot_performance_trends(show_plot=True)
        # if trends_path:
        #      print(f"\nTrends plot generation successful: {trends_path}")
        # else:
        #      print("\nFailed to generate trends plot (likely not enough data files).")

    else:
        print("\nVisualizer could not load results. Cannot run visualization tests.")