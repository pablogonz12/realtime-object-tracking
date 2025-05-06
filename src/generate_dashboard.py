#!/usr/bin/env python3
"""
Generate metrics dashboard from existing evaluation results.
This script lets you regenerate the dashboard visualization without running a new evaluation.
"""

import argparse
import os
from pathlib import Path
# Fix the import - import directly from metrics_visualizer in the same directory
from metrics_visualizer import MetricsVisualizer, RESULTS_DIR

def main():
    """Generate a metrics dashboard from existing results"""
    parser = argparse.ArgumentParser(
        description="Generate a metrics dashboard from existing model evaluation results"
    )
    parser.add_argument(
        "--results", 
        type=str, 
        help="Path to specific results JSON file. If not specified, uses the latest file."
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save the dashboard image. If not specified, saves to the results directory."
    )
    parser.add_argument(
        "--show", 
        action="store_true", 
        help="Show the dashboard in a window (in addition to saving it)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpg", "pdf", "svg"],
        default="png",
        help="Output file format (default: png)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI (resolution) for the output image (default: 150)"
    )
    parser.add_argument(
        "--precision-recall",
        action="store_true",
        help="Also generate precision-recall curves"
    )
    args = parser.parse_args()

    results_file = args.results if args.results else None
    
    print(f"Using results directory: {RESULTS_DIR}")
    print(f"Generating dashboard from {'specified file: ' + results_file if results_file else 'latest results file'}")
    
    visualizer = MetricsVisualizer(results_file=results_file)
    
    # Set custom output path if specified
    output_path = args.output
    if output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # Generate dashboard
    dashboard_path = visualizer.create_metrics_dashboard(
        show_plot=args.show,
        output_path=output_path,
        dpi=args.dpi,
        file_format=args.format
    )
    
    if dashboard_path:
        print(f"Dashboard generated successfully at: {dashboard_path}")
    else:
        print("Failed to generate dashboard.")
    
    # Generate precision-recall curves if requested
    if args.precision_recall:
        pr_path = visualizer.plot_precision_recall_curves(show_plot=args.show)
        if pr_path:
            print(f"Precision-recall curves generated at: {pr_path}")
        else:
            print("Failed to generate precision-recall curves.")

if __name__ == "__main__":
    main()