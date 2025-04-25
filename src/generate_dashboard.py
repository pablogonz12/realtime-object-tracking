#!/usr/bin/env python3
"""
Generate metrics dashboard from existing evaluation results.
This script lets you regenerate the dashboard visualization without running a new evaluation.
"""

import argparse
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
        "--show", 
        action="store_true", 
        help="Show the dashboard in a window (in addition to saving it)"
    )
    args = parser.parse_args()

    results_file = args.results if args.results else None
    
    print(f"Using results directory: {RESULTS_DIR}")
    print(f"Generating dashboard from {'specified file: ' + results_file if results_file else 'latest results file'}")
    
    visualizer = MetricsVisualizer(results_file=results_file)
    dashboard_path = visualizer.create_metrics_dashboard(show_plot=args.show)
    
    if dashboard_path:
        print(f"Dashboard generated successfully at: {dashboard_path}")
    else:
        print("Failed to generate dashboard.")

if __name__ == "__main__":
    main()