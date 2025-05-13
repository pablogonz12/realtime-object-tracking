# Usage and Expected Outputs

This guide provides instructions on how to use the various components of the Computer Vision Project, including setting up the necessary data and models, running command-line tools, and using the GUI application.

## Prerequisites and Setup

Before you can run the scripts or the application, ensure you have completed the following setup steps:

1.  **Python Environment and Dependencies:**
    *   Make sure you have Python installed (version 3.8 or higher is recommended).
    *   Install all required project dependencies by running:
        ```bash
        pip install -r requirements.txt
        ```
    *   For detailed installation instructions, please refer to `docs/setup_installation.md`.

2.  **Downloading Models:**
    *   The object detection models (e.g., YOLOv8, YOLOv9) are generally downloaded automatically by the underlying libraries (like Ultralytics) when a script or the GUI requests a specific model for the first time.
    *   These models are typically cached in a standard directory used by the library (e.g., `~/AppData/Local/Ultralytics` on Windows or `~/.cache/Ultralytics` on Linux/macOS).
    *   If the project uses a custom model management system, pre-trained model files (`.pt` files) might be expected in the `models/pts/` directory. If a specific model is not found, the scripts will attempt to download it.

3.  **Downloading Datasets (COCO for Evaluation):**
    *   The `evaluate_models.py` script and evaluation features in the GUI rely on the COCO (Common Objects in Context) dataset.
    *   If the COCO dataset is not found in the expected location (typically within the `data_sets/` directory, e.g., `data_sets/coco/`), the `DatasetManager` (invoked by the evaluation scripts or GUI) should automatically attempt to download and extract it. This can be a large download (several gigabytes) and may take a significant amount of time.
    *   The `DatasetManager` is responsible for organizing the dataset into the correct structure (e.g., `data_sets/coco/images/val2017/` and `data_sets/coco/annotations/instances_val2017.json`).
    *   You can also manually download the COCO 2017 validation images and annotations and place them in the appropriate subdirectories within `data_sets/coco/` if you prefer or if automatic download fails.
        *   COCO 2017 Val Images: [http://images.cocodataset.org/zips/val2017.zip](http://images.cocodataset.org/zips/val2017.zip)
        *   COCO 2017 Annotations: [http://images.cocodataset.org/annotations/annotations_trainval2017.zip](http://images.cocodataset.org/annotations/annotations_trainval2017.zip) (extract `instances_val2017.json` from this).

4.  **Sample Videos and Images:**
    *   The project may include sample videos in `data_sets/video_data/` and sample images in `data_sets/image_data/`. You can use these for testing or provide your own.

Once these prerequisites are met, you can proceed to use the command-line tools or the GUI application.

## GUI Application (Beta Version)

The GUI application provides an interactive way to:
- Run object detection models
- Visualize the detection results
- Generate performance metrics

**⚠️ IMPORTANT: USE WITH CAUTION**

The GUI application (`app.py`) is currently in **beta testing** and has several stability issues:
- May crash unexpectedly during operation
- Contains bugs that affect functionality
- Has inconsistent or inaccurate interface elements

**For reliable results:** We strongly recommend using the command-line tools described in the following sections instead of the GUI.

To launch the GUI application (despite the warnings):

```
python src/app.py
```

**Expected Output:**
- Interactive application window for loading videos/images
- Real-time object detection and segmentation visualization
- Option to export results

**⚠️ RECOMMENDED APPROACH:**
For stable evaluation and reliable results, use the command-line tools described in the following sections. They have been thoroughly tested and provide consistent, reproducible outcomes.

## Command-Line Usage

This section details each command-line script, its purpose, generic syntax, and runnable examples.

### 1. Creating Demo Videos
Process a video with object detection/segmentation to generate a demonstration video.

**Generic Command:**
```bash
python src/create_demo_video.py --model <model_name> --video <path_to_input_video> --output <path_to_output_video> [options]
```

**Runnable Example (default model, specific video):**
```bash
python src/create_demo_video.py --video data_sets/video_data/people-detection.mp4 --output inference/output_videos/people_detection_default_model_demo.mp4
```

**Runnable Example (specific model, video, and output):**
```bash
python src/create_demo_video.py --model yolov9e-seg --video data_sets/video_data/people-detection.mp4 --output inference/output_videos/sample_video_yolov9e_demo.mp4 --conf-threshold 0.3
```

**Expected Output:**
- Processed video file with visualized detections saved to specified output path
  - Default: `inference/output_videos/[input_filename]_[model_name]_demo.mp4`
  - Format: MP4 with H.264 encoding at original resolution and framerate
  - Contents: Original video with overlaid object masks, bounding boxes, class names, and confidence scores
- Performance summary saved to `inference/results/demo_performance_[timestamp].json`
  - Contains average FPS, model info, and processing parameters
- Console output showing:
  - Processing progress (frames/second)
  - Final statistics (total objects detected, processing time)

Options:
- `--model` or `-m`: Model type to use (default: yolov8n-seg)
- `--video` or `-v`: Input video path (if not specified, lists available samples)
- `--output` or `-o`: Output video path (optional)
- `--conf-threshold`: Confidence threshold (default: 0.25)
- `--iou-threshold`: IoU threshold for NMS (default: 0.45)

### 2. Evaluating Models
Evaluate object detection/segmentation models on the COCO dataset. Assumes COCO dataset is set up (see Prerequisites).

**Generic Command:**
```bash
python src/evaluate_models.py [--images <num_images>] [--models <model1_name> <model2_name>...] [options]
```

**Runnable Example (evaluate default SoA models on 50 images):**
```bash
python src/evaluate_models.py --images 50
```

**Runnable Example (evaluate specific models on 100 images):**
```bash
python src/evaluate_models.py --images 100 --models yolov8n-seg yolov8s-seg
```

**Expected Output:**
- JSON results file saved to `inference/results/evaluation_results_[timestamp].json`
- Visualization images saved to `inference/results/[model_name]_visualizations/` (unless --no-vis is used)
- Summary statistics printed to console
- Progress bar showing evaluation status

If no models are specified using the `--models` argument, three default State-of-the-Art (SoA) models (`yolov8n-seg`, `yolov8x-seg`, `yolov9e-seg`) will be evaluated. You can specify any number of models from the available list to evaluate them.

**Default SoA Models Justification:**
These three models were chosen to provide a representative sample of current SoA capabilities, balancing performance, speed, and architectural variety:
-   `yolov9e-seg`: An extended version from the YOLOv9 family, typically offering the highest accuracy and representing the cutting edge in the YOLO series, suitable for scenarios where performance is the top priority.
-   `yolo11m-seg`: A medium-sized model from the YOLOv11 family, aimed at providing a good balance between accuracy and inference speed, making it a versatile choice for various applications.
-   `yolov8n-seg`: A nano version from the YOLOv8 family, optimized for very high speed and minimal resource consumption, ideal for real-time applications on edge devices or where computational power is limited.

This selection allows for a comparison across different model generations (YOLOv8, YOLOv9, YOLOv11) and sizes, highlighting trade-offs between performance, speed, and resource requirements.

Options:
- `--images`: Number of COCO validation images to use (default: 50)
- `--models`: Specific models to evaluate (e.g., `yolov8n-seg yolov8s-seg`). If not provided, the three default SoA models are used. Accepts one or more model names.
- `--no-vis`: Skip saving individual detection visualizations
- `--conf-threshold`: Confidence threshold (default: 0.25)
- `--iou-threshold`: IoU threshold (default: 0.45)

### 3. Generating Metrics Dashboard
Generate a comprehensive performance dashboard from evaluation results.

**Generic Command:**
```bash
python src/generate_dashboard.py [--results <path_to_results_json>] [--output <path_to_dashboard_png>] [--show]
```

**Runnable Example (generate from latest results, save to default, and show interactively):**
```bash
python src/generate_dashboard.py --show
```

**Runnable Example (generate from latest results, specify output PNG file):**
```bash
python src/generate_dashboard.py --output inference/results/visualizations/custom_metrics_dashboard.png
```

**Expected Output:**
- Interactive dashboard window (if `--show` is used)
  - Contains multiple tabs for different analysis views
  - Supports zoom, pan, and export functionality
- Dashboard image saved to specified output path
  - Default: `inference/results/dashboard_[timestamp].png`
  - High-resolution (3840x2160) PNG file suitable for presentations
- Performance metrics report saved to `inference/results/metrics_report_[timestamp].pdf`
  - Detailed breakdown of all metrics with explanations
  - Model comparison tables
  - Size vs. performance analysis charts
- Raw data export in CSV format at `inference/results/metrics_export_[timestamp].csv`
  - Contains all numerical data for external analysis

The dashboard provides:
- mAP (mean Average Precision) scores across various IoU thresholds
- Per-category performance analysis
- Speed comparisons (FPS)
- Size vs. performance tradeoffs
- Confidence threshold sensitivity analysis

Options:
- `--results`: Path to evaluation results JSON file (uses latest if not specified)
- `--output`: Path to save dashboard image (optional)
- `--show`: Display dashboard interactively

### 4. Printing Summary Table
Print a simple text-based summary of model performance metrics from evaluation results.

**Generic Command:**
```bash
python src/print_model_summary.py [--results <path_to_results_json>] [--format <text|markdown|csv>] [--output <path_to_output_file>]
```

**Runnable Example (print summary of latest results to console in text format):**
```bash
python src/print_model_summary.py
```

**Runnable Example (save summary of latest results to a Markdown file):**
```bash
python src/print_model_summary.py --format markdown --output inference/results/model_summary_report.md
```
  
Command uses the latest evaluation results file by default if no specific file is provided.

### 5. Running a Complete Pipeline
Run a complete pipeline: evaluation, visualization, and demo video generation.

**Generic Command:**
```bash
python src/pipeline.py [--images <num_images>] [--models <model1_name> <model2_name>...] [--demo-video <path_to_video>] [--output-dir <path_to_output_directory>]
```

**Runnable Example (run pipeline with specified models and video):**
```bash
python src/pipeline.py --images 50 --models yolov8n-seg yolov8s-seg --demo-video data_sets/video_data/people-detection.mp4 --output-dir inference/pipeline_outputs
```

**Runnable Example (run pipeline with minimal options, using some defaults):**
```bash
python src/pipeline.py --models yolov8n-seg --demo-video data_sets/video_data/sample_video.mp4
```

**Expected Output:**
- Model evaluation results JSON file in `inference/results/`
- Performance visualizations for each model
- Metrics dashboard saved as PNG
- Processed demo video with the best-performing model
- Summary table printed to console

# Usage Guide

This section provides a collection of ready-to-use command-line examples for common tasks. Ensure you have met the prerequisites (Python environment, dependencies, and potentially datasets/models as described earlier).

## Command-Line Interface (Recommended)

This project provides several command-line tools for model evaluation, visualization, and video processing. The following examples are designed to be runnable.

### Evaluating Models

Assumes the COCO dataset is available or can be downloaded.

```bash
# Evaluate default SoA models on a small set of COCO images (e.g., 50)
python src/evaluate_models.py --images 50

# Evaluate specific models (e.g., yolov8n-seg, yolov8s-seg) on 100 COCO images
python src/evaluate_models.py --models yolov8n-seg yolov8s-seg --images 100

# Adjust batch size if you encounter memory issues (e.g., to 4)
python src/evaluate_models.py --batch-size 4 --images 50

# Example with more options specified, using CUDA if available
python src/evaluate_models.py --models yolov8n-seg --images 200 --batch-size 8 --confidence 0.25 --iou 0.45 --device cuda
```

### Creating Demo Videos

Assumes sample videos like `data_sets/video_data/people-detection.mp4` and `data_sets/video_data/sample_video.mp4` exist.

```bash
# Create a demo video using the default model and a sample video
python src/create_demo_video.py --video data_sets/video_data/people-detection.mp4 --output inference/output_videos/people_detection_default_demo.mp4

# Use the best model from prior evaluations (if available) on a sample video
python src/create_demo_video.py --best-model --video data_sets/video_data/sample_video.mp4 --output inference/output_videos/sample_video_best_model_demo.mp4

# Specify a model (e.g., yolov8s-seg), a sample video, and an output location
python src/create_demo_video.py --model yolov8s-seg --video data_sets/video_data/people-detection.mp4 --output inference/output_videos/people_detection_yolov8s_demo.mp4

# Adjust detection parameters for a specific model and video
python src/create_demo_video.py --model yolov8n-seg --video data_sets/video_data/sample_video.mp4 --conf-threshold 0.4 --iou-threshold 0.5 --output inference/output_videos/sample_video_yolov8n_custom_params_demo.mp4
```

### Generating Visualizations

Assumes model evaluations have been run and `evaluation_results_*.json` files exist in `inference/results/`.

```bash
# Generate and show the performance visualization dashboard using the latest results
python src/generate_dashboard.py --show

# Generate a dashboard from a specific (example) results file and save it
# (Replace YYYYMMDD_HHMMSS with an actual timestamp from your results file if needed)
python src/generate_dashboard.py --results-file inference/results/evaluation_results_YYYYMMDD_HHMMSS.json --output inference/results/visualizations/dashboard_from_specific_file.png
```

### Viewing Model Summary

Assumes model evaluations have been run and `evaluation_results_*.json` files exist.

```bash
# Print a summary of model stats from the latest evaluation to the console
python src/print_model_summary.py

# View summary for specific models from the latest evaluation
python src/print_model_summary.py --models yolov8n-seg yolov8s-seg

# Save the summary for all models from the latest evaluation to a Markdown file
python src/print_model_summary.py --format markdown --output inference/results/latest_model_summary.md
```

## GUI Application (Alpha)

The GUI application provides an interactive interface for exploring model performance and testing on images/videos.

```bash
# Launch the GUI application
python src/app.py
```

**Note:** The GUI is currently in alpha state and not recommended for evaluation tasks. Use the command-line tools for reliable results.

## Generating the Metrics Dashboard

To generate the model performance metrics dashboard from the command line:

1.  Ensure you have run model evaluations using the "Evaluate Models" feature in the GUI or by running `src/evaluate_models.py` script. This will generate the necessary `evaluation_results_*.json` files in the `inference/results/` directory.
2.  Navigate to the project's root directory in your terminal.
3.  Run the following command:

    ```bash
    python src/generate_dashboard.py
    ```

This script will:
-   Automatically find the latest `evaluation_results_*.json` file.
-   Process the data for all models found in that file.
-   Generate a comprehensive dashboard image (e.g., `metrics_dashboard_YYYYMMDD_HHMMSS.png`).
-   Save the dashboard to the `inference/results/visualizations/` directory.
-   By default, it will also attempt to display the generated dashboard image.
