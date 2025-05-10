# Usage and Expected Outputs

This section outlines the main scripts, their purposes, and what outputs to expect from each.

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

### 1. Creating Demo Videos
Process a video with object detection/segmentation:

**Command:**
```
python src/create_demo_video.py --model yolov8n-seg --video path/to/video.mp4 --output path/to/output.mp4
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
Evaluate models on the COCO dataset. 

**Command (default models):**
```
python src/evaluate_models.py --images 100 
```

**Command (specific models):**
```
python src/evaluate_models.py --images 100 --models yolov8s-seg yolov9c-seg
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
Generate a comprehensive performance dashboard:

**Command:**
```
python src/generate_dashboard.py --results path/to/results.json --output path/to/dashboard.png
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
Print a simple text-based summary of model performance metrics:

**Command:**
```
python src/print_model_summary.py --results path/to/results.json --format [text|markdown|csv]
```

**Expected Output:**
- Formatted text table printed to console showing key metrics for all evaluated models
  - Includes mAP, F1-Score, object size performance, and speed metrics
  - Models are sorted by overall performance by default
- Optional output file (when using `--output` flag):
  - Text format: `inference/results/model_summary_[timestamp].txt`
  - Markdown format: `inference/results/model_summary_[timestamp].md`
  - CSV format: `inference/results/model_summary_[timestamp].csv`
  
Command uses the latest evaluation results file by default if no specific file is provided.

### 5. Running a Complete Pipeline
Run a complete evaluation, visualization, and demo video generation pipeline:

**Command:**
```
python src/pipeline.py --images 100 --models yolov8n-seg yolov8s-seg --demo-video path/to/video.mp4
```

**Expected Output:**
- Model evaluation results JSON file in `inference/results/`
- Performance visualizations for each model
- Metrics dashboard saved as PNG
- Processed demo video with the best-performing model
- Summary table printed to console

Options:
- `--images`: Number of images for evaluation (default: 50)
- `--models`: Models to evaluate (default: top 3 by size)
- `--demo-video`: Video to process with best model (optional)
- `--output-dir`: Directory to save outputs (optional)
