# Computer Vision Object Recognition System

## Project Overview
This project implements a computer vision system for detecting and segmenting objects in videos and images. The system evaluates multiple state-of-the-art object detection and segmentation algorithms on the COCO dataset, analyzing their performance for validity, reliability, and objectivity.

### Key Features
- Object detection and segmentation in videos
- Performance evaluation of multiple models
- Interactive GUI for visualization and analysis
- Comprehensive metrics dashboard
- Command-line interfaces for automation and scripting

## Requirements
- Python 3.8+ 
- PyTorch 1.10+
- OpenCV 4.5+
- CUDA-capable GPU (recommended for optimal performance)

See `requirements.txt` for a complete list of dependencies.

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/computer-vision-project.git
cd computer-vision-project
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download pre-trained models (will be downloaded automatically when needed):
```
python src/models.py --download-models
```

## Usage

### GUI Application (Beta)
The GUI application provides an interactive interface to run models, visualize results, and generate metrics.

**Note:** The GUI application is currently in beta and may have some limitations.

To launch the GUI:
```
python src/app.py
```

### Command-Line Usage

#### 1. Creating Demo Videos
Process a video with object detection/segmentation:

```
python src/create_demo_video.py --model yolov8n-seg --video path/to/video.mp4 --output path/to/output.mp4
```

Options:
- `--model` or `-m`: Model type to use (default: yolov8n-seg)
- `--video` or `-v`: Input video path (if not specified, lists available samples)
- `--output` or `-o`: Output video path (optional)
- `--conf-threshold`: Confidence threshold (default: 0.25)
- `--iou-threshold`: IoU threshold for NMS (default: 0.45)

#### 2. Evaluating Models
Evaluate models on the COCO dataset. 

If no models are specified using the `--models` argument, three default State-of-the-Art (SoA) models (`yolov8n-seg`, `yolov8x-seg`, `yolov9e-seg`) will be evaluated. You can specify any number of models from the available list to evaluate them.

**Default SoA Models Justification:**
These three models were chosen to provide a representative sample of current SoA capabilities, balancing performance, speed, and architectural variety:
-   `yolov9e-seg`: An extended version from the YOLOv9 family, typically offering the highest accuracy and representing the cutting edge in the YOLO series, suitable for scenarios where performance is the top priority.
-   `yolo11m-seg`: A medium-sized model from the YOLOv11 family, aimed at providing a good balance between accuracy and inference speed, making it a versatile choice for various applications.
-   `yolov8n-seg`: A nano version from the YOLOv8 family, optimized for very high speed and minimal resource consumption, ideal for real-time applications on edge devices or where computational power is limited.

This selection allows for a comparison across different model generations (YOLOv8, YOLOv9, YOLOv11) and sizes, highlighting trade-offs between performance, speed, and resource requirements.

```
python src/evaluate_models.py --images 100 
```
To evaluate specific models (e.g., two models):
```
python src/evaluate_models.py --images 100 --models yolov8s-seg yolov9c-seg
```
To evaluate more than three models:
```
python src/evaluate_models.py --images 100 --models yolov8n-seg yolov8s-seg yolov8m-seg yolov8l-seg
```

Options:
- `--images`: Number of COCO validation images to use (default: 50)
- `--models`: Specific models to evaluate (e.g., `yolov8n-seg yolov8s-seg`). If not provided, the three default SoA models are used. Accepts one or more model names.
- `--no-vis`: Skip saving individual detection visualizations
- `--conf-threshold`: Confidence threshold (default: 0.25)
- `--iou-threshold`: IoU threshold (default: 0.45)

#### 3. Generating Metrics Dashboard
Generate a comprehensive performance dashboard:

```
python src/generate_dashboard.py --results path/to/results.json --output path/to/dashboard.png
```

Options:
- `--results`: Path to evaluation results JSON file (uses latest if not specified)
- `--output`: Path to save dashboard image (optional)
- `--show`: Display dashboard interactively

#### 4. Running a Complete Pipeline
Run a complete evaluation, visualization, and demo video generation pipeline:

```
python src/pipeline.py --images 100 --models yolov8n-seg yolov8s-seg --demo-video path/to/video.mp4
```

Options:
- `--images`: Number of images for evaluation (default: 50)
- `--models`: Models to evaluate (default: top 3 by size)
- `--demo-video`: Video to process with best model (optional)
- `--output-dir`: Directory to save outputs (optional)

## Task #2 Solution
This project addresses Task #2 "Recognizing Objects in Video Sequences" by:

1. Implementing multiple state-of-the-art object detection and segmentation algorithms
2. Evaluating these algorithms on the COCO dataset for validity, reliability, and objectivity
3. Generating quantitative metrics (mAP, FPS, etc.) and qualitative visualizations
4. Creating demonstration videos showing object positions, shapes, and names

### Implemented Models
- YOLOv8 (various sizes)
- YOLOv9
- YOLO11 (various sizes)
- Mask R-CNN

### Running the Full Task #2 Pipeline
To reproduce the complete pipeline for Task #2:

```
# Step 1: Evaluate models
python src/evaluate_models.py --images 200 --models yolov8n-seg yolov8m-seg yolov8x-seg

# Step 2: Generate metrics dashboard
python src/generate_dashboard.py --show

# Step 3: Create demo video with best model
python src/create_demo_video.py --best-model
```

The best performing model will be automatically selected based on the evaluation results.

## Project Structure
- `src/`: Source code
  - `app.py`: GUI application
  - `models.py`: Model wrappers and utilities
  - `evaluate_models.py`: Model evaluation
  - `create_demo_video.py`: Demo video creation
  - `generate_dashboard.py`: Metrics visualization
  - `pipeline.py`: End-to-end pipeline
- `data_sets/`: Dataset management
- `models/`: Pre-trained model weights
- `inference/`: Output files and results

## Notes
- The first run will download necessary model weights and dataset files
- For best performance, use a CUDA-capable GPU
- Full COCO dataset evaluation requires significant disk space (~20GB)