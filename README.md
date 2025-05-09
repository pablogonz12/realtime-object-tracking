# Computer Vision Object Recognition System

## Project Overview
This project implements a computer vision system for detecting and segmenting objects in videos and images. The system evaluates multiple state-of-the-art object detection and segmentation algorithms on the COCO dataset, analyzing their performance for validity, reliability, and objectivity.

### Dataset Justification
The COCO (Common Objects in Context) dataset was chosen as a representative dataset for this project due to its:
- **Diversity**: Contains 80 common object categories covering a wide range of everyday scenarios
- **Scale**: Includes over 200,000 labeled images with more than 1.5 million object instances
- **Complexity**: Features objects in their natural contexts with varying scales, occlusions, and viewpoints
- **Industry Standard**: Widely used benchmark in computer vision research, enabling direct comparison with state-of-the-art methods
- **Transferability**: Performance on COCO generally translates well to real-world video applications, making it ideal for our object recognition task

### Key Features
- Object detection and segmentation in videos
- Performance evaluation of multiple models
- Interactive GUI for visualization and analysis
- Comprehensive metrics dashboard
- Command-line interfaces for automation and scripting

## Requirements and Setup

### Software Requirements
- Python 3.8+ (tested with Python 3.8, 3.9, and 3.10)
- PyTorch 1.10+
- OpenCV 4.5+
- CUDA-capable GPU (recommended for optimal performance)
- 20GB+ free disk space for full dataset evaluation

### System Requirements
- 16GB+ RAM recommended
- NVIDIA GPU with 4GB+ VRAM (8GB+ recommended for larger models)
- Operating Systems: Windows 10/11, Ubuntu 20.04+, or macOS 12+

### Dataset Preparation
The COCO dataset is used for model evaluation and will be automatically downloaded when running the evaluation scripts for the first time. The dataset is managed by the `dataset_manager.py` module, which handles:
- Downloading the validation set (~1GB)
- Caching images for faster subsequent use
- Organizing data for proper evaluation

No manual dataset preparation is needed, but ensure you have sufficient disk space (~20GB for full dataset).

## Installation

Follow these steps to set up the project environment:

1. Clone the repository:
```
git clone https://github.com/yourusername/computer-vision-project.git
cd computer-vision-project
```

2. Create and activate a virtual environment (recommended):
```
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Verify installation:
```
python src/print_model_summary.py
```
This command should run without errors. If you see a message about missing results files, that's normal before running an evaluation.

### First-time Setup Notes
- On first run, model weights will be automatically downloaded (~1-2GB depending on selected models)
- The COCO validation dataset will be downloaded automatically (~1GB)
- GPU drivers should be properly installed for CUDA support
- For macOS users: PyTorch with MPS acceleration is supported for Apple Silicon

## Detailed Analysis and Report

For a comprehensive analysis of the State-of-the-Art models evaluated in this project, including:
- Detailed qualitative comparison of each model
- Analysis of performance in different conditions
- Strengths and weaknesses of each approach
- Recommendations for different use cases

Please see our [detailed analysis report](REPORT.md).

## Usage and Expected Outputs

This section outlines the main scripts, their purposes, and what outputs to expect from each.

### GUI Application (Beta)
The GUI application provides an interactive interface to run models, visualize results, and generate metrics.

**Command:**
```
python src/app.py
```

**Expected Output:**
- Interactive application window for loading videos/images
- Real-time object detection and segmentation visualization
- Option to export results

**Note:** The GUI application is currently in beta and may have some limitations.

### Command-Line Usage

#### 1. Creating Demo Videos
Process a video with object detection/segmentation:

**Command:**
```
python src/create_demo_video.py --model yolov8n-seg --video path/to/video.mp4 --output path/to/output.mp4
```

**Expected Output:**
- Processed video file with visualized detections saved to specified output path
- Performance statistics printed to console
- If no output path is specified, video will be saved to `inference/output_videos/`

Options:
- `--model` or `-m`: Model type to use (default: yolov8n-seg)
- `--video` or `-v`: Input video path (if not specified, lists available samples)
- `--output` or `-o`: Output video path (optional)
- `--conf-threshold`: Confidence threshold (default: 0.25)
- `--iou-threshold`: IoU threshold for NMS (default: 0.45)

#### 2. Evaluating Models
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

#### 3. Generating Metrics Dashboard
Generate a comprehensive performance dashboard:

**Command:**
```
python src/generate_dashboard.py --results path/to/results.json --output path/to/dashboard.png
```

**Expected Output:**
- Interactive dashboard window (if --show is used)
- Dashboard image saved to specified output path (or `inference/results/dashboard_[timestamp].png` by default)
- Performance metrics displayed in a visually informative way

The dashboard provides:
- mAP (mean Average Precision) scores across various IoU thresholds
- Per-category performance analysis
- Speed comparisons (FPS)
- Size vs. performance tradeoffs

Options:
- `--results`: Path to evaluation results JSON file (uses latest if not specified)
- `--output`: Path to save dashboard image (optional)
- `--show`: Display dashboard interactively

#### 4. Printing Summary Table
Print a simple text-based summary of model performance metrics:

**Command:**
```
python src/print_model_summary.py
```

**Expected Output:**
- Formatted text table printed to console showing key metrics for all evaluated models
- Uses the latest evaluation results file by default

#### 5. Running a Complete Pipeline
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

## Task #2 Solution and Deliverables
This project addresses Task #2 "Recognizing Objects in Video Sequences" by:

1. Implementing multiple state-of-the-art object detection and segmentation algorithms
2. Evaluating these algorithms on the COCO dataset for validity, reliability, and objectivity
3. Generating quantitative metrics (mAP, FPS, etc.) and qualitative visualizations
4. Creating demonstration videos showing object positions, shapes, and names

### Key Deliverables
1. **Source Code**: Complete implementation of all components described in this README
2. **Detailed Analysis Report** (`REPORT.md`): Comprehensive qualitative and quantitative analysis of model performance
3. **Demo Videos**: Processed videos demonstrating object recognition capabilities
4. **Evaluation Results**: JSON files containing raw metrics and comparisons

The **detailed analysis report** (`REPORT.md`) is a critical deliverable that contains:
- In-depth comparison of model architectures
- Analysis of strengths and weaknesses in different scenarios
- Recommendations for optimal model selection based on use case
- Links to demo videos and supplementary materials
- References to relevant research and methodologies

### Shape Determination and Visualization
Object "shape" is determined and visualized using the instance segmentation masks produced by the models. Unlike bounding box detection, these pixel-precise segmentation masks accurately outline the exact shape of each detected object, enabling more detailed analysis of object morphology, size, and orientation. The segmentation masks create a precise silhouette of objects that follows their actual contours rather than approximating them with rectangles.

### Implemented Models
- YOLOv8 (various sizes)
- YOLOv9
- YOLO11 (various sizes)
- Mask R-CNN

### Reproducing the Full Task #2 Pipeline
To reproduce the complete pipeline for Task #2, follow these steps in order:

```
# Step 1: Ensure environment is set up properly
python -c "import torch; print(f'PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# Step 2: Evaluate models (this will download models and dataset if not present)
python src/evaluate_models.py --images 200 --models yolov8n-seg yolov8m-seg yolov8x-seg

# Step 3: Generate and display metrics dashboard
python src/generate_dashboard.py --show

# Step 4: Create demo video with best model
python src/create_demo_video.py --best-model

# Step 5: Print summary table to console
python src/print_model_summary.py
```

**Expected Outputs from Pipeline:**
1. **Step 1**: Verification of PyTorch installation and CUDA availability
2. **Step 2**: 
   - Model weights downloaded to `models/pts/` (first run only)
   - COCO validation images downloaded to `data_sets/image_data/coco/` (first run only)
   - Evaluation results saved to `inference/results/evaluation_results_[timestamp].json`
   - Detection visualizations saved to `inference/results/[model_name]_visualizations/`
3. **Step 3**:
   - Interactive dashboard window
   - Dashboard image saved to `inference/results/dashboard_[timestamp].png`
4. **Step 4**:
   - Demo video processed with best-performing model
   - Output saved to `inference/output_videos/[video_name]_[best_model]_demo.mp4`
5. **Step 5**:
   - Text table with model metrics printed to console

The best performing model will be automatically selected based on the evaluation results.

**Execution Time Note:** The full pipeline may take 1-3 hours to run depending on hardware, GPU availability, and number of images processed.

## Project Structure and Components
The project is organized into the following main directories:

- `src/`: Source code for all components
  - `app.py`: GUI application for interactive visualization
  - `models.py`: Model loading, inference, and management utilities
  - `evaluate_models.py`: Comprehensive model evaluation on COCO dataset
  - `create_demo_video.py`: Demo video creation with object detection/segmentation
  - `generate_dashboard.py`: Performance metrics visualization dashboard
  - `metrics_visualizer.py`: Utilities for visualizing metrics and results
  - `pipeline.py`: End-to-end pipeline combining all components
  - `print_model_summary.py`: Console summary table generator
  - `compare_models.py`: Side-by-side model comparison utilities
  - `video_utils.py`: Video processing utilities

- `data_sets/`: Dataset management
  - `dataset_manager.py`: Handles dataset download, caching, and loading
  - `image_data/`: Storage for image datasets (COCO)
  - `video_data/`: Sample videos for demo and testing

- `models/`: Pre-trained model storage
  - `pts/`: Model weights in PyTorch format
  - `configs/`: Configuration files for models

- `inference/`: Output files and results
  - `results/`: Evaluation metrics and JSON result files
    - `visualizations/`: Per-model detection visualizations
  - `output_videos/`: Processed video outputs

- `docs/`: Documentation and reports
  - Contains project requirements and specifications

- `REPORT.md`: Detailed qualitative analysis report (key deliverable)
- `README.md`: This file, providing overview and instructions
- `requirements.txt`: Required Python packages for installation

## Notes on Reproducibility
To ensure full reproducibility of results:

- The first run will download necessary model weights and dataset files automatically
- All random seeds are fixed for reproducible results
- For best performance, use a CUDA-capable GPU (evaluation on CPU is possible but significantly slower)
- Full COCO dataset evaluation requires significant disk space (~20GB)
- The same hardware should be used for meaningful speed (FPS) comparisons between runs
- All evaluation parameters (confidence thresholds, IoU settings) are saved with results
- Models are versioned and pinned to specific releases to ensure consistent results
- Log files are generated for each run in the `inference/results/` directory

### Troubleshooting Common Issues

**Model download fails:**
```
# Manual download and placement in the correct directory
cd models/pts/
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt -o yolov8n-seg.pt
```

**CUDA out of memory:**
```
# Reduce batch size for evaluation
python src/evaluate_models.py --images 100 --batch-size 1
```

**Dataset download issues:**
```
# Manual COCO validation set download
cd data_sets/image_data/
mkdir -p coco/val2017
# Download from https://cocodataset.org/ and extract to coco/val2017
```

## License and Attribution
This project is licensed under the MIT License - see the LICENSE file for details.

The COCO dataset is used under the terms specified by its creators. See [cocodataset.org](https://cocodataset.org/) for more information.

Model architectures are based on the original implementations by their respective authors, with appropriate citations in the code and report.