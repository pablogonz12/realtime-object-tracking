# Requirements and Setup

## Software Requirements
- Python 3.13 (current development version)
- PyTorch 1.10+
- OpenCV 4.5+
- CUDA-capable GPU (recommended for optimal performance)
- 20GB+ free disk space for full dataset evaluation

## System Requirements
- 16GB+ RAM recommended
- NVIDIA GPU with 4GB+ VRAM (8GB+ recommended for larger models)
- Operating Systems: Windows 10/11, Ubuntu 20.04+, or macOS 12+

## Dataset Preparation
The COCO dataset is used for model evaluation and will be automatically downloaded when running the evaluation scripts for the first time. The dataset is managed by the `dataset_manager.py` module, which handles:
- Downloading the validation set (~1GB)
- Caching images for faster subsequent use
- Organizing data for proper evaluation

No manual dataset preparation is needed, but ensure you have sufficient disk space (~20GB for full dataset).

# Installation

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

## First-time Setup Notes
- On first run, model weights will be automatically downloaded (~1-2GB depending on selected models)
- The COCO validation dataset will be downloaded automatically (~1GB)
- GPU drivers should be properly installed for CUDA support
- For macOS users: PyTorch with MPS acceleration is supported for Apple Silicon
