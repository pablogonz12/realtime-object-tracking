# Project Structure and Components
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
