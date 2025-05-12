# Notes on Reproducibility
To ensure full reproducibility of results:

- The first run will download necessary model weights and dataset files automatically
- All random seeds are fixed for reproducible results
- For best performance, use a CUDA-capable GPU (evaluation on CPU is possible but significantly slower)
- Full COCO dataset evaluation requires significant disk space (~20GB)
- The same hardware should be used for meaningful speed (FPS) comparisons between runs
- All evaluation parameters (confidence thresholds, IoU settings) are saved with results
- Models are versioned and pinned to specific releases to ensure consistent results
- Log files are generated for each run in the `inference/results/` directory

## Troubleshooting Common Issues

**Model download fails:**
```bash
# Manual download and placement in the correct directory
mkdir -p models/pts/
cd models/pts/
# For YOLOv8n-seg model
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt -o yolov8n-seg.pt
# For YOLOv9c-seg model
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov9c-seg.pt -o yolov9c-seg.pt
```

**CUDA out of memory:**
```bash
# Reduce batch size for evaluation
python src/evaluate_models.py --images 100 --batch-size 1

# Or use CPU if no GPU available (much slower)
python src/evaluate_models.py --device cpu
```

**Dataset download issues:**
```bash
# Manual COCO validation set download
mkdir -p data_sets/image_data/coco/val2017
mkdir -p data_sets/image_data/coco/annotations

# Download validation images
cd data_sets/image_data/coco
curl -L http://images.cocodataset.org/zips/val2017.zip -o val2017.zip
unzip val2017.zip

# Download annotations
curl -L http://images.cocodataset.org/annotations/annotations_trainval2017.zip -o annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d .
```

**Permission errors when saving results:**
```bash
# Ensure the inference directory exists with write permissions
mkdir -p inference/results
mkdir -p inference/output_videos
chmod -R 755 inference  # On Linux/macOS
```
