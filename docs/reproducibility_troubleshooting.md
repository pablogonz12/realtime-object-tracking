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
