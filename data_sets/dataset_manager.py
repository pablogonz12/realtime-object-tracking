import os
import sys
import json
import random
import zipfile
import requests
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

class DatasetManager:
    """Smart manager for COCO dataset and video samples"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.image_dir = self.base_dir / "image_data"
        self.video_dir = self.base_dir / "video_data"
        self.coco_dir = self.image_dir / "coco"
        self.samples_dir = self.video_dir / "samples"
        
    def download_file(self, url, destination, headers=None):
        """Download file with progress bar"""
        if headers is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
        try:
            response = requests.get(url, stream=True, headers=headers)
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024
            
            with open(destination, 'wb') as f, tqdm(
                    desc=os.path.basename(destination),
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    f.write(data)
            
            # Verify the file was downloaded correctly
            if os.path.getsize(destination) < 1000:  # Less than 1KB is suspicious
                print(f"Warning: {destination} may be empty or invalid")
                
            return True
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            # Remove the potentially corrupt file
            if os.path.exists(destination):
                os.remove(destination)
            return False

    def setup_coco(self, subset_size=None):
        """Download and set up COCO validation dataset"""
        self.coco_dir.mkdir(parents=True, exist_ok=True)
        
        # Download validation images
        val_img_url = "http://images.cocodataset.org/zips/val2017.zip"
        val_img_path = self.coco_dir / "val2017.zip"
        
        # Download annotations
        ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        ann_path = self.coco_dir / "annotations_trainval2017.zip"
        
        # Download annotations first (needed for both full and subset paths)
        print("Downloading COCO 2017 annotations...")
        self.download_file(ann_url, ann_path)
        
        print("Extracting annotations...")
        with zipfile.ZipFile(ann_path, 'r') as zip_ref:
            zip_ref.extractall(self.coco_dir)
        
        # If val2017 directory already exists and subset is requested, remove it
        if subset_size and (self.coco_dir / "val2017").exists():
            print(f"Removing existing val2017 directory to create subset of {subset_size} images...")
            shutil.rmtree(self.coco_dir / "val2017")
        
        # Handle subset vs full dataset
        if subset_size:
            # Create val2017 directory for subset
            val2017_dir = self.coco_dir / "val2017"
            val2017_dir.mkdir(parents=True, exist_ok=True)
            
            # Read annotations
            with open(self.coco_dir / "annotations" / "instances_val2017.json", 'r') as f:
                data = json.load(f)
            
            # Select random images
            all_image_ids = [img['id'] for img in data['images']]
            selected_ids = set(random.sample(all_image_ids, min(subset_size, len(all_image_ids))))
            
            # Create subset annotations
            subset_data = {
                'info': data['info'],
                'licenses': data['licenses'],
                'categories': data['categories'],
                'images': [img for img in data['images'] if img['id'] in selected_ids],
                'annotations': [ann for ann in data['annotations'] if ann['image_id'] in selected_ids]
            }
            
            # We need to download the full dataset to extract only selected images
            print("Downloading COCO 2017 validation images for subset extraction...")
            self.download_file(val_img_url, val_img_path)
            
            # Extract to temporary directory
            temp_dir = self.coco_dir / "temp_extract"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            print("Extracting validation images to temporary directory...")
            with zipfile.ZipFile(val_img_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Copy only the selected images
            print(f"Creating subset with {subset_size} images...")
            for img in tqdm(subset_data['images'], desc="Copying images"):
                src = temp_dir / "val2017" / img['file_name']
                dst = val2017_dir / img['file_name']
                shutil.copy2(src, dst)
            
            # Save modified annotations
            with open(self.coco_dir / "annotations" / "instances_val2017_subset.json", 'w') as f:
                json.dump(subset_data, f)
                
            # Create symlink to make it easier to find
            src_file = self.coco_dir / "annotations" / "instances_val2017_subset.json"
            dst_file = self.coco_dir / "annotations" / "instances_val2017.json"
            
            # Make backup of original annotations
            if not (self.coco_dir / "annotations" / "instances_val2017_original.json").exists():
                shutil.copy2(dst_file, self.coco_dir / "annotations" / "instances_val2017_original.json")
            
            # Replace with subset annotations
            shutil.copy2(src_file, dst_file)
            
            # Remove temporary directory
            shutil.rmtree(temp_dir)
            
            print(f"Created COCO subset with {subset_size} images in val2017 directory")
        else:
            # Full dataset
            print("Downloading COCO 2017 validation images (full set)...")
            self.download_file(val_img_url, val_img_path)
            
            print("Extracting validation images...")
            with zipfile.ZipFile(val_img_path, 'r') as zip_ref:
                zip_ref.extractall(self.coco_dir)
        
        # Remove zip files to save space
        os.remove(val_img_path)
        os.remove(ann_path)
        
        print(f"COCO dataset ready at {self.coco_dir}")
        
    def setup_video_samples(self):
        """Download sample videos for testing"""
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        
def setup_video_samples(self):
    """Download sample videos for testing"""
    self.samples_dir.mkdir(parents=True, exist_ok=True)
    
    # List of videos with reliable direct download links
    videos = [
        # Video 1: Sample MP4 from GitHub (very reliable)
        {
            "url": "https://github.com/bower-media-samples/big-buck-bunny-1080p-60fps-30s/raw/master/video.mp4",
            "name": "big_buck_bunny.mp4"
        },
        # Video 2: Sample video of traffic from GitHub (reliable)
        {
            "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
            "name": "car_detection.mp4"
        },
        # Video 3: Sample video of people walking from GitHub (reliable)
        {
            "url": "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4",
            "name": "person_bicycle_car.mp4"
        },
        # Video 4: Wildlife video sample (medium-sized file)
        {
            "url": "https://vjs.zencdn.net/v/oceans.mp4",
            "name": "oceans.mp4" 
        }
    ]
    
    # Download each video
    for video in videos:
        video_path = self.samples_dir / video["name"]
        if not video_path.exists():
            print(f"Downloading {video['name']}...")
            success = self.download_file(video["url"], video_path)
            if not success:
                print(f"Failed to download {video['name']}, please download manually.")
                # Create a text file with instructions
                with open(self.samples_dir / f"{video['name']}.txt", 'w') as f:
                    f.write(f"Please download a sample video and rename it to {video['name']}\n")
                    f.write("You can find free videos at:\n")
                    f.write("1. https://www.pexels.com/videos/\n")
                    f.write("2. https://pixabay.com/videos/\n")
                    f.write("3. https://www.videvo.net/free-stock-videos/")
            else:
                print(f"Successfully downloaded {video['name']}")
        else:
            print(f"{video['name']} already exists.")
            
    print(f"Sample videos ready at {self.samples_dir}")

    def compress_datasets(self):
        """Compress datasets to save space"""
        if (self.coco_dir / "val2017").exists():
            print("Compressing COCO validation images...")
            with zipfile.ZipFile(self.coco_dir / "val2017_compressed.zip", 'w', 
                                zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(self.coco_dir / "val2017"):
                    for file in tqdm(files):
                        zipf.write(
                            os.path.join(root, file),
                            os.path.relpath(os.path.join(root, file), 
                                           os.path.join(self.coco_dir, '..'))
                        )
            
            shutil.rmtree(self.coco_dir / "val2017")
            print("COCO dataset compressed. Use decompress command to restore.")
    
    def decompress_datasets(self):
        """Decompress datasets for use"""
        compressed_file = self.coco_dir / "val2017_compressed.zip"
        if compressed_file.exists():
            print("Decompressing COCO validation images...")
            with zipfile.ZipFile(compressed_file, 'r') as zip_ref:
                zip_ref.extractall(self.coco_dir / "..")
            print("COCO dataset decompressed and ready for use.")
    
    def delete_datasets(self):
        """Delete all downloaded datasets"""
        compressed_file = self.coco_dir / "val2017_compressed.zip"
        if compressed_file.exists():
            os.remove(compressed_file)
            
        if self.image_dir.exists():
            shutil.rmtree(self.image_dir)
            print("Deleted image dataset.")
        else:
            print("Image dataset not found.")
            
        if self.video_dir.exists():
            shutil.rmtree(self.video_dir)
            print("Deleted video dataset.")
        else:
            print("Video dataset not found.")

def main():
    parser = argparse.ArgumentParser(description='Dataset Manager for Computer Vision Project')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Download COCO dataset
    coco_parser = subparsers.add_parser('coco', help='Download COCO dataset')
    coco_parser.add_argument('--subset', type=int, help='Create a subset with specified number of images')
    
    # Download video samples
    subparsers.add_parser('videos', help='Download sample videos')
    
    # Download all datasets
    all_parser = subparsers.add_parser('all', help='Download all datasets')
    all_parser.add_argument('--subset', type=int, help='Create a subset with specified number of images')
    
    # Compress datasets
    subparsers.add_parser('compress', help='Compress datasets to save space')
    
    # Decompress datasets
    subparsers.add_parser('decompress', help='Decompress datasets for use')
    
    # Delete all datasets
    subparsers.add_parser('delete', help='Delete all downloaded datasets')
    
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    manager = DatasetManager()
    
    if args.command == 'coco':
        manager.setup_coco(args.subset)
    elif args.command == 'videos':
        manager.setup_video_samples()
    elif args.command == 'all':
        subset = getattr(args, 'subset', None)
        manager.setup_coco(subset)
        manager.setup_video_samples()
    elif args.command == 'compress':
        manager.compress_datasets()
    elif args.command == 'decompress':
        manager.decompress_datasets()
    elif args.command == 'delete':
        manager.delete_datasets()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()