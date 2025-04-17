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

# Version identifier to verify we're running the correct file
VERSION = "1.2.0"  # Unified download command

class DatasetManager:
    """Smart manager for COCO dataset"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.image_dir = self.base_dir / "image_data"
        self.coco_dir = self.image_dir / "coco"
        
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

    def compress_datasets(self):
        """Compress datasets to save space"""
        # Compress val2017 images
        if (self.coco_dir / "val2017").exists():
            print("Compressing COCO validation images...")
            with zipfile.ZipFile(self.coco_dir / "val2017_compressed.zip", 'w', 
                                zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(self.coco_dir / "val2017"):
                    for file in tqdm(files):
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, self.image_dir)
                        zipf.write(file_path, arc_name)
            
            shutil.rmtree(self.coco_dir / "val2017")
            print("COCO images compressed.")
        
        # Optimize annotation compression by reducing redundant operations
        annotations_dir = self.coco_dir / "annotations"
        if annotations_dir.exists():
            print("Compressing COCO annotations...")
            try:
                zip_path = self.coco_dir / "annotations_compressed.zip"
                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for dirpath, _, filenames in os.walk(annotations_dir):
                        for filename in tqdm(filenames, desc="Compressing annotation files"):
                            filepath = os.path.join(dirpath, filename)
                            arcname = os.path.relpath(filepath, annotations_dir)
                            zipf.write(filepath, arcname)
                
                if zip_path.exists() and zip_path.stat().st_size > 0:
                    print(f"Successfully compressed annotations to {zip_path}")
                    shutil.rmtree(annotations_dir)
                    print("Original annotations directory removed")
                else:
                    print("Error: Zip file was not created successfully")
            except Exception as e:
                print(f"Error compressing annotations: {e}")
                return
        print("Compression complete. Use decompress command to restore.")

    def decompress_datasets(self):
        """Decompress datasets for use"""
        # Optimize decompression by checking file existence before extraction
        img_compressed_file = self.coco_dir / "val2017_compressed.zip"
        if img_compressed_file.exists():
            print("Decompressing COCO validation images...")
            with zipfile.ZipFile(img_compressed_file, 'r') as zip_ref:
                zip_ref.extractall(self.coco_dir / "val2017")
            print("COCO images decompressed.")
            img_compressed_file.unlink()
            print("Removed compressed image file.")
        
        # Decompress annotations
        ann_compressed_file = self.coco_dir / "annotations_compressed.zip"
        if ann_compressed_file.exists():
            print("Decompressing COCO annotations...")
            with zipfile.ZipFile(ann_compressed_file, 'r') as zip_ref:
                zip_ref.extractall(self.coco_dir / "..")
            print("COCO annotations decompressed.")
            # Delete the compressed file after decompression
            os.remove(ann_compressed_file)
            print("Removed compressed annotation file.")
            
        print("Decompression complete. Datasets ready for use.")

    def delete_datasets(self):
        """Delete all downloaded datasets"""
        # Remove compressed files
        img_compressed_file = self.coco_dir / "val2017_compressed.zip"
        if img_compressed_file.exists():
            os.remove(img_compressed_file)
            
        ann_compressed_file = self.coco_dir / "annotations_compressed.zip"
        if ann_compressed_file.exists():
            os.remove(ann_compressed_file)
            
        # Remove dataset directories
        if self.image_dir.exists():
            shutil.rmtree(self.image_dir)
            print("Deleted image dataset including any compressed files.")
        else:
            print("Image dataset not found.")

def main():
    # Print version to confirm we're running the right script
    print(f"Dataset Manager version {VERSION}")
    
    manager = DatasetManager()
    
    while True:
        # Display options menu
        print("\nWhat would you like to do?")
        print("1. Download full COCO dataset (all)")
        print("2. Download subset of COCO dataset (subset)")
        print("3. Compress datasets to save space (compress)")
        print("4. Decompress datasets for use (decompress)")
        print("5. Delete all downloaded datasets (delete)")
        print("6. Exit")
        
        # Get user input
        choice = input("\nEnter command or number: ").strip().lower()
        
        # Process user choice
        if choice in ('1', 'all'):
            manager.setup_coco(None)  # Full dataset
            break
        elif choice in ('2', 'subset'):
            try:
                size = int(input("Enter number of images for subset: ").strip())
                if size <= 0:
                    print("Error: Number of images must be positive.")
                    continue
                manager.setup_coco(size)  # Subset of dataset
                break
            except ValueError:
                print("Error: Please enter a valid number.")
                continue
        elif choice in ('3', 'compress'):
            manager.compress_datasets()
            break
        elif choice in ('4', 'decompress'):
            manager.decompress_datasets()
            break
        elif choice in ('5', 'delete'):
            confirm = input("Are you sure you want to delete all datasets? (yes/no): ").strip().lower()
            if confirm in ('yes', 'y'):
                manager.delete_datasets()
                break
            else:
                print("Deletion cancelled.")
                continue
        elif choice in ('6', 'exit', 'quit'):
            print("Exiting dataset manager.")
            break
        else:
            print(f"Invalid choice: '{choice}'. Please try again.")

    print("\nDataset manager operation completed.")

if __name__ == "__main__":
    main()