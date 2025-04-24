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
from typing import Optional, Dict, Any

# Version identifier to verify we're running the correct file
VERSION = "1.3.0"  # Added argparse CLI

# Constants for URLs
COCO_BASE_URL = "http://images.cocodataset.org/"
COCO_VAL_IMG_URL = f"{COCO_BASE_URL}zips/val2017.zip"
COCO_ANN_URL = f"{COCO_BASE_URL}annotations/annotations_trainval2017.zip"

class DatasetManager:
    """Smart manager for COCO dataset"""

    def __init__(self):
        self.base_dir: Path = Path(__file__).parent
        self.image_dir: Path = self.base_dir / "image_data"
        self.coco_dir: Path = self.image_dir / "coco"
        self.coco_val_dir: Path = self.coco_dir / "val2017"
        self.coco_ann_dir: Path = self.coco_dir / "annotations"

    def download_file(self, url: str, destination: Path, headers: Optional[Dict[str, str]] = None) -> bool:
        """Download file with progress bar"""
        if headers is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

        print(f"Downloading {os.path.basename(destination)} from {url}...")
        try:
            response = requests.get(url, stream=True, headers=headers, timeout=60) # Added timeout
            response.raise_for_status()  # Raise an exception for 4XX/5XX responses

            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 8 # Increased block size for potentially faster downloads

            with open(destination, 'wb') as f, tqdm(
                    desc=os.path.basename(destination),
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=1, # Update progress bar more frequently
                ) as bar:
                for data in response.iter_content(block_size):
                    bar.update(len(data))
                    f.write(data)

            # Verify the file was downloaded correctly (basic size check)
            downloaded_size = destination.stat().st_size
            if total_size != 0 and downloaded_size < total_size:
                 print(f"Warning: Downloaded size ({downloaded_size} B) is less than expected size ({total_size} B) for {destination}")
            elif downloaded_size < 1000: # Less than 1KB is suspicious
                print(f"Warning: {destination} seems small ({downloaded_size} B), download might be incomplete or invalid.")

            return True
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {url}: {e}")
            # Remove the potentially corrupt file
            if destination.exists():
                destination.unlink()
            return False
        except Exception as e:
            print(f"An unexpected error occurred during download of {url}: {e}")
            if destination.exists():
                destination.unlink()
            return False

    def _extract_zip(self, zip_path: Path, extract_to: Path, description: str = "Extracting"):
        """Helper to extract zip files with progress."""
        print(f"{description} {zip_path.name} to {extract_to}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Note: tqdm doesn't easily track zip extraction progress by default
                zip_ref.extractall(extract_to)
            print(f"Successfully extracted {zip_path.name}.")
            return True
        except zipfile.BadZipFile:
            print(f"Error: {zip_path} is not a valid zip file or is corrupted.")
            return False
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")
            return False

    def _create_subset(self, subset_size: int):
        """Creates a subset of the COCO validation dataset."""
        print(f"Creating COCO subset with {subset_size} images...")
        val2017_subset_dir = self.coco_val_dir # Use the standard val2017 dir name
        val2017_subset_dir.mkdir(parents=True, exist_ok=True)

        # Define annotation paths
        full_ann_file = self.coco_ann_dir / "instances_val2017.json"
        subset_ann_file_temp = self.coco_ann_dir / "instances_val2017_subset.json"
        original_ann_backup = self.coco_ann_dir / "instances_val2017_original.json"

        if not full_ann_file.exists():
             # Check if original backup exists from a previous subset operation
            if original_ann_backup.exists():
                print("Using backed-up original annotations file.")
                shutil.copy2(original_ann_backup, full_ann_file)
            else:
                print(f"Error: Full annotation file not found at {full_ann_file} and no backup exists.")
                return False # Indicate failure

        try:
            with open(full_ann_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading annotation file {full_ann_file}: {e}")
            return False

        # Select random images
        all_image_ids = [img['id'] for img in data['images']]
        if subset_size >= len(all_image_ids):
            print("Subset size requested is >= total images. Using all validation images.")
            selected_ids = set(all_image_ids)
            subset_size = len(all_image_ids) # Adjust size for messages
        else:
            selected_ids = set(random.sample(all_image_ids, subset_size))

        # Create subset annotations data
        subset_data = {
            'info': data.get('info', {}),
            'licenses': data.get('licenses', []),
            'categories': data.get('categories', []),
            'images': [img for img in data['images'] if img['id'] in selected_ids],
            'annotations': [ann for ann in data.get('annotations', []) if ann['image_id'] in selected_ids]
        }

        # Download the full image dataset if not already extracted
        val_img_zip_path = self.coco_dir / "val2017.zip"
        if not val_img_zip_path.exists() and not any(val2017_subset_dir.iterdir()):
             print("Downloading COCO 2017 validation images for subset extraction...")
             if not self.download_file(COCO_VAL_IMG_URL, val_img_zip_path):
                 print("Failed to download validation images. Cannot create subset.")
                 return False

        # Extract needed images from the zip file directly if possible, or extract all then copy
        temp_extract_dir = None
        source_image_dir = None

        if val_img_zip_path.exists():
            print("Extracting required validation images...")
            try:
                with zipfile.ZipFile(val_img_zip_path, 'r') as zip_ref:
                    # Extract only necessary files
                    required_files = [f"val2017/{img['file_name']}" for img in subset_data['images']]
                    for member in tqdm(required_files, desc="Extracting subset images"):
                        try:
                            zip_ref.extract(member, path=self.coco_dir)
                        except KeyError:
                            print(f"Warning: Image file {member} not found in zip archive.")
                        except Exception as e:
                            print(f"Error extracting file {member}: {e}")
                source_image_dir = self.coco_val_dir # Images are extracted directly into val2017
                print("Finished extracting subset images.")
                # Optionally remove the zip after extracting needed files
                # val_img_zip_path.unlink()
                # print(f"Removed {val_img_zip_path.name}")
            except Exception as e:
                 print(f"Error during selective extraction: {e}. Falling back to full extraction.")
                 # Fallback: Extract all to temp then copy (original logic)
                 temp_extract_dir = self.coco_dir / "temp_extract"
                 temp_extract_dir.mkdir(parents=True, exist_ok=True)
                 if self._extract_zip(val_img_zip_path, temp_extract_dir, "Extracting full set for subset"):
                     source_image_dir = temp_extract_dir / "val2017"
                 else:
                     print("Failed to extract images. Cannot create subset.")
                     if temp_extract_dir.exists(): shutil.rmtree(temp_extract_dir)
                     return False
        elif self.coco_val_dir.exists() and any(self.coco_val_dir.iterdir()):
             print("Using existing extracted images in val2017 directory.")
             # Need to potentially prune existing images if they don't match the new subset
             print("Pruning existing val2017 directory for new subset...")
             required_filenames = {img['file_name'] for img in subset_data['images']}
             for item in tqdm(list(self.coco_val_dir.iterdir()), desc="Checking existing images"):
                 if item.is_file() and item.name not in required_filenames:
                     item.unlink() # Remove files not in the selected subset
             source_image_dir = self.coco_val_dir # Source is the (now pruned) directory itself
        else:
             print("Error: Cannot find validation images zip or extracted directory.")
             return False


        # If we used a temporary directory for full extraction, copy selected images
        if temp_extract_dir and source_image_dir:
            print(f"Copying {len(subset_data['images'])} selected images to {val2017_subset_dir}...")
            val2017_subset_dir.mkdir(parents=True, exist_ok=True) # Ensure target exists
            for img in tqdm(subset_data['images'], desc="Copying images"):
                src = source_image_dir / img['file_name']
                dst = val2017_subset_dir / img['file_name']
                if src.exists():
                    shutil.copy2(src, dst)
                else:
                    print(f"Warning: Source image {src} not found during copy.")
            # Clean up temporary extraction directory
            print(f"Removing temporary directory {temp_extract_dir}...")
            shutil.rmtree(temp_extract_dir)

        # Save modified annotations
        try:
            with open(subset_ann_file_temp, 'w') as f:
                json.dump(subset_data, f)
        except Exception as e:
            print(f"Error writing subset annotation file {subset_ann_file_temp}: {e}")
            return False

        # Backup original annotations if not already done
        if not original_ann_backup.exists() and full_ann_file.exists():
            print(f"Backing up original annotations to {original_ann_backup.name}")
            shutil.copy2(full_ann_file, original_ann_backup)

        # Replace original annotations with subset annotations
        print(f"Replacing {full_ann_file.name} with subset annotations.")
        shutil.move(str(subset_ann_file_temp), str(full_ann_file)) # Use move for atomicity

        print(f"Successfully created COCO subset with {subset_size} images in {self.coco_val_dir}")
        return True


    def setup_coco(self, subset_size: Optional[int] = None):
        """Download and set up COCO validation dataset (full or subset)"""
        self.coco_dir.mkdir(parents=True, exist_ok=True)
        self.coco_ann_dir.mkdir(parents=True, exist_ok=True)

        # Define paths
        ann_zip_path = self.coco_dir / "annotations_trainval2017.zip"
        val_img_zip_path = self.coco_dir / "val2017.zip"
        full_ann_file = self.coco_ann_dir / "instances_val2017.json"
        original_ann_backup = self.coco_ann_dir / "instances_val2017_original.json"

        # 1. Download and Extract Annotations (needed for both full and subset)
        # Check if annotations are already extracted or if original backup exists (from previous subset op)
        if not full_ann_file.exists() and not original_ann_backup.exists():
            if not ann_zip_path.exists():
                if not self.download_file(COCO_ANN_URL, ann_zip_path):
                    print("Failed to download annotations. Aborting setup.")
                    return
            if not self._extract_zip(ann_zip_path, self.coco_dir, "Extracting annotations"):
                 print("Failed to extract annotations. Aborting setup.")
                 # Clean up potentially corrupted zip
                 if ann_zip_path.exists(): ann_zip_path.unlink()
                 return
            # Remove annotations zip file after successful extraction
            if ann_zip_path.exists(): ann_zip_path.unlink()
            print(f"Removed {ann_zip_path.name}")
        elif original_ann_backup.exists() and not full_ann_file.exists():
             print("Restoring original annotations from backup for setup...")
             shutil.copy2(original_ann_backup, full_ann_file)
        else:
            print("Annotations already seem to be present.")


        # 2. Handle Subset Creation if requested
        if subset_size is not None and subset_size > 0:
            # If val2017 directory exists, check if it needs removal/pruning for the new subset
            if self.coco_val_dir.exists():
                print(f"Existing val2017 directory found. Will be checked/pruned for subset of {subset_size} images.")
                # The _create_subset method now handles pruning/using existing files

            if not self._create_subset(subset_size):
                 print("Failed to create subset. Aborting setup.")
                 return # Stop if subset creation failed

        # 3. Handle Full Dataset Download/Extraction if not doing subset
        elif subset_size is None:
            # Restore original annotations if we were previously in a subset state
            if original_ann_backup.exists() and full_ann_file.exists():
                 print("Restoring original annotations from backup...")
                 shutil.move(str(original_ann_backup), str(full_ann_file)) # Move backup back
                 print("Original annotations restored.")
            elif original_ann_backup.exists() and not full_ann_file.exists():
                 print("Error: Backup annotation file exists but main one is missing. Cannot proceed.")
                 return

            # Check if images need downloading/extracting
            if not self.coco_val_dir.exists() or not any(self.coco_val_dir.iterdir()):
                 if not val_img_zip_path.exists():
                     print("Downloading COCO 2017 validation images (full set)...")
                     if not self.download_file(COCO_VAL_IMG_URL, val_img_zip_path):
                         print("Failed to download full validation images. Aborting setup.")
                         return

                 if not self._extract_zip(val_img_zip_path, self.coco_dir, "Extracting full validation images"):
                     print("Failed to extract full validation images. Aborting setup.")
                     # Clean up potentially corrupted zip
                     if val_img_zip_path.exists(): val_img_zip_path.unlink()
                     return
                 # Remove image zip file after successful extraction
                 if val_img_zip_path.exists(): val_img_zip_path.unlink()
                 print(f"Removed {val_img_zip_path.name}")
            else:
                 print("Full validation images seem to be already extracted.")

        print(f"COCO dataset setup complete at {self.coco_dir}")


    def compress_datasets(self):
        """Compress datasets to save space"""
        img_compressed_file = self.coco_dir / "val2017_compressed.zip"
        ann_compressed_file = self.coco_dir / "annotations_compressed.zip"

        # Compress val2017 images
        if self.coco_val_dir.exists() and any(self.coco_val_dir.iterdir()):
            print(f"Compressing COCO validation images to {img_compressed_file.name}...")
            try:
                with zipfile.ZipFile(img_compressed_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Walk relative to coco_dir to get correct archive names like 'val2017/...'
                    for root, _, files in os.walk(self.coco_val_dir):
                        for file in tqdm(files, desc=f"Compressing {self.coco_val_dir.name}"):
                            file_path = Path(root) / file
                            # Archive name relative to coco_dir (e.g., 'val2017/000...jpg')
                            arc_name = file_path.relative_to(self.coco_dir)
                            zipf.write(file_path, arc_name)

                print(f"Successfully compressed images to {img_compressed_file}")
                print(f"Removing original directory: {self.coco_val_dir}")
                shutil.rmtree(self.coco_val_dir)
            except Exception as e:
                print(f"Error compressing images: {e}")
                # Clean up potentially incomplete zip file
                if img_compressed_file.exists(): img_compressed_file.unlink()
        elif img_compressed_file.exists():
             print("Images seem to be already compressed.")
        else:
            print("COCO validation image directory not found or empty, nothing to compress.")

        # Compress annotations
        if self.coco_ann_dir.exists() and any(self.coco_ann_dir.iterdir()):
            print(f"Compressing COCO annotations to {ann_compressed_file.name}...")
            try:
                with zipfile.ZipFile(ann_compressed_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                     # Walk relative to coco_dir to get correct archive names like 'annotations/...'
                    for root, _, files in os.walk(self.coco_ann_dir):
                        for file in tqdm(files, desc=f"Compressing {self.coco_ann_dir.name}"):
                            file_path = Path(root) / file
                            # Archive name relative to coco_dir (e.g., 'annotations/instances_val2017.json')
                            arc_name = file_path.relative_to(self.coco_dir)
                            zipf.write(file_path, arc_name)

                print(f"Successfully compressed annotations to {ann_compressed_file}")
                print(f"Removing original directory: {self.coco_ann_dir}")
                shutil.rmtree(self.coco_ann_dir)
            except Exception as e:
                print(f"Error compressing annotations: {e}")
                # Clean up potentially incomplete zip file
                if ann_compressed_file.exists(): ann_compressed_file.unlink()
        elif ann_compressed_file.exists():
            print("Annotations seem to be already compressed.")
        else:
            print("COCO annotations directory not found or empty, nothing to compress.")

        if not img_compressed_file.exists() and not ann_compressed_file.exists():
             print("No data found to compress.")
        else:
             print("Compression complete. Use decompress command to restore.")


    def decompress_datasets(self):
        """Decompress datasets for use"""
        img_compressed_file = self.coco_dir / "val2017_compressed.zip"
        ann_compressed_file = self.coco_dir / "annotations_compressed.zip"
        decompressed = False

        # Decompress images
        if img_compressed_file.exists():
            if not self.coco_val_dir.exists() or not any(self.coco_val_dir.iterdir()):
                print(f"Decompressing {img_compressed_file.name}...")
                # Extract directly into coco_dir, zip paths should be relative (e.g., 'val2017/...')
                if self._extract_zip(img_compressed_file, self.coco_dir, "Decompressing images"):
                    print("COCO images decompressed.")
                    img_compressed_file.unlink()
                    print(f"Removed {img_compressed_file.name}.")
                    decompressed = True
                else:
                    print(f"Failed to decompress {img_compressed_file.name}.")
            else:
                print("Image directory val2017 already exists. Skipping image decompression.")
                # Optionally offer to remove the compressed file anyway
                # img_compressed_file.unlink()
        else:
             if self.coco_val_dir.exists() and any(self.coco_val_dir.iterdir()):
                 print("Images seem to be already decompressed (directory exists).")
             else:
                 print("Compressed image file not found.")


        # Decompress annotations
        if ann_compressed_file.exists():
            if not self.coco_ann_dir.exists() or not any(self.coco_ann_dir.iterdir()):
                print(f"Decompressing {ann_compressed_file.name}...")
                 # Extract directly into coco_dir, zip paths should be relative (e.g., 'annotations/...')
                if self._extract_zip(ann_compressed_file, self.coco_dir, "Decompressing annotations"):
                    print("COCO annotations decompressed.")
                    ann_compressed_file.unlink()
                    print(f"Removed {ann_compressed_file.name}.")
                    decompressed = True
                else:
                    print(f"Failed to decompress {ann_compressed_file.name}.")
            else:
                print("Annotations directory already exists. Skipping annotation decompression.")
                # Optionally offer to remove the compressed file anyway
                # ann_compressed_file.unlink()
        else:
            if self.coco_ann_dir.exists() and any(self.coco_ann_dir.iterdir()):
                 print("Annotations seem to be already decompressed (directory exists).")
            else:
                 print("Compressed annotation file not found.")

        if decompressed:
            print("Decompression finished. Datasets ready for use.")
        else:
            print("No datasets were decompressed.")


    def delete_datasets(self):
        """Delete downloaded COCO dataset (images and annotations), leaving other data intact."""
        deleted = False
        # Specifically target the COCO directory for deletion
        if self.coco_dir.exists():
            print(f"Deleting COCO dataset directory: {self.coco_dir}...")
            try:
                shutil.rmtree(self.coco_dir)
                print("Successfully deleted COCO dataset directory.")
                deleted = True
            except Exception as e:
                print(f"Error deleting {self.coco_dir}: {e}")
        else:
            print("COCO dataset directory not found.")

        # Also check for stray COCO-related zip files at the base level and within coco_dir (if it still exists partially)
        stray_files = list(self.base_dir.glob("*.zip")) # Check in data_sets/
        if self.coco_dir.exists(): # Check inside coco_dir as well
             stray_files.extend(list(self.coco_dir.glob("*.zip")))
        else: # If coco_dir was deleted or never existed, check image_dir for zips
             stray_files.extend(list(self.image_dir.glob("*.zip")))


        for stray_zip in stray_files:
             # Only delete zips clearly related to COCO
             if "val2017" in stray_zip.name or "annotations" in stray_zip.name:
                 print(f"Deleting stray COCO zip file: {stray_zip}")
                 try:
                     stray_zip.unlink()
                     deleted = True
                 except Exception as e:
                     print(f"Error deleting stray file {stray_zip}: {e}")

        if deleted:
            print("COCO dataset deletion process finished.")
        else:
            print("No COCO data found to delete.")


def main():
    # Print version to confirm we're running the right script
    print(f"Dataset Manager version {VERSION}")

    parser = argparse.ArgumentParser(description=f"COCO Dataset Manager (v{VERSION})")
    parser.add_argument(
        'command',
        choices=['download', 'compress', 'decompress', 'delete', 'status'],
        help="Action to perform: 'download', 'compress', 'decompress', 'delete', 'status'."
    )
    parser.add_argument(
        '--subset',
        type=int,
        metavar='N',
        default=None,
        help="Download only a subset of N images (only valid with 'download' command)."
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help="Force action even if seemingly unnecessary (e.g., re-download, force delete confirmation)."
    )
    # Add status command later if needed

    args = parser.parse_args()
    manager = DatasetManager()

    if args.command == 'download':
        if args.subset is not None:
            if args.subset <= 0:
                print("Error: Subset size must be a positive integer.")
                sys.exit(1)
            print(f"Starting download and setup for a subset of {args.subset} images...")
            manager.setup_coco(subset_size=args.subset)
        else:
            print("Starting download and setup for the full COCO validation dataset...")
            manager.setup_coco(subset_size=None) # Explicitly None for full set
    elif args.command == 'compress':
        print("Starting dataset compression...")
        manager.compress_datasets()
    elif args.command == 'decompress':
        print("Starting dataset decompression...")
        manager.decompress_datasets()
    elif args.command == 'delete':
        confirm = False
        if args.force:
            confirm = True
        else:
            user_input = input("Are you sure you want to delete all downloaded/extracted datasets and compressed files? (yes/no): ").strip().lower()
            if user_input in ('yes', 'y'):
                confirm = True

        if confirm:
            print("Starting dataset deletion...")
            manager.delete_datasets()
        else:
            print("Deletion cancelled.")
    elif args.command == 'status':
        # Basic status check
        print("\nDataset Status:")
        print(f" Base Directory: {manager.base_dir}")
        print(f" Image Directory: {manager.image_dir} {'(Exists)' if manager.image_dir.exists() else '(Not Found)'}")
        if manager.image_dir.exists():
             print(f"  COCO Directory: {manager.coco_dir} {'(Exists)' if manager.coco_dir.exists() else '(Not Found)'}")
             if manager.coco_dir.exists():
                 # Check for extracted data
                 val_dir_exists = manager.coco_val_dir.exists() and any(manager.coco_val_dir.iterdir())
                 ann_dir_exists = manager.coco_ann_dir.exists() and any(manager.coco_ann_dir.iterdir())
                 print(f"   val2017 Images: {'Extracted' if val_dir_exists else 'Not Extracted'}")
                 print(f"   Annotations: {'Extracted' if ann_dir_exists else 'Not Extracted'}")

                 # Check for compressed files
                 img_zip = manager.coco_dir / "val2017_compressed.zip"
                 ann_zip = manager.coco_dir / "annotations_compressed.zip"
                 print(f"   Compressed Images: {'Exists' if img_zip.exists() else 'Not Found'}")
                 print(f"   Compressed Annotations: {'Exists' if ann_zip.exists() else 'Not Found'}")

                 # Check for subset indicator (backup annotation file)
                 backup_ann = manager.coco_ann_dir / "instances_val2017_original.json"
                 if backup_ann.exists():
                     print("   Mode: Likely configured as a SUBSET (original annotations backed up)")
                 elif val_dir_exists and ann_dir_exists:
                      print("   Mode: Likely configured as FULL dataset")

    print("\nDataset manager operation completed.")

if __name__ == "__main__":
    main()