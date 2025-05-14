import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import traceback

# Function to create COCO annotations from mask files
def create_coco_annotations_from_masks(masks_dir, images_dir, output_file, include_classes=None):
    """
    Create COCO format annotations from mask files, adjusting for the difference
    between mask dimensions (960x960) and original image dimensions (1280x960).
    
    Args:
        masks_dir: Directory containing the mask .npy files
        images_dir: Directory containing the original images
        output_file: Path to save the COCO JSON file
        include_classes: List of class indices to include (None means include all)
    """
    print(f"Creating COCO annotations from masks in {masks_dir}")
    print(f"Using images from {images_dir}")
    
    # If include_classes is None, include all classes
    if include_classes is None:
        include_classes = [0, 1, 2, 3]
    
    # Define class mapping - starting at 1 for CVAT compatibility
    class_mapping = {}
    for i, original_id in enumerate(sorted(include_classes), 1):
        class_mapping[original_id] = i
    
    # Define class names
    original_class_names = ["Ductile", "Brittle", "Background", "Pores"]
    
    # Initialize COCO data structure
    coco_data = {
        "licenses": [{"name": "", "id": 0, "url": ""}],
        "info": {
            "contributor": "",
            "date_created": "",
            "description": "",
            "url": "",
            "version": "",
            "year": ""
        },
        "categories": [],
        "images": [],
        "annotations": []
    }
    
    # Add categories based on included classes
    for original_id, new_id in class_mapping.items():
        coco_data["categories"].append({
            "id": new_id,
            "name": original_class_names[original_id],
            "supercategory": ""
        })
    
    # Find all mask files (.npy)
    mask_files = list(Path(masks_dir).glob('*_mask.npy'))
    
    if not mask_files:
        print(f"No mask files found in {masks_dir}")
        return False
    
    print(f"Found {len(mask_files)} mask files")
    
    # Process each mask file
    annotation_id = 1
    
    try:
        # Import needed for mask processing
        import pycocotools.mask as mask_util
    except ImportError:
        print("Error: pycocotools module not found. Please install with:")
        print("  pip install pycocotools")
        return False
    
    # Original image size and mask size
    original_width, original_height = 1280, 960
    mask_width, mask_height = 960, 960
    
    # Calculate horizontal offset (mask is centered in the original image)
    x_offset = (original_width - mask_width) // 2  # This should be 160
    
    for img_id, mask_file in enumerate(mask_files, 1):
        # Get image filename from mask filename
        img_stem = mask_file.stem.replace('_mask', '')
        
        # Try to find the original image file
        img_file = None
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            potential_file = Path(images_dir) / f"{img_stem}{ext}"
            if potential_file.exists():
                img_file = potential_file
                break
                
            # Try uppercase extension
            potential_file = Path(images_dir) / f"{img_stem}{ext.upper()}"
            if potential_file.exists():
                img_file = potential_file
                break
        
        if img_file is None:
            print(f"Warning: Original image not found for mask {mask_file.name}")
            img_filename = f"{img_stem}.tif"  # Use .tif extension to match sample
        else:
            img_filename = img_file.name
        
        # Load mask
        try:
            mask = np.load(mask_file)
            
            # Ensure mask has expected dimensions
            if mask.shape[:2] != (mask_height, mask_width):
                print(f"Warning: Mask {mask_file.name} has unexpected dimensions {mask.shape[:2]}")
                # Resize mask if needed
        except Exception as e:
            print(f"Error loading mask {mask_file.name}: {str(e)}")
            continue
        
        # Add image info
        coco_data["images"].append({
            "id": img_id,
            "width": original_width,
            "height": original_height,
            "file_name": img_filename,
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })
        
        # Process each class
        for original_class_id in include_classes:
            new_class_id = class_mapping[original_class_id]
            
            # Create binary mask for this class
            binary_mask = (mask == original_class_id).astype(np.uint8)
            
            # Skip if empty
            if not np.any(binary_mask):
                continue
            
            # Create a full-sized binary mask with the proper offset
            full_mask = np.zeros((original_height, original_width), dtype=np.uint8)
            full_mask[:mask_height, x_offset:x_offset+mask_width] = binary_mask
            
            # Convert to RLE format
            fortran_full_mask = np.asfortranarray(full_mask)
            encoded_mask = mask_util.encode(fortran_full_mask)
            
            # Calculate area
            area = float(np.sum(full_mask))
            
            # Calculate bounding box [x, y, width, height]
            y_indices, x_indices = np.where(full_mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
            bbox = [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]
            
            # Add annotation
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": img_id,
                "category_id": new_class_id,
                "segmentation": {
                    "counts": encoded_mask['counts'].decode('utf-8'),
                    "size": encoded_mask['size']
                },
                "area": area,
                "bbox": bbox,
                "iscrowd": 1,
                "attributes": {"occluded": False}
            })
            
            annotation_id += 1
        
        print(f"Processed {mask_file.name}")
    
    # Save COCO annotations
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)
        
    print(f"\nCOCO annotations created successfully!")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Included classes: {[original_class_names[c] for c in include_classes]}")
    print(f"Saved to: {output_file}")
    
    return True

if __name__ == "__main__":
    # Paths to set
    MASKS_DIR = r"D:\TEAM\Sneha\D_SEM_segmentation\test_results\masks"
    IMAGES_DIR = r"D:\TEAM\Sneha\D_SEM_segmentation\test_images"
    OUTPUT_FILE = r"D:\TEAM\Sneha\D_SEM_segmentation\test_results\coco_annotations_1.json"
    
    # Classes to include (exclude Ductile - class 0)
    # 1: Brittle, 2: Background, 3: Pores
    INCLUDE_CLASSES = [1, 2, 3]
    
    try:
        success = create_coco_annotations_from_masks(
            masks_dir=MASKS_DIR,
            images_dir=IMAGES_DIR,
            output_file=OUTPUT_FILE,
            include_classes=INCLUDE_CLASSES
        )
        
        if success:
            print("\nAnnotations created successfully!")
            print("You can now import these into CVAT.")
        else:
            print("\nFailed to create annotations.")
    except Exception as e:
        print(f"Error: {str(e)}")
        traceback.print_exc()