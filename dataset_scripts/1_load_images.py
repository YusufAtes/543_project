"""
Step 1: Load and validate images from archive/artwork/
Extracts all JPG files and creates a mapping of image IDs to file paths.
"""
import os
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image

def load_and_validate_images(archive_dir="../../archive/artwork", output_file="../../dataset/image_mapping.json"):
    """
    Load all JPG images from archive directory and validate they can be opened.
    Creates a mapping of image IDs to file paths.
    """
    archive_path = Path(archive_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(list(archive_path.glob("*.jpg")))
    print(f"Found {len(image_files)} JPG files")
    
    image_mapping = {}
    valid_images = []
    corrupted_count = 0
    
    for img_path in tqdm(image_files, desc="Validating images"):
        try:
            # Try to open and verify the image
            with Image.open(img_path) as img:
                img.verify()
            
            # Get image ID from filename (without extension)
            image_id = img_path.stem
            image_mapping[image_id] = str(img_path.resolve())
            valid_images.append(image_id)
            
        except Exception as e:
            corrupted_count += 1
            print(f"\nWarning: Could not open {img_path.name}: {e}")
            continue
    
    print(f"\nValid images: {len(valid_images)}")
    print(f"Corrupted images: {corrupted_count}")
    
    # Save mapping
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(image_mapping, f, indent=2, ensure_ascii=False)
    
    print(f"Saved image mapping to {output_path}")
    return image_mapping, valid_images

if __name__ == "__main__":
    mapping, valid = load_and_validate_images()
    print(f"\nTotal valid images: {len(valid)}")

