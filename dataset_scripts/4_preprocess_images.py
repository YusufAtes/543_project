"""
Step 4: Preprocess images - resize to 128x128, normalize to [-1,1], save to dataset/images/
"""
import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


# Resolve paths relative to the project root (two levels above this script)
BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_MAPPING = BASE_DIR / "dataset" / "image_mapping.json"
DEFAULT_CAPTIONS = BASE_DIR / "dataset" / "captions.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "dataset" / "images"


def preprocess_image(input_path, output_path, target_size=(128, 128)):
    """
    Resize image to target_size maintaining aspect ratio with padding,
    then normalize to [-1, 1] range.
    """
    # Open and convert to RGB if needed
    img = Image.open(input_path).convert('RGB')
    
    # Calculate scaling to fit image in target size while maintaining aspect ratio
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]
    
    if img_ratio > target_ratio:
        # Image is wider - fit to width
        new_width = target_size[0]
        new_height = int(target_size[0] / img_ratio)
    else:
        # Image is taller - fit to height
        new_height = target_size[1]
        new_width = int(target_size[1] * img_ratio)
    
    # Resize image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size and paste resized image centered
    img_padded = Image.new('RGB', target_size, (0, 0, 0))  # Black background
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    img_padded.paste(img_resized, (paste_x, paste_y))
    
    # Convert to numpy array and normalize to [-1, 1]
    img_array = np.array(img_padded, dtype=np.float32)
    img_normalized = (img_array / 127.5) - 1.0  # Scale from [0, 255] to [-1, 1]
    
    # Convert back to [0, 255] for saving as JPG (we'll denormalize on load during training)
    img_for_save = ((img_normalized + 1.0) * 127.5).astype(np.uint8)
    img_final = Image.fromarray(img_for_save)
    
    # Save preprocessed image
    img_final.save(output_path, 'JPEG', quality=95)
    
    return img_normalized


def preprocess_all_images(
    image_mapping_file=DEFAULT_IMAGE_MAPPING,
    captions_file=DEFAULT_CAPTIONS,
    output_dir=DEFAULT_OUTPUT_DIR,
    target_size=(128, 128),
):
    """
    Preprocess images that have captions: resize, normalize, and save to output directory.
    Creates standardized naming (00001.jpg, 00002.jpg, etc.).
    """
    # Ensure all paths are Path objects
    image_mapping_file = Path(image_mapping_file)
    captions_path = Path(captions_file)
    output_path = Path(output_dir)

    print("Loading image mapping...")
    with open(image_mapping_file, "r", encoding="utf-8") as f:
        image_mapping = json.load(f)

    # Load captions and keep only image IDs that have captions
    if captions_path.exists():
        print("Loading captions...")
        with open(captions_path, "r", encoding="utf-8") as f:
            captions = json.load(f)

        caption_ids = set(captions.keys())
        # Only keep image IDs that appear in captions
        image_ids = [img_id for img_id in image_mapping.keys() if img_id in caption_ids]
        image_ids = sorted(image_ids)
        print(
            f"Found {len(image_ids)} images that have captions "
            f"(out of {len(image_mapping)} total images)."
        )
    else:
        # Fallback: process all images if captions file is missing
        image_ids = sorted(image_mapping.keys())
        print(
            f"Captions file not found at {captions_path}. "
            f"Preprocessing all {len(image_ids)} images."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create mapping from old IDs to new standardized IDs
    id_mapping = {}

    print(
        f"Preprocessing {len(image_ids)} images to "
        f"{target_size[0]}x{target_size[1]}..."
    )

    for idx, old_id in enumerate(tqdm(image_ids, desc="Preprocessing images")):
        # Create new standardized ID (zero-padded 5 digits)
        new_id = f"{idx + 1:05d}"
        old_path = Path(image_mapping[old_id])
        new_path = output_path / f"{new_id}.jpg"

        try:
            preprocess_image(old_path, new_path, target_size)
            id_mapping[old_id] = new_id
        except Exception as e:
            print(f"\nError processing {old_id}: {e}")
            continue

    # Save ID mapping for later use
    mapping_file = Path(output_dir).parent / "id_mapping.json"
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(id_mapping, f, indent=2, ensure_ascii=False)

    print(f"\nPreprocessed {len(id_mapping)} images")
    print(f"Saved to {output_path}")
    print(f"ID mapping saved to {mapping_file}")

    return id_mapping


if __name__ == "__main__":
    id_mapping = preprocess_all_images()

