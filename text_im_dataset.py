import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TextToImageDataset(Dataset):
    """
    Dataset for Text-to-Image generation.
    Loads pre-tokenized captions from tokenized_captions.json and corresponding images.
    Uses the same splits as the image-to-text scripts for compatibility.
    """
    
    def __init__(self, root, split="train", max_len=256, image_size=128):
        """
        Args:
            root: Path to dataset directory containing images
            split: One of "train", "val", "test"
            max_len: Maximum sequence length for captions
            image_size: Output image size (128x128 for this project)
        """
        self.root = root
        self.split = split
        self.max_len = max_len
        self.image_size = image_size

        # --- paths setup ---
        root_abs = os.path.abspath(root)
        project_root = os.path.dirname(root_abs)  # Go up one level from dataset/ to project root
        
        # --- read split ids ---
        split_path = os.path.join(project_root, "dataset_splits", f"{split_ids_name(split)}")
        with open(split_path, "r", encoding="utf-8") as f:
            all_ids = [line.strip() for line in f if line.strip()]

        # --- read tokenized captions ---
        # tokenized_captions.json uses image IDs as keys (e.g., "00001", "03048")
        tokenized_path = os.path.join(project_root, "tokenized_captions.json")
        with open(tokenized_path, "r", encoding="utf-8") as f:
            self.tokenized_captions = json.load(f)
        
        # Filter IDs to only include those with tokenized captions
        # tokenized_captions.json already uses image IDs as keys
        self.ids = []
        for img_id in all_ids:
            if img_id in self.tokenized_captions:
                self.ids.append(img_id)
        
        if len(self.ids) < len(all_ids):
            missing = len(all_ids) - len(self.ids)
            print(f"Info: {missing} image IDs in {split} split filtered (no matching entry in tokenized_captions.json). Using {len(self.ids)} samples.")
        
        if len(self.ids) == 0:
            raise ValueError(f"No valid image-caption pairs found in {split} split after filtering!")

        # --- image transforms ---
        # For text-to-image, we normalize to [-1, 1] for tanh output
        # No augmentation for text-to-image (augmentation only for image-to-text per project description)
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # [-1, 1] range
        ])

        # Images are directly in root/ (dataset/ folder)
        self.img_dir = root

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.tf(image)

        # Get tokenized caption directly using image_id as key
        tokenized = self.tokenized_captions[img_id]
        
        # Get input_ids and attention_mask from pre-tokenized data
        input_ids = torch.tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tokenized["attention_mask"], dtype=torch.long)
        
        # Truncate or pad to max_len
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            attention_mask = attention_mask[:self.max_len]
        elif len(input_ids) < self.max_len:
            # Pad with zeros (GPT-2 uses eos_token as pad)
            pad_len = self.max_len - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "image_id": img_id,
        }


def split_ids_name(split: str) -> str:
    """Return the filename for the given split."""
    if split == "train":
        return "train_ids.txt"
    if split == "val":
        return "val_ids.txt"
    if split == "test":
        return "test_ids.txt"
    raise ValueError(f"Unknown split: {split}")


def denormalize_image(tensor):
    """
    Denormalize image tensor from [-1, 1] to [0, 1] for visualization.
    """
    return (tensor + 1) / 2


if __name__ == "__main__":
    # Quick test
    ds = TextToImageDataset("dataset", split="train")
    print(f"Dataset size: {len(ds)}")
    
    sample = ds[0]
    print(f"Input IDs shape: {sample['input_ids'].shape}")
    print(f"Attention mask shape: {sample['attention_mask'].shape}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Image ID: {sample['image_id']}")
    print(f"Image min/max: {sample['image'].min():.2f} / {sample['image'].max():.2f}")
