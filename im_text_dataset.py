import json, os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ArtCaptionDataset(Dataset):
    def __init__(self, root, split="train", tokenizer_name="gpt2", max_len=256, image_size=224):
        self.root = root
        self.split = split
        self.max_len = max_len

        # --- read ids ---
        # Splits, captions, and id_mapping are in project root (one level up from root="dataset")
        root_abs = os.path.abspath(root)
        project_root = os.path.dirname(root_abs)  # Go up one level from dataset/ to project root
        split_path = os.path.join(project_root, "dataset_splits", f"{split_ids_name(split)}")
        with open(split_path, "r", encoding="utf-8") as f:
            all_ids = [line.strip() for line in f if line.strip()]

        # --- read captions ---
        cap_path = os.path.join(project_root, "captions.json")
        with open(cap_path, "r", encoding="utf-8") as f:
            self.captions = json.load(f)
        
        # --- read id_mapping (caption_key -> image_filename) and reverse it ---
        id_mapping_path = os.path.join(project_root, "id_mapping.json")
        with open(id_mapping_path, "r", encoding="utf-8") as f:
            caption_to_image = json.load(f)  # caption_key -> image_id
        
        # Reverse mapping: image_id -> caption_key
        image_to_caption = {v: k for k, v in caption_to_image.items()}
        
        # Create mapping from image_id to caption_key for this dataset
        self.id_to_caption_key = {}
        
        # Filter IDs to only include those with captions
        self.ids = []
        for img_id in all_ids:
            if img_id in image_to_caption:
                caption_key = image_to_caption[img_id]
                if caption_key in self.captions:
                    self.ids.append(img_id)
                    self.id_to_caption_key[img_id] = caption_key
        
        if len(self.ids) < len(all_ids):
            missing = len(all_ids) - len(self.ids)
            print(f"Info: {missing} image IDs in {split} split filtered (no matching caption). Using {len(self.ids)} samples.")
        
        if len(self.ids) == 0:
            raise ValueError(f"No valid image-caption pairs found in {split} split after filtering!")

        # --- tokenizer (not a captioning model) ---
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # GPT2 has no pad by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- image transforms ---
        # Build transform list conditionally (avoid lambda for Windows multiprocessing compatibility)
        transform_list = [
            transforms.Resize((image_size, image_size)),
        ]
        if split == "train":
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.tf = transforms.Compose(transform_list)

        # Images are directly in root/ (not root/dataset/)
        self.img_dir = root

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.tf(image)

        # Use the mapped caption key from id_mapping.json
        caption_key = self.id_to_caption_key[img_id]
        text = self.captions[caption_key]
        enc = self.tokenizer(
            text,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)          # (L,)
        attention_mask = enc["attention_mask"].squeeze(0) # (L,)

        # We train next-token prediction:
        # decoder_in: tokens[:-1], targets: tokens[1:]
        decoder_in = input_ids[:-1].clone()
        targets   = input_ids[1:].clone()
        dec_mask  = attention_mask[:-1].clone()

        return {
            "image": image,
            "decoder_in": decoder_in,
            "targets": targets,
            "dec_pad_mask": (dec_mask == 0)  # True where padding
        }

def split_ids_name(split: str) -> str:
    if split == "train":
        return "train_ids.txt"
    if split == "val":
        return "val_ids.txt"
    if split == "test":
        return "test_ids.txt"
    raise ValueError(f"Unknown split: {split}")
