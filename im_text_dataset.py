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
        split_path = os.path.join(root, "splits", f"{split_ids_name(split)}")
        with open(split_path, "r", encoding="utf-8") as f:
            self.ids = [line.strip() for line in f if line.strip()]

        # --- read captions ---
        cap_path = os.path.join(root, "captions.json")
        with open(cap_path, "r", encoding="utf-8") as f:
            self.captions = json.load(f)

        # --- tokenizer (not a captioning model) ---
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # GPT2 has no pad by default
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # --- image transforms ---
        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5) if split == "train" else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.img_dir = os.path.join(root, "images")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.jpg")
        image = Image.open(img_path).convert("RGB")
        image = self.tf(image)

        text = self.captions[img_id]
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
