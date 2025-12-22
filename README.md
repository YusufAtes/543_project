# Dataset Creation for Bidirectional Vision-Language Networks

This directory contains scripts to create a training-ready dataset from artwork images and metadata.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

For caption generation, you can optionally set a HuggingFace API token (free tier available):
```bash
# On Windows PowerShell
$env:HF_API_TOKEN = "your_token_here"

# On Linux/Mac
export HF_API_TOKEN="your_token_here"
```

Get a free token at: https://huggingface.co/settings/tokens

## Dataset Structure

The final dataset will have this structure:

```
dataset/
    images/                    # Preprocessed 128x128 images (normalized to [-1,1])
        00001.jpg
        00002.jpg
        ...
    captions.json              # Long-form captions (100-200 words per image)
    metadata.csv               # Cleaned and merged metadata
    tokenized_captions.json    # Tokenized captions with input_ids and attention_mask
    id_mapping.json            # Mapping from original image IDs to standardized IDs
    splits/
        train_ids.txt          # Training set IDs (80%)
        val_ids.txt            # Validation set IDs (10%)
        test_ids.txt           # Test set IDs (10%)
```

## Usage

### Option 1: Run All Steps at Once

```bash
cd dataset/scripts
python create_dataset.py
```

### Option 2: Run Steps Individually

1. **Load and validate images:**
   ```bash
   cd dataset/scripts
   python 1_load_images.py
   ```

2. **Process metadata:**
   ```bash
   python 2_process_metadata.py
   ```

3. **Generate captions** (this step takes the longest):
   ```bash
   python 3_generate_captions.py
   ```
   
   To test with a smaller subset first:
   ```bash
   python 3_generate_captions.py 100  # Generate captions for first 100 images
   ```

4. **Preprocess images:**
   ```bash
   python 4_preprocess_images.py
   ```

5. **Tokenize captions:**
   ```bash
   python 5_tokenize_captions.py
   ```

6. **Create splits:**
   ```bash
   python 6_create_splits.py
   ```

## Notes

- **Image preprocessing**: Images are resized to 128×128 pixels while maintaining aspect ratio (with padding). Pixel values are normalized to [-1, 1] range.

- **Captions**: Long-form captions (100-200 words) are generated using HuggingFace Inference API with GPT-2 model. Each caption focuses on composition, colors, subjects, details, textures, and symbolism.

- **Tokenizer**: GPT-2 tokenizer from HuggingFace is used with max_length=256 tokens.

- **Splits**: Dataset is split 80/10/10 (train/val/test) with stratification by artist when possible.

## Expected Output

After running all steps, you should have:
- ~45,574 preprocessed images (128×128)
- Corresponding long-form captions
- Tokenized captions ready for model training
- Train/val/test splits

## Troubleshooting

- If HuggingFace API is slow or rate-limited, the script includes delays and retries.
- If caption generation fails, fallback descriptions are generated from metadata.
- Corrupted images are automatically skipped during preprocessing.

