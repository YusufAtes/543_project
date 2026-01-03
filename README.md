### Project Title
**Bidirectional Vision–Language Networks for Cross-Modal Generation and Reconstruction**

### Project Description
This project trains vision and language models to communicate through long-form natural language captions. Using an artwork dataset with LLM-generated captions, we implement:

- **Image → Text**: Generate detailed captions from images
- **Text → Image**: Generate images from captions (using pretrained Stable Diffusion for final pipeline)
- **Complete Pipeline**: Image → Caption → Reconstructed Image

---

### Project Status
- **Dataset creation**: ✅ Achieved (4,353 image-caption pairs)
- **Image → Text**: ✅ Achieved (ResNet-18/34/50 backbones)
- **Text → Image (trained)**: ⚠️ Attempted with conditional VAE; quality limited
- **Complete Pipeline**: ✅ Achieved (uses pretrained Stable Diffusion v1.5)

---

### Part 1 — Dataset Creation
**Pipeline**: Load images → Process metadata → Generate captions (GPT-4o-mini) → Preprocess images (128×128) → Tokenize captions (GPT-2) → Create splits (80/10/10)

**Dataset**: 4,353 image-caption pairs (Train: 3,760, Val: 493, Test: 100)

---

### Part 2 — Image → Text
**Architecture**: ResNet encoder (18/34/50) → Transformer decoder → GPT-2 vocabulary

**Training**: Cross-entropy loss, label smoothing 0.1, AdamW optimizer, early stopping

**Results** (test set):
- ResNet-18: PPL 6.09, BLEU-1 0.34, BLEU-4 0.047
- ResNet-50: PPL 6.22, BLEU-1 0.34, BLEU-4 0.046

---

### Part 3 — Text → Image
**Trained Model (Conditional VAE)**: Transformer text encoder → VAE decoder with FiLM conditioning → 128×128 images

**Loss**: L1 + ResNet perceptual + style + Sobel edge + KL divergence + GAN (ResNet discriminator)

**Results** (test set): MSE ~0.06, PSNR ~12.5 dB, SSIM ~0.14 (quality limited; outputs tend to be blurry/average)

**Note**: For the complete pipeline, we use **pretrained Stable Diffusion v1.5** instead of the trained VAE for better generation quality.

---

### Complete Pipeline (Image → Text → Image)
**Script**: `complete_test_pipeline.py`

**Process**: Test image → Image→Text model (generates caption) → Stable Diffusion v1.5 (generates image from caption)

**Output**: 3-panel figures saved to `results/complete_pipeline/<backbone>/<image_id>_pipeline.png`
- Left: Original image
- Middle: Generated caption (first 85 words)
- Right: Generated image from Stable Diffusion

---

### How to Run

**Environment**: Install dependencies with `pip install -r requirements.txt`

**Dataset Creation**:
```bash
python dataset_scripts/1_load_images.py
python dataset_scripts/2_process_metadata.py
python dataset_scripts/3_generate_captions.py  # Requires OPENAI_API_KEY
python dataset_scripts/4_preprocess_images.py
python dataset_scripts/5_tokenize_captions.py
python dataset_scripts/6_create_splits.py
```

**Train Models**:
- Image→Text: `python im_text_train.py`
- Text→Image: `python text_im_train.py`

**Run Complete Pipeline**:
```bash
python complete_test_pipeline.py  # Uses all backbones, full test set
# Or with options:
python complete_test_pipeline.py --max_samples 10 --backbones resnet18 --num_inference_steps 50
```

