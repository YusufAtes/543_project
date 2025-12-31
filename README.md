### Project Title
**Bidirectional Vision–Language Networks for Cross-Modal Generation and Reconstruction**

### Project Description (for report)
This project investigates whether **independently trained** vision and language models can form a usable “communication channel” through **long-form natural language**. Using an artwork dataset paired with LLM-generated paragraph captions, we train two complementary models:

- **Image → Text (captioning)**: generate a detailed description given an image.
- **Text → Image (generation / reconstruction)**: generate an image conditioned on a caption (attempted with multiple approaches).

Finally, we aim to connect the two into a **cycle**:

- **Image → Caption → Reconstructed Image**

This README is written as a **research-report template**: it includes technical details (architectures, losses, training/evaluation) and also explicitly documents **what was achieved vs. what failed** in the current project state.

---

### Repository Structure (what to cite in the report)
- **Dataset creation**: `dataset_scripts/` (Steps 1–6)
- **Image → Text**:
  - `im_text_dataset.py`, `im_text_model.py`, `im_text_train.py`, `im_text_test.py`
  - Outputs: `checkpoints/im2text_*`, `runs/im2text_*`, `results/im2text_*_best_test.jsonl`
- **Text → Image (3 attempted methods)**:
  - **Method A (conditional VAE)**: `text_im_dataset.py`, `text_im_model.py`, `text_im_train.py`, `text_im_test.py`
  - **Method B (small diffusion from scratch)**: `text_im_small_diff_dataset.py`, `text_im_small_diff_train.py`, `text_im_small_diff_test.py`
  - **Method C (Stable Diffusion v1.5 LoRA)**: `text_im_sd_dataset.py`, `text_im_sd_train.py`, `text_im_sd_test.py`
  - Outputs: `checkpoints/text2im_*`, `runs/text2im_*`, `results/text2im_*`

---

### Current Project Status (high-level)
- **Dataset creation**: **Achieved**
- **Image → Text**: **Achieved** (3 backbones trained + test results saved)
- **Text → Image**: **Not achieved** (3 methods attempted; generation quality collapsed toward “average-looking” / overly smooth outputs, highlighting the difficulty of high-quality image generation under this setup)
- **Full cycle (Image → Text → Image)**: **Planned next step** (see “Next Steps”)

---

### Part 1 — Dataset Creation (Achieved)
This stage produces a training-ready dataset of aligned (image, caption) pairs and splits.

- **Goals**:
  - Standardize images for training (**resize + pad to 128×128**)
  - Create consistent paragraph captions for each artwork using an LLM
  - Tokenize captions for model training
  - Create train/val/test splits

- **Data pipeline (implemented)**:
- **Step 1 — Load & validate images** (`dataset_scripts/1_load_images.py`)
  - Scans `archive/artwork/*.jpg`, verifies images can be opened, writes a mapping (`dataset/image_mapping.json`).
- **Step 2 — Process metadata** (`dataset_scripts/2_process_metadata.py`)
  - Merges `archive/artwork_dataset.csv` and `archive/info_dataset.csv` (join via cleaned artist field), extracts year/medium where possible, writes `dataset/metadata.csv`.
- **Step 3 — Caption generation (LLM)** (`dataset_scripts/3_generate_captions.py`)
  - Uses **OpenAI `gpt-4o-mini` (vision)** via `https://api.openai.com/v1/chat/completions`
  - Fixed instruction prompt: generate **100–120 words**, end with explicit “type + style” sentence.
  - Outputs a captions JSON (in this repo, training consumes `captions.json` at the project root).
- **Step 4 — Image preprocessing** (`dataset_scripts/4_preprocess_images.py`)
  - Aspect-ratio preserving resize + padding to **128×128**
  - Pixel normalization convention: images are prepared for a **[-1, 1]** training range (the script saves JPGs and training code applies normalization on load).
  - Writes standardized IDs (`00001.jpg`, …) and an ID mapping (in this repo, training consumes `id_mapping.json` at the project root).
- **Step 5 — Caption tokenization** (`dataset_scripts/5_tokenize_captions.py`)
  - Tokenizer: **HuggingFace GPT‑2 (`gpt2`)**, `max_length=256`, truncation + padding
  - Output: tokenized captions JSON (in this repo, training consumes `tokenized_captions.json` at the project root).
- **Step 6 — Split creation** (`dataset_scripts/6_create_splits.py`)
  - Target split: **80/10/10**
  - Attempts **artist-stratified split** using `metadata.csv`; falls back to random if stratification fails.
  - Output: split ID files (in this repo, training consumes `dataset_splits/{train,val,test}_ids.txt`).

- **Dataset size (current run outputs)**:
  - These numbers are measured from the files in this repository (≈ **4.3k**, consistent with the “~4.2k processed” project target).
  - **Total paired samples**: **4,353**
  - **Train / Val / Test**: **3,760 / 493 / 100**

- **Where the “final” files live in this repo**:
  - This project’s training scripts reference these files at the project root.

```
dataset/                   # preprocessed images used by models (e.g., 00001.jpg ...)
captions.json              # raw captions keyed by caption-IDs
tokenized_captions.json    # GPT-2 tokenized captions keyed by standardized image IDs
id_mapping.json            # caption-key -> standardized image ID mapping
dataset_splits/            # {train,val,test}_ids.txt
metadata.csv               # merged metadata
```

---

### Part 2 — Image → Text (Achieved)
This stage trains an image captioning model that reconstructs the long-form caption given an input artwork image.

- **Model family (3 backbones)**:
  - Backbones trained and evaluated:
- **ResNet‑18**
- **ResNet‑34**
- **ResNet‑50**

- **Architecture (implemented in `im_text_model.py`)**:
- **Image encoder**: torchvision ResNet (ImageNet-pretrained weights allowed), truncated to the final conv feature map.
- **Image → token memory**: feature map flattened into \(S = H \times W\) tokens, projected with a linear layer to \(d_{model}\).
- **Text decoder**: Transformer Decoder (causal self-attention + cross-attention to image memory)
- **Output head**: linear projection to GPT‑2 vocabulary size.

- **Training objective and optimization (implemented in `im_text_train.py`)**:
- **Loss**: next-token **Cross Entropy** with:
  - `ignore_index = pad_token_id`
  - **label smoothing = 0.1**
- **Teacher forcing**: implemented via shifted tokens (`decoder_in = input_ids[:-1]`, `targets = input_ids[1:]`)
- **Optimizer**: AdamW (`weight_decay=0.01`)
- **Gradient clipping**: global norm `1.0`
- **Early stopping**: patience `5`
- **LR scheduling**: default `ReduceLROnPlateau` (optionally cosine restarts)

- **Evaluation (implemented in `im_text_test.py`)**:
- **Token-level metrics**: test loss, perplexity
- **Text similarity metric**: BLEU‑1/2/3/4 (simple smoothing)
- **Outputs saved**:
  - `results/im2text_<backbone>_best_test.jsonl` (first line is a summary JSON object; following lines are per-sample generations)
  - Validation curves and logs: `runs/im2text_<backbone>/history.csv` and plots (loss / perplexity / LR).

- **Current reported test summaries (from `results/*.jsonl`)**:
- **ResNet‑18**: test ppl **6.09**, avg BLEU‑1 **0.3396**, BLEU‑4 **0.0465**
- **ResNet‑50**: test ppl **6.22**, avg BLEU‑1 **0.3411**, BLEU‑4 **0.0458**

---

### Part 3 — Text → Image (Not Achieved; 3 methods attempted)
This stage attempts to generate an artwork image conditioned on a caption. Despite multiple approaches, **image generation was not achieved successfully** in this project phase: outputs tended to be overly smooth and visually similar, consistent with a collapse toward an “average” image solution. This section documents the attempted methods and training details for the report.

- **Method A — Conditional VAE (implemented; did not reach satisfactory generation quality)**:
Files: `text_im_dataset.py`, `text_im_model.py`, `text_im_train.py`, `text_im_test.py`

- **Text encoder**: Transformer Encoder (GPT‑2 vocab), sinusoidal positional encoding, mean pooling over non-padding tokens.
- **Image decoder**: conditional decoder with:
  - latent sampling \(z\) from \( \mu(text), \log\sigma^2(text) \)
  - FiLM conditioning injected into residual blocks
  - progressive upsampling \(4\rightarrow8\rightarrow16\rightarrow32\rightarrow64\rightarrow128\) (updated to **nearest-neighbor upsampling** to reduce blur)
  - output `tanh` producing pixels in **[-1, 1]**

**Loss function (implemented in `text_im_model.py`)**
- **Base reconstruction**: **L1 loss**
- **Perceptual term (two options)**:
  - **ResNet perceptual + style** (new default): feature-space L1 using ImageNet-pretrained ResNet blocks + optional Gram/style loss (`ResNetPerceptualLoss`)
  - **VGG‑19 perceptual** (older option): feature-space L1 using ImageNet-pretrained VGG19 blocks (`PerceptualLoss`)
- **Edge sharpening term** (new): **Sobel edge loss** (L1 distance between edge maps), controlled by `edge_weight`
- **Regularization**: **KL divergence**, weighted by \(\beta\)
- Total (generator): \(L_G = L_{L1} + \lambda_{perc} L_{perc} + \lambda_{style} L_{style} + \lambda_{edge} L_{edge} + \beta L_{KL}\)
- **Adversarial component (new)**: optional conditional GAN hinge loss using a **ResNet projection discriminator** (`TextImageDiscriminator`):
  - Discriminator: \(L_D = \mathbb{E}[\max(0, 1 - D(x, t))] + \mathbb{E}[\max(0, 1 + D(\hat{x}, t))]\)
  - Generator: adds \( \lambda_{adv}\, \mathbb{E}[-D(\hat{x}, t)] \)

**Training (implemented in `text_im_train.py`)**
- Backbones: `resnet18`, `resnet34`, `resnet50` are used as *capacity presets* (controls decoder channel width)
- Epochs: up to **100**
- LR: **1e‑4**, AdamW, grad clip 1.0
- KL annealing: \(\beta: 0 \rightarrow 1e\!-\!4\) over **15** epochs
- Perceptual weight: default **0.1**
- Edge weight: default **0.10**
- GAN (enabled by default in current code):
  - discriminator backbone: **ResNet‑18**
  - discriminator LR: default **2e‑4**
  - adversarial weight: default **0.02**
- Scheduler: default **OneCycleLR**
- Early stopping: patience **15**

**Evaluation outputs**
- Metrics: MSE / PSNR / SSIM (simple global SSIM)
- Saves per-sample comparisons and JSONL summaries:
  - `results/text2im_<backbone>_test.jsonl`
  - `results/text2im_<backbone>_images/*_comparison.png` (+ grid image)

**Current reported test summaries (from `results/*.jsonl`)**
- **ResNet‑18 preset**: avg MSE **0.06146**, PSNR **12.93 dB**, SSIM **0.1517**
- **ResNet‑34 preset**: avg MSE **0.06561**, PSNR **12.45 dB**, SSIM **0.1161**
- **ResNet‑50 preset**: avg MSE **0.06321**, PSNR **12.66 dB**, SSIM **0.1420**

- **Method B — Small diffusion from scratch (implemented; did not reach satisfactory generation quality)**:
Files: `text_im_small_diff_dataset.py`, `text_im_small_diff_train.py`, `text_im_small_diff_test.py`

- **Conditioning**: frozen **CLIP text encoder** (`openai/clip-vit-base-patch32`) + CLIP tokenizer (max length **77**)
- **Backbones** (capacity configs):
  - `tiny`: `block_out_channels=(64,128,192)`, `layers_per_block=1`
  - `small`: `block_out_channels=(96,192,384)`, `layers_per_block=2`
  - `base`: `block_out_channels=(128,256,512)`, `layers_per_block=2`
- **Model**: `diffusers.UNet2DConditionModel` with cross-attention blocks (pixel-space, 128×128)
- **Noise schedule**: DDPM, `beta_schedule="squaredcos_cap_v2"`, 1000 timesteps
- **Training objective**: predict \(\epsilon\) with **MSE**, using **Min‑SNR weighting** (gamma default **5.0**)
- **Sampling**: classifier-free guidance (default guidance scale **5.0**)
- Outputs: `runs/text2im_small_diff_*` and `checkpoints/text2im_small_diff_*`

- **Method C — Stable Diffusion v1.5 LoRA finetune (attempted; did not reach satisfactory generation quality)**:
Files: `text_im_sd_dataset.py`, `text_im_sd_train.py`, `text_im_sd_test.py`

- **Base model**: `runwayml/stable-diffusion-v1-5`
- **Frozen**: VAE + text encoder
- **Trainable**: LoRA adapters on UNet (conservative target modules: `["to_out.0"]` only)
- **Objective**: latent-space noise prediction with **weighted MSE** (Min‑SNR gamma default **5.0**)
- **Training settings (defaults)**:
  - batch size **2**, gradient accumulation **4**
  - LR **5e‑6**, scheduler **constant**
  - mixed precision **fp16**
  - save LoRA weights to `checkpoints/text2im_sd/*_lora`

---

### Cross-Modal Pipeline (Image → Text → Image)
**Intended evaluation**: run the captioner on a test image, then feed the generated caption into the text-to-image generator to reconstruct an image.

**Current status**: the pipeline connection is conceptually defined, but **end-to-end reconstruction quality is blocked by Text→Image performance**.

---

### Next Steps (planned improvements)
To complete and connect the two pipelines for a compelling report demo, the planned approach is to use a **pretrained, high-capacity generator**:
- **Use `runwayml/stable-diffusion-v1-5` (pretrained)** as the generation backbone for a stable Image→Image / Caption→Image comparison pipeline.
- Use this pretrained generator to produce visually meaningful reconstructions for qualitative cycle evaluation and side-by-side comparisons.

---

### How to Run (reproducibility checklist)
#### Environment
Install Python dependencies:

```bash
pip install -r requirements.txt
```

Notes:
- Some experiments (small diffusion, Stable Diffusion LoRA) require additional packages referenced in imports (e.g., `diffusers`, `peft`, `skimage`). If missing, install them before running those scripts.

#### Caption generation (OpenAI)
Set your API key:

```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your_key_here"
```

#### Dataset creation
Run dataset steps in order:
- `python dataset_scripts/1_load_images.py`
- `python dataset_scripts/2_process_metadata.py`
- `python dataset_scripts/3_generate_captions.py`
- `python dataset_scripts/4_preprocess_images.py`
- `python dataset_scripts/5_tokenize_captions.py`
- `python dataset_scripts/6_create_splits.py`

#### Train / test Image → Text
- Train: `python im_text_train.py`
- Test: `python im_text_test.py`

#### Train / test Text → Image (Method A: VAE)
- Train: `python text_im_train.py`
- Test: `python text_im_test.py`

#### Train / test Text → Image (Method B: small diffusion)
- Train: `python text_im_small_diff_train.py --backbone small`
- Test: `python text_im_small_diff_test.py --backbone small`

#### Train / test Text → Image (Method C: SD LoRA)
- Train: `python text_im_sd_train.py`
- Test: `python text_im_sd_test.py`

---

### Report-Writing Template (fill these in)
- **Abstract**: [Problem, approach, results in 3–5 sentences]
- **Introduction & Motivation**: [Why bidirectional vision-language mapping?]
- **Dataset**:
  - [Source, preprocessing, caption generation prompt, tokenization, split strategy]
  - [Dataset size and any filtering]
- **Methods**:
  - [Image→Text architecture + loss + training schedule]
  - [Text→Image methods attempted + loss functions + training schedule]
- **Experiments**:
  - [Compute, epochs, batch sizes, early stopping, schedulers]
- **Results**:
  - [Quantitative tables from JSONL summaries]
  - [Qualitative figures from `runs/` and `results/`]
- **Failure analysis (required here)**:
  - [Why Text→Image collapsed; what symptoms; what evidence]
- **Conclusion & Future Work**:
  - [Using pretrained Stable Diffusion for a stronger end-to-end pipeline demo]

