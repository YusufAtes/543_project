"""
Complete test-time pipeline:

Image (test split) -> Image-to-Text (im2text) -> generated caption -> Stable Diffusion (text2im) -> generated image

Outputs:
For every test image and for each im2text backbone (resnet18/resnet34/resnet50),
saves a composite figure:
  [Original image]  |  [First ~N words of generated caption]  |  [Generated image]

Saved under:
  results/complete_pipeline/<backbone>/<image_id>_pipeline.png

Run (recommended via conda env):
  conda run -n eee543 python complete_test_pipeline.py
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import AutoTokenizer

from diffusers import StableDiffusionPipeline

from im_text_model import ImageCaptioner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="dataset", help="Path to dataset/ images dir (00001.jpg, ...)")
    p.add_argument("--splits_dir", type=str, default="dataset_splits", help="Directory containing test_ids.txt")
    p.add_argument("--backbones", type=str, default="resnet18,resnet34,resnet50")

    p.add_argument("--sd_model_id", type=str, default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=7.5)

    p.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens to generate for im2text")
    p.add_argument("--caption_words_for_figure", type=int, default=85, help="Words shown in the center panel")

    p.add_argument("--max_samples", type=int, default=None, help="Limit number of test samples (default: all)")
    p.add_argument("--seed", type=int, default=123)

    p.add_argument("--sd_dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (default: auto)")

    p.add_argument("--panel_size", type=int, default=512, help="Size of left/right image panels (pixels)")
    return p.parse_args()


def load_test_ids(splits_dir: str) -> List[str]:
    test_path = Path(splits_dir) / "test_ids.txt"
    with open(test_path, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    return ids


def truncate_words(text: str, n_words: int) -> str:
    words = text.split()
    if len(words) <= n_words:
        return text.strip()
    return " ".join(words[:n_words]).strip() + "..."


def wrap_text(text: str, max_chars_per_line: int = 42, max_lines: int = 14) -> List[str]:
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
            continue
        if len(cur) + 1 + len(w) <= max_chars_per_line:
            cur = f"{cur} {w}"
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break
    if len(lines) < max_lines and cur:
        lines.append(cur)
    if len(lines) == max_lines and words:
        # indicate truncation if we hit max lines
        if not lines[-1].endswith("..."):
            lines[-1] = lines[-1][: max(0, max_chars_per_line - 3)] + "..."
    return lines


def get_font(size: int = 18):
    # Windows: try arial; fall back to default
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def make_composite(original: Image.Image, caption: str, generated: Image.Image, panel_size: int) -> Image.Image:
    """
    Create a 3-panel composite: original | caption text | generated
    """
    pad = 16
    title_h = 42
    W = panel_size
    H = panel_size

    canvas_w = W * 3 + pad * 4
    canvas_h = H + pad * 3 + title_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")

    # Resize images
    orig = original.convert("RGB").resize((W, H), Image.Resampling.BICUBIC)
    gen = generated.convert("RGB").resize((W, H), Image.Resampling.BICUBIC)

    # Paste images
    y0 = pad + title_h
    canvas.paste(orig, (pad, y0))
    canvas.paste(gen, (pad * 3 + W * 2, y0))

    draw = ImageDraw.Draw(canvas)
    title_font = get_font(22)
    # Slightly larger caption text for readability (user requested)
    text_font = get_font(18)

    # Titles
    draw.text((pad + W // 2 - 40, pad), "Original", fill="black", font=title_font)
    draw.text((pad * 2 + W + W // 2 - 35, pad), "Generated Caption", fill="black", font=title_font)
    draw.text((pad * 3 + W * 2 + W // 2 - 50, pad), "Generated", fill="black", font=title_font)

    # Caption box
    cap_x0 = pad * 2 + W
    cap_y0 = y0
    cap_x1 = cap_x0 + W
    cap_y1 = cap_y0 + H
    draw.rectangle([cap_x0, cap_y0, cap_x1, cap_y1], outline="black", width=1)

    # Wrapped caption text
    # With a larger font, use slightly fewer characters per line and a bit more line spacing.
    lines = wrap_text(caption, max_chars_per_line=40, max_lines=18)
    line_h = 20
    tx = cap_x0 + 12
    ty = cap_y0 + 12
    for i, line in enumerate(lines):
        draw.text((tx, ty + i * line_h), line, fill="black", font=text_font)

    return canvas


@torch.no_grad()
def generate_caption_batch(
    model: ImageCaptioner,
    tokenizer,
    image_tensors: torch.Tensor,
    max_new_tokens: int,
) -> List[str]:
    # model.generate already returns list[str]
    return model.generate(image_tensors, tokenizer, max_new_tokens=max_new_tokens)


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()]

    data_root = Path(args.data_root)
    splits_dir = Path(args.splits_dir)
    out_root = Path("results") / "complete_pipeline"
    out_root.mkdir(parents=True, exist_ok=True)

    # --- test ids ---
    test_ids = load_test_ids(str(splits_dir))
    if args.max_samples is not None:
        test_ids = test_ids[: args.max_samples]
    print(f"Test samples: {len(test_ids)}")

    # --- tokenizer (GPT-2 to match training) ---
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- image transform for im2text model ---
    # Must match im_text_dataset.py normalization (ImageNet stats)
    im2text_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # --- Stable Diffusion pipeline ---
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    sd_dtype = torch.float16 if args.sd_dtype == "fp16" else torch.float32
    print(f"Loading Stable Diffusion: {args.sd_model_id} (dtype={args.sd_dtype}, device={device})")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_model_id,
        torch_dtype=sd_dtype,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    # --- iterate over each im2text backbone ---
    for backbone in backbones:
        ckpt_path = Path("checkpoints") / f"im2text_{backbone}" / "best.pt"
        if not ckpt_path.exists():
            print(f"[WARN] Missing checkpoint: {ckpt_path} (skipping backbone {backbone})")
            continue

        print(f"\nLoading im2text backbone: {backbone} from {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

        model = ImageCaptioner(
            vocab_size=len(tokenizer),
            max_len=ckpt.get("max_len", 256),
            backbone=backbone,
            backbone_pretrained=ckpt.get("backbone_pretrained", True),
        ).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        bb_out = out_root / backbone
        bb_out.mkdir(parents=True, exist_ok=True)

        # Optional: save captions as JSONL for later report analysis
        captions_log_path = bb_out / "generated_captions.jsonl"
        cap_log_f = open(captions_log_path, "w", encoding="utf-8")

        for idx, image_id in enumerate(test_ids, start=1):
            img_path = data_root / f"{image_id}.jpg"
            if not img_path.exists():
                print(f"[WARN] Missing image: {img_path} (skipping)")
                continue

            # Load original image for display
            orig_pil = Image.open(img_path).convert("RGB")

            # Prepare tensor for im2text
            img_tensor = im2text_tf(orig_pil).unsqueeze(0).to(device)

            # Generate caption
            gen_caption = generate_caption_batch(
                model=model,
                tokenizer=tokenizer,
                image_tensors=img_tensor,
                max_new_tokens=args.max_new_tokens,
            )[0]

            # SD prompt (SD internally truncates to CLIP max length; keep prompt reasonably sized)
            # Also avoid CLIP tokenizer warnings by shortening very long generated captions.
            # Keeping the first ~60 words is usually enough for conditioning.
            sd_prompt = truncate_words(gen_caption.strip(), 60)

            # Generate image from caption
            generator = torch.Generator(device=device).manual_seed(args.seed + idx)
            with torch.autocast(device_type="cuda", dtype=sd_dtype, enabled=(device == "cuda" and sd_dtype == torch.float16)):
                out = pipe(
                    prompt=sd_prompt,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                )
            gen_pil = out.images[0]

            # Build composite figure
            caption_for_fig = truncate_words(gen_caption, args.caption_words_for_figure)
            composite = make_composite(orig_pil, caption_for_fig, gen_pil, panel_size=args.panel_size)

            save_path = bb_out / f"{image_id}_pipeline.png"
            composite.save(save_path)

            # Log
            cap_log_f.write(json.dumps(
                {"image_id": image_id, "backbone": backbone, "generated_caption": gen_caption},
                ensure_ascii=False
            ) + "\n")

            if idx % 10 == 0 or idx == 1 or idx == len(test_ids):
                print(f"[{backbone}] {idx}/{len(test_ids)} saved: {save_path}")

        cap_log_f.close()
        print(f"[{backbone}] Captions saved to: {captions_log_path}")

    print("\nDone. Outputs saved under:", out_root)


if __name__ == "__main__":
    main()


