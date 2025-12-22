import json
import os
import base64
import time
from pathlib import Path

import requests

# ============================================================
# CONFIG
# ============================================================

# Path to your image_mappings.json (generated earlier)
IMAGE_MAPPING_PATH = Path(__file__).resolve().parents[2] / "dataset" / "image_mapping.json"

# Where to save captions
OUTPUT_CAPTIONS_PATH = Path(__file__).resolve().parents[2] / "dataset" / "captions.json"

# OpenAI API configuration for gpt-4o-mini vision
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # set this in your env
MODEL_NAME = "gpt-4o-mini"

# Fixed prompt (target 100–120 words)
BASE_PROMPT = (
    "Generate a caption of 100-120 words.\n"
    "1. Describe clearly what is pictured in the image (people, objects, background, colors, lighting).\n"
    "2. At the end, state the type and style of the image (for example: 'It is an oil painting, a portrait in a mannerist style.').\n"
    "Use simple, direct sentences."
)

# How long to sleep between requests (seconds) to avoid rate limits.
# This is a conservative value to reduce 429 errors; you can lower it later
# if you see very few rate-limit responses.
REQUEST_SLEEP = 3.0


# ============================================================
# HELPER: encode image as base64
# ============================================================

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode("utf-8")


# ============================================================
# HELPER: call OpenAI vision model
# ============================================================

def call_openai_gpt4o_mini(image_path: str, prompt: str) -> str:
    """
    Sends an image + prompt to OpenAI gpt-4o-mini and returns the caption text.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    img_b64 = encode_image_to_base64(image_path)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }

    # Build payload once (image is constant for this call)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        },
                    },
                ],
            }
        ],
        "temperature": 0.4,
        # Rough cap; 120 words ~ 170–200 tokens, plus prompt context
        "max_tokens": 220,
    }

    # Simple retry with backoff for 429/rate limit
    max_retries = 20
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                OPENAI_API_URL, headers=headers, json=payload, timeout=60
            )

            if resp.status_code == 429:
                # Too many requests – back off and retry
                wait = 4
                print(f"  [Rate limit] 429 from OpenAI, slept {wait * (attempt + 1)}s and retrying...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            caption = data["choices"][0]["message"]["content"].strip()

            # Enforce 100–120 words by truncation if needed
            words = caption.split()
            if len(words) > 120:
                caption = " ".join(words[:120])
            return caption

        except requests.exceptions.RequestException as e:
            # Network / HTTP issues – small backoff and retry
            wait = 3 * (attempt + 1)
            print(f"  [HTTP error] {e}. Sleeping {wait}s and retrying...")
            time.sleep(wait)
            continue

    # If we reach here, all retries failed
    raise RuntimeError("Failed to get caption from OpenAI after multiple retries.")


# ============================================================
# MAIN: read mappings, generate captions, save
# ============================================================

def main():
    # --- load image mappings ---
    if not IMAGE_MAPPING_PATH.exists():
        raise FileNotFoundError(f"IMAGE_MAPPING_PATH not found: {IMAGE_MAPPING_PATH}")

    with open(IMAGE_MAPPING_PATH, "r", encoding="utf-8") as f:
        image_mappings = json.load(f)

    # If you're resuming, load existing captions to avoid re-querying
    if OUTPUT_CAPTIONS_PATH.exists():
        try:
            with open(OUTPUT_CAPTIONS_PATH, "r", encoding="utf-8") as f:
                captions = json.load(f)
        except json.JSONDecodeError:
            print("captions.json is empty or corrupted, starting fresh.")
            captions = {}
    else:
        captions = {}

    total = len(image_mappings)
    print(f"Found {total} images in mapping.")

    for idx, (image_id, image_path) in enumerate(image_mappings.items(), start=1):
        # Skip if we already have a caption for this image_id
        if image_id in captions:
            if idx % 1000 == 0:
                print(f"[{idx}/{total}] {image_id} already captioned, skipping.")
            continue

        # Normalize path
        image_path = str(Path(image_path))

        if not os.path.exists(image_path):
            print(f"[{idx}/{total}] WARNING: image not found at {image_path}, skipping.")
            continue

        print(f"[{idx}/{total}] Processing {image_id} -> {image_path}")

        try:
            caption = call_openai_gpt4o_mini(image_path, BASE_PROMPT)
            captions[image_id] = caption

            # Save after each image (safer if script crashes)
            OUTPUT_CAPTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_CAPTIONS_PATH, "w", encoding="utf-8") as f:
                json.dump(captions, f, ensure_ascii=False, indent=2)

            print(f"  ✓ Caption length: {len(caption.split())} words")
        except Exception as e:
            print(f"  ✗ Error captioning image {image_id}: {e}")

        # Respect rate limits
        time.sleep(REQUEST_SLEEP)

    print("Done. Captions saved to:", OUTPUT_CAPTIONS_PATH)


if __name__ == "__main__":
    main()
