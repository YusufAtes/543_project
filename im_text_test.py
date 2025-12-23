import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

from im_text_dataset import ArtCaptionDataset
from im_text_model import ImageCaptioner

def compute_bleu_scores(reference, candidate):
    """Compute BLEU-1, BLEU-2, BLEU-3, BLEU-4 scores."""
    ref_tokens = word_tokenize(reference.lower())
    cand_tokens = word_tokenize(candidate.lower())
    
    smoothing = SmoothingFunction().method1
    
    bleu_1 = sentence_bleu([ref_tokens], cand_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu_2 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu_3 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
    bleu_4 = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
    
    return {
        "bleu_1": float(bleu_1),
        "bleu_2": float(bleu_2),
        "bleu_3": float(bleu_3),
        "bleu_4": float(bleu_4),
    }

def evaluate_test_set(backbone, root="dataset", tokenizer_name="gpt2", max_len=256, 
                     image_size=224, batch_size=16, max_new_tokens=200):
    """Evaluate model on test set and compute metrics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    ckpt_path = f"checkpoints/im2text_{backbone}/best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Load dataset
    test_ds = ArtCaptionDataset(root, "test", tokenizer_name, max_len, image_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model
    vocab_size = len(test_ds.tokenizer)
    model = ImageCaptioner(
        vocab_size=vocab_size,
        max_len=max_len,
        backbone=ckpt["backbone"],
        backbone_pretrained=ckpt.get("backbone_pretrained", True),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    pad_id = test_ds.tokenizer.pad_token_id
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    # Evaluation metrics
    total_loss, total_tokens = 0.0, 0
    all_results = []
    all_bleu_scores = {"bleu_1": [], "bleu_2": [], "bleu_3": [], "bleu_4": []}
    
    print(f"\nEvaluating {backbone} on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = batch["image"].to(device)
            decoder_in = batch["decoder_in"].to(device)
            targets = batch["targets"].to(device)
            dec_pad_mask = batch["dec_pad_mask"].to(device)
            
            # Compute loss
            logits = model(images, decoder_in, dec_pad_mask)
            B, L, V = logits.shape
            loss = loss_fn(logits.reshape(B * L, V), targets.reshape(B * L))
            
            nonpad = (targets != pad_id).sum().item()
            total_loss += loss.item() * nonpad
            total_tokens += nonpad
            
            # Generate captions
            generated_texts = model.generate(images, test_ds.tokenizer, max_new_tokens=max_new_tokens)
            
            # Get reference captions
            batch_ids = test_ds.ids[batch_idx * batch_size:(batch_idx + 1) * batch_size]
            
            for i, img_id in enumerate(batch_ids):
                if i >= len(generated_texts):
                    break
                    
                generated = generated_texts[i]
                reference = test_ds.captions[img_id]
                
                # Compute BLEU scores
                bleu_scores = compute_bleu_scores(reference, generated)
                for key in all_bleu_scores:
                    all_bleu_scores[key].append(bleu_scores[key])
                
                # Store result
                result = {
                    "image_id": img_id,
                    "generated_caption": generated,
                    "reference_caption": reference,
                    **bleu_scores,
                }
                all_results.append(result)
    
    # Compute average metrics
    avg_loss = total_loss / max(total_tokens, 1)
    avg_ppl = np.exp(avg_loss)
    
    avg_bleu = {
        "bleu_1": np.mean(all_bleu_scores["bleu_1"]),
        "bleu_2": np.mean(all_bleu_scores["bleu_2"]),
        "bleu_3": np.mean(all_bleu_scores["bleu_3"]),
        "bleu_4": np.mean(all_bleu_scores["bleu_4"]),
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Results for {backbone}")
    print(f"{'='*60}")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Perplexity: {avg_ppl:.2f}")
    print(f"BLEU-1: {avg_bleu['bleu_1']:.4f}")
    print(f"BLEU-2: {avg_bleu['bleu_2']:.4f}")
    print(f"BLEU-3: {avg_bleu['bleu_3']:.4f}")
    print(f"BLEU-4: {avg_bleu['bleu_4']:.4f}")
    print(f"{'='*60}\n")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results_file = f"results/im2text_{backbone}_best_test.jsonl"
    
    # Add summary metrics as first entry
    summary = {
        "summary": True,
        "backbone": backbone,
        "test_loss": float(avg_loss),
        "test_ppl": float(avg_ppl),
        **{f"avg_{k}": float(v) for k, v in avg_bleu.items()},
        "num_samples": len(all_results),
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        # Write summary first
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        # Write individual results
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Results saved to {results_file}")
    
    return avg_loss, avg_ppl, avg_bleu, all_results

def main():
    """Evaluate all trained backbones on test set."""
    backbones = ["resnet18", "resnet34", "resnet50"]
    
    for backbone in backbones:
        try:
            evaluate_test_set(backbone)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print(f"Skipping {backbone}\n")
            continue
    
    print("\nAll evaluations complete!")

if __name__ == "__main__":
    main()

