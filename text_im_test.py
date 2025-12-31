import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from PIL import Image

from text_im_dataset import TextToImageDataset, denormalize_image
from text_im_model import TextToImageVAE, vae_loss


def compute_psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1, img2: Tensors of shape (C, H, W) in [0, 1] range
    Returns:
        psnr: PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(1.0 / mse)


def compute_ssim(img1, img2, window_size=11, C1=0.01**2, C2=0.03**2):
    """
    Compute Structural Similarity Index between two images.
    Simplified implementation without Gaussian weighting.
    
    Args:
        img1, img2: Tensors of shape (C, H, W) in [0, 1] range
    Returns:
        ssim: SSIM value
    """
    # Convert to numpy for easier computation
    img1 = img1.numpy()
    img2 = img2.numpy()
    
    # Compute means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Compute variances and covariance
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    # SSIM formula
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    return numerator / denominator


def save_comparison_image(original, generated, image_id, save_dir):
    """
    Save side-by-side comparison of original and generated images.
    
    Args:
        original: Original image tensor (C, H, W) in [0, 1] range
        generated: Generated image tensor (C, H, W) in [0, 1] range
        image_id: Image identifier
        save_dir: Directory to save the comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    # Original image
    axes[0].imshow(original.permute(1, 2, 0).numpy())
    axes[0].set_title("Original")
    axes[0].axis('off')
    
    # Generated image
    axes[1].imshow(generated.permute(1, 2, 0).numpy())
    axes[1].set_title("Generated")
    axes[1].axis('off')
    
    plt.suptitle(f"Image ID: {image_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{image_id}_comparison.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()


def save_grid_comparison(originals, generateds, save_path, num_cols=5):
    """
    Save a grid comparison of multiple original and generated image pairs.
    
    Args:
        originals: List of original image tensors
        generateds: List of generated image tensors
        save_path: Path to save the grid
        num_cols: Number of columns in the grid
    """
    num_images = len(originals)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows * 2, num_cols, figsize=(3 * num_cols, 3 * num_rows * 2))
    
    for i in range(num_images):
        row_orig = (i // num_cols) * 2
        row_gen = row_orig + 1
        col = i % num_cols
        
        # Handle single row case
        if num_rows * 2 == 2 and num_cols == 1:
            ax_orig = axes[0]
            ax_gen = axes[1]
        elif num_rows * 2 == 2:
            ax_orig = axes[0, col]
            ax_gen = axes[1, col]
        elif num_cols == 1:
            ax_orig = axes[row_orig]
            ax_gen = axes[row_gen]
        else:
            ax_orig = axes[row_orig, col]
            ax_gen = axes[row_gen, col]
        
        ax_orig.imshow(originals[i].permute(1, 2, 0).numpy())
        ax_orig.axis('off')
        if i < num_cols:
            ax_orig.set_title("Original", fontsize=8)
        
        ax_gen.imshow(generateds[i].permute(1, 2, 0).numpy())
        ax_gen.axis('off')
        if i < num_cols:
            ax_gen.set_title("Generated", fontsize=8)
    
    # Hide empty subplots
    for i in range(num_images, num_rows * num_cols):
        row_orig = (i // num_cols) * 2
        row_gen = row_orig + 1
        col = i % num_cols
        
        if num_rows * 2 > 2 and num_cols > 1:
            axes[row_orig, col].axis('off')
            axes[row_gen, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def evaluate_test_set(
    backbone,
    root="dataset",
    max_len=256,
    image_size=128,
    batch_size=16,
    save_individual=True,
    max_individual_saves=50,
):
    """
    Evaluate model on test set and compute metrics.
    
    Args:
        backbone: Model backbone name
        root: Path to dataset directory
        max_len: Maximum sequence length
        image_size: Image size (128x128)
        batch_size: Batch size for evaluation
        save_individual: Whether to save individual comparison images
        max_individual_saves: Maximum number of individual comparisons to save
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    ckpt_path = f"checkpoints/text2im_{backbone}/best.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Load dataset
    test_ds = TextToImageDataset(root, "test", max_len, image_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    # Initialize model
    model = TextToImageVAE(
        vocab_size=50257,
        max_len=ckpt.get("max_len", max_len),
        backbone=ckpt["backbone"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # Create results directories
    results_dir = f"results/text2im_{backbone}_images"
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluation metrics
    all_results = []
    all_mse = []
    all_psnr = []
    all_ssim = []
    
    # For grid comparison
    grid_originals = []
    grid_generateds = []
    
    print(f"\nEvaluating {backbone} on test set...")
    saved_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            target_images = batch["image"].to(device)
            image_ids = batch["image_id"]
            
            # Generate images
            generated_images, mu, logvar = model(input_ids, attention_mask, sample=False)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss(
                generated_images, target_images, mu, logvar, 
                beta=ckpt.get("beta", 0.001)
            )
            
            # Process each sample
            for i in range(len(image_ids)):
                img_id = image_ids[i]
                
                # Denormalize images to [0, 1]
                orig_img = denormalize_image(target_images[i].cpu())
                gen_img = denormalize_image(generated_images[i].cpu())
                
                # Clamp to valid range
                orig_img = torch.clamp(orig_img, 0, 1)
                gen_img = torch.clamp(gen_img, 0, 1)
                
                # Compute metrics
                mse = torch.mean((orig_img - gen_img) ** 2).item()
                psnr = compute_psnr(orig_img, gen_img)
                ssim = compute_ssim(orig_img, gen_img)
                
                all_mse.append(mse)
                all_psnr.append(psnr)
                all_ssim.append(ssim)
                
                # Store result
                result = {
                    "image_id": img_id,
                    "mse": float(mse),
                    "psnr": float(psnr),
                    "ssim": float(ssim),
                }
                all_results.append(result)
                
                # Save individual comparison (up to max_individual_saves)
                if save_individual and saved_count < max_individual_saves:
                    save_comparison_image(orig_img, gen_img, img_id, results_dir)
                    saved_count += 1
                
                # Collect for grid (first 25 samples)
                if len(grid_originals) < 25:
                    grid_originals.append(orig_img)
                    grid_generateds.append(gen_img)
    
    # Compute average metrics
    avg_mse = np.mean(all_mse)
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Test Results for {backbone}")
    print(f"{'='*60}")
    print(f"Number of test samples: {len(all_results)}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"{'='*60}\n")
    
    # Save grid comparison
    grid_path = os.path.join(results_dir, "grid_comparison.png")
    save_grid_comparison(grid_originals, grid_generateds, grid_path)
    print(f"Grid comparison saved to {grid_path}")
    
    # Save results to JSONL
    results_file = f"results/text2im_{backbone}_test.jsonl"
    
    # Add summary metrics as first entry
    summary = {
        "summary": True,
        "backbone": backbone,
        "avg_mse": float(avg_mse),
        "avg_psnr": float(avg_psnr),
        "avg_ssim": float(avg_ssim),
        "num_samples": len(all_results),
    }
    
    with open(results_file, "w", encoding="utf-8") as f:
        # Write summary first
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        # Write individual results
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print(f"Results saved to {results_file}")
    print(f"Individual comparisons saved to {results_dir}/")
    
    return avg_mse, avg_psnr, avg_ssim, all_results


def main():
    """Evaluate all trained backbones on test set."""
    backbones = ["resnet18", "resnet34", "resnet50"]
    
    all_summaries = []
    
    for backbone in backbones:
        try:
            avg_mse, avg_psnr, avg_ssim, _ = evaluate_test_set(backbone)
            all_summaries.append({
                "backbone": backbone,
                "mse": avg_mse,
                "psnr": avg_psnr,
                "ssim": avg_ssim,
            })
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print(f"Skipping {backbone}\n")
            continue
    
    # Print comparison table
    if all_summaries:
        print(f"\n{'='*60}")
        print("Summary Comparison of All Backbones")
        print(f"{'='*60}")
        print(f"{'Backbone':<12} {'MSE':<12} {'PSNR (dB)':<12} {'SSIM':<12}")
        print("-" * 48)
        for s in all_summaries:
            print(f"{s['backbone']:<12} {s['mse']:<12.6f} {s['psnr']:<12.2f} {s['ssim']:<12.4f}")
        print(f"{'='*60}")
    
    print("\nAll evaluations complete!")


if __name__ == "__main__":
    main()
