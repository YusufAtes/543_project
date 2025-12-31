import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from text_im_dataset import TextToImageDataset, denormalize_image
from text_im_model import TextToImageVAE, vae_loss, PerceptualLoss, ResNetPerceptualLoss, TextImageDiscriminator


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=5, min_delta=0.0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.should_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience} (no improvement)")
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


class KLAnnealer:
    """
    Gradually increase KL weight (beta) during training.
    This helps the model first learn to reconstruct images, 
    then gradually regularize the latent space.
    """
    
    def __init__(self, start_beta=0.0, end_beta=0.0001, warmup_epochs=10):
        self.start_beta = start_beta
        self.end_beta = end_beta
        self.warmup_epochs = warmup_epochs
    
    def get_beta(self, epoch):
        """Get current beta value based on epoch."""
        if epoch >= self.warmup_epochs:
            return self.end_beta
        # Linear interpolation
        progress = epoch / self.warmup_epochs
        return self.start_beta + progress * (self.end_beta - self.start_beta)


def run_epoch(model, loader, optimizer, device, beta, perceptual_loss_fn,
              perceptual_weight, edge_weight=0.0, train=True):
    """
    Run one epoch of training or validation.
    
    Args:
        model: TextToImageVAE model
        loader: DataLoader
        optimizer: Optimizer (only used if train=True)
        device: Device to use
        beta: KL divergence weight
        perceptual_loss_fn: PerceptualLoss module
        perceptual_weight: Weight for perceptual loss
        train: If True, train mode; if False, eval mode
    Returns:
        avg_loss: Average total loss
        avg_recon_loss: Average reconstruction loss
        avg_kl_loss: Average KL divergence loss
    """
    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_recon, total_kl, total_samples = 0.0, 0.0, 0.0, 0

    pbar = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        target_image = batch["image"].to(device)

        with torch.set_grad_enabled(train):
            # Forward pass
            recon_image, mu, logvar = model(input_ids, attention_mask, sample=train)
            
            # Compute loss with perceptual component
            loss, recon_loss, kl_loss = vae_loss(
                recon_image, target_image, mu, logvar,
                perceptual_loss_fn=perceptual_loss_fn,
                beta=beta,
                perceptual_weight=perceptual_weight,
                edge_weight=edge_weight,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        batch_size = input_ids.size(0)
        total_loss += loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_kl += kl_loss.item() * batch_size
        total_samples += batch_size
        
        pbar.set_postfix(
            loss=total_loss / total_samples,
            recon=total_recon / total_samples,
            kl=total_kl / total_samples
        )

    avg_loss = total_loss / max(total_samples, 1)
    avg_recon = total_recon / max(total_samples, 1)
    avg_kl = total_kl / max(total_samples, 1)
    
    return avg_loss, avg_recon, avg_kl


def save_sample_images(model, val_loader, device, run_dir, epoch, num_samples=8):
    """Generate and save sample reconstructions."""
    model.eval()
    
    # Get a batch of samples
    batch = next(iter(val_loader))
    input_ids = batch["input_ids"][:num_samples].to(device)
    attention_mask = batch["attention_mask"][:num_samples].to(device)
    target_images = batch["image"][:num_samples].to(device)
    
    with torch.no_grad():
        recon_images, _, _ = model(input_ids, attention_mask, sample=False)
    
    # Denormalize images for visualization
    target_vis = denormalize_image(target_images.cpu())
    recon_vis = denormalize_image(recon_images.cpu())
    
    # Clamp to valid range
    target_vis = torch.clamp(target_vis, 0, 1)
    recon_vis = torch.clamp(recon_vis, 0, 1)
    
    # Create comparison grid: top row = original, bottom row = reconstructed
    comparison = torch.cat([target_vis, recon_vis], dim=0)
    grid = vutils.make_grid(comparison, nrow=num_samples, padding=2, normalize=False)
    
    # Save image
    plt.figure(figsize=(16, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.title(f'Epoch {epoch}: Original (top) vs Reconstructed (bottom)')
    plt.savefig(os.path.join(run_dir, f"samples_epoch_{epoch:03d}.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()


def train_backbone(
    backbone,
    root="dataset",
    max_len=256,
    image_size=128,
    batch_size=64,
    epochs=100,
    lr=1e-4,
    beta_start=0.0,
    beta_end=0.0001,
    beta_warmup=15,
    perceptual_weight=0.1,
    edge_weight=0.10,
    adv_weight=0.02,
    d_lr=2e-4,
    use_resnet_perceptual=True,
    use_gan=False,
    patience=5,
    lr_scheduler_type="onecycle",
    save_samples_every=2,
):
    """
    Train a single backbone model with improved settings.
    
    Args:
        backbone: Model backbone name (resnet18, resnet34, resnet50)
        root: Path to dataset directory
        max_len: Maximum sequence length for captions
        image_size: Target image size (128x128)
        batch_size: Training batch size
        epochs: Maximum number of epochs
        lr: Learning rate
        beta_start: Initial KL weight
        beta_end: Final KL weight
        beta_warmup: Number of epochs for KL annealing
        perceptual_weight: Weight for perceptual loss
        patience: Early stopping patience
        lr_scheduler_type: "plateau", "cosine", or "onecycle"
        save_samples_every: Save sample reconstructions every N epochs
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Create directories for this backbone
    run_dir = f"runs/text2im_{backbone}"
    ckpt_dir = f"checkpoints/text2im_{backbone}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # CSV logging setup
    history_file = os.path.join(run_dir, "history.csv")
    with open(history_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_recon", "train_kl", 
                        "val_loss", "val_recon", "val_kl", "lr", "beta"])
    
    # Track history for plotting
    history = {
        "epoch": [], 
        "train_loss": [], "train_recon": [], "train_kl": [],
        "val_loss": [], "val_recon": [], "val_kl": [], 
        "lr": [], "beta": []
    }
    
    print(f"\n{'='*60}")
    print(f"Training Text-to-Image VAE with {backbone}")
    print(f"  - LR: {lr}, Scheduler: {lr_scheduler_type}")
    print(f"  - Beta: {beta_start} -> {beta_end} (warmup: {beta_warmup} epochs)")
    print(f"  - Perceptual weight: {perceptual_weight}")
    print(f"  - Edge weight: {edge_weight}")
    print(f"  - GAN: {use_gan} (adv_weight={adv_weight}, d_lr={d_lr})")
    print(f"  - Early stopping patience: {patience}")
    print(f"  - Image size: {image_size}x{image_size}")
    print(f"{'='*60}")
    
    # ---- datasets ----
    train_ds = TextToImageDataset(root, "train", max_len, image_size)
    val_ds = TextToImageDataset(root, "val", max_len, image_size)
    
    # num_workers=0 avoids slow multiprocessing startup on Windows
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # ---- model ----
    model = TextToImageVAE(
        vocab_size=50257,  # GPT-2 vocab size
        max_len=max_len,
        backbone=backbone,
        latent_dim=512,  # Increased latent dimension
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # ---- perceptual loss ----
    # VGG perceptual is decent, but ResNet perceptual is cheaper and aligns with "use resnet backbones".
    if use_resnet_perceptual:
        perceptual_loss_fn = ResNetPerceptualLoss(backbone="resnet18", style_layers=True).to(device)
    else:
        perceptual_loss_fn = PerceptualLoss(device=device).to(device)

    # ---- discriminator (optional) ----
    discriminator = None
    d_optimizer = None
    if use_gan:
        discriminator = TextImageDiscriminator(text_dim=model.text_dim, backbone="resnet18").to(device)
        d_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=d_lr, weight_decay=0.01, betas=(0.0, 0.9))
    
    # ---- optimizer (generator) ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.999))
    
    # ---- learning rate scheduler ----
    if lr_scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, 
            verbose=True, min_lr=1e-6
        )
    elif lr_scheduler_type == "cosine":
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    elif lr_scheduler_type == "onecycle":
        scheduler = OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, 
            steps_per_epoch=len(train_loader),
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos'
        )
    else:
        scheduler = None
    
    # ---- KL annealer ----
    kl_annealer = KLAnnealer(start_beta=beta_start, end_beta=beta_end, warmup_epochs=beta_warmup)
    
    # ---- early stopping ----
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Get current beta from annealer
        current_beta = kl_annealer.get_beta(epoch)
        
        # Training (with OneCycleLR stepping per batch)
        if lr_scheduler_type == "onecycle":
            # For OneCycleLR, we need to step after each batch
            model.train()
            total_loss, total_recon, total_kl, total_samples = 0.0, 0.0, 0.0, 0
            
            pbar = tqdm(train_loader, desc="train", leave=False)
            for batch in pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_image = batch["image"].to(device)

                # text embedding for conditional discriminator
                text_emb = model.encode_text(input_ids, attention_mask)

                recon_image, mu, logvar = model(input_ids, attention_mask, sample=True)

                # -------------------------
                # Discriminator step (hinge)
                # -------------------------
                if discriminator is not None:
                    discriminator.train()
                    d_optimizer.zero_grad(set_to_none=True)
                    real_logits = discriminator(target_image, text_emb.detach())
                    fake_logits = discriminator(recon_image.detach(), text_emb.detach())
                    d_loss = F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                    d_optimizer.step()

                # -------------------------
                # Generator step
                # -------------------------
                g_loss, recon_loss, kl_loss = vae_loss(
                    recon_image, target_image, mu, logvar,
                    perceptual_loss_fn=perceptual_loss_fn,
                    beta=current_beta,
                    perceptual_weight=perceptual_weight,
                    edge_weight=edge_weight,
                )

                if discriminator is not None:
                    discriminator.eval()
                    fake_logits_g = discriminator(recon_image, text_emb)
                    adv_loss = (-fake_logits_g).mean()
                    g_loss = g_loss + adv_weight * adv_loss

                optimizer.zero_grad(set_to_none=True)
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()  # Step OneCycleLR per batch

                batch_size_curr = input_ids.size(0)
                total_loss += g_loss.item() * batch_size_curr
                total_recon += recon_loss.item() * batch_size_curr
                total_kl += kl_loss.item() * batch_size_curr
                total_samples += batch_size_curr
                
                pbar.set_postfix(
                    loss=total_loss / total_samples,
                    recon=total_recon / total_samples,
                    kl=total_kl / total_samples
                )
            
            tr_loss = total_loss / max(total_samples, 1)
            tr_recon = total_recon / max(total_samples, 1)
            tr_kl = total_kl / max(total_samples, 1)
        else:
            tr_loss, tr_recon, tr_kl = run_epoch(
                model, train_loader, optimizer, device, current_beta, 
                perceptual_loss_fn, perceptual_weight, train=True
            )
        
        # Validation
        va_loss, va_recon, va_kl = run_epoch(
            model, val_loader, optimizer, device, current_beta, 
            perceptual_loss_fn, perceptual_weight, edge_weight=edge_weight, train=False
        )
        
        # Update learning rate scheduler (except OneCycleLR which steps per batch)
        if scheduler is not None and lr_scheduler_type != "onecycle":
            if lr_scheduler_type == "plateau":
                scheduler.step(va_loss)
            else:
                scheduler.step()
        
        # Log to history
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_recon"].append(tr_recon)
        history["train_kl"].append(tr_kl)
        history["val_loss"].append(va_loss)
        history["val_recon"].append(va_recon)
        history["val_kl"].append(va_kl)
        history["lr"].append(current_lr)
        history["beta"].append(current_beta)
        
        # Write to CSV
        with open(history_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, tr_recon, tr_kl, 
                           va_loss, va_recon, va_kl, current_lr, current_beta])
        
        print(f"Epoch {epoch:02d} | "
              f"train loss {tr_loss:.4f} (recon {tr_recon:.4f}, kl {tr_kl:.4f}) | "
              f"val loss {va_loss:.4f} (recon {va_recon:.4f}, kl {va_kl:.4f}) | "
              f"lr {current_lr:.2e} | beta {current_beta:.6f}")
        
        # Save checkpoint
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": va_loss,
            "max_len": max_len,
            "image_size": image_size,
            "backbone": backbone,
            "beta": current_beta,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))
        
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
            print("  -> saved best.pt")
        
        # Generate plots after each epoch
        plot_training_curves(history, run_dir)
        
        # Save sample reconstructions periodically (every epoch for first 10, then every N)
        if epoch <= 10 or epoch % save_samples_every == 0:
            save_sample_images(model, val_loader, device, run_dir, epoch)
        
        # Check early stopping
        if early_stopping(va_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}!")
            break
    
    # Save final sample reconstructions
    save_sample_images(model, val_loader, device, run_dir, epoch)
    
    print(f"\nTraining complete for {backbone}. Results saved to {run_dir}")
    print(f"Best validation loss: {best_val:.4f}")
    return model, history


def plot_training_curves(history, run_dir):
    """Generate and save loss and learning rate curves."""
    epochs = history["epoch"]
    
    # Combined plot: Total Loss + Reconstruction Loss
    plt.figure(figsize=(15, 5))
    
    # Total loss curves
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Total Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Reconstruction loss curves
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_recon"], label="Train Recon", marker="o")
    plt.plot(epochs, history["val_recon"], label="Val Recon", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (L1 + Perceptual)")
    plt.title("Reconstruction Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # KL divergence curves
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_kl"], label="Train KL", marker="o")
    plt.plot(epochs, history["val_kl"], label="Val KL", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Separate reconstruction loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_recon"], label="Train Recon", marker="o", linewidth=2)
    plt.plot(epochs, history["val_recon"], label="Val Recon", marker="s", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Reconstruction Loss", fontsize=12)
    plt.title("Reconstruction Loss Over Training", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(run_dir, "recon_loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Learning rate and beta curves
    if "lr" in history and history["lr"]:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(epochs, history["lr"], label="Learning Rate", marker="o", color="green", linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Learning Rate", fontsize=12)
        ax1.set_title("Learning Rate Schedule", fontsize=14)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(epochs, history["beta"], label="Beta (KL weight)", marker="o", color="purple", linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Beta", fontsize=12)
        ax2.set_title("KL Annealing Schedule", fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "lr_beta_curves.png"), dpi=150, bbox_inches="tight")
        plt.close()


def main():
    # ---- config ----
    root = "dataset"
    max_len = 256
    image_size = 128
    batch_size = 64
    epochs = 100  # More epochs since we have better loss
    lr = 1e-4  # Lower learning rate for stability
    
    # KL annealing: start at 0, gradually increase to small value
    beta_start = 0.0
    beta_end = 0.0001
    beta_warmup = 15  # Warmup over 15 epochs
    
    # Perceptual loss weight
    perceptual_weight = 0.1
    
    # Training improvements
    patience = 5  # More patience since improvements may be gradual
    lr_scheduler = "onecycle"  # OneCycleLR often works best
    save_samples_every = 5
    
    # Train only resnet18 first (faster to validate the fix)
    backbones = ["resnet18","resnet34","resnet50"]
    
    for backbone in backbones:
        train_backbone(
            backbone=backbone,
            root=root,
            max_len=max_len,
            image_size=image_size,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_warmup=beta_warmup,
            perceptual_weight=perceptual_weight,
            patience=patience,
            lr_scheduler_type=lr_scheduler,
            save_samples_every=save_samples_every,
        )
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
