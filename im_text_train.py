import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
import matplotlib.pyplot as plt

from im_text_dataset import ArtCaptionDataset
from im_text_model import ImageCaptioner


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

def run_epoch(model, loader, optimizer, device, pad_token_id, loss_fn, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_tokens = 0.0, 0

    pbar = tqdm(loader, desc="train" if train else "val", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        decoder_in = batch["decoder_in"].to(device)
        targets = batch["targets"].to(device)
        dec_pad_mask = batch["dec_pad_mask"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(images, decoder_in, dec_pad_mask)  # (B,L,V)
            B, L, V = logits.shape
            loss = loss_fn(logits.reshape(B * L, V), targets.reshape(B * L))

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        # token count excludes padding
        nonpad = (targets != pad_token_id).sum().item()
        total_loss += loss.item() * nonpad
        total_tokens += nonpad
        ppl = torch.exp(torch.tensor(total_loss / max(total_tokens, 1))).item()
        pbar.set_postfix(loss=(total_loss / max(total_tokens, 1)), ppl=ppl)

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl

def train_backbone(backbone, root="dataset", tokenizer_name="gpt2", max_len=256, image_size=224,
                   batch_size=16, epochs=20, lr=2e-4, backbone_pretrained=True,
                   patience=5, label_smoothing=0.1, lr_scheduler_type="plateau"):
    """
    Train a single backbone model with logging, plotting, early stopping, and LR scheduling.
    
    Args:
        backbone: CNN backbone name (resnet18, resnet34, resnet50)
        patience: Early stopping patience (stop if val loss doesn't improve for N epochs)
        label_smoothing: Label smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
        lr_scheduler_type: "plateau" (ReduceLROnPlateau) or "cosine" (CosineAnnealingWarmRestarts)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    
    # Create directories for this backbone
    run_dir = f"runs/im2text_{backbone}"
    ckpt_dir = f"checkpoints/im2text_{backbone}"
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # CSV logging setup (with learning rate column)
    history_file = os.path.join(run_dir, "history.csv")
    with open(history_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_ppl", "val_loss", "val_ppl", "lr"])
    
    # Track history for plotting
    history = {"epoch": [], "train_loss": [], "train_ppl": [], "val_loss": [], "val_ppl": [], "lr": []}
    
    print(f"\n{'='*60}")
    print(f"Training {backbone}")
    print(f"  - LR: {lr}, Scheduler: {lr_scheduler_type}")
    print(f"  - Label smoothing: {label_smoothing}")
    print(f"  - Early stopping patience: {patience}")
    print(f"{'='*60}")
    
    # ---- datasets ----
    train_ds = ArtCaptionDataset(root, "train", tokenizer_name, max_len, image_size)
    val_ds   = ArtCaptionDataset(root, "val",   tokenizer_name, max_len, image_size)
    
    pad_id = train_ds.tokenizer.pad_token_id
    
    # num_workers=0 avoids slow multiprocessing startup on Windows
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # ---- model ----
    vocab_size = len(train_ds.tokenizer)
    model = ImageCaptioner(
        vocab_size=vocab_size,
        max_len=max_len,
        backbone=backbone,
        backbone_pretrained=backbone_pretrained,
    ).to(device)
    
    # ---- loss function with label smoothing ----
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=label_smoothing)
    
    # ---- optimizer ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # ---- learning rate scheduler ----
    if lr_scheduler_type == "plateau":
        # Reduce LR when validation loss plateaus
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, 
            verbose=True, min_lr=1e-6
        )
    elif lr_scheduler_type == "cosine":
        # Cosine annealing with warm restarts
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-6)
    else:
        scheduler = None
    
    # ---- early stopping ----
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        tr_loss, tr_ppl = run_epoch(model, train_loader, optimizer, device, pad_id, loss_fn, train=True)
        va_loss, va_ppl = run_epoch(model, val_loader,   optimizer, device, pad_id, loss_fn, train=False)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if lr_scheduler_type == "plateau":
                scheduler.step(va_loss)
            else:
                scheduler.step()
        
        # Log to history
        history["epoch"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_ppl"].append(tr_ppl)
        history["val_loss"].append(va_loss)
        history["val_ppl"].append(va_ppl)
        history["lr"].append(current_lr)
        
        # Write to CSV
        with open(history_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, tr_ppl, va_loss, va_ppl, current_lr])
        
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} ppl {tr_ppl:.2f} | val loss {va_loss:.4f} ppl {va_ppl:.2f} | lr {current_lr:.2e}")
        
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": va_loss,
            "tokenizer_name": tokenizer_name,
            "max_len": max_len,
            "image_size": image_size,
            "backbone": backbone,
            "backbone_pretrained": backbone_pretrained,
        }
        torch.save(ckpt, os.path.join(ckpt_dir, "last.pt"))
        
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, os.path.join(ckpt_dir, "best.pt"))
            print("  -> saved best.pt")
        
        # Generate plots after each epoch
        plot_training_curves(history, run_dir)
        
        # Check early stopping
        if early_stopping(va_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}!")
            break
    
    print(f"\nTraining complete for {backbone}. Results saved to {run_dir}")
    print(f"Best validation loss: {best_val:.4f}")
    return model, history

def plot_training_curves(history, run_dir):
    """Generate and save loss, perplexity, and learning rate curves."""
    epochs = history["epoch"]
    
    # Combined plot: Loss + Perplexity
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Perplexity curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_ppl"], label="Train PPL", marker="o")
    plt.plot(epochs, history["val_ppl"], label="Val PPL", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.title("Training and Validation Perplexity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "loss_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Separate perplexity plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_ppl"], label="Train PPL", marker="o", linewidth=2)
    plt.plot(epochs, history["val_ppl"], label="Val PPL", marker="s", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.title("Training and Validation Perplexity", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(run_dir, "ppl_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Learning rate curve
    if "lr" in history and history["lr"]:
        plt.figure(figsize=(8, 4))
        plt.plot(epochs, history["lr"], label="Learning Rate", marker="o", color="green", linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Learning Rate", fontsize=12)
        plt.title("Learning Rate Schedule", fontsize=14)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(run_dir, "lr_curve.png"), dpi=150, bbox_inches="tight")
        plt.close()

def main():
    # ---- config ----
    root = "dataset"
    tokenizer_name = "gpt2"
    max_len = 256
    image_size = 224
    batch_size = 16
    epochs = 50  # Increased max epochs (early stopping will prevent overfitting)
    lr = 2e-4
    backbone_pretrained = True   # allowed: pretrained on ImageNet, but NOT a pretrained captioner
    
    # Training improvements
    patience = 5           # Early stopping: stop if val loss doesn't improve for 5 epochs
    label_smoothing = 0.1  # Label smoothing to prevent overconfident predictions
    lr_scheduler = "plateau"  # "plateau" or "cosine"
    
    # Train all three backbones
    backbones = ["resnet18", "resnet34", "resnet50"]
    
    for backbone in backbones:
        train_backbone(
            backbone=backbone,
            root=root,
            tokenizer_name=tokenizer_name,
            max_len=max_len,
            image_size=image_size,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            backbone_pretrained=backbone_pretrained,
            patience=patience,
            label_smoothing=label_smoothing,
            lr_scheduler_type=lr_scheduler,
        )
    
    print(f"\n{'='*60}")
    print("All backbones trained successfully!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
