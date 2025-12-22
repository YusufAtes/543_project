import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from im_text_dataset import ArtCaptionDataset
from im_text_model import ImageCaptioner

def run_epoch(model, loader, optimizer, device, pad_token_id, train=True):
    if train:
        model.train()
    else:
        model.eval()

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
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

def main():
    # ---- config ----
    root = "dataset"
    tokenizer_name = "gpt2"
    max_len = 256
    image_size = 224
    batch_size = 16
    epochs = 20
    lr = 2e-4
    backbone = "resnet50"
    backbone_pretrained = True   # allowed: pretrained on ImageNet, but NOT a pretrained captioner

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("checkpoints", exist_ok=True)

    # ---- datasets ----
    train_ds = ArtCaptionDataset(root, "train", tokenizer_name, max_len, image_size)
    val_ds   = ArtCaptionDataset(root, "val",   tokenizer_name, max_len, image_size)

    pad_id = train_ds.tokenizer.pad_token_id

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ---- model ----
    vocab_size = len(train_ds.tokenizer)
    model = ImageCaptioner(
        vocab_size=vocab_size,
        max_len=max_len,
        backbone=backbone,
        backbone_pretrained=backbone_pretrained,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, optimizer, device, pad_id, train=True)
        va_loss, va_ppl = run_epoch(model, val_loader,   optimizer, device, pad_id, train=False)

        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} ppl {tr_ppl:.2f} | val loss {va_loss:.4f} ppl {va_ppl:.2f}")

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
        torch.save(ckpt, f"checkpoints/last.pt")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, f"checkpoints/best.pt")
            print("  -> saved best.pt")

if __name__ == "__main__":
    main()
