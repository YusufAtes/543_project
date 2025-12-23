import math
import torch
import torch.nn as nn
import torchvision.models as models

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (B, L, D)
        L = x.size(1)
        return x + self.pe[:, :L, :]

def generate_square_subsequent_mask(sz: int, device):
    # causal mask for decoder self-attention
    mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
    return mask  # True means "blocked" for nn.Transformer modules if used as key_padding? We’ll use float mask below.

class ImageCaptioner(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 256,
        backbone: str = "resnet50",
        backbone_pretrained: bool = True,
    ):
        super().__init__()

        # ---- CNN encoder (feature map -> tokens) ----
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if backbone_pretrained else None)
            enc_dim = 512
        elif backbone == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if backbone_pretrained else None)
            enc_dim = 512
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if backbone_pretrained else None)
            enc_dim = 2048
        else:
            raise ValueError("backbone must be resnet18, resnet34, or resnet50")

        self.cnn = nn.Sequential(*list(net.children())[:-2])  # up to conv5 feature map (B, C, H, W)
        self.enc_proj = nn.Linear(enc_dim, d_model)

        # ---- Token embedding + positional encoding ----
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        # ---- Transformer decoder ----
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def encode_image(self, images):
        # images: (B,3,H,W)
        feat = self.cnn(images)              # (B,C,h,w)
        B, C, h, w = feat.shape
        feat = feat.flatten(2).transpose(1, 2)  # (B, h*w, C)
        mem = self.enc_proj(feat)              # (B, S, D)  S=h*w
        return mem

    def forward(self, images, decoder_in, dec_pad_mask):
        """
        images: (B,3,H,W)
        decoder_in: (B,L) token ids (shifted)
        dec_pad_mask: (B,L) bool, True where padding
        """
        device = images.device
        mem = self.encode_image(images)  # (B,S,D)

        x = self.tok_emb(decoder_in)     # (B,L,D)
        x = self.pos_enc(x)

        L = x.size(1)
        # nn.TransformerDecoder expects a float attn mask with -inf for blocked positions
        causal = torch.triu(torch.ones(L, L, device=device), diagonal=1)
        tgt_mask = causal.masked_fill(causal == 1, float("-inf"))

        out = self.decoder(
            tgt=x,
            memory=mem,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=dec_pad_mask,   # (B,L) True = ignore
            memory_key_padding_mask=None,
        )
        logits = self.lm_head(out)  # (B,L,V)
        return logits

    @torch.no_grad()
    def generate(self, images, tokenizer, max_new_tokens=200):
        self.eval()
        device = images.device
        mem = self.encode_image(images)

        # start with BOS-like token: for GPT2 we use eos as start
        start_id = tokenizer.eos_token_id
        cur = torch.full((images.size(0), 1), start_id, dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            x = self.tok_emb(cur)
            x = self.pos_enc(x)
            L = x.size(1)
            causal = torch.triu(torch.ones(L, L, device=device), diagonal=1)
            tgt_mask = causal.masked_fill(causal == 1, float("-inf"))

            out = self.decoder(tgt=x, memory=mem, tgt_mask=tgt_mask)
            logits = self.lm_head(out[:, -1, :])  # last token (B,V)
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            cur = torch.cat([cur, next_id], dim=1)

        # decode
        texts = []
        for i in range(cur.size(0)):
            texts.append(tokenizer.decode(cur[i].tolist(), skip_special_tokens=True))
        return texts
