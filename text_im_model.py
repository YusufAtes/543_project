import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def _imgnet_normalize_from_tanh(x: torch.Tensor) -> torch.Tensor:
    """
    Convert image tensor from [-1, 1] to ImageNet normalized space.
    Args:
        x: (B,3,H,W) in [-1,1]
    Returns:
        (B,3,H,W) normalized with ImageNet mean/std
    """
    x01 = (x + 1) / 2
    mean = x01.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x01.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x01 - mean) / std

def _gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    """
    Compute Gram matrix for style loss.
    Args:
        feat: (B, C, H, W)
    Returns:
        gram: (B, C, C)
    """
    B, C, H, W = feat.shape
    f = feat.view(B, C, H * W)
    gram = torch.bmm(f, f.transpose(1, 2)) / (C * H * W + 1e-8)
    return gram

def _sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """
    Approximate edge maps using Sobel filters.
    Args:
        x: (B,3,H,W) in [-1,1]
    Returns:
        edges: (B,1,H,W) in arbitrary scale
    """
    # convert to grayscale in [-1,1]
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    kx = x.new_tensor([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]]).view(1, 1, 3, 3)
    ky = x.new_tensor([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]]).view(1, 1, 3, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-8)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
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


class TextEncoder(nn.Module):
    """
    Transformer-based text encoder that converts tokenized captions to embeddings.
    Uses mean pooling to get a fixed-size global text representation.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2 vocab size
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 256,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Token embedding
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, input_ids, attention_mask):
        """
        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) 1 for valid tokens, 0 for padding
        Returns:
            text_emb: (B, d_model) global text embedding
        """
        # Embed tokens
        x = self.tok_emb(input_ids)  # (B, L, D)
        x = self.pos_enc(x)
        
        # Create padding mask for transformer (True = ignore)
        src_key_padding_mask = (attention_mask == 0)
        
        # Encode
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = self.norm(x)
        
        # Mean pooling over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        sum_embeddings = torch.sum(x * mask_expanded, dim=1)  # (B, D)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)  # (B, 1)
        text_emb = sum_embeddings / sum_mask  # (B, D)
        
        return text_emb


class ResidualBlock(nn.Module):
    """Residual block with optional conditional normalization."""
    
    def __init__(self, channels, cond_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.activation = nn.SiLU(inplace=True)
        
        # FiLM conditioning (Feature-wise Linear Modulation)
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.film_scale = nn.Linear(cond_dim, channels)
            self.film_shift = nn.Linear(cond_dim, channels)
    
    def forward(self, x, cond=None):
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)
        
        # Apply FiLM conditioning
        if cond is not None and self.cond_dim is not None:
            scale = self.film_scale(cond).unsqueeze(-1).unsqueeze(-1)
            shift = self.film_shift(cond).unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + scale) + shift
        
        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)
        
        return x + h


class UpsampleBlock(nn.Module):
    """Upsample block with residual connections and text conditioning."""
    
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        # Bilinear upsampling tends to blur; nearest + conv usually preserves sharper details.
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.res_block1 = ResidualBlock(out_channels, cond_dim)
        self.res_block2 = ResidualBlock(out_channels, cond_dim)
    
    def forward(self, x, cond):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.res_block1(x, cond)
        x = self.res_block2(x, cond)
        return x


class ImageDecoder(nn.Module):
    """
    Improved conditional VAE decoder with:
    - Residual blocks for better gradient flow
    - FiLM conditioning to inject text information at every layer
    - GroupNorm instead of BatchNorm (more stable)
    """
    
    def __init__(
        self,
        text_dim: int = 512,
        latent_dim: int = 512,  # Increased latent dimension
        base_channels: int = 512,
        image_size: int = 128,
    ):
        super().__init__()
        
        self.text_dim = text_dim
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        
        # Latent space parameters (for VAE)
        self.fc_mu = nn.Linear(text_dim, latent_dim)
        self.fc_logvar = nn.Linear(text_dim, latent_dim)
        
        # Conditioning dimension
        cond_dim = latent_dim + text_dim
        
        # Project [z; text_emb] to initial spatial feature map
        # Initial size: 4x4
        self.init_size = 4
        self.fc_decode = nn.Sequential(
            nn.Linear(cond_dim, base_channels * self.init_size * self.init_size),
            nn.SiLU(inplace=True),
        )
        
        # Progressive upsampling with residual blocks: 4 -> 8 -> 16 -> 32 -> 64 -> 128
        ch = base_channels
        self.up1 = UpsampleBlock(ch, ch, cond_dim)           # 4 -> 8
        self.up2 = UpsampleBlock(ch, ch // 2, cond_dim)      # 8 -> 16
        self.up3 = UpsampleBlock(ch // 2, ch // 4, cond_dim) # 16 -> 32
        self.up4 = UpsampleBlock(ch // 4, ch // 8, cond_dim) # 32 -> 64
        self.up5 = UpsampleBlock(ch // 8, ch // 16, cond_dim) # 64 -> 128
        
        # Final output layer
        self.final = nn.Sequential(
            nn.GroupNorm(8, ch // 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch // 16, 3, 3, padding=1),
            nn.Tanh(),  # Output in [-1, 1]
        )
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, text_emb, sample=True):
        """
        Args:
            text_emb: (B, text_dim) text embedding from encoder
            sample: If True, sample from latent space; if False, use mean
        Returns:
            image: (B, 3, 128, 128) generated image
            mu: (B, latent_dim) mean of latent distribution
            logvar: (B, latent_dim) log variance of latent distribution
        """
        B = text_emb.size(0)
        
        # Get latent distribution parameters
        mu = self.fc_mu(text_emb)
        logvar = self.fc_logvar(text_emb)
        
        # Sample or use mean
        if sample:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        
        # Concatenate z and text_emb as conditioning
        cond = torch.cat([z, text_emb], dim=1)  # (B, latent_dim + text_dim)
        
        # Project to spatial feature map
        x = self.fc_decode(cond)  # (B, base_channels * 4 * 4)
        x = x.view(B, self.base_channels, self.init_size, self.init_size)  # (B, C, 4, 4)
        
        # Progressive upsampling with conditioning
        x = self.up1(x, cond)  # 8x8
        x = self.up2(x, cond)  # 16x16
        x = self.up3(x, cond)  # 32x32
        x = self.up4(x, cond)  # 64x64
        x = self.up5(x, cond)  # 128x128
        
        # Final output
        image = self.final(x)  # (B, 3, 128, 128)
        
        return image, mu, logvar


class TextToImageVAE(nn.Module):
    """
    Complete Text-to-Image Conditional VAE model.
    Combines TextEncoder and ImageDecoder.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 256,
        latent_dim: int = 512,  # Increased from 256
        backbone: str = "resnet50",
    ):
        super().__init__()
        
        self.backbone = backbone
        self.text_dim = d_model
        
        # Determine base channels based on backbone
        if backbone == "resnet18":
            base_channels = 512
        elif backbone == "resnet34":
            base_channels = 512
        elif backbone == "resnet50":
            base_channels = 1024
        else:
            raise ValueError(f"backbone must be resnet18, resnet34, or resnet50, got {backbone}")
        
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_ff=dim_ff,
            dropout=dropout,
            max_len=max_len,
        )
        
        self.image_decoder = ImageDecoder(
            text_dim=d_model,
            latent_dim=latent_dim,
            base_channels=base_channels,
            image_size=128,
        )
    
    def forward(self, input_ids, attention_mask, sample=True):
        """
        Args:
            input_ids: (B, L) token ids
            attention_mask: (B, L) attention mask
            sample: If True, sample from latent space
        Returns:
            image: (B, 3, 128, 128) generated image
            mu: (B, latent_dim) mean
            logvar: (B, latent_dim) log variance
        """
        # Encode text
        text_emb = self.text_encoder(input_ids, attention_mask)
        
        # Decode to image
        image, mu, logvar = self.image_decoder(text_emb, sample=sample)
        
        return image, mu, logvar

    def encode_text(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)
    
    @torch.no_grad()
    def generate(self, input_ids, attention_mask, sample=False):
        """
        Generate images from tokenized captions (inference mode).
        """
        self.eval()
        image, _, _ = self.forward(input_ids, attention_mask, sample=sample)
        return image


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    This loss compares high-level features instead of raw pixels,
    which helps generate sharper, more realistic images.
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Extract features at different layers
        # These layers capture different levels of abstraction
        self.blocks = nn.ModuleList([
            vgg[:4].eval(),   # relu1_2 - low-level features (edges, colors)
            vgg[4:9].eval(),  # relu2_2 - textures
            vgg[9:18].eval(), # relu3_4 - patterns
            vgg[18:27].eval(), # relu4_4 - semantic content
        ])
        
        # Freeze VGG weights
        for block in self.blocks:
            for param in block.parameters():
                param.requires_grad = False
        
        # VGG normalization (ImageNet stats)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        # Layer weights (higher weight for higher-level features)
        self.layer_weights = [1.0, 1.0, 1.0, 1.0]
    
    def normalize(self, x):
        """Convert from [-1, 1] to VGG input range."""
        # First convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Then normalize with ImageNet stats
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        """
        Compute perceptual loss between predicted and target images.
        
        Args:
            pred: (B, 3, H, W) predicted image in [-1, 1]
            target: (B, 3, H, W) target image in [-1, 1]
        Returns:
            loss: Scalar perceptual loss
        """
        pred = self.normalize(pred)
        target = self.normalize(target)
        
        loss = 0.0
        for i, block in enumerate(self.blocks):
            pred = block(pred)
            target = block(target)
            loss += self.layer_weights[i] * F.l1_loss(pred, target)
        
        return loss


class ResNetPerceptualLoss(nn.Module):
    """
    Perceptual + (optional) style loss using ResNet feature maps.
    This is cheaper than VGG19 and also satisfies "use resnet backbones".
    """
    def __init__(self, backbone: str = "resnet18", style_layers: bool = True):
        super().__init__()

        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"backbone must be resnet18/resnet34/resnet50, got {backbone}")

        # feature stages
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool).eval()
        self.layer1 = net.layer1.eval()
        self.layer2 = net.layer2.eval()
        self.layer3 = net.layer3.eval()
        self.layer4 = net.layer4.eval()

        for m in [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]:
            for p in m.parameters():
                p.requires_grad = False

        self.style_layers = style_layers
        self.layer_weights = [1.0, 1.0, 1.0, 1.0]  # l1..l4

    def forward(self, pred, target):
        """
        Args:
            pred/target: (B,3,H,W) in [-1,1]
        Returns:
            feat_loss: scalar
            style_loss: scalar (0 if disabled)
        """
        pred_n = _imgnet_normalize_from_tanh(pred)
        tgt_n = _imgnet_normalize_from_tanh(target)

        feat_loss = pred.new_zeros(())
        style_loss = pred.new_zeros(())

        # stage 0: after maxpool
        p = self.stem(pred_n)
        t = self.stem(tgt_n)

        for w, layer in zip(self.layer_weights, [self.layer1, self.layer2, self.layer3, self.layer4]):
            p = layer(p)
            t = layer(t)
            feat_loss = feat_loss + w * F.l1_loss(p, t)
            if self.style_layers:
                style_loss = style_loss + w * F.l1_loss(_gram_matrix(p), _gram_matrix(t))

        return feat_loss, style_loss


class TextImageDiscriminator(nn.Module):
    """
    Conditional projection discriminator using a ResNet backbone.

    D(x, y) = u(h(x)) + <v(y), h(x)>
    where h(x) is image feature vector, v(y) projects text embedding to same dim.
    """
    def __init__(self, text_dim: int = 512, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feat_dim = 512
        elif backbone == "resnet34":
            net = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            feat_dim = 512
        elif backbone == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            feat_dim = 2048
        else:
            raise ValueError(f"backbone must be resnet18/resnet34/resnet50, got {backbone}")

        # keep conv stem + layers; remove classifier head
        self.features = nn.Sequential(*list(net.children())[:-1])  # (B,feat_dim,1,1)

        # projection discriminator head
        self.img_linear = nn.Linear(feat_dim, 1)
        self.txt_proj = nn.Linear(text_dim, feat_dim, bias=False)

    def forward(self, images, text_emb):
        """
        Args:
            images: (B,3,H,W) in [-1,1]
            text_emb: (B,text_dim)
        Returns:
            logits: (B,)
        """
        x = _imgnet_normalize_from_tanh(images)
        h = self.features(x).flatten(1)  # (B,feat_dim)
        out = self.img_linear(h).squeeze(1)
        proj = self.txt_proj(text_emb)
        out = out + torch.sum(h * proj, dim=1)
        return out


def vae_loss(
    recon_image,
    target_image,
    mu,
    logvar,
    perceptual_loss_fn=None,
    beta=0.0001,
    perceptual_weight=0.1,
    edge_weight: float = 0.0,
):
    """
    Improved VAE loss combining:
    - L1 reconstruction loss (sharper than MSE)
    - Perceptual loss (semantic similarity)
    - KL divergence (regularization)
    - Optional edge loss (helps reduce blur)
    
    Args:
        recon_image: (B, 3, H, W) reconstructed image
        target_image: (B, 3, H, W) target image
        mu: (B, latent_dim) mean of latent distribution
        logvar: (B, latent_dim) log variance of latent distribution
        perceptual_loss_fn: PerceptualLoss module (optional)
        beta: Weight for KL divergence term
        perceptual_weight: Weight for perceptual loss
    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss (L1)
        kl_loss: KL divergence loss
    """
    # L1 reconstruction loss (better than MSE for sharp images)
    recon_loss = F.l1_loss(recon_image, target_image, reduction='mean')

    # Edge loss (Sobel) encourages sharper boundaries
    edge_loss = recon_image.new_zeros(())
    if edge_weight and edge_weight > 0:
        edge_pred = _sobel_edges(recon_image)
        edge_tgt = _sobel_edges(target_image)
        edge_loss = F.l1_loss(edge_pred, edge_tgt)
        recon_loss = recon_loss + edge_weight * edge_loss
    
    # Perceptual loss (if provided)
    if perceptual_loss_fn is not None:
        # Allow either VGG perceptual (returns scalar) OR ResNet perceptual (returns tuple)
        percep_out = perceptual_loss_fn(recon_image, target_image)
        if isinstance(percep_out, tuple):
            percep_loss, style_loss = percep_out
            recon_loss = recon_loss + perceptual_weight * percep_loss + (perceptual_weight * 0.25) * style_loss
        else:
            percep_loss = percep_out
            recon_loss = recon_loss + perceptual_weight * percep_loss
    
    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


if __name__ == "__main__":
    # Quick test
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test perceptual loss
    print("Testing PerceptualLoss...")
    perceptual_loss_fn = PerceptualLoss().to(device)
    dummy_pred = torch.randn(2, 3, 128, 128).to(device)
    dummy_target = torch.randn(2, 3, 128, 128).to(device)
    p_loss = perceptual_loss_fn(dummy_pred, dummy_target)
    print(f"  Perceptual loss: {p_loss.item():.4f}")
    
    # Test each backbone
    for backbone in ["resnet18", "resnet34", "resnet50"]:
        print(f"\nTesting {backbone}...")
        model = TextToImageVAE(backbone=backbone).to(device)
        
        # Dummy input
        batch_size = 4
        seq_len = 256
        input_ids = torch.randint(0, 50257, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones(batch_size, seq_len).to(device)
        target_image = torch.randn(batch_size, 3, 128, 128).to(device)
        
        # Forward pass
        recon_image, mu, logvar = model(input_ids, attention_mask)
        print(f"  Output shape: {recon_image.shape}")
        print(f"  Mu shape: {mu.shape}")
        print(f"  Logvar shape: {logvar.shape}")
        
        # Compute loss
        loss, recon_loss, kl_loss = vae_loss(
            recon_image, target_image, mu, logvar, 
            perceptual_loss_fn=perceptual_loss_fn
        )
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Recon loss: {recon_loss.item():.4f}")
        print(f"  KL loss: {kl_loss.item():.4f}")
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
    
    print("\nAll backbone tests passed!")
