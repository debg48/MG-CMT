"""
Custom Lightweight Vision Transformer with 2D Rotary Positional Embeddings
Optimized for M4 Air (MPS backend support)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE2D(nn.Module):
    """
    2D Rotary Positional Embeddings for image patches.
    Extends RoPE to 2D by applying rotation in both x and y dimensions.
    
    This captures relative spatial relationships critical for medical imaging
    (e.g., "lesion is above the clavicle").
    """
    def __init__(self, dim, max_resolution=(14, 14)):
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D RoPE"
        
        # Split dimension for x and y rotations
        half_dim = dim // 4
        
        # Create frequency bands
        freqs = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer('freqs', freqs)
        
        # Precompute max position embeddings
        self.max_resolution = max_resolution
        
    def forward(self, x, positions=None):
        """
        Apply 2D rotary embeddings to input tensor.
        
        Args:
            x: [batch, num_patches, dim] - input features
            positions: [num_patches, 2] with (row, col) indices
                      If None, assumes grid layout
        
        Returns:
            x with rotary embeddings applied
        """
        batch_size, num_patches, dim = x.shape
        
        # Generate positions if not provided
        if positions is None:
            h, w = self.max_resolution
            row_pos = torch.arange(h, device=x.device).repeat_interleave(w)
            col_pos = torch.arange(w, device=x.device).repeat(h)
            positions = torch.stack([row_pos, col_pos], dim=1)  # [num_patches, 2]
        
        # Split features into 4 parts: [x_cos, x_sin, y_cos, y_sin]
        quarter_dim = dim // 4
        x_parts = x.reshape(batch_size, num_patches, 4, quarter_dim)
        
        # Compute angles for x and y dimensions
        row_pos = positions[:, 0].float()  # [num_patches]
        col_pos = positions[:, 1].float()  # [num_patches]
        
        # Create rotation matrices
        # For half-dimension pairs: rotate adjacent dimensions
        freqs = self.freqs[:quarter_dim // 2]  # Only use needed frequencies
        
        row_angles = row_pos.unsqueeze(1) * freqs.unsqueeze(0)  # [num_patches, half_dim/2]
        col_angles = col_pos.unsqueeze(1) * freqs.unsqueeze(0)  # [num_patches, half_dim/2]
        
        # Apply rotation to x-dimension components (first half)
        x_0, x_1 = x_parts[:, :, 0, :].chunk(2, dim=-1)  # Split into pairs
        x_rotated_0 = x_0 * torch.cos(row_angles).unsqueeze(0) - x_1 * torch.sin(row_angles).unsqueeze(0)
        x_rotated_1 = x_0 * torch.sin(row_angles).unsqueeze(0) + x_1 * torch.cos(row_angles).unsqueeze(0)
        x_parts[:, :, 0, :] = torch.cat([x_rotated_0, x_rotated_1], dim=-1)
        
        # Apply rotation to y-dimension components (second half)
        y_0, y_1 = x_parts[:, :, 2, :].chunk(2, dim=-1)
        y_rotated_0 = y_0 * torch.cos(col_angles).unsqueeze(0) - y_1 * torch.sin(col_angles).unsqueeze(0)
        y_rotated_1 = y_0 * torch.sin(col_angles).unsqueeze(0) + y_1 * torch.cos(col_angles).unsqueeze(0)
        x_parts[:, :, 2, :] = torch.cat([y_rotated_0, y_rotated_1], dim=-1)
        
        return x_parts.reshape(batch_size, num_patches, dim)


class MultiHeadAttentionWithRoPE(nn.Module):
    """Multi-head attention with integrated 2D-RoPE."""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, x, rope, positions):
        """
        Args:
            x: [batch, num_patches, dim]
            rope: RoPE2D module
            positions: [num_patches, 2] position indices
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, num_heads, N, head_dim]
        
        # Apply RoPE to Q and K (not V)
        # Reshape for RoPE: [B * num_heads, N, head_dim]
        q_rope = q.reshape(B * self.num_heads, N, self.head_dim)
        k_rope = k.reshape(B * self.num_heads, N, self.head_dim)
        
        q_rope = rope(q_rope, positions)
        k_rope = rope(k_rope, positions)
        
        # Reshape back: [B, num_heads, N, head_dim]
        q = q_rope.reshape(B, self.num_heads, N, self.head_dim)
        k = k_rope.reshape(B, self.num_heads, N, self.head_dim)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class MLP(nn.Module):
    """Feed-forward network."""
    def __init__(self, in_features, hidden_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with RoPE-augmented attention."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionWithRoPE(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        
    def forward(self, x, rope, positions):
        x = x + self.attn(self.norm1(x), rope, positions)
        x = x + self.mlp(self.norm2(x))
        return x


class LightweightViT(nn.Module):
    """
    Custom lightweight Vision Transformer with 2D-RoPE.
    
    Designed for small medical imaging datasets (~1000 images).
    Optimized for Apple M4 Air (MPS backend).
    
    Architecture:
    - 4-5 transformer layers (vs 12 in ViT-Base)
    - Embedding dim: 256 (vs 768 in ViT-Base)
    - 8 attention heads
    - Total params: ~5M (vs 86M in ViT-Base)
    """
    def __init__(
        self, 
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        
        self.img_size = img_size
        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding (convolutional)
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # 2D-RoPE (no learnable positional embeddings needed)
        self.rope = RoPE2D(
            dim=embed_dim, 
            max_resolution=(img_size // patch_size, img_size // patch_size)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Uncertainty estimation head (auxiliary classifier)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2),  # Binary logits for uncertainty
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for better training on small datasets."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: [batch, channels, height, width] - input images
        
        Returns:
            features: [batch, embed_dim] - global features
            uncertainty: [batch] - entropy-based uncertainty [0, 1]
        """
        # Patch embedding: [B, 3, 224, 224] -> [B, 256, 14, 14]
        x = self.patch_embed(x)
        
        # Flatten patches: [B, 256, 14, 14] -> [B, 196, 256]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Generate position indices for RoPE
        # Grid positions: (0,0), (0,1), ..., (H-1, W-1)
        row_pos = torch.arange(H, device=x.device).repeat_interleave(W)
        col_pos = torch.arange(W, device=x.device).repeat(H)
        positions = torch.stack([row_pos, col_pos], dim=1)  # [num_patches, 2]
        
        # Apply transformer blocks with RoPE
        for block in self.blocks:
            x = block(x, self.rope, positions)
        
        x = self.norm(x)
        
        # Global average pooling over patches
        features = x.mean(dim=1)  # [B, embed_dim]
        
        # Compute uncertainty via entropy
        logits = self.uncertainty_head(features)  # [B, 2]
        probs = F.softmax(logits, dim=-1)  # [B, 2]
        
        # Shannon entropy: H = -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [B]
        
        # Normalize to [0, 1] (max entropy for binary is log(2))
        uncertainty = entropy / math.log(2)  # [B]
        
        return features, uncertainty


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model on M4 Air (MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = LightweightViT(
        img_size=224,
        patch_size=16,
        embed_dim=256,
        num_layers=4,
        num_heads=8
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    features, uncertainty = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")
    print(f"Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
