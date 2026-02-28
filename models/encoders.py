"""
Custom Lightweight Vision Transformer with 2D Rotary Positional Embeddings
Optimized for M4 Air (MPS backend support)
Includes Convolutional Stem and Stochastic Depth for small data regimes.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class RoPE2D(nn.Module):
    """
    2D Rotary Positional Embeddings for image patches.
    Extends RoPE to 2D by applying rotation in both x and y dimensions.
    """
    def __init__(self, dim, max_resolution=(14, 14)):
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "Dimension must be divisible by 4 for 2D RoPE"
        
        half_dim = dim // 4
        freqs = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer('freqs', freqs)
        self.max_resolution = max_resolution
        
    def forward(self, x, positions=None):
        batch_size, num_patches, dim = x.shape
        
        if positions is None:
            h, w = self.max_resolution
            row_pos = torch.arange(h, device=x.device).repeat_interleave(w)
            col_pos = torch.arange(w, device=x.device).repeat(h)
            positions = torch.stack([row_pos, col_pos], dim=1)
        
        quarter_dim = dim // 4
        x_parts = x.reshape(batch_size, num_patches, 4, quarter_dim)
        
        row_pos = positions[:, 0].float()
        col_pos = positions[:, 1].float()
        
        freqs = self.freqs[:quarter_dim // 2]
        
        row_angles = row_pos.unsqueeze(1) * freqs.unsqueeze(0)
        col_angles = col_pos.unsqueeze(1) * freqs.unsqueeze(0)
        
        x_0, x_1 = x_parts[:, :, 0, :].chunk(2, dim=-1)
        x_rotated_0 = x_0 * torch.cos(row_angles).unsqueeze(0) - x_1 * torch.sin(row_angles).unsqueeze(0)
        x_rotated_1 = x_0 * torch.sin(row_angles).unsqueeze(0) + x_1 * torch.cos(row_angles).unsqueeze(0)
        x_parts[:, :, 0, :] = torch.cat([x_rotated_0, x_rotated_1], dim=-1)
        
        y_0, y_1 = x_parts[:, :, 2, :].chunk(2, dim=-1)
        y_rotated_0 = y_0 * torch.cos(col_angles).unsqueeze(0) - y_1 * torch.sin(col_angles).unsqueeze(0)
        y_rotated_1 = y_0 * torch.sin(col_angles).unsqueeze(0) + y_1 * torch.cos(col_angles).unsqueeze(0)
        x_parts[:, :, 2, :] = torch.cat([y_rotated_0, y_rotated_1], dim=-1)
        
        return x_parts.reshape(batch_size, num_patches, dim)


class ConvolutionalStem(nn.Module):
    """
    3-Layer Convolutional Stem for early feature extraction.
    Replaces simple Patch Embedding to introduce inductive bias.
    """
    def __init__(self, in_channels=3, embed_dim=256, patch_size=16):
        super().__init__()
        # Target downsampling factor = patch_size (typically 16)
        # 16 = 2 * 2 * 2 * 2 (4 strides of 2)
        # We need to map 224x224 -> 14x14
        
        self.conv1 = nn.Conv2d(in_channels, embed_dim // 4, kernel_size=3, stride=2, padding=1)
        self.norm1 = nn.BatchNorm2d(embed_dim // 4)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(embed_dim // 4, embed_dim // 2, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.BatchNorm2d(embed_dim // 2)
        self.act2 = nn.ReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.BatchNorm2d(embed_dim)
        self.act3 = nn.ReLU(inplace=True)
        
        # Last downsampling (if patch_size is 16, we need one more 2x downsample)
        # Current stride: 2*2*2 = 8. Need 16.
        self.conv4 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.LayerNorm(embed_dim) # LayerNorm for Transformer input
        
    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        x = self.act3(self.norm3(self.conv3(x)))
        x = self.conv4(x)
        
        # [B, C, H, W] -> [B, H, W, C] for LayerNorm
        x = x.permute(0, 2, 3, 1)
        x = self.norm4(x)
        # [B, H, W, C] -> [B, C, H, W]
        x = x.permute(0, 3, 1, 2)
        
        return x


class MultiHeadAttentionWithRoPE(nn.Module):
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
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q_rope = q.reshape(B * self.num_heads, N, self.head_dim)
        k_rope = k.reshape(B * self.num_heads, N, self.head_dim)
        
        q_rope = rope(q_rope, positions)
        k_rope = rope(k_rope, positions)
        
        q = q_rope.reshape(B, self.num_heads, N, self.head_dim)
        k = k_rope.reshape(B, self.num_heads, N, self.head_dim)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
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
    """Transformer block with RoPE and DropPath."""
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionWithRoPE(dim, num_heads, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        
    def forward(self, x, rope, positions):
        x = x + self.drop_path(self.attn(self.norm1(x), rope, positions))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class LightweightViT(nn.Module):
    """
    Custom lightweight Vision Transformer with Convolutional Stem and 2D-RoPE.
    Optimized for small medical imaging datasets.
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
        dropout=0.1,
        drop_path_rate=0.1  # Stochastic depth rate
    ):
        super().__init__()
        
        assert img_size % patch_size == 0
        
        self.img_size = img_size
        self.patch_size = patch_size
        
        # 1. Convolutional Stem (replaces linear patch embed)
        self.stem = ConvolutionalStem(in_channels, embed_dim, patch_size)
        
        # 2. 2D-RoPE
        self.rope = RoPE2D(
            dim=embed_dim, 
            max_resolution=(img_size // patch_size, img_size // patch_size)
        )
        
        # 3. Transformer blocks with DropPath
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, drop_path=dpr[i])
            for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 4. Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 2),
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Apply Convolutional Stem -> [B, 256, 14, 14]
        x = self.stem(x)
        
        # Flatten patches -> [B, 196, 256]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        
        # Generate position indices for RoPE
        row_pos = torch.arange(H, device=x.device).repeat_interleave(W)
        col_pos = torch.arange(W, device=x.device).repeat(H)
        positions = torch.stack([row_pos, col_pos], dim=1)
        
        # Apply transformers
        for block in self.blocks:
            x = block(x, self.rope, positions)
        
        x = self.norm(x)
        
        # Global Pooling
        features = x.mean(dim=1)
        
        # Uncertainty
        logits = self.uncertainty_head(features)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        uncertainty = entropy / math.log(2)
        
        return features, uncertainty



import torchvision.models as models

class CNNEncoder(nn.Module):
    """
    Wrapper for standard CNN backbones (ResNet, EfficientNet, MobileNet).
    Adapts them to the MG-CMT interface (features + uncertainty).
    """
    def __init__(self, backbone_name='resnet50', in_channels=3, embed_dim=256, dropout=0.1, pretrained=False):
        super().__init__()
        
        # Load backbone
        weights = 'DEFAULT' if pretrained else None
        
        if backbone_name == 'resnet50':
            base_model = models.resnet50(weights=weights)
            self.feature_dim = base_model.fc.in_features
            # Remove FC
            self.features = nn.Sequential(*list(base_model.children())[:-1])
            
        elif backbone_name == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights=weights)
            self.feature_dim = base_model.classifier[1].in_features
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            
        elif backbone_name == 'efficientnet_v2_s':
            base_model = models.efficientnet_v2_s(weights=weights)
            self.feature_dim = base_model.classifier[1].in_features
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            
        elif backbone_name == 'densenet121':
            base_model = models.densenet121(weights=weights)
            self.feature_dim = base_model.classifier.in_features
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            
        elif backbone_name == 'mobilenet_v2':
            base_model = models.mobilenet_v2(weights=weights)
            self.feature_dim = base_model.classifier[1].in_features
            self.features = base_model.features
            self.pool = nn.AdaptiveAvgPool2d(1)
            
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")
            
        self.backbone_name = backbone_name
        
        
        # Projection to shared embedding dim
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Returns:
            features: [batch, embed_dim]
        """
        x = self.features(x)
        
        # Handle pooling differences
        if self.backbone_name in ['efficientnet_b0', 'efficientnet_v2_s', 'mobilenet_v2']:
            x = self.pool(x)
        elif self.backbone_name in ['densenet121']:
            x = F.relu(x, inplace=True)
            x = self.pool(x)
            
        x = torch.flatten(x, 1)  # [B, feature_dim]
        features = self.proj(x)  # [B, embed_dim]
        
        return features


class TransformerEncoder(nn.Module):
    """
    Wrapper for standard Transformer backbones (Swin, ViT, CvT).
    Adapts them to the MG-CMT interface (extract features).
    """
    def __init__(self, backbone_name='swin_tiny', embed_dim=256, dropout=0.1, pretrained=False):
        super().__init__()
        
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm to use Transformer backbones: pip install timm")
            
        self.backbone_name = backbone_name
        
        # Mapping custom names to timm names
        timm_names = {
            'vit_tiny': 'vit_tiny_patch16_224',
            'swin_tiny': 'swin_tiny_patch4_window7_224',
        }

        if backbone_name == 'levit_tiny':
            # LeViT-128s with depth=(1,1,2) = 4 total attention blocks
            self.model = timm.create_model(
                'levit_128s',
                pretrained=pretrained,
                num_classes=0,
                depth=(1, 1, 2)
            )
        else:
            if backbone_name not in timm_names:
                raise ValueError(f"Backbone {backbone_name} not supported. Try one of {list(timm_names.keys()) + ['levit_tiny']}")
            self.model = timm.create_model(
                timm_names[backbone_name],
                pretrained=pretrained,
                num_classes=0
            )

        self.feature_dim = self.model.num_features

        # Projection to shared embedding dim
        self.proj = nn.Sequential(
            nn.Linear(self.feature_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        """
        Returns:
            features: [batch, embed_dim]
        """
        features = self.model(x)  # [B, feature_dim]
        features = self.proj(features)  # [B, embed_dim]
        
        return features

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test Encoders
    print("\n--- ViT with Conv Stem ---")
    vit = LightweightViT(img_size=224, embed_dim=256, num_layers=4).to(device)
    print(f"ViT params: {count_parameters(vit):,}")
    
    print("\n--- CNN Backbones ---")
    for bb in ['resnet50', 'efficientnet_v2_s']:
        cnn = CNNEncoder(backbone_name=bb, embed_dim=256).to(device)
        print(f"{bb}: {count_parameters(cnn):,} params")


