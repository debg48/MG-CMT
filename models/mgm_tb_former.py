"""
Mamdani-Gated Cross-Modal Transformer (MG-CMT)
Complete architecture integrating all components
"""
import torch
import torch.nn as nn

from .encoders import LightweightViT
from .fis import MamdaniFIS
from .fusion import FMCA


class MGMTBFormer(nn.Module):
    """
    MGM-TB-Former: Mamdani-Gated Multimodal Transformer for Robust TB Detection.
    
    Architecture:
    1. Custom lightweight ViT encoders (4 layers, 256-dim) with 2D-RoPE
    2. Differentiable Mamdani FIS for uncertainty-aware gating
    3. Gated Residual Fusion (FMCA) for robustness (Fallback mechanism)
    4. Binary classification head
    
    Total Parameters: ~10.5M
    Optimized for M4 Air (MPS backend)
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_transformer_layers=4,
        embed_dim=256,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=2,  # TB vs. Normal
        fmca_modulation='logit'  # 'logit' or 'post'
    ):
        super().__init__()
        
        # Two independent encoders (no weight sharing)
        self.cxr_encoder = LightweightViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        self.sputum_encoder = LightweightViT(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout
        )
        
        # Fuzzy Inference System
        self.fis = MamdaniFIS(num_inputs=2, num_membership_funcs=3)
        
        # Fuzzy-Modulated Cross-Attention
        self.fmca = FMCA(
            dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            modulation_type=fmca_modulation
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
        
        self.num_classes = num_classes
        
    def forward(self, cxr_img, sputum_img, return_attention=False):
        """
        Args:
            cxr_img: [batch, 3, 224, 224] - CXR images
            sputum_img: [batch, 3, 224, 224] - Sputum microscopy images
            return_attention: Whether to return attention maps
        
        Returns:
            Dictionary containing:
            - logits: [batch, num_classes]
            - alpha: [batch] - CXR confidence
            - beta: [batch] - Sputum confidence (gating value)
            - uncertainty_cxr: [batch]
            - uncertainty_sputum: [batch]
            - attention: [batch, num_heads, 1, 1] (if return_attention=True)
        """
        # 1. Encode both modalities
        cxr_feats, uncertainty_cxr = self.cxr_encoder(cxr_img)  # [B, 256], [B]
        spt_feats, uncertainty_spt = self.sputum_encoder(sputum_img)  # [B, 256], [B]
        
        # 2. Fuzzy gating: Map uncertainties to confidence scalars
        alpha, beta = self.fis(uncertainty_cxr, uncertainty_spt)  # [B], [B]
        
        # 3. Cross-modal fusion with fuzzy modulation
        # Query = CXR (primary modality)
        # Key/Value = Sputum (confirmatory modality)
        # β gates how much the CXR attends to Sputum
        fused_feats, attn_map = self.fmca(
            query=cxr_feats.unsqueeze(1),      # [B, 1, 256]
            key=spt_feats.unsqueeze(1),        # [B, 1, 256]
            value=spt_feats.unsqueeze(1),      # [B, 1, 256]
            alpha=beta                         # [B] - gate sputum influence
        )
        
        fused_feats = fused_feats.squeeze(1)  # [B, 256]
        
        # CRITICAL: Add residual connection from CXR
        # This allows the model to fall back to CXR-only when sputum is unreliable
        # fused = CXR_features + beta * cross_attention(CXR, Sputum)
        # Note: beta is already applied inside FMCA if modulation='post'
        # So we just add the residual here for fallback capability
        fused_feats = cxr_feats + fused_feats  # Residual connection
        
        # 4. Classification
        logits = self.classifier(fused_feats)  # [B, num_classes]
        
        # Prepare output dictionary
        output = {
            'logits': logits,
            'alpha': alpha,                    # CXR confidence
            'beta': beta,                      # Sputum confidence (gating value)
            'uncertainty_cxr': uncertainty_cxr,
            'uncertainty_sputum': uncertainty_spt,
            'cxr_features': cxr_feats,
            'sputum_features': spt_feats,
            'fused_features': fused_feats
        }
        
        if return_attention:
            output['attention'] = attn_map
        
        return output


def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total


if __name__ == "__main__":
    # Test full MGM-TB-Former model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    model = MGMTBFormer(
        img_size=224,
        patch_size=16,
        num_transformer_layers=4,
        embed_dim=256,
        num_heads=8,
        num_classes=2
    ).to(device)
    
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Component breakdown
    cxr_params = count_parameters(model.cxr_encoder)
    spt_params = count_parameters(model.sputum_encoder)
    fis_params = count_parameters(model.fis)
    fmca_params = count_parameters(model.fmca)
    classifier_params = count_parameters(model.classifier)
    
    print(f"\nParameter breakdown:")
    print(f"  CXR Encoder:    {cxr_params:,}")
    print(f"  Sputum Encoder: {spt_params:,}")
    print(f"  FIS:            {fis_params:,}")
    print(f"  FMCA:           {fmca_params:,}")
    print(f"  Classifier:     {classifier_params:,}")
    
    # Test forward pass
    print(f"\nTesting forward pass...")
    batch_size = 2
    cxr_imgs = torch.randn(batch_size, 3, 224, 224).to(device)
    sputum_imgs = torch.randn(batch_size, 3, 224, 224).to(device)
    
    with torch.no_grad():
        outputs = model(cxr_imgs, sputum_imgs, return_attention=True)
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if torch.is_tensor(value):
            print(f"  {key:20s}: {tuple(value.shape)}")
    
    print(f"\nGating values:")
    print(f"  Alpha (CXR conf):    {outputs['alpha']}")
    print(f"  Beta (Sputum conf):  {outputs['beta']}")
    print(f"  Uncertainty (CXR):   {outputs['uncertainty_cxr']}")
    print(f"  Uncertainty (Spt):   {outputs['uncertainty_sputum']}")
    
    print(f"\n[OK] Model successfully initialized and tested on {device}!")
