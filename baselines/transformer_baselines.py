"""
Baseline models for comparison
"""
import torch
import torch.nn as nn
from models.encoders import LightweightViT
from models.fusion import StandardCrossAttention
from models.fis import MamdaniFIS
from models.fusion import FMCA


class UnimodalModel(nn.Module):
    """
    Unimodal baseline (CXR-only or Sputum-only).
    Uses single encoder without fusion.
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_layers=4,
        embed_dim=256,
        num_heads=8,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.encoder = LightweightViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, img, dummy_img=None):
        """
        Args:
            img: Primary modality image
            dummy_img: Ignored (for interface compatibility)
        """
        features, uncertainty = self.encoder(img)
        logits = self.classifier(features)
        
        return {
            'logits': logits,
            'features': features,
            'uncertainty': uncertainty
        }


class ConcatFusion(nn.Module):
    """Late fusion via feature concatenation."""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_layers=4,
        embed_dim=256,
        num_heads=8,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.cxr_encoder = LightweightViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.sputum_encoder = LightweightViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Classifier takes concatenated features
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, num_classes)
        )
    
    def forward(self, cxr_img, sputum_img):
        cxr_feats, uncertainty_cxr = self.cxr_encoder(cxr_img)
        spt_feats, uncertainty_spt = self.sputum_encoder(sputum_img)
        
        # Concatenate features
        fused = torch.cat([cxr_feats, spt_feats], dim=1)
        logits = self.classifier(fused)
        
        return {
            'logits': logits,
            'cxr_features': cxr_feats,
            'sputum_features': spt_feats,
            'uncertainty_cxr': uncertainty_cxr,
            'uncertainty_sputum': uncertainty_spt
        }


class VanillaCMT(nn.Module):
    """Vanilla Cross-Modal Transformer (no gating)."""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_layers=4,
        embed_dim=256,
        num_heads=8,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()
        
        self.cxr_encoder = LightweightViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.sputum_encoder = LightweightViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Standard cross-attention (no gating)
        self.cross_attn = StandardCrossAttention(
            dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, cxr_img, sputum_img):
        cxr_feats, uncertainty_cxr = self.cxr_encoder(cxr_img)
        spt_feats, uncertainty_spt = self.sputum_encoder(sputum_img)
        
        # Cross-attention fusion (no gating)
        fused, attn = self.cross_attn(
            query=cxr_feats.unsqueeze(1),
            key=spt_feats.unsqueeze(1),
            value=spt_feats.unsqueeze(1)
        )
        fused = fused.squeeze(1)
        
        logits = self.classifier(fused)
        
        return {
            'logits': logits,
            'attention': attn,
            'cxr_features': cxr_feats,
            'sputum_features': spt_feats,
            'uncertainty_cxr': uncertainty_cxr,
            'uncertainty_sputum': uncertainty_spt
        }


class ScalarGateFusion(nn.Module):
    """Fusion with learnable scalar gate (non-fuzzy baseline)."""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_layers=4,
        embed_dim=256,
        num_heads=8,
        num_classes=2,
        dropout=0.1,
        gate_type='mlp'  # 'mlp' or 'sigmoid'
    ):
        super().__init__()
        
        self.cxr_encoder = LightweightViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.sputum_encoder = LightweightViT(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Learnable scalar gate
        if gate_type == 'mlp':
            self.gate = nn.Sequential(
                nn.Linear(2, 16),  # Input: [uncertainty_cxr, uncertainty_sputum]
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )
        else:  # sigmoid
            self.gate = nn.Sequential(
                nn.Linear(2, 1),
                nn.Sigmoid()
            )
        
        # FMCA with scalar gating
        self.fmca = FMCA(
            dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            modulation_type='logit'
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes)
        )
    
    def forward(self, cxr_img, sputum_img):
        cxr_feats, uncertainty_cxr = self.cxr_encoder(cxr_img)
        spt_feats, uncertainty_spt = self.sputum_encoder(sputum_img)
        
        # Compute scalar gate from uncertainties
        uncertainties = torch.stack([uncertainty_cxr, uncertainty_spt], dim=1)  # [B, 2]
        beta = self.gate(uncertainties).squeeze(1)  # [B]
        
        # Cross-attention with scalar gating
        fused, attn = self.fmca(
            query=cxr_feats.unsqueeze(1),
            key=spt_feats.unsqueeze(1),
            value=spt_feats.unsqueeze(1),
            alpha=beta
        )
        fused = fused.squeeze(1)
        
        logits = self.classifier(fused)
        
        return {
            'logits': logits,
            'beta': beta,
            'attention': attn,
            'cxr_features': cxr_feats,
            'sputum_features': spt_feats,
            'uncertainty_cxr': uncertainty_cxr,
            'uncertainty_sputum': uncertainty_spt
        }
