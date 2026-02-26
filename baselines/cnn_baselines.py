
import torch
import torch.nn as nn
from models.encoders import CNNEncoder

class StandardCNNFusion(nn.Module):
    """
    Standard Fusion Baseline for CNN Backbones.
    
    Architecture:
    1. CXR CNN Encoder (ResNet/EfficientNet/MobileNet) -> Features
    2. Sputum CNN Encoder (ResNet/EfficientNet/MobileNet) -> Features
    3. Concatenation
    4. Two-Layer MLP Classifier
    
    This represents the 'standard way' to do multimodal fusion with CNNs,
    providing a strong baseline to compare MGM-TB-Net against.
    """
    def __init__(
        self,
        backbone='resnet50',
        embed_dim=256,
        num_classes=2,
        dropout=0.1
    ):
        super().__init__()
        
        # Encoders (Shared weights option could be added, but here independent)
        self.cxr_encoder = CNNEncoder(
            backbone_name=backbone,
            embed_dim=embed_dim,
            dropout=dropout,
            pretrained=False  # Explicitly disable pretraining per user request
        )
        
        self.sputum_encoder = CNNEncoder(
            backbone_name=backbone,
            embed_dim=embed_dim,
            dropout=dropout,
            pretrained=False  # Explicitly disable pretraining per user request
        )
        
        # Fusion Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, cxr, sputum):
        """
        Args:
            cxr: [B, 3, 224, 224]
            sputum: [B, 3, 224, 224]
        
        Returns:
            dict with 'logits'
        """
        # Feature Extraction
        cxr_feats = self.cxr_encoder(cxr)      # [B, 256]
        spt_feats = self.sputum_encoder(sputum)# [B, 256]
        
        # Concatenation
        combined = torch.cat([cxr_feats, spt_feats], dim=1) # [B, 512]
        
        # Classification
        logits = self.classifier(combined)
        
        return {'logits': logits}


class StandardCNNUnimodal(nn.Module):
    """
    Standard Unimodal Baseline for CNN Backbones.
    """
    def __init__(self, backbone='resnet50', embed_dim=256, num_classes=2, dropout=0.1):
        super().__init__()
        self.encoder = CNNEncoder(backbone_name=backbone, embed_dim=embed_dim, dropout=dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x, dummy=None):
        feats = self.encoder(x)
        logits = self.classifier(feats)
        return {'logits': logits}
