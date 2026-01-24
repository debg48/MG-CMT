"""
Fuzzy-Modulated Cross-Attention (FMCA)
Cross-modal attention with fuzzy gating to prevent feature wash-out
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FMCA(nn.Module):
    """
    Fuzzy-Modulated Cross-Attention.
    
    Standard cross-attention:
        Attention = softmax(QK^T / sqrt(d)) V
    
    FMCA (logit scaling):
        Attention = softmax((QK^T · α) / sqrt(d)) V
    
    Where α is the fuzzy gating scalar from the FIS.
    Scaling BEFORE softmax provides stronger, non-linear gating.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1, modulation_type='logit'):
        """
        Args:
            dim: Feature dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            modulation_type: 'logit' (scale before softmax) or 'post' (scale after)
        """
        super().__init__()
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert modulation_type in ['logit', 'post'], "modulation_type must be 'logit' or 'post'"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.modulation_type = modulation_type
        
        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, query, key, value, alpha):
        """
        Args:
            query: [batch, num_queries, dim] - Query features (from CXR)
            key: [batch, num_keys, dim] - Key features (from Sputum)
            value: [batch, num_keys, dim] - Value features (from Sputum)
            alpha: [batch, 1] or [batch] - Fuzzy gating scalar (β from FIS)
        
        Returns:
            output: [batch, num_queries, dim] - Fused features
            attn_weights: [batch, num_heads, num_queries, num_keys] - Attention map
        """
        B, Nq, C = query.shape
        _, Nk, _ = key.shape
        
        # Ensure alpha has correct shape
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(1)  # [batch, 1]
        
        # Project to Q, K, V
        q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        # q, k, v: [batch, num_heads, num_patches, head_dim]
        
        # Compute attention logits
        logits = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, Nq, Nk]
        
        # Apply fuzzy modulation
        if self.modulation_type == 'logit':
            # Scale BEFORE softmax (stronger gating effect)
            # Expand alpha to match logits shape
            alpha_expanded = alpha.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 1]
            logits = logits * alpha_expanded
            attn = F.softmax(logits, dim=-1)
        else:  # 'post'
            # Scale AFTER softmax (weaker gating)
            attn = F.softmax(logits, dim=-1)
            alpha_expanded = alpha.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, 1]
            attn = attn * alpha_expanded
        
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)  # [B, Nq, C]
        
        # Final projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out, attn


class StandardCrossAttention(nn.Module):
    """
    Standard cross-attention without fuzzy modulation.
    Used as a baseline for ablation studies.
    """
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        """Standard cross-attention (no gating)."""
        B, Nq, C = query.shape
        _, Nk, _ = key.shape
        
        q = self.q_proj(query).reshape(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        
        logits = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(logits, dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out, attn


if __name__ == "__main__":
    # Test FMCA
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    fmca = FMCA(dim=256, num_heads=8, modulation_type='logit').to(device)
    
    # Dummy features
    query = torch.randn(2, 1, 256).to(device)  # CXR features
    key = torch.randn(2, 1, 256).to(device)    # Sputum features
    value = torch.randn(2, 1, 256).to(device)
    
    # Fuzzy gating scalars
    alpha_high = torch.tensor([0.9, 0.9]).to(device)  # Trust sputum
    alpha_low = torch.tensor([0.1, 0.1]).to(device)   # Don't trust sputum
    
    # Test with high confidence
    out_high, attn_high = fmca(query, key, value, alpha_high)
    print("High α (trust sputum):")
    print(f"  Output norm: {out_high.norm():.3f}")
    print(f"  Attention norm: {attn_high.norm():.3f}")
    
    # Test with low confidence
    out_low, attn_low = fmca(query, key, value, alpha_low)
    print("\nLow α (ignore sputum):")
    print(f"  Output norm: {out_low.norm():.3f}")
    print(f"  Attention norm: {attn_low.norm():.3f}")
    
    print("\nExpected: Low α should produce smaller output (gating effect)")
