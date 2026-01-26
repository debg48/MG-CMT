"""
Differentiable Mamdani Fuzzy Inference System
Converts uncertainty metrics to fuzzy gating scalars
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MamdaniFIS(nn.Module):
    """
    Learnable Mamdani Fuzzy Inference System.
    
    Architecture:
    1. Fuzzification: Maps crisp uncertainty values to fuzzy sets (Low, Medium, High)
    2. Inference: Computes rule firing strengths
    3. Defuzzification: Aggregates rules to produce confidence scalars (α, β)
    
    Key Innovation: All operations are differentiable, allowing end-to-end training.
    """
    def __init__(self, num_inputs=2, num_membership_funcs=3):
        """
        Args:
            num_inputs: Number of uncertainty inputs (2 for CXR + Sputum)
            num_membership_funcs: Number of fuzzy sets per input (Low, Medium, High)
        """
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_membership_funcs = num_membership_funcs
        
        # Learnable Gaussian membership function parameters
        # Initialize with reasonable spread across [0, 1]
        init_means = torch.linspace(0.0, 1.0, num_membership_funcs)
        init_std = torch.full((num_membership_funcs,), 0.2)
        
        # Parameters for each input modality
        self.mu_cxr = nn.Parameter(init_means.clone())      # [3]
        self.sigma_cxr = nn.Parameter(init_std.clone())     # [3]
        
        self.mu_sputum = nn.Parameter(init_means.clone())   # [3]
        self.sigma_sputum = nn.Parameter(init_std.clone())  # [3]
        
        # Learnable rule consequents (output fuzzy sets)
        # For each rule, define the output membership function
        # Learnable rule consequents (output fuzzy sets)
        # Initialize with logic priors:
        # Rule index maps to: (CXR_MF, SPT_MF)
        # 0=(L,L), 1=(L,M), 2=(L,H), 3=(M,L), 4=(M,M), 5=(M,H), 6=(H,L), 7=(H,M), 8=(H,H)
        
        # Alpha (Trust in CXR): High when CXR is Low Uncertainty (indices 0,1,2)
        # Beta (Trust in Sputum): High when Sputum is Low Uncertainty (indices 0,3,6)
        
        # Logic: If Uncertainty is Low -> Trust, If High -> Distrust
        init_alpha = torch.tensor([
            0.9, 0.9, 0.9,  # CXR Low Unc -> High Trust
            0.5, 0.5, 0.5,  # CXR Med Unc -> Med Trust
            0.1, 0.1, 0.1   # CXR High Unc -> Low Trust
        ])
        
        init_beta = torch.tensor([
            0.9, 0.5, 0.1,
            0.9, 0.5, 0.1,
            0.9, 0.5, 0.1
        ])
        
        self.rule_outputs_alpha = nn.Parameter(init_alpha)
        self.rule_outputs_beta = nn.Parameter(init_beta)
        
    def gaussian_membership(self, x, mu, sigma):
        """
        Gaussian membership function.
        
        Args:
            x: [batch] - input values
            mu: [num_funcs] - means
            sigma: [num_funcs] - standard deviations
        
        Returns:
            membership: [batch, num_funcs] - membership degrees
        """
        x = x.unsqueeze(1)  # [batch, 1]
        mu = mu.unsqueeze(0)  # [1, num_funcs]
        sigma = sigma.unsqueeze(0)  # [1, num_funcs]
        
        # Gaussian: exp(-0.5 * ((x - mu) / sigma)^2)
        membership = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
        
        return membership  # [batch, num_funcs]
    
    def forward(self, uncertainty_cxr, uncertainty_sputum):
        """
        Args:
            uncertainty_cxr: [batch] - CXR uncertainty values [0, 1]
            uncertainty_sputum: [batch] - Sputum uncertainty values [0, 1]
        
        Returns:
            alpha: [batch] - CXR confidence (high when CXR is reliable)
            beta: [batch] - Sputum confidence (high when Sputum is reliable)
        """
        batch_size = uncertainty_cxr.shape[0]
        
        # Step 1: Fuzzification
        # Convert crisp inputs to fuzzy membership degrees
        mu_cxr = self.gaussian_membership(
            uncertainty_cxr, 
            self.mu_cxr, 
            torch.clamp(self.sigma_cxr, min=0.05)  # Prevent collapse
        )  # [batch, 3]
        
        mu_sputum = self.gaussian_membership(
            uncertainty_sputum, 
            self.mu_sputum, 
            torch.clamp(self.sigma_sputum, min=0.05)
        )  # [batch, 3]
        
        # Step 2: Rule Inference
        # Compute firing strength for each rule using t-norm (min operator)
        # Rules: IF (CXR is X) AND (Sputum is Y) THEN ...
        
        firing_strengths = []
        for i in range(self.num_membership_funcs):  # CXR fuzzy set
            for j in range(self.num_membership_funcs):  # Sputum fuzzy set
                # AND operator: min(mu_cxr[i], mu_sputum[j])
                strength = torch.min(
                    mu_cxr[:, i], 
                    mu_sputum[:, j]
                )  # [batch]
                firing_strengths.append(strength)
        
        firing_strengths = torch.stack(firing_strengths, dim=1)  # [batch, 9]
        
        # Step 3: Defuzzification (Weighted Average)
        # α = sum(firing_strength * rule_output) / sum(firing_strength)
        
        # Normalize firing strengths (avoid divide by zero)
        firing_sum = firing_strengths.sum(dim=1, keepdim=True) + 1e-8  # [batch, 1]
        normalized_strengths = firing_strengths / firing_sum  # [batch, 9]
        
        # Compute output confidence
        alpha = (normalized_strengths * self.rule_outputs_alpha.unsqueeze(0)).sum(dim=1)  # [batch]
        beta = (normalized_strengths * self.rule_outputs_beta.unsqueeze(0)).sum(dim=1)   # [batch]
        
        # Clamp to [0, 1] for stability
        alpha = torch.sigmoid(alpha)  # Ensure [0, 1]
        beta = torch.sigmoid(beta)
        
        return alpha, beta
    
    def get_membership_functions(self):
        """Return learned membership function parameters for visualization."""
        return {
            'cxr': {
                'mu': self.mu_cxr.detach().cpu(),
                'sigma': torch.clamp(self.sigma_cxr, min=0.05).detach().cpu()
            },
            'sputum': {
                'mu': self.mu_sputum.detach().cpu(),
                'sigma': torch.clamp(self.sigma_sputum, min=0.05).detach().cpu()
            }
        }


if __name__ == "__main__":
    # Test FIS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    fis = MamdaniFIS(num_inputs=2, num_membership_funcs=3).to(device)
    
    # Test with dummy uncertainties
    uncertainty_cxr = torch.tensor([0.1, 0.5, 0.9]).to(device)
    uncertainty_sputum = torch.tensor([0.2, 0.6, 0.95]).to(device)
    
    alpha, beta = fis(uncertainty_cxr, uncertainty_sputum)
    
    print("Uncertainties (CXR):", uncertainty_cxr)
    print("Uncertainties (Sputum):", uncertainty_sputum)
    print("Alpha (CXR confidence):", alpha)
    print("Beta (Sputum confidence):", beta)
    print("\nExpected behavior:")
    print("  Low uncertainty -> High confidence (α or β close to 1)")
    print("  High uncertainty -> Low confidence (α or β close to 0)")
