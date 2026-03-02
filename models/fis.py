"""
Differentiable Mamdani Fuzzy Inference System
Converts uncertainty metrics to fuzzy gating scalars using true Mamdani inference
with Gaussian fuzzy consequents and Centroid-of-Area defuzzification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MamdaniFIS(nn.Module):
    """
    True Differentiable Mamdani Fuzzy Inference System.
    
    Unlike a Takagi-Sugeno-Kang (TSK) system which uses singleton/polynomial
    consequents, a Mamdani FIS uses fuzzy sets for both antecedents AND consequents.
    
    Architecture:
    1. Fuzzification: Maps crisp uncertainty values to fuzzy sets via Gaussian MFs
    2. Inference: Computes rule firing strengths (product t-norm)
    3. Implication: Clips each consequent fuzzy set by its firing strength (soft-min)
    4. Aggregation: Combines implied fuzzy sets via fuzzy OR (soft-max)
    5. Defuzzification: Centroid-of-Area over the aggregated output surface
    
    Anti-Semantic-Collapse: A monotonicity regularization loss ensures that the
    output MF centers preserve their linguistic ordering (Low Uncertainty → High
    Confidence, High Uncertainty → Low Confidence).
    """
    def __init__(self, num_inputs=2, num_membership_funcs=3, num_output_points=100,
                 softmax_temp=10.0):
        """
        Args:
            num_inputs: Number of uncertainty inputs (2 for CXR + Sputum)
            num_membership_funcs: Number of fuzzy sets per input (Low, Medium, High)
            num_output_points: Discretization resolution for CoA defuzzification
            softmax_temp: Temperature for soft-min/max approximations (higher = harder)
        """
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_mf = num_membership_funcs
        self.num_rules = num_membership_funcs ** num_inputs  # 3x3 = 9
        self.num_output_points = num_output_points
        self.softmax_temp = softmax_temp
        
        # ─── Antecedent (Input) Membership Functions ────────────────────
        # Learnable Gaussian MFs: 3 per input (Low / Medium / High uncertainty)
        init_means = torch.linspace(0.0, 1.0, num_membership_funcs)
        init_std = torch.full((num_membership_funcs,), 0.2)
        
        self.mu_cxr = nn.Parameter(init_means.clone())       # [3]
        self.sigma_cxr = nn.Parameter(init_std.clone())       # [3]
        
        self.mu_sputum = nn.Parameter(init_means.clone())     # [3]
        self.sigma_sputum = nn.Parameter(init_std.clone())    # [3]
        
        # ─── Consequent (Output) Membership Functions ───────────────────
        # Each of the 9 rules has a Gaussian output MF in [0, 1]
        # Rule index: (CXR_MF, SPT_MF) → 0=(L,L), 1=(L,M), 2=(L,H),
        #             3=(M,L), 4=(M,M), 5=(M,H), 6=(H,L), 7=(H,M), 8=(H,H)
        
        # Alpha (CXR confidence): High when CXR uncertainty is Low
        init_out_mu_alpha = torch.tensor([
            0.9, 0.9, 0.9,   # CXR Low Unc  → High Confidence
            0.5, 0.5, 0.5,   # CXR Med Unc  → Med Confidence
            0.1, 0.1, 0.1    # CXR High Unc → Low Confidence
        ])
        
        # Beta (Sputum confidence): High when Sputum uncertainty is Low
        init_out_mu_beta = torch.tensor([
            0.9, 0.5, 0.1,   # SPT Low/Med/High Unc (CXR Low)
            0.9, 0.5, 0.1,   # SPT Low/Med/High Unc (CXR Med)
            0.9, 0.5, 0.1    # SPT Low/Med/High Unc (CXR High)
        ])
        
        init_out_sigma = torch.full((self.num_rules,), 0.15)
        
        self.output_mu_alpha = nn.Parameter(init_out_mu_alpha)      # [9]
        self.output_sigma_alpha = nn.Parameter(init_out_sigma.clone())  # [9]
        
        self.output_mu_beta = nn.Parameter(init_out_mu_beta)        # [9]
        self.output_sigma_beta = nn.Parameter(init_out_sigma.clone())   # [9]
        
        # ─── Discretized output universe for CoA ────────────────────────
        # Fixed grid over [0, 1] — not a parameter, just a buffer
        self.register_buffer(
            'output_universe',
            torch.linspace(0.0, 1.0, num_output_points)  # [N]
        )
    
    def gaussian_membership(self, x, mu, sigma):
        """
        Gaussian membership function.
        
        Args:
            x: [batch] - input values
            mu: [K] - means
            sigma: [K] - standard deviations
        
        Returns:
            membership: [batch, K] - membership degrees
        """
        x = x.unsqueeze(1)       # [B, 1]
        mu = mu.unsqueeze(0)     # [1, K]
        sigma = sigma.unsqueeze(0)  # [1, K]
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)  # [B, K]
    
    def _soft_min(self, a, b):
        """
        Differentiable approximation of min(a, b) using negative-temperature logsumexp.
        soft_min(a, b) ≈ -1/T * log(exp(-T*a) + exp(-T*b))
        
        Args:
            a: [batch, num_rules, N]  (firing strengths expanded)
            b: [batch, num_rules, N]  (output MF values)
        Returns:
            [batch, num_rules, N]
        """
        T = self.softmax_temp
        stacked = torch.stack([a, b], dim=-1)  # [B, R, N, 2]
        return -torch.logsumexp(-T * stacked, dim=-1) / T  # [B, R, N]
    
    def _soft_max(self, x, dim):
        """
        Differentiable approximation of max over a dimension using logsumexp.
        
        Args:
            x: tensor
            dim: dimension to take max over
        Returns:
            soft max over dim
        """
        T = self.softmax_temp
        return torch.logsumexp(T * x, dim=dim) / T
    
    def _defuzzify_coa(self, firing_strengths, output_mu, output_sigma):
        """
        Full Mamdani defuzzification via Centroid-of-Area.
        
        Steps:
        1. Evaluate each rule's consequent Gaussian over the output universe
        2. Apply Mamdani implication: clip each consequent by its firing strength (soft-min)
        3. Aggregate all implied sets via fuzzy OR (soft-max)
        4. Compute centroid of the aggregated surface
        
        Args:
            firing_strengths: [B, 9] - rule firing strengths
            output_mu: [9] - centers of consequent Gaussians
            output_sigma: [9] - widths of consequent Gaussians
        
        Returns:
            crisp_output: [B] - defuzzified value in [0, 1]
        """
        B = firing_strengths.shape[0]
        N = self.num_output_points
        R = self.num_rules
        
        # 1. Evaluate consequent MFs over the output universe
        # output_universe: [N], output_mu: [R], output_sigma: [R]
        y = self.output_universe.unsqueeze(0)          # [1, N]
        mu = output_mu.unsqueeze(1)                    # [R, 1]
        sigma = torch.clamp(output_sigma, min=0.02).unsqueeze(1)  # [R, 1]
        
        # consequent_mf: [R, N] — the shape of each rule's output fuzzy set
        consequent_mf = torch.exp(-0.5 * ((y - mu) / sigma) ** 2)  # [R, N]
        
        # Expand for batch: [B, R, N]
        consequent_mf = consequent_mf.unsqueeze(0).expand(B, -1, -1)
        
        # 2. Mamdani Implication: clip each consequent by its firing strength
        # firing_strengths: [B, R] → [B, R, 1] for broadcasting
        fs_expanded = firing_strengths.unsqueeze(2).expand(-1, -1, N)  # [B, R, N]
        
        # implied[k](y) = min(firing_strength_k, consequent_mf_k(y))
        implied = self._soft_min(fs_expanded, consequent_mf)  # [B, R, N]
        
        # 3. Aggregation: fuzzy OR (max) across all rules
        # aggregated(y) = max_k(implied[k](y))
        aggregated = self._soft_max(implied, dim=1)  # [B, N]
        
        # 4. Centroid-of-Area defuzzification
        # output = ∫ y · aggregated(y) dy  /  ∫ aggregated(y) dy
        y_grid = self.output_universe.unsqueeze(0)  # [1, N]
        
        numerator = (y_grid * aggregated).sum(dim=1)       # [B]
        denominator = aggregated.sum(dim=1) + 1e-8         # [B]
        
        crisp_output = numerator / denominator  # [B]
        
        return crisp_output
    
    def forward(self, uncertainty_cxr, uncertainty_sputum):
        """
        Args:
            uncertainty_cxr: [batch] - CXR uncertainty values [0, 1]
            uncertainty_sputum: [batch] - Sputum uncertainty values [0, 1]
        
        Returns:
            alpha: [batch] - CXR confidence (high when CXR is reliable)
            beta: [batch] - Sputum confidence (high when Sputum is reliable)
        """
        # ── Step 1: Fuzzification ──────────────────────────────────────
        mu_cxr = self.gaussian_membership(
            uncertainty_cxr,
            self.mu_cxr,
            torch.clamp(self.sigma_cxr, min=0.05)
        )  # [B, 3]
        
        mu_sputum = self.gaussian_membership(
            uncertainty_sputum,
            self.mu_sputum,
            torch.clamp(self.sigma_sputum, min=0.05)
        )  # [B, 3]
        
        # ── Step 2: Rule Inference (product t-norm) ────────────────────
        firing_strengths = []
        for i in range(self.num_mf):        # CXR fuzzy set index
            for j in range(self.num_mf):    # Sputum fuzzy set index
                strength = mu_cxr[:, i] * mu_sputum[:, j]  # [B]
                firing_strengths.append(strength)
        
        firing_strengths = torch.stack(firing_strengths, dim=1)  # [B, 9]
        
        # Normalize firing strengths
        firing_sum = firing_strengths.sum(dim=1, keepdim=True) + 1e-8
        firing_strengths = firing_strengths / firing_sum  # [B, 9]
        
        # ── Steps 3-5: Implication + Aggregation + CoA Defuzzification ─
        alpha = self._defuzzify_coa(
            firing_strengths, self.output_mu_alpha, self.output_sigma_alpha
        )  # [B]
        
        beta = self._defuzzify_coa(
            firing_strengths, self.output_mu_beta, self.output_sigma_beta
        )  # [B]
        
        # Clamp for numerical stability
        alpha = torch.clamp(alpha, 0.0, 1.0)
        beta = torch.clamp(beta, 0.0, 1.0)
        
        return alpha, beta
    
    def monotonicity_loss(self):
        """
        Regularization loss that penalizes semantic collapse.
        
        Ensures the linguistic ordering of consequent centers is preserved:
        - Alpha: Low CXR uncertainty → High confidence (rules 0-2 > rules 6-8)
        - Beta:  Low SPT uncertainty → High confidence (rules 0,3,6 > rules 2,5,8)
        
        Returns:
            loss: scalar ≥ 0 (zero when ordering is fully satisfied)
        """
        # Alpha: rules 0-2 (Low CXR unc) should have HIGHER centers than rules 6-8 (High CXR unc)
        # Penalize when high-unc rules have centers ≥ low-unc rules
        alpha_violations = F.relu(self.output_mu_alpha[6:9] - self.output_mu_alpha[0:3])
        
        # Beta: rules {0,3,6} (Low SPT unc) should have HIGHER centers than {2,5,8} (High SPT unc)
        low_spt_rules = self.output_mu_beta[torch.tensor([0, 3, 6])]
        high_spt_rules = self.output_mu_beta[torch.tensor([2, 5, 8])]
        beta_violations = F.relu(high_spt_rules - low_spt_rules)
        
        return alpha_violations.sum() + beta_violations.sum()
    
    def get_membership_functions(self):
        """Return learned membership function parameters for visualization."""
        return {
            'cxr_input': {
                'mu': self.mu_cxr.detach().cpu(),
                'sigma': torch.clamp(self.sigma_cxr, min=0.05).detach().cpu()
            },
            'sputum_input': {
                'mu': self.mu_sputum.detach().cpu(),
                'sigma': torch.clamp(self.sigma_sputum, min=0.05).detach().cpu()
            },
            'alpha_output': {
                'mu': self.output_mu_alpha.detach().cpu(),
                'sigma': torch.clamp(self.output_sigma_alpha, min=0.02).detach().cpu()
            },
            'beta_output': {
                'mu': self.output_mu_beta.detach().cpu(),
                'sigma': torch.clamp(self.output_sigma_beta, min=0.02).detach().cpu()
            }
        }


if __name__ == "__main__":
    # Test FIS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    fis = MamdaniFIS(num_inputs=2, num_membership_funcs=3).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in fis.parameters())
    print(f"FIS parameters: {num_params}")
    
    # Test with dummy uncertainties
    uncertainty_cxr = torch.tensor([0.1, 0.5, 0.9]).to(device)
    uncertainty_sputum = torch.tensor([0.2, 0.6, 0.95]).to(device)
    
    alpha, beta = fis(uncertainty_cxr, uncertainty_sputum)
    
    print(f"\nUncertainties (CXR):    {uncertainty_cxr}")
    print(f"Uncertainties (Sputum): {uncertainty_sputum}")
    print(f"Alpha (CXR confidence): {alpha}")
    print(f"Beta (Sputum confidence): {beta}")
    print(f"\nExpected behavior:")
    print(f"  Low uncertainty  -> High confidence (α or β close to 1)")
    print(f"  High uncertainty -> Low confidence (α or β close to 0)")
    
    # Test monotonicity loss
    mono_loss = fis.monotonicity_loss()
    print(f"\nMonotonicity loss (should be 0 with default init): {mono_loss.item():.6f}")
    
    # Test gradient flow
    alpha.sum().backward()
    grads_ok = all(p.grad is not None for p in fis.parameters())
    print(f"Gradient flow OK: {grads_ok}")
