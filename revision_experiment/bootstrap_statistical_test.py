"""
Bootstrap Statistical Test for TB Detection (Reviewer #3)
Calculates 95% Confidence Intervals via Bootstrapping and
performs McNemar's Test between proposed model and a comparative baseline.
"""
import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# Add project root to sys.path to resolve imports when run directly
project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Try imports, fallback to dummy for mock/dry-runs
try:
    from data.dataset import get_dataloaders
    from train import create_model
except ImportError:
    get_dataloaders = None
    create_model = None


def compute_metrics(y_true, y_pred, y_probs=None):
    """Compute standard metrics for binary classification."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Sensitivity and Specificity
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        metrics['sensitivity'] = metrics['recall']
        metrics['specificity'] = 0.0
        
    if y_probs is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_probs)
        except:
            metrics['auc_roc'] = 0.0
    else:
        metrics['auc_roc'] = 0.0
        
    return metrics


def run_evaluation(model, dataloader, device, modality='both'):
    """Evaluate model on the test loader and return true labels, predictions, and probabilities."""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for cxr, sputum, labels, _ in tqdm(dataloader, desc="Evaluating Model"):
            cxr = cxr.to(device)
            sputum = sputum.to(device)
            labels = labels.to(device)
            
            if modality == 'cxr':
                outputs = model(cxr, None)
            elif modality == 'sputum':
                outputs = model(sputum, None)
            else:
                outputs = model(cxr, sputum)
                
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def bootstrap_ci(y_true, y_pred, y_probs=None, B=1000, ci_level=0.95, seed=42):
    """Compute bootstrap confidence intervals for standard metrics."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    
    metrics_list = []
    for _ in tqdm(range(B), desc="Bootstrapping CIs"):
        # Resample indices with replacement
        indices = rng.choice(n, size=n, replace=True)
        boot_y_true = y_true[indices]
        boot_y_pred = y_pred[indices]
        boot_y_probs = y_probs[indices] if y_probs is not None else None
        
        try:
            m = compute_metrics(boot_y_true, boot_y_pred, boot_y_probs)
            metrics_list.append(m)
        except Exception:
            continue
            
    # Calculate bounds
    lower_pct = (1.0 - ci_level) / 2.0 * 100
    upper_pct = (1.0 + ci_level) / 2.0 * 100
    
    ci_results = {}
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'sensitivity', 'specificity', 'auc_roc']
    
    # Calculate actual metric
    actual_metrics = compute_metrics(y_true, y_pred, y_probs)
    
    for key in metric_keys:
        vals = [m[key] for m in metrics_list if key in m]
        if len(vals) > 0:
            ci_results[key] = {
                'actual': actual_metrics[key],
                'mean': np.mean(vals),
                'lower': np.percentile(vals, lower_pct),
                'upper': np.percentile(vals, upper_pct),
                'boot_vals': vals
            }
            
    return ci_results


def mcnemar_significance_test(y_true, y_pred_proposed, y_pred_baseline):
    """
    Perform McNemar's statistical significance test between two models.
    Handles exact binomial test for small discordant counts (< 25).
    """
    y_true = np.array(y_true)
    y_pred_proposed = np.array(y_pred_proposed)
    y_pred_baseline = np.array(y_pred_baseline)
    
    # Contingency Table:
    #                     Baseline Correct    Baseline Incorrect
    # Proposed Correct           n00                  n01
    # Proposed Incorrect         n10                  n11
    n00 = n01 = n10 = n11 = 0
    for t, p_prop, p_base in zip(y_true, y_pred_proposed, y_pred_baseline):
        c_prop = (p_prop == t)
        c_base = (p_base == t)
        if c_prop and c_base:
            n00 += 1
        elif c_prop and not c_base:
            n01 += 1
        elif not c_prop and c_base:
            n10 += 1
        else:
            n11 += 1
            
    discordant_sum = n01 + n10
    
    if discordant_sum == 0:
        p_value = 1.0
        stat = 0.0
        test_type = "Exact Binomial"
    elif discordant_sum < 25:
        # Exact Binomial Test (highly recommended for small sample sizes)
        from scipy.stats import binom
        k = min(n01, n10)
        p_value = 2.0 * binom.cdf(k, discordant_sum, 0.5)
        p_value = min(1.0, p_value)
        stat = discordant_sum
        test_type = "Exact Binomial (Discordant < 25)"
    else:
        # Chi-Squared Test with Yates' continuity correction
        from scipy.stats import chi2
        stat = ((abs(n01 - n10) - 1.0) ** 2) / discordant_sum
        p_value = chi2.sf(stat, 1)
        test_type = "Asymptotic Chi-Square (with Yates' correction)"
        
    return {
        'table': [[n00, n01], [n10, n11]],
        'statistic': stat,
        'p_value': p_value,
        'test_type': test_type,
        'discordant_count': discordant_sum,
        'proposed_only_correct': n01,
        'baseline_only_correct': n10
    }


def plot_bootstrap_distributions(ci_results, model_name, save_path):
    """Plot histograms of the boot distributions with 95% CI bands for a single model."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    keys_to_plot = ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc_roc']
    colors = ['#1f77b4', '#8c564b', '#e377c2', '#ff7f0e', '#d62728', '#9467bd']
    
    for idx, (key, color) in enumerate(zip(keys_to_plot, colors)):
        if key not in ci_results:
            continue
        
        ax = axes[idx]
        # Use exact raw bootstrapped values without any modifications (no jittering or noise added)
        data = np.array(ci_results[key]['boot_vals'])
        actual = ci_results[key]['actual']
        lower = ci_results[key]['lower']
        upper = ci_results[key]['upper']
        
        unique_vals = np.unique(data)
        
        if len(unique_vals) <= 20:
            # Discrete distribution - compute bin edges centered on unique values
            sorted_vals = np.sort(unique_vals)
            diffs = np.diff(sorted_vals)
            step = np.min(diffs) if len(diffs) > 0 else 0.01
            bins = np.append(sorted_vals - step / 2.0, sorted_vals[-1] + step / 2.0)
            ax.hist(data, bins=bins, color=color, alpha=0.6, edgecolor='white', rwidth=0.85)
        else:
            # Continuous distribution
            ax.hist(data, bins=25, color=color, alpha=0.6, edgecolor='white')
            
        # Dynamically set x-axis limits for a clean layout without altering any data values
        min_val = np.min(data)
        max_val = np.max(data)
        if min_val == max_val:
            ax.set_xlim(min_val - 0.03, min_val + 0.03)
        else:
            range_val = max_val - min_val
            padding = max(0.005, range_val * 0.15)
            ax.set_xlim(min_val - padding, min(1.01, max_val + padding))
            
        ax.axvline(actual, color='black', linestyle='-', linewidth=2.5, label=f'Actual: {actual:.3f}')
        ax.axvline(lower, color='red', linestyle='--', linewidth=2, label=f'95% CI Lower: {lower:.3f}')
        ax.axvline(upper, color='red', linestyle='--', linewidth=2, label=f'95% CI Upper: {upper:.3f}')
        
        ax.set_title(f'Bootstrap Distribution for {key.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Metric Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.grid(True, alpha=0.3)
        # Position legend dynamically to avoid overlapping the distribution
        loc = 'upper right' if actual < np.mean(data) else 'upper left'
        ax.legend(fontsize=9, loc=loc)
        
    plt.suptitle(f"Bootstrap 95% Confidence Interval Distributions ({model_name})", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()



def load_model_from_ckpt(ckpt_path, device):
    """Load model architecture and weights from checkpoint path."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # Find config inside checkpoint directory
    ckpt_dir = Path(ckpt_path).parent
    config_path = ckpt_dir / 'config.yaml'
    if not config_path.exists():
        # Fallback to default
        config_path = Path('configs/default.yaml')
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Merge overrides from checkpoint if available
    if 'config' in ckpt:
        config.update(ckpt['config'])
        
    model = create_model(config, device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, config


def run_mock_experiment(seed=42):
    """Run simulated predictions to test stats and return simulated data."""
    rng = np.random.default_rng(seed)
    
    # 100 samples (50 TB, 50 normal)
    y_true = np.array([0] * 50 + [1] * 50)
    
    # Proposed model predictions (99% accurate on test, let's say 1 mistake)
    # y_pred_proposed will have 99% accuracy
    y_pred_proposed = y_true.copy()
    y_pred_proposed[rng.choice(100)] = 1 - y_pred_proposed[rng.choice(100)]  # Introduce 1 error
    
    # Proposed probs (high trust)
    y_probs_proposed = np.zeros(100)
    for i in range(100):
        if y_true[i] == 1:
            y_probs_proposed[i] = rng.uniform(0.75, 0.99) if y_pred_proposed[i] == 1 else rng.uniform(0.1, 0.3)
        else:
            y_probs_proposed[i] = rng.uniform(0.01, 0.25) if y_pred_proposed[i] == 0 else rng.uniform(0.7, 0.9)
            
    # Baseline model predictions (e.g., 94% accurate on test, let's say 6 mistakes)
    y_pred_baseline = y_true.copy()
    errors_base = rng.choice(100, size=6, replace=False)
    for idx in errors_base:
        y_pred_baseline[idx] = 1 - y_pred_baseline[idx]
        
    y_probs_baseline = np.zeros(100)
    for i in range(100):
        if y_true[i] == 1:
            y_probs_baseline[i] = rng.uniform(0.6, 0.95) if y_pred_baseline[i] == 1 else rng.uniform(0.1, 0.4)
        else:
            y_probs_baseline[i] = rng.uniform(0.05, 0.4) if y_pred_baseline[i] == 0 else rng.uniform(0.6, 0.8)
            
    return y_true, y_pred_proposed, y_probs_proposed, y_pred_baseline, y_probs_baseline


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CI and McNemar significance test")
    parser.add_argument("--proposed_ckpt", type=str, default=None,
                        help="Path to proposed model (mgm_tb_net) checkpoint")
    parser.add_argument("--proposed_name", type=str, default=None,
                        help="Override name for proposed model in plots/reports")
    parser.add_argument("--baseline_ckpt", type=str, default=None,
                        help="Path to comparative baseline model checkpoint")
    parser.add_argument("--baseline_name", type=str, default=None,
                        help="Override name for baseline model in plots/reports")
    parser.add_argument("--data_root", type=str, default="data/JU-LDD-task-b",
                        help="Dataset root path")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--B", type=int, default=1000,
                        help="Number of bootstrap resamples")
    parser.add_argument("--confidence_level", type=float, default=0.95,
                        help="Confidence level for CIs")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save statistical reports and plots")
    parser.add_argument("--mock", action="store_true",
                        help="Run mock analysis with synthetic data (for test/dry-run)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for bootstrapping")
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("REVISION STATISTICAL TEST: BOOTSTRAP CI & MCNEMAR TEST")
    print("="*70)
    
    if args.mock:
        print("\n--> [MOCK MODE] Generating synthetic predictions...")
        y_true, y_pred_p, y_probs_p, y_pred_b, y_probs_b = run_mock_experiment(seed=args.seed)
        proposed_name = args.proposed_name if args.proposed_name else "MGM-TB-Net (Ours - Mock)"
        baseline_name = args.baseline_name if args.baseline_name else "Late Fusion Baseline (Mock)"
    else:
        # Load weights and run evaluation
        if not args.proposed_ckpt or not args.baseline_ckpt:
            print("[ERROR] Please provide --proposed_ckpt and --baseline_ckpt OR run with --mock")
            return
            
        device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Device: {device}")
        
        # Load datasets
        print(f"Loading test dataset from: {args.data_root}")
        _, _, test_loader = get_dataloaders(
            data_root=args.data_root,
            batch_size=args.batch_size,
            num_workers=0,
            is_unimodal=False
        )
        
        # Load Proposed Model
        print(f"Loading proposed model checkpoint: {args.proposed_ckpt}")
        model_p, config_p = load_model_from_ckpt(args.proposed_ckpt, device)
        modality_p = config_p.get('modality', 'both')
        y_true, y_pred_p, y_probs_p = run_evaluation(model_p, test_loader, device, modality=modality_p)
        proposed_name = args.proposed_name if args.proposed_name else config_p.get('exp_name', 'MGM-TB-Net (Ours)')
        
        # Load Baseline Model
        print(f"Loading baseline model checkpoint: {args.baseline_ckpt}")
        model_b, config_b = load_model_from_ckpt(args.baseline_ckpt, device)
        modality_b = config_b.get('modality', 'both')
        _, y_pred_b, y_probs_b = run_evaluation(model_b, test_loader, device, modality=modality_b)
        baseline_name = args.baseline_name if args.baseline_name else config_b.get('exp_name', 'Baseline Model')
        
    print(f"\nTest set size: {len(y_true)} total (50 normal, 50 TB)")
    
    # 1. Compute Bootstrapped CIs for Proposed Model
    print(f"\nComputing Bootstrap CIs for {proposed_name}...")
    ci_results_p = bootstrap_ci(y_true, y_pred_p, y_probs_p, B=args.B, ci_level=args.confidence_level, seed=args.seed)
    
    # 2. Compute Bootstrapped CIs for Baseline Model
    print(f"\nComputing Bootstrap CIs for {baseline_name}...")
    ci_results_b = bootstrap_ci(y_true, y_pred_b, y_probs_b, B=args.B, ci_level=args.confidence_level, seed=args.seed)
    
    # 3. Perform McNemar significance test
    print("\nRunning McNemar statistical significance test...")
    mcnemar_results = mcnemar_significance_test(y_true, y_pred_p, y_pred_b)
    
    # Print results to stdout
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY: {proposed_name} vs {baseline_name}")
    print(f"{'='*70}")
    
    print(f"\n{proposed_name} Metrics with 95% Confidence Intervals:")
    for metric, ci in ci_results_p.items():
        print(f"  {metric.upper():12s}: {ci['actual']:.4f}  [95% CI: {ci['lower']:.4f} - {ci['upper']:.4f}]")
        
    print(f"\n{baseline_name} Metrics with 95% Confidence Intervals:")
    for metric, ci in ci_results_b.items():
        print(f"  {metric.upper():12s}: {ci['actual']:.4f}  [95% CI: {ci['lower']:.4f} - {ci['upper']:.4f}]")
        
    print(f"\nMcNemar Significance Test:")
    print(f"  Test type:         {mcnemar_results['test_type']}")
    print(f"  Contingency Table:")
    print(f"                         {baseline_name} Correct  {baseline_name} Incorrect")
    print(f"    {proposed_name:20s} Correct            {mcnemar_results['table'][0][0]:3d}                {mcnemar_results['table'][0][1]:3d}")
    print(f"    {proposed_name:20s} Incorrect          {mcnemar_results['table'][1][0]:3d}                {mcnemar_results['table'][1][1]:3d}")
    print(f"  Test statistic:    {mcnemar_results['statistic']:.4f}")
    print(f"  P-value:           {mcnemar_results['p_value']:.4g}")
    sig_str = "SIGNIFICANT (p < 0.05)" if mcnemar_results['p_value'] < 0.05 else "NOT SIGNIFICANT (p >= 0.05)"
    print(f"  Verdict:           {sig_str}")
    

    # Save Proposed Plot
    plot_file_p = output_path / "bootstrap_distributions_proposed.png"
    plot_bootstrap_distributions(ci_results_p, proposed_name, plot_file_p)
    print(f"[SAVED] Proposed bootstrap distributions plot saved to: {plot_file_p}")
    
    # Save Baseline Plot
    plot_file_b = output_path / "bootstrap_distributions_baseline.png"
    plot_bootstrap_distributions(ci_results_b, baseline_name, plot_file_b)
    print(f"[SAVED] Baseline bootstrap distributions plot saved to: {plot_file_b}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
