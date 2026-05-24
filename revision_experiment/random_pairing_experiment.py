"""
Random Sputum Pairing Experiment for TB Detection (Reviewer #3)
Shuffles sputum microscopy images to test if the model relies on true multimodal
co-adaptation or artificial class correlation.
Includes both:
1. Within-Class Shuffling (preserves class alignment, breaks patient pairing)
2. Across-Class Shuffling (breaks both class alignment and patient pairing)
"""
import os
import argparse
import copy
import yaml
import torch
import random
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
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

try:
    from data.dataset import TBMultimodalDataset
    from train import create_model
except ImportError:
    TBMultimodalDataset = None
    create_model = None


def compute_metrics(y_true, y_pred, y_probs=None):
    """Compute standard metrics."""
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
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


def load_model_from_ckpt(ckpt_path, device):
    """Load model architecture and weights from checkpoint path."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    ckpt_dir = Path(ckpt_path).parent
    config_path = ckpt_dir / 'config.yaml'
    if not config_path.exists():
        config_path = Path('configs/default.yaml')
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    if 'config' in ckpt:
        config.update(ckpt['config'])
        
    model = create_model(config, device)
    
    # Handle state dict loading
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else (ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    return model, config


def plot_pairing_comparison(clean_m, within_m, across_m, save_path):
    """Generate bar chart comparing all three pairing scenarios."""
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
    clean_vals = [clean_m[k] * 100 for k in metrics_to_plot]
    within_vals = [within_m[k] * 100 for k in metrics_to_plot]
    across_vals = [across_m[k] * 100 for k in metrics_to_plot]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(11, 6))
    rects1 = ax.bar(x - width, clean_vals, width, label='Clean Paired (Original)', color='#1f77b4', alpha=0.8)
    rects2 = ax.bar(x, within_vals, width, label='Within-Class Mismatch', color='#2ca02c', alpha=0.8)
    rects3 = ax.bar(x + width, across_vals, width, label='Across-Class Mismatch', color='#d62728', alpha=0.8)
    
    ax.set_ylabel('Score (%)')
    ax.set_title('Sputum Scrambling Comparison (Within vs. Across-Class)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics_to_plot])
    ax.legend(loc='lower left')
    ax.set_ylim(0, 115)
    
    # Add values on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def shuffle_sputum_within_class(samples, seed=42):
    """Shuffle sputum paths only within the same class (TB with TB, Normal with Normal)."""
    rng = random.Random(seed)
    shuffled_samples = copy.deepcopy(samples)
    
    # Separate indices by class label
    tb_indices = [i for i, s in enumerate(samples) if s['label'] == 1]
    normal_indices = [i for i, s in enumerate(samples) if s['label'] == 0]
    
    # Extract sputum paths for each class
    tb_sputum = [samples[i]['sputum_path'] for i in tb_indices]
    normal_sputum = [samples[i]['sputum_path'] for i in normal_indices]
    
    # Shuffle paths
    rng.shuffle(tb_sputum)
    rng.shuffle(normal_sputum)
    
    # Assign shuffled paths back
    for idx, path in zip(tb_indices, tb_sputum):
        shuffled_samples[idx]['sputum_path'] = path
    for idx, path in zip(normal_indices, normal_sputum):
        shuffled_samples[idx]['sputum_path'] = path
        
    return shuffled_samples


def shuffle_sputum_across_class(samples, seed=42):
    """Completely shuffle sputum paths across all classes."""
    rng = random.Random(seed)
    shuffled_samples = copy.deepcopy(samples)
    
    all_sputum_paths = [s['sputum_path'] for s in samples]
    rng.shuffle(all_sputum_paths)
    
    for i in range(len(shuffled_samples)):
        shuffled_samples[i]['sputum_path'] = all_sputum_paths[i]
        
    return shuffled_samples


def main():
    parser = argparse.ArgumentParser(description="Random Sputum Pairing Ablation Study")
    parser.add_argument("--proposed_ckpt", type=str, required=True, help="Path to proposed model checkpoint")
    parser.add_argument("--data_root", type=str, default="data/JU-LDD-task-b", help="Path to data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save output files")
    parser.add_argument("--seed", type=int, default=42, help="Seed for shuffling")
    
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
    
    print("\n" + "="*80)
    print("REVISION STUDY: RANDOM SPUTUM PAIRING SHUFFLING (3 SCENARIOS)")
    print(f"Device: {device}")
    print(f"Proposed Checkpoint: {args.proposed_ckpt}")
    print(f"Data Root: {args.data_root}")
    print("="*80 + "\n")
    
    # 1. Load test dataloader
    clean_dataset = TBMultimodalDataset(
        data_root=args.data_root,
        split='test',
        img_size=224
    )
    
    clean_loader = torch.utils.data.DataLoader(
        clean_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 2. Load model
    model, config = load_model_from_ckpt(args.proposed_ckpt, device)
    modality = config.get('modality', 'both')
    
    # 3. Clean Evaluation
    print("Running Clean Control Evaluation...")
    y_true, y_pred_c, y_probs_c = run_evaluation(model, clean_loader, device, modality=modality)
    clean_m = compute_metrics(y_true, y_pred_c, y_probs_c)
    
    # 4. Within-Class Evaluation
    print("\nRunning Within-Class Mismatch Evaluation...")
    within_dataset = copy.deepcopy(clean_dataset)
    within_dataset.samples = shuffle_sputum_within_class(clean_dataset.samples, seed=args.seed)
    within_loader = torch.utils.data.DataLoader(
        within_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    y_true_w, y_pred_w, y_probs_w = run_evaluation(model, within_loader, device, modality=modality)
    within_m = compute_metrics(y_true_w, y_pred_w, y_probs_w)
    
    # 5. Across-Class Evaluation
    print("\nRunning Across-Class Mismatch Evaluation...")
    across_dataset = copy.deepcopy(clean_dataset)
    across_dataset.samples = shuffle_sputum_across_class(clean_dataset.samples, seed=args.seed)
    across_loader = torch.utils.data.DataLoader(
        across_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    y_true_a, y_pred_a, y_probs_a = run_evaluation(model, across_loader, device, modality=modality)
    across_m = compute_metrics(y_true_a, y_pred_a, y_probs_a)
    
    # Print metrics table
    print("\n" + "="*85)
    print("RESULTS COMPARISON")
    print("="*85)
    print(f"Metric       |  Clean Paired  |  Within-Class Mismatch  |  Across-Class Mismatch")
    print("-" * 85)
    for key in ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']:
        print(f"{key.upper():12s} |     {clean_m[key]*100:.2f}%     |         {within_m[key]*100:.2f}%         |         {across_m[key]*100:.2f}%")
    print("="*85 + "\n")
    
    # Ensure output directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot results
    plot_file = Path(args.output_dir) / 'pairing_comparison.png'
    plot_pairing_comparison(clean_m, within_m, across_m, str(plot_file))
    print(f"[SAVED] Comparison plot saved to: {plot_file}")
    



if __name__ == "__main__":
    main()
