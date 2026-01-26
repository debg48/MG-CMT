"""
Robustness & Interpretability Analysis Script
Generates:
1. Alpha vs Noise Level Plot
2. Failure Case Visualization (Noisy CXR, Clean Sputum, Alpha)
3. Missing Modality Simulation Results
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from PIL import Image

from data.dataset import TBMultimodalDataset, AddGaussianNoise
from models.mg_cmt import MGCMT
from baselines.transformer_baselines import ConcatFusion

def load_model(checkpoint_path, config, device):
    """Load trained MG-CMT model."""
    model = MGCMT(
        img_size=config['img_size'],
        patch_size=config['patch_size'],
        num_transformer_layers=config['num_layers'],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_classes=2,
        fmca_modulation=config.get('fmca_modulation', 'logit')
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model



def visualize_failure_cases(model, data_loader, device):
    """Visualize 3-4 interesting cases."""
    print("\n[Analysis 2] Visualizing Cases...")
    
    # We want to find a High Uncertainty case
    # Inject heavy noise to find one
    noise_transform = AddGaussianNoise(std=0.8)
    
    found = False
    
    for batch in data_loader:
        if found: break
        cxr, sputum, labels, ids = batch
        
        # Corrupt the CXR
        noisy_cxr = torch.stack([noise_transform(img) for img in cxr]).to(device)
        sputum = sputum.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(noisy_cxr, sputum)
            preds = torch.argmax(outputs['logits'], dim=1)
            alphas = outputs.get('alpha_cxr', outputs.get('alpha', torch.zeros_like(preds).float()))
            
        # Look for a correctly classified example with low alpha (high CXR uncertainty)
        for i in range(len(labels)):
            if preds[i] == labels[i] and alphas[i] < 0.3: # Threshold for "low trust in CXR"
                # Found one!
                print(f"  Found interesting case: ID {ids['id'][i]}")
                print(f"  Correct: {preds[i]==labels[i]}")
                print(f"  Alpha (CXR): {alphas[i]:.4f}")
                
                # Plot
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                
                # Show Noisy CXR (denormalize for viz)
                img_cxr = noisy_cxr[i].cpu().permute(1,2,0) * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                img_cxr = torch.clamp(img_cxr, 0, 1)
                
                # Show Clean Sputum
                img_spt = sputum[i].cpu().permute(1,2,0) * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
                img_spt = torch.clamp(img_spt, 0, 1)
                
                axes[0].imshow(img_cxr)
                axes[0].set_title(f"Noisy CXR (Alpha={alphas[i]:.2f})")
                axes[0].axis('off')
                
                axes[1].imshow(img_spt)
                axes[1].set_title(f"Clean Sputum (Trusted)")
                axes[1].axis('off')
                
                plt.suptitle(f"MG-CMT Handling Sensor Failure (True Label: {labels[i].item()})")
                plt.savefig('results/failure_case_viz.png', dpi=300)
                print("  [Saved] results/failure_case_viz.png")
                found = True
                break

def simulate_missing_modality(model, data_loader, device):
    """Simulate missing modality (zeroed out input)."""
    print("\n[Analysis 3] Missing Modality Simulation...")
    
    accuracies = {}
    
    # 1. Full Data
    correct = 0
    total = 0
    with torch.no_grad():
        for cxr, sputum, labels, _ in data_loader:
            cxr, sputum, labels = cxr.to(device), sputum.to(device), labels.to(device)
            out = model(cxr, sputum)
            preds = torch.argmax(out['logits'], dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    accuracies['Full'] = correct / total
    
    # 2. Missing CXR (Zeros)
    correct = 0
    total = 0
    with torch.no_grad():
        for cxr, sputum, labels, _ in data_loader:
            cxr_missing = torch.zeros_like(cxr).to(device)
            sputum, labels = sputum.to(device), labels.to(device)
            out = model(cxr_missing, sputum)
            preds = torch.argmax(out['logits'], dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    accuracies['Missing CXR'] = correct / total
    
    # 3. Missing Sputum (Zeros)
    correct = 0
    total = 0
    with torch.no_grad():
        for cxr, sputum, labels, _ in data_loader:
            cxr, labels = cxr.to(device), labels.to(device)
            sputum_missing = torch.zeros_like(sputum).to(device)
            out = model(cxr, sputum_missing)
            preds = torch.argmax(out['logits'], dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    accuracies['Missing Sputum'] = correct / total
    
    print("\nResults:")
    for k, v in accuracies.items():
        print(f"  {k}: {v*100:.2f}%")
        
    # Bar plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(accuracies.keys(), [v*100 for v in accuracies.values()], color=['green', 'orange', 'red'])
    plt.ylim(0, 100)
    plt.ylabel('Accuracy (%)')
    plt.title('MG-CMT Performance under Missing Modality')
    plt.grid(axis='y', alpha=0.3)
    
    # Add numbers on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}%', ha='center', va='bottom')
                 
    plt.savefig('results/missing_modality.png', dpi=300)
    print("  [Saved] results/missing_modality.png")

if __name__ == "__main__":
    # Load config and model
    # Assuming standard config layout
    config_path = "configs/default.yaml" 
    # Use best checkpoint (Assuming MG-CMT run is done)
    # Check if checkpoint exists
    # Find actual MG-CMT checkpoints (not ablation variants like mg_cmt_no_gate, mg_cmt_sigmoid_gate)
    # Pattern: mg_cmt_YYYYMMDD_HHMMSS (exactly 8+6 digits after mg_cmt_)
    checkpoint_glob = [p for p in Path("checkpoints").glob("mg_cmt_*/checkpoint_best.pth")
                       if len(p.parent.name) == len("mg_cmt_20260125_145550")]  # Only exact timestamp format
    
    if not checkpoint_glob:
        print("Error: No MG-CMT checkpoint found! Run MG-CMT training first.")
        print("Looking for: checkpoints/mg_cmt_YYYYMMDD_HHMMSS/checkpoint_best.pth")
        exit(1)
        
    checkpoint_path = sorted(checkpoint_glob)[-1] # Take latest
    print(f"Using checkpoint: {checkpoint_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load Dataset (Validation Set basically)
    # Use Dataset class directly to avoid recreating infinite dataloaders
    test_set = TBMultimodalDataset("data/JU-LDD-task-b", split='test', img_size=224, augment=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)
    
    # Load Model
    model = load_model(checkpoint_path, config, device)
    
    # Run Analyses
    visualize_failure_cases(model, test_loader, device)
    simulate_missing_modality(model, test_loader, device)
