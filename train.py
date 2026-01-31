"""
Enhanced training script with plotting and multi-model support
Supports all baselines and ablations
"""
import os
import argparse
import yaml
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from models.mgm_tb_former import MGMTBFormer
from baselines.transformer_baselines import (
    UnimodalModel, ConcatFusion, VanillaCMT, ScalarGateFusion
)
from models.encoders import CNNEncoder
from baselines.cnn_baselines import StandardCNNFusion, StandardCNNUnimodal
from data.dataset import get_dataloaders
from utils.metrics import compute_metrics, print_metrics
from utils.visualization import (
    plot_training_curves, plot_confusion_matrix
)


def create_model(config, device):
    """Create model based on config."""
    model_type = config.get('model_type', 'mg_cmt')
    
    common_args = {
        'img_size': config['img_size'],
        'patch_size': config['patch_size'],
        'num_layers': config['num_layers'],
        'embed_dim': config['embed_dim'],
        'num_heads': config['num_heads'],
        'num_classes': 2,
        'dropout': config.get('dropout', 0.1)
    }
    
    if model_type == 'unimodal':
        model = UnimodalModel(**common_args)
    elif model_type == 'concat':
        model = ConcatFusion(**common_args)
    elif model_type == 'vanilla_cmt':
        model = VanillaCMT(
            **common_args,
            use_residual=config.get('use_residual', False)
        )
    elif model_type == 'scalar_gate':
        model = ScalarGateFusion(
            **common_args,
            gate_type=config.get('gate_type', 'mlp'),
            use_residual=config.get('use_residual', False)
        )
    elif model_type in ['mg_cmt', 'mgm_tb_former']:
        model = MGMTBFormer(
            img_size=config['img_size'],
            patch_size=config['patch_size'],
            num_transformer_layers=config['num_layers'],
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            num_classes=2,
            fmca_modulation=config.get('fmca_modulation', 'logit')
        )
    elif model_type == 'cnn':
        # To maintain backward compatibility if user uses 'cnn', default to cnn_fusion
        backbone_name = config.get('backbone', 'resnet50')
        model = StandardCNNFusion(
            backbone=backbone_name,
            embed_dim=config['embed_dim'],
            num_classes=2,
            dropout=config.get('dropout', 0.1)
        )
    elif model_type == 'cnn_fusion':
        # Standard CNN Fusion (Concat)
        backbone_name = config.get('backbone', 'resnet50')
        model = StandardCNNFusion(
            backbone=backbone_name,
            embed_dim=config['embed_dim'],
            num_classes=2,
            dropout=config.get('dropout', 0.1)
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, modality='both'):
    """
    Train for one epoch.
    
    Args:
        modality: 'cxr', 'sputum', or 'both'
    """
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for cxr, sputum, labels, _ in pbar:
        cxr = cxr.to(device)
        sputum = sputum.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass (handle different model interfaces)
        if modality == 'cxr':
            outputs = model(cxr, None)
        elif modality == 'sputum':
            outputs = model(sputum, None)
        else:  # both
            outputs = model(cxr, sputum)
        
        logits = outputs['logits']
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{running_loss/(pbar.n+1):.4f}'
        })
    
    metrics = compute_metrics(all_labels, all_preds)
    metrics['loss'] = running_loss / len(train_loader)
    
    return metrics


@torch.no_grad()
def validate(model, val_loader, criterion, device, epoch, split='Val', modality='both'):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch} [{split}]')
    for cxr, sputum, labels, _ in pbar:
        cxr = cxr.to(device)
        sputum = sputum.to(device)
        labels = labels.to(device)
        
        # Forward pass
        if modality == 'cxr':
            outputs = model(cxr, None)
        elif modality == 'sputum':
            outputs = model(sputum, None)
        else:
            outputs = model(cxr, sputum)
        
        logits = outputs['logits']
        loss = criterion(logits, labels)
        
        running_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics['loss'] = running_loss / len(val_loader)
    
    return metrics, all_labels, all_preds


def save_checkpoint(model, optimizer, epoch, metrics, save_dir, is_best=False):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    latest_path = save_dir / 'checkpoint_latest.pth'
    torch.save(checkpoint, latest_path)
    
    if is_best:
        best_path = save_dir / 'checkpoint_best.pth'
        torch.save(checkpoint, best_path)
        print(f"  [BEST] Saved best model (F1: {metrics['f1']:.4f})")


def train(config):
    """Main training function with plotting."""
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{config['exp_name']}_{timestamp}"
    save_dir = Path(config['save_dir']) / exp_name
    plots_dir = save_dir / 'plots'
    save_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(exist_ok=True)
    
    # Save config
    with open(save_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    writer = SummaryWriter(log_dir=save_dir / 'logs')
    
    # Data loaders
    print("="*60)
    print("LOADING DATASETS")
    print("="*60)
    print(f"Data root: {config['data_root']}")
    print(f"Image size: {config['img_size']}")
    print(f"Batch size: {config['batch_size']}")
    print("")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=config['data_root'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=config['num_workers']
    )
    
    print(f"Dataset loading complete!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches:   {len(val_loader)}")
    print(f"   Test batches:  {len(test_loader)}")
    print("")
    
    # Create model
    print("="*60)
    print("INITIALIZING MODEL")
    print("="*60)
    print(f"Model type: {config.get('model_type', 'mg_cmt')}")
    print(f"Architecture: {config['num_layers']} layers, {config['embed_dim']}-dim, {config['num_heads']} heads")
    print(f"Modality: {config.get('modality', 'both')}")
    print("")
    
    model = create_model(config, device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized successfully!")
    print(f"   Total parameters: {num_params:,}")
    print(f"   Memory estimate: ~{num_params * 4 / 1e6:.1f} MB")
    print("")
    
    # Loss and optimizer
    print("="*60)
    print("TRAINING SETUP")
    print("="*60)
    print(f"Loss function: CrossEntropyLoss")
    print(f"Optimizer: AdamW")
    print(f"  - Learning rate: {config['learning_rate']}")
    print(f"  - Weight decay: {config['weight_decay']}")
    print(f"Scheduler: CosineAnnealingLR")
    print(f"  - T_max: {config['num_epochs']} epochs")
    print("")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )
    
    # Training history for plotting
    history = {
        'train': defaultdict(list),
        'val': defaultdict(list)
    }
    
    # Determine modality
    modality = config.get('modality', 'both')
    
    # Training loop
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Total epochs: {config['num_epochs']}")
    print(f"Device: {device}")
    print(f"Experiment: {exp_name}")
    print(f"Results will be saved to: {save_dir}")
    print("")
    
    best_f1 = 0.0
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, modality
        )
        
        # Validate
        val_metrics, _, _ = validate(
            model, val_loader, criterion, device, epoch, 'Val', modality
        )
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        for key in ['loss', 'accuracy', 'f1']:
            history['train'][key].append(train_metrics[key])
            history['val'][key].append(val_metrics[key])
        if 'auc_roc' in val_metrics:
            history['val']['auc_roc'].append(val_metrics['auc_roc'])
        
        # Print metrics
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, "
              f"AUC: {val_metrics.get('auc_roc', 0):.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # TensorBoard
        writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        writer.add_scalars('Metrics/Accuracy', {
            'train': train_metrics['accuracy'],
            'val': val_metrics['accuracy']
        }, epoch)
        writer.add_scalars('Metrics/F1', {
            'train': train_metrics['f1'],
            'val': val_metrics['f1']
        }, epoch)
        
        # Save checkpoint
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
        
        save_checkpoint(model, optimizer, epoch, val_metrics, save_dir, is_best)
        
        # Plot training curves every 10 epochs
        if epoch % 10 == 0 or epoch == config['num_epochs']:
            print(f"  Generating training curves...")
            plot_training_curves(history, plots_dir)
            print(f"  [SAVED] Plots saved to {plots_dir}")
    
    # Final test evaluation
    print(f"\n{'='*60}")
    print("FINAL TEST EVALUATION")
    print(f"{'='*60}")
    print("Loading best model checkpoint...")
    
    best_checkpoint = torch.load(save_dir / 'checkpoint_best.pth', weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    print(f"[LOADED] Model from epoch {best_checkpoint['epoch']}")
    print(f"   Best validation F1: {best_checkpoint['metrics']['f1']:.4f}")
    print("")
    
    print("Running test evaluation...")
    test_metrics, test_labels, test_preds = validate(
        model, test_loader, criterion, device, 0, 'Test', modality
    )
    
    print(f"\n{'='*60}")
    print("FINAL TEST RESULTS")
    print(f"{'='*60}")
    print_metrics(test_metrics)
    
    print(f"\nGenerating final plots...")
    # Plot final confusion matrix
    cm = np.array(test_metrics['confusion_matrix'])
    plot_confusion_matrix(
        cm,
        class_names=['No Finding', 'TB'],
        save_path=plots_dir / 'confusion_matrix_test.png',
        title='Test Set Confusion Matrix'
    )
    print(f"  [SAVED] Confusion matrix saved")
    
    # Plot final training curves
    plot_training_curves(history, plots_dir)
    print(f"  [SAVED] Training curves saved")
    
    # Save results
    with open(save_dir / 'test_results.yaml', 'w') as f:
        yaml.dump(test_metrics, f)
    print(f"  [SAVED] Test results saved")
    
    writer.close()
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Results directory: {save_dir}")
    print(f"Plots directory: {plots_dir}")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Final test F1: {test_metrics['f1']:.4f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Train MG-CMT')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--model_type', type=str, default='mg_cmt',
                        choices=['unimodal', 'concat', 'vanilla_cmt', 'scalar_gate', 'mg_cmt'])
    parser.add_argument('--modality', type=str, default='both',
                        choices=['cxr', 'sputum', 'both'])
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    
    args = parser.parse_args()
    
    # Load/create config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'exp_name': 'mg_cmt',
            'data_root': 'data/JU-LDD-task-b',
            'save_dir': 'checkpoints',
            'img_size': 224,
            'patch_size': 16,
            'num_layers': 4,
            'embed_dim': 256,
            'num_heads': 8,
            'batch_size': 4,
            'num_epochs': 50,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
            'num_workers': 2
        }
    
    # Override with args
    config['model_type'] = args.model_type
    config['modality'] = args.modality
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.lr:
        config['learning_rate'] = args.lr
    
    print("\n" + "="*60)
    print("Configuration")
    print("="*60)
    for key, value in config.items():
        print(f"  {key:20s}: {value}")
    print("="*60 + "\n")
    
    train(config)


if __name__ == '__main__':
    main()
