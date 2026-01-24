"""
Visualization utilities for training and results
"""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_training_curves(history, save_dir):
    """
    Plot training and validation curves.
    
    Args:
        history: dict with 'train' and 'val' lists of metrics per epoch
        save_dir: Directory to save plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train']['loss']) + 1)
    
    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train']['loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, history['val']['loss'], 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train']['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, history['val']['accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_dir / 'accuracy_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # F1 curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train']['f1'], 'b-', label='Train F1', linewidth=2)
    plt.plot(epochs, history['val']['f1'], 'r-', label='Val F1', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1-Score', fontsize=12)
    plt.title('Training and Validation F1-Score', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_dir / 'f1_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Combined metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train']['loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val']['loss'], 'r-', label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['train']['accuracy'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val']['accuracy'], 'r-', label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([0, 1])
    
    # F1
    axes[1, 0].plot(epochs, history['train']['f1'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(epochs, history['val']['f1'], 'r-', label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_title('F1-Score', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim([0, 1])
    
    # AUC-ROC
    if 'auc_roc' in history['val']:
        axes[1, 1].plot(epochs, history['val']['auc_roc'], 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC-ROC')
        axes[1, 1].set_title('Validation AUC-ROC', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 1])
    
    plt.suptitle('Training Metrics', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_dir / 'all_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """
    Plot confusion matrix as heatmap.
    
    Args:
        cm: Confusion matrix (2x2 for binary)
        class_names: List of class names
        save_path: Path to save figure
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(
        cm_norm,
        annot=cm,  # Show raw counts
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Value'},
        linewidths=2,
        linecolor='white',
        square=True
    )
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(results_dict, save_path, metric='f1'):
    """
    Compare metrics across different models.
    
    Args:
        results_dict: {model_name: metrics_dict}
        save_path: Path to save figure
        metric: Metric to compare
    """
    models = list(results_dict.keys())
    values = [results_dict[m][metric] for m in models]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(models)), values, color='steelblue', alpha=0.8)
    
    # Color best model
    best_idx = np.argmax(values)
    bars[best_idx].set_color('darkgreen')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.title(f'{metric.upper()} Comparison Across Models', fontsize=14, fontweight='bold')
    plt.xticks(range(len(models)), models, rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.01,
            f'{val:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(fpr, tpr, auc, save_path, model_name='Model'):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
