"""
Experiment runner for all MGM-TB-Net experiments
Coordinates training of all models from the experimental plan
"""
import os
import yaml
import argparse
from pathlib import Path
from datetime import datetime

# All experiment configurations
EXPERIMENTS = {
    # Tier 1: Transformer-based experiments
    'tier1': {
        'cxr_only': {
            'model_type': 'unimodal',
            'modality': 'cxr',
            'description': 'CXR-only baseline'
        },
        'sputum_only': {
            'model_type': 'unimodal',
            'modality': 'sputum',
            'description': 'Sputum-only baseline'
        },
        'concat_fusion': {
            'model_type': 'concat',
            'description': 'Late fusion via concatenation'
        },
        'vanilla_cmt': {
            'model_type': 'vanilla_cmt',
            'description': 'Cross-attention without gating',
            'use_residual': False
        },
        'scalar_gate_mlp': {
            'model_type': 'scalar_gate',
            'gate_type': 'mlp',
            'description': 'Learnable MLP gate',
            'use_residual': False
        },
        'scalar_gate_sigmoid': {
            'model_type': 'scalar_gate',
            'gate_type': 'sigmoid',
            'description': 'Learnable sigmoid gate',
            'use_residual': False
        },
        'mgm_tb_net': {
            'model_type': 'mgm_tb_net',
            'fmca_modulation': 'logit',
            'description': 'Full MGM-TB-Net (ours)'
        }
    },
    
    # Tier 2: CNN Backbone Comparisons (Standard Baselines)
    'tier2': {
        # Multimodal Fusion Baselines (Concat)
        'resnet_fusion': {
            'model_type': 'cnn_fusion',
            'backbone': 'resnet50',
            'description': 'ResNet-50 Concat Fusion'
        },
        'efficientnet_fusion': {
            'model_type': 'cnn_fusion',
            'backbone': 'efficientnet_b0',
            'description': 'EfficientNet-B0 Concat Fusion'
        },
        'mobilenet_fusion': {
            'model_type': 'cnn_fusion',
            'backbone': 'mobilenet_v2',
            'description': 'MobileNetV2 Concat Fusion'
        }
    },
    
    # -------------------------------------------------------
    # Tier 4: Robustness (Noise Analysis)
    # -------------------------------------------------------
    'robustness': {
        # Concat Fusion Baselines
        'concat_noise_1': {'model_type': 'concat', 'modality': 'both', 'kwargs': {'cxr_noise': 0.1}, 'desc': 'Concat Fusion (Noise 0.1)'},
        'concat_noise_2': {'model_type': 'concat', 'modality': 'both', 'kwargs': {'cxr_noise': 0.2}, 'desc': 'Concat Fusion (Noise 0.2)'},
        'concat_noise_3': {'model_type': 'concat', 'modality': 'both', 'kwargs': {'cxr_noise': 0.3}, 'desc': 'Concat Fusion (Noise 0.3)'},
        
        # MGM-TB-Net (Ours)
        'mgm_tb_net_noise_1': {'model_type': 'mgm_tb_net', 'modality': 'both', 'kwargs': {'cxr_noise': 0.1}, 'desc': 'MGM-TB-Net (Noise 0.1)'},
        'mgm_tb_net_noise_2': {'model_type': 'mgm_tb_net', 'modality': 'both', 'kwargs': {'cxr_noise': 0.2}, 'desc': 'MGM-TB-Net (Noise 0.2)'},
        'mgm_tb_net_noise_3': {'model_type': 'mgm_tb_net', 'modality': 'both', 'kwargs': {'cxr_noise': 0.3}, 'desc': 'MGM-TB-Net (Noise 0.3)'},
    },

    # -------------------------------------------------------
    # Dataset 2 Comparative Analysis (CXR-only)
    # -------------------------------------------------------
    'dataset2_comparison': {
        'densenet121': {
            'model_type': 'cnn_unimodal',
            'backbone': 'densenet121',
            'data_root': 'data/Dataset of Tuberculosis Chest X-rays Images',
            'is_unimodal': True,
            'modality': 'cxr',
            'description': 'DenseNet121 Dataset 2 Comparison'
        },
        'resnet_50': {
            'model_type': 'cnn_unimodal',
            'backbone': 'resnet50',
            'data_root': 'data/Dataset of Tuberculosis Chest X-rays Images',
            'is_unimodal': True,
            'modality': 'cxr',
            'description': 'ResNet-50 Dataset 2 Comparison'
        },
        'efficientnet_v2_s': {
            'model_type': 'cnn_unimodal',
            'backbone': 'efficientnet_v2_s',
            'data_root': 'data/Dataset of Tuberculosis Chest X-rays Images',
            'is_unimodal': True,
            'modality': 'cxr',
            'description': 'EfficientNetV2-S Dataset 2 Comparison'
        },
        'vit_tiny': {
            'model_type': 'transformer_unimodal',
            'backbone': 'vit_tiny',
            'data_root': 'data/Dataset of Tuberculosis Chest X-rays Images',
            'is_unimodal': True,
            'modality': 'cxr',
            'description': 'ViT-Tiny Dataset 2 Comparison'
        },
        'swin_tiny': {
            'model_type': 'transformer_unimodal',
            'backbone': 'swin_tiny',
            'data_root': 'data/Dataset of Tuberculosis Chest X-rays Images',
            'is_unimodal': True,
            'modality': 'cxr',
            'description': 'Swin-Tiny Dataset 2 Comparison'
        },
        'cvt_tiny': {
            'model_type': 'transformer_unimodal',
            'backbone': 'cvt_tiny',
            'data_root': 'data/Dataset of Tuberculosis Chest X-rays Images',
            'is_unimodal': True,
            'modality': 'cxr',
            'description': 'CvT-Tiny Dataset 2 Comparison'
        },
        'mgm_tb_net_dataset2': {
            'model_type': 'mgm_tb_net',
            'data_root': 'data/Dataset of Tuberculosis Chest X-rays Images',
            'is_unimodal': True,
            'modality': 'both',
            'description': 'MGM-TB-Net Dataset 2 Comparison (Sputum=0)'
        }
    },

    # -------------------------------------------------------
    # Ablation Studies
    # -------------------------------------------------------
    # Ablation A: FIS variants
    'ablation_fis': {
        'mgm_tb_net_mamdani': {
            'model_type': 'mgm_tb_net',
            'fis_type': 'mamdani',
            'description': 'Full Mamdani FIS'
        },
        'mgm_tb_net_no_gate': {
            'model_type': 'vanilla_cmt',
            'description': 'No gating (alpha=1)',
            'use_residual': True
        },
        'mgm_tb_net_mlp_gate': {
            'model_type': 'scalar_gate',
            'gate_type': 'mlp',
            'description': 'MLP gate instead of FIS',
            'use_residual': True
        },
        'mgm_tb_net_sigmoid_gate': {
            'model_type': 'scalar_gate',
            'gate_type': 'sigmoid',
            'description': 'Sigmoid gate instead of FIS',
            'use_residual': True
        }
    },
    
    # Ablation B: FMCA mechanism (comparing modulation strategies)
    # All experiments here use residual connections for fair comparison:
    # - fmca_standard: uses vanilla_cmt + use_residual=True (explicit flag)
    # - fmca_logit_scale/post_scale: use mgm_tb_net (residual is hardcoded in MGCMT class)
    'ablation_fmca': {
        'fmca_standard': {
            'model_type': 'vanilla_cmt',
            'description': 'Standard cross-attention',
            'use_residual': True  # Matches mgm_tb_net residual for fair comparison
        },
        # Note: fmca_logit_scale removed — identical to mgm_tb_net (both use mgm_tb_net + logit)
        'fmca_post_scale': {
            'model_type': 'mgm_tb_net',  # mgm_tb_net has residual built-in
            'fmca_modulation': 'post',
            'description': 'FMCA with post-softmax scaling'
        }
    },

    # Ablation C: Lambda Sensitivity (Auxiliary Loss Weight)
    'ablation_lambda': {
        'mgm_tb_net_lambda_0_01': {
            'model_type': 'mgm_tb_net',
            'lambda_aux': 0.01,
            'description': 'Lambda=0.01 (Weak Regularization)'
        },
        'mgm_tb_net_lambda_1_0': {
            'model_type': 'mgm_tb_net',
            'lambda_aux': 1.0,
            'description': 'Lambda=1.0 (Strong Regularization)'
        },
        'mgm_tb_net_lambda_0_5': {
            'model_type': 'mgm_tb_net',
            'lambda_aux': 0.5,
            'description': 'Lambda=0.5 (Moderate Regularization)'
        }
        # Note: Lambda=0.1 is the default 'mgm_tb_net' experiment
    }
}


def create_experiment_config(base_config, exp_config):
    """Create config for specific experiment."""
    config = base_config.copy()
    config.update(exp_config)
    return config


def run_single_experiment(exp_name, exp_config, base_config, dry_run=False):
    """Run a single experiment."""
    from train import train
    
    # Create experiment-specific config
    config = create_experiment_config(base_config, exp_config)
    config['exp_name'] = exp_name
    
    print(f"\n{'='*70}")
    print(f"Starting Experiment: {exp_name}")
    print(f"   Description: {exp_config.get('description', 'N/A')}")
    print(f"   Model Type: {exp_config.get('model_type', 'N/A')}")
    print(f"{'='*70}\n")
    
    if dry_run:
        print(f"[DRY RUN] Would train with config:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        return
    
    # Run training
    try:
        train(config)
        print(f"\n[OK] Experiment '{exp_name}' completed successfully!\n")
    except Exception as e:
        print(f"\n[ERROR] Experiment '{exp_name}' failed: {e}\n")
        raise


def run_experiment_suite(suite_name, base_config, dry_run=False):
    """Run a suite of experiments."""
    if suite_name not in EXPERIMENTS:
        print(f"Error: Suite '{suite_name}' not found.")
        print(f"Available suites: {list(EXPERIMENTS.keys())}")
        return
    
    suite = EXPERIMENTS[suite_name]
    
    print(f"\n{'#'*70}")
    print(f"# EXPERIMENT SUITE: {suite_name.upper()}")
    print(f"# Total experiments: {len(suite)}")
    print(f"{'#'*70}\n")
    
    for exp_name, exp_config in suite.items():
        run_single_experiment(exp_name, exp_config, base_config, dry_run)


def main():
    parser = argparse.ArgumentParser(description='Run MGM-TB-Net experiments')
    parser.add_argument('--suite', type=str, choices=list(EXPERIMENTS.keys()) + ['all'],
                        default='tier1', help='Experiment suite to run')
    parser.add_argument('--experiment', type=str, default=None,
                        help='Run specific experiment only')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Base configuration file')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print configs without running')
    parser.add_argument('--list', action='store_true',
                        help='List all available experiments')
    
    args = parser.parse_args()
    
    # List experiments
    if args.list:
        print("\nAvailable Experiment Suites:")
        for suite_name, experiments in EXPERIMENTS.items():
            print(f"\n  {suite_name.upper()}:")
            for exp_name, exp_config in experiments.items():
                desc = exp_config.get('description', 'N/A')
                print(f"    - {exp_name:25s} : {desc}")
        return
    
    # Load base config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            base_config = yaml.safe_load(f)
    else:
        base_config = {
            'data_root': 'data/JU-LDD-task-b',
            'save_dir': 'checkpoints',
            'img_size': 224,
            'patch_size': 16,
            'num_layers': 4,
            'embed_dim': 192,
            'num_heads': 8,
            'batch_size': 4,
            'num_epochs': 30,
            'learning_rate': 0.0003,
            'weight_decay': 0.15,
            'num_workers': 2
        }
    
    # Override with command line args
    base_config['num_epochs'] = args.epochs
    base_config['batch_size'] = args.batch_size
    
    # Run specific experiment
    if args.experiment:
        # Find experiment across all suites
        found = False
        for suite_name, suite in EXPERIMENTS.items():
            if args.experiment in suite:
                exp_config = suite[args.experiment]
                run_single_experiment(args.experiment, exp_config, base_config, args.dry_run)
                found = True
                break
        
        if not found:
            print(f"Error: Experiment '{args.experiment}' not found.")
            print(f"Use --list to see all available experiments.")
        return
    
    # Run experiment suite(s)
    if args.suite == 'all':
        for suite_name in EXPERIMENTS.keys():
            run_experiment_suite(suite_name, base_config, args.dry_run)
    else:
        run_experiment_suite(args.suite, base_config, args.dry_run)


if __name__ == '__main__':
    main()
