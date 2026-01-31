"""
Experiment runner for all MG-CMT experiments
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
        'mg_cmt': {
            'model_type': 'mgm_tb_former',
            'fmca_modulation': 'logit',
            'description': 'Full MGM-TB-Former (ours)'
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
        
        # MG-CMT (Ours)
        'mg_cmt_noise_1': {'model_type': 'mg_cmt', 'modality': 'both', 'kwargs': {'cxr_noise': 0.1}, 'desc': 'MG-CMT (Noise 0.1)'},
        'mg_cmt_noise_2': {'model_type': 'mg_cmt', 'modality': 'both', 'kwargs': {'cxr_noise': 0.2}, 'desc': 'MG-CMT (Noise 0.2)'},
        'mg_cmt_noise_3': {'model_type': 'mg_cmt', 'modality': 'both', 'kwargs': {'cxr_noise': 0.3}, 'desc': 'MG-CMT (Noise 0.3)'},
    },

    # -------------------------------------------------------
    # Ablation Studies
    # -------------------------------------------------------
    # Ablation A: FIS variants
    'ablation_fis': {
        'mg_cmt_mamdani': {
            'model_type': 'mg_cmt',
            'fis_type': 'mamdani',
            'description': 'Full Mamdani FIS'
        },
        'mg_cmt_no_gate': {
            'model_type': 'vanilla_cmt',
            'description': 'No gating (alpha=1)',
            'use_residual': True
        },
        'mg_cmt_mlp_gate': {
            'model_type': 'scalar_gate',
            'gate_type': 'mlp',
            'description': 'MLP gate instead of FIS',
            'use_residual': True
        },
        'mg_cmt_sigmoid_gate': {
            'model_type': 'scalar_gate',
            'gate_type': 'sigmoid',
            'description': 'Sigmoid gate instead of FIS',
            'use_residual': True
        }
    },
    
    # Ablation B: FMCA mechanism (comparing modulation strategies)
    # All experiments here use residual connections for fair comparison:
    # - fmca_standard: uses vanilla_cmt + use_residual=True (explicit flag)
    # - fmca_logit_scale/post_scale: use mg_cmt (residual is hardcoded in MGCMT class)
    'ablation_fmca': {
        'fmca_standard': {
            'model_type': 'vanilla_cmt',
            'description': 'Standard cross-attention',
            'use_residual': True  # Matches mg_cmt residual for fair comparison
        },
        'fmca_logit_scale': {
            'model_type': 'mgm_tb_former',  # mg_cmt has residual built-in
            'fmca_modulation': 'logit',
            'description': 'FMCA with logit scaling (ours)'
        },
        'fmca_post_scale': {
            'model_type': 'mgm_tb_former',  # mgm_tb_former has residual built-in
            'fmca_modulation': 'post',
            'description': 'FMCA with post-softmax scaling'
        }
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
    parser = argparse.ArgumentParser(description='Run MG-CMT experiments')
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
            'embed_dim': 256,
            'num_heads': 8,
            'batch_size': 4,
            'num_epochs': 50,
            'learning_rate': 1e-4,
            'weight_decay': 0.01,
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
