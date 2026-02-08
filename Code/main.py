import os
import sys
import argparse
from datetime import datetime
# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR, RESULTS_DIR, FIGURES_DIR, DEVICE,
    DEFAULT_TRAINING_CONFIG, get_config_summary
)
from dataset import PneumoniaDataset, CrossValidationHandler
from models import ModelFactory, CustomCNN, PretrainedModel
from trainer import Trainer, CrossValidationTrainer
from experiments import ExperimentRunner, ExperimentConfig, get_full_experiments
from visualization import generate_experiment_report, plot_experiment_comparison
from utils import (
    set_seed, setup_logger, save_results_csv, 
    generate_conclusions, Timer, format_time
)
def parse_args():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description='Pneumonia X-Ray Classification Experiments'
    )
    
    parser.add_argument(
        '--data_dir', type=str, default=DATA_DIR,
        help='Path to chest_xray dataset'
    )
    
    parser.add_argument(
        '--results_dir', type=str, default=RESULTS_DIR,
        help='Path to save results'
    )
    
    parser.add_argument(
        '--mode', type=str, default='full',
        choices=['quick', 'full', 'custom', 'single'],
        help='Experiment mode: quick (fast test), full (all experiments), custom, single'
    )
    
    parser.add_argument(
        '--n_folds', type=int, default=5,
        help='Number of cross-validation folds'
    )
    
    parser.add_argument(
        '--epochs', type=int, default=25,
        help='Number of training epochs per fold'
    )
    
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size'
    )
    
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--experiments', type=str, nargs='+',
        default=['activation', 'dropout', 'architecture', 'batchnorm', 'pretrained', 'optimizer'],
        help='List of experiment types to run'
    )
    
    return parser.parse_args()

def run_quick_experiments(runner: ExperimentRunner, args) -> None:
    """Run quick test experiments (reduced epochs and folds)"""
    print("\n" + "="*70)
    print("RUNNING QUICK EXPERIMENTS (Test Mode)")
    print("="*70)
    # Quick configurations - only custom CNN models for quick test
    quick_configs = [
        ExperimentConfig(
            name='quick_baseline_relu',
            model_type='custom_cnn',
            model_kwargs={
                'activation': 'relu',
                'channels': [32, 64, 128],
                'fc_sizes': [256],
                'dropout_fc': 0.5
            },
            epochs=5,
            n_folds=2
        ),
        ExperimentConfig(
            name='quick_baseline_gelu',
            model_type='custom_cnn',
            model_kwargs={
                'activation': 'gelu',
                'channels': [32, 64, 128],
                'fc_sizes': [256],
                'dropout_fc': 0.5
            },
            epochs=5,
            n_folds=2
        ),
        ExperimentConfig(
            name='quick_dropout_0.3',
            model_type='custom_cnn',
            model_kwargs={
                'activation': 'relu',
                'channels': [32, 64, 128],
                'fc_sizes': [256],
                'dropout_fc': 0.3
            },
            epochs=5,
            n_folds=2
        ),
    ]
    
    for config in quick_configs:
        runner.run_experiment(config, verbose=True)

def run_full_experiments(runner: ExperimentRunner, args) -> None:
    """Run full comprehensive experiments"""
    print("\n" + "="*70)
    print("RUNNING FULL EXPERIMENTS")
    print("="*70)
    print(f"Experiments to run: {args.experiments}")
    print(f"Epochs: {args.epochs}, Folds: {args.n_folds}")
    # Update default training config
    DEFAULT_TRAINING_CONFIG.epochs = args.epochs
    DEFAULT_TRAINING_CONFIG.n_folds = args.n_folds
    # Run requested experiments
    if 'activation' in args.experiments:
        print("\n" + "-"*50)
        print("Running Activation Function Comparison...")
        print("-"*50)
        runner.run_activation_comparison(
            activations=['relu', 'leaky_relu', 'elu', 'gelu', 'mish']
        )
    if 'dropout' in args.experiments:
        print("\n" + "-"*50)
        print("Running Dropout Rate Comparison...")
        print("-"*50)
        runner.run_dropout_comparison(
            dropout_rates=[0.0, 0.25, 0.5, 0.7]
        )
    
    if 'architecture' in args.experiments:
        print("\n" + "-"*50)
        print("Running Architecture Depth Comparison...")
        print("-"*50)
        runner.run_architecture_comparison()
    
    if 'batchnorm' in args.experiments:
        print("\n" + "-"*50)
        print("Running Batch Normalization Comparison...")
        print("-"*50)
        runner.run_batch_norm_comparison()
    
    if 'pretrained' in args.experiments:
        print("\n" + "-"*50)
        print("Running Pretrained Model Comparison...")
        print("-"*50)
        runner.run_pretrained_comparison(
            models=['resnet18', 'resnet34', 'densenet121', 'efficientnet_b0']
        )
    
    if 'optimizer' in args.experiments:
        print("\n" + "-"*50)
        print("Running Optimizer Comparison...")
        print("-"*50)
        runner.run_optimizer_comparison()
    
    if 'transfer' in args.experiments:
        print("\n" + "-"*50)
        print("Running Transfer Learning Comparison...")
        print("-"*50)
        runner.run_transfer_learning_comparison()


def run_single_experiment(runner: ExperimentRunner, args) -> None:
    """Run a single experiment for testing"""
    print("\n" + "="*70)
    print("RUNNING SINGLE EXPERIMENT")
    print("="*70)
    config = ExperimentConfig(
        name='single_test',
        model_type='custom_cnn',
        model_kwargs={
            'activation': 'relu',
            'channels': [32, 64, 128, 256],
            'fc_sizes': [512, 256],
            'dropout_fc': 0.5,
            'use_batch_norm': True
        },
        epochs=args.epochs,
        n_folds=args.n_folds,
        batch_size=args.batch_size
    )
    runner.run_experiment(config, verbose=True)


def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    # Set random seed
    set_seed(args.seed)
    # Print configuration
    print("\n" + "#"*70)
    print("#" + " "*20 + "PNEUMONIA CLASSIFICATION" + " "*20 + "#")
    print("#" + " "*15 + "Comprehensive Experiment Suite" + " "*15 + "#")
    print("#"*70)
    get_config_summary()
    print(f"\nCommand line arguments:")
    print(f"  Mode: {args.mode}")
    print(f"  Data directory: {args.data_dir}")
    print(f"  Results directory: {args.results_dir}")
    print(f"  Folds: {args.n_folds}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Seed: {args.seed}")
    # Verify data directory exists
    if not os.path.exists(args.data_dir):
        print(f"\n ERROR: Data directory not found: {args.data_dir}")
        print("Please ensure the chest_xray dataset is in the correct location.")
        return
    # Create results directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    # Initialize experiment runner
    runner = ExperimentRunner(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        device=DEVICE
    )
    # Start timer
    with Timer("Total experiment time"):
        # Run experiments based on mode
        if args.mode == 'quick':
            run_quick_experiments(runner, args)
        elif args.mode == 'full':
            run_full_experiments(runner, args)
        elif args.mode == 'single':
            run_single_experiment(runner, args)
        else:
            print(f"Unknown mode: {args.mode}")
            return
        # Check if we have results
        if not runner.all_results:
            print("\n No experiments were completed successfully.")
            return
        # Generate summary
        print("\n" + "="*70)
        print("GENERATING RESULTS SUMMARY")
        print("="*70)
        # Print summary table
        summary_df = runner.print_summary()
        # Save results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(args.results_dir, f'results_{timestamp}.csv')
        save_results_csv(runner.all_results, csv_path)
        # Generate visual report
        print("\n" + "="*70)
        print("GENERATING VISUAL REPORT")
        print("="*70)
        generate_experiment_report(
            runner.all_results,
            output_dir=FIGURES_DIR,
            prefix=f'{timestamp}_'
        )
        # Generate conclusions
        conclusions = generate_conclusions(runner.all_results)
        print(conclusions)
        # Save conclusions to file
        conclusions_path = os.path.join(args.results_dir, f'conclusions_{timestamp}.txt')
        with open(conclusions_path, 'w') as f:
            f.write(conclusions)
        print(f"\nConclusions saved to {conclusions_path}")
    print("\n" + "="*70)
    print("EXPERIMENT SUITE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nResults saved to: {args.results_dir}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
