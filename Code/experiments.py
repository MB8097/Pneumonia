import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import numpy as np
import pandas as pd
from copy import deepcopy
from config import (
    DEVICE, RESULTS_DIR, LOGS_DIR,
    DEFAULT_DATA_CONFIG, DEFAULT_TRAINING_CONFIG,
    ACTIVATIONS, DROPOUT_RATES, AUGMENTATION_STRATEGIES,
    ARCHITECTURE_CONFIGS, PRETRAINED_MODELS, OPTIMIZERS
)
from dataset import CrossValidationHandler, AugmentationFactory
from models import ModelFactory, CustomCNN, CustomResNet, PretrainedModel
from trainer import CrossValidationTrainer, MetricsCalculator
# EXPERIMENT CONFIGURATION
class ExperimentConfig:
    """Configuration for a single experiment"""
    
    def __init__(
        self,
        name: str,
        model_type: str = 'custom_cnn',
        model_kwargs: Dict = None,
        augmentation: str = 'moderate',
        optimizer: str = 'adam',
        learning_rate: float = DEFAULT_TRAINING_CONFIG.learning_rate,
        epochs: int = DEFAULT_TRAINING_CONFIG.epochs,
        batch_size: int = DEFAULT_DATA_CONFIG.batch_size,
        n_folds: int = DEFAULT_TRAINING_CONFIG.n_folds
    ):
        self.name = name
        self.model_type = model_type
        self.model_kwargs = model_kwargs or {}
        self.augmentation = augmentation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_folds = n_folds
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'model_type': self.model_type,
            'model_kwargs': self.model_kwargs,
            'augmentation': self.augmentation,
            'optimizer': self.optimizer,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'n_folds': self.n_folds
        }

# EXPERIMENT RUNNER
class ExperimentRunner:
    """
    Run experiments with cross-validation
    """
    def __init__(
        self,
        data_dir: str,
        results_dir: str = RESULTS_DIR,
        device = DEVICE
    ):
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.device = device
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        # Store all results
        self.all_results = []
        # Timestamp for this run
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"ExperimentRunner initialized")
        print(f"  Data directory: {data_dir}")
        print(f"  Results directory: {results_dir}")
        print(f"  Device: {device}")
    
    def _get_model_factory(self, model_type: str) -> Callable:
        """Get the appropriate model factory function"""
        if model_type == 'custom_cnn':
            return CustomCNN
        elif model_type == 'custom_resnet':
            return CustomResNet
        elif model_type == 'pretrained':
            return PretrainedModel
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def run_experiment(
        self,
        config: ExperimentConfig,
        verbose: bool = True
    ) -> Dict:
        start_time = time.time()
        if verbose:
            print("\n" + "#"*70)
            print(f"# EXPERIMENT: {config.name}")
            print("#"*70)
            print(f"Configuration: {config.to_dict()}")
        # Create cross-validation handler
        cv_handler = CrossValidationHandler(
            data_dir=self.data_dir,
            n_splits=config.n_folds,
            img_size=DEFAULT_DATA_CONFIG.img_size,
            grayscale=DEFAULT_DATA_CONFIG.grayscale
        )
        
        # Get model factory
        model_factory = self._get_model_factory(config.model_type)
        # Prepare model kwargs with default values
        model_kwargs = {
            'in_channels': DEFAULT_DATA_CONFIG.in_channels,
            'num_classes': 1,
            **config.model_kwargs
        }
        # Add img_size only for custom models that need it (NOT for pretrained)
        if config.model_type in ['custom_cnn', 'custom_resnet']:
            model_kwargs['img_size'] = DEFAULT_DATA_CONFIG.img_size
        # Create CV trainer
        cv_trainer = CrossValidationTrainer(
            model_factory=model_factory,
            model_kwargs=model_kwargs,
            device=self.device,
            n_folds=config.n_folds
        )
        # Run cross-validation
        for fold, train_loader, val_loader, class_weights in cv_handler.get_fold_loaders(
            batch_size=config.batch_size
        ):
            cv_trainer.train_fold(
                fold=fold,
                train_loader=train_loader,
                val_loader=val_loader,
                class_weights=class_weights,
                epochs=config.epochs,
                learning_rate=config.learning_rate,
                patience=DEFAULT_TRAINING_CONFIG.patience,
                optimizer_name=config.optimizer,
                verbose=verbose
            )
        # Get aggregated results
        aggregated = cv_trainer.get_aggregated_results()
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        # Create result entry
        result = {
            'experiment_name': config.name,
            'config': config.to_dict(),
            'accuracy_mean': aggregated['accuracy_mean'],
            'accuracy_std': aggregated['accuracy_std'],
            'precision_mean': aggregated['precision_mean'],
            'precision_std': aggregated['precision_std'],
            'recall_mean': aggregated['recall_mean'],
            'recall_std': aggregated['recall_std'],
            'f1_mean': aggregated['f1_mean'],
            'f1_std': aggregated['f1_std'],
            'auc_mean': aggregated['auc_mean'],
            'auc_std': aggregated['auc_std'],
            'specificity_mean': aggregated['specificity_mean'],
            'specificity_std': aggregated['specificity_std'],
            'fold_results': cv_trainer.fold_results,
            'fold_histories': cv_trainer.fold_histories,
            'elapsed_time': elapsed_time
        }
        self.all_results.append(result)
        if verbose:
            cv_trainer.print_summary()
            print(f"\nExperiment completed in {elapsed_time:.1f} seconds")
        # Save intermediate results
        self._save_results()
        return result
    
    def run_activation_comparison(
        self,
        activations: List[str] = None,
        base_config: Dict = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Compare different activation functions
        """
        
        if activations is None:
            activations = ACTIVATIONS
        
        if verbose:
            print("\n" + "="*70)
            print("EXPERIMENT SET: Activation Function Comparison")
            print("="*70)
        
        base = base_config or {
            'channels': [32, 64, 128, 256],
            'fc_sizes': [512, 256],
            'dropout_fc': 0.5,
            'use_batch_norm': True
        }
        
        results = []
        for activation in activations:
            config = ExperimentConfig(
                name=f'activation_{activation}',
                model_type='custom_cnn',
                model_kwargs={**base, 'activation': activation}
            )
            result = self.run_experiment(config, verbose=verbose)
            results.append(result)
        
        return results
    
    def run_dropout_comparison(
        self,
        dropout_rates: List[float] = None,
        base_config: Dict = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Compare different dropout rates
        """
        
        if dropout_rates is None:
            dropout_rates = DROPOUT_RATES
        
        if verbose:
            print("\n" + "="*70)
            print("EXPERIMENT SET: Dropout Rate Comparison")
            print("="*70)
        
        base = base_config or {
            'channels': [32, 64, 128, 256],
            'fc_sizes': [512, 256],
            'activation': 'relu',
            'use_batch_norm': True
        }
        
        results = []
        for dropout in dropout_rates:
            config = ExperimentConfig(
                name=f'dropout_{dropout}',
                model_type='custom_cnn',
                model_kwargs={**base, 'dropout_fc': dropout}
            )
            result = self.run_experiment(config, verbose=verbose)
            results.append(result)
        return results
    
    def run_architecture_comparison(
        self,
        architectures: Dict = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Compare different architecture depths
        """
        
        if architectures is None:
            architectures = ARCHITECTURE_CONFIGS
        
        if verbose:
            print("\n" + "="*70)
            print("EXPERIMENT SET: Architecture Depth Comparison")
            print("="*70)
        
        results = []
        for arch_name, arch_config in architectures.items():
            config = ExperimentConfig(
                name=f'architecture_{arch_name}',
                model_type='custom_cnn',
                model_kwargs={
                    **arch_config,
                    'activation': 'relu',
                    'dropout_fc': 0.5,
                    'use_batch_norm': True
                }
            )
            result = self.run_experiment(config, verbose=verbose)
            results.append(result)
        return results
    
    def run_batch_norm_comparison(self, verbose: bool = True) -> List[Dict]:
        """
        Compare with and without batch normalization
        """
        
        if verbose:
            print("\n" + "="*70)
            print("EXPERIMENT SET: Batch Normalization Comparison")
            print("="*70)
        
        base = {
            'channels': [32, 64, 128, 256],
            'fc_sizes': [512, 256],
            'activation': 'relu',
            'dropout_fc': 0.5
        }
        
        results = []
        for use_bn in [True, False]:
            config = ExperimentConfig(
                name=f'batch_norm_{use_bn}',
                model_type='custom_cnn',
                model_kwargs={**base, 'use_batch_norm': use_bn}
            )
            result = self.run_experiment(config, verbose=verbose)
            results.append(result)
        return results
    
    def run_pretrained_comparison(
        self,
        models: List[str] = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Compare different pretrained models
        """
        
        if models is None:
            models = PRETRAINED_MODELS
        
        if verbose:
            print("\n" + "="*70)
            print("EXPERIMENT SET: Pretrained Model Comparison")
            print("="*70)
        
        results = []
        for model_name in models:
            try:
                config = ExperimentConfig(
                    name=f'pretrained_{model_name}',
                    model_type='pretrained',
                    model_kwargs={
                        'model_name': model_name,
                        'pretrained': True,
                        'dropout': 0.5
                    },
                    learning_rate=1e-4  # Lower LR for pretrained
                )
                result = self.run_experiment(config, verbose=verbose)
                results.append(result)
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                continue
        
        return results
    
    def run_optimizer_comparison(
        self,
        optimizers: Dict = None,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Compare different optimizers
        """
        
        if optimizers is None:
            optimizers = OPTIMIZERS
        
        if verbose:
            print("\n" + "="*70)
            print("EXPERIMENT SET: Optimizer Comparison")
            print("="*70)
        
        base = {
            'channels': [32, 64, 128, 256],
            'fc_sizes': [512, 256],
            'activation': 'relu',
            'dropout_fc': 0.5,
            'use_batch_norm': True
        }
        
        results = []
        for opt_name, opt_config in optimizers.items():
            config = ExperimentConfig(
                name=f'optimizer_{opt_name}',
                model_type='custom_cnn',
                model_kwargs=base,
                optimizer=opt_name,
                learning_rate=opt_config.get('lr', 0.0001)
            )
            result = self.run_experiment(config, verbose=verbose)
            results.append(result)
        
        return results
    
    def run_transfer_learning_comparison(self, verbose: bool = True) -> List[Dict]:
        """
        Compare transfer learning strategies
        """
        
        if verbose:
            print("\n" + "="*70)
            print("EXPERIMENT SET: Transfer Learning Strategy Comparison")
            print("="*70)
        
        strategies = [
            {
                'name': 'from_scratch',
                'kwargs': {'model_name': 'resnet18', 'pretrained': False, 'dropout': 0.5},
                'lr': 1e-3
            },
            {
                'name': 'pretrained_frozen',
                'kwargs': {'model_name': 'resnet18', 'pretrained': True, 'freeze_backbone': True, 'dropout': 0.5},
                'lr': 1e-3
            },
            {
                'name': 'pretrained_finetuned',
                'kwargs': {'model_name': 'resnet18', 'pretrained': True, 'freeze_backbone': False, 'dropout': 0.5},
                'lr': 1e-4
            },
        ]
        
        results = []
        for strategy in strategies:
            config = ExperimentConfig(
                name=f'transfer_{strategy["name"]}',
                model_type='pretrained',
                model_kwargs=strategy['kwargs'],
                learning_rate=strategy['lr']
            )
            result = self.run_experiment(config, verbose=verbose)
            results.append(result)
        
        return results
    
    def _save_results(self):
        """Save current results to JSON file"""
        filepath = os.path.join(self.results_dir, f'results_{self.run_timestamp}.json')
        # Prepare serializable results
        results_to_save = []
        for r in self.all_results:
            r_copy = deepcopy(r)
            # Remove non-serializable items
            if 'fold_histories' in r_copy:
                del r_copy['fold_histories']
            # Convert numpy arrays in fold_results
            if 'fold_results' in r_copy:
                for fr in r_copy['fold_results']:
                    if 'confusion_matrix' in fr and fr['confusion_matrix'] is not None:
                        if hasattr(fr['confusion_matrix'], 'tolist'):
                            fr['confusion_matrix'] = fr['confusion_matrix'].tolist()
            results_to_save.append(r_copy)
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a pandas DataFrame
        """
        
        rows = []
        for r in self.all_results:
            rows.append({
                'Experiment': r['experiment_name'],
                'Accuracy': f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}",
                'Precision': f"{r['precision_mean']:.4f} ± {r['precision_std']:.4f}",
                'Recall': f"{r['recall_mean']:.4f} ± {r['recall_std']:.4f}",
                'F1': f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f}",
                'AUC': f"{r['auc_mean']:.4f} ± {r['auc_std']:.4f}",
                'Accuracy_Mean': r['accuracy_mean'],
                'F1_Mean': r['f1_mean'],
                'AUC_Mean': r['auc_mean'],
                'Time (s)': r['elapsed_time']
            })
        
        df = pd.DataFrame(rows)
        return df.sort_values('F1_Mean', ascending=False).reset_index(drop=True)
    
    def print_summary(self):
        """Print summary of all experiments"""
        df = self.get_results_dataframe()
        print("\n" + "="*100)
        print("EXPERIMENT SUMMARY (Sorted by F1 Score)")
        print("="*100)
        # Print formatted table
        print(df[['Experiment', 'Accuracy', 'F1', 'AUC', 'Time (s)']].to_string(index=False))
        print("\n" + "="*100)
        # Best experiment
        if len(df) > 0:
            best = df.iloc[0]
            print(f"\n BEST CONFIGURATION: {best['Experiment']}")
            print(f"   Accuracy: {best['Accuracy']}")
            print(f"   F1-Score: {best['F1']}")
            print(f"   AUC: {best['AUC']}")
        return df



# EXPERIMENT PRESETS
def get_quick_experiments() -> List[ExperimentConfig]:
    """
    Get a quick set of experiments for testing
    (Reduced epochs and folds)
    """
    
    return [
        ExperimentConfig(
            name='baseline_relu',
            model_type='custom_cnn',
            model_kwargs={'activation': 'relu', 'channels': [32, 64, 128], 'dropout_fc': 0.5},
            epochs=10,
            n_folds=3
        ),
        ExperimentConfig(
            name='baseline_gelu',
            model_type='custom_cnn',
            model_kwargs={'activation': 'gelu', 'channels': [32, 64, 128], 'dropout_fc': 0.5},
            epochs=10,
            n_folds=3
        ),
    ]
def get_full_experiments() -> List[ExperimentConfig]:
    """
    Get a comprehensive set of experiments for full comparison
    """
    experiments = []
    # Base configuration
    base_cnn = {
        'channels': [32, 64, 128, 256],
        'fc_sizes': [512, 256],
        'use_batch_norm': True
    }
    
    # 1. Activation functions
    for act in ['relu', 'leaky_relu', 'elu', 'gelu', 'mish']:
        experiments.append(ExperimentConfig(
            name=f'activation_{act}',
            model_type='custom_cnn',
            model_kwargs={**base_cnn, 'activation': act, 'dropout_fc': 0.5}
        ))
    
    # 2. Dropout rates
    for dropout in [0.0, 0.25, 0.5, 0.7]:
        experiments.append(ExperimentConfig(
            name=f'dropout_{dropout}',
            model_type='custom_cnn',
            model_kwargs={**base_cnn, 'activation': 'relu', 'dropout_fc': dropout}
        ))
    
    # 3. Architecture depths
    for arch_name, arch_config in ARCHITECTURE_CONFIGS.items():
        experiments.append(ExperimentConfig(
            name=f'arch_{arch_name}',
            model_type='custom_cnn',
            model_kwargs={**arch_config, 'activation': 'relu', 'dropout_fc': 0.5, 'use_batch_norm': True}
        ))
    
    # 4. Batch normalization
    for use_bn in [True, False]:
        experiments.append(ExperimentConfig(
            name=f'batchnorm_{use_bn}',
            model_type='custom_cnn',
            model_kwargs={**base_cnn, 'activation': 'relu', 'dropout_fc': 0.5, 'use_batch_norm': use_bn}
        ))
    
    # 5. Pretrained models
    for model_name in ['resnet18', 'resnet50', 'densenet121']:
        experiments.append(ExperimentConfig(
            name=f'pretrained_{model_name}',
            model_type='pretrained',
            model_kwargs={'model_name': model_name, 'pretrained': True, 'dropout': 0.5},
            learning_rate=1e-4
        ))
    
    return experiments

# TESTING
if __name__ == "__main__":
    from config import DATA_DIR
    print("Testing ExperimentRunner...")
    runner = ExperimentRunner(data_dir=DATA_DIR)
    # Run a quick test experiment
    config = ExperimentConfig(
        name='test_experiment',
        model_type='custom_cnn',
        model_kwargs={
            'activation': 'relu',
            'channels': [16, 32],
            'fc_sizes': [64],
            'dropout_fc': 0.5
        },
        epochs=2,
        n_folds=2
    )
    result = runner.run_experiment(config, verbose=True)
    runner.print_summary()
    print("\nExperimentRunner test completed!")