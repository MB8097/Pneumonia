import os
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional
# PATH CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'chest_xray')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')
# Create directories if they don't exist
for dir_path in [RESULTS_DIR, FIGURES_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)
# DEVICE CONFIGURATION
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 2 if torch.cuda.is_available() else 0
PIN_MEMORY = torch.cuda.is_available()
# DATA CONFIGURATION
@dataclass
class DataConfig:
    img_size: int = 150
    batch_size: int = 32
    num_classes: int = 2
    class_names: List[str] = field(default_factory=lambda: ['NORMAL', 'PNEUMONIA'])
    grayscale: bool = True
    in_channels: int = 1
# TRAINING CONFIGURATION
@dataclass
class TrainingConfig:
    epochs: int = 25
    learning_rate: float = 0.0001
    weight_decay: float = 1e-5
    patience: int = 7
    min_delta: float = 0.001
    # Learning rate scheduler
    scheduler_factor: float = 0.2
    scheduler_patience: int = 3
    min_lr: float = 1e-7
    # Cross-validation
    n_folds: int = 5
    random_state: int = 42
# MODEL CONFIGURATIONS
@dataclass
class ModelConfig:
    """Base model configuration"""
    name: str = "CustomCNN"
    activation: str = "relu"
    dropout: float = 0.5
    use_batch_norm: bool = True
@dataclass 
class CNNConfig(ModelConfig):
    """Custom CNN configuration"""
    channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    fc_sizes: List[int] = field(default_factory=lambda: [512, 256])
    dropout_conv: float = 0.0
@dataclass
class PretrainedConfig(ModelConfig):
    """Pretrained model configuration"""
    model_name: str = "resnet18"
    pretrained: bool = True
    freeze_backbone: bool = False
# EXPERIMENT CONFIGURATIONS
# Activation functions to compare
ACTIVATIONS = ['relu', 'leaky_relu', 'elu', 'gelu', 'mish', 'selu']
# Dropout rates to compare
DROPOUT_RATES = [0.0, 0.25, 0.5, 0.7]
# Augmentation strategies to compare
AUGMENTATION_STRATEGIES = ['none', 'light', 'moderate', 'aggressive', 'medical']
# Architecture depths to compare
ARCHITECTURE_CONFIGS = {
    'shallow': {'channels': [32, 64], 'fc_sizes': [256]},
    'medium': {'channels': [32, 64, 128], 'fc_sizes': [512, 256]},
    'deep': {'channels': [32, 64, 128, 256], 'fc_sizes': [512, 256]},
    'very_deep': {'channels': [32, 64, 128, 256, 512], 'fc_sizes': [512, 256, 128]},
}
# Pretrained models to compare
PRETRAINED_MODELS = ['resnet18', 'resnet34', 'resnet50', 'densenet121', 'efficientnet_b0']
# Optimizers to compare
OPTIMIZERS = {
    'adam': {'lr': 0.0001},
    'adamw': {'lr': 0.0001, 'weight_decay': 0.01},
    'sgd': {'lr': 0.001, 'momentum': 0.9},
}
# DEFAULT CONFIGURATIONS
DEFAULT_DATA_CONFIG = DataConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_MODEL_CONFIG = CNNConfig()
def get_config_summary():
    """Print configuration summary"""
    print("=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"\nData Config:")
    print(f"  Image Size: {DEFAULT_DATA_CONFIG.img_size}")
    print(f"  Batch Size: {DEFAULT_DATA_CONFIG.batch_size}")
    print(f"  Grayscale: {DEFAULT_DATA_CONFIG.grayscale}")
    print(f"\nTraining Config:")
    print(f"  Epochs: {DEFAULT_TRAINING_CONFIG.epochs}")
    print(f"  Learning Rate: {DEFAULT_TRAINING_CONFIG.learning_rate}")
    print(f"  K-Folds: {DEFAULT_TRAINING_CONFIG.n_folds}")
    print(f"  Early Stopping Patience: {DEFAULT_TRAINING_CONFIG.patience}")
    print("=" * 60)
if __name__ == "__main__":
    get_config_summary()