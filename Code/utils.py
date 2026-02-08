import os
import json
import random
import numpy as np
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from config import RESULTS_DIR, LOGS_DIR



# REPRODUCIBILITY
def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

# LOGGING
def setup_logger(
    name: str = 'pneumonia_classification',
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logger with console and file handlers
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Clear existing handlers
    logger.handlers = []
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# FILE I/O
def save_json(data: Any, filepath: str):
    """Save data to JSON file"""
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    data = convert_numpy(data)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Data saved to {filepath}")

def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def save_results_csv(results: List[Dict], filepath: str):
    """Save results to CSV file"""
    import pandas as pd
    rows = []
    for r in results:
        rows.append({
            'Experiment': r['experiment_name'],
            'Accuracy_Mean': r['accuracy_mean'],
            'Accuracy_Std': r['accuracy_std'],
            'Precision_Mean': r['precision_mean'],
            'Precision_Std': r['precision_std'],
            'Recall_Mean': r['recall_mean'],
            'Recall_Std': r['recall_std'],
            'F1_Mean': r['f1_mean'],
            'F1_Std': r['f1_std'],
            'AUC_Mean': r['auc_mean'],
            'AUC_Std': r['auc_std'],
            'Time_Seconds': r.get('elapsed_time', 0)
        })
    df = pd.DataFrame(rows)
    df = df.sort_values('F1_Mean', ascending=False)
    df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")
    return df


# MODEL UTILITIES
def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable
    }


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Get model size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_summary(model: torch.nn.Module, input_size: tuple = (1, 1, 150, 150)):
    """
    Print a summary of the model architecture
    """
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    print(model)
    params = count_parameters(model)
    size_mb = get_model_size_mb(model)
    print("\n" + "-"*60)
    print(f"Total Parameters: {params['total']:,}")
    print(f"Trainable Parameters: {params['trainable']:,}")
    print(f"Non-trainable Parameters: {params['non_trainable']:,}")
    print(f"Model Size: {size_mb:.2f} MB")
    print("="*60)



# TIMING UTILITIES
class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        self.elapsed = None
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start_time).total_seconds()
        if self.name:
            print(f"[{self.name}] Elapsed time: {self.elapsed:.2f} seconds")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"



# STATISTICAL UTILITIES
def calculate_confidence_interval(values: List[float], confidence: float = 0.95) -> tuple:
    """
    Calculate confidence interval for a list of values
    """
    from scipy import stats
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h

def perform_statistical_test(
    values1: List[float],
    values2: List[float],
    test: str = 'ttest'
) -> Dict:
    """
    Perform statistical significance test between two sets of results
    """
    from scipy import stats
    if test == 'ttest':
        statistic, p_value = stats.ttest_ind(values1, values2)
        test_name = "Independent t-test"
    elif test == 'paired_ttest':
        statistic, p_value = stats.ttest_rel(values1, values2)
        test_name = "Paired t-test"
    elif test == 'wilcoxon':
        statistic, p_value = stats.wilcoxon(values1, values2)
        test_name = "Wilcoxon signed-rank test"
    elif test == 'mannwhitney':
        statistic, p_value = stats.mannwhitneyu(values1, values2)
        test_name = "Mann-Whitney U test"
    else:
        raise ValueError(f"Unknown test: {test}")
    return {
        'test': test_name,
        'statistic': statistic,
        'p_value': p_value,
        'significant_0.05': p_value < 0.05,
        'significant_0.01': p_value < 0.01
    }

def compare_experiments_statistically(
    results1: Dict,
    results2: Dict,
    metric: str = 'f1'
) -> Dict:
    """
    Compare two experiments using statistical tests
    """
    
    values1 = [fold[metric] for fold in results1['fold_results']]
    values2 = [fold[metric] for fold in results2['fold_results']]
    
    comparison = {
        'experiment1': results1['experiment_name'],
        'experiment2': results2['experiment_name'],
        'metric': metric,
        'mean1': np.mean(values1),
        'mean2': np.mean(values2),
        'std1': np.std(values1),
        'std2': np.std(values2),
    }
    
    # Perform paired t-test (since we're comparing same folds)
    test_result = perform_statistical_test(values1, values2, test='paired_ttest')
    comparison.update(test_result)
    
    return comparison

# RESULTS ANALYSIS
def analyze_experiment_category(
    results: List[Dict],
    category_key: str,
    metric: str = 'f1_mean'
) -> Dict:
    """
    Analyze results grouped by a category (e.g., activation function, dropout rate)
    """
    
    # Group results by category
    grouped = {}
    for r in results:
        # Extract category from experiment name or config
        exp_name = r['experiment_name']
        if category_key in exp_name:
            # Parse category value from name
            parts = exp_name.split('_')
            for i, part in enumerate(parts):
                if part == category_key.split('_')[-1] and i + 1 < len(parts):
                    category_value = parts[i + 1]
                    break
            else:
                category_value = exp_name
        else:
            category_value = r.get('config', {}).get(category_key, 'unknown')
        
        if category_value not in grouped:
            grouped[category_value] = []
        grouped[category_value].append(r[metric])
    
    # Calculate statistics for each category
    analysis = {}
    for category, values in grouped.items():
        analysis[category] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    return analysis


def generate_conclusions(results: List[Dict]) -> str:
    """
    Generate text conclusions from experiment results
    """
    if not results:
        return "No results available for analysis."
    
    # Sort by F1 score
    sorted_results = sorted(results, key=lambda x: x['f1_mean'], reverse=True)
    best = sorted_results[0]
    worst = sorted_results[-1]
    conclusions = []
    conclusions.append("="*60)
    conclusions.append("EXPERIMENT CONCLUSIONS")
    conclusions.append("="*60)
    # Best configuration
    conclusions.append(f"\n BEST CONFIGURATION: {best['experiment_name']}")
    conclusions.append(f"   • F1 Score: {best['f1_mean']:.4f} ± {best['f1_std']:.4f}")
    conclusions.append(f"   • Accuracy: {best['accuracy_mean']:.4f} ± {best['accuracy_std']:.4f}")
    conclusions.append(f"   • AUC: {best['auc_mean']:.4f} ± {best['auc_std']:.4f}")
    conclusions.append(f"   • Recall: {best['recall_mean']:.4f} (important for medical diagnosis)")
    # Worst configuration
    conclusions.append(f"\n WORST CONFIGURATION: {worst['experiment_name']}")
    conclusions.append(f"   • F1 Score: {worst['f1_mean']:.4f} ± {worst['f1_std']:.4f}")
    # Performance improvement
    improvement = ((best['f1_mean'] - worst['f1_mean']) / worst['f1_mean']) * 100
    conclusions.append(f"\n Performance Range:")
    conclusions.append(f"   • Best vs Worst F1 improvement: {improvement:.1f}%")
    # Analyze by category if possible
    conclusions.append("\n KEY FINDINGS:")
    # Find patterns in experiment names
    activation_results = [r for r in results if 'activation' in r['experiment_name'].lower()]
    if activation_results:
        best_act = max(activation_results, key=lambda x: x['f1_mean'])
        conclusions.append(f"   • Best activation function: {best_act['experiment_name']} (F1: {best_act['f1_mean']:.4f})")
    
    dropout_results = [r for r in results if 'dropout' in r['experiment_name'].lower()]
    if dropout_results:
        best_drop = max(dropout_results, key=lambda x: x['f1_mean'])
        conclusions.append(f"   • Best dropout configuration: {best_drop['experiment_name']} (F1: {best_drop['f1_mean']:.4f})")
    
    pretrained_results = [r for r in results if 'pretrained' in r['experiment_name'].lower()]
    if pretrained_results:
        best_pre = max(pretrained_results, key=lambda x: x['f1_mean'])
        conclusions.append(f"   • Best pretrained model: {best_pre['experiment_name']} (F1: {best_pre['f1_mean']:.4f})")
    
    arch_results = [r for r in results if 'arch' in r['experiment_name'].lower()]
    if arch_results:
        best_arch = max(arch_results, key=lambda x: x['f1_mean'])
        conclusions.append(f"   • Best architecture: {best_arch['experiment_name']} (F1: {best_arch['f1_mean']:.4f})")
    
    # Recommendations
    conclusions.append("\n💡 RECOMMENDATIONS:")
    conclusions.append(f"   1. Use configuration: {best['experiment_name']}")
    if best['recall_mean'] < 0.9:
        conclusions.append("   2. Consider optimizing for higher recall (critical for medical diagnosis)")
    if best['f1_std'] > 0.05:
        conclusions.append("   3. High variance observed - consider more training data or regularization")
    conclusions.append("\n" + "="*60)
    
    return "\n".join(conclusions)


# TESTING
if __name__ == "__main__":
    print("Testing utility functions...")
    # Test seed setting
    set_seed(42)
    # Test timer
    with Timer("Test operation"):
        import time
        time.sleep(0.5)
    # Test time formatting
    print(f"Format 45 seconds: {format_time(45)}")
    print(f"Format 125 seconds: {format_time(125)}")
    print(f"Format 3725 seconds: {format_time(3725)}")
    # Test confidence interval
    values = [0.85, 0.87, 0.83, 0.86, 0.84]
    ci = calculate_confidence_interval(values)
    print(f"95% CI for {values}: ({ci[0]:.4f}, {ci[1]:.4f})")
    # Test statistical test
    values1 = [0.85, 0.87, 0.83, 0.86, 0.84]
    values2 = [0.80, 0.82, 0.78, 0.81, 0.79]
    test_result = perform_statistical_test(values1, values2)
    print(f"Statistical test: {test_result}")
    print("\nAll utility tests passed!")