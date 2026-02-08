import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix, roc_curve, auc
from config import FIGURES_DIR
# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
# TRAINING HISTORY PLOTS
def plot_training_history(
    history: Dict,
    title: str = 'Training History',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation metrics over epochs
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(history['train_loss']) + 1)
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    # Accuracy
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    # F1, Precision, Recall
    axes[1, 0].plot(epochs, history['val_precision'], 'g-', label='Precision', linewidth=2)
    axes[1, 0].plot(epochs, history['val_recall'], 'm-', label='Recall', linewidth=2)
    axes[1, 0].plot(epochs, history['val_f1'], 'c-', label='F1-Score', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision, Recall & F1')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    # AUC and Learning Rate
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    line1 = ax1.plot(epochs, history['val_auc'], 'b-', label='AUC', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('AUC', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    if 'lr' in history:
        line2 = ax2.plot(epochs, history['lr'], 'r--', label='Learning Rate', linewidth=1)
        ax2.set_ylabel('Learning Rate', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_yscale('log')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')
    
    ax1.set_title('AUC & Learning Rate')
    ax1.grid(True, alpha=0.3)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    return fig

def plot_fold_histories(
    fold_histories: List[Dict],
    metric: str = 'val_f1',
    title: str = 'Training Across Folds',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot a metric across all folds
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_histories)))
    for fold, history in enumerate(fold_histories):
        epochs = range(1, len(history[metric]) + 1)
        ax.plot(epochs, history[metric], color=colors[fold], 
                label=f'Fold {fold + 1}', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig



# EXPERIMENT COMPARISON PLOTS
def plot_experiment_comparison(
    results: List[Dict],
    metric: str = 'f1_mean',
    title: str = 'Experiment Comparison',
    top_n: int = 15,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Bar chart comparing experiments by a specific metric
    """
    # Sort by metric
    sorted_results = sorted(results, key=lambda x: x[metric], reverse=True)[:top_n]
    names = [r['experiment_name'] for r in sorted_results]
    values = [r[metric] for r in sorted_results]
    stds = [r.get(metric.replace('_mean', '_std'), 0) for r in sorted_results]
    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.4)))
    # Horizontal bar chart
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, xerr=stds, capsize=3, color='steelblue', edgecolor='navy')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    # Add value labels
    for bar, val, std in zip(bars, values, stds):
        ax.text(bar.get_width() + std + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

def plot_metric_comparison_grouped(
    results: List[Dict],
    metrics: List[str] = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean', 'auc_mean'],
    title: str = 'Multi-Metric Comparison',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Grouped bar chart comparing multiple metrics across experiments
    """
    # Limit to top experiments
    sorted_results = sorted(results, key=lambda x: x['f1_mean'], reverse=True)[:8]
    names = [r['experiment_name'][:20] for r in sorted_results]
    metric_labels = [m.replace('_mean', '').title() for m in metrics]
    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(metrics)))
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [r[metric] for r in sorted_results]
        offset = (i - len(metrics)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color)
    
    ax.set_xlabel('Experiment')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_radar_comparison(
    results: List[Dict],
    metrics: List[str] = ['accuracy_mean', 'precision_mean', 'recall_mean', 'f1_mean', 'auc_mean'],
    title: str = 'Radar Comparison',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Radar/spider chart comparing experiments across multiple metrics
    """
    # Limit to top 6 experiments
    sorted_results = sorted(results, key=lambda x: x['f1_mean'], reverse=True)[:6]
    labels = [m.replace('_mean', '').title() for m in metrics]
    num_vars = len(labels)
    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    colors = plt.cm.tab10(np.linspace(0, 1, len(sorted_results)))
    for idx, result in enumerate(sorted_results):
        values = [result[m] for m in metrics]
        values += values[:1]  # Complete the loop
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=result['experiment_name'][:25], color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title(title, y=1.08)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_fold_variance(
    results: List[Dict],
    metric: str = 'f1',
    title: str = 'Score Variance Across Folds',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Box plot showing variance across folds for each experiment
    """
    # Prepare data
    data = []
    for r in results:
        exp_name = r['experiment_name'][:25]
        for fold_result in r['fold_results']:
            data.append({
                'Experiment': exp_name,
                'Score': fold_result[metric]
            })
    df = pd.DataFrame(data)
    # Order by median score
    order = df.groupby('Experiment')['Score'].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df, x='Experiment', y='Score', order=order, ax=ax, palette='Set2')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

# CONFUSION MATRIX PLOTS
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['NORMAL', 'PNEUMONIA'],
    title: str = 'Confusion Matrix',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={'size': 14})
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

def plot_confusion_matrices_comparison(
    results: List[Dict],
    class_names: List[str] = ['NORMAL', 'PNEUMONIA'],
    title: str = 'Confusion Matrices Comparison',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrices for multiple experiments
    """
    n = min(6, len(results))
    sorted_results = sorted(results, key=lambda x: x['f1_mean'], reverse=True)[:n]
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, result in enumerate(sorted_results):
        # Average confusion matrix across folds
        cm_sum = np.zeros((2, 2))
        count = 0
        for fold_result in result['fold_results']:
            if fold_result.get('confusion_matrix') is not None:
                cm = np.array(fold_result['confusion_matrix'])
                cm_sum += cm
                count += 1
        if count > 0:
            cm_avg = cm_sum / count
        else:
            cm_avg = np.zeros((2, 2))
        
        sns.heatmap(cm_avg, annot=True, fmt='.1f', cmap='Blues', ax=axes[idx],
                    xticklabels=class_names, yticklabels=class_names)
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_title(f"{result['experiment_name'][:20]}\nF1: {result['f1_mean']:.3f}")
    # Hide unused axes
    for idx in range(n, len(axes)):
        axes[idx].axis('off')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

# ROC CURVE PLOTS
def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = 'ROC Curve',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

# SAMPLE VISUALIZATION
def plot_sample_predictions(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = ['NORMAL', 'PNEUMONIA'],
    num_samples: int = 8,
    title: str = 'Sample Predictions',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize sample predictions
    """
    num_samples = min(num_samples, len(images))
    rows = 2
    cols = num_samples // 2
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    for i in range(num_samples):
        img = images[i]
        true_label = class_names[int(y_true[i])]
        pred_label = class_names[int(y_pred[i])]
        prob = y_prob[i] if y_pred[i] == 1 else 1 - y_prob[i]
        # Denormalize if needed
        if img.max() <= 1:
            img = (img * 0.5) + 0.5
        axes[i].imshow(img.squeeze(), cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({prob:.1%})', 
                          color=color, fontsize=10)
        axes[i].axis('off')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig

def plot_misclassified_samples(
    images: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = ['NORMAL', 'PNEUMONIA'],
    num_samples: int = 8,
    title: str = 'Misclassified Samples',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize misclassified samples
    """
    # Find misclassified indices
    misclassified_idx = np.where(y_true != y_pred)[0]
    if len(misclassified_idx) == 0:
        print("No misclassified samples found!")
        return None
    num_samples = min(num_samples, len(misclassified_idx))
    rows = 2
    cols = num_samples // 2
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()
    for i, idx in enumerate(misclassified_idx[:num_samples]):
        img = images[idx]
        true_label = class_names[int(y_true[idx])]
        pred_label = class_names[int(y_pred[idx])]
        prob = y_prob[idx]
        # Denormalize if needed
        if img.max() <= 1:
            img = (img * 0.5) + 0.5
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label} ({prob:.1%})', 
                          color='red', fontsize=10, fontweight='bold')
        axes[i].axis('off')
    # Hide unused axes
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig

# COMPREHENSIVE REPORT GENERATION
def generate_experiment_report(
    results: List[Dict],
    output_dir: str = FIGURES_DIR,
    prefix: str = ''
) -> None:
    """
    Generate a comprehensive visual report of all experiments
    """
    print(f"\nGenerating experiment report in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    # 1. Overall comparison bar chart
    fig = plot_experiment_comparison(
        results, metric='f1_mean', title='Experiment Comparison (F1 Score)',
        save_path=os.path.join(output_dir, f'{prefix}comparison_f1.png')
    )
    plt.close(fig)
    # 2. Multi-metric grouped bar chart
    fig = plot_metric_comparison_grouped(
        results, title='Multi-Metric Comparison',
        save_path=os.path.join(output_dir, f'{prefix}comparison_metrics.png')
    )
    plt.close(fig)
    # 3. Radar chart
    fig = plot_radar_comparison(
        results, title='Radar Comparison (Top 6)',
        save_path=os.path.join(output_dir, f'{prefix}radar_comparison.png')
    )
    plt.close(fig)
    # 4. Fold variance box plot
    fig = plot_fold_variance(
        results, metric='f1', title='F1 Score Variance Across Folds',
        save_path=os.path.join(output_dir, f'{prefix}fold_variance.png')
    )
    plt.close(fig)
    # 5. Confusion matrices comparison
    fig = plot_confusion_matrices_comparison(
        results, title='Confusion Matrices (Top 6)',
        save_path=os.path.join(output_dir, f'{prefix}confusion_matrices.png')
    )
    plt.close(fig)
    # 6. Training histories for top experiments
    sorted_results = sorted(results, key=lambda x: x['f1_mean'], reverse=True)[:3]
    for result in sorted_results:
        if 'fold_histories' in result and result['fold_histories']:
            # Plot first fold's history as representative
            exp_name = result['experiment_name'].replace('/', '_')
            fig = plot_training_history(
                result['fold_histories'][0],
                title=f"Training History: {result['experiment_name']}",
                save_path=os.path.join(output_dir, f'{prefix}history_{exp_name}.png')
            )
            plt.close(fig)
    
    print(f"Report generated with {6 + len(sorted_results)} figures!")

# TESTING
if __name__ == "__main__":
    print("Testing visualization functions...")
    # Create dummy data for testing
    np.random.seed(42)
    dummy_results = [
        {
            'experiment_name': f'experiment_{i}',
            'accuracy_mean': 0.8 + np.random.uniform(-0.1, 0.1),
            'accuracy_std': np.random.uniform(0.01, 0.05),
            'precision_mean': 0.75 + np.random.uniform(-0.1, 0.1),
            'precision_std': np.random.uniform(0.01, 0.05),
            'recall_mean': 0.85 + np.random.uniform(-0.1, 0.1),
            'recall_std': np.random.uniform(0.01, 0.05),
            'f1_mean': 0.8 + np.random.uniform(-0.1, 0.1),
            'f1_std': np.random.uniform(0.01, 0.05),
            'auc_mean': 0.85 + np.random.uniform(-0.1, 0.1),
            'auc_std': np.random.uniform(0.01, 0.05),
            'fold_results': [
                {'f1': 0.8 + np.random.uniform(-0.05, 0.05), 
                 'confusion_matrix': [[40 + np.random.randint(-5, 5), 10 + np.random.randint(-3, 3)],
                                      [5 + np.random.randint(-2, 2), 45 + np.random.randint(-5, 5)]]}
                for _ in range(5)
            ]
        }
        for i in range(10)
    ]
    # Test bar chart
    print("  Testing bar chart...")
    fig = plot_experiment_comparison(dummy_results)
    plt.close(fig)
    # Test radar chart
    print("  Testing radar chart...")
    fig = plot_radar_comparison(dummy_results)
    plt.close(fig)
    # Test fold variance
    print("  Testing fold variance plot...")
    fig = plot_fold_variance(dummy_results)
    plt.close(fig)
    print("  All visualization tests passed!")