import os
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'chest_xray')
MODEL_PATH = os.path.join(BASE_DIR, 'trained_model.pth')
# Model configuration (must match the trained model)
IMG_SIZE = 150
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
# MODEL DEFINITION (same as in models.py)
class PneumoniaCNN(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_classes=1,
        activation='relu',
        channels=[32, 64, 128, 256],
        fc_sizes=[512, 256],
        dropout_fc=0.5,
        use_batch_norm=True
    ):
        super(PneumoniaCNN, self).__init__()
        
        # Activation function
        def get_activation():
            activations = {
                'relu': nn.ReLU(inplace=True),
                'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
                'elu': nn.ELU(inplace=True),
                'gelu': nn.GELU(),
                'mish': nn.Mish(inplace=True),
            }
            return activations.get(activation, nn.ReLU(inplace=True))
        
        # Build convolutional layers
        conv_layers = []
        curr_channels = in_channels
        for out_channels in channels:
            conv_layers.append(nn.Conv2d(curr_channels, out_channels, kernel_size=3, padding=1))
            if use_batch_norm:
                conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(get_activation())
            conv_layers.append(nn.MaxPool2d(2, 2))
            curr_channels = out_channels
        self.conv_layers = nn.Sequential(*conv_layers)
        # Calculate flattened size
        self._to_linear = self._get_conv_output_size(in_channels, IMG_SIZE)
        # Build fully connected layers
        fc_layers = []
        fc_input = self._to_linear
        for fc_size in fc_sizes:
            fc_layers.extend([
                nn.Linear(fc_input, fc_size),
                get_activation(),
                nn.Dropout(dropout_fc)
            ])
            fc_input = fc_size
        fc_layers.append(nn.Linear(fc_input, num_classes))
        fc_layers.append(nn.Sigmoid())
        self.fc_layers = nn.Sequential(*fc_layers)
    def _get_conv_output_size(self, in_channels, img_size):
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, img_size, img_size)
            dummy = self.conv_layers(dummy)
            return dummy.view(1, -1).size(1)
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# DATA LOADING
def get_transforms():
    """Get image transforms for inference"""
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

def load_test_data():
    """Load test dataset"""
    test_dir = os.path.join(DATA_DIR, 'test')
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found at {test_dir}")
        print("Please ensure the chest_xray dataset is in the 'data' folder")
        sys.exit(1)
    test_dataset = datasets.ImageFolder(
        root=test_dir,
        transform=get_transforms()
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    return test_loader, test_dataset

# MODEL LOADING
def load_model():
    """Load the trained model"""
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Trained model not found at {MODEL_PATH}")
        print("\nTo train a model, run: python main.py --mode quick")
        print("Then save the model using the provided training scripts.")
        sys.exit(1)
    # Load checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    # Get model configuration from checkpoint
    model_config = checkpoint.get('model_config', {
        'activation': 'relu',
        'channels': [32, 64, 128, 256],
        'fc_sizes': [512, 256],
        'dropout_fc': 0.5,
        'use_batch_norm': True
    })
    # Create model
    model = PneumoniaCNN(
        in_channels=1,
        num_classes=1,
        **model_config
    )
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")
    print(f"Model configuration: {model_config}")
    return model, model_config

# EVALUATION
def evaluate_model(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    print("\nEvaluating model on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = outputs.cpu().numpy().flatten()
            preds = (outputs > 0.5).float().cpu().numpy().flatten()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)

def print_results(y_true, y_pred):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 NORMAL  PNEUMONIA")
    print(f"Actual NORMAL    {cm[0][0]:6d}  {cm[0][1]:6d}")
    print(f"Actual PNEUMONIA {cm[1][0]:6d}  {cm[1][1]:6d}")
    return cm

def plot_results(y_true, y_pred, y_prob, test_dataset):
    """Generate and save visualization plots"""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    axes[0, 0].set_title('Confusion Matrix')
    # 2. Class Distribution in Test Set
    unique, counts = np.unique(y_true, return_counts=True)
    axes[0, 1].bar(CLASS_NAMES, counts, color=['green', 'red'], edgecolor='black')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Test Set Class Distribution')
    for i, (name, count) in enumerate(zip(CLASS_NAMES, counts)):
        axes[0, 1].text(i, count + 5, str(count), ha='center', fontweight='bold')
    # 3. Prediction Confidence Distribution
    axes[1, 0].hist(y_prob[y_true == 0], bins=30, alpha=0.7, label='NORMAL', color='green')
    axes[1, 0].hist(y_prob[y_true == 1], bins=30, alpha=0.7, label='PNEUMONIA', color='red')
    axes[1, 0].axvline(x=0.5, color='black', linestyle='--', label='Threshold (0.5)')
    axes[1, 0].set_xlabel('Prediction Probability (Pneumonia)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Prediction Confidence Distribution')
    axes[1, 0].legend()
    # 4. Metrics Summary
    axes[1, 1].axis('off')
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = 0.0
    # Specificity (True Negative Rate)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    metrics_text = f"""
    ╔══════════════════════════════════════╗
    ║     PERFORMANCE METRICS SUMMARY      ║
    ╠══════════════════════════════════════╣
    ║  Accuracy:    {accuracy:>6.2%}               ║
    ║  Precision:   {precision:>6.2%}               ║
    ║  Recall:      {recall:>6.2%}               ║
    ║  F1-Score:    {f1:>6.2%}               ║
    ║  Specificity: {specificity:>6.2%}               ║
    ║  AUC-ROC:     {auc:>6.4f}               ║
    ╠══════════════════════════════════════╣
    ║  True Positives:  {tp:>5}              ║
    ║  True Negatives:  {tn:>5}              ║
    ║  False Positives: {fp:>5}              ║
    ║  False Negatives: {fn:>5}              ║
    ╚══════════════════════════════════════╝
    """
    axes[1, 1].text(0.1, 0.5, metrics_text, fontsize=12, fontfamily='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Performance Metrics')
    plt.suptitle('Pneumonia Classification - Test Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    # Save figure
    output_path = os.path.join(BASE_DIR, 'results_demo.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults visualization saved to: {output_path}")
    plt.show()

def visualize_sample_predictions(model, test_dataset, num_samples=8):
    """Visualize sample predictions"""
    print("\nGenerating sample predictions visualization...")
    # Get random samples
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    model.eval()
    for i, idx in enumerate(indices):
        image, true_label = test_dataset[idx]
        # Get prediction
        with torch.no_grad():
            image_tensor = image.unsqueeze(0).to(DEVICE)
            output = model(image_tensor)
            prob = output.item()
            pred_label = 1 if prob > 0.5 else 0
        # Convert image for display
        img_np = image.squeeze().numpy()
        img_np = (img_np * 0.5) + 0.5  # Denormalize
        # Plot
        axes[i].imshow(img_np, cmap='gray')
        true_name = CLASS_NAMES[true_label]
        pred_name = CLASS_NAMES[pred_label]
        confidence = prob if pred_label == 1 else 1 - prob
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(
            f'True: {true_name}\nPred: {pred_name} ({confidence:.1%})',
            color=color, fontsize=10
        )
        axes[i].axis('off')
    
    plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    # Save
    output_path = os.path.join(BASE_DIR, 'sample_predictions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Sample predictions saved to: {output_path}")
    plt.show()

# MAIN
def main():
    print("="*60)
    print("   PNEUMONIA X-RAY CLASSIFICATION - DEMO")
    print("="*60)
    print(f"\nDevice: {DEVICE}")
    print(f"Model path: {MODEL_PATH}")
    print(f"Data directory: {DATA_DIR}")
    # Load model
    model, model_config = load_model()
    # Load test data
    test_loader, test_dataset = load_test_data()
    print(f"\nTest set: {len(test_dataset)} images")
    # Evaluate
    y_true, y_pred, y_prob = evaluate_model(model, test_loader)
    # Print results
    cm = print_results(y_true, y_pred)
    # Plot results
    plot_results(y_true, y_pred, y_prob, test_dataset)
    # Show sample predictions
    visualize_sample_predictions(model, test_dataset)
    print("\n" + "="*60)
    print("   DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == '__main__':
    main()