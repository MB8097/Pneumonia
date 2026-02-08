import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'chest_xray')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'trained_model.pth')
# Training parameters
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0001
PATIENCE = 7
# Model configuration
MODEL_CONFIG = {
    'activation': 'relu',
    'channels': [32, 64, 128, 256],
    'fc_sizes': [512, 256],
    'dropout_fc': 0.5,
    'use_batch_norm': True
}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# MODEL (same as in run.py)
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
        def get_activation():
            activations = {
                'relu': nn.ReLU(inplace=True),
                'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
                'elu': nn.ELU(inplace=True),
                'gelu': nn.GELU(),
                'mish': nn.Mish(inplace=True),
            }
            return activations.get(activation, nn.ReLU(inplace=True))
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
        self._to_linear = self._get_conv_output_size(in_channels, IMG_SIZE)
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
def get_data_loaders():
    """Create data loaders with augmentation"""
    train_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'train'), 
        transform=train_transforms
    )
    val_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'val'), 
        transform=val_transforms
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'test'), 
        transform=val_transforms
    )
    
    # Class weights for imbalanced data
    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.bincount(targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Class distribution: NORMAL={class_counts[0]}, PNEUMONIA={class_counts[1]}")
    return train_loader, val_loader, test_loader

# TRAINING
def train_model():
    """Train the model and save it"""
    print("="*60)
    print("TRAINING PNEUMONIA CLASSIFICATION MODEL")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Configuration: {MODEL_CONFIG}")
    # Load data
    train_loader, val_loader, test_loader = get_data_loaders()
    # Create model
    model = PneumoniaCNN(in_channels=1, num_classes=1, **MODEL_CONFIG).to(DEVICE)
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}')
        for images, labels in pbar:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.float().unsqueeze(1).to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                predicted = (outputs > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        # Update scheduler
        scheduler.step(val_loss)
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print("   New best model!")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered!")
                break
    # Restore best model
    model.load_state_dict(best_model_state)
    # Final evaluation on test set
    print("\n" + "="*60)
    print("FINAL EVALUATION ON TEST SET")
    print("="*60)
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.float().unsqueeze(1).to(DEVICE)
            outputs = model(images)
            predicted = (outputs > 0.5).float()
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    # Save model
    checkpoint = {
        'model_state_dict': best_model_state,
        'model_config': MODEL_CONFIG,
        'history': history,
        'test_accuracy': test_acc,
        'img_size': IMG_SIZE
    }
    torch.save(checkpoint, MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")
    return model, history
if __name__ == '__main__':
    train_model()