import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm
import time
import copy
from config import DEVICE, DEFAULT_TRAINING_CONFIG
# EARLY STOPPING
class EarlyStopping:
    """
    Early stopping handler to prevent overfitting
    """
    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        mode: str = 'min',
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    def __call__(self, score: float, model: nn.Module) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            return False
        
        improved = False
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.verbose:
                print(f"   Improvement! Best score: {self.best_score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"  No improvement. Patience: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def restore_best_model(self, model: nn.Module):
        """Restore model to best state"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)

# METRICS CALCULATOR
class MetricsCalculator:
    """
    Calculate and store metrics for classification
    """
    
    @staticmethod
    def calculate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict:

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'specificity': 0.0,  # Will be calculated from confusion matrix
        }
        
        # Calculate AUC if probabilities are available
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob)
            except:
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0
        
        # Calculate confusion matrix and specificity
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['confusion_matrix'] = cm
        return metrics
    
    @staticmethod
    def print_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
        """Print classification report"""
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))



# TRAINER CLASS
class Trainer:
    """
    Training handler for pneumonia classification models
    """
    def __init__(
        self,
        model: nn.Module,
        device: torch.device = DEVICE,
        learning_rate: float = DEFAULT_TRAINING_CONFIG.learning_rate,
        weight_decay: float = DEFAULT_TRAINING_CONFIG.weight_decay,
        optimizer_name: str = 'adam',
        scheduler_name: str = 'plateau',
        class_weights: Optional[np.ndarray] = None
    ):
        self.model = model.to(device)
        self.device = device
        # Loss function
        self.criterion = nn.BCELoss()
        # Optimizer
        self.optimizer = self._create_optimizer(optimizer_name, learning_rate, weight_decay)
        # Scheduler
        self.scheduler = self._create_scheduler(scheduler_name)
        self.scheduler_name = scheduler_name
        # History
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_precision': [], 'val_recall': [],
            'val_f1': [], 'val_auc': [],
            'lr': []
        }
    
    def _create_optimizer(
        self,
        name: str,
        lr: float,
        weight_decay: float
    ) -> optim.Optimizer:
        """Create optimizer"""
        if name == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        elif name == 'rmsprop':
            return optim.RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def _create_scheduler(self, name: str):
        """Create learning rate scheduler"""
        if name == 'plateau':
            # Note: 'verbose' parameter removed in PyTorch 2.0+
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=DEFAULT_TRAINING_CONFIG.scheduler_factor,
                patience=DEFAULT_TRAINING_CONFIG.scheduler_patience,
                min_lr=DEFAULT_TRAINING_CONFIG.min_lr
            )
        elif name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=DEFAULT_TRAINING_CONFIG.epochs
            )
        elif name == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        elif name == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {name}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Returns:
            Tuple of (loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)
            # Zero gradients
            self.optimizer.zero_grad()
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Gradient clipping (optional)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            # Statistics
            running_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict:
        """
        Validate the model
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        for images, labels in val_loader:
            images = images.to(self.device)
            labels_tensor = labels.float().unsqueeze(1).to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels_tensor)
            running_loss += loss.item() * images.size(0)
            probs = outputs.cpu().numpy().flatten()
            preds = (outputs > 0.5).float().cpu().numpy().flatten()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
        
        # Calculate metrics
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_prob = np.array(all_probs)
        metrics = MetricsCalculator.calculate(y_true, y_pred, y_prob)
        metrics['loss'] = running_loss / len(y_true)
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = DEFAULT_TRAINING_CONFIG.epochs,
        patience: int = DEFAULT_TRAINING_CONFIG.patience,
        verbose: bool = True
    ) -> Dict:
       
        early_stopping = EarlyStopping(patience=patience, mode='min', verbose=verbose)
        for epoch in range(epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print("-" * 40)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            # Validate
            val_metrics = self.validate(val_loader)
            # Update scheduler
            if self.scheduler is not None:
                if self.scheduler_name == 'plateau':
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['val_precision'].append(val_metrics['precision'])
            self.history['val_recall'].append(val_metrics['recall'])
            self.history['val_f1'].append(val_metrics['f1'])
            self.history['val_auc'].append(val_metrics['auc'])
            self.history['lr'].append(current_lr)
            if verbose:
                print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
                print(f"  Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
                print(f"  Learning Rate: {current_lr:.2e}")
            
            # Early stopping check
            if early_stopping(val_metrics['loss'], self.model):
                if verbose:
                    print("\nEarly stopping triggered!")
                break
        # Restore best model
        early_stopping.restore_best_model(self.model)
        return self.history
    @torch.no_grad()
    def evaluate(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval()
        all_labels = []
        all_preds = []
        all_probs = []
        for images, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(self.device)
            outputs = self.model(images)
            probs = outputs.cpu().numpy().flatten()
            preds = (outputs > 0.5).float().cpu().numpy().flatten()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)
        return np.array(all_labels), np.array(all_preds), np.array(all_probs)



# CROSS-VALIDATION TRAINER
class CrossValidationTrainer:
    """
    K-Fold Cross-Validation trainer for robust model evaluation
    """
    def __init__(
        self,
        model_factory: Callable,
        model_kwargs: Dict,
        device: torch.device = DEVICE,
        n_folds: int = DEFAULT_TRAINING_CONFIG.n_folds
    ):
        
        self.model_factory = model_factory
        self.model_kwargs = model_kwargs
        self.device = device
        self.n_folds = n_folds
        self.fold_results = []
        self.fold_histories = []
    def train_fold(
        self,
        fold: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_weights: np.ndarray,
        epochs: int = DEFAULT_TRAINING_CONFIG.epochs,
        learning_rate: float = DEFAULT_TRAINING_CONFIG.learning_rate,
        patience: int = DEFAULT_TRAINING_CONFIG.patience,
        optimizer_name: str = 'adam',
        verbose: bool = True
    ) -> Dict:
        if verbose:
            print(f"\n{'='*50}")
            print(f"FOLD {fold + 1}/{self.n_folds}")
            print(f"{'='*50}")
        # Create fresh model for each fold
        model = self.model_factory(**self.model_kwargs)
        # Create trainer
        trainer = Trainer(
            model=model,
            device=self.device,
            learning_rate=learning_rate,
            optimizer_name=optimizer_name
        )
        # Train
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            patience=patience,
            verbose=verbose
        )
        # Final evaluation on validation set
        y_true, y_pred, y_prob = trainer.evaluate(val_loader)
        metrics = MetricsCalculator.calculate(y_true, y_pred, y_prob)
        # Store results
        fold_result = {
            'fold': fold,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'auc': metrics['auc'],
            'specificity': metrics['specificity'],
            'confusion_matrix': metrics.get('confusion_matrix', None),
            'best_val_loss': min(history['val_loss']),
            'best_val_acc': max(history['val_acc']),
            'best_val_f1': max(history['val_f1']),
            'epochs_trained': len(history['train_loss'])
        }
        self.fold_results.append(fold_result)
        self.fold_histories.append(history)
        if verbose:
            print(f"\nFold {fold + 1} Results:")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1']:.4f}")
            print(f"  AUC: {metrics['auc']:.4f}")
        
        return fold_result
    
    def get_aggregated_results(self) -> Dict:
        if not self.fold_results:
            raise ValueError("No fold results available. Run training first.")
        metrics_to_aggregate = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'specificity']
        results = {}
        for metric in metrics_to_aggregate:
            values = [fold[metric] for fold in self.fold_results]
            results[f'{metric}_mean'] = np.mean(values)
            results[f'{metric}_std'] = np.std(values)
            results[f'{metric}_values'] = values
        # Additional stats
        results['n_folds'] = len(self.fold_results)
        results['epochs_trained'] = [fold['epochs_trained'] for fold in self.fold_results]
        return results
    
    def print_summary(self):
        """Print summary of cross-validation results"""
        results = self.get_aggregated_results()
        print("\n" + "="*60)
        print("CROSS-VALIDATION SUMMARY")
        print("="*60)
        print(f"Number of folds: {results['n_folds']}")
        print(f"\nMetrics (mean ± std):")
        print(f"  Accuracy:    {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
        print(f"  Precision:   {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
        print(f"  Recall:      {results['recall_mean']:.4f} ± {results['recall_std']:.4f}")
        print(f"  F1-Score:    {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
        print(f"  AUC:         {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
        print(f"  Specificity: {results['specificity_mean']:.4f} ± {results['specificity_std']:.4f}")
        print("="*60)
        return results



# UTILITY FUNCTIONS
def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    history: Dict,
    filepath: str
):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: torch.device = DEVICE
) -> Tuple[nn.Module, Optional[optim.Optimizer], int, Dict]:
    """Load model checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    history = checkpoint['history']
    print(f"Checkpoint loaded from {filepath} (epoch {epoch})")
    return model, optimizer, epoch, history


# TESTING
if __name__ == "__main__":
    from models import CustomCNN
    from dataset import PneumoniaDataset
    print("Testing Trainer...")
    # Create a small model for testing
    model = CustomCNN(channels=[16, 32], fc_sizes=[64])
    # Create dummy data
    dataset = PneumoniaDataset(augmentation_strategy='light')
    train_loader, val_loader, test_loader, class_weights = dataset.get_data_loaders(batch_size=16)
    # Create trainer
    trainer = Trainer(model=model, learning_rate=0.001)
    # Train for a few epochs
    print("\nTraining for 2 epochs...")
    history = trainer.train(train_loader, val_loader, epochs=2, patience=5)
    # Evaluate
    print("\nEvaluating...")
    y_true, y_pred, y_prob = trainer.evaluate(test_loader)
    metrics = MetricsCalculator.calculate(y_true, y_pred, y_prob)
    print(f"\nTest Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")
    print(f"  AUC: {metrics['auc']:.4f}")
    print("\nTrainer test completed!")