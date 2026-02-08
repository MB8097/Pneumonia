import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedKFold
from typing import Tuple, List, Dict, Optional, Generator
from config import DEFAULT_DATA_CONFIG, DATA_DIR, NUM_WORKERS, PIN_MEMORY
class AugmentationFactory:
    """Factory class to create different augmentation strategies"""
    
    @staticmethod
    def get_transforms(
        strategy: str = 'moderate',
        img_size: int = 150,
        grayscale: bool = True
    ) -> Tuple[transforms.Compose, transforms.Compose]:
        # Base transforms (always applied)
        base = []
        if grayscale:
            base.append(transforms.Grayscale(num_output_channels=1))
        base.append(transforms.Resize((img_size, img_size)))
        # Normalization
        if grayscale:
            normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])
        
        # Final transforms (always applied)
        final = [transforms.ToTensor(), normalize]
        # Validation transform (no augmentation)
        val_transform = transforms.Compose(base + final)
        # Augmentation based on strategy
        if strategy == 'none':
            aug = []
        elif strategy == 'light':
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        elif strategy == 'moderate':
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    shear=10
                ),
            ]
        elif strategy == 'aggressive':
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.1),
                transforms.RandomRotation(20),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),
                    shear=20,
                    scale=(0.8, 1.2)
                ),
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
        elif strategy == 'medical':
            # Conservative augmentation suitable for medical imaging
            aug = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),
                    shear=5
                ),
                transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
                transforms.RandomAutocontrast(p=0.3),
            ]
        else:
            raise ValueError(f"Unknown augmentation strategy: {strategy}")
        
        train_transform = transforms.Compose(base + aug + final)
        return train_transform, val_transform

class PneumoniaDataset:
    """
    Handler for the Pneumonia dataset with support for different splits and augmentation
    """
    
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        img_size: int = DEFAULT_DATA_CONFIG.img_size,
        grayscale: bool = DEFAULT_DATA_CONFIG.grayscale,
        augmentation_strategy: str = 'moderate'
    ):
        self.data_dir = data_dir
        self.img_size = img_size
        self.grayscale = grayscale
        self.augmentation_strategy = augmentation_strategy
        # Get transforms
        self.train_transform, self.val_transform = AugmentationFactory.get_transforms(
            strategy=augmentation_strategy,
            img_size=img_size,
            grayscale=grayscale
        )
        # Load datasets
        self._load_datasets()
        
    def _load_datasets(self):
        """Load train, validation, and test datasets"""
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        test_dir = os.path.join(self.data_dir, 'test')
        # Training data with augmentation
        self.train_dataset = datasets.ImageFolder(
            root=train_dir, 
            transform=self.train_transform
        )
        # Validation data without augmentation
        self.val_dataset = datasets.ImageFolder(
            root=val_dir, 
            transform=self.val_transform
        )
        # Test data without augmentation
        self.test_dataset = datasets.ImageFolder(
            root=test_dir, 
            transform=self.val_transform
        )
        # Store class info
        self.classes = self.train_dataset.classes
        self.class_to_idx = self.train_dataset.class_to_idx
        # Print info
        print(f"Dataset loaded from: {self.data_dir}")
        print(f"  Classes: {self.classes}")
        print(f"  Training samples: {len(self.train_dataset)}")
        print(f"  Validation samples: {len(self.val_dataset)}")
        print(f"  Test samples: {len(self.test_dataset)}")
    def get_combined_train_val(self) -> Tuple[ConcatDataset, np.ndarray]:
        # Create datasets with validation transform for proper CV
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        train_data = datasets.ImageFolder(root=train_dir, transform=self.val_transform)
        val_data = datasets.ImageFolder(root=val_dir, transform=self.val_transform)
        combined = ConcatDataset([train_data, val_data])
        # Get all labels
        labels = np.array(
            [sample[1] for sample in train_data.samples] + 
            [sample[1] for sample in val_data.samples]
        )
        
        return combined, labels
    
    def get_class_weights(self, dataset=None) -> Tuple[np.ndarray, np.ndarray]:
        if dataset is None:
            dataset = self.train_dataset
            
        if hasattr(dataset, 'samples'):
            targets = [sample[1] for sample in dataset.samples]
        else:
            # For Subset or ConcatDataset
            targets = []
            for i in range(len(dataset)):
                _, label = dataset[i]
                targets.append(label)
        
        targets = np.array(targets)
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_weights)
        sample_weights = class_weights[targets]
        print(f"  Class distribution: {dict(zip(self.classes, class_counts))}")
        print(f"  Class weights: {dict(zip(self.classes, class_weights))}")
        return sample_weights, class_weights

    def get_data_loaders(
        self,
        batch_size: int = DEFAULT_DATA_CONFIG.batch_size,
        use_weighted_sampler: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        # Get class weights
        sample_weights, class_weights = self.get_class_weights(self.train_dataset)
        # Create sampler for training
        if use_weighted_sampler:
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        
        return train_loader, val_loader, test_loader, class_weights


class CrossValidationHandler:
    def __init__(
        self,
        data_dir: str = DATA_DIR,
        n_splits: int = 5,
        random_state: int = 42,
        img_size: int = DEFAULT_DATA_CONFIG.img_size,
        grayscale: bool = DEFAULT_DATA_CONFIG.grayscale
    ):
        self.data_dir = data_dir
        self.n_splits = n_splits
        self.random_state = random_state
        self.img_size = img_size
        self.grayscale = grayscale
        # Initialize stratified k-fold
        self.skf = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=random_state
        )
        # Load combined data
        self._load_combined_data()
        
    def _load_combined_data(self):
        """Load and combine train+val data for CV"""
        
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'val')
        
        # Basic transform for data loading (augmentation applied dynamically)
        basic_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1) if self.grayscale else transforms.Lambda(lambda x: x),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) if self.grayscale else 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_data = datasets.ImageFolder(root=train_dir, transform=basic_transform)
        val_data = datasets.ImageFolder(root=val_dir, transform=basic_transform)
        
        self.full_dataset = ConcatDataset([train_data, val_data])
        self.labels = np.array(
            [sample[1] for sample in train_data.samples] + 
            [sample[1] for sample in val_data.samples]
        )
        
        self.classes = train_data.classes
        self.class_to_idx = train_data.class_to_idx
        
        print(f"Cross-validation dataset: {len(self.full_dataset)} samples")
        print(f"  Class distribution: {np.bincount(self.labels)}")
        
    def get_fold_loaders(
        self,
        batch_size: int = DEFAULT_DATA_CONFIG.batch_size,
        use_weighted_sampler: bool = True
    ) -> Generator[Tuple[int, DataLoader, DataLoader, np.ndarray], None, None]:
        
        indices = np.arange(len(self.labels))
        
        for fold, (train_idx, val_idx) in enumerate(self.skf.split(indices, self.labels)):
            print(f"\n  Fold {fold + 1}/{self.n_splits}")
            print(f"    Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
            # Create subsets
            train_subset = Subset(self.full_dataset, train_idx)
            val_subset = Subset(self.full_dataset, val_idx)
            # Calculate class weights for this fold
            fold_labels = self.labels[train_idx]
            class_counts = np.bincount(fold_labels)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum() * len(class_weights)
            # Create weighted sampler
            if use_weighted_sampler:
                sample_weights = class_weights[fold_labels]
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
            else:
                sampler = None
            
            # Create loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=False if sampler else True,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY
            )
            
            yield fold, train_loader, val_loader, class_weights
    
    def get_test_loader(self, batch_size: int = DEFAULT_DATA_CONFIG.batch_size) -> DataLoader:
        """Get test data loader"""
        
        test_dir = os.path.join(self.data_dir, 'test')
        
        test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1) if self.grayscale else transforms.Lambda(lambda x: x),
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) if self.grayscale else 
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY
        )
        
        return test_loader



# UTILITY FUNCTIONS
def visualize_augmentations(data_dir: str = DATA_DIR, img_size: int = 150):
    """Visualize different augmentation strategies on sample images"""
    import matplotlib.pyplot as plt
    
    strategies = ['none', 'light', 'moderate', 'aggressive', 'medical']
    
    # Load a sample image
    train_dir = os.path.join(data_dir, 'train', 'PNEUMONIA')
    sample_img_path = os.path.join(train_dir, os.listdir(train_dir)[0])
    
    fig, axes = plt.subplots(len(strategies), 5, figsize=(15, 3*len(strategies)))
    
    for i, strategy in enumerate(strategies):
        train_transform, _ = AugmentationFactory.get_transforms(
            strategy=strategy, img_size=img_size, grayscale=True
        )
        
        for j in range(5):
            img = Image.open(sample_img_path)
            transformed = train_transform(img)
            
            # Convert tensor to numpy for display
            img_np = transformed.squeeze().numpy()
            img_np = (img_np * 0.5) + 0.5  # Denormalize
            
            axes[i, j].imshow(img_np, cmap='gray')
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_title(f'{strategy}', fontsize=12)
    
    plt.suptitle('Augmentation Strategies Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'augmentation_comparison.png'), dpi=150)
    plt.show()


if __name__ == "__main__":
    # Test dataset loading
    print("Testing PneumoniaDataset...")
    dataset = PneumoniaDataset(augmentation_strategy='moderate')
    train_loader, val_loader, test_loader, class_weights = dataset.get_data_loaders()
    
    print(f"\nDataLoader sizes:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    print("\nTesting CrossValidationHandler...")
    cv_handler = CrossValidationHandler(n_splits=3)
    
    for fold, train_loader, val_loader, weights in cv_handler.get_fold_loaders():
        print(f"  Fold {fold}: {len(train_loader)} train batches, {len(val_loader)} val batches")
        break  # Just test first fold