"""
Dataset loader for JU-LDD-task-b TB Detection
Loads paired CXR and Sputum microscopy images
"""
import os
import re
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AddGaussianNoise(object):
    """Add Gaussian noise to the image tensor."""
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # tensor is [C, H, W]
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class TBMultimodalDataset(Dataset):
    """
    Dataset for paired CXR and Sputum microscopy images.
    
    Directory structure:
    data_root/
        train/
            tb/
                cxr/      # CXR-train-1.png, CXR-train-2.png, ...
                sputum/   # Sputum-train-1.png, Sputum-train-2.png, ...
            no_finding/
                cxr/
                sputum/
        val/
        test/
    """
    def __init__(
        self, 
        data_root, 
        split='train',
        img_size=224,
        augment=True
    ):
        """
        Args:
            data_root: Path to JU-LDD-task-b directory
            split: 'train', 'val', or 'test'
            img_size: Target image size
            augment: Whether to apply data augmentation
        """
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        self.img_size = img_size
        
        # Collect all paired samples
        self.samples = []
        self._load_samples()
        
        # Define transforms
        self.cxr_transform = self._get_cxr_transforms(augment and split == 'train')
        self.sputum_transform = self._get_sputum_transforms(augment and split == 'train')
        
    def _load_samples(self):
        """Load all CXR-Sputum pairs from the dataset."""
        split_dir = self.data_root / self.split
        
        # Class mapping: tb=1, no_finding=0
        class_dirs = {
            'tb': 1,
            'no_finding': 0
        }
        
        for class_name, label in class_dirs.items():
            cxr_dir = split_dir / class_name / 'cxr'
            sputum_dir = split_dir / class_name / 'sputum'
            
            if not cxr_dir.exists() or not sputum_dir.exists():
                print(f"Warning: {class_name} directory not found in {self.split}")
                continue
            
            # Get all CXR files
            cxr_files = sorted(cxr_dir.glob('*.png'))
            
            for cxr_path in cxr_files:
                # Extract ID from filename (e.g., CXR-train-1.png -> 1)
                match = re.search(r'-(\d+)\.png$', cxr_path.name)
                if not match:
                    continue
                
                img_id = match.group(1)
                
                # Find corresponding sputum file
                sputum_path = sputum_dir / f'Sputum-{self.split}-{img_id}.png'
                
                if sputum_path.exists():
                    self.samples.append({
                        'cxr_path': str(cxr_path),
                        'sputum_path': str(sputum_path),
                        'label': label,
                        'class_name': class_name,
                        'id': img_id
                    })
                else:
                    print(f"Warning: Missing sputum pair for {cxr_path.name}")
        
        print(f"Loaded {len(self.samples)} paired samples from {self.split} split")
        
        # Print class distribution
        tb_count = sum(1 for s in self.samples if s['label'] == 1)
        normal_count = len(self.samples) - tb_count
        print(f"  TB: {tb_count}, No Finding: {normal_count}")
    
    def _get_cxr_transforms(self, augment=False):
        """CXR-specific transforms."""
        if augment:
            # Training: augmentation + noise for regularization
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
                # Gaussian noise to simulate radiographic sensor noise / exposure variations
                AddGaussianNoise(std=0.6)
            ])
        else:
            # Validation/Test: NO augmentation, NO noise - just preprocessing
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def _get_sputum_transforms(self, augment=False):
        """Sputum microscopy-specific transforms."""
        if augment:
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  # Convert to grayscale
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),  # Microscopy is rotation-invariant
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]),
                # Random noise on sputum to simulate microscopy artifacts (Robustness)
                transforms.RandomApply([AddGaussianNoise(std=0.3)], p=0.5)
            ])
        else:
            return transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  # Convert to grayscale
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            cxr: [3, 224, 224] tensor
            sputum: [3, 224, 224] tensor
            label: 0 (no_finding) or 1 (tb)
            metadata: dict with paths and ID
        """
        sample = self.samples[idx]
        
        # Load images
        cxr = Image.open(sample['cxr_path']).convert('RGB')
        sputum = Image.open(sample['sputum_path']).convert('RGB')
        
        # Apply transforms
        cxr = self.cxr_transform(cxr)
        sputum = self.sputum_transform(sputum)
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        metadata = {
            'id': sample['id'],
            'class_name': sample['class_name'],
            'cxr_path': sample['cxr_path'],
            'sputum_path': sample['sputum_path']
        }
        
        return cxr, sputum, label, metadata


class UnimodalCXRDataset(Dataset):
    """
    Dataset for CXR-only datasets (like "Dataset of Tuberculosis Chest X-rays Images").
    Splits the directory deterministically and returns fake (zero) sputum to maintain compatibility.
    """
    def __init__(
        self, 
        data_root, 
        split='train',
        img_size=224,
        augment=True,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size
        
        # Load all samples
        all_samples = []
        
        class_dirs = {
            'TB Chest X-rays': 1,
            'Normal Chest X-rays': 0,
            'Tuberculosis': 1,
            'Normal': 0
        }
        
        for class_name, label in class_dirs.items():
            class_dir = self.data_root / class_name
            if not class_dir.exists():
                print(f"Warning: {class_name} directory not found in {data_root}")
                continue
                
            for ext in ['*.png', '*.jpg', '*.jpeg']:
                for img_path in sorted(class_dir.glob(ext)):
                    all_samples.append({
                        'cxr_path': str(img_path),
                        'sputum_path': 'none',
                        'label': label,
                        'class_name': class_name,
                        'id': img_path.stem
                    })
                
        # Split data deterministically
        import random
        random.seed(seed)
        random.shuffle(all_samples)
        
        n_total = len(all_samples)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        if split == 'train':
            self.samples = all_samples[:n_train]
        elif split == 'val':
            self.samples = all_samples[n_train:n_train+n_val]
        else: # test
            self.samples = all_samples[n_train+n_val:]
            
        print(f"Loaded {len(self.samples)} CXR-only samples for {split} split")
        
        # CXR transforms (reused logic)
        if augment and split == 'train':
            self.cxr_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                AddGaussianNoise(std=0.6)
            ])
        else:
            self.cxr_transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        cxr = Image.open(sample['cxr_path']).convert('RGB')
        cxr = self.cxr_transform(cxr)
        
        # Fake sputum tensor (all zeros) for model compatibility
        sputum = torch.zeros_like(cxr)
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        metadata = {
            'id': sample['id'],
            'class_name': sample['class_name'],
            'cxr_path': sample['cxr_path'],
            'sputum_path': sample['sputum_path']
        }
        
        return cxr, sputum, label, metadata



def get_dataloaders(
    data_root,
    batch_size=4,
    img_size=224,
    num_workers=2,
    pin_memory=False,
    is_unimodal=False
):
    """
    Create train, val, and test dataloaders.
    
    Args:
        data_root: Path to JU-LDD-task-b directory
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (faster for GPU)
        is_unimodal: Use UnimodalCXRDataset instead of TBMultimodalDataset
    
    Returns:
        train_loader, val_loader, test_loader
    """
    dataset_class = UnimodalCXRDataset if is_unimodal else TBMultimodalDataset
    
    # Create datasets
    train_dataset = dataset_class(
        data_root, split='train', img_size=img_size, augment=True
    )
    val_dataset = dataset_class(
        data_root, split='val', img_size=img_size, augment=False
    )
    test_dataset = dataset_class(
        data_root, split='test', img_size=img_size, augment=False
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # For stable batch norm
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing dataset loader...")
    
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root='data/JU-LDD-task-b',
        batch_size=2,
        num_workers=0  # For testing
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val:   {len(val_loader.dataset)} samples")
    print(f"  Test:  {len(test_loader.dataset)} samples")
    
    # Test one batch
    print(f"\nTesting batch loading...")
    cxr, sputum, labels, metadata = next(iter(train_loader))
    
    print(f"  CXR shape: {cxr.shape}")
    print(f"  Sputum shape: {sputum.shape}")
    print(f"  Labels: {labels}")
    print(f"  IDs: {metadata['id']}")
    
    print("\n[OK] Dataset loader working correctly!")
