from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np

from PIL import Image


class SEMSegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size, augment):
        self.img_size = img_size
        self.augment = augment
        
        # Create list of all images and their augmentations
        self.samples = []
        
        self.mean = [0.449]  # Average of ImageNet RGB means
        self.std = [0.226] 
        
        for img_path, mask_path in zip(image_paths, mask_paths):
            # Original
            self.samples.append({
                'image_path': img_path,
                'mask_path': mask_path
            })
        
        
        if self.augment:
            import albumentations as A
            
            self.aug_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                # A.Rotate(limit=180, p=0.5)
                #A.RandomScale(scale_limit=(-0.1, 0.1), p=0.5),
                #A.RandomCrop(height=self.img_size[0], width=self.img_size[1], p=0.5),
                
            
                
               
                
                # A.OneOf([
                #     A.GridDropout(ratio=0.01, unit_size_min=1, unit_size_max=2, random_offset=True, p=0.2),
                #     A.GaussNoise(var_limit=(5.0, 20.0), p=0.5),
                #     A.MultiplicativeNoise(multiplier=[0.95, 1.05], elementwise=True, p=0.3),
                # ], p=0.5),
                
                # A.GaussianBlur(blur_limit=(1, 3), p=0.3),
                # A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.2),
            ])
                     
        # Base transforms
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=transforms.InterpolationMode.NEAREST)
        ])
        
    def extract_metadata(self, image_path):
            """Extract metadata from image path and convert to tensor"""
            meta_f_s = Path(image_path).stem[:2]
            
            # Material type one-hot encoding (2 materials)
            material_type = torch.zeros(2)  # [SS316L, IN718]
            
            # Heat treatment one-hot encoding (2 treatments)
            heat_treatment = torch.zeros(2)  # [None, Solution treated]
            
            # For F1: SS316L with no heat treatment
            if meta_f_s == "F1":
                material_type[0] = 1.0  # SS316L
                heat_treatment[0] = 1.0  # No heat treatment
                
            # For F2: SS316L with solution treatment at 1200°C
            elif meta_f_s == "F2":
                material_type[0] = 1.0  # SS316L
                heat_treatment[1] = 1.0  # Solution treated
                
            # For F3: IN718 with no heat treatment
            elif meta_f_s == "F3":
                material_type[1] = 1.0  # IN718
                heat_treatment[0] = 1.0  # No heat treatment
                
            # For F4: IN718 with solution treatment at 980°C
            elif meta_f_s == "F4":
                material_type[1] = 1.0  # IN718
                heat_treatment[1] = 1.0  # Solution treated
                
            # Combine all metadata features
            metadata = torch.cat([material_type, heat_treatment], dim=0)
            return metadata    
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            image =  np.array(Image.open(sample['image_path']).convert('L'))
            mask = np.load(sample['mask_path'])
            
            # Extract metadata
            metadata = self.extract_metadata(sample['image_path'])
            
            if self.augment:
                augmented = self.aug_transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
                
            image = Image.fromarray(image)
            image = self.image_transform(image)
            
          
            mask = torch.from_numpy(mask).float().permute(2, 0, 1)
            mask = self.mask_transform(mask)
            
            assert image.size()[-2:] == mask.size()[-2:], f"Image and mask size don't match: {image.size()} vs {mask.size()}"

            return image, mask, metadata 
            
        except Exception as e:
            print(f"Error loading {sample['image_path']} or {sample['mask_path']}: {str(e)}")
            raise


def get_data_loaders(base_dir, batch_size, augment, img_size , num_workers=0,pin_memory=False ):
    base_dir = Path(base_dir)
    splits = ['train', 'val', 'test']
    loaders = {}
    
    print(f"Loading data from: {base_dir}")
    
    for split in splits:
        split_dir = base_dir / split
        images_dir = split_dir / 'images'
        masks_dir = split_dir / 'masks'
        
        if not images_dir.exists() or not masks_dir.exists():
            raise ValueError(f"Images or masks directory not found for {split} split")
        
        # Get all image and mask files
        image_files = sorted(list(images_dir.glob('*.tif')))
        mask_files = sorted(list(masks_dir.glob('*_mask.npy')))
        
        assert len(image_files)==len(mask_files) , "Some images doesn't have masks"
        
        # Create dataset with augmentation only for training set
        dataset = SEMSegmentationDataset(
            image_paths=image_files,
            mask_paths=mask_files,
            img_size=img_size,
            augment=(split == 'train' and augment)
        )
        
        print(f"\n{split} split:")
        print(f"Found {len(image_files)} original images")
        # print(f"Total dataset size after augmentation: {len(dataset)}")
        
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return loaders['train'], loaders['val'], loaders['test']