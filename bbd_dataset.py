import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
import random

class COCOMultiCropDataset(Dataset):
    """
    COCO dataset with multi-crop transformations for DINO training
    """
    def __init__(self, annFile, dataDir, global_crop_size=224, local_crop_size=96, 
                 num_local_crops=4, transform=None):
        self.coco = COCO(annFile)
        self.dataDir = dataDir
        self.img_ids = self.coco.getImgIds()
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.num_local_crops = num_local_crops
        
        # Normalization (ImageNet stats)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        # Default transformations if none provided
        if transform is None:
            self.transform = self.get_default_transforms()
        else:
            self.transform = transform
    
    def get_default_transforms(self):
        """
        Default DINO multi-crop transformations
        """
        # Global crops (2x)
        global_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.global_crop_size, scale=(0.4, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
            transforms.ToTensor(),
            self.normalize
        ])
        
        # Local crops (multiple)
        local_transform = transforms.Compose([
            transforms.RandomResizedCrop(self.local_crop_size, scale=(0.05, 0.4)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((self.global_crop_size, self.global_crop_size)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
            transforms.ToTensor(),
            self.normalize
        ])
        
        return {
            'global': global_transform,
            'local': local_transform
        }
    
    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        
        # Load image
        img_path = f"{self.dataDir}/{img_info['file_name']}"
        image = cv2.imread(img_path)
        
        if image is None:
            # If image loading fails, return a random image
            image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image for transformations
        image_pil = transforms.ToPILImage()(image)
        
        # Apply transformations
        crops = []
        
        # Global crops (2x)
        for _ in range(2):
            crops.append(self.transform['global'](image_pil))
        
        # Local crops (num_local_crops x)
        for _ in range(self.num_local_crops):
            crops.append(self.transform['local'](image_pil))
        
        return crops

# Create dataset and data loader
def create_coco_dataloader(annFile, dataDir, batch_size=4, num_workers=4, 
                          global_crop_size=224, local_crop_size=96, num_local_crops=4):
    """
    Create COCO data loader for DINO training
    """
    dataset = COCOMultiCropDataset(
        annFile=annFile,
        dataDir=dataDir,
        global_crop_size=global_crop_size,
        local_crop_size=local_crop_size,
        num_local_crops=num_local_crops
    )
    
    # Custom collate function for multi-crop data
    def collate_fn(batch):
        # batch is a list of lists of crops
        # We need to transpose it to group crops by type
        transposed = list(zip(*batch))
        return [torch.stack(crops) for crops in transposed]
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader

# Usage example
if __name__ == "__main__":
    # Initialize COCO dataset
    dataDir = "/content/object-detection-BBD/data/100k/test/"
    annFile = f"{dataDir}/_annotations.coco.json"
    
    # Create data loader
    dataloader = create_coco_dataloader(
        annFile=annFile,
        dataDir=dataDir,
        batch_size=4,
        num_workers=4,
        global_crop_size=224,
        local_crop_size=96,
        num_local_crops=4
    )
    
    # Test the data loader
    for batch_idx, crops in enumerate(dataloader):
        print(f"Batch {batch_idx}:")
        for i, crop_batch in enumerate(crops):
            print(f"  Crop {i}: shape {crop_batch.shape}")
        
        if batch_idx >= 2:  # Just test a few batches
            break