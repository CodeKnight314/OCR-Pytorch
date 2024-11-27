import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Tuple, Dict
from glob import glob 
from torchvision import transforms as T
from PIL import Image

class OCRDataset(Dataset):
    def __init__(self, root_dir: str, mode: str, csv: str, size: Tuple[int, int] = (64, 128), augment: bool = True):
        super().__init__()
        self.root_dir = root_dir
        self.img_dir = sorted(glob(os.path.join(self.root_dir, mode, "*")))
        self.csv = pd.read_csv(csv)
        self.height, self.width = size
        self.augment = augment and mode == 'train'
        
        # Base transforms
        self.transform = T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
        ])
        
        # Augmentation transforms
        self.augment_transforms = T.Compose([
            T.RandomRotation(3),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            T.RandomPerspective(distortion_scale=0.2, p=0.5),
        ])
        
        # Character mapping
        self.char_dict, self.idx_dict = self._create_char_mappings()
        
    def _create_char_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        df = self.csv["labels"]
        all_characters = ''.join(df.astype(str))
        unique_characters = sorted(set(all_characters))
        
        # Add blank token at index 0 for CTC loss
        char_dict = {'<blank>': 0}
        char_dict.update({char: idx + 1 for idx, char in enumerate(unique_characters)})
        idx_dict = {idx: char for char, idx in char_dict.items()}
        
        return char_dict, idx_dict
    
    def __len__(self) -> int:
        return len(self.img_dir)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_dir[index]
        filename = os.path.basename(img_path)
        
        label_row = self.csv[self.csv['images'] == filename]
        if len(label_row) == 0:
            raise ValueError(f"Label not found for image {filename}")
        
        label = str(label_row['labels'].iloc[0])
        img = Image.open(img_path).convert("RGB")
        
        if self.augment:
            img = self.augment_transforms(img)
        
        img = self.transform(img)
        encoded_label = torch.tensor([self.char_dict[char] for char in label], dtype=torch.long)
        
        return img, encoded_label

def collate_fn(batch):
    """
    Args:
        batch (List[Tuple[torch.Tensor, torch.Tensor]]): Batch of samples from the dataset, each containing an image tensor and a label tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple containing stacked image tensors and padded label tensors.
    """
    images, labels = zip(*batch)

    max_len = max(len(label) for label in labels)
    padded_labels = torch.full((len(labels), max_len), fill_value = -1, dtype=torch.long)

    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = label

    images = torch.stack(images)

    return images, padded_labels

def load_dataset(root_dir: str, mode: str, csv: str, size: Tuple[int, int], batch_size: int):
    ds = OCRDataset(root_dir, mode, csv, size, augment=(mode == 'train'))
    return DataLoader(
        ds, 
        batch_size=batch_size,
        shuffle=(mode == 'train'),
        num_workers=os.cpu_count(),
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )