import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from typing import Tuple
from glob import glob 
from torchvision import transforms as T
from PIL import Image 

class OCRDataset(Dataset): 
    def __init__(self, root_dir: str, mode: str, csv: str, size: Tuple[int, int] = (64, 128)): 
        super().__init__()
        self.root_dir = root_dir 
        self.img_dir = glob(os.path.join(self.root_dir, mode, "*"))
        
        self.csv = pd.read_csv(csv)
        
        self.height, self.width = size 
        
        self.transform = T.Compose([T.Resize(size), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        
        df = self.csv["labels"]
        all_characters = ''.join(df.astype(str))
        unique_characters = sorted(set(all_characters))
        
        self.char_dict = {}
        blank_character = ''  # Use space as the blank character
        if blank_character in unique_characters:
            unique_characters.remove(blank_character)
        unique_characters.append(blank_character)
        
        self.char_dict = {char: idx for idx, char in enumerate(unique_characters)}
        self.idx_dict = {idx: char for idx, char in enumerate(unique_characters)}
        
    def __len__(self): 
        return len(self.img_dir)
    
    def __getitem__(self, index: int): 
        img_path = self.img_dir[index]
        filename = os.path.basename(img_path)
        
        label_row = self.csv.loc[self.csv['images'] == filename]
        
        if len(label_row) == 0:
            raise ValueError(f"Label not found for image {filename}")
        
        label = label_row['labels'].iloc[0]
        
        img = self.transform(Image.open(img_path).convert("RGB"))
        
        encoded_label = [self.char_dict[char] for char in label]
        
        return img, torch.tensor(encoded_label, dtype=torch.long)
    
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
    """
    Args:
        root_dir (str): Root directory containing the dataset.
        mode (str): Mode of the dataset ('train' or 'val').
        csv (str): Path to the CSV file containing image file names and their corresponding labels.
        size (Tuple[int, int]): Tuple specifying the height and width for resizing images.
        batch_size (int): Number of samples per batch.

    Returns:
        DataLoader: PyTorch DataLoader instance for loading the dataset with batching and shuffling.
    """
    ds = OCRDataset(root_dir, mode, csv, size)
    return DataLoader(ds, batch_size, shuffle=True, num_workers=os.cpu_count())
        
        