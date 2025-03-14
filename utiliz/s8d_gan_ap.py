import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import tifffile as tiff

class TIFFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.tiff')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = tiff.imread(img_path)
        
        return img


if __name__ == "__main__":
    # Define dataset and dataloader
    root_dir = "/lustre/orion/mat268/world-shared/RIKEN/simulation_XCT/Noisy0.5_900views_detector800x800_12um/FBP"  # Change this to your directory
    dataset = TIFFDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Example of iterating through the DataLoader
    for batch in dataloader:
        print(batch.shape)  # Should print torch.Size([4, 3, 256, 256]) if batch_size=4