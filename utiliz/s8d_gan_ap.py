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
    # Time cost:0.5032467756952558, total samples 111
    # root_dir = "/lustre/orion/mat268/world-shared/RIKEN/simulation_XCT/Noisy0.5_900views_detector800x800_12um/FBP"  # Change this to your directory
    # Time cost:0.5032467756952558, total samples 111
    root_dir = "/lustre/orion/mat268/world-shared/RIKEN/simulation_XCT/Noisy0.35_300views_detector1200x1200_12um/FBP"  # Change this to your directory
    # root_dir = "/lustre/orion/mat268/world-shared/RIKEN/simulation_XCT/high_packingFactor/Noisy0.35_300views_detector1200x1200_12um_HPF/FBP"
    dataset = TIFFDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=32)

    # Example of iterating through the DataLoader
    import time
    start_time = time.time()
    for batch in dataloader:
        print(batch.shape)  # Should print torch.Size([4, 3, 256, 256]) if batch_size=4
    print(f"Time cost:{(time.time() - start_time)/len(dataloader)}, total samples {len(dataset)}")