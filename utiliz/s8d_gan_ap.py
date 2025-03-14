import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class TIFFDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.tiff')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Convert to RGB if needed
        
        if self.transform:
            image = self.transform(image)
        
        return image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize if needed
    transforms.ToTensor(),
])

if __name__ == "__main__":
    # Define dataset and dataloader
    root_dir = "/lustre/orion/mat268/world-shared/RIKEN/simulation_XCT/Noisy0.5_900views_detector800x800_12um/FBP"  # Change this to your directory
    import pdb;pdb.set_trace()
    
    from PIL import Image
    img_path = "/lustre/orion/mat268/world-shared/RIKEN/simulation_XCT/Noisy0.5_900views_detector800x800_12um/FBP/VOL_104_ConcA_angular_flack33_sph_p70Vf_dm_p15_pm_p15_r_p15_UC60_reconFBPsimul_Crop150.tiff"

    try:
        img = Image.open(img_path)
        img.verify()  # Verify if it's a valid image
        print("Image format:", img.format)
    except Exception as e:
        print("Error:", e)


    image_files = [f for f in os.listdir(root_dir) if f.endswith('.tiff')]
    idx = 0
    img_path = os.path.join(root_dir, image_files[idx])
    image = Image.open(img_path).convert('RGB')  # Convert to RGB if needed
    
# dataset = TIFFDataset(root_dir, transform=transform)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Example of iterating through the DataLoader
# for batch in dataloader:
#     print(batch.shape)  # Should print torch.Size([4, 3, 256, 256]) if batch_size=4