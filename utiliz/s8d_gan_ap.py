import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import tifffile as tiff

class S8DGANAP(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.pdb_dir = os.path.join(root_dir, "FBP")
        self.labels_dir = os.path.join(root_dir, "labels")
        
        self.image_files = [f for f in os.listdir(self.pdb_dir) if f.endswith('.tiff')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        label_name = image_name.replace("reconFBPsimul", "label")
        
        img_path = os.path.join(self.pdb_dir, image_name)
        label_path = os.path.join(self.labels_dir, label_name)
        
        image =  tiff.imread(img_path)  # May need to Convert to RGB if needed
        label =  tiff.imread(label_path)  # May need to Convert to grayscale
        
        if self.transform:
            image = self.transform(image)
            label = transforms.ToTensor()(label)  # Convert label to tensor
        
        return image, label
    

if __name__ == "__main__":
    # Define dataset and dataloader
    # Time cost:0.5032467756952558, total samples 111, torch.Size([4, 768, 768, 768])
    # root_dir = "/lustre/orion/mat268/world-shared/RIKEN/simulation_XCT/Noisy0.5_900views_detector800x800_12um"  # Change this to your directory
    # Time cost:0.8656143597194127, total samples 140, torch.Size([4, 768, 768, 768])
    root_dir = "/lustre/orion/mat268/world-shared/RIKEN/simulation_XCT/Noisy0.35_300views_detector1200x1200_12um"  # Change this to your directory
    # Time cost:0.9388706513813564, total samples 56, torch.Size([4, 768, 768, 768]) 768*16x768*16x3 ?
    # root_dir = "/lustre/orion/mat268/world-shared/RIKEN/simulation_XCT/high_packingFactor/Noisy0.35_300views_detector1200x1200_12um_HPF"
    dataset = S8DGANAP(root_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=32)

    # Example of iterating through the DataLoader
    import time
    start_time = time.time()
    for (img, mask) in dataloader:
        import pdb;pdb.set_trace()
        print(img.shape, mask.shape)  # Should print torch.Size([4, 3, 256, 256]) if batch_size=4
    print(f"Time cost:{(time.time() - start_time)/len(dataloader)}, total samples {len(dataset)}")
    
    # # Save the last batch of images and labels as grayscale slices
    # last_images_np = img.squeeze().numpy()  # Remove batch dimension
    # last_labels_np = mask.squeeze().numpy()

    # # Normalize and convert to uint8
    # last_images_np = ((last_images_np - last_images_np.min()) / (last_images_np.max() - last_images_np.min()) * 255).astype(np.uint8)
    # last_labels_np = ((last_labels_np - last_labels_np.min()) / (last_labels_np.max() - last_labels_np.min()) * 255).astype(np.uint8)

    # output_dir = "saved_slices"
    # os.makedirs(output_dir, exist_ok=True)

    # for i in range(last_images_np.shape[0]):
    #     Image.fromarray(last_images_np[i]).convert('L').save(os.path.join(output_dir, f"image_slice_{i}.png"))
    #     Image.fromarray(last_labels_np[i]).convert('L').save(os.path.join(output_dir, f"label_slice_{i}.png"))

    # print(f"Saved grayscale slices in {output_dir}")