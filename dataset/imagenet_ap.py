import argparse
import sys
sys.path.append("./")
import torch
import os
import glob
import numpy as np
from PIL import Image
import cv2 as cv
from torchvision import datasets, transforms
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
    
from torch.utils.data import Dataset
from map.transform import Patchify
class ImageNetDataset(Dataset):
    def __init__(self, root_dir, sths=[1,3,5,7], cannys=[50, 100], fixed_length=196, patch_size=16, transform=None):
        """
        Custom dataset to load ImageNet data using glob.
        Args:
            root_dir (str): Path to the root directory (train or val).
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.patchify = Patchify(sths=sths, fixed_length=fixed_length, cannys=cannys, patch_size=patch_size)
        self.image_paths = []  # List to store image paths
        self.labels = []       # List to store corresponding labels
        # Glob all images and map class indices
        classes = sorted(os.listdir(root_dir))  # Get class directories
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        
        for cls_name in classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_path in glob.glob(os.path.join(cls_dir, "*.JPEG")):  # Adjust extension if needed
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Open image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        image = cv.resize(image, dsize=[256,256])
        seq_img, seq_size, _ = self.patchify(image)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return seq_size, label
    
def test_ap(root_dir):
    patchify = Patchify()
    classes = sorted(os.listdir(root_dir))  # Get class directories
    image_paths = []
    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        for img_path in glob.glob(os.path.join(cls_dir, "*.JPEG")):  # Adjust extension if needed
            image_paths.append(img_path)
    for idx in range(len(image_paths)):
        img_path = image_paths[idx]
        # Open image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        image = cv.resize(image, dsize=[256,256])
        seq_img, seq_size, _ = patchify(image)
        if len(seq_size) != 196:
            print(len(seq_size), img_path, idx)
    
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet DataLoader Example')
    parser.add_argument('--task', type=str, default='imagenet', help='Type of task')
    # parser.add_argument('--data_dir', type=str, default='/Volumes/data/dataset/imagenet', help='Path to the ImageNet dataset directory')
    parser.add_argument('--data_dir', type=str, default='/Volumes/Extreme/dataset/imagenet', help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_epochs', type=int, default=3, help='Epochs for iteration')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for DataLoader')
    
    args = parser.parse_args()
    return args        
if __name__ == "__main__":
    
    args = parse_args()
    # Paths to the ImageNet directories
    train_dir = os.path.join(args.data_dir, "train")
    
    # test_ap(train_dir)
    
    val_dir = os.path.join(args.data_dir,"val")

    # Create datasets
    train_set = ImageNetDataset(train_dir)
    val_set = ImageNetDataset(val_dir)
    
    train_size = len(train_set)
    val_size = len(val_set)
    print("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, val_size))
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    # Example usage:
    # Iterate through the dataloaders
    import time
    start_time = time.time()
    for phase in ['train', 'val']:
        for step, data in enumerate(train_loader):
            if step%500==0:
                print("Step:{} Time Step:{}, Time Image:{}".format(step, 
                                                                (time.time() - start_time)//(step+1), 
                      (time.time() - start_time)//((step+1)*args.batch_size)))
    print("Time cost for loading {}".format(time.time() - start_time))