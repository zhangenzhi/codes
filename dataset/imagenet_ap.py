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
from map.transform import ImagePatchify
class ImageNetDataset(Dataset):
    def __init__(self, root_dir, sths=[0,1,3,5,7], cannys=[50, 100], fixed_length=196, patch_size=16, transform=None):
        """
        Custom dataset to load ImageNet data using glob.
        Args:
            root_dir (str): Path to the root directory (train or val).
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root_dir = root_dir
        self.patchify = ImagePatchify(sths=sths, fixed_length=fixed_length, cannys=cannys, patch_size=patch_size)
        self.transform =  transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize([224,224]),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float16),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.seq_transform= transforms.Compose([
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float16)
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
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
        np_image = np.array(image)
        np_image = cv.resize(np_image, dsize=[256,256])
        seq_img, seq_size, seq_pos = self.patchify(np_image)
        seq_img = self.seq_transform(seq_img)
        seq_size = torch.Tensor(seq_size)
        seq_pos = torch.Tensor(seq_pos)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        return image, seq_img, seq_size, seq_pos, label
    
def test_ap(root_dir, fixed_length=4096,res=256):
    patchify = ImagePatchify(fixed_length=fixed_length, patch_size=4, sths=[3,5,7], cannys=[70, 120])
    classes = sorted(os.listdir(root_dir))  # Get class directories
    image_paths = []
    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        for img_path in glob.glob(os.path.join(cls_dir, "*.JPEG")):  # Adjust extension if needed
            image_paths.append(img_path)
    avg_size = 0 
    avg_length = 0
    for idx in range(len(image_paths)):
        img_path = image_paths[idx]
        # Open image
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        image = cv.resize(image, dsize=[res,res])
        seq_img, seq_size, seq_pos = patchify(image)
        avg_size += np.sum(seq_size)
        import pdb
        pdb.set_trace()
        avg_length += len(seq_pos)
        if len(seq_pos) != fixed_length:
            print(len(seq_pos), img_path, idx)
        if (idx+1)%500==0:
            
            avg_fpatch = avg_size/idx/fixed_length
            fcr = ((res/avg_fpatch)*(res/avg_fpatch))/fixed_length 
            
            avg_true_length = avg_length/(idx+1)
            avg_tpatch = avg_size/idx/avg_true_length
            tcr = ((res/avg_tpatch)*(res/avg_tpatch))/avg_true_length 
            print("avg_true_length:{}, Avg_FPatch:{}, Avg_TPatch:{}, FCR:{}, TCR:{}".format(avg_true_length, avg_fpatch,avg_tpatch, fcr, tcr))
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet DataLoader Example')
    parser.add_argument('--task', type=str, default='imagenet', help='Type of task')
    parser.add_argument('--data_dir', type=str, default='/Volumes/data/dataset/imagenet', help='Path to the ImageNet dataset directory')
    # parser.add_argument('--data_dir', type=str, default='/Volumes/Extreme/dataset/imagenet', help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_epochs', type=int, default=3, help='Epochs for iteration')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for DataLoader')
    
    args = parser.parse_args()
    return args        
if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("TkAgg")
    args = parse_args()
    # # Paths to the ImageNet directories
    train_dir = os.path.join(args.data_dir, "train")
    
    test_ap(train_dir)
    
    # val_dir = os.path.join(args.data_dir,"val")

    # # Create datasets
    # train_set = ImageNetDataset(train_dir, fixed_length=196, patch_size=16)
    # val_set = ImageNetDataset(val_dir, fixed_length=196, patch_size=16)
    
    # train_size = len(train_set)
    # val_size = len(val_set)
    # print("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, val_size))
    
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=args.batch_size, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    # test_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    
    # # Example usage:
    # # Iterate through the dataloaders
    # import time
    # start_time = time.time()
    # for phase in ['train', 'val']:
    #     for step, data in enumerate(train_loader):
    #         gd, image, label = data
    #         image = torch.reshape(image,shape=(-1,3,224, 224))
    #         image = image[0]
    #         image = image.permute(1, 2, 0).numpy()
    #         image = np.float32(image)
    #         gd = gd[0]
    #         gd = gd.permute(1, 2, 0).numpy()
    #         gd = np.float32(gd)
    #         # import pdb
    #         # pdb.set_trace()

    #         # Plot the image
    #         fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns
    #         axs[0].imshow(gd)
    #         axs[0].axis('off')  # Turn off axes for better visualization
    #         axs[0].set_title(f"gd")
            
    #         axs[1].imshow(image)
    #         axs[1].axis('off')  # Turn off axes for better visualization
    #         axs[1].set_title(f"seq img")
    
    #         plt.tight_layout()
    #         plt.show()
    #         plt.close()
            
    #         if step%500==0:
    #             print("Step:{} Time Step:{}, Time Image:{}".format(step, 
    #                                                             (time.time() - start_time)/(step+1), 
    #                   (time.time() - start_time)/((step+1)*args.batch_size)))
    # print("Time cost for loading {}".format(time.time() - start_time))