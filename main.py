import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


from dataset.imagenet import imagenet
from train.trainer import vit_train

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet DataLoader Example')
    parser.add_argument('--data_dir', type=str, default='/Volumes/data/dataset/imagenet', help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_epochs', type=int, default=3, help='Epochs for iteration')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    args = parser.parse_args()
    return args

        
if __name__ == '__main__':
    args = parse_args()
    vit_train(args=args)