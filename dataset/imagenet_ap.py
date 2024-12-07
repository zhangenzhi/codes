import argparse
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def imagenet(args):
    
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(args.data_dir,"train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_dir,"val"), transform=transform)

    image_datasets = {'train':train_dataset, 'val':val_dataset}
    
    # Create data loaders
    shuffle = True
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=shuffle, 
                                 num_workers=args.num_workers, pin_memory=False, prefetch_factor=2)
                   for x in ['train', 'val']}
    return dataloaders

# epoch iteration
def imagenet_iter(args):
    dataloaders = imagenet(args=args)
    
    # Example usage:
    # Iterate through the dataloaders
    import time
    for e in range(args.num_epochs):
        start_time = time.time()
        for phase in ['train', 'val']:
            for step, (inputs, labels) in enumerate(dataloaders[phase]):
                if step%1000==0:
                    print(step)
        print("Time cost for loading {}".format(time.time() - start_time))
        
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet DataLoader Example')
    parser.add_argument('--task', type=str, default='imagenet', help='Type of task')
    parser.add_argument('--data_dir', type=str, default='/Volumes/data/dataset/imagenet', help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_epochs', type=int, default=3, help='Epochs for iteration')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for DataLoader')
    
    args = parser.parse_args()
    return args        
if __name__ == "__main__":
    args = parse_args()
    dataloaders = imagenet(args)
    # Example usage:
    # Iterate through the dataloaders
    import time
    start_time = time.time()
    for phase in ['train', 'val']:
        for step, data in enumerate(dataloaders[phase]):
            if step%500==0:
                print(step)
    print("Time cost for loading {}".format(time.time() - start_time))