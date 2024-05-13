import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


from dataset.imagenet import imagenet

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet DataLoader Example')
    parser.add_argument('--data_dir', type=str, default='/Volumes/data/dataset/imagenet', help='Path to the ImageNet dataset directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataloaders = imagenet(args=args)
    
    # Example usage:
    # Iterate through the dataloaders
    import time
    start_time = time.time()
    for phase in ['train', 'val']:
        for step, (inputs, labels) in enumerate(dataloaders[phase]):
            # Your training/validation/inference code goes here
            # For example:
            # model_output = model(inputs)
            # loss = loss_function(model_output, labels)
            # ...
            if step%500==0:
                print(step)
    print("Time cost for loading {}".format(time.time() - start_time))