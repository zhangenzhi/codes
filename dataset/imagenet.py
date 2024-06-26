import argparse
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define a custom InMemoryDataset class
# class InMemoryDataset(torch.utils.data.Dataset):
#     def __init__(self, data):
#         self.data = data
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data[idx]
    
def imagenet(args):

    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets
    image_datasets = {x: datasets.ImageNet(root=args.data_dir, split=x, transform=data_transforms[x])
                      for x in ['train', 'val']}
    
    # Convert datasets to InMemoryDataset
    # in_memory_datasets = {x: InMemoryDataset(image_datasets[x]) for x in ['train', 'val']}


    # Create data loaders
    shuffle = True
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=shuffle, 
                                 num_workers=args.num_workers,pin_memory=False)
                   for x in ['train', 'val']}
    return dataloaders

def imagenet_distribute(args):
    # Define data transformations
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Create datasets
    image_datasets = {x: datasets.ImageNet(root=args.data_dir, split=x, transform=data_transforms[x])
                      for x in ['train', 'val']}
    sampler = {x:torch.utils.data.distributed.DistributedSampler(image_datasets[x]) for x in ['train', 'val']}

    # Create data loaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=args.batch_size,
                                #  num_workers=args.num_workers, 
                                 pin_memory=False, sampler=sampler[x])
                   for x in ['train', 'val']}
    return dataloaders
    
import os
import shutil
import random

def imagenet_subset(args):
    # Define the paths
    original_data_dir = args.original_data_dir
    subset_data_dir = args.subset_data_dir 

    # Define the number of samples you want in your subset
    subset_size = 1000  # Adjust this as per your requirement

    # Create the subset directory if it doesn't exist
    if not os.path.exists(subset_data_dir):
        os.makedirs(subset_data_dir)

    # List all the classes (folders) in the original dataset directory
    classes = os.listdir(original_data_dir)

    # Randomly select classes for the subset
    selected_classes = random.sample(classes, k=min(subset_size, len(classes)))

    # Iterate through the selected classes
    for class_name in selected_classes:
        # Create the class directory in the subset directory
        subset_class_dir = os.path.join(subset_data_dir, class_name)
        os.makedirs(subset_class_dir)

        # List all the images in the class directory of the original dataset
        class_images = os.listdir(os.path.join(original_data_dir, class_name))

        # Randomly select images for the subset
        selected_images = random.sample(class_images, k=min(subset_size, len(class_images)))

        # Copy the selected images to the subset directory
        for image_name in selected_images:
            original_image_path = os.path.join(original_data_dir, class_name, image_name)
            subset_image_path = os.path.join(subset_class_dir, image_name)
            shutil.copy(original_image_path, subset_image_path)

    print("Subset creation completed.")

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
        
if __name__ == "__main__":
    dataloaders = imagenet()
    # Example usage:
    # Iterate through the dataloaders
    import time
    start_time = time.time()
    for phase in ['train', 'val']:
        for step, inputs, labels in enumerate(dataloaders[phase]):
            # Your training/validation/inference code goes here
            # For example:
            # model_output = model(inputs)
            # loss = loss_function(model_output, labels)
            # ...
            if step%500==0:
                print(step)
    print("Time cost for loading {}".format(time.time() - start_time))