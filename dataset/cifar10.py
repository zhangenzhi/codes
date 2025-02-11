import os
import numpy as np
from PIL import Image
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from map.transform import ImagePatchify

class CIFAR10Dataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        """
        Args:
            root (str): Root directory of the CIFAR-10 dataset.
            train (bool): If True, loads the training set, else loads the test set.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.root = root
        self.train = train
        self.transform = transform

        # CIFAR-10 data files
        if self.train:
            self.data_file = os.path.join(self.root, 'data_batch_')
            self.num_batches = 5  # CIFAR-10 training set has 5 batches
        else:
            self.data_file = os.path.join(self.root, 'test_batch')

        self.data, self.labels = self._load_data()

    def _load_data(self):
        """Loads CIFAR-10 data from binary files."""
        data = []
        labels = []

        if self.train:
            for i in range(1, self.num_batches + 1):
                batch_file = f"{self.data_file}{i}"
                with open(batch_file, 'rb') as f:
                    batch = self._unpickle(f)
                    data.append(batch[b'data'])
                    labels.extend(batch[b'labels'])
        else:
            with open(self.data_file, 'rb') as f:
                batch = self._unpickle(f)
                data.append(batch[b'data'])
                labels.extend(batch[b'labels'])

        data = np.concatenate(data).reshape(-1, 3, 32, 32)  # Convert to [N, C, H, W]
        data = np.transpose(data, (0, 2, 3, 1))  # Convert to [N, H, W, C]
        return data, labels

    def _unpickle(self, file):
        """Unpickles the CIFAR-10 binary file."""
        import pickle
        return pickle.load(file, encoding='bytes')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # Convert to PIL Image
        image = Image.fromarray(image)

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label

class CIFAR10DatasetAP(Dataset):
    def __init__(self, root, train=True, sths=[0,1,3,5,7], cannys=[50, 100], fixed_length=64, patch_size=4, transform=None):
        """
        Args:
            root (str): Root directory of the CIFAR-10 dataset.
            train (bool): If True, loads the training set, else loads the test set.
            transform (callable, optional): A function/transform to apply to the data.
        """
        self.root = root
        self.train = train
        self.transform = transform
        self.patchify = ImagePatchify(sths=sths, fixed_length=fixed_length, cannys=cannys, patch_size=patch_size)

        # CIFAR-10 data files
        if self.train:
            self.data_file = os.path.join(self.root, 'data_batch_')
            self.num_batches = 5  # CIFAR-10 training set has 5 batches
        else:
            self.data_file = os.path.join(self.root, 'test_batch')

        self.data, self.labels = self._load_data()

    def _load_data(self):
        """Loads CIFAR-10 data from binary files."""
        data = []
        labels = []

        if self.train:
            for i in range(1, self.num_batches + 1):
                batch_file = f"{self.data_file}{i}"
                with open(batch_file, 'rb') as f:
                    batch = self._unpickle(f)
                    data.append(batch[b'data'])
                    labels.extend(batch[b'labels'])
        else:
            with open(self.data_file, 'rb') as f:
                batch = self._unpickle(f)
                data.append(batch[b'data'])
                labels.extend(batch[b'labels'])

        data = np.concatenate(data).reshape(-1, 3, 32, 32)  # Convert to [N, C, H, W]
        data = np.transpose(data, (0, 2, 3, 1))  # Convert to [N, H, W, C]
        return data, labels

    def _unpickle(self, file):
        """Unpickles the CIFAR-10 binary file."""
        import pickle
        return pickle.load(file, encoding='bytes')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        # Convert to PIL Image
        image = Image.fromarray(image)

        np_image = np.array(image)
        np_image = cv.resize(np_image, dsize=[32,32])
        seq_img, seq_size, seq_pos = self.patchify(np_image)
        seq_img = self.seq_transform(seq_img)
        seq_size = torch.Tensor(seq_size)
        seq_pos = torch.Tensor(seq_pos)
        # print(seq_img.shape, seq_size.shape, seq_pos.shape)
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        return image, seq_img, seq_size, seq_pos, label
    
# Example usage
if __name__ == "__main__":
    root = "./cifar-10-batches-py"  # Path to the CIFAR-10 dataset

    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # Create dataset and dataloader
    train_dataset = CIFAR10Dataset(root=root, train=True, transform=transform)
    test_dataset = CIFAR10Dataset(root=root, train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    # Iterate through the dataloader
    for images, labels in train_loader:
        print(f"Batch of images: {images.shape}")
        print(f"Batch of labels: {labels.shape}")
        break