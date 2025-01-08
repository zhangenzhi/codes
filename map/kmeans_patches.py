import numpy as np
from sklearn.cluster import KMeans
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def extract_patches(image, patch_size):
    """
    Extracts non-overlapping patches from an image.

    Args:
        image (numpy.ndarray): Input image as a 3D array (C, H, W).
        patch_size (int): The size of the patch (patch_size x patch_size).
        
    Returns:
        list: A list of patches flattened to 1D arrays.
    """
    C, H, W = image.shape
    patches = []
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = image[:, i:i+patch_size, j:j+patch_size]
            if patch.shape[1] == patch_size and patch.shape[2] == patch_size:
                patches.append(patch.flatten())
    return patches

def aggregate_patches_with_kmeans(dataset, patch_size, N):
    """
    Aggregates patches into N pivots using K-means clustering.

    Args:
        dataset (torch.utils.data.Dataset): The ImageNet dataset or similar.
        patch_size (int): The size of the patch (patch_size x patch_size).
        N (int): Number of clusters (pivots).
        
    Returns:
        np.ndarray: Cluster centers representing the N pivots.
    """
    all_patches = []

    # Extract patches from all images
    for image, _ in dataset:
        image_np = np.array(image).transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
        patches = extract_patches(image_np, patch_size)
        all_patches.extend(patches)
    
    # Convert to numpy array for clustering
    all_patches = np.array(all_patches)
    
    # Perform K-means clustering
    print(f"Performing K-means clustering on {len(all_patches)} patches...")
    kmeans = KMeans(n_clusters=N, random_state=42, verbose=1)
    kmeans.fit(all_patches)
    
    # Cluster centers are the pivot patches
    pivots = kmeans.cluster_centers_
    
    return pivots

# Parameters
patch_size = 4
N = 100

# Load ImageNet (Subset or Simulated)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fixed dimensions
    transforms.ToTensor(),  # Convert to Tensor
])
imagenet_dataset = datasets.FakeData(transform=transform)  # Simulated dataset
dataloader = DataLoader(imagenet_dataset, batch_size=16)

# Aggregate patches into N pivots
pivot_patches = aggregate_patches_with_kmeans(dataloader, patch_size, N)
print(f"Pivot patches (shape {pivot_patches.shape}):")
