import os
import torch
from torch import nn
import torch.utils.data as data  # For custom dataset (optional)
import torchvision.transforms as transforms
import timm

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# from model.vit import create_vit_model
from dataset.imagenet import imagenet_distribute

def create_vit_model(pretrained, num_classes=1000):
    """
    Creates a ViT model for ImageNet classification.

    Args:
        pretrained (bool): If True, loads pre-trained weights. Defaults to False.
        num_classes (int, optional): Number of output classes (defaults to 1000 for ImageNet). Defaults to 1000.

    Returns:
        nn.Module: The created ViT model.
    """

    if pretrained:
        # Fine-tune a pre-trained model (freeze early layers if desired)
        model = timm.create_model("vit_base_patch16_224", pretrained=True)
        for param in model.parameters():
            param.requires_grad = False  # Optionally freeze early layers

        # Modify the final classification head
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    else:
        # Create a ViT model with randomly initialized weights
        model = timm.create_model("vit_base_patch16_224", pretrained=False)
        # Modify the final classification head
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    return model


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device_id):
    """
    Trains the ViT model on the ImageNet dataset with validation.

    Args:
        model (nn.Module): The ViT model to train.
        train_loader (DataLoader): The DataLoader for the training data.
        val_loader (DataLoader): The DataLoader for the validation data.
        criterion (nn.Module): The loss function (e.g., CrossEntropyLoss).
        optimizer (Optimizer): The optimizer (e.g., Adam).
        num_epochs (int): The number of epochs to train.

    Returns:
        None
    """
    model.train()  # Set model to training mode
    total_step = len(train_loader)
    best_val_acc = 0.0
    print("Training the ViT model for {} epochs...".format(num_epochs))

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device_id)
            labels = labels.to(device_id)

            # Forward pass, calculate loss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print training progress (optional)
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        # Validate after each epoch
        val_acc = evaluate_model(model, val_loader)
        print("Validation Accuracy: {:.4f}".format(val_acc))

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_vit_model.pth")

        print('Finished Training Step %d' % (epoch + 1))

    print('Finished Training. Best Validation Accuracy: {:.4f}'.format(best_val_acc))

def evaluate_model(model, val_loader, device_id):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The trained model.
        val_loader (DataLoader): The
            # Put model in evaluation mode
    """
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device_id)
            labels = labels.to(device_id)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def vit_train(gpu, args):
    rank = gpu
    os.environ['MASTER_ADDR'] = 'localhost' #str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = "29500"
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=rank                                               
    )
    print("cal:{},get:{}".format(rank, dist.get_rank()))

    print(f"Start running basic DDP example on rank {rank}.")
    device_id = rank % torch.cuda.device_count()
    
    # Create DataLoader for training and validation
    dataloaders = imagenet_distribute(args=args)

    
    # Create ViT model
    model = create_vit_model(args.pretrained)
    model.to(device_id)
    model = DDP(model, device_ids=[device_id])
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    train_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, args.num_epochs, device_id=device_id)

def vit_ddp(args):
    args.world_size = args.gpus * args.nodes                
    mp.spawn(vit_train, nprocs=args.gpus, args=(args,))