import torch
from torch import nn
import torch.utils.data as data  # For custom dataset (optional)
import torchvision.transforms as transforms
import timm

from model.unet import create_unet_model
from dataset.btcv import btcv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
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
            images = images.to(device)
            labels = labels.to(device)

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

def evaluate_model(model, val_loader):
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
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

def unet_btcv(args):

    # Create DataLoader for training and validation
    dataloaders = btcv(args=args)
    
    # Create Unet model
    model = create_unet_model(args.pretrained)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    train_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, args.num_epochs)

