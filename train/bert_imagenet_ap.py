import os
import torch
from torch import nn
import torch.utils.data as data  # For custom dataset (optional)
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time

from dataset.imagenet_ap import ImageNetDataset
from model.bert import BERTClassifier

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
    # Enable mixed precision
    scaler = torch.cuda.amp.GradScaler()
    model.train()  # Set model to training mode
    total_step = len(train_loader)
    best_val_acc = 0.0
    print("Training the BERT model for {} epochs...".format(num_epochs))

    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        running_loss = 0.0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            atten = torch.ones((images.size(0), images.size(1))).to(device, non_blocking=True)
            optimizer.zero_grad()   
            
            # Forward pass, calculate loss
            with torch.cuda.amp.autocast():
                outputs = model(images, atten)
                loss = criterion(outputs, labels)

            # Backward pass and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Print training progress (optional)
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                print('[%d, %5d] train loss: %.3f train acc: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100,  100 * correct / labels.size(0)))
                running_loss = 0.0
                correct = 0

        # Validate after each epoch
        model.eval()
        val_acc,val_loss = evaluate_model(model, val_loader, criterion)
        print("Val_Acc: {:.4f},Val_Loss: {:.4f}, Time Cost:{}".format(val_acc, val_loss, time.time()-start_time))

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_vit_model.pth")

        print('Finished Training Step %d' % (epoch + 1))

    print('Finished Training. Best Validation Accuracy: {:.4f}'.format(best_val_acc))

def evaluate_model(model, val_loader, criterion):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The trained model.
        val_loader (DataLoader): The
            # Put model in evaluation mode
    """
    correct = 0
    total = 0
    val_loss = 0.0
    num_iter = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            atten = torch.ones((images.size(0), images.size(1))).to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(images,atten)
                loss = criterion(outputs, labels)
                
            num_iter += 1
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_loss /= num_iter
    accuracy = 100 * correct / total
    return accuracy, val_loss

def bert_train(args):

    # Create DataLoader for training and validation
    train_dir = os.path.join(args.data_dir, "train")
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
    
    
    # Create Bert model
    seq_length = 512
    num_classes = 1000
    model = BERTClassifier(seq_length=seq_length, num_classes=num_classes)
    model = nn.DataParallel(model)
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs)

