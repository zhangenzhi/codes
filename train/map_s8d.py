import torch
from torch import nn
import torch.utils.data as data  # For custom dataset (optional)
import torchvision.transforms as transforms
import timm
import time
import os
from torch.utils.data import DataLoader
import time

import sys
sys.path.append("./")

import os
import logging

# Configure logging
def log(args):
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output, "out.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
from model.map import map_vit_base_patch16_dec512d8b
from dataset.imagenet import imagenet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretrain_model(model, train_loader, val_loader, optimizer, num_epochs, output):
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
    # scaler = torch.cuda.amp.GradScaler()

    model.train()  # Set model to training mode
    best_val_loss = 0.0
    print("Training the ViT model for {} epochs...".format(num_epochs))

    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        running_loss = 0.0
        for i, (seq_img, seq_size, seq_pos) in enumerate(train_loader):
            seq_img = seq_img.to(device, non_blocking=True)
            # labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()   
            
            # Forward pass, calculate loss
            # with torch.cuda.amp.autocast():
            loss, pred, mask = model(seq_img)
            # print(loss)
            loss.sum().backward()
            optimizer.step()

            # Backward pass and optimize
            # scaler.scale(loss).backward()
            # scaler.step(optimizer)
            # scaler.update()

            # Print training progress (optional)
            running_loss += loss.mean().item()
            if i % 100 == 99:  # Print every 100 mini-batches
                logging.info('[%d, %5d] train loss: %.3f train acc: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100,  100 * 0.0 / labels.size(0)))
                running_loss = 0.0

        # Validate after each epoch
        val_total = 0
        val_loss = 0.0
        num_iter = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                loss, pred, mask = model(images)
    
                num_iter += 1
                val_loss += loss.mean().item()
     
                val_total += labels.size(0)
        val_loss /= num_iter
        logging.info("Val_Acc: {:.4f},Val_Loss: {:.4f}, Time Cost:{}".format(0.0, val_loss, time.time()-start_time))

        # Save the best model based on validation accuracy
        if val_loss > best_val_loss:
            val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(output,"best_mae_model.pth"))

        logging.info('Finished Pre-Training Step %d' % (epoch + 1))

    logging.info('Finished Pre-Training. Best Validation Accuracy: {:.4f}'.format(best_val_loss))

def map_s8d_pretrain(args):
    log(args=args)

    # Create datasets
    dataloaders = imagenet(args=args)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    
    train_size = len(train_loader)
    val_size = len(val_loader)
    logging.info("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, val_size))
    
    # Create ViT model
    model = map_vit_base_patch16_dec512d8b()
    # model = nn.DataParallel(model)
    model = model.to(device)
    
    # Define loss function and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    pretrain_model(model, train_loader, val_loader, optimizer, args.num_epochs, args.output)


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, output):
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
    best_val_acc = 0.0
    print("Training the MAE model for {} epochs...".format(num_epochs))

    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        running_loss = 0.0
        correct = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()   
            
            # Forward pass, calculate loss
            # with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.sum().backward()
            optimizer.step()

            # Print training progress (optional)
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                logging.info('[%d, %5d] train loss: %.3f train acc: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100,  100 * correct / labels.size(0)))
                running_loss = 0.0
                correct = 0

        # Validate after each epoch
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        num_iter = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                    
                num_iter += 1
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss /= num_iter
        val_acc = 100 * val_correct / val_total
        logging.info("Val_Acc: {:.4f},Val_Loss: {:.4f}, Time Cost:{}".format(val_acc, val_loss, time.time()-start_time))

        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output,"best_mae_model.pth"))

        logging.info('Finished Training Step %d' % (epoch + 1))

    logging.info('Finished Training. Best Validation Accuracy: {:.4f}'.format(best_val_acc))

def map_finetune(args):
    log(args=args)

    # Create datasets
    dataloaders = imagenet(args=args)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    
    train_size = len(train_loader)
    val_size = len(val_loader)
    logging.info("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, val_size))
    
    # Create ViT model
    model = map_vit_base_patch16_dec512d8b()
    # model = nn.DataParallel(model)
    if args.reload:
        if os.path.exists(os.path.join(args.output, "best_mae_model.pth")):
            model.load_state_dict(torch.load(os.path.join(args.output, "best_mae_model.pth")))
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs)

