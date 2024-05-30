import os
import torch
from torch import nn
import torch.utils.data as data  # For custom dataset (optional)
import torchvision.transforms as transforms

import sys
sys.path.append("./")
import matplotlib.pyplot as plt

from dataset.btcv import btcv

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import (
    decollate_batch,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
post_label = AsDiscrete(to_onehot=14)
post_pred = AsDiscrete(argmax=True, to_onehot=14)

# Configure logging
import logging
def log(args):
    logging.basicConfig(
        filename=args.logname,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
def train_model(model, train_loader, val_loader, criterion, dice_metric, optimizer, num_epochs):
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
    best_val_dice = 0.0
    logging.info("Training the Unetr model for {} epochs...".format(num_epochs))

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        running_loss = 0.0
        for i, sample in enumerate(train_loader):
            images = sample["image"]
            labels = sample["label"]
            
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
            if i % 3 == 2:  # Print every 3 mini-batches
                logging.info('[%d, %5d] dice loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 3))
                running_loss = 0.0

        # Validate after each epoch
        mean_dice_val = evaluate_model(model, val_loader, dice_metric)
        logging.info("Validation Dice: {:.4f}".format(mean_dice_val))

        # Save the best model based on validation accuracy
        if mean_dice_val > best_val_dice:
            best_val_dice = mean_dice_val
            torch.save(model.state_dict(), "best_unetr_model.pth")

        logging.info('Finished Training Step %d' % (epoch + 1))

    print('Finished Training. Best Validation Accuracy: {:.4f}'.format(best_val_dice))

def evaluate_model(model, val_loader, dice_metric):
    """
    Evaluates the model on the validation set.

    Args:
        model (nn.Module): The trained model.
        val_loader (DataLoader): The
            # Put model in evaluation mode
    """
    
    model.eval()
    
    with torch.no_grad():
        for batch in val_loader:
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model)
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
    return mean_dice_val

def visualize(val_ds, model):
    slice_map = {
        "img0035.nii.gz": 170,
        "img0036.nii.gz": 230,
        "img0037.nii.gz": 204,
        "img0038.nii.gz": 204,
        "img0039.nii.gz": 204,
        "img0040.nii.gz": 180,
    }
    case_num = 4
    model.load_state_dict(torch.load(os.path.join("./", "best_metric_model.pth")))
    model.eval()
    with torch.no_grad():
        img_name = os.path.split(val_ds[case_num]["image"].meta["filename_or_obj"])[1]
        img = val_ds[case_num]["image"]
        label = val_ds[case_num]["label"]
        val_inputs = torch.unsqueeze(img, 1).cuda()
        val_labels = torch.unsqueeze(label, 1).cuda()
        val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=0.8)
        plt.figure("check", (18, 6))
        plt.subplot(1, 3, 1)
        plt.title("image")
        plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("label")
        plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
        plt.subplot(1, 3, 3)
        plt.title("output")
        plt.savefig("btcv_best-4.png")
        plt.close()
        
def unetr_btcv(args):
    
    log(args=args)
    from monai.networks.nets import UNETR
    
    # Create DataLoader for training and validation
    dataloaders = btcv(args=args)
    
    # Create Unet model
    model = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=(96, 96, 96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    res_block=True,
    dropout_rate=0.0,
    ).to(device)
    
    # Define loss function and optimizer
    criterion = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters())
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    # Train the model
    train_model(model, dataloaders['train'], dataloaders['val'], criterion, dice_metric, optimizer, args.num_epochs)

    # Visualize prediction
    visualize(dataloaders["val"], model=model)
    
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch ImageNet DataLoader Example')
    parser.add_argument('--logname', type=str, default='unetr_btcv.log', help='logging of task.')
    parser.add_argument('--data_dir', type=str, default='/Volumes/data/dataset/btcv/data', help='Path to the BTCV dataset directory')
    parser.add_argument('--num_epochs', type=int, default=10, help='Epochs for iteration')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for DataLoader')
    args = parser.parse_args()
    
    unetr_btcv(args=args)