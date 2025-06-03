import torch
from torch import nn
import time
import os
import time

import sys
sys.path.append("./")

import os
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# from model.vit import create_vit_model
from dataset.imagenet import imagenet_distribute
# Configure logging
def log(args):
    os.makedirs(args.output, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.output, "out.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
from model.mae import mae_vit_base_patch16
from dataset.imagenet import imagenet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pretrain_model(model, train_loader, val_loader, optimizer, num_epochs, output):
    
    model.train()  # Set model to training mode
    best_val_loss = 0.0
    print("Training the ViT model for {} epochs...".format(num_epochs))

    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()   
            
            # Forward pass, calculate loss
            loss, pred, mask = model(images)
            # print(loss)
            loss.sum().backward()
            optimizer.step()

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


def mae_pretrain(args, device_id):
    # Create DataLoader for training and validation
    dataloaders = imagenet_distribute(args=args)
    logging.info("train_size:{}, val_size:{}, test_size:{}".format(len( dataloaders['train']), len( dataloaders['val']), len( dataloaders['val'])))
    

    # Create ViT model
    model = mae_vit_base_patch16(args.pretrained)
    model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=False)

    
    save_path = os.path.join(args.output, args.savefile)
    if args.reload:
        if os.path.exists(os.path.join(save_path, "best_mae_pretrain_model.pth")):
            model.load_state_dict(torch.load(os.path.join(save_path, "best_mae_pretrain_model.pth")))
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Pretrain the model
    pretrain_model(model, dataloaders['train'], dataloaders['val'], criterion, optimizer, args.num_epochs, device_id=device_id)
    dist.destroy_process_group()

def mae_pretrain_ddp(args):
    log(args=args)
    args.world_size = int(os.environ['SLURM_NTASKS'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME']) #str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = "29500"
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']
    print("MASTER_ADDR:{}, MASTER_PORT:{}, WORLD_SIZE:{}, WORLD_RANK:{}, local_rank:{}".format(os.environ['MASTER_ADDR'], 
                                                    os.environ['MASTER_PORT'], 
                                                    os.environ['WORLD_SIZE'], 
                                                    os.environ['RANK'],
                                                    local_rank))
    dist.init_process_group(                                   
    	backend='nccl',                                         
   		init_method='env://',                                   
    	world_size=args.world_size,                              
    	rank=int(os.environ['RANK'])                                               
    )
    print("SLURM_LOCALID/lcoal_rank:{}, dist_rank:{}".format(local_rank, dist.get_rank()))

    print(f"Start running basic DDP example on rank {local_rank}.")
    device_id = local_rank % torch.cuda.device_count()
    mae_pretrain(args=args, device_id=device_id)
    
    dist.destroy_process_group()
    

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, output):
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

def mae_finetune(args):
    log(args=args)

    # Create datasets
    dataloaders = imagenet(args=args)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    
    train_size = len(train_loader)
    val_size = len(val_loader)
    logging.info("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, val_size))
    
    # Create ViT model
    model = mae_vit_base_patch16()
    model = nn.DataParallel(model)
    if args.reload:
        if os.path.exists(os.path.join(args.output, "best_mae_model.pth")):
            model.load_state_dict(torch.load(os.path.join(args.output, "best_mae_model.pth")))
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs)

