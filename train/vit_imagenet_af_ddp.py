import os
import torch
from torch import nn
import torch.utils.data as data  # For custom dataset (optional)
import torchvision.transforms as transforms
import timm
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
import time
from torch.utils.data import DataLoader

# from model.vit import create_vit_model
from dataset.imagenet_ap import ImageNetDataset
from model.vit import AF_ViT

# Configure logging
def log(args):
    os.makedirs(os.path.join(args.output,args.savefile), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(os.path.join(args.output,args.savefile), "out.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device_id, save_path, seq_length):
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
    logging.info("Training the ViT model for %d epochs...", num_epochs)

    for epoch in range(num_epochs):
        start_time = time.time()
        logging.info("Epoch %d/%d", epoch + 1, num_epochs)
        running_loss = 0.0
        correct = 0
        for i, (image, seq_img, seq_size, seq_pos, labels) in enumerate(train_loader):
            # import pdb 
            # pdb.set_trace()
            seq_img = seq_img.to(device_id, non_blocking=True)
            seq_img = seq_img.view(-1, seq_length, 8*8*3) 
            seq_img = seq_img[:, :, :64]
            seq_pos = seq_pos.to(device_id, non_blocking=True)
            seq_size = seq_size.view(-1, seq_length, 1)
            seq_size = seq_size.to(device_id, non_blocking=True)
            # images = torch.reshape(images,shape=(-1,3,224, 224))
            labels = labels.to(device_id, non_blocking=True)
            optimizer.zero_grad()   
            
            # Forward pass, calculate loss
            # with torch.cuda.amp.autocast():
            # outputs = model(torch.cat([seq_img],dim=-1))
            outputs = model(seq_img)
            loss = criterion(outputs, labels)
                
            if torch.isnan(loss):
                import pdb
                pdb.set_trace()
            # Backward pass and optimize
            loss.backward()
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
            for image, seq_img, seq_size, seq_pos, labels in val_loader:
                seq_img = seq_img.to(device_id, non_blocking=True)
                seq_img = seq_img.view(-1, seq_length, 8*8*3) 
                seq_img = seq_img[:, :, :64]
                # images = torch.reshape(images,shape=(-1,3,224, 224))
                labels = labels.to(device_id, non_blocking=True)
                seq_pos = seq_pos.to(device_id, non_blocking=True)
                seq_size = seq_size.view(-1, seq_length, 1)
                seq_size = seq_size.to(device_id, non_blocking=True)
                
                # outputs = model(torch.cat([seq_img], dim=-1))
                outputs = model(seq_img)
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
            torch.save(model.state_dict(), os.path.join(save_path, "best_score_model.pth"))
            
        logging.info('Finished Training Step %d' % (epoch + 1))


    logging.info('Finished Training. Best Validation Accuracy: %.4f', best_val_acc)

def af_train(args, device_id):
    # Create DataLoader for training and validation
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir,"val")   
    train_set = ImageNetDataset(train_dir, fixed_length=args.seq_length, patch_size=8, sths=[1,3,5,7,9])
    val_set = ImageNetDataset(val_dir, fixed_length=args.seq_length, patch_size=8, sths=[1,3,5,7,9])
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    train_size = len(train_set)
    val_size = len(val_set)
    logging.info("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, val_size))
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, sampler=val_sampler)
    test_loader = val_loader

    # Create ViT model
    model = AF_ViT(num_classes=1000, seq_length=args.seq_length)
    model.to(device_id)
    model = DDP(model, device_ids=[device_id], find_unused_parameters=False)

    save_path = os.path.join(args.output, args.savefile)
    if args.reload:
        if os.path.exists(os.path.join(save_path, "best_score_model.pth")):
            model.load_state_dict(torch.load(os.path.join(save_path, "best_score_model.pth")))
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, device_id=device_id, save_path=save_path, seq_length=args.seq_length)

def vit_imagenet_af_ddp_train(args):
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
    af_train(args=args, device_id=device_id)
    
    dist.destroy_process_group()
