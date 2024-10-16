import os
import sys
sys.path.append("./")
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from model.vae import VAE, vae_loss

import logging


# Configure logging
def log(args):
    os.makedirs(args.savefile, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.savefile, "out.log"),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
def main(args, device_id):
    log(args=args)
    
    # Create an instance of the U-Net model and other necessary components
    latent_dim = 128
    hidden_dim = 512 * 14 * 14  # This is flexible based on the decoder architecture
    output_channels = 3  # For RGB images
    img_size = 224  # Size of ImageNet images

    model = VAE(latent_dim, hidden_dim, output_channels, img_size)
    # criterion = vae_loss
    # best_val_score = 0.0
    
    # Move the model to GPU
    model.to(device_id)
    if args.reload:
        if os.path.exists(os.path.join(args.savefile, "best_score_model.pth")):
            model.load_state_dict(torch.load(os.path.join(args.savefile, "best_score_model.pth")))
    model = DDP(model, device_ids=[device_id], find_unused_parameters=False)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Define the learning rate scheduler
    milestones =[int(args.epoch*r) for r in [0.5, 0.75, 0.875]]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    
    # Split the dataset into train, validation, and test sets
    # Define data transformations for ImageNet
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load ImageNet dataset
    data_path = args.data_dir
    train_dataset = datasets.ImageFolder(root=data_path, transform=transform)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # dataset = PAIPTrans(data_path, args.resolution, fixed_length=args.fixed_length, patch_size=patch_size, normalize=False)
    # eval_set = MICCAIDataset(data_path, args.resolution, normalize=True, eval_mode=True)
    train_size = len(train_dataset)
    # train_size = int(0.85 * dataset_size)
    # val_size = dataset_size - train_size
    # test_size = val_size
    # logging.info("train_size:{}, val_size:{}, test_size:{}".format(train_size, val_size, test_size))
    
    # train_indices = list(range(0, train_size))
    # val_indices = list(range(train_size, dataset_size))
    # train_set = Subset(dataset, train_indices)
    # val_set = Subset(dataset, val_indices)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    # val_loader = DataLoader(val_set, batch_size=args.batch_size,  sampler=val_sampler)
    # test_loader = val_loader

    # Training loop
    num_epochs = args.epoch
    train_losses = []
    val_losses = []
    output_dir = args.savefile  # Change this to the desired directory
    os.makedirs(output_dir, exist_ok=True)
    import time
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        train_loader.sampler.set_epoch(epoch)
        start_time = time.time()
        step=1
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device_id)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            # print("train step loss:{}, sec/step:{}".format(loss, (time.time()-start_time)/step))
            epoch_train_loss += loss.item()
            step+=1
            if batch_idx % 100 == 0 and device_id==0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item() / len(data)}')
        end_time = time.time()
        logging.info("epoch cost:{}, sec/img:{}, lr:{}".format(end_time-start_time, (end_time-start_time)/train_size, optimizer.param_groups[0]['lr']))

        epoch_train_loss /= len(train_loader)
        train_losses.append(epoch_train_loss)
        scheduler.step()
                        
    # Save train and validation losses
    train_losses_path = os.path.join(output_dir, 'train_losses.pth')
    val_losses_path = os.path.join(output_dir, 'val_losses.pth')
    torch.save(train_losses, train_losses_path)
    torch.save(val_losses, val_losses_path)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default="paip", help='name of the dataset.')
    parser.add_argument('--data_dir', default="./dataset/paip/output_images_and_masks", 
                        help='base path of dataset.')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='resolution of img.')
    parser.add_argument('--pretrain', default="sam-b", type=str,
                        help='Use ResNet pretrained weigths.')
    parser.add_argument('--reload', default=True, type=bool,
                        help='Use reload val weigths.')
    parser.add_argument('--epoch', default=10, type=int,
                        help='Epoch of training.')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch_size for training')
    parser.add_argument('--savefile', default="./apt",
                        help='save visualized and loss filename')
    args = parser.parse_args()

    args.world_size = int(os.environ['SLURM_NTASKS'])
    
    log(args=args)
    
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
    main(args, device_id)
    
    dist.destroy_process_group()
