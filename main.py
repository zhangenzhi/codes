import os
import argparse

from dataset.imagenet import imagenet_iter
from dataset.btcv import btcv_iter

from train.vit_imagenet import vit_train
from train.vit_imagenet_ddp import vit_ddp
from train.unet3d_btcv import unet3d_btcv
from train.unetr_btcv import unetr_btcv

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet DataLoader Example')
    parser.add_argument('--task', type=str, default='imagenet', help='Type of task')
    parser.add_argument('--logname', type=str, default='train.log', help='logging of task.')
    parser.add_argument('--output', type=str, default='./output', help='output dir')
    parser.add_argument('--gpus', type=int, default=8, help='Epochs for iteration')
    parser.add_argument('--nodes', type=int, default=1, help='Epochs for iteration')
    parser.add_argument('--data_dir', type=str, default='/Volumes/data/dataset/imagenet', help='Path to the ImageNet dataset directory')
    parser.add_argument('--num_epochs', type=int, default=3, help='Epochs for iteration')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for DataLoader')
    parser.add_argument('--pretrained', type=bool, default=False, help='Use pretrained weights')
    parser.add_argument('--reload', type=bool, default=False, help='Reuse previous weights')
    parser.add_argument('--2d', type=bool, default=True, help='Use flat the 3d mri image to 2d.')
    
    args = parser.parse_args()
    return args

def main(args):
    args.output = os.path.join(args.output, args.task)
    os.makedirs(args.output, exist_ok=True)
    
    if args.task == "imagenet":
        imagenet_iter(args=args)
    elif args.task == "vit_imagenet":
        vit_train(args=args)
    elif args.task == "vit_imagenet_ddp":
        vit_ddp(args=args)
    elif args.task == "btcv":
        btcv_iter(args=args)
    elif args.task == "unet3d_btcv":
        unet3d_btcv(args=args)
    elif args.task == "unet3d_btcv_ddp":
        unet3d_btcv(args=args)
    elif args.task == "unetr_btcv":
        unetr_btcv(args=args)
    else:
        raise "No such task."
    
if __name__ == '__main__':
    args = parse_args()
    main(args=args)