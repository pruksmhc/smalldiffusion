import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from torchvision.datasets import FashionMNIST
from torchvision.utils import make_grid, save_image
from torch_ema import ExponentialMovingAverage as EMA
import argparse
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

from smalldiffusion import (
    ScheduleDDPM, samples, training_loop, MappedDataset, DiT,
    img_train_transform, img_normalize, GrayscaleImageFolder
)

def main(train_batch_size=1024, epochs=300, sample_batch_size=64, patch_size=2, depth=6, num_heads=6, mlp_ratio=4.0, head_dim=32, personal_img_folder=None):
    # Setup
    a = Accelerator()
    
    # Setup TensorBoard
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs', f'fashion_mnist_{current_time}')
    writer = SummaryWriter(log_dir)
    
    if personal_img_folder:
        dataset = MappedDataset(GrayscaleImageFolder(personal_img_folder, transform=img_train_transform),
                            lambda x: x[0])
    else:
        dataset = MappedDataset(FashionMNIST('datasets', train=True, download=True,
                                         transform=img_train_transform),
                            lambda x: x[0])
    in_dim = dataset[0].shape[-1]
    loader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)
    schedule = ScheduleDDPM(beta_start=0.0001, beta_end=0.02, N=1000)
    model = DiT(in_dim=in_dim, channels=1, patch_size=patch_size, depth=depth, head_dim=head_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)

    # Train
    ema = EMA(model.parameters(), decay=0.99)
    ema.to(a.device)
    step = 0
    for idx, ns in enumerate(training_loop(loader, model, schedule, epochs=epochs, lr=1e-3, accelerator=a)):
        ns.pbar.set_description(f'Loss={ns.loss.item():.5}')
        # Log loss to TensorBoard
        writer.add_scalar('Loss/train', ns.loss.item(), step)
        step += 1
        ema.update()
        if idx % 1000 == 0:
            with ema.average_parameters():
                _, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                                  batchsize=sample_batch_size, accelerator=a)
                sample_grid = img_normalize(make_grid(x0))
                save_image(sample_grid, f'samples_{idx}.png')

    # Sample
    with ema.average_parameters():
        *xt, x0 = samples(model, schedule.sample_sigmas(20), gam=1.6,
                          batchsize=sample_batch_size, accelerator=a)
        # Save sample images
        sample_grid = img_normalize(make_grid(x0))
        save_image(sample_grid, 'samples.png')
        # Log sample images to TensorBoard
        writer.add_image('Samples', sample_grid, 0)
        torch.save(model.state_dict(), 'checkpoint.pth')
    
    writer.close()

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Train DiT on FashionMNIST')
    parser.add_argument('--patch_size', type=int, default=2, help='Patch size for DiT')
    parser.add_argument('--depth', type=int, default=6, help='Number of transformer layers (depth) for DiT')
    parser.add_argument('--num_heads', type=int, default=6, help='Number of attention heads for DiT')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio for DiT')
    parser.add_argument('--train_batch_size', type=int, default=1024, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--sample_batch_size', type=int, default=64, help='Batch size for sampling')
    parser.add_argument('--personal_img_folder', type=str, default=None, help='Path to personal image folder')
    args = parser.parse_args()
    main(args.train_batch_size, args.epochs, args.sample_batch_size, args.patch_size, args.depth, args.num_heads, args.mlp_ratio, args.head_dim, args.personal_img_folder)
