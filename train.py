from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.config import settings
from models.edsr import Discriminator, Generator
from models.loss import VGGLoss
from utils.dataset import DIV2K, Compose, Normalize, ToTensor


def train():
    # Setup data
    # Normalizing with standard values
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = Compose([ToTensor(), Normalize(mean, std)])

    # Placeholder path. User needs to provide DIV2K or similar dataset
    dataset_path = "Data/DIV2K_train_HR"
    if not Path(dataset_path).exists():
        print(f"Dataset path {dataset_path} not found. Please download DIV2K dataset.")
        print("Expected structure:")
        print("  Data/DIV2K_train_HR/img/ (Low res or high res images depending on setup)")
        print("  Data/DIV2K_train_HR/label/ (High res ground truth)")
        return

    dataset = DIV2K(root_dir=dataset_path, im_size=settings.high_res, scale=4, transform=transform)
    loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=True)

    # Models
    gen = Generator(in_channels=settings.in_channels, num_channels=settings.num_channels, num_blocks=settings.num_blocks).to(
        settings.device
    )

    disc = Discriminator(in_channels=settings.in_channels).to(settings.device)

    # Losses
    vgg_loss = VGGLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    # Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=settings.learning_rate, betas=(0.9, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=settings.learning_rate, betas=(0.9, 0.999))

    # Ensure weights dir exists
    Path(settings.weights_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(settings.num_epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{settings.num_epochs}")
        for _idx, (low_res, high_res) in enumerate(loop):
            low_res = low_res.to(settings.device)
            high_res = high_res.to(settings.device)

            # Train Discriminator
            fake = gen(low_res)
            disc_real = disc(high_res)
            disc_fake = disc(fake.detach())

            loss_disc_real = bce_loss(disc_real, torch.ones_like(disc_real))
            loss_disc_fake = bce_loss(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = loss_disc_fake + loss_disc_real

            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            disc_fake = disc(fake)
            # Adversarial loss
            loss_gen_adv = bce_loss(disc_fake, torch.ones_like(disc_fake))
            # VGG loss
            loss_gen_vgg = vgg_loss(fake, high_res)

            loss_gen = loss_gen_vgg + 1e-3 * loss_gen_adv

            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            loop.set_postfix(loss_gen=loss_gen.item(), loss_disc=loss_disc.item())

        # Save checkpoints
        if (epoch + 1) % 10 == 0 or epoch == settings.num_epochs - 1:
            torch.save(gen.state_dict(), Path(settings.weights_dir) / settings.checkpoint_gen)
            torch.save(disc.state_dict(), Path(settings.weights_dir) / "disc.pth.tar")
            print(f"Saved checkpoints at epoch {epoch + 1}")


if __name__ == "__main__":
    train()
