import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import config
from dataset import MapDataset
from discriminator import Discriminator
from generator import Generator
from utils import load_checkpoint, save_checkpoint, save_some_examples


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce):
    loop = tqdm(loader, leave=True)

    for i, (x, y) in enumerate(loop):

        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        generated_img = gen(x)
        D_real = disc(x, y)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = disc(x, generated_img.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        if i%5 == 0 :
            disc.zero_grad()
            D_loss.backward()
            opt_disc.step()

        # Train generator
        D_fake = disc(x, generated_img)
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(generated_img, y)  # * config.L1_LAMBDA
        G_loss = G_fake_loss + L1 / 100

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        if i % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
                D_fake_loss=D_fake_loss.mean().item(),
                D_real_loss=D_real_loss.mean().item(),
                D_loss=D_loss.mean().item(),

                G_fake_loss=G_fake_loss.mean().item(),
                L1_loss=L1.mean().item(),
                G_loss=G_loss.mean().item(),
            )


def train():
    disc = Discriminator(in_channel=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE*10, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    train_dataset = MapDataset(root_dir='../datasets/facades/train')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    val_dataset = MapDataset(root_dir='../datasets/facades/val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE)
        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
        save_some_examples(gen, val_loader, epoch, folder='evaluation')


def test():
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
    train_dataset = MapDataset(root_dir='../datasets/facades/train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)
    gen.eval()
    for idx, (x, y) in enumerate(train_loader):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        y_fake = gen(x)
        save_image(torch.concat((x, y, y_fake), 3), f'./evaluation/train_output/train_{idx}.png')
    val_dataset = MapDataset(root_dir='../datasets/facades/val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    for idx, (x, y) in enumerate(val_loader):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        y_fake = gen(x)
        save_image(torch.concat((x, y, y_fake), 3), f'./evaluation/val_output/val_{idx}.png')


if __name__ == '__main__':
    train()
    # test()
