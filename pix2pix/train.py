import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics.image
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import MapDataset
from discriminator import Discriminator
from generator import Generator
from utils import load_checkpoint, save_checkpoint, save_some_examples

generator_losses = []
discriminator_losses = []


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, epoch):
    loop = tqdm(loader, leave=True)

    for i, (label_img, real_img) in enumerate(loop):

        label_img = label_img.to(config.DEVICE)
        real_img = real_img.to(config.DEVICE)

        # Train Discriminator
        generated_img = gen(label_img)
        D_real = disc(label_img, real_img)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = disc(label_img, generated_img.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # Train generator
        D_fake = disc(label_img, generated_img)
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
        L1 = l1_loss(generated_img, real_img) * config.L1_LAMBDA
        G_loss = G_fake_loss + L1

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
        if (i == loop.__len__() - 1):
            generator_losses.append(G_loss.mean().item() / 10)
            discriminator_losses.append(D_loss.mean().item())


def train():
    disc = Discriminator(in_channel=1).to(config.DEVICE)
    gen = Generator(in_channels=1).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)

    train_dataset = MapDataset(source_dir='../datasets/Private_Dataset_NC_ART_PV/nc_img',
                               target_dir='../datasets/Private_Dataset_NC_ART_PV/pv_img')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    val_dataset = MapDataset(source_dir='../datasets/Private_Dataset_NC_ART_PV/nc_img_test',
                             target_dir='../datasets/Private_Dataset_NC_ART_PV/pv_img_test')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(config.NUM_EPOCHS + 1):
        train_fn(disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, epoch)
        if config.SAVE_MODEL and epoch % 20 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
        save_some_examples(gen, val_loader, epoch, folder='evaluation')

        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(generator_losses, label="Generator Loss")
        plt.plot(discriminator_losses, label="Discriminator Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()


def test():
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
    train_dataset = MapDataset(source_dir='../datasets/facades/train')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=config.NUM_WORKERS)
    gen.eval()
    mse_loss = torch.nn.MSELoss()
    rmse_train = []
    # for idx, (x, y) in enumerate(train_loader):
    #     x = x.to(config.DEVICE)
    #     y = y.to(config.DEVICE)
    #     y_fake = gen(x)
    #     rmse = torch.sqrt(mse_loss(y, y_fake))
    #     rmse_train.append(rmse)
    #     save_image(torch.concat((x, y, y_fake), 3), f'./evaluation/train_output/train_{idx}_{rmse:.3f}.png')
    # print(f'rmse_train={torch.tensor(rmse_train).mean().item()}')

    val_dataset = MapDataset(source_dir='../datasets/facades/val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    rmse_val = []
    ssim_img = []
    for idx, (x, y) in enumerate(val_loader):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)
        y_fake = gen(x)
        rmse = torch.sqrt(mse_loss(y, y_fake))
        rmse_val.append(rmse)
        ssim_loss = torchmetrics.image.StructuralSimilarityIndexMeasure()
        ssim = ssim_loss(y_fake, x)
        ssim_img.append(ssim)

    # save_image(torch.concat((x, y, y_fake), 3), f'./evaluation/val_output/val_{idx}_{rmse:.3f}.png')
    print(f'rmse_val={torch.tensor(rmse_val).mean().item()}')
    print(f'{torch.tensor(ssim_img).mean().item()}')


def evaluate():
    img_path = './images'
    mse_loss = torch.nn.MSELoss()
    ssim_loss = torchmetrics.image.StructuralSimilarityIndexMeasure()
    mse_img = []
    ssim_img = []
    for img in (os.listdir(img_path)):
        prefix = img[0:-10]
        suffix = img[-10:]
        if (suffix == 'real_B.png'):
            real_img = np.array(Image.open(img_path + '/' + prefix + 'real_B.png')).astype(dtype=np.float32) / 255
            real_img = torch.tensor(real_img).permute(2, 0, 1)
            fake_img = np.array(Image.open(img_path + '/' + prefix + 'fake_B.png')).astype(dtype=np.float32) / 255
            fake_img = torch.tensor(fake_img).permute(2, 0, 1)
            loss = mse_loss(real_img, fake_img)
            real_img = real_img.unsqueeze(0)
            fake_img = fake_img.unsqueeze(0)
            ssim = ssim_loss(real_img, fake_img)
            mse_img.append(loss)
            ssim_img.append(ssim)
    print(f'{torch.tensor(mse_img).mean().item()}')
    print(f'{torch.tensor(ssim_img).mean().item()}')


if __name__ == '__main__':
    train()
    # test()
    # evaluate()
    # psnr_val = 10 * math.log10(1 / (0.2246 ** 2))
    # print(psnr_val)
