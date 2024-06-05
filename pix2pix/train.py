import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from torchvision.utils import save_image
from tqdm import tqdm

import config
from dataset import MapDataset
from discriminator import Discriminator
from generator import Generator
from utils import load_checkpoint, save_checkpoint, save_some_examples, denormalize

generator_losses = []
discriminator_losses = []


def mae(img1, img2):
    mae = np.mean(abs(img1 - img2))
    return mae


def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = np.max(img1)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def train_fn(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, epoch):
    loop = tqdm(loader, leave=True)

    for i, (label_img, real_img, name) in enumerate(loop):

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
    output_path = './output'
    gen = Generator(in_channels=1).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    chcekpoint_gen = 'gen.pth.nc2pv.tar'
    load_checkpoint(chcekpoint_gen, gen, opt_gen, config.LEARNING_RATE)
    gen.eval()
    val_dataset = MapDataset(source_dir='../datasets/Private_Dataset_NC_ART_PV/nc_img_test',
                             target_dir='../datasets/Private_Dataset_NC_ART_PV/pv_img_test')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    for index, (nc_image, pv_image, image_name) in enumerate(val_loader):
        nc_image = nc_image.to(config.DEVICE)
        pv_fake = gen(nc_image)
        save_image(denormalize(pv_fake), os.path.join(output_path, image_name[0]))


def evaluate():
    pv_images_path = '../datasets/Private_Dataset_NC_ART_PV/pv_img_test'
    generated_images_path = './output'
    generated_images_dir = os.listdir(generated_images_path)
    sum_mse = 0
    sum_psnr = 0
    sum_ssim = 0
    for image_name in generated_images_dir:
        generated_image = cv2.imread(os.path.join(generated_images_path, image_name), cv2.IMREAD_GRAYSCALE)
        pv_test_image = cv2.imread(os.path.join(pv_images_path, image_name), cv2.IMREAD_GRAYSCALE)
        sum_mse += mean_squared_error(generated_image, pv_test_image)
        sum_psnr += psnr(generated_image, pv_test_image)
        sum_ssim += ssim(generated_image, pv_test_image)
    image_count = len(generated_images_dir)
    print(f"mse={sum_mse / image_count} psnr={sum_psnr / image_count} ssim={sum_ssim / image_count}")


if __name__ == '__main__':
    # train()
    # test()
    evaluate()
    # psnr_val = 10 * math.log10(1 / (0.2246 ** 2))
    # print(psnr_val)
