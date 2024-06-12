import torch
from torchvision.utils import save_image

import config


def denormalize(tensor):
    return (tensor + 1) / 2


def save_some_examples(gen, val_loader, epoch, folder):
    x, y, name = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    with torch.no_grad():
        y_fake = gen(x)
        # print(f"y_source_{epoch}.max={torch.max(y_fake)} y_source_{epoch}.min={torch.min(y_fake)}")
        y_fake = denormalize(y_fake)
        # print(f"y_gen_{epoch}.max={torch.max(y_fake)} y_gen_{epoch}.min={torch.min(y_fake)}")
        save_image(y_fake, folder + f"/y_gen_{epoch}.png")
        if epoch == 0:
            # print(f"label.max={torch.max(x)} label.min={torch.min(x)}")
            # print(f"real.max={torch.max(y)} real.min={torch.min(y)}")
            save_image(denormalize(x), folder + f"/label.jpeg")
            save_image(denormalize(y), folder + f"/real.jpeg")
    gen.train()


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device=config.DEVICE):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

class WGAN_GP_Loss:
    def __init__(self, discriminator, lambda_gp=10):
        self.discriminator = discriminator
        self.lambda_gp = lambda_gp

    def gradient_penalty(self, real_images, fake_images):
        batch_size, c, h, w = real_images.size()
        alpha = torch.rand(batch_size, 1, 1, 1).to(real_images.device)
        alpha = alpha.expand_as(real_images)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)

        d_interpolated = self.discriminator(interpolated)
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                        grad_outputs=torch.ones(d_interpolated.size()).to(real_images.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        return gradient_penalty

    def discriminator_loss(self, real_output, fake_output, real_images, fake_images):
        wasserstein_distance = torch.mean(fake_output) - torch.mean(real_output)
        gp = self.gradient_penalty(real_images, fake_images)
        return wasserstein_distance + gp

    def generator_loss(self, fake_output):
        return -torch.mean(fake_output)