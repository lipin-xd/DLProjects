import torch
from torchvision.utils import save_image

import config


def denormalize(tensor):
    return (tensor + 1) / 2

def save_some_examples(gen, val_loader, epoch, folder):
    x, y = next(iter(val_loader))
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


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
