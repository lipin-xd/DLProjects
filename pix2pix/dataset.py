import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.utils import save_image


class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path)).astype(dtype=np.float32) / 255
        image = torch.tensor(image).permute(2, 0, 1)
        transform = transforms.Resize((256, 512))
        image = transform(image)

        input_image = image[:, :, 256:].clone().detach()
        target_image = image[:, :, :256].clone().detach()

        # augmentations = config.both_transform(image=input_image, image0=target_image)
        # input_image, target_image = augmentations['image'], augmentations['image0']
        # input_image = config.transform_only_input(image=input_image)['image']
        # target_image = config.transform_only_input(image=target_image)['image']

        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("../datasets/facades/val")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()
