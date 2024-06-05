import os

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # 将图像转换为张量并自动添加通道维度
    transforms.Normalize(0.5, 0.5)
])


def get_image(images_dir, index):
    images = os.listdir(images_dir)
    image_file = images[index]
    img_path = os.path.join(images_dir, image_file)
    image = Image.open(img_path).convert('L')
    image = transform(image)
    image = np.array(image)

    return image


class MapDataset(Dataset):

    def __init__(self, source_dir, target_dir):
        self.source_dir = source_dir
        self.target_dir = target_dir

    def __len__(self):
        images_list = os.listdir(self.source_dir)
        return len(images_list)

    def __getitem__(self, index):
        source_image = get_image(self.source_dir, index)
        target_image = get_image(self.target_dir, index)

        return source_image, target_image


if __name__ == "__main__":

    dataset = MapDataset(source_dir="../datasets/Private_Dataset_NC_ART_PV/nc_img_test",
                         target_dir="../datasets/Private_Dataset_NC_ART_PV/pv_img_test")
    loader = DataLoader(dataset, batch_size=1)
    i = 0
    for x, y in loader:
        # save_image(x, f"x{i}.png")
        # save_image(y, f"y{i}.png")
        i += 1
