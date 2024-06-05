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


class MapDataset(Dataset):

    def get_image(self, images_dir, index):
        images = os.listdir(images_dir)
        self.image_name = images[index]
        img_path = os.path.join(images_dir, self.image_name)
        image = Image.open(img_path).convert('L')
        image = transform(image)
        image = np.array(image)
        return image

    def __init__(self, source_dir, target_dir):
        self.image_name = None
        self.source_dir = source_dir
        self.target_dir = target_dir

    def __len__(self):
        images_list = os.listdir(self.source_dir)
        return len(images_list)

    def __getitem__(self, index):
        source_image = self.get_image(self.source_dir, index)
        target_image = self.get_image(self.target_dir, index)

        return source_image, target_image, self.image_name


if __name__ == "__main__":

    dataset = MapDataset(source_dir="../datasets/Private_Dataset_NC_ART_PV/nc_img_test",
                         target_dir="../datasets/Private_Dataset_NC_ART_PV/pv_img_test")
    loader = DataLoader(dataset, batch_size=1)
    i = 0
    for x, y, name in loader:
        print(x, y, name)
        # save_image(x, f"x{i}.png")
        # save_image(y, f"y{i}.png")
        i += 1
