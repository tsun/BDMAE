from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms, models
from tqdm import tqdm
from PIL import Image

import os

mean = [0.489, 0.409, 0.372]
std = [1, 1, 1]

val_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])

class VGGFace2_Dataset(Dataset):
    def __init__(self, dataset_path, split):
        self.dataset_path = dataset_path
        self.split = split
        self.transforms = val_transforms

    def build(self):
        self.get_image_list()
        targets = []
        images = []
        for i in tqdm(range(len(self.image_list))):
            img_path = self.image_list[i][0]
            if os.path.exists(img_path):
                img = Image.open(img_path)
                img_tmp = img.copy()
                images.append(img_tmp)
                img.close()
                targets.append(self.image_list[i][1])
            else:
                raise Exception('pic is not existed')

        self.length = len(targets)
        self.X = images
        self.y = targets

    def copy(self):
        return VGGFace2_Dataset(self.dataset_path, self.split)

    def get_image_list(self):
        classes = sorted(os.listdir(os.path.join(self.dataset_path, self.split)))
        self.image_list = []
        for i, cls in enumerate(classes):
            images = os.listdir(os.path.join(self.dataset_path, self.split, cls))
            for img in images:
                self.image_list.append((os.path.join(self.dataset_path, self.split, cls, img), i))

    def __getitem__(self, index):
        data = self.X[index] # PIL Image
        label = self.y[index] # np uint8
        # if self.transforms is not None:
        #     data = self.transforms(data)
        return (data, label)

    def __len__(self):
        return self.length



