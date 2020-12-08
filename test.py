import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt


my_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 再去搞懂这个normalize是怎么回事
])



# 来获取我们的数据集
# 在这里决定要不要使用裁剪它形状的transformer
class DogCat(Dataset):
    def __init__(self, root, transform=None):
        images = os.listdir(root)
        self.img = [os.path.join(root, img) for img in images]
        self.transforms = transform

    def __getitem__(self, index):
        image = self.img[index]
        # label, postion, dog = [0], cat = [1]
        label = 0 if 'dog' in image.split('/')[-1] else 1
        data = Image.open(image)
        if self.transforms:
            data = self.transforms(data)
        return data, label
    def __len__(self):
        return len(self.img)


data = DogCat('./train_new', my_transform)
img_path = data.img[0]
img = Image.open(img_path)
img.show()
img_numpy = np.array(img)
plt.figure()
plt.imshow(img_numpy)
