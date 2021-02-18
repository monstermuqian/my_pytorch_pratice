import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# 此代码用于定义我的validation过程

# 定义我的图片裁剪器
my_transform = transforms.Compose([
    transforms.Resize((500, 500)),
    #transforms.ColorJitter(),
    #transforms.RandomCrop(224),
    #transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
])

# 定义我的设备
my_gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 来获取我们的数据集
# 在这里决定要不要使用裁剪它形状的transformer
class DogCat(Dataset):
    def __init__(self, root, transform=None):
        images = os.listdir(root)
        self.img = [os.path.join(root, img) for img in images]
        self.transforms = transform

    def __getitem__(self, index):
        image = self.img[index]
        label = torch.tensor([0, 0], dtype=float)
        # label, postion, dog = [0], cat = [1]
        if 'dog' in image.split('/')[-1]:
            label[0] = 1
            # label = 0
        else:
            label[1] = 1
        data = Image.open(image)
        if self.transforms:
            data = self.transforms(data)
        return data, label
    def __len__(self):
        return len(self.img)




# 定义我们的DataLoader for testing
dataset_valid = DogCat('./valid_new', transform=my_transform)
size_valid = dataset_valid.__len__()
validloader = DataLoader(dataset_valid, batch_size=1, shuffle=True)


# 加载模型
net_valid = torchvision.models.densenet121(pretrained=True)
num_ftrs = net_valid.classifier.in_features
net_valid.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 500),
    nn.ReLU(),
    nn.Linear(500, 100),
    nn.Linear(100, 2),
    nn.Softmax(dim=1)
)
net_valid = net_valid.to(my_gpu)
criterion = nn.BCEWithLogitsLoss()


if __name__ == "__main__":
    PATH_model = './my_model_desnet121.pth'
    net_valid.load_state_dict(torch.load(PATH_model))



    total_correct = 0.0
    total_loss = 0.0
    num_count = 0
<<<<<<< HEAD
    loss = 0.0
    with torch.no_grad():
        net_valid.eval()
        for i, data in enumerate(validloader, 0):
            img, labels = data[0].to(my_gpu), data[1].to(my_gpu)
            outputs = net_valid(img)
            loss = criterion(outputs.float(), labels)
            total_loss += loss
            # print("output is :{}".format(outputs))
            _, predicted = torch.max(outputs, 1)
            # print("predicted is :{}".format(predicted))
            # print("labels is :{}".format(labels))
            _, labels = torch.max(labels, 1)
            result = torch.eq(predicted, labels)
            result = result.sum().item()



            num_count += outputs.size()[0]
            total_correct += result
            if i % 200 == 199:
                print('当前准确率是： %.5f%%' %
                      (100 * total_correct / num_count))
                print('当前的loss为： %.5f ' % (total_loss / 200))
                total_loss = 0.0
=======
    for i, data in enumerate(validloader, 0):
        img, labels = data[0].to(my_gpu), data[1].to(my_gpu)
        outputs = net_valid(img)
        __, predicted = torch.max(outputs, 1)
        #print(outputs)
        #print(predicted)
        #print(labels)
        result = (predicted == labels).sum().item()
        num_count += outputs.size()[0]
        total_correct += result
        if i % 200 == 199:
            print('当前准确率是： %.5f%%' %
                  (100 * total_correct / num_count))
>>>>>>> b24f77756fa9be94c0592bb8f58162e71ea43139

    print('The correct rate on the validation set is: %.3f%%' %
          (100 * total_correct / size_valid))



