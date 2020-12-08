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
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 再去搞懂这个normalize是怎么回事
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
        # label, postion, dog = [0], cat = [1]
        label = 0 if 'dog' in image.split('/')[-1] else 1
        data = Image.open(image)
        if self.transforms:
            data = self.transforms(data)
        return data, label
    def __len__(self):
        return len(self.img)


# 定义神经网络结构
# conv1->relu->maxpool->conv2->relu->fullconnet1->fullconnect2->fc3->fc4

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 5)
        self.conv2 = nn.Conv2d(12, 6, 5)
        self.pool = nn.MaxPool2d(5, 5)
        self.fc1 = nn.Linear(11094, 80)
        self.fc2 = nn.Linear(80, 20)
        self.fc3 = nn.Linear(20, 2)
    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 11094)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 定义我们的DataLoader for testing
dataset_valid = DogCat('./valid_new', transform=my_transform)
size_valid = dataset_valid.__len__()
validloader = DataLoader(dataset_valid, batch_size=2, shuffle=True)


# 加载模型
net_valid = torchvision.models.densenet121(pretrained=True)
num_ftrs = net_valid.classifier.in_features
net_valid.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 500),
    nn.ReLU(),
    nn.Linear(500, 2)
)
net_valid = net_valid.to(my_gpu)

if __name__ == "__main__":
    PATH_model = './my_model_desnet121.pth'
    net_valid.load_state_dict(torch.load(PATH_model))

    # 定义迭代器
    data_valid_iter = iter(validloader)
    count_true = 0
    # 先验证一波输出是个什么玩意
    # imgs, labels = next(data_valid_iter)[0].to(my_gpu), next(data_valid_iter)[1].to(my_gpu)
    # outputs = net_valid(imgs)
    # __, predicted = torch.max(outputs, 1)
    # print(outputs)
    # print(predicted)
    # print(labels)
    # result = (predicted == labels).sum().item()
    # print(result)
    # 输出的东西是一个4×2（4行，2列） 的结果张量，符合4个batch有4个输出的表现，其中每一行都包括
    # 对两个类的预测结果，选出较高的那个作为最终预测结果


    '''
    while True:
        try:
            img, label = next(data_valid_iter)[0].to(my_gpu), next(data_valid_iter)[1].to(my_gpu)
        except StopIteration:
            break
        # 从迭代器中获得了图片数据以及其label后，怼入网络模型中
        outputs = net_valid(img)
        __, predicted = torch.max(outputs, 1)
        result = (predicted == label).sum().item()
        total_correct += result
    '''
    total_correct = 0
    num_count = 0
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
            print('操你妈的当前准确率是： %.5f%%' %
                  (100 * total_correct / num_count))

    print('The correct rate on the validation set is: %.3f%%' %
          (100 * total_correct / size_valid))



