import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
#import matplotlib.pyplot as plt

# 定义我的图片裁剪器
my_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 再去搞懂这个normalize是怎么回事
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

classes = ('dog', 'cat')



# 定义我们的DataLoader for training
dataset_train = DogCat('./train_new', my_transform)
train_size = dataset_train.__len__()
trainloader = DataLoader(dataset_train, batch_size=1, shuffle=True)

# 测试trainloader
# print(len(trainloader))
# dataiter = iter(trainloader)
# img, label = next(dataiter)
# print(img.size())
# print(label)

# 定义神经网络结构
# conv1->relu->maxpool->conv2->relu->fullconnet1->fullconnect2->fc3

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 72, 5)
        self.conv2 = nn.Conv2d(72, 48, 5)
        self.conv3 = nn.Conv2d(48, 36, 5)
        self.conv4 = nn.Conv2d(36, 18, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(36450, 2000)
        self.fc2 = nn.Linear(2000, 160)
        self.fc3 = nn.Linear(160, 80)
        self.fc4 = nn.Linear(80, 20)
        self.fc5 = nn.Linear(20, 2)
    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = x.view(-1, 36450)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        x = self.fc5(x)
        return x

# 实例化我的网络
net = Net()
net = net.to(my_gpu)



# 定义优化器
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()



# 开始训练
if __name__ == '__main__':
    running_loss = 0.0
    dataiter = iter(trainloader)
    count = 0

    # for epoch in range(2):
    if False:
        for i, data in enumerate(dataiter, 0):
            imgs, labels = data[0].to(my_gpu), data[1].to(my_gpu)
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss
            if i % 2000 == 0:
                print('第%d批，经过%d次训练，得到的loss为： %.3f' %
                      (epoch + 1, i, running_loss / 2000))
                running_loss = 0.0
        # while False:
        #     try:
        #         imgs, labels = next(dataiter)
        #     except StopIteration:
        #         print('All training data are loaded, exit from loop')
        #         break
        #     imgs = imgs.to(my_gpu)
        #     labels = labels.to(my_gpu)

        # 开始训练
        #     optimizer.zero_grad()

        #    outputs = net(imgs)
        #     loss = criterion(outputs, labels)
        #     loss.backward()
        #     optimizer.step()

        #     running_loss += loss
        #     count += 1
        #     if count % 2000 == 0:
        #         print('训练了%d次，其误差为 %.3f' %
        #                 (count, running_loss / 2000))
        #        running_loss = 0
    print(count)
    # for i, data in enumerate(trainloader, 0):
      #   inputs, labels = data[0].to(my_gpu), data[1].to(my_gpu)

        # 开始训练
        # optimizer.zero_grad()
        # outputs = net(inputs)
        # loss = criterion(outputs, labels)
        # loss.backward()
        # optimizer.step()

        # 记录每次的误差
        # running_loss += loss.item()

        # if i % 200 == 199:
        #     print('[%d] loss: %.3f' %
        #         (i + 1, running_loss / 2000))
        #     running_loss = 0
    print('Training finished')

    # 计算训练集上的正确率
    dataiter_train = iter(trainloader)
    print(dataiter_train.__sizeof__())
    total_correct = 0


    # 保存模型
    PATH = './my_model_net.pth'
    torch.save(net.state_dict(), PATH)

    # 加载模型给检验训练
    net_test = Net()
    net_test.load_state_dict(torch.load(PATH))
    net_test.to(my_gpu)
    '''
    语法
以下是 enumerate() 方法的语法:

enumerate(sequence, [start=0])
参数
sequence -- 一个序列、迭代器或其他支持迭代对象。
start -- 下标起始位置。
    '''
    for i, data in enumerate(dataiter_train, 0):
        imgs, labels = data[0].to(my_gpu), data[1].to(my_gpu)
        outputs = net_test(imgs)
        __, predicted = torch.max(outputs, 1)
        #print(predicted)
        #print(labels)
        result = (predicted == labels).sum().item()
        total_correct += result

    # while True:
    #     try:
    #         imgs, labels = next(dataiter_train)[0].to(my_gpu), next(dataiter_train)[1].to(my_gpu)
    #         # imgs, labels = next(dataiter_train)
    #     except StopIteration:
    #         print('all data from training set are loaed, exit from the loop')
    #         break
    #         outputs = net(imgs)
    #         __, predicted = torch.max(outputs, 1)
    #         print(predicted)
    #         print(labels)
    #         result = (predicted == labels).sum().item()
    #         total_correct += result

    print('the correct rate on training set is : %.3f%%'%
          (100 * total_correct / train_size))








