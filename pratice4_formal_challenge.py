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
import matplotlib.pyplot as plt

# 定义我的图片裁剪器
my_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ColorJitter(),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(128),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 再去搞懂这个normalize是怎么回事
])

# 定义我的设备
my_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 来获取我们的数据集
# 在这里决定要不要使用裁剪它形状的transformer
class DogCat(Dataset):
    def __init__(self, root, transform=None):
        images = os.listdir(root)
        self.img = [os.path.join(root, img) for img in images]
        self.transforms = transform

    def __getitem__(self, index):
        image = self.img[index]
        label = torch.tensor([0])
        # label, postion, dog = [0], cat = [1]
        if 'dog' in image.split('/')[-1]:
            label = 0
        else:
            label = 1
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
trainloader = DataLoader(dataset_train, batch_size=4, shuffle=True, num_workers=2)



# 定义神经网络结构
# conv1->relu->maxpool->conv2->relu->fullconnet1->fullconnect2->fc3


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 94, 3)
        self.conv2 = nn.Conv2d(94, 10, 3)
        self.conv3 = nn.Conv2d(10, 3, 3)
        self.conv4 = nn.Conv2d(3, 5, 3)
        #self.conv5 = nn.Conv2d(5, 10, 2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.fc1 = nn.Linear(5*19*19, 100)
        self.fc2 = nn.Linear(100, 2)
    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = f.relu(self.conv2(x))
        x = self.pool1(x)
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = self.pool2(x)
        #print(x.size())
        x = x.view(x.size()[0], 5*19*19)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化我的网络
# net = Net()


net = torchvision.models.densenet121(pretrained=True)
num_ftrs = net.classifier.in_features
net.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 500),
    nn.ReLU(),
    nn.Linear(500, 2)
)

net = net.to(my_gpu)



# 定义优化器
optimizer = optim.Adam(net.parameters(), lr=0.002, amsgrad=True)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[200, 400, 600], gamma=0.5)
criterion = nn.CrossEntropyLoss()




# 开始训练
if __name__ == '__main__':
    # 测试trainloader
    # print(len(trainloader))
    test = False
    if test:
        data_iter = iter(trainloader)
        img, labels = next(data_iter)
        img, labels = img.to(my_gpu), labels.to(my_gpu)
        output = net(img)
        #print(net)
        print(img.size())
        print(output.size())
        print(output)
        print(labels.size())
        print(labels)
        loss = criterion(output, labels)
        print(loss)
        #output = torch.squeeze(output)
        #print(output.size())
        #loss = criterion(output, labels)
        #print(loss)

    # 决定是否继续承接上一个模型继续训练
    training_conti = False
    if training_conti:
        PATH = './my_model_desnet121.pth'
        net.load_state_dict(torch.load(PATH))
    if False:
        dataiter = iter(trainloader)
        img, label = next(dataiter)
        plt.figure(figsize=(8, 16))
        grid_imgs = torchvision.utils.make_grid(img)
        np_grid_imgs = grid_imgs.numpy()
        print(np_grid_imgs[0])
        # 在tensor中，图片的格式是(batch, width, height),现在要转成(width, height, batch), 就是将原来的维度（0，1，2）
        # 换成维度（1，2，0）
        plt.imshow(np.transpose(np_grid_imgs, (1, 2, 0)))
        print(img.size())
        print(label)



    running_loss = 0.0
    # dataiter = iter(trainloader)


    #for j in range(8):
    if False:
        print("第%d次循环" % (j+1))
        for i, data in enumerate(trainloader, 0):
            imgs, labels = data[0].to(my_gpu), data[1].to(my_gpu)
            #imgs, labels = data
            optimizer.zero_grad()
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()
            scheduler.step()
            if i % 200 == 199:
                print('第%d批，经过%d次训练，得到的loss为： %.5f' %
                      (j + 1, i+1, running_loss / 200))
                running_loss = 0.0

    print('Training finished')

    # 计算训练集上的正确率
    # dataiter_train = iter(trainloader)
    # print(dataiter_train.__sizeof__())
    total_correct = 0


    # 保存模型
    PATH = './my_model_desnet121.pth'
    # PATH = './my_model_net.pth'
    torch.save(net.state_dict(), PATH)

    # 加载模型给检验训练
    # net_test = Net()

    net_test = torchvision.models.densenet121(pretrained=True)
    num_ftrs = net_test.classifier.in_features
    net_test.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        nn.ReLU(),
        nn.Linear(500, 2)
    )


    num_count = 0
    if True:
        net_test.load_state_dict(torch.load(PATH))
        net_test.to(my_gpu)
        for i, data in enumerate(trainloader, 0):
            imgs, labels = data[0].to(my_gpu), data[1].to(my_gpu)
            #print(imgs.size())
            #print(labels)
            #imgs, labels = data
            outputs = net_test(imgs)
            #print(outputs)
            __, predicted = torch.max(outputs, 1)
            #print(predicted)
            #print(labels)
            result = (predicted == labels).sum().item()
            #print(result)

            num_count += outputs.size()[0]
            total_correct += result
            if i % 200 == 199:
                print("current correct rate is : %.5f%% " %
                      (100 * total_correct / num_count))
        print('the correct rate on training set is : %.5f%%'%
              (100 * total_correct / train_size))








