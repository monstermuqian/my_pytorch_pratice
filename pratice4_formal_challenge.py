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
    transforms.Resize((500, 500)),
    #transforms.ColorJitter(),
    #transforms.RandomCrop(224),
    #transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(degrees=15),
    transforms.Resize(256),
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
        label = torch.tensor([0, 0], dtype=float)
        # label, postion, dog = [0], cat = [1]
        if 'dog' in image.split('/')[-1]:
            label[0] = 1
        else:
            label[1] = 1
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
trainloader = DataLoader(dataset_train, batch_size=1, shuffle=True, num_workers=2)






# 开始训练
if __name__ == '__main__':


    # torch.cuda.empty_cache()
    #
    if False:
        running_loss = 0.0
        net = torchvision.models.densenet121(pretrained=True)
        num_ftrs = net.classifier.in_features
        net.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )

        # 定义优化器
        optimizer = optim.SGD(net.parameters(), lr=0.0003, momentum=0.9)
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCEWithLogitsLoss()


        # 决定是否继续承接上一个模型继续训练
        training_conti = True
        if training_conti:
            PATH = './my_model_desnet121.pth'
            net.load_state_dict(torch.load(PATH))

        net = net.to(my_gpu)

        for j in range(3):
            print("第%d次循环" % (j+1))
            for i, data in enumerate(trainloader, 0):
                imgs, labels = data[0].to(my_gpu), data[1].to(my_gpu)
                #imgs, labels = data
                optimizer.zero_grad()
                outputs = net(imgs)
                loss = criterion(outputs.float(), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                # scheduler1.step(loss)
                if i % 200 == 199:
                    print('第%d批，经过%d次训练，得到的loss为： %.5f' %
                          (j + 1, i+1, running_loss / 200))
                    running_loss = 0.0

        print('Training finished')

        # 计算训练集上的正确率
        # dataiter_train = iter(trainloader)
        # print(dataiter_train.__sizeof__())

        # 保存模型
        PATH = './my_model_desnet121.pth'
        # PATH = './my_model_net.pth'
        torch.save(net.state_dict(), PATH)


    if True:
        num_count = 0
        total_correct = 0
        net_test = torchvision.models.densenet121(pretrained=True)
        num_ftrs = net_test.classifier.in_features
        net_test.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.Linear(100, 2),
            nn.Softmax(dim=1)
        )
        PATH = './my_model_desnet121.pth'
        net_test.load_state_dict(torch.load(PATH))
        net_test.to(my_gpu)
        print('Training data set is : {}'.format(train_size))
        with torch.no_grad():
            net_test.eval()
            for i, data in enumerate(trainloader, 0):
                imgs, labels = data[0].to(my_gpu), data[1].to(my_gpu)
                #print(imgs.size())
                #print(labels)
                #imgs, labels = data
                outputs = net_test(imgs)
                # print("output is :{}".format(outputs))
                a, predicted = torch.max(outputs, 1)
                # print("predicted is :{}".format(predicted))

                b, labels = torch.max(labels, 1)
                # print("labels is :{}".format(labels))
                result = torch.eq(predicted, labels)
                result = result.sum().item()
                # print("result is :{}".format(result))
                #print(result)

                num_count += outputs.size()[0]
                total_correct += result
                # print('current correct classification is: {}'.format(total_correct))
                if i % 200 == 199:
                    print("current correct rate is : %.5f%% " %
                          (100 * total_correct / num_count))
        print('the correct rate on training set is : %.5f%%'%
              (100 * total_correct / train_size))








