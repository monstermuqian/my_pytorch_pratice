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

# 定义神经网络结构
# conv1->relu->maxpool->conv2->relu->fullconnet1->fullconnect2->fc3->fc4

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 12, 5)
#         self.conv2 = nn.Conv2d(12, 6, 5)
#         self.pool = nn.MaxPool2d(5, 5)
#         self.fc1 = nn.Linear(11094, 80)
#         self.fc2 = nn.Linear(80, 20)
#         self.fc3 = nn.Linear(20, 2)
#     def forward(self, x):
#         x = f.relu(self.conv1(x))
#         x = f.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.view(-1, 11094)
#         x = f.relu(self.fc1(x))
#         x = f.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


data = DogCat('./train_new', my_transform)
img_path = data.img[0]
img = Image.open(img_path)
img.show()
img_numpy = np.array(img)
plt.figure()
plt.imshow(img_numpy)

# 测试trainloader
# print(len(trainloader))
# test = True
# if test:
#     data_iter = iter(trainloader)
#     img, labels = next(data_iter)
#     #labels = labels.unsqueeze(1)
#     img, labels = img.to(my_gpu), labels.to(my_gpu)
#     output = net(img)
#     #print(net)
#     # print(img.size())
#     #         # print(output.size())
#     print(output)
#     #         # print(labels.size())
#     print(labels)
#     #output = output.float()
#     #labels = labels.float()
#     loss = criterion(output, labels)
#     print(loss)
#
#     # test for validation method
#     a, predicted = torch.max(output, 1)
#     #b, labels = torch.max(labels, 1)
#     print(predicted)
#     print(labels)
#     results = torch.eq(predicted, labels)
#     print(results)
#     results = results.sum().item()
#     print(results)
# output = torch.squeeze(output)
# print(output.size())
# loss = criterion(output, labels)
# print(loss)

# if False:
#     for i in range(5):
#         dataiter = iter(trainloader)
#         img, label = next(dataiter)
#         plt.figure(figsize=(8, 16))
#         grid_imgs = torchvision.utils.make_grid(img)
#         np_grid_imgs = grid_imgs.numpy()
#         print(np_grid_imgs[0])
#         # 在tensor中，图片的格式是(batch, width, height),现在要转成(width, height, batch), 就是将原来的维度（0，1，2）
#         # 换成维度（1，2，0）
#         plt.imshow(np.transpose(np_grid_imgs, (1, 2, 0)))
#         plt.show()
#         print(img.size())
#         print(label)


# running_correct = 0.0
# total_correct = 0.0
# loss = 0.0
# dataiter = iter(trainloader)


# 定义神经网络结构
# conv1->relu->maxpool->conv2->relu->fullconnet1->fullconnect2->fc3


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 94, 3)
#         self.conv2 = nn.Conv2d(94, 10, 3)
#         self.conv3 = nn.Conv2d(10, 3, 3)
#         self.conv4 = nn.Conv2d(3, 5, 3)
#         #self.conv5 = nn.Conv2d(5, 10, 2)
#         self.pool1 = nn.MaxPool2d(2, 2)
#         self.pool2 = nn.MaxPool2d(3, 3)
#         self.fc1 = nn.Linear(5*19*19, 100)
#         self.fc2 = nn.Linear(100, 2)
#     def forward(self, x):
#         x = f.relu(self.conv1(x))
#         x = f.relu(self.conv2(x))
#         x = self.pool1(x)
#         x = f.relu(self.conv3(x))
#         x = f.relu(self.conv4(x))
#         x = self.pool2(x)
#         #print(x.size())
#         x = x.view(x.size()[0], 5*19*19)
#         x = f.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# 实例化我的网络
# net = Net()


# 定义迭代器
# data_valid_iter = iter(validloader)
# count_true = 0
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


# scheduler2.step()
# torch.cuda.empty_cache()
# for i, data in enumerate(trainloader, 0):
#     imgs, labels =data[0].to(my_gpu), data[1].to(my_gpu)
#     outputs = net(imgs)
#     _, predicted = torch.max(outputs, 1)
#     results = torch.eq(predicted, labels)
#     total_correct = results.sum().item()
#
#     running_correct += total_correct
#
#     if i % 200 == 199:
#         print("第%d批训练后，当前准确率为 %.5f%%" % (j+1, running_correct / 200))
#         running_correct = 0


# 加载模型给检验训练
# net_test = Net()


# trainloader = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=2)