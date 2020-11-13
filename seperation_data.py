import os
import shutil

root = './train/train'
imgs = os.listdir(root)
imgs_path = [os.path.join(root, img) for img in imgs]
length = len(imgs_path)

# 将猫狗都分开
imgs_path_cat = list()
imgs_path_dog = list()
for i in range(length):
    if 'cat' in imgs_path[i]:
        imgs_path_cat.append(imgs_path[i])
    else:
        imgs_path_dog.append(imgs_path[i])


max_cat = int(len(imgs_path_cat)/2)
max_dog = int(len(imgs_path_dog)/2)

# 各取一半合并为训练集同验证集
imgs_path_train = imgs_path_cat[0:max_cat] + imgs_path_dog[0:max_dog]
imgs_path_valid = imgs_path_cat[max_cat:] + imgs_path_dog[max_dog:]

# print(len(imgs_path_dog))
# print(len(imgs_path_cat))
print(len(imgs_path_train))
print(len(imgs_path_valid))
# 成功获取训练集和验证集的所有图片路径

# 开始创建文件夹以及将数据移入
new_path_train = './train_new'
new_path_valid = './valid_new'

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
    else:
        print('---- there is already a folder -----')

mkdir(new_path_train)
mkdir(new_path_valid)

for i in range(len(imgs_path_train)):
    shutil.move(imgs_path_train[i], new_path_train)
for i in range(len(imgs_path_valid)):
    shutil.move(imgs_path_valid[i], new_path_valid)


