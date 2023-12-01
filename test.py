# from random import random
#
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import torch
# import torchvision.transforms as transforms
# # 加载CIFAR-10数据集
# import torchvision
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import numpy as np
#
#
# def load_cifar_batch(filename):
#     with open(filename, 'rb') as f:
#         datadict = pickle.load(f, encoding='bytes')
#         X = datadict[b'data']
#         Y = datadict[b'labels']
#         X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
#         Y = np.array(Y)
#         return X, Y
#
#
#
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
# # 加载训练集和测试集
# train_files = ['D:\PROJECT\pseudo_label\ABC-main\ABCcode0920\data\cifar-10-batches-py\data_batch_1', 'D:\PROJECT\pseudo_label\ABC-main\ABCcode0920\data\cifar-10-batches-py\data_batch_2', 'D:\PROJECT\pseudo_label\ABC-main\ABCcode0920\data\cifar-10-batches-py\data_batch_3', 'D:\PROJECT\pseudo_label\ABC-main\ABCcode0920\data\cifar-10-batches-py\data_batch_4', 'D:\PROJECT\pseudo_label\ABC-main\ABCcode0920\data\cifar-10-batches-py\data_batch_5']
# test_file = r'D:\PROJECT\pseudo_label\ABC-main\ABCcode0920\data\cifar-10-batches-py\test_batch'
# num_classes = 10
# train_data = []
# train_labels = []
# for train_file in train_files:
#     train_X, train_Y = load_cifar_batch(train_file)
#     train_data.append(train_X)
#     train_labels.append(train_Y)
# train_data = np.concatenate(train_data)
# train_data = transform_train(train_data)
# train_labels = np.concatenate(train_labels)
# test_data, test_labels = load_cifar_batch(test_file)
#
# # 可视化前20张图像
# fig, axes1 = plt.subplots(5,5,figsize=(3,3))
# for j in range(5):
#     for k in range(5):
#         i = np.random.choice(range(len(train_data)))
#         axes1[j][k].set_axis_off()
#         axes1[j][k].imshow(train_data[i:i+1][0])
#
# plt.show()

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 数据增强的变换
transform_train = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(15),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 加载CIFAR10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

# 对图像进行增强，并保存为npy格式
images = []
labels = []
for i, (image, label) in enumerate(trainloader):
    for j in range(image.size(0)):
        img = Image.fromarray(np.uint8(image[j].numpy() * 255))
        img = transform_train(img)
        images.append(img)
        labels.append(label[j])
images = np.array(images)
labels = np.array(labels)

# 可视化增强后的图像
fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
for i in range(2):
    for j in range(5):
        idx = np.random.randint(len(images))
        img = images[idx].numpy().transpose((1, 2, 0))
        img = (img * 0.5 + 0.5) * 255
        img = img.astype(np.uint8)
        ax[i, j].imshow(img)
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])
        ax[i, j].set_title("Label: {}".format(labels[idx]))
plt.show()















# # 数据增强的变换
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(15),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])
#
# # 加载CIFAR10数据集
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
#
# # 对图像进行增强，并保存为npy格式
# images = []
# labels = []
# for i, (image, label) in enumerate(trainloader):
#     for j in range(image.size(0)):
#         img = Image.fromarray(np.uint8(image[j].numpy() * 255))
#         img = transform_train(img)
#         images.append(img.numpy())
#         labels.append(label[j])
# images = np.array(images)
# labels = np.array(labels)
# np.save("cifar10_train_images.npy", images)
# np.save("cifar10_train_labels.npy", labels)
