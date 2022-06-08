#-*- coding = utf-8 -*- 
#@time:2022/6/7 16:15
#@Author:ice

import model
import torch
import torchvision
from torch import nn as nn
from torch import optim

import os
from torch.utils.data import DataLoader
from torchvision import datasets,transforms

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
'''
制作数据集并加载
1.数据路径data_dir
2.一次传入多少张图片
3.类型：train、val、test
'''
def get_data_loaders(data_dir,batch_size=64,type=None):
    if type=="train":
        #转换为张量格式
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
        #先制作数据集
        train_data=datasets.ImageFolder(os.path.join(data_dir,"train/"),transform=transform)
        print(len(train_data),len(train_data.classes),train_data.type())
        #然后加载数据集
        train_loader=DataLoader(train_data,batch_size,shuffle=True)
        return train_loader

    elif type=="val":
        #定义转换器，转换为张量格式
        transform=transforms.Compose([
            transforms.ToTensor()
        ])
        #传入图像数据，制作数据集
        val_list=datasets.ImageFolder(os.path.join(data_dir,'val/'),transform=transform)
        #加载数据集
        val_loader=DataLoader(val_list,batch_size=batch_size,shuffle=False)
        return val_loader

    elif type=="test":
        #定义转化器
        transform=torchvision.transforms.Compose([
            transforms.ToTensor()
        ])
        #制作测试集
        test_list=torchvision.datasets.ImageFolder(os.path.join(data_dir,"test/"),transform=transform)
        # print(len(test_list),len(test_list.classes))
        #加载测试集
        test_loader=torch.utils.data.DataLoader(test_list,batch_size=batch_size,shuffle=False)
        return test_loader

def showFig(dir):
    data_loader = get_data_loaders(dir, batch_size=10, type='test')

    class_dict = pd.read_csv('./data/class_dict.csv')
    classes = list(class_dict['class'])
    print(len(classes))

    # dateiter = iter(data_loader)
    # img, labels = dateiter.next()
    # img = img.numpy()

    for i,(img,labels) in enumerate(data_loader):
        print(labels)
        print(labels.size())
    # fig = plt.figure(figsize=(16, 4))
    # for idx in np.arange(6):
    #     ax = fig.add_subplot(2, int(6 / 2), idx + 1, xticks=[], yticks=[])
    #     # 输入pytorch为 c，h,w 需要转换为imgshow()的 h,w,c
    #     plt.imshow(np.transpose(img[idx], (1, 2, 0)))
    #     ax.set_title(classes[labels[idx]])
    # plt.show()


from tqdm import tqdm

if __name__ == '__main__':
    dir= 'data/'
    #获取训练数据
    train_loader=get_data_loaders(dir,10,'train')
    val_loader=get_data_loaders(dir,10,'val')
    test_loader=get_data_loaders(dir,10,'test')
    #设置训练参数
    num_epoch=100
    batch_size=64
    lr=0.01

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model=model.resnet18().to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)

    for epoch in range(1,num_epoch+1):
        print("\nEpoch:%d"%(epoch))
        #test_loader是将数据一个批次一个批次传进来 label表示每个批次的图片的类别，比如一个批次为6，label为[0, 0, 0, 0, 0, 1]
        for i,(img,label) in enumerate(tqdm(test_loader)):
            img=img.to(device)
            label=label.to(device)
            #训练模式
            model.train()
            #优化器梯度归零
            optimizer.zero_grad()
            #模型预测
            output=model(img)
            #计算loss
            loss=criterion(output,label)
            #梯度回传
            loss.backward()
            #更新梯度
            optimizer.step()
            #每迭代10个batch_size 10*10张图的时候判断一次，打印一下结果
            if i%10==0:
                correct=0
                total=0
                #_表示每一行最大的数值，predicted表示每一行最大数值的下标
                _,predicted=torch.max(output.data,dim=1)
                #表示训练集中的送入网络的数据个数，及准确率
                total+=label.size(0)
                correct+=(predicted==label).sum()

                print('[epoch:%d,iter:%d]Loss:%.3f%|acc:%.3f%'
                      %(epoch,i*batch_size,loss.item(),(100*correct/total)))

        #每经历一次epoch 则使用验证集来验证一下模型的准确性
        with torch.no_grad():
            correct=0
            toral=0
            for img,label in tqdm(val_loader):
                model.eval()
                img,label=img.to(device),label.to(device)
                output=model(img)
                _,predicted=torch.max(output.data,dim=1)
                total+=label.size(0)
                correct+=(predicted==label).sum()
            print('Valid\'s acc is %.3f%'%(100*total/correct))























