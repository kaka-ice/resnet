#-*- coding = utf-8 -*- 
#@time:2022/6/6 19:58
#@Author:ice
import torch
from torch import nn as nn

import torchvision.models.resnet


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.conv2=nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU(inplace=True)

        self.downsample=downsample

    def forward(self,x):
        residual=x
        #用作第二个layer后下采样的conv2
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.conv2(x)
        x=self.bn2(x)

        #如果默认为None，则不进行任何操作
        if self.downsample:
            residual=self.downsample(residual)

        x+=residual
        x=self.relu(x)

        return x


class BottleNeck(nn.Module):
    expansion=4
    #out_channel指示的是第二层的channels，并非指最后一层的channels
    def __init__(self,in_channels,out_channels,stride=1,downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels)
        #用作分辨率下采样的操作
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.conv3=nn.Conv2d(out_channels,out_channels*self.expansion,kernel_size=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels*self.expansion)

        self.relu=nn.ReLU(inplace=True)

        #传递到每一个大layer层中，使用第一个layer来做stride=2下采样(第一个make_layer的第一层不需要下采样，后三个make_layer的第一层进行下采样)
        #如果需要下采样，则在resnet网络构造中给downsample赋予意义，
        self.downsample=downsample

    def forward(self,x):
        residual=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample:
            residual=self.downsample(residual)

        x+=residual
        x=self.relu(x)

        return x

class ResNet(nn.Module):
    #101层num_layers:[3,4,23,3] 每一个layer层的数量，input_channels输入通道数
    def __init__(self,block,num_layes,n_classes=10,input_channels=3):
        super(ResNet, self).__init__()
        #正式输入维度
        self.in_channels=64
        self.conv1=nn.Conv2d(input_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.relu=nn.ReLU(inplace=True)

        self.layer1=self._make_layer(block,64,num_layes[0],stride=1)
        self.layer2 = self._make_layer(block, 128, num_layes[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_layes[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_layes[3], stride=2)

        #平均池化层
        self.avgpool=nn.AvgPool2d(kernel_size=7,stride=1)
        #全连接层
        self.fc=nn.Linear(block.expansion*512,n_classes)

        #初始化值
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif issubclass(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1.0)
                nn.init.constant_(m.bias,0.0)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.maxpool(x)
        x=self.relu(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=self.avgpool(x)
        #全连接层要进行view进行展平 一张图展开成为一个向量
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return  x

    #out_channels:输出通道数，是每一个小层的第二个通道数的输出层 1x1,64 3x3,64
    def _make_layer(self,block,out_channels,num_block,stride=1):
        downsample=None
        '''downsample是对输入层特征图进行
        1.调整特征图的分辨率，在第二、三、四个layer的第一个conv进行调整（顺便还调整了通道数）
        2.调整通道数，一般是在第一个layer的第一个conv进行调整，
        '''
        if stride!=1 or self.in_channels!=out_channels*block.expansion:
            downsample=nn.Sequential(
                nn.Conv2d(self.in_channels,out_channels*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels*block.expansion)
            )
            #搭建每一个layer层的个数
        layers=[]
        layers.append(block(self.in_channels,out_channels,stride,downsample))
        self.in_channels=out_channels*block.expansion
        for _ in range(1,num_block):
            layers.append(block(self.in_channels,out_channels,1,None))
        return nn.Sequential(*layers)


def resnet18():
    model=ResNet(BasicBlock,[2,2,2,2])
    return model

def resnet34():
    model=ResNet(BasicBlock,[3,4,6,3])
    return model

def resnet50():
    model=ResNet(BottleNeck,[3,4,6,3])
    return model

def resnet101():
    model=ResNet(BottleNeck,[3,4,23,3])
    return model

def resnet152():
    model=ResNet(BottleNeck,[3,8,36,3])
    return model

if __name__ == '__main__':
    a=torch.randn(5,3,224,224)
    m=resnet18()
    o=m(a)
    print(o.size())







