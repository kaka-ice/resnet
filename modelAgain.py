#-*- coding = utf-8 -*- 
#@time:2022/6/7 16:32
#@Author:ice

import torch
from torch import nn as nn

class BasicBlock(nn.Module):
    expansion=1
    #in_channel表示输入通道数，out_channel表示第二层的通道数，不是指最后一层的通道数
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(BasicBlock, self).__init__()
        #通过传入stride，方便对特征图的分辨率进行调整
        self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        self.conv2=nn.Conv2d(out_channel,out_channel*self.expansion,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel*self.expansion)
        self.relu=nn.ReLU(inplace=True)

        self.downsample=downsample

    def forward(self,x):
        residual=x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.conv2(x)
        x=self.bn2(x)

        #如果需要进行下采样，则将残差边加入下采样操作
        if self.downsample:
            residual=self.downsample(residual)

        x+=residual
        x=self.relu(x)
        return x

class BottleNeck(nn.Module):
    #第一个卷积的通道数和第三个卷积的通道数倍数是4
    expansion=4
    #in_channel表示输入通道数，out_channel表示第二个卷积的通道数
    def __init__(self,in_channel,out_channel,stride=1,downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1=nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channel)
        #通过传入stride，方便对特征图的分辨率进行调整
        self.conv2=nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channel)
        self.conv3=nn.Conv2d(out_channel,out_channel*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channel*self.expansion)

        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample

    def forward(self,x):
        residual=x
        x=self.conv1(x)
        x=self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample:
            residual=self.downsample(residual)

        x+=residual
        x=self.relu(x)

        return x

class ResNet(nn.Module):
    """观察ResNet网络结构:
    1.需要定义的以上两个基本结构block,
    2.需要layer1/2/3/4层的数量(num_layers[2,2,2,2]),
    3.需要输入通道数init_channels=3,
    4.需要类别数num_class=10(手写)
    """
    def __init__(self,block_layer,num_layers,init_channels=3,num_class=10):
        super(ResNet, self).__init__()
        self.conv1=nn.Conv2d(init_channels,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(64)
        self.max_pool=nn.MaxPool2d(kernel_size=3,stride=2)

        #输入四个layer层的通道数均为64
        self.in_channel=64

        """
        定义4个layer
        1.第一个layer不需要进行下采样，只需要给第一个block_layer的downsample进行调整通道数
        2.第二三四个layer的第一个block_layer的downsample需要下采样和调整通道数
        """
        self.layer1=self._make_layer(block_layer,64,stride=1,num_layer=num_layers[0])
        self.layer2 = self._make_layer(block_layer, 128, stride=2, num_layer=num_layers[1])
        self.layer3 = self._make_layer(block_layer, 256, stride=2, num_layer=num_layers[2])
        self.layer4 = self._make_layer(block_layer, 512, stride=2, num_layer=num_layers[3])

        #平均池化层,刚好转换为1x1大小的图
        self.avg_pool=nn.AvgPool2d(kernel_size=7,stride=1)
        #全连接层，输入通道数为512*expansion
        self.fc=nn.Linear(512*block_layer.expansion,num_class)

    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.max_pool(x)
        x=self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x=self.avg_pool(x)

        #进行全连接层之前进行转换，[batch_size,num_class]
        x=x.view(x.size(0),-1)
        x=self.fc(x)

        return x

    """
    1.需要传入block构建网络，
    2.每个block的第二个卷积的通道数，
    3.一般不需要进行下采样，所以步长stride=1，
    4.每个layer的层数num_layer
    """
    def _make_layer(self,block_layer,out_channel,num_layer,stride=1):
        downsample = None
        """
        1.每一个layer的第一个block比较特殊，先构建第一个block
        2.如果stride!=1,即当stride=2需要进行下采样时候（位于两个layer交界处时），需要对残差边进行通道数与分辨率调整
        3.如果in_channel!=out_channel*block_layer.expansion，表示如果处于每个layer的第一层时候，需要进行通道数调整
        """
        if stride!=1 or self.in_channel!=out_channel*block_layer.expansion:
            downsample=nn.Sequential(
                #注意，此处传入的stride可以调节，即可以根据是否需要分辨率调整进行传值,卷积大小为1x1
                nn.Conv2d(self.in_channel,out_channel*block_layer.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channel*block_layer.expansion)
            )
        layer=[]
        #构建每一个layer的第一层，其是否需要进行进行下采样，可以由stride来调节,
        layer.append(block_layer(self.in_channel,out_channel,stride=stride,downsample=downsample))
        #layer的第二层输入通道数即为第一层的输出通道数，输出通道数也为第一层的输出通道数，注意此处的outchannel为第二个卷积的通道数
        self.in_channel=out_channel*block_layer.expansion
        for i in range(1,num_layer):
            layer.append(block_layer(self.in_channel,out_channel,stride=1,downsample=None))

        return nn.Sequential(*layer)

def resnet18():
    model=ResNet(BasicBlock,[2,2,2,2],num_class=10)
    return model

def resnet34():
    model=ResNet(BasicBlock,[3,4,6,3],num_class=10)
    return model

def resnet50():
    model=ResNet(BottleNeck,[3,4,6,3],num_class=10)
    return model

def resnet101():
    model=ResNet(BasicBlock,[3,4,23,3],num_class=10)
    return model

def resnet152():
    model=ResNet(BasicBlock,[3,8,36,3],num_class=10)
    return model

if __name__ == '__main__':
    a=torch.randn(5,3,224,224)
    m=resnet18()
    o=m(a)
    #_,返回的是最大值，pre返回的索引
    _,pre=torch.max(o,1)

    print(_,pre)
    print(o.size())
