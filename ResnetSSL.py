import torch
import torch.nn as nn

from torch import Tensor
from typing import Type,List

class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0,
            bias=False
        )
        
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=1, 
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        
        self.stride = stride
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity # add the original tensor to the output, this is the skip connection
        out = self.relu(out)
        return  out

        
class ResNet(nn.Module):
    def __init__(
        self, 
        block: Type[BasicBlock],
        layers: List[int],
        img_channels: int,
        num_classes: int
    ) -> None:
        
        super(ResNet, self).__init__()
        
        self.in_channels = 64
        self.expansion = 4
        
        self.conv1 = nn.Conv2d(
                in_channels=img_channels,
                out_channels=64,
                kernel_size=7, 
                stride=2,
                padding=3,
                bias=False
            )
        
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
        #resnet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels= 64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels= 128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels= 256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels= 512, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
     
        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc(x)
        return x
        
    def _make_layer (self, block:Type[BasicBlock], num_residual_blocks:int, out_channels: int, stride):
        
        downsample= None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels * 4:
            downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(out_channels * 4),)
        
        layers.append(block(self.in_channels,out_channels,downsample=downsample))
        self.in_channels = out_channels*4
            
        for i in range(num_residual_blocks-1):
            layers.append(block(
                self.in_channels,
                out_channels,
            ))
        return nn.Sequential(*layers)
    
    
def ResNet50(img_channels, num_classes):
    return ResNet(block=BasicBlock, layers=[3,4,6,3], img_channels=img_channels,num_classes=num_classes)

def test():
    net = ResNet50(img_channels=3, num_classes=1000)
    x = torch.randn(2,3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)

test()