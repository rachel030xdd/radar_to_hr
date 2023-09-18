import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, ELU, Sequential, Conv1d, MaxPool1d, Module, BatchNorm1d, AvgPool1d, Dropout
from torch.nn import Conv2d, MaxPool2d, BatchNorm2d, AvgPool2d
from typing import Any, Callable, List, Optional, Type, Union
from torch import Tensor


class CNNRES_Monitor_Dilated3_hrrr(nn.Module):
    def __init__(self, args, device, in_channels = 2):
        super().__init__()
        self.args = args
        self.device = device
        self.in_channels = in_channels
        fc_size = 64
        out_class = 1
        
        self.maxpool2 = MaxPool1d(kernel_size=2, stride=2)
        self.avgpool1 = AvgPool1d(kernel_size=2, stride=2)

        self.conv5_32 = Sequential(Conv1d(32, 32, 5, padding=2), ELU(), Conv1d(32, 32, 5, padding=2))
        self.conv5_64 = Sequential(Conv1d(64, 64, 5, padding=4, dilation=2), ELU(), Conv1d(64, 64, 5, padding=4, dilation=2))
        self.conv3_128 = Sequential(Conv1d(128,128,3, padding=2, dilation=2), ELU(), Conv1d(128, 128, 3, padding=2, dilation=2))
        self.conv3_256 = Sequential(Conv1d(256, 256, 3, padding=4, dilation=4), ELU(), Conv1d(256, 256, 3, padding=4, dilation=4))
        self.conv3_512 = Sequential(Conv1d(512, 512, 3, padding=4, dilation=4), ELU(), Conv1d(512,512, 3, padding=4, dilation=4))
        self.conv3_1024 = Sequential(Conv1d(1024, 1024, 3, padding=8, dilation=8), ELU(), Conv1d(1024,1024, 3, padding=8, dilation=8))

        self.conv32to64 = Conv1d(32, 64, 3, padding=1)
        self.conv64to128 = Conv1d(64, 128, 3, padding=1)
        self.conv128to256 = Conv1d(128, 256, 3, padding=1)
        self.conv256to512 = Conv1d(256, 512, 3, padding=1)
        self.conv512to1024 = Conv1d(512, 1024, 3, padding=1)

        self.batch32 = BatchNorm1d(32)
        self.batch64 = BatchNorm1d(64)
        self.batch128 = BatchNorm1d(128)
        self.batch256 = BatchNorm1d(256)
        self.batch512 = BatchNorm1d(512)
        self.batch1024 = BatchNorm1d(1024)
        
        self.conv1_layers = Sequential(
            Conv1d(self.in_channels, 16, 7),
            BatchNorm1d(16),
            ELU(),
            MaxPool1d(kernel_size=2, stride=2))

        self.conv2_layers = Sequential(
            Conv1d(16, 32, 7),
            BatchNorm1d(32),
            ELU(),
            MaxPool1d(kernel_size=2, stride=2),
            )
        self.avgpool = AvgPool1d(kernel_size=2, stride=2)
        self.linear_layers = Sequential(
            Dropout(0.2),
            Linear(59392, 2),
        )
    
    def forward(self, inp_comb):
        x = self.conv1_layers(inp_comb)
        x = self.conv2_layers(x)
        x1 = x
        x = self.conv5_32(x)
        x = self.conv32to64(self.maxpool2(self.batch32(self.conv5_32(x)+x1)))
        x2 = x
        x = self.conv5_64(x)
        x = self.conv64to128(self.maxpool2(self.batch64(self.conv5_64(x)+x2)))
        x3 = x 
        x = self.conv3_128(x)
        x = self.conv128to256(self.maxpool2(self.batch128(self.conv3_128(x)+x3)))
        x4 = x
        x = self.conv3_256(x)
        x = self.conv256to512(self.maxpool2(self.batch256(self.conv3_256(x)+x4)))
        x5 = x 
        x = self.conv3_512(x)
        x = self.maxpool2(self.batch512(self.conv3_512(x)+x5))
        x = self.conv512to1024(x)
        x6 = x 
        x = self.conv3_1024(x)
        x = self.batch1024(self.conv3_1024(x)+x6)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.linear_layers(x)
        return out
    
class CNNRES_Monitor_Dilated3(nn.Module):
    def __init__(self, args, device, in_channels = 2):
        super().__init__()
        self.args = args
        self.device = device
        self.in_channels = in_channels
        fc_size = 64
        out_class = 1
        
        self.maxpool2 = MaxPool1d(kernel_size=2, stride=2)
        self.avgpool1 = AvgPool1d(kernel_size=2, stride=2)

        self.conv5_32 = Sequential(Conv1d(32, 32, 5, padding=2), ELU(), Conv1d(32, 32, 5, padding=2))
        self.conv5_64 = Sequential(Conv1d(64, 64, 5, padding=4, dilation=2), ELU(), Conv1d(64, 64, 5, padding=4, dilation=2))
        self.conv3_128 = Sequential(Conv1d(128,128,3, padding=2, dilation=2), ELU(), Conv1d(128, 128, 3, padding=2, dilation=2))
        self.conv3_256 = Sequential(Conv1d(256, 256, 3, padding=4, dilation=4), ELU(), Conv1d(256, 256, 3, padding=4, dilation=4))
        self.conv3_512 = Sequential(Conv1d(512, 512, 3, padding=4, dilation=4), ELU(), Conv1d(512,512, 3, padding=4, dilation=4))
        self.conv3_1024 = Sequential(Conv1d(1024, 1024, 3, padding=8, dilation=8), ELU(), Conv1d(1024,1024, 3, padding=8, dilation=8))

        self.conv32to64 = Conv1d(32, 64, 3, padding=1)
        self.conv64to128 = Conv1d(64, 128, 3, padding=1)
        self.conv128to256 = Conv1d(128, 256, 3, padding=1)
        self.conv256to512 = Conv1d(256, 512, 3, padding=1)
        self.conv512to1024 = Conv1d(512, 1024, 3, padding=1)

        self.batch32 = BatchNorm1d(32)
        self.batch64 = BatchNorm1d(64)
        self.batch128 = BatchNorm1d(128)
        self.batch256 = BatchNorm1d(256)
        self.batch512 = BatchNorm1d(512)
        self.batch1024 = BatchNorm1d(1024)
        
        self.conv1_layers = Sequential(
            Conv1d(self.in_channels, 16, 7),
            BatchNorm1d(16),
            ELU(),
            MaxPool1d(kernel_size=2, stride=2))

        self.conv2_layers = Sequential(
            Conv1d(16, 32, 7),
            BatchNorm1d(32),
            ELU(),
            MaxPool1d(kernel_size=2, stride=2),
            )
        self.avgpool = AvgPool1d(kernel_size=2, stride=2)
        self.linear_layers = Sequential(
            Dropout(0.2),
            Linear(59392, 1),
        )
    
    def forward(self, inp_comb):
        x = self.conv1_layers(inp_comb)
        x = self.conv2_layers(x)
        x1 = x
        x = self.conv5_32(x)
        x = self.conv32to64(self.maxpool2(self.batch32(self.conv5_32(x)+x1)))
        x2 = x
        x = self.conv5_64(x)
        x = self.conv64to128(self.maxpool2(self.batch64(self.conv5_64(x)+x2)))
        x3 = x 
        x = self.conv3_128(x)
        x = self.conv128to256(self.maxpool2(self.batch128(self.conv3_128(x)+x3)))
        x4 = x
        x = self.conv3_256(x)
        x = self.conv256to512(self.maxpool2(self.batch256(self.conv3_256(x)+x4)))
        x5 = x 
        x = self.conv3_512(x)
        x = self.maxpool2(self.batch512(self.conv3_512(x)+x5))
        x = self.conv512to1024(x)
        x6 = x 
        x = self.conv3_1024(x)
        x = self.batch1024(self.conv3_1024(x)+x6)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.linear_layers(x)
        return out