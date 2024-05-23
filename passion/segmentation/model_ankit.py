from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
from torch.nn import Dropout2d
import torch.nn as nn
from torch.nn import Conv2d, BatchNorm2d, Module, ModuleList, ReLU, Dropout2d
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

NUM_CLASSES = 18
INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH = 512, 512

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Aggiungi dropout qui nel SELayer
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, w, h = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(outChannels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.3)  # Aumentato il dropout qui
        self.conv2 = nn.Conv2d(outChannels, outChannels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(outChannels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Applicare dropout dopo la prima attivazione
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        self.encBlocks = ModuleList([Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)])
        self.pool = MaxPool2d(2)

    def forward(self, x):
        blockOutputs = []
        for block in self.encBlocks:
            x = block(x)
            blockOutputs.append(x)
            x = self.pool(x)
        return blockOutputs


class Decoder(Module):
    def __init__(self, channels=(64, 32, 16), dropout_probs=(0.2,0.2, 0.2)):
        super().__init__()
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
             for i in range(len(channels) - 1)])
        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i + 1], dropout_prob=dropout_probs[i])
             for i in range(len(channels) - 1)])

    def forward(self, x, encFeatures):
        for i in range(len(self.channels) - 1):
            x = self.upconvs[i](x)
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, encFeatures, x):
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        return encFeatures


class UNet(Module):
    def __init__(self, encChannels=(3, 16, 32, 64, 128), decChannels=(128, 64, 32, 16), nbClasses=NUM_CLASSES, retainDim=True, outSize=(INPUT_IMAGE_HEIGHT,  INPUT_IMAGE_WIDTH)):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)
        # initialize the regression head and store the class variables
        self.head = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize
    
    def forward(self, x):
        # grab the features from the encoder
        encFeatures = self.encoder(x)
        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0],
            encFeatures[::-1][1:])
        # pass the decoder features through the regression head to
        # obtain the segmentation mask
        map = self.head(decFeatures)
        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = F.interpolate(map, self.outSize)
        # return the segmentation map
        return map


import torch.nn as nn
import torchvision.models


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

import torchvision.models

class ResNetUNet(nn.Module):
    def __init__(self, n_class=NUM_CLASSES):
        super().__init__()

        self.base_model = torchvision.models.resnet152(weights='ResNet152_Weights.DEFAULT')
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) 
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) 
        self.layer1_1x1 = convrelu(256, 64, 1, 0)  # Da 256 a 64 
        self.SE_layer_1 = SELayer(64)
        self.layer2 = self.base_layers[5]  
        self.layer2_1x1 = convrelu(512, 128, 1, 0)  # Da 512 a 128
        self.SE_layer_2 = SELayer(128)
        self.layer3 = self.base_layers[6]  
        self.layer3_1x1 = convrelu(1024, 256, 1, 0)  # Da 1024 a 256
        self.SE_layer_3 = SELayer(256)
        self.layer4 = self.base_layers[7]  
        self.layer4_1x1 = convrelu(2048, 512, 1, 0)  # Da 2048 a 512

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Aggiustamento delle dimensioni dei canali in base alla concatenazione
        self.conv_up3 = convrelu(512 + 256, 256, 3, 1)  # Concatena layer4_1x1 e layer3_1x1
        self.conv_up2 = convrelu(256 + 128, 128, 3, 1)  # Concatena l'output di conv_up3 e layer2_1x1
        self.conv_up1 = convrelu(128 + 64, 64, 3, 1)   # Concatena l'output di conv_up2 e layer1_1x1
        self.conv_up0 = convrelu(64 + 64, 64, 3, 1)    # Concatena l'output di conv_up1 e layer0_1x1

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 64, 64, 3, 1) # Da 64 + 64 a 64
        
        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        SE_layer3 = self.SE_layer_3(layer3)
        x = torch.cat([x, SE_layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        SE_layer2 = self.SE_layer_2(layer2)
        x = torch.cat([x, SE_layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        SE_layer1 = self.SE_layer_1(layer1)
        x = torch.cat([x, SE_layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out 
    