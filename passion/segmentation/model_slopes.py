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
from torch.nn import Conv2d, BatchNorm2d, Module, ModuleList, ReLU, Dropout2d
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

NUM_CLASSES = 18
INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH = 512, 512

class Block(Module):
    def __init__(self, inChannels, outChannels, dropout_prob=0.1):
        super().__init__()
        self.conv1 = Conv2d(inChannels, outChannels, 3, padding=1)
        self.bn1 = BatchNorm2d(outChannels)  # Aggiunto layer di batch normalization
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(outChannels, outChannels, 3, padding=1)
        self.bn2 = BatchNorm2d(outChannels)  # Aggiunto layer di batch normalization
        self.dropout = Dropout2d(dropout_prob) if dropout_prob > 0 else None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)  # Batch normalization
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)  # Batch normalization
        x = self.relu(x)
        if self.dropout:
            x = self.dropout(x)
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
    def __init__(self, channels=(64, 32, 16), dropout_probs=(0.1,0.1,0.1)):
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


# def convrelu(in_channels, out_channels, kernel, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
#         nn.ReLU(inplace=True),
#     )

# import torchvision.models

# class ResNetUNet(nn.Module):
#     def __init__(self, n_class_orient=18, n_class_slope=1):
#         super().__init__()
#         self.base_model = torchvision.models.resnet152(weights='ResNet152_Weights.DEFAULT')
#         self.base_layers = list(self.base_model.children())

#         self.layer0 = nn.Sequential(*self.base_layers[:3]) 
#         self.layer0_1x1 = convrelu(64, 64, 1, 0)
#         self.layer1 = nn.Sequential(*self.base_layers[3:5]) 
#         self.layer1_1x1 = convrelu(256, 64, 1, 0)
#         self.layer2 = self.base_layers[5]  
#         self.layer2_1x1 = convrelu(512, 128, 1, 0)
#         self.layer3 = self.base_layers[6]  
#         self.layer3_1x1 = convrelu(1024, 256, 1, 0)
#         self.layer4 = self.base_layers[7]  
#         self.layer4_1x1 = convrelu(2048, 512, 1, 0)

#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.conv_up3 = convrelu(512 + 256, 256, 3, 1)
#         self.conv_up2 = convrelu(256 + 128, 128, 3, 1)
#         self.conv_up1 = convrelu(128 + 64, 64, 3, 1)
#         self.conv_up0 = convrelu(64 + 64, 64, 3, 1)

#         self.conv_original_size0 = convrelu(3, 64, 3, 1)
#         self.conv_original_size1 = convrelu(64, 64, 3, 1)
#         self.conv_original_size2 = convrelu(64 + 64, 64, 3, 1)

#         # Output per orientamento
#         self.conv_last_orient = nn.Conv2d(64, n_class_orient, 1)
#         # Output per pendenza (regressione)
#         self.conv_last_slope = nn.Conv2d(64, n_class_slope, 1)


#     def forward(self, input):
#         x_original = self.conv_original_size0(input)
#         x_original = self.conv_original_size1(x_original)

#         layer0 = self.layer0(input)
#         layer1 = self.layer1(layer0)
#         layer2 = self.layer2(layer1)
#         layer3 = self.layer3(layer2)
#         layer4 = self.layer4(layer3)

#         layer4 = self.layer4_1x1(layer4)
#         x = self.upsample(layer4)
#         layer3 = self.layer3_1x1(layer3)
#         x = torch.cat([x, layer3], dim=1)
#         x = self.conv_up3(x)

#         x = self.upsample(x)
#         layer2 = self.layer2_1x1(layer2)
#         x = torch.cat([x, layer2], dim=1)
#         x = self.conv_up2(x)

#         x = self.upsample(x)
#         layer1 = self.layer1_1x1(layer1)
#         x = torch.cat([x, layer1], dim=1)
#         x = self.conv_up1(x)

#         x = self.upsample(x)
#         layer0 = self.layer0_1x1(layer0)
#         x = torch.cat([x, layer0], dim=1)
#         x = self.conv_up0(x)

#         x = self.upsample(x)
#         x = torch.cat([x, x_original], dim=1)
#         x = self.conv_original_size2(x)

#         # Due output diversi per i due compiti
#         out_orient = self.conv_last_orient(x)  # classificazione
#         out_slope = self.conv_last_slope(x) 
#         out_slope = out_slope

#         return out_orient, out_slope

import torch
import torch.nn as nn
import torchvision.models


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class ResNetUNet(nn.Module):
    def __init__(self, n_class_orient=18, n_class_slope=8):
        super().__init__()
        self.base_model = torchvision.models.resnet152(weights='ResNet152_Weights.DEFAULT')
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # 64 channels output
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # 256 channels output
        self.layer1_1x1 = convrelu(256, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # 512 channels output
        self.layer2_1x1 = convrelu(512, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # 1024 channels output
        self.layer3_1x1 = convrelu(1024, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # 2048 channels output
        self.layer4_1x1 = convrelu(2048, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Original size processing layers to capture fine details
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)

        # Orient Decoder Path
        self.orient_conv_up3 = convrelu(512 + 256, 256, 3, 1)
        self.orient_conv_up2 = convrelu(256 + 128, 128, 3, 1)
        self.orient_conv_up1 = convrelu(128 + 64, 64, 3, 1)
        self.orient_conv_up0 = convrelu(64 + 64, 64, 3, 1)
        self.orient_final = nn.Conv2d(64, n_class_orient, 1)

        # Slope Decoder Path
        self.slope_conv_up3 = convrelu(512 + 256, 256, 3, 1)
        self.slope_conv_up2 = convrelu(256 + 128, 128, 3, 1)
        self.slope_conv_up1 = convrelu(128 + 64, 64, 3, 1)
        self.slope_conv_up0 = convrelu(64 + 64, 64, 3, 1)
        self.slope_final = nn.Conv2d(64, n_class_slope, 1)

    def forward(self, x):
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(x)
        layer0_1x1 = self.layer0_1x1(layer0)

        layer1 = self.layer1(layer0)
        layer1_1x1 = self.layer1_1x1(layer1)

        layer2 = self.layer2(layer1)
        layer2_1x1 = self.layer2_1x1(layer2)

        layer3 = self.layer3(layer2)
        layer3_1x1 = self.layer3_1x1(layer3)

        layer4 = self.layer4(layer3)
        layer4_1x1 = self.layer4_1x1(layer4)

        # Orient Path
        x_orient = self.upsample(layer4_1x1)
        x_orient = torch.cat([x_orient, layer3_1x1], dim=1)
        x_orient = self.orient_conv_up3(x_orient)
        x_orient = self.upsample(x_orient)
        x_orient = torch.cat([x_orient, layer2_1x1], dim=1)
        x_orient = self.orient_conv_up2(x_orient)
        x_orient = self.upsample(x_orient)
        x_orient = torch.cat([x_orient, layer1_1x1], dim=1)
        x_orient = self.orient_conv_up1(x_orient)
        x_orient = self.upsample(x_orient)
        x_orient = torch.cat([x_orient, layer0_1x1], dim=1)
        x_orient = self.orient_conv_up0(x_orient)
        x_orient = self.upsample(x_orient)
        out_orient = self.orient_final(x_orient)

        # Slope Path
        x_slope = self.upsample(layer4_1x1)
        x_slope = torch.cat([x_slope, layer3_1x1], dim=1)
        x_slope = self.slope_conv_up3(x_slope)
        x_slope = self.upsample(x_slope)
        x_slope = torch.cat([x_slope, layer2_1x1], dim=1)
        x_slope = self.slope_conv_up2(x_slope)
        x_slope = self.upsample(x_slope)
        x_slope = torch.cat([x_slope, layer1_1x1], dim=1)
        x_slope = self.slope_conv_up1(x_slope)
        x_slope = self.upsample(x_slope)
        x_slope = torch.cat([x_slope, layer0_1x1], dim=1)
        x_slope = self.slope_conv_up0(x_slope)
        x_slope = self.upsample(x_slope)
        out_slope = self.slope_final(x_slope)

        return out_orient, out_slope


# def convrelu(in_channels, out_channels, kernel, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
#         nn.ReLU(inplace=True),
#     )

# class ResNetUNet(nn.Module):
#     def __init__(self, n_class_orient=18, n_class_slope=1):
#         super().__init__()
#         self.base_model = torchvision.models.resnet152(weights='ResNet152_Weights.DEFAULT')
#         self.base_layers = list(self.base_model.children())

#         self.layer0 = nn.Sequential(*self.base_layers[:3])  # 64 channels output
#         self.layer0_1x1 = convrelu(64, 64, 1, 0)
#         self.layer1 = nn.Sequential(*self.base_layers[3:5])  # 256 channels output
#         self.layer1_1x1 = convrelu(256, 64, 1, 0)
#         self.layer2 = self.base_layers[5]  # 512 channels output
#         self.layer2_1x1 = convrelu(512, 128, 1, 0)
#         self.layer3 = self.base_layers[6]  # 1024 channels output
#         self.layer3_1x1 = convrelu(1024, 256, 1, 0)
#         self.layer4 = self.base_layers[7]  # 2048 channels output
#         self.layer4_1x1 = convrelu(2048, 512, 1, 0)

#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
#         # Original size processing layers to capture fine details
#         self.conv_original_size0 = convrelu(3, 64, 3, 1)
#         self.conv_original_size1 = convrelu(64, 64, 3, 1)

#         # Orient Decoder Path
#         self.orient_conv_up3 = convrelu(512 + 256, 256, 3, 1)
#         self.orient_conv_up2 = convrelu(256 + 128, 128, 3, 1)
#         self.orient_conv_up1 = convrelu(128 + 64, 64, 3, 1)
#         self.orient_conv_up0 = convrelu(64 + 64, 64, 3, 1)
#         self.orient_final = nn.Conv2d(64, n_class_orient, 1)

#         # Slope Decoder Path
#         self.slope_conv_up3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.slope_conv_up2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.slope_conv_up1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.slope_conv_up0 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.slope_final = nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

#     def forward(self, x):
#         x_original = self.conv_original_size0(x)
#         x_original = self.conv_original_size1(x_original)

#         layer0 = self.layer0(x)
#         layer0_1x1 = self.layer0_1x1(layer0)

#         layer1 = self.layer1(layer0)
#         layer1_1x1 = self.layer1_1x1(layer1)

#         layer2 = self.layer2(layer1)
#         layer2_1x1 = self.layer2_1x1(layer2)

#         layer3 = self.layer3(layer2)
#         layer3_1x1 = self.layer3_1x1(layer3)

#         layer4 = self.layer4(layer3)
#         layer4_1x1 = self.layer4_1x1(layer4)

#         # Orient Path
#         x_orient = self.upsample(layer4_1x1)
#         x_orient = torch.cat([x_orient, layer3_1x1], dim=1)
#         x_orient = self.orient_conv_up3(x_orient)
#         x_orient = self.upsample(x_orient)
#         x_orient = torch.cat([x_orient, layer2_1x1], dim=1)
#         x_orient = self.orient_conv_up2(x_orient)
#         x_orient = self.upsample(x_orient)
#         x_orient = torch.cat([x_orient, layer1_1x1], dim=1)
#         x_orient = self.orient_conv_up1(x_orient)
#         x_orient = self.upsample(x_orient)
#         x_orient = torch.cat([x_orient, layer0_1x1], dim=1)
#         x_orient = self.orient_conv_up0(x_orient)
#         x_orient = self.upsample(x_orient)
#         out_orient = self.orient_final(x_orient)

#         # Slope Path
#         # x_slope = self.upsample(layer4_1x1)
#         # x_slope = torch.cat([x_slope, layer3_1x1], dim=1)
#         x_slope = self.slope_conv_up3(layer4_1x1)
#         # x_slope = self.upsample(x_slope)
#         # x_slope = torch.cat([x_slope, layer2_1x1], dim=1)
#         x_slope = self.slope_conv_up2(x_slope)
#         # x_slope = self.upsample(x_slope)
#         # x_slope = torch.cat([x_slope, layer1_1x1], dim=1)
#         x_slope = self.slope_conv_up1(x_slope)
#         # x_slope = self.upsample(x_slope)
#         # x_slope = torch.cat([x_slope, layer0_1x1], dim=1)
#         x_slope = self.slope_conv_up0(x_slope)
#         # x_slope = self.upsample(x_slope)
#         out_slope = self.slope_final(x_slope)

#         return out_orient, out_slope

