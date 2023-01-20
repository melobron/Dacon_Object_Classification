import timm
import torch
import torch.nn as nn

# from efficientnet_pytorch import EfficientNet
from vit_pytorch import ViT
from timm import create_model


# class EfficientNetModel(nn.Module):
#     def __init__(self, out_channels=10):
#         super(EfficientNetModel, self).__init__()
#         self.model = EfficientNet.from_name('efficientnet-b7', num_classes=out_channels)
#
#     def forward(self, x):
#         x = self.model(x)
#         return x


class ViTModel(nn.Module):
    def __init__(self, image_size, in_channels=3, num_classes=10):
        super(ViTModel, self).__init__()
        self.model = ViT(image_size=image_size, patch_size=32, num_classes=num_classes, dim=1024, depth=6, heads=16,
                             mlp_dim=2048, dropout=0.1, emb_dropout=0.1)

    def forward(self, x):
        output = self.model(x)
        return output

# class ViTModel(nn.Module):
#     def __init__(self):
#         super(ViTModel, self).__init__()
#         self.model = timm.create_model('vit_base_resnet50_384', pretrained=False)
#
#     def forward(self, x):
#         output = self.model(x)
#         return output


class ConvNextModel(nn.Module):
    def __init__(self, version=1):
        super(ConvNextModel, self).__init__()
        if version == 1:
            self.model = timm.create_model('convnext_small', pretrained=False)
        elif version == 2:
            self.model = timm.create_model('convnext_large', pretrained=False)
        elif version == 3:
            self.model = timm.create_model('convnext_xlarge_384_in22ft1k', pretrained=False)

    def forward(self, x):
        output = self.model(x)
        return output


# model_names = timm.list_models(pretrained=False)
# print(model_names)
#
# model_names = [model for model in model_names if "convnext" in model]
# print(model_names)

# ['convnext_base', 'convnext_base_384_in22ft1k', 'convnext_base_in22ft1k', 'convnext_base_in22k', 'convnext_large',
#  'convnext_large_384_in22ft1k', 'convnext_large_in22ft1k', 'convnext_large_in22k', 'convnext_small', 'convnext_tiny',
#  'convnext_tiny_hnf', 'convnext_xlarge_384_in22ft1k', 'convnext_xlarge_in22ft1k', 'convnext_xlarge_in22k']

