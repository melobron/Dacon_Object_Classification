from efficientnet_pytorch import EfficientNet
import torch.nn as nn
from timm import create_model
import torchvision.models as models

class Classifier(nn.Module):
    def __init__(self, dim, n_way):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(dim, n_way)

    def forward(self, x):
        x = self.fc(x)
        return x

class Label_propagation_RESNET18(nn.Module):
    def __init__(self, out_channels):
        super(Label_propagation_RESNET18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=False)
        self.classifier = Classifier(1000, out_channels)

    def forward(self, x):
        outputs = self.resnet18(x)
        predict = self.classifier(outputs)
        return predict, outputs

class ConvNextModel(nn.Module):
    def __init__(self, version=1, in_channels=3, num_classes=10):
        super(ConvNextModel, self).__init__()
        if version == 1:
            self.model = create_model('convnext_small', pretrained=False, in_chans=in_channels, num_classes=num_classes)
        elif version == 2:
            self.model = create_model('convnext_large', pretrained=False, in_chans=in_channels, num_classes=num_classes)
        elif version == 3:
            self.model = create_model('convnext_xlarge_384_in22ft1k', pretrained=False, in_chans=in_channels, num_classes=num_classes)

    def forward(self, x):
        output = self.model(x)
        return output

class EfficientNet_MultiLabel(nn.Module):
    def __init__(self, out_channels):
        super(EfficientNet_MultiLabel, self).__init__()
        self.network = EfficientNet.from_name('efficientnet-b0', num_classes=out_channels)
    def forward(self, x):
        x = self.network(x)
        return x

class ViT_model(nn.Module):
    def __init__(self, in_channels=3, num_classes=25):
        super(ViT_model, self).__init__()
        self.model = create_model('vit_small_patch16_224', pretrained=False, in_chans=in_channels, num_classes=num_classes, img_size=224)#, patch_size=8)
    def forward(self, x):
        output = self.model(x)
        return output

class RESNET18(nn.Module):
    def __init__(self, out_channels):
        super(RESNET18, self).__init__()
        self.res18 = models.resnet18(pretrained=False)
        #self.res18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False) #36
        self.res18.fc = nn.Linear(in_features=self.res18.fc.in_features, out_features=out_channels)
        self.feature1 = nn.Sequential(*(list(self.res18.children())[0:8]))
        self.feature2 = nn.Sequential(list(self.res18.children())[8])
        self.feature3 = nn.Sequential(list(self.res18.children())[9])

    def forward(self, x):
        map = self.feature1(x)
        h1 = self.feature2(map)
        output = self.feature3(h1.reshape(h1.shape[0], -1))
        return output#, map

class RESNET161(nn.Module):
    def __init__(self, out_channels):
        super(RESNET18, self).__init__()
        self.res161 = models.RESNET161(pretrained=False)
        #self.res18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,bias=False) #36
        self.res161.fc = nn.Linear(in_features=self.res161.fc.in_features, out_features=out_channels)
        self.feature1 = nn.Sequential(*(list(self.res161.children())[0:8]))
        self.feature2 = nn.Sequential(list(self.res161.children())[8])
        self.feature3 = nn.Sequential(list(self.res161.children())[9])

    def forward(self, x):
        map = self.feature1(x)
        h1 = self.feature2(map)
        output = self.feature3(h1.reshape(h1.shape[0], -1))
        return output#, map