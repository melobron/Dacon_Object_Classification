from efficientnet_pytorch import EfficientNet

class EfficientNet_MultiLabel(nn.Module):
    def __init__(self, out_channels):
        super(EfficientNet_MultiLabel, self).__init__()
        self.network = EfficientNet.from_name(name="efficientnet-b7", num_classes=out_channels)
    def forward(self, x):
        x = self.network(x)
        return x, 