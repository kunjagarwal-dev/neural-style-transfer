import torch.nn as nn
import torch
import torchvision.models as models

names = {0: "conv1_1", 5: "conv2_1", 10: "conv3_1", 19: "conv4_1", 21: "conv4_2", 28: "conv5_1"}

class VGGFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features

        for layer in self.model.children():
            if isinstance(layer, nn.ReLU):
                layer.inplace = False

        self.model.eval()
        self.model.requires_grad_(False)

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        features = {}

        for i, layer in enumerate(self.model.children()):
            x = layer(x)
            if i in names:
                features[names[i]] = x.shape

            if i == 28:
                break

        return features

