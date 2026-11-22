import torch.nn.functional as F
import torch

def content_loss(generated, target):
    # Target must already be detached because it is a fixed reference and no gradient should flow into it
    loss = F.mse_loss(generated, target)
    return loss

def gram_matrix(features):
    b,c,h,w = features.size()
    features = features.view(c, h*w)
    gram = features @ features.t()
    gram = gram / (c * h * w)
    return gram