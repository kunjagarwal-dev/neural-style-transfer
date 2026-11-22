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

style_list = [
    "conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"
]

def style_grams(features):
    grams = {}
    for layer in style_list:
        grams[layer] = gram_matrix(features[layer])
    return grams

def style_loss(generated_features, target_grams, weights=None):
    if weights is None:
        weights = {layer: 1.0 for layer in style_list}

    loss=0
    for layer in style_list:
        generated_gram = gram_matrix(generated_features[layer])
        target_gram = target_grams[layer]
        layer_loss = F.mse_loss(generated_gram, target_gram)
        loss += weights[layer] * layer_loss

    return loss
    