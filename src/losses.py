import torch.nn.functional as F

def content_loss(generated, target):
    # Target must already be detached because it is a fixed reference and no gradient should flow into it
    loss = F.mse_loss(generated, target)
    return loss