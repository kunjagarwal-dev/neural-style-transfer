import torch

from src.losses import content_loss, style_grams, style_loss, total_variation_loss


def run_style_transfer(extractor, content, style, steps=500,
                       content_weight=1.0, style_weight=1e4,
                       tv_weight=0.0 ,lr=0.02, log_every=100):
    
    with torch.no_grad():
        content_target = extractor(content)["conv4_2"].detach()
        target_grams = {k: v.detach() for k, v in style_grams(extractor(style)).items()}

    image = content.clone().requires_grad_(True)
    optimiser = torch.optim.Adam([image], lr=lr)

    history = []
    for i in range(steps + 1):
        features = extractor(image)
        c = content_loss(features["conv4_2"], content_target)
        s = style_loss(features, target_grams)
        tv = total_variation_loss(image)
        total = content_weight * c + style_weight * s + tv_weight * tv

        optimiser.zero_grad()
        total.backward()
        optimiser.step()

        with torch.no_grad():
            image.clamp_(0, 1)

        history.append((i, c.item(), s.item(), total.item(), tv.item()))
        if log_every and i % log_every == 0:
            print(f"step {i}: c={c.item():.4f}  s={s.item():.4f}  total={total.item():.4f}  tv={tv.item():.4f}")

    return image.detach(), history