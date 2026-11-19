from PIL import Image, ImageOps
from torchvision import transforms

def load_image(path, max_size, device):
    image = Image.open(path).convert("RGB")

    image = ImageOps.contain(image, (max_size, max_size), method=Image.Resampling.LANCZOS)

    image_tensor = transforms.ToTensor()(image).view(1, 3, image.size[1], image.size[0]).to(device)
    return image_tensor

def save_image(tensor, path):
    tensor = tensor.clone().detach().cpu().squeeze(0)
    image = transforms.ToPILImage()(tensor)
    image.save(path)