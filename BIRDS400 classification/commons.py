import io


from PIL import Image
from torchvision import models
import torchvision.transforms as transforms
import torch


def get_model():
    
    model = models.resnet50(weights=True)
    model = torch.load(r"C:\Users\yigit\Masaüstü\BIRDS400 classification\BIRDS400 CLASSIFICATION.pth", map_location="cpu")
    model.eval()
    return model


def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)


