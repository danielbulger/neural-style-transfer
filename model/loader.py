import torch
import torchvision.transforms as transforms
from PIL import Image


def image_to_tensor(device, name, image_size):
	image = Image.open(name)
	loader = transforms.Compose([
		transforms.Resize(image_size),
		transforms.ToTensor(),
		transforms.Normalize(
			(0.485, 0.456, 0.406),
			(0.229, 0.224, 0.225)
		)
	])

	image = loader(image).unsqueeze(0)
	return image.to(device, torch.float)


def tensor_to_image(tensor):
	image = tensor.cpu().clone()
	image = image.squeeze(0)
	return transforms.ToPILImage()(image)
