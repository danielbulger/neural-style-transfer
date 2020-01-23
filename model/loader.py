import torch
import torchvision.transforms as transforms
from PIL import Image


def pil_image_to_tensor(device, image, image_size):
	image = image.convert('RGB')
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


def image_to_tensor(device, name, image_size):
	return pil_image_to_tensor(device, Image.open(name), image_size)


def tensor_to_image(tensor):
	image = tensor.cpu().clone()
	image = image.squeeze(0)
	return transforms.ToPILImage()(image)
