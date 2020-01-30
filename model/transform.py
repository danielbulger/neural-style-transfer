import torchvision.transforms as transforms
import torch
from PIL import Image
from torch.autograd import Variable


def pil_image_to_tensor(image, image_size):
	image = image.convert('RGB')
	loader = get_transforms(image_size)

	image = Variable(loader(image))
	image = image.unsqueeze(0)

	return image


def get_transforms(image_size):
	return transforms.Compose([
		transforms.Resize(image_size),
		transforms.ToTensor(),
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225]
		),
		transforms.Lambda(lambda x: x.mul_(255)),
	])


def image_to_tensor(name, image_size):
	return pil_image_to_tensor(Image.open(name), image_size)


def tensor_to_image(tensor):
	transform = transforms.Compose([
		transforms.Lambda(lambda x: x.mul_(1.0 / 255)),
		transforms.Normalize(
			mean=[-0.40760392, -0.45795686, -0.48501961],
			std=[1, 1, 1]
		)
	])
	image = tensor.cpu().clone()
	image = image.squeeze(0)
	image = transform(image)
	image = image.clamp_(0, 1)
	return transforms.ToPILImage()(image)
