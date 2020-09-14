import torchvision.transforms as transforms

from PIL import Image


def pil_image_to_tensor(image, image_size):
	image = image.convert('RGB')
	loader = get_transforms(image_size)

	image = loader(image)
	image = image.unsqueeze(0)

	return image


def get_transforms(image_size):
	return transforms.Compose([
		transforms.Resize(image_size),
		transforms.ToTensor(),
	])


def image_to_tensor(name, image_size):
	return pil_image_to_tensor(Image.open(name), image_size)


def tensor_to_image(tensor):
	image = tensor.cpu().clone()
	image = image.squeeze(0)
	return transforms.ToPILImage()(image)
