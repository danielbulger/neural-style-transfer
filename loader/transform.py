import torchvision.transforms as transforms
from PIL import Image


def pil_image_to_tensor(image, image_size, add_batch=True):
	image = image.convert('RGB')
	loader = get_transforms(image_size)

	image = loader(image)

	if add_batch:
		image = image.unsqueeze(0)

	return image


def get_transforms(image_size):
	return transforms.Compose([
		transforms.Resize(image_size),
		transforms.ToTensor(),
		transforms.Normalize(
			(0.485, 0.456, 0.406),
			(0.229, 0.224, 0.225)
		)
	])


def image_to_tensor(name, image_size, add_batch=True):
	return pil_image_to_tensor(Image.open(name), image_size, add_batch)


def tensor_to_image(tensor):
	image = tensor.cpu().clone()
	image = image.squeeze(0)
	return transforms.ToPILImage()(image)
