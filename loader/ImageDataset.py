import os
from torch.utils import data
from loader import transform


def get_images(root_dir):
	"""
	Gets all the images from a directory recursively.
	:param root_dir: The starting directory to search for images.
	:return: A list of image files.
	"""
	images = []
	for dir, subdirs, files in os.walk(root_dir):
		for file in files:
			if file.endswith(('jpg', 'jpeg', 'png')):
				images.append(os.path.join(dir, file))

	return images


class ImageDataset(data.Dataset):

	def __init__(self, root):
		self.files = get_images(root)

	def __getitem__(self, item):
		return transform.image_to_tensor(
			self.files[item],
			(224, 224),
			False
		)

	def __len__(self):
		return len(self.files)
