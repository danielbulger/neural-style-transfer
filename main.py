import torch
from argparse import ArgumentParser

from PIL import Image

from model.ConvNet import ConvNet
from model import transform


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--cuda', default=False, type=lambda x: str(x).lower() == 'true')
	parser.add_argument('--model', type=str, required=True, help='The PyTorch model to use')
	parser.add_argument('--content', type=str, required=True, help='The content image to transfer to')
	parser.add_argument('--output', type=str, required=True, help='The output file to save the result to')
	return parser.parse_args()


def main():
	args = parse_args()

	if args.cuda and not torch.cuda.is_available():
		raise Exception('CUDA Device not found')

	device = torch.device('cuda' if args.cuda else 'cpu')

	image = Image.open(args.content)
	image_size = image.size

	input_tensor = transform.pil_image_to_tensor(image, (224, 224)).to(device)

	model = ConvNet()
	model.load_state_dict(torch.load(args.model))
	model.to(device)

	output_tensor = model(input_tensor)

	image = transform.tensor_to_image(output_tensor['output'])
	image = image.resize(image_size)
	image.save(args.output)


if __name__ == '__main__':
	main()
