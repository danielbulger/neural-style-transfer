import torch
from torch import optim

from model import loader
from model.ConvNet import ConvNet
from argparse import ArgumentParser


def get_args():
	parser = ArgumentParser()
	parser.add_argument('--cuda', default=False, type=lambda x: str(x).lower() == 'true')
	parser.add_argument('--model', type=str, help='The PyTorch model to use')
	parser.add_argument('--style', type=str, required=True, help='The style image to learn')
	parser.add_argument('--content', type=str, required=True, help='The content image to transfer to')
	parser.add_argument('--iterations', type=int, default=50, help='The number of training iterations')
	parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')
	parser.add_argument('--checkpoints', type=int, default=5, help='The number of iterations before checkpoints')
	return parser.parse_args()


def main():
	args = get_args()

	if args.cuda and not torch.cuda.is_available():
		raise Exception('CUDA Device not found')

	device = torch.device('cuda' if args.cuda else 'cpu')

	style_tensor = loader.image_to_tensor(device, args.style, 224)
	content_tensor = loader.image_to_tensor(device, args.content, 224)

	model = ConvNet()

	# White noise
	input_image = torch.randn(content_tensor.data.size(), device=device)

	if args.model:
		model.load_state_dict(torch.load(args.model))

	model.to(device)

	optimiser = optim.Adam(model.parameters(), lr=args.lr)

	for x in range(args.iterations):

		def closure():
			optimiser.zero_grad()

			model(input_image)

			style_loss = 0
			content_loss = 0

			loss = style_loss + content_loss

			return loss

		optimiser.step(closure)


if __name__ == '__main__':
	main()
