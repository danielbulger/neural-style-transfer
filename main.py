import torch
from argparse import ArgumentParser
from model.ConvNet import ConvNet


def parse_args():
	parser = ArgumentParser()
	parser.add_argument('--cuda', default=False, type=lambda x: str(x).lower() == 'true')
	parser.add_argument('--model', type=str, help='The PyTorch model to use')
	parser.add_argument('--content', type=str, help='The content image to transfer to')
	parser.add_argument('--output', type=str, help='The output file to save the result to')
	return parser.parse_args()


def main():
	args = parse_args()

	if args.cuda and not torch.cuda.is_available():
		raise Exception('CUDA Device not found')

	device = torch.device('cuda' if args.cuda else 'cpu')
	model = ConvNet()
	model.load_state_dict(torch.load(args.model))
	model.to(device)


if __name__ == '__main__':
	main()
