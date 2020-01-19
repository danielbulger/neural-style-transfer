import os
import torch
import torch.nn.functional as F
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
	parser.add_argument('--style-weight', type=float, default=1, help='The amount to scale the style loss')
	parser.add_argument('--content-weight', type=float, default=1, help='The amount to scale the content loss')
	parser.add_argument('--log-dir', type=str, required=True, help='The directory to save the model checkpoints')
	return parser.parse_args()


def gram_matrix(tensor):
	(batch, channels, height, width) = tensor.size()
	features = tensor.view(batch, channels, width * height)
	transpose = features.transpose(1, 2)
	gram = features.bmm(transpose) / (channels * height * width)
	return gram


def main():
	args = get_args()

	if args.cuda and not torch.cuda.is_available():
		raise Exception('CUDA Device not found')

	device = torch.device('cuda' if args.cuda else 'cpu')

	style_tensor = loader.image_to_tensor(device, args.style, (224, 224))
	content_tensor = loader.image_to_tensor(device, args.content, (224, 224))

	model = ConvNet()

	# White noise
	input_image = torch.randn(content_tensor.data.size(), device=device)

	if args.model:
		model.load_state_dict(torch.load(args.model))

	model.to(device)

	optimiser = optim.Adam(model.parameters(), lr=args.lr)
	losses = {
		'total': 0,
		'content': 0,
		'style': 0
	}

	for epoch in range(1, args.iterations + 1):

		input_image.data.clamp_(0, 1)

		optimiser.zero_grad()

		style = model(style_tensor)
		content = model(input_image)

		style_loss = 0
		content_loss = 0

		for name in style['content_features'].keys():
			content_loss += F.mse_loss(
				content['content_features'][name],
				style['content_features'][name]
			)

		style_loss *= args.style_weight
		content_loss *= args.content_weight

		loss = style_loss + content_loss
		loss.backward()
		optimiser.step()

		losses['style'] += style_loss
		losses['content'] += content_loss
		losses['total'] += loss

		if epoch % args.checkpoints == 0:
			print(
				f"Epoch: {epoch}\tContent: {losses['content']:.6f}\tStyle: {losses['style']}\tTotal: {losses['total']}")
			# Save the current model to the checkpoint directory.
			torch.save(model.state_dict(), os.path.join(args.log_dir, f'checkpoint{epoch}.pt'))
			# Reset the losses tally.
			losses['style'] = losses['content'] = losses['total'] = 0


if __name__ == '__main__':
	main()
