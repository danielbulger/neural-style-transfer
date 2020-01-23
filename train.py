import os
import torch
import torch.nn.functional as F
from PIL import Image
from torch import optim

from model import transform
from model.ConvNet import ConvNet
from argparse import ArgumentParser


def get_args():
	parser = ArgumentParser()
	parser.add_argument('--cuda', default=False, type=lambda x: str(x).lower() == 'true')
	parser.add_argument('--model', type=str, help='The PyTorch model to use')
	parser.add_argument('--style', type=str, required=True, help='The style image to learn')
	parser.add_argument('--content', type=str, required=True, help='The content image to stylise')
	parser.add_argument('--iterations', type=int, default=50, help='The number of training iterations')
	parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')
	parser.add_argument('--checkpoints', type=int, default=5, help='The number of iterations before checkpoints')
	parser.add_argument('--style-weight', type=float, default=1, help='The amount to scale the style loss')
	parser.add_argument('--content-weight', type=float, default=1, help='The amount to scale the content loss')
	parser.add_argument('--log-dir', type=str, required=True, help='The directory to save the model checkpoints')
	return parser.parse_args()


def gram_matrix(tensor):
	(batch, channel, width, height) = tensor.size()
	features = tensor.view(batch * channel, width * height)
	gram = torch.mm(features, features.t())
	return gram.div(batch * channel * width * height)


def save_image(directory, filename, tensor, size):
	image = transform.tensor_to_image(tensor.cpu().detach())
	image = image.resize(size)
	image.save(os.path.join(directory, filename))


def main():
	args = get_args()

	if args.cuda and not torch.cuda.is_available():
		raise Exception('CUDA Device not found')

	device = torch.device('cuda' if args.cuda else 'cpu')

	style_tensor = transform.image_to_tensor(args.style, (224, 224)).to(device)
	content_image = Image.open(args.content)
	content_dimensions = content_image.size
	content_tensor = transform.pil_image_to_tensor(content_image, (224, 224)).to(device)

	model = ConvNet()

	for param in model.parameters():
		param.requires_grad = False

	model.to(device)

	style_output = model(style_tensor)
	style_features = [gram_matrix(x.detach()) for x in style_output['style']]

	content_output = model(content_tensor)
	content_features = [x.detach() for x in content_output['content']]

	input_noise = torch.randn(1, 3, 224, 224, device=device, requires_grad=True)

	optimiser = optim.Adam(params=[input_noise], lr=args.lr)

	losses = {
		'content': 0,
		'style': 0,
		'total': 0
	}

	def closure():

		return loss

	for iteration in range(1, args.iterations + 1):
		input_noise.data.clamp_(0, 1)
		optimiser.zero_grad()

		output = model(input_noise)

		style_loss = 0
		content_loss = 0

		for x in range(len(output['content'])):
			content_loss += F.mse_loss(
				output['content'][x],
				content_features[x],
			)

		for x in range(len(output['style'])):
			style_loss += F.mse_loss(
				gram_matrix(output['style'][x]),
				style_features[x]
			)

		style_loss *= args.style_weight
		content_loss *= args.content_weight

		style_loss.to(device)
		content_loss.to(device)

		loss = style_loss + content_loss
		loss.backward()

		losses['style'] += style_loss
		losses['content'] += content_loss
		losses['total'] += loss

		if iteration % args.checkpoints == 0:
			print(f"Iteration {iteration}")

			for key, value in losses.items():
				print(f"\t{key}: {value}")
				# Reset the losses for the next epoch
				losses[key] = 0

			save_image(
				args.log_dir,
				f'checkpoint-{iteration}.png',
				input_noise,
				content_dimensions
			)

	optimiser.step(closure)

	input_noise.data.clamp_(0, 1)
	save_image(
		args.log_dir,
		'final.png',
		input_noise,
		content_dimensions
	)


if __name__ == '__main__':
	main()
