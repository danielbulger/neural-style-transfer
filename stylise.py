import os
import torch
import torch.nn.functional as F
from PIL import Image
from torch import optim
from torch.autograd import Variable

from model import transform
from model.ConvNet import ConvNet
from argparse import ArgumentParser


def get_args():
	parser = ArgumentParser()
	parser.add_argument('--cuda', default=False, type=lambda x: str(x).lower() == 'true')
	parser.add_argument('--style', type=str, required=True, help='The style image to learn')
	parser.add_argument('--content', type=str, required=True, help='The content image to stylise')
	parser.add_argument('--iterations', type=int, default=50, help='The number of training iterations')
	parser.add_argument('--checkpoints', type=int, default=5, help='The number of iterations before checkpoints')
	parser.add_argument('--style-weight', type=float, default=1, help='The amount to scale the style loss')
	parser.add_argument('--content-weight', type=float, default=1, help='The amount to scale the content loss')
	parser.add_argument('--variation-weight', type=float, default=1, help='The amount to scale the variation loss')
	parser.add_argument('--log-dir', type=str, required=True, help='The directory to save the model checkpoints')
	return parser.parse_args()


def gram_matrix(tensor):
	(batch, channel, width, height) = tensor.size()
	features = tensor.view(batch, channel, width * height)
	transpose = features.transpose(1, 2)
	gram = features.bmm(transpose)
	return gram.div(channel * width * height)


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

	input_img = Variable(content_tensor.clone(), requires_grad=True)

	optimiser = optim.LBFGS(params=[input_img])

	losses = {
		'content': 0,
		'style': 0,
		'variation': 0,
		'total': 0
	}

	for iteration in range(1, args.iterations + 1):

		def closure():
			output = model(input_img)

			style_loss = 0
			content_loss = 0

			for x in range(len(output['content'])):
				content_loss += F.mse_loss(
					output['content'][x],
					content_features[x],
				) * args.content_weight

			for x in range(len(output['style'])):
				style_loss += F.mse_loss(
					gram_matrix(output['style'][x]),
					style_features[x]
				) * args.style_weight

			x_var = torch.abs(input_img[:, :, 1:, :] - input_img[:, :, :-1, :])
			y_var = torch.abs(input_img[:, 1:, :, :] - input_img[:, :-1, :, :])
			variation_loss = (torch.mean(x_var) + torch.mean(y_var)) * args.variation_weight

			loss = style_loss + content_loss + variation_loss

			optimiser.zero_grad()
			loss.backward()

			losses['style'] += style_loss
			losses['content'] += content_loss
			losses['variation'] += variation_loss
			losses['total'] += loss

			return loss

		if iteration % args.checkpoints == 0:
			print(f"Iteration {iteration}")

			for key, value in losses.items():
				print(f"\t{key}: {value}")
				# Reset the losses for the next epoch
				losses[key] = 0

			save_image(
				args.log_dir,
				f"checkpoint-{iteration}.png",
				input_img,
				content_dimensions
			)

		optimiser.step(closure)

	save_image(
		args.log_dir,
		'final.png',
		input_img,
		content_dimensions
	)


if __name__ == '__main__':
	main()
