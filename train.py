import os
import torch
import torch.nn.functional as F
from torch import optim

from loader import transform
from loader.ImageDataset import ImageDataset
from model.ConvNet import ConvNet
from argparse import ArgumentParser


def get_args():
	parser = ArgumentParser()
	parser.add_argument('--cuda', default=False, type=lambda x: str(x).lower() == 'true')
	parser.add_argument('--images', type=str, required=True, help='The folder that contains the training images')
	parser.add_argument('--workers', type=int, help='The number of threads to use when loading the training set')
	parser.add_argument('--model', type=str, help='The PyTorch model to use')
	parser.add_argument('--style', type=str, required=True, help='The style image to learn')
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

	workers = args.workers
	if workers is None:
		from multiprocessing import cpu_count
		workers = cpu_count()

	device = torch.device('cuda' if args.cuda else 'cpu')

	style_tensor = transform.image_to_tensor(args.style, (224, 224)).to(device)

	model = ConvNet()

	if args.model:
		model.load_state_dict(torch.load(args.model))

	model.to(device)

	optimiser = optim.Adam(model.parameters(), lr=args.lr)
	losses = {
		'content': 0,
		'style': 0,
		'total': 0
	}

	train_dataset = ImageDataset(args.images)

	train_loader = torch.utils.data.DataLoader(
		train_dataset,
		batch_size=1,
		num_workers=workers,
		shuffle=False
	)

	for epoch in range(1, args.iterations + 1):
		for index, data in enumerate(train_loader):

			data = data.to(device)

			optimiser.zero_grad()

			style = model(style_tensor)
			content = model(data)

			style_loss = 0
			content_loss = 0

			for x in range(len(style['content'])):
				content_loss += F.mse_loss(
					content['content'][x],
					style['content'][x],
				)

			for x in range(len(style['style'])):
				style_loss += F.mse_loss(
					gram_matrix(style['style'][x]),
					gram_matrix(content['style'][x])
				)

			style_loss *= args.style_weight
			content_loss *= args.content_weight

			loss = style_loss + content_loss

			loss.backward()
			optimiser.step()

			losses['style'] += style_loss
			losses['content'] += content_loss
			losses['total'] += loss

			if ((index + 1) * epoch) % args.checkpoints == 0:
				print(f"Epoch {epoch}: Iteration {index}")

				for key, value in losses.items():
					print(f"\t{key}: {value}")
					# Reset the losses for the next epoch
					losses[key] = 0
				# Save the current model to the checkpoint directory.
				torch.save(model.state_dict(), os.path.join(args.log_dir, f'checkpoint{epoch}.pt'))

	# Save the finished model
	torch.save(model.state_dict(), os.path.join(args.log_dir, 'model-final.pt'))


if __name__ == '__main__':
	main()
