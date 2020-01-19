from torch import nn


class ConvNet(nn.Module):
	_layers = [
		[[3, 64], [64, 64]],
		[[64, 128], [128, 128]],
		[[128, 256], [256, 256], [256, 256]],
		[[256, 512], [512, 512], [512, 512]],
		[[512, 512], [512, 512], [512, 512]]
	]

	_content_layers = [
		'block4_relu3'
	]

	_style_layers = [
		'block1_relu2',
		'block2_relu2',
		'block3_relu3',
		'block4_relu3',
		'block5_relu3'
	]

	def __init__(self):
		super(ConvNet, self).__init__()

		self.sequential = nn.Sequential()

		for index, layer in enumerate(self._layers):
			block = 'block{}'.format(index + 1)
			for num, module in enumerate(layer):
				self.sequential.add_module(
					'{}_conv{}'.format(block, num + 1),
					nn.Conv2d(module[0], module[1], kernel_size=3)
				)

				self.sequential.add_module(
					'{}_relu{}'.format(block, num + 1),
					nn.ReLU(inplace=False)
				)
			self.sequential.add_module(block, nn.MaxPool2d(kernel_size=1, stride=1))

	def forward(self, x):

		content_layers = {}
		style_layers = {}

		for name, module in self.sequential.named_children():
			x = module(x)

			if name in self._content_layers:
				content_layers[name] = x.clone().detach()

			if name in self._style_layers:
				style_layers[name] = x.clone().detach()
		return {
			'output': x,
			'content_features': content_layers,
			'style_features': style_layers
		}
