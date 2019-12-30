from torch import nn


class ConvNet(nn.Module):
	_layers = [
		[[3, 64], [64, 64]],
		[[64, 128], [128, 128]],
		[[128, 256], [256, 256], [256, 256]],
		[[256, 512], [512, 512], [512, 512]],
		[[512, 512], [512, 512], [512, 512]]
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
		return self.sequential(x)

	def get_module(self, name: str):
		return self.sequential.__getattr__(name)
